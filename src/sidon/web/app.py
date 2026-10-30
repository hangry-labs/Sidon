# SPDX-License-Identifier: Apache-2.0
"""One-process local studio and bounded asynchronous audio API."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import gc
import hmac
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Literal
import uuid
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import soundfile as sf
import torch

from sidon import __version__
from .gpu import GpuMonitor
from .audio import OutputFormat, decode_input, encode_file, encode_audio
from .queue import JobQueue, InstanceLock, finish_in_thread
from .protocol import MODELS, AudioJob, JobList, AUDIO_RESPONSES, PUBLIC_ERRORS, api_error

LOG = logging.getLogger("sidon")
STATIC = Path(__file__).parent / "static"
TERMINAL = {"done", "failed", "cancelled"}


class JobCancelled(Exception):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    strength: float = Field(default=1, ge=0, le=1)
    output_format: OutputFormat = "wav"
    chunk_seconds: float = Field(default=4, ge=2, le=16)
    overlap_seconds: float = Field(default=1, ge=0.1, le=2)

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("Overlap must be shorter than the separation window")
        return self


class ModelAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["load", "unload"]


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        self.settings = Settings()
        if (root / "settings.json").exists():
            self.settings = Settings.model_validate_json((root / "settings.json").read_text())
        self.jobs: dict[str, dict] = {}
        self.active_ids: set[str] = set()
        self.state_lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.models: dict[str, object] = {}
        self.model_states = {"enhance": "unloaded", "separate": "unloaded"}
        self.model_errors: dict[str, str] = {}
        self.admissions = {}
        self.clocks = {}
        self.capacity = int(os.environ.get("SIDON_MAX_ACTIVE_JOBS", "4"))
        if not 1 <= self.capacity <= 64:
            raise ValueError("SIDON_MAX_ACTIVE_JOBS must be between 1 and 64")
        self.device = os.environ.get("SIDON_DEVICE", "auto")
        self.retention = int(os.environ.get("SIDON_RETENTION_HOURS", "24")) * 3600
        self.max_bytes = int(os.environ.get("SIDON_MAX_UPLOAD_MB", "100")) * 1024 * 1024
        self.max_seconds = int(os.environ.get("SIDON_MAX_AUDIO_SECONDS", "600"))
        if min(self.retention, self.max_bytes, self.max_seconds) <= 0:
            raise ValueError("Retention, upload and duration limits must be positive")
        self.lease = InstanceLock(root)
        self.pool = JobQueue()
        self.gpu = GpuMonitor()
        for path in (root / "jobs").glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if str(uuid.UUID(job["id"])) != path.parent.name:
                    continue
                if job["status"] in {"uploading", "queued", "running"}:
                    job.update(status="failed", error="The service restarted. Submit the audio again.")
                    write_json(path, job)
                self.jobs[job["id"]] = job
                if job['status'] not in TERMINAL:
                    self.active_ids.add(job['id'])
            except (ValueError, KeyError, OSError):
                LOG.warning("Ignoring unreadable job metadata")
        self.cleanup()

    def cleanup(self):
        with self.state_lock:
            for job_id, job in list(self.jobs.items()):
                if job["status"] in TERMINAL and time.time() - job["created"] > self.retention:
                    shutil.rmtree(self.root / "jobs" / job_id, ignore_errors=True)
                    del self.jobs[job_id]
                    self.active_ids.discard(job_id)
            # Interrupted uploads never become queued jobs.
            for path in (self.root / "jobs").iterdir():
                if path.is_dir() and path.name not in self.jobs and time.time() - path.stat().st_mtime > 3600:
                    shutil.rmtree(path, ignore_errors=True)

    def reserve(self):
        with self.state_lock:
            if self.pool.closed:
                raise HTTPException(503, 'The service is shutting down', headers={'Retry-After':'1'})
            active = len(self.active_ids)
            if active + len(self.admissions) >= self.capacity:
                raise HTTPException(429, 'The audio queue is full. Retry after capacity is available.', headers={'Retry-After':'1'})
            job_id = str(uuid.uuid4())
            self.admissions[job_id] = dict(created=time.time(), received=time.monotonic())
            return job_id

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.gpu.close()
        self.lease.close()

    def snapshot(self, job_id):
        with self.state_lock:
            if job_id not in self.jobs:
                raise HTTPException(404, 'Job not found or expired')
            job = dict(self.jobs[job_id])
            job['queue_position'] = self.pool.positions().get(job_id, 1) if job['status'] == 'queued' else 0 if job['status'] == 'running' else None
            return job

    def queue_state(self):
        with self.state_lock:
            states = [self.jobs[job_id]['status'] for job_id in self.active_ids]
            active = len(states) + len(self.admissions)
            return dict(capacity=self.capacity, active=active, available=self.capacity-active,
                        uploading=states.count('uploading')+len(self.admissions), queued=states.count('queued'),
                        running=states.count('running'), ordering='ready_fifo', workers=1)

    def check_cancel(self, job_id):
        with self.state_lock:
            if self.jobs[job_id].get('cancel_requested') or self.jobs[job_id]['status'] == 'cancelled':
                raise JobCancelled()

    def finish(self, job_id, **fields):
        with self.state_lock:
            fields.update(self.timings(job_id))
            self.clocks.pop(job_id, None)
            self.update(job_id, **fields)

    def timings(self, job_id):
        clock = self.clocks.get(job_id, {})
        now = time.monotonic()
        processing = max(0, now-clock.get('started', now))
        return dict(finished_at=time.time(), elapsed_seconds=round(processing, 3),
                    processing_seconds=round(processing, 3), total_seconds=round(now-clock.get('received', now), 3))

    def cancel(self, job_id):
        with self.state_lock:
            if job_id not in self.jobs:
                raise HTTPException(404, 'Job not found or expired')
            job = self.jobs[job_id]
            if job['status'] == 'queued':
                self.pool.cancel(job_id)
                self.finish(job_id, status='cancelled', stage='Cancelled', cancel_requested=True)
            elif job['status'] == 'running':
                self.update(job_id, cancel_requested=True, stage='Cancellation requested')
            elif job['status'] == 'uploading':
                raise HTTPException(409, 'The upload is still being received')
            return self.snapshot(job_id)

    def update(self, job_id: str, **fields):
        with self.state_lock:
            self.jobs[job_id].update(fields)
            if self.jobs[job_id]['status'] in TERMINAL:
                self.active_ids.discard(job_id)
            else:
                self.active_ids.add(job_id)
            write_json(self.root / "jobs" / job_id / "job.json", self.jobs[job_id])

    def load(self, name: str):
        if name in self.models:
            return self.models[name]
        self.model_states[name] = "loading"
        try:
            if name == "enhance":
                from sidon.inference.restoration import SidonRestorer
                model = SidonRestorer(self.device)
            else:
                from sidon.inference.assets import resolve_asset
                from sidon.inference.mossformer import MossFormer2Separator
                model = MossFormer2Separator(device=self.device, checkpoint=resolve_asset("last_best_checkpoint.pt"))
            self.models[name] = model
            self.model_states[name] = "ready"
            self.model_errors.pop(name, None)
            return model
        except Exception:
            self.model_states[name] = "error"
            self.model_errors[name] = "Could not load the model. Check available memory and the model cache; tiny images need a first online download."
            raise

    def process(self, job_id: str):
        started = time.monotonic()
        with self.state_lock:
            job = self.jobs.get(job_id)
            if not job or job['status'] == 'cancelled':
                return
            clock = self.clocks[job_id]
            clock['started'] = started
            self.update(job_id, status="running", started_at=time.time(), queue_wait_seconds=round(started-clock['ready'],3),
                        stage="Converting audio" if job["mode"] == "convert" else "Loading models")
        directory = self.root / "jobs" / job_id
        try:
            with self.model_lock:
                self.check_cancel(job_id)
                options = job["settings"]
                if job["mode"] == "convert":
                    tracks = []
                else:
                    audio, rate = sf.read(directory / "input.wav", dtype="float32")
                if job["mode"] in {"separate", "separate_enhance"}:
                    separator = self.load("separate")
                    self.check_cancel(job_id)
                    self.update(job_id, stage="Splitting speakers")
                    sources, source_rate = separator.separate(audio, rate, chunk_seconds=options["chunk_seconds"], overlap_seconds=options["overlap_seconds"], cancel_check=lambda:self.check_cancel(job_id))
                    tracks = [("speaker_1", sources[0], source_rate), ("speaker_2", sources[1], source_rate)]
                elif job["mode"] != "convert":
                    tracks = [("enhanced", audio, rate)]
                if job["mode"] in {"enhance", "separate_enhance"}:
                    restorer = self.load("enhance")
                    self.check_cancel(job_id)
                    restored = []
                    for name, data, source_rate in tracks:
                        self.check_cancel(job_id)
                        self.update(job_id, stage=f"Enhancing {name.replace('_', ' ')}")
                        data, source_rate = restorer.enhance(data, source_rate, strength=options["strength"], cancel_check=lambda:self.check_cancel(job_id))
                        restored.append((name, data, source_rate))
                    tracks = restored
                outputs = []
                self.check_cancel(job_id)
                if job["mode"] == "convert":
                    track = encode_file(directory / "input.wav", directory / f"converted.{options['output_format']}", options['output_format'])
                    outputs.append(dict(name="converted", **track))
                for name, data, source_rate in tracks:
                    self.check_cancel(job_id)
                    filename = f"{name}.{options['output_format']}"
                    track = encode_audio(data, source_rate, directory / filename, options['output_format'])
                    outputs.append(dict(name=name, **track))
                used = [] if job["mode"] == "convert" else ["enhance"] if job["mode"] == "enhance" else ["separate"] if job["mode"] == "separate" else ["separate", "enhance"]
                backends = {name: getattr(self.models[name], "backend", "torch-fp32") for name in used}
                self.check_cancel(job_id)
                self.update(job_id, stage="Packaging results", outputs=outputs, backends=backends,
                            runtime_version=__version__, elapsed_seconds=round(time.monotonic() - started, 3))
                # An archive is finalized before the job becomes downloadable.
                with zipfile.ZipFile(directory / "results.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                    for output in outputs:
                        self.check_cancel(job_id)
                        archive.write(directory / output["file"], output["file"])
                    with self.state_lock:
                        self.check_cancel(job_id)
                        self.update(job_id, **self.timings(job_id))
                        archive.writestr("run.json", json.dumps({**self.jobs[job_id], "status": "done", "stage": "Complete"}, indent=2))
                with self.state_lock:
                    self.check_cancel(job_id)
                    self.clocks.pop(job_id, None)
                    self.update(job_id, status="done", stage="Complete")
        except JobCancelled:
            with self.state_lock:
                for path in directory.iterdir():
                    if path.name not in {'input.wav', 'job.json'} and path.is_file():
                        path.unlink(missing_ok=True)
                self.finish(job_id, status='cancelled', stage='Cancelled', outputs=[])
        except Exception:
            LOG.exception("Audio job %s failed", job_id)
            error = "Audio conversion failed. Try WAV/FLAC, or check FFmpeg and codec support in the service logs." if job["mode"] == "convert" else "Audio processing failed. Check model readiness and codec support in the service logs, then retry with a shorter clip."
            self.finish(job_id, status="failed", stage="Failed", error=error)


def create_app(data_dir: Path | None = None) -> FastAPI:
    runtime: Runtime | None = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime
        runtime = Runtime(data_dir or Path(os.environ.get("SIDON_DATA_DIR", Path.home() / ".local/share/sidon")))
        app.state.runtime = runtime

        async def expire():
            while True:
                await asyncio.sleep(60)
                await asyncio.to_thread(runtime.cleanup)

        task = asyncio.create_task(expire())
        yield
        task.cancel()
        await asyncio.to_thread(runtime.close)

    app = FastAPI(title="Sidon Studio API", version=__version__, lifespan=lifespan,
                  swagger_ui_parameters={"persistAuthorization": False}, docs_url=None, redoc_url=None)
    security = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        expected = os.environ.get("SIDON_API_KEY")
        if expected and (not credentials or not hmac.compare_digest(credentials.credentials.encode(), expected.encode())):
            raise HTTPException(401, "A valid bearer API key is required", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        if getattr(request.state, 'body_too_large', False):
            return api_error(request, 413, 'Request exceeds the upload limit')
        return api_error(request, 400 if exc.status_code == 422 and request.url.path.startswith('/v1/') else exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if not request.url.path.startswith('/v1/'):
            from fastapi.exception_handlers import request_validation_exception_handler
            return await request_validation_exception_handler(request, exc)
        first = exc.errors()[0]
        param = '.'.join(str(x) for x in first.get('loc', ())[1:]) or None
        return api_error(request, 400, first['msg'], code='invalid_parameter', param=param)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        LOG.error('Request %s failed', getattr(request.state, 'request_id', 'unknown'), exc_info=exc)
        return api_error(request, 500, 'Internal server error')

    @app.middleware("http")
    async def request_policy(request, call_next):
        request.state.request_id = str(uuid.uuid4())
        protected = request.url.path.startswith(('/api/', '/v1/'))
        reservation = None
        if protected:
            expected = os.environ.get('SIDON_API_KEY')
            scheme, _, token = request.headers.get('authorization', '').partition(' ')
            if expected and (scheme.lower() != 'bearer' or not hmac.compare_digest(token.encode(), expected.encode())):
                return api_error(request, 401, 'A valid bearer API key is required', headers={'WWW-Authenticate':'Bearer'})
            if request.method in {'POST','PUT','DELETE'}:
                origin = request.headers.get('origin')
                if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
                    return api_error(request, 403, 'Cross-origin changes are not allowed')
            maximum = runtime.max_bytes + 1024 * 1024
            try:
                declared = int(request.headers.get('content-length', '0'))
                if declared < 0:
                    raise ValueError()
                if declared > maximum:
                    return api_error(request, 413, 'Request exceeds the upload limit')
            except ValueError:
                return api_error(request, 400, 'Invalid Content-Length')
            submissions = {'/api/jobs','/v1/audio/jobs','/v1/audio/enhancements',
                           '/v1/audio/separations','/v1/audio/conversions'}
            if request.method == 'POST' and request.url.path.rstrip('/') in submissions:
                try:
                    reservation = runtime.reserve()
                    request.state.reservation = reservation
                except HTTPException as exc:
                    return api_error(request, exc.status_code, str(exc.detail), headers=exc.headers)
            original_receive = request._receive
            received = 0

            async def limited_receive():
                nonlocal received
                event = await original_receive()
                received += len(event.get('body', b''))
                if received > maximum:
                    request.state.body_too_large = True
                    raise HTTPException(413, 'Request exceeds the upload limit')
                return event

            request._receive = limited_receive
        try:
            response = await call_next(request)
        finally:
            if reservation:
                with runtime.state_lock:
                    runtime.admissions.pop(reservation, None)
        response.headers['X-Request-ID'] = request.state.request_id
        if protected:
            response.headers['Cache-Control'] = 'no-store'
        return response

    dependencies = [Depends(authenticate)]

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/status", dependencies=dependencies)
    def status():
        return dict(version=__version__, device=("cuda:0" if torch.cuda.is_available() else "cpu") if runtime.device == "auto" else runtime.device,
                    models=dict(runtime.model_states), model_errors=dict(runtime.model_errors), busy=runtime.model_lock.locked(),
                    backends={name: getattr(model, "backend", "torch-fp32") for name, model in dict(runtime.models).items()},
                    queue=runtime.queue_state(),
                    limits=dict(active_jobs=runtime.capacity, upload_mb=runtime.max_bytes // 1024**2, audio_seconds=runtime.max_seconds, retention_hours=runtime.retention / 3600),
                    settings=runtime.settings.model_dump(), modes=["enhance", "separate", "separate_enhance", "convert"])

    @app.put("/api/settings", dependencies=dependencies)
    def settings(value: Settings):
        if value.overlap_seconds >= value.chunk_seconds:
            raise HTTPException(422, "Overlap must be shorter than the separation window")
        with runtime.state_lock:
            write_json(runtime.root / "settings.json", value.model_dump())
            runtime.settings = value
        return value

    @app.post("/api/models/{name}", dependencies=dependencies)
    def model_action(name: Literal["enhance", "separate"], value: ModelAction):
        if not runtime.model_lock.acquire(blocking=False):
            raise HTTPException(409, "Models are busy processing audio")
        try:
            if value.action == "load":
                try:
                    runtime.load(name)
                except Exception:
                    LOG.exception("Model load failed")
                    raise HTTPException(503, runtime.model_errors[name])
            else:
                runtime.model_states[name] = "unloading"
                runtime.models.pop(name, None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                runtime.model_states[name] = "unloaded"
            return {"state": runtime.model_states[name]}
        finally:
            runtime.model_lock.release()

    @app.get("/api/gpu", dependencies=dependencies)
    def gpu():
        from fastapi.responses import JSONResponse
        return JSONResponse(runtime.gpu.request_snapshot(), headers={"Cache-Control": "no-store"})

    def options_for(request, mode, model, strength, output_format, response_format, chunk_seconds, overlap_seconds):
        expected = 'sidon' if mode == 'enhance' else 'audio-converter' if mode == 'convert' else 'mossformer2'
        if model and model != expected:
            raise HTTPException(400, f"Use model '{expected}' for this operation")
        if response_format and output_format and response_format != output_format:
            raise HTTPException(422, 'response_format and output_format must agree')
        with runtime.state_lock:
            base = Settings() if request.url.path.startswith('/v1/') else runtime.settings
            values = base.model_dump()
        values.update({key:value for key,value in dict(strength=strength, output_format=response_format or output_format,
                      chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds).items() if value is not None})
        try:
            return Settings.model_validate(values)
        except ValueError:
            raise HTTPException(422, 'Invalid processing settings; overlap must be shorter than the separation window')

    async def submit_audio(request, file, mode, options):
        job_id = request.state.reservation
        with runtime.state_lock:
            admission = runtime.admissions.pop(job_id)
            runtime.clocks[job_id] = admission
            job = dict(id=job_id, status='uploading', stage='Receiving audio', created=admission['created'], mode=mode,
                       filename=Path(file.filename or 'audio').name[:180], settings=options.model_dump(), outputs=[],
                       cancel_requested=False, request_id=request.state.request_id)
            runtime.jobs[job_id] = job
            runtime.active_ids.add(job_id)
        directory = runtime.root / 'jobs' / job_id
        try:
            directory.mkdir()
            size = 0
            with (directory / 'upload').open('wb') as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > runtime.max_bytes:
                        raise HTTPException(413, 'Audio exceeds the upload limit')
                    stream.write(chunk)
            duration = await finish_in_thread(decode_input, directory / 'upload', directory / 'input.wav',
                                              runtime.max_seconds, preserve_channels=mode == 'convert')
            (directory / 'upload').unlink()
            with runtime.state_lock:
                runtime.clocks[job_id]['ready'] = time.monotonic()
                runtime.update(job_id, status='queued', stage='Queued', duration=duration, queued_at=time.time())
                runtime.pool.submit(runtime.process, job_id)
            return runtime.snapshot(job_id)
        except BaseException as exc:
            with runtime.state_lock:
                runtime.jobs.pop(job_id, None)
                runtime.active_ids.discard(job_id)
                runtime.clocks.pop(job_id, None)
                shutil.rmtree(directory, ignore_errors=True)
            if not isinstance(exc, Exception) or isinstance(exc, HTTPException):
                raise
            LOG.info('Rejected audio upload', exc_info=True)
            raise HTTPException(422, 'Could not decode this recording. Use WAV, FLAC, Opus, MP3, AAC/M4A or OGG; native installs need FFmpeg for additional formats.') from exc
        finally:
            await file.close()

    @app.post('/api/jobs', status_code=202, dependencies=dependencies)
    @app.post('/v1/audio/jobs', status_code=202, dependencies=dependencies, response_model=AudioJob, responses=PUBLIC_ERRORS)
    async def submit(request: Request, response: Response, file: UploadFile = File(...),
                     mode: Literal['enhance','separate','separate_enhance','convert'] = Form('enhance'),
                     strength: float | None = Form(None, ge=0, le=1, allow_inf_nan=False),
                     output_format: OutputFormat | None = Form(None), response_format: OutputFormat | None = Form(None),
                     model: str | None = Form(None), chunk_seconds: float | None = Form(None, ge=2, le=16, allow_inf_nan=False),
                     overlap_seconds: float | None = Form(None, ge=.1, le=2, allow_inf_nan=False), stream: bool = Form(False)):
        if stream:
            raise HTTPException(422, 'Streaming is not supported. Submit complete chunks as independent jobs.')
        options = options_for(request, mode, model, strength, output_format, response_format, chunk_seconds, overlap_seconds)
        job = await submit_audio(request, file, mode, options)
        prefix = '/v1/audio/jobs' if request.url.path.startswith('/v1/') else '/api/jobs'
        response.headers['Location'] = f"{prefix}/{job['id']}"
        response.headers['X-Job-ID'] = job['id']
        return public_job(job) if request.url.path.startswith('/v1/') else job

    def public_job(job):
        base = f"/v1/audio/jobs/{job['id']}"
        outputs = [{**track, 'url':f"{base}/files/{track['file']}"} for track in job['outputs']]
        return {**job, 'object':'audio.job', 'outputs':outputs, 'url':base,
                'result_url': f"{base}/files/{outputs[0]['file']}" if job['status'] == 'done' and len(outputs) == 1
                              else f'{base}/files/results.zip' if job['status'] == 'done' else None}

    @app.get('/v1/models', dependencies=dependencies, responses=PUBLIC_ERRORS)
    def models():
        return dict(object='list', data=list(MODELS.values()))

    @app.get('/v1/models/{model_id}', dependencies=dependencies, responses=PUBLIC_ERRORS)
    def model_info(model_id: str):
        if model_id not in MODELS:
            raise HTTPException(404, 'Model not found')
        return MODELS[model_id]

    @app.get('/v1/audio/capabilities', dependencies=dependencies, responses=PUBLIC_ERRORS)
    def capabilities():
        return dict(object='audio.capabilities', streaming=False, response_modes=['audio','job'],
                    formats=['wav','flac','opus','mp3','m4a','aac','ogg'], models=list(MODELS),
                    queue=runtime.queue_state(), api_style='native_audio_extensions')

    @app.get('/v1/audio/queue', dependencies=dependencies, responses=PUBLIC_ERRORS)
    def queue_status():
        return runtime.queue_state()

    @app.post('/v1/audio/enhancements', dependencies=dependencies, response_class=FileResponse, responses=AUDIO_RESPONSES)
    @app.post('/v1/audio/separations', dependencies=dependencies, response_class=FileResponse, responses=AUDIO_RESPONSES)
    @app.post('/v1/audio/conversions', dependencies=dependencies, response_class=FileResponse, responses=AUDIO_RESPONSES)
    async def transform(request: Request, file: UploadFile = File(...), model: str | None = Form(None),
                        response_format: OutputFormat = Form('wav'),
                        response_mode: Literal['audio','job'] = Form('audio'), enhance: bool = Form(False),
                        strength: float = Form(1, ge=0, le=1, allow_inf_nan=False),
                        chunk_seconds: float = Form(4, ge=2, le=16, allow_inf_nan=False),
                        overlap_seconds: float = Form(1, ge=.1, le=2, allow_inf_nan=False), stream: bool = Form(False)):
        from fastapi.responses import JSONResponse
        if stream:
            raise HTTPException(422, 'Streaming is not supported. Submit complete chunks as independent requests.')
        mode = {'enhancements':'enhance','separations':'separate','conversions':'convert'}[request.url.path.rsplit('/',1)[1]]
        if mode == 'separate' and enhance:
            mode = 'separate_enhance'
        options = options_for(request, mode, model, strength, None, response_format, chunk_seconds, overlap_seconds)
        job = await submit_audio(request, file, mode, options)
        job_id = job['id']
        headers = {'X-Job-ID':job_id, 'Location':f'/v1/audio/jobs/{job_id}'}
        if response_mode == 'job':
            return JSONResponse(public_job(job), status_code=202, headers=headers)
        try:
            while job['status'] not in TERMINAL:
                if await request.is_disconnected():
                    runtime.cancel(job_id)
                    raise HTTPException(499, 'Client disconnected; cancellation requested')
                await asyncio.sleep(.025)
                job = runtime.snapshot(job_id)
        except asyncio.CancelledError:
            runtime.cancel(job_id)
            raise
        if job['status'] == 'cancelled':
            return api_error(request, 409, 'Audio job was cancelled', code='job_cancelled', headers=headers)
        if job['status'] == 'failed':
            return api_error(request, 500, job['error'], code='processing_failed', headers=headers)
        filename = job['outputs'][0]['file'] if len(job['outputs']) == 1 else 'results.zip'
        return FileResponse(runtime.root / 'jobs' / job_id / filename, filename=filename, headers=headers)

    def find(job_id: uuid.UUID):
        with runtime.state_lock:
            if str(job_id) not in runtime.jobs:
                raise HTTPException(404, "Job not found or expired")
            return runtime.snapshot(str(job_id))

    @app.get('/v1/audio/jobs', dependencies=dependencies, response_model=JobList, responses=PUBLIC_ERRORS)
    @app.get("/api/jobs", dependencies=dependencies)
    def jobs(request: Request):
        with runtime.state_lock:
            rows = [runtime.snapshot(j['id']) for j in sorted(runtime.jobs.values(), key=lambda j: j['created'], reverse=True) if j['status'] != 'uploading']
            return dict(object='list', data=[public_job(j) for j in rows]) if request.url.path.startswith('/v1/') else rows

    @app.get('/v1/audio/jobs/{job_id}', dependencies=dependencies, response_model=AudioJob, responses=PUBLIC_ERRORS)
    @app.get("/api/jobs/{job_id}", dependencies=dependencies)
    def job(request: Request, job_id: uuid.UUID):
        found = find(job_id)
        return public_job(found) if request.url.path.startswith('/v1/') else found

    @app.post('/v1/audio/jobs/{job_id}/cancel', dependencies=dependencies, response_model=AudioJob, responses=PUBLIC_ERRORS)
    @app.post('/api/jobs/{job_id}/cancel', dependencies=dependencies)
    def cancel_job(request: Request, job_id: uuid.UUID):
        found = runtime.cancel(str(job_id))
        return public_job(found) if request.url.path.startswith('/v1/') else found

    @app.delete('/v1/audio/jobs/{job_id}', dependencies=dependencies)
    @app.delete("/api/jobs/{job_id}", dependencies=dependencies)
    def delete_job(job_id: uuid.UUID):
        with runtime.state_lock:
            item = find(job_id)
            if item["status"] not in TERMINAL:
                raise HTTPException(409, "Wait for processing to finish before deleting this job")
            shutil.rmtree(runtime.root / "jobs" / str(job_id))
            del runtime.jobs[str(job_id)]
        return {"deleted": True}

    @app.get('/v1/audio/jobs/{job_id}/files/{filename}', dependencies=dependencies)
    @app.get("/api/jobs/{job_id}/files/{filename}", dependencies=dependencies)
    def download(job_id: uuid.UUID, filename: str):
        item = find(job_id)
        allowed = {"input.wav"} | ({"results.zip", "job.json"} | {o["file"] for o in item["outputs"]} if item["status"] == "done" else set())
        if filename not in allowed:
            raise HTTPException(404, "File not found")
        return FileResponse(runtime.root / "jobs" / str(job_id) / filename, filename=filename, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("SIDON_HOST", "127.0.0.1"), port=int(os.environ.get("SIDON_PORT", "7868")))


if __name__ == "__main__":
    main()
