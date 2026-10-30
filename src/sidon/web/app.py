# SPDX-License-Identifier: Apache-2.0
"""One-process local studio and bounded asynchronous audio API."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import gc
import hmac
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Literal
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
import torch

from sidon import __version__
from .audio import OutputFormat, decode_input
from .queue import finish_in_thread
from .models import Settings, TERMINAL
from .runtime import Runtime, write_json
from .protocol import MODELS, AudioJob, JobList, AUDIO_RESPONSES, PUBLIC_ERRORS, api_error

LOG = logging.getLogger("sidon")
STATIC = Path(__file__).parent / "static"


class ModelAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["load", "unload"]


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
                try:
                    await finish_in_thread(runtime.cleanup)
                except OSError:
                    LOG.exception('Could not expire retained jobs; retrying at the next cleanup interval')

        task = asyncio.create_task(expire())
        try:
            yield
        finally:
            task.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await task
            finally:
                await finish_in_thread(runtime.close)

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
