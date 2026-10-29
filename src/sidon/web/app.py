# SPDX-License-Identifier: Apache-2.0
"""One-process local studio and bounded asynchronous audio API."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
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

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import soundfile as sf
import torch

from sidon import __version__
from .gpu import GpuMonitor

LOG = logging.getLogger("sidon")
STATIC = Path(__file__).parent / "static"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    strength: float = Field(default=1, ge=0, le=1)
    output_format: Literal["wav", "flac"] = "wav"
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


def decode_input(source: Path, destination: Path, max_seconds: int) -> float:
    """Downmix in bounded blocks; compressed multichannel input cannot inflate RAM."""
    with sf.SoundFile(source) as stream:
        if stream.frames <= 0 or not 8000 <= stream.samplerate <= 192000 or not 1 <= stream.channels <= 8 or stream.frames / stream.samplerate > max_seconds:
            raise HTTPException(422, "Use a nonempty clip within the duration limit, 8–192 kHz, and at most 8 channels")
        total = 0
        with sf.SoundFile(destination, "w", samplerate=stream.samplerate, channels=1, subtype="FLOAT", format="WAV") as output:
            for block in stream.blocks(blocksize=65536, dtype="float32", always_2d=True):
                total += len(block)
                if total > max_seconds * stream.samplerate or not np.isfinite(block).all():
                    raise HTTPException(422, "Audio exceeds the duration limit or contains nonfinite samples")
                output.write(block.mean(axis=1))
        if not total:
            raise HTTPException(422, "The audio file is empty")
        return total / stream.samplerate


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        self.settings = Settings()
        if (root / "settings.json").exists():
            self.settings = Settings.model_validate_json((root / "settings.json").read_text())
        self.jobs: dict[str, dict] = {}
        self.state_lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.models: dict[str, object] = {}
        self.model_states = {"enhance": "unloaded", "separate": "unloaded"}
        self.model_errors: dict[str, str] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidon-audio")
        self.gpu = GpuMonitor()
        self.device = os.environ.get("SIDON_DEVICE", "auto")
        self.retention = int(os.environ.get("SIDON_RETENTION_HOURS", "24")) * 3600
        self.max_bytes = int(os.environ.get("SIDON_MAX_UPLOAD_MB", "100")) * 1024 * 1024
        self.max_seconds = int(os.environ.get("SIDON_MAX_AUDIO_SECONDS", "600"))
        if min(self.retention, self.max_bytes, self.max_seconds) <= 0:
            raise ValueError("Retention, upload and duration limits must be positive")
        for path in (root / "jobs").glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if str(uuid.UUID(job["id"])) != path.parent.name:
                    continue
                if job["status"] in {"queued", "running"}:
                    job.update(status="failed", error="The service restarted. Submit the audio again.")
                    write_json(path, job)
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                LOG.warning("Ignoring unreadable job metadata")
        self.cleanup()

    def cleanup(self):
        with self.state_lock:
            for job_id, job in list(self.jobs.items()):
                if job["status"] in {"done", "failed"} and time.time() - job["created"] > self.retention:
                    shutil.rmtree(self.root / "jobs" / job_id, ignore_errors=True)
                    del self.jobs[job_id]
            # Interrupted uploads never become queued jobs.
            for path in (self.root / "jobs").iterdir():
                if path.is_dir() and path.name not in self.jobs and time.time() - path.stat().st_mtime > 3600:
                    shutil.rmtree(path, ignore_errors=True)

    def update(self, job_id: str, **fields):
        with self.state_lock:
            self.jobs[job_id].update(fields)
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
        job = self.jobs[job_id]
        directory = self.root / "jobs" / job_id
        self.update(job_id, status="running", stage="Loading models")
        try:
            with self.model_lock:
                audio, rate = sf.read(directory / "input.wav", dtype="float32")
                options = job["settings"]
                if job["mode"] in {"separate", "separate_enhance"}:
                    separator = self.load("separate")
                    self.update(job_id, stage="Splitting speakers")
                    sources, source_rate = separator.separate(audio, rate, chunk_seconds=options["chunk_seconds"], overlap_seconds=options["overlap_seconds"])
                    tracks = [("speaker_1", sources[0], source_rate), ("speaker_2", sources[1], source_rate)]
                else:
                    tracks = [("enhanced", audio, rate)]
                if job["mode"] != "separate":
                    restorer = self.load("enhance")
                    restored = []
                    for name, data, source_rate in tracks:
                        self.update(job_id, stage=f"Enhancing {name.replace('_', ' ')}")
                        data, source_rate = restorer.enhance(data, source_rate, strength=options["strength"])
                        restored.append((name, data, source_rate))
                    tracks = restored
                outputs = []
                for name, data, source_rate in tracks:
                    filename = f"{name}.{options['output_format']}"
                    clipped = int(np.count_nonzero(np.abs(data) > 1))
                    sf.write(directory / filename, data, source_rate, subtype="FLOAT" if options["output_format"] == "wav" else "PCM_24")
                    outputs.append(dict(name=name, file=filename, sample_rate=source_rate, duration=len(data) / source_rate, samples_over_full_scale=clipped))
                used = ["enhance"] if job["mode"] == "enhance" else ["separate"] if job["mode"] == "separate" else ["separate", "enhance"]
                backends = {name: getattr(self.models[name], "backend", "torch-fp32") for name in used}
                self.update(job_id, stage="Packaging results", outputs=outputs, backends=backends,
                            runtime_version=__version__, elapsed_seconds=round(time.monotonic() - started, 3))
                # An archive is finalized before the job becomes downloadable.
                with zipfile.ZipFile(directory / "results.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("run.json", json.dumps({**self.jobs[job_id], "status": "done", "stage": "Complete"}, indent=2))
                    for output in outputs:
                        archive.write(directory / output["file"], output["file"])
                self.update(job_id, status="done", stage="Complete")
        except Exception:
            LOG.exception("Audio job %s failed", job_id)
            self.update(job_id, status="failed", stage="Failed", error="Audio processing failed. Check model readiness in System and the service logs, then retry with a shorter clip.")


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
        runtime.pool.shutdown(wait=True, cancel_futures=True)
        runtime.gpu.close()

    app = FastAPI(title="Sidon Studio API", version=__version__, lifespan=lifespan,
                  swagger_ui_parameters={"persistAuthorization": False}, docs_url=None, redoc_url=None)
    security = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        expected = os.environ.get("SIDON_API_KEY")
        if expected and (not credentials or not hmac.compare_digest(credentials.credentials, expected)):
            raise HTTPException(401, "A valid bearer API key is required", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def request_policy(request, call_next):
        from fastapi.responses import JSONResponse
        if request.url.path.startswith("/api/"):
            expected = os.environ.get("SIDON_API_KEY")
            bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
            if expected and not hmac.compare_digest(bearer, expected):
                return JSONResponse({"detail": "A valid bearer API key is required"}, status_code=401)
            if request.method in {"POST", "PUT", "DELETE"}:
                origin = request.headers.get("origin")
                if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                    return JSONResponse({"detail": "Cross-origin changes are not allowed"}, status_code=403)
            maximum = runtime.max_bytes + 1024 * 1024
            try:
                if int(request.headers.get("content-length", "0")) > maximum:
                    return JSONResponse({"detail": "Request exceeds the upload limit"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            original_receive = request._receive
            received = 0

            async def limited_receive():
                nonlocal received
                event = await original_receive()
                received += len(event.get("body", b""))
                if received > maximum:
                    raise HTTPException(413, "Request exceeds the upload limit")
                return event

            request._receive = limited_receive
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
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
                    limits=dict(upload_mb=runtime.max_bytes // 1024**2, audio_seconds=runtime.max_seconds, retention_hours=runtime.retention / 3600),
                    settings=runtime.settings.model_dump(), modes=["enhance", "separate", "separate_enhance"])

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

    @app.post("/api/jobs", status_code=202, dependencies=dependencies)
    async def submit(file: UploadFile = File(...), mode: Literal["enhance", "separate", "separate_enhance"] = Form("enhance"),
                     strength: float | None = Form(None, ge=0, le=1, allow_inf_nan=False), output_format: Literal["wav", "flac"] | None = Form(None)):
        with runtime.state_lock:
            if sum(j["status"] in {"uploading", "queued", "running"} for j in runtime.jobs.values()) >= 4:
                raise HTTPException(429, "The audio queue is full. Retry when a job finishes.")
            job_id = str(uuid.uuid4())
            options = runtime.settings.model_copy(update={k: v for k, v in dict(strength=strength, output_format=output_format).items() if v is not None})
            options = Settings.model_validate(options.model_dump())
            job = dict(id=job_id, status="uploading", stage="Receiving audio", created=time.time(), mode=mode,
                       filename=Path(file.filename or "audio").name[:180], settings=options.model_dump(), outputs=[])
            runtime.jobs[job_id] = job
        directory = runtime.root / "jobs" / job_id
        directory.mkdir()
        try:
            size = 0
            with (directory / "upload").open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > runtime.max_bytes:
                        raise HTTPException(413, "Audio exceeds the upload limit")
                    stream.write(chunk)
            duration = await asyncio.to_thread(decode_input, directory / "upload", directory / "input.wav", runtime.max_seconds)
            (directory / "upload").unlink()
            runtime.update(job_id, status="queued", stage="Queued", duration=duration)
            runtime.pool.submit(runtime.process, job_id)
            return dict(job)
        except HTTPException:
            runtime.jobs.pop(job_id, None)
            shutil.rmtree(directory, ignore_errors=True)
            raise
        except Exception:
            runtime.jobs.pop(job_id, None)
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(422, "Could not decode this file. Upload WAV, FLAC, OGG or another libsndfile-supported audio file.")
        finally:
            await file.close()

    def find(job_id: uuid.UUID):
        with runtime.state_lock:
            if str(job_id) not in runtime.jobs:
                raise HTTPException(404, "Job not found or expired")
            return dict(runtime.jobs[str(job_id)])

    @app.get("/api/jobs", dependencies=dependencies)
    def jobs():
        with runtime.state_lock:
            return [dict(j) for j in sorted(runtime.jobs.values(), key=lambda j: j["created"], reverse=True) if j["status"] != "uploading"]

    @app.get("/api/jobs/{job_id}", dependencies=dependencies)
    def job(job_id: uuid.UUID):
        return find(job_id)

    @app.delete("/api/jobs/{job_id}", dependencies=dependencies)
    def delete_job(job_id: uuid.UUID):
        with runtime.state_lock:
            item = find(job_id)
            if item["status"] not in {"done", "failed"}:
                raise HTTPException(409, "Wait for processing to finish before deleting this job")
            shutil.rmtree(runtime.root / "jobs" / str(job_id))
            del runtime.jobs[str(job_id)]
        return {"deleted": True}

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
