# SPDX-License-Identifier: Apache-2.0
"""Owned job state, model residency and serial audio processing."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
import zipfile

from fastapi import HTTPException
import soundfile as sf

from sidon import __version__
from .audio import encode_audio, encode_file
from .gpu import GpuMonitor
from .models import Settings, StoredJob, TERMINAL
from .queue import InstanceLock, JobQueue

LOG = logging.getLogger('sidon')


class JobCancelled(Exception):
    pass


def write_json(path: Path, value: dict):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
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
        try:
            self.recover()
            self.cleanup()
            self.gpu = GpuMonitor()
            self.pool = JobQueue()
        except BaseException:
            self.lease.close()
            raise

    def recover(self):
        """Ignore malformed records without breaking startup or the public job list."""
        root = self.root
        for path in (root / "jobs").glob("*/job.json"):
            try:
                job = StoredJob.model_validate_json(path.read_text(encoding='utf-8')).model_dump(mode='json')
                if job["id"] != path.parent.name:
                    continue
                if job["status"] in {"uploading", "queued", "running"}:
                    job.update(status="failed", stage="Failed", error="The service restarted. Submit the audio again.")
                    write_json(path, job)
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                LOG.warning("Ignoring invalid job metadata for %s", path.parent.name)

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
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                self.gpu.close()
            finally:
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


