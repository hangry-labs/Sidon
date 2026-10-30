# SPDX-License-Identifier: Apache-2.0
import io
import time
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from sidon.web.app import create_app


def audio_file(rate=16000):
    stream = io.BytesIO()
    sf.write(stream, np.sin(np.arange(1600) * .1).astype(np.float32) * .1, rate, format="WAV")
    return stream.getvalue()


class Restorer:
    def enhance(self, data, rate, **kwargs):
        return np.repeat(data, 3), rate * 3


class Separator:
    def separate(self, data, rate, **kwargs):
        return np.stack([data * .6, data * .4]), rate


def wait_job(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(.01)
    pytest.fail("Job did not finish")


@pytest.mark.parametrize("mode,tracks,rate", [("enhance", 1, 48000), ("separate", 2, 16000), ("separate_enhance", 2, 48000)])
@pytest.mark.parametrize("output_format", ["wav", "flac", "opus"])
def test_processing_download_and_delete(tmp_path, mode, tracks, rate, output_format):
    if output_format == "opus":
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg required for Opus")
    app = create_app(tmp_path)
    with TestClient(app) as client:
        app.state.runtime.models.update(enhance=Restorer(), separate=Separator())
        response = client.post("/api/jobs", files={"file": ("voice.wav", audio_file())}, data={"mode": mode, "output_format": output_format})
        assert response.status_code == 202, response.text
        job = wait_job(client, response.json()["id"])
        assert job["status"] == "done", job
        assert set(job["backends"]) == ({"enhance"} if mode == "enhance" else {"separate"} if mode == "separate" else {"separate", "enhance"})
        assert len(job["outputs"]) == tracks
        for track in job["outputs"]:
            data, sr = sf.read(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/{track['file']}").content))
            assert sr == (48000 if output_format == "opus" else rate) and np.isfinite(data).all()
        archive = client.get(f"/api/jobs/{job['id']}/files/results.zip")
        assert len(zipfile.ZipFile(io.BytesIO(archive.content)).namelist()) == tracks + 1
        assert client.get(f"/api/jobs/{job['id']}/files/settings.json").status_code == 404
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert not (tmp_path / "jobs" / job["id"]).exists()


def test_authentication_and_cross_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("SIDON_API_KEY", "test-only-key")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/status").status_code == 401
        headers = {"Authorization": "Bearer test-only-key"}
        assert client.get("/api/status", headers=headers).status_code == 200
        assert client.post("/api/jobs", headers={**headers, "Origin": "https://unrelated.invalid"}).status_code == 403


def test_invalid_inputs_limits_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("SIDON_MAX_UPLOAD_MB", "1")
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/jobs", files={"file": ("bad.wav", b"bad")}).status_code == 422
        assert client.post("/api/jobs", files={"file": ("big.wav", b"0" * (2 * 1024**2 + 1))}).status_code == 413
        for strength in ("nan", "inf", "-1", "2"):
            assert client.post("/api/jobs", files={"file": ("voice.wav", audio_file())}, data={"strength": strength}).status_code == 422
        assert client.get("/api/jobs").json() == []
        settings = client.get("/api/status").json()["settings"]
        assert client.put("/api/settings", json={**settings, "chunk_seconds": 2, "overlap_seconds": 2}).status_code == 422
        assert client.put("/api/settings", json={**settings, "strength": .5}).status_code == 200
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/status").json()["settings"]["strength"] == .5


def test_expiry_and_restart_recovery(tmp_path):
    from sidon.web.models import Settings
    from sidon.web.runtime import Runtime, write_json
    import uuid
    job_id = str(uuid.uuid4()); directory = tmp_path / "jobs" / job_id; directory.mkdir(parents=True)
    write_json(directory / "job.json", {"id": job_id, "status": "running", "created": time.time(), "outputs": [],
               "mode": "enhance", "stage": "Enhancing audio", "filename": "voice.wav", "settings": Settings().model_dump()})
    runtime = Runtime(tmp_path)
    assert runtime.jobs[job_id]["status"] == "failed"
    runtime.jobs[job_id]["created"] = time.time() - runtime.retention - 1
    runtime.cleanup()
    assert not directory.exists() and not runtime.jobs
    runtime.close()


def test_queue_and_residency_are_bounded(tmp_path):
    import threading
    app = create_app(tmp_path)
    with TestClient(app) as client:
        runtime = app.state.runtime
        runtime.model_lock.acquire()
        try:
            assert client.post("/api/models/enhance", json={"action": "unload"}).status_code == 409
            for _ in range(4):
                assert client.post("/api/jobs", files={"file": ("voice.wav", audio_file())}).status_code == 202
            assert client.post("/api/jobs", files={"file": ("voice.wav", audio_file())}).status_code == 429
            for job_id in runtime.jobs:
                assert client.delete(f"/api/jobs/{job_id}").status_code == 409
            runtime.models["enhance"] = Restorer()
        finally:
            runtime.model_lock.release()
