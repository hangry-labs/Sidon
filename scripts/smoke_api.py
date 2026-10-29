# SPDX-License-Identifier: Apache-2.0
"""Exercise real models via a running service; provide your own licensed sample."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import httpx
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7868")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["enhance", "separate", "separate_enhance"])
    parser.add_argument("--api-key")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    records = []
    with httpx.Client(base_url=args.url, headers=headers, timeout=180) as client:
        for _ in range(60):
            try:
                client.get("/healthz").raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        for mode in args.modes:
            response = client.post("/api/jobs", files={"file": (args.input.name, args.input.read_bytes())}, data={"mode": mode})
            response.raise_for_status()
            job_id = response.json()["id"]
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                response = client.get(f"/api/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                if job["status"] in {"done", "failed"}:
                    break
                time.sleep(.5)
            assert job["status"] == "done", job
            assert len(job["outputs"]) == (1 if mode == "enhance" else 2)
            expected_rate = 16000 if mode == "separate" else 48000
            for track in job["outputs"]:
                response = client.get(f"/api/jobs/{job_id}/files/{track['file']}")
                response.raise_for_status()
                audio, rate = sf.read(io.BytesIO(response.content))
                assert rate == expected_rate and audio.ndim == 1 and np.isfinite(audio).all()
                assert abs(len(audio) - round(job["duration"] * rate)) <= 3
                input_audio, _ = sf.read(args.input)
                # Catch arbitrary separator gain without imposing a peak limiter.
                assert np.sqrt(np.mean(audio ** 2)) <= max(1e-6, np.sqrt(np.mean(input_audio ** 2))) * 4
            response = client.get(f"/api/jobs/{job_id}/files/results.zip")
            response.raise_for_status()
            archive = zipfile.ZipFile(io.BytesIO(response.content))
            assert json.loads(archive.read("run.json"))["id"] == job_id
            records.append(job)
            print(json.dumps({"mode": mode, "status": job["status"], "elapsed_seconds": job["elapsed_seconds"], "outputs": job["outputs"]}), flush=True)
        state = client.get("/api/status").json()
        settings = {**state["settings"], "strength": .75}
        client.put("/api/settings", json=settings).raise_for_status()
        assert client.get("/api/status").json()["settings"]["strength"] == .75
        report = {"status": state, "jobs": records}
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
