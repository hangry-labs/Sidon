# SPDX-License-Identifier: Apache-2.0
"""Round-trip actual audio codecs through a running service; optionally run models."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7868')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--api-key')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    rows = []
    cases = [('convert', f) for f in ('wav','flac','opus','mp3','m4a','aac','ogg')]
    if args.inference:
        cases += [(m, 'opus') for m in ('enhance','separate','separate_enhance')]
    headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=180) as client:
        for attempt in range(60):
            try:
                client.get('/healthz').raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 59:
                    raise
                time.sleep(1)
        for mode, output_format in cases:
            result = client.post('/api/jobs', files={'file': (args.input.name, args.input.read_bytes())},
                                 data={'mode':mode, 'output_format':output_format})
            result.raise_for_status()
            job_id = result.json()['id']
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                job = client.get(f'/api/jobs/{job_id}').json()
                if job['status'] in {'done','failed'}:
                    break
                time.sleep(.2)
            assert job['status'] == 'done', job
            try:
                assert len(job['outputs']) == (2 if mode.startswith('separate') else 1)
                if mode == 'convert':
                    assert job['backends'] == {}
                for track in job['outputs']:
                    payload = client.get(f"/api/jobs/{job_id}/files/{track['file']}")
                    payload.raise_for_status()
                    with tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary)/track['file']
                        path.write_bytes(payload.content)
                        decoded = subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-'])
                        audio = np.frombuffer(decoded, dtype='<f4')
                        assert audio.size and np.isfinite(audio).all()
                        assert abs(audio.size / track['channels'] / track['sample_rate'] - track['duration']) < .08
                archive = zipfile.ZipFile(io.BytesIO(client.get(f'/api/jobs/{job_id}/files/results.zip').content))
                assert json.loads(archive.read('run.json'))['outputs'] == job['outputs']
                rows.append(job)
                print(json.dumps(dict(mode=mode, format=output_format, elapsed=job['elapsed_seconds'], backends=job['backends'])), flush=True)
            finally:
                client.delete(f'/api/jobs/{job_id}').raise_for_status()
        state = client.get('/api/status').json()
    if args.report:
        args.report.write_text(json.dumps(dict(status=state, jobs=rows), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
