# SPDX-License-Identifier: Apache-2.0
"""Exercise real versioned responses through a running Sidon instance."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7868')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--api-key')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    rows = []
    headers = {'Authorization':f'Bearer {args.api_key}'} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=180) as client:
        for attempt in range(60):
            try:
                client.get('/healthz').raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 59:
                    raise
                time.sleep(1)
        models = client.get('/v1/models')
        models.raise_for_status()
        assert models.json()['object'] == 'list'
        cases = [('conversions', {})]
        if args.inference:
            cases += [('enhancements', {}), ('separations', {}), ('separations', {'enhance':'true'})]
        for operation, extra in cases:
            response = client.post('/v1/audio/'+operation, files={'file':(args.input.name,args.input.read_bytes())},
                                   data={'response_format':'opus',**extra})
            response.raise_for_status()
            assert response.status_code == 200 and response.headers['x-request-id']
            job_id = response.headers['x-job-id']
            try:
                job = client.get('/v1/audio/jobs/'+job_id).json()
                assert job['object'] == 'audio.job' and job['status'] == 'done', job
                if operation == 'separations':
                    archive = zipfile.ZipFile(io.BytesIO(response.content))
                    assert len(json.loads(archive.read('run.json'))['outputs']) == 2
                else:
                    assert response.content.startswith(b'OggS')
                rows.append(dict(operation=operation, enhanced=bool(extra), job=job,
                                 request_id=response.headers['x-request-id'], bytes=len(response.content)))
                print(json.dumps(dict(operation=operation, enhanced=bool(extra), status='done', bytes=len(response.content))), flush=True)
            finally:
                client.delete('/v1/audio/jobs/'+job_id).raise_for_status()
        response = client.post('/v1/audio/conversions', files={'file':(args.input.name,args.input.read_bytes())},
                               data={'response_mode':'job'})
        response.raise_for_status()
        assert response.status_code == 202
        job_id = response.json()['id']
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            job = client.get('/v1/audio/jobs/'+job_id).json()
            if job['status'] in {'done','failed','cancelled'}:
                break
            time.sleep(.05)
        assert job['status'] == 'done', job
        client.delete('/v1/audio/jobs/'+job_id).raise_for_status()
        queue = client.get('/v1/audio/queue').json()
        assert queue['active'] == 0 and queue['available'] == queue['capacity'], queue
    if args.report:
        args.report.write_text(json.dumps(dict(models=models.json(), cases=rows, queue=queue), indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
