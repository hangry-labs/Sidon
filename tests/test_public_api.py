# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import json
import threading
import time
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from sidon.web.app import create_app


def audio(value=.1):
    stream = io.BytesIO()
    sf.write(stream, np.full(1600, value, dtype='float32'), 16000, format='WAV', subtype='FLOAT')
    return stream.getvalue()


def submit(client, **fields):
    return client.post('/v1/audio/jobs', files={'file':('voice.wav', audio())}, data=fields)


def wait(client, job_id):
    for _ in range(500):
        job = client.get('/v1/audio/jobs/'+job_id).json()
        if job['status'] in {'done','failed','cancelled'}:
            return job
        time.sleep(.01)
    pytest.fail('Job did not finish')


class GateRestorer:
    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
        self.calls = []
        self.active = self.peak = 0

    def enhance(self, data, rate, **kwargs):
        self.calls.append(float(data[0]))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if len(self.calls) == 1:
                self.started.set()
                assert self.release.wait(10), 'Test processing gate timed out'
            return np.repeat(data, 3), rate*3
        finally:
            self.active -= 1


def test_direct_response_model_catalog_and_independent_defaults(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        defaults = client.get('/api/status').json()['settings']
        client.put('/api/settings', json={**defaults, 'strength':.5, 'output_format':'flac'})
        models = client.get('/v1/models').json()
        assert models['object'] == 'list' and len(models['data']) == 3
        assert client.get('/v1/models/sidon').json()['object'] == 'model'
        assert client.get('/v1/audio/capabilities').json()['streaming'] is False
        response = client.post('/v1/audio/conversions', files={'file':('data.wav',audio())}, data={'model':'audio-converter'})
        assert response.status_code == 200, response.text
        assert response.headers['x-request-id'] and response.headers['x-job-id']
        assert response.headers['cache-control'] == 'no-store'
        values, rate = sf.read(io.BytesIO(response.content))
        assert rate == 16000 and len(values) == 1600
        job = client.get(response.headers['location']).json()
        assert job['object'] == 'audio.job' and job['settings']['output_format'] == 'wav'
        assert job['settings']['strength'] == 1
        assert job['outputs'][0]['url'] == job['result_url']
        assert job['total_seconds'] >= job['processing_seconds'] >= 0
        assert job['queue_wait_seconds'] >= 0
        assert app.state.runtime.models == {}
        schema = client.get('/openapi.json').json()
        assert 'audio/*' in schema['paths']['/v1/audio/conversions']['post']['responses']['200']['content']
        assert schema['paths']['/v1/audio/jobs']['post']['responses']['202']['content']['application/json']['schema']['$ref'].endswith('/AudioJob')


def test_versioned_errors_and_legacy_shapes_release_admission(tmp_path, monkeypatch):
    monkeypatch.setenv('SIDON_API_KEY', 'test-only')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.get('/v1/models')
        assert response.status_code == 401
        assert set(response.json()['error']) == {'message','type','param','code'}
        assert response.headers['x-request-id']
        assert client.get('/api/status').json().keys() == {'detail'}
        client.headers['Authorization'] = 'bearer test-only'
        assert client.get('/v1/models').status_code == 200
        assert client.get('/v1/audio/speech').status_code == 404
        assert client.get('/v1/models/unknown').status_code == 404
        for data in ({'stream':'true'}, {'response_format':'bad'}, {'model':'unknown'},
                     {'chunk_seconds':'2','overlap_seconds':'2'}, {'response_format':'wav','output_format':'flac'}):
            response = submit(client, mode='convert', **data)
            assert response.status_code == 400, response.text
            assert response.json()['error']['type'] == 'invalid_request_error'
            assert app.state.runtime.queue_state()['active'] == 0
        assert client.post('/v1/audio/jobs', data={}).status_code == 400
        assert app.state.runtime.admissions == {}
        assert client.post('/api/jobs', data={}).status_code == 422
        assert client.post('/v1/audio/jobs', headers={'Origin':'https://unrelated.invalid'}).status_code == 403
        # Non-ASCII raw headers must be rejected as credentials, not crash HMAC.
        assert client.get('/v1/models', headers={b'Authorization':b'Bearer \xff'}).status_code == 401


def test_fifo_cancellation_and_removable_pending_work(tmp_path):
    app = create_app(tmp_path)
    gate = GateRestorer()
    with TestClient(app) as client:
        app.state.runtime.models['enhance'] = gate
        try:
            first = submit(client).json()['id']
            assert gate.started.wait(2)
            second = submit(client).json()['id']
            third = submit(client).json()['id']
            assert client.get('/v1/audio/jobs/'+second).json()['queue_position'] == 1
            assert client.get('/v1/audio/jobs/'+third).json()['queue_position'] == 2
            cancelled = client.post('/v1/audio/jobs/'+second+'/cancel').json()
            assert cancelled['status'] == 'cancelled'
            assert client.delete('/v1/audio/jobs/'+second).status_code == 200
            assert client.get('/v1/audio/jobs/'+third).json()['queue_position'] == 1
            for _ in range(12):
                pending = submit(client).json()['id']
                assert client.post('/v1/audio/jobs/'+pending+'/cancel').json()['status'] == 'cancelled'
            assert len(app.state.runtime.pool.positions()) == 1
            time.sleep(.03)
        finally:
            gate.release.set()
        assert wait(client, first)['status'] == wait(client, third)['status'] == 'done'
        assert len(gate.calls) == 2 and gate.peak == 1
        assert wait(client, third)['queue_wait_seconds'] > .02


def test_parallel_admission_and_running_cancellation_keep_gpu_slot(tmp_path, monkeypatch):
    monkeypatch.setenv('SIDON_MAX_ACTIVE_JOBS','2')
    app = create_app(tmp_path)
    gate = GateRestorer()
    with TestClient(app) as client:
        app.state.runtime.models['enhance'] = gate
        try:
            first = submit(client).json()['id']
            assert gate.started.wait(2)
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda _:submit(client), range(8)))
            assert [r.status_code for r in responses].count(202) == 1
            assert [r.status_code for r in responses].count(429) == 7
            assert all(r.headers.get('retry-after') == '1' for r in responses if r.status_code == 429)
            # Missing multipart boundary would fail parsing if overload admission were late.
            rejected = client.post('/v1/audio/jobs', content=b'bad', headers={'Content-Type':'multipart/form-data'})
            assert rejected.status_code == 429 and rejected.json()['error']['code'] == 'queue_full'
            assert client.post('/api/models/enhance', json={'action':'unload'}).status_code == 409
            cancelling = client.post('/v1/audio/jobs/'+first+'/cancel').json()
            assert cancelling['cancel_requested'] and cancelling['status'] == 'running'
            assert app.state.runtime.queue_state()['active'] == 2
        finally:
            gate.release.set()
        assert wait(client, first)['status'] == 'cancelled'
        assert client.get('/v1/audio/jobs/'+first+'/files/results.zip').status_code == 404
        second = next(r.json()['id'] for r in responses if r.status_code == 202)
        assert wait(client, second)['status'] == 'done'
        assert gate.peak == 1


def test_async_transform_and_separated_archive(tmp_path):
    class Separator:
        def separate(self, data, rate, **kwargs):
            return np.stack([data*.6, data*.4]), rate
    app = create_app(tmp_path)
    with TestClient(app) as client:
        app.state.runtime.models['separate'] = Separator()
        response = client.post('/v1/audio/separations', files={'file':('voice.wav',audio())}, data={'response_mode':'job'})
        assert response.status_code == 202
        assert wait(client, response.json()['id'])['status'] == 'done'
        direct = client.post('/v1/audio/separations', files={'file':('voice.wav',audio())})
        assert direct.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(direct.content))
        assert len(json.loads(archive.read('run.json'))['outputs']) == 2


def test_data_directory_has_one_process_owner_and_releases_on_shutdown(tmp_path):
    with TestClient(create_app(tmp_path)) as first:
        with pytest.raises(RuntimeError, match='Another Sidon server'):
            with TestClient(create_app(tmp_path)):
                pass
        assert first.get('/healthz').status_code == 200
    with TestClient(create_app(tmp_path)) as next_server:
        assert next_server.get('/healthz').status_code == 200


def test_direct_waiting_request_observes_cooperative_cancellation(tmp_path):
    app = create_app(tmp_path)
    gate = GateRestorer()
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        app.state.runtime.models['enhance'] = gate
        pending = pool.submit(lambda:client.post('/v1/audio/enhancements', files={'file':('voice.wav',audio())}))
        try:
            assert gate.started.wait(2)
            job_id = client.get('/v1/audio/jobs').json()['data'][0]['id']
            client.post('/v1/audio/jobs/'+job_id+'/cancel')
        finally:
            gate.release.set()
        result = pending.result(timeout=5)
        assert result.status_code == 409
        assert result.json()['error']['code'] == 'job_cancelled'


def test_streamed_request_size_limit_releases_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv('SIDON_MAX_UPLOAD_MB','1')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        request = client.build_request('POST','/v1/audio/jobs', files={'file':('big.wav',b'x'*(2*1024**2+1))})
        request.headers.pop('content-length',None)
        response = client.send(request)
        assert response.status_code == 413, response.text
        assert response.json()['error']['code'] == 'payload_too_large'
        assert app.state.runtime.queue_state()['active'] == 0


def test_cancelled_upload_waits_for_its_decoder_before_cleanup():
    from sidon.web.queue import finish_in_thread
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def decode():
        started.set()
        assert release.wait(5)
        finished.set()
    async def scenario():
        task = asyncio.create_task(finish_in_thread(decode))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(.01)
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done() and not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(scenario())
