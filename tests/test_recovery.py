# SPDX-License-Identifier: Apache-2.0
"""Persisted state must not turn a single damaged job into a service outage."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import pytest

from sidon.web.app import create_app
from sidon.web.models import Settings
from sidon.web.runtime import Runtime
from sidon.web.queue import InstanceLock


def record(job_id):
    return dict(id=job_id, status='done', mode='convert', created=time.time(),
                stage='Complete', filename='voice.wav', settings=Settings().model_dump(), outputs=[])


def test_damaged_jobs_are_isolated_and_valid_history_remains_available(tmp_path):
    good_id = str(uuid.uuid4())
    records = [record(good_id), [], {'id':'broken'},
               {**record(str(uuid.uuid4())), 'created':'yesterday'},
               {**record(str(uuid.uuid4())), 'created':float('nan')},
               {**record(str(uuid.uuid4())), 'status':'unknown'},
               {**record(str(uuid.uuid4())), 'settings':{'strength':2}},
               {**record(str(uuid.uuid4())), 'outputs':[dict(name='audio', file='../private.wav', sample_rate=16000, duration=1)]}]
    for value in records:
        job_id = value['id'] if isinstance(value, dict) and value.get('id') != 'broken' else str(uuid.uuid4())
        directory = tmp_path/'jobs'/job_id
        directory.mkdir(parents=True)
        (directory/'job.json').write_text(json.dumps(value), encoding='utf-8')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/healthz').status_code == 200
        listed = client.get('/v1/audio/jobs')
        assert listed.status_code == 200, listed.text
        assert [job['id'] for job in listed.json()['data']] == [good_id]
        assert client.get('/v1/audio/queue').json()['active'] == 0


def test_startup_failure_releases_the_data_directory_lease(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        def fail(_):
            raise OSError('Simulated inaccessible job directory')
        patch.setattr(Runtime, 'cleanup', fail)
        with pytest.raises(OSError, match='Simulated'):
            Runtime(tmp_path)
    runtime = Runtime(tmp_path)
    runtime.close()


def test_lifespan_exception_still_shuts_down_the_worker_and_releases_ownership(tmp_path):
    app = create_app(tmp_path)
    async def scenario():
        with pytest.raises(RuntimeError, match='Simulated'):
            async with app.router.lifespan_context(app):
                raise RuntimeError('Simulated lifespan failure')
        assert not app.state.runtime.pool.worker.is_alive()
    asyncio.run(scenario())
    runtime = Runtime(tmp_path)
    runtime.close()


def test_shutdown_keeps_ownership_until_background_expiry_finishes(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original_cleanup = Runtime.cleanup
    calls = 0
    def cleanup(runtime):
        nonlocal calls
        calls += 1
        if calls > 1:  # Constructor cleanup is synchronous; expiry runs in a thread.
            started.set()
            assert release.wait(5), 'Test expiry gate timed out'
        original_cleanup(runtime)
    monkeypatch.setattr(Runtime, 'cleanup', cleanup)
    # Wake the periodic task immediately without replacing the test's asyncio clock.
    monkeypatch.setattr('sidon.web.app.asyncio', SimpleNamespace(
        create_task=asyncio.create_task, CancelledError=asyncio.CancelledError,
        sleep=lambda _:asyncio.sleep(0)))
    app = create_app(tmp_path)
    async def scenario():
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        shutdown = None
        try:
            assert await asyncio.to_thread(started.wait, 2)
            shutdown = asyncio.create_task(manager.__aexit__(None, None, None))
            await asyncio.sleep(.02)
            assert not shutdown.done()
            with pytest.raises(RuntimeError, match='Another Sidon server'):
                InstanceLock(tmp_path)
        finally:
            release.set()
            if shutdown is None:
                await manager.__aexit__(None, None, None)
            else:
                await shutdown
        assert not app.state.runtime.pool.worker.is_alive()
    asyncio.run(scenario())
    runtime = Runtime(tmp_path)
    runtime.close()
