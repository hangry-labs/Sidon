# SPDX-License-Identifier: Apache-2.0
"""One FIFO worker with removable pending work, rather than cancelled futures."""
from collections import deque
import asyncio
import logging
import threading
import os


async def finish_in_thread(function, *args, **kwargs):
    """A disconnected upload must not release admission while decoding continues."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Consume decoder errors; cancellation remains primary.
        raise


class InstanceLock:
    """Prevent two server processes from mutating the same job directory."""
    def __init__(self, root):
        self.file = (root / 'server.lock').open('a+b')
        try:
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b'0')
                self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError('Another Sidon server owns this data directory. Run one server worker per data directory and GPU allocation.') from exc

    def close(self):
        if self.file.closed:
            return
        self.file.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


class JobQueue:
    def __init__(self):
        self.pending = deque()
        self.condition = threading.Condition()
        self.closed = False
        self.worker = threading.Thread(target=self._run, name='sidon-audio', daemon=True)
        self.worker.start()

    def submit(self, function, job_id):
        with self.condition:
            if self.closed:
                raise RuntimeError('The processing queue is shutting down')
            self.pending.append((job_id, function))
            self.condition.notify()

    def cancel(self, job_id):
        with self.condition:
            self.pending = deque(item for item in self.pending if item[0] != job_id)

    def positions(self):
        with self.condition:
            return {item[0]: index + 1 for index, item in enumerate(self.pending)}

    def shutdown(self, wait=True, cancel_futures=False):
        with self.condition:
            self.closed = True
            if cancel_futures:
                self.pending.clear()
            self.condition.notify_all()
        if wait:
            self.worker.join()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closed)
                if not self.pending:
                    return
                job_id, function = self.pending.popleft()
            try:
                function(job_id)
            except Exception:
                logging.getLogger('sidon').exception('Unexpected queue failure for job %s', job_id)
