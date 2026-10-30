# SPDX-License-Identifier: Apache-2.0
"""Local Opus previews must support duration probing and seeking."""
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts import serve_pages


def test_preview_byte_ranges_head_and_project_prefix(tmp_path, monkeypatch):
    content = bytes(range(64))
    (tmp_path / 'clip.opus').write_bytes(content)
    monkeypatch.setattr(serve_pages, 'ROOT', tmp_path)

    class QuietHandler(serve_pages.Handler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for requested, expected, span in [('bytes=0-31', content[:32], '0-31'),
                                          ('bytes=-12', content[-12:], '52-63')]:
            with urlopen(Request(base + '/Sidon/clip.opus', headers={'Range': requested}), timeout=5) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {span}/64'
                assert response.read() == expected
        with urlopen(Request(base + '/Sidon/clip.opus', method='HEAD'), timeout=5) as response:
            assert response.status == 200
            assert response.headers['Content-Length'] == '64'
            assert response.read() == b''
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(base + '/Sidon/clip.opus', headers={'Range': 'bytes=64-'}), timeout=5)
        assert invalid.value.code == 416
        with pytest.raises(HTTPError) as outside:
            urlopen(base + '/clip.opus', timeout=5)
        assert outside.value.code == 404
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
