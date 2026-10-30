# SPDX-License-Identifier: MIT
# Adapted from faster-qwen3-tts-studio; see licenses/faster-qwen-upstream-MIT.txt.
"""Serve the generated website with its production project-path prefix."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / '.ai/tmp/pages/output'
PREFIX = '/Sidon'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def route_project(self):
        if self.path == '/' or self.path == PREFIX:
            self.send_response(302)
            self.send_header('Location', PREFIX + '/')
            self.end_headers()
            return False
        if not urlsplit(self.path).path.startswith(PREFIX + '/'):
            self.send_error(404)
            return False
        self.path = self.path[len(PREFIX):]
        return True

    def do_GET(self):
        if self.route_project():
            super().do_GET()

    def do_HEAD(self):
        if self.route_project():
            super().do_HEAD()

    def send_head(self):
        # Browsers need byte ranges to read Ogg's final granule and seek in Opus.
        self.range_remaining = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get('Range', '')
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        if not match or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last or '0'))
        end = min(int(last), size - 1) if first and last else size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream = path.open('rb')
        stream.seek(start)
        self.range_remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.range_remaining))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.range_remaining is None:
            return super().copyfile(source, outputfile)
        while self.range_remaining:
            chunk = source.read(min(65536, self.range_remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self.range_remaining -= len(chunk)


if __name__ == '__main__':
    print('Serving http://127.0.0.1:9005' + PREFIX + '/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 9005), Handler).serve_forever()
