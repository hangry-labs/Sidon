# SPDX-License-Identifier: Apache-2.0
"""Public API conventions; audio transforms remain native Sidon extensions."""
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal
from .models import StoredJob, StoredOutput


class AudioOutput(StoredOutput):
    url: str


class AudioJob(StoredJob):
    object: Literal['audio.job'] = 'audio.job'
    outputs: list[AudioOutput]
    url: str
    result_url: str | None = None
    queue_position: int | None = None


class JobList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[AudioJob]


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str


class APIError(BaseModel):
    error: ErrorDetail


PUBLIC_ERRORS = {status: {'model': APIError} for status in (400,401,403,404,409,413,429,500,503)}
PUBLIC_ERRORS['4XX'] = {'model':APIError, 'description':'Client error'}
AUDIO_RESPONSES = {
    **PUBLIC_ERRORS,
    200: {'description':'Audio file, or a ZIP containing both separated tracks and a manifest',
          'content': {kind:{'schema':{'type':'string','format':'binary'}} for kind in ('audio/*','application/zip')}},
    202: {'model': AudioJob, 'description':'Accepted for asynchronous processing'},
}

MODELS = {
    'sidon': dict(id='sidon', object='model', created=0, owned_by='sarulab-speech'),
    'mossformer2': dict(id='mossformer2', object='model', created=0, owned_by='alibabasglab'),
    'audio-converter': dict(id='audio-converter', object='model', created=0, owned_by='hangry-labs'),
}


def api_error(request, status, message, *, code=None, param=None, headers=None):
    headers = {**(headers or {}), 'Cache-Control': 'no-store'}
    request_id = getattr(request.state, 'request_id', None)
    if request_id:
        headers['X-Request-ID'] = request_id
    if request.url.path.startswith('/v1/'):
        kind = ('authentication_error' if status == 401 else 'permission_error' if status == 403
                else 'rate_limit_error' if status == 429 else 'server_error' if status >= 500
                else 'invalid_request_error')
        body = dict(error=dict(message=message, type=kind, param=param,
                               code=code or {401:'invalid_api_key',403:'forbidden',404:'not_found',409:'conflict',413:'payload_too_large',429:'queue_full'}.get(status, 'server_error' if status >= 500 else 'invalid_request')))
    else:
        body = dict(detail=message)
    return JSONResponse(body, status_code=status, headers=headers)
