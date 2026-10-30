# Public audio API and processing queue

Sidon exposes a versioned native audio-processing API with familiar OpenAI-style
conventions: bearer authentication, `model`, `response_format`, model discovery,
request IDs and structured errors. Enhancement, separation and conversion are
**Sidon extensions**, not OpenAI speech/transcription endpoints. The OpenAI
`audio.speech.create()` method is not a restoration client. No streaming is
implemented; submit each complete client-selected chunk as an independent request.

## Direct audio responses

| POST route | Model (optional; inferred when omitted) | Response |
| --- | --- | --- |
| `/v1/audio/enhancements` | `sidon` | Restored audio |
| `/v1/audio/separations` | `mossformer2` | ZIP with two tracks and `run.json` |
| `/v1/audio/conversions` | `audio-converter` | Converted audio |

All three accept multipart `file`, `response_format` and `response_mode`.
`response_format` is `wav` (default), `flac`, `opus`, `mp3`, `m4a`, `aac` or `ogg`.
`response_mode=audio` (default) waits for completion and returns the file with
HTTP 200. A separated pair is a ZIP, not an unlabelled concatenated audio stream.
Responses include `X-Request-ID`, `X-Job-ID` and a `Location` pointing to job status.
Results remain available until deletion/expiry, including after direct responses.

```bash
curl http://localhost:7868/v1/models
curl -X POST http://localhost:7868/v1/audio/enhancements -F file=@voice.wav -F model=sidon -F response_format=opus --output restored.opus
curl -X POST http://localhost:7868/v1/audio/conversions -F file=@voice.m4a -F response_format=flac --output converted.flac
```

Optional processing fields: `strength` (0–1, default 1), `chunk_seconds` (2–16,
default 4) and `overlap_seconds` (0.1–2, default 1; shorter than the separation
window). Set `enhance=true` on separations to restore both tracks after splitting.
The model parameter then still selects the separator; restoration uses Sidon.
Fields that do not affect an operation have no effect, and harmless unknown form
fields are ignored. `stream=true` is explicitly rejected.

Versioned requests have fixed documented defaults, independent of the Studio's
saved settings. Every request has its own normalization and speaker assignment;
there is no cross-request state. Clients control chunk boundaries, ordering and
joining, including matching speaker labels between separate requests. See
[audio formats](audio-formats.md) for rates, channels and encoding behavior.

## Asynchronous jobs

Add `-F response_mode=job` to a transform request to receive HTTP 202 and an
`audio.job` object immediately after upload/validation. Alternatively, submit to
`POST /v1/audio/jobs` with `mode=enhance|separate|separate_enhance|convert`.
That route accepts `response_format` or the legacy `output_format` alias;
supplying both with different values is an error.

```bash
curl -X POST http://localhost:7868/v1/audio/enhancements -F file=@voice.wav -F response_format=opus -F response_mode=job
curl http://localhost:7868/v1/audio/jobs/JOB_ID
curl -OJ http://localhost:7868/v1/audio/jobs/JOB_ID/files/enhanced.opus
curl -X POST http://localhost:7868/v1/audio/jobs/JOB_ID/cancel
curl -X DELETE http://localhost:7868/v1/audio/jobs/JOB_ID
```

Poll `status`: `queued`, `running`, `done`, `failed` or `cancelled`. Download only
when `done`; filenames and URLs are in `outputs`, with `result_url` for the direct
result. `GET /v1/audio/jobs` lists retained jobs as `{object:"list", data:[...]}`.
Deleting a running job returns 409: request cancellation first, then delete after
it becomes terminal. Cancellation is idempotent; cancelling a completed job keeps
its completed result. Direct requests can also observe cancellation and return a
409 `job_cancelled` error. A detected disconnect while waiting requests cancellation.

## Admission, FIFO and cancellation

- One worker processes jobs sequentially. The model lock also prevents manual
  load/unload from overlapping processing. Concurrent clients share this queue.
- `SIDON_MAX_ACTIVE_JOBS` defaults to **4**, configurable from 1 to 64. It counts
  requests receiving/parsing uploads, decoding, queued jobs and running work,
  including running work awaiting cancellation. It is not four waiting plus one
  running. Admission is reserved **before multipart parsing**.
  A disconnected upload keeps its reservation until an already-started decoder
  finishes, so abandoned requests cannot leave extra decoder work outside the cap.
- When full, requests receive **429** with `Retry-After: 1`. Retry after at least
  that delay, adding jitter for multiple clients. The limit is capacity control,
  not a requests-per-minute quota. Completed file downloads do not occupy slots.
- Ready jobs run FIFO in the order that their uploads finish validation. A slow
  upload does not reserve a processing-order position ahead of already-ready work.
- Pending cancellation removes the queued item immediately and frees its slot.
  Running cancellation is cooperative, checked between processing windows, tracks and
  output stages. It does not interrupt a CUDA forward or codec process mid-call.
- `/v1/audio/queue`, `/v1/audio/capabilities`, and Studio status expose capacity,
  active/available slots and uploading/queued/running counts. Each job has a live
  `queue_position` (one-based while waiting, zero while running, null if terminal).
- New completed jobs/manifests include `queue_wait_seconds`, `processing_seconds`
  and `total_seconds`. Total includes receiving/decoding/queueing; processing
  includes model loading, inference, encoding and archive payload copying.
  `elapsed_seconds` remains the processing-time alias for older clients.

One server process owns each data directory through an OS lock. Run one instance
per GPU allocation, with separate data volumes for separate instances. Different
instances do not share an admission queue or coordinate GPU use. There is no
distributed scheduler, parallel GPU batching or tenant-specific fairness here.
Interrupted queued/running jobs become failed after restart; they are not replayed.
Retry ambiguous transport failures carefully: POST requests are not deduplicated.

## Authentication, discovery and errors

Set `SIDON_API_KEY` to protect both `/api/*` and `/v1/*`; use
`Authorization: Bearer YOUR_KEY`. Without a configured key the local API is open.
All authenticated clients of an instance share jobs/settings. Keep tenant
isolation and GPU allocation at the deployment boundary.

`GET /v1/models` and `/v1/models/{id}` expose model objects. `created: 0` means the
catalog does not specify a creation timestamp. `GET /v1/audio/capabilities`
describes native operations, formats and the no-streaming contract. The full
request/response schema is `/openapi.json`; health is `/healthz`.

Versioned errors use this envelope, with a server-generated `X-Request-ID`:

```json
{"error":{"message":"The audio queue is full. Retry after capacity is available.","type":"rate_limit_error","param":null,"code":"queue_full"}}
```

Typical statuses are 400 (invalid request), 401/403 (access), 404 (missing
model/job/file), 409 (job conflict/cancellation), 413 (upload too large), 429
(capacity), and 500/503 (processing/server failure). Validation errors identify
the offending parameter where available. Keep job IDs for asynchronous recovery.

## Existing Studio API

`/api/jobs`, settings, model controls and GPU telemetry remain supported.
Legacy errors retain their `detail` shape; legacy job submission uses saved
Studio defaults. Both API families share the same admission queue, jobs and
cancellation, so the new routes cannot bypass Studio capacity limits.
