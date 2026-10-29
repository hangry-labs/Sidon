<p align="center"><img src="src/sidon/web/static/assets/sidon-wordmark.svg" width="560" alt="Sidon · Voice restoration studio"></p>

# Sidon

Restore noisy or degraded speech, split a two-speaker mixture, and listen to the results in a local Hangry Labs studio. One Docker container serves the browser UI, audio-job API and models.

This is an inference-focused fork of [sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon). It uses Sidon's single-speaker restoration exports and Apache-licensed MossFormer2 for separation. Training and dataset preparation are outside this product's scope.

## Run locally

Build the full image with baked, checksum-verified CPU and CUDA weights:

```bash
docker build --target baked -t sidon:local .
docker run -d --name sidon --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent sidon:local
```

Open **http://localhost:7868**. NVIDIA GPU support requires a compatible host driver and NVIDIA container support. The image targets **Linux x86-64, Python 3.13 and CUDA 12.8 / PyTorch 2.10**. It can also run on CPU: omit `--gpus all` and add `-e SIDON_DEVICE=cpu`. CPU processing is slower.

Or use `docker compose up -d --build`. The supplied Compose configuration requests a GPU and binds the UI to localhost.

The full image reads weights directly from its baked model directory and does not need a first-run download. The five assets occupy about 2.68 GB before runtime dependencies. Keep the single data volume for settings and audio jobs.

## What the studio does

| Workflow | Result |
| --- | --- |
| Enhance | Sidon restores a single voice; mono 48 kHz output, with an original/restored mix control |
| Split speakers | MossFormer2 separates a mixture into two mono 16 kHz tracks |
| Split, then enhance | Separates first, then restores each track; two mono 48 kHz outputs |

Upload WAV, FLAC, OGG, AIFF or another format supported by the installed libsndfile. Input is downmixed to mono. Compare the original and results with waveform players, seek, volume and speed controls. Download individual tracks or a ZIP with a processing manifest. Choose float32 WAV to retain samples above full scale, or 24-bit FLAC; FLAC clips samples outside its PCM range and the UI reports that condition.

System provides model load/unload controls, server defaults, retained-job history and demand-driven GPU charts. Models also load automatically when a job needs them. Settings survive container replacement through the data volume.

![Sidon studio processing a two-speaker mixture](docs/assets/studio.webp)

Restoration resynthesizes speech. It can change timbre or add artifacts and does not recover missing words reliably. Separation always returns two tracks; speaker labels are relative to the first window, and ambiguous overlaps can swap them. Neither path is speaker identification. Compare representative generated voices before adopting a preset.

## Smaller image and cache

```bash
docker build --target tiny -t sidon:tiny .
docker run -d --name sidon-tiny --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent sidon:tiny
```

The tiny image has the same UI and runtime but downloads a model's pinned weights on first use. Keep `/app/persistent` to reuse the cache. Download every model/device variant you need before disconnecting: CPU and CUDA restoration exports are different files. Restart with the same volume and `-e HF_HUB_OFFLINE=1` for cached operation.

## Image publication

The GitHub workflow is configured to publish to `hangrylabs/sidon` and `ghcr.io/hangry-labs/sidon`. These are publication targets; this change does not itself publish images or assert a first release exists. Build locally until a successful publication run is available.

| Trigger | Full | Tiny | Corresponding sources |
| --- | --- | --- | --- |
| Push to main | `latest` | `latest_tiny` | `latest_sources` plus an immutable build identity |
| Existing stable Git tag | `vX.Y` or `vX.Y.Z` | matching `_tiny` | matching `_sources` plus an immutable build identity |

Stable tags are never overwritten. Tag builds do not move `latest`; manual publication can explicitly opt in. Sources publish before binary images, and both registries' resulting index digests are checked. Publication evidence includes runtime notices, exact installed versions, model hashes and a verified source manifest. See [Docker packaging](docs/docker-packaging.md) and the [Docker Hub description](docs/dockerhub.md).

## API

The studio uses a native job API; it does not claim an OpenAI audio-restoration standard. Examples below use Bash/cURL (`curl.exe` avoids PowerShell's legacy alias).

```bash
curl -X POST http://localhost:7868/api/jobs -F file=@voice.wav -F mode=enhance -F strength=1 -F output_format=wav
curl http://localhost:7868/api/jobs/JOB_ID
curl -OJ http://localhost:7868/api/jobs/JOB_ID/files/results.zip
curl -X DELETE http://localhost:7868/api/jobs/JOB_ID
```

Submission returns HTTP 202 with an `id`. Poll until `status` is `done` or `failed`. Completed jobs list each track's filename, sample rate, duration and count of samples above full scale. Use `mode=separate` or `mode=separate_enhance` for the other workflows. Defaults are read when a job is submitted.

Discovery: `GET /healthz`, `GET /api/status`, `GET /api/jobs`, and `/openapi.json`. Persist defaults with `PUT /api/settings`. Manage residency with `POST /api/models/enhance` or `/api/models/separate` and JSON `{"action":"load"}` or `{"action":"unload"}`. GPU telemetry is `/api/gpu`. The UI's API tab has the same examples.

When `SIDON_API_KEY` is set, all `/api/*` requests require `Authorization: Bearer YOUR_KEY`; enter it in System for browser use. Health, static assets and the API schema remain public. All authorized clients of an instance share its jobs and settings. For hosted workloads, isolate tenants in separate instances and place TLS/access controls at your reverse proxy.

## Runtime settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SIDON_DEVICE` | `auto` | CUDA when available, otherwise CPU; explicit `cpu` or logical `cuda:0` |
| `SIDON_HOST` | `127.0.0.1` locally; `0.0.0.0` in Docker | Server bind address |
| `SIDON_PORT` | `7868` | HTTP port; update mapping and healthcheck if changed |
| `SIDON_DATA_DIR` | `/app/persistent/studio` in Docker | Defaults and retained audio jobs |
| `HF_HOME` | `/app/persistent/huggingface` in Docker | Tiny-image model cache |
| `SIDON_MODEL_DIR` | `/app/baked-models` in full image | Verified baked model assets |
| `HF_HUB_OFFLINE` | full `1`, tiny `0` | Disallow or allow model downloads |
| `SIDON_MAX_UPLOAD_MB` | `100` | Maximum file size |
| `SIDON_MAX_AUDIO_SECONDS` | `600` | Maximum decoded clip duration |
| `SIDON_RETENTION_HOURS` | `24` | Automatic uploaded/generated audio expiry |
| `SIDON_API_KEY` | unset | Optional shared bearer key |

At most four jobs may be active, with one inference worker per process. Run one server process per GPU allocation. Use `--gpus device=0` to select a physical GPU; the published Sidon CUDA export uses logical `cuda:0`. Separation windows default to four seconds with one-second overlaps; adjust and save them in System.

## Development

```bash
uv sync --locked
uv run --locked sidon-studio
uv run --locked pytest -q
```

The project lock uses the CUDA 12.8 PyTorch index explicitly. Docker installs a separate fully hashed Linux/CPython 3.13 runtime lock. Regenerate it with `pwsh -File scripts/lock-runtime.ps1`; check it with `-Check 1`. `Taskfile.yml` provides matching setup, studio, test and image tasks.

The separation CLI remains available:

```bash
uv run --locked sidon-separate --input dialogue.wav --output-dir separated
```

Its output folders contain two float WAVs and `run.json`. It rejects existing output folders. See [separation details](docs/mossformer-runtime.md) and [validation notes](docs/validation.md).

For cold/warm latency measurements and the audio.cpp implementation comparison,
see [performance notes](docs/performance.md).

## Privacy, attribution and licenses

Processing runs on your server. The full image's core UI and inference work without an external network. Tiny images contact Hugging Face to download weights. Audio and results are kept in the server data volume until deletion or expiry; expiry runs at startup and about once per minute. Deleting a job removes its server files, but not copies already downloaded by a client. Browser input files stay in page memory until the page closes.

Sidon restoration is credited to sarulab-speech; MossFormer2 to its authors and ClearerVoice-Studio. The UI shell and packaging conventions are adapted from Hangry Labs faster-qwen3-tts-studio, retaining its MIT notice. Lucide icons retain their license.

Our contributions use [Apache-2.0](LICENSE). Retained source includes MIT material and ISC-licensed Lucide assets, so the package expression is **Apache-2.0 AND MIT AND ISC**. Sidon restoration weights are publisher-declared MIT; MossFormer2 weights are publisher-declared Apache-2.0. Model grants are distinct from training-data provenance or rights to uploaded audio.

Complete containers also include third-party LGPL/GPL runtime components and NVIDIA binaries under their own terms. Corresponding sources, build inputs and notices are supplied by the `sources` target; the application license does not replace those terms. See [licensing](docs/licensing.md), [runtime notices](RUNTIME_NOTICES.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Feedback and support: [contact@hangrylabs.app](mailto:contact@hangrylabs.app).
