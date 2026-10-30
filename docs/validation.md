# Product qualification, 2026-10-29

This records functional qualification of the Python 3.13 studio and Docker
runtime. It is not a perceptual restoration benchmark or a production load test.

## Runtime and audio

- Security-updated Windows development environment: Python 3.13.13,
  PyTorch 2.14.1+cu130 / torchaudio 2.11.1; NVIDIA RTX 5070 Ti.
- Linux x86-64 Docker images built from the pinned Python 3.13 base and fully
  hashed runtime requirements. `pip check` passed in full and tiny images.
- Full image: all three real-model API workflows completed with Docker networking
  disabled and a fresh data volume. No host model cache was mounted.
- CPU restoration also completed in the full image with networking disabled and
  a fresh volume, selecting the separate CPU TorchScript exports.
- Tiny image: all three workflows completed with a fresh online volume, then
  completed again after recreating the container with that same volume,
  `HF_HUB_OFFLINE=1` and Docker networking disabled.
- A 4.4085625-second real speech mixture produced 211,611-sample 48 kHz restored
  tracks and 70,537-sample 16 kHz separated tracks. Outputs were finite; downloadable
  WAVs, ZIP manifests and saved defaults were checked through the HTTP API.
- Separation testing caught arbitrary raw model gain. The wrapper now uses the
  upstream input operating level and whole-track RMS output calibration. This
  preserves duration and avoids gain changes at joins, but does not estimate each
  speaker's original loudness. Some separated peaks can still exceed full scale;
  float WAV retains them, and FLAC encoding clips them with a reported warning.

The short sample is a local evaluation input, not a bundled public fixture.
Durations above are correctness checks, not throughput measurements. Cold model
loading, concurrent GPU use and compiler warm-up affect elapsed time.

## Tests and interface

- 123 tests passed on the Windows Python 3.13 environment, with five skips
  (four CUDA/Triton cases and one unavailable symlink case).
- All 14 native-backend tests passed separately in the Linux GPU container,
  including four bit-exact fused-kernel cases and initialization fallback checks.
- The packaged Linux CPU runtime passed 121 tests, with seven skips (four GPU kernel
  cases and three tests needing Git). The kernel cases are separately exercised in
  the GPU container. Tests cover chunk tails and swaps, silence, resampling,
  model hash rejection, source identity/tamper checks, arbitrary gain, API jobs,
  queue/residency conflicts, auth, invalid inputs, limits, settings and expiry.
- The Linux hash lock was checked against its input pins; wheel and sdist built.
- Browser checks used Chromium at widths 1440, 768 and 390 pixels, in expanded and
  compact header states, including System. No horizontal overflow was detected.
- Real packaged-service uploads, all three processing modes, waveform rendering,
  playback, ZIP download, tab state and GPU telemetry were exercised. JavaScript
  page errors were monitored during these workflows.

## Distribution evidence

The runtime collector records 67 Python distributions and 326 Debian packages,
with no missing-notice flags. Metadata omissions were checked against included
license texts; notably pathspec's MPL-2.0 notice is supplied with its matching
source despite its empty license metadata. The source target checks exact Debian
source identities and reviewed wheel native hashes, downloads matching sources,
and verifies sizes and SHA256 values before marking its manifest complete.

The current image's source manifest contains 721 verified files, including
FFmpeg/codecs and the compiler/header sources needed for the local Triton launcher. Original
audio.cpp conversion attribution is retained alongside the other source notices.

The qualified wheel-source policy covers torch, numpy, scipy, soundfile, certifi,
tqdm and pathspec. Original notices and the Intel oneMKL supplemental notice are
included. This does not turn the entire NVIDIA/native runtime into Apache-licensed
software or independently guarantee model training provenance.

Use `scripts/smoke_api.py --input YOUR_AUDIO --url http://localhost:7868` to repeat
real-model API checks. Provide your own authorized audio; weights are not mocked
by that script. CPU-only CI unit tests do not run the real checkpoints.

## Audio conversion follow-up, 2026-10-30

All seven output formats passed conversion, download, ZIP manifest and reupload
checks. WAV conversion preserves stereo samples exactly; compressed round trips
are checked for channels, rate, finite audio and duration with codec padding.
Tests cover peak handling, MP3 downmix and playlist rejection. The full and tiny
images both passed real codec conversion with Docker networking disabled and no
model downloads. The full image additionally ran all three inference workflows
with Opus exports, using the optimized restoration backend.

Browser checks exercised Opus upload to M4A conversion, download, zero model
loads and the new Convert tab at 320, 390, 768 and 1440 pixels without horizontal
overflow. Repeat with `python scripts/smoke_formats.py --input YOUR_AUDIO` and add
`--inference` to exercise actual models. This script deletes its completed jobs.

## Public API and queue follow-up, 2026-10-30

The versioned API and queue follow-up exercised eight simultaneous submissions
competing for one free slot: one was accepted and seven received 429 with retry
headers. Tests cover FIFO positions, removable queued cancellation, cooperative
running cancellation, decoder cleanup after disconnect, shared data-directory
ownership, structured errors, body-size limits and legacy API behavior.
Cancellation checks leave outputs bit-identical on the tested 23.112-second
restoration input and 7.704-second separation input when no cancellation occurs.

Real offline full-image checks exercised the versioned direct enhancement,
separation, split-then-enhance and conversion routes, plus asynchronous responses.
The tiny image passed versioned conversion without network/model downloads.
The official OpenAI Python SDK successfully listed/retrieved our model catalog
and parsed a structured missing-model error with its request ID. This validates
those conventions, not OpenAI speech-endpoint compatibility. Browser checks
exercised queue position, pending cancellation/deletion and System queue counts.
Repeat native endpoint checks with `scripts/smoke_public_api.py`.

## Recovery and interface review, 2026-10-30

Runtime ownership, persistence and model processing now live separately from HTTP
routing. Saved jobs use the same validated settings/output contract as the public
API. Malformed records are ignored without taking valid history or the service
offline. Regression tests cover invalid metadata, startup failures, exceptional
lifespan exits and shutdown during background expiry. Expiry work finishes before
directory ownership is released; transient filesystem errors retry next interval.

Browser checks cover completed, failed and cancelled history, clearing old audio
and download actions, player resets, expired jobs and a delayed status response
arriving after cancellation and input replacement. These checks used Chromium at
320, 390, 768 and 1440 pixels. The rebuilt full image also passed real offline
conversion, restoration, separation, split-then-enhance and asynchronous API checks.

## Dependency security follow-up, 2026-10-30

Both dependency locks were audited after the [security fixes](security.md):
66 development-lock packages and 65 runtime-lock packages reported no known
advisories. The audit uses the public package version for CUDA-local wheel
versions; the actual installed wheel is separately hash locked and tested.

The PyTorch 2.13.0+cu130 candidate still crashed on the upstream malformed JIT
annotation reproducer despite the advisory marking that version as fixed.
PyTorch 2.14.1+cu130 safely rejects it on Windows and Linux. The new child-process
regression test checks that behavior. The final stack uses TorchAudio 2.11.1 and
Triton 3.8.0 with a CUDA 13-compatible host driver.

Final full-image checks, without networking or a mounted model cache, exercised
all four versioned operations, asynchronous jobs, all seven format round trips
and real GPU models. CPU restoration also passed using the baked CPU exports.
The tiny image passed offline conversion without loading/downloading models.
The full corresponding-source image verified all 721 files. Its notice inventory
has 67 Python distributions and 326 Debian packages with no missing-notice flags.
The dependency-only CUDA metapackage is recorded explicitly, with exact metadata
and payload checks; it is not assigned an invented software license.

One public 7.704-second speech sample and a threefold repeat were compared with
the earlier PyTorch 2.10/CUDA 12.8 stack. Sampled filterbank features matched
exactly; output lengths and finite samples were preserved. Restoration difference
SNR was 51.8 dB and 55.7 dB for the short and repeated clips; separation difference
SNR was 55.7 dB. The operator subsequently listened in the updated Docker build
and confirmed that quality was still fine on the tested material. This is informal
listening acceptance limited to that material; the outputs remain numerically
different.

A separate sequential timing check used the same RTX 5070 Ti, ten CPU threads,
adaptive backend and instrumented benchmark script. Each length had one cold
call followed by three warmed calls; the table reports warmed medians:

| Input duration | PyTorch 2.10 / CUDA 12.8 | PyTorch 2.14.1 / CUDA 13.0 |
| --- | --- | --- |
| 7.704 seconds | 66.45 ms | 77.15 ms |
| 23.112 seconds | 217.96 ms | 245.09 ms |

This run's medians differed by about 16% and 12% (roughly 11 ms and 27 ms).
The workstation and GPU were shared with other local workloads; background
traffic, contention and clock variation were not controlled. These observations
do not establish a repeatable slowdown or attribute the difference to the upgrade.
The operator explicitly noted local traffic/noise as a possible explanation.
Performance impact remains inconclusive pending controlled, repeated measurements.
Earlier optimization ratios in performance.md were measured on the original stack
and are not reasserted for the upgraded stack.

## Pre-release review, 2026-10-30

Release preparation now includes a tested `task release` command, following the
faster-qwen3-tts-studio process. The dry run exercises stable and next-snapshot
document/version transformations without writing files or creating refs. Both
transformed source trees passed version/lock and release-contract checks. Real
Git tests use disposable local repositories to verify annotated tagging, atomic
push, restoration after validation failure, and retry after a failed push.

The final preparation tree passed 144 Windows tests (five skips), 139 packaged
Linux CPU tests (ten skips), six JavaScript tests and all 14 GPU-native tests.
Linux skips cover GPU kernels and host-Git checks; GPU kernels and real Git
transitions are exercised separately. The source archive now includes its lock,
Docker context exclusions and workflow/build inputs needed by its included tests.
Wheel/source archives were inspected for required notices and excluded private
files, caches and model weights.

Fresh full/tiny images were inspected directly: installed application files match
the packaged source; all five baked assets pass their pinned SHA256 checks;
no user recordings or private workspace files are included. The actual notice
inventory remains 67 Python and 326 Debian packages, with no missing-notice flags.
All 721 corresponding-source files verified, and the source manifest's inventory
hash exactly matches the packaged runtime.

The rebuilt full image passed offline GPU direct/async operations, all three
model workflows and seven-format conversion/reupload checks, with no mounted
model cache. The tiny image passed offline conversion with no model loads.
Authentication, static UI assets and empty queues after job deletion were checked.
The release guide also rendered in the website preview without browser errors.
GitHub reported zero open Dependabot and CodeQL alerts at this checkpoint.

These are preparation checks on `1.0.0-snapshot`; the version transition was
qualified in isolated source copies. No stable tag, release commit or publication
was performed by this review. Publishing CI must qualify the eventual tagged
checkout and supply its immutable full/tiny/source digests. See [releasing](releasing.md).

## Stable release publication, 2026-10-30

[Sidon v1.0.0](https://github.com/hangry-labs/Sidon/releases/tag/v1.0.0) is published
as an immutable GitHub Release. Its annotated tag identifies source commit
`c62528ae8e2722768e26303432a438de56b698fa`; development continues at
`1.0.1-snapshot`. The [stable publishing workflow](https://github.com/hangry-labs/Sidon/actions/runs/41065109270)
passed its source/package checks, packaged runtime tests, corresponding-source
collection, binary publication and registry verification.

Independent anonymous registry requests confirmed identical full, tiny and source
OCI indexes in Docker Hub and GHCR. Binary configuration labels identify the
stable version, exact source commit and unique matching source image. The
published evidence contains 67 Python and 326 Debian records, no missing-notice
flags, and 721 verified source files with a matching runtime inventory hash.
The GitHub Release also provides a wheel, source archive and SHA256 checksums
built directly from the immutable tag. Stable deployment references are recorded
in the [release notes](releases/v1.0.0.md).

## v1.0.1 authentication patch, 2026-10-30

[v1.0.1](https://github.com/hangry-labs/Sidon/releases/tag/v1.0.1) fixes repeated
navigation to System after unauthorized background requests and adds masked key
retention, Show/Hide and Copy controls. Server authentication remains optional
through `SIDON_API_KEY`; the server's configured secret is never returned to the
browser. Inference, model weights and runtime dependency pins are unchanged.

An image built directly from the tag passed Chromium checks for missing/wrong-key
navigation, paused unauthorized polling, successful reconnection, masked copy,
automatic masking and clearing on reload. Packaged layouts at 1440, 390 and 320
pixels had no horizontal overflow or JavaScript errors. Both clipboard paths
were also checked during frontend validation, including plain HTTP fallback.
The tagged full image passed offline GPU enhancement, separation, split/enhance,
direct/async APIs and all seven format round trips without a model-cache mount.

The [stable publishing workflow](https://github.com/hangry-labs/Sidon/actions/runs/41076843359)
passed. Independent anonymous checks confirmed matching full/tiny/source OCI
indexes in Docker Hub and GHCR, correct version/commit labels and exact source
identity. The published inventory contains 67 Python and 326 Debian records,
zero missing-notice flags and 721 verified corresponding-source files. The
GitHub Release is immutable, with tagged Python artifacts and SHA256 checksums.

## Remaining qualification limits

Listening acceptance covers the operator's tested material; a broader controlled
perceptual evaluation has not been performed. Model checks used the short real
speech fixture, including repeated input up to
32 seconds for the [performance follow-up](performance.md). Long-input paths and
speaker permutations have unit coverage, without a long-form listening study.
Other GPUs, operating systems, architectures, multi-tenant isolation and sustained
production load have not been qualified. The deployment is one shared instance
with an optional shared API key, rather than a tenant-management product.

The security-updated `main` subsequently passed its
[publishing workflow](https://github.com/hangry-labs/Sidon/actions/runs/41039867103).
On 2026-10-30, rolling full, tiny and source OCI indexes matched between Docker
Hub and GHCR, and all 14 reviewed Dependabot alerts were marked fixed. No stable
release tag or GitHub Release was created during this qualification. Stable
documentation digests must come from the eventual stable publication.
