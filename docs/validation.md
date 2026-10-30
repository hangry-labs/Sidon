# Product qualification, 2026-10-29

This records functional qualification of the Python 3.13 studio and Docker
runtime. It is not a perceptual restoration benchmark or a production load test.

## Runtime and audio

- Windows development environment: Python 3.13.13, PyTorch / torchaudio
  2.10.0+cu128; NVIDIA RTX 5070 Ti.
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

- 105 tests passed on the Windows Python 3.13 environment, with five skips
  (four CUDA/Triton cases and one unavailable symlink case).
- All 13 native-backend tests passed separately in the Linux GPU container,
  including four bit-exact fused-kernel cases and initialization fallback checks.
- The packaged Linux CPU runtime passed 103 tests, with seven skips (four GPU kernel
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

The runtime collector records 66 Python distributions and 326 Debian packages,
with no missing-notice flags. Metadata omissions were checked against included
license texts; notably pathspec's MPL-2.0 notice is supplied with its matching
source despite its empty license metadata. The source target checks exact Debian
source identities and reviewed wheel native hashes, downloads matching sources,
and verifies sizes and SHA256 values before marking its manifest complete.

The current image's source manifest contains 719 verified files, including
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

## Remaining qualification limits

Quality on representative generated voices still needs listening comparisons.
Model checks used the short real speech fixture, including repeated input up to
32 seconds for the [performance follow-up](performance.md). Long-input paths and
speaker permutations have unit coverage, without a long-form listening study.
Other GPUs, operating systems, architectures, multi-tenant isolation and sustained
production load have not been qualified. The deployment is one shared instance
with an optional shared API key, rather than a tenant-management product.

No registry image, Docker Hub description, release or Git tag was published by
this local implementation work. Publication needs configured registry secrets and
a successful workflow run; stable documentation digests must come from that run.
