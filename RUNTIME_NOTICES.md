# Inference runtime notices

The default Sidon wheel/source distribution includes the studio/API, restoration
wrapper and selected MossFormer2 SS architecture. Its source license
expression is **Apache-2.0 AND MIT AND ISC**. This fork's own contributions use
Apache-2.0, as set out in the root `LICENSE`. Installed dependencies, CUDA libraries
and separately downloaded model weights retain their own terms.

## Sidon

Copyright (c) 2025 sarulab-speech. The upstream MIT license is preserved in
`licenses/Sidon-upstream-MIT.txt`, including its copyright, grant and disclaimer.
Upstream material retains that grant.

Copyright (c) 2026 Hangry Labs contributors. New integration code and this
fork's own contributions are offered under Apache-2.0, in the root `LICENSE`.
This project-level choice does not relicense third-party material or replace
its retained notices.

## ClearerVoice-Studio / MossFormer2 SS

The four model files under `sidon/inference/_mossformer2` are adapted from
ClearerVoice-Studio revision `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`:
https://github.com/modelscope/ClearerVoice-Studio

Authors credited upstream include Shengkui Zhao, Zexu Pan and Trung Hieu Nguyen.
The architecture is offered under Apache-2.0. The original text is included as
`licenses/ClearerVoice-Apache-2.0.txt` in the source distribution and under the
wheel's `.dist-info/licenses` directory.

Changes made on 2026-10-29: separation-only packaging; removal of unused imports;
replacement of the channel-layer-normalization wrapper with PyTorch LayerNorm
and explicit axis transposes. The original third-party `layer_norm.py`, training,
metrics, video and other model families are not included. Source hashes and URLs
for each imported file are recorded in `_mossformer2/provenance.json`.

The model implementation credits SpeechBrain's dual-path implementation
(Apache-2.0) and FLASH-pytorch (MIT). Their license texts are preserved in
`licenses/SpeechBrain-Apache-2.0.txt` and `licenses/FLASH-MIT.txt`, respectively,
and in the wheel's license directory. FLASH's notice is:

Copyright (c) 2022 Phil Wang.

## Separately downloaded weights

Model: `alibabasglab/MossFormer2_SS_16K`

Publisher model card, declaring Apache-2.0:
https://huggingface.co/alibabasglab/MossFormer2_SS_16K/blob/407cb030cd66340918ebb6c8cc63b18f8592cdbe/README.md

Revision: `407cb030cd66340918ebb6c8cc63b18f8592cdbe`

File: `last_best_checkpoint.pt`

SHA-256: `00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a`

Checkpoint bytes are verified before loading; loading uses PyTorch's
`weights_only=True` with strict parameter matching. Model weights are downloaded
separately into the Hugging Face cache and are not embedded in the wheel/sdist.

The full Docker image bakes in this pinned asset for offline inference.
Its distribution must include the model license and attribution alongside the
application and dependency notices; see `docs/docker-packaging.md`.

## Binary dependencies

Python package licenses do not replace the terms on bundled native libraries.
In particular, SoundFile wheels can bundle LGPL-licensed libsndfile and related
codec libraries, while PyTorch GPU builds include NVIDIA components. Preserve
their installed notices and meet the applicable source/relinking and NVIDIA
redistribution requirements before distributing a complete binary environment.
These components allow commercial use subject to their terms; they are not all
licensed under the source package's MIT/Apache expression.

The repository contains only the selected inference implementation. The upstream
research code, old demo, training/data pipelines and experimental dependency
group have been removed. See `docs/licensing.md` for the current source review
and `docs/mossformer-runtime.md` for runtime validation.

## Sidon restoration weights and feature preparation

Publisher: sarulab-speech. Model repository `sarulab-speech/sidon-v0.1`,
revision `b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52`, publisher-declared MIT.
The pinned publisher card is preserved as `licenses/Sidon-MODEL-CARD.md`.
The original Sidon MIT grant and attribution are in `licenses/Sidon-upstream-MIT.txt`.

CPU and CUDA `feature_extractor_*.pt` / `decoder_*.pt` TorchScript exports are
loaded directly, without training/model Python code or flow-matching. Exact
file hashes are recorded in `sidon/inference/assets.py` and the baked image's
model manifest; every asset is verified before loading. TorchScript is executable
model content, so only the pinned, reviewed publisher exports are accepted.

Feature preparation follows upstream Sidon's Kaldi-filterbank normalization and
frame stacking under its MIT grant. This fork adds bounded overlapping inference,
right context, silence handling, original-gain restoration and an optional dry/wet
mix. These modifications are not claimed to reproduce the upstream demo exactly.
Publisher model licenses are not independent guarantees about training provenance.

## Studio and build helpers

The shared shell/audio styles, GPU monitor, source/notice collectors and publication
planner are adapted from faster-qwen3-tts-studio, MIT. Its complete original notice
is retained in `licenses/faster-qwen-upstream-MIT.txt`. New Sidon integration uses
Apache-2.0. The packaged Lucide subset carries its own complete license at
`sidon/web/static/vendor/lucide/LICENSE`. See `THIRD_PARTY_NOTICES.md` for the
supplemental Intel oneMKL notice and `docs/docker-packaging.md` for corresponding sources.

The separation wrapper now follows upstream active-region input normalization
and whole-track output RMS calibration. Raw model gain is not a usable playback
level. Each separated track is matched to mixture RMS, not an estimate of the
original speaker's loudness; no peak limiter is applied.

## Optimized F32 restoration

On supported Linux CUDA systems, restoration can bind the pinned exports into
an inference-only PyTorch implementation. The automatic path preserves the
reference encoder/arithmetic for shorter windows and uses a folded encoder with
strict F32 math for windows of at least 512 latent frames. All windows use a fused
F32 Snake activation in the vocoder. The folded encoder merges 16 LoRA branches
once at model load and projects relative-position embeddings before gathering
attention bias. Weights, normalization, chunking and output rates are preserved.

Tensor bindings adapt audio.cpp's Sidon converter at revision
`1f5dc02257379b790fae9f3a0e0d28065d9baf2f`, copyright 2026 ShugoAI LLC,
Apache-2.0; its complete notice is in `licenses/audio.cpp-Apache-2.0.txt`.
The original TorchScript path remains available, including CPU and unsupported
platform fallback. Forced `native` mode uses strict IEEE F32 for every window.
Precision flags are restored afterwards; service inference is serialized.

Triton uses the installed C compiler for a local launcher. Compiler/header notices
and corresponding source are included with the image's dependency records.
Cached kernels contain compiled program code, not audio.

## Audio conversion

FFmpeg and ffprobe are installed as Debian runtime packages for additional upload
containers and Opus, MP3, AAC/M4A and Vorbis exports. The application invokes them
as separate processes; their GPL/LGPL component licenses remain in force. The
runtime inventory retains their Debian copyright notices and exact source
identities, and the matching source image includes their corresponding sources.
Native Python installs use an operator-installed FFmpeg rather than a bundled
executable. No conversion codec or model weights are added to the Python wheel.
