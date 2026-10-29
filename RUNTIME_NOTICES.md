# Inference runtime notices

The default Sidon wheel/source distribution includes only the new inference
wrapper and the selected MossFormer2 SS architecture. Its source license
expression is **Apache-2.0 AND MIT**. This fork's own contributions use
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

The planned Docker image will bake in this pinned asset for offline inference.
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
