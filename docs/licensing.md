# Licensing and redistribution

Reviewed for this inference-focused product on 2026-10-29. Source, model grants and
complete binary-image dependencies are separate layers; the root Apache license
does not replace third-party terms.

| Shipped component | Basis and retained notices |
| --- | --- |
| Hangry Labs application contributions | Apache-2.0, root LICENSE |
| Original Sidon material / filterbank preparation | MIT, licenses/Sidon-upstream-MIT.txt |
| Sidon restoration exports | Publisher-declared MIT; pinned revision b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52, licenses/Sidon-MODEL-CARD.md |
| MossFormer2 architecture | ClearerVoice-Studio Apache-2.0; revision 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61, provenance.json and retained license |
| Credited SpeechBrain / FLASH origins | Apache-2.0 / MIT, original notices retained |
| MossFormer2 SS weights | Publisher-declared Apache-2.0; revision 407cb030cd66340918ebb6c8cc63b18f8592cdbe, licenses/MossFormer2-MODEL-CARD.md |
| UI shell, GPU monitor, packaging/publication helpers | Adapted from faster-qwen3-tts-studio, MIT; licenses/faster-qwen-upstream-MIT.txt |
| Lucide icon subset | Complete packaged vendor/lucide/LICENSE retained, including applicable upstream notices |
| Optimized restoration tensor bindings | Adapted from audio.cpp Sidon converter, revision 1f5dc02257379b790fae9f3a0e0d28065d9baf2f, Apache-2.0; licenses/audio.cpp-Apache-2.0.txt |

The package's aggregate source expression is **Apache-2.0 AND MIT AND ISC**. Model binaries
are baked into the full Docker image or downloaded by the tiny image; they are
not committed or included in the wheel/sdist. Revisions and SHA256 values live in
`src/sidon/inference/assets.py`; the image also carries a verified model manifest.
No arbitrary model-repository Python is downloaded or executed. Pinned TorchScript
exports are executable model artifacts and are checksum verified before loading.

Model-card grants document the publisher's terms. They do not independently prove
all training-data rights or grant rights to uploaded recordings. Upstream issue
[#19](https://github.com/sarulab-speech/Sidon/issues/19) discusses a Japanese
training-law exception; that discussion is distinct from a model's distribution
license and does not relicense non-commercial libraries or DialogueSidon weights.

## Excluded research components

DialogueSidon weights, GENESES/DiT research implementations, unresolved helper
provenance and CC-BY-NC flow-matching are absent from the current runtime and its
dependencies. Training, preprocessing, dataset tools and the original Gradio demo
are removed. Historical Git commits are not rewritten by the source cleanup.

## Runtime and corresponding sources

`requirements.runtime.txt` locks and hashes the qualified Linux CPython 3.13 / CUDA
12.8 environment. `uv.lock` supports local development with explicit PyTorch-index
selection. The image collector inventories actual installed wheels and Debian
packages, copies their notices, and hashes bundled native files. See
[the generated dependency summary](runtime-dependency-licenses.tsv).

Native libraries retain licenses including LGPL libsndfile/codecs and GCC runtime
terms. NVIDIA CUDA libraries use their own redistribution agreements; Intel oneMKL
uses the Intel Simplified Software License, reproduced in THIRD_PARTY_NOTICES.md.
These are not collectively Apache-licensed containers. Commercial use is subject
to the component terms, including notice/source obligations where applicable.

The optimized runtime includes GCC and C development headers for Triton's local
launcher compilation. Their Debian notices and exact corresponding sources are
collected with the other runtime packages. The custom activation kernel is this
project's Apache-2.0 code; it does not contain model weights or training code.

The `sources` image contains matching Debian source packages (including patches),
reviewed wheel-bundled native component sources and build recipes. Collection
fails on missing notices, changed reviewed binary hashes, or an unreviewed
copyleft package. Source artifacts are size/hash verified. Published binaries
identify their exact immutable source image, which publishes first. Local builders
must retain/distribute the matching source target with their binaries. Keep
libraries replaceable; no application restriction limits LGPL debugging or
modification rights. Full details: [Docker packaging](docker-packaging.md).

This is an engineering record of declared grants, retained notices and matching
source provision, not a guarantee of every upstream rights claim. Qualification
results and remaining limits are in [validation](validation.md).
