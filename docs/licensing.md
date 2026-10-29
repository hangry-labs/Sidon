# Current source license review

Reviewed 2026-10-29. Scope: the current inference-only source tree, declared
dependencies, selected pretrained checkpoint and built wheel/source package.

## Retained source and model

| Component | Verified license basis |
|---|---|
| Hangry Labs integration code | Owner-selected Apache-2.0, root `LICENSE`; explicit SPDX headers on new Python modules/tests |
| Original Sidon attribution | Original MIT copyright, grant and disclaimer preserved in `licenses/Sidon-upstream-MIT.txt` |
| Four imported MossFormer2 architecture files | ClearerVoice-Studio [revision 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61](https://github.com/modelscope/ClearerVoice-Studio/tree/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61), Apache-2.0; source hashes/URLs in the retained provenance manifest |
| Architecture's credited SpeechBrain/FLASH origins | Apache-2.0 / MIT; original licenses and attribution preserved |
| `alibabasglab/MossFormer2_SS_16K` | Publisher's [Apache-2.0 model card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/blob/407cb030cd66340918ebb6c8cc63b18f8592cdbe/README.md), revision `407cb030cd66340918ebb6c8cc63b18f8592cdbe`; card preserved locally |

The checkpoint SHA-256 is
`00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a`.
The runtime checks it before strict, weights-only parameter loading. A model-card
declaration documents the publisher's grant; it is not independent proof of all
training-data rights. Model binaries are not part of the source repository or
Python distributions.

The upstream channel-normalization wrapper is not copied into the selected
architecture. Its needed operation uses standard PyTorch LayerNorm with explicit
axis transposes. Numerical comparison with the original model passed exactly on
the tested input. See [runtime notices](../RUNTIME_NOTICES.md) for modification
notices and [third-party notices](../THIRD_PARTY_NOTICES.md) for license texts.

## Removed from the current tree

The original training/preprocessing code, dataset tooling, experimental model
tree, cluster scripts, notebooks, Gradio demo and optional research dependency
group have been removed. This includes GENESES/DiT-derived code, unresolved
positional helper provenance, DialogueSidon loading paths and the non-commercial
`flow-matching` dependency. They are no longer selectable package extras or
retained source files. Historical Git commits are not rewritten by this cleanup.

License texts used only by those removed components were removed too. Attribution
and license terms for the components still shipped are preserved.

## Dependencies and artifact scope

The only runtime requirements are PyTorch, NumPy, SciPy, SoundFile, einops,
rotary-embedding-torch and huggingface-hub, with their transitive dependencies.
The lockfile and [current inventory](runtime-dependency-licenses.tsv) record
exact versions and publisher metadata. The rotary-embedding-torch 0.8.3 wheel's
included MIT license was also inspected directly, beyond its metadata label.

No known non-commercial or missing-license source component remains in the
selected current tree based on this review. Native-library obligations are
separate: SoundFile wheels may bundle LGPL libsndfile/codecs, and GPU PyTorch
builds contain NVIDIA components. Those licenses permit commercial use subject
to their terms; the application does not relicense them.

The wheel/sdist are checked for the required notices, licenses, provenance record
and declared dependency set. The aggregate license expression is
`Apache-2.0 AND MIT`, reflecting retained MIT portions as well as Apache code.
Publishing this cleaned source is distinct from distributing a complete runtime
image. The future Docker image still needs its exact base-image/native-binary
inventory, notices, applicable source/relinking arrangements and baked-model
verification. No Docker/UI/API implementation is claimed in this milestone.
