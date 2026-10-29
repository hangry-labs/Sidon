# Third-party notices

Our contributions are licensed under Apache-2.0 in the root [LICENSE](LICENSE).
The following third-party grants and notices remain in effect:

| Material | License and preserved notice |
|---|---|
| Original Sidon project attribution | MIT; copyright (c) 2025 sarulab-speech, in [licenses/Sidon-upstream-MIT.txt](licenses/Sidon-upstream-MIT.txt) |
| Selected ClearerVoice-Studio MossFormer2 architecture | Apache-2.0; [licenses/ClearerVoice-Apache-2.0.txt](licenses/ClearerVoice-Apache-2.0.txt) |
| SpeechBrain-derived dual-path model portions | Apache-2.0; [licenses/SpeechBrain-Apache-2.0.txt](licenses/SpeechBrain-Apache-2.0.txt) |
| FLASH-pytorch-derived model portions | MIT; copyright (c) 2022 Phil Wang, in [licenses/FLASH-MIT.txt](licenses/FLASH-MIT.txt) |
| Pinned MossFormer2 SS weights, downloaded separately | Publisher declares Apache-2.0; preserved [model card](licenses/MossFormer2-MODEL-CARD.md) |

The exact imported source revision, file hashes and URLs are recorded in
[provenance.json](src/sidon/inference/_mossformer2/provenance.json). Local changes
and the model asset's revision/hash are described in [RUNTIME_NOTICES.md](RUNTIME_NOTICES.md).
The notices and license files accompany the built distributions.

Dependencies and bundled native libraries retain their own terms. Their top-level
metadata is recorded in [the dependency inventory](docs/runtime-dependency-licenses.tsv);
it is not a complete binary-distribution notice bundle for a future Docker image.
