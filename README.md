# Sidon

An inference-focused audio project maintained by Hangry Labs. Our target is a
local Docker application with baked-in model weights, a browser UI and APIs for
restoring, converting and serving generated audio.

**Current milestone:** a tested two-speaker separation runtime using
MossFormer2 SS, with reviewed source/model licenses. Docker, UI, serving APIs and
single-speaker restoration are subsequent work; they are not implemented yet.

## Run separation

Python 3.10+ and `uv` are required. Validation used Python 3.11.

```bash
uv sync --python 3.11
uv run sidon-separate --input dialogue.wav --output-dir separated
```

For a directory of WAV/FLAC files with a compatible CUDA-enabled PyTorch build:

```bash
uv run sidon-separate --input-dir wavs --output-dir separated --device cuda
```

The first run downloads approximately 670 MB of pinned, checksum-verified model
weights to the Hugging Face cache. Later runs can use `--offline`.
`--device cpu` selects CPU inference. A local copy of the same audited checkpoint
can be supplied with `--checkpoint`.

Each input produces `speaker_1.wav`, `speaker_2.wav` and `run.json` in its own
output folder. Audio is downmixed, resampled to 16 kHz, and separated in overlapping
windows with speaker-order matching and crossfades. Output folders are never
overwritten. The WAV outputs use float32 samples to avoid PCM16 clipping.

Use `--chunk-seconds` and `--overlap-seconds` to adjust chunking. Speaker numbers
are relative to the first window, not identified people; ambiguous or silent
overlaps can still cause swaps. The model always returns two tracks.

Separation is not a demonstrated replacement for speech restoration. We have
not established perceptual improvements on our generated voices yet. See
[runtime validation](docs/mossformer-runtime.md).

## Licensing

Our contributions use **Apache-2.0**, in [LICENSE](LICENSE). Retained upstream
components have MIT/Apache terms, so the aggregate package expression is
**Apache-2.0 AND MIT**. The pinned MossFormer2 weights are declared Apache-2.0 by
their publisher.

The training/data pipelines, old demo, restricted research implementations and
their optional dependencies have been removed from the working source tree.
Required notices for retained components are preserved. See the
[source license review](docs/licensing.md), [third-party notices](THIRD_PARTY_NOTICES.md),
[runtime notices](RUNTIME_NOTICES.md) and
[dependency inventory](docs/runtime-dependency-licenses.tsv).

Native dependencies retain their own terms, including LGPL and NVIDIA terms
where applicable. The planned Docker image will need a separate review of its
actual base image, binaries, baked weights and required notices/source provisions.
Software licenses do not supply rights to input recordings or override the output
terms of the models that generated them.

## Development

```bash
uv run pytest
uv build
```

The repository and build artifacts contain only the inference runtime, tests,
documentation and required license/provenance records.

## Attribution and contact

- Original [Sidon project](https://github.com/sarulab-speech/Sidon), sarulab-speech;
  its [MIT notice](licenses/Sidon-upstream-MIT.txt) is preserved.
- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) and
  [MossFormer2 SS weights](https://huggingface.co/alibabasglab/MossFormer2_SS_16K).
- SpeechBrain and FLASH-pytorch, credited by the selected architecture.

Project questions and privacy requests: [contact@hangrylabs.app](mailto:contact@hangrylabs.app).
