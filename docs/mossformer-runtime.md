# MossFormer2 inference runtime

The current CLI separates a mono mixture into two 16 kHz tracks using the
Apache-2.0 MossFormer2 SS checkpoint. The studio also offers a separate
Sidon restoration stage and a combined split-then-enhance workflow; see the README.

## Processing

- Accepts mono or multichannel audio supported by libsndfile, downmixing to mono.
- Resamples to 16 kHz and normalizes active-region RMS to the upstream operating level.
- Restores each complete output track to input RMS. The SI-SNR-trained model has
  arbitrary raw output gain; one gain per track avoids pumping at chunk joins.
  This is level calibration, not an estimate of each speaker's original loudness.
- Uses four-second windows with one-second overlap by default. Durations are
  configurable; overlap must be positive and shorter than the window.
- Matches speaker order by overlap similarity and crossfades boundaries.
- Preserves the final partial chunk and exact resampled duration.
- Returns silence without running the model for a fully silent input.
- Rejects empty, nonfinite or invalid audio, settings and unverified checkpoints.
- Writes two float32 WAV tracks and a run metadata record per input. Existing
  output folders are rejected rather than overwritten.

Speaker labels refer to the first chunk, not identified people. Silence,
speaker handoffs and ambiguous overlaps may still cause identity swaps. Two
tracks are returned even for one speaker. The CLI does not invoke FFmpeg or
process video; directory mode scans WAV/FLAC files.

The approximately 670 MB checkpoint downloads on first use and can be reused
with `--offline`. Its revision, hash and licenses are recorded in
[runtime notices](../RUNTIME_NOTICES.md). CPU and compatible CUDA devices are
supported; the validation below used CUDA.

## Validation, 2026-10-29

- 21 audio-handling tests passed, including swaps, silence, resampling, short
  clips, final partial chunks and checkpoint rejection.
- Strict weights-only checkpoint loading succeeded on RTX 5070 Ti with
  PyTorch 2.8.0+cu128.
- The adapted full model matched the original selected upstream architecture
  on a seeded two-second input: maximum absolute difference 0.0.
- A real upstream demo mixture produced two finite tracks, each exactly 70,537
  samples at 16 kHz, through the overlapping-chunk path.
- The installed wheel's CLI completed the same real-audio job offline.
- Model binaries, demo audio and generated evaluation outputs are not bundled
  in the source or wheel/sdist.

These checks establish functional and numerical behavior, not perceptual quality
or equivalence to a joint restoration/separation system. Representative generated
audio needs a listening comparison before claiming improvement.
