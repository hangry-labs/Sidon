---
title: Hear Sidon restore a voice
permalink: /examples/
sidebar: false
wide: true
stylesheets:
  - /examples/comparison.css
scripts:
  - /examples/comparison.js
---

# Hear the difference

Compare three low-quality VoxCPMTTS samples before and after Sidon restoration.
Play either version in each pair to hear the difference.

## Spoken text

> Give your voice a cleaner finish with Sidon, local audio restoration that helps rough recordings sound their best.


## Sample 1

<div class="audio-comparison">
  <section class="comparison-card" aria-labelledby="sample-1-before-title">
    <span class="comparison-label">Before</span>
    <h3 id="sample-1-before-title">Original · low quality</h3>
    <p>VoxCPMTTS output before restoration.</p>
    <audio controls preload="metadata" aria-label="Sample 1: Original · low quality" src="audio/voxcpmtts-sample-1-before-restoration-low-quality.opus"></audio>
    <a href="audio/voxcpmtts-sample-1-before-restoration-low-quality.opus" download>Download sample 1 · Opus</a>
  </section>
  <section class="comparison-card" aria-labelledby="sample-1-after-title">
    <span class="comparison-label">After Sidon</span>
    <h3 id="sample-1-after-title">Restored voice</h3>
    <p>The recording with Sidon applied.</p>
    <audio controls preload="metadata" aria-label="Sample 1: Restored voice" src="audio/voxcpmtts-sample-1-after-sidon-restoration.opus"></audio>
    <a href="audio/voxcpmtts-sample-1-after-sidon-restoration.opus" download>Download sample 1 · Opus</a>
  </section>
</div>


## Sample 2

<div class="audio-comparison">
  <section class="comparison-card" aria-labelledby="sample-2-before-title">
    <span class="comparison-label">Before</span>
    <h3 id="sample-2-before-title">Original · low quality</h3>
    <p>VoxCPMTTS output before restoration.</p>
    <audio controls preload="metadata" aria-label="Sample 2: Original · low quality" src="audio/voxcpmtts-sample-2-before-restoration-low-quality.opus"></audio>
    <a href="audio/voxcpmtts-sample-2-before-restoration-low-quality.opus" download>Download sample 2 · Opus</a>
  </section>
  <section class="comparison-card" aria-labelledby="sample-2-after-title">
    <span class="comparison-label">After Sidon</span>
    <h3 id="sample-2-after-title">Restored voice</h3>
    <p>The recording with Sidon applied.</p>
    <audio controls preload="metadata" aria-label="Sample 2: Restored voice" src="audio/voxcpmtts-sample-2-after-sidon-restoration.opus"></audio>
    <a href="audio/voxcpmtts-sample-2-after-sidon-restoration.opus" download>Download sample 2 · Opus</a>
  </section>
</div>


## Sample 3

<div class="audio-comparison">
  <section class="comparison-card" aria-labelledby="sample-3-before-title">
    <span class="comparison-label">Before</span>
    <h3 id="sample-3-before-title">Original · low quality</h3>
    <p>VoxCPMTTS output before restoration.</p>
    <audio controls preload="metadata" aria-label="Sample 3: Original · low quality" src="audio/voxcpmtts-sample-3-before-restoration-low-quality.opus"></audio>
    <a href="audio/voxcpmtts-sample-3-before-restoration-low-quality.opus" download>Download sample 3 · Opus</a>
  </section>
  <section class="comparison-card" aria-labelledby="sample-3-after-title">
    <span class="comparison-label">After Sidon</span>
    <h3 id="sample-3-after-title">Restored voice</h3>
    <p>The recording with Sidon applied.</p>
    <audio controls preload="metadata" aria-label="Sample 3: Restored voice" src="audio/voxcpmtts-sample-3-after-sidon-restoration.opus"></audio>
    <a href="audio/voxcpmtts-sample-3-after-sidon-restoration.opus" download>Download sample 3 · Opus</a>
  </section>
</div>

Only one recording plays at a time, including across samples. Use the players to
pause, seek and adjust volume.

<details markdown="1">
<summary>About these recordings</summary>

All recordings are mono at 48 kHz. Sample 1 is approximately 7.7 seconds long;
samples 2 and 3 are approximately 5.8 seconds each. Both original and restored
recordings are served as Opus, encoded with the same 96 kbps variable-bitrate
settings. Their decoded durations differ by about two milliseconds, retained
from the supplied files. No gallery loudness
matching was applied. These are listening examples, not a general quality benchmark.

</details>

Sidon resynthesizes speech and may change timbre or introduce artifacts. Compare
your own voices and settings. See the [studio overview](../README.md),
[performance measurements](../docs/performance.md) and
[runtime qualification](../docs/validation.md).
