# Restoration latency and audio.cpp comparison

Investigated 2026-10-29 on an RTX 5070 Ti, Python 3.13 and PyTorch 2.10.0+cu128.
The comparison below profiles our own pipeline before/after a targeted fix. It is
not a same-machine speed benchmark against an audio.cpp executable.

These measurements describe the original CUDA 12.8 stack. The later security
upgrade uses PyTorch 2.14.1 / CUDA 13.0; its separate numerical and latency checks
are recorded in the [security follow-up](validation.md#dependency-security-follow-up-2026-10-30).

## What audio.cpp changes

Reviewed [audio.cpp revision 1f5dc02](https://github.com/0xShug0/audio.cpp/tree/1f5dc02257379b790fae9f3a0e0d28065d9baf2f):

- Its [Sidon conversion](https://github.com/0xShug0/audio.cpp/blob/1f5dc02257379b790fae9f3a0e0d28065d9baf2f/tests/sidon/convert_gguf.py)
  extracts the same released TorchScript weights into an F32 GGUF and folds LoRA
  adapters into base linear weights during conversion.
- Its [runtime](https://github.com/0xShug0/audio.cpp/blob/1f5dc02257379b790fae9f3a0e0d28065d9baf2f/src/models/sidon/runtime.cpp)
  retains weights and reuses its decoder graph for a matching frame count. It
  represents pointwise convolutions as linear operations and tiles large decoder
  inputs with context halos to bound working memory.
- Its [encoder](https://github.com/0xShug0/audio.cpp/blob/1f5dc02257379b790fae9f3a0e0d28065d9baf2f/src/framework/modules/speech_encoders/wav2vec2_bert_encoder.cpp)
  projects queries against the small relative-position embedding table before
  gathering pairwise attention bias. It can choose Flash Attention on a supported
  backend and uses linear operations for pointwise convolutions.
- Its [Sidon processing path](https://github.com/0xShug0/audio.cpp/blob/1f5dc02257379b790fae9f3a0e0d28065d9baf2f/src/models/sidon/session.cpp)
  uses 96-second encoder chunks with one-frame carry. Our restoration wrapper
  keeps 16-second windows and one-second crossfades to bound memory. Chunking and
  frontend details differ, so outputs need not match across implementations.

These are concrete execution and graph changes. There is no denoising-step or
sampler-count knob in this model's inference path. F32 weights are used by the
reviewed audio.cpp Sidon package; broad TTS speed claims from its README are not
Sidon-specific measurements.

## The measured problem in our runtime

The released exports are already frozen, but PyTorch's profiling executor runs
additional optimization passes on early calls and new input lengths. Stage timing
put almost all multi-second stalls inside the encoder and decoder. High-pass and
filterbank preparation took milliseconds on the short fixture.

We now scope `torch.jit.optimized_execution(False)` to the two export forwards.
This bypasses those extra passes. It preserves the weights, F32 precision,
preprocessing, chunking and output sample counts. The context is restored on exit,
including exceptions, so the separation model retains its own execution policy.

One isolated-process run per policy, using repeated real speech as input:

| Call | Input | Previous forward | Updated forward |
| --- | --- | --- | --- |
| First invocation after loading | 4.4 seconds | 3.827 s | 0.517 s |
| Second invocation | 4.4 seconds | 8.021 s | 0.078 s |
| Third invocation | 4.4 seconds | 0.063 s | 0.072 s |
| Repeated changed length | 4.825 seconds | 4.158 s | 0.078 s |
| Repeated longer length | 16 seconds | 5.005 s | 0.184 s |
| Multiple processing windows | 32 seconds | 0.352 s | 0.408 s |

Times exclude model loading, checksums, uploads, downloads, job polling and browser
decoding. They include stage synchronization for profiling. The small warm-call
overhead is the tradeoff for avoiding repeated multi-second optimization stalls.
This shared workstation was not an otherwise idle benchmark machine; rerun on
deployment hardware before treating these as capacity figures.

On tested first-shape calls, the updated output was bit-identical to the original
unprofiled cold output. The old optimized warm path differed slightly from that
cold output due to floating-point execution changes. Numerical checks do not
replace listening comparisons on representative generated voices.

Model loading remains a separate cost: reading/checking assets, CUDA initialization
and moving weights still take time after an unload or service restart. Keeping the
model resident avoids repeating those costs.

An end-to-end API check on a 4.826-second clip took 4.094 seconds on the first
request after restart, including model loading, then 0.081 and 0.079 seconds of
processing on subsequent requests. These figures include API-side processing;
they are distinct from the forward-only table above. A repeat under concurrent
CPU load took 5.826 seconds on startup and 0.099 seconds on each subsequent call.

The browser also waited up to one second between job-status polls. A warmed
0.097-second job took 1.225 seconds from clicking Process to populated results.
The UI now polls every 250 ms for the first ten seconds of a job, then backs off
to one second for longer jobs. Output transfer and browser decoding still add
their own latency. A controlled browser check using its monotonic clock measured
1.198 seconds with one-second polling and 0.452 seconds with the updated polling.

## Reproduce current measurements

```bash
uv run --locked python scripts/benchmark_restoration.py --input voice.wav --seconds 4.4 4.825 16 32 --repeat 3 --report measurements.json
```

Or copy the script and an authorized audio file into the image and run it there.
Use `--device cpu` for CPU measurements. Start a fresh process to measure the first
forward, and compare later calls separately. The script reports load time and
individual stage times; it does not retain input/output audio in its JSON report.

## Folded encoder and fused vocoder follow-up

The optimized CUDA path now implements the useful audio.cpp-style transformations
inside the existing PyTorch service. It binds only the pinned Sidon exports;
there are no new models, training code, FP16 weights or quantized weights.

- Sixteen LoRA output projections are folded once at load time.
- The long-window encoder projects queries against the 73-entry relative-position
  table before gathering attention bias. It uses fused QKV, SDPA and linear
  pointwise encoder convolutions.
- A custom F32 Triton kernel computes the vocoder's Snake activation in one
  operation. Floating-point contraction is disabled to match the original
  operation sequence. The fused kernel passed bit-exact CUDA tests across batches,
  channels and partial blocks.
- Vocoder pointwise-linear replacements, channels-last convolution and CUDA graph
  capture were tested and left out: their costs did not justify a default change
  on the qualified GPU.

Forcing strict IEEE F32 everywhere was slower on the shortest clips. `auto`
therefore keeps the reference encoder and original arithmetic below 512 latent
frames (about 8.7 seconds of input with the current context padding). Longer
windows use the folded encoder and strict F32 arithmetic. All windows use the
fused vocoder activation. All tensors/weights remain F32; shorter windows retain
the original cuDNN TF32 allowance. No precision flag is changed for other models.

Five-run warmed medians, same RTX 5070 Ti and process, repeated speech fixture:

| Input | Reference | Automatic optimized | Speedup |
| --- | --- | --- | --- |
| 4.825 s | 73.96 ms | 59.38 ms | 1.25× |
| 8 s | 95.57 ms | 74.12 ms | 1.29× |
| 9 s | 114.60 ms | 92.52 ms | 1.24× |
| 16 s | 172.58 ms | 144.51 ms | 1.19× |
| 32 s | 381.23 ms | 325.87 ms | 1.17× |

Short-window outputs were bit-identical to the reference on tested speech and TTS
clips. Longer/mixed windows measured 90–95 dB SNR against the unchanged TorchScript
operations under the same per-window precision policy; maximum absolute error
was below 0.000055 on the repeated fixture. A separate forced strict-F32 check
also passed. These are numerical parity checks, not a new perceptual quality
benchmark. Performance varies with GPU clocks, input and concurrent workloads.

The previous production path allowed TF32 convolution arithmetic. Comparing its
long-window outputs directly with the new default measured 45–48 dB SNR, with
maximum absolute differences up to 0.0112 on the repeated fixture. This is distinct
from implementation parity under matched arithmetic. Strict F32 avoids reduced
mantissa arithmetic, but these measurements alone do not establish perceptual
equivalence on generated voices. The comparison report includes both references;
`torchscript` remains available for listening comparisons with previous behavior.

The adaptive encoder shares unchanged weights with the reference but retains
about 256 MiB of additional folded matrices. Some working allocations increase on
short windows and decrease on long windows, so this is not a total-VRAM reduction
claim. The first model load also compiles a small activation/launcher kernel;
`/app/persistent/triton` caches it for later loads. Input lengths and channel counts
are runtime parameters, so each new clip length does not trigger recompilation.

`SIDON_RESTORATION_BACKEND=auto` selects this path on supported Linux CUDA hosts
with Triton and a C compiler, falling back to the exports if unavailable.
`torchscript` forces the previous path. `native` forces the folded encoder and
strict F32 for every window; it can be slower on short inputs. CPU and unsupported
platforms keep the reference path.

```bash
uv run --locked python scripts/compare_restoration.py --input voice.wav --seconds 4.4 4.825 8 9 16 32 --repeat 5 --report comparison.json
```

Use `--candidate native` to compare forced strict F32. The script gates parity at
80 dB SNR and 0.001 maximum absolute error, measures resident processing separately
from loading, and reports extra working allocations separately from resident
weights. Run it inside the qualified Docker environment when the host lacks Triton.
