# Restoration latency and audio.cpp comparison

Investigated 2026-10-29 on an RTX 5070 Ti, Python 3.13 and PyTorch 2.10.0+cu128.
The comparison below profiles our own pipeline before/after a targeted fix. It is
not a same-machine speed benchmark against an audio.cpp executable.

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
