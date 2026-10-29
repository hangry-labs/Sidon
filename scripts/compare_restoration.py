# SPDX-License-Identifier: Apache-2.0
"""Compare resident latency and numerical parity on operator-provided audio."""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import soundfile as sf
import torch

from sidon.inference.native import NATIVE_MIN_FRAMES, fp32_execution
from sidon.inference.restoration import SidonRestorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seconds", type=float, nargs="+")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--candidate", choices=["auto", "native"], default="auto")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or any(not math.isfinite(v) or v <= 0 for v in args.seconds or []):
        parser.error("Repeat and durations must be positive and finite")
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size:
        parser.error("Input audio is empty")
    reference = SidonRestorer("cuda:0", backend="torchscript")
    native = SidonRestorer("cuda:0", backend=args.candidate)
    if native.backend == "torchscript-fp32":
        raise RuntimeError("The optimized CUDA backend is unavailable")

    def quality_forward(function, axis):
        def call(value):
            if native.backend == "native-fp32" or value.shape[axis] >= NATIVE_MIN_FRAMES:
                with fp32_execution(value.device):
                    return function(value)
            return function(value)
        return call

    rows = []
    for seconds in args.seconds or [len(audio) / sr]:
        count = max(1, round(seconds * sr))
        signal = np.tile(audio, (count + len(audio) - 1) // len(audio))[:count]
        encoder, decoder = reference.encoder, reference.decoder
        try:
            reference.encoder = quality_forward(encoder, 1)
            reference.decoder = quality_forward(decoder, 2)
            expected, rate = reference.enhance(signal, sr)
        finally:
            reference.encoder, reference.decoder = encoder, decoder
        actual, actual_rate = native.enhance(signal, sr)
        assert actual_rate == rate == 48000 and actual.shape == expected.shape and np.isfinite(actual).all()
        error = actual.astype(np.float64) - expected
        rms = float(np.sqrt(np.mean(error ** 2)))
        signal_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
        exact = not np.any(error)
        snr = None if exact else 20 * math.log10(max(signal_rms, 1e-30) / max(rms, 1e-30))
        maximum = float(np.abs(error).max())
        # A strict F32 parity gate, not a perceptual quality score.
        assert (exact or snr >= 80) and maximum <= .001, (seconds, snr, maximum)
        previous, _ = reference.enhance(signal, sr)
        previous_error = actual.astype(np.float64) - previous
        previous_exact = not np.any(previous_error)
        previous_rms = float(np.sqrt(np.mean(previous.astype(np.float64) ** 2)))
        previous_error_rms = float(np.sqrt(np.mean(previous_error ** 2)))
        previous_snr = None if previous_exact else 20 * math.log10(max(previous_rms, 1e-30) / max(previous_error_rms, 1e-30))
        timings, working = {}, {}
        for name, model in (("reference", reference), ("native", native)):
            model.enhance(signal, sr)  # discard first-use kernels for this shape
            values, peaks = [], []
            for _ in range(args.repeat):
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                model.enhance(signal, sr)
                torch.cuda.synchronize()
                values.append(time.perf_counter() - start)
                peaks.append(max(0, torch.cuda.max_memory_allocated() - allocated) / 2**20)
            timings[name] = statistics.median(values)
            working[name] = max(peaks)
        row = dict(seconds=count / sr, median_seconds=timings, working_mib=working,
                   speedup=timings["reference"] / timings["native"],
                   fp32_snr_db=snr, bit_exact=exact, max_abs_error=maximum, output_samples=len(actual),
                   previous_production=dict(bit_exact=previous_exact, snr_db=previous_snr,
                                            max_abs_error=float(np.abs(previous_error).max())))
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(python=platform.python_version(), torch=torch.__version__,
                  candidate=native.backend,
                  gpu=torch.cuda.get_device_name(), rows=rows,
                  timing_reference="Previous production precision policy",
                  quality_reference="Same TorchScript pipeline with matching per-window precision policy")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
