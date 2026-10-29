# SPDX-License-Identifier: Apache-2.0
"""Measure model load and cold/warm restoration stages with your own audio."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import sidon.inference.restoration as restoration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seconds", type=float, nargs="+", help="Repeat/truncate the input to test these lengths; default: original length")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or any(not np.isfinite(s) or s <= 0 for s in args.seconds or []):
        parser.error("Repeat and durations must be positive and finite")
    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size:
        parser.error("The input audio is empty")
    start = time.perf_counter()
    model = restoration.SidonRestorer(args.device)
    load_seconds = time.perf_counter() - start
    use_cuda = model.device.type == "cuda"

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(model.device)

    stages = {}

    def measured(name, function):
        def call(*a, **kw):
            synchronize()
            start = time.perf_counter()
            result = function(*a, **kw)
            synchronize()
            stages[name] = stages.get(name, 0) + time.perf_counter() - start
            return result
        return call

    # Instrument only this benchmark process, preserving the product's policy.
    model.encoder = measured("encoder_seconds", model.encoder)
    model.decoder = measured("decoder_seconds", model.decoder)
    restoration.features = measured("features_seconds", restoration.features)
    restoration.torchaudio.functional.highpass_biquad = measured("highpass_seconds", restoration.torchaudio.functional.highpass_biquad)
    runs = []
    for seconds in args.seconds or [len(audio) / sample_rate]:
        count = max(1, round(seconds * sample_rate))
        signal = np.tile(audio, (count + len(audio) - 1) // len(audio))[:count]
        for repeat in range(args.repeat):
            stages.clear()
            synchronize()
            start = time.perf_counter()
            result, rate = model.enhance(signal, sample_rate)
            synchronize()
            elapsed = time.perf_counter() - start
            assert rate == 48000 and len(result) == round(count * 48000 / sample_rate)
            assert np.isfinite(result).all()
            row = dict(audio_seconds=count / sample_rate, repeat=repeat,
                       forward_seconds=elapsed, rtf=elapsed / (count / sample_rate),
                       stages=dict(stages), output_samples=len(result))
            runs.append(row)
            print(json.dumps(row), flush=True)
    report = dict(python=platform.python_version(), torch=torch.__version__,
                  device=str(model.device), threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name(model.device) if use_cuda else None,
                  load_seconds=load_seconds, runs=runs)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"load_seconds": load_seconds}), flush=True)


if __name__ == "__main__":
    main()
