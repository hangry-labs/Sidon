# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hangry Labs contributors.
"""File and directory CLI for the commercial-use MossFormer2 separation path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from .mossformer import MODEL_REPO, MODEL_REVISION, MODEL_SHA256, MossFormer2Separator


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate two voices with MossFormer2 SS (16 kHz).")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Input audio supported by libsndfile")
    inputs.add_argument("--input-dir", type=Path, help="Directory of WAV/FLAC files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the audited checkpoint (checksum required)")
    parser.add_argument("--offline", action="store_true", help="Use local/cached weights; never download")
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.input is not None:
        paths = [args.input]
    else:
        if not args.input_dir.is_dir():
            parser.error("--input-dir must be an existing directory")
        paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {".wav", ".flac"} and p.is_file())
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No input audio files found")
    # Separate folders prevent equal stems from different formats colliding.
    destinations = [args.output_dir / path.name for path in paths]
    for dest in destinations:
        if dest.exists():
            parser.error(f"Output already exists: {dest}; choose another output directory")
    separator = MossFormer2Separator(device=args.device, checkpoint=args.checkpoint, offline=args.offline)
    for path, dest in zip(paths, destinations):
        audio, sample_rate = sf.read(path, dtype="float32")
        separated, out_rate = separator.separate(audio, sample_rate,
                chunk_seconds=args.chunk_seconds, overlap_seconds=args.overlap_seconds)
        dest.mkdir(parents=True)
        for speaker in range(2):
            sf.write(dest / f"speaker_{speaker + 1}.wav", separated[speaker], out_rate, subtype="FLOAT")
        metadata = {"input": path.name, "input_sample_rate": sample_rate,
                    "output_sample_rate": out_rate, "output_samples": separated.shape[1],
                    "model": MODEL_REPO, "revision": MODEL_REVISION, "sha256": MODEL_SHA256,
                    "chunk_seconds": args.chunk_seconds, "overlap_seconds": args.overlap_seconds,
                    "speaker_labels": "Relative to first chunk; ambiguous overlaps may swap identities"}
        (dest / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{path} -> {dest} (2 tracks, {out_rate} Hz)")


if __name__ == "__main__":
    main()
