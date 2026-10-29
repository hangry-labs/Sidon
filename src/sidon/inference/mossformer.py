# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hangry Labs contributors.
"""Two-speaker separation with the pinned Apache-2.0 MossFormer2 SS checkpoint."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

MODEL_REPO = "alibabasglab/MossFormer2_SS_16K"
MODEL_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
MODEL_FILENAME = "last_best_checkpoint.pt"
MODEL_SHA256 = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
SAMPLE_RATE = 16_000


def resolve_checkpoint(checkpoint: str | Path | None = None, *, offline: bool = False) -> Path:
    """Resolve only the audited asset and verify its bytes before loading it."""
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            MODEL_REPO, MODEL_FILENAME, revision=MODEL_REVISION,
            local_files_only=offline,
        )
    path = Path(checkpoint)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}. Expected the pinned {MODEL_REPO} asset.")
    return path


def prepare_audio(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    """Accept mono or soundfile-style [time, channels], retaining input gain."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2 and signal.shape[1] > 0:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or not signal.size:
        raise ValueError("Expected nonempty mono audio or [time, channels] audio")
    if not np.isfinite(signal).all():
        raise ValueError("Audio contains NaN or infinite samples")
    if sample_rate != SAMPLE_RATE:
        common = math.gcd(int(sample_rate), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // common, int(sample_rate) // common)
    return torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))


def _similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    norms = first.norm(dim=-1) * second.norm(dim=-1)
    return (first * second).sum(dim=-1) / norms.clamp_min(1e-12)


@torch.inference_mode()
def separate_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    *,
    device: torch.device,
    chunk_seconds: float = 4.0,
    overlap_seconds: float = 1.0,
) -> torch.Tensor:
    """Return [2, time] at 16 kHz; align speakers and crossfade all overlaps.

    Labels are relative to the first chunk, not identified people. Silent or
    ambiguous overlaps cannot guarantee consistent identity assignment.
    """
    if not math.isfinite(chunk_seconds) or not math.isfinite(overlap_seconds):
        raise ValueError("Chunk and overlap durations must be finite")
    if chunk_seconds <= 0 or overlap_seconds <= 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Require 0 < overlap_seconds < chunk_seconds")
    window = round(chunk_seconds * SAMPLE_RATE)
    overlap = round(overlap_seconds * SAMPLE_RATE)
    if window < 16 or overlap < 1 or overlap >= window:
        raise ValueError("Chunk/overlap durations are too small at 16 kHz")
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    waveform = waveform.detach().float().cpu()
    total = waveform.numel()
    output = torch.zeros(2, total)
    if waveform.abs().max() == 0:
        return output
    previous_end = 0
    model.eval()
    for start in range(0, total, window - overlap):
        end = min(start + window, total)
        chunk = torch.nn.functional.pad(waveform[start:end], (0, window - (end - start)))
        if chunk.abs().max() == 0:
            predicted = torch.zeros(2, end - start)
        else:
            sources = model(chunk.unsqueeze(0).to(device))
            if len(sources) != 2:
                raise ValueError("MossFormer2 must return exactly two sources")
            predicted = torch.stack([source.reshape(-1).cpu() for source in sources])
            if predicted.shape[1] < end - start or not torch.isfinite(predicted).all():
                raise ValueError("Separator returned invalid or truncated audio")
            predicted = predicted[:, :end - start]
        shared = max(0, previous_end - start)
        if shared:
            previous = output[:, start:start + shared]
            current = predicted[:, :shared]
            direct = _similarity(previous, current).sum()
            swapped = _similarity(previous, current.flip(0)).sum()
            if swapped > direct:
                predicted = predicted.flip(0)
            fade = torch.linspace(0, 1, shared).unsqueeze(0)
            output[:, start:start + shared] = previous * (1 - fade) + predicted[:, :shared] * fade
        output[:, start + shared:end] = predicted[:, shared:]
        previous_end = end
        if end == total:
            break
    return output


class MossFormer2Separator:
    """Load only the inference architecture and the checksum-verified weights."""

    def __init__(self, *, device: str = "auto", checkpoint: str | Path | None = None, offline: bool = False):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        from ._mossformer2.mossformer2 import MossFormer

        self.model = MossFormer()
        state = torch.load(resolve_checkpoint(checkpoint, offline=offline), map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        # A partial load would silently mix trained and random weights.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def separate(self, audio: np.ndarray, sample_rate: int, *, chunk_seconds: float = 4.0,
                 overlap_seconds: float = 1.0) -> tuple[np.ndarray, int]:
        waveform = prepare_audio(audio, sample_rate)
        result = separate_chunks(self.model, waveform, device=self.device,
                                 chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
        return result.numpy(), SAMPLE_RATE
