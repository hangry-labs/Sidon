# SPDX-License-Identifier: Apache-2.0 AND MIT
# Feature preparation adapted from sarulab-speech/Sidon (MIT).
# See licenses/Sidon-upstream-MIT.txt.
"""Inference-only restoration with the publisher's pinned TorchScript exports."""
from __future__ import annotations

import math
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly

from .assets import resolve_asset
from .mossformer import prepare_audio


def features(wave: torch.Tensor) -> torch.Tensor:
    mel = torchaudio.compliance.kaldi.fbank(
        wave.unsqueeze(0), sample_frequency=16000, num_mel_bins=80,
        frame_length=25, frame_shift=10, dither=0.0, energy_floor=1.192092955e-7,
        window_type="povey", use_energy=False,
    )
    mel = (mel - mel.mean(0)) / torch.sqrt(mel.var(0) + 1e-5)
    if mel.shape[0] % 2:
        mel = F.pad(mel, (0, 0, 0, 1))
    return mel.reshape(1, -1, 160)


class SidonRestorer:
    def __init__(self, device: str = "auto", *, backend: str | None = None):
        self.device = torch.device("cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        requested = backend or os.environ.get("SIDON_RESTORATION_BACKEND", "auto")
        if requested not in {"auto", "native", "torchscript"}:
            raise ValueError("Restoration backend must be auto, native or torchscript")
        from .native import available, load_native

        if requested == "native" and (self.device.type != "cuda" or not available()):
            raise RuntimeError("The native backend requires CUDA, Triton and a C compiler on Linux")
        if requested == "native" or (requested == "auto" and self.device.type == "cuda" and available()):
            encoder_path = resolve_asset("feature_extractor_cuda.pt")
            decoder_path = resolve_asset("decoder_cuda.pt")
            try:
                adaptive = requested == "auto" and self.device.index in {None, 0}
                self.encoder, self.decoder = load_native(encoder_path, decoder_path, self.device, adaptive=adaptive)
                self.backend = "adaptive-fp32" if adaptive else "native-fp32"
                return
            except Exception:
                if requested == "native":
                    raise
                logging.getLogger("sidon").exception("Native restoration initialization failed; using the reference exports")
        if self.device.type == "cuda" and self.device.index not in {None, 0}:
            raise ValueError("The published CUDA export requires logical cuda:0. Use CUDA_VISIBLE_DEVICES to choose a physical GPU.")
        suffix = "cuda" if self.device.type == "cuda" else "cpu"
        self.encoder = torch.jit.load(str(resolve_asset(f"feature_extractor_{suffix}.pt")), map_location=self.device).eval()
        self.decoder = torch.jit.load(str(resolve_asset(f"decoder_{suffix}.pt")), map_location=self.device).eval()
        self.backend = "torchscript-fp32"

    @torch.inference_mode()
    def enhance(self, audio: np.ndarray, sample_rate: int, *, strength: float = 1.0, cancel_check=None) -> tuple[np.ndarray, int]:
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Strength must be between zero and one")
        wave = prepare_audio(audio, sample_rate)
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        common = math.gcd(int(sample_rate), 48000)
        original = resample_poly(mono, 48000 // common, int(sample_rate) // common).astype(np.float32)
        target_length = round(len(mono) * 48000 / sample_rate)
        original = original[:target_length]
        peak = wave.abs().max().item()
        if peak < 1e-8 or strength == 0:
            return original, 48000
        wave = torchaudio.functional.highpass_biquad(wave * (0.9 / peak), 16000, 50)
        # Overlap-add keeps memory bounded and avoids hard cuts on long inputs.
        window, overlap = 16 * 16000, 16000
        result = torch.zeros(wave.numel() * 3)
        weights = torch.zeros_like(result)
        for start in range(0, wave.numel(), window - overlap):
            if cancel_check:
                cancel_check()
            length = min(window, wave.numel() - start)
            # The export needs right context, as in the publisher's demo.
            chunk = F.pad(wave[start:start + length], (0, 24000))
            inputs = features(F.pad(chunk, (160, 160))).to(self.device)
            # These publisher exports are already frozen. The profiling executor
            # otherwise spends seconds optimizing early calls and new lengths,
            # while the actual forward takes milliseconds. Keep the policy local
            # to these calls so other models retain their own execution settings.
            with torch.jit.optimized_execution(False):
                encoded = self.encoder(inputs)["last_hidden_state"]
                output = self.decoder(encoded.transpose(1, 2)).reshape(-1)[:-960].float().cpu()
            if output.numel() < length * 3:
                raise RuntimeError("Restoration export returned too little audio")
            output = output[:length * 3] * (peak / 0.9)
            if not torch.isfinite(output).all():
                raise RuntimeError("Restoration produced nonfinite audio")
            fade = torch.ones_like(output)
            edge = min(overlap * 3, output.numel())
            if start:
                fade[:edge] *= torch.linspace(0, 1, edge)
            if start + length < wave.numel():
                fade[-edge:] *= torch.linspace(1, 0, edge)
            position = start * 3
            result[position:position + output.numel()] += output * fade
            weights[position:position + output.numel()] += fade
            if start + length == wave.numel():
                break
        enhanced = (result / weights.clamp_min(1e-8)).numpy()[:target_length]
        return enhanced * strength + original * (1 - strength), 48000
