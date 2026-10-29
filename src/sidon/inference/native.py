# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2026 Hangry Labs contributors.
# Tensor binding adapted from audio.cpp revision 1f5dc02257379b790fae9f3a0e0d28065d9baf2f,
# Copyright 2026 ShugoAI LLC, Apache-2.0 (licenses/audio.cpp-Apache-2.0.txt).
# The architecture/weights retain Sidon's MIT grant (licenses/Sidon-upstream-MIT.txt).
"""Inference-only F32 implementation of the two pinned Sidon CUDA exports.

Adapters are folded once at model load. No training/model-repository Python is loaded.
The service serializes model execution; precision flags are scoped and restored.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil
import sys

import torch
from torch import nn
from torch.nn import functional as F

NATIVE_MIN_FRAMES = 512


def available() -> bool:
    return sys.platform == "linux" and importlib.util.find_spec("triton") is not None and bool(shutil.which("cc") or shutil.which("gcc"))


@contextmanager
def fp32_execution(device: torch.device):
    """Use IEEE F32 arithmetic without changing subsequent models' policies."""
    with torch.autocast(device_type=device.type, enabled=False):
        if device.type != "cuda":
            yield
            return
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                benchmark_limit=torch.backends.cudnn.benchmark_limit,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False,
            ):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


def fold_lora(base: torch.Tensor, left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
    if base.dtype != torch.float32 or left.dtype != base.dtype or right.dtype != base.dtype:
        raise ValueError("Adapter folding requires F32 weights")
    if base.ndim != 2 or left.ndim != 2 or right.ndim != 2 or (left.shape[0], right.shape[1]) != base.shape or left.shape[1] != right.shape[0]:
        raise ValueError("Adapter dimensions do not match the base weight")
    with torch.autocast(device_type=base.device.type, enabled=False):
        return base + scale * (left @ right)


def extract_weights(encoder_path: Path, decoder_path: Path, *, exports=None, share=False) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Bind the audited frozen layout. Callers verify asset hashes before loading."""
    if exports is None:
        encoder = torch.jit.load(str(encoder_path), map_location="cpu")
        decoder = torch.jit.load(str(decoder_path), map_location="cpu")
    else:
        encoder, decoder = exports
    ec = encoder.code_with_constants[1].const_mapping
    dc = decoder.code_with_constants[1].const_mapping
    if len(ec) != 264 or len(dc) != 146:
        raise ValueError("Unsupported Sidon export layout; use the TorchScript backend")
    weights: dict[str, torch.Tensor] = {}
    used: set[int] = set()

    def get(index, shape):
        value = ec[f"c{index}"]
        if tuple(value.shape) != shape or (shape and value.dtype != torch.float32):
            raise ValueError(f"Unexpected encoder constant c{index}")
        used.add(index)
        return value

    def norm(name, index, channels=1024):
        weights[name + ".weight"] = get(index, (channels,))
        weights[name + ".bias"] = get(index + 1, (channels,))

    def linear(name, index, inputs, outputs, adapter=None):
        value = get(index, (inputs, outputs))
        if adapter is not None:
            value = fold_lora(value, get(adapter, (inputs, 64)), get(adapter + 1, (64, outputs)), get(12, ()).item())
        weights[name + ".weight"] = value.T if share else value.T.contiguous()
        weights[name + ".bias"] = get(index + 1, (outputs,))

    for index, expected in ((12, .25), (13, .5), (18, 8.), (19, 64.)):
        if get(index, ()).item() != expected:
            raise ValueError(f"Unexpected encoder scale c{index}")
    norm("feature.norm", 0, 160)
    linear("feature.projection", 2, 160, 1024)
    for layer in range(8):
        ids = ([4, 6, 8, 10, 14, 16, 20, 21, 23, 25, 26, 27, 29, 30, 32, 34, 36, 38] if layer == 0 else
               [40 + 32 * (layer - 1) + i for i in (0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 18, 19, 21, 22, 24, 26, 28, 30)])
        p = f"layers.{layer}."
        norm(p + "ff1.norm", ids[0])
        linear(p + "ff1.in", ids[1], 1024, 4096)
        linear(p + "ff1.out", ids[2], 4096, 1024, ids[3])
        norm(p + "attention.norm", ids[4])
        linear(p + "attention.qkv", ids[5], 1024, 3072)
        weights[p + "attention.relative"] = get(ids[6], (73, 64))
        linear(p + "attention.out", ids[7], 1024, 1024)
        norm(p + "conv.norm", ids[8])
        weights[p + "conv.in"] = get(ids[9], (2048, 1024, 1)).squeeze(-1)
        weights[p + "conv.depth"] = get(ids[10], (1024, 1, 31))
        norm(p + "conv.depth_norm", ids[11])
        weights[p + "conv.out"] = get(ids[12], (1024, 1024, 1)).squeeze(-1)
        norm(p + "ff2.norm", ids[13])
        linear(p + "ff2.in", ids[14], 1024, 4096)
        linear(p + "ff2.out", ids[15], 4096, 1024, ids[16])
        norm(p + "final", ids[17])
    if used != set(range(264)):
        raise ValueError("Some encoder constants were not bound")

    def check_decoder(index, shape):
        if tuple(dc[f"c{index}"].shape) != shape or dc[f"c{index}"].dtype != torch.float32:
            raise ValueError(f"Unexpected decoder constant c{index}")

    check_decoder(0, (1536, 1024, 7)); check_decoder(1, (1536,))
    channels = 1536
    for stage, stride in enumerate((8, 5, 4, 3, 2)):
        base = 2 + 28 * stage
        for index in (base, base + 1):
            check_decoder(index, (1, channels, 1))
        check_decoder(base + 2, (channels, channels // 2, stride * 2))
        channels //= 2
        check_decoder(base + 3, (channels,))
        for unit in range(3):
            offset = base + 4 + unit * 8
            for index in (offset, offset + 1, offset + 4, offset + 5):
                check_decoder(index, (1, channels, 1))
            check_decoder(offset + 2, (channels, channels, 7))
            check_decoder(offset + 3, (channels,))
            check_decoder(offset + 6, (channels, channels, 1))
            check_decoder(offset + 7, (channels,))
    check_decoder(142, (1, channels, 1)); check_decoder(143, (1, channels, 1))
    check_decoder(144, (1, channels, 7)); check_decoder(145, (1,))
    if share:
        return ({k: v.detach() for k, v in weights.items()}, {k: v.detach() for k, v in dc.items()})
    # Clone to release the ScriptModules and their unused adapter tensors.
    return ({k: v.detach().clone().contiguous() for k, v in weights.items()},
            {k: v.detach().clone().contiguous() for k, v in dc.items()})


class _Weights(nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        for name, value in weights.items():
            self.register_buffer(name.replace(".", "__"), value)

    def weight(self, name):
        return getattr(self, name.replace(".", "__"))

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.weight(name + ".weight"), self.weight(name + ".bias"), 1e-5)

    def linear(self, x, name):
        return F.linear(x, self.weight(name + ".weight"), self.weight(name + ".bias"))


def relative_bias(query: torch.Tensor, embedding: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Project against 73 relative positions before gathering pairwise bias."""
    table = F.linear(query, embedding)
    b, h, t, _ = query.shape
    return table.gather(-1, indices[None, None].expand(b, h, t, t)) * (query.shape[-1] ** -.5)


class NativeEncoder(_Weights):
    def __init__(self, weights):
        super().__init__(weights)
        self.register_buffer("distance_indices", torch.empty(0, dtype=torch.long), persistent=False)

    def forward(self, features):
        with fp32_execution(features.device):
            return self._forward(features.float())

    def _forward(self, features):
        b, t, _ = features.shape
        if self.distance_indices.shape != (t, t):
            positions = torch.arange(t, device=features.device)
            self.distance_indices = (positions[None, :] - positions[:, None]).clamp(-64, 8) + 64
        x = self.linear(self.norm(features, "feature.norm"), "feature.projection")
        for layer in range(8):
            p = f"layers.{layer}."
            x = x + .5 * self.linear(F.silu(self.linear(self.norm(x, p + "ff1.norm"), p + "ff1.in")), p + "ff1.out")
            q, k, v = self.linear(self.norm(x, p + "attention.norm"), p + "attention.qkv").reshape(b, t, 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
            bias = relative_bias(q, self.weight(p + "attention.relative"), self.distance_indices)
            attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0., scale=.125)
            x = x + self.linear(attention.transpose(1, 2).reshape(b, t, 1024), p + "attention.out")
            del q, k, v, bias, attention
            hidden = F.glu(F.linear(self.norm(x, p + "conv.norm"), self.weight(p + "conv.in")), -1).transpose(1, 2)
            hidden = F.conv1d(F.pad(hidden, (30, 0)), self.weight(p + "conv.depth"), groups=1024).transpose(1, 2)
            x = x + F.linear(F.silu(self.norm(hidden, p + "conv.depth_norm")), self.weight(p + "conv.out"))
            del hidden
            x = x + .5 * self.linear(F.silu(self.linear(self.norm(x, p + "ff2.norm"), p + "ff2.in")), p + "ff2.out")
            x = self.norm(x, p + "final")
        return {"last_hidden_state": x}


class NativeDecoder(_Weights):
    def __init__(self, weights, *, adaptive=False):
        super().__init__(weights)
        self.adaptive = adaptive

    def _snake(self, x, index):
        from ._triton_snake import fused_snake
        return fused_snake(x, self.weight(f"c{index}"), self.weight(f"c{index + 1}"))

    def _conv(self, x, index, dilation=1):
        weight = self.weight(f"c{index}")
        return F.conv1d(x, weight, self.weight(f"c{index + 1}"), padding=weight.shape[-1] // 2 * dilation, dilation=dilation)

    def warmup(self):
        from ._triton_snake import fused_snake
        device = self.weight("c0").device
        fused_snake(torch.zeros(1, 1, 512, device=device), torch.ones(1, 1, 1, device=device), torch.ones(1, 1, 1, device=device))
        torch.cuda.synchronize(device)

    def forward(self, x):
        if self.adaptive and x.shape[-1] < NATIVE_MIN_FRAMES:
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self._forward(x.float())
        with fp32_execution(x.device):
            return self._forward(x.float())

    def _forward(self, x):
        x = self._conv(x, 0)
        for stage, stride in enumerate((8, 5, 4, 3, 2)):
            base = 2 + 28 * stage
            x = F.conv_transpose1d(self._snake(x, base), self.weight(f"c{base + 2}"), self.weight(f"c{base + 3}"), stride=stride, padding=(stride + 1) // 2)
            for unit, dilation in enumerate((1, 3, 9)):
                index = base + 4 + unit * 8
                hidden = self._conv(self._snake(x, index), index + 2, dilation)
                x = x + self._conv(self._snake(hidden, index + 4), index + 6)
                del hidden
        return torch.tanh(self._conv(self._snake(x, 142), 144))


class AdaptiveEncoder(nn.Module):
    """Keep the exact short-input encoder where F32 rewriting doesn't pay off."""
    def __init__(self, reference, native):
        super().__init__()
        self.reference = reference
        self.native = native

    def forward(self, inputs):
        if inputs.shape[1] < NATIVE_MIN_FRAMES:
            return self.reference(inputs)
        return self.native(inputs)


def load_native(encoder_path: Path, decoder_path: Path, device: torch.device, *, adaptive=False):
    if device.type != "cuda" or not available():
        raise RuntimeError("The native backend requires CUDA, Triton and a C compiler on Linux")
    with torch.inference_mode():
        if adaptive:
            reference = torch.jit.load(str(encoder_path), map_location=device).eval()
            decoder_export = torch.jit.load(str(decoder_path), map_location=device).eval()
            with fp32_execution(device):
                encoder_weights, decoder_weights = extract_weights(encoder_path, decoder_path, exports=(reference, decoder_export), share=True)
            encoder = AdaptiveEncoder(reference, NativeEncoder(encoder_weights)).eval()
            decoder = NativeDecoder(decoder_weights, adaptive=True).eval()
            decoder.warmup()
            return encoder, decoder
        encoder_weights, decoder_weights = extract_weights(encoder_path, decoder_path)
        encoder = NativeEncoder(encoder_weights).to(device).eval()
        decoder = NativeDecoder(decoder_weights).to(device).eval()
        decoder.warmup()
        return encoder, decoder
