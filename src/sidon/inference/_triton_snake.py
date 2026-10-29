# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hangry Labs contributors.
"""One F32 pointwise kernel, shared by all vocoder stages and input lengths."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["count", "length", "channels"])
def _snake(X, Alpha, Inverse, Y, count, length, channels, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    channel = (index // length) % channels
    x = tl.load(X + index, valid, other=0)
    alpha = tl.load(Alpha + channel, valid, other=0)
    inverse = tl.load(Inverse + channel, valid, other=0)
    sine = libdevice.sin(alpha * x)
    result = x + inverse * (sine * sine)
    tl.store(Y + index, result, valid)


def fused_snake(x: torch.Tensor, alpha: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("The fused activation requires contiguous F32 CUDA input")
    if x.ndim != 3 or alpha.shape != (1, x.shape[1], 1) or inverse.shape != alpha.shape:
        raise ValueError("Expected [batch, channels, time] and channel activation weights")
    if alpha.dtype != x.dtype or inverse.dtype != x.dtype or alpha.device != x.device or inverse.device != x.device or not alpha.is_contiguous() or not inverse.is_contiguous():
        raise ValueError("Activation weights must be F32 on the input device")
    output = torch.empty_like(x)
    _snake[(triton.cdiv(x.numel(), 512),)](
        x, alpha, inverse, output, x.numel(), x.shape[-1], x.shape[-2],
        BLOCK=512, enable_fp_fusion=False,
    )
    return output
