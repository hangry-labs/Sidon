# SPDX-License-Identifier: Apache-2.0
import importlib.util
import numpy as np
import pytest
import torch

from sidon.inference.native import AdaptiveEncoder, NATIVE_MIN_FRAMES, fold_lora, fp32_execution, relative_bias


def test_folded_adapter_matches_unfolded_forward_without_autocast():
    generator = torch.Generator().manual_seed(42)
    base = torch.randn(17, 11, generator=generator)
    left = torch.randn(17, 3, generator=generator)
    right = torch.randn(3, 11, generator=generator)
    x = torch.randn(2, 5, 17, generator=generator)
    expected = x @ base + .25 * ((x @ left) @ right)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        folded = fold_lora(base, left, right, .25)
    assert folded.dtype == torch.float32
    torch.testing.assert_close(x @ folded, expected, atol=4e-6, rtol=1e-5)


def test_adapter_shape_and_dtype_fail_closed():
    with pytest.raises(ValueError, match="dimensions"):
        fold_lora(torch.zeros(4, 3), torch.zeros(5, 2), torch.zeros(2, 3), .25)
    with pytest.raises(ValueError, match="F32"):
        fold_lora(torch.zeros(4, 3).half(), torch.zeros(4, 2), torch.zeros(2, 3), .25)


def test_adaptive_encoder_routes_short_long_and_short_again():
    class Mark(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def forward(self, x):
            return {"last_hidden_state": x + self.value}
    encoder = AdaptiveEncoder(Mark(1), Mark(2))
    for frames, expected in [(NATIVE_MIN_FRAMES - 1, 1), (NATIVE_MIN_FRAMES, 2), (3, 1)]:
        result = encoder(torch.zeros(1, frames, 4))["last_hidden_state"]
        torch.testing.assert_close(result, torch.full_like(result, expected))


def test_relative_projection_matches_pairwise_embedding_at_clamped_distances():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 3, 81, 4, generator=generator)
    embedding = torch.randn(73, 4, generator=generator)
    positions = torch.arange(81)
    indices = (positions[None, :] - positions[:, None]).clamp(-64, 8) + 64
    pairwise = embedding[indices]
    expected = torch.einsum("bhld,lrd->bhlr", query, pairwise) / 2
    actual = relative_bias(query, embedding, indices)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fail", [False, True])
def test_f32_scope_restores_precision_after_success_and_error(fail):
    matmul = torch.backends.cuda.matmul.allow_tf32
    conv = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    try:
        with fp32_execution(torch.device("cuda")):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.backends.cudnn.benchmark == benchmark
            if fail:
                raise RuntimeError("test")
    except RuntimeError:
        assert fail
    assert torch.backends.cuda.matmul.allow_tf32 == matmul
    assert torch.backends.cudnn.allow_tf32 == conv
    assert torch.backends.cudnn.benchmark == benchmark


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None, reason="CUDA/Triton unavailable")
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 17), (2, 7, 513), (1, 48, 4801)])
def test_fused_snake_matches_original_f32_across_channels_batches_and_tail(shape):
    from sidon.inference._triton_snake import fused_snake
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(*shape, device="cuda", generator=generator)
    alpha = torch.rand(1, shape[1], 1, device="cuda", generator=generator) * 10 + .1
    inverse = 1 / (alpha + 1e-9)
    expected = x + inverse * torch.sin(alpha * x).square()
    actual = fused_snake(x, alpha, inverse)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if shape[1] > 1:
        strided = torch.ones(1, shape[1] * 2, 1, device="cuda")[:, ::2]
        with pytest.raises(ValueError):
            fused_snake(x, strided, inverse)


def test_explicit_native_cpu_fails_before_downloading_assets(monkeypatch):
    from sidon.inference.restoration import SidonRestorer
    monkeypatch.setattr("sidon.inference.restoration.resolve_asset", lambda *a, **k: pytest.fail("Assets should not be downloaded"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        SidonRestorer("cpu", backend="native")


@pytest.mark.parametrize("backend", ["auto", "native"])
def test_unavailable_kernel_falls_back_only_in_automatic_mode(monkeypatch, backend):
    from pathlib import Path
    from sidon.inference.restoration import SidonRestorer
    class Export:
        def eval(self):
            return self
    def unavailable(*a, **kw):
        raise RuntimeError("Kernel unavailable")
    monkeypatch.setattr("sidon.inference.native.available", lambda: True)
    monkeypatch.setattr("sidon.inference.native.load_native", unavailable)
    monkeypatch.setattr("sidon.inference.restoration.resolve_asset", lambda name: Path(name))
    monkeypatch.setattr(torch.jit, "load", lambda *a, **kw: Export())
    if backend == "native":
        with pytest.raises(RuntimeError, match="Kernel unavailable"):
            SidonRestorer("cuda:0", backend=backend)
    else:
        assert SidonRestorer("cuda:0", backend=backend).backend == "torchscript-fp32"
