# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from sidon.inference.restoration import SidonRestorer, features


def test_features_match_export_shape_and_are_finite():
    value = features(torch.sin(torch.arange(16000) * .04))
    assert value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 160
    assert torch.isfinite(value).all()


def test_silence_and_zero_mix_do_not_invoke_models():
    restorer = SidonRestorer.__new__(SidonRestorer)
    for audio, strength in [(np.zeros(101), 1), (np.sin(np.arange(101)), 0)]:
        result, rate = restorer.enhance(audio, 44100, strength=strength)
        assert rate == 48000 and len(result) == round(101 * 48000 / 44100)
        assert np.isfinite(result).all()
    for strength in (float("nan"), float("inf"), -1, 2):
        with pytest.raises(ValueError):
            restorer.enhance(np.ones(100), 16000, strength=strength)


def test_asset_digest_rejects_changed_bytes(tmp_path, monkeypatch):
    from sidon.inference.assets import resolve_asset
    monkeypatch.setenv("SIDON_MODEL_DIR", str(tmp_path))
    (tmp_path / "decoder_cpu.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_asset("decoder_cpu.pt", offline=True)
