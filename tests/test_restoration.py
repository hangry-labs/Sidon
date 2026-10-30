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


def test_restoration_cancellation_stops_before_next_window():
    restorer = SidonRestorer.__new__(SidonRestorer)
    restorer.device = torch.device('cpu')
    calls = []
    def encoder(inputs):
        calls.append(1)
        return {'last_hidden_state':torch.zeros(1, inputs.shape[1], 1024)}
    def cancel():
        if calls:
            raise RuntimeError('cancelled between windows')
    restorer.encoder = encoder
    restorer.decoder = lambda hidden:torch.zeros(1, hidden.shape[-1]*960)
    with pytest.raises(RuntimeError, match='cancelled between windows'):
        restorer.enhance(np.sin(np.arange(32*16000)*.1), 16000, cancel_check=cancel)
    assert len(calls) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_export_execution_policy_does_not_leak_to_other_models(fail):
    restorer = SidonRestorer.__new__(SidonRestorer)
    restorer.device = torch.device("cpu")

    def encoder(inputs):
        assert not torch._C._get_graph_executor_optimize()
        if fail:
            raise RuntimeError("Export failed")
        return {"last_hidden_state": torch.zeros(1, inputs.shape[1], 1024)}

    def decoder(hidden):
        assert not torch._C._get_graph_executor_optimize()
        return torch.zeros(1, hidden.shape[-1] * 960)

    restorer.encoder, restorer.decoder = encoder, decoder
    with torch.jit.optimized_execution(True):
        if fail:
            with pytest.raises(RuntimeError, match="Export failed"):
                restorer.enhance(np.sin(np.arange(1600) * .1), 16000)
        else:
            result, rate = restorer.enhance(np.sin(np.arange(1600) * .1), 16000)
            assert rate == 48000 and len(result) == 4800 and np.isfinite(result).all()
        assert torch._C._get_graph_executor_optimize()
