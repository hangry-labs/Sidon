# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hangry Labs contributors.
"""Audio correctness and packaging boundaries for the separation runtime."""
from pathlib import Path

import numpy as np
import pytest
import torch

from sidon.inference.mossformer import prepare_audio, resolve_checkpoint, separate_chunks


class AlternatingSeparator(torch.nn.Module):
    """Simulate a model swapping its two output channels at every window."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, waveform):
        self.calls += 1
        sources = [waveform, -waveform]
        return sources if self.calls % 2 else sources[::-1]


@pytest.mark.parametrize("length", [1, 15, 1599, 1600, 1601, 2799, 4001, 10000])
def test_chunking_preserves_tail_and_aligns_swapped_speakers(length):
    waveform = torch.sin(torch.arange(length) * 0.04) * 0.3
    model = AlternatingSeparator()
    output = separate_chunks(model, waveform, device=torch.device("cpu"),
                             chunk_seconds=0.1, overlap_seconds=0.025)
    assert output.shape == (2, length)
    torch.testing.assert_close(output[0], waveform)
    torch.testing.assert_close(output[1], -waveform)
    expected_calls = 0 if length == 1 else 1 + (max(0, length - 1600) + 1199) // 1200
    assert model.calls == expected_calls


def test_silence_needs_no_model_calls():
    model = AlternatingSeparator()
    output = separate_chunks(model, torch.zeros(12345), device=torch.device("cpu"))
    assert model.calls == 0
    assert output.shape == (2, 12345)
    assert torch.count_nonzero(output) == 0


def test_resampling_preserves_duration_and_downmixes_without_peak_normalizing():
    stereo = np.full((44100, 2), [0.1, 0.3], dtype=np.float32)
    audio = prepare_audio(stereo, 44100)
    assert audio.shape == (16000,)
    np.testing.assert_allclose(audio[100:-100].numpy().mean(), 0.2, atol=1e-4)


@pytest.mark.parametrize("audio,rate", [(np.array([]), 16000), (np.ones(2), 0),
    (np.array([np.nan]), 16000), (np.array([np.inf]), 16000), (np.ones((2, 2, 2)), 16000)])
def test_invalid_audio_is_rejected(audio, rate):
    with pytest.raises(ValueError):
        prepare_audio(audio, rate)


@pytest.mark.parametrize("chunk,overlap", [(0, 1), (1, 1), (1, 0), (1, -1), (float("nan"), 1)])
def test_invalid_chunk_parameters_fail_before_inference(chunk, overlap):
    model = AlternatingSeparator()
    with pytest.raises(ValueError):
        separate_chunks(model, torch.ones(100), device=torch.device("cpu"),
                        chunk_seconds=chunk, overlap_seconds=overlap)
    assert model.calls == 0


def test_unknown_checkpoint_bytes_are_rejected(tmp_path):
    path = tmp_path / "unverified.pt"
    path.write_bytes(b"not the audited checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint(path)
