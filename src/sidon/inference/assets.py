# SPDX-License-Identifier: Apache-2.0
"""Immutable model assets. Downloads never execute repository Python code."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

SIDON_REPO = "sarulab-speech/sidon-v0.1"
SIDON_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
ASSETS = {
    "feature_extractor_cpu.pt": (SIDON_REPO, SIDON_REVISION, "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019"),
    "decoder_cpu.pt": (SIDON_REPO, SIDON_REVISION, "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4"),
    "feature_extractor_cuda.pt": (SIDON_REPO, SIDON_REVISION, "c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce"),
    "decoder_cuda.pt": (SIDON_REPO, SIDON_REVISION, "a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc"),
    "last_best_checkpoint.pt": ("alibabasglab/MossFormer2_SS_16K", "407cb030cd66340918ebb6c8cc63b18f8592cdbe", "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"),
}


def resolve_asset(name: str, *, offline: bool | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    repo, revision, expected = ASSETS[name]
    directory = os.environ.get("SIDON_MODEL_DIR")
    baked = Path(directory, name) if directory else None
    if baked and baked.is_file():
        path = baked
    else:
        if offline is None:
            offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true"}
        path = Path(hf_hub_download(repo, name, revision=revision, local_files_only=offline))
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {name}; remove the damaged cached file and retry.")
    return path
