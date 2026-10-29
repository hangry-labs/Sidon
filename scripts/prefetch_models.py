# SPDX-License-Identifier: Apache-2.0
"""Bake the exact CPU/CUDA restoration exports and separator into an image."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sidon.inference.assets import ASSETS, resolve_asset

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
manifest = []
for name, (repo, revision, sha256) in ASSETS.items():
    path = resolve_asset(name, offline=False)
    shutil.copyfile(path, destination / name)
    manifest.append(dict(file=name, repository=repo, revision=revision, sha256=sha256))
    print(f"Verified {name}", flush=True)
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
