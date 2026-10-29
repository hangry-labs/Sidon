# SPDX-License-Identifier: Apache-2.0
"""Fail before publication if restricted research code or required notices drift."""
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_product_runtime_and_notices():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.13,<3.14"
    assert project["license"] == "Apache-2.0 AND MIT AND ISC"
    for name in ("Sidon-upstream-MIT.txt", "ClearerVoice-Apache-2.0.txt", "FLASH-MIT.txt", "SpeechBrain-Apache-2.0.txt", "faster-qwen-upstream-MIT.txt"):
        assert (ROOT / "licenses" / name).is_file()
    assert (ROOT / "src/sidon/web/static/vendor/lucide/LICENSE").is_file()
    for dep in project["dependencies"]:
        assert not re.search(r"flow[-_]matching|geneses|dialogue", dep, re.I)
    assert not (ROOT / "src/sidon/model").exists()


def test_runtime_pins_have_hashes_and_match_direct_requirements():
    source = (ROOT / "requirements.runtime.in").read_text()
    lock = (ROOT / "requirements.runtime.txt").read_text()
    entries = re.split(r"\n(?=[a-zA-Z0-9])", lock)
    pinned = {}
    for entry in entries:
        match = re.match(r"([\w-]+)==([^\s\\]+)", entry)
        if match:
            assert re.search(r"--hash=sha256:[a-f0-9]{64}", entry)
            pinned[match[1]] = match[2]
    for name, version in re.findall(r"^([\w-]+)==([^\s]+)", source, re.M):
        assert pinned[name] == version
    assert pinned["torch"] == pinned["torchaudio"] == "2.10.0+cu128"
