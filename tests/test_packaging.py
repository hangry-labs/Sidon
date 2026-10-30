# SPDX-License-Identifier: Apache-2.0
"""Fail before publication if restricted research code or required notices drift."""
from pathlib import Path
import re
import tomllib
from packaging.version import Version

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
    assert pinned["torch"] == "2.14.1+cu130"
    assert pinned["torchaudio"] == "2.11.1"


def test_reviewed_security_floors_cover_both_locks_and_test_runners():
    # Dependabot advisories reviewed on 2026-10-30; see docs/security.md.
    minimums = dict(certifi="2026.10.18", filelock="4.2.1", fsspec="2026.6.0",
                    setuptools="84.1.0", idna="3.15", pygments="2.20.0",
                    pytest="9.0.3", torch="2.14.1")
    development = {p['name']:p['version'] for p in tomllib.loads((ROOT/'uv.lock').read_text())['package']}
    runtime = dict(re.findall(r'^([\w-]+)==([^\s\\]+)', (ROOT/'requirements.runtime.txt').read_text(), re.M))
    for name, minimum in minimums.items():
        assert Version(development[name]) >= Version(minimum), f'uv.lock: {name}'
        if name in runtime:
            assert Version(runtime[name]) >= Version(minimum), f'runtime lock: {name}'
    for path in ('pyproject.toml', 'Dockerfile', '.github/workflows/docker-build.yml'):
        for pinned in re.findall(r'pytest==([\d.]+)', (ROOT/path).read_text()):
            assert Version(pinned) >= Version(minimums['pytest']), path
