# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from types import SimpleNamespace
import hashlib
from scripts.collect_runtime_licenses import notice_license_hints, verified_metadata_only, METADATA_ONLY
from scripts.collect_runtime_sources import wheel_sources


def test_mpl_notice_is_detected_when_wheel_metadata_is_empty():
    assert notice_license_hints("Mozilla Public License Version 2.0\n===") == {"MPL-2.0"}
    assert not notice_license_hints("BSD license; see Mozilla Public License for an unrelated work")
    inventory = {"python_packages": [{"name": "pathspec", "version": "1.1.1", "notice_license_hints": ["MPL-2.0"]}]}
    with pytest.raises(ValueError, match="Unreviewed copyleft"):
        wheel_sources(inventory, {"wheels": {}})


def test_metadata_only_exception_rejects_changed_metadata_or_added_implementation(tmp_path, monkeypatch):
    directory = 'cuda_toolkit-13.0.3.0.dist-info'
    metadata = tmp_path/directory/'METADATA'
    metadata.parent.mkdir()
    metadata.write_bytes(b'reviewed dependency metadata')
    monkeypatch.setitem(METADATA_ONLY, ('cuda-toolkit', '13.0.3.0'),
                        (directory, hashlib.sha256(metadata.read_bytes()).hexdigest()))
    dist = SimpleNamespace(metadata={'Name':'cuda-toolkit'}, version='13.0.3.0',
                           files=[Path(directory)/name for name in ('METADATA','WHEEL','RECORD','INSTALLER')],
                           locate_file=lambda relative:tmp_path/relative)
    assert verified_metadata_only(dist)
    dist.files.append(Path('cuda_toolkit/implementation.py'))
    assert not verified_metadata_only(dist)
    dist.files.pop()
    metadata.write_bytes(b'changed package')
    assert not verified_metadata_only(dist)
