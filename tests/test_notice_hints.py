# SPDX-License-Identifier: Apache-2.0
import pytest
from scripts.collect_runtime_licenses import notice_license_hints
from scripts.collect_runtime_sources import wheel_sources


def test_mpl_notice_is_detected_when_wheel_metadata_is_empty():
    assert notice_license_hints("Mozilla Public License Version 2.0\n===") == {"MPL-2.0"}
    assert not notice_license_hints("BSD license; see Mozilla Public License for an unrelated work")
    inventory = {"python_packages": [{"name": "pathspec", "version": "1.1.1", "notice_license_hints": ["MPL-2.0"]}]}
    with pytest.raises(ValueError, match="Unreviewed copyleft"):
        wheel_sources(inventory, {"wheels": {}})
