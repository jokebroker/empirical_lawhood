# SPDX-License-Identifier: MPL-2.0
"""A selected release profile cannot silently become a suite of skips."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("collection", (False, True))
def test_selected_profile_skip_has_nonzero_exit(tmp_path, collection):
    path = tmp_path / "test_missing.py"
    path.write_text(
        "import pytest\npytest.skip('required input absent', allow_module_level=True)\n"
        if collection else
        "import pytest\ndef test_missing():\n    pytest.skip('required input absent')\n"
    )
    (tmp_path / "test_present.py").write_text("def test_present():\n    assert True\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "--fail-on-skip", str(tmp_path)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "Required release-profile check skipped" in result.stdout + result.stderr
