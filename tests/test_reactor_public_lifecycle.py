# SPDX-License-Identifier: MPL-2.0
"""Real public lifecycle in a clean checkout; synthetic roles cannot qualify science."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.test_source_origin import ROOT, _git, build_source_repo


@pytest.mark.parametrize("mode", ("lifecycle", "scale", "exposed"))
def test_public_reactor_issue_execute_reveal_and_recover(tmp_path, mode):
    repo, _, _ = build_source_repo(tmp_path)
    for name in (".gitignore", "pyproject.toml", "uv.lock", "scripts/operator_records.py"):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "Isolated lifecycle environment and helpers")
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo))),
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("reactor_lifecycle_scenario.py")),
         str(repo), str(tmp_path / "private-test-trust"), mode],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        "lifecycle": "21 receipted tasks; same-identity recovery verified",
        "scale": "realistic census and production member bounds verified",
        "exposed": 'exposed assignment refused base and extension study issue despite contradictory attestations',
    }[mode]
    assert expected in result.stdout
    assert not _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
