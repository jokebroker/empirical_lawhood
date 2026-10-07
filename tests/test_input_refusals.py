# SPDX-License-Identifier: MPL-2.0
"""Malformed inputs refuse at the public boundary, without native or write effects."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest
from typer.testing import CliRunner

from empirical_lawhood.cli.app import app
from tests.test_doctor_profile import _profile


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(("command", "limit", "exit_code"), (
    ('electron-gas-reference-check', 32 * 1024, 3),
    ("rc-ladder-native-check", 64 * 1024, 2),
    ("brian2-native-check", 16 * 1024, 3),
))
def test_native_check_oversize_refuses_without_contact(tmp_path, monkeypatch, command, limit, exit_code):
    from empirical_lawhood.adapters.simulators.uniform_electron_gas_response import native_quickstart as electron_gas
    from empirical_lawhood.adapters.simulators.rc_ladder_response import native_quickstart as rc
    from empirical_lawhood.adapters.simulators.brian2_neuron_current_response import native_quickstart as brian

    def forbidden(*args, **kwargs):
        pytest.fail("oversized config reached a native computation")

    monkeypatch.setattr(electron_gas, "run_analytic_development_check", forbidden)
    monkeypatch.setattr(brian, "run_native_development_check", forbidden)
    # The RC entry point decodes before constructing either solver.
    monkeypatch.setattr(rc, "decode_canonical_bytes", forbidden)
    config = tmp_path / "oversized.json"
    config.write_bytes(b" " * (limit + 1))
    args = ["campaign", command, "--config", str(config)]
    if command == "brian2-native-check":
        args += ["--native-python", sys.executable]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == exit_code, result.output
    assert result.stdout == ""
    assert "exceeds its byte limit" in result.stderr
    assert "Traceback" not in result.output
    assert sorted(tmp_path.iterdir()) == [config]


def test_actual_cli_refuses_fifo_without_waiting_for_writer(tmp_path):
    fifo = tmp_path / "input.json"
    os.mkfifo(fifo)
    result = subprocess.run(
        [str(Path(sys.executable).parent / "empirical-lawhood"), "campaign",
         'electron-gas-reference-check', "--config", str(fifo)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert result.stdout == ""
    assert "not a regular file" in result.stderr
    assert "Traceback" not in result.stderr
    assert sorted(tmp_path.iterdir()) == [fifo]


@pytest.mark.parametrize("document", ([], None, False, 0, "profile", {"schema": []}))
def test_fresh_author_refuses_nonobject_or_invalid_schema_without_writes(tmp_path, monkeypatch, document):
    from empirical_lawhood.cli import platform

    def forbidden(*args, **kwargs):
        pytest.fail("invalid profile reached authoring/source inspection")

    monkeypatch.setattr(platform, "author_fresh_reactor", forbidden)
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(document))
    with TemporaryDirectory(prefix="empirical-lawhood-refusal-", dir="/dev/shm") as temporary:
        storage = Path(temporary)
        (storage / "artifacts").mkdir()
        (storage / "scratch").mkdir()
        profile_path = tmp_path / "storage.json"
        profile_path.write_bytes(_profile(storage).canonical_bytes())
        before = sorted(storage.rglob("*"))
        result = CliRunner().invoke(app, [
            "--project-root", str(ROOT), "--operator-profile", str(profile_path),
            "campaign", 'reactor-author', "--profile", str(config),
            "--output-dir", str(storage / "artifacts" / "new-authoring"),
        ])
        assert result.exit_code == 5, result.output
        assert result.stdout == ""
        assert "REACTOR_AUTHORING_PROFILE_" in result.stderr
        assert "Traceback" not in result.output
        assert sorted(storage.rglob("*")) == before
