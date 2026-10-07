# SPDX-License-Identifier: MPL-2.0
"""Human projections preserve operational/scientific and environment boundaries."""

import json
import subprocess

from typer.testing import CliRunner

from empirical_lawhood.api.execution import AttemptHistorySummary, RunStatusSummary
from empirical_lawhood.cli.app import app
from empirical_lawhood.cli.presentation import render_campaign_status


def test_doctor_text_separates_public_dependencies_and_issued_readiness(tmp_path):
    result = CliRunner().invoke(app, ["doctor", "--route", "reactor"])
    assert result.exit_code == 0, result.output
    assert "Environment diagnosis completed" in result.stdout
    assert "Dependency observations do not run a native task" in result.stdout
    assert "public example needs no project or operator profile" in result.stdout
    assert "infrastructure_ready=" in result.stdout
    assert "authority_required=" in result.stdout


def test_doctor_version_hint_requires_explicit_project_and_keeps_json(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text('[project]\nversion = "999.0.0"\n')
    runner = CliRunner()
    before = sorted(tmp_path.rglob("*"))
    implicit = runner.invoke(app, ["doctor"])
    assert "Metadata hint:" not in implicit.stdout
    explicit = runner.invoke(app, ["--project-root", str(tmp_path), "doctor"])
    assert explicit.exit_code == 0, explicit.output
    assert "selected source declares 999.0.0" in explicit.stdout
    machine = runner.invoke(app, ["--project-root", str(tmp_path), "doctor", "--format", "json"])
    assert machine.exit_code == 0
    assert "package_version" in json.loads(machine.stdout)["payload"]
    assert "Metadata hint:" not in machine.stdout
    assert sorted(tmp_path.rglob("*")) == before


def test_status_uses_only_bounded_existing_projection_and_receipt_pointers():
    summary = RunStatusSummary(
        "test-run", "FAILED", ("done",), ("failed",), (), (),
        attempt_history=(AttemptHistorySummary(
            "attempt-one", "failed", 1, "FAILED", "INPUT_INVALID", None,
            failure_class="INPUT_INVALID", retryable=False,
            recovery_event_relative_path="runs/test-run/recovery/event.json",
        ),),
        recovery_index_relative_path="runs/test-run/recovery/index.json",
        artifact_receipt_relative_paths=("runs/test-run/receipts/one.json",),
    )
    text = render_campaign_status(summary)
    assert "operational_status=FAILED" in text
    assert "retryable=False" in text
    assert "not read by status" in text
    assert "runs/test-run/recovery/index.json" in text
    assert "runs/test-run/receipts/one.json" in text
    assert "status does not replay tasks or reveal outcomes" in text
