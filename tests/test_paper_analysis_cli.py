# SPDX-License-Identifier: MPL-2.0
"""Actual editable public preparation/proof journeys with no scientific campaign."""

import json
from pathlib import Path

from typer.testing import CliRunner

from empirical_lawhood.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def invoke(arguments):
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    return json.loads(result.stdout)


def test_geometry_public_fresh_prepare_edit_canonicalize_and_full_resource_proof(tmp_path, monkeypatch):
    from empirical_lawhood.api import matrix_geometry
    def forbidden():
        raise AssertionError("no-contact preparation/proof entered native numerical runtime")
    monkeypatch.setattr(matrix_geometry, "matrix_native_runtime_observation", forbidden)
    first, renamed, changed = (tmp_path / name for name in ("first.json", "renamed.json", "changed.json"))
    for output, label, seed in ((first, "test.geometry", 845001), (renamed, "test.geometry.renamed", 845001), (changed, "test.geometry.changed", 845002)):
        report = invoke(["--project-root", str(ROOT), "campaign", "matrix-geometry-prepare", "--config-id", label,
                         "--master-seed", str(seed), "--output", str(output)])
        assert report["native_contact"] == "none" and report["eligibility_granted"] is False
    bodies = [json.loads(path.read_bytes()) for path in (first, renamed, changed)]
    assert bodies[0]["value"]["allocation"]["value"]["cells"] == bodies[1]["value"]["allocation"]["value"]["cells"]
    assert bodies[0]["value"]["allocation"]["value"]["cells"] != bodies[2]["value"]["allocation"]["value"]["cells"]
    # A permitted resource edit keeps the scientific allocation intact.
    body = bodies[0]
    body["value"]["workers"] = 2
    pretty = tmp_path / "edited.json"
    pretty.write_text(json.dumps(body, indent=2))
    readiness = invoke(["config", "validate", "--consumer", "matrix-geometry", "--config", str(pretty), "--format", "json"])
    assert readiness["canonical_byte_ready"] is False
    prepared = tmp_path / "prepared.json"
    invoke(["config", "prepare", "--consumer", "matrix-geometry", "--config", str(pretty), "--output-file", str(prepared), "--format", "json"])
    attempt = tmp_path / "geometry-attempt"
    proof = invoke(["campaign", "matrix-geometry-prove", "--input", str(prepared), "--attempt-dir", str(attempt)])
    assert proof["potential_task_count"] == 17118 and proof["maximum_workers"] == 2
    assert proof["scientific_execution_performed"] is False
    retained = json.loads((attempt / "invocation.json").read_bytes())
    assert retained["operation_completed"] and retained["native_contact"] == "none"


def test_history_public_allocation_current_source_and_invalid_edit_refusal(tmp_path):
    allocation = tmp_path / "allocation.json"
    report = invoke(["campaign", "matrix-history-allocation", "--allocation-id", "test.public.history", "--namespace", "test.public.history", "--master-seed", "913562", "--output", str(allocation)])
    assert report["roots"] == 256 and report["eligibility_granted"] is False
    source = tmp_path / "source.json"
    invoke(["--project-root", str(ROOT), "campaign", "matrix-history-source-config", "--config-id", "test.public.history.source", "--allocation", str(allocation), "--output", str(source)])
    readiness = invoke(["config", "validate", "--consumer", "matrix-history-source", "--config", str(source), "--format", "json"])
    assert readiness["canonical_byte_ready"]
    body = json.loads(allocation.read_bytes())
    body["value"]["roots"] = body["value"]["roots"][:-1]
    invalid = tmp_path / "missing-root.json"
    invalid.write_text(json.dumps(body))
    rejected = runner.invoke(app, ["config", "validate", "--consumer", "matrix-history-allocation", "--config", str(invalid), "--format", "json"])
    assert rejected.exit_code == 2 and not (tmp_path / "invented-source").exists()


def test_existing_prepare_output_refuses_before_source_or_native_contact(tmp_path):
    output = tmp_path / "occupied.json"
    output.write_bytes(b"preserved")
    result = runner.invoke(app, ["campaign", "matrix-geometry-prepare", "--config-id", "test.refused", "--master-seed", "91562", "--output", str(output)])
    assert result.exit_code == 2 and "exist" in result.stderr.lower()
    assert output.read_bytes() == b"preserved"
