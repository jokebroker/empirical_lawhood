"""Static discovery preserves task, input and installed-resource boundaries.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from empirical_lawhood.api.workflows import list_workflows, show_workflow
from empirical_lawhood.cli.metadata import COMMAND_METADATA, invocation_contracts
from empirical_lawhood.cli.workflows import workflow_app, workflow_details
from scripts.generate_workflow_index import render_table

ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_accounts_for_every_bundle_and_input():
    workflows = list_workflows()
    ids = [w.workflow_id for w in workflows]
    expected = {p.parent.name for p in (ROOT / "experiments").glob("*/guide.md")}
    assert len(ids) == len(set(ids))
    assert set(ids) == expected
    declared = {item.path: item for workflow in workflows for item in workflow.inputs}
    files = sorted((ROOT / "experiments").rglob("*.json"))
    assert len(files) == len(declared)
    assert set(declared) == {p.relative_to(ROOT).as_posix() for p in files}
    for path in files:
        assert (
            json.loads(path.read_text())["schema"]
            == declared[path.relative_to(ROOT).as_posix()].schema_id
        )
    for workflow in workflows:
        assert (ROOT / workflow.guide).is_file()
        assert all((ROOT / owner).is_file() for owner in workflow.verification_owners)
        assert (
            workflow.substrates
            and workflow.methods
            and workflow.first_missing_prerequisite
        )
        assert workflow.input_availability and workflow.scientific_ceiling
        assert "not the installed wheel" in workflow.resources
        assert {p for op in workflow.operations for p in op.inputs} <= {
            p.path for p in workflow.inputs
        }


def test_command_effects_and_invocation_requirements_have_existing_owners():
    metadata = {m.command: m for m in COMMAND_METADATA}
    contracts = invocation_contracts()
    for workflow in list_workflows():
        detail = workflow_details(workflow)
        assert detail["prerequisites_inspected"] is False
        for entry in detail["operations"]:
            command = entry["command"]
            assert entry["metadata"]["effects"] == metadata[command].effects
            assert entry["invocation"]["project"] == contracts[command].project


@pytest.mark.parametrize(
    "args",
    (
        ["list"],
        ["list", "--format", "json"],
        ["show", "rc-ladder-response"],
        ["show", "rc-ladder-response", "--format", "json"],
    ),
)
def test_discovery_from_unrelated_directory_reads_no_inputs_and_writes_nothing(
    tmp_path, monkeypatch, args
):
    monkeypatch.chdir(tmp_path)
    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        assert "experiments/" not in str(file)
        return original_open(file, mode, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("discovery attempted path I/O")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    monkeypatch.setattr(Path, "write_text", refuse)
    result = CliRunner().invoke(workflow_app, args)
    assert result.exit_code == 0, result.output
    assert not tuple(tmp_path.iterdir())
    if "json" in args:
        body = json.loads(result.stdout)
        records = body["workflows"] if "workflows" in body else [body]
        assert all(record["prerequisites_inspected"] is False for record in records)
        assert all(
            "not the installed wheel" in record["resources"] for record in records
        )
    else:
        assert (
            "not inspected" in result.output
            or "have not been inspected" in result.output
        )


@pytest.mark.parametrize(
    "identifier", ("../../other", "os.system('touch x')", "missing")
)
def test_unknown_workflow_is_a_closed_lookup_refusal(identifier):
    with pytest.raises(ValueError, match="Unknown workflow"):
        show_workflow(identifier)
    result = CliRunner().invoke(workflow_app, ["show", identifier])
    assert result.exit_code == 2
    assert "Unknown workflow" in result.stderr
    assert not result.stdout


def test_fresh_discovery_does_not_import_optional_native_packages(tmp_path):
    script = """
import builtins, json
blocked = {'pybamm', 'brian2', 'cantera', 'fipy', 'grid2op', 'torax', 'gymtorax', 'jax', 'jaxlib'}
old_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('optional native import: ' + name)
    return old_import(name, *args, **kwargs)
builtins.__import__ = guarded
from empirical_lawhood.cli.workflows import workflow_details
from empirical_lawhood.api.workflows import list_workflows
print(json.dumps([workflow_details(w)['workflow_id'] for w in list_workflows()]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert set(json.loads(completed.stdout)) == {item.workflow_id for item in list_workflows()}
    assert not tuple(tmp_path.iterdir())


def test_index_matches_its_single_owner_and_all_paper_families_are_mapped():
    text = (ROOT / "experiments/README.md").read_text()
    assert render_table() in text
    paper = (ROOT / "paper/SOURCES.md").read_text()
    claims = json.loads(
        (ROOT / "paper/source/claim-evidence-inventory.json").read_text()
    )["claims"]
    rows = [line for line in paper.splitlines() if line.startswith("| R")]
    assert len(rows) == len(claims) == 22
    for claim in claims:
        assert sum(row.startswith(f"| {claim['id']} ") for row in rows) == 1
    assert "does not reproduce" in paper
