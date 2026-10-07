# SPDX-License-Identifier: MPL-2.0
"""Generated invocation contracts agree with both Click and observed refusals."""

import json
import importlib
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from empirical_lawhood.cli.app import app
from empirical_lawhood.cli.introspection import canonical_click_tree
from empirical_lawhood.cli.metadata import invocation_contracts

ROOT = Path(__file__).resolve().parents[1]


def test_every_leaf_has_exact_output_format_and_error_metadata():
    leaves = {
        fact.command: fact
        for fact in canonical_click_tree(get_command(app))
        if fact.command_kind == "command"
    }
    contracts = invocation_contracts()
    assert set(contracts) == set(leaves)
    for name, fact in leaves.items():
        has_format = any(
            "--format" in parameter.declarations for parameter in fact.parameters
        )
        assert has_format is (
            contracts[name].errors == "api"
            or name in {"workflow list", "workflow show", "config validate", "config prepare"}
        ), name


@pytest.mark.parametrize(
    ("command", "config", "reason"),
    (
        (
            "dependent-response-input-check",
            "prepared-response/prepared-dependent-refinement-input.json",
            "SOURCE_ROOT_REQUIRED",
        ),
        (
            "finite-response-stage-input-check",
            "finite-response-law/calibration-stage-input.json",
            "SOURCE_ROOT_REQUIRED",
        ),
        (
            "response-composition-input-check",
            "causal-transfer-audit/analysis-input.json",
            "SOURCE_ROOT_REQUIRED",
        ),
    ),
)
def test_conditional_parent_commands_reach_source_refusal_without_project_or_profile(
    tmp_path,
    monkeypatch,
    command,
    config,
    reason,
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["campaign", command, "--config", str(ROOT / "experiments" / config)]
    )
    assert result.exit_code == 3, result.output
    assert reason in result.output
    assert "--project-root" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_native_schema_errors_use_their_documented_exit_and_stream(tmp_path):
    config = json.loads(
        (ROOT / "experiments/prepared-response/prepared-canary.json").read_text()
    )
    config["value"]["unknown_field"] = "invalid"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(
        app, ["campaign", "prepared-response-native-check", "--config", str(path)]
    )
    assert result.exit_code == 3, result.output
    assert result.stdout == ""
    assert "fields differ" in result.stderr
    contract = invocation_contracts()["campaign prepared-response-native-check"]
    assert contract.errors == "refusal-3"


def test_portable_api_errors_use_common_envelope_and_argument_errors_use_click(
    tmp_path,
):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema":"unsupported"}')
    result = CliRunner().invoke(
        app, ["system", "validate", "--spec", str(path), "--format", "json"]
    )
    assert result.exit_code == 2, result.output
    document = json.loads(result.stderr)
    assert document["status"] == "INVALID"
    argument = CliRunner().invoke(app, ["campaign", "prepared-response-native-check"])
    assert argument.exit_code == 2
    assert "Missing option" in argument.output


@pytest.mark.parametrize(
    ("route", "module", "entry", "contacts"),
    (
        ("prepared", "prepared_response", "prepare_native_prefix", 2),
        ("finite", "finite_response_law", "execute_native_phase", 12),
    ),
)
def test_canary_cli_observed_native_contact_and_read_only_effects(
    tmp_path,
    monkeypatch,
    route,
    module,
    entry,
    contacts,
):
    from empirical_lawhood.cli.metadata import COMMAND_METADATA

    native = importlib.import_module(
        f"empirical_lawhood.adapters.simulators.{module}.native_quickstart"
    )
    original = getattr(native, entry)
    calls = []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(native, entry, observed)
    monkeypatch.chdir(tmp_path)
    command = f"{route}-response-native-check"
    bundle = "prepared-response" if route == "prepared" else "finite-response-law"
    config = ROOT / "experiments" / bundle / f"{route}-canary.json"
    result = CliRunner().invoke(app, ["campaign", command, "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["independent_roots"] == 1
    assert report["nested_numerical_views"] == 2
    assert report["development_only"] is True
    assert report["campaign_issued"] is False
    assert len(calls) == contacts
    assert list(tmp_path.iterdir()) == []
    metadata = next(m for m in COMMAND_METADATA if m.command == f"campaign {command}")
    assert metadata.effects.startswith("read-only. Executes")
    assert metadata.native_software and "native canary" in metadata.native_status

    calls.clear()
    missing = CliRunner().invoke(app, ["campaign", command])
    assert missing.exit_code == 2 and "Missing option" in missing.stderr
    malformed = tmp_path / "invalid.json"
    document = json.loads(config.read_bytes())
    document["value"]["evidence_role"] = "PROSPECTIVE"
    malformed.write_text(json.dumps(document))
    refused = CliRunner().invoke(app, ["campaign", command, "--config", str(malformed)])
    assert refused.exit_code == 3 and refused.stdout == ""
    assert "refused" in refused.stderr
    assert calls == []
    assert list(tmp_path.iterdir()) == [malformed]
