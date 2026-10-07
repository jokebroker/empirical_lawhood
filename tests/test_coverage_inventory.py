# SPDX-License-Identifier: MPL-2.0
"""Configured paths and actual critical measurements have separate gates."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def inventory():
    path = Path(__file__).resolve().parents[1] / "scripts/check_coverage_inventory.py"
    spec = importlib.util.spec_from_file_location("coverage_inventory_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configured_sources_exist(inventory):
    inventory.validate_configuration(Path(__file__).resolve().parents[1])


def test_missing_literal_source_refuses(inventory, tmp_path):
    (tmp_path / ".coveragerc").write_text("[run]\ninclude = missing.py\n")
    with pytest.raises(ValueError, match="no source match: missing.py"):
        inventory.validate_configuration(tmp_path)


def test_glob_requires_a_source_match(inventory, tmp_path):
    (tmp_path / ".coveragerc").write_text("[run]\ninclude = owner_*.py\n")
    with pytest.raises(ValueError, match="no source match"):
        inventory.validate_configuration(tmp_path)
    (tmp_path / "owner_present.py").write_text("pass\n")
    inventory.validate_configuration(tmp_path)


@pytest.mark.parametrize("absolute", [False, True])
def test_measured_inventory_normalizes_report_paths(inventory, tmp_path, absolute):
    names = {str(tmp_path / name) if absolute else name: {"executed_lines": [1], "summary": {"covered_lines": 1}}
             for name in inventory.CRITICAL_MEASURED_MODULES}
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": names}))
    inventory.validate_measured_inventory(tmp_path, report)
    names.pop(next(iter(names)))
    report.write_text(json.dumps({"files": names}))
    with pytest.raises(ValueError, match="required measured coverage owners missing"):
        inventory.validate_measured_inventory(tmp_path, report)


def test_outside_measurement_cannot_substitute_for_selected_source(inventory, tmp_path):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {
        str(tmp_path.parent / "other-checkout" / name): {"executed_lines": [1], "summary": {"covered_lines": 1}}
        for name in inventory.CRITICAL_MEASURED_MODULES}}))
    with pytest.raises(ValueError, match="required measured coverage owners missing"):
        inventory.validate_measured_inventory(tmp_path, report)


@pytest.mark.parametrize("details", [
    {},
    {"executed_lines": [], "summary": {"covered_lines": 0}},
    {"executed_lines": [], "summary": {"covered_lines": 1}},
    {"executed_lines": [1], "summary": {"covered_lines": 0}},
    {"executed_lines": [1]},
    {"executed_lines": [1], "summary": {"covered_lines": True}},
    {"executed_lines": [True], "summary": {"covered_lines": 1}},
    {"executed_lines": [-1], "summary": {"covered_lines": 1}},
])
def test_listed_but_unexecuted_required_owner_refuses(inventory, tmp_path, details):
    names = {name: {"executed_lines": [1], "summary": {"covered_lines": 1}}
             for name in inventory.CRITICAL_MEASURED_MODULES}
    names[next(iter(names))] = details
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": names}))
    with pytest.raises(ValueError, match="no executed statements"):
        inventory.validate_measured_inventory(tmp_path, report)


def test_empty_required_owner_inventory_cannot_claim_measurement(inventory, tmp_path):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {name: {} for name in inventory.CRITICAL_MEASURED_MODULES}}))
    with pytest.raises(ValueError, match="no executed statements"):
        inventory.validate_measured_inventory(tmp_path, report)


def test_unexecuted_alias_cannot_hide_behind_executed_selected_path(inventory, tmp_path):
    names = {name: {"executed_lines": [1], "summary": {"covered_lines": 1}}
             for name in inventory.CRITICAL_MEASURED_MODULES}
    owner = next(iter(names))
    names[str(tmp_path / owner)] = {"executed_lines": [], "summary": {"covered_lines": 0}}
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": names}))
    with pytest.raises(ValueError, match="no executed statements"):
        inventory.validate_measured_inventory(tmp_path, report)
