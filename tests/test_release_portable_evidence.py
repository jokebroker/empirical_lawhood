# SPDX-License-Identifier: MPL-2.0
"""Reject mismatched source/evidence before an artifact-only release gate."""

import hashlib
import json

import pytest

from scripts import release_portable_evidence as evidence


@pytest.fixture
def packet(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"actual locked dependencies")
    (root / ".coveragerc").write_bytes(b"selected owner config")
    inventory = [{"path": "uv.lock", "mode": "100644", "size_bytes": 26, "sha256": "1"*64}]
    monkeypatch.setattr(evidence, "candidate_inventory", lambda selected: inventory)
    monkeypatch.setattr(evidence.subprocess, "check_output", lambda *a, **k: "abc123\n")
    log = tmp_path / "bound.log"
    log.write_bytes(b"completed check output\n")
    binding = {"path": log.name, "size_bytes": log.stat().st_size,
               "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    checks = [{"name": name, "exit_code": 0, "disposition": "PASSED", "log": log.name,
               "size_bytes": binding["size_bytes"], "sha256": binding["sha256"],
               "command": ["uv", "run", "--no-sync", "python", "-m", "coverage", "run", "-m", "pytest", "-q", "-ra", "--fail-on-skip", "-o", f"cache_dir={tmp_path / 'pytest-cache'}", "tests", "-m", "not native and not held"]}
              for name in sorted(evidence.REQUIRED_CHECKS)]
    inventory_file = tmp_path / "candidate-inventory.json"
    inventory_file.write_text(json.dumps(inventory))
    inventory_binding = {"path": inventory_file.name, "size_bytes": inventory_file.stat().st_size,
                         "sha256": hashlib.sha256(inventory_file.read_bytes()).hexdigest()}
    record = {"schema": evidence.SCHEMA, "status": "PASSED", "full_suite_invocations": 1,
              "candidate_tree": "abc123", "inventory_before_sha256": evidence.inventory_digest(inventory),
              "inventory_after_sha256": evidence.inventory_digest(inventory),
              "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
              "test_environment_policy": {"pytest_addopts": None, "coverage_rcfile_sha256": hashlib.sha256((root / ".coveragerc").read_bytes()).hexdigest()},
              "checks": checks, "coverage": binding, "candidate_inventory": inventory_binding}
    path = tmp_path / "precommit.json"
    def write():
        path.write_text(json.dumps(record))
        return path
    return root, record, write, log


def test_exact_tree_join_authenticates_existing_bytes_without_test_execution(packet):
    root, record, write, _ = packet
    assert evidence.authenticate_portable_evidence(write(), root) == record


@pytest.mark.parametrize("field,value", [("candidate_tree", "other"), ("status", "FAILED"),
    ("inventory_after_sha256", "0"*64), ("uv_lock_sha256", "0"*64), ("full_suite_invocations", 2)])
def test_rejects_source_change_failure_or_duplicate_suite(packet, field, value):
    root, record, write, _ = packet
    record[field] = value
    with pytest.raises(ValueError):
        evidence.authenticate_portable_evidence(write(), root)


def test_bound_log_mutation_invalidates_precommit_pass(packet):
    root, _, write, log = packet
    path = write()
    log.write_bytes(b"altered check output!!\n")
    with pytest.raises(ValueError):
        evidence.authenticate_portable_evidence(path, root)


def test_incomplete_selection_or_check_owners_are_not_a_full_portable_pass(packet):
    root, record, write, _ = packet
    record["checks"] = [item for item in record["checks"] if item["name"] != "critical-coverage-inventory"]
    with pytest.raises(ValueError, match="owners"):
        evidence.authenticate_portable_evidence(write(), root)


def test_missing_required_skip_policy_refuses_reuse(packet):
    root, record, write, _ = packet
    next(item for item in record["checks"] if item["name"] == "tests")["command"].remove("--fail-on-skip")
    with pytest.raises(ValueError, match="selector"):
        evidence.authenticate_portable_evidence(write(), root)


def test_evidence_path_escape_refuses_file_access(packet):
    root, record, write, _ = packet
    record["coverage"]["path"] = "../outside.log"
    with pytest.raises(ValueError, match="contained"):
        evidence.authenticate_portable_evidence(write(), root)


@pytest.mark.parametrize("extra", [["--collect-only"], ["-k", "favourable"], ["--deselect", "tests/test_bad.py"], ["--ignore", "tests/test_bad.py"], ["tests/test_good.py::test_one"]])
def test_filtered_or_collect_only_command_is_not_full_suite_evidence(packet, extra):
    root, record, write, _ = packet
    next(item for item in record["checks"] if item["name"] == "tests")["command"].extend(extra)
    with pytest.raises(ValueError, match="exact supported"):
        evidence.authenticate_portable_evidence(write(), root)


def test_unbound_or_filtered_environment_refuses_full_suite_reuse(packet):
    root, record, write, _ = packet
    record["test_environment_policy"]["pytest_addopts"] = "--collect-only"
    with pytest.raises(ValueError, match="environment"):
        evidence.authenticate_portable_evidence(write(), root)
