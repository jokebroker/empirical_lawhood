# SPDX-License-Identifier: MPL-2.0
"""Authenticate an unchanged tested candidate for separate postcommit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "empirical-lawhood/release/precommit-portable-evidence"
REQUIRED_CHECKS = frozenset({"environment", "coverage-configuration", "tests", "critical-coverage",
                             "critical-coverage-json", "critical-coverage-inventory"})


def candidate_inventory(root: Path) -> list[dict[str, object]]:
    """Describe the selected index's actual loaded bytes and executable modes."""
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).split(b"\0")
    result = []
    for entry in entries:
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, _, stage = metadata.decode().split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError("Release candidate refuses unresolved or non-file index entries")
        path = name.decode()
        payload = (root / path).read_bytes()
        executable = bool((root / path).stat().st_mode & 0o111)
        if executable != (mode == "100755"):
            raise ValueError(f"Loaded candidate executable mode differs: {path}")
        result.append({"path": path, "mode": mode, "size_bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()})
    return sorted(result, key=lambda row: row["path"])


def inventory_digest(inventory) -> str:
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bound_member(root: Path, binding: dict) -> Path:
    name = binding["path"]
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Portable evidence member must be a contained relative file")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Portable evidence refuses symlink members")
    if not path.is_file() or path.stat().st_size != binding["size_bytes"]:
        raise ValueError("Portable evidence member size differs")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != binding["sha256"]:
        raise ValueError("Portable evidence member checksum differs")
    return path


def authenticate_portable_evidence(path: Path, root: Path) -> dict:
    """Verify evidence bytes and exact final-tree join; never execute tests."""
    if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Portable evidence requires a bounded ordinary file")
    record = json.loads(path.read_bytes())
    if record["schema"] != SCHEMA or record["status"] != "PASSED" or record["full_suite_invocations"] != 1:
        raise ValueError("Portable evidence is not one completed passing precommit suite")
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    if record["candidate_tree"] != tree:
        raise ValueError("Final commit tree differs from the tested candidate")
    current = inventory_digest(candidate_inventory(root))
    if current != record["inventory_before_sha256"] or current != record["inventory_after_sha256"]:
        raise ValueError("Candidate loaded bytes changed before or after its suite")
    if record["uv_lock_sha256"] != hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest():
        raise ValueError("Portable evidence dependency lock differs")
    if record.get("test_environment_policy") != {
        "pytest_addopts": None,
        "coverage_rcfile_sha256": hashlib.sha256((root / ".coveragerc").read_bytes()).hexdigest(),
    }:
        raise ValueError("Portable evidence does not bind the unfiltered test/coverage environment")
    checks = record["checks"]
    if len({item["name"] for item in checks}) != len(checks) or not REQUIRED_CHECKS <= {item["name"] for item in checks}:
        raise ValueError("Portable evidence is missing required check owners")
    for item in checks:
        if item["exit_code"] != 0 or item["disposition"] != "PASSED":
            raise ValueError("Portable evidence contains a failed or incomplete check")
        _bound_member(path.parent, {"path": item["log"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    suite = next(item for item in checks if item["name"] == "tests")
    command = suite["command"]
    prefix = ["uv", "run", "--no-sync", "python", "-m", "coverage", "run", "-m", "pytest",
              "-q", "-ra", "--fail-on-skip", "-o"]
    suffix = ["tests", "-m", "not native and not held"]
    retention = ["-o", "tmp_path_retention_policy=failed", "-o", "tmp_path_retention_count=1"]
    if (command[:len(prefix)] != prefix or len(command) <= len(prefix)
        or not command[len(prefix)].startswith("cache_dir=")
        or not Path(command[len(prefix)].removeprefix("cache_dir=")).is_absolute()
        or command[len(prefix)+1:] not in (suffix, retention + suffix)):
        raise ValueError("Portable evidence does not cover the exact supported required-skip selector")
    _bound_member(path.parent, record["coverage"])
    selected_inventory = _bound_member(path.parent, record["candidate_inventory"])
    if inventory_digest(json.loads(selected_inventory.read_bytes())) != current:
        raise ValueError("Retained candidate inventory does not describe the tested source bytes")
    return record
