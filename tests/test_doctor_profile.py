# SPDX-License-Identifier: MPL-2.0
"""Actual CLI diagnostics honor profiles without creating an execution plane."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from typer.testing import CliRunner

from empirical_lawhood.cli.app import app
from empirical_lawhood.runtime.operator_profile import OperatorStorageAccessMode, OperatorStorageProfile


def _profile(root: Path) -> OperatorStorageProfile:
    # /dev/shm supplies an isolated, real Linux mount for software diagnostics.
    # These empty temporary directories contain no scientific run or authority.
    return OperatorStorageProfile(
        profile_id="test-doctor-storage",
        backend_key="external-filesystem",
        backend_version="1.0.0",
        external_root_locator=str(root),
        required_mount_path="/dev/shm",
        artifact_namespace="artifacts",
        scientific_scratch_namespace="scratch",
        access_mode=OperatorStorageAccessMode.READ_WRITE,
        expected_mount_source=None,
        expected_volume_identity=None,
        allowed_filesystem_types=("tmpfs",),
        containment_policy_key="strict-mount-contained-no-symlink",
        minimum_free_bytes=1,
        read_only_mirror_ids=(),
        maximum_parallel_tasks=1,
        authority_granted=False,
    )


def _invoke(tmp_path: Path, raw: bytes | None):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    if not (project / ".git").exists():
        subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    args = ["--project-root", str(project)]
    if raw is not None:
        profile = tmp_path / "profile.json"
        profile.write_bytes(raw)
        args += ["--operator-profile", str(profile)]
    result = CliRunner().invoke(app, [*args, "doctor", "--format", "json"])
    assert sorted(p.name for p in project.iterdir()) == [".git"], "doctor wrote local state"
    return result


def test_no_profile_is_portable_and_malformed_profile_is_not_ignored(tmp_path: Path) -> None:
    result = _invoke(tmp_path, None)
    assert result.exit_code == 0, result.output
    assert "OPERATOR_STORAGE_UNCONFIGURED" in json.loads(result.stdout)["reason_codes"]
    malformed = _invoke(tmp_path, b'{"not":"an operator storage profile"}')
    assert malformed.exit_code == 2, malformed.output
    assert "operator-profile" in malformed.output


@pytest.mark.parametrize("read_only", (False, True))
def test_valid_profile_reports_real_mount_and_honors_access_mode(
    tmp_path: Path, read_only: bool
) -> None:
    with TemporaryDirectory(prefix="empirical-lawhood-doctor-", dir="/dev/shm") as directory:
        root = Path(directory)
        (root / "artifacts").mkdir()
        profile = _profile(root)
        if read_only:
            profile = replace(profile, access_mode=OperatorStorageAccessMode.READ_ONLY)
        before = sorted(root.rglob("*"))
        result = _invoke(tmp_path, profile.canonical_bytes())
        assert result.exit_code == 0, result.output
        document = json.loads(result.stdout)
        payload = document["payload"]
        assert payload["external_root"] == str(root / "artifacts")
        assert payload["mount_active"] is True
        assert payload["storage_read_ready"] is True
        assert payload["storage_write_ready"] is not read_only
        assert ("OPERATOR_STORAGE_READ_ONLY" in document["reason_codes"]) is read_only
        assert "OPERATOR_STORAGE_UNCONFIGURED" not in document["reason_codes"]
        assert sorted(root.rglob("*")) == before


@pytest.mark.parametrize("failure", ("filesystem", "space", "missing", "mount", "permission"))
def test_profile_unavailability_is_diagnosed_without_initialization(
    tmp_path: Path, failure: str
) -> None:
    with TemporaryDirectory(prefix="empirical-lawhood-doctor-", dir="/dev/shm") as directory:
        root = Path(directory)
        profile = _profile(root)
        if failure != "missing":
            (root / "artifacts").mkdir()
        if failure == "filesystem":
            profile = replace(profile, allowed_filesystem_types=("unavailable-filesystem",))
            expected = "EXTERNAL_FILESYSTEM_NOT_ALLOWED"
        elif failure == "space":
            profile = replace(profile, minimum_free_bytes=2**60)
            expected = "EXTERNAL_FREE_SPACE_BELOW_FLOOR"
        elif failure == "mount":
            profile = replace(profile, required_mount_path=str(root))
            expected = "EXTERNAL_MOUNT_INACTIVE"
        elif failure == "permission":
            (root / "artifacts").chmod(0o500)
            expected = "EXTERNAL_STORAGE_NOT_WRITABLE"
        else:
            expected = "EXTERNAL_STORAGE_ABSENT"
        before = sorted(root.rglob("*"))
        result = _invoke(tmp_path, profile.canonical_bytes())
        assert result.exit_code == 0, result.output
        document = json.loads(result.stdout)
        assert expected in document["reason_codes"]
        assert document["payload"]["storage_write_ready"] is False
        assert sorted(root.rglob("*")) == before


def test_profile_containment_is_checked_before_mount_inspection(tmp_path: Path) -> None:
    profile = replace(_profile(tmp_path), required_mount_path=str(tmp_path))
    result = _invoke(tmp_path, profile.canonical_bytes())
    assert result.exit_code == 2
    assert "forbidden local root" in result.output
