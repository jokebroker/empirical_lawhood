"""Guarded external I/O for held simulator selector authoring."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

from empirical_lawhood.planning.simulator_inventory import (
    SimulatorDirectoryInventoryAudit,
    SimulatorHeldInventorySpec,
    SimulatorInventorySelectionKind,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_relative_locator
from empirical_lawhood.planning.dataset_manifests import (
    DatasetDirectoryMemberExpectation,
    DatasetDirectorySelectorManifest,
    dataset_directory_content_sha256,
)

from .artifacts import ArtifactIdentityConflict, GuardedExternalRoot
from .bounded_process import BoundedProcessError, run_bounded_command


def _git(root: Path, *arguments: str, maximum_bytes: int) -> bytes:
    try:
        result = run_bounded_command(
            ("git", *arguments),
            cwd=root,
            timeout_seconds=30,
            maximum_stdout_bytes=maximum_bytes,
            maximum_stderr_bytes=64 * 1024,
        )
    except (BoundedProcessError, OSError, ValueError) as error:
        raise ArtifactIdentityConflict("bounded simulator source Git audit failed") from error
    if result.returncode != 0:
        raise ArtifactIdentityConflict("simulator source Git audit returned nonzero")
    return result.stdout


def _tracked_locators(root: Path, expected_commit: str, maximum_files: int) -> tuple[str, ...]:
    observed = _git(root, "rev-parse", "HEAD", maximum_bytes=128).decode("ascii").strip()
    if observed != expected_commit:
        raise ArtifactIdentityConflict("simulator source checkout has another Git commit")
    if _git(root, "status", "--porcelain=v1", maximum_bytes=1024 * 1024):
        raise ArtifactIdentityConflict("simulator source checkout is not clean")
    raw = _git(root, "ls-files", "-z", maximum_bytes=4 * 1024 * 1024)
    values = tuple(value.decode("utf-8") for value in raw.split(b"\0") if value)
    if not values or len(values) > maximum_files or tuple(sorted(set(values))) != values:
        raise ArtifactIdentityConflict("tracked simulator inventory is unbounded or unsorted")
    for value in values:
        validate_relative_locator(value)
    return values


def _tree_locators(root: Path, maximum_files: int) -> tuple[str, ...]:
    values: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for name in tuple(directory_names):
            if (directory_path / name).is_symlink():
                raise ArtifactIdentityConflict("simulator inventory tree contains a symlink")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ArtifactIdentityConflict(
                    "simulator inventory member is a symlink or not regular"
                )
            values.append(path.relative_to(root).as_posix())
            if len(values) > maximum_files:
                raise ArtifactIdentityConflict("simulator inventory exceeds its file bound")
    result = tuple(sorted(values))
    if not result or len(set(result)) != len(result):
        raise ArtifactIdentityConflict("simulator inventory tree is empty or ambiguous")
    return result


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_regular_file(path: Path, maximum_remaining_bytes: int) -> tuple[int, str]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ArtifactIdentityConflict("simulator inventory no-follow reads are unavailable")
    descriptor = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_remaining_bytes:
            raise ArtifactIdentityConflict("simulator inventory member exceeds its byte bound")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ArtifactIdentityConflict("simulator inventory member was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ArtifactIdentityConflict("simulator inventory member grew during hashing")
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise ArtifactIdentityConflict("simulator inventory member changed during hashing")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


class ExternalSimulatorDirectoryInventoryAuthor:
    """Author literal selectors beneath one authenticated external root."""

    def __init__(self, root: GuardedExternalRoot) -> None:
        if not isinstance(root, GuardedExternalRoot):
            raise TypeError("root must be a GuardedExternalRoot")
        self.root = root

    def author(
        self,
        spec: SimulatorHeldInventorySpec,
    ) -> tuple[DatasetDirectorySelectorManifest, SimulatorDirectoryInventoryAudit]:
        if not isinstance(spec, SimulatorHeldInventorySpec):
            raise TypeError("spec must be a SimulatorHeldInventorySpec")
        source = self.root.resolve(spec.source_relative_prefix, for_write=False)
        if source.is_symlink() or not source.is_dir():
            raise ArtifactIdentityConflict("simulator inventory source must be a real directory")
        observed_git_commit: str | None = None
        if spec.selection_kind is SimulatorInventorySelectionKind.GIT_TRACKED_TREE:
            assert spec.expected_git_commit is not None
            locators = _tracked_locators(source, spec.expected_git_commit, spec.maximum_files)
            observed_git_commit = spec.expected_git_commit
        elif spec.selection_kind is SimulatorInventorySelectionKind.COMPLETE_REGULAR_TREE:
            locators = _tree_locators(source, spec.maximum_files)
        else:
            locators = spec.literal_relative_locators
        members: list[DatasetDirectoryMemberExpectation] = []
        total_bytes = 0
        for index, locator in enumerate(locators, start=1):
            combined = PurePosixPath(spec.source_relative_prefix, locator).as_posix()
            path = self.root.resolve(combined, for_write=False)
            size, digest = _hash_regular_file(path, spec.maximum_bytes - total_bytes)
            total_bytes += size
            members.append(
                DatasetDirectoryMemberExpectation(
                    member_id=f"member.{spec.subject_id}.{index:05d}",
                    relative_locator=locator,
                    expected_physical_sha256=digest,
                    expected_size_bytes=size,
                    media_type="application/octet-stream",
                )
            )
        values = tuple(members)
        selector = DatasetDirectorySelectorManifest(
            selector_id=spec.selector_id,
            subject_id=spec.subject_id,
            complete_subject=spec.complete_subject,
            members=values,
            expected_content_sha256=dataset_directory_content_sha256(values),
            expected_total_size_bytes=total_bytes,
        )
        digest = hashlib.sha256(
            f"{spec.fingerprint()}:{selector.fingerprint()}".encode("ascii")
        ).hexdigest()
        return selector, SimulatorDirectoryInventoryAudit(
            audit_id=f"audit.simulator-inventory.{digest[:24]}",
            inventory_spec=ObjectIdentity.from_record(spec.inventory_id, spec),
            selector=ObjectIdentity.from_record(selector.selector_id, selector),
            observed_files=len(values),
            observed_bytes=total_bytes,
            observed_git_commit=observed_git_commit,
            source_mutated=False,
            network_bytes=0,
        )


__all__ = ["ExternalSimulatorDirectoryInventoryAuthor"]
