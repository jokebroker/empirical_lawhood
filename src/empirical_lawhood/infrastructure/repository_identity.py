"""Bounded identity proof for exact files in one clean Git worktree."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Final

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    validate_relative_locator,
    validate_schema,
    validate_stable_id,
)

from .bounded_process import BoundedProcessError, run_bounded_command


_MAX_REPOSITORY_STATUS_BYTES: Final[int] = 1024 * 1024
_READ_CHUNK_BYTES: Final[int] = 1024 * 1024


class RepositoryIdentityError(RuntimeError):
    """The worktree or a requested tracked file lacks an exact clean identity."""


@dataclass(frozen=True, slots=True)
class RepositoryArtifactSpec:
    """Code-owned request for one bounded tracked-file artifact identity."""

    relative_path: str
    artifact_id: str
    role: str
    payload_schema: str
    media_type: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        if ":" in self.relative_path or "\\" in self.relative_path:
            raise ValueError("repository artifact path contains a forbidden separator")
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        validate_stable_id(self.role, field_name="role")
        validate_schema(self.payload_schema)
        if (
            not isinstance(self.media_type, str)
            or not self.media_type
            or self.media_type != self.media_type.strip()
            or "\x00" in self.media_type
        ):
            raise ValueError("media_type must be a nonblank exact string")
        if (
            not isinstance(self.maximum_bytes, int)
            or isinstance(self.maximum_bytes, bool)
            or not 1 <= self.maximum_bytes <= 64 * 1024**2
        ):
            raise ValueError("maximum_bytes must be positive and within 64 MiB")


def _git(repo_root: Path, *arguments: str, maximum_stdout_bytes: int = 4096) -> bytes:
    try:
        result = run_bounded_command(
            ("git", *arguments),
            cwd=repo_root,
            timeout_seconds=10,
            maximum_stdout_bytes=maximum_stdout_bytes,
            maximum_stderr_bytes=64 * 1024,
        )
    except (BoundedProcessError, OSError, ValueError) as error:
        raise RepositoryIdentityError("bounded Git identity check failed") from error
    if result.returncode != 0:
        raise RepositoryIdentityError("Git identity check returned a nonzero status")
    return result.stdout


def _verify_clean_commit(repo_root: Path, expected_commit: str) -> None:
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ValueError("expected_commit must be a lowercase Git SHA-1")
    try:
        observed = _git(repo_root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise RepositoryIdentityError("Git HEAD is not ASCII") from error
    if observed != expected_commit:
        raise RepositoryIdentityError("worktree HEAD differs from the expected commit")
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        maximum_stdout_bytes=_MAX_REPOSITORY_STATUS_BYTES,
    )
    if status:
        raise RepositoryIdentityError("dataset composition requires a clean worktree")


def _open_relative_read_only(root_path: str, relative_path: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RepositoryIdentityError("platform cannot enforce no-follow repository reads")
    components = tuple(relative_path.split("/"))
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor = -1
    parent_descriptor = -1
    try:
        root_descriptor = os.open(root_path, directory_flags)
        parent_descriptor = root_descriptor
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=parent_descriptor)
            if parent_descriptor != root_descriptor:
                os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        return os.open(
            components[-1],
            os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise RepositoryIdentityError("tracked repository file cannot be opened safely") from error
    finally:
        if parent_descriptor >= 0 and parent_descriptor != root_descriptor:
            os.close(parent_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _read_exact_file(repo_root: Path, spec: RepositoryArtifactSpec) -> bytes:
    descriptor = _open_relative_read_only(str(repo_root), spec.relative_path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise RepositoryIdentityError("tracked repository input is not a regular file")
        if before.st_size > spec.maximum_bytes:
            raise RepositoryIdentityError("tracked repository input exceeds its byte ceiling")
        chunks: list[bytes] = []
        remaining = spec.maximum_bytes
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if not remaining and os.read(descriptor, 1):
            raise RepositoryIdentityError("tracked repository input exceeds its byte ceiling")
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise RepositoryIdentityError("tracked repository input changed during read")
        return b"".join(chunks)
    except OSError as error:
        raise RepositoryIdentityError("tracked repository input read failed") from error
    finally:
        os.close(descriptor)


def load_clean_repository_artifacts(
    repo_root: Path,
    *,
    expected_commit: str,
    specifications: tuple[RepositoryArtifactSpec, ...],
) -> tuple[ArtifactIdentity, ...]:
    """Authenticate exact tracked worktree bytes against one clean Git commit."""

    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a Path")
    if not isinstance(specifications, tuple) or not specifications:
        raise ValueError("repository artifact specifications must be a nonempty tuple")
    if any(not isinstance(value, RepositoryArtifactSpec) for value in specifications):
        raise ValueError("repository artifact specifications contain another type")
    artifact_ids = tuple(value.artifact_id for value in specifications)
    if tuple(sorted(set(artifact_ids))) != artifact_ids:
        raise ValueError("repository artifact specifications must be sorted and unique")
    relative_paths = tuple(value.relative_path for value in specifications)
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("repository artifact specifications repeat a tracked path")

    root = repo_root.resolve(strict=True)
    if repo_root.is_symlink() or not root.is_dir():
        raise RepositoryIdentityError("repository root must be a real directory")
    _verify_clean_commit(root, expected_commit)
    values: list[ArtifactIdentity] = []
    for spec in specifications:
        _git(root, "ls-files", "--error-unmatch", "--", spec.relative_path)
        worktree_payload = _read_exact_file(root, spec)
        committed_payload = _git(
            root,
            "show",
            f"{expected_commit}:{spec.relative_path}",
            maximum_stdout_bytes=spec.maximum_bytes,
        )
        if committed_payload != worktree_payload:
            raise RepositoryIdentityError("tracked worktree bytes differ from the expected commit")
        values.append(
            ArtifactIdentity(
                artifact_id=spec.artifact_id,
                role=spec.role,
                payload_schema=spec.payload_schema,
                sha256=hashlib.sha256(worktree_payload).hexdigest(),
                media_type=spec.media_type,
                size_bytes=len(worktree_payload),
            )
        )
    _verify_clean_commit(root, expected_commit)
    return tuple(values)


__all__ = [
    "RepositoryArtifactSpec",
    "RepositoryIdentityError",
    "load_clean_repository_artifacts",
]
