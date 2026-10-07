"""Capture and authenticate the clean Git source actually executing.

SPDX-License-Identifier: MPL-2.0
"""

from hashlib import sha256
from pathlib import Path

from empirical_lawhood.infrastructure.bounded_process import BoundedProcessError, run_bounded_command
from empirical_lawhood.infrastructure.study_issue import GitStudySourceClosureInspector
from empirical_lawhood.infrastructure.source_origin import (
    require_executing_target_source,
)
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _git(root: Path, *arguments: str) -> bytes:
    limits = {
        "rev-parse": 128,
        "status": GitStudySourceClosureInspector._MAX_STATUS_BYTES,
        "ls-tree": GitStudySourceClosureInspector._MAX_TREE_BYTES,
        "archive": _MAX_ARCHIVE_BYTES,
    }
    if not arguments or arguments[0] not in limits:
        raise ValueError("unsupported target source capture operation")
    try:
        result = run_bounded_command(
            ("git", *arguments),
            cwd=root,
            timeout_seconds=60,
            maximum_stdout_bytes=limits[arguments[0]],
            maximum_stderr_bytes=64 * 1024,
        )
    except BoundedProcessError as error:
        raise PermissionError(f"target source capture failed: {error}") from error
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise PermissionError(
            f"target source capture failed: {message or result.returncode}"
        )
    return result.stdout


def capture_clean_target_closure(
    root: Path, closure_id: str
) -> ImplementationSourceClosure:
    root = root.resolve(strict=True)
    require_executing_target_source(root)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PermissionError("fresh native authoring requires a clean target commit")
    archive = _git(root, "archive", "HEAD")
    if len(archive) > _MAX_ARCHIVE_BYTES:
        raise ValueError("target source archive exceeds the bounded authoring limit")
    closure = ImplementationSourceClosure(
        closure_id,
        SourceClosureKind.CLEAN_GIT_COMMIT,
        _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        sha256(archive).hexdigest(),
        sha256(_git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")).hexdigest(),
        True,
    )
    GitStudySourceClosureInspector(root).observe(closure)
    return closure
