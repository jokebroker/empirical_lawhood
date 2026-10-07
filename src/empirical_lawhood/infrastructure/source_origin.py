# SPDX-License-Identifier: MPL-2.0
"""Bind a clean Git claim to the Python files that are actually executing."""

from __future__ import annotations

import sys
from pathlib import Path

from empirical_lawhood.infrastructure.bounded_process import (
    BoundedProcessError,
    run_bounded_command,
)


class ExecutingSourceMismatch(PermissionError):
    """The executing package is not the tracked package in the inspected checkout."""


def require_executing_target_source(repo_root: Path) -> Path:
    """Refuse a source claim from a wheel or a different source checkout.

    This checks origin and tracked membership. Clean HEAD and tree identity are
    checked separately by the caller's existing source-closure inspector.
    """

    try:
        root = repo_root.resolve(strict=True)
    except OSError as error:
        raise ExecutingSourceMismatch("inspected project root is unavailable") from error
    package_root = root / "src" / "empirical_lawhood"
    initializer = package_root / "__init__.py"
    package = sys.modules.get("empirical_lawhood")
    package_file = getattr(package, "__file__", None)
    if package_file is None or Path(package_file).resolve() != initializer.resolve():
        raise ExecutingSourceMismatch(
            "executing empirical_lawhood package is not the inspected target source tree"
        )
    try:
        found = run_bounded_command(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=root,
            timeout_seconds=5,
            maximum_stdout_bytes=4096,
            maximum_stderr_bytes=4096,
        )
        if found.returncode or Path(found.stdout.decode().strip()).resolve() != root:
            raise ExecutingSourceMismatch("inspected project root is not a Git checkout root")
        listed = run_bounded_command(
            ("git", "ls-files", "-z", "--", "src/empirical_lawhood"),
            cwd=root,
            timeout_seconds=5,
            maximum_stdout_bytes=1024 * 1024,
            maximum_stderr_bytes=4096,
        )
        if listed.returncode:
            raise ExecutingSourceMismatch("tracked target source cannot be enumerated")
        tracked_paths = set(listed.stdout.decode("utf-8").split("\0"))
        for name, module in tuple(sys.modules.items()):
            if name != "empirical_lawhood" and not name.startswith("empirical_lawhood."):
                continue
            origin = getattr(module, "__file__", None)
            if origin is None or not origin.endswith(".py"):
                continue
            path = Path(origin).resolve()
            if not path.is_relative_to(package_root):
                raise ExecutingSourceMismatch(
                    f"loaded module {name} is outside the inspected target source tree"
                )
            relative = path.relative_to(root).as_posix()
            if relative not in tracked_paths:
                raise ExecutingSourceMismatch(f"loaded module {name} is not tracked")
    except (BoundedProcessError, OSError, UnicodeDecodeError, ValueError) as error:
        raise ExecutingSourceMismatch("executing source origin cannot be verified") from error
    return package_root


__all__ = ["ExecutingSourceMismatch", "require_executing_target_source"]
