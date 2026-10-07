"""Bounded process-wide serialization for production catalog mutations."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from empirical_lawhood.infrastructure.sql.database import production_catalog_path


CATALOG_MUTATION_LOCK_FILENAME: Final = "experiment_catalog.mutation.lock"
DEFAULT_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS: Final = 5.0
MAX_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS: Final = 30.0
_CATALOG_MUTATION_LOCK_POLL_SECONDS: Final = 0.01


class CatalogMutationLockError(RuntimeError):
    """The production catalog mutation lock could not be used safely."""


class CatalogMutationLockUnavailable(CatalogMutationLockError):
    """Another cooperative process retained the lock through the bounded wait."""


@dataclass(frozen=True, slots=True)
class CatalogMutationLock:
    """Identity of one held lock; the descriptor remains private to the context."""

    lock_path: Path
    device: int
    inode: int


def production_catalog_mutation_lock_path(repo_root: Path) -> Path:
    """Return the persistent, separate lock path for the exact production catalog."""

    return production_catalog_path(repo_root).parent / CATALOG_MUTATION_LOCK_FILENAME


def _validate_timeout(timeout_seconds: float) -> float:
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 <= timeout_seconds <= MAX_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS
    ):
        raise ValueError("timeout_seconds must be finite and within the catalog lock bound")
    value = float(timeout_seconds)
    if not value < float("inf"):
        raise ValueError("timeout_seconds must be finite and within the catalog lock bound")
    return value


@contextmanager
def production_catalog_mutation_lock(
    repo_root: Path,
    *,
    timeout_seconds: float = DEFAULT_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS,
) -> Iterator[CatalogMutationLock]:
    """Hold the non-symlink production mutation lock for a bounded interval.

    The file is persistent and never unlinked: replacing a lock inode while it
    is held would split cooperative writers across different advisory locks.
    """

    timeout = _validate_timeout(timeout_seconds)
    lock_path = production_catalog_mutation_lock_path(repo_root)
    runtime_directory = lock_path.parent
    try:
        runtime_directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    except OSError as error:
        raise CatalogMutationLockError(
            "production catalog lock directory could not be prepared"
        ) from error
    if runtime_directory.is_symlink() or not runtime_directory.is_dir():
        raise CatalogMutationLockError("production catalog lock directory must be a real directory")

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise CatalogMutationLockError("platform cannot enforce a non-symlink lock file")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | no_follow
    try:
        directory_descriptor = os.open(runtime_directory, directory_flags)
    except OSError as error:
        raise CatalogMutationLockError(
            "production catalog lock directory could not be opened safely"
        ) from error

    descriptor: int | None = None
    acquired = False
    try:
        try:
            descriptor = os.open(
                CATALOG_MUTATION_LOCK_FILENAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | no_follow,
                stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            raise CatalogMutationLockError(
                "production catalog mutation lock must not be a symlink"
            ) from error
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_nlink != 1
            or identity.st_uid != os.getuid()
            or stat.S_IMODE(identity.st_mode) & 0o077
        ):
            raise CatalogMutationLockError(
                "production catalog mutation lock has an unsafe identity or mode"
            )

        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise CatalogMutationLockError(
                        "production catalog mutation lock acquisition failed"
                    ) from error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CatalogMutationLockUnavailable(
                        "production catalog mutation lock remained held"
                    ) from error
                time.sleep(min(_CATALOG_MUTATION_LOCK_POLL_SECONDS, remaining))

        path_identity = os.stat(
            CATALOG_MUTATION_LOCK_FILENAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_identity.st_mode) or (
            path_identity.st_dev,
            path_identity.st_ino,
        ) != (identity.st_dev, identity.st_ino):
            raise CatalogMutationLockError(
                "production catalog mutation lock changed during acquisition"
            )
        yield CatalogMutationLock(
            lock_path=lock_path,
            device=identity.st_dev,
            inode=identity.st_ino,
        )
    finally:
        if descriptor is not None:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        os.close(directory_descriptor)


__all__ = [
    "CATALOG_MUTATION_LOCK_FILENAME",
    "CatalogMutationLock",
    "CatalogMutationLockError",
    "CatalogMutationLockUnavailable",
    "DEFAULT_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS",
    "MAX_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS",
    "production_catalog_mutation_lock",
    "production_catalog_mutation_lock_path",
]
