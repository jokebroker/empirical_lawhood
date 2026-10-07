"""Checked advisory locks without opening unwritten lock files for writing."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold a regular, no-follow lock file, retaining NFS write-mode support.

    Local Linux flock accepts a read-only descriptor. Avoiding FMODE_WRITE
    prevents VFAT's flush-on-close delay for these never-written files. NFS
    emulates exclusive flock with a write lock and reports EBADF instead; only
    that refusal permits reopening for writing. Every other lock error escapes.
    """

    flags = os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    locked = False
    try:
        for mode in (os.O_RDONLY, os.O_RDWR):
            descriptor = os.open(path, flags | mode, stat.S_IRUSR | stat.S_IWUSR)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("advisory lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as error:
                if mode != os.O_RDONLY or error.errno != errno.EBADF:
                    raise
                os.close(descriptor)
                descriptor = None
            else:
                locked = True
                break
        yield
    finally:
        if descriptor is not None:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
