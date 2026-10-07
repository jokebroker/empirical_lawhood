"""Exclusive atomic file publication on the supported Linux filesystems.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import ctypes
import errno
import os


def rename_noreplace(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    source_dir_fd: int | None = None,
    destination_dir_fd: int | None = None,
) -> None:
    """Move a staged file atomically, refusing any occupied destination.

    Linux renameat2 works on vfat without hard links. Directory descriptors
    keep publication anchored to an already opened directory. Unsupported
    platforms fail closed rather than checking existence before replacement.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "exclusive atomic publication is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100 if source_dir_fd is None else source_dir_fd,
        os.fsencode(source),
        -100 if destination_dir_fd is None else destination_dir_fd,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
