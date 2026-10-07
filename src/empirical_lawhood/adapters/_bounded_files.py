"""Small race-aware file primitives for adapter-owned configuration and fixtures."""

from __future__ import annotations

import os as os
from pathlib import Path

from empirical_lawhood.infrastructure import bounded_io


_CHUNK_BYTES = 1024 * 1024


class AdapterFileBoundError(ValueError):
    """An adapter file violated its explicit type, size, or stability bound."""


_identity = bounded_io._identity


def _open_regular(path: Path, maximum_bytes: int) -> tuple[int, os.stat_result]:
    if maximum_bytes < 0:
        raise ValueError("adapter file maximum_bytes must be nonnegative")
    try:
        return bounded_io._open_regular(
            path, maximum_bytes=maximum_bytes,
            open_error=lambda error: AdapterFileBoundError("adapter file cannot be opened safely"),
        )
    except bounded_io.BoundedFileIOError as error:
        raise AdapterFileBoundError(str(error).replace("bounded", "adapter")) from error


def read_bounded_regular(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one adapter-owned file only after its descriptor passes the byte limit."""

    descriptor, before = _open_regular(path, maximum_bytes)
    return _read_regular_descriptor(descriptor, before)


def _read_regular_descriptor(descriptor: int, before: os.stat_result) -> bytes:
    """Consume with the adapter's exact opened-size policy and exception boundary."""

    try:
        return bounded_io._read_regular_descriptor(
            descriptor, before, maximum_bytes=before.st_size,
            opened_size_bound=True, chunk_bytes=_CHUNK_BYTES,
        )
    except bounded_io.BoundedFileIOError as error:
        raise AdapterFileBoundError(str(error).replace("bounded", "adapter")) from error


def read_bounded_contained(root: Path, path: Path, *, maximum_bytes: int) -> bytes:
    """Read a regular member through directory descriptors with no symlink traversal.

    Adapted from the repository identity reader: each component is opened
    relative to the preceding directory descriptor. Path replacement cannot
    redirect the final open outside the selected directory tree. Bounds and
    stability checks are the same as for other adapter-owned file reads.
    """

    descriptor, before = _open_contained_regular(root, path, maximum_bytes)
    try:
        return _read_regular_descriptor(descriptor, before)
    except OSError as error:
        raise AdapterFileBoundError("adapter contained file cannot be read safely") from error


def _open_contained_regular(
    root: Path, path: Path, maximum_bytes: int
) -> tuple[int, os.stat_result]:
    if maximum_bytes < 0:
        raise ValueError("adapter file maximum_bytes must be nonnegative")
    try:
        return bounded_io._open_contained_regular(root, path, maximum_bytes)
    except bounded_io.BoundedFileIOError as error:
        raise AdapterFileBoundError(str(error).replace("bounded", "adapter")) from error


def sha256_bounded_regular(path: Path, *, maximum_bytes: int) -> str:
    """Hash one adapter-owned regular file under an explicit work ceiling."""

    descriptor, before = _open_regular(path, maximum_bytes)
    return _hash_regular_descriptor(descriptor, before)


def sha256_bounded_contained(
    root: Path, path: Path, *, maximum_bytes: int
) -> tuple[int, str]:
    """Stream a contained member under its opened size bound, without buffering it."""

    descriptor, before = _open_contained_regular(root, path, maximum_bytes)
    try:
        return before.st_size, _hash_regular_descriptor(descriptor, before)
    except OSError as error:
        raise AdapterFileBoundError("adapter contained file cannot be hashed safely") from error


def _hash_regular_descriptor(descriptor: int, before: os.stat_result) -> str:
    try:
        return bounded_io._hash_regular_descriptor(
            descriptor, before, maximum_bytes=before.st_size,
            opened_size_bound=True, chunk_bytes=_CHUNK_BYTES,
        )[1]
    except bounded_io.BoundedFileIOError as error:
        raise AdapterFileBoundError(str(error).replace("bounded", "adapter")) from error


__all__ = [
    "AdapterFileBoundError",
    "read_bounded_contained",
    "read_bounded_regular",
    "sha256_bounded_contained",
    "sha256_bounded_regular",
]
