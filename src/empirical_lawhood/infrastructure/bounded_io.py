"""Race-aware bounded file primitives for current control-plane data."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
import os
import stat
from pathlib import Path


CONTROL_PLANE_CHUNK_BYTES = 1024 * 1024
MAX_CONTROL_PLANE_JSON_BYTES = 16 * 1024 * 1024
# Full typed run/execution DAGs have a separate finite envelope. The current
# 64-root/128-view graph is 38,642,100 bytes; compact receipts/control stay 16 MiB.
MAX_RUNTIME_PLAN_JSON_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_ARTIFACT_PUBLICATION_MARKER_BYTES = 1024 * 1024
MAX_ARTIFACT_PUBLICATION_INTENT_BYTES = 2 * 1024 * 1024
MAX_CATALOG_DATABASE_BYTES = 512 * 1024 * 1024


class BoundedFileIOError(RuntimeError):
    """A file violated its declared size, type, identity, or stability bound."""


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _open_regular(
    path: Path,
    *,
    maximum_bytes: int,
    open_error: Callable[[OSError], Exception] | None = None,
) -> tuple[int, os.stat_result]:
    if maximum_bytes < 0:
        raise ValueError("maximum_bytes must be nonnegative")
    # Validate the descriptor before any read. Without O_NONBLOCK, opening a
    # FIFO can wait for a writer forever, before fstat can refuse its type.
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if open_error is not None:
            raise open_error(error) from error
        raise
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise BoundedFileIOError("bounded input is not a regular file")
        if observed.st_size > maximum_bytes:
            raise BoundedFileIOError("bounded input exceeds its byte limit")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, observed


@contextmanager
def _stable_descriptor(
    descriptor: int, before: os.stat_result, *, operation: str
) -> Iterator[None]:
    """Own one consumed descriptor, closing before reporting identity drift."""

    try:
        yield
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise BoundedFileIOError(f"bounded input changed during {operation}")


def _regular_chunks(
    descriptor: int,
    before: os.stat_result,
    *,
    maximum_bytes: int,
    chunk_bytes: int,
    opened_size_bound: bool,
    operation: str,
) -> Iterator[bytes]:
    """Consume the declared maximum or exact opened size without reopening paths."""

    consumed = 0
    while not opened_size_bound or consumed < before.st_size:
        if opened_size_bound:
            request = min(chunk_bytes, before.st_size - consumed)
        elif operation == "read":
            request = min(chunk_bytes, maximum_bytes - consumed + 1)
        else:
            # Streaming hashes retain their fixed-chunk work contract.
            request = chunk_bytes
        chunk = os.read(descriptor, request)
        if not chunk:
            if opened_size_bound:
                raise BoundedFileIOError(f"bounded input was truncated during {operation}")
            break
        consumed += len(chunk)
        if not opened_size_bound and consumed > maximum_bytes:
            raise BoundedFileIOError("bounded input exceeds its byte limit")
        yield chunk
    if opened_size_bound and os.read(descriptor, 1):
        raise BoundedFileIOError("bounded input grew beyond its byte limit")


def _read_regular_descriptor(
    descriptor: int,
    before: os.stat_result,
    *,
    maximum_bytes: int,
    opened_size_bound: bool = False,
    chunk_bytes: int | None = None,
) -> bytes:
    chunks: list[bytes] = []
    count = 0
    with _stable_descriptor(descriptor, before, operation="read"):
        for chunk in _regular_chunks(
            descriptor, before, maximum_bytes=maximum_bytes,
            chunk_bytes=CONTROL_PLANE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes,
            opened_size_bound=opened_size_bound, operation="read",
        ):
            chunks.append(chunk)
            count += len(chunk)
        if count != before.st_size:
            raise BoundedFileIOError("bounded input changed during read")
    return b"".join(chunks)


def _hash_regular_descriptor(
    descriptor: int,
    before: os.stat_result,
    *,
    maximum_bytes: int,
    opened_size_bound: bool = False,
    chunk_bytes: int | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with _stable_descriptor(descriptor, before, operation="hashing"):
        for chunk in _regular_chunks(
            descriptor, before, maximum_bytes=maximum_bytes,
            chunk_bytes=CONTROL_PLANE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes,
            opened_size_bound=opened_size_bound, operation="hashing",
        ):
            digest.update(chunk)
            count += len(chunk)
        if count != before.st_size:
            raise BoundedFileIOError("bounded input changed during hashing")
    return count, digest.hexdigest()


def _open_contained_regular(
    root: Path, path: Path, maximum_bytes: int
) -> tuple[int, os.stat_result]:
    """Open once, relative to no-follow directories; the caller owns the descriptor."""

    if maximum_bytes < 0:
        raise ValueError("bounded file maximum_bytes must be nonnegative")
    if not root.is_absolute() or not path.is_absolute():
        raise BoundedFileIOError("bounded root and member must be absolute")
    if ".." in root.parts or ".." in path.parts:
        raise BoundedFileIOError("bounded member cannot traverse parent directories")
    if not path.is_relative_to(root) or path == root:
        raise BoundedFileIOError("bounded member is outside its declared root")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise BoundedFileIOError("platform cannot enforce no-follow contained reads")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    parent = -1
    descriptor = -1
    try:
        # Also reject links in the root's ancestors, not only inside the root.
        parent = os.open(path.anchor, flags | directory)
        for component in path.parts[1:-1]:
            child = os.open(component, flags | directory, dir_fd=parent)
            os.close(parent)
            parent = child
        # O_NONBLOCK ensures a concurrently substituted FIFO cannot hang before
        # fstat rejects it. It has no effect on ordinary file reads.
        descriptor = os.open(
            path.name, flags | getattr(os, "O_NONBLOCK", 0), dir_fd=parent
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BoundedFileIOError("bounded input is not a regular file")
        if before.st_size > maximum_bytes:
            raise BoundedFileIOError("bounded input exceeds its byte limit")
        selected, descriptor = descriptor, -1
        return selected, before
    except OSError as error:
        raise BoundedFileIOError("bounded contained file cannot be read safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent >= 0:
            os.close(parent)



def read_bounded_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read at most ``maximum_bytes`` and reject identity drift during the read."""

    descriptor, before = _open_regular(path, maximum_bytes=maximum_bytes)
    return _read_regular_descriptor(descriptor, before, maximum_bytes=maximum_bytes)


def file_matches_bytes(
    path: Path, expected: bytes, *, maximum_bytes: int | None = None,
) -> bool:
    """Compare stable actual bytes with an independently established expectation.

    An explicit consumer bound applies to both the expectation and the actual
    regular file. Without it, retain the original expected-length bound.
    Nothing is cached across reads, and equality consumes all expected bytes.
    """

    limit = len(expected) if maximum_bytes is None else maximum_bytes
    if limit < 0:
        raise ValueError("maximum_bytes must be nonnegative")
    if len(expected) > limit:
        raise BoundedFileIOError("expected input exceeds its byte limit")
    descriptor, before = _open_regular(path, maximum_bytes=limit)
    offset = 0
    matches = before.st_size == len(expected)
    with _stable_descriptor(descriptor, before, operation="comparison"):
        while matches:
            chunk = os.read(descriptor, CONTROL_PLANE_CHUNK_BYTES)
            if not chunk:
                break
            if chunk != expected[offset : offset + len(chunk)]:
                matches = False
                break
            offset += len(chunk)
    return matches and offset == len(expected)


def bounded_file_sha256(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
    """Stream one regular file through SHA-256 under an explicit total-byte bound."""

    descriptor, before = _open_regular(path, maximum_bytes=maximum_bytes)
    return _hash_regular_descriptor(descriptor, before, maximum_bytes=maximum_bytes)


def bounded_file_contains_any(
    path: Path,
    needles: tuple[bytes, ...],
    *,
    maximum_bytes: int,
) -> bool:
    """Search a bounded file incrementally, including matches across chunk boundaries."""

    values = tuple(needle for needle in needles if needle)
    if not values:
        bounded_file_sha256(path, maximum_bytes=maximum_bytes)
        return False
    maximum_needle = max(len(needle) for needle in values)
    if maximum_needle > MAX_CONTROL_PLANE_JSON_BYTES:
        raise BoundedFileIOError("forbidden-payload probe exceeds its byte limit")
    descriptor, before = _open_regular(path, maximum_bytes=maximum_bytes)
    tail = b""
    observed_bytes = 0
    found = False
    with _stable_descriptor(descriptor, before, operation="search"):
        while True:
            chunk = os.read(descriptor, CONTROL_PLANE_CHUNK_BYTES)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise BoundedFileIOError("bounded input exceeds its byte limit")
            window = tail + chunk
            if any(needle in window for needle in values):
                found = True
                break
            tail = window[-(maximum_needle - 1) :] if maximum_needle > 1 else b""
        if found:
            os.lseek(descriptor, 0, os.SEEK_END)
    return found
