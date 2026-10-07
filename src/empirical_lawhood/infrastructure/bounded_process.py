"""Small, timeout- and byte-bounded boundary for fixed local diagnostics."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast


PROCESS_IO_CHUNK_BYTES = 64 * 1024


class BoundedProcessError(RuntimeError):
    """A fixed local command exceeded or violated its process-I/O contract."""


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def _close_registered(selector: selectors.BaseSelector, stream: IO[bytes]) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    stream.close()


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    group_kill_failed = False
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - the supported production host is POSIX
            process.kill()
    except OSError:
        group_kill_failed = True
    if group_kill_failed and process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_bounded_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    input_bytes: bytes = b"",
    timeout_seconds: float = 5.0,
    maximum_input_bytes: int = 0,
    maximum_stdout_bytes: int = 64 * 1024,
    maximum_stderr_bytes: int = 64 * 1024,
) -> BoundedProcessResult:
    """Run one fixed argv while incrementally bounding every buffered channel."""

    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("bounded process argv must be a nonempty string sequence")
    if not isinstance(input_bytes, bytes):
        raise ValueError("bounded process input must be immutable bytes")
    if timeout_seconds <= 0:
        raise ValueError("bounded process timeout must be positive")
    if min(maximum_input_bytes, maximum_stdout_bytes, maximum_stderr_bytes) < 0:
        raise ValueError("bounded process I/O ceilings must be nonnegative")
    if len(input_bytes) > maximum_input_bytes:
        raise BoundedProcessError("bounded local command input exceeds its byte ceiling")

    try:
        process = subprocess.Popen(
            tuple(argv),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise BoundedProcessError("bounded local command could not start") from error
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    streams = (process.stdin, process.stdout, process.stderr)
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    input_offset = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        if input_bytes:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BoundedProcessError("bounded local command timed out")
            events = selector.select(remaining)
            if not events:
                raise BoundedProcessError("bounded local command timed out")
            for key, _mask in events:
                stream = cast(IO[bytes], key.fileobj)
                if key.data == "stdin":
                    try:
                        written = os.write(
                            process.stdin.fileno(),
                            input_bytes[input_offset : input_offset + PROCESS_IO_CHUNK_BYTES],
                        )
                    except BrokenPipeError:
                        written = 0
                    input_offset += written
                    if written == 0 or input_offset == len(input_bytes):
                        _close_registered(selector, process.stdin)
                    continue

                target = stdout if key.data == "stdout" else stderr
                limit = maximum_stdout_bytes if key.data == "stdout" else maximum_stderr_bytes
                chunk = os.read(
                    stream.fileno(),
                    min(PROCESS_IO_CHUNK_BYTES, max(1, limit - len(target) + 1)),
                )
                if not chunk:
                    _close_registered(selector, stream)
                    continue
                target.extend(chunk)
                if len(target) > limit:
                    raise BoundedProcessError(
                        f"bounded local command {key.data} exceeds its byte ceiling"
                    )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BoundedProcessError("bounded local command timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise BoundedProcessError("bounded local command timed out") from error
        # A fixed diagnostic has no authority to leave background descendants.
        # The parent may already be reaped while a child still owns the process
        # group, so cleanup must not be conditional on ``process.poll()``.
        _kill_process_tree(process)
        return BoundedProcessResult(
            returncode=returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    except BaseException:
        _kill_process_tree(process)
        raise
    finally:
        selector.close()
        for stream in streams:
            if not stream.closed:
                stream.close()


__all__ = [
    "BoundedProcessError",
    "BoundedProcessResult",
    "run_bounded_command",
]
