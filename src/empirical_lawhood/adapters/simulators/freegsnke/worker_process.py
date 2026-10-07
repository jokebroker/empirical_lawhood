"""Bounded canonical subprocess transport with optional safe progress streaming."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .progress import FREEGSNKE_PROGRESS_PREFIX, FreeGsnkeProgressEvent, FreeGsnkeProgressStage


FreeGsnkeProgressSink = Callable[[FreeGsnkeProgressEvent], None]


def _emit_runtime_progress(
    progress_sink: FreeGsnkeProgressSink,
    *,
    subject_id: str,
    stage: FreeGsnkeProgressStage,
    ordinal: int,
    elapsed_seconds: int,
) -> None:
    """Keep a reporting failure from changing scientific execution."""

    event = FreeGsnkeProgressEvent(
        event_id=(f"progress.{subject_id}.{stage.value.lower().replace('_', '-')}.{ordinal:08d}"),
        subject_id=subject_id,
        stage=stage,
        ordinal=ordinal,
        elapsed_seconds=elapsed_seconds,
    )
    try:
        progress_sink(event)
    except Exception:  # noqa: BLE001 - progress is explicitly non-authoritative
        return


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def _progress_lines(
    pending: bytearray,
    chunk: bytes,
    *,
    subject_id: str,
    progress_sink: FreeGsnkeProgressSink,
    final: bool = False,
) -> None:
    pending.extend(chunk)
    lines = pending.splitlines(keepends=True)
    pending.clear()
    if lines and not lines[-1].endswith((b"\n", b"\r")) and not final:
        pending.extend(lines.pop())
    prefix = FREEGSNKE_PROGRESS_PREFIX.encode("ascii")
    for line in lines:
        stripped = line.rstrip(b"\r\n")
        if not stripped.startswith(prefix):
            continue
        event = decode_canonical_bytes(
            bytes(stripped[len(prefix) :]) + b"\n",
            FreeGsnkeProgressEvent,
            maximum_bytes=64 * 1024,
        )
        if event.subject_id != subject_id:
            raise ValueError("FreeGSNKE progress event changed its subject")
        try:
            progress_sink(event)
        except Exception:  # noqa: BLE001 - reporting cannot alter the task
            continue


def run_freegsnke_canonical_process(
    *,
    command: Sequence[str],
    payload: bytes,
    working_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    progress_subject_id: str,
    progress_sink: FreeGsnkeProgressSink | None = None,
    heartbeat_seconds: float = 60.0,
) -> tuple[int, bytes, bytes]:
    """Run one canonical worker and optionally emit only typed safe heartbeats."""

    if timeout_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("FreeGSNKE worker timing bounds must be positive")
    if progress_sink is None:
        completed = subprocess.run(
            command,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_directory,
            env=dict(environment),
            timeout=timeout_seconds,
            check=False,
        )
        if len(completed.stdout) > maximum_stdout_bytes:
            raise ValueError("FreeGSNKE worker stdout exceeds its bounded contract")
        if len(completed.stderr) > maximum_stderr_bytes:
            raise ValueError("FreeGSNKE worker stderr exceeds its bounded contract")
        return completed.returncode, completed.stdout, completed.stderr

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory,
        env=dict(environment),
        start_new_session=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    progress_pending = bytearray()
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_seconds
    progress_ordinal = 0
    _emit_runtime_progress(
        progress_sink,
        subject_id=progress_subject_id,
        stage=FreeGsnkeProgressStage.WORKER_STARTED,
        ordinal=progress_ordinal,
        elapsed_seconds=0,
    )
    try:
        process.stdin.write(payload)
        process.stdin.close()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            wait_seconds = min(remaining, max(0.0, next_heartbeat - now), 1.0)
            for key, _mask in selector.select(timeout=wait_seconds):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = stdout if key.data == "stdout" else stderr
                target.extend(chunk)
                limit = maximum_stdout_bytes if key.data == "stdout" else maximum_stderr_bytes
                if len(target) > limit:
                    raise ValueError(f"FreeGSNKE worker {key.data} exceeds its bounded contract")
                if key.data == "stderr":
                    _progress_lines(
                        progress_pending,
                        chunk,
                        subject_id=progress_subject_id,
                        progress_sink=progress_sink,
                    )
            now = time.monotonic()
            if now >= next_heartbeat and process.poll() is None:
                progress_ordinal += 1
                _emit_runtime_progress(
                    progress_sink,
                    subject_id=progress_subject_id,
                    stage=FreeGsnkeProgressStage.WORKER_HEARTBEAT,
                    ordinal=progress_ordinal,
                    elapsed_seconds=int(now - started),
                )
                next_heartbeat = now + heartbeat_seconds
        _progress_lines(
            progress_pending,
            b"",
            subject_id=progress_subject_id,
            progress_sink=progress_sink,
            final=True,
        )
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        progress_ordinal += 1
        _emit_runtime_progress(
            progress_sink,
            subject_id=progress_subject_id,
            stage=FreeGsnkeProgressStage.WORKER_EXITED,
            ordinal=progress_ordinal,
            elapsed_seconds=int(time.monotonic() - started),
        )
        return returncode, bytes(stdout), bytes(stderr)
    except BaseException:
        _terminate(process)
        progress_ordinal += 1
        _emit_runtime_progress(
            progress_sink,
            subject_id=progress_subject_id,
            stage=FreeGsnkeProgressStage.WORKER_EXITED,
            ordinal=progress_ordinal,
            elapsed_seconds=int(time.monotonic() - started),
        )
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


__all__ = [
    "FreeGsnkeProgressSink",
    "run_freegsnke_canonical_process",
]
