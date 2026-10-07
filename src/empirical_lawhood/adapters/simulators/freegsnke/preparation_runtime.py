"""Bounded process boundary for FreeGSNKE preparation qualification."""

from __future__ import annotations

import os
from typing import Sequence

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .preparation import FreeGsnkePreparationWorkerInput, FreeGsnkePreparedUnit
from .runtime import (
    FREEGSNKE_MAX_REQUEST_BYTES,
    FREEGSNKE_MAX_STDERR_BYTES,
    FREEGSNKE_PREPARATION_TIMEOUT_SECONDS,
    FreeGsnkeWorkerInvocation,
    task_local_freegsnke_invocation,
)
from .worker_process import FreeGsnkeProgressSink, run_freegsnke_canonical_process


def run_freegsnke_preparation_worker(
    *,
    worker_input: FreeGsnkePreparationWorkerInput,
    invocation: FreeGsnkeWorkerInvocation,
    progress_sink: FreeGsnkeProgressSink | None = None,
) -> tuple[FreeGsnkePreparedUnit, bytes]:
    """Execute one exact recipe and retain a typed qualification stop."""

    payload = worker_input.canonical_bytes()
    if len(payload) > 2 * FREEGSNKE_MAX_REQUEST_BYTES:
        raise ValueError("FreeGSNKE preparation request exceeds its bounded contract")
    task_invocation = task_local_freegsnke_invocation(invocation)
    command: Sequence[str] = (
        os.fspath(task_invocation.python_executable),
        "-S",
        os.fspath(task_invocation.worker_script),
    )
    returncode, stdout, stderr = run_freegsnke_canonical_process(
        command=command,
        payload=payload,
        working_directory=task_invocation.working_directory,
        environment=task_invocation.environment,
        timeout_seconds=FREEGSNKE_PREPARATION_TIMEOUT_SECONDS,
        maximum_stdout_bytes=32 * 1024**2,
        maximum_stderr_bytes=FREEGSNKE_MAX_STDERR_BYTES,
        progress_subject_id=worker_input.input_id,
        progress_sink=progress_sink,
    )
    if returncode != 0:
        excerpt = stderr[-4096:].decode("utf-8", errors="replace")
        raise RuntimeError(f"FreeGSNKE preparation worker failed ({returncode}): {excerpt}")
    result = decode_canonical_bytes(
        stdout,
        FreeGsnkePreparedUnit,
        maximum_bytes=32 * 1024**2,
    )
    if (
        result.worker_input != ObjectIdentity.from_record(worker_input.input_id, worker_input)
        or result.recipe != worker_input.recipe
    ):
        raise ValueError("FreeGSNKE preparation worker changed its frozen input")
    return result, stderr


__all__ = ["run_freegsnke_preparation_worker"]
