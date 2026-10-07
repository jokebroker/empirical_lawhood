"""Standard-library-only environment setup before a worker interpreter starts."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.process import BaseProcess


NUMERICAL_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_SPAWN_ENVIRONMENT_LOCK = threading.Lock()


def join_worker_process(process: BaseProcess, *, timeout: float | None = None) -> None:
    """Collect exit status without racing multiprocessing's start-time cleanup.

    Process.start() polls every child in its global registry. A concurrent
    waitpid in join() can otherwise see ECHILD before that poll has stored the
    exit code. Use the same short lifecycle boundary as owned process starts.
    """

    with _SPAWN_ENVIRONMENT_LOCK:
        process.join(timeout=timeout)


@contextmanager
def numerical_spawn_environment() -> Iterator[None]:
    """Serialize owned launches and restore the host environment even on failure.

    Spawned Python interpreters inherit these values before importing runner
    modules. Forked or previously started forkserver processes additionally
    restrict already loaded numerical libraries at the worker boundary.
    """

    with _SPAWN_ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in NUMERICAL_THREAD_ENV_VARS}
        try:
            os.environ.update(dict.fromkeys(NUMERICAL_THREAD_ENV_VARS, "1"))
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
