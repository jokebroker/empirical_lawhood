# SPDX-License-Identifier: MPL-2.0

"""Single-use task accounting and OS limits for a declared local campaign."""

import json
import os
import resource
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import ClassVar

from empirical_lawhood.infrastructure.artifacts import GuardedExternalRoot
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.time import parse_utc_timestamp


@dataclass(frozen=True, slots=True)
class LocalCampaignAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/local-campaign-allocation'
    allocation_id: str
    execution_authority: ObjectIdentity
    resource_envelope: ObjectIdentity
    started_at_utc: str
    wall_seconds: int
    cpu_seconds: int
    worker_memory_bytes: int
    task_cpu_reservations: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.allocation_id)
        parse_utc_timestamp(self.started_at_utc)
        if (
            self.execution_authority.object_schema
            != 'empirical-lawhood/planning/study-operation-authority'
            or self.resource_envelope.object_schema
            != 'empirical-lawhood/runtime/execution-resource-envelope-spec'
            or min(self.wall_seconds, self.cpu_seconds, self.worker_memory_bytes) <= 0
            or not self.task_cpu_reservations
            or self.task_cpu_reservations != tuple(sorted(self.task_cpu_reservations))
            or len(dict(self.task_cpu_reservations)) != len(self.task_cpu_reservations)
            or any(seconds <= 1 for _, seconds in self.task_cpu_reservations)
            or sum(seconds for _, seconds in self.task_cpu_reservations)
            > self.cpu_seconds
        ):
            raise ValueError(
                "campaign allocation lacks exact authority or bounded task reservations"
            )
        for task_id, _ in self.task_cpu_reservations:
            validate_stable_id(task_id)


@dataclass(frozen=True)
class LocalCampaignProcessLimits:
    allocation: LocalCampaignAllocation
    root: GuardedExternalRoot
    relative_root: str

    def _cost(self, task_id: str, stage: str, data: dict[str, object]) -> None:
        path = self.root.resolve(
            f"{self.relative_root}/{task_id}.{stage}.json",
            for_write=True,
            operation_minimum_free_bytes=100 * 1024**3,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(raw) > 16384:
            raise ValueError("task accounting exceeds its bound")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def task(self, task_id: str) -> Iterator[None]:
        allocation = self.allocation
        seconds = dict(allocation.task_cpu_reservations).get(task_id)
        if seconds is None:
            raise ValueError("task outside frozen campaign allocation")
        remaining = (
            allocation.wall_seconds
            - (
                datetime.now(UTC) - datetime.fromisoformat(allocation.started_at_utc)
            ).total_seconds()
        )
        if not 0 < remaining <= allocation.wall_seconds:
            raise RuntimeError("retained campaign wall allocation expired")
        if any(
            os.environ.get(k) != "1"
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            raise RuntimeError("worker numerical thread limit absent")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("worker has an unaccounted wall timer")
        old_cpu, old_memory = (
            resource.getrlimit(resource.RLIMIT_CPU),
            resource.getrlimit(resource.RLIMIT_AS),
        )
        used = resource.getrusage(resource.RUSAGE_SELF)
        if used.ru_utime + used.ru_stime >= seconds - 1:
            raise RuntimeError("worker used its CPU reservation before task entry")
        target = (
            seconds - 1
            if old_cpu[1] == resource.RLIM_INFINITY
            else min(seconds - 1, old_cpu[1])
        )
        memory = (
            allocation.worker_memory_bytes
            if old_memory[1] == resource.RLIM_INFINITY
            else min(allocation.worker_memory_bytes, old_memory[1])
        )
        common = {
            "allocation_sha256": allocation.fingerprint(),
            "task_id": task_id,
            "reserved_cpu_seconds": seconds,
        }
        # Exclusive start prevents reacquisition after an interrupted scientific task.
        self._cost(
            task_id, "started", {**common, "at_utc": datetime.now(UTC).isoformat()}
        )
        started = monotonic()
        returned = False
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (target, old_cpu[1]))
            resource.setrlimit(resource.RLIMIT_AS, (memory, old_memory[1]))
            signal.setitimer(signal.ITIMER_REAL, remaining)
            yield
            returned = True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            resource.setrlimit(resource.RLIMIT_CPU, old_cpu)
            resource.setrlimit(resource.RLIMIT_AS, old_memory)
            own = resource.getrusage(resource.RUSAGE_SELF)
            self._cost(
                task_id,
                "finished",
                {
                    **common,
                    "body_returned": returned,
                    "at_utc": datetime.now(UTC).isoformat(),
                    "worker_lifetime_cpu_seconds": own.ru_utime + own.ru_stime,
                    "task_cpu_seconds": own.ru_utime
                    + own.ru_stime
                    - used.ru_utime
                    - used.ru_stime,
                    "task_wall_seconds": monotonic() - started,
                    "worker_peak_rss_bytes": own.ru_maxrss * 1024,
                },
            )
