"""Outcome-blind operational progress events for bounded FreeGSNKE workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import sys
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id


FREEGSNKE_PROGRESS_PREFIX = "EMPIRICAL_LAWHOOD_FREEGSNKE_PROGRESS "


class FreeGsnkeProgressStage(StrEnum):
    WORKER_STARTED = "WORKER_STARTED"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    STATIC_RESTORE_COMPLETED = "STATIC_RESTORE_COMPLETED"
    LINEARIZATION_COMPLETED = "LINEARIZATION_COMPLETED"
    DYNAMIC_STEPS_STARTED = "DYNAMIC_STEPS_STARTED"
    DYNAMIC_STEP_MILESTONE = "DYNAMIC_STEP_MILESTONE"
    LINEARIZATION_DOMAIN_STOP = "LINEARIZATION_DOMAIN_STOP"
    DYNAMIC_GS_CONVERGENCE_STOP = "DYNAMIC_GS_CONVERGENCE_STOP"
    WORKER_RESPONSE_READY = "WORKER_RESPONSE_READY"
    WORKER_EXITED = "WORKER_EXITED"


@dataclass(frozen=True, slots=True)
class FreeGsnkeProgressEvent(CanonicalRecord):
    """Non-scientific heartbeat containing no receiver or verdict operand."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-progress-event'

    event_id: str
    subject_id: str
    stage: FreeGsnkeProgressStage
    ordinal: int
    elapsed_seconds: int

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.subject_id, field_name="subject_id")
        if self.ordinal < 0 or self.elapsed_seconds < 0:
            raise ValueError("FreeGSNKE progress counters must be nonnegative")
        if self.stage is FreeGsnkeProgressStage.WORKER_STARTED and (
            self.ordinal or self.elapsed_seconds
        ):
            raise ValueError("FreeGSNKE worker start counters must be zero")


def emit_freegsnke_progress(
    *,
    subject_id: str,
    stage: FreeGsnkeProgressStage,
    ordinal: int,
    elapsed_seconds: int,
) -> None:
    """Emit one flushed, outcome-blind stderr event from a worker process."""

    event = FreeGsnkeProgressEvent(
        event_id=(f"progress.{subject_id}.{stage.value.lower().replace('_', '-')}.{ordinal:08d}"),
        subject_id=subject_id,
        stage=stage,
        ordinal=ordinal,
        elapsed_seconds=elapsed_seconds,
    )
    write_freegsnke_progress(event)


def write_freegsnke_progress(event: FreeGsnkeProgressEvent) -> None:
    """Write one already validated event as a flushed canonical stderr line."""

    # Canonical records already carry their terminal newline.  Writing through
    # ``print`` would add a second one and make the payload non-canonical after
    # the transport splits it into lines.
    sys.stderr.buffer.write(FREEGSNKE_PROGRESS_PREFIX.encode("ascii") + event.canonical_bytes())
    sys.stderr.buffer.flush()


__all__ = [
    "FREEGSNKE_PROGRESS_PREFIX",
    'FreeGsnkeProgressEvent',
    "FreeGsnkeProgressStage",
    "emit_freegsnke_progress",
    "write_freegsnke_progress",
]
