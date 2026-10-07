"""Authenticate an audit word against all twelve nominal native deliveries."""

from decimal import Decimal as D
import numpy as np
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.controller_study import ImplementationBinding
from empirical_lawhood.runtime.controller_runtime import ExactActionDeliveryTrace


def measured_trace(
    *,
    data: dict[str, np.ndarray],
    key: str,
    word: OccurrenceActionWord,
    implementation: ImplementationBinding,
    commitment: ObjectIdentity,
    kind: ScientificCommitmentKind,
    trace_id: str,
    guard_s: int = 120,
) -> ExactActionDeliveryTrace | None:
    """The dt=.5 measurement is a numerical view of this nominal delivered word."""
    if guard_s not in (120, 240):
        raise ValueError("undeclared bounded native trace")
    if not all(f"{key}_{f}" in data for f in ("requests", "stages", "exposure")):
        return None
    requested, stages, exposure = (data[f"{key}_{f}"] for f in ("requests", "stages", "exposure"))
    if (
        requested.shape != (guard_s // 10, 2)
        or stages.shape != (guard_s // 10, 4)
        or exposure.shape != (guard_s // 10, 10, 4)
    ):
        return None
    origin = word.occurrences[0].requested.coordinate.coordinate
    elapsed = D(0)
    for occurrence in word.occurrences:
        native_time = occurrence.realized.coordinate.coordinate
        start = native_time - origin
        callback = int(start / 10)
        command_time = origin + callback * 10
        duration = occurrence.duration
        if (
            start != elapsed
            or duration <= 0
            or start != int(start)
            or duration != int(duration)
            or not 0 <= callback < guard_s // 10
            or occurrence.requested.coordinate.coordinate != command_time
            or occurrence.accepted.coordinate.coordinate != command_time
            or occurrence.applied.coordinate.coordinate != command_time
        ):
            return None
        if any(
            actual != D(repr(float(expected)))
            for actual, expected in (
                (occurrence.requested.value, requested[callback, 0]),
                (occurrence.accepted.value, stages[callback, 0]),
                (occurrence.applied.value, stages[callback, 2]),
            )
        ):
            return None
        flat = exposure.reshape(-1, 4)
        for step in range(int(start), int(start + duration)):
            if (
                step >= guard_s
                or D(repr(float(flat[step, 0]))) != origin + step
                or flat[step, 1] != 1
                or D(repr(float(flat[step, 2]))) != occurrence.realized.value
            ):
                return None
        elapsed += duration
    if elapsed != guard_s:
        return None
    observed = tuple(
        ObservedActionOccurrence(
            f"observed.{o.occurrence_id}",
            o.occurrence_id,
            o.requested,
            o.accepted,
            o.applied,
            o.realized,
            (),
        )
        for o in word.occurrences
    )
    return ExactActionDeliveryTrace(
        trace_id,
        commitment,
        implementation,
        kind,
        word,
        observed,
        OperationalDeliveryState.DELIVERED,
        (),
    )
