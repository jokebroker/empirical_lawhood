"""Authenticate the scalar feed word against saved native delivery stages."""
from decimal import Decimal as D

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.controller_study import ImplementationBinding
from empirical_lawhood.runtime.controller_runtime import ExactActionDeliveryTrace
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode


def measured_feed_trace(
    *, episode: NativeEpisode | None, callback: int, word: OccurrenceActionWord,
    implementation: ImplementationBinding, commitment: ObjectIdentity,
    kind: ScientificCommitmentKind, trace_id: str,
) -> ExactActionDeliveryTrace | None:
    if episode is None or min(len(episode.exposure), len(episode.requests), len(episode.stages)) <= callback:
        return None
    exposure = episode.exposure[callback]
    requested = D(repr(float(episode.requests[callback, 0])))
    accepted = D(repr(float(episode.stages[callback, 0])))
    applied = D(repr(float(episode.stages[callback, 2])))
    dt, clock = D(repr(episode.dt)), D(0)
    for occurrence in word.occurrences:
        start = occurrence.realized.coordinate.coordinate - D(callback * 10)
        duration = occurrence.duration
        if (start != clock or duration <= 0
                or start / dt != int(start / dt) or duration / dt != int(duration / dt)
                or occurrence.requested.coordinate.coordinate != D(callback * 10)
                or occurrence.accepted.coordinate.coordinate != D(callback * 10)
                or occurrence.requested.value != requested
                or occurrence.accepted.value != accepted or occurrence.applied.value != applied):
            return None
        first, last = int(start / dt), int((start + duration) / dt)
        if last > len(exposure) or any(
            D(repr(float(row[0]))) != D(callback * 10) + D(step) * dt
            or D(repr(float(row[1]))) != dt
            or D(repr(float(row[2]))) != occurrence.realized.value
            for step, row in enumerate(exposure[first:last], start=first)
        ):
            return None
        clock += duration
    if clock != D(10) or len(exposure) * dt != D(10):
        return None
    observed = tuple(ObservedActionOccurrence(
        f"observed.{o.occurrence_id}", o.occurrence_id,
        o.requested, o.accepted, o.applied, o.realized, (),
    ) for o in word.occurrences)
    return ExactActionDeliveryTrace(trace_id, commitment, implementation, kind,
                                      word, observed, OperationalDeliveryState.DELIVERED, ())
