"""Outcome-blind NBI event nomination for one physical MAST shot."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .contracts import (
    MastActionLabel,
    MastActionTrace,
    MastEvent,
    MastEventDisposition,
    MastScalarSample,
)


@dataclass(frozen=True, slots=True)
class MastEventRule(CanonicalRecord):
    """Development rule; issuing final numeric science requires a later freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-event-rule'

    rule_id: str
    pre_window_start_s: Decimal
    pre_window_end_s: Decimal
    post_window_start_s: Decimal
    post_window_end_s: Decimal
    leading_boundary_exclusion_s: Decimal
    trailing_boundary_exclusion_s: Decimal
    active_change_threshold_w: Decimal
    hold_change_tolerance_w: Decimal
    simultaneous_actuator_threshold: Decimal
    interpolation_record_id: str
    development_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.rule_id, field_name="rule_id")
        validate_stable_id(self.interpolation_record_id, field_name="interpolation_record_id")
        for name, value in (
            ("leading_boundary_exclusion_s", self.leading_boundary_exclusion_s),
            ("trailing_boundary_exclusion_s", self.trailing_boundary_exclusion_s),
            ("active_change_threshold_w", self.active_change_threshold_w),
            ("hold_change_tolerance_w", self.hold_change_tolerance_w),
            ("simultaneous_actuator_threshold", self.simultaneous_actuator_threshold),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        for name, value in (
            ("pre_window_start_s", self.pre_window_start_s),
            ("pre_window_end_s", self.pre_window_end_s),
            ("post_window_start_s", self.post_window_start_s),
            ("post_window_end_s", self.post_window_end_s),
        ):
            validate_decimal(value, field_name=name)
        if not (
            self.pre_window_start_s < self.pre_window_end_s < 0
            and 0 < self.post_window_start_s < self.post_window_end_s
        ):
            raise ValueError("event windows must lie strictly before/after t0")
        if self.active_change_threshold_w <= self.hold_change_tolerance_w:
            raise ValueError("active-action threshold must exceed HOLD tolerance")
        if not self.development_only:
            raise ValueError("Phase 4 event rule is development-only and unissued")


def decode_mast_event_rule(payload: bytes) -> MastEventRule:
    return decode_canonical_bytes(payload, MastEventRule, maximum_bytes=64 * 1024)


def build_mast_event(trace: MastActionTrace, rule: MastEventRule) -> MastEvent:
    """Select at most one event using only the realized NBI/auxiliary traces."""

    valid = tuple(value for value in trace.nbi_power if value.valid and value.value is not None)
    rule_identity = ObjectIdentity.from_record(rule.rule_id, rule)
    clock_id = trace.nbi_power[0].clock_id
    if len(valid) < 4 or any(value.clock_id != clock_id for value in valid):
        return _empty_event(trace, rule, rule_identity, clock_id, "ACTION_TRACE_INVALID")
    start = valid[0].coordinate_s
    end = valid[-1].coordinate_s
    candidates: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    boundary_seen = False
    # Nominate only native action discontinuities.  Sliding every sample over a
    # step would manufacture many nested "events" from one physical shot.
    pivots = tuple(
        current
        for previous, current in zip(valid, valid[1:], strict=False)
        if current.value is not None
        and previous.value is not None
        and abs(current.value - previous.value) > rule.hold_change_tolerance_w
    )
    for pivot in pivots:
        t0 = pivot.coordinate_s
        if (
            t0 < start + rule.leading_boundary_exclusion_s
            or t0 > end - rule.trailing_boundary_exclusion_s
        ):
            boundary_seen = True
            continue
        before = _window_values(
            valid,
            t0 + rule.pre_window_start_s,
            t0 + rule.pre_window_end_s,
        )
        after = _window_values(
            valid,
            t0 + rule.post_window_start_s,
            t0 + rule.post_window_end_s,
        )
        if not before or not after:
            continue
        pre = Decimal(median(before))
        post = Decimal(median(after))
        candidates.append((t0, pre, post, post - pre))

    active = tuple(value for value in candidates if abs(value[3]) >= rule.active_change_threshold_w)
    if active:
        maximum = max(abs(value[3]) for value in active)
        strongest = tuple(value for value in active if abs(value[3]) == maximum)
        if len(strongest) != 1:
            return _empty_event(trace, rule, rule_identity, clock_id, "MULTIPLE_EQUAL_EVENTS")
        t0, pre, post, change = strongest[0]
        if _simultaneous_actuator_change(trace, t0, rule):
            return MastEvent(
                event_id=f"mast-event-{trace.shot_id}",
                shot_id=trace.shot_id,
                campaign=trace.campaign,
                t0_s=t0,
                action=MastActionLabel.UP if change > 0 else MastActionLabel.DOWN,
                disposition=MastEventDisposition.AMBIGUOUS,
                pre_power_w=pre,
                post_power_w=post,
                delta_power_w=change,
                native_clock_id=clock_id,
                interpolation_record_id=rule.interpolation_record_id,
                rule=rule_identity,
                reason_codes=("SIMULTANEOUS_ACTUATOR_CHANGE",),
            )
        return MastEvent(
            event_id=f"mast-event-{trace.shot_id}",
            shot_id=trace.shot_id,
            campaign=trace.campaign,
            t0_s=t0,
            action=MastActionLabel.UP if change > 0 else MastActionLabel.DOWN,
            disposition=MastEventDisposition.ACCEPTED,
            pre_power_w=pre,
            post_power_w=post,
            delta_power_w=change,
            native_clock_id=clock_id,
            interpolation_record_id=rule.interpolation_record_id,
            rule=rule_identity,
            reason_codes=(),
        )

    if not pivots:
        pivot = valid[len(valid) // 2]
        t0 = pivot.coordinate_s
        before = _window_values(
            valid,
            t0 + rule.pre_window_start_s,
            t0 + rule.pre_window_end_s,
        )
        after = _window_values(
            valid,
            t0 + rule.post_window_start_s,
            t0 + rule.post_window_end_s,
        )
        if not before or not after:
            return _empty_event(
                trace,
                rule,
                rule_identity,
                clock_id,
                "BOUNDARY_EFFECT",
            )
        pre = Decimal(median(before))
        post = Decimal(median(after))
        change = post - pre
    else:
        change = rule.hold_change_tolerance_w + Decimal(1)
        t0 = pre = post = Decimal(0)
    if abs(change) <= rule.hold_change_tolerance_w:
        return MastEvent(
            event_id=f"mast-event-{trace.shot_id}",
            shot_id=trace.shot_id,
            campaign=trace.campaign,
            t0_s=t0,
            action=MastActionLabel.HOLD,
            disposition=MastEventDisposition.CANDIDATE,
            pre_power_w=pre,
            post_power_w=post,
            delta_power_w=change,
            native_clock_id=clock_id,
            interpolation_record_id=rule.interpolation_record_id,
            rule=rule_identity,
            reason_codes=(),
        )
    reason = "BOUNDARY_EFFECT" if boundary_seen and not candidates else "NO_ELIGIBLE_EVENT"
    return _empty_event(trace, rule, rule_identity, clock_id, reason)


def _window_values(
    samples: tuple[MastScalarSample, ...],
    lower: Decimal,
    upper: Decimal,
) -> tuple[Decimal, ...]:
    return tuple(
        value.value
        for value in samples
        if lower <= value.coordinate_s <= upper and value.value is not None
    )


def _simultaneous_actuator_change(
    trace: MastActionTrace,
    t0: Decimal,
    rule: MastEventRule,
) -> bool:
    before = _window_values(
        tuple(value for value in trace.auxiliary_actuator if value.valid),
        t0 + rule.pre_window_start_s,
        t0 + rule.pre_window_end_s,
    )
    after = _window_values(
        tuple(value for value in trace.auxiliary_actuator if value.valid),
        t0 + rule.post_window_start_s,
        t0 + rule.post_window_end_s,
    )
    if not before or not after:
        return False
    return abs(Decimal(median(after)) - Decimal(median(before))) >= (
        rule.simultaneous_actuator_threshold
    )


def _empty_event(
    trace: MastActionTrace,
    rule: MastEventRule,
    rule_identity: ObjectIdentity,
    clock_id: str,
    reason: str,
) -> MastEvent:
    disposition = (
        MastEventDisposition.AMBIGUOUS
        if reason in {"MULTIPLE_EQUAL_EVENTS", "ACTION_TRACE_INVALID"}
        else MastEventDisposition.EXCLUDED
    )
    return MastEvent(
        event_id=f"mast-event-{trace.shot_id}",
        shot_id=trace.shot_id,
        campaign=trace.campaign,
        t0_s=None,
        action=None,
        disposition=disposition,
        pre_power_w=None,
        post_power_w=None,
        delta_power_w=None,
        native_clock_id=clock_id,
        interpolation_record_id=rule.interpolation_record_id,
        rule=rule_identity,
        reason_codes=(reason,),
    )


__all__ = ["MastEventRule", "build_mast_event", "decode_mast_event_rule"]
