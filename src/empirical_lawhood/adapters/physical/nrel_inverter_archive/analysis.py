"""Preparation-level reduction for safely decoded NREL inverter records."""

from __future__ import annotations

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateActionDeliveryStatus, IndependentSubstrateCompleteTargetPanel, IndependentSubstrateCompleteTargetUnit, IndependentSubstrateNativeActionLedger, IndependentSubstrateNativeReceiverPoint, IndependentSubstrateUnitStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import NRELActionKind, NRELArchiveDecodeResult, NRELArchiveSourceBinding, NRELNativeSample, NRELPreparationTrace, NRELSafeDecodeProfile


def _delivery(sample: NRELNativeSample) -> IndependentSubstrateActionDeliveryStatus:
    if not sample.action_accepted:
        return IndependentSubstrateActionDeliveryStatus.NOT_ACCEPTED
    complete = all(
        value is not None
        for value in (
            sample.accepted_action_code,
            sample.applied_action_code,
            sample.realized_action_code,
            sample.acceptance_tick,
            sample.application_tick,
            sample.realization_tick,
        )
    )
    if complete:
        return IndependentSubstrateActionDeliveryStatus.REALIZED
    applied = all(
        value is not None
        for value in (
            sample.accepted_action_code,
            sample.applied_action_code,
            sample.acceptance_tick,
            sample.application_tick,
        )
    )
    if applied and sample.realized_action_code is None and sample.realization_tick is None:
        return IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED
    return IndependentSubstrateActionDeliveryStatus.SEMANTICS_INCOMPLETE


def reduce_nrel_preparation(
    *,
    source: NRELArchiveSourceBinding,
    profile: NRELSafeDecodeProfile,
    result: NRELArchiveDecodeResult,
    trace: NRELPreparationTrace,
) -> IndependentSubstrateCompleteTargetUnit:
    """Reduce nested samples/channels while preserving one preparation unit."""

    if (
        result.source_binding != ObjectIdentity.from_record(source.binding_id, source)
        or result.decode_profile != ObjectIdentity.from_record(profile.profile_id, profile)
        or trace not in result.traces
        or result.outcome_access is not profile.outcome_access
    ):
        raise ValueError("NREL reduction operands differ")
    ledgers: list[IndependentSubstrateNativeActionLedger] = []
    points: list[IndependentSubstrateNativeReceiverPoint] = []
    sinks: set[str] = set()
    reasons: set[str] = set()
    observation_failure = False
    action_failure = False
    for sample in trace.samples:
        delivery = _delivery(sample)
        ledger_reasons: set[str] = set()
        if delivery is IndependentSubstrateActionDeliveryStatus.NOT_ACCEPTED:
            ledger_reasons.add("NREL_ACTION_NOT_ACCEPTED")
        elif delivery is IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED:
            ledger_reasons.add("NREL_REALIZED_ACTION_UNOBSERVED")
        elif delivery is IndependentSubstrateActionDeliveryStatus.SEMANTICS_INCOMPLETE:
            ledger_reasons.add("NREL_ACTION_SEMANTICS_INCOMPLETE")
        ledgers.append(
            IndependentSubstrateNativeActionLedger(
                ledger_id=f"ledger.{trace.preparation_id}.{sample.sample_id}",
                native_action_id=sample.action_id,
                requested_action_code=sample.requested_action_code,
                accepted_action_code=sample.accepted_action_code,
                applied_action_code=sample.applied_action_code,
                realized_action_code=sample.realized_action_code,
                request_tick=sample.request_tick,
                acceptance_tick=sample.acceptance_tick,
                application_tick=sample.application_tick,
                realization_tick=sample.realization_tick,
                legal=True,
                ambiguous=False,
                hold_requested=sample.action_kind is NRELActionKind.HOLD,
                delivery_status=delivery,
                reason_codes=tuple(sorted(ledger_reasons)),
            )
        )
        if delivery is not IndependentSubstrateActionDeliveryStatus.REALIZED:
            action_failure = True
            reasons.update(ledger_reasons)
        if sample.tripped:
            sinks.add("NREL_INVERTER_PROTECTION_TRIP")
        if not sample.observation_valid:
            observation_failure = True
            reasons.update(sample.reason_codes)
            continue
        assert sample.ac_power_w is not None
        assert sample.dc_power_w is not None
        assert sample.ac_uncertainty_w is not None
        assert sample.dc_uncertainty_w is not None
        points.extend(
            (
                IndependentSubstrateNativeReceiverPoint(
                    point_id=f"point.{trace.preparation_id}.{sample.sample_id}.ac-power",
                    receiver_id="ac-power",
                    gauge_id="ac-electrical",
                    native_unit="W",
                    horizon_tick=sample.sequence_index,
                    value=sample.ac_power_w,
                    uncertainty=sample.ac_uncertainty_w,
                    topology_state_id=None,
                    valid=True,
                    backaction_class=source.ac_receiver_backaction,
                    reason_codes=(),
                ),
                IndependentSubstrateNativeReceiverPoint(
                    point_id=f"point.{trace.preparation_id}.{sample.sample_id}.dc-power",
                    receiver_id="dc-power",
                    gauge_id="dc-electrical",
                    native_unit="W",
                    horizon_tick=sample.sequence_index,
                    value=sample.dc_power_w,
                    uncertainty=sample.dc_uncertainty_w,
                    topology_state_id=None,
                    valid=True,
                    backaction_class=source.dc_receiver_backaction,
                    reason_codes=(),
                ),
            )
        )
    if sinks:
        status = IndependentSubstrateUnitStatus.PHYSICAL_OR_NUMERICAL_SINK
        reasons.add("NREL_PHYSICAL_SINK_UNIT_RETAINED")
    elif observation_failure or action_failure:
        status = IndependentSubstrateUnitStatus.OBSERVATION_FAILURE
        reasons.add("NREL_INCOMPLETE_PHYSICAL_UNIT_RETAINED")
    else:
        status = IndependentSubstrateUnitStatus.COMPLETE
        reasons.clear()
    return IndependentSubstrateCompleteTargetUnit(
        unit_id=trace.preparation_id,
        slot=IndependentSubstrateTargetKind.NREL_INVERTER,
        phase=profile.phase,
        denominator_id=(f"denominator.nrel.{source.apparatus_id}.{source.site_id}"),
        history_id=f"history.nrel.{trace.preparation_id}.{trace.run_id}",
        source_view_id=f"view.nrel.{profile.fingerprint()[:16]}.ac-dc",
        preparation_or_chronic_id=trace.preparation_id,
        action_ledgers=tuple(sorted(ledgers, key=lambda value: value.ledger_id)),
        receiver_points=tuple(sorted(points, key=lambda value: value.point_id)),
        sink_codes=tuple(sorted(sinks)),
        nested_view_count=len(ledgers),
        status=status,
        outcome_access=profile.outcome_access,
        reason_codes=tuple(sorted(reasons)),
    )


def build_nrel_panel(
    *,
    source: NRELArchiveSourceBinding,
    profile: NRELSafeDecodeProfile,
    result: NRELArchiveDecodeResult,
) -> IndependentSubstrateCompleteTargetPanel:
    units = tuple(
        sorted(
            (
                reduce_nrel_preparation(
                    source=source,
                    profile=profile,
                    result=result,
                    trace=trace,
                )
                for trace in result.traces
            ),
            key=lambda value: value.unit_id,
        )
    )
    return IndependentSubstrateCompleteTargetPanel(
        panel_id=f"panel.independent-substrate-grounding.nrel.{profile.phase.value.lower()}",
        slot=IndependentSubstrateTargetKind.NREL_INVERTER,
        phase=profile.phase,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        planned_unit_ids=tuple(value.unit_id for value in units),
        units=units,
        postissue_outcome_exclusion_allowed=False,
    )


__all__ = ["build_nrel_panel", "reduce_nrel_preparation"]
