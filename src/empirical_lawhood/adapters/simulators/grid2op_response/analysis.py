"""Target-native Grid2Op complete-chronic reduction for independent substrate grounding."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateActionDeliveryStatus, IndependentSubstrateCompleteTargetPanel, IndependentSubstrateCompleteTargetUnit, IndependentSubstrateNativeActionLedger, IndependentSubstrateNativeReceiverPoint, IndependentSubstrateReceiverBackactionClass, IndependentSubstrateTargetPhase, IndependentSubstrateUnitStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import Grid2OpActionKind, Grid2OpEpisodeRequest, Grid2OpEpisodeResult, Grid2OpNativeStep, Grid2OpSourceBinding


def _delivery(step: Grid2OpNativeStep) -> IndependentSubstrateActionDeliveryStatus:
    if not step.realization_observed:
        return IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED
    if step.action_illegal or step.action_ambiguous:
        return IndependentSubstrateActionDeliveryStatus.REPLACED_WITH_HOLD
    return IndependentSubstrateActionDeliveryStatus.REALIZED


def reduce_grid2op_episode(
    *,
    source: Grid2OpSourceBinding,
    request: Grid2OpEpisodeRequest,
    result: Grid2OpEpisodeResult,
) -> IndependentSubstrateCompleteTargetUnit:
    """Reduce nested branch/horizon views without inflating chronic count."""

    if (
        result.request != ObjectIdentity.from_record(request.request_id, request)
        or result.source_binding != ObjectIdentity.from_record(source.binding_id, source)
        or result.chronic != request.chronic
        or result.outcome_access is not request.outcome_access
    ):
        raise ValueError("Grid2Op reduction operands differ")
    branch_by_id = {value.branch_id: value for value in request.branches}
    ledgers: list[IndependentSubstrateNativeActionLedger] = []
    points: list[IndependentSubstrateNativeReceiverPoint] = []
    sinks: set[str] = set()
    reasons: set[str] = set()
    observation_failure = False
    for trace in result.traces:
        branch = branch_by_id.get(trace.branch.object_id)
        if branch is None:
            raise ValueError("Grid2Op result contains an unissued branch")
        first = trace.steps[0]
        final = trace.steps[-1]
        delivery = _delivery(first)
        ledger_reasons: set[str] = set()
        if first.action_illegal:
            ledger_reasons.add("GRID2OP_ACTION_ILLEGAL_REPLACED_WITH_HOLD")
        if first.action_ambiguous:
            ledger_reasons.add("GRID2OP_ACTION_AMBIGUOUS_REPLACED_WITH_HOLD")
        if not first.realization_observed:
            ledger_reasons.add("GRID2OP_REALIZED_ACTION_UNOBSERVED")
        if first.has_error:
            ledger_reasons.add("GRID2OP_ACTION_DELIVERY_FAILED")
        ledgers.append(
            IndependentSubstrateNativeActionLedger(
                ledger_id=f"ledger.{request.unit_id}.{branch.branch_id}",
                native_action_id=branch.action.action_id,
                requested_action_code=first.requested_action_code,
                accepted_action_code=first.accepted_action_code,
                applied_action_code=first.applied_action_code,
                realized_action_code=first.realized_action_code,
                request_tick=request.initial_step,
                acceptance_tick=request.initial_step,
                application_tick=request.initial_step + 1,
                realization_tick=(request.initial_step + 1 if first.realization_observed else None),
                legal=not first.action_illegal and not first.action_ambiguous,
                ambiguous=first.action_ambiguous,
                hold_requested=branch.action.kind is Grid2OpActionKind.HOLD,
                delivery_status=delivery,
                reason_codes=tuple(sorted(ledger_reasons)),
            )
        )
        for step in trace.steps:
            if step.has_error or step.done:
                sinks.add("GRID2OP_EPISODE_TERMINAL_SINK")
            if not step.observation_valid:
                observation_failure = True
                reasons.update(step.reason_codes)
                continue
            assert step.maximum_rho is not None
            points.extend(
                (
                    IndependentSubstrateNativeReceiverPoint(
                        point_id=f"point.{request.unit_id}.{branch.branch_id}.{step.horizon_step:06d}.thermal",
                        receiver_id="maximum-rho",
                        gauge_id="thermal",
                        native_unit="1",
                        value=step.maximum_rho,
                        horizon_tick=step.environment_tick,
                        uncertainty=Decimal("0"),
                        topology_state_id=step.topology_state_id,
                        valid=True,
                        backaction_class=IndependentSubstrateReceiverBackactionClass.PASSIVE_BOUNDED,
                        reason_codes=(),
                    ),
                    IndependentSubstrateNativeReceiverPoint(
                        point_id=f"point.{request.unit_id}.{branch.branch_id}.{step.horizon_step:06d}.topology",
                        receiver_id="connected-components",
                        gauge_id="topology",
                        native_unit="count",
                        value=Decimal(step.connected_component_count),
                        horizon_tick=step.environment_tick,
                        uncertainty=Decimal("0"),
                        topology_state_id=step.topology_state_id,
                        valid=True,
                        backaction_class=IndependentSubstrateReceiverBackactionClass.PASSIVE_BOUNDED,
                        reason_codes=(),
                    ),
                )
            )
        if final.has_error:
            reasons.add("GRID2OP_BACKEND_ERROR_RETAINED")
    if sinks:
        status = IndependentSubstrateUnitStatus.PHYSICAL_OR_NUMERICAL_SINK
        reasons.add("GRID2OP_SINK_UNIT_RETAINED")
    elif observation_failure:
        status = IndependentSubstrateUnitStatus.OBSERVATION_FAILURE
        reasons.add("GRID2OP_OBSERVATION_FAILURE_RETAINED")
    elif any(
        ledger.delivery_status
        in {
            IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED,
            IndependentSubstrateActionDeliveryStatus.DELIVERY_FAILED,
        }
        for ledger in ledgers
    ):
        status = IndependentSubstrateUnitStatus.OBSERVATION_FAILURE
        reasons.add("GRID2OP_ACTION_REALIZATION_INCOMPLETE")
    else:
        status = IndependentSubstrateUnitStatus.COMPLETE
    if status is IndependentSubstrateUnitStatus.COMPLETE:
        reasons.clear()
    source_token = source.fingerprint()[:16]
    chronic_token = request.chronic.object_fingerprint[:16]
    return IndependentSubstrateCompleteTargetUnit(
        unit_id=request.unit_id,
        slot=IndependentSubstrateTargetKind.GRID2OP,
        phase=request.phase,
        denominator_id=f"denominator.grid2op.{source_token}.{chronic_token}",
        history_id=(f"history.grid2op.{chronic_token}.step-{request.initial_step:06d}"),
        source_view_id=(f"view.grid2op.{source.observation_operator_sha256[:16]}.native-graph"),
        preparation_or_chronic_id=request.chronic.object_id,
        action_ledgers=tuple(sorted(ledgers, key=lambda value: value.ledger_id)),
        receiver_points=tuple(sorted(points, key=lambda value: value.point_id)),
        sink_codes=tuple(sorted(sinks)),
        nested_view_count=len(ledgers),
        status=status,
        outcome_access=request.outcome_access,
        reason_codes=tuple(sorted(reasons)),
    )


def build_grid2op_panel(
    *,
    source: Grid2OpSourceBinding,
    phase: IndependentSubstrateTargetPhase,
    planned_unit_ids: tuple[str, ...],
    units: tuple[IndependentSubstrateCompleteTargetUnit, ...],
) -> IndependentSubstrateCompleteTargetPanel:
    return IndependentSubstrateCompleteTargetPanel(
        panel_id=f"panel.independent-substrate-grounding.grid2op.{phase.value.lower()}",
        slot=IndependentSubstrateTargetKind.GRID2OP,
        phase=phase,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        planned_unit_ids=planned_unit_ids,
        units=units,
        postissue_outcome_exclusion_allowed=False,
    )


__all__ = ["build_grid2op_panel", "reduce_grid2op_episode"]
