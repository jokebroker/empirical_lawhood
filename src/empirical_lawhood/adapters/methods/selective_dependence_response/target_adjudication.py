"""Independent-axis target adjudication and compact handoff for selective dependence response."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS, SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID, SelectiveDependenceResponseAxisAdjudication, SelectiveDependenceResponseAxisName, SelectiveDependenceResponseAxisState, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseTargetAdjudication, SelectiveDependenceResponseTargetHandoff


def axis(
    target_id: str,
    name: SelectiveDependenceResponseAxisName,
    state: SelectiveDependenceResponseAxisState,
    *,
    decisive_case_ids: tuple[str, ...] = (),
    reason: str,
) -> SelectiveDependenceResponseAxisAdjudication:
    return SelectiveDependenceResponseAxisAdjudication(
        adjudication_id=f"axis.{target_id}.{name.value.lower().replace('_', '-')}",
        axis=name,
        state=state,
        decisive_case_ids=tuple(sorted(decisive_case_ids)),
        reason=reason,
    )


def adjudicate_target(
    *,
    target_id: str,
    forecast: ObjectIdentity,
    evaluation_unit_ids_sha256: str,
    issued_unit_count: int,
    source_valid_unit_count: int,
    evaluable_unit_count: int,
    stopped_unit_count: int,
    missing_unit_count: int,
    axes: tuple[SelectiveDependenceResponseAxisAdjudication, ...],
    exact_counterexample_ids: tuple[str, ...] = (),
) -> SelectiveDependenceResponseTargetAdjudication:
    """Apply construct/inference/counterexample precedence without scalar voting."""

    by_axis = {value.axis: value for value in axes}
    if set(by_axis) != set(SelectiveDependenceResponseAxisName) or len(by_axis) != len(axes):
        raise ValueError("target adjudication requires exactly one state per axis")
    construct_valid = by_axis[SelectiveDependenceResponseAxisName.CONSTRUCT].state is SelectiveDependenceResponseAxisState.VALID
    closed = issued_unit_count == evaluable_unit_count + stopped_unit_count + missing_unit_count
    if not construct_valid:
        dependent = (
            SelectiveDependenceResponseAxisName.LOCAL_RESPONSE_LAW,
            SelectiveDependenceResponseAxisName.SELECTIVE_DEPENDENCE,
            SelectiveDependenceResponseAxisName.SUPPORT_BOUNDARY,
            SelectiveDependenceResponseAxisName.RECEIVER_ADMISSION,
            SelectiveDependenceResponseAxisName.HOLD_VIABILITY,
            SelectiveDependenceResponseAxisName.PREDICTIVE_DISTINCTIVENESS,
        )
        if any(by_axis[name].state is not SelectiveDependenceResponseAxisState.UNEVALUABLE for name in dependent):
            raise ValueError("claim axes must be unevaluable when construct validity fails")
    if not closed and any(
        by_axis[name].state is not SelectiveDependenceResponseAxisState.UNEVALUABLE
        for name in (
            SelectiveDependenceResponseAxisName.LOCAL_RESPONSE_LAW,
            SelectiveDependenceResponseAxisName.SELECTIVE_DEPENDENCE,
            SelectiveDependenceResponseAxisName.PREDICTIVE_DISTINCTIVENESS,
        )
    ):
        raise ValueError("complete-unit accounting must close before scientific adjudication")
    if exact_counterexample_ids:
        opposed = {
            SelectiveDependenceResponseAxisName.LOCAL_RESPONSE_LAW,
            SelectiveDependenceResponseAxisName.SELECTIVE_DEPENDENCE,
            SelectiveDependenceResponseAxisName.SUPPORT_BOUNDARY,
            SelectiveDependenceResponseAxisName.RECEIVER_ADMISSION,
            SelectiveDependenceResponseAxisName.HOLD_VIABILITY,
        }
        if not construct_valid or not any(
            by_axis[name].state in {SelectiveDependenceResponseAxisState.OPPOSED, SelectiveDependenceResponseAxisState.NOT_VIABLE}
            for name in opposed
        ):
            raise ValueError("exact counterexamples require construct-valid opposition")
    if not construct_valid:
        maximum_claim = "The target relation was unevaluable because construct validity failed."
    elif any(value.state is SelectiveDependenceResponseAxisState.UNEVALUABLE for value in axes):
        maximum_claim = "At least one non-substitutable target axis was unevaluable."
    elif exact_counterexample_ids:
        maximum_claim = "At least one exact construct-valid case opposed the target relation."
    elif any(
        value.state in {SelectiveDependenceResponseAxisState.OPPOSED, SelectiveDependenceResponseAxisState.NOT_VIABLE} for value in axes
    ):
        maximum_claim = "At least one non-substitutable target axis opposed the forecast."
    elif by_axis[SelectiveDependenceResponseAxisName.PREDICTIVE_DISTINCTIVENESS].state is not (
        SelectiveDependenceResponseAxisState.DISTINGUISHED
    ):
        maximum_claim = "The target pattern was not distinguished from every frozen comparator."
    else:
        maximum_claim = "The frozen selective-lawhood grammar was supported in this simulator task."
    return SelectiveDependenceResponseTargetAdjudication(
        adjudication_id=f"adjudication.{target_id}",
        target_id=target_id,
        forecast=forecast,
        evaluation_unit_ids_sha256=evaluation_unit_ids_sha256,
        issued_unit_count=issued_unit_count,
        source_valid_unit_count=source_valid_unit_count,
        evaluable_unit_count=evaluable_unit_count,
        stopped_unit_count=stopped_unit_count,
        missing_unit_count=missing_unit_count,
        axes=tuple(sorted(axes, key=lambda value: value.adjudication_id)),
        exact_counterexample_ids=tuple(sorted(exact_counterexample_ids)),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_claim=maximum_claim,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def compact_handoff(
    adjudication: SelectiveDependenceResponseTargetAdjudication,
    *,
    source_family_id: str,
    solver_family_id: str,
    forecast_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...],
    observed_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...],
) -> SelectiveDependenceResponseTargetHandoff:
    by_axis = {value.axis: value for value in adjudication.axes}
    components = {
        "finite-law": SelectiveDependenceResponseAxisName.LOCAL_RESPONSE_LAW,
        "selective-exchange": SelectiveDependenceResponseAxisName.SELECTIVE_DEPENDENCE,
        "support-boundary": SelectiveDependenceResponseAxisName.SUPPORT_BOUNDARY,
        "action-fibre": SelectiveDependenceResponseAxisName.RECEIVER_ADMISSION,
    }
    if tuple(sorted(components)) != SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS:
        raise AssertionError("component map differs from the frozen relation")
    component_states = tuple(
        sorted(
            (
                SelectiveDependenceResponseAxisAdjudication(
                    adjudication_id=f"component.{adjudication.target_id}.{component_id}",
                    axis=by_axis[axis_name].axis,
                    state=by_axis[axis_name].state,
                    decisive_case_ids=by_axis[axis_name].decisive_case_ids,
                    reason=by_axis[axis_name].reason,
                )
                for component_id, axis_name in components.items()
            ),
            key=lambda value: value.adjudication_id,
        )
    )
    construct_valid = by_axis[SelectiveDependenceResponseAxisName.CONSTRUCT].state is SelectiveDependenceResponseAxisState.VALID
    measurement_state = by_axis[SelectiveDependenceResponseAxisName.MEASUREMENT_ACTION_CHAIN].state
    target_eligible = construct_valid and measurement_state is SelectiveDependenceResponseAxisState.QUALIFIED
    ineligibility_reasons = []
    if not construct_valid:
        ineligibility_reasons.append("construct-invalid")
    if measurement_state is not SelectiveDependenceResponseAxisState.QUALIFIED:
        ineligibility_reasons.append("measurement-action-chain-unqualified")
    return SelectiveDependenceResponseTargetHandoff(
        handoff_id=f"handoff.{adjudication.target_id}",
        target_id=adjudication.target_id,
        source_family_id=source_family_id,
        solver_family_id=solver_family_id,
        primary_relation_id=SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID,
        component_states=component_states,
        measurement_action_state=measurement_state,
        hold_state=by_axis[SelectiveDependenceResponseAxisName.HOLD_VIABILITY].state,
        distinctiveness_state=by_axis[SelectiveDependenceResponseAxisName.PREDICTIVE_DISTINCTIVENESS].state,
        forecast_policy_dispositions=tuple(
            sorted(set(forecast_policy_dispositions), key=lambda value: value.value)
        ),
        observed_policy_dispositions=tuple(
            sorted(set(observed_policy_dispositions), key=lambda value: value.value)
        ),
        construct_valid=construct_valid,
        target_eligible=target_eligible,
        ineligibility_reason_codes=tuple(sorted(ineligibility_reasons)),
        exact_counterexample_ids=adjudication.exact_counterexample_ids,
        native_numeric_value_count=0,
        target_adjudication=ObjectIdentity.from_record(adjudication.adjudication_id, adjudication),
        maximum_claim=adjudication.maximum_claim,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = ["adjudicate_target", "axis", "compact_handoff"]
