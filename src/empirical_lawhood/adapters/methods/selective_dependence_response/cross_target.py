"""Non-pooling selective dependence response cross-target adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID, SelectiveDependenceResponseAxisState, SelectiveDependenceResponseCrossTargetAdjudication, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseRelationState, SelectiveDependenceResponseTargetHandoff


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetEligibility(CanonicalRecord):
    """Compact exact-two-target gate before relation adjudication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cross-target-eligibility'

    eligibility_id: str
    target_handoffs: tuple[ObjectIdentity, ...]
    target_ids: tuple[str, ...]
    eligible_target_ids: tuple[str, ...]
    distinct_source_family_count: int
    distinct_solver_family_count: int
    native_numeric_value_count: int
    study_forecast_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    study_observed_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    all_three_policy_dispositions_present: bool
    eligible: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.eligibility_id, field_name="eligibility_id")
        require_sorted_unique_ids(
            self.target_handoffs,
            attribute="object_id",
            field_name="target_handoffs",
        )
        for name in ("target_ids", "eligible_target_ids", "reason_codes"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if len(self.target_handoffs) != 2 or len(self.target_ids) != 2:
            raise ValueError("cross-target eligibility requires two exact handoffs")
        if self.distinct_source_family_count != 2 or self.distinct_solver_family_count != 2:
            raise ValueError("cross-target eligibility requires distinct sources and solvers")
        if self.native_numeric_value_count:
            raise ValueError("cross-target eligibility cannot carry native numeric values")
        expected_dispositions = (
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
            SelectiveDependenceResponseDisposition.HOLD_ONLY,
            SelectiveDependenceResponseDisposition.NONATTEMPT,
        )
        if (
            tuple(
                sorted(
                    set(self.study_forecast_policy_dispositions),
                    key=lambda value: value.value,
                )
            )
            != self.study_forecast_policy_dispositions
        ):
            raise ValueError("programme forecast policy dispositions must be sorted and unique")
        if (
            tuple(
                sorted(
                    set(self.study_observed_policy_dispositions),
                    key=lambda value: value.value,
                )
            )
            != self.study_observed_policy_dispositions
        ):
            raise ValueError("programme observed policy dispositions must be sorted and unique")
        expected_three = set(expected_dispositions).issubset(
            self.study_forecast_policy_dispositions
        )
        if self.all_three_policy_dispositions_present != expected_three:
            raise ValueError("programme policy-diversity flag is not data-derived")
        expected = len(self.eligible_target_ids) == 2 and expected_three
        if self.eligible != expected:
            raise ValueError("cross-target eligibility flag is not target-derived")
        if self.eligible == bool(self.reason_codes):
            raise ValueError("cross-target eligibility reasons are inconsistent")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("cross-target eligibility must remain outcome visible")


def qualify_cross_target(
    handoffs: tuple[SelectiveDependenceResponseTargetHandoff, SelectiveDependenceResponseTargetHandoff],
) -> SelectiveDependenceResponseCrossTargetEligibility:
    ordered = tuple(sorted(handoffs, key=lambda value: value.target_id))
    if len({value.target_id for value in ordered}) != 2:
        raise ValueError("cross-target handoff identities repeat")
    if len({value.source_family_id for value in ordered}) != 2:
        raise ValueError("cross-target sources are not distinct")
    if len({value.solver_family_id for value in ordered}) != 2:
        raise ValueError("cross-target solvers are not distinct")
    if any(value.native_numeric_value_count for value in ordered):
        raise ValueError("cross-target input contains native numeric values")
    eligible_ids = tuple(value.target_id for value in ordered if value.target_eligible)
    forecast_dispositions = tuple(
        sorted(
            {item for value in ordered for item in value.forecast_policy_dispositions},
            key=lambda value: value.value,
        )
    )
    observed_dispositions = tuple(
        sorted(
            {item for value in ordered for item in value.observed_policy_dispositions},
            key=lambda value: value.value,
        )
    )
    expected_dispositions = {
        SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
        SelectiveDependenceResponseDisposition.HOLD_ONLY,
        SelectiveDependenceResponseDisposition.NONATTEMPT,
    }
    reasons_list = [
        f"{value.target_id}.{reason}"
        for value in ordered
        for reason in value.ineligibility_reason_codes
    ]
    if not expected_dispositions.issubset(forecast_dispositions):
        reasons_list.append("programme-policy-disposition-roster-incomplete")
    reasons = tuple(sorted(reasons_list))
    return SelectiveDependenceResponseCrossTargetEligibility(
        eligibility_id="selective-dependence-response.cross-target-eligibility",
        target_handoffs=tuple(
            sorted(
                (ObjectIdentity.from_record(value.handoff_id, value) for value in ordered),
                key=lambda value: value.object_id,
            )
        ),
        target_ids=tuple(value.target_id for value in ordered),
        eligible_target_ids=eligible_ids,
        distinct_source_family_count=2,
        distinct_solver_family_count=2,
        native_numeric_value_count=0,
        study_forecast_policy_dispositions=forecast_dispositions,
        study_observed_policy_dispositions=observed_dispositions,
        all_three_policy_dispositions_present=expected_dispositions.issubset(forecast_dispositions),
        eligible=len(eligible_ids) == 2 and expected_dispositions.issubset(forecast_dispositions),
        reason_codes=reasons,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def adjudicate_cross_target(
    handoffs: tuple[SelectiveDependenceResponseTargetHandoff, SelectiveDependenceResponseTargetHandoff],
) -> SelectiveDependenceResponseCrossTargetAdjudication:
    ordered = tuple(sorted(handoffs, key=lambda value: value.target_id))
    eligibility = qualify_cross_target(handoffs)

    exact_counterexamples = tuple(
        sorted(value.target_id for value in ordered if value.exact_counterexample_ids)
    )
    component_unevaluable = tuple(
        sorted(
            value.target_id
            for value in ordered
            if value.hold_state is SelectiveDependenceResponseAxisState.UNEVALUABLE
            or value.distinctiveness_state is SelectiveDependenceResponseAxisState.UNEVALUABLE
            or any(
                component.state is SelectiveDependenceResponseAxisState.UNEVALUABLE
                for component in value.component_states
            )
        )
    )
    if not eligibility.eligible:
        state = SelectiveDependenceResponseRelationState.UNEVALUABLE
        decisive = tuple(sorted(set(eligibility.target_ids) - set(eligibility.eligible_target_ids)))
        claim = "The two-target relation was unevaluable because target eligibility did not close."
    elif component_unevaluable:
        state = SelectiveDependenceResponseRelationState.UNEVALUABLE
        decisive = component_unevaluable
        claim = "The primary relation was unevaluable in at least one target."
    elif exact_counterexamples:
        state = SelectiveDependenceResponseRelationState.OPPOSED
        decisive = exact_counterexamples
        claim = "At least one construct-valid target opposed a primary relation component."
    elif any(
        value.distinctiveness_state is not SelectiveDependenceResponseAxisState.DISTINGUISHED for value in ordered
    ):
        state = SelectiveDependenceResponseRelationState.NOT_DISTINGUISHED
        decisive = tuple(
            sorted(
                value.target_id
                for value in ordered
                if value.distinctiveness_state is not SelectiveDependenceResponseAxisState.DISTINGUISHED
            )
        )
        claim = "The two simulator tasks did not distinguish selective-response from the frozen comparators."
    else:
        states = [component.state for value in ordered for component in value.component_states]
        if all(value is SelectiveDependenceResponseAxisState.SUPPORTED for value in states):
            state = SelectiveDependenceResponseRelationState.SUPPORTED
            decisive = tuple(value.target_id for value in ordered)
            claim = (
                "The predeclared selective-context and action-fibre grammar recurred "
                "across two outcome-visible-selected independently maintained simulators."
            )
        else:
            state = SelectiveDependenceResponseRelationState.MIXED
            decisive = tuple(value.target_id for value in ordered)
            claim = "The two target-local component patterns were mixed."

    return SelectiveDependenceResponseCrossTargetAdjudication(
        adjudication_id="selective-dependence-response.cross-target-adjudication",
        primary_relation_id=SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID,
        eligibility=ObjectIdentity.from_record(eligibility.eligibility_id, eligibility),
        target_handoffs=tuple(
            sorted(
                (ObjectIdentity.from_record(value.handoff_id, value) for value in ordered),
                key=lambda value: value.object_id,
            )
        ),
        target_ids=tuple(value.target_id for value in ordered),
        relation_state=state,
        decisive_target_ids=decisive,
        pooled_native_numeric_value_count=0,
        maximum_claim=claim,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = [
    'SelectiveDependenceResponseCrossTargetEligibility',
    "adjudicate_cross_target",
    "qualify_cross_target",
]
