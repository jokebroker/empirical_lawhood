"""Nonactuating receiver-information and control-quotient planning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.receiver_geometry_control import (
    BoundedControlEquivalenceAssessment,
    CandidateDecisionDisposition,
    DecisionEquivalenceClass,
    ReceiverFiberAssessment,
    ReceiverObservationExperiment,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ReceiverQuotientControlPlan(CanonicalRecord):
    "Frozen compiler input for one bounded candidate/view roster.\n\n    The plan carries local law receiver/control evidence and a predeclared safe\n    fallback choice.  The compiler, rather than this record, constructs the admission\n    ambiguity certificate after replaying admission and reachability.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receiver-quotient-control-plan'

    plan_id: str
    observation_experiment: ReceiverObservationExperiment
    fiber_assessment: ReceiverFiberAssessment
    equivalence_assessment: BoundedControlEquivalenceAssessment
    observer_binding_id: str
    fallback_word: OccurrenceActionWord | None
    termination_contract: ObjectIdentity | None
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(
            self.observer_binding_id,
            field_name="observer_binding_id",
        )
        if self.fiber_assessment.observation_experiment != self.observation_experiment:
            raise ValueError("receiver quotient plan changes its observation experiment")
        if self.equivalence_assessment.fiber_assessment != self.fiber_assessment:
            raise ValueError("receiver quotient plan changes its exact fiber")

        decision_class = self.equivalence_assessment.decision_class
        if decision_class is DecisionEquivalenceClass.SHARED_ACTION:
            if self.fallback_word is not None or self.termination_contract is not None:
                raise ValueError("shared-action plan cannot replace action with abstention")
        elif decision_class is DecisionEquivalenceClass.SHARED_QUALIFIED_HOLD:
            fallbacks = {
                (value.fallback_word.word_id, value.fallback_word.fingerprint())
                for value in self.equivalence_assessment.decisions
                if value.fallback_word is not None
            }
            if (
                self.fallback_word is None
                or (
                    self.fallback_word.word_id,
                    self.fallback_word.fingerprint(),
                )
                not in fallbacks
                or self.termination_contract is not None
            ):
                raise ValueError("shared-hold plan lacks its exact qualified fallback")
        elif decision_class is DecisionEquivalenceClass.SHARED_TERMINATION:
            contracts = {
                (
                    value.termination_contract.object_id,
                    value.termination_contract.object_fingerprint,
                )
                for value in self.equivalence_assessment.decisions
                if value.termination_contract is not None
            }
            if (
                self.fallback_word is not None
                or self.termination_contract is None
                or (
                    self.termination_contract.object_id,
                    self.termination_contract.object_fingerprint,
                )
                not in contracts
            ):
                raise ValueError("shared-termination plan lacks its exact contract")
        elif (self.fallback_word is None) == (self.termination_contract is None):
            raise ValueError(
                "decision-distinct or unevaluable plan requires exactly one safe response"
            )

        if self.fallback_word is not None:
            if any(
                value.disposition is CandidateDecisionDisposition.ACTION_SET_AVAILABLE
                and value.fallback_word == self.fallback_word
                for value in self.equivalence_assessment.decisions
            ):
                raise ValueError("fallback word is mislabeled as a candidate action")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("receiver quotient planning cannot promote its local law evidence")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("receiver quotient plan must remain outcome-blind")
        if (
            not self.visibility_ceiling.is_promotable
            or not self.visibility_ceiling.is_at_least_as_restrictive_as(
                self.equivalence_assessment.visibility_ceiling
            )
        ):
            raise ValueError("receiver quotient plan lowers inherited visibility")


__all__ = ["ReceiverQuotientControlPlan"]
