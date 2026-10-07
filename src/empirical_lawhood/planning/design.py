"""Advisory fresh-evidence proposals and explicit selection decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.time import InformationCutoff


@dataclass(frozen=True, slots=True)
class ExperimentProposal(CanonicalRecord):
    """Advisory design; unlike AnalysisProposal it requests fresh evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-proposal'

    proposal_id: str
    nomination: ObjectIdentity
    candidate_experiment: ExperimentSpec
    alternative_design_ids: tuple[str, ...]
    expected_discrimination: str
    risk_codes: tuple[str, ...]
    budget: ResourceBudget
    decision_cutoff: InformationCutoff
    proposed_by: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        validate_stable_id(self.proposed_by, field_name="proposed_by")
        require_sorted_unique_strings(
            self.alternative_design_ids, field_name="alternative_design_ids"
        )
        validate_nonempty(self.expected_discrimination, field_name="expected_discrimination")
        require_sorted_unique_strings(self.risk_codes, field_name="risk_codes", allow_empty=False)
        if self.candidate_experiment.authorization_record_id is not None:
            raise ValueError("an experiment proposal cannot contain an authorization")
        if self.candidate_experiment.readiness is not ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("candidate experiment must await separate authority")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("experiment-proposal visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("a nomination-derived proposal remains non-promotable")
        if self.candidate_experiment.design_visibility_ceiling is not self.visibility_ceiling:
            raise ValueError("candidate design visibility differs from its proposal")
        require_extensions(self.extensions)


class DecisionDisposition(StrEnum):
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class DecisionRecord(CanonicalRecord):
    """Explicit outcome-blind disposition of an experiment proposal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-record'

    decision_id: str
    proposal: ObjectIdentity
    disposition: DecisionDisposition
    selected_experiment_id: str | None
    reason_codes: tuple[str, ...]
    decision_maker_id: str
    authority_policy_id: str
    decision_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("decision_id", self.decision_id),
            ("decision_maker_id", self.decision_maker_id),
            ("authority_policy_id", self.authority_policy_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if self.disposition is DecisionDisposition.SELECTED:
            if self.selected_experiment_id is None:
                raise ValueError("selected decision must name an experiment")
        elif self.selected_experiment_id is not None:
            raise ValueError("only a selected decision may name an experiment")
        if self.selected_experiment_id is not None:
            validate_stable_id(self.selected_experiment_id, field_name="selected_experiment_id")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("experiment selection must be outcome-blind")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("decision visibility cannot be lowered")
        require_extensions(self.extensions)
