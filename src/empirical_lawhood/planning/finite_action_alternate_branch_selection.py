"Typed alternate-branch selection bridge for finite-action owners.\n\nThis record carries the sealed tokamak-control alternate-branch decision and\nits controlling input-output operator feasibility obstruction into the\nfinite-action occurrence and recurrence owners. It cannot select a branch,\nissue a parent, grant authority, or read measurement through local law outcomes.\n"

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)


FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID = 'TOKAMAK-FINITE-ACTION-RECURRENCE'
FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_REASON_PRECEDENCE = (
    "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE",
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE",
    "CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE",
    "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE",
)


@dataclass(frozen=True, slots=True)
class FiniteActionAlternateBranchSelectionBridge(CanonicalRecord):
    "Outcome-visible branch-selection metadata projected without measurement through local law outcome access."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-alternate-branch-selection-bridge'

    receipt_id: str
    source_selection: ObjectIdentity
    g2_operator_feasibility: ObjectIdentity
    selected_branch_id: str
    controlling_reason_code: str
    hfr_contract_evidence: tuple[ObjectIdentity, ...]
    parent_issue_started: bool
    measurement_through_law_qualification_outcome_used: bool
    issues_parent: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.hfr_contract_evidence,
            attribute="object_id",
            field_name="hfr_contract_evidence",
        )
        if (
            self.selected_branch_id != FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID
            or self.controlling_reason_code not in FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_REASON_PRECEDENCE
            or not self.hfr_contract_evidence
        ):
            raise ValueError("finite-action bridge requires one closed finite-action recurrence selection")
        if self.parent_issue_started or self.measurement_through_law_qualification_outcome_used:
            raise ValueError("finite-action recurrence selection must precede parent outcomes")
        if self.issues_parent or self.grants_authority:
            raise ValueError("finite-action recurrence bridge cannot issue or authorize")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("finite-action recurrence bridge changes its branch-selection metadata custody")


__all__ = [
    "FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID",
    "FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_REASON_PRECEDENCE",
    'FiniteActionAlternateBranchSelectionBridge',
]
