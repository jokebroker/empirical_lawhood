"""Evidence rungs, visibility ceilings and claim contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


class EvidenceRung(StrEnum):
    MEASUREMENT = "MEASUREMENT"
    ORDER_RELATION = "ORDER_RELATION"
    RESPONSE = "RESPONSE"
    LOCAL_LAW = "LOCAL_LAW"
    ADMISSION = "ADMISSION"
    CONTROLLER_USE = "CONTROLLER_USE"


_RUNG_RANK = {rung: index for index, rung in enumerate(EvidenceRung)}


class EvidenceCeiling(StrEnum):
    NON_PROMOTABLE = "NON_PROMOTABLE"
    MEASUREMENT = "MEASUREMENT"
    ORDER_RELATION = "ORDER_RELATION"
    RESPONSE = "RESPONSE"
    LOCAL_LAW = "LOCAL_LAW"
    ADMISSION = "ADMISSION"
    CONTROLLER_USE = "CONTROLLER_USE"

    def allows(self, rung: EvidenceRung) -> bool:
        if self is EvidenceCeiling.NON_PROMOTABLE:
            return False
        return _RUNG_RANK[rung] <= _CEILING_RANK[self]

    @classmethod
    def lowest(cls, *ceilings: EvidenceCeiling) -> EvidenceCeiling:
        if not ceilings:
            raise ValueError("at least one evidence ceiling is required")
        return min(ceilings, key=lambda ceiling: _CEILING_RANK[ceiling])


_CEILING_RANK = {ceiling: index - 1 for index, ceiling in enumerate(EvidenceCeiling)}


class OutcomeAccess(StrEnum):
    OUTCOME_BLIND = "outcome-blind"
    DEVELOPMENT_VISIBLE = "development-visible"
    EVALUATION_SEALED = "evaluation-sealed"
    EVALUATOR_REVEAL = "evaluator-reveal"
    EVALUATION_REVEALED = "evaluation-revealed"
    PRIVILEGED_TRUTH = "privileged-truth"


class VisibilityCeiling(StrEnum):
    PROSPECTIVE = "PROSPECTIVE"
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    OUTCOME_VISIBLE = "OUTCOME_VISIBLE"
    PRIVILEGED_TRUTH = "PRIVILEGED_TRUTH"
    UNBOUND_HISTORICAL = "UNBOUND_HISTORICAL"

    @property
    def is_promotable(self) -> bool:
        return self in {
            VisibilityCeiling.PROSPECTIVE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
        }

    @classmethod
    def most_restrictive(cls, *ceilings: VisibilityCeiling) -> VisibilityCeiling:
        if not ceilings:
            raise ValueError("at least one visibility ceiling is required")
        return max(ceilings, key=lambda ceiling: _VISIBILITY_RANK[ceiling])

    def is_at_least_as_restrictive_as(self, other: VisibilityCeiling) -> bool:
        return _VISIBILITY_RANK[self] >= _VISIBILITY_RANK[other]


_VISIBILITY_RANK = {
    VisibilityCeiling.PROSPECTIVE: 0,
    VisibilityCeiling.DEVELOPMENT_ONLY: 1,
    VisibilityCeiling.OUTCOME_VISIBLE: 2,
    VisibilityCeiling.PRIVILEGED_TRUTH: 3,
    VisibilityCeiling.UNBOUND_HISTORICAL: 4,
}

_ACCESS_VISIBILITY = {
    OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
    OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
    OutcomeAccess.EVALUATION_SEALED: VisibilityCeiling.PROSPECTIVE,
    OutcomeAccess.EVALUATOR_REVEAL: VisibilityCeiling.PROSPECTIVE,
    OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
    OutcomeAccess.PRIVILEGED_TRUTH: VisibilityCeiling.PRIVILEGED_TRUTH,
}


def inherited_visibility(
    parent_ceilings: tuple[VisibilityCeiling, ...], outcome_access: OutcomeAccess
) -> VisibilityCeiling:
    """Return the minimum visibility ceiling a derived object must retain."""

    return VisibilityCeiling.most_restrictive(_ACCESS_VISIBILITY[outcome_access], *parent_ceilings)


@dataclass(frozen=True, slots=True)
class ClaimSpec(CanonicalRecord):
    """One proposition and the evidence scope it is permitted to answer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/claim-spec'

    claim_id: str
    world_id: str
    relation_id: str
    proposition: str
    estimand: str
    physical_independent_unit_id: str
    requested_rung: EvidenceRung | None
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    promotion_rule: str
    assumption_ids: tuple[str, ...]
    derivation_parent_ids: tuple[str, ...] = ()
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...] = ()
    numerical_view_ids: tuple[str, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("claim_id", self.claim_id),
            ("world_id", self.world_id),
            ("relation_id", self.relation_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.proposition, field_name="proposition")
        validate_nonempty(self.estimand, field_name="estimand")
        validate_nonempty(self.promotion_rule, field_name="promotion_rule")
        require_sorted_unique_strings(
            self.assumption_ids, field_name="assumption_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.derivation_parent_ids, field_name="derivation_parent_ids"
        )
        require_sorted_unique_strings(self.numerical_view_ids, field_name="numerical_view_ids")
        if len(self.derivation_parent_ids) != len(self.parent_visibility_ceilings):
            raise ValueError("each derivation parent must have one visibility ceiling")
        required_visibility = inherited_visibility(
            self.parent_visibility_ceilings, self.outcome_access
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("visibility ceiling cannot be lowered by derivation")
        if not self.visibility_ceiling.is_promotable:
            if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
                raise ValueError("outcome-visible, privileged or unbound claims are non-promotable")
            if self.requested_rung is not None:
                raise ValueError("a non-promotable claim cannot request an evidence stage")
        elif self.requested_rung is not None and not self.evidence_ceiling.allows(
            self.requested_rung
        ):
            raise ValueError("requested evidence rung exceeds the claim ceiling")
        require_extensions(self.extensions)
