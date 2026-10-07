"""Construct-validity assessment and independent-review separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)

from .native_dossier import TargetConstructValidationIndependenceDossier, TargetConstructValidationTargetNativeDossier


@dataclass(frozen=True, slots=True)
class TargetConstructValidationConstructThreat(CanonicalRecord):
    """One pre-outcome threat and its non-rescuing disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-construct-threat'

    threat_id: str
    target_id: str
    threat_kind: str
    description: str
    decisive_case_ids: tuple[str, ...]
    blocks_claim_if_observed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("threat_id", "target_id", "threat_kind"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.description, field_name="description")
        require_sorted_unique_strings(
            self.decisive_case_ids,
            field_name="decisive_case_ids",
            allow_empty=False,
        )
        if not self.blocks_claim_if_observed:
            raise ValueError("a construct threat cannot be made compensating")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct threats must be authored outcome-blindly")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationConstructValidityAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-construct-validity-assessment'

    assessment_id: str
    target_id: str
    native_dossier: ObjectIdentity
    independence_dossier: ObjectIdentity
    passed: bool
    reason_codes: tuple[str, ...]
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.target_id, field_name="target_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("construct pass status and reason codes are inconsistent")
        if self.protected_outcome_access_count != 0:
            raise ValueError("construct assessment cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct assessment must remain outcome-blind")


def assess_construct_validity(
    native: TargetConstructValidationTargetNativeDossier,
    independence: TargetConstructValidationIndependenceDossier,
) -> TargetConstructValidationConstructValidityAssessment:
    """Assess independence without importing later mapping or outcome records."""

    if native.target_id != independence.target_id:
        raise ValueError("native and independence dossiers name different targets")
    if native.source_candidate != independence.source_candidate:
        raise ValueError("native and independence dossiers bind different sources")
    reasons: list[str] = []
    checks = (
        (
            independence.generator_not_authored_for_construct_validation,
            "GENERATOR_AUTHORED_FOR_INDEPENDENT_RECURRENCE_PROGRAMME",
        ),
        (independence.generator_predates_construct_validation, "GENERATOR_CHRONOLOGY_UNRESOLVED"),
        (
            independence.domain_independent_of_other_target,
            "TARGET_DOMAIN_NOT_INDEPENDENT",
        ),
        (
            independence.solver_family_independent_of_other_target,
            "SOLVER_FAMILY_NOT_INDEPENDENT",
        ),
        (
            independence.generator_family_independent_of_other_target,
            "GENERATOR_FAMILY_NOT_INDEPENDENT",
        ),
        (not independence.direct_echo_of_structural_recurrence_fixture, 'DIRECT_STRUCTURAL_RECURRENCE_FIXTURE_ECHO'),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return TargetConstructValidationConstructValidityAssessment(
        assessment_id=f"{native.target_id}.construct-validity",
        target_id=native.target_id,
        native_dossier=ObjectIdentity.from_record(native.dossier_id, native),
        independence_dossier=ObjectIdentity.from_record(
            independence.dossier_id,
            independence,
        ),
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationConstructReviewAttestation(CanonicalRecord):
    """Identity/exposure record enforcing an independent construct reviewer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-construct-review-attestation'

    attestation_id: str
    target_id: str
    assessment: ObjectIdentity
    dossier_author_id: str
    mapping_author_id: str
    projector_owner_id: str
    evaluator_owner_id: str
    reviewer_id: str
    reviewer_role: str
    reviewed_case_ids: tuple[str, ...]
    exposure_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    review_passed: bool
    review_reason: str
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "attestation_id",
            "target_id",
            "dossier_author_id",
            "mapping_author_id",
            "projector_owner_id",
            "evaluator_owner_id",
            "reviewer_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.reviewer_role, field_name="reviewer_role")
        validate_nonempty(self.review_reason, field_name="review_reason")
        require_sorted_unique_strings(
            self.reviewed_case_ids,
            field_name="reviewed_case_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.exposure_ids, field_name="exposure_ids")
        require_sorted_unique_strings(self.conflict_ids, field_name="conflict_ids")
        prohibited = {
            self.dossier_author_id,
            self.mapping_author_id,
            self.projector_owner_id,
            self.evaluator_owner_id,
        }
        if self.reviewer_id in prohibited:
            raise ValueError("construct reviewer is not independent of authoring/evaluation")
        if self.review_passed and self.conflict_ids:
            raise ValueError("a conflicted construct review cannot pass")
        if self.protected_outcome_access_count != 0:
            raise ValueError("construct review cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct review must remain outcome-blind")
