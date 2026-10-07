"""Common target construct validation relation and typed-stop contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID = 'margin-structural-recurrence-forecast-categorical-action-fibre-restriction'
TARGET_CONSTRUCT_VALIDATION_RELATION_STATEMENT = (
    "For a fixed prepared denominator D, retained causal history H, native "
    'finite action A, receiver gauge R, and horizon tau, the frozen margin structural recurrence forecast '
    "categorical restriction relation recurs only when the target-native "
    "compatibility map preserves the complete physical or preparation unit, "
    "causal cutoff, action identity, receiver direction, units, frame, and "
    "support semantics."
)
TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS = ("A", "D", "H", "R", "tau")


class TargetConstructValidationStopCode(StrEnum):
    """Scientific stops that must not be converted into operational failure."""

    CONSTRUCT_REVIEW_INDEPENDENCE_UNRESOLVED = "CONSTRUCT_REVIEW_INDEPENDENCE_UNRESOLVED"
    CROSS_TARGET_RELATION_NOT_PREDECLARED = "CROSS_TARGET_RELATION_NOT_PREDECLARED"
    DENOMINATOR_UNRESOLVED = "DENOMINATOR_UNRESOLVED"
    EVALUATION_REVEAL_BARRIER_VIOLATED = "EVALUATION_REVEAL_BARRIER_VIOLATED"
    MAPPING_AMBIGUOUS = "MAPPING_AMBIGUOUS"
    MAPPING_UNRESOLVED = "MAPPING_UNRESOLVED"
    NO_ELIGIBLE_SOURCE = "NO_ELIGIBLE_SOURCE"
    NO_VALID_FORECAST = "NO_VALID_FORECAST"
    PARENT_RECEIPT_SUBSTITUTION = "PARENT_RECEIPT_SUBSTITUTION"
    TARGET_SPECIFIC_RESCUE_FORBIDDEN = "TARGET_SPECIFIC_RESCUE_FORBIDDEN"
    UNEVALUABLE_COMPLETE_UNIT_INFERENCE = "UNEVALUABLE_COMPLETE_UNIT_INFERENCE"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationStructuralRelationCodebook(CanonicalRecord):
    """The sole claim-bearing relation shared by both construct-validation targets."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-structural-relation-codebook'

    codebook_id: str
    primary_relation_id: str
    relation_statement: str
    required_role_ids: tuple[str, ...]
    claim_bearing_relation_ids: tuple[str, ...]
    diagnostic_relation_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.codebook_id, field_name="codebook_id")
        validate_stable_id(self.primary_relation_id, field_name="primary_relation_id")
        validate_nonempty(self.relation_statement, field_name="relation_statement")
        require_sorted_unique_strings(
            self.required_role_ids,
            field_name="required_role_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.claim_bearing_relation_ids,
            field_name="claim_bearing_relation_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.diagnostic_relation_ids,
            field_name="diagnostic_relation_ids",
        )
        if (
            self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID
            or self.relation_statement != TARGET_CONSTRUCT_VALIDATION_RELATION_STATEMENT
            or self.required_role_ids != TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS
            or self.claim_bearing_relation_ids != (TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,)
            or self.primary_relation_id in self.diagnostic_relation_ids
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("target construct validation must freeze exactly one outcome-blind primary relation")


def primary_relation_codebook() -> TargetConstructValidationStructuralRelationCodebook:
    return TargetConstructValidationStructuralRelationCodebook(
        codebook_id="target-construct-validation.structural-relation-codebook",
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        relation_statement=TARGET_CONSTRUCT_VALIDATION_RELATION_STATEMENT,
        required_role_ids=TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS,
        claim_bearing_relation_ids=(TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,),
        diagnostic_relation_ids=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetRelationBinding(CanonicalRecord):
    """Pre-development binding of one target to the common primary relation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-relation-binding'

    binding_id: str
    target_id: str
    relation_codebook: ObjectIdentity
    primary_relation_id: str
    claim_bearing_cell_ids: tuple[str, ...]
    diagnostic_relation_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.target_id, field_name="target_id")
        validate_stable_id(self.primary_relation_id, field_name="primary_relation_id")
        require_sorted_unique_strings(
            self.claim_bearing_cell_ids,
            field_name="claim_bearing_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.diagnostic_relation_ids,
            field_name="diagnostic_relation_ids",
        )
        if (
            self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID
            or self.primary_relation_id in self.diagnostic_relation_ids
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("target relation binding differs from the construct-validation primary relation")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationMethodQuestionFreeze(CanonicalRecord):
    """C01 scientific question/axis/falsifier freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-method-question-freeze'

    freeze_id: str
    donor_binding: ObjectIdentity
    relation_codebook: ObjectIdentity
    primary_question: str
    primary_relation_id: str
    primary_estimand_ids: tuple[str, ...]
    independent_axis_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    noncompensating_gate_ids: tuple[str, ...]
    excluded_scope_ids: tuple[str, ...]
    target_specific_threshold_ids: tuple[str, ...]
    maximum_claim: str
    frozen: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(self.primary_relation_id, field_name="primary_relation_id")
        validate_nonempty(self.primary_question, field_name="primary_question")
        validate_nonempty(self.maximum_claim, field_name="maximum_claim")
        for name in (
            "primary_estimand_ids",
            "independent_axis_ids",
            "decisive_falsifier_ids",
            "noncompensating_gate_ids",
            "excluded_scope_ids",
            "target_specific_threshold_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name == "target_specific_threshold_ids",
            )
        if self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID:
            raise ValueError("method question freeze names a nonprimary relation")
        if self.target_specific_threshold_ids:
            raise ValueError("C01 cannot freeze target-specific rescue thresholds")
        if not self.frozen:
            raise ValueError("method question record must be immutable before nomination")
        if self.protected_outcome_access_count != 0:
            raise ValueError("method question freeze cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("method question freeze must remain outcome-blind")


def method_question_freeze(
    donor_binding: CanonicalRecord,
    relation_codebook: TargetConstructValidationStructuralRelationCodebook,
) -> TargetConstructValidationMethodQuestionFreeze:
    return TargetConstructValidationMethodQuestionFreeze(
        freeze_id="target-construct-validation.method-question-freeze",
        donor_binding=ObjectIdentity.from_record(
            'target-construct-validation.frozen-margin-structural-recurrence-forecast-donor',
            donor_binding,
        ),
        relation_codebook=ObjectIdentity.from_record(
            relation_codebook.codebook_id,
            relation_codebook,
        ),
        primary_question=(
            'Does the frozen margin structural recurrence forecast categorical action-fibre restriction recur '
            "across two independently maintained simulator substrates under "
            "target-native, construct-valid finite mappings?"
        ),
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        primary_estimand_ids=(
            "complete-unit-forecast-coverage",
            "complete-unit-prediction-error",
            "target-local-predictive-restrictiveness",
            "two-target-primary-relation-recurrence",
        ),
        independent_axis_ids=(
            "construct-validity",
            "predictive-restrictiveness",
            "target-prediction",
            "topology-vs-metric",
        ),
        decisive_falsifier_ids=(
            "construct-valid-exact-counterexample",
            "direct-action-or-receiver-echo",
            "incomplete-unit-inference",
            "safer-or-more-exact-simple-comparator",
        ),
        noncompensating_gate_ids=(
            "complete-unit-power",
            "construct-review-independence",
            "mapping-semantic-preservation",
            "sealed-evaluation-reveal",
        ),
        excluded_scope_ids=(
            "cpaim",
            "nature-publication",
            "physical-claim",
            "prospective-controller-validation",
            "scale-covariance",
        ),
        target_specific_threshold_ids=(),
        maximum_claim=(
            "The named structural relation recurred across two independently "
            "maintained software substrates under construct-valid mappings."
        ),
        frozen=True,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
