"""Finite outcome-blind map roster and development-only denominator order."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Iterable

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.adapters.methods.scientific_description_code import scientific_description_octets

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .contracts import TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS


class TargetConstructValidationMappingConstructor(StrEnum):
    DIRECT_NATIVE_ROLE_BINDING = "direct-native-role-binding"
    DECLARED_TARGET_NATIVE_AGGREGATE = "declared-target-native-aggregate"
    DECLARED_TARGET_NATIVE_RESTRICTION = "declared-target-native-restriction"
    DECLARED_RECEIVER_GAUGE_PROJECTION = "declared-receiver-gauge-projection"
    DECLARED_HORIZON_SELECTION = "declared-horizon-selection"


class TargetConstructValidationDenominatorAlternativeKind(StrEnum):
    PROPOSED = "proposed"
    SPLIT = "split"
    MERGE = "merge"
    OMISSION = "omission"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationSemanticLoss(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-semantic-loss'

    loss_id: str
    target_id: str
    role_id: str
    description: str
    blocks_selection: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("loss_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.role_id not in set(TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS):
            raise ValueError("semantic loss names an unknown target construction validation role")
        validate_nonempty(self.description, field_name="description")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("semantic loss must be declared before outcomes")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationRoleBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-role-binding'

    binding_id: str
    structural_recurrence_role_id: str
    target_native_role_ids: tuple[str, ...]
    constructor: TargetConstructValidationMappingConstructor
    native_units_preserved: bool
    native_frame_preserved: bool
    receiver_direction_preserved: bool
    causal_cutoff_preserved: bool
    action_stage_semantics_preserved: bool
    outcome_derived: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.structural_recurrence_role_id not in set(TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS):
            raise ValueError("role binding names an unknown target construction validation role")
        require_sorted_unique_strings(
            self.target_native_role_ids,
            field_name="target_native_role_ids",
            allow_empty=False,
        )
        if not all(
            (
                self.native_units_preserved,
                self.native_frame_preserved,
                self.receiver_direction_preserved,
                self.causal_cutoff_preserved,
                self.action_stage_semantics_preserved,
            )
        ):
            raise ValueError("mapping candidate loses required native semantics")
        if self.outcome_derived:
            raise ValueError("mapping bindings cannot be outcome-derived")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCompatibilityMapCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-compatibility-map-candidate'

    candidate_id: str
    target_id: str
    finite_roster_id: str
    native_dossier: ObjectIdentity
    role_bindings: tuple[TargetConstructValidationRoleBinding, ...]
    semantic_losses: tuple[TargetConstructValidationSemanticLoss, ...]
    requested_accepted_applied_realized_distinct: bool
    target_native_action_chart_preserved: bool
    authored_before_development: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("candidate_id", "target_id", "finite_roster_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.role_bindings,
            attribute="binding_id",
            field_name="role_bindings",
        )
        require_sorted_unique_ids(
            self.semantic_losses,
            attribute="loss_id",
            field_name="semantic_losses",
        )
        if tuple(binding.structural_recurrence_role_id for binding in self.role_bindings) != (
            TARGET_CONSTRUCT_VALIDATION_REQUIRED_ROLE_IDS
        ):
            raise ValueError("mapping candidate must bind exact D/H/A/R/tau roles")
        if any(loss.target_id != self.target_id for loss in self.semantic_losses):
            raise ValueError("mapping semantic loss crosses targets")
        if not all(
            (
                self.requested_accepted_applied_realized_distinct,
                self.target_native_action_chart_preserved,
                self.authored_before_development,
            )
        ):
            raise ValueError("mapping candidate violates the predevelopment firewall")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("map candidates must be authored outcome-blindly")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationMappingDevelopmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-mapping-development-assessment'

    assessment_id: str
    candidate: TargetConstructValidationCompatibilityMapCandidate
    construct_review: ObjectIdentity
    same_complete_unit_ids_sha256: str
    construct_review_passed: bool
    unsafe_false_admission_count: int
    categorical_mismatch_count: int
    prediction_set_cardinality: int
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )
        for name in (
            "unsafe_false_admission_count",
            "categorical_mismatch_count",
            "prediction_set_cardinality",
            "evaluation_outcome_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.evaluation_outcome_count:
            raise ValueError("mapping selection cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("mapping assessment must be development-only")

    @property
    def eligible(self) -> bool:
        return self.construct_review_passed and not any(
            loss.blocks_selection for loss in self.candidate.semantic_losses
        )

    @property
    def lexicographic_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.unsafe_false_admission_count,
            self.categorical_mismatch_count,
            self.prediction_set_cardinality,
            scientific_description_octets(self.candidate) * 8,
            self.candidate.candidate_id,
        )


def select_mapping_candidate(
    assessments: Iterable[TargetConstructValidationMappingDevelopmentAssessment],
) -> TargetConstructValidationMappingDevelopmentAssessment:
    """Select one of at most three frozen maps with no evaluation rescue."""

    values = tuple(assessments)
    if not values or len(values) > 3:
        raise ValueError("MAPPING_UNRESOLVED: roster must contain one to three maps")
    candidates = tuple(value.candidate for value in values)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("mapping roster contains duplicate identities")
    if len({candidate.target_id for candidate in candidates}) != 1:
        raise ValueError("mapping candidates cannot cross targets")
    if len({candidate.finite_roster_id for candidate in candidates}) != 1:
        raise ValueError("mapping candidates must belong to one frozen roster")
    if len({value.same_complete_unit_ids_sha256 for value in values}) != 1:
        raise ValueError("mapping candidates must use identical complete units")
    eligible = tuple(value for value in values if value.eligible)
    if not eligible:
        raise ValueError("MAPPING_UNRESOLVED: no construct-valid candidate")
    return min(eligible, key=lambda value: value.lexicographic_key)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDenominatorCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-denominator-candidate'

    candidate_id: str
    target_id: str
    alternative_kind: TargetConstructValidationDenominatorAlternativeKind
    factor_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    causal_loss_ids: tuple[str, ...]
    impossible: bool
    impossible_reason: str | None
    authored_before_development: bool

    def __post_init__(self) -> None:
        for name in ("candidate_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("factor_ids", "stratum_ids", "causal_loss_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.impossible != (self.impossible_reason is not None):
            raise ValueError("impossible denominator requires exactly one reason")
        if self.impossible_reason is not None:
            validate_nonempty(self.impossible_reason, field_name="impossible_reason")
        if not self.authored_before_development:
            raise ValueError("denominator alternatives must freeze before development")

    @property
    def factor_and_stratum_count(self) -> int:
        return len(self.factor_ids) + len(self.stratum_ids)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDenominatorDevelopmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-denominator-development-assessment'

    assessment_id: str
    candidate: TargetConstructValidationDenominatorCandidate
    same_complete_unit_ids_sha256: str
    legal_action_alphabet_preserved: bool
    forecast_roster_preserved: bool
    target_native_response_preserved: bool
    target_native_policy_preserved: bool
    unsafe_error_count: int
    simultaneous_precision_passed: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )
        if self.unsafe_error_count < 0 or self.evaluation_outcome_count < 0:
            raise ValueError("denominator counts must be nonnegative")
        if self.evaluation_outcome_count:
            raise ValueError("denominator selection cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("denominator assessment must be development-only")

    @property
    def eligible(self) -> bool:
        return not self.candidate.impossible and all(
            (
                self.legal_action_alphabet_preserved,
                self.forecast_roster_preserved,
                self.target_native_response_preserved,
                self.target_native_policy_preserved,
                not self.candidate.causal_loss_ids,
                self.unsafe_error_count == 0,
                self.simultaneous_precision_passed,
            )
        )


def select_minimal_denominator(
    assessments: Iterable[TargetConstructValidationDenominatorDevelopmentAssessment],
) -> TargetConstructValidationDenominatorDevelopmentAssessment:
    """Select the smallest safe causal denominator on identical units."""

    values = tuple(assessments)
    if not values:
        raise ValueError("DENOMINATOR_UNRESOLVED: empty candidate lattice")
    if len({value.candidate.candidate_id for value in values}) != len(values):
        raise ValueError("denominator candidate identities are duplicated")
    if len({value.candidate.target_id for value in values}) != 1:
        raise ValueError("denominator candidates cannot cross targets")
    if len({value.same_complete_unit_ids_sha256 for value in values}) != 1:
        raise ValueError("denominator candidates must use identical complete units")
    kinds = {value.candidate.alternative_kind for value in values}
    if kinds != set(TargetConstructValidationDenominatorAlternativeKind):
        raise ValueError("denominator lattice must include proposed/split/merge/omission")
    eligible = tuple(value for value in values if value.eligible)
    if not eligible:
        raise ValueError("DENOMINATOR_UNRESOLVED: no safe causal candidate")
    return min(
        eligible,
        key=lambda value: (
            value.candidate.factor_and_stratum_count,
            scientific_description_octets(value.candidate),
            value.candidate.candidate_id,
        ),
    )
