"""Typed Six-matrix response role, law, structural and statistical method contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_schema,
    validate_semantic_version,
    validate_stable_id,
)


class ConstitutiveLawFamily(StrEnum):
    L_UP = "L_UP"
    L_DOWN = "L_DOWN"


class RoleSpaceKind(StrEnum):
    CONFIGURATION = "CONFIGURATION"
    PHASE_SPACE = "PHASE_SPACE"
    HERMITIAN_PROBE = "HERMITIAN_PROBE"
    RECEIVER_RESPONSE = "RECEIVER_RESPONSE"
    LOCAL_LAW_LATENT = "LOCAL_LAW_LATENT"
    HERMITIAN_FORCE = "HERMITIAN_FORCE"


class CrossSizeMapKind(StrEnum):
    EXACT_SCALAR_NORMALIZATION = "EXACT_SCALAR_NORMALIZATION"
    QUOTIENT_INVARIANT_SUMMARY = "QUOTIENT_INVARIANT_SUMMARY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseRoleSpace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-role-space'
    VERSION: ClassVar[str] = "1.0.0"

    role_space_id: str
    kind: RoleSpaceKind
    real_dimension_formula: str
    ambient_space_id_formula: str
    basis_frame_id: str
    scalar_convention: str
    inner_product_id: str
    gauge_quotient_id: str
    cross_size_map_kind: CrossSizeMapKind
    degeneracy_rule_id: str
    projector_rank_rule_id: str
    projector_tolerance_rule_id: str
    response_map_id: str
    realizable_force_map_id: str

    def __post_init__(self) -> None:
        for name in (
            "role_space_id",
            "basis_frame_id",
            "inner_product_id",
            "gauge_quotient_id",
            "degeneracy_rule_id",
            "projector_rank_rule_id",
            "projector_tolerance_rule_id",
            "response_map_id",
            "realizable_force_map_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.real_dimension_formula not in {
            "q^4",
            "6*q^4",
            "12*q^4",
            "24",
            "36",
        }:
            raise ValueError("Six-matrix response role-space dimension formula is not closed")
        if self.ambient_space_id_formula not in {
            "six-matrix-response.ambient.configuration.q{q}",
            "six-matrix-response.ambient.phase-space.q{q}",
            "six-matrix-response.ambient.hermitian-probe.q{q}",
            "six-matrix-response.ambient.receiver-response",
            "six-matrix-response.ambient.local-law-latent",
            "six-matrix-response.ambient.hermitian-force.q{q}",
        }:
            raise ValueError("Six-matrix response role-space ambient identity formula is unknown")
        if self.scalar_convention not in {"real", "complex-hermitian-realification"}:
            raise ValueError("Six-matrix response role-space scalar convention is unknown")


@dataclass(frozen=True, slots=True)
class MatrixResponseRoleEquivarianceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-role-equivariance-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    role_spaces: tuple[MatrixResponseRoleSpace, ...]
    transformation_ids: tuple[str, ...]
    invariant_feature_ids: tuple[str, ...]
    factor_anonymization_rule_id: str
    degenerate_subspace_rule_id: str
    same_q_projector_comparison_id: str
    cross_q_projector_disposition: CrossSizeMapKind
    adversarial_conformance_ids: tuple[str, ...]
    maximum_role_slots_per_constitution: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for name in (
            "factor_anonymization_rule_id",
            "degenerate_subspace_rule_id",
            "same_q_projector_comparison_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.role_spaces,
            attribute="role_space_id",
            field_name="role_spaces",
        )
        if {value.kind for value in self.role_spaces} != set(RoleSpaceKind):
            raise ValueError("Six-matrix response role-space family must contain every typed ambient space")
        for name in (
            "transformation_ids",
            "invariant_feature_ids",
            "adversarial_conformance_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if self.cross_q_projector_disposition is not CrossSizeMapKind.NOT_APPLICABLE:
            raise ValueError("Six-matrix response baseline forbids direct projector comparison across q")
        if self.maximum_role_slots_per_constitution != 4:
            raise ValueError("Six-matrix response baseline freezes four anonymous lower-world role slots")


@dataclass(frozen=True, slots=True)
class MatrixResponseLawFamilyConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-law-family-config'
    VERSION: ClassVar[str] = "1.0.0"

    law_family_id: str
    family: ConstitutiveLawFamily
    system_id: str
    prepared_denominator_id: str
    retained_history_id: str
    action_chart_id: str
    receiver_id: str
    horizon_id: str
    relation_id: str
    action_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    candidate_payload_schema: str
    candidate_payload_version: str
    candidate_decoder_id: str
    method_kind: LawMethodKind
    representation_kind: LawRepresentationKind
    causal_strength_ceiling: CausalStrength
    mapping_assumption_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    candidate_version_ids: tuple[str, ...]
    numerical_member_ids: tuple[str, ...]
    qualification_view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "law_family_id",
            "system_id",
            "prepared_denominator_id",
            "retained_history_id",
            "action_chart_id",
            "receiver_id",
            "horizon_id",
            "relation_id",
            "candidate_decoder_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.candidate_payload_schema)
        validate_semantic_version(self.candidate_payload_version)
        for name in (
            "action_quantity_ids",
            "receiver_quantity_ids",
            "mapping_assumption_ids",
            "decisive_falsifier_ids",
            "candidate_version_ids",
            "numerical_member_ids",
            "qualification_view_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if self.method_kind is not LawMethodKind.NONLINEAR_LOCAL:
            raise ValueError("Six-matrix response baseline qualifies only predeclared nonlinear-local candidates")
        expected_representation = (
            LawRepresentationKind.FINITE_ACTION_OPERATOR
            if self.family is ConstitutiveLawFamily.L_UP
            else LawRepresentationKind.LOCAL_STATE_SPACE
        )
        if self.representation_kind is not expected_representation:
            raise ValueError("Six-matrix response law family uses the wrong representation kind")
        if self.causal_strength_ceiling is not CausalStrength.SIMULATOR_INTERVENTION:
            raise ValueError("Six-matrix response law evidence cannot exceed simulator intervention")


@dataclass(frozen=True, slots=True)
class MatrixResponseLawMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-law-method-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    parent_law: MatrixResponseLawFamilyConfig
    lower_laws: tuple[MatrixResponseLawFamilyConfig, ...]
    qualification_profile_id: str
    method_evidence_producer_id: str
    sole_finalizer_service_id: str
    batch_owner_id: str
    atlas_owner_id: str
    leakage_statistic_id: str
    leakage_upper_bound: Decimal
    grants_law_truth: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "qualification_profile_id",
            "method_evidence_producer_id",
            "sole_finalizer_service_id",
            "batch_owner_id",
            "atlas_owner_id",
            "leakage_statistic_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.parent_law.family is not ConstitutiveLawFamily.L_UP:
            raise ValueError("Six-matrix response method config requires one parent L_up contract")
        require_sorted_unique_ids(
            self.lower_laws,
            attribute="law_family_id",
            field_name="lower_laws",
        )
        if not self.lower_laws or any(
            value.family is not ConstitutiveLawFamily.L_DOWN for value in self.lower_laws
        ):
            raise ValueError("Six-matrix response method config requires exact lower-world laws")
        validate_decimal(
            self.leakage_upper_bound,
            field_name="leakage_upper_bound",
            minimum=Decimal(0),
        )
        if self.leakage_upper_bound >= Decimal(1):
            raise ValueError("leakage upper bound must be a proper fraction")
        if self.grants_law_truth:
            raise ValueError("method adapter cannot grant ResponseLaw truth")


@dataclass(frozen=True, slots=True)
class MatrixResponsePropertyFace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-property-face'
    VERSION: ClassVar[str] = "1.0.0"

    face_id: str
    property_kind_id: str
    method_id: str
    ambient_map_id: str
    target_roster_rule_id: str
    metric_id: str
    uncertainty_operation_id: str
    categorical_companion_id: str
    decisive_falsifier_ids: tuple[str, ...]
    cross_q_applicability: CrossSizeMapKind

    def __post_init__(self) -> None:
        for name in (
            "face_id",
            "property_kind_id",
            "method_id",
            "ambient_map_id",
            "target_roster_rule_id",
            "metric_id",
            "uncertainty_operation_id",
            "categorical_companion_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class MatrixResponseStructuralFacePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-structural-face-plan'
    VERSION: ClassVar[str] = "1.0.0"

    plan_id: str
    faces: tuple[MatrixResponsePropertyFace, ...]
    complete_unit_roster_rule_id: str
    evidence_dependence_id: str
    prediction_cutoff_id: str
    supported_terminal_id: str
    opposed_terminal_id: str
    unevaluable_terminal_id: str
    missing_terminal_id: str
    optional_admission_and_controller_evaluation_independent: bool

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "complete_unit_roster_rule_id",
            "evidence_dependence_id",
            "prediction_cutoff_id",
            "supported_terminal_id",
            "opposed_terminal_id",
            "unevaluable_terminal_id",
            "missing_terminal_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.faces, attribute="face_id", field_name="faces")
        if len(self.faces) != 6:
            raise ValueError("Six-matrix response structural plan freezes exactly six independent faces")
        if not self.optional_admission_and_controller_evaluation_independent:
            raise ValueError("structural recurrence must remain parallel to programme admission/prospective evaluation")


@dataclass(frozen=True, slots=True)
class MatrixResponsePairedPanelReducerConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-paired-panel-reducer-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    independent_unit_id: str
    panel_count: int
    outer_arm_ids: tuple[str, ...]
    primary_contrast_id: str
    primary_contrast_direction: str
    paired_binary_interval_id: str
    continuous_interval_id: str
    bootstrap_resample_unit: str
    bootstrap_replicates: int
    familywise_alpha: Decimal
    multiplicity_method_id: str
    missingness_rule_id: str
    invalidity_rule_id: str
    intent_to_treat: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "independent_unit_id",
            "primary_contrast_id",
            "paired_binary_interval_id",
            "continuous_interval_id",
            "multiplicity_method_id",
            "missingness_rule_id",
            "invalidity_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.outer_arm_ids,
            field_name="outer_arm_ids",
            allow_empty=False,
        )
        if self.panel_count != 80 or len(self.outer_arm_ids) != 5:
            raise ValueError("Six-matrix response outer evaluation is exactly 80 panels and five arms")
        if self.primary_contrast_direction not in {"greater", "less"}:
            raise ValueError("primary contrast direction is not closed")
        if self.bootstrap_resample_unit != "whole-panel":
            raise ValueError("Six-matrix response bootstrap must resample independent panels")
        if self.bootstrap_replicates < 9999:
            raise ValueError("Six-matrix response panel bootstrap requires at least 9,999 replicates")
        validate_decimal(
            self.familywise_alpha,
            field_name="familywise_alpha",
            minimum=Decimal(0),
        )
        if not Decimal(0) < self.familywise_alpha < Decimal(1):
            raise ValueError("familywise alpha must lie strictly inside (0,1)")
        if not self.intent_to_treat:
            raise ValueError("Six-matrix response paired-panel inference is intent-to-treat")
