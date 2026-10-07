"""Strict Six-matrix response action authoring, prospective evaluation topology and protected-field contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_stable_id,
)


class MatrixResponseActionDomain(StrEnum):
    PARENT_COUPLING = "PARENT_COUPLING"
    LOWER_WORLD_FORCE = "LOWER_WORLD_FORCE"


class MatrixResponseOuterArmDisposition(StrEnum):
    QUALIFIED_PROSPECTIVE_PROGRAMME = "QUALIFIED_PROSPECTIVE_PROGRAMME"
    COMPARATOR_ONLY = "COMPARATOR_ONLY"
    HOLD_CONTROL = "HOLD_CONTROL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ProtectedAccessClaim(StrEnum):
    PROCEDURAL_BLINDNESS_ONLY = "PROCEDURAL_BLINDNESS_ONLY"


@dataclass(frozen=True, slots=True)
class MatrixResponsePrimitiveAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-primitive-action'
    VERSION: ClassVar[str] = "1.0.0"

    primitive_id: str
    domain: MatrixResponseActionDomain
    delta_x: Decimal
    delta_y: Decimal
    simultaneous_channels: bool
    measured_hold: bool
    native_unit: str
    native_frame: str

    def __post_init__(self) -> None:
        validate_stable_id(self.primitive_id, field_name="primitive_id")
        validate_decimal(self.delta_x, field_name="delta_x")
        validate_decimal(self.delta_y, field_name="delta_y")
        if self.native_unit not in {"dimensionless-alpha-tilde", "dimensionless-force"}:
            raise ValueError("Six-matrix response action uses an unknown native unit")
        if self.native_frame not in {
            "anonymous-parent-factor-frame",
            "anonymous-hermitian-mode-frame",
        }:
            raise ValueError("Six-matrix response action uses an unknown native frame")
        if self.measured_hold:
            if self.delta_x != 0 or self.delta_y != 0 or self.simultaneous_channels:
                raise ValueError("measured HOLD must be a nonempty zero-ramp fibre")
        elif self.delta_x == 0 and self.delta_y == 0:
            raise ValueError("active primitive cannot have zero amplitude")
        if self.domain is MatrixResponseActionDomain.LOWER_WORLD_FORCE and self.delta_y != 0:
            raise ValueError("lower-world force uses one anonymous signed coordinate")


@dataclass(frozen=True, slots=True)
class MatrixResponseActionGrammar(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-action-grammar'
    VERSION: ClassVar[str] = "1.0.0"

    chart_id: str
    domain: MatrixResponseActionDomain
    primitives: tuple[MatrixResponsePrimitiveAction, ...]
    maximum_active_depth: int
    allow_all_ordered_active_compositions: bool
    hold_standalone_only: bool
    maximum_occurrences_per_word: int
    ramp_steps: int
    dwell_steps: int
    qualification_steps: int
    maximum_rate_per_step: Decimal
    generalized_work_includes_explicit_parameter_term: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        require_sorted_unique_ids(
            self.primitives,
            attribute="primitive_id",
            field_name="primitives",
        )
        if any(value.domain is not self.domain for value in self.primitives):
            raise ValueError("Six-matrix response chart mixes parent and lower-world actions")
        holds = tuple(value for value in self.primitives if value.measured_hold)
        active = tuple(value for value in self.primitives if not value.measured_hold)
        if len(holds) != 1 or not active:
            raise ValueError("Six-matrix response chart requires one measured HOLD and active primitives")
        expected_depth = 3 if self.domain is MatrixResponseActionDomain.PARENT_COUPLING else 2
        if self.maximum_active_depth != expected_depth:
            raise ValueError("Six-matrix response chart uses the wrong frozen composition depth")
        if not self.allow_all_ordered_active_compositions or not self.hold_standalone_only:
            raise ValueError("Six-matrix response baseline freezes complete active words and standalone HOLD")
        expected_occurrences = self.maximum_active_depth * max(
            2 if value.simultaneous_channels else 1 for value in active
        )
        if self.maximum_occurrences_per_word != expected_occurrences:
            raise ValueError("Six-matrix response occurrence ceiling differs from its exact grammar")
        for name in ("ramp_steps", "dwell_steps", "qualification_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        validate_decimal(
            self.maximum_rate_per_step,
            field_name="maximum_rate_per_step",
            minimum=Decimal(0),
        )
        if self.maximum_rate_per_step == 0:
            raise ValueError("Six-matrix response maximum rate must be positive")
        if not self.generalized_work_includes_explicit_parameter_term:
            raise ValueError("Six-matrix response work must include partial-S/partial-alpha delivery work")

    @property
    def action_fibre_count(self) -> int:
        active_count = len(tuple(value for value in self.primitives if not value.measured_hold))
        count = 1
        for depth in range(1, self.maximum_active_depth + 1):
            count += active_count**depth
        return count


@dataclass(frozen=True, slots=True)
class MatrixResponseOuterStudyAuthoringConfig(CanonicalRecord):
    """Compact outcome-blind input that expands to the unchanged programme compiler owner."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-outer-study-authoring-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    config_version: str
    law_family_id: str
    outer_action_grammar: MatrixResponseActionGrammar
    inner_action_grammar: MatrixResponseActionGrammar
    numerical_member_ids: tuple[str, ...]
    candidate_version_ids: tuple[str, ...]
    outer_support_cell_ids: tuple[str, ...]
    inner_support_cell_ids: tuple[str, ...]
    outer_admission_cartesian_count: int
    inner_admission_cartesian_count_per_law: int
    admission_gate_ids: tuple[str, ...]
    controller_study_schema: str
    controller_compiler_schema: str
    controller_runtime_id: str
    arbitrary_chart_binder_id: str
    expected_study_identity_rule_id: str
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "law_family_id",
            "controller_runtime_id",
            "arbitrary_chart_binder_id",
            'expected_study_identity_rule_id',
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        for name in (
            "numerical_member_ids",
            "candidate_version_ids",
            "outer_support_cell_ids",
            "inner_support_cell_ids",
            'admission_gate_ids',
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        outer_count = (
            len(self.numerical_member_ids)
            * len(self.candidate_version_ids)
            * self.outer_action_grammar.action_fibre_count
            * len(self.outer_support_cell_ids)
        )
        inner_count = (
            len(self.numerical_member_ids)
            * len(self.candidate_version_ids)
            * self.inner_action_grammar.action_fibre_count
            * len(self.inner_support_cell_ids)
        )
        if self.outer_admission_cartesian_count != outer_count:
            raise ValueError("outer programme admission cardinality differs from the exact Cartesian roster")
        if self.inner_admission_cartesian_count_per_law != inner_count:
            raise ValueError("inner programme admission cardinality differs from the exact Cartesian roster")
        if self.controller_study_schema != 'empirical-lawhood/planning/admission-controller-study':
            raise ValueError("Six-matrix response must author the sole ControllerProgramme")
        if self.controller_compiler_schema != 'empirical-lawhood/runtime/compiled-admission-controller-study':
            raise ValueError("Six-matrix response must use the sole compiled controller")
        if len(self.admission_gate_ids) != 9:
            raise ValueError("Six-matrix response programme admission admission is the noncompensating nine-gate intersection")
        if self.grants_authority:
            raise ValueError("programme authoring cannot grant authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseOuterArm(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-outer-arm'
    VERSION: ClassVar[str] = "1.0.0"

    arm_id: str
    disposition: MatrixResponseOuterArmDisposition
    information_contract_id: str
    capacity_contract_id: str
    law_family_id: str | None
    study_id_rule: str | None
    inner_controller_use_applicable: bool

    def __post_init__(self) -> None:
        for name in ("arm_id", "information_contract_id", "capacity_contract_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.law_family_id is not None:
            validate_stable_id(self.law_family_id, field_name="law_family_id")
        if self.disposition is MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME:
            if self.law_family_id is None or self.study_id_rule is None:
                raise ValueError("prospective evaluation arm requires a law and programme")
        elif self.law_family_id is not None or self.study_id_rule is not None:
            raise ValueError("non-prospective evaluation arms cannot claim a qualified law/programme")
        if (
            self.inner_controller_use_applicable
            and self.disposition is not MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME
        ):
            raise ValueError("inner prospective evaluation requires an independently qualified outer arm")


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveEvaluationTopology(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-prospective-evaluation-topology'
    VERSION: ClassVar[str] = "1.0.0"

    topology_id: str
    physical_panel_count: int
    target_constitution_ids: tuple[str, ...]
    panels_per_target_constitution: int
    outer_arms: tuple[MatrixResponseOuterArm, ...]
    possible_parent_action_word_count: int
    contingent_template_rule_id: str
    activation_rule_id: str
    within_stratum_role_ids: tuple[str, ...]
    minimum_efficacy_units_per_stratum: int
    minimum_hold_controls_per_stratum: int
    minimum_reserves_per_stratum: int
    below_minimum_terminal_id: str
    matched_hold_rule_id: str
    common_random_number_rule_id: str
    sealed_reference_rule_id: str
    inner_constitution_ids: tuple[str, ...]
    inner_modes_per_constitution: int
    inner_action_hold_child_per_applicable_cell: bool
    zero_constitution_inner_terminal_id: str
    panel_is_sole_inference_unit: bool
    frozen_before_admission_outcome_visibility: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "topology_id",
            "contingent_template_rule_id",
            "activation_rule_id",
            "below_minimum_terminal_id",
            "matched_hold_rule_id",
            "common_random_number_rule_id",
            "sealed_reference_rule_id",
            "zero_constitution_inner_terminal_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.target_constitution_ids,
            field_name="target_constitution_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.outer_arms, attribute="arm_id", field_name="outer_arms")
        require_sorted_unique_strings(
            self.within_stratum_role_ids,
            field_name="within_stratum_role_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.inner_constitution_ids,
            field_name="inner_constitution_ids",
            allow_empty=False,
        )
        if (
            self.physical_panel_count != 80
            or self.target_constitution_ids != ("00", "01", "10", "11")
            or self.panels_per_target_constitution != 20
            or len(self.outer_arms) != 5
        ):
            raise ValueError("Six-matrix response freezes 80 panels, four targets and five paired arms")
        if self.possible_parent_action_word_count != 585:
            raise ValueError("Six-matrix response prospective evaluation topology must cover the complete outer chart")
        if self.within_stratum_role_ids != ("efficacy", "hold-control", "reserve"):
            raise ValueError("Six-matrix response prospective evaluation strata require efficacy, HOLD-control and reserve roles")
        if (
            min(
                self.minimum_efficacy_units_per_stratum,
                self.minimum_hold_controls_per_stratum,
                self.minimum_reserves_per_stratum,
            )
            < 1
        ):
            raise ValueError("every activated prospective evaluation stratum requires all allocation roles")
        if self.inner_constitution_ids != ("01", "10", "11"):
            raise ValueError("00 has no lower-world child and must remain absent")
        if self.inner_modes_per_constitution != 4:
            raise ValueError("Six-matrix response freezes four anonymous role slots per lower world")
        if not (
            self.inner_action_hold_child_per_applicable_cell
            and self.panel_is_sole_inference_unit
            and self.frozen_before_admission_outcome_visibility
        ):
            raise ValueError("Six-matrix response nested prospective evaluation topology violates freeze or unit semantics")
        if self.grants_authority:
            raise ValueError("prospective evaluation topology cannot grant issue/execution/reveal authority")


@dataclass(frozen=True, slots=True)
class ProtectedFieldAccessRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/protected-field-access-rule'
    VERSION: ClassVar[str] = "1.0.0"

    rule_id: str
    principal_id: str
    field_ids: tuple[str, ...]
    access_mode: str
    earliest_stage_id: str

    def __post_init__(self) -> None:
        for name in ("rule_id", "principal_id", "earliest_stage_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.field_ids, field_name="field_ids", allow_empty=False)
        if self.access_mode not in {"read", "write-only"}:
            raise ValueError("protected-field access mode is not closed")


@dataclass(frozen=True, slots=True)
class ProtectedObservableAccessManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/protected-observable-access-manifest'
    VERSION: ClassVar[str] = "1.0.0"

    manifest_id: str
    access_claim: ProtectedAccessClaim
    operating_system_boundary_id: str
    rules: tuple[ProtectedFieldAccessRule, ...]
    q4_excluded_allowed_field_ids: tuple[str, ...]
    protected_outcome_field_ids: tuple[str, ...]
    prediction_policy_field_ids: tuple[str, ...]
    reveal_authority_id: str
    confidentiality_claimed: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "operating_system_boundary_id",
            "reveal_authority_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.rules, attribute="rule_id", field_name="rules")
        for name in (
            "q4_excluded_allowed_field_ids",
            "protected_outcome_field_ids",
            "prediction_policy_field_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if set(self.q4_excluded_allowed_field_ids) & set(self.protected_outcome_field_ids):
            raise ValueError("excluded q=4 qualification cannot expose protected outcomes")
        if not set(self.prediction_policy_field_ids).isdisjoint(self.protected_outcome_field_ids):
            raise ValueError("prediction/policy input cannot expose protected outcomes")
        if self.access_claim is not ProtectedAccessClaim.PROCEDURAL_BLINDNESS_ONLY:
            raise ValueError("Six-matrix response baseline has no enforced multi-principal storage boundary")
        if self.confidentiality_claimed or self.grants_authority:
            raise ValueError("procedural blindness grants neither confidentiality nor authority")
