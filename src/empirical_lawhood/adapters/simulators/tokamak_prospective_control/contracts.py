"""Fresh Gym--TORAX task, preparation, action and assay contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)


class TaskArchetype(StrEnum):
    RECEIVER_SINK = 'RECEIVER_SINK'
    HIDDEN_HISTORY = 'HIDDEN_HISTORY'
    ACTION_ORDER = 'ACTION_ORDER'
    MEMBER_HOLD = 'MEMBER_HOLD'


class PreparationPoolRole(StrEnum):
    ACQUISITION = "ACQUISITION"
    ACTION_ALGEBRA = "ACTION_ALGEBRA"
    CONFIRMATORY = "CONFIRMATORY"
    OUTSIDE_SUPPORT_CALIBRATION = "OUTSIDE_SUPPORT_CALIBRATION"
    REFERENCE = "REFERENCE"


class GeneratedArmKind(StrEnum):
    WITNESS_GATED_IO = "WITNESS_GATED_IO"
    IO_ONLY = "IO_ONLY"
    FINITE_MPC = "FINITE_MPC"
    ROBUST_MAXIMIN = "ROBUST_MAXIMIN"


class ActionSegmentPort(StrEnum):
    CURRENT = "CURRENT"
    HEATING = "HEATING"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class GeneratedActionSegment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/generated-action-segment'

    segment_id: str
    port: ActionSegmentPort
    signed_amplitude: Decimal
    native_unit: str
    onset_seconds: Decimal
    duration_seconds: Decimal
    occurrence_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("segment_id", self.segment_id),
            ("occurrence_id", self.occurrence_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.signed_amplitude, field_name="signed_amplitude")
        validate_decimal(self.onset_seconds, field_name="onset_seconds", minimum=Decimal(0))
        validate_decimal(
            self.duration_seconds,
            field_name="duration_seconds",
            minimum=Decimal(0),
        )
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.port is ActionSegmentPort.HOLD:
            if self.signed_amplitude != 0:
                raise ValueError("HOLD segment changes a native source")
        elif self.signed_amplitude == 0 or self.duration_seconds == 0:
            raise ValueError("active action segment is null")


@dataclass(frozen=True, slots=True)
class GeneratedActionWordSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/generated-action-word-spec'

    word_id: str
    label: str
    segments: tuple[GeneratedActionSegment, ...]
    receiver_clock_id: str
    is_native_hold: bool
    requested_accepted_applied_realized_distinct: bool
    realized_effort_is_controlling: bool
    clipping_invalidates_delivery: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        validate_nonempty(self.label, field_name="label")
        validate_stable_id(self.receiver_clock_id, field_name="receiver_clock_id")
        require_sorted_unique_ids(self.segments, attribute="segment_id", field_name="segments")
        if not self.segments:
            raise ValueError("generated action word has no native segment")
        if self.is_native_hold is not all(
            value.port is ActionSegmentPort.HOLD for value in self.segments
        ):
            raise ValueError("generated HOLD semantics differ from native identity")
        if (
            not self.requested_accepted_applied_realized_distinct
            or not self.realized_effort_is_controlling
            or not self.clipping_invalidates_delivery
        ):
            raise ValueError("generated action collapses delivery or effort stages")


@dataclass(frozen=True, slots=True)
class GeneratedReceiverDimension(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/generated-receiver-dimension'

    receiver_id: str
    native_quantity: str
    native_unit: str
    direction: int
    role_ids: tuple[str, ...]
    horizon_seconds: Decimal
    materiality: Decimal
    numeric_floor: Decimal
    validity_rule_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("receiver_id", self.receiver_id),
            ("validity_rule_id", self.validity_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_quantity, field_name="native_quantity")
        validate_nonempty(self.native_unit, field_name="native_unit")
        require_sorted_unique_strings(self.role_ids, field_name="role_ids", allow_empty=False)
        if self.direction not in {-1, 1}:
            raise ValueError("receiver direction must be a signed native gauge")
        for decimal_name, decimal_value in (
            ("horizon_seconds", self.horizon_seconds),
            ("materiality", self.materiality),
            ("numeric_floor", self.numeric_floor),
        ):
            validate_decimal(
                decimal_value,
                field_name=decimal_name,
                minimum=Decimal(0),
            )
        if self.horizon_seconds == 0 or self.materiality == 0:
            raise ValueError("receiver lacks a horizon or native materiality")


@dataclass(frozen=True, slots=True)
class LawToActionTaskSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/law-to-action-task-spec'

    task_id: str
    archetype: TaskArchetype
    source_origin_id: str
    formula_origin_id: str
    grid_origin_id: str
    denominator_id: str
    native_action_words: tuple[GeneratedActionWordSpec, ...]
    receiver_dimensions: tuple[GeneratedReceiverDimension, ...]
    model_member_ids: tuple[str, ...]
    preparation_generator: ObjectIdentity
    causal_cutoff: ObjectIdentity
    generated_before_outcomes: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("source_origin_id", self.source_origin_id),
            ("formula_origin_id", self.formula_origin_id),
            ("grid_origin_id", self.grid_origin_id),
            ("denominator_id", self.denominator_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.native_action_words,
            attribute="word_id",
            field_name="native_action_words",
        )
        require_sorted_unique_ids(
            self.receiver_dimensions,
            attribute="receiver_id",
            field_name="receiver_dimensions",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        if (
            len(self.native_action_words) != 4
            or sum(value.is_native_hold for value in self.native_action_words) != 1
            or len(self.model_member_ids) != 2
            or len(self.receiver_dimensions) < 2
            or not self.generated_before_outcomes
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("generated task lacks the four-word/two-member receiver block")


@dataclass(frozen=True, slots=True)
class PreparationOccurrence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/preparation-occurrence'

    occurrence_id: str
    pool_role: PreparationPoolRole
    seed_domain_id: str
    requested_seed: str
    clone_group_id: str
    realized_source_identity_required: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("occurrence_id", self.occurrence_id),
            ("seed_domain_id", self.seed_domain_id),
            ("requested_seed", self.requested_seed),
            ("clone_group_id", self.clone_group_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.realized_source_identity_required:
            raise ValueError("preparation distinctness cannot be seed-only")


@dataclass(frozen=True, slots=True)
class TaskPreparationPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/task-preparation-plan'

    plan_id: str
    task_id: str
    occurrences: tuple[PreparationOccurrence, ...]
    action_algebra_sentinel_task: bool
    pool_seed_domains_disjoint: bool
    cross_pool_clone_groups_disjoint: bool
    independent_unit_count: int
    collapse_disposition: str

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.task_id, field_name="task_id")
        require_sorted_unique_ids(
            self.occurrences,
            attribute="occurrence_id",
            field_name="occurrences",
        )
        counts = Counter(value.pool_role for value in self.occurrences)
        required = {
            PreparationPoolRole.ACQUISITION: 2,
            PreparationPoolRole.REFERENCE: 2,
            PreparationPoolRole.CONFIRMATORY: 3,
            PreparationPoolRole.OUTSIDE_SUPPORT_CALIBRATION: 2,
        }
        if any(counts[key] != count for key, count in required.items()):
            raise ValueError("task preparation plan changes acquisition/reference/controller-use nesting")
        expected_algebra = 2 if self.action_algebra_sentinel_task else 0
        if counts[PreparationPoolRole.ACTION_ALGEBRA] != expected_algebra:
            raise ValueError("task action-algebra preparation roster differs")
        if (
            not self.pool_seed_domains_disjoint
            or not self.cross_pool_clone_groups_disjoint
            or self.independent_unit_count != 1
            or self.collapse_disposition != "UNEVALUABLE_PREPARATION_DISTINCTNESS"
        ):
            raise ValueError("task preparation plan inflates or aliases replication")


@dataclass(frozen=True, slots=True)
class DenominatorAuditSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/denominator-audit-spec'

    audit_spec_id: str
    denominator_axis_ids: tuple[str, ...]
    numerical_interaction_axes: tuple[str, ...]
    interaction_levels: tuple[str, ...]
    source_and_receiver_exact_product_required: bool
    member_drop_allowed: bool
    convergence_claim_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_spec_id, field_name="audit_spec_id")
        require_sorted_unique_strings(
            self.denominator_axis_ids,
            field_name="denominator_axis_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.numerical_interaction_axes,
            field_name="numerical_interaction_axes",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.interaction_levels,
            field_name="interaction_levels",
            allow_empty=False,
        )
        required = {
            "backend",
            "closure",
            "grid",
            "observation-operator",
            "precision",
            "solver",
            "timestep",
        }
        if (
            not required.issubset(self.denominator_axis_ids)
            or self.numerical_interaction_axes
            != ("corrector-depth", "radial-cell-count", "timestep")
            or self.interaction_levels != ("central", "high", "low")
            or not self.source_and_receiver_exact_product_required
            or self.member_drop_allowed
            or self.convergence_claim_authorized
        ):
            raise ValueError("TORAX denominator/numerical interaction audit differs")


@dataclass(frozen=True, slots=True)
class ActionAlgebraCoveragePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/action-algebra-coverage-plan'

    plan_id: str
    sentinel_task_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    preparation_count_per_task: int
    member_count: int
    relation_ids: tuple[str, ...]
    total_episode_count: int
    can_supply_candidate_to_admission_commitment_or_controller_evaluation: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        for name, values in (
            ("sentinel_task_ids", self.sentinel_task_ids),
            ("word_ids", self.word_ids),
            ("relation_ids", self.relation_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if (
            len(self.sentinel_task_ids) != 2
            or len(self.word_ids) != 9
            or self.preparation_count_per_task != 2
            or self.member_count != 2
            or set(self.relation_ids)
            != {"composition", "duration", "future-null", "inverse", "order", "repetition", "sign"}
            or self.total_episode_count != 72
            or self.can_supply_candidate_to_admission_commitment_or_controller_evaluation
        ):
            raise ValueError("protected action-algebra product differs from 2x9x2x2")


@dataclass(frozen=True, slots=True)
class OutsideSupportCalibrationPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/outside-support-calibration-plan'

    plan_id: str
    preparation_count_per_task: int
    member_count: int
    physical_episode_count_per_task: int
    logical_arm_decision_count_per_task: int
    permitted_dispositions: tuple[str, ...]
    active_action_is_failure: bool
    false_safe_hold_is_failure: bool
    can_rescue_efficacy: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        require_sorted_unique_strings(
            self.permitted_dispositions,
            field_name="permitted_dispositions",
            allow_empty=False,
        )
        if (
            self.preparation_count_per_task != 2
            or self.member_count != 2
            or self.physical_episode_count_per_task != 4
            or self.logical_arm_decision_count_per_task != 8
            or self.permitted_dispositions != ("HOLD", "NONATTEMPT")
            or not self.active_action_is_failure
            or not self.false_safe_hold_is_failure
            or self.can_rescue_efficacy
        ):
            raise ValueError("outside-support calibration product differs")


@dataclass(frozen=True, slots=True)
class GeneratedArmBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/generated-arm-binding'

    binding_id: str
    arm: GeneratedArmKind
    method_config_schema: str
    method_config: ObjectIdentity
    shared_capability_ids: tuple[str, ...]
    qualification_profile: ObjectIdentity
    admission_service: ObjectIdentity
    controller_bridge: ObjectIdentity
    local_decision_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_schema(self.method_config_schema)
        require_sorted_unique_strings(
            self.shared_capability_ids,
            field_name="shared_capability_ids",
            allow_empty=False,
        )
        required = {
            "response-law.controller_compile",
            "response-law.law_qualification",
            "response-law.admission_receipts",
        }
        if (
            self.method_config.object_schema != self.method_config_schema
            or not required.issubset(self.shared_capability_ids)
            or self.local_decision_allowed
        ):
            raise ValueError("generated arm forks a shared law/admission/controller decision")


@dataclass(frozen=True, slots=True)
class GeneratedChildScientificConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/generated-child-scientific-config'

    config_id: str
    roster_options: tuple[int, ...]
    archetypes: tuple[TaskArchetype, ...]
    model_member_count: int
    action_count_per_task: int
    arm_bindings: tuple[GeneratedArmBinding, ...]
    acquisition_preparation_count: int
    common_bundle_count: int
    adaptive_round_count: int
    reference_preparation_count: int
    confirmatory_preparation_count: int
    confirmatory_seed_reducer: str
    member_reducer: str
    denominator_audit: DenominatorAuditSpec
    action_algebra: ActionAlgebraCoveragePlan
    outside_support_calibration: OutsideSupportCalibrationPlan

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.arm_bindings,
            attribute="binding_id",
            field_name="arm_bindings",
        )
        if (
            self.roster_options != (8, 12)
            or self.archetypes != tuple(TaskArchetype)
            or self.model_member_count != 2
            or self.action_count_per_task != 4
            or {value.arm for value in self.arm_bindings} != set(GeneratedArmKind)
            or self.acquisition_preparation_count != 2
            or self.common_bundle_count != 4
            or self.adaptive_round_count != 2
            or self.reference_preparation_count != 2
            or self.confirmatory_preparation_count != 3
            or self.confirmatory_seed_reducer != "MEDIAN"
            or self.member_reducer != "MINIMUM"
        ):
            raise ValueError("generated child design differs from the frozen C8/C12 route")


def validate_generated_task_roster(
    tasks: tuple[LawToActionTaskSpec, ...],
    *,
    selected_count: int,
) -> None:
    require_sorted_unique_ids(tasks, attribute="task_id", field_name="tasks")
    if selected_count not in {8, 12} or len(tasks) != selected_count:
        raise ValueError("generated task roster is outside C8/C12")
    counts = Counter(value.archetype for value in tasks)
    expected = selected_count // 4
    if set(counts) != set(TaskArchetype) or any(value != expected for value in counts.values()):
        raise ValueError("generated task roster is not archetype-balanced")
    source_formula_grid = {
        (value.source_origin_id, value.formula_origin_id, value.grid_origin_id) for value in tasks
    }
    if len(source_formula_grid) != len(tasks):
        raise ValueError("generated task roster reuses a source/formula/grid identity")


__all__ = [
    'ActionAlgebraCoveragePlan',
    "ActionSegmentPort",
    'DenominatorAuditSpec',
    'GeneratedActionSegment',
    'GeneratedActionWordSpec',
    'GeneratedArmBinding',
    "GeneratedArmKind",
    'GeneratedChildScientificConfig',
    'GeneratedReceiverDimension',
    "LawToActionTaskSpec",
    'OutsideSupportCalibrationPlan',
    'PreparationOccurrence',
    "PreparationPoolRole",
    "TaskArchetype",
    'TaskPreparationPlan',
    "validate_generated_task_roster",
]
