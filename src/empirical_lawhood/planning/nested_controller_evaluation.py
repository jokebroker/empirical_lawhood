"""Outcome-blind nested prospective controller-evaluation authoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import EvaluationStratumSpec, EvaluatorBoundarySpec, ImplementationBinding, ImplementationRole, UtilityDirection


from empirical_lawhood.planning.finite_response_geometry import FiniteResponseCoordinate


MAX_NESTED_EVALUATION_PLAN_BYTES = 8 * 1024 * 1024


class NestedRepeatReducer(StrEnum):
    ARITHMETIC_MEAN = "ARITHMETIC_MEAN"
    MEDIAN = "MEDIAN"
    WORST_REPEAT = "WORST_REPEAT"


class NestedMemberReducer(StrEnum):
    WORST_MEMBER = "WORST_MEMBER"


class NestedInferenceMethod(StrEnum):
    PREDECLARED_ONE_SIDED_LOWER_BOUND = "PREDECLARED_ONE_SIDED_LOWER_BOUND"


NESTED_REDUCTION_ORDER = (
    "REPEATED_DELIVERY",
    "MODEL_MEMBER",
    "PHYSICAL_INDEPENDENT_UNIT",
)

NESTED_PREPARATION_REDUCTION_ORDER = (
    "PREPARATION_OCCURRENCE",
    "MODEL_MEMBER",
    "PHYSICAL_INDEPENDENT_UNIT",
)

NESTED_CONTROLLER_EVALUATION_BRANCHES = (
    "CANONICAL_CONFORMANCE",
    "COMMITTED_ACTION",
    "QUALIFIED_HOLD_BASELINE",
)

NESTED_CONTROLLER_USE_TERMINAL_MATRIX = (
    "CONTROLLER_USE_UNSAFE",
    "CONTROLLER_USE_DELIVERY_INVALID",
    "CONTROLLER_USE_TECHNICAL_FAILURE",
    "CONTROLLER_USE_UNEVALUABLE",
    "CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS",
    "CONTROLLER_USE_HOLD_DOMINANT",
    "CONTROLLER_USE_VALIDATED",
    "CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY",
    "CONTROLLER_USE_NEGATIVE",
)


@dataclass(frozen=True, slots=True)
class NestedPhysicalUnitSpec(CanonicalRecord):
    """One physical independent unit; nested coordinates never replace it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-physical-unit-spec'

    independent_unit_id: str
    task_id: str
    preparation_unit_id: str
    stratum_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("preparation_unit_id", self.preparation_unit_id),
            ("stratum_id", self.stratum_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class NestedReplicateSpec(CanonicalRecord):
    """One seed/repeated-delivery coordinate nested inside a physical unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-replicate-spec'

    replicate_id: str
    independent_unit_id: str
    task_id: str
    preparation_unit_id: str
    nested_seed_id: str
    repetition_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("replicate_id", self.replicate_id),
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("preparation_unit_id", self.preparation_unit_id),
            ("nested_seed_id", self.nested_seed_id),
            ("repetition_id", self.repetition_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class RepeatedDeliveryReducerRegistration(CanonicalRecord):
    """Frozen registered reducer/inference semantics, never a callable config."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/repeated-delivery-reducer-registration'

    registration_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str
    repeat_reducer: NestedRepeatReducer
    member_reducer: NestedMemberReducer
    inference_method: NestedInferenceMethod
    reduction_order: tuple[str, ...]
    deterministic: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.reduction_order != NESTED_REDUCTION_ORDER:
            raise ValueError("nested controller use reducer order differs from the frozen axis order")
        if not self.deterministic:
            raise ValueError("current nested controller use reducer registration must be deterministic")


@dataclass(frozen=True, slots=True)
class RepeatedDeliveryControllerEvaluationPlan(CanonicalRecord):
    "Separately bound nested controller-use design frozen before any outcome reveal."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/repeated-delivery-controller-evaluation-plan'

    evaluation_plan_id: str
    evaluator_boundary: EvaluatorBoundarySpec
    physical_independent_unit_type_id: str
    independent_units: tuple[NestedPhysicalUnitSpec, ...]
    replicates: tuple[NestedReplicateSpec, ...]
    model_member_ids: tuple[str, ...]
    controller_branch_id: str
    reference_branch_id: str
    hold_baseline_branch_id: str | None
    finite_chart_reference_plan: ObjectIdentity
    reference_action_word: OccurrenceActionWord
    measured_hold_fibre: ObjectIdentity | None
    causal_cutoff_id: str
    commitment_point_id: str
    post_cutoff_action_window_id: str
    effect_quantity_id: str
    effect_native_unit: str
    favorable_direction: UtilityDirection
    reducer: RepeatedDeliveryReducerRegistration
    strata: tuple[EvaluationStratumSpec, ...]
    minimum_evaluable_units: int
    minimum_active_coverage: Decimal
    alpha: Decimal
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    multiplicity_family_id: str
    coverage_rule_id: str
    materiality_rule_id: str
    maximum_claim_ceiling: str
    intent_to_treat: bool
    terminal_matrix: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_plan_id", self.evaluation_plan_id),
            ("physical_independent_unit_type_id", self.physical_independent_unit_type_id),
            ("controller_branch_id", self.controller_branch_id),
            ("reference_branch_id", self.reference_branch_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("commitment_point_id", self.commitment_point_id),
            ("post_cutoff_action_window_id", self.post_cutoff_action_window_id),
            ("effect_quantity_id", self.effect_quantity_id),
            ("multiplicity_family_id", self.multiplicity_family_id),
            ("coverage_rule_id", self.coverage_rule_id),
            ("materiality_rule_id", self.materiality_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.effect_native_unit, field_name="effect_native_unit")
        validate_nonempty(self.maximum_claim_ceiling, field_name="maximum_claim_ceiling")
        require_sorted_unique_ids(
            self.independent_units,
            attribute="independent_unit_id",
            field_name="independent_units",
        )
        require_sorted_unique_ids(
            self.replicates,
            attribute="replicate_id",
            field_name="replicates",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.strata, attribute="stratum_id", field_name="strata")
        if not self.independent_units or not self.replicates:
            raise ValueError("nested controller use requires physical units and repeated-delivery coordinates")
        if self.controller_branch_id == self.reference_branch_id:
            raise ValueError("controller and reference controller use branches must be distinct")
        if self.hold_baseline_branch_id is not None:
            validate_stable_id(
                self.hold_baseline_branch_id,
                field_name="hold_baseline_branch_id",
            )
            if self.hold_baseline_branch_id in {
                self.controller_branch_id,
                self.reference_branch_id,
            }:
                raise ValueError("HOLD baseline branch must be distinct")
        if (self.hold_baseline_branch_id is None) != (self.measured_hold_fibre is None):
            raise ValueError("nested HOLD branch and measured-HOLD identity must appear together")
        if self.measured_hold_fibre is not None and self.measured_hold_fibre.object_schema != (
            'empirical-lawhood/planning/admission-measured-hold-fibre'
        ):
            raise ValueError("nested efficacy baseline must bind measured HOLD")
        if self.finite_chart_reference_plan.object_schema != (
            'empirical-lawhood/planning/finite-chart-reference-plan'
        ):
            raise ValueError("nested correctness requires a finite-chart reference plan")
        units = {value.independent_unit_id: value for value in self.independent_units}
        if len(
            {(value.task_id, value.preparation_unit_id) for value in self.independent_units}
        ) != len(self.independent_units):
            raise ValueError("nested controller use physical-unit identities reuse a task/preparation unit")
        seen_nested: set[tuple[str, str, str]] = set()
        replicate_units: set[str] = set()
        for replicate in self.replicates:
            unit = units.get(replicate.independent_unit_id)
            if (
                unit is None
                or replicate.task_id != unit.task_id
                or replicate.preparation_unit_id != unit.preparation_unit_id
            ):
                raise ValueError("nested replicate rewrites its physical unit/task/preparation")
            coordinate = (
                replicate.independent_unit_id,
                replicate.nested_seed_id,
                replicate.repetition_id,
            )
            if coordinate in seen_nested:
                raise ValueError("nested seed/repetition coordinate is duplicated")
            seen_nested.add(coordinate)
            replicate_units.add(replicate.independent_unit_id)
        if replicate_units != set(units):
            raise ValueError("every physical unit requires at least one nested replicate")
        unit_ids = tuple(sorted(units))
        if self.strata:
            covered = {value for stratum in self.strata for value in stratum.independent_unit_ids}
            if covered != set(unit_ids):
                raise ValueError("nested controller use strata must cover the exact physical-unit roster")
            if any(
                units[unit_id].stratum_id != stratum.stratum_id
                for stratum in self.strata
                for unit_id in stratum.independent_unit_ids
            ):
                raise ValueError("nested controller use stratum assignments differ from physical units")
        if not 1 <= self.minimum_evaluable_units <= len(unit_ids):
            raise ValueError("invalid nested controller use minimum evaluable physical-unit count")
        for name, amount in (
            ("minimum_active_coverage", self.minimum_active_coverage),
            ("alpha", self.alpha),
            ("one_sided_critical_value", self.one_sided_critical_value),
        ):
            validate_decimal(amount, field_name=name, minimum=Decimal(0))
        if self.minimum_active_coverage > 1:
            raise ValueError("nested controller use active coverage cannot exceed one")
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("nested controller use alpha must lie strictly between zero and one")
        if self.one_sided_critical_value <= 0:
            raise ValueError("nested controller use critical value must be positive")
        if self.materiality.unit != self.effect_native_unit or self.materiality.value < 0:
            raise ValueError("nested controller use materiality uses another unit or is negative")
        if self.effect_quantity_id not in self.evaluator_boundary.raw_outcome_quantity_ids:
            raise ValueError("nested controller use effect quantity is absent from the reveal boundary")
        if self.terminal_matrix != NESTED_CONTROLLER_USE_TERMINAL_MATRIX:
            raise ValueError("nested controller use terminal matrix differs from frozen precedence")
        if not self.intent_to_treat:
            raise ValueError("nested controller use requires intent-to-treat physical-unit accounting")
        if self.evidence_ceiling is not EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("nested prospective evaluation must remain controller use")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("nested controller use plan must remain sealed before reveal")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible development cannot author nested controller use")


def decode_repeated_delivery_controller_evaluation_plan(
    payload: bytes,
) -> RepeatedDeliveryControllerEvaluationPlan:
    return decode_canonical_bytes(
        payload,
        RepeatedDeliveryControllerEvaluationPlan,
        maximum_bytes=MAX_NESTED_EVALUATION_PLAN_BYTES,
    )


class NestedPreparationRole(StrEnum):
    CONFIRMATORY_CONTROLLER_USE = "CONFIRMATORY_CONTROLLER_USE"


class NestedEvaluationBranch(StrEnum):
    CANONICAL_CONFORMANCE = "CANONICAL_CONFORMANCE"
    COMMITTED_ACTION = "COMMITTED_ACTION"
    QUALIFIED_HOLD_BASELINE = "QUALIFIED_HOLD_BASELINE"


@dataclass(frozen=True, slots=True)
class NestedTaskUnitSpec(CanonicalRecord):
    """One task row; nested preparations and members never add replication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-task-unit-spec'

    independent_unit_id: str
    task_id: str
    arm_id: str
    stratum_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
            ("stratum_id", self.stratum_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class NestedPreparationOccurrence(CanonicalRecord):
    """One fresh confirmatory preparation nested in its task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-preparation-occurrence'

    occurrence_id: str
    independent_unit_id: str
    task_id: str
    preparation_unit_id: str
    seed_id: str
    clone_group_id: str
    source_fingerprint: str
    role: NestedPreparationRole

    def __post_init__(self) -> None:
        for name, value in (
            ("occurrence_id", self.occurrence_id),
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("preparation_unit_id", self.preparation_unit_id),
            ("seed_id", self.seed_id),
            ("clone_group_id", self.clone_group_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.source_fingerprint, field_name="source_fingerprint")
        if self.role is not NestedPreparationRole.CONFIRMATORY_CONTROLLER_USE:
            raise ValueError("action-aware nested preparation must be confirmatory controller use")


@dataclass(frozen=True, slots=True)
class NestedEvaluationCellLocator(CanonicalRecord):
    """Logical branch alias onto one immutable physical rollout cell."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nested-evaluation-cell-locator'

    locator_id: str
    physical_cell_id: str
    branch: NestedEvaluationBranch
    independent_unit_id: str
    task_id: str
    occurrence_id: str
    action_word_id: str
    model_member_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("locator_id", self.locator_id),
            ("physical_cell_id", self.physical_cell_id),
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("occurrence_id", self.occurrence_id),
            ("action_word_id", self.action_word_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class PreparationMedianReducerRegistration(CanonicalRecord):
    """Frozen median-then-member-minimum reducer registration."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/preparation-median-reducer-registration'

    registration_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str
    repeat_reducer: NestedRepeatReducer
    member_reducer: NestedMemberReducer
    inference_method: NestedInferenceMethod
    reduction_order: tuple[str, ...]
    deterministic: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.repeat_reducer is not NestedRepeatReducer.MEDIAN:
            raise ValueError("action-aware nested repeat reducer must be MEDIAN")
        if self.member_reducer is not NestedMemberReducer.WORST_MEMBER:
            raise ValueError("action-aware nested member reducer must be WORST_MEMBER")
        if self.reduction_order != NESTED_PREPARATION_REDUCTION_ORDER:
            raise ValueError("action-aware nested reducer order differs from the frozen axis order")
        if not self.deterministic:
            raise ValueError("action-aware nested reducer must be deterministic")


@dataclass(frozen=True, slots=True)
class ActionAwareControllerEvaluationPlan(CanonicalRecord):
    "Controller-independent exact confirmatory topology for action-aware controller use."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/action-aware-controller-evaluation-plan'

    evaluation_plan_id: str
    evaluator_boundary: EvaluatorBoundarySpec
    finite_chart_reference_design: ObjectIdentity
    task_units: tuple[NestedTaskUnitSpec, ...]
    preparation_occurrences: tuple[NestedPreparationOccurrence, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    model_member_ids: tuple[str, ...]
    cell_locators: tuple[NestedEvaluationCellLocator, ...]
    reducer: PreparationMedianReducerRegistration
    measured_hold_word_id: str | None
    causal_cutoff_id: str
    reference_domain_id: str
    confirmatory_domain_id: str
    effect_quantity_id: str
    effect_native_unit: str
    favorable_direction: UtilityDirection
    minimum_evaluable_units: int
    materiality: NamedDecimal
    intent_to_treat: bool
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_plan_id", self.evaluation_plan_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("reference_domain_id", self.reference_domain_id),
            ("confirmatory_domain_id", self.confirmatory_domain_id),
            ("effect_quantity_id", self.effect_quantity_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.effect_native_unit, field_name="effect_native_unit")
        if self.reference_domain_id == self.confirmatory_domain_id:
            raise ValueError("reference and confirmatory causal domains must be disjoint")
        if self.finite_chart_reference_design.object_schema != (
            'empirical-lawhood/planning/finite-chart-reference-design'
        ):
            raise ValueError("action-aware nested requires a set-valued reference design")
        require_sorted_unique_ids(
            self.task_units,
            attribute="independent_unit_id",
            field_name="task_units",
        )
        require_sorted_unique_ids(
            self.preparation_occurrences,
            attribute="occurrence_id",
            field_name="preparation_occurrences",
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.cell_locators,
            attribute="locator_id",
            field_name="cell_locators",
        )
        if not self.task_units or not self.action_words:
            raise ValueError("action-aware nested requires task units and action words")
        if len({value.arm_id for value in self.task_units}) != 1:
            raise ValueError("one action-aware nested plan must belong to exactly one arm")
        units = {value.independent_unit_id: value for value in self.task_units}
        occurrences = {value.occurrence_id: value for value in self.preparation_occurrences}
        by_unit: dict[str, list[NestedPreparationOccurrence]] = {unit_id: [] for unit_id in units}
        for occurrence in self.preparation_occurrences:
            unit = units.get(occurrence.independent_unit_id)
            if unit is None or occurrence.task_id != unit.task_id:
                raise ValueError("nested preparation rewrites or omits its task unit")
            by_unit[occurrence.independent_unit_id].append(occurrence)
        if any(len(values) != 3 for values in by_unit.values()):
            raise ValueError("action-aware nested requires exactly three fresh preparations per task")
        if any(len({value.seed_id for value in values}) != 3 for values in by_unit.values()):
            raise ValueError("action-aware nested preparation seeds must be distinct within a task")
        if any(
            len({value.preparation_unit_id for value in values}) != 3 for values in by_unit.values()
        ):
            raise ValueError("action-aware nested preparations must be physically distinct")
        action_ids = {value.word_id for value in self.action_words}
        if self.measured_hold_word_id is not None:
            validate_stable_id(self.measured_hold_word_id, field_name="measured_hold_word_id")
            if self.measured_hold_word_id not in action_ids:
                raise ValueError("action-aware nested measured HOLD is outside the action chart")
        expected_physical = {
            (unit.independent_unit_id, occurrence.occurrence_id, action_id, member)
            for unit in self.task_units
            for occurrence in by_unit[unit.independent_unit_id]
            for action_id in action_ids
            for member in self.model_member_ids
        }
        expected_logical = {
            (branch, *coordinate)
            for branch in NestedEvaluationBranch
            for coordinate in expected_physical
        }
        observed_logical: set[tuple[object, ...]] = set()
        physical_semantics: dict[str, tuple[str, str, str, str]] = {}
        for locator in self.cell_locators:
            located_occurrence = occurrences.get(locator.occurrence_id)
            unit = units.get(locator.independent_unit_id)
            if (
                unit is None
                or located_occurrence is None
                or located_occurrence.independent_unit_id != locator.independent_unit_id
                or locator.task_id != unit.task_id
                or locator.action_word_id not in action_ids
                or locator.model_member_id not in self.model_member_ids
            ):
                raise ValueError("action-aware nested locator changes a frozen axis")
            coordinate = (
                locator.independent_unit_id,
                locator.occurrence_id,
                locator.action_word_id,
                locator.model_member_id,
            )
            observed_logical.add((locator.branch, *coordinate))
            prior = physical_semantics.setdefault(locator.physical_cell_id, coordinate)
            if prior != coordinate:
                raise ValueError("one physical cell is aliased to different scientific axes")
        if observed_logical != expected_logical or len(self.cell_locators) != len(expected_logical):
            raise ValueError("action-aware nested logical branch/action/member/seed grid is incomplete")
        if set(physical_semantics.values()) != expected_physical or len(physical_semantics) != len(
            expected_physical
        ):
            raise ValueError("action-aware nested physical action/member/seed union is incomplete")
        if not 1 <= self.minimum_evaluable_units <= len(self.task_units):
            raise ValueError("action-aware nested minimum evaluable task count is invalid")
        if self.materiality.unit != self.effect_native_unit or self.materiality.value < 0:
            raise ValueError("action-aware nested materiality uses another unit or is negative")
        if self.effect_quantity_id not in self.evaluator_boundary.raw_outcome_quantity_ids:
            raise ValueError("action-aware nested effect is absent from the evaluator boundary")
        if not self.intent_to_treat:
            raise ValueError("action-aware nested requires intent-to-treat task accounting")
        if self.evidence_ceiling is not EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("action-aware nested plan must remain controller use")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("action-aware nested plan must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("action-aware nested plan must remain prospective")


@dataclass(frozen=True, slots=True)
class ProspectiveEvaluationPrecommitment(CanonicalRecord):
    """Acyclic evaluator design frozen before programme authoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-evaluation-precommitment'

    precommitment_id: str
    issued_manifest: ObjectIdentity
    extension_set: ObjectIdentity
    sealed_acquisition_trace: ObjectIdentity
    nested_plan: ActionAwareControllerEvaluationPlan
    reference_design: ObjectIdentity
    evaluator_binding: ImplementationBinding
    logical_roster_rule_id: str
    causal_cutoff_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    confirmatory_domain_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.precommitment_id, field_name="precommitment_id")
        for name, value in (
            ("logical_roster_rule_id", self.logical_roster_rule_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("acquisition_domain_id", self.acquisition_domain_id),
            ("reference_domain_id", self.reference_domain_id),
            ("confirmatory_domain_id", self.confirmatory_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            len(
                {
                    self.acquisition_domain_id,
                    self.reference_domain_id,
                    self.confirmatory_domain_id,
                }
            )
            != 3
        ):
            raise ValueError("prospective evaluation causal domains must be disjoint")
        plan_identity = ObjectIdentity.from_record(
            self.nested_plan.evaluation_plan_id,
            self.nested_plan,
        )
        if (
            self.reference_design != self.nested_plan.finite_chart_reference_design
            or self.causal_cutoff_id != self.nested_plan.causal_cutoff_id
            or self.reference_domain_id != self.nested_plan.reference_domain_id
            or self.confirmatory_domain_id != self.nested_plan.confirmatory_domain_id
            or plan_identity.object_schema != ActionAwareControllerEvaluationPlan.SCHEMA
        ):
            raise ValueError("precommitment changes its nested/reference/causal design")
        if self.evaluator_binding.role is not ImplementationRole.OUTCOME_EVALUATOR:
            raise ValueError("precommitment evaluator binding has another role")
        reducer = self.nested_plan.reducer
        if (
            self.evaluator_binding.reference.capability_key != reducer.capability_key
            or self.evaluator_binding.reference.capability_version != reducer.capability_version
            or self.evaluator_binding.config_sha256 != reducer.config_sha256
            or self.evaluator_binding.implementation_sha256 != reducer.implementation_sha256
        ):
            raise ValueError("precommitment evaluator differs from the frozen reducer")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("prospective precommitment must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("prospective precommitment cannot be outcome-visible")


def decode_action_aware_controller_evaluation_plan(
    payload: bytes,
) -> ActionAwareControllerEvaluationPlan:
    return decode_canonical_bytes(
        payload,
        ActionAwareControllerEvaluationPlan,
        maximum_bytes=MAX_NESTED_EVALUATION_PLAN_BYTES,
    )


__all__ = [
    "MAX_NESTED_EVALUATION_PLAN_BYTES",
    'NESTED_CONTROLLER_USE_TERMINAL_MATRIX',
    "NESTED_REDUCTION_ORDER",
    'NESTED_PREPARATION_REDUCTION_ORDER',
    'NESTED_CONTROLLER_EVALUATION_BRANCHES',
    "NestedInferenceMethod",
    "NestedMemberReducer",
    "NestedPhysicalUnitSpec",
    'RepeatedDeliveryReducerRegistration',
    "NestedRepeatReducer",
    'RepeatedDeliveryControllerEvaluationPlan',
    "NestedReplicateSpec",
    'NestedPreparationRole',
    'NestedEvaluationBranch',
    'NestedTaskUnitSpec',
    'NestedPreparationOccurrence',
    'NestedEvaluationCellLocator',
    'PreparationMedianReducerRegistration',
    'ActionAwareControllerEvaluationPlan',
    'ProspectiveEvaluationPrecommitment',
    'decode_repeated_delivery_controller_evaluation_plan',
    'decode_action_aware_controller_evaluation_plan',
]


class PreparedFutureRole(StrEnum):
    AUDIT_PROBE = "AUDIT_PROBE"
    COMMITTED_TASK = "COMMITTED_TASK"
    MATCHED_HOLD = "MATCHED_HOLD"


@dataclass(frozen=True, slots=True)
class PreparedRootAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-root-assignment'

    root_id: str
    context_id: str
    problem_id: str
    initial_stream_id: str

    def __post_init__(self) -> None:
        for name in ("root_id", "context_id", "problem_id", "initial_stream_id"):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class PreparedPolicyAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-policy-assignment'

    policy_id: str
    parent_recipe: ObjectIdentity
    child_recipe: ObjectIdentity
    pre_parent_feature_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        require_sorted_unique_strings(
            self.pre_parent_feature_ids, field_name="pre_parent_feature_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class PreparedFutureSlot(CanonicalRecord):
    """One preassigned future; numerical views do not create another acquisition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-future-slot'

    slot_id: str
    root_id: str
    policy_id: str
    role: PreparedFutureRole
    ordinal: int
    acquisition_id: str
    stream_id: str
    assigned_probe_word: ObjectIdentity | None
    acquisition_precommitment: ObjectIdentity
    innovation_coupling: ObjectIdentity | None

    def __post_init__(self) -> None:
        for name in ("slot_id", "root_id", "policy_id", "acquisition_id", "stream_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not isinstance(self.role, PreparedFutureRole) or self.ordinal < 0:
            raise ValueError("prepared future requires a typed role and nonnegative ordinal")
        if self.role is PreparedFutureRole.COMMITTED_TASK:
            if self.ordinal != 0 or self.assigned_probe_word is not None:
                raise ValueError("task slot cannot preselect a late child action")
        elif (
            self.assigned_probe_word is None
            or self.assigned_probe_word.object_schema != OccurrenceActionWord.SCHEMA
        ):
            raise ValueError(
                "evaluator future must bind an exact native word before parent contact"
            )
        elif self.role is PreparedFutureRole.MATCHED_HOLD and self.ordinal != 0:
            raise ValueError("matched task HOLD must have ordinal zero")


@dataclass(frozen=True, slots=True)
class PreparedInterfaceReducerRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-interface-reducer-registration'

    registration_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str
    reduction_order: tuple[str, ...]
    statistics_specification: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.reduction_order != ("NUMERICAL_VIEW", "FUTURE_ROLE", "ROOT_POLICY", "ROOT"):
            raise ValueError("prepared controller use reducer changes the frozen root reduction order")


@dataclass(frozen=True, slots=True)
class PreparedReadoutTolerance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-readout-tolerance'

    tolerance_id: str
    coordinate: FiniteResponseCoordinate
    numerical_tolerance: Decimal
    adequacy_epsilon: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.tolerance_id, field_name="tolerance_id")
        validate_decimal(
            self.numerical_tolerance, field_name="numerical_tolerance", minimum=Decimal(0)
        )
        validate_decimal(self.adequacy_epsilon, field_name="adequacy_epsilon", minimum=Decimal(0))
        if self.numerical_tolerance <= 0 or self.adequacy_epsilon <= 0:
            raise ValueError("prepared readout tolerances must be positive in the native unit")


@dataclass(frozen=True, slots=True)
class CommonStartControllerEvaluationPlan(CanonicalRecord):
    """Common-start census without an unobserved full-action reference estimand."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/common-start-controller-evaluation-plan'

    evaluation_plan_id: str
    roots: tuple[PreparedRootAssignment, ...]
    policies: tuple[PreparedPolicyAssignment, ...]
    futures: tuple[PreparedFutureSlot, ...]
    numerical_view_ids: tuple[str, ...]
    readout_tolerances: tuple[PreparedReadoutTolerance, ...]
    audit_action_words: tuple[OccurrenceActionWord, ...]
    audit_probes_per_policy: int
    audit_distribution: ObjectIdentity | None
    matched_hold_word: OccurrenceActionWord
    target_family: ObjectIdentity
    parent_public_feature_ids: tuple[str, ...]
    late_target_feature_ids: tuple[str, ...]
    evaluator_boundary: EvaluatorBoundarySpec
    parent_preservation_predicate_ids: tuple[str, ...]
    future_preservation_predicate_ids: tuple[str, ...]
    reducer: PreparedInterfaceReducerRegistration
    causal_cutoff_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_plan_id, field_name="evaluation_plan_id")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        require_sorted_unique_ids(self.roots, attribute="root_id", field_name="roots")
        require_sorted_unique_ids(self.policies, attribute="policy_id", field_name="policies")
        require_sorted_unique_ids(self.futures, attribute="slot_id", field_name="futures")
        for name in ("numerical_view_ids", "parent_public_feature_ids", "late_target_feature_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.readout_tolerances, attribute="tolerance_id", field_name="readout_tolerances"
        )
        if not self.readout_tolerances or len(
            {t.coordinate.coordinate_id for t in self.readout_tolerances}
        ) != len(self.readout_tolerances):
            raise ValueError("prepared controller use requires one tolerance per exact native readout")
        if not self.roots or not self.policies or len(self.roots) > 4096 or len(self.policies) > 16:
            raise ValueError("prepared controller use requires a bounded nonempty root/policy census")
        if len({r.initial_stream_id for r in self.roots}) != len(self.roots):
            raise ValueError("distinct independent roots cannot reuse an initial stream")
        if not 0 <= self.audit_probes_per_policy <= 16:
            raise ValueError("prepared controller use probe roster is outside its bounded contract")
        if (self.audit_distribution is None) != (self.audit_probes_per_policy == 0):
            raise ValueError("probe distribution and positive probe count must appear together")
        allowed = set(self.parent_public_feature_ids)
        if allowed.intersection(self.late_target_feature_ids):
            raise ValueError("late target is available to the parent policy")
        if any(not set(p.pre_parent_feature_ids) <= allowed for p in self.policies):
            raise ValueError("parent policy reads outside the public pre-parent feature contract")
        for name in (
            "parent_preservation_predicate_ids",
            "future_preservation_predicate_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        parent_predicates = set(self.parent_preservation_predicate_ids)
        future_predicates = set(self.future_preservation_predicate_ids)
        boundary_predicates = {
            predicate.predicate_id for predicate in self.evaluator_boundary.outcome_predicates
        }
        if parent_predicates.intersection(future_predicates):
            raise ValueError("parent and future preservation predicates must be role-separated")
        if parent_predicates.union(future_predicates) != boundary_predicates:
            raise ValueError(
                "prepared preservation scopes must cover the exact evaluator predicates"
            )
        roots = {r.root_id: r for r in self.roots}
        policies = {p.policy_id: p for p in self.policies}
        expected = {
            (root, policy, role, ordinal)
            for root in roots
            for policy in policies
            for role, ordinals in (
                (PreparedFutureRole.COMMITTED_TASK, range(1)),
                (PreparedFutureRole.MATCHED_HOLD, range(1)),
                (PreparedFutureRole.AUDIT_PROBE, range(self.audit_probes_per_policy)),
            )
            for ordinal in ordinals
        }
        observed = {(f.root_id, f.policy_id, f.role, f.ordinal) for f in self.futures}
        if observed != expected or len(observed) != len(self.futures):
            raise ValueError("prepared controller use future census is incomplete, duplicated or expanded")
        require_sorted_unique_ids(
            self.audit_action_words, attribute="word_id", field_name="audit_action_words"
        )
        words = {ObjectIdentity.from_record(word.word_id, word) for word in self.audit_action_words}
        if (self.audit_probes_per_policy == 0) != (not self.audit_action_words):
            raise ValueError("audit word roster and probes must appear together")
        if any(
            f.assigned_probe_word not in words
            for f in self.futures
            if f.role is PreparedFutureRole.AUDIT_PROBE
        ):
            raise ValueError("audit assignment substitutes an unregistered native word")
        matched_hold_identity = ObjectIdentity.from_record(
            self.matched_hold_word.word_id, self.matched_hold_word
        )
        if (
            not self.matched_hold_word.occurrences
            or any(
                value.value != 0
                for occurrence in self.matched_hold_word.occurrences
                for value in (
                    occurrence.requested,
                    occurrence.accepted,
                    occurrence.applied,
                    occurrence.realized,
                )
            )
            or any(
                future.assigned_probe_word != (
                    self.matched_word_identity(future.root_id)
                    if isinstance(self, CoupledRealizationControllerEvaluationPlan)
                    else matched_hold_identity
                )
                for future in self.futures
                if future.role is PreparedFutureRole.MATCHED_HOLD
            )
        ):
            raise ValueError("matched task HOLD substitutes its frozen native HOLD word")
        for root_id in roots:
            for policy_id in policies:
                task = next(
                    future
                    for future in self.futures
                    if future.root_id == root_id
                    and future.policy_id == policy_id
                    and future.role is PreparedFutureRole.COMMITTED_TASK
                )
                matched = next(
                    future
                    for future in self.futures
                    if future.root_id == root_id
                    and future.policy_id == policy_id
                    and future.role is PreparedFutureRole.MATCHED_HOLD
                )
                if (
                    task.stream_id != matched.stream_id
                    or task.innovation_coupling is None
                    or task.innovation_coupling != matched.innovation_coupling
                ):
                    raise ValueError(
                        "task and matched HOLD must share their exact declared innovations"
                    )
        acquisitions: dict[str, tuple[object, ...]] = {}
        streams: dict[str, PreparedFutureSlot] = {}
        for future in self.futures:
            policy = policies[future.policy_id]
            meaning = (
                future.root_id,
                future.role,
                future.ordinal,
                future.stream_id,
                future.assigned_probe_word,
                future.acquisition_precommitment,
                policy.parent_recipe,
                policy.child_recipe,
            )
            if acquisitions.setdefault(future.acquisition_id, meaning) != meaning:
                raise ValueError("acquisition alias changes a precommitted future")
            key = future.stream_id
            previous = streams.setdefault(key, future)
            if previous.acquisition_id != future.acquisition_id and (
                future.innovation_coupling is None
                or previous.innovation_coupling != future.innovation_coupling
                or previous.root_id != future.root_id
                or (
                    previous.role is not future.role
                    and {previous.role, future.role}
                    != {
                        PreparedFutureRole.COMMITTED_TASK,
                        PreparedFutureRole.MATCHED_HOLD,
                    }
                )
            ) and not (
                isinstance(self, CoupledRealizationControllerEvaluationPlan)
                and self.allows_shared_realization(previous, future)
            ):
                raise ValueError(
                    "independent futures cannot reuse a native random stream without exact coupling"
                )
            if future.stream_id == roots[future.root_id].initial_stream_id and not (
                isinstance(self, CoupledRealizationControllerEvaluationPlan)
                and self.allows_shared_initial_stream(future)
            ):
                raise ValueError("post-handoff future reuses the initial-state stream")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("prepared controller use design must be outcome-sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("prepared controller use design must be prospective")
        if self.evidence_ceiling is not EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("prepared controller use design must retain its controller use ceiling")

    @property
    def primary_future_count(self) -> int:
        return len({f.acquisition_id for f in self.futures})

    @property
    def logical_view_count(self) -> int:
        return len(self.futures) * len(self.numerical_view_ids)


@dataclass(frozen=True, slots=True)
class PreparedNativeRealizationCoupling(CanonicalRecord):
    """One assigned native scenario shared by initial, task, reference and audits."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-native-realization-coupling'

    coupling_id: str
    root_id: str
    stream_id: str
    scenario: ObjectIdentity

    def __post_init__(self) -> None:
        for name in ("coupling_id", "root_id", "stream_id"):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class PreparedRootHoldWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-root-hold-word'

    root_id: str
    word: OccurrenceActionWord

    def __post_init__(self) -> None:
        validate_stable_id(self.root_id, field_name="root_id")
        if not self.word.occurrences or any(
            value.value != 0
            for occurrence in self.word.occurrences
            for value in (
                occurrence.requested, occurrence.accepted,
                occurrence.applied, occurrence.realized,
            )
        ):
            raise ValueError("root-specific reference is not a measured zero-feed word")


@dataclass(frozen=True, slots=True)
class CoupledRealizationControllerEvaluationPlan(CommonStartControllerEvaluationPlan):
    "Census with an explicit common native realization per assigned root."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coupled-realization-controller-evaluation-plan'

    realization_couplings: tuple[PreparedNativeRealizationCoupling, ...]
    root_hold_words: tuple[PreparedRootHoldWord, ...]

    def matched_word_identity(self, root_id: str) -> ObjectIdentity:
        value = next((item for item in self.root_hold_words if item.root_id == root_id), None)
        if value is None:
            raise ValueError("coupled realization controller evaluation lacks the root's exact zero-feed reference")
        return ObjectIdentity.from_record(value.word.word_id, value.word)

    def allows_shared_realization(
        self, left: PreparedFutureSlot, right: PreparedFutureSlot
    ) -> bool:
        coupling = next(
            (value for value in self.realization_couplings if value.root_id == left.root_id),
            None,
        )
        return bool(
            coupling is not None
            and left.root_id == right.root_id
            and left.stream_id == right.stream_id == coupling.stream_id
            and left.innovation_coupling
            == right.innovation_coupling
            == ObjectIdentity.from_record(coupling.coupling_id, coupling)
        )

    def allows_shared_initial_stream(self, future: PreparedFutureSlot) -> bool:
        coupling = next(
            (value for value in self.realization_couplings if value.root_id == future.root_id),
            None,
        )
        return bool(
            coupling is not None
            and future.stream_id == coupling.stream_id
            and future.innovation_coupling
            == ObjectIdentity.from_record(coupling.coupling_id, coupling)
        )

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.realization_couplings,
            attribute="root_id",
            field_name="realization_couplings",
        )
        require_sorted_unique_ids(
            self.root_hold_words, attribute="root_id", field_name="root_hold_words"
        )
        by_root = {value.root_id: value for value in self.realization_couplings}
        if (
            set(by_root) != {value.root_id for value in self.roots}
            or {value.root_id for value in self.root_hold_words} != set(by_root)
            or self.matched_hold_word != self.root_hold_words[0].word
        ) or any(
            root.initial_stream_id != by_root[root.root_id].stream_id
            for root in self.roots
        ) or any(
            not self.allows_shared_initial_stream(future)
            for future in self.futures
        ):
            raise ValueError("coupled realization controller evaluation changes its assigned shared native realization")
        CommonStartControllerEvaluationPlan.__post_init__(self)


def decode_coupled_realization_controller_evaluation_plan(
    payload: bytes,
) -> CoupledRealizationControllerEvaluationPlan:
    return decode_canonical_bytes(
        payload,
        CoupledRealizationControllerEvaluationPlan,
        maximum_bytes=MAX_NESTED_EVALUATION_PLAN_BYTES,
    )


def decode_common_start_controller_evaluation_plan(
    payload: bytes,
) -> CommonStartControllerEvaluationPlan:
    return decode_canonical_bytes(
        payload,
        CommonStartControllerEvaluationPlan,
        maximum_bytes=MAX_NESTED_EVALUATION_PLAN_BYTES,
    )


@dataclass(frozen=True, slots=True)
class PreparedInterfaceEvaluationSlice(CanonicalRecord):
    """Exact bounded root/policy slice of a separately custodied complete census."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-interface-evaluation-slice'

    slice_id: str
    source_plan: ObjectIdentity
    root: PreparedRootAssignment
    policy: PreparedPolicyAssignment
    futures: tuple[PreparedFutureSlot, ...]
    numerical_view_ids: tuple[str, ...]
    readout_tolerances: tuple[PreparedReadoutTolerance, ...]
    audit_probes_per_policy: int
    audit_distribution: ObjectIdentity | None
    matched_hold_word: OccurrenceActionWord
    target_family: ObjectIdentity
    evaluator_boundary: EvaluatorBoundarySpec
    parent_preservation_predicate_ids: tuple[str, ...]
    future_preservation_predicate_ids: tuple[str, ...]
    reducer: PreparedInterfaceReducerRegistration
    causal_cutoff_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.slice_id, field_name="slice_id")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        if self.source_plan.object_schema not in (
            CommonStartControllerEvaluationPlan.SCHEMA,
            CoupledRealizationControllerEvaluationPlan.SCHEMA,
        ):
            raise ValueError("prepared slice must bind the complete pre-parent census")
        require_sorted_unique_ids(self.futures, attribute="slot_id", field_name="futures")
        require_sorted_unique_strings(
            self.numerical_view_ids, field_name="numerical_view_ids", allow_empty=False
        )
        require_sorted_unique_ids(
            self.readout_tolerances, attribute="tolerance_id", field_name="readout_tolerances"
        )
        if not 0 <= self.audit_probes_per_policy <= 16 or (self.audit_distribution is None) != (
            self.audit_probes_per_policy == 0
        ):
            raise ValueError("prepared slice changes its bounded audit distribution")
        for name in (
            "parent_preservation_predicate_ids",
            "future_preservation_predicate_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        parent_predicates = set(self.parent_preservation_predicate_ids)
        future_predicates = set(self.future_preservation_predicate_ids)
        boundary_predicates = {
            predicate.predicate_id for predicate in self.evaluator_boundary.outcome_predicates
        }
        if (
            parent_predicates.intersection(future_predicates)
            or parent_predicates.union(future_predicates) != boundary_predicates
        ):
            raise ValueError("prepared slice changes the role-scoped preservation predicates")
        if not self.matched_hold_word.occurrences or any(
            value.value != 0
            for occurrence in self.matched_hold_word.occurrences
            for value in (
                occurrence.requested,
                occurrence.accepted,
                occurrence.applied,
                occurrence.realized,
            )
        ):
            raise ValueError("prepared slice loses its exact matched task HOLD word")
        expected = {
            (PreparedFutureRole.COMMITTED_TASK, 0),
            (PreparedFutureRole.MATCHED_HOLD, 0),
        } | {
            (PreparedFutureRole.AUDIT_PROBE, index) for index in range(self.audit_probes_per_policy)
        }
        if (
            {(future.role, future.ordinal) for future in self.futures} != expected
            or len(self.futures) != len(expected)
            or any(
                future.root_id != self.root.root_id or future.policy_id != self.policy.policy_id
                for future in self.futures
            )
        ):
            raise ValueError("prepared slice drops or substitutes an assigned future")

    @property
    def identity(self) -> ObjectIdentity:
        return self.source_plan

    @property
    def evaluation_plan_id(self) -> str:
        return self.source_plan.object_id

    @property
    def roots(self) -> tuple[PreparedRootAssignment, ...]:
        return (self.root,)

    @property
    def policies(self) -> tuple[PreparedPolicyAssignment, ...]:
        return (self.policy,)


def prepared_interface_evaluation_slice(
    plan: CommonStartControllerEvaluationPlan,
    *,
    root_id: str,
    policy_id: str,
) -> PreparedInterfaceEvaluationSlice:
    root = next((root for root in plan.roots if root.root_id == root_id), None)
    policy = next((policy for policy in plan.policies if policy.policy_id == policy_id), None)
    if root is None or policy is None:
        raise ValueError("prepared slice requires an assigned root and policy")
    return PreparedInterfaceEvaluationSlice(
        slice_id=f"prepared-slice.{root_id}.{policy_id}",
        source_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
        root=root,
        policy=policy,
        futures=tuple(
            future
            for future in plan.futures
            if future.root_id == root_id and future.policy_id == policy_id
        ),
        numerical_view_ids=plan.numerical_view_ids,
        readout_tolerances=plan.readout_tolerances,
        audit_probes_per_policy=plan.audit_probes_per_policy,
        audit_distribution=plan.audit_distribution,
        matched_hold_word=(
            next(value.word for value in plan.root_hold_words if value.root_id == root_id)
            if isinstance(plan, CoupledRealizationControllerEvaluationPlan)
            else plan.matched_hold_word
        ),
        target_family=plan.target_family,
        evaluator_boundary=plan.evaluator_boundary,
        parent_preservation_predicate_ids=plan.parent_preservation_predicate_ids,
        future_preservation_predicate_ids=plan.future_preservation_predicate_ids,
        reducer=plan.reducer,
        causal_cutoff_id=plan.causal_cutoff_id,
    )
