"""Canonical interchange records for the history budget phase diagram history-budget phase diagram.

These records are the deliberately small shared surface between the independent
observer and generator implementations.  They contain no solver, receiver,
targeting or adjudication implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


class HistoryBudgetPhaseDiagramPhase(StrEnum):
    NOMINATION = "NOMINATION"
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class HistoryBudgetPhaseDiagramDisorderFamily(StrEnum):
    SMOOTH_PERIODIC_12 = "smooth-periodic-12"
    CORRELATED_FIELD_12 = "correlated-field-12"
    MULTISCALE_WAVELET_12 = "multiscale-wavelet-12"


class HistoryBudgetPhaseDiagramGate(StrEnum):
    DYNAMICAL = "DYNAMICAL"
    TARGET = "TARGET"
    SINK = "SINK"
    RANDOM = "RANDOM"


class HistoryBudgetPhaseDiagramPairKind(StrEnum):
    DYNAMICAL_BOUNDARY = "DYNAMICAL_BOUNDARY"
    TARGET_BOUNDARY = "TARGET_BOUNDARY"
    SINK_BOUNDARY = "SINK_BOUNDARY"
    RANDOM_FIBRE = "RANDOM_FIBRE"


class HistoryBudgetPhaseDiagramCohort(StrEnum):
    TARGETED = "TARGETED"
    UNTOUCHED = "UNTOUCHED"


class HistoryBudgetPhaseDiagramCoordinateKind(StrEnum):
    ABSOLUTE_DEPTH = "ABSOLUTE_DEPTH"
    NORMALIZED_BUDGET = "NORMALIZED_BUDGET"


class HistoryBudgetPhaseDiagramRankCompatibility(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    SAMPLING_ALIAS = "SAMPLING_ALIAS"
    NUMERICAL_CANCELLATION = "NUMERICAL_CANCELLATION"
    RANK_UNRESOLVED = "RANK_UNRESOLVED"


class HistoryBudgetPhaseDiagramDecisionDisposition(StrEnum):
    CORRECT_ACTION = "CORRECT_ACTION"
    CORRECT_HOLD = "CORRECT_HOLD"
    UNSAFE_FALSE_PROMOTION = "UNSAFE_FALSE_PROMOTION"
    FALSE_HOLD = "FALSE_HOLD"
    INVALID = "INVALID"


class HistoryBudgetPhaseDiagramOptimizationState(StrEnum):
    WITNESS = "WITNESS"
    BELOW_MARGIN = "BELOW_MARGIN"
    INFEASIBLE = "INFEASIBLE"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"


class HistoryBudgetPhaseDiagramScientificState(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    INFORMATIVE_NONADVERSE = "INFORMATIVE_NONADVERSE"
    TARGETABILITY_LIMITED = "TARGETABILITY_LIMITED"
    NO_COLLISION_ENCOUNTER_WITH_512_PREPARATIONS = "NO_COLLISION_ENCOUNTER_WITH_512_PREPARATIONS"
    MIXED = "MIXED"
    NONMONOTONE_PHASE_PATTERN = "NONMONOTONE_PHASE_PATTERN"
    LOW_RATE_BOUNDED = "LOW_RATE_BOUNDED"
    MAJORITY_RECURS = "MAJORITY_RECURS"
    ABSOLUTE_DEPTH_ALIGNED = "ABSOLUTE_DEPTH_ALIGNED"
    NORMALIZED_BUDGET_ALIGNED = "NORMALIZED_BUDGET_ALIGNED"
    NOT_DISTINGUISHED = "NOT_DISTINGUISHED"
    RESOLUTION_STABLE = "RESOLUTION_STABLE"
    RESOLUTION_SHIFTED = "RESOLUTION_SHIFTED"
    TARGETABILITY_SHIFTED = "TARGETABILITY_SHIFTED"
    UNSAFE_FALSE_PROMOTION = "UNSAFE_FALSE_PROMOTION"
    FALSE_HOLD = "FALSE_HOLD"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"
    RIGHT_CENSORED = "RIGHT_CENSORED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class HistoryBudgetPhaseDiagramTerminal(StrEnum):
    HISTORY_BUDGET_PHASE_DIAGRAM_EVALUATED = "HISTORY_BUDGET_PHASE_DIAGRAM_EVALUATED"
    IMPLEMENTATION_STOP = "IMPLEMENTATION_STOP"
    DEVELOPMENT_STOP = "DEVELOPMENT_STOP"
    OPERATIONAL_STOP = "OPERATIONAL_STOP"


def _sorted_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{field_name} must be sorted and unique")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    validate_decimal(value, field_name=field_name, minimum=Decimal(0))
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramCoordinateLabel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-coordinate-label'

    coordinate_id: str
    kind: HistoryBudgetPhaseDiagramCoordinateKind
    scale_cells: int
    depth: int
    budget: Decimal | None
    resolution_epsilon: Decimal
    primary: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.scale_cells not in {16, 32, 64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("history budget phase diagram coordinate scale/depth differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if self.kind is HistoryBudgetPhaseDiagramCoordinateKind.ABSOLUTE_DEPTH:
            if self.budget is not None:
                raise ValueError("absolute-depth coordinate cannot carry a budget")
        else:
            if self.budget is None:
                raise ValueError("normalized-budget coordinate requires a budget")
            _positive_decimal(self.budget, field_name="budget")
            if Decimal(8 * (self.depth + 1)) / Decimal(self.scale_cells) != self.budget:
                raise ValueError("normalized-budget coordinate mapping differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramStructuralRankStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-structural-rank-step'

    coordinate_id: str
    structural_rank: int
    row_count: int
    state_count: int
    support_sha256: str
    algorithm_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.algorithm_id, field_name="algorithm_id")
        validate_sha256(self.support_sha256, field_name="support_sha256")
        if not 0 <= self.structural_rank <= min(self.row_count, self.state_count):
            raise ValueError("history budget phase diagram structural rank exceeds its matrix dimensions")
        if self.row_count <= 0 or self.state_count <= 0:
            raise ValueError("history budget phase diagram structural rank dimensions must be positive")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramDiscreteRankBracket(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-discrete-rank-bracket'

    coordinate_id: str
    lower_rank: int
    upper_rank: int
    state_count: int
    precision_bits: int
    residual_norm: Decimal
    separation_ratio: Decimal | None
    matrix_sha256: str
    compatibility: HistoryBudgetPhaseDiagramRankCompatibility
    algorithm_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.algorithm_id, field_name="algorithm_id")
        validate_sha256(self.matrix_sha256, field_name="matrix_sha256")
        if not 0 <= self.lower_rank <= self.upper_rank <= self.state_count:
            raise ValueError("history budget phase diagram discrete-rank bracket is invalid")
        if self.precision_bits < 256:
            raise ValueError("history budget phase diagram discrete rank requires at least 256 bits")
        validate_decimal(self.residual_norm, field_name="residual_norm", minimum=Decimal(0))
        if self.separation_ratio is not None:
            _positive_decimal(self.separation_ratio, field_name="separation_ratio")
        if (
            self.compatibility is not HistoryBudgetPhaseDiagramRankCompatibility.RANK_UNRESOLVED
            and self.lower_rank != self.upper_rank
        ):
            raise ValueError("resolved history budget phase diagram rank requires a singleton bracket")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramConditioningStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-conditioning-step'

    coordinate_id: str
    resolution_epsilon: Decimal
    largest_singular_value: Decimal
    smallest_singular_value: Decimal
    condition_number: Decimal | None
    stable_rank: Decimal
    effective_rank: int
    right_censored: bool
    algorithm_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.algorithm_id, field_name="algorithm_id")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        _positive_decimal(self.largest_singular_value, field_name="largest_singular_value")
        validate_decimal(
            self.smallest_singular_value,
            field_name="smallest_singular_value",
            minimum=Decimal(0),
        )
        if self.condition_number is not None:
            _positive_decimal(self.condition_number, field_name="condition_number")
        validate_decimal(self.stable_rank, field_name="stable_rank", minimum=Decimal(0))
        if self.effective_rank < 0:
            raise ValueError("history budget phase diagram effective rank cannot be negative")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramPreparationMapManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-preparation-map-manifest'

    manifest_id: str
    unit_id: str
    preparation_count: int
    fine_scale_cells: int
    coarse_scale_cells: tuple[int, ...]
    fine_states_sha256: str
    mapped_states_sha256: str
    collision_edges_sha256: str
    prefix_counts: tuple[int, ...]
    charge_preservation_max_error: Decimal
    outcome_count_at_freeze: int
    algorithm_id: str

    def __post_init__(self) -> None:
        for name in ("manifest_id", "unit_id", "algorithm_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "fine_states_sha256",
            "mapped_states_sha256",
            "collision_edges_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.preparation_count != 512
            or self.fine_scale_cells != 256
            or self.coarse_scale_cells != (64, 128)
            or self.prefix_counts != (64, 128, 256)
        ):
            raise ValueError("history budget phase diagram preparation-map roster differs")
        validate_decimal(
            self.charge_preservation_max_error,
            field_name="charge_preservation_max_error",
            minimum=Decimal(0),
        )
        if self.outcome_count_at_freeze != 0:
            raise ValueError("history budget phase diagram preparation map must precede outcomes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramOptimizationCertificate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-optimization-certificate'

    certificate_id: str
    coordinate_id: str
    gate: HistoryBudgetPhaseDiagramGate
    action_id: str | None
    state: HistoryBudgetPhaseDiagramOptimizationState
    objective_lower_bound: Decimal
    objective_upper_bound: Decimal
    required_margin: Decimal
    primal_residual: Decimal
    duality_gap: Decimal
    solver_id: str
    complete_family_certificate: bool
    outcome_count_at_certificate: int

    def __post_init__(self) -> None:
        for name in ("certificate_id", "coordinate_id", "solver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_id is not None:
            validate_stable_id(self.action_id, field_name="action_id")
        for name in (
            "objective_lower_bound",
            "objective_upper_bound",
            "required_margin",
            "primal_residual",
            "duality_gap",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.objective_lower_bound > self.objective_upper_bound:
            raise ValueError("history budget phase diagram optimization bounds are reversed")
        if self.state is HistoryBudgetPhaseDiagramOptimizationState.WITNESS and (
            self.objective_lower_bound < self.required_margin
        ):
            raise ValueError("history budget phase diagram witness does not reach the required margin")
        if self.state is HistoryBudgetPhaseDiagramOptimizationState.BELOW_MARGIN and (
            self.objective_upper_bound >= self.required_margin
        ):
            raise ValueError("history budget phase diagram below-margin certificate does not exclude the margin")
        if (
            self.state
            in {
                HistoryBudgetPhaseDiagramOptimizationState.WITNESS,
                HistoryBudgetPhaseDiagramOptimizationState.BELOW_MARGIN,
                HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE,
            }
            and not self.complete_family_certificate
        ):
            raise ValueError("resolved history budget phase diagram optimization must be complete")
        if self.outcome_count_at_certificate != 0:
            raise ValueError("history budget phase diagram optimization certificate cannot observe outcomes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramConfig(CanonicalRecord):
    """Strict phase configuration shared by authoring and scientific code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-config'

    config_id: str
    phase: HistoryBudgetPhaseDiagramPhase
    plan_id: str
    parent_result_id: str
    scale_cells: tuple[int, ...]
    excluded_resource_canary_scale_cells: tuple[int, ...]
    disorder_families: tuple[HistoryBudgetPhaseDiagramDisorderFamily, ...]
    unit_ids: tuple[str, ...]
    seed_roster_commitment_sha256: str | None
    receiver_bin_count: int
    history_max_depth: int
    history_lag_t_star: Decimal
    rank_relative_threshold: Decimal
    coordinate_collision_epsilon: Decimal
    future_divergence_epsilon: Decimal
    generator_observer_metric_tolerance: Decimal
    optimization_tolerance: Decimal
    midpoint_boundary_tie_epsilon: Decimal
    hidden_amplitude_max_volts: Decimal
    hidden_voltage_envelope: tuple[Decimal, Decimal]
    hidden_voltage_interior_margin: Decimal
    target_pairs_per_depth: int
    sink_pairs_per_depth: int
    random_pairs_per_depth: int
    action_amplitudes_u_star: tuple[Decimal, ...]
    action_durations_t_star: tuple[Decimal, ...]
    panel_endpoint_t_star: Decimal
    diagnostic_times_t_star: tuple[Decimal, ...]
    target_charge_minimum_q_star: Decimal
    sink_voltage_maximum_v_star: Decimal
    absolute_depths: tuple[int, ...]
    normalized_history_budgets: tuple[Decimal, ...]
    primary_absolute_depth_contrast: tuple[int, int]
    primary_normalized_budget_contrast: tuple[Decimal, Decimal]
    receiver_resolution_panel: tuple[Decimal, ...]
    resolution_panel_depths: tuple[int, ...]
    target_decision_margin: Decimal
    sink_decision_margin: Decimal
    untouched_preparation_count: int
    untouched_prefix_counts: tuple[int, ...]
    untouched_distribution_id: str
    untouched_scale_map_id: str
    untouched_earliest_lag: int
    untouched_low_rate_threshold: Decimal
    bootstrap_resamples: int
    bootstrap_seed: int
    untouched_bootstrap_seed: int
    alignment_bootstrap_seed: int
    simultaneous_alpha: Decimal
    majority_opposition_threshold: Decimal
    descriptor_algorithm_id: str
    history_rank_algorithm_id: str
    targeting_tie_break_rule_id: str
    random_comparator_algorithm_id: str
    inference_algorithm_id: str
    structural_rank_algorithm_id: str
    discrete_rank_algorithm_id: str
    conditioning_algorithm_id: str
    preparation_sampler_algorithm_id: str
    alignment_algorithm_id: str
    false_promotion_rule_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_execution_authorized: bool
    requests_controller: bool
    nonactuating: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "parent_result_id",
            "descriptor_algorithm_id",
            "history_rank_algorithm_id",
            "targeting_tie_break_rule_id",
            "random_comparator_algorithm_id",
            "inference_algorithm_id",
            "structural_rank_algorithm_id",
            "discrete_rank_algorithm_id",
            "conditioning_algorithm_id",
            "preparation_sampler_algorithm_id",
            "alignment_algorithm_id",
            "false_promotion_rule_id",
            "untouched_distribution_id",
            "untouched_scale_map_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.plan_id != "history-budget-phase-diagram":
            raise ValueError("history budget phase diagram config binds the wrong plan")
        if self.parent_result_id != "simulator-morphism-challenges-result":
            raise ValueError("history budget phase diagram config binds the wrong parent result")
        expected_scales = {
            HistoryBudgetPhaseDiagramPhase.NOMINATION: (64, 128, 256),
            HistoryBudgetPhaseDiagramPhase.CANARY: (16, 32),
            HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: (64, 128, 256),
            HistoryBudgetPhaseDiagramPhase.EVALUATION: (64, 128, 256),
        }[self.phase]
        if self.scale_cells != expected_scales:
            raise ValueError("history budget phase diagram phase scale roster differs")
        expected_resource_scales = (64, 128, 256) if self.phase is HistoryBudgetPhaseDiagramPhase.CANARY else ()
        if self.excluded_resource_canary_scale_cells != expected_resource_scales:
            raise ValueError("history budget phase diagram excluded resource-canary scale roster differs")
        if self.disorder_families != tuple(HistoryBudgetPhaseDiagramDisorderFamily):
            raise ValueError("history budget phase diagram disorder-family roster differs")
        _sorted_unique(self.unit_ids, field_name="unit_ids")
        expected_units = {
            HistoryBudgetPhaseDiagramPhase.NOMINATION: 90,
            HistoryBudgetPhaseDiagramPhase.CANARY: 2,
            HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: 12,
            HistoryBudgetPhaseDiagramPhase.EVALUATION: 90,
        }[self.phase]
        if len(self.unit_ids) != expected_units:
            raise ValueError("history budget phase diagram phase complete-unit count differs")
        if self.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
            if self.seed_roster_commitment_sha256 is None:
                raise ValueError("evaluation config requires the pre-development seed commitment")
            validate_sha256(
                self.seed_roster_commitment_sha256,
                field_name="seed_roster_commitment_sha256",
            )
        elif self.seed_roster_commitment_sha256 is not None:
            raise ValueError("only the evaluation config may bind the seed commitment")
        if self.receiver_bin_count != 8 or self.history_max_depth != 31:
            raise ValueError("history budget phase diagram receiver/history roster differs")
        for name in (
            "history_lag_t_star",
            "rank_relative_threshold",
            "coordinate_collision_epsilon",
            "future_divergence_epsilon",
            "generator_observer_metric_tolerance",
            "optimization_tolerance",
            "midpoint_boundary_tie_epsilon",
            "hidden_amplitude_max_volts",
            "hidden_voltage_interior_margin",
            "panel_endpoint_t_star",
            "target_charge_minimum_q_star",
            "sink_voltage_maximum_v_star",
            "target_decision_margin",
            "sink_decision_margin",
            "simultaneous_alpha",
            "majority_opposition_threshold",
        ):
            _positive_decimal(getattr(self, name), field_name=name)
        if self.rank_relative_threshold != Decimal("1e-12"):
            raise ValueError("history budget phase diagram declared continuity threshold differs")
        if self.coordinate_collision_epsilon != Decimal(
            "0.005"
        ) or self.future_divergence_epsilon != Decimal("0.005"):
            raise ValueError("history budget phase diagram collision/divergence margins differ")
        if self.target_decision_margin != Decimal("0.005") or self.sink_decision_margin != Decimal(
            "0.005"
        ):
            raise ValueError("history budget phase diagram target/sink decision margins differ")
        if self.generator_observer_metric_tolerance != Decimal("1e-10"):
            raise ValueError("history budget phase diagram generator/observer metric tolerance differs")
        if self.optimization_tolerance != Decimal("1e-10"):
            raise ValueError("history budget phase diagram optimization tolerance differs")
        if self.midpoint_boundary_tie_epsilon != Decimal("1e-10"):
            raise ValueError("history budget phase diagram midpoint boundary tie convention differs")
        if self.hidden_voltage_envelope != (Decimal("0"), Decimal("1")):
            raise ValueError("history budget phase diagram hidden-state envelope differs")
        if self.hidden_voltage_interior_margin != Decimal("1e-8"):
            raise ValueError("history budget phase diagram hidden-state interior margin differs")
        if (
            self.target_pairs_per_depth != 2
            or self.sink_pairs_per_depth != 2
            or self.random_pairs_per_depth != 4
        ):
            raise ValueError("history budget phase diagram targeting roster differs")
        if self.absolute_depths != (0, 2, 4, 6, 8, 12):
            raise ValueError("history budget phase diagram absolute-depth roster differs")
        if self.normalized_history_budgets != (
            Decimal("0.125"),
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("1"),
        ):
            raise ValueError("history budget phase diagram normalized-budget roster differs")
        if self.primary_absolute_depth_contrast != (
            4,
            8,
        ) or self.primary_normalized_budget_contrast != (
            Decimal("0.5"),
            Decimal("1"),
        ):
            raise ValueError("history budget phase diagram primary contrasts differ")
        if self.receiver_resolution_panel != (
            Decimal("0.0025"),
            Decimal("0.005"),
            Decimal("0.01"),
        ) or self.resolution_panel_depths != (4, 6, 8):
            raise ValueError("history budget phase diagram resolution panel differs")
        if self.untouched_preparation_count != 512 or self.untouched_prefix_counts != (
            64,
            128,
            256,
        ):
            raise ValueError("history budget phase diagram untouched-preparation roster differs")
        if (
            self.untouched_distribution_id != "shifted-scaled-beta-2-2"
            or self.untouched_scale_map_id != "dyadic-charge-preserving-block-mean"
            or self.untouched_earliest_lag != 31
            or self.untouched_low_rate_threshold != Decimal("0.20")
        ):
            raise ValueError("history budget phase diagram untouched-preparation semantics differ")
        for name in ("action_amplitudes_u_star", "action_durations_t_star"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or not values:
                raise ValueError(f"{name} must be nonempty, sorted and unique")
            for index, value in enumerate(values):
                _positive_decimal(value, field_name=f"{name}[{index}]")
        if len(self.action_amplitudes_u_star) != 5 or len(self.action_durations_t_star) != 5:
            raise ValueError("history budget phase diagram requires the complete 5 x 5 action panel")
        if self.panel_endpoint_t_star != Decimal("0.20"):
            raise ValueError("history budget phase diagram panel endpoint differs")
        if self.diagnostic_times_t_star != (
            Decimal("0.005"),
            Decimal("0.01"),
            Decimal("0.02"),
        ):
            raise ValueError("history budget phase diagram diagnostic-time roster differs")
        if (
            self.bootstrap_resamples != 10_000
            or self.bootstrap_seed != 3_140_159
            or self.untouched_bootstrap_seed != 3_140_160
            or self.alignment_bootstrap_seed != 3_140_161
        ):
            raise ValueError("history budget phase diagram bootstrap contract differs")
        if not Decimal(0) < self.simultaneous_alpha < Decimal(1):
            raise ValueError("history budget phase diagram simultaneous alpha is invalid")
        if self.majority_opposition_threshold != Decimal("0.5"):
            raise ValueError("history budget phase diagram majority-opposition threshold differs")
        if (
            self.descriptor_algorithm_id != "history-budget-phase-diagram-latent-field-centre-edge"
            or self.history_rank_algorithm_id != "history-budget-phase-diagram-fixed-k-b-history"
            or self.targeting_tie_break_rule_id != "history-budget-phase-diagram-stable-lexical-boundary"
            or self.random_comparator_algorithm_id != "history-budget-phase-diagram-outcome-blind-fibre-pcg64"
            or self.inference_algorithm_id != "history-budget-phase-diagram-phase-diagram-complete-unit"
            or self.structural_rank_algorithm_id != "history-budget-phase-diagram-observability-jet-matching"
            or self.discrete_rank_algorithm_id != "arb-inertia-exact-dyadic-v2"
            or self.conditioning_algorithm_id != "history-budget-phase-diagram-absolute-resolution-svd"
            or self.preparation_sampler_algorithm_id != "history-budget-phase-diagram-scale-coupled-untouched-pcg64"
            or self.alignment_algorithm_id != "history-budget-phase-diagram-paired-k-b-alignment"
            or self.false_promotion_rule_id != "history-budget-phase-diagram-unsafe-first-lexicographic"
        ):
            raise ValueError("history budget phase diagram algorithm identity differs")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("history budget phase diagram evidence ceiling differs")
        expected_access = {
            HistoryBudgetPhaseDiagramPhase.NOMINATION: OutcomeAccess.OUTCOME_BLIND,
            HistoryBudgetPhaseDiagramPhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            HistoryBudgetPhaseDiagramPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        expected_visibility = {
            HistoryBudgetPhaseDiagramPhase.NOMINATION: VisibilityCeiling.PROSPECTIVE,
            HistoryBudgetPhaseDiagramPhase.CANARY: VisibilityCeiling.DEVELOPMENT_ONLY,
            HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: VisibilityCeiling.DEVELOPMENT_ONLY,
            HistoryBudgetPhaseDiagramPhase.EVALUATION: VisibilityCeiling.PROSPECTIVE,
        }[self.phase]
        if (
            self.outcome_access is not expected_access
            or self.visibility_ceiling is not expected_visibility
        ):
            raise ValueError("history budget phase diagram phase outcome/visibility contract differs")
        if self.physical_execution_authorized or self.requests_controller or not self.nonactuating:
            raise ValueError("history budget phase diagram must remain nonactuating simulator-only work")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramSeedRosterCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-seed-roster-commitment'

    roster_id: str
    evaluation_unit_ids: tuple[str, ...]
    seed_payload_sha256: str
    commitment_sha256: str
    seed_bytes_per_unit: int
    public_seed_count: int
    outcome_count_at_commitment: int

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        _sorted_unique(self.evaluation_unit_ids, field_name="evaluation_unit_ids")
        if len(self.evaluation_unit_ids) != 90:
            raise ValueError("history budget phase diagram evaluation roster must contain 90 complete units")
        validate_sha256(self.seed_payload_sha256, field_name="seed_payload_sha256")
        validate_sha256(self.commitment_sha256, field_name="commitment_sha256")
        if self.seed_bytes_per_unit != 32 or self.public_seed_count != 0:
            raise ValueError("history budget phase diagram commitment must not disclose evaluation seeds")
        if self.outcome_count_at_commitment != 0:
            raise ValueError("history budget phase diagram roster commitment must precede outcomes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramDenominatorDescriptor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-denominator-descriptor'

    descriptor_id: str
    unit_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    seed_sha256: str
    latent_field_sha256: str
    capacitances_farads: tuple[Decimal, ...]
    interior_resistances_ohms: tuple[Decimal, ...]
    left_source_resistance_ohms: Decimal
    right_termination_resistance_ohms: Decimal
    resistance_bar_ohms: Decimal
    capacitance_bar_farads: Decimal
    voltage_reference_volts: Decimal
    time_scale_seconds: Decimal
    component_variation_fraction: Decimal
    dtype_id: str

    def __post_init__(self) -> None:
        for name in ("descriptor_id", "unit_id", "dtype_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.seed_sha256, field_name="seed_sha256")
        validate_sha256(self.latent_field_sha256, field_name="latent_field_sha256")
        if self.scale_cells not in {16, 32, 64, 128, 256}:
            raise ValueError("history budget phase diagram descriptor scale differs")
        if len(self.capacitances_farads) != self.scale_cells:
            raise ValueError("history budget phase diagram capacitance roster differs")
        if len(self.interior_resistances_ohms) != self.scale_cells - 1:
            raise ValueError("history budget phase diagram resistance roster differs")
        for name, values in (
            ("capacitances_farads", self.capacitances_farads),
            ("interior_resistances_ohms", self.interior_resistances_ohms),
        ):
            for index, value in enumerate(values):
                _positive_decimal(value, field_name=f"{name}[{index}]")
        for name in (
            "left_source_resistance_ohms",
            "right_termination_resistance_ohms",
            "resistance_bar_ohms",
            "capacitance_bar_farads",
            "voltage_reference_volts",
            "time_scale_seconds",
            "component_variation_fraction",
        ):
            _positive_decimal(getattr(self, name), field_name=name)
        if self.component_variation_fraction != Decimal("0.12") or self.dtype_id != "float64":
            raise ValueError("history budget phase diagram descriptor numerical contract differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramRankStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-rank-step'

    depth: int
    effective_rank: int
    algebraic_comparator_rank: int
    row_count_ceiling: int
    largest_singular_value: Decimal
    smallest_retained_relative_singular_value: Decimal
    condition_number: Decimal | None
    nonmonotone_effective_rank: bool

    def __post_init__(self) -> None:
        if not 0 <= self.depth <= 31:
            raise ValueError("history budget phase diagram history depth differs")
        if not 0 <= self.effective_rank <= self.row_count_ceiling:
            raise ValueError("history budget phase diagram effective rank exceeds its ceiling")
        if not 0 <= self.algebraic_comparator_rank <= self.row_count_ceiling:
            raise ValueError("history budget phase diagram algebraic rank exceeds its ceiling")
        _positive_decimal(self.largest_singular_value, field_name="largest_singular_value")
        validate_decimal(
            self.smallest_retained_relative_singular_value,
            field_name="smallest_retained_relative_singular_value",
            minimum=Decimal(0),
        )
        if self.condition_number is not None:
            _positive_decimal(self.condition_number, field_name="condition_number")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramHistoryRankForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-history-rank-forecast'

    forecast_id: str
    descriptor_sha256: str
    rank_steps: tuple[HistoryBudgetPhaseDiagramRankStep, ...]
    singular_spectra_sha256: str
    k_full_effective: int | None
    effective_rank_right_censored: bool
    probe_depths: tuple[int, ...]
    algebraic_prediction_state: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        validate_sha256(self.descriptor_sha256, field_name="descriptor_sha256")
        validate_sha256(self.singular_spectra_sha256, field_name="singular_spectra_sha256")
        if tuple(value.depth for value in self.rank_steps) != tuple(range(32)):
            raise ValueError("history budget phase diagram rank forecast must contain depths 0..31")
        if self.k_full_effective is None:
            if not self.effective_rank_right_censored:
                raise ValueError("missing full-rank depth must be right-censored")
        elif not 0 <= self.k_full_effective <= 31 or self.effective_rank_right_censored:
            raise ValueError("history budget phase diagram full-rank depth/censoring differs")
        if self.probe_depths != tuple(sorted(set(self.probe_depths))):
            raise ValueError("history budget phase diagram probe depths must be sorted and unique")
        if self.algebraic_prediction_state not in {
            HistoryBudgetPhaseDiagramScientificState.SUPPORTED,
            HistoryBudgetPhaseDiagramScientificState.OPPOSED,
        }:
            raise ValueError("history budget phase diagram algebraic prediction state differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramActionPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-action-prediction'

    action_id: str
    amplitude_u_star: Decimal
    duration_t_star: Decimal
    requested_action_id: str
    predicted_plus_q_star: Decimal
    predicted_minus_q_star: Decimal
    predicted_plus_e_star: Decimal
    predicted_minus_e_star: Decimal
    predicted_plus_v_max_star: Decimal
    predicted_minus_v_max_star: Decimal
    predicted_midpoint_q_star: Decimal
    predicted_midpoint_e_star: Decimal
    predicted_midpoint_v_max_star: Decimal
    predicted_plus_target_pass: bool
    predicted_minus_target_pass: bool
    predicted_midpoint_target_pass: bool
    predicted_plus_sink_pass: bool
    predicted_minus_sink_pass: bool
    predicted_midpoint_sink_pass: bool
    predicted_plus_admit: bool
    predicted_minus_admit: bool
    predicted_midpoint_admit: bool

    def __post_init__(self) -> None:
        for name in ("action_id", "requested_action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "amplitude_u_star",
            "duration_t_star",
            "predicted_plus_q_star",
            "predicted_minus_q_star",
            "predicted_plus_e_star",
            "predicted_minus_e_star",
            "predicted_plus_v_max_star",
            "predicted_minus_v_max_star",
            "predicted_midpoint_q_star",
            "predicted_midpoint_e_star",
            "predicted_midpoint_v_max_star",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.predicted_plus_admit
            != (self.predicted_plus_target_pass and self.predicted_plus_sink_pass)
            or self.predicted_minus_admit
            != (self.predicted_minus_target_pass and self.predicted_minus_sink_pass)
            or self.predicted_midpoint_admit
            != (self.predicted_midpoint_target_pass and self.predicted_midpoint_sink_pass)
        ):
            raise ValueError("history budget phase diagram predicted admission is not a gate intersection")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramChallengeNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-challenge-nomination'

    nomination_id: str
    coordinate_id: str
    cohort: HistoryBudgetPhaseDiagramCohort
    unit_id: str
    scale_cells: int
    depth: int
    resolution_epsilon: Decimal
    pair_index: int
    pair_kind: HistoryBudgetPhaseDiagramPairKind
    gate: HistoryBudgetPhaseDiagramGate
    targeting_action_id: str | None
    carrier_volts: Decimal
    center_values: tuple[Decimal, ...]
    amplitude_volts: Decimal
    mode_values: tuple[Decimal, ...]
    predicted_coordinate_defect: Decimal
    predicted_maximum_future_receiver_defect: Decimal
    predicted_maximum_future_metric_defect: Decimal
    action_predictions: tuple[HistoryBudgetPhaseDiagramActionPrediction, ...]
    observer_implementation_sha256: str
    outcome_count_at_nomination: int

    def __post_init__(self) -> None:
        for name in ("nomination_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.targeting_action_id is not None:
            validate_stable_id(self.targeting_action_id, field_name="targeting_action_id")
        if self.scale_cells not in {16, 32, 64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("history budget phase diagram nomination scale/depth differs")
        if self.cohort is not HistoryBudgetPhaseDiagramCohort.TARGETED:
            raise ValueError("history budget phase diagram challenge nomination must remain in cohort T")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if (
            self.pair_index < 0
            or len(self.mode_values) != self.scale_cells
            or len(self.center_values) != self.scale_cells
        ):
            raise ValueError("history budget phase diagram nomination mode roster differs")
        for field_name, values in (
            ("mode_values", self.mode_values),
            ("center_values", self.center_values),
        ):
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{field_name}[{index}]")
        for name in (
            "carrier_volts",
            "amplitude_volts",
            "predicted_coordinate_defect",
            "predicted_maximum_future_receiver_defect",
            "predicted_maximum_future_metric_defect",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.amplitude_volts <= 0:
            raise ValueError("history budget phase diagram nomination amplitude must be positive")
        if len(self.action_predictions) != 25:
            raise ValueError("history budget phase diagram nomination must carry the complete action panel")
        action_ids = tuple(value.action_id for value in self.action_predictions)
        _sorted_unique(action_ids, field_name="action_predictions")
        validate_sha256(
            self.observer_implementation_sha256,
            field_name="observer_implementation_sha256",
        )
        if self.outcome_count_at_nomination != 0:
            raise ValueError("history budget phase diagram nomination cannot observe generator outcomes")
        expected_gate = {
            HistoryBudgetPhaseDiagramPairKind.DYNAMICAL_BOUNDARY: HistoryBudgetPhaseDiagramGate.DYNAMICAL,
            HistoryBudgetPhaseDiagramPairKind.TARGET_BOUNDARY: HistoryBudgetPhaseDiagramGate.TARGET,
            HistoryBudgetPhaseDiagramPairKind.SINK_BOUNDARY: HistoryBudgetPhaseDiagramGate.SINK,
            HistoryBudgetPhaseDiagramPairKind.RANDOM_FIBRE: HistoryBudgetPhaseDiagramGate.RANDOM,
        }[self.pair_kind]
        if self.gate is not expected_gate:
            raise ValueError("history budget phase diagram pair kind and gate differ")
        if (
            self.pair_kind in {HistoryBudgetPhaseDiagramPairKind.RANDOM_FIBRE, HistoryBudgetPhaseDiagramPairKind.DYNAMICAL_BOUNDARY}
            and self.targeting_action_id is not None
        ) or (
            self.pair_kind in {HistoryBudgetPhaseDiagramPairKind.TARGET_BOUNDARY, HistoryBudgetPhaseDiagramPairKind.SINK_BOUNDARY}
            and self.targeting_action_id is None
        ):
            raise ValueError("history budget phase diagram targeting action and pair kind differ")
        if any(value.requested_action_id != value.action_id for value in self.action_predictions):
            raise ValueError("history budget phase diagram requested action identity differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramMethodFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-method-freeze'

    freeze_id: str
    development_config_sha256: str
    observer_implementation_sha256: str
    generator_implementation_sha256: str
    evaluator_implementation_sha256: str
    development_receipt_closure_sha256: str
    frozen_threshold_ids: tuple[str, ...]
    evaluation_outcome_count: int
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for name in (
            "development_config_sha256",
            "observer_implementation_sha256",
            "generator_implementation_sha256",
            "evaluator_implementation_sha256",
            "development_receipt_closure_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        _sorted_unique(self.frozen_threshold_ids, field_name="frozen_threshold_ids")
        if self.evaluation_outcome_count != 0:
            raise ValueError("history budget phase diagram method freeze must precede evaluation outcomes")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("history budget phase diagram method-freeze ceiling differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramGeneratorActionOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-generator-action-outcome'

    action_id: str
    requested_action_id: str
    accepted_action_id: str
    applied_action_id: str
    realized_action_id: str
    amplitude_u_star: Decimal
    duration_t_star: Decimal
    plus_q_star: Decimal
    minus_q_star: Decimal
    plus_e_star: Decimal
    minus_e_star: Decimal
    plus_v_max_star: Decimal
    minus_v_max_star: Decimal
    midpoint_q_star: Decimal
    midpoint_e_star: Decimal
    midpoint_v_max_star: Decimal
    plus_target_pass: bool
    minus_target_pass: bool
    midpoint_target_pass: bool
    plus_sink_pass: bool
    minus_sink_pass: bool
    midpoint_sink_pass: bool
    plus_admit: bool
    minus_admit: bool
    midpoint_admit: bool
    plus_left_current_amperes: Decimal
    minus_left_current_amperes: Decimal
    plus_right_current_amperes: Decimal
    minus_right_current_amperes: Decimal
    requested_accepted_applied_realized_parity: bool

    def __post_init__(self) -> None:
        for name in (
            "action_id",
            "requested_action_id",
            "accepted_action_id",
            "applied_action_id",
            "realized_action_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "amplitude_u_star",
            "duration_t_star",
            "plus_q_star",
            "minus_q_star",
            "plus_e_star",
            "minus_e_star",
            "plus_v_max_star",
            "minus_v_max_star",
            "midpoint_q_star",
            "midpoint_e_star",
            "midpoint_v_max_star",
            "plus_left_current_amperes",
            "minus_left_current_amperes",
            "plus_right_current_amperes",
            "minus_right_current_amperes",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.plus_admit != (self.plus_target_pass and self.plus_sink_pass)
            or self.minus_admit != (self.minus_target_pass and self.minus_sink_pass)
            or self.midpoint_admit != (self.midpoint_target_pass and self.midpoint_sink_pass)
        ):
            raise ValueError("history budget phase diagram realized admission is not a gate intersection")
        exact_parity = (
            len(
                {
                    self.requested_action_id,
                    self.accepted_action_id,
                    self.applied_action_id,
                    self.realized_action_id,
                }
            )
            == 1
        )
        if self.requested_accepted_applied_realized_parity != exact_parity:
            raise ValueError("history budget phase diagram action parity flag is not derived")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramGeneratorPairOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-generator-pair-outcome'

    nomination_id: str
    realized_coordinate_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    future_receiver_defects: tuple[Decimal, ...]
    action_outcomes: tuple[HistoryBudgetPhaseDiagramGeneratorActionOutcome, ...]
    voltage_envelope_valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        for name in (
            "realized_coordinate_defect",
            "maximum_future_receiver_defect",
            "maximum_future_metric_defect",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if len(self.future_receiver_defects) != 3:
            raise ValueError("history budget phase diagram generator future diagnostic roster differs")
        for index, value in enumerate(self.future_receiver_defects):
            validate_decimal(
                value, field_name=f"future_receiver_defects[{index}]", minimum=Decimal(0)
            )
        if len(self.action_outcomes) != 25:
            raise ValueError("history budget phase diagram generator outcome requires the complete action panel")
        action_ids = tuple(value.action_id for value in self.action_outcomes)
        _sorted_unique(action_ids, field_name="action_outcomes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramGeneratorOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-generator-outcome'

    outcome_id: str
    unit_id: str
    descriptor_sha256: str
    nomination_set_sha256: str
    generator_implementation_sha256: str
    pair_outcomes: tuple[HistoryBudgetPhaseDiagramGeneratorPairOutcome, ...]
    arrays_sha256: str
    array_count: int
    outcome_access: OutcomeAccess
    scientific_verdict_assigned: bool

    def __post_init__(self) -> None:
        for name in ("outcome_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "descriptor_sha256",
            "nomination_set_sha256",
            "generator_implementation_sha256",
            "arrays_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        pair_ids = tuple(value.nomination_id for value in self.pair_outcomes)
        _sorted_unique(pair_ids, field_name="pair_outcomes")
        if self.array_count <= 0 or self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("history budget phase diagram generator outcome access/array roster differs")
        if self.scientific_verdict_assigned:
            raise ValueError("history budget phase diagram generator cannot assign a scientific verdict")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramUntouchedGeneratorOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-untouched-generator-outcome'

    outcome_id: str
    unit_id: str
    scale_cells: int
    descriptor_sha256: str
    preparation_manifest_sha256: str
    generator_implementation_sha256: str
    preparation_count: int
    action_ids: tuple[str, ...]
    arrays_sha256: str
    outcome_access: OutcomeAccess
    requested_accepted_applied_realized_parity: bool
    scientific_verdict_assigned: bool

    def __post_init__(self) -> None:
        for name in ("outcome_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "descriptor_sha256",
            "preparation_manifest_sha256",
            "generator_implementation_sha256",
            "arrays_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256} or self.preparation_count != 512:
            raise ValueError("history budget phase diagram untouched generator roster differs")
        _sorted_unique(self.action_ids, field_name="action_ids")
        if len(self.action_ids) != 25:
            raise ValueError("history budget phase diagram untouched generator requires the complete action panel")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("history budget phase diagram untouched generator access differs")
        if not self.requested_accepted_applied_realized_parity:
            raise ValueError("history budget phase diagram untouched generator action identity differs")
        if self.scientific_verdict_assigned:
            raise ValueError("history budget phase diagram untouched generator cannot assign a verdict")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-untouched-coordinate-adjudication'

    adjudication_id: str
    coordinate_id: str
    unit_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    depth: int
    resolution_epsilon: Decimal
    preparation_count: int
    valid: bool
    collision_edge_count: int | None
    prefix_edge_counts: tuple[int, ...]
    dynamic_adverse_edge_count: int | None
    decision_adverse_edge_count: int | None
    any_adverse_edge_count: int | None
    unsafe_false_promotion_edge_count: int | None
    false_hold_edge_count: int | None
    maximum_future_receiver_defect: Decimal | None
    maximum_future_metric_defect: Decimal | None
    scientific_state: HistoryBudgetPhaseDiagramScientificState
    unsafe_first: bool

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.scale_cells not in {64, 128, 256}
            or not 0 <= self.depth <= 31
            or self.preparation_count != 512
            or len(self.prefix_edge_counts) != 4
        ):
            raise ValueError("history budget phase diagram untouched adjudication roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        counts = (
            self.collision_edge_count,
            self.dynamic_adverse_edge_count,
            self.decision_adverse_edge_count,
            self.any_adverse_edge_count,
            self.unsafe_false_promotion_edge_count,
            self.false_hold_edge_count,
        )
        metrics = (
            self.maximum_future_receiver_defect,
            self.maximum_future_metric_defect,
        )
        if not self.valid:
            if any(value is not None for value in counts + metrics):
                raise ValueError("invalid history budget phase diagram untouched unit cannot encode zero evidence")
            if self.scientific_state is not HistoryBudgetPhaseDiagramScientificState.INVALID:
                raise ValueError("invalid history budget phase diagram untouched unit has another state")
            return
        if any(value is None for value in counts + metrics):
            raise ValueError("valid history budget phase diagram untouched adjudication is incomplete")
        integer_counts = tuple(int(value) for value in counts if value is not None)
        edge_count = integer_counts[0]
        if min((*integer_counts, *self.prefix_edge_counts)) < 0:
            raise ValueError("history budget phase diagram untouched counts cannot be negative")
        if tuple(sorted(self.prefix_edge_counts)) != self.prefix_edge_counts:
            raise ValueError("history budget phase diagram prefix encounter counts must be nested")
        if self.prefix_edge_counts[-1] != edge_count:
            raise ValueError("history budget phase diagram 512-preparation prefix must equal the full graph")
        if any(value > edge_count for value in integer_counts[1:]):
            raise ValueError("history budget phase diagram adverse edge count exceeds collision edges")
        for index, value in enumerate(metrics):
            assert value is not None
            validate_decimal(value, field_name=f"maximum_defect[{index}]", minimum=Decimal(0))
        expected = (
            HistoryBudgetPhaseDiagramScientificState.UNSAFE_FALSE_PROMOTION
            if integer_counts[4]
            else HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if integer_counts[1] or integer_counts[2]
            else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
            if edge_count
            else HistoryBudgetPhaseDiagramScientificState.NO_COLLISION_ENCOUNTER_WITH_512_PREPARATIONS
        )
        if self.scientific_state is not expected or not self.unsafe_first:
            raise ValueError("history budget phase diagram untouched state violates unsafe-first precedence")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramPairAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-pair-adjudication'

    nomination_id: str
    coordinate_id: str
    depth: int
    resolution_epsilon: Decimal
    gate: HistoryBudgetPhaseDiagramGate
    realized_coordinate_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    target_mismatch_count: int
    sink_mismatch_count: int
    decision_mismatch_count: int
    false_safe_count: int
    unsafe_false_promotion_count: int
    false_hold_count: int
    collision_confirmed: bool
    dynamically_adverse: bool
    decision_adverse: bool
    decision_disposition: HistoryBudgetPhaseDiagramDecisionDisposition
    prediction_agreement: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if not 0 <= self.depth <= 31:
            raise ValueError("history budget phase diagram pair depth differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        for name in (
            "realized_coordinate_defect",
            "maximum_future_receiver_defect",
            "maximum_future_metric_defect",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        counts = (
            self.target_mismatch_count,
            self.sink_mismatch_count,
            self.decision_mismatch_count,
            self.false_safe_count,
            self.unsafe_false_promotion_count,
            self.false_hold_count,
        )
        if min(counts) < 0 or max(counts) > 25:
            raise ValueError("history budget phase diagram mismatch count differs")
        if self.false_safe_count + self.false_hold_count != self.decision_mismatch_count:
            raise ValueError("history budget phase diagram false-safe/hold counts do not partition decisions")
        if self.unsafe_false_promotion_count != self.false_safe_count:
            raise ValueError("history budget phase diagram unsafe-promotion count differs from false-safe continuity")
        if self.decision_adverse != (self.collision_confirmed and self.decision_mismatch_count > 0):
            raise ValueError("history budget phase diagram decision-adverse state is not derived")
        expected_disposition = (
            HistoryBudgetPhaseDiagramDecisionDisposition.UNSAFE_FALSE_PROMOTION
            if self.unsafe_false_promotion_count
            else (
                HistoryBudgetPhaseDiagramDecisionDisposition.FALSE_HOLD
                if self.false_hold_count
                else HistoryBudgetPhaseDiagramDecisionDisposition.CORRECT_ACTION
            )
        )
        if self.decision_disposition is not expected_disposition:
            raise ValueError("history budget phase diagram decision disposition violates unsafe-first precedence")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramDepthAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-depth-adjudication'

    depth: int
    requested_pair_count: int
    realized_collision_count: int
    dynamic_adverse_count: int
    decision_adverse_count: int
    false_safe_count: int
    dynamical_state: HistoryBudgetPhaseDiagramScientificState
    decision_state: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        counts = (
            self.requested_pair_count,
            self.realized_collision_count,
            self.dynamic_adverse_count,
            self.decision_adverse_count,
            self.false_safe_count,
        )
        if not 0 <= self.depth <= 31 or min(counts) < 0:
            raise ValueError("history budget phase diagram depth adjudication differs")
        if (
            not self.dynamic_adverse_count
            <= self.realized_collision_count
            <= self.requested_pair_count
        ):
            raise ValueError("history budget phase diagram dynamical depth counts differ")
        if not self.decision_adverse_count <= self.realized_collision_count:
            raise ValueError("history budget phase diagram decision depth counts differ")
        expected_dynamical_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if self.dynamic_adverse_count
            else HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if self.realized_collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if self.decision_adverse_count
            else HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if self.realized_collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("history budget phase diagram depth scientific states are not derived")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication(CanonicalRecord):
    """Bounded T-cohort state for one scale, coordinate and endpoint pair."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-targeted-coordinate-adjudication'

    adjudication_id: str
    coordinate_id: str
    unit_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_kind: HistoryBudgetPhaseDiagramCoordinateKind
    depth: int
    budget: Decimal | None
    resolution_epsilon: Decimal
    primary: bool
    valid: bool
    nominated_pair_count: int | None
    realized_collision_count: int | None
    dynamic_adverse_count: int | None
    decision_adverse_count: int | None
    unsafe_false_promotion_count: int | None
    false_hold_count: int | None
    dynamical_state: HistoryBudgetPhaseDiagramScientificState
    decision_state: HistoryBudgetPhaseDiagramScientificState
    dynamical_certificate_complete: bool
    decision_certificate_complete: bool
    generator_observer_agreement: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("history budget phase diagram targeted coordinate roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if self.coordinate_kind is HistoryBudgetPhaseDiagramCoordinateKind.ABSOLUTE_DEPTH:
            if self.budget is not None:
                raise ValueError("absolute targeted coordinate cannot carry a budget")
        elif self.budget is None:
            raise ValueError("budget targeted coordinate lacks its budget")
        else:
            _positive_decimal(self.budget, field_name="budget")
        counts = (
            self.nominated_pair_count,
            self.realized_collision_count,
            self.dynamic_adverse_count,
            self.decision_adverse_count,
            self.unsafe_false_promotion_count,
            self.false_hold_count,
        )
        if not self.valid:
            if any(value is not None for value in counts):
                raise ValueError("invalid targeted coordinate cannot encode zero evidence")
            if (
                self.dynamical_state is not HistoryBudgetPhaseDiagramScientificState.INVALID
                or self.decision_state is not HistoryBudgetPhaseDiagramScientificState.INVALID
                or self.generator_observer_agreement is not None
            ):
                raise ValueError("invalid targeted coordinate has evidentiary state")
            return
        if any(value is None for value in counts) or self.generator_observer_agreement is None:
            raise ValueError("valid targeted coordinate is incomplete")
        integer_counts = tuple(int(value) for value in counts if value is not None)
        nominated, realized, dynamic, decision, unsafe, false_hold = integer_counts
        if (
            min(integer_counts) < 0
            or realized > nominated
            or max(dynamic, decision, unsafe, false_hold) > realized
        ):
            raise ValueError("targeted coordinate denominator chain differs")
        expected_dynamic = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if dynamic
            else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
            if self.dynamical_certificate_complete
            else HistoryBudgetPhaseDiagramScientificState.TARGETABILITY_LIMITED
        )
        expected_decision = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if decision
            else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
            if self.decision_certificate_complete
            else HistoryBudgetPhaseDiagramScientificState.TARGETABILITY_LIMITED
        )
        if (
            self.dynamical_state is not expected_dynamic
            or self.decision_state is not expected_decision
        ):
            raise ValueError("targeted coordinate state is not certificate-derived")
        _sorted_unique(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramUnitAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-unit-adjudication'

    adjudication_id: str
    unit_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    descriptor_valid: bool
    nominated_pair_count: int
    realized_collision_count: int
    evaluable_pair_count: int
    pair_adjudications: tuple[HistoryBudgetPhaseDiagramPairAdjudication, ...]
    depth_adjudications: tuple[HistoryBudgetPhaseDiagramDepthAdjudication, ...]
    dynamical_state: HistoryBudgetPhaseDiagramScientificState
    decision_state: HistoryBudgetPhaseDiagramScientificState
    target_mismatch_count: int
    sink_mismatch_count: int
    false_safe_count: int
    false_hold_count: int
    k_full_effective: int | None
    b_full: Decimal | None
    rank_curve_distance: Decimal
    generator_observer_agreement: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.nominated_pair_count,
            self.realized_collision_count,
            self.evaluable_pair_count,
            self.target_mismatch_count,
            self.sink_mismatch_count,
            self.false_safe_count,
            self.false_hold_count,
        )
        if self.scale_cells not in {16, 32, 64, 128, 256} or min(counts) < 0:
            raise ValueError("history budget phase diagram unit counts/scale differ")
        if (
            not self.evaluable_pair_count
            <= self.realized_collision_count
            <= self.nominated_pair_count
        ):
            raise ValueError("history budget phase diagram unit denominator chain differs")
        pair_ids = tuple(value.nomination_id for value in self.pair_adjudications)
        _sorted_unique(pair_ids, field_name="pair_adjudications")
        if (
            self.nominated_pair_count != len(self.pair_adjudications)
            or self.realized_collision_count
            != sum(value.collision_confirmed for value in self.pair_adjudications)
            or self.evaluable_pair_count != self.realized_collision_count
            or self.target_mismatch_count
            != sum(value.target_mismatch_count for value in self.pair_adjudications)
            or self.sink_mismatch_count
            != sum(value.sink_mismatch_count for value in self.pair_adjudications)
            or self.false_safe_count
            != sum(value.false_safe_count for value in self.pair_adjudications)
            or self.false_hold_count
            != sum(value.false_hold_count for value in self.pair_adjudications)
        ):
            raise ValueError("history budget phase diagram unit counts do not match pair adjudications")
        if tuple(value.depth for value in self.depth_adjudications) != tuple(
            sorted({value.depth for value in self.depth_adjudications})
        ):
            raise ValueError("history budget phase diagram depth adjudications must be sorted and unique")
        if (
            sum(value.requested_pair_count for value in self.depth_adjudications)
            != self.nominated_pair_count
            or sum(value.realized_collision_count for value in self.depth_adjudications)
            != self.realized_collision_count
            or sum(value.dynamic_adverse_count for value in self.depth_adjudications)
            != sum(value.dynamically_adverse for value in self.pair_adjudications)
            or sum(value.decision_adverse_count for value in self.depth_adjudications)
            != sum(value.decision_adverse for value in self.pair_adjudications)
            or sum(value.false_safe_count for value in self.depth_adjudications)
            != self.false_safe_count
        ):
            raise ValueError("history budget phase diagram depth ledger does not aggregate the pair ledger")
        expected_dynamical_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if any(value.dynamically_adverse for value in self.pair_adjudications)
            else HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if self.realized_collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if any(value.decision_adverse for value in self.pair_adjudications)
            else HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if self.realized_collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("history budget phase diagram unit scientific states are not derived")
        if self.generator_observer_agreement != all(
            value.prediction_agreement for value in self.pair_adjudications
        ):
            raise ValueError("history budget phase diagram generator/observer agreement is not derived")
        if self.k_full_effective is not None and not 0 <= self.k_full_effective <= 31:
            raise ValueError("history budget phase diagram full-rank depth differs")
        if self.b_full is not None:
            validate_decimal(self.b_full, field_name="b_full", minimum=Decimal(0))
            if self.b_full > 1:
                raise ValueError("history budget phase diagram normalized history budget exceeds one")
        validate_decimal(
            self.rank_curve_distance, field_name="rank_curve_distance", minimum=Decimal(0)
        )
        _sorted_unique(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramCellRecurrence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-cell-recurrence'

    cell_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_label: str
    coordinate_kind: HistoryBudgetPhaseDiagramCoordinateKind
    depth: int
    budget: Decimal | None
    endpoint_id: str
    requested_count: int
    invalid_count: int
    opposed_count: int
    informative_nonadverse_count: int
    targetability_limited_count: int
    unsafe_false_promotion_unit_count: int
    false_hold_unit_count: int
    opposition_lower_bound: Decimal
    opposition_upper_bound: Decimal
    informative_nonadverse_lower_bound: Decimal
    informative_nonadverse_upper_bound: Decimal
    existence_recurs: bool
    majority_opposition: bool
    majority_informative_nonadverse: bool

    def __post_init__(self) -> None:
        for name in ("cell_id", "endpoint_id", "coordinate_label"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.requested_count,
            self.invalid_count,
            self.opposed_count,
            self.informative_nonadverse_count,
            self.targetability_limited_count,
            self.unsafe_false_promotion_unit_count,
            self.false_hold_unit_count,
        )
        if self.scale_cells not in {64, 128, 256} or not 0 <= self.depth <= 31 or min(counts) < 0:
            raise ValueError("history budget phase diagram recurrence cell differs")
        if self.requested_count != 30 or sum(counts[1:5]) != self.requested_count:
            raise ValueError("history budget phase diagram primary cell denominator must remain 30")
        if max(self.unsafe_false_promotion_unit_count, self.false_hold_unit_count) > 30:
            raise ValueError("history budget phase diagram decision-error unit count exceeds requests")
        if self.coordinate_kind is HistoryBudgetPhaseDiagramCoordinateKind.ABSOLUTE_DEPTH:
            if self.budget is not None:
                raise ValueError("absolute recurrence cell cannot carry a budget")
        else:
            if self.budget is None:
                raise ValueError("budget recurrence cell lacks its budget")
            _positive_decimal(self.budget, field_name="budget")
        for name in (
            "opposition_lower_bound",
            "opposition_upper_bound",
            "informative_nonadverse_lower_bound",
            "informative_nonadverse_upper_bound",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) > 1:
                raise ValueError("history budget phase diagram exact proportion interval exceeds one")
        if (
            self.opposition_lower_bound > self.opposition_upper_bound
            or self.informative_nonadverse_lower_bound > self.informative_nonadverse_upper_bound
        ):
            raise ValueError("history budget phase diagram exact proportion interval is reversed")
        if self.existence_recurs != (self.opposed_count > 0):
            raise ValueError("history budget phase diagram cell existence flag is not derived")
        if self.majority_opposition != (self.opposition_lower_bound > Decimal("0.5")):
            raise ValueError("history budget phase diagram cell majority flag is not derived")
        if self.majority_informative_nonadverse != (
            self.informative_nonadverse_lower_bound > Decimal("0.5")
        ):
            raise ValueError("history budget phase diagram informative-nonadverse majority flag is not derived")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramUntouchedPrevalenceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-untouched-prevalence-cell'

    cell_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_label: str
    depth: int
    budget: Decimal | None
    requested_count: int
    invalid_count: int
    encounter_count: int
    dynamic_adverse_count: int
    decision_adverse_count: int
    any_adverse_count: int
    unsafe_false_promotion_count: int
    false_hold_count: int
    any_adverse_lower_bound: Decimal | None
    any_adverse_upper_bound: Decimal | None
    unsafe_lower_bound: Decimal | None
    unsafe_upper_bound: Decimal | None
    any_adverse_disposition: HistoryBudgetPhaseDiagramScientificState
    unsafe_disposition: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        for name in ("cell_id", "coordinate_label"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.invalid_count,
            self.encounter_count,
            self.dynamic_adverse_count,
            self.decision_adverse_count,
            self.any_adverse_count,
            self.unsafe_false_promotion_count,
            self.false_hold_count,
        )
        if (
            self.scale_cells not in {64, 128, 256}
            or not 0 <= self.depth <= 31
            or self.requested_count != 30
            or min(counts) < 0
            or max(counts) > self.requested_count
        ):
            raise ValueError("history budget phase diagram U prevalence cell differs")
        if max(counts[2:]) > self.encounter_count:
            raise ValueError("history budget phase diagram U adverse count exceeds encounters")
        intervals = (
            self.any_adverse_lower_bound,
            self.any_adverse_upper_bound,
            self.unsafe_lower_bound,
            self.unsafe_upper_bound,
        )
        if self.invalid_count:
            if any(value is not None for value in intervals):
                raise ValueError("invalid history budget phase diagram U denominator cannot encode an interval")
            if (
                self.any_adverse_disposition is not HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
                or self.unsafe_disposition is not HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
            ):
                raise ValueError("invalid history budget phase diagram U denominator must remain unevaluable")
            return
        if any(value is None for value in intervals):
            raise ValueError("valid history budget phase diagram U denominator lacks an interval")
        for name in (
            "any_adverse_lower_bound",
            "any_adverse_upper_bound",
            "unsafe_lower_bound",
            "unsafe_upper_bound",
        ):
            value = getattr(self, name)
            assert value is not None
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError("history budget phase diagram U interval exceeds one")
        assert self.any_adverse_lower_bound is not None
        assert self.any_adverse_upper_bound is not None
        assert self.unsafe_lower_bound is not None
        assert self.unsafe_upper_bound is not None
        if (
            self.any_adverse_lower_bound > self.any_adverse_upper_bound
            or self.unsafe_lower_bound > self.unsafe_upper_bound
        ):
            raise ValueError("history budget phase diagram U interval is reversed")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramUntouchedDescriptiveSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-untouched-descriptive-summary'

    summary_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_id: str
    requested_count: int
    valid_count: int
    prefix_encounter_unit_counts: tuple[int, ...]
    mean_collision_edge_fraction: Decimal | None
    collision_edge_fraction_lower_95: Decimal | None
    collision_edge_fraction_upper_95: Decimal | None
    adverse_given_collision_evaluable_count: int
    mean_adverse_given_collision_fraction: Decimal | None
    adverse_given_collision_lower_95: Decimal | None
    adverse_given_collision_upper_95: Decimal | None
    whole_unit_bootstrap: bool

    def __post_init__(self) -> None:
        for name in ("summary_id", "coordinate_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.scale_cells not in {64, 128, 256}
            or self.requested_count != 30
            or not 0 <= self.valid_count <= 30
            or len(self.prefix_encounter_unit_counts) != 4
            or any(not 0 <= value <= 30 for value in self.prefix_encounter_unit_counts)
            or tuple(sorted(self.prefix_encounter_unit_counts)) != self.prefix_encounter_unit_counts
            or not 0 <= self.adverse_given_collision_evaluable_count <= self.valid_count
            or not self.whole_unit_bootstrap
        ):
            raise ValueError("history budget phase diagram U descriptive summary roster differs")
        collision_estimates = (
            self.mean_collision_edge_fraction,
            self.collision_edge_fraction_lower_95,
            self.collision_edge_fraction_upper_95,
        )
        if self.valid_count == 0:
            if any(value is not None for value in collision_estimates):
                raise ValueError("empty valid-unit denominator has a collision estimate")
        elif any(value is None for value in collision_estimates):
            raise ValueError("evaluable collision denominator lacks its interval")
        else:
            for name in (
                "mean_collision_edge_fraction",
                "collision_edge_fraction_lower_95",
                "collision_edge_fraction_upper_95",
            ):
                value = getattr(self, name)
                assert value is not None
                validate_decimal(value, field_name=name, minimum=Decimal(0))
                if value > 1:
                    raise ValueError("history budget phase diagram collision-edge fraction exceeds one")
        optional = (
            self.mean_adverse_given_collision_fraction,
            self.adverse_given_collision_lower_95,
            self.adverse_given_collision_upper_95,
        )
        if self.adverse_given_collision_evaluable_count == 0:
            if any(value is not None for value in optional):
                raise ValueError("empty adverse-edge denominator has an estimate")
        elif any(value is None for value in optional):
            raise ValueError("adverse-edge estimate lacks its interval")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramAlignmentSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-alignment-summary'

    summary_id: str
    endpoint_id: str
    family_id: str
    requested_unit_count: int
    evaluable_unit_count: int
    mean_delta: Decimal | None
    lower_95: Decimal | None
    upper_95: Decimal | None
    disposition: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id", "family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.requested_unit_count not in {30, 90}
            or not 0 <= self.evaluable_unit_count <= self.requested_unit_count
        ):
            raise ValueError("history budget phase diagram alignment denominator differs")
        values = (self.mean_delta, self.lower_95, self.upper_95)
        if self.evaluable_unit_count == 0:
            if (
                any(value is not None for value in values)
                or self.disposition is not HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
            ):
                raise ValueError("empty history budget phase diagram alignment is evidentiary")
        elif any(value is None for value in values):
            raise ValueError("evaluable history budget phase diagram alignment lacks its interval")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramResolutionPanelSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-resolution-panel-summary'

    summary_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    depth: int
    endpoint_id: str
    requested_count: int
    invalid_count: int
    changed_state_count: int
    targetability_changed_count: int
    disposition: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.scale_cells not in {64, 128, 256}
            or self.depth not in {4, 6, 8}
            or self.requested_count != 30
            or min(self.invalid_count, self.changed_state_count, self.targetability_changed_count)
            < 0
            or max(self.invalid_count, self.changed_state_count, self.targetability_changed_count)
            > 30
        ):
            raise ValueError("history budget phase diagram resolution-panel summary differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramRankObjectSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-rank-object-summary'

    summary_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_id: str
    depth: int
    resolution_epsilon: Decimal
    requested_count: int
    structural_rank_median: Decimal
    discrete_lower_rank_median: Decimal
    discrete_upper_rank_median: Decimal
    effective_rank_median: Decimal
    compatible_count: int
    sampling_alias_count: int
    numerical_cancellation_count: int
    unresolved_count: int
    conditioning_right_censored_count: int

    def __post_init__(self) -> None:
        for name in ("summary_id", "coordinate_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.compatible_count,
            self.sampling_alias_count,
            self.numerical_cancellation_count,
            self.unresolved_count,
        )
        if (
            self.scale_cells not in {64, 128, 256}
            or not 0 <= self.depth <= 31
            or self.requested_count != 30
            or sum(counts) != 30
            or not 0 <= self.conditioning_right_censored_count <= 30
        ):
            raise ValueError("history budget phase diagram rank-object summary roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        for name in (
            "structural_rank_median",
            "discrete_lower_rank_median",
            "discrete_upper_rank_median",
            "effective_rank_median",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramTransitionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-transition-summary'

    summary_id: str
    family: HistoryBudgetPhaseDiagramDisorderFamily
    scale_cells: int
    coordinate_kind: HistoryBudgetPhaseDiagramCoordinateKind
    endpoint_id: str
    ordered_coordinate_labels: tuple[str, ...]
    last_opposed_coordinate: str | None
    first_informative_nonadverse_coordinate: str | None
    targetability_limited_coordinates: tuple[str, ...]
    reversal_pairs: tuple[str, ...]
    disposition: HistoryBudgetPhaseDiagramScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256}:
            raise ValueError("history budget phase diagram transition scale differs")
        if not self.ordered_coordinate_labels:
            raise ValueError("history budget phase diagram transition grid is empty")
        for values, field_name in (
            (self.targetability_limited_coordinates, "targetability_limited_coordinates"),
            (self.reversal_pairs, "reversal_pairs"),
        ):
            _sorted_unique(values, field_name=field_name)
        for value in (
            self.last_opposed_coordinate,
            self.first_informative_nonadverse_coordinate,
        ):
            if value is not None:
                validate_stable_id(value, field_name="transition_coordinate")
        if (
            self.disposition is HistoryBudgetPhaseDiagramScientificState.NONMONOTONE_PHASE_PATTERN
            and not self.reversal_pairs
        ):
            raise ValueError("history budget phase diagram transition reversal disposition differs")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramRecurrenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-recurrence-result'

    result_id: str
    method_freeze_sha256: str
    cells: tuple[HistoryBudgetPhaseDiagramCellRecurrence, ...]
    secondary_cells: tuple[HistoryBudgetPhaseDiagramCellRecurrence, ...]
    untouched_cells: tuple[HistoryBudgetPhaseDiagramUntouchedPrevalenceCell, ...]
    untouched_descriptive_summaries: tuple[HistoryBudgetPhaseDiagramUntouchedDescriptiveSummary, ...]
    alignment_summaries: tuple[HistoryBudgetPhaseDiagramAlignmentSummary, ...]
    resolution_summaries: tuple[HistoryBudgetPhaseDiagramResolutionPanelSummary, ...]
    rank_summaries: tuple[HistoryBudgetPhaseDiagramRankObjectSummary, ...]
    transition_summaries: tuple[HistoryBudgetPhaseDiagramTransitionSummary, ...]
    fixed_depth_dynamical_bracket: HistoryBudgetPhaseDiagramScientificState
    fixed_depth_decision_bracket: HistoryBudgetPhaseDiagramScientificState
    normalized_budget_dynamical_bracket: HistoryBudgetPhaseDiagramScientificState
    normalized_budget_decision_bracket: HistoryBudgetPhaseDiagramScientificState
    bootstrap_summary_sha256: str
    decisive_counterexample_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    physical_claim_allowed: bool
    controller_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        validate_sha256(self.bootstrap_summary_sha256, field_name="bootstrap_summary_sha256")
        for field_name, values in (
            ("cells", self.cells),
            ("secondary_cells", self.secondary_cells),
            ("untouched_cells", self.untouched_cells),
            ("untouched_descriptive_summaries", self.untouched_descriptive_summaries),
            ("alignment_summaries", self.alignment_summaries),
            ("resolution_summaries", self.resolution_summaries),
            ("rank_summaries", self.rank_summaries),
            ("transition_summaries", self.transition_summaries),
        ):
            ids = tuple(
                getattr(value, "cell_id", getattr(value, "summary_id", "")) for value in values
            )
            _sorted_unique(ids, field_name=field_name)
        if len(self.cells) != 72 or len(self.untouched_cells) != 36:
            raise ValueError("history budget phase diagram result requires the frozen 72 T and 36 U primary cells")
        if len(self.secondary_cells) != 216:
            raise ValueError("history budget phase diagram result requires all frozen secondary T cells")
        if len(self.untouched_descriptive_summaries) != 144:
            raise ValueError("history budget phase diagram result requires all frozen U descriptive cells")
        if len(self.alignment_summaries) != 8 or len(self.resolution_summaries) != 54:
            raise ValueError("history budget phase diagram alignment or resolution-panel roster differs")
        if len(self.rank_summaries) != 144:
            raise ValueError("history budget phase diagram rank-object summary roster differs")
        if len(self.transition_summaries) != 36:
            raise ValueError("history budget phase diagram transition-summary roster differs")
        _sorted_unique(self.decisive_counterexample_ids, field_name="decisive_counterexample_ids")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("history budget phase diagram recurrence ceiling differs")
        if self.physical_claim_allowed or self.controller_claim_allowed:
            raise ValueError("history budget phase diagram cannot claim physical recurrence or control")


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramTerminalCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/history-budget-phase-diagram/history-budget-phase-diagram-terminal-closeout'

    closeout_id: str
    terminal: HistoryBudgetPhaseDiagramTerminal
    recurrence_result_sha256: str | None
    artifact_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    external_device_count: int
    physical_action_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        if self.recurrence_result_sha256 is not None:
            validate_sha256(self.recurrence_result_sha256, field_name="recurrence_result_sha256")
        for name in ("artifact_ids", "receipt_ids", "limitation_ids"):
            _sorted_unique(getattr(self, name), field_name=name)
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("history budget phase diagram closeout ceiling differs")
        if self.external_device_count != 0 or self.physical_action_count != 0:
            raise ValueError("history budget phase diagram closeout must remain simulator-only")


__all__ = [name for name in globals() if name.startswith("HistoryBudgetPhaseDiagram")]
