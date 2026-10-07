"""Canonical interchange records for the receiver-history history-budget phase diagram.

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


RECEIVER_HISTORY_REDUCED_ACTION_MENU_PLAN_ID = 'receiver-history-reduced-action-menu-metatheory'
RECEIVER_HISTORY_REDUCED_ACTION_MENU_INFERENCE_ID = 'receiver-history-reduced-action-menu-complete-unit-phase-diagram'
RECEIVER_HISTORY_REDUCED_ACTION_MENU_EVALUATION_UNITS_PER_FAMILY = 12
RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_AMPLITUDES = (
    Decimal("0.2"),
    Decimal("0.6"),
    Decimal("1"),
)
RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_DURATIONS = (
    Decimal("0.01"),
    Decimal("0.05"),
    Decimal("0.2"),
)


RECEIVER_HISTORY_HYPOTHESIS_IDS = ('development-power-transport', 'fixed-absolute-depth-bracket', 'normalized-budget-bracket', 'paired-coordinate-alignment', 'selective-contextual-locality', 'typed-endpoint-dependence')
RECEIVER_HISTORY_METATHEORY_PROPOSITION_IDS = ('contextual-lawhood', 'development-power-transport', 'endpoint-relative-lawhood', 'evidence-gate-separation', 'history-coordinate-bounds', 'prospective-discipline')
RECEIVER_HISTORY_EVIDENCE_GATE_IDS = ('measurement-provenance', 'observable-coordinate', 'response-identification', 'law-qualification', 'admission', 'prospective-use')

class ReceiverHistoryPhase(StrEnum):
    NOMINATION = "NOMINATION"
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    TARGETED_EVALUATION = 'TARGETED_EVALUATION'
    UNTOUCHED_EVALUATION = 'UNTOUCHED_EVALUATION'


class ReceiverHistoryDisorderFamily(StrEnum):
    SMOOTH_PERIODIC_12 = "smooth-periodic-12"
    CORRELATED_FIELD_12 = "correlated-field-12"
    MULTISCALE_WAVELET_12 = "multiscale-wavelet-12"


class ReceiverHistoryGate(StrEnum):
    DYNAMICAL = "DYNAMICAL"
    TARGET = "TARGET"
    SINK = "SINK"
    RANDOM = "RANDOM"


class ReceiverHistoryEndpoint(StrEnum):
    DYNAMICAL = "DYNAMICAL"
    TARGET_DECISION = "TARGET_DECISION"
    SINK_DECISION = "SINK_DECISION"


class ReceiverHistoryOrientation(StrEnum):
    UNSAFE_PROMOTION = "UNSAFE_PROMOTION"
    FALSE_HOLD = "FALSE_HOLD"


class ReceiverHistoryPowerState(StrEnum):
    WITNESS_CAPABLE = "WITNESS_CAPABLE"
    NONADVERSITY_CAPABLE = "NONADVERSITY_CAPABLE"
    MIXED_CAPABLE = "MIXED_CAPABLE"
    UNPOWERED = "UNPOWERED"
    INVALID = "INVALID"


class ReceiverHistoryPairKind(StrEnum):
    DYNAMICAL_BOUNDARY = "DYNAMICAL_BOUNDARY"
    TARGET_LEXICAL = "TARGET_LEXICAL"
    SINK_LEXICAL = "SINK_LEXICAL"
    RANDOM_FIBRE = "RANDOM_FIBRE"


class ReceiverHistoryCohort(StrEnum):
    TARGETED = "TARGETED"
    UNTOUCHED = "UNTOUCHED"


class ReceiverHistoryCoordinateKind(StrEnum):
    ABSOLUTE_DEPTH = "ABSOLUTE_DEPTH"
    NORMALIZED_BUDGET = "NORMALIZED_BUDGET"


class ReceiverHistoryRankCompatibility(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    SAMPLING_ALIAS = "SAMPLING_ALIAS"
    NUMERICAL_CANCELLATION = "NUMERICAL_CANCELLATION"
    RANK_UNRESOLVED = "RANK_UNRESOLVED"


class ReceiverHistoryDecisionDisposition(StrEnum):
    CORRECT_ACTION = "CORRECT_ACTION"
    CORRECT_HOLD = "CORRECT_HOLD"
    UNSAFE_FALSE_PROMOTION = "UNSAFE_FALSE_PROMOTION"
    FALSE_HOLD = "FALSE_HOLD"
    INVALID = "INVALID"


class ReceiverHistoryOptimizationState(StrEnum):
    WITNESS = "WITNESS"
    BELOW_MARGIN = "BELOW_MARGIN"
    INFEASIBLE = "INFEASIBLE"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"


class ReceiverHistoryScientificState(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    INFORMATIVE_NONADVERSE = "INFORMATIVE_NONADVERSE"
    TARGETABILITY_LIMITED = "TARGETABILITY_LIMITED"
    NO_ENCOUNTER_AT_512_PREPARATIONS = 'NO_ENCOUNTER_AT_512_PREPARATIONS'
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


class ReceiverHistoryMetatheoryDisposition(StrEnum):
    """Closed terminal vocabulary for receiver-history metatheory dispositions."""

    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    NOT_DISTINGUISHED = "NOT_DISTINGUISHED"
    TARGETABILITY_LIMITED = "TARGETABILITY_LIMITED"
    UNEVALUABLE = "UNEVALUABLE"
    INVALID = "INVALID"
    EXCLUDED = "EXCLUDED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class ReceiverHistoryLocalityCandidate(StrEnum):
    ACTION_ONLY = 'ACTION_ONLY'
    EIGHT_BIN_RECEIVER_AND_ACTION = 'EIGHT_BIN_RECEIVER_AND_ACTION'
    FIXED_HISTORY_DEPTH = 'FIXED_HISTORY_DEPTH'
    NORMALIZED_HISTORY_BUDGET = 'NORMALIZED_HISTORY_BUDGET'
    ENDPOINT_CONTEXT_ATLAS = "ENDPOINT_CONTEXT_ATLAS"


class ReceiverHistoryTerminal(StrEnum):
    ALL_ENDPOINTS_UNPOWERED = "ALL_ENDPOINTS_UNPOWERED"
    TARGETED_EVALUATION_COMPLETE = "TARGETED_EVALUATION_COMPLETE"
    INTEGRATED_EVALUATION_COMPLETE = "INTEGRATED_EVALUATION_COMPLETE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    HISTORY_BUDGET_PHASE_DIAGRAM_EVALUATED = "HISTORY_BUDGET_PHASE_DIAGRAM_EVALUATED"
    IMPLEMENTATION_STOP = "IMPLEMENTATION_STOP"
    DEVELOPMENT_STOP = "DEVELOPMENT_STOP"
    OPERATIONAL_STOP = "OPERATIONAL_STOP"


def _sorted_unique(values: tuple[str, ...], *, field_name: str, allow_empty: bool = True) -> None:
    if values != tuple(sorted(set(values))) or (not allow_empty and not values):
        raise ValueError(f"{field_name} must be sorted and unique")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    validate_decimal(value, field_name=field_name, minimum=Decimal(0))
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryCoordinateLabel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-coordinate-label'

    coordinate_id: str
    kind: ReceiverHistoryCoordinateKind
    scale_cells: int
    depth: int
    budget: Decimal | None
    resolution_epsilon: Decimal
    primary: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.scale_cells not in {16, 32, 64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("receiver-history coordinate scale/depth differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if self.kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH:
            if self.budget is not None:
                raise ValueError("absolute-depth coordinate cannot carry a budget")
        else:
            if self.budget is None:
                raise ValueError("normalized-budget coordinate requires a budget")
            _positive_decimal(self.budget, field_name="budget")
            if Decimal(8 * (self.depth + 1)) / Decimal(self.scale_cells) != self.budget:
                raise ValueError("normalized-budget coordinate mapping differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryStructuralRankStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-structural-rank-step'

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
            raise ValueError("receiver-history structural rank exceeds its matrix dimensions")
        if self.row_count <= 0 or self.state_count <= 0:
            raise ValueError("receiver-history structural rank dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDiscreteRankBracket(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-discrete-rank-bracket'

    coordinate_id: str
    lower_rank: int
    upper_rank: int
    state_count: int
    precision_bits: int
    residual_norm: Decimal
    separation_ratio: Decimal | None
    matrix_sha256: str
    compatibility: ReceiverHistoryRankCompatibility
    algorithm_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.algorithm_id, field_name="algorithm_id")
        validate_sha256(self.matrix_sha256, field_name="matrix_sha256")
        if not 0 <= self.lower_rank <= self.upper_rank <= self.state_count:
            raise ValueError("receiver-history discrete-rank bracket is invalid")
        if self.precision_bits < 256:
            raise ValueError("receiver-history discrete rank requires at least 256 bits")
        validate_decimal(self.residual_norm, field_name="residual_norm", minimum=Decimal(0))
        if self.separation_ratio is not None:
            _positive_decimal(self.separation_ratio, field_name="separation_ratio")
        if (
            self.compatibility is not ReceiverHistoryRankCompatibility.RANK_UNRESOLVED
            and self.lower_rank != self.upper_rank
        ):
            raise ValueError("resolved receiver-history rank requires a singleton bracket")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryConditioningStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-conditioning-step'

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
            raise ValueError("receiver-history effective rank cannot be negative")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryPreparationMapManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-preparation-map-manifest'

    manifest_id: str
    unit_id: str
    preparation_count: int
    fine_scale_cells: int
    coarse_scale_cells: tuple[int, ...]
    fine_states_sha256: str
    mapped_states_sha256: str
    collision_edges_sha256: str
    prefix_counts: tuple[int, ...]
    valid_preparation_counts: tuple[int, ...]
    invalid_reason_counts: tuple[tuple[str, int], ...]
    pre_cast_charge_preservation_max_error: Decimal
    charge_preservation_max_error: Decimal
    invalid_raw_bits_sha256: str
    validity_mask_sha256: str
    outcome_count_at_freeze: int
    algorithm_id: str

    def __post_init__(self) -> None:
        for name in ("manifest_id", "unit_id", "algorithm_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "fine_states_sha256",
            "mapped_states_sha256",
            "collision_edges_sha256",
            "invalid_raw_bits_sha256",
            "validity_mask_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.preparation_count != 512
            or self.fine_scale_cells != 256
            or self.coarse_scale_cells != (64, 128)
            or self.prefix_counts != (64, 128, 256)
            or len(self.valid_preparation_counts) != 3
            or any(
                not 0 <= value <= self.preparation_count for value in self.valid_preparation_counts
            )
        ):
            raise ValueError("receiver-history preparation-map roster differs")
        if tuple(key for key, _ in self.invalid_reason_counts) != tuple(
            sorted({key for key, _ in self.invalid_reason_counts})
        ) or any(value <= 0 for _, value in self.invalid_reason_counts):
            raise ValueError("receiver-history invalid-preparation reasons differ")
        validate_decimal(
            self.pre_cast_charge_preservation_max_error,
            field_name="pre_cast_charge_preservation_max_error",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.charge_preservation_max_error,
            field_name="charge_preservation_max_error",
            minimum=Decimal(0),
        )
        if self.outcome_count_at_freeze != 0:
            raise ValueError("receiver-history preparation map must precede outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryOptimizationCertificate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-optimization-certificate'

    certificate_id: str
    coordinate_id: str
    gate: ReceiverHistoryGate
    action_id: str | None
    orientation: ReceiverHistoryOrientation | None
    state: ReceiverHistoryOptimizationState
    objective_lower_bound: Decimal
    objective_upper_bound: Decimal
    required_margin: Decimal
    primal_residual: Decimal
    duality_gap: Decimal
    solver_id: str
    constraint_family_sha256: str
    complete_family_certificate: bool
    outcome_count_at_certificate: int

    def __post_init__(self) -> None:
        for name in ("certificate_id", "coordinate_id", "solver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_id is not None:
            validate_stable_id(self.action_id, field_name="action_id")
        decision_gate = self.gate in {ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK}
        if decision_gate != (self.orientation is not None):
            raise ValueError("receiver-history optimization orientation differs")
        validate_sha256(self.constraint_family_sha256, field_name="constraint_family_sha256")
        for name in (
            "objective_lower_bound",
            "objective_upper_bound",
            "required_margin",
            "primal_residual",
            "duality_gap",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.objective_lower_bound > self.objective_upper_bound:
            raise ValueError("receiver-history optimization bounds are reversed")
        if self.state is ReceiverHistoryOptimizationState.WITNESS and (
            self.objective_lower_bound < self.required_margin
        ):
            raise ValueError("receiver-history witness does not reach the required margin")
        if self.state is ReceiverHistoryOptimizationState.BELOW_MARGIN and (
            self.objective_upper_bound >= self.required_margin
        ):
            raise ValueError(
                "receiver-history below-margin certificate does not exclude the margin"
            )
        if (
            self.state
            in {
                ReceiverHistoryOptimizationState.WITNESS,
                ReceiverHistoryOptimizationState.BELOW_MARGIN,
                ReceiverHistoryOptimizationState.INFEASIBLE,
            }
            and not self.complete_family_certificate
        ):
            raise ValueError("resolved receiver-history optimization must be complete")
        if self.outcome_count_at_certificate != 0:
            raise ValueError("receiver-history optimization certificate cannot observe outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistorySparseCertificateCheck(CanonicalRecord):
    """Generator-closure reconstruction of one complete objective family."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-sparse-certificate-check'

    objective_key: str
    coordinate_id: str
    gate: ReceiverHistoryGate
    action_id: str | None
    orientation: ReceiverHistoryOrientation | None
    state: ReceiverHistoryOptimizationState
    objective_lower_bound: Decimal
    objective_upper_bound: Decimal
    required_margin: Decimal
    primal_residual: Decimal
    duality_gap: Decimal
    interval_width: Decimal
    constraint_family_sha256: str
    solver_id: str
    precision_bits: int
    complete_family_certificate: bool

    def __post_init__(self) -> None:
        for name in ("objective_key", "coordinate_id", "solver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_id is not None:
            validate_stable_id(self.action_id, field_name="action_id")
        decision_gate = self.gate in {ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK}
        if decision_gate != (self.orientation is not None):
            raise ValueError("receiver-history sparse-certificate orientation differs")
        validate_sha256(self.constraint_family_sha256, field_name="constraint_family_sha256")
        for name in (
            "objective_lower_bound",
            "objective_upper_bound",
            "required_margin",
            "primal_residual",
            "duality_gap",
            "interval_width",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.objective_lower_bound > self.objective_upper_bound:
            raise ValueError("receiver-history sparse certificate bounds are reversed")
        if self.precision_bits < 53:
            raise ValueError("receiver-history sparse certificate precision differs")
        if (
            self.state
            in {
                ReceiverHistoryOptimizationState.WITNESS,
                ReceiverHistoryOptimizationState.BELOW_MARGIN,
                ReceiverHistoryOptimizationState.INFEASIBLE,
            }
            and not self.complete_family_certificate
        ):
            raise ValueError("resolved sparse certificate must cover its complete family")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryConfig(CanonicalRecord):
    """Strict phase configuration shared by authoring and scientific code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-config'

    config_id: str
    phase: ReceiverHistoryPhase
    plan_id: str
    parent_result_id: str
    scale_cells: tuple[int, ...]
    excluded_resource_canary_scale_cells: tuple[int, ...]
    disorder_families: tuple[ReceiverHistoryDisorderFamily, ...]
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
    maximum_parallel_tasks: int
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
        reduced_runtime_profile = self.plan_id == RECEIVER_HISTORY_REDUCED_ACTION_MENU_PLAN_ID
        if reduced_runtime_profile and self.phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
            raise ValueError('receiver-history reduced-action-menu excludes conditional untouched evaluation')
        if self.maximum_parallel_tasks != (6 if reduced_runtime_profile else 4):
            raise ValueError("receiver-history maximum parallel task count differs")
        expected_scales = {
            ReceiverHistoryPhase.NOMINATION: (64, 128, 256),
            ReceiverHistoryPhase.CANARY: (16, 32),
            ReceiverHistoryPhase.DEVELOPMENT: (64, 128, 256),
            ReceiverHistoryPhase.TARGETED_EVALUATION: (64, 128, 256),
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: (64, 128, 256),
        }[self.phase]
        if self.scale_cells != expected_scales:
            raise ValueError("receiver-history phase scale roster differs")
        expected_resource_scales = (
            (64, 128, 256) if self.phase is ReceiverHistoryPhase.CANARY else ()
        )
        if self.excluded_resource_canary_scale_cells != expected_resource_scales:
            raise ValueError("receiver-history excluded resource-canary scale roster differs")
        if self.disorder_families != tuple(ReceiverHistoryDisorderFamily):
            raise ValueError("receiver-history disorder-family roster differs")
        _sorted_unique(self.unit_ids, field_name="unit_ids")
        evaluation_unit_count = (
            len(ReceiverHistoryDisorderFamily) * RECEIVER_HISTORY_REDUCED_ACTION_MENU_EVALUATION_UNITS_PER_FAMILY
            if reduced_runtime_profile
            else 90
        )
        expected_units = {
            ReceiverHistoryPhase.NOMINATION: evaluation_unit_count,
            ReceiverHistoryPhase.CANARY: 2,
            ReceiverHistoryPhase.DEVELOPMENT: 12,
            ReceiverHistoryPhase.TARGETED_EVALUATION: evaluation_unit_count,
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: evaluation_unit_count,
        }[self.phase]
        if len(self.unit_ids) != expected_units:
            raise ValueError("receiver-history phase complete-unit count differs")
        if self.phase in {ReceiverHistoryPhase.TARGETED_EVALUATION, ReceiverHistoryPhase.UNTOUCHED_EVALUATION}:
            if self.seed_roster_commitment_sha256 is None:
                raise ValueError("evaluation config requires the pre-development seed commitment")
            validate_sha256(
                self.seed_roster_commitment_sha256,
                field_name="seed_roster_commitment_sha256",
            )
        elif self.seed_roster_commitment_sha256 is not None:
            raise ValueError("only the evaluation config may bind the seed commitment")
        if self.receiver_bin_count != 8 or self.history_max_depth != 31:
            raise ValueError("receiver-history receiver/history roster differs")
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
            raise ValueError('receiver-history frozen continuity threshold differs')
        if self.coordinate_collision_epsilon != Decimal(
            "0.005"
        ) or self.future_divergence_epsilon != Decimal("0.005"):
            raise ValueError("receiver-history collision/divergence margins differ")
        if self.target_decision_margin != Decimal("0.005") or self.sink_decision_margin != Decimal(
            "0.005"
        ):
            raise ValueError("receiver-history target/sink decision margins differ")
        if self.generator_observer_metric_tolerance != Decimal("1e-10"):
            raise ValueError("receiver-history generator/observer metric tolerance differs")
        if self.optimization_tolerance != Decimal("1e-9"):
            raise ValueError("receiver-history optimization tolerance differs")
        if self.midpoint_boundary_tie_epsilon != Decimal("1e-10"):
            raise ValueError("receiver-history midpoint boundary tie convention differs")
        if self.hidden_voltage_envelope != (Decimal("0"), Decimal("1")):
            raise ValueError("receiver-history hidden-state envelope differs")
        if self.hidden_voltage_interior_margin != Decimal("1e-8"):
            raise ValueError("receiver-history hidden-state interior margin differs")
        if (
            self.target_pairs_per_depth != 2
            or self.sink_pairs_per_depth != 2
            or self.random_pairs_per_depth != 4
        ):
            raise ValueError("receiver-history targeting roster differs")
        if self.absolute_depths != (0, 4, 8):
            raise ValueError("receiver-history absolute-depth roster differs")
        if self.normalized_history_budgets != (
            Decimal("0.5"),
            Decimal("1"),
        ):
            raise ValueError("receiver-history normalized-budget roster differs")
        if self.primary_absolute_depth_contrast != (
            4,
            8,
        ) or self.primary_normalized_budget_contrast != (
            Decimal("0.5"),
            Decimal("1"),
        ):
            raise ValueError("receiver-history primary contrasts differ")
        if self.receiver_resolution_panel != (Decimal("0.005"),) or self.resolution_panel_depths:
            raise ValueError("receiver-history resolution panel differs")
        if self.untouched_preparation_count != 0 or self.untouched_prefix_counts:
            raise ValueError("receiver-history untouched-preparation roster differs")
        if (
            self.untouched_distribution_id != 'not-requested'
            or self.untouched_scale_map_id != 'not-requested'
            or self.untouched_earliest_lag != 0
            or self.untouched_low_rate_threshold != Decimal("0.20")
        ):
            raise ValueError("receiver-history untouched-preparation semantics differ")
        for name in ("action_amplitudes_u_star", "action_durations_t_star"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or not values:
                raise ValueError(f"{name} must be nonempty, sorted and unique")
            for index, value in enumerate(values):
                _positive_decimal(value, field_name=f"{name}[{index}]")
        expected_action_amplitudes = (
            RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_AMPLITUDES
            if reduced_runtime_profile
            else tuple(Decimal(value) for value in ("0.2", "0.4", "0.6", "0.8", "1"))
        )
        expected_action_durations = (
            RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_DURATIONS
            if reduced_runtime_profile
            else tuple(Decimal(value) for value in ("0.01", "0.025", "0.05", "0.1", "0.2"))
        )
        if (
            self.action_amplitudes_u_star != expected_action_amplitudes
            or self.action_durations_t_star != expected_action_durations
        ):
            raise ValueError("receiver-history action panel differs from its frozen profile")
        if self.panel_endpoint_t_star != Decimal("0.20"):
            raise ValueError("receiver-history panel endpoint differs")
        if self.diagnostic_times_t_star != (
            Decimal("0.005"),
            Decimal("0.01"),
            Decimal("0.02"),
        ):
            raise ValueError("receiver-history diagnostic-time roster differs")
        if (
            self.bootstrap_resamples != 10_000
            or self.bootstrap_seed != 7_031_419
            or self.untouched_bootstrap_seed != 7_031_420
            or self.alignment_bootstrap_seed != 7_031_421
        ):
            raise ValueError("receiver-history bootstrap contract differs")
        if not Decimal(0) < self.simultaneous_alpha < Decimal(1):
            raise ValueError("receiver-history simultaneous alpha is invalid")
        if self.majority_opposition_threshold != Decimal("0.5"):
            raise ValueError("receiver-history majority-opposition threshold differs")
        if (
            self.descriptor_algorithm_id != 'receiver-history-latent-field-centre-edge'
            or self.history_rank_algorithm_id != 'receiver-history-fixed-k-b-history'
            or self.targeting_tie_break_rule_id != 'receiver-history-stable-lexical-boundary'
            or self.random_comparator_algorithm_id
            != 'receiver-history-outcome-blind-fibre-pcg64'
            or self.inference_algorithm_id
            not in {
                'receiver-history-full-action-menu-complete-unit-phase-diagram',
                RECEIVER_HISTORY_REDUCED_ACTION_MENU_INFERENCE_ID,
            }
            or self.structural_rank_algorithm_id != 'receiver-history-observability-jet-matching'
            or self.discrete_rank_algorithm_id
            != 'arb-inertia-exact-dyadic-v2'
            or self.conditioning_algorithm_id != 'receiver-history-absolute-resolution-svd'
            or self.preparation_sampler_algorithm_id != 'not-requested'
            or self.alignment_algorithm_id != 'receiver-history-bracket-alignment'
            or self.false_promotion_rule_id != 'receiver-history-unsafe-first-lexicographic'
        ):
            raise ValueError("receiver-history algorithm identity differs")
        if reduced_runtime_profile != (
            self.inference_algorithm_id == RECEIVER_HISTORY_REDUCED_ACTION_MENU_INFERENCE_ID
        ):
            raise ValueError("receiver-history reduced runtime profile/inference identity differs")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("receiver-history evidence ceiling differs")
        expected_access = {
            ReceiverHistoryPhase.NOMINATION: OutcomeAccess.OUTCOME_BLIND,
            ReceiverHistoryPhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            ReceiverHistoryPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            ReceiverHistoryPhase.TARGETED_EVALUATION: OutcomeAccess.EVALUATION_SEALED,
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        expected_visibility = {
            ReceiverHistoryPhase.NOMINATION: VisibilityCeiling.PROSPECTIVE,
            ReceiverHistoryPhase.CANARY: VisibilityCeiling.DEVELOPMENT_ONLY,
            ReceiverHistoryPhase.DEVELOPMENT: VisibilityCeiling.DEVELOPMENT_ONLY,
            ReceiverHistoryPhase.TARGETED_EVALUATION: VisibilityCeiling.PROSPECTIVE,
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: VisibilityCeiling.PROSPECTIVE,
        }[self.phase]
        if (
            self.outcome_access is not expected_access
            or self.visibility_ceiling is not expected_visibility
        ):
            raise ValueError("receiver-history phase outcome/visibility contract differs")
        if self.physical_execution_authorized or self.requests_controller or not self.nonactuating:
            raise ValueError("receiver-history must remain nonactuating simulator-only work")


@dataclass(frozen=True, slots=True)
class ReceiverHistorySeedRosterCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-seed-roster-commitment'

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
        if len(self.evaluation_unit_ids) not in {36, 90}:
            raise ValueError(
                "receiver-history evaluation roster must contain an accepted complete-unit count"
            )
        validate_sha256(self.seed_payload_sha256, field_name="seed_payload_sha256")
        validate_sha256(self.commitment_sha256, field_name="commitment_sha256")
        if self.seed_bytes_per_unit != 32 or self.public_seed_count != 0:
            raise ValueError("receiver-history commitment must not disclose evaluation seeds")
        if self.outcome_count_at_commitment != 0:
            raise ValueError("receiver-history roster commitment must precede outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDenominatorDescriptor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-denominator-descriptor'

    descriptor_id: str
    unit_id: str
    family: ReceiverHistoryDisorderFamily
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
            raise ValueError("receiver-history descriptor scale differs")
        if len(self.capacitances_farads) != self.scale_cells:
            raise ValueError("receiver-history capacitance roster differs")
        if len(self.interior_resistances_ohms) != self.scale_cells - 1:
            raise ValueError("receiver-history resistance roster differs")
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
            raise ValueError("receiver-history descriptor numerical contract differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryRankStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-rank-step'

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
            raise ValueError("receiver-history history depth differs")
        if not 0 <= self.effective_rank <= self.row_count_ceiling:
            raise ValueError("receiver-history effective rank exceeds its ceiling")
        if not 0 <= self.algebraic_comparator_rank <= self.row_count_ceiling:
            raise ValueError("receiver-history algebraic rank exceeds its ceiling")
        _positive_decimal(self.largest_singular_value, field_name="largest_singular_value")
        validate_decimal(
            self.smallest_retained_relative_singular_value,
            field_name="smallest_retained_relative_singular_value",
            minimum=Decimal(0),
        )
        if self.condition_number is not None:
            _positive_decimal(self.condition_number, field_name="condition_number")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryHistoryRankForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-history-rank-forecast'

    forecast_id: str
    descriptor_sha256: str
    rank_steps: tuple[ReceiverHistoryRankStep, ...]
    singular_spectra_sha256: str
    k_full_effective: int | None
    effective_rank_right_censored: bool
    probe_depths: tuple[int, ...]
    algebraic_prediction_state: ReceiverHistoryScientificState

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        validate_sha256(self.descriptor_sha256, field_name="descriptor_sha256")
        validate_sha256(self.singular_spectra_sha256, field_name="singular_spectra_sha256")
        if tuple(value.depth for value in self.rank_steps) != self.probe_depths:
            raise ValueError("receiver-history rank forecast must contain only frozen probe depths")
        if self.k_full_effective is None:
            if not self.effective_rank_right_censored:
                raise ValueError("missing full-rank depth must be right-censored")
        elif not 0 <= self.k_full_effective <= 31 or self.effective_rank_right_censored:
            raise ValueError("receiver-history full-rank depth/censoring differs")
        if self.probe_depths != tuple(sorted(set(self.probe_depths))):
            raise ValueError("receiver-history probe depths must be sorted and unique")
        if self.algebraic_prediction_state not in {
            ReceiverHistoryScientificState.SUPPORTED,
            ReceiverHistoryScientificState.OPPOSED,
        }:
            raise ValueError("receiver-history algebraic prediction state differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryActionPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-action-prediction'

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
            raise ValueError("receiver-history predicted admission is not a gate intersection")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryChallengeGeometry(CanonicalRecord):
    'Generator-facing targeted geometry with no observer prediction or certificate.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-challenge-geometry'

    nomination_id: str
    coordinate_id: str
    unit_id: str
    scale_cells: int
    depth: int
    resolution_epsilon: Decimal
    pair_index: int
    pair_kind: ReceiverHistoryPairKind
    gate: ReceiverHistoryGate
    orientation: ReceiverHistoryOrientation | None
    targeting_action_id: str | None
    center_values: tuple[Decimal, ...]
    amplitude_volts: Decimal
    mode_values: tuple[Decimal, ...]
    lexical_reference_side: str
    constraint_family_sha256: str
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        for name in ("nomination_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.targeting_action_id is not None:
            validate_stable_id(self.targeting_action_id, field_name="targeting_action_id")
        decision_pair = self.pair_kind in {
            ReceiverHistoryPairKind.TARGET_LEXICAL,
            ReceiverHistoryPairKind.SINK_LEXICAL,
        }
        if decision_pair != (self.orientation is not None):
            raise ValueError("receiver-history lexical geometry orientation differs")
        if (
            self.scale_cells not in {16, 32, 64, 128, 256}
            or not 0 <= self.depth <= 31
            or self.pair_index < 0
            or len(self.center_values) != self.scale_cells
            or len(self.mode_values) != self.scale_cells
        ):
            raise ValueError("receiver-history challenge geometry roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        _positive_decimal(self.amplitude_volts, field_name="amplitude_volts")
        for field_name, values in (
            ("center_values", self.center_values),
            ("mode_values", self.mode_values),
        ):
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{field_name}[{index}]")
        if self.lexical_reference_side != "PLUS":
            raise ValueError("receiver-history challenge geometry must freeze PLUS as reference")
        validate_sha256(self.constraint_family_sha256, field_name="constraint_family_sha256")
        if self.outcome_count_at_freeze != 0:
            raise ValueError("receiver-history challenge geometry must precede outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryChallengeNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-challenge-nomination'

    nomination_id: str
    coordinate_id: str
    cohort: ReceiverHistoryCohort
    unit_id: str
    scale_cells: int
    depth: int
    resolution_epsilon: Decimal
    pair_index: int
    pair_kind: ReceiverHistoryPairKind
    gate: ReceiverHistoryGate
    orientation: ReceiverHistoryOrientation | None
    targeting_action_id: str | None
    carrier_volts: Decimal
    center_values: tuple[Decimal, ...]
    amplitude_volts: Decimal
    mode_values: tuple[Decimal, ...]
    predicted_coordinate_defect: Decimal
    predicted_maximum_future_receiver_defect: Decimal
    predicted_maximum_future_metric_defect: Decimal
    action_predictions: tuple[ReceiverHistoryActionPrediction, ...]
    observer_implementation_sha256: str
    outcome_count_at_nomination: int

    def __post_init__(self) -> None:
        for name in ("nomination_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.targeting_action_id is not None:
            validate_stable_id(self.targeting_action_id, field_name="targeting_action_id")
        if self.scale_cells not in {16, 32, 64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("receiver-history nomination scale/depth differs")
        if self.cohort is not ReceiverHistoryCohort.TARGETED:
            raise ValueError('receiver-history challenge nomination must remain in cohort targeted')
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if (
            self.pair_index < 0
            or len(self.mode_values) != self.scale_cells
            or len(self.center_values) != self.scale_cells
        ):
            raise ValueError("receiver-history nomination mode roster differs")
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
            raise ValueError("receiver-history nomination amplitude must be positive")
        if len(self.action_predictions) not in {9, 25}:
            raise ValueError("receiver-history nomination must carry the complete action panel")
        action_ids = tuple(value.action_id for value in self.action_predictions)
        _sorted_unique(action_ids, field_name="action_predictions")
        validate_sha256(
            self.observer_implementation_sha256,
            field_name="observer_implementation_sha256",
        )
        if self.outcome_count_at_nomination != 0:
            raise ValueError("receiver-history nomination cannot observe generator outcomes")
        expected_gate = {
            ReceiverHistoryPairKind.DYNAMICAL_BOUNDARY: ReceiverHistoryGate.DYNAMICAL,
            ReceiverHistoryPairKind.TARGET_LEXICAL: ReceiverHistoryGate.TARGET,
            ReceiverHistoryPairKind.SINK_LEXICAL: ReceiverHistoryGate.SINK,
            ReceiverHistoryPairKind.RANDOM_FIBRE: ReceiverHistoryGate.RANDOM,
        }[self.pair_kind]
        if self.gate is not expected_gate:
            raise ValueError("receiver-history pair kind and gate differ")
        if (
            self.pair_kind
            in {ReceiverHistoryPairKind.RANDOM_FIBRE, ReceiverHistoryPairKind.DYNAMICAL_BOUNDARY}
            and self.targeting_action_id is not None
        ) or (
            self.pair_kind
            in {ReceiverHistoryPairKind.TARGET_LEXICAL, ReceiverHistoryPairKind.SINK_LEXICAL}
            and self.targeting_action_id is None
        ):
            raise ValueError("receiver-history targeting action and pair kind differ")
        decision_pair = self.pair_kind in {
            ReceiverHistoryPairKind.TARGET_LEXICAL,
            ReceiverHistoryPairKind.SINK_LEXICAL,
        }
        if decision_pair != (self.orientation is not None):
            raise ValueError("receiver-history lexical nomination orientation differs")
        if any(value.requested_action_id != value.action_id for value in self.action_predictions):
            raise ValueError("receiver-history requested action identity differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryMethodFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-method-freeze'

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
            raise ValueError("receiver-history method freeze must precede evaluation outcomes")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("receiver-history method-freeze ceiling differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryGeneratorActionOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-generator-action-outcome'

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
            raise ValueError("receiver-history realized admission is not a gate intersection")
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
            raise ValueError("receiver-history action parity flag is not derived")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryGeneratorPairOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-generator-pair-outcome'

    nomination_id: str
    realized_coordinate_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    future_receiver_defects: tuple[Decimal, ...]
    action_outcomes: tuple[ReceiverHistoryGeneratorActionOutcome, ...]
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
            raise ValueError("receiver-history generator future diagnostic roster differs")
        for index, value in enumerate(self.future_receiver_defects):
            validate_decimal(
                value, field_name=f"future_receiver_defects[{index}]", minimum=Decimal(0)
            )
        if len(self.action_outcomes) not in {9, 25}:
            raise ValueError(
                "receiver-history generator outcome requires the complete action panel"
            )
        action_ids = tuple(value.action_id for value in self.action_outcomes)
        _sorted_unique(action_ids, field_name="action_outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryGeneratorOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-generator-outcome'

    outcome_id: str
    unit_id: str
    descriptor_sha256: str
    nomination_set_sha256: str
    generator_implementation_sha256: str
    pair_outcomes: tuple[ReceiverHistoryGeneratorPairOutcome, ...]
    sparse_certificate_checks: tuple[ReceiverHistorySparseCertificateCheck, ...]
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
        objective_keys = tuple(value.objective_key for value in self.sparse_certificate_checks)
        _sorted_unique(objective_keys, field_name="sparse_certificate_checks")
        if self.array_count <= 0 or self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("receiver-history generator outcome access/array roster differs")
        if self.scientific_verdict_assigned:
            raise ValueError("receiver-history generator cannot assign a scientific verdict")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedGeneratorOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-generator-outcome'

    outcome_id: str
    unit_id: str
    scale_cells: int
    descriptor_sha256: str
    preparation_manifest_sha256: str
    generator_implementation_sha256: str
    preparation_count: int
    valid_preparation_count: int
    action_ids: tuple[str, ...]
    complete_collision_graph_sha256: str
    validity_mask_sha256: str
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
            "complete_collision_graph_sha256",
            "validity_mask_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256} or self.preparation_count != 512:
            raise ValueError("receiver-history untouched generator roster differs")
        if not 0 <= self.valid_preparation_count <= self.preparation_count:
            raise ValueError("receiver-history untouched generator valid count differs")
        _sorted_unique(self.action_ids, field_name="action_ids")
        if len(self.action_ids) != 25:
            raise ValueError(
                "receiver-history untouched generator requires the complete action panel"
            )
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("receiver-history untouched generator access differs")
        if not self.requested_accepted_applied_realized_parity:
            raise ValueError("receiver-history untouched generator action identity differs")
        if self.scientific_verdict_assigned:
            raise ValueError("receiver-history untouched generator cannot assign a verdict")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedCoordinateAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-coordinate-adjudication'

    adjudication_id: str
    coordinate_id: str
    unit_id: str
    family: ReceiverHistoryDisorderFamily
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
    scientific_state: ReceiverHistoryScientificState
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
            raise ValueError("receiver-history untouched adjudication roster differs")
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
                raise ValueError(
                    "invalid receiver-history untouched unit cannot encode zero evidence"
                )
            if self.scientific_state is not ReceiverHistoryScientificState.INVALID:
                raise ValueError("invalid receiver-history untouched unit has another state")
            return
        if any(value is None for value in counts + metrics):
            raise ValueError("valid receiver-history untouched adjudication is incomplete")
        integer_counts = tuple(int(value) for value in counts if value is not None)
        edge_count = integer_counts[0]
        if min((*integer_counts, *self.prefix_edge_counts)) < 0:
            raise ValueError("receiver-history untouched counts cannot be negative")
        if tuple(sorted(self.prefix_edge_counts)) != self.prefix_edge_counts:
            raise ValueError("receiver-history prefix encounter counts must be nested")
        if self.prefix_edge_counts[-1] != edge_count:
            raise ValueError('receiver-history 512-preparation prefix must equal the full graph')
        if any(value > edge_count for value in integer_counts[1:]):
            raise ValueError("receiver-history adverse edge count exceeds collision edges")
        for index, value in enumerate(metrics):
            assert value is not None
            validate_decimal(value, field_name=f"maximum_defect[{index}]", minimum=Decimal(0))
        expected = (
            ReceiverHistoryScientificState.UNSAFE_FALSE_PROMOTION
            if integer_counts[4]
            else ReceiverHistoryScientificState.OPPOSED
            if integer_counts[1] or integer_counts[2]
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if edge_count
            else ReceiverHistoryScientificState.NO_ENCOUNTER_AT_512_PREPARATIONS
        )
        if self.scientific_state is not expected or not self.unsafe_first:
            raise ValueError("receiver-history untouched state violates unsafe-first precedence")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryPairAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-pair-adjudication'

    nomination_id: str
    coordinate_id: str
    depth: int
    resolution_epsilon: Decimal
    gate: ReceiverHistoryGate
    endpoint: ReceiverHistoryEndpoint | None
    orientation: ReceiverHistoryOrientation | None
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
    endpoint_adverse: bool
    decision_disposition: ReceiverHistoryDecisionDisposition
    prediction_agreement: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if not 0 <= self.depth <= 31:
            raise ValueError("receiver-history pair depth differs")
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
            raise ValueError("receiver-history mismatch count differs")
        if self.false_safe_count + self.false_hold_count != self.decision_mismatch_count:
            raise ValueError("receiver-history false-safe/hold counts do not partition decisions")
        if self.unsafe_false_promotion_count != self.false_safe_count:
            raise ValueError(
                "receiver-history unsafe-promotion count differs from false-safe continuity"
            )
        if self.decision_adverse != (self.collision_confirmed and self.decision_mismatch_count > 0):
            raise ValueError("receiver-history decision-adverse state is not derived")
        expected_endpoint = {
            ReceiverHistoryGate.DYNAMICAL: ReceiverHistoryEndpoint.DYNAMICAL,
            ReceiverHistoryGate.TARGET: ReceiverHistoryEndpoint.TARGET_DECISION,
            ReceiverHistoryGate.SINK: ReceiverHistoryEndpoint.SINK_DECISION,
            ReceiverHistoryGate.RANDOM: None,
        }[self.gate]
        if self.endpoint is not expected_endpoint:
            raise ValueError("receiver-history pair endpoint is not gate-derived")
        decision_endpoint = self.endpoint in {
            ReceiverHistoryEndpoint.TARGET_DECISION,
            ReceiverHistoryEndpoint.SINK_DECISION,
        }
        if decision_endpoint != (self.orientation is not None):
            raise ValueError("receiver-history pair orientation differs from its endpoint")
        expected_endpoint_adverse = (
            self.dynamically_adverse
            if self.endpoint is ReceiverHistoryEndpoint.DYNAMICAL
            else self.decision_adverse
            if decision_endpoint
            else False
        )
        if self.endpoint_adverse != expected_endpoint_adverse:
            raise ValueError("receiver-history pair endpoint-adverse state is not derived")
        expected_disposition = (
            ReceiverHistoryDecisionDisposition.UNSAFE_FALSE_PROMOTION
            if self.unsafe_false_promotion_count
            else (
                ReceiverHistoryDecisionDisposition.FALSE_HOLD
                if self.false_hold_count
                else ReceiverHistoryDecisionDisposition.CORRECT_ACTION
            )
        )
        if self.decision_disposition is not expected_disposition:
            raise ValueError(
                "receiver-history decision disposition violates unsafe-first precedence"
            )


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDepthAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-depth-adjudication'

    depth: int
    requested_pair_count: int
    realized_collision_count: int
    dynamic_adverse_count: int
    decision_adverse_count: int
    false_safe_count: int
    dynamical_state: ReceiverHistoryScientificState
    decision_state: ReceiverHistoryScientificState

    def __post_init__(self) -> None:
        counts = (
            self.requested_pair_count,
            self.realized_collision_count,
            self.dynamic_adverse_count,
            self.decision_adverse_count,
            self.false_safe_count,
        )
        if not 0 <= self.depth <= 31 or min(counts) < 0:
            raise ValueError("receiver-history depth adjudication differs")
        if (
            not self.dynamic_adverse_count
            <= self.realized_collision_count
            <= self.requested_pair_count
        ):
            raise ValueError("receiver-history dynamical depth counts differ")
        if not self.decision_adverse_count <= self.realized_collision_count:
            raise ValueError("receiver-history decision depth counts differ")
        expected_dynamical_state = (
            ReceiverHistoryScientificState.OPPOSED
            if self.dynamic_adverse_count
            else ReceiverHistoryScientificState.SUPPORTED
            if self.realized_collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            ReceiverHistoryScientificState.OPPOSED
            if self.decision_adverse_count
            else ReceiverHistoryScientificState.SUPPORTED
            if self.realized_collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("receiver-history depth scientific states are not derived")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryTargetedCoordinateAdjudication(CanonicalRecord):
    'Bounded targeted-cohort state for one scale, coordinate and endpoint pair.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-targeted-coordinate-adjudication'

    adjudication_id: str
    coordinate_id: str
    unit_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    coordinate_kind: ReceiverHistoryCoordinateKind
    depth: int
    budget: Decimal | None
    resolution_epsilon: Decimal
    primary: bool
    valid: bool
    eligible_endpoints: tuple[ReceiverHistoryEndpoint, ...]
    nominated_pair_count: int | None
    realized_collision_count: int | None
    dynamic_adverse_count: int | None
    target_adverse_count: int | None
    sink_adverse_count: int | None
    decision_adverse_count: int | None
    target_unsafe_false_promotion_count: int | None
    target_false_hold_count: int | None
    sink_unsafe_false_promotion_count: int | None
    sink_false_hold_count: int | None
    unsafe_false_promotion_count: int | None
    false_hold_count: int | None
    dynamical_state: ReceiverHistoryScientificState
    target_state: ReceiverHistoryScientificState
    sink_state: ReceiverHistoryScientificState
    decision_state: ReceiverHistoryScientificState
    dynamical_certificate_complete: bool
    target_certificate_complete: bool
    sink_certificate_complete: bool
    decision_certificate_complete: bool
    generator_observer_agreement: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "coordinate_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.eligible_endpoints != tuple(
            sorted(set(self.eligible_endpoints), key=lambda value: value.value)
        ):
            raise ValueError("receiver-history eligible endpoint mask is not canonical")
        if self.scale_cells not in {64, 128, 256} or not 0 <= self.depth <= 31:
            raise ValueError("receiver-history targeted coordinate roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        if self.coordinate_kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH:
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
            self.target_adverse_count,
            self.sink_adverse_count,
            self.decision_adverse_count,
            self.target_unsafe_false_promotion_count,
            self.target_false_hold_count,
            self.sink_unsafe_false_promotion_count,
            self.sink_false_hold_count,
            self.unsafe_false_promotion_count,
            self.false_hold_count,
        )
        _sorted_unique(self.reason_codes, field_name="reason_codes")
        if not self.valid:
            if any(value is not None for value in counts):
                raise ValueError("invalid targeted coordinate cannot encode zero evidence")
            if (
                self.dynamical_state is not ReceiverHistoryScientificState.INVALID
                or self.target_state is not ReceiverHistoryScientificState.INVALID
                or self.sink_state is not ReceiverHistoryScientificState.INVALID
                or self.decision_state is not ReceiverHistoryScientificState.INVALID
                or self.generator_observer_agreement is not None
                or self.dynamical_certificate_complete
                or self.target_certificate_complete
                or self.sink_certificate_complete
                or self.decision_certificate_complete
            ):
                raise ValueError("invalid targeted coordinate has evidentiary state")
            if not self.reason_codes:
                raise ValueError("invalid targeted coordinate lacks a reason")
            return
        if any(value is None for value in counts) or self.generator_observer_agreement is None:
            raise ValueError("valid targeted coordinate is incomplete")
        integer_counts = tuple(int(value) for value in counts if value is not None)
        (
            nominated,
            realized,
            dynamic,
            target,
            sink,
            decision,
            target_unsafe,
            target_hold,
            sink_unsafe,
            sink_hold,
            unsafe,
            false_hold,
        ) = integer_counts
        if (
            min(integer_counts) < 0
            or realized > nominated
            or max(
                dynamic,
                target,
                sink,
                decision,
                target_unsafe,
                target_hold,
                sink_unsafe,
                sink_hold,
                unsafe,
                false_hold,
            )
            > realized
            or decision != target + sink
            or unsafe != target_unsafe + sink_unsafe
            or false_hold != target_hold + sink_hold
            or target != target_unsafe + target_hold
            or sink != sink_unsafe + sink_hold
        ):
            raise ValueError("targeted coordinate denominator chain differs")
        if (
            (
                ReceiverHistoryEndpoint.DYNAMICAL not in self.eligible_endpoints
                and (dynamic or self.dynamical_certificate_complete)
            )
            or (
                ReceiverHistoryEndpoint.TARGET_DECISION not in self.eligible_endpoints
                and (target or target_unsafe or target_hold or self.target_certificate_complete)
            )
            or (
                ReceiverHistoryEndpoint.SINK_DECISION not in self.eligible_endpoints
                and (sink or sink_unsafe or sink_hold or self.sink_certificate_complete)
            )
        ):
            raise ValueError("ineligible endpoint cannot encode evaluation evidence")
        expected_dynamic = (
            ReceiverHistoryScientificState.NOT_ATTEMPTED
            if ReceiverHistoryEndpoint.DYNAMICAL not in self.eligible_endpoints
            else ReceiverHistoryScientificState.OPPOSED
            if dynamic
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if self.dynamical_certificate_complete
            else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
        )
        expected_decision = (
            ReceiverHistoryScientificState.NOT_ATTEMPTED
            if not {
                ReceiverHistoryEndpoint.TARGET_DECISION,
                ReceiverHistoryEndpoint.SINK_DECISION,
            }.intersection(self.eligible_endpoints)
            else ReceiverHistoryScientificState.OPPOSED
            if decision
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if self.decision_certificate_complete
            else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
        )
        expected_target = (
            ReceiverHistoryScientificState.NOT_ATTEMPTED
            if ReceiverHistoryEndpoint.TARGET_DECISION not in self.eligible_endpoints
            else ReceiverHistoryScientificState.OPPOSED
            if target
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if self.target_certificate_complete
            else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
        )
        expected_sink = (
            ReceiverHistoryScientificState.NOT_ATTEMPTED
            if ReceiverHistoryEndpoint.SINK_DECISION not in self.eligible_endpoints
            else ReceiverHistoryScientificState.OPPOSED
            if sink
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if self.sink_certificate_complete
            else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
        )
        if (
            self.dynamical_state is not expected_dynamic
            or self.target_state is not expected_target
            or self.sink_state is not expected_sink
            or self.decision_state is not expected_decision
        ):
            raise ValueError("targeted coordinate state is not certificate-derived")
        if (
            (ReceiverHistoryEndpoint.DYNAMICAL not in self.eligible_endpoints and dynamic)
            or (
                ReceiverHistoryEndpoint.TARGET_DECISION not in self.eligible_endpoints
                and (target or target_unsafe or target_hold)
            )
            or (
                ReceiverHistoryEndpoint.SINK_DECISION not in self.eligible_endpoints
                and (sink or sink_unsafe or sink_hold)
            )
        ):
            raise ValueError("receiver-history ineligible endpoint contains evidence")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUnitAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-unit-adjudication'

    adjudication_id: str
    unit_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    descriptor_valid: bool
    nominated_pair_count: int
    realized_collision_count: int
    evaluable_pair_count: int
    pair_adjudications: tuple[ReceiverHistoryPairAdjudication, ...]
    depth_adjudications: tuple[ReceiverHistoryDepthAdjudication, ...]
    dynamical_state: ReceiverHistoryScientificState
    decision_state: ReceiverHistoryScientificState
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
            raise ValueError("receiver-history unit counts/scale differ")
        if (
            not self.evaluable_pair_count
            <= self.realized_collision_count
            <= self.nominated_pair_count
        ):
            raise ValueError("receiver-history unit denominator chain differs")
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
            raise ValueError("receiver-history unit counts do not match pair adjudications")
        if tuple(value.depth for value in self.depth_adjudications) != tuple(
            sorted({value.depth for value in self.depth_adjudications})
        ):
            raise ValueError("receiver-history depth adjudications must be sorted and unique")
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
            raise ValueError("receiver-history depth ledger does not aggregate the pair ledger")
        expected_dynamical_state = (
            ReceiverHistoryScientificState.OPPOSED
            if any(value.dynamically_adverse for value in self.pair_adjudications)
            else ReceiverHistoryScientificState.SUPPORTED
            if self.realized_collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            ReceiverHistoryScientificState.OPPOSED
            if any(value.decision_adverse for value in self.pair_adjudications)
            else ReceiverHistoryScientificState.SUPPORTED
            if self.realized_collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("receiver-history unit scientific states are not derived")
        if self.generator_observer_agreement != all(
            value.prediction_agreement for value in self.pair_adjudications
        ):
            raise ValueError("receiver-history generator/observer agreement is not derived")
        if self.k_full_effective is not None and not 0 <= self.k_full_effective <= 31:
            raise ValueError("receiver-history full-rank depth differs")
        if self.b_full is not None:
            validate_decimal(self.b_full, field_name="b_full", minimum=Decimal(0))
            if self.b_full > 1:
                raise ValueError("receiver-history normalized history budget exceeds one")
        validate_decimal(
            self.rank_curve_distance, field_name="rank_curve_distance", minimum=Decimal(0)
        )
        _sorted_unique(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryCellRecurrence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-cell-recurrence'

    cell_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    coordinate_label: str
    coordinate_kind: ReceiverHistoryCoordinateKind
    depth: int
    budget: Decimal | None
    endpoint_id: str
    requested_count: int
    invalid_count: int
    opposed_count: int
    informative_nonadverse_count: int
    targetability_limited_count: int
    prerequisite_nonattempt_count: int
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
            self.prerequisite_nonattempt_count,
            self.unsafe_false_promotion_unit_count,
            self.false_hold_unit_count,
        )
        if self.scale_cells not in {64, 128, 256} or not 0 <= self.depth <= 31 or min(counts) < 0:
            raise ValueError("receiver-history recurrence cell differs")
        if self.requested_count not in {12, 30} or sum(counts[1:6]) != self.requested_count:
            raise ValueError("receiver-history primary cell denominator differs")
        if (
            max(self.unsafe_false_promotion_unit_count, self.false_hold_unit_count)
            > self.requested_count
        ):
            raise ValueError("receiver-history decision-error unit count exceeds requests")
        if self.coordinate_kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH:
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
                raise ValueError("receiver-history exact proportion interval exceeds one")
        if (
            self.opposition_lower_bound > self.opposition_upper_bound
            or self.informative_nonadverse_lower_bound > self.informative_nonadverse_upper_bound
        ):
            raise ValueError("receiver-history exact proportion interval is reversed")
        if self.existence_recurs != (self.opposed_count > 0):
            raise ValueError("receiver-history cell existence flag is not derived")
        if self.majority_opposition != (self.opposition_lower_bound > Decimal("0.5")):
            raise ValueError("receiver-history cell majority flag is not derived")
        if self.majority_informative_nonadverse != (
            self.informative_nonadverse_lower_bound > Decimal("0.5")
        ):
            raise ValueError("receiver-history informative-nonadverse majority flag is not derived")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedPrevalenceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-prevalence-cell'

    cell_id: str
    family: ReceiverHistoryDisorderFamily
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
    any_adverse_disposition: ReceiverHistoryScientificState
    unsafe_disposition: ReceiverHistoryScientificState

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
            raise ValueError('receiver-history untouched prevalence cell differs')
        if max(counts[2:]) > self.encounter_count:
            raise ValueError('receiver-history untouched adverse count exceeds encounters')
        intervals = (
            self.any_adverse_lower_bound,
            self.any_adverse_upper_bound,
            self.unsafe_lower_bound,
            self.unsafe_upper_bound,
        )
        if self.invalid_count:
            if any(value is not None for value in intervals):
                raise ValueError('invalid receiver-history untouched denominator cannot encode an interval')
            if (
                self.any_adverse_disposition is not ReceiverHistoryScientificState.UNEVALUABLE
                or self.unsafe_disposition is not ReceiverHistoryScientificState.UNEVALUABLE
            ):
                raise ValueError('invalid receiver-history untouched denominator must remain unevaluable')
            return
        if any(value is None for value in intervals):
            raise ValueError('valid receiver-history untouched denominator lacks an interval')
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
                raise ValueError('receiver-history untouched interval exceeds one')
        assert self.any_adverse_lower_bound is not None
        assert self.any_adverse_upper_bound is not None
        assert self.unsafe_lower_bound is not None
        assert self.unsafe_upper_bound is not None
        if (
            self.any_adverse_lower_bound > self.any_adverse_upper_bound
            or self.unsafe_lower_bound > self.unsafe_upper_bound
        ):
            raise ValueError('receiver-history untouched interval is reversed')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedDescriptiveSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-descriptive-summary'

    summary_id: str
    family: ReceiverHistoryDisorderFamily
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
            raise ValueError('receiver-history untouched descriptive summary roster differs')
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
                    raise ValueError("receiver-history collision-edge fraction exceeds one")
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
class ReceiverHistoryAlignmentSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-alignment-summary'

    summary_id: str
    endpoint_id: str
    family_id: str
    requested_unit_count: int
    evaluable_unit_count: int
    mean_delta: Decimal | None
    lower_95: Decimal | None
    upper_95: Decimal | None
    disposition: ReceiverHistoryScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id", "family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.requested_unit_count not in {12, 30, 36, 90}
            or not 0 <= self.evaluable_unit_count <= self.requested_unit_count
        ):
            raise ValueError("receiver-history alignment denominator differs")
        values = (self.mean_delta, self.lower_95, self.upper_95)
        if self.evaluable_unit_count == 0:
            if any(value is not None for value in values) or self.disposition not in {
                ReceiverHistoryScientificState.INVALID,
                ReceiverHistoryScientificState.UNEVALUABLE,
            }:
                raise ValueError("empty receiver-history alignment is evidentiary")
        elif any(value is None for value in values):
            raise ValueError("evaluable receiver-history alignment lacks its interval")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryResolutionPanelSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-resolution-panel-summary'

    summary_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    depth: int
    endpoint_id: str
    requested_count: int
    invalid_count: int
    changed_state_count: int
    targetability_changed_count: int
    disposition: ReceiverHistoryScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.scale_cells not in {64, 128, 256}
            or self.depth not in {4, 6, 8}
            or self.requested_count not in {12, 30}
            or min(self.invalid_count, self.changed_state_count, self.targetability_changed_count)
            < 0
            or max(self.invalid_count, self.changed_state_count, self.targetability_changed_count)
            > self.requested_count
        ):
            raise ValueError("receiver-history resolution-panel summary differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryRankObjectSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-rank-object-summary'

    summary_id: str
    family: ReceiverHistoryDisorderFamily
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
            or self.requested_count not in {12, 30}
            or sum(counts) != self.requested_count
            or not 0 <= self.conditioning_right_censored_count <= self.requested_count
        ):
            raise ValueError("receiver-history rank-object summary roster differs")
        _positive_decimal(self.resolution_epsilon, field_name="resolution_epsilon")
        for name in (
            "structural_rank_median",
            "discrete_lower_rank_median",
            "discrete_upper_rank_median",
            "effective_rank_median",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class ReceiverHistoryTransitionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-transition-summary'

    summary_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    coordinate_kind: ReceiverHistoryCoordinateKind
    endpoint_id: str
    ordered_coordinate_labels: tuple[str, ...]
    last_opposed_coordinate: str | None
    first_informative_nonadverse_coordinate: str | None
    targetability_limited_coordinates: tuple[str, ...]
    reversal_pairs: tuple[str, ...]
    disposition: ReceiverHistoryScientificState

    def __post_init__(self) -> None:
        for name in ("summary_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256}:
            raise ValueError("receiver-history transition scale differs")
        if not self.ordered_coordinate_labels:
            raise ValueError("receiver-history transition grid is empty")
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
            self.disposition is ReceiverHistoryScientificState.NONMONOTONE_PHASE_PATTERN
            and not self.reversal_pairs
        ):
            raise ValueError("receiver-history transition reversal disposition differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryRecurrenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-recurrence-result'

    result_id: str
    method_freeze_sha256: str
    power_atlas_sha256: str
    cells: tuple[ReceiverHistoryCellRecurrence, ...]
    secondary_cells: tuple[ReceiverHistoryCellRecurrence, ...]
    untouched_cells: tuple[ReceiverHistoryUntouchedPrevalenceCell, ...]
    untouched_descriptive_summaries: tuple[ReceiverHistoryUntouchedDescriptiveSummary, ...]
    alignment_summaries: tuple[ReceiverHistoryAlignmentSummary, ...]
    resolution_summaries: tuple[ReceiverHistoryResolutionPanelSummary, ...]
    rank_summaries: tuple[ReceiverHistoryRankObjectSummary, ...]
    transition_summaries: tuple[ReceiverHistoryTransitionSummary, ...]
    fixed_depth_dynamical_bracket: ReceiverHistoryScientificState
    fixed_depth_target_bracket: ReceiverHistoryScientificState
    fixed_depth_sink_bracket: ReceiverHistoryScientificState
    normalized_budget_dynamical_bracket: ReceiverHistoryScientificState
    normalized_budget_target_bracket: ReceiverHistoryScientificState
    normalized_budget_sink_bracket: ReceiverHistoryScientificState
    bootstrap_summary_sha256: str
    decisive_counterexample_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    physical_claim_allowed: bool
    controller_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        validate_sha256(self.power_atlas_sha256, field_name="power_atlas_sha256")
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
            _sorted_unique(
                ids,
                field_name=field_name,
                allow_empty=field_name
                in {
                    "secondary_cells",
                    "untouched_cells",
                    "untouched_descriptive_summaries",
                    "resolution_summaries",
                    "rank_summaries",
                },
            )
        if len(self.cells) != 135 or len(self.untouched_cells) not in {0, 36}:
            raise ValueError(
                'receiver-history result requires 135 targeted and zero-or-36 untouched primary cells'
            )
        if self.secondary_cells:
            raise ValueError("receiver-history primary profile excludes secondary coordinate cells")
        if len(self.untouched_descriptive_summaries) not in {0, 144} or bool(
            self.untouched_descriptive_summaries
        ) != bool(self.untouched_cells):
            raise ValueError('receiver-history result has an incomplete staged untouched roster')
        if len(self.alignment_summaries) != 12 or self.resolution_summaries:
            raise ValueError(
                "receiver-history primary alignment or excluded-resolution roster differs"
            )
        if self.rank_summaries:
            raise ValueError("receiver-history primary profile excludes rank-object summaries")
        if len(self.transition_summaries) != 54:
            raise ValueError("receiver-history transition-summary roster differs")
        _sorted_unique(self.decisive_counterexample_ids, field_name="decisive_counterexample_ids")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("receiver-history recurrence ceiling differs")
        if self.physical_claim_allowed or self.controller_claim_allowed:
            raise ValueError("receiver-history cannot claim physical recurrence or control")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryPowerTransportCell(CanonicalRecord):
    'Prospective comparison of one frozen development power class with one targeted cell.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-power-transport-cell'

    cell_id: str
    power_cell_id: str
    recurrence_cell_id: str
    endpoint_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    coordinate_label: str
    frozen_power_state: ReceiverHistoryPowerState
    disposition: ReceiverHistoryMetatheoryDisposition
    counterexample_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "power_cell_id",
            "recurrence_cell_id",
            "endpoint_id",
            "coordinate_label",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256}:
            raise ValueError("receiver-history transport scale differs")
        _sorted_unique(self.counterexample_ids, field_name="counterexample_ids")
        _sorted_unique(self.reason_codes, field_name="reason_codes")
        if (
            self.disposition
            in {
                ReceiverHistoryMetatheoryDisposition.OPPOSED,
                ReceiverHistoryMetatheoryDisposition.INVALID,
                ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
            }
            and not self.reason_codes
        ):
            raise ValueError("non-supporting power transport lacks a typed reason")
        if (
            self.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
            and not self.counterexample_ids
        ):
            raise ValueError("opposed power transport lacks a held-out counterexample")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryLocalityCandidateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-locality-candidate-assessment'

    assessment_id: str
    candidate: ReceiverHistoryLocalityCandidate
    coordinate_labels: tuple[str, ...]
    disposition: ReceiverHistoryMetatheoryDisposition
    counterexample_cell_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        _sorted_unique(self.coordinate_labels, field_name="coordinate_labels")
        _sorted_unique(self.counterexample_cell_ids, field_name="counterexample_cell_ids")
        _sorted_unique(self.reason_codes, field_name="reason_codes")
        if self.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED and not (
            self.counterexample_cell_ids and self.reason_codes
        ):
            raise ValueError("opposed locality candidate lacks held-out counterevidence")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryLocalityTournamentResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-locality-tournament-result'

    tournament_id: str
    endpoint_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    assessments: tuple[ReceiverHistoryLocalityCandidateAssessment, ...]
    winning_candidate: ReceiverHistoryLocalityCandidate | None
    disposition: ReceiverHistoryMetatheoryDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("tournament_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {64, 128, 256}:
            raise ValueError("receiver-history tournament scale differs")
        if tuple(value.candidate for value in self.assessments) != tuple(
            ReceiverHistoryLocalityCandidate
        ):
            raise ValueError("receiver-history tournament candidate antichain/order differs")
        _sorted_unique(self.reason_codes, field_name="reason_codes")
        supported = tuple(
            value.candidate
            for value in self.assessments
            if value.disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED
        )
        if self.winning_candidate is not None and self.winning_candidate not in supported:
            raise ValueError("receiver-history tournament winner did not survive")
        if self.disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED:
            if (
                self.winning_candidate
                is not ReceiverHistoryLocalityCandidate.ENDPOINT_CONTEXT_ATLAS
            ):
                raise ValueError("contextual lawhood support requires the richest frozen candidate")
            if any(
                value.disposition is not ReceiverHistoryMetatheoryDisposition.OPPOSED
                for value in self.assessments[:-1]
            ):
                raise ValueError(
                    "contextual winner lacks counterexamples to every simpler candidate"
                )
        elif self.disposition is ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED:
            if not supported:
                raise ValueError("not-distinguished tournament lacks a surviving candidate")
        elif self.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED:
            contextual = self.assessments[-1]
            if contextual.disposition is not ReceiverHistoryMetatheoryDisposition.OPPOSED:
                raise ValueError("opposed tournament lacks a contextual counterexample")
            if self.winning_candidate is not None and self.winning_candidate not in supported:
                raise ValueError("opposed tournament names a nonsurviving simpler candidate")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryHypothesisDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-hypothesis-disposition'

    hypothesis_id: str
    disposition: ReceiverHistoryMetatheoryDisposition
    evidence_ids: tuple[str, ...]
    counterexample_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.hypothesis_id, field_name="hypothesis_id")
        for name in ("evidence_ids", "counterexample_ids", "limitation_ids"):
            _sorted_unique(getattr(self, name), field_name=name)
        if (
            self.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
            and not self.counterexample_ids
        ):
            raise ValueError("opposed receiver-history hypothesis lacks a counterexample")
        if (
            self.disposition
            in {
                ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
                ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
                ReceiverHistoryMetatheoryDisposition.INVALID,
            }
            and not self.limitation_ids
        ):
            raise ValueError("limited receiver-history hypothesis lacks a limitation")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryInferenceDispositionMatrix(CanonicalRecord):
    'Pure phase-diagram synthesis output binding cell transport, tournament, and the six hypotheses results.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-inference-disposition-matrix'

    matrix_id: str
    recurrence_result_sha256: str
    power_atlas_sha256: str
    power_transport_cells: tuple[ReceiverHistoryPowerTransportCell, ...]
    locality_tournaments: tuple[ReceiverHistoryLocalityTournamentResult, ...]
    hypotheses: tuple[ReceiverHistoryHypothesisDisposition, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.matrix_id, field_name="matrix_id")
        validate_sha256(self.recurrence_result_sha256, field_name="recurrence_result_sha256")
        validate_sha256(self.power_atlas_sha256, field_name="power_atlas_sha256")
        if len(self.power_transport_cells) != 135:
            raise ValueError('receiver-history phase-diagram synthesis matrix requires all 135 transport cells')
        if len(self.locality_tournaments) != 27:
            raise ValueError('receiver-history phase-diagram synthesis matrix requires all 27 tournaments')
        if tuple(value.hypothesis_id for value in self.hypotheses) != RECEIVER_HISTORY_HYPOTHESIS_IDS:
            raise ValueError('receiver-history phase-diagram synthesis hypothesis roster differs')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryMetatheoryPropositionDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-metatheory-proposition-disposition'

    proposition_id: str
    disposition: ReceiverHistoryMetatheoryDisposition
    hypothesis_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.proposition_id, field_name="proposition_id")
        for name in ("hypothesis_ids", "evidence_ids", "limitation_ids"):
            _sorted_unique(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class ReceiverHistoryNonclaimDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-nonclaim-disposition'

    nonclaim_id: str
    disposition: ReceiverHistoryMetatheoryDisposition
    reason_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.nonclaim_id, field_name="nonclaim_id")
        validate_stable_id(self.reason_id, field_name="reason_id")
        if self.disposition is not ReceiverHistoryMetatheoryDisposition.EXCLUDED:
            raise ValueError("receiver-history nonclaim must remain explicitly excluded")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryEvidenceGateDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-evidence-gate-disposition'

    evidence_gate_id: str
    disposition: ReceiverHistoryMetatheoryDisposition
    evidence_ceiling: EvidenceCeiling
    reason_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_gate_id, field_name='evidence_gate_id')
        validate_stable_id(self.reason_id, field_name="reason_id")
        if self.evidence_gate_id not in {'measurement-provenance', 'observable-coordinate', 'response-identification', 'law-qualification', 'admission', 'prospective-use'}:
            raise ValueError('receiver-history evidence-gate roster differs')
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError('receiver-history evidence-gate record exceeds its simulator ceiling')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDevelopmentMetatheoryGate(CanonicalRecord):
    'Outcome-blind branch record; terminal empirical metatheory content when no cell is powered.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-development-metatheory-gate'

    gate_id: str
    metatheory_id: str
    power_atlas_sha256: str
    method_freeze_sha256: str
    evaluation_design_sha256: str
    source_closure_sha256: str
    capability_registry_sha256: str
    execution_plan_sha256: str
    eligible_cell_ids: tuple[str, ...]
    all_endpoints_unpowered: bool
    issue_evaluation: bool
    terminal: ReceiverHistoryTerminal | None
    propositions: tuple[ReceiverHistoryMetatheoryPropositionDisposition, ...]
    nonclaims: tuple[ReceiverHistoryNonclaimDisposition, ...]
    evidence_gates: tuple[ReceiverHistoryEvidenceGateDisposition, ...]
    excluded_experiment_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    physical_claim_allowed: bool
    controller_claim_allowed: bool

    def __post_init__(self) -> None:
        for name in ("gate_id", 'metatheory_id'):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "power_atlas_sha256",
            "method_freeze_sha256",
            "evaluation_design_sha256",
            "source_closure_sha256",
            "capability_registry_sha256",
            "execution_plan_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in ("eligible_cell_ids", "excluded_experiment_ids", "limitation_ids"):
            _sorted_unique(getattr(self, name), field_name=name)
        if self.all_endpoints_unpowered != (not self.eligible_cell_ids):
            raise ValueError('development empirical metatheory all-unpowered predicate is not atlas-derived')
        if self.issue_evaluation == self.all_endpoints_unpowered:
            raise ValueError('development empirical metatheory branch decision conflicts with power')
        expected_terminal = (
            ReceiverHistoryTerminal.ALL_ENDPOINTS_UNPOWERED
            if self.all_endpoints_unpowered
            else None
        )
        if self.terminal is not expected_terminal:
            raise ValueError('development empirical metatheory terminal label conflicts with its branch')
        if tuple(value.proposition_id for value in self.propositions) != RECEIVER_HISTORY_METATHEORY_PROPOSITION_IDS:
            raise ValueError('development empirical metatheory proposition roster differs')
        if len(self.nonclaims) != 14 or any(
            value.disposition is not ReceiverHistoryMetatheoryDisposition.EXCLUDED
            for value in self.nonclaims
        ):
            raise ValueError('development empirical metatheory nonclaim register differs')
        if tuple(value.evidence_gate_id for value in self.evidence_gates) != RECEIVER_HISTORY_EVIDENCE_GATE_IDS:
            raise ValueError('development empirical metatheory measurement through prospective-use gates matrix differs')
        empirical = {
            value.proposition_id: value.disposition
            for value in self.propositions
            if value.proposition_id in {'contextual-lawhood', 'development-power-transport', 'endpoint-relative-lawhood', 'history-coordinate-bounds'}
        }
        expected_empirical = (
            ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
            if self.all_endpoints_unpowered
            else ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED
        )
        if set(empirical.values()) != {expected_empirical}:
            raise ValueError('development empirical metatheory empirical dispositions conflict with its branch')
        procedural = {
            value.proposition_id: value.disposition
            for value in self.propositions
            if value.proposition_id in {'evidence-gate-separation', 'prospective-discipline'}
        }
        if set(procedural.values()) != {ReceiverHistoryMetatheoryDisposition.SUPPORTED}:
            raise ValueError('development empirical metatheory procedural dispositions differ')
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError('development empirical metatheory exceeds its simulator evidence ceiling')
        if self.physical_claim_allowed or self.controller_claim_allowed:
            raise ValueError('development empirical metatheory cannot assign physical or controller claims')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryEmpiricalMetatheoryDossier(CanonicalRecord):
    'Immutable empirical metatheory scientific readout, including claims and nonclaims.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-empirical-metatheory-dossier'

    dossier_id: str
    metatheory_id: str
    base_law_id: str
    recurrence_result_sha256: str
    power_atlas_sha256: str
    method_freeze_sha256: str
    source_closure_sha256: str
    capability_registry_sha256: str
    execution_plan_sha256: str
    power_transport_cells: tuple[ReceiverHistoryPowerTransportCell, ...]
    locality_tournaments: tuple[ReceiverHistoryLocalityTournamentResult, ...]
    hypotheses: tuple[ReceiverHistoryHypothesisDisposition, ...]
    propositions: tuple[ReceiverHistoryMetatheoryPropositionDisposition, ...]
    nonclaims: tuple[ReceiverHistoryNonclaimDisposition, ...]
    evidence_gates: tuple[ReceiverHistoryEvidenceGateDisposition, ...]
    excluded_experiment_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    physical_claim_allowed: bool
    controller_claim_allowed: bool

    def __post_init__(self) -> None:
        for name in ("dossier_id", 'metatheory_id', "base_law_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "recurrence_result_sha256",
            "power_atlas_sha256",
            "method_freeze_sha256",
            "source_closure_sha256",
            "capability_registry_sha256",
            "execution_plan_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if len(self.power_transport_cells) != 135:
            raise ValueError('empirical metatheory requires every power-transport cell')
        if len(self.locality_tournaments) != 27:
            raise ValueError('empirical metatheory requires 27 endpoint/family/scale tournaments')
        if tuple(value.hypothesis_id for value in self.hypotheses) != RECEIVER_HISTORY_HYPOTHESIS_IDS:
            raise ValueError('empirical metatheory hypothesis roster differs')
        if tuple(value.proposition_id for value in self.propositions) != RECEIVER_HISTORY_METATHEORY_PROPOSITION_IDS:
            raise ValueError('empirical metatheory proposition roster differs')
        if len(self.nonclaims) != 14 or any(
            value.disposition is not ReceiverHistoryMetatheoryDisposition.EXCLUDED
            for value in self.nonclaims
        ):
            raise ValueError('empirical metatheory nonclaim register differs')
        if tuple(value.evidence_gate_id for value in self.evidence_gates) != RECEIVER_HISTORY_EVIDENCE_GATE_IDS:
            raise ValueError('empirical metatheory measurement through prospective-use gates matrix differs')
        for name in ("excluded_experiment_ids", "limitation_ids"):
            _sorted_unique(getattr(self, name), field_name=name)
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError('empirical metatheory exceeds the entered simulator evidence world')
        if self.physical_claim_allowed or self.controller_claim_allowed:
            raise ValueError('empirical metatheory cannot assign physical or controller claims')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryTargetedContinuationGateRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-targeted-continuation-gate-record'

    gate_id: str
    predicate_id: str
    targeted_result_sha256: str
    all_primary_cells_valid: bool
    independence_failure_count: int
    continuation_endpoint_coordinate_ids: tuple[str, ...]
    activate_evaluation_u: bool
    eligible_unit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        validate_sha256(self.targeted_result_sha256, field_name="targeted_result_sha256")
        _sorted_unique(
            self.continuation_endpoint_coordinate_ids,
            field_name="continuation_endpoint_coordinate_ids",
            allow_empty=True,
        )
        _sorted_unique(self.reason_codes, field_name="reason_codes", allow_empty=True)
        expected = (
            self.all_primary_cells_valid
            and self.independence_failure_count == 0
            and bool(self.continuation_endpoint_coordinate_ids)
        )
        if self.activate_evaluation_u != expected or self.eligible_unit_count not in {36, 90}:
            raise ValueError('receiver-history targeted continuation predicate is not derived')
        if self.activate_evaluation_u == bool(self.reason_codes):
            raise ValueError('receiver-history targeted continuation reasons conflict with its result')


@dataclass(frozen=True, slots=True)
class ReceiverHistoryTerminalCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-terminal-closeout'

    closeout_id: str
    terminal: ReceiverHistoryTerminal
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
            raise ValueError("receiver-history closeout ceiling differs")
        if self.external_device_count != 0 or self.physical_action_count != 0:
            raise ValueError("receiver-history closeout must remain simulator-only")


__all__ = [name for name in globals() if name.startswith("ReceiverHistory")]
