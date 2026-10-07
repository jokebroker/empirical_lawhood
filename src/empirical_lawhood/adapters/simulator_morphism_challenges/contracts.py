"""Canonical interchange records for simulator morphism challenges.

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


class SimulatorMorphismChallengePhase(StrEnum):
    NOMINATION = "NOMINATION"
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class SimulatorMorphismChallengeDisorderFamily(StrEnum):
    SMOOTH_PERIODIC_12 = "smooth-periodic-12"
    CORRELATED_FIELD_12 = "correlated-field-12"
    MULTISCALE_WAVELET_12 = "multiscale-wavelet-12"


class SimulatorMorphismChallengeGate(StrEnum):
    TARGET = "TARGET"
    SINK = "SINK"
    RANDOM = "RANDOM"


class SimulatorMorphismChallengePairKind(StrEnum):
    TARGET_BOUNDARY = "TARGET_BOUNDARY"
    SINK_BOUNDARY = "SINK_BOUNDARY"
    RANDOM_FIBRE = "RANDOM_FIBRE"


class SimulatorMorphismChallengeScientificState(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"
    RIGHT_CENSORED = "RIGHT_CENSORED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class SimulatorMorphismChallengeTerminal(StrEnum):
    BOUNDED_SIMULATOR_MORPHISM_CHALLENGES_EVALUATED = (
        "BOUNDED_SIMULATOR_MORPHISM_CHALLENGES_EVALUATED"
    )
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
class SimulatorMorphismChallengeConfig(CanonicalRecord):
    """Strict phase configuration shared by authoring and scientific code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-config'

    config_id: str
    phase: SimulatorMorphismChallengePhase
    plan_id: str
    parent_result_id: str
    scale_cells: tuple[int, ...]
    excluded_resource_canary_scale_cells: tuple[int, ...]
    disorder_families: tuple[SimulatorMorphismChallengeDisorderFamily, ...]
    unit_ids: tuple[str, ...]
    seed_roster_commitment_sha256: str | None
    receiver_bin_count: int
    history_max_depth: int
    history_lag_t_star: Decimal
    rank_relative_threshold: Decimal
    coordinate_collision_epsilon: Decimal
    future_divergence_epsilon: Decimal
    generator_observer_metric_tolerance: Decimal
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
    bootstrap_resamples: int
    bootstrap_seed: int
    simultaneous_alpha: Decimal
    majority_opposition_threshold: Decimal
    descriptor_algorithm_id: str
    history_rank_algorithm_id: str
    targeting_tie_break_rule_id: str
    random_comparator_algorithm_id: str
    inference_algorithm_id: str
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
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.plan_id != "simulator-morphism-challenges":
            raise ValueError("simulator morphism challenges config binds the wrong plan")
        expected_scales = {
            SimulatorMorphismChallengePhase.NOMINATION: (64, 128, 256),
            SimulatorMorphismChallengePhase.CANARY: (16, 32),
            SimulatorMorphismChallengePhase.DEVELOPMENT: (16, 32, 64),
            SimulatorMorphismChallengePhase.EVALUATION: (64, 128, 256),
        }[self.phase]
        if self.scale_cells != expected_scales:
            raise ValueError("simulator morphism challenges phase scale roster differs")
        expected_resource_scales = (64, 128, 256) if self.phase is SimulatorMorphismChallengePhase.CANARY else ()
        if self.excluded_resource_canary_scale_cells != expected_resource_scales:
            raise ValueError("simulator morphism challenges excluded resource-canary scale roster differs")
        if self.disorder_families != tuple(SimulatorMorphismChallengeDisorderFamily):
            raise ValueError("simulator morphism challenges disorder-family roster differs")
        _sorted_unique(self.unit_ids, field_name="unit_ids")
        expected_units = {
            SimulatorMorphismChallengePhase.NOMINATION: 36,
            SimulatorMorphismChallengePhase.CANARY: 2,
            SimulatorMorphismChallengePhase.DEVELOPMENT: 6,
            SimulatorMorphismChallengePhase.EVALUATION: 36,
        }[self.phase]
        if len(self.unit_ids) != expected_units:
            raise ValueError("simulator morphism challenges phase complete-unit count differs")
        if self.phase is SimulatorMorphismChallengePhase.EVALUATION:
            if self.seed_roster_commitment_sha256 is None:
                raise ValueError("evaluation config requires the pre-development seed commitment")
            validate_sha256(
                self.seed_roster_commitment_sha256,
                field_name="seed_roster_commitment_sha256",
            )
        elif self.seed_roster_commitment_sha256 is not None:
            raise ValueError("only the evaluation config may bind the seed commitment")
        if self.receiver_bin_count != 8 or self.history_max_depth != 35:
            raise ValueError("simulator morphism challenges receiver/history roster differs")
        for name in (
            "history_lag_t_star",
            "rank_relative_threshold",
            "coordinate_collision_epsilon",
            "future_divergence_epsilon",
            "generator_observer_metric_tolerance",
            "midpoint_boundary_tie_epsilon",
            "hidden_amplitude_max_volts",
            "hidden_voltage_interior_margin",
            "panel_endpoint_t_star",
            "target_charge_minimum_q_star",
            "sink_voltage_maximum_v_star",
            "simultaneous_alpha",
            "majority_opposition_threshold",
        ):
            _positive_decimal(getattr(self, name), field_name=name)
        if self.rank_relative_threshold != Decimal("1e-12"):
            raise ValueError("simulator morphism challenges rank threshold differs")
        if self.coordinate_collision_epsilon != Decimal(
            "0.005"
        ) or self.future_divergence_epsilon != Decimal("0.005"):
            raise ValueError("simulator morphism challenges collision/divergence margins differ")
        if self.generator_observer_metric_tolerance != Decimal("1e-10"):
            raise ValueError("simulator morphism challenges generator/observer metric tolerance differs")
        if self.midpoint_boundary_tie_epsilon != Decimal("1e-10"):
            raise ValueError("simulator morphism challenges midpoint boundary tie convention differs")
        if self.hidden_voltage_envelope != (Decimal("0"), Decimal("1")):
            raise ValueError("simulator morphism challenges amended hidden-state envelope differs")
        if self.hidden_voltage_interior_margin != Decimal("1e-8"):
            raise ValueError("simulator morphism challenges hidden-state interior margin differs")
        if (
            self.target_pairs_per_depth != 2
            or self.sink_pairs_per_depth != 2
            or self.random_pairs_per_depth != 4
        ):
            raise ValueError("simulator morphism challenges targeting roster differs")
        for name in ("action_amplitudes_u_star", "action_durations_t_star"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or not values:
                raise ValueError(f"{name} must be nonempty, sorted and unique")
            for index, value in enumerate(values):
                _positive_decimal(value, field_name=f"{name}[{index}]")
        if len(self.action_amplitudes_u_star) != 5 or len(self.action_durations_t_star) != 5:
            raise ValueError("simulator morphism challenges requires the complete 5 x 5 action panel")
        if self.panel_endpoint_t_star != Decimal("0.20"):
            raise ValueError("simulator morphism challenges panel endpoint differs")
        if self.diagnostic_times_t_star != (
            Decimal("0.005"),
            Decimal("0.01"),
            Decimal("0.02"),
        ):
            raise ValueError("simulator morphism challenges diagnostic-time roster differs")
        if self.bootstrap_resamples != 10_000 or self.bootstrap_seed < 0:
            raise ValueError("simulator morphism challenges bootstrap contract differs")
        if not Decimal(0) < self.simultaneous_alpha < Decimal(1):
            raise ValueError("simulator morphism challenges simultaneous alpha is invalid")
        if self.majority_opposition_threshold != Decimal("0.5"):
            raise ValueError("simulator morphism challenges majority-opposition threshold differs")
        if (
            self.descriptor_algorithm_id != "simulator-morphism-challenges-latent-field-centre-edge"
            or self.history_rank_algorithm_id != "simulator-morphism-challenges-relative-svd-row-equilibrated"
            or self.targeting_tie_break_rule_id != "simulator-morphism-challenges-stable-lexical-boundary"
            or self.random_comparator_algorithm_id != "simulator-morphism-challenges-outcome-blind-fibre-pcg64"
            or self.inference_algorithm_id != "simulator-morphism-challenges-complete-unit-bonferroni-bootstrap"
        ):
            raise ValueError("simulator morphism challenges algorithm identity differs")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("simulator morphism challenges evidence ceiling differs")
        expected_access = {
            SimulatorMorphismChallengePhase.NOMINATION: OutcomeAccess.OUTCOME_BLIND,
            SimulatorMorphismChallengePhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SimulatorMorphismChallengePhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SimulatorMorphismChallengePhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        expected_visibility = {
            SimulatorMorphismChallengePhase.NOMINATION: VisibilityCeiling.PROSPECTIVE,
            SimulatorMorphismChallengePhase.CANARY: VisibilityCeiling.DEVELOPMENT_ONLY,
            SimulatorMorphismChallengePhase.DEVELOPMENT: VisibilityCeiling.DEVELOPMENT_ONLY,
            SimulatorMorphismChallengePhase.EVALUATION: VisibilityCeiling.PROSPECTIVE,
        }[self.phase]
        if (
            self.outcome_access is not expected_access
            or self.visibility_ceiling is not expected_visibility
        ):
            raise ValueError("simulator morphism challenges phase outcome/visibility contract differs")
        if self.physical_execution_authorized or self.requests_controller or not self.nonactuating:
            raise ValueError("simulator morphism challenges must remain nonactuating simulator-only work")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeSeedRosterCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-seed-roster-commitment'

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
        if len(self.evaluation_unit_ids) != 36:
            raise ValueError("simulator morphism challenges evaluation roster must contain 36 complete units")
        validate_sha256(self.seed_payload_sha256, field_name="seed_payload_sha256")
        validate_sha256(self.commitment_sha256, field_name="commitment_sha256")
        if self.seed_bytes_per_unit != 32 or self.public_seed_count != 0:
            raise ValueError("simulator morphism challenges commitment must not disclose evaluation seeds")
        if self.outcome_count_at_commitment != 0:
            raise ValueError("simulator morphism challenges roster commitment must precede outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDenominatorDescriptor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-denominator-descriptor'

    descriptor_id: str
    unit_id: str
    family: SimulatorMorphismChallengeDisorderFamily
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
            raise ValueError("simulator morphism challenges descriptor scale differs")
        if len(self.capacitances_farads) != self.scale_cells:
            raise ValueError("simulator morphism challenges capacitance roster differs")
        if len(self.interior_resistances_ohms) != self.scale_cells - 1:
            raise ValueError("simulator morphism challenges resistance roster differs")
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
            raise ValueError("simulator morphism challenges descriptor numerical contract differs")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeRankStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-rank-step'

    depth: int
    effective_rank: int
    algebraic_comparator_rank: int
    row_count_ceiling: int
    largest_singular_value: Decimal
    smallest_retained_relative_singular_value: Decimal
    condition_number: Decimal | None
    nonmonotone_effective_rank: bool

    def __post_init__(self) -> None:
        if not 0 <= self.depth <= 35:
            raise ValueError("simulator morphism challenges history depth differs")
        if not 0 <= self.effective_rank <= self.row_count_ceiling:
            raise ValueError("simulator morphism challenges effective rank exceeds its ceiling")
        if not 0 <= self.algebraic_comparator_rank <= self.row_count_ceiling:
            raise ValueError("simulator morphism challenges algebraic rank exceeds its ceiling")
        _positive_decimal(self.largest_singular_value, field_name="largest_singular_value")
        validate_decimal(
            self.smallest_retained_relative_singular_value,
            field_name="smallest_retained_relative_singular_value",
            minimum=Decimal(0),
        )
        if self.condition_number is not None:
            _positive_decimal(self.condition_number, field_name="condition_number")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeHistoryRankForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-history-rank-forecast'

    forecast_id: str
    descriptor_sha256: str
    rank_steps: tuple[SimulatorMorphismChallengeRankStep, ...]
    singular_spectra_sha256: str
    k_full_effective: int | None
    effective_rank_right_censored: bool
    probe_depths: tuple[int, ...]
    algebraic_prediction_state: SimulatorMorphismChallengeScientificState

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        validate_sha256(self.descriptor_sha256, field_name="descriptor_sha256")
        validate_sha256(self.singular_spectra_sha256, field_name="singular_spectra_sha256")
        if tuple(value.depth for value in self.rank_steps) != tuple(range(36)):
            raise ValueError("simulator morphism challenges rank forecast must contain depths 0..35")
        if self.k_full_effective is None:
            if not self.effective_rank_right_censored:
                raise ValueError("missing full-rank depth must be right-censored")
        elif not 0 <= self.k_full_effective <= 35 or self.effective_rank_right_censored:
            raise ValueError("simulator morphism challenges full-rank depth/censoring differs")
        if self.probe_depths != tuple(sorted(set(self.probe_depths))):
            raise ValueError("simulator morphism challenges probe depths must be sorted and unique")
        if self.algebraic_prediction_state not in {
            SimulatorMorphismChallengeScientificState.SUPPORTED,
            SimulatorMorphismChallengeScientificState.OPPOSED,
        }:
            raise ValueError("simulator morphism challenges algebraic prediction state differs")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeActionPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-action-prediction'

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
            raise ValueError("simulator morphism challenges predicted admission is not a gate intersection")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeChallengeNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-challenge-nomination'

    nomination_id: str
    unit_id: str
    scale_cells: int
    depth: int
    pair_index: int
    pair_kind: SimulatorMorphismChallengePairKind
    gate: SimulatorMorphismChallengeGate
    targeting_action_id: str | None
    carrier_volts: Decimal
    center_values: tuple[Decimal, ...]
    amplitude_volts: Decimal
    mode_values: tuple[Decimal, ...]
    predicted_coordinate_defect: Decimal
    predicted_maximum_future_receiver_defect: Decimal
    predicted_maximum_future_metric_defect: Decimal
    action_predictions: tuple[SimulatorMorphismChallengeActionPrediction, ...]
    observer_implementation_sha256: str
    outcome_count_at_nomination: int

    def __post_init__(self) -> None:
        for name in ("nomination_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.targeting_action_id is not None:
            validate_stable_id(self.targeting_action_id, field_name="targeting_action_id")
        if self.scale_cells not in {16, 32, 64, 128, 256} or not 0 <= self.depth <= 35:
            raise ValueError("simulator morphism challenges nomination scale/depth differs")
        if (
            self.pair_index < 0
            or len(self.mode_values) != self.scale_cells
            or len(self.center_values) != self.scale_cells
        ):
            raise ValueError("simulator morphism challenges nomination mode roster differs")
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
            raise ValueError("simulator morphism challenges nomination amplitude must be positive")
        if len(self.action_predictions) != 25:
            raise ValueError("simulator morphism challenges nomination must carry the complete action panel")
        action_ids = tuple(value.action_id for value in self.action_predictions)
        _sorted_unique(action_ids, field_name="action_predictions")
        validate_sha256(
            self.observer_implementation_sha256,
            field_name="observer_implementation_sha256",
        )
        if self.outcome_count_at_nomination != 0:
            raise ValueError("simulator morphism challenges nomination cannot observe generator outcomes")
        expected_gate = {
            SimulatorMorphismChallengePairKind.TARGET_BOUNDARY: SimulatorMorphismChallengeGate.TARGET,
            SimulatorMorphismChallengePairKind.SINK_BOUNDARY: SimulatorMorphismChallengeGate.SINK,
            SimulatorMorphismChallengePairKind.RANDOM_FIBRE: SimulatorMorphismChallengeGate.RANDOM,
        }[self.pair_kind]
        if self.gate is not expected_gate:
            raise ValueError("simulator morphism challenges pair kind and gate differ")
        if (
            self.pair_kind is SimulatorMorphismChallengePairKind.RANDOM_FIBRE and self.targeting_action_id is not None
        ) or (
            self.pair_kind is not SimulatorMorphismChallengePairKind.RANDOM_FIBRE and self.targeting_action_id is None
        ):
            raise ValueError("simulator morphism challenges targeting action and pair kind differ")
        if any(value.requested_action_id != value.action_id for value in self.action_predictions):
            raise ValueError("simulator morphism challenges requested action identity differs")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeMethodFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-method-freeze'

    freeze_id: str
    development_config_sha256: str
    evaluation_config_sha256: str
    seed_roster_commitment_sha256: str
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
            "evaluation_config_sha256",
            "seed_roster_commitment_sha256",
            "observer_implementation_sha256",
            "generator_implementation_sha256",
            "evaluator_implementation_sha256",
            "development_receipt_closure_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        _sorted_unique(self.frozen_threshold_ids, field_name="frozen_threshold_ids")
        if self.evaluation_outcome_count != 0:
            raise ValueError("simulator morphism challenges method freeze must precede evaluation outcomes")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("simulator morphism challenges method-freeze ceiling differs")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeGeneratorActionOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-generator-action-outcome'

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
            raise ValueError("simulator morphism challenges realized admission is not a gate intersection")
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
            raise ValueError("simulator morphism challenges action parity flag is not derived")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeGeneratorPairOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-generator-pair-outcome'

    nomination_id: str
    realized_coordinate_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    future_receiver_defects: tuple[Decimal, ...]
    action_outcomes: tuple[SimulatorMorphismChallengeGeneratorActionOutcome, ...]
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
            raise ValueError("simulator morphism challenges generator future diagnostic roster differs")
        for index, value in enumerate(self.future_receiver_defects):
            validate_decimal(
                value, field_name=f"future_receiver_defects[{index}]", minimum=Decimal(0)
            )
        if len(self.action_outcomes) != 25:
            raise ValueError("simulator morphism challenges generator outcome requires the complete action panel")
        action_ids = tuple(value.action_id for value in self.action_outcomes)
        _sorted_unique(action_ids, field_name="action_outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeGeneratorOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-generator-outcome'

    outcome_id: str
    unit_id: str
    descriptor_sha256: str
    nomination_set_sha256: str
    generator_implementation_sha256: str
    pair_outcomes: tuple[SimulatorMorphismChallengeGeneratorPairOutcome, ...]
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
            raise ValueError("simulator morphism challenges generator outcome access/array roster differs")
        if self.scientific_verdict_assigned:
            raise ValueError("simulator morphism challenges generator cannot assign a scientific verdict")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengePairAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-pair-adjudication'

    nomination_id: str
    depth: int
    gate: SimulatorMorphismChallengeGate
    realized_coordinate_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    target_mismatch_count: int
    sink_mismatch_count: int
    decision_mismatch_count: int
    false_safe_count: int
    false_hold_count: int
    collision_confirmed: bool
    dynamically_adverse: bool
    decision_adverse: bool
    prediction_agreement: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        if not 0 <= self.depth <= 35:
            raise ValueError("simulator morphism challenges pair depth differs")
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
            self.false_hold_count,
        )
        if min(counts) < 0 or max(counts) > 25:
            raise ValueError("simulator morphism challenges mismatch count differs")
        if self.false_safe_count + self.false_hold_count != self.decision_mismatch_count:
            raise ValueError("simulator morphism challenges false-safe/hold counts do not partition decisions")
        if self.decision_adverse != (self.collision_confirmed and self.decision_mismatch_count > 0):
            raise ValueError("simulator morphism challenges decision-adverse state is not derived")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDepthAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-depth-adjudication'

    depth: int
    requested_pair_count: int
    realized_collision_count: int
    dynamic_adverse_count: int
    decision_adverse_count: int
    false_safe_count: int
    dynamical_state: SimulatorMorphismChallengeScientificState
    decision_state: SimulatorMorphismChallengeScientificState

    def __post_init__(self) -> None:
        counts = (
            self.requested_pair_count,
            self.realized_collision_count,
            self.dynamic_adverse_count,
            self.decision_adverse_count,
            self.false_safe_count,
        )
        if not 0 <= self.depth <= 35 or min(counts) < 0:
            raise ValueError("simulator morphism challenges depth adjudication differs")
        if (
            not self.dynamic_adverse_count
            <= self.realized_collision_count
            <= self.requested_pair_count
        ):
            raise ValueError("simulator morphism challenges dynamical depth counts differ")
        if not self.decision_adverse_count <= self.realized_collision_count:
            raise ValueError("simulator morphism challenges decision depth counts differ")
        expected_dynamical_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if self.dynamic_adverse_count
            else SimulatorMorphismChallengeScientificState.SUPPORTED
            if self.realized_collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if self.decision_adverse_count
            else SimulatorMorphismChallengeScientificState.SUPPORTED
            if self.realized_collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("simulator morphism challenges depth scientific states are not derived")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeUnitAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-unit-adjudication'

    adjudication_id: str
    unit_id: str
    family: SimulatorMorphismChallengeDisorderFamily
    scale_cells: int
    descriptor_valid: bool
    nominated_pair_count: int
    realized_collision_count: int
    evaluable_pair_count: int
    pair_adjudications: tuple[SimulatorMorphismChallengePairAdjudication, ...]
    depth_adjudications: tuple[SimulatorMorphismChallengeDepthAdjudication, ...]
    dynamical_state: SimulatorMorphismChallengeScientificState
    decision_state: SimulatorMorphismChallengeScientificState
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
            raise ValueError("simulator morphism challenges unit counts/scale differ")
        if (
            not self.evaluable_pair_count
            <= self.realized_collision_count
            <= self.nominated_pair_count
        ):
            raise ValueError("simulator morphism challenges unit denominator chain differs")
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
            raise ValueError("simulator morphism challenges unit counts do not match pair adjudications")
        if tuple(value.depth for value in self.depth_adjudications) != tuple(
            sorted({value.depth for value in self.depth_adjudications})
        ):
            raise ValueError("simulator morphism challenges depth adjudications must be sorted and unique")
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
            raise ValueError("simulator morphism challenges depth ledger does not aggregate the pair ledger")
        expected_dynamical_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if any(value.dynamically_adverse for value in self.pair_adjudications)
            else SimulatorMorphismChallengeScientificState.SUPPORTED
            if self.realized_collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
        )
        expected_decision_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if any(value.decision_adverse for value in self.pair_adjudications)
            else SimulatorMorphismChallengeScientificState.SUPPORTED
            if self.realized_collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
        )
        if (
            self.dynamical_state is not expected_dynamical_state
            or self.decision_state is not expected_decision_state
        ):
            raise ValueError("simulator morphism challenges unit scientific states are not derived")
        if self.generator_observer_agreement != all(
            value.prediction_agreement for value in self.pair_adjudications
        ):
            raise ValueError("simulator morphism challenges generator/observer agreement is not derived")
        if self.k_full_effective is not None and not 0 <= self.k_full_effective <= 35:
            raise ValueError("simulator morphism challenges full-rank depth differs")
        if self.b_full is not None:
            validate_decimal(self.b_full, field_name="b_full", minimum=Decimal(0))
            if self.b_full > 1:
                raise ValueError("simulator morphism challenges normalized history budget exceeds one")
        validate_decimal(
            self.rank_curve_distance, field_name="rank_curve_distance", minimum=Decimal(0)
        )
        _sorted_unique(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeCellRecurrence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-cell-recurrence'

    cell_id: str
    family: SimulatorMorphismChallengeDisorderFamily
    scale_cells: int
    endpoint_id: str
    requested_count: int
    descriptor_valid_count: int
    nominated_count: int
    realized_collision_count: int
    evaluable_count: int
    opposed_count: int
    supported_count: int
    unevaluable_count: int
    evaluable_only_opposition_rate: Decimal | None
    opposition_lower_bound: Decimal
    existence_recurs: bool
    majority_opposition: bool

    def __post_init__(self) -> None:
        for name in ("cell_id", "endpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.requested_count,
            self.descriptor_valid_count,
            self.nominated_count,
            self.realized_collision_count,
            self.evaluable_count,
            self.opposed_count,
            self.supported_count,
            self.unevaluable_count,
        )
        if self.scale_cells not in {64, 128, 256} or min(counts) < 0:
            raise ValueError("simulator morphism challenges recurrence cell differs")
        if self.requested_count != 12:
            raise ValueError("simulator morphism challenges primary cell denominator must remain 12")
        if not (
            self.requested_count
            >= self.descriptor_valid_count
            >= self.nominated_count
            >= self.realized_collision_count
            >= self.evaluable_count
        ):
            raise ValueError("simulator morphism challenges recurrence denominator chain differs")
        if self.opposed_count + self.supported_count + self.unevaluable_count != 12:
            raise ValueError("simulator morphism challenges recurrence states do not close the requested denominator")
        if self.evaluable_count != self.opposed_count + self.supported_count:
            raise ValueError("simulator morphism challenges evaluable denominator differs from terminal states")
        if self.evaluable_count == 0:
            if self.evaluable_only_opposition_rate is not None:
                raise ValueError("empty evaluable denominator cannot report a rate")
        else:
            if self.evaluable_only_opposition_rate is None:
                raise ValueError("nonempty evaluable denominator requires its sensitivity rate")
            validate_decimal(
                self.evaluable_only_opposition_rate,
                field_name="evaluable_only_opposition_rate",
                minimum=Decimal(0),
            )
            if self.evaluable_only_opposition_rate != (
                Decimal(self.opposed_count) / Decimal(self.evaluable_count)
            ):
                raise ValueError("evaluable-only opposition rate is not derived")
        validate_decimal(
            self.opposition_lower_bound, field_name="opposition_lower_bound", minimum=Decimal(0)
        )
        if self.opposition_lower_bound > 1:
            raise ValueError("simulator morphism challenges opposition lower bound exceeds one")
        if self.existence_recurs != (self.opposed_count > 0):
            raise ValueError("simulator morphism challenges cell existence flag is not derived")
        if self.majority_opposition != (self.opposition_lower_bound > Decimal("0.5")):
            raise ValueError("simulator morphism challenges cell majority flag is not derived")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeRecurrenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-recurrence-result'

    result_id: str
    method_freeze_sha256: str
    cells: tuple[SimulatorMorphismChallengeCellRecurrence, ...]
    dynamical_existence_recurs: bool
    decision_existence_recurs: bool
    dynamical_majority_opposition_recurs: bool
    decision_majority_opposition_recurs: bool
    bootstrap_summary_sha256: str
    decisive_counterexample_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    physical_claim_allowed: bool
    controller_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        validate_sha256(self.bootstrap_summary_sha256, field_name="bootstrap_summary_sha256")
        cell_ids = tuple(value.cell_id for value in self.cells)
        _sorted_unique(cell_ids, field_name="cells")
        if len(self.cells) != 18:
            raise ValueError("simulator morphism challenges recurrence result requires 18 primary cells/endpoints")
        dynamical = tuple(value for value in self.cells if value.endpoint_id == "dynamical-closure")
        decision = tuple(value for value in self.cells if value.endpoint_id == "decision-closure")
        if len(dynamical) != 9 or len(decision) != 9:
            raise ValueError("simulator morphism challenges recurrence endpoint roster differs")
        if (
            self.dynamical_existence_recurs != all(value.existence_recurs for value in dynamical)
            or self.decision_existence_recurs != all(value.existence_recurs for value in decision)
            or self.dynamical_majority_opposition_recurs
            != all(value.majority_opposition for value in dynamical)
            or self.decision_majority_opposition_recurs
            != all(value.majority_opposition for value in decision)
        ):
            raise ValueError("simulator morphism challenges recurrence conjunction is not derived")
        _sorted_unique(self.decisive_counterexample_ids, field_name="decisive_counterexample_ids")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("simulator morphism challenges recurrence ceiling differs")
        if self.physical_claim_allowed or self.controller_claim_allowed:
            raise ValueError("simulator morphism challenges cannot claim physical recurrence or control")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeTerminalCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-terminal-closeout'

    closeout_id: str
    terminal: SimulatorMorphismChallengeTerminal
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
            raise ValueError("simulator morphism challenges closeout ceiling differs")
        if self.external_device_count != 0 or self.physical_action_count != 0:
            raise ValueError("simulator morphism challenges closeout must remain simulator-only")


__all__ = [name for name in globals() if name.startswith("SimulatorMorphismChallenge")]
