"""Immutable scientific records for the quantum trajectory preparation qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Mapping

import numpy as np
from numpy.typing import NDArray


PLAN_ID: Final = "quantum-trajectory-preparation-qualification"
CAMPAIGN_TOKEN: Final = "quantum-trajectory-preparation-qualification"
EXTERNAL_ROOT: Final = "runs/quantum-trajectory-preparation-qualification"
CAPABILITY_VERSION: Final = "1.0.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.preparation-qualification"
METHOD_CAPABILITY_KEY: Final = "method.quantum-source-preparation.preparation-qualification"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-source-preparation.preparation-qualification"

ComplexVector = NDArray[np.complex128]
FloatVector = NDArray[np.float64]


class Stage(StrEnum):
    DESIGN = "design"
    FREEZE = "freeze"
    BASE_COMPATIBILITY = "base-compatibility"
    STATISTICAL_CONFORMANCE = "statistical-conformance"
    DEVELOPMENT = "development"
    EVALUATION_FREEZE = "evaluation-freeze"
    EVALUATION = "evaluation"
    CLOSEOUT = "closeout"


class Action(StrEnum):
    HOLD = "hold"
    MINUS = "minus"
    PLUS = "plus"

    @property
    def sign(self) -> int:
        return {Action.HOLD: 0, Action.MINUS: -1, Action.PLUS: 1}[self]


class Denominator(StrEnum):
    STRONG = "strong"
    WEAK = "weak"

    @property
    def gamma(self) -> float:
        return {Denominator.STRONG: 1.0, Denominator.WEAK: 0.1}[self]


class View(StrEnum):
    ANALYTIC = "analytic"
    SPARSE = "sparse"
    DENSE = "dense"


class Validity(StrEnum):
    VALID = "VALID"
    ZERO_PROJECTED_MASS = "ZERO_PROJECTED_MASS"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class Verdict(StrEnum):
    BASE_COMPATIBLE = "BASE_COMPATIBLE"
    BASE_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_BASE_COMPATIBILITY_STOP"
    STATISTICAL_VALID = "JOINT_PREPARATION_INSTRUMENT_VALID"
    DESIGN_INADEQUATE = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_DESIGN_INADEQUATE_BEFORE_FREEZE"
    STATISTICAL_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_JOINT_GRID_INFERENCE_STOP"
    NO_WINDOW = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_NO_PREPARATION_WINDOW_SELECTED"
    EVALUATION_FREEZE_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_EVALUATION_FREEZE_STOP"
    FRESH_STATIONARITY_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_FRESH_STATIONARITY_STOP"
    PREPARATION_MEMORY_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_PREPARATION_MEMORY_STOP"
    EVALUATION_UNEVALUABLE = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_EVALUATION_UNEVALUABLE"
    STORAGE_AUTHORITY_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_STORAGE_OR_AUTHORITY_STOP"
    IMPLEMENTATION_DEFECT_STOP = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_IMPLEMENTATION_DEFECT_STOP"
    QUALIFIED = "QUANTUM_TRAJECTORY_PREPARATION_QUALIFICATION_STRONG_SOURCE_PREPARATION_QUALIFIED"


@dataclass(frozen=True, slots=True)
class PreparationUnit:
    roster_id: str
    unit_id: str
    unit_index: int
    l_sites: int
    particles: int
    k_left: int
    initial_state: int
    boundary_occupation: int
    translation_orbit: int
    preparation_seed: int


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_index: int
    event_time: float
    site: int


@dataclass(frozen=True, slots=True)
class JumpDiagnostic:
    event_index: int
    event_time: float
    site: int
    propagator_norm_residual: float
    projected_mass_expectation: float
    projected_vector_norm_squared: float
    projected_mass_identity_residual: float
    selected_mark_probability: float
    mass_sum_residual: float
    mark_probability_sum_residual: float
    post_jump_norm_residual: float
    particle_number_residual: float


@dataclass(frozen=True, slots=True)
class TrajectoryCheckpoint:
    unit_id: str
    denominator: Denominator
    action: Action
    l_sites: int
    particles: int
    clock: float
    state: ComplexVector
    events: tuple[EventRecord, ...]
    diagnostics: tuple[JumpDiagnostic, ...]
    rng_state: Mapping[str, object]
    rng_draw_count: int
    maximum_propagator_norm_residual: float
    maximum_post_jump_norm_residual: float
    maximum_projected_mass_identity_residual: float
    maximum_mark_probability_sum_residual: float
    maximum_particle_number_residual: float


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    unit_id: str
    denominator: Denominator
    action: Action
    view: View
    l_sites: int
    particles: int
    initial_state: int
    preparation_seed: int
    end_time: float
    terminal_state: ComplexVector
    events: tuple[EventRecord, ...]
    diagnostics: tuple[JumpDiagnostic, ...]
    snapshots: Mapping[float, ComplexVector]
    checkpoint_rng_states: Mapping[float, Mapping[str, object]]
    checkpoint_rng_draw_counts: Mapping[float, int]
    maximum_propagator_norm_residual: float
    maximum_post_jump_norm_residual: float
    maximum_projected_mass_identity_residual: float
    maximum_mark_probability_sum_residual: float
    maximum_particle_number_residual: float


@dataclass(frozen=True, slots=True)
class FixtureRow:
    fixture_id: str
    view: View
    input_state_sha256: str
    site: int | None
    expected_projected_mass: float | None
    observed_projected_mass_expectation: float | None
    observed_projected_vector_norm_squared: float | None
    projected_mass_identity_residual: float | None
    expected_mark_probability: float | None
    observed_mark_probability: float | None
    mark_sum_residual: float | None
    propagator_norm_residual: float | None
    post_jump_norm_residual: float | None
    phase_gauge_infidelity: float | None
    prng_counter_before: int
    prng_counter_after: int
    validity: Validity
    reason_code: str
    requested_action: str | None = None
    accepted_action: str | None = None
    applied_action: str | None = None
    realized_action_start: float | None = None
    realized_action_end: float | None = None
    endpoint_included: bool | None = None


@dataclass(frozen=True, slots=True)
class ObservableSupport:
    index: int
    coordinate: str
    lower: float
    upper: float
    operator_sha256: str
    support_sha256: str
    route: str

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class BoundedInterval:
    coordinate: str
    sample_size: int
    estimate: float
    unbiased_variance_normalized: float
    delta: float
    normalized_halfwidth: float
    native_halfwidth: float
    lower_unclipped: float
    upper_unclipped: float
    lower_display: float
    upper_display: float
    support_lower: float
    support_upper: float
    validity: Validity
    reason_code: str


@dataclass(frozen=True, slots=True)
class ReferenceEnvelope:
    coordinate: str
    estimate: float
    radius: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class EnsembleConformanceRungDecision:
    denominator: Denominator
    sample_size: int
    precision_passed: bool
    selected_by_width: bool
    coverage_passed: bool | None
    density_passed: bool | None
    passed: bool | None
    validity: Validity
    reason_code: str


@dataclass(frozen=True, slots=True)
class CoordinateInterval:
    stage: Stage
    denominator: Denominator
    comparison_id: str
    sample_size: int
    family: str
    coordinate: str
    estimate: float
    standard_error: float
    scale: float
    critical_value: float
    upper_bound: float
    margin: float
    passed: bool
    validity: Validity = Validity.VALID
    reason_code: str = "OK"


@dataclass(frozen=True, slots=True)
class MemoryResult:
    stage: Stage
    denominator: Denominator
    clock_id: str
    sample_size: int
    factor: str
    coordinate: str
    eta_squared: float
    adjusted_p_value: float
    material: bool
    invariant: bool
    validity: Validity


__all__ = [
    "Action",
    "BoundedInterval",
    "CAMPAIGN_TOKEN",
    "CAPABILITY_VERSION",
    "EnsembleConformanceRungDecision",
    "ComplexVector",
    "CoordinateInterval",
    "Denominator",
    "EVALUATOR_CAPABILITY_KEY",
    "EXTERNAL_ROOT",
    "EventRecord",
    "FixtureRow",
    "FloatVector",
    "JumpDiagnostic",
    "METHOD_CAPABILITY_KEY",
    "MemoryResult",
    "ObservableSupport",
    "PLAN_ID",
    "PreparationUnit",
    "ReferenceEnvelope",
    "SOURCE_CAPABILITY_KEY",
    "Stage",
    "TrajectoryCheckpoint",
    "TrajectoryResult",
    "Validity",
    "Verdict",
    "View",
]
