"""Immutable local scientific records for quantum trajectory burnin screen."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Mapping

import numpy as np
from numpy.typing import NDArray


PLAN_ID: Final = "quantum-trajectory-burnin-screen"
CAMPAIGN_TOKEN: Final = "quantum-trajectory-burnin-screen"
EXTERNAL_ROOT: Final = "runs/quantum-trajectory-burnin-screen"
CAPABILITY_VERSION: Final = "1.0.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.burnin-screen"
METHOD_CAPABILITY_KEY: Final = "method.quantum-source-qualification.burnin-screen"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-source-qualification.burnin-screen"

ComplexVector = NDArray[np.complex128]
FloatVector = NDArray[np.float64]


class Stage(StrEnum):
    DESIGN = "design"
    FREEZE = "freeze"
    CONFORMANCE = "conformance"
    COMPARATOR = "comparator"
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
    SEMANTIC_VALID = "SEMANTIC_CONTRACT_VALID"
    SEMANTIC_STOP = "SEMANTIC_CONTRACT_STOP"
    NUMERICAL_SOURCE_STOP = "NUMERICAL_SOURCE_STOP"
    NO_BURNIN = "NO_BURNIN_WINDOW_SELECTED"
    EVALUATION_FREEZE_STOP = "EVALUATION_FREEZE_STOP"
    FRESH_STATIONARITY_STOP = "FRESH_STATIONARITY_STOP"
    PREPARATION_MEMORY_STOP = "PREPARATION_MEMORY_STOP"
    EVALUATION_UNEVALUABLE = "EVALUATION_UNEVALUABLE"
    STORAGE_AUTHORITY_STOP = "STORAGE_OR_AUTHORITY_STOP"
    BOTH_QUALIFIED = "SOURCE_QUALIFIED_BOTH_DENOMINATORS"
    STRONG_ONLY = "SOURCE_QUALIFIED_STRONG_ONLY"
    WEAK_ONLY = "WEAK_ONLY_QUALIFIED_NO_RECEIVER_LAW_RELEASE"


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


@dataclass(frozen=True, slots=True)
class CoordinateInterval:
    stage: Stage
    denominator: Denominator
    comparison_id: str
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
    factor: str
    coordinate: str
    eta_squared: float
    adjusted_p_value: float
    material: bool
    invariant: bool
    validity: Validity


__all__ = [
    "Action",
    "CAMPAIGN_TOKEN",
    "CAPABILITY_VERSION",
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
    "PLAN_ID",
    "PreparationUnit",
    "SOURCE_CAPABILITY_KEY",
    "Stage",
    "TrajectoryCheckpoint",
    "TrajectoryResult",
    "Validity",
    "Verdict",
    "View",
]
