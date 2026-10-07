"""Simulator morphism challenges following the physical-scale numerical result.

The four experiments in this module are deliberately finite and scientific:

1. a receiver/refinement/scale observation cube;
2. gate-owned finite boundary-stratum transport;
3. adversarial hidden-mode and finite-history closure; and
4. a coordinate-collision tournament on the same hidden preparations.

They reuse the frozen physical-scale RC denominator and do not alter or promote its result.
Every output remains development-visible numerical-twin evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import ClassVar, Iterable

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, null_space
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderModelConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.matrix_exponential import solve_matrix_exponential
from empirical_lawhood.adapters.simulators.rc_ladder_response.mna import build_operator
from empirical_lawhood.adapters.simulators.rc_ladder_response.refinement import solve_backward_euler
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal, validate_stable_id

from .evaluation_evidence import PhysicalScaleMorphismEvidenceState
from .numerical_experiment import PHYSICAL_SCALE_MORPHISM_REFERENCE_BLAS_THREAD_COUNT, PhysicalScaleMorphismNumericalMorphismFamilyConfig, _model_config as _reference_model_config, _time_scale as _reference_time_scale, default_numerical_morphism_family_config
from .receivers import equal_width_receiver_spec, receiver_matrix


FloatArray = npt.NDArray[np.float64]


class PhysicalScaleMorphismCubeFace(StrEnum):
    RECEIVER_NUMERICAL = "RECEIVER_NUMERICAL"
    RECEIVER_SCALE = "RECEIVER_SCALE"
    NUMERICAL_SCALE = "NUMERICAL_SCALE"
    FULL_CUBE = "FULL_CUBE"


class PhysicalScaleMorphismBoundaryAxis(StrEnum):
    NUMERICAL = "NUMERICAL"
    RECEIVER = "RECEIVER"
    SCALE = "SCALE"


class PhysicalScaleMorphismCoordinateCandidate(StrEnum):
    ACTION_ONLY = "ACTION_ONLY"
    EIGHT_BIN_RECEIVER_CURRENT = "EIGHT_BIN_RECEIVER_CURRENT"
    EIGHT_BIN_RECEIVER_CURRENT_ONE_LAG = "EIGHT_BIN_RECEIVER_CURRENT_ONE_LAG"
    GUARDED_EIGHT_BIN_RECEIVER_CURRENT = "GUARDED_EIGHT_BIN_RECEIVER_CURRENT"
    FINE_CURRENT = "FINE_CURRENT"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSimulatorChallengesConfig(CanonicalRecord):
    """Frozen finite design for the four simulator-only challenges."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-simulator-challenges-config'

    config_id: str
    parent_config: ObjectIdentity
    scale_cells: tuple[int, ...]
    receiver_bin_count: int
    numerical_views: tuple[str, ...]
    metric_ids: tuple[str, ...]
    metric_commutator_margin: Decimal
    exact_commutator_margin: Decimal
    present_collision_margin: Decimal
    coordinate_collision_margin: Decimal
    future_divergence_floor: Decimal
    history_lag_t_star: Decimal
    future_times_t_star: tuple[Decimal, ...]
    hidden_mode_count: int
    hidden_amplitude_volts: Decimal
    hidden_voltage_envelope: tuple[Decimal, Decimal]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_execution_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.parent_config.object_schema != PhysicalScaleMorphismNumericalMorphismFamilyConfig.SCHEMA:
            raise ValueError("simulator challenges must bind the physical-scale numerical design")
        if self.scale_cells != (16, 32, 64):
            raise ValueError("simulator challenges require the frozen N16/N32/N64 scales")
        if self.receiver_bin_count != 8:
            raise ValueError("simulator challenges require the frozen R8 receiver")
        if self.numerical_views != ("exact", "h0-quarter"):
            raise ValueError("cube requires exact and accepted h0-quarter views")
        if self.metric_ids != ("e-star", "q-star", "v-max-star"):
            raise ValueError("cube metric roster changed")
        for name in (
            "metric_commutator_margin",
            "exact_commutator_margin",
            "present_collision_margin",
            "coordinate_collision_margin",
            "future_divergence_floor",
            "history_lag_t_star",
            "hidden_amplitude_volts",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if tuple(sorted(set(self.future_times_t_star))) != self.future_times_t_star:
            raise ValueError("future times must be sorted and unique")
        if not self.future_times_t_star or self.future_times_t_star[0] <= 0:
            raise ValueError("future times must be strictly positive")
        if self.history_lag_t_star >= self.future_times_t_star[0]:
            raise ValueError("history lag must precede the first future diagnostic")
        if self.hidden_mode_count != 4:
            raise ValueError("challenge freezes four modes per nonempty hidden subspace")
        lower, upper = self.hidden_voltage_envelope
        for name, value in (("hidden_voltage_lower", lower), ("hidden_voltage_upper", upper)):
            validate_decimal(value, field_name=name)
        if not Decimal(0) <= lower < Decimal("0.5") < upper <= Decimal(1):
            raise ValueError("hidden-mode envelope must contain the 0.5 V carrier")
        if (
            self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        ):
            raise ValueError("simulator challenge evidence ceiling changed")
        if self.physical_execution_authorized:
            raise ValueError("simulator challenge cannot authorize a physical action")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCubeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-cube-result'

    cube_result_id: str
    scale_map_id: str
    receiver_id: str
    object_id: str
    face: PhysicalScaleMorphismCubeFace
    maximum_defect: Decimal
    margin: Decimal
    interaction_count: int
    state: PhysicalScaleMorphismEvidenceState

    def __post_init__(self) -> None:
        for name in ("cube_result_id", "scale_map_id", "receiver_id", "object_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("maximum_defect", "margin"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.interaction_count < 0:
            raise ValueError("cube interaction count must be nonnegative")
        expected = (
            PhysicalScaleMorphismEvidenceState.SUPPORTED
            if self.maximum_defect <= self.margin and self.interaction_count == 0
            else PhysicalScaleMorphismEvidenceState.OPPOSED
        )
        if self.state is not expected:
            raise ValueError("cube disposition is not derived from its defect")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryPanelResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-panel-result'

    panel_id: str
    scale_cells: int
    numerical_view_id: str
    receiver_id: str
    boundary_cell_ids: tuple[str, ...]
    target_boundary_cell_ids: tuple[str, ...]
    sink_boundary_cell_ids: tuple[str, ...]
    codimension_two_cell_ids: tuple[str, ...]
    admit_vertex_count: int

    def __post_init__(self) -> None:
        for name in ("panel_id", "numerical_view_id", "receiver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {16, 32, 64}:
            raise ValueError("boundary panel uses an undeclared scale")
        for name in (
            "boundary_cell_ids",
            "target_boundary_cell_ids",
            "sink_boundary_cell_ids",
            "codimension_two_cell_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if not 0 <= self.admit_vertex_count <= 25:
            raise ValueError("boundary panel admit count lies outside the frozen grid")
        if set(self.boundary_cell_ids) != set(self.target_boundary_cell_ids) | set(
            self.sink_boundary_cell_ids
        ):
            raise ValueError("boundary union differs from its gate-owned strata")
        if set(self.codimension_two_cell_ids) != set(self.target_boundary_cell_ids) & set(
            self.sink_boundary_cell_ids
        ):
            raise ValueError("codimension-two stratum differs from the gate intersection")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryTransportResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-transport-result'

    transport_result_id: str
    axis: PhysicalScaleMorphismBoundaryAxis
    map_id: str
    boundary_symmetric_difference_count: int
    target_stratum_symmetric_difference_count: int
    sink_stratum_symmetric_difference_count: int
    owner_switch_count: int
    orientation_mismatch_count: int
    category_mismatch_count: int
    false_safe_count: int
    false_hold_count: int
    topology_state: PhysicalScaleMorphismEvidenceState
    stratum_state: PhysicalScaleMorphismEvidenceState
    decision_state: PhysicalScaleMorphismEvidenceState

    def __post_init__(self) -> None:
        for name in ("transport_result_id", "map_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        counts = (
            self.boundary_symmetric_difference_count,
            self.target_stratum_symmetric_difference_count,
            self.sink_stratum_symmetric_difference_count,
            self.owner_switch_count,
            self.orientation_mismatch_count,
            self.category_mismatch_count,
            self.false_safe_count,
            self.false_hold_count,
        )
        if min(counts) < 0:
            raise ValueError("boundary transport counts must be nonnegative")
        expected_topology = (
            PhysicalScaleMorphismEvidenceState.SUPPORTED
            if self.boundary_symmetric_difference_count == 0
            else PhysicalScaleMorphismEvidenceState.OPPOSED
        )
        expected_stratum = (
            PhysicalScaleMorphismEvidenceState.SUPPORTED if not any(counts[1:5]) else PhysicalScaleMorphismEvidenceState.OPPOSED
        )
        expected_decision = (
            PhysicalScaleMorphismEvidenceState.SUPPORTED if not any(counts[5:]) else PhysicalScaleMorphismEvidenceState.OPPOSED
        )
        if (
            self.topology_state is not expected_topology
            or self.stratum_state is not expected_stratum
            or self.decision_state is not expected_decision
        ):
            raise ValueError("boundary transport disposition is not derived from counts")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismHiddenClosureResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-hidden-closure-result'

    closure_result_id: str
    scale_cells: int
    mode_family_id: str
    mode_index: int
    present_receiver_defect: Decimal
    lagged_receiver_defect: Decimal
    maximum_future_receiver_defect: Decimal
    maximum_future_metric_defect: Decimal
    category_mismatch_count: int
    false_safe_count: int
    present_collision_state: PhysicalScaleMorphismEvidenceState
    memoryless_closure_state: PhysicalScaleMorphismEvidenceState
    one_lag_closure_state: PhysicalScaleMorphismEvidenceState

    def __post_init__(self) -> None:
        for name in ("closure_result_id", "mode_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {16, 32, 64} or self.mode_index < 0:
            raise ValueError("hidden closure scale or mode index is invalid")
        for name in (
            "present_receiver_defect",
            "lagged_receiver_defect",
            "maximum_future_receiver_defect",
            "maximum_future_metric_defect",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.category_mismatch_count < 0 or self.false_safe_count < 0:
            raise ValueError("hidden closure mismatch counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCollisionTournamentResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-collision-tournament-result'

    tournament_result_id: str
    scale_cells: int
    candidate: PhysicalScaleMorphismCoordinateCandidate
    entered_pair_count: int
    collision_pair_count: int
    adverse_collision_count: int
    categorical_adverse_collision_count: int
    false_safe_collision_count: int
    state: PhysicalScaleMorphismEvidenceState

    def __post_init__(self) -> None:
        validate_stable_id(self.tournament_result_id, field_name="tournament_result_id")
        counts = (
            self.entered_pair_count,
            self.collision_pair_count,
            self.adverse_collision_count,
            self.categorical_adverse_collision_count,
            self.false_safe_collision_count,
        )
        if self.scale_cells not in {16, 32, 64} or min(counts) < 0:
            raise ValueError("coordinate tournament count or scale is invalid")
        if not (
            self.adverse_collision_count <= self.collision_pair_count <= self.entered_pair_count
        ):
            raise ValueError("coordinate tournament counts are inconsistent")
        if self.categorical_adverse_collision_count > self.adverse_collision_count:
            raise ValueError("categorical adverse collisions exceed all adverse collisions")
        if self.false_safe_collision_count > self.categorical_adverse_collision_count:
            raise ValueError("false-safe collisions exceed categorical adverse collisions")
        expected = (
            PhysicalScaleMorphismEvidenceState.UNEVALUABLE
            if self.collision_pair_count == 0
            else (
                PhysicalScaleMorphismEvidenceState.OPPOSED
                if self.adverse_collision_count > 0
                else PhysicalScaleMorphismEvidenceState.SUPPORTED
            )
        )
        if self.state is not expected:
            raise ValueError("coordinate tournament state is not derived from collisions")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSimulatorChallengesResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-simulator-challenges-result'

    result_id: str
    config: ObjectIdentity
    cube_results: tuple[PhysicalScaleMorphismCubeResult, ...]
    boundary_panels: tuple[PhysicalScaleMorphismBoundaryPanelResult, ...]
    boundary_transports: tuple[PhysicalScaleMorphismBoundaryTransportResult, ...]
    hidden_closure_results: tuple[PhysicalScaleMorphismHiddenClosureResult, ...]
    collision_tournament_results: tuple[PhysicalScaleMorphismCollisionTournamentResult, ...]
    result_wording_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_claim_allowed: bool
    independent_recurrence_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != PhysicalScaleMorphismSimulatorChallengesConfig.SCHEMA:
            raise ValueError("simulator challenge result binds the wrong config")
        for name, attribute in (
            ("cube_results", "cube_result_id"),
            ("boundary_panels", "panel_id"),
            ("boundary_transports", "transport_result_id"),
            ("hidden_closure_results", "closure_result_id"),
            ("collision_tournament_results", "tournament_result_id"),
        ):
            values = getattr(self, name)
            identifiers = tuple(getattr(value, attribute) for value in values)
            if identifiers != tuple(sorted(set(identifiers))) or not identifiers:
                raise ValueError(f"{name} must be nonempty, sorted and unique")
        validate_stable_id(self.result_wording_id, field_name="result_wording_id")
        if self.result_wording_id != "bounded-simulator-morphism-challenges-evaluated":
            raise ValueError("simulator result wording exceeds its scope")
        if (
            self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        ):
            raise ValueError("simulator result changed its evidence ceiling")
        if self.physical_claim_allowed or self.independent_recurrence_claim_allowed:
            raise ValueError("simulator result cannot claim physical or independent recurrence")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSimulatorChallengesExecution:
    result: PhysicalScaleMorphismSimulatorChallengesResult
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class _Observation:
    metrics: dict[str, float]
    target_pass: bool
    sink_pass: bool
    admit: bool


@dataclass(frozen=True, slots=True)
class _BoundaryPanel:
    result: PhysicalScaleMorphismBoundaryPanelResult
    observations: tuple[_Observation, ...]
    owners: dict[str, tuple[str, ...]]
    orientations: dict[tuple[str, str], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _HiddenPair:
    pair_id: str
    scale_cells: int
    mode_family_id: str
    mode_index: int
    plus: FloatArray
    minus: FloatArray
    lag_plus: FloatArray
    lag_minus: FloatArray
    receiver: FloatArray
    capacitances: FloatArray
    model: ResistorCapacitorLadderModelConfig


def _decimal(value: float) -> Decimal:
    if not isfinite(float(value)):
        raise ValueError("scientific output must be finite")
    return Decimal(str(float(value)))


def default_simulator_challenges_config() -> PhysicalScaleMorphismSimulatorChallengesConfig:
    """Return the frozen, outcome-blind simulator-only design."""

    parent = default_numerical_morphism_family_config()
    return PhysicalScaleMorphismSimulatorChallengesConfig(
        config_id="physical-scale-morphism-simulator-challenges",
        parent_config=ObjectIdentity.from_record(parent.config_id, parent),
        scale_cells=(16, 32, 64),
        receiver_bin_count=8,
        numerical_views=("exact", "h0-quarter"),
        metric_ids=("e-star", "q-star", "v-max-star"),
        metric_commutator_margin=Decimal("0.01"),
        exact_commutator_margin=Decimal("1e-10"),
        present_collision_margin=Decimal("1e-10"),
        coordinate_collision_margin=Decimal("0.005"),
        future_divergence_floor=Decimal("0.005"),
        history_lag_t_star=Decimal("0.00001"),
        future_times_t_star=(Decimal("0.005"), Decimal("0.01"), Decimal("0.02")),
        hidden_mode_count=4,
        hidden_amplitude_volts=Decimal("0.15"),
        hidden_voltage_envelope=(Decimal("0.10"), Decimal("0.90")),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        physical_execution_authorized=False,
    )


def _receiver_for_model(model: ResistorCapacitorLadderModelConfig, bin_count: int) -> FloatArray:
    capacitances = np.asarray(model.capacitances_farads, dtype=np.float64)
    spec = equal_width_receiver_spec(
        receiver_id=f"r{bin_count}.n{model.scale_cells}",
        scale_cells=model.scale_cells,
        bin_count=bin_count,
    )
    return receiver_matrix(spec, capacitances)


def _observation(
    *,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    model: ResistorCapacitorLadderModelConfig,
    state: FloatArray,
    receiver_id: str,
) -> _Observation:
    capacitances = np.asarray(model.capacitances_farads, dtype=np.float64)
    if receiver_id == "fine" or receiver_id == "r8-guard":
        values = np.asarray(state, dtype=np.float64)
        weights = capacitances
    elif receiver_id == "r8-mean":
        matrix = _receiver_for_model(model, 8)
        values = matrix @ state
        weights = np.asarray(
            [np.sum(capacitances[np.flatnonzero(row)]) for row in matrix],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown receiver: {receiver_id}")
    v_ref = float(design.voltage_reference_volts)
    denominator_charge = model.scale_cells * float(design.capacitance_bar_farads) * v_ref
    denominator_energy = denominator_charge * v_ref
    metrics = {
        "e-star": float(np.sum(weights * values**2) / denominator_energy),
        "q-star": float(np.sum(weights * values) / denominator_charge),
        "v-max-star": float(np.max(values) / v_ref),
    }
    target_pass = metrics["q-star"] >= float(design.target_charge_minimum_q_star)
    sink_pass = metrics["v-max-star"] <= float(design.sink_voltage_maximum_v_star)
    return _Observation(
        metrics=metrics,
        target_pass=target_pass,
        sink_pass=sink_pass,
        admit=target_pass and sink_pass,
    )


def _solve_action_panel(
    config: PhysicalScaleMorphismSimulatorChallengesConfig,
) -> tuple[
    PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    dict[tuple[int, str, str, int, int], _Observation],
    dict[tuple[int, str, int, int], FloatArray],
    dict[int, ResistorCapacitorLadderModelConfig],
]:
    design = default_numerical_morphism_family_config()
    observations: dict[tuple[int, str, str, int, int], _Observation] = {}
    states: dict[tuple[int, str, int, int], FloatArray] = {}
    templates: dict[int, ResistorCapacitorLadderModelConfig] = {}
    for scale_cells in config.scale_cells:
        for amplitude_index, amplitude in enumerate(design.action_amplitudes_u_star):
            for duration_index, duration in enumerate(design.action_durations_t_star):
                model = _reference_model_config(
                    design,
                    scale_cells=scale_cells,
                    amplitude_u_star=amplitude,
                    duration_t_star=duration,
                    output_times_t_star=(Decimal(0), duration),
                    suffix=f"action-panel.a{amplitude_index}.d{duration_index}",
                )
                templates.setdefault(scale_cells, model)
                exact = solve_matrix_exponential(model)
                coarse = solve_backward_euler(
                    model,
                    maximum_step_seconds=(
                        float(design.refinement_base_step_t_star)
                        * _reference_time_scale(design, scale_cells)
                        / 4
                    ),
                )
                for view_id, state in (
                    ("exact", exact.voltages_volts[-1]),
                    ("h0-quarter", coarse.voltages_volts[-1]),
                ):
                    states[(scale_cells, view_id, amplitude_index, duration_index)] = state
                    for receiver_id in ("fine", "r8-guard", "r8-mean"):
                        observations[
                            (
                                scale_cells,
                                view_id,
                                receiver_id,
                                amplitude_index,
                                duration_index,
                            )
                        ] = _observation(
                            design=design,
                            model=model,
                            state=state,
                            receiver_id=receiver_id,
                        )
    return design, observations, states, templates


def _mixed_metric(
    values: dict[tuple[int, str, str], float],
    *,
    source_scale: int,
    target_scale: int,
    receiver_id: str,
    face: PhysicalScaleMorphismCubeFace,
) -> float:
    def value(scale: int, view: str, receiver: str) -> float:
        return values[(scale, view, receiver)]

    if face is PhysicalScaleMorphismCubeFace.RECEIVER_NUMERICAL:
        defects = [
            value(scale, "h0-quarter", receiver_id)
            - value(scale, "exact", receiver_id)
            - value(scale, "h0-quarter", "fine")
            + value(scale, "exact", "fine")
            for scale in (source_scale, target_scale)
        ]
        return max(abs(item) for item in defects)
    if face is PhysicalScaleMorphismCubeFace.RECEIVER_SCALE:
        defects = [
            value(target_scale, view, receiver_id)
            - value(source_scale, view, receiver_id)
            - value(target_scale, view, "fine")
            + value(source_scale, view, "fine")
            for view in ("exact", "h0-quarter")
        ]
        return max(abs(item) for item in defects)
    if face is PhysicalScaleMorphismCubeFace.NUMERICAL_SCALE:
        defects = [
            value(target_scale, "h0-quarter", receiver)
            - value(source_scale, "h0-quarter", receiver)
            - value(target_scale, "exact", receiver)
            + value(source_scale, "exact", receiver)
            for receiver in ("fine", receiver_id)
        ]
        return max(abs(item) for item in defects)
    defect = (
        value(target_scale, "h0-quarter", receiver_id)
        - value(source_scale, "h0-quarter", receiver_id)
        - value(target_scale, "exact", receiver_id)
        + value(source_scale, "exact", receiver_id)
        - value(target_scale, "h0-quarter", "fine")
        + value(source_scale, "h0-quarter", "fine")
        + value(target_scale, "exact", "fine")
        - value(source_scale, "exact", "fine")
    )
    return abs(defect)


def _mixed_category(
    values: dict[tuple[int, str, str], bool],
    *,
    source_scale: int,
    target_scale: int,
    receiver_id: str,
    face: PhysicalScaleMorphismCubeFace,
) -> bool:
    def parity(keys: Iterable[tuple[int, str, str]]) -> bool:
        result = False
        for key in keys:
            result ^= values[key]
        return result

    if face is PhysicalScaleMorphismCubeFace.RECEIVER_NUMERICAL:
        return any(
            parity(
                (
                    (scale, "exact", "fine"),
                    (scale, "exact", receiver_id),
                    (scale, "h0-quarter", "fine"),
                    (scale, "h0-quarter", receiver_id),
                )
            )
            for scale in (source_scale, target_scale)
        )
    if face is PhysicalScaleMorphismCubeFace.RECEIVER_SCALE:
        return any(
            parity(
                (
                    (source_scale, view, "fine"),
                    (source_scale, view, receiver_id),
                    (target_scale, view, "fine"),
                    (target_scale, view, receiver_id),
                )
            )
            for view in ("exact", "h0-quarter")
        )
    if face is PhysicalScaleMorphismCubeFace.NUMERICAL_SCALE:
        return any(
            parity(
                (
                    (source_scale, "exact", receiver),
                    (source_scale, "h0-quarter", receiver),
                    (target_scale, "exact", receiver),
                    (target_scale, "h0-quarter", receiver),
                )
            )
            for receiver in ("fine", receiver_id)
        )
    return parity(
        (
            (source_scale, "exact", "fine"),
            (source_scale, "exact", receiver_id),
            (source_scale, "h0-quarter", "fine"),
            (source_scale, "h0-quarter", receiver_id),
            (target_scale, "exact", "fine"),
            (target_scale, "exact", receiver_id),
            (target_scale, "h0-quarter", "fine"),
            (target_scale, "h0-quarter", receiver_id),
        )
    )


def _cube_results(
    config: PhysicalScaleMorphismSimulatorChallengesConfig,
    observations: dict[tuple[int, str, str, int, int], _Observation],
) -> tuple[PhysicalScaleMorphismCubeResult, ...]:
    design = default_numerical_morphism_family_config()
    rows: list[PhysicalScaleMorphismCubeResult] = []
    for source_scale, target_scale in ((16, 32), (32, 64), (16, 64)):
        scale_map_id = f"n{source_scale}-to-n{target_scale}"
        for receiver_id in ("r8-guard", "r8-mean"):
            for face in PhysicalScaleMorphismCubeFace:
                for object_id in config.metric_ids:
                    maximum = 0.0
                    for amplitude_index in range(len(design.action_amplitudes_u_star)):
                        for duration_index in range(len(design.action_durations_t_star)):
                            corner_values = {
                                (scale, view, receiver): observations[
                                    (scale, view, receiver, amplitude_index, duration_index)
                                ].metrics[object_id]
                                for scale in (source_scale, target_scale)
                                for view in config.numerical_views
                                for receiver in ("fine", receiver_id)
                            }
                            maximum = max(
                                maximum,
                                _mixed_metric(
                                    corner_values,
                                    source_scale=source_scale,
                                    target_scale=target_scale,
                                    receiver_id=receiver_id,
                                    face=face,
                                ),
                            )
                    margin = (
                        float(config.exact_commutator_margin)
                        if receiver_id == "r8-guard" or object_id == "q-star"
                        else float(config.metric_commutator_margin)
                    )
                    state = (
                        PhysicalScaleMorphismEvidenceState.SUPPORTED
                        if maximum <= margin
                        else PhysicalScaleMorphismEvidenceState.OPPOSED
                    )
                    rows.append(
                        PhysicalScaleMorphismCubeResult(
                            cube_result_id=(
                                f"cube.{scale_map_id}.{receiver_id}.{face.value.lower()}"
                                f".{object_id}"
                            ),
                            scale_map_id=scale_map_id,
                            receiver_id=receiver_id,
                            object_id=object_id,
                            face=face,
                            maximum_defect=_decimal(maximum),
                            margin=_decimal(margin),
                            interaction_count=0,
                            state=state,
                        )
                    )
                interactions = 0
                for amplitude_index in range(len(design.action_amplitudes_u_star)):
                    for duration_index in range(len(design.action_durations_t_star)):
                        corner_labels = {
                            (scale, view, receiver): observations[
                                (scale, view, receiver, amplitude_index, duration_index)
                            ].admit
                            for scale in (source_scale, target_scale)
                            for view in config.numerical_views
                            for receiver in ("fine", receiver_id)
                        }
                        interactions += int(
                            _mixed_category(
                                corner_labels,
                                source_scale=source_scale,
                                target_scale=target_scale,
                                receiver_id=receiver_id,
                                face=face,
                            )
                        )
                rows.append(
                    PhysicalScaleMorphismCubeResult(
                        cube_result_id=(
                            f"cube.{scale_map_id}.{receiver_id}.{face.value.lower()}.admission"
                        ),
                        scale_map_id=scale_map_id,
                        receiver_id=receiver_id,
                        object_id="admission",
                        face=face,
                        maximum_defect=Decimal(0),
                        margin=Decimal(0),
                        interaction_count=interactions,
                        state=(
                            PhysicalScaleMorphismEvidenceState.SUPPORTED
                            if interactions == 0
                            else PhysicalScaleMorphismEvidenceState.OPPOSED
                        ),
                    )
                )
    return tuple(sorted(rows, key=lambda value: value.cube_result_id))


def _cell_id(amplitude_index: int, duration_index: int) -> str:
    return f"cell.a{amplitude_index:02d}.d{duration_index:02d}"


def _orientation(corners: tuple[bool, bool, bool, bool]) -> tuple[int, int]:
    # Corner order is low/low, low/high, high/low, high/high.
    scores = tuple(1 if value else -1 for value in corners)
    return (
        int(np.sign(scores[2] + scores[3] - scores[0] - scores[1])),
        int(np.sign(scores[1] + scores[3] - scores[0] - scores[2])),
    )


def _boundary_panel(
    *,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    scale_cells: int,
    view_id: str,
    receiver_id: str,
    observations: dict[tuple[int, str, str, int, int], _Observation],
) -> _BoundaryPanel:
    owners: dict[str, tuple[str, ...]] = {}
    orientations: dict[tuple[str, str], tuple[int, int]] = {}
    target_cells: list[str] = []
    sink_cells: list[str] = []
    for amplitude_index in range(len(design.action_amplitudes_u_star) - 1):
        for duration_index in range(len(design.action_durations_t_star) - 1):
            indices = (
                (amplitude_index, duration_index),
                (amplitude_index, duration_index + 1),
                (amplitude_index + 1, duration_index),
                (amplitude_index + 1, duration_index + 1),
            )
            corners = tuple(
                observations[(scale_cells, view_id, receiver_id, a_index, d_index)]
                for a_index, d_index in indices
            )
            cell_id = _cell_id(amplitude_index, duration_index)
            active: list[str] = []
            target_signs = (
                corners[0].target_pass,
                corners[1].target_pass,
                corners[2].target_pass,
                corners[3].target_pass,
            )
            sink_signs = (
                corners[0].sink_pass,
                corners[1].sink_pass,
                corners[2].sink_pass,
                corners[3].sink_pass,
            )
            for gate_id, signs in (("target", target_signs), ("sink", sink_signs)):
                if len(set(signs)) > 1:
                    active.append(gate_id)
                    orientations[(cell_id, gate_id)] = _orientation(signs)
                    (target_cells if gate_id == "target" else sink_cells).append(cell_id)
            if active:
                owners[cell_id] = tuple(active)
    boundary_cells = tuple(sorted(owners))
    target_tuple = tuple(sorted(target_cells))
    sink_tuple = tuple(sorted(sink_cells))
    panel_id = f"boundary.n{scale_cells}.{view_id}.{receiver_id}"
    panel_observations = tuple(
        observations[(scale_cells, view_id, receiver_id, amplitude_index, duration_index)]
        for amplitude_index in range(len(design.action_amplitudes_u_star))
        for duration_index in range(len(design.action_durations_t_star))
    )
    return _BoundaryPanel(
        result=PhysicalScaleMorphismBoundaryPanelResult(
            panel_id=panel_id,
            scale_cells=scale_cells,
            numerical_view_id=view_id,
            receiver_id=receiver_id,
            boundary_cell_ids=boundary_cells,
            target_boundary_cell_ids=target_tuple,
            sink_boundary_cell_ids=sink_tuple,
            codimension_two_cell_ids=tuple(sorted(set(target_tuple) & set(sink_tuple))),
            admit_vertex_count=sum(value.admit for value in panel_observations),
        ),
        observations=panel_observations,
        owners=owners,
        orientations=orientations,
    )


def _boundary_transport(
    *,
    result_id: str,
    axis: PhysicalScaleMorphismBoundaryAxis,
    map_id: str,
    source: _BoundaryPanel,
    target: _BoundaryPanel,
) -> PhysicalScaleMorphismBoundaryTransportResult:
    source_cells = set(source.result.boundary_cell_ids)
    target_cells = set(target.result.boundary_cell_ids)
    common = source_cells & target_cells
    owner_switches = sum(source.owners[cell_id] != target.owners[cell_id] for cell_id in common)
    orientation_mismatches = 0
    for cell_id in common:
        for gate_id in set(source.owners[cell_id]) & set(target.owners[cell_id]):
            orientation_mismatches += int(
                source.orientations[(cell_id, gate_id)] != target.orientations[(cell_id, gate_id)]
            )
    mismatches = sum(
        left.admit != right.admit
        for left, right in zip(source.observations, target.observations, strict=True)
    )
    false_safe = sum(
        not left.admit and right.admit
        for left, right in zip(source.observations, target.observations, strict=True)
    )
    false_hold = sum(
        left.admit and not right.admit
        for left, right in zip(source.observations, target.observations, strict=True)
    )
    boundary_diff = len(source_cells ^ target_cells)
    target_diff = len(
        set(source.result.target_boundary_cell_ids) ^ set(target.result.target_boundary_cell_ids)
    )
    sink_diff = len(
        set(source.result.sink_boundary_cell_ids) ^ set(target.result.sink_boundary_cell_ids)
    )
    return PhysicalScaleMorphismBoundaryTransportResult(
        transport_result_id=result_id,
        axis=axis,
        map_id=map_id,
        boundary_symmetric_difference_count=boundary_diff,
        target_stratum_symmetric_difference_count=target_diff,
        sink_stratum_symmetric_difference_count=sink_diff,
        owner_switch_count=owner_switches,
        orientation_mismatch_count=orientation_mismatches,
        category_mismatch_count=mismatches,
        false_safe_count=false_safe,
        false_hold_count=false_hold,
        topology_state=(
            PhysicalScaleMorphismEvidenceState.SUPPORTED if boundary_diff == 0 else PhysicalScaleMorphismEvidenceState.OPPOSED
        ),
        stratum_state=(
            PhysicalScaleMorphismEvidenceState.SUPPORTED
            if not any((target_diff, sink_diff, owner_switches, orientation_mismatches))
            else PhysicalScaleMorphismEvidenceState.OPPOSED
        ),
        decision_state=(
            PhysicalScaleMorphismEvidenceState.SUPPORTED
            if not any((mismatches, false_safe, false_hold))
            else PhysicalScaleMorphismEvidenceState.OPPOSED
        ),
    )


def _boundary_results(
    config: PhysicalScaleMorphismSimulatorChallengesConfig,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    observations: dict[tuple[int, str, str, int, int], _Observation],
) -> tuple[tuple[PhysicalScaleMorphismBoundaryPanelResult, ...], tuple[PhysicalScaleMorphismBoundaryTransportResult, ...]]:
    panels = {
        (scale, view, receiver): _boundary_panel(
            design=design,
            scale_cells=scale,
            view_id=view,
            receiver_id=receiver,
            observations=observations,
        )
        for scale in config.scale_cells
        for view in config.numerical_views
        for receiver in ("fine", "r8-guard", "r8-mean")
    }
    transports: list[PhysicalScaleMorphismBoundaryTransportResult] = []
    for scale in config.scale_cells:
        for view in config.numerical_views:
            for receiver in ("r8-guard", "r8-mean"):
                transports.append(
                    _boundary_transport(
                        result_id=f"boundary-map.receiver.n{scale}.{view}.{receiver}",
                        axis=PhysicalScaleMorphismBoundaryAxis.RECEIVER,
                        map_id=f"n{scale}.{view}.fine-to-{receiver}",
                        source=panels[(scale, view, "fine")],
                        target=panels[(scale, view, receiver)],
                    )
                )
    for scale in config.scale_cells:
        for receiver in ("fine", "r8-guard", "r8-mean"):
            transports.append(
                _boundary_transport(
                    result_id=f"boundary-map.numerical.n{scale}.{receiver}",
                    axis=PhysicalScaleMorphismBoundaryAxis.NUMERICAL,
                    map_id=f"n{scale}.{receiver}.exact-to-h0-quarter",
                    source=panels[(scale, "exact", receiver)],
                    target=panels[(scale, "h0-quarter", receiver)],
                )
            )
    for source_scale, target_scale in ((16, 32), (32, 64), (16, 64)):
        for view in config.numerical_views:
            for receiver in ("fine", "r8-guard", "r8-mean"):
                transports.append(
                    _boundary_transport(
                        result_id=(
                            f"boundary-map.scale.n{source_scale}.n{target_scale}.{view}.{receiver}"
                        ),
                        axis=PhysicalScaleMorphismBoundaryAxis.SCALE,
                        map_id=f"n{source_scale}-to-n{target_scale}.{view}.{receiver}",
                        source=panels[(source_scale, view, receiver)],
                        target=panels[(target_scale, view, receiver)],
                    )
                )
    return (
        tuple(
            sorted((panel.result for panel in panels.values()), key=lambda value: value.panel_id)
        ),
        tuple(sorted(transports, key=lambda value: value.transport_result_id)),
    )


def _canonicalize_mode_sign(mode: FloatArray) -> FloatArray:
    result = np.asarray(mode, dtype=np.float64)
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0:
        result = -result
    maximum = float(np.max(np.abs(result)))
    if maximum <= 0:
        raise ValueError("hidden mode is numerically zero")
    return result / maximum


def _hidden_modes(
    *,
    operator: FloatArray,
    receiver: FloatArray,
    time_scale: float,
    future_times: tuple[Decimal, ...],
    constraints: FloatArray,
    count: int,
) -> tuple[FloatArray, ...]:
    basis = null_space(constraints)
    if basis.shape[1] == 0:
        return ()
    observability = np.concatenate(
        [receiver @ expm(operator * (float(value) * time_scale)) for value in future_times],
        axis=0,
    )
    _, _, right = np.linalg.svd(observability @ basis, full_matrices=False)
    directions = []
    for index in range(min(count, right.shape[0])):
        directions.append(_canonicalize_mode_sign(basis @ right[index]))
    return tuple(directions)


def _hidden_pairs(
    config: PhysicalScaleMorphismSimulatorChallengesConfig,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    templates: dict[int, ResistorCapacitorLadderModelConfig],
) -> tuple[_HiddenPair, ...]:
    pairs: list[_HiddenPair] = []
    for scale_cells in config.scale_cells:
        model = templates[scale_cells]
        operator = build_operator(model).state_matrix
        receiver = _receiver_for_model(model, config.receiver_bin_count)
        time_scale = _reference_time_scale(design, scale_cells)
        lag_propagator = expm(-operator * (float(config.history_lag_t_star) * time_scale))
        constraint_families = (
            ("present-kernel", receiver),
            ("present-one-lag-kernel", np.concatenate((receiver, receiver @ lag_propagator))),
        )
        for family_id, constraints in constraint_families:
            modes = _hidden_modes(
                operator=operator,
                receiver=receiver,
                time_scale=time_scale,
                future_times=config.future_times_t_star,
                constraints=constraints,
                count=config.hidden_mode_count,
            )
            for mode_index, mode in enumerate(modes):
                lag_mode = lag_propagator @ mode
                maximum = max(float(np.max(np.abs(mode))), float(np.max(np.abs(lag_mode))))
                envelope_half_width = min(
                    Decimal("0.5") - config.hidden_voltage_envelope[0],
                    config.hidden_voltage_envelope[1] - Decimal("0.5"),
                )
                amplitude = min(
                    float(config.hidden_amplitude_volts),
                    float(envelope_half_width) / maximum,
                )
                present_plus = np.full(scale_cells, 0.5) + amplitude * mode
                present_minus = np.full(scale_cells, 0.5) - amplitude * mode
                lag_plus = np.full(scale_cells, 0.5) + amplitude * lag_mode
                lag_minus = np.full(scale_cells, 0.5) - amplitude * lag_mode
                lower = float(config.hidden_voltage_envelope[0]) - 1e-12
                upper = float(config.hidden_voltage_envelope[1]) + 1e-12
                if any(
                    np.min(values) < lower or np.max(values) > upper
                    for values in (present_plus, present_minus, lag_plus, lag_minus)
                ):
                    raise RuntimeError("hidden-mode envelope construction failed")
                pairs.append(
                    _HiddenPair(
                        pair_id=f"hidden.n{scale_cells}.{family_id}.m{mode_index}",
                        scale_cells=scale_cells,
                        mode_family_id=family_id,
                        mode_index=mode_index,
                        plus=present_plus,
                        minus=present_minus,
                        lag_plus=lag_plus,
                        lag_minus=lag_minus,
                        receiver=receiver,
                        capacitances=np.asarray(model.capacitances_farads, dtype=np.float64),
                        model=model,
                    )
                )
    return tuple(sorted(pairs, key=lambda value: value.pair_id))


def _forced_action_states(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    templates: dict[int, ResistorCapacitorLadderModelConfig],
) -> dict[tuple[int, int, int], tuple[FloatArray, FloatArray]]:
    values: dict[tuple[int, int, int], tuple[FloatArray, FloatArray]] = {}
    for scale_cells, template in templates.items():
        operator = build_operator(template).state_matrix
        for amplitude_index, amplitude in enumerate(design.action_amplitudes_u_star):
            for duration_index, duration in enumerate(design.action_durations_t_star):
                model = _reference_model_config(
                    design,
                    scale_cells=scale_cells,
                    amplitude_u_star=amplitude,
                    duration_t_star=duration,
                    output_times_t_star=(Decimal(0), duration),
                    suffix=f"hidden-profile-action.a{amplitude_index}.d{duration_index}",
                )
                forced = solve_matrix_exponential(model).voltages_volts[-1]
                propagator = expm(
                    operator * (float(duration) * _reference_time_scale(design, scale_cells))
                )
                values[(scale_cells, amplitude_index, duration_index)] = (forced, propagator)
    return values


def _guard_coordinates(
    pair: _HiddenPair,
    state: FloatArray,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
) -> FloatArray:
    c_bar = float(design.capacitance_bar_farads)
    n_cells = pair.scale_cells
    q_star = float(np.sum(pair.capacitances * state) / (n_cells * c_bar))
    e_star = float(np.sum(pair.capacitances * state**2) / (n_cells * c_bar))
    left_current_star = 0.5 - float(state[0])
    right_current_star = 0.5 - float(state[-1])
    return np.concatenate(
        (
            pair.receiver @ state,
            np.asarray(
                [
                    q_star,
                    e_star,
                    float(np.max(state)),
                    float(np.min(state)),
                    left_current_star,
                    right_current_star,
                ]
            ),
        )
    )


def _candidate_coordinates(
    pair: _HiddenPair,
    candidate: PhysicalScaleMorphismCoordinateCandidate,
    state: FloatArray,
    lag_state: FloatArray,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
) -> FloatArray:
    if candidate is PhysicalScaleMorphismCoordinateCandidate.ACTION_ONLY:
        return np.asarray([0.0, 0.0], dtype=np.float64)
    if candidate is PhysicalScaleMorphismCoordinateCandidate.EIGHT_BIN_RECEIVER_CURRENT:
        return pair.receiver @ state
    if candidate is PhysicalScaleMorphismCoordinateCandidate.EIGHT_BIN_RECEIVER_CURRENT_ONE_LAG:
        return np.concatenate((pair.receiver @ state, pair.receiver @ lag_state))
    if candidate is PhysicalScaleMorphismCoordinateCandidate.GUARDED_EIGHT_BIN_RECEIVER_CURRENT:
        return _guard_coordinates(pair, state, design)
    return state


@dataclass(frozen=True, slots=True)
class _PairOutcome:
    maximum_receiver_defect: float
    maximum_metric_defect: float
    category_mismatch_count: int
    false_safe_count: int


def _pair_outcome(
    *,
    pair: _HiddenPair,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    forced_states: dict[tuple[int, int, int], tuple[FloatArray, FloatArray]],
) -> _PairOutcome:
    maximum_receiver = 0.0
    maximum_metric = 0.0
    category_mismatches = 0
    false_safe = 0
    for amplitude_index in range(len(design.action_amplitudes_u_star)):
        for duration_index in range(len(design.action_durations_t_star)):
            forced, propagator = forced_states[(pair.scale_cells, amplitude_index, duration_index)]
            plus_state = forced + propagator @ pair.plus
            minus_state = forced + propagator @ pair.minus
            plus_observation = _observation(
                design=design,
                model=pair.model,
                state=plus_state,
                receiver_id="fine",
            )
            minus_observation = _observation(
                design=design,
                model=pair.model,
                state=minus_state,
                receiver_id="fine",
            )
            maximum_receiver = max(
                maximum_receiver,
                float(np.max(np.abs(pair.receiver @ (plus_state - minus_state)))),
            )
            maximum_metric = max(
                maximum_metric,
                *(
                    abs(plus_observation.metrics[key] - minus_observation.metrics[key])
                    for key in design_metric_ids()
                ),
            )
            if plus_observation.admit != minus_observation.admit:
                category_mismatches += 1
                false_safe += 1
    return _PairOutcome(
        maximum_receiver_defect=maximum_receiver,
        maximum_metric_defect=maximum_metric,
        category_mismatch_count=category_mismatches,
        false_safe_count=false_safe,
    )


def design_metric_ids() -> tuple[str, ...]:
    """Small public helper used by scientific tests to pin the metric roster."""

    return ("e-star", "q-star", "v-max-star")


def _hidden_and_collision_results(
    config: PhysicalScaleMorphismSimulatorChallengesConfig,
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    templates: dict[int, ResistorCapacitorLadderModelConfig],
) -> tuple[
    tuple[PhysicalScaleMorphismHiddenClosureResult, ...],
    tuple[PhysicalScaleMorphismCollisionTournamentResult, ...],
    dict[str, FloatArray],
]:
    pairs = _hidden_pairs(config, design, templates)
    forced_states = _forced_action_states(design, templates)
    hidden_rows: list[PhysicalScaleMorphismHiddenClosureResult] = []
    arrays: dict[str, FloatArray] = {}
    pair_outcomes: dict[str, _PairOutcome] = {}
    for pair in pairs:
        outcome = _pair_outcome(pair=pair, design=design, forced_states=forced_states)
        pair_outcomes[pair.pair_id] = outcome
        present_defect = float(np.max(np.abs(pair.receiver @ (pair.plus - pair.minus))))
        lag_defect = float(np.max(np.abs(pair.receiver @ (pair.lag_plus - pair.lag_minus))))
        future_adverse = (
            outcome.maximum_receiver_defect >= float(config.future_divergence_floor)
            or outcome.maximum_metric_defect >= float(config.future_divergence_floor)
            or outcome.category_mismatch_count > 0
        )
        present_collision = present_defect <= float(config.present_collision_margin)
        one_lag_collision = max(present_defect, lag_defect) <= float(
            config.present_collision_margin
        )
        hidden_rows.append(
            PhysicalScaleMorphismHiddenClosureResult(
                closure_result_id=pair.pair_id,
                scale_cells=pair.scale_cells,
                mode_family_id=pair.mode_family_id,
                mode_index=pair.mode_index,
                present_receiver_defect=_decimal(present_defect),
                lagged_receiver_defect=_decimal(lag_defect),
                maximum_future_receiver_defect=_decimal(outcome.maximum_receiver_defect),
                maximum_future_metric_defect=_decimal(outcome.maximum_metric_defect),
                category_mismatch_count=outcome.category_mismatch_count,
                false_safe_count=outcome.false_safe_count,
                present_collision_state=(
                    PhysicalScaleMorphismEvidenceState.SUPPORTED
                    if present_collision
                    else PhysicalScaleMorphismEvidenceState.OPPOSED
                ),
                memoryless_closure_state=(
                    PhysicalScaleMorphismEvidenceState.OPPOSED
                    if present_collision and future_adverse
                    else (
                        PhysicalScaleMorphismEvidenceState.SUPPORTED
                        if present_collision
                        else PhysicalScaleMorphismEvidenceState.UNEVALUABLE
                    )
                ),
                one_lag_closure_state=(
                    PhysicalScaleMorphismEvidenceState.OPPOSED
                    if one_lag_collision and future_adverse
                    else (
                        PhysicalScaleMorphismEvidenceState.SUPPORTED
                        if one_lag_collision
                        else PhysicalScaleMorphismEvidenceState.UNEVALUABLE
                    )
                ),
            )
        )
        arrays[f"{pair.pair_id}.plus"] = pair.plus
        arrays[f"{pair.pair_id}.minus"] = pair.minus
        arrays[f"{pair.pair_id}.lag-plus"] = pair.lag_plus
        arrays[f"{pair.pair_id}.lag-minus"] = pair.lag_minus

    tournament_rows: list[PhysicalScaleMorphismCollisionTournamentResult] = []
    for scale_cells in config.scale_cells:
        scale_pairs = tuple(pair for pair in pairs if pair.scale_cells == scale_cells)
        for candidate in PhysicalScaleMorphismCoordinateCandidate:
            collision_count = 0
            adverse_count = 0
            categorical_count = 0
            false_safe_count = 0
            for pair in scale_pairs:
                plus_coordinates = _candidate_coordinates(
                    pair, candidate, pair.plus, pair.lag_plus, design
                )
                minus_coordinates = _candidate_coordinates(
                    pair, candidate, pair.minus, pair.lag_minus, design
                )
                margin = (
                    float(config.present_collision_margin)
                    if candidate
                    in {
                        PhysicalScaleMorphismCoordinateCandidate.ACTION_ONLY,
                        PhysicalScaleMorphismCoordinateCandidate.FINE_CURRENT,
                        PhysicalScaleMorphismCoordinateCandidate.EIGHT_BIN_RECEIVER_CURRENT,
                    }
                    else float(config.coordinate_collision_margin)
                )
                collision = float(np.max(np.abs(plus_coordinates - minus_coordinates))) <= margin
                if not collision:
                    continue
                collision_count += 1
                outcome = pair_outcomes[pair.pair_id]
                adverse = (
                    outcome.maximum_receiver_defect >= float(config.future_divergence_floor)
                    or outcome.maximum_metric_defect >= float(config.future_divergence_floor)
                    or outcome.category_mismatch_count > 0
                )
                adverse_count += int(adverse)
                categorical_count += int(outcome.category_mismatch_count > 0)
                false_safe_count += int(outcome.false_safe_count > 0)
            state = (
                PhysicalScaleMorphismEvidenceState.UNEVALUABLE
                if collision_count == 0
                else (
                    PhysicalScaleMorphismEvidenceState.OPPOSED
                    if adverse_count > 0
                    else PhysicalScaleMorphismEvidenceState.SUPPORTED
                )
            )
            tournament_rows.append(
                PhysicalScaleMorphismCollisionTournamentResult(
                    tournament_result_id=f"tournament.n{scale_cells}.{candidate.value.lower().replace('_', '-')}",
                    scale_cells=scale_cells,
                    candidate=candidate,
                    entered_pair_count=len(scale_pairs),
                    collision_pair_count=collision_count,
                    adverse_collision_count=adverse_count,
                    categorical_adverse_collision_count=categorical_count,
                    false_safe_collision_count=false_safe_count,
                    state=state,
                )
            )
    return (
        tuple(sorted(hidden_rows, key=lambda value: value.closure_result_id)),
        tuple(sorted(tournament_rows, key=lambda value: value.tournament_result_id)),
        arrays,
    )


def _run_simulator_morphism_challenges(
    config: PhysicalScaleMorphismSimulatorChallengesConfig | None = None,
) -> PhysicalScaleMorphismSimulatorChallengesExecution:
    """Execute the four bounded simulator-only morphism challenges."""

    design = config or default_simulator_challenges_config()
    parent, observations, states, templates = _solve_action_panel(design)
    cube = _cube_results(design, observations)
    boundary_panels, boundary_transports = _boundary_results(design, parent, observations)
    hidden, tournament, hidden_arrays = _hidden_and_collision_results(design, parent, templates)
    arrays = {
        f"panel.n{scale}.{view}.a{a_index}.d{d_index}": state
        for (scale, view, a_index, d_index), state in states.items()
    }
    arrays.update(hidden_arrays)
    result = PhysicalScaleMorphismSimulatorChallengesResult(
        result_id="physical-scale-morphism-simulator-challenges-result",
        config=ObjectIdentity.from_record(design.config_id, design),
        cube_results=cube,
        boundary_panels=boundary_panels,
        boundary_transports=boundary_transports,
        hidden_closure_results=hidden,
        collision_tournament_results=tournament,
        result_wording_id="bounded-simulator-morphism-challenges-evaluated",
        evidence_ceiling=design.evidence_ceiling,
        outcome_access=design.outcome_access,
        visibility_ceiling=design.visibility_ceiling,
        physical_claim_allowed=False,
        independent_recurrence_claim_allowed=False,
    )
    return PhysicalScaleMorphismSimulatorChallengesExecution(result=result, arrays=arrays)


def run_simulator_morphism_challenges(
    config: PhysicalScaleMorphismSimulatorChallengesConfig | None = None,
) -> PhysicalScaleMorphismSimulatorChallengesExecution:
    """Execute the challenges under a deterministic linear-algebra budget."""

    with threadpool_limits(limits=PHYSICAL_SCALE_MORPHISM_REFERENCE_BLAS_THREAD_COUNT):
        return _run_simulator_morphism_challenges(config)


__all__ = [
    'PhysicalScaleMorphismBoundaryAxis',
    'PhysicalScaleMorphismBoundaryPanelResult',
    'PhysicalScaleMorphismBoundaryTransportResult',
    'PhysicalScaleMorphismCollisionTournamentResult',
    'PhysicalScaleMorphismCoordinateCandidate',
    'PhysicalScaleMorphismCubeFace',
    'PhysicalScaleMorphismCubeResult',
    'PhysicalScaleMorphismHiddenClosureResult',
    'PhysicalScaleMorphismSimulatorChallengesConfig',
    'PhysicalScaleMorphismSimulatorChallengesExecution',
    'PhysicalScaleMorphismSimulatorChallengesResult',
    "default_simulator_challenges_config",
    "design_metric_ids",
    "run_simulator_morphism_challenges",
]
