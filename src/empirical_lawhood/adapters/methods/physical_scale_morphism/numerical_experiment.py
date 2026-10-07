"""Executable numerical morphism-family experiment for physical scale morphism.

This module is deliberately scientific rather than orchestration infrastructure.
It runs the already implemented RC operator, receiver reductions,
nondimensionalization and refinement methods on a finite predeclared panel.  Its
results are numerical-twin evidence only: they do not stand in for fabricated
boards or physical recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import cos, pi, sin
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderActionSegment, ResistorCapacitorLadderModelConfig, ResistorCapacitorLadderNumericalSummary
from empirical_lawhood.adapters.simulators.rc_ladder_response.matrix_exponential import ResistorCapacitorLadderTrajectory, solve_matrix_exponential
from empirical_lawhood.adapters.simulators.rc_ladder_response.mna import build_operator, slowest_decay_rate
from empirical_lawhood.adapters.simulators.rc_ladder_response.refinement import refinement_chain
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismEvidenceWorld, PhysicalScaleMorphismMapFamily
from .dimensionless import PhysicalScaleMorphismDimensionlessRecord, PhysicalScaleMorphismNormalizationSpec, PhysicalScaleMorphismNormalizationUncertainty, compute_dimensionless_record
from .evaluation_evidence import PhysicalScaleMorphismEvidenceState
from .receivers import composition_defect, equal_width_receiver_spec, receiver_matrix


FloatArray = npt.NDArray[np.float64]

# The original accepted fingerprints were produced with this OpenBLAS worker count.
# Preserve that historical numerical view explicitly instead of inheriting
# collection order or the caller's process-wide thread-pool state.
PHYSICAL_SCALE_MORPHISM_REFERENCE_BLAS_THREAD_COUNT = 16


class PhysicalScaleMorphismSurvivalPrediction(StrEnum):
    NONSURVIVOR = "NONSURVIVOR"
    SURVIVOR = "SURVIVOR"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNumericalMorphismFamilyConfig(CanonicalRecord):
    """Finite, result-blind design for the first executable morphism panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-numerical-morphism-family-config'

    config_id: str
    scale_cells: tuple[int, ...]
    receiver_bin_counts: tuple[int, ...]
    resistance_bar_ohms: Decimal
    capacitance_bar_farads: Decimal
    voltage_reference_volts: Decimal
    component_variation_fraction: Decimal
    output_times_t_star: tuple[Decimal, ...]
    probe_time_t_star: Decimal
    refinement_base_step_t_star: Decimal
    numerical_voltage_margin_volts: Decimal
    receiver_exact_margin: Decimal
    receiver_loss_margin: Decimal
    scale_relative_margin: Decimal
    negative_control_minimum_defect: Decimal
    action_amplitudes_u_star: tuple[Decimal, ...]
    action_durations_t_star: tuple[Decimal, ...]
    target_charge_minimum_q_star: Decimal
    sink_voltage_maximum_v_star: Decimal
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_execution_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.scale_cells != (16, 32, 64):
            raise ValueError("first numerical morphism panel requires N=16,32,64")
        if self.receiver_bin_counts != (4, 8, 16):
            raise ValueError("first numerical morphism panel requires R4,R8,R16")
        for name in (
            "resistance_bar_ohms",
            "capacitance_bar_farads",
            "voltage_reference_volts",
            "component_variation_fraction",
            "probe_time_t_star",
            "refinement_base_step_t_star",
            "numerical_voltage_margin_volts",
            "receiver_exact_margin",
            "receiver_loss_margin",
            "scale_relative_margin",
            "negative_control_minimum_defect",
            "target_charge_minimum_q_star",
            "sink_voltage_maximum_v_star",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            min(
                self.resistance_bar_ohms,
                self.capacitance_bar_farads,
                self.voltage_reference_volts,
                self.probe_time_t_star,
                self.refinement_base_step_t_star,
                self.numerical_voltage_margin_volts,
                self.receiver_exact_margin,
                self.receiver_loss_margin,
                self.scale_relative_margin,
                self.negative_control_minimum_defect,
            )
            <= 0
        ):
            raise ValueError("numerical morphism scales and margins must be positive")
        if not Decimal(0) < self.component_variation_fraction < Decimal(1):
            raise ValueError("component variation must lie inside (0,1)")
        if tuple(sorted(set(self.output_times_t_star))) != self.output_times_t_star:
            raise ValueError("normalized output times must be sorted and unique")
        if (
            len(self.output_times_t_star) < 3
            or self.output_times_t_star[0] != 0
            or self.output_times_t_star[-1] != self.probe_time_t_star
        ):
            raise ValueError("output grid must run from zero to the probe time")
        for name in ("action_amplitudes_u_star", "action_durations_t_star"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or len(values) < 3 or values[0] <= 0:
                raise ValueError(f"{name} must be a sorted positive finite grid")
        if self.action_durations_t_star[-1] > self.probe_time_t_star:
            raise ValueError("action grid extends beyond the normalized horizon")
        if not Decimal(0) < self.sink_voltage_maximum_v_star <= Decimal(1):
            raise ValueError("sink threshold must lie inside (0,1]")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("numerical morphism development has the exact local law ceiling")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("numerical morphism results are development-visible")
        if self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY:
            raise ValueError("numerical morphism results are development-only")
        if self.physical_execution_authorized:
            raise ValueError("numerical configuration cannot authorize physical execution")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMorphismPropertyResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-property-result'

    property_result_id: str
    map_family: PhysicalScaleMorphismMapFamily
    map_id: str
    property_id: str
    prediction: PhysicalScaleMorphismSurvivalPrediction
    observed_survival: bool
    prediction_state: PhysicalScaleMorphismEvidenceState
    source_value: Decimal
    target_value: Decimal
    absolute_defect: Decimal
    relative_defect: Decimal
    margin: Decimal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("property_result_id", "map_id", "property_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "source_value",
            "target_value",
            "absolute_defect",
            "relative_defect",
            "margin",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.absolute_defect < 0 or self.relative_defect < 0 or self.margin < 0:
            raise ValueError("morphism defects and margins must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = self.observed_survival == (self.prediction is PhysicalScaleMorphismSurvivalPrediction.SURVIVOR)
        expected_state = PhysicalScaleMorphismEvidenceState.SUPPORTED if expected else PhysicalScaleMorphismEvidenceState.OPPOSED
        if self.prediction_state is not expected_state:
            raise ValueError("property prediction state is not derived from observed survival")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryMapResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-map-result'

    boundary_result_id: str
    map_family: PhysicalScaleMorphismMapFamily
    map_id: str
    source_boundary_cell_ids: tuple[str, ...]
    target_boundary_cell_ids: tuple[str, ...]
    symmetric_difference_cell_ids: tuple[str, ...]
    category_mismatch_count: int
    false_safe_count: int
    false_hold_count: int
    state: PhysicalScaleMorphismEvidenceState
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("boundary_result_id", "map_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "source_boundary_cell_ids",
            "target_boundary_cell_ids",
            "symmetric_difference_cell_ids",
            "reason_codes",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if min(self.category_mismatch_count, self.false_safe_count, self.false_hold_count) < 0:
            raise ValueError("boundary mismatch counts must be nonnegative")
        if not self.source_boundary_cell_ids or not self.target_boundary_cell_ids:
            expected = PhysicalScaleMorphismEvidenceState.UNEVALUABLE
        elif (
            not self.symmetric_difference_cell_ids
            and self.category_mismatch_count == 0
            and self.false_safe_count == 0
        ):
            expected = PhysicalScaleMorphismEvidenceState.SUPPORTED
        else:
            expected = PhysicalScaleMorphismEvidenceState.OPPOSED
        if self.state is not expected:
            raise ValueError("boundary state is not derived from its finite comparison")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNumericalMorphismFamilyResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-numerical-morphism-family-result'

    result_id: str
    config: ObjectIdentity
    evidence_world: PhysicalScaleMorphismEvidenceWorld
    numerical_summaries: tuple[ResistorCapacitorLadderNumericalSummary, ...]
    property_results: tuple[PhysicalScaleMorphismMorphismPropertyResult, ...]
    boundary_results: tuple[PhysicalScaleMorphismBoundaryMapResult, ...]
    scale_ids: tuple[str, ...]
    result_wording_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_claim_allowed: bool
    independent_recurrence_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != PhysicalScaleMorphismNumericalMorphismFamilyConfig.SCHEMA:
            raise ValueError("numerical result binds the wrong configuration schema")
        if self.evidence_world is not PhysicalScaleMorphismEvidenceWorld.NUMERICAL_TWIN:
            raise ValueError("numerical experiment cannot claim another evidence world")
        require_sorted_unique_ids(
            self.numerical_summaries, attribute="summary_id", field_name="numerical_summaries"
        )
        require_sorted_unique_ids(
            self.property_results,
            attribute="property_result_id",
            field_name="property_results",
        )
        require_sorted_unique_ids(
            self.boundary_results,
            attribute="boundary_result_id",
            field_name="boundary_results",
        )
        require_sorted_unique_strings(self.scale_ids, field_name="scale_ids", allow_empty=False)
        validate_stable_id(self.result_wording_id, field_name="result_wording_id")
        if self.result_wording_id != "bounded-numerical-morphism-family-evaluated":
            raise ValueError("numerical result wording exceeds its frozen scope")
        if (
            self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        ):
            raise ValueError("numerical result changed its claim/access ceiling")
        if self.physical_claim_allowed or self.independent_recurrence_claim_allowed:
            raise ValueError("numerical results cannot claim physical or independent recurrence")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNumericalMorphismExecution:
    """In-memory scientific payload accompanying the compact result record."""

    result: PhysicalScaleMorphismNumericalMorphismFamilyResult
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class _BoundaryPanel:
    labels: tuple[str, ...]
    boundary_cells: tuple[str, ...]


def default_numerical_morphism_family_config() -> PhysicalScaleMorphismNumericalMorphismFamilyConfig:
    """Return the one frozen first-tranche design, without observing results."""

    return PhysicalScaleMorphismNumericalMorphismFamilyConfig(
        config_id="physical-scale-morphism-numerical-family",
        scale_cells=(16, 32, 64),
        receiver_bin_counts=(4, 8, 16),
        resistance_bar_ohms=Decimal("100"),
        capacitance_bar_farads=Decimal("0.001"),
        voltage_reference_volts=Decimal("1"),
        component_variation_fraction=Decimal("0.12"),
        output_times_t_star=(
            Decimal("0"),
            Decimal("0.05"),
            Decimal("0.10"),
            Decimal("0.15"),
            Decimal("0.20"),
        ),
        probe_time_t_star=Decimal("0.20"),
        refinement_base_step_t_star=Decimal("0.04"),
        numerical_voltage_margin_volts=Decimal("0.03"),
        receiver_exact_margin=Decimal("1e-12"),
        receiver_loss_margin=Decimal("0.001"),
        scale_relative_margin=Decimal("0.05"),
        negative_control_minimum_defect=Decimal("0.001"),
        action_amplitudes_u_star=(
            Decimal("0.20"),
            Decimal("0.40"),
            Decimal("0.60"),
            Decimal("0.80"),
            Decimal("1.00"),
        ),
        action_durations_t_star=(
            Decimal("0.01"),
            Decimal("0.025"),
            Decimal("0.05"),
            Decimal("0.10"),
            Decimal("0.20"),
        ),
        target_charge_minimum_q_star=Decimal("0.08"),
        sink_voltage_maximum_v_star=Decimal("0.80"),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        physical_execution_authorized=False,
    )


def _decimal(value: float) -> Decimal:
    return Decimal(str(float(value)))


def _time_scale(config: PhysicalScaleMorphismNumericalMorphismFamilyConfig, scale_cells: int) -> float:
    return float(config.resistance_bar_ohms) * float(config.capacitance_bar_farads) * scale_cells**2


def _model_config(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    *,
    scale_cells: int,
    amplitude_u_star: Decimal,
    duration_t_star: Decimal,
    output_times_t_star: tuple[Decimal, ...],
    suffix: str,
) -> ResistorCapacitorLadderModelConfig:
    variation = float(design.component_variation_fraction)
    c_bar = float(design.capacitance_bar_farads)
    r_bar = float(design.resistance_bar_ohms)
    capacitances = tuple(
        _decimal(c_bar * (1 + variation * sin(2 * pi * (index + 0.5) / scale_cells)))
        for index in range(scale_cells)
    )
    resistances = tuple(
        _decimal(r_bar * (1 + variation * cos(2 * pi * (index + 1) / scale_cells)))
        for index in range(scale_cells - 1)
    )
    time_scale = _time_scale(design, scale_cells)
    return ResistorCapacitorLadderModelConfig(
        config_id=f"physical-scale-morphism-numerical.n{scale_cells}.{suffix}",
        scale_cells=scale_cells,
        capacitances_farads=capacitances,
        interior_resistances_ohms=resistances,
        left_source_resistance_ohms=design.resistance_bar_ohms,
        right_termination_resistance_ohms=design.resistance_bar_ohms,
        initial_voltages_volts=tuple(Decimal(0) for _ in range(scale_cells)),
        action_segments=(
            ResistorCapacitorLadderActionSegment(
                segment_id=f"drive.{suffix}",
                start_seconds=Decimal(0),
                end_seconds=_decimal(float(duration_t_star) * time_scale),
                left_voltage_volts=amplitude_u_star * design.voltage_reference_volts,
                right_voltage_volts=Decimal(0),
            ),
        ),
        output_times_seconds=tuple(
            _decimal(float(value) * time_scale) for value in output_times_t_star
        ),
        component_metrology_frozen_before_response=True,
    )


def _normalization(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    model: ResistorCapacitorLadderModelConfig,
) -> tuple[PhysicalScaleMorphismNormalizationSpec, PhysicalScaleMorphismNormalizationUncertainty]:
    component_bytes = b"".join(
        value.to_eng_string().encode("ascii") + b"\0"
        for value in (*model.capacitances_farads, *model.interior_resistances_ohms)
    )
    normalization_id = f"normalization.n{model.scale_cells}"
    spec = PhysicalScaleMorphismNormalizationSpec(
        normalization_id=normalization_id,
        scale_cells=model.scale_cells,
        component_basis_sha256=sha256(component_bytes).hexdigest(),
        resistance_summary_rule_id="nominal-interior-resistance",
        capacitance_summary_rule_id="nominal-cell-capacitance",
        resistance_bar_ohms=design.resistance_bar_ohms,
        capacitance_bar_farads=design.capacitance_bar_farads,
        voltage_reference_volts=design.voltage_reference_volts,
        realized_component_weights=True,
        diffusive_time_exponent=2,
    )
    uncertainty = PhysicalScaleMorphismNormalizationUncertainty(
        uncertainty_id=f"uncertainty.n{model.scale_cells}.analytic-zero",
        normalization_id=normalization_id,
        voltage_uncertainty_volts=tuple(Decimal(0) for _ in range(model.scale_cells)),
        capacitance_uncertainty_farads=tuple(Decimal(0) for _ in range(model.scale_cells)),
        resistance_bar_uncertainty_ohms=Decimal(0),
        capacitance_bar_uncertainty_farads=Decimal(0),
        voltage_reference_uncertainty_volts=Decimal(0),
        terminal_current_uncertainty_amperes=Decimal(0),
        native_time_uncertainty_seconds=Decimal(0),
        decay_rate_uncertainty_per_second=Decimal(0),
        dissipated_energy_uncertainty_joules=None,
        penetration_depth_uncertainty_cells=None,
        first_passage_time_uncertainty_seconds=None,
        simultaneous_rectangular_bound=True,
    )
    return spec, uncertainty


def _endpoint_dimensionless(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    model: ResistorCapacitorLadderModelConfig,
    trajectory: ResistorCapacitorLadderTrajectory,
    *,
    amplitude_u_star: Decimal,
) -> PhysicalScaleMorphismDimensionlessRecord:
    spec, uncertainty = _normalization(design, model)
    state = trajectory.voltages_volts[-1]
    applied_voltage = float(amplitude_u_star * design.voltage_reference_volts)
    terminal_current = (applied_voltage - state[0]) / float(model.left_source_resistance_ohms)
    operator = build_operator(model)
    return compute_dimensionless_record(
        record_id=f"dimensionless.n{model.scale_cells}.probe",
        spec=spec,
        uncertainty=uncertainty,
        voltages=state,
        capacitances_farads=np.asarray(model.capacitances_farads, dtype=np.float64),
        terminal_current_amperes=terminal_current,
        native_time_seconds=float(trajectory.times_seconds[-1]),
        slowest_decay_rate_per_second=slowest_decay_rate(operator),
    )


def _values(record: PhysicalScaleMorphismDimensionlessRecord) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {value.value_id: float(value.value) for value in record.dimensionless_values},
        {value.value_id: float(value.value) for value in record.native_values},
    )


def _relative_defect(source: float, target: float) -> float:
    return abs(source - target) / max(abs(source), abs(target), 1e-15)


def _property_result(
    *,
    result_id: str,
    map_family: PhysicalScaleMorphismMapFamily,
    map_id: str,
    property_id: str,
    prediction: PhysicalScaleMorphismSurvivalPrediction,
    source: float,
    target: float,
    margin: float,
    use_relative_margin: bool,
    reason_codes: tuple[str, ...] = (),
) -> PhysicalScaleMorphismMorphismPropertyResult:
    source_value = float(source)
    target_value = float(target)
    absolute = abs(source_value - target_value)
    relative = _relative_defect(source_value, target_value)
    observed = (relative if use_relative_margin else absolute) <= margin
    prediction_matches = observed == (prediction is PhysicalScaleMorphismSurvivalPrediction.SURVIVOR)
    return PhysicalScaleMorphismMorphismPropertyResult(
        property_result_id=result_id,
        map_family=map_family,
        map_id=map_id,
        property_id=property_id,
        prediction=prediction,
        observed_survival=observed,
        prediction_state=(
            PhysicalScaleMorphismEvidenceState.SUPPORTED if prediction_matches else PhysicalScaleMorphismEvidenceState.OPPOSED
        ),
        source_value=_decimal(source_value),
        target_value=_decimal(target_value),
        absolute_defect=_decimal(absolute),
        relative_defect=_decimal(relative),
        margin=_decimal(margin),
        reason_codes=tuple(sorted(reason_codes)),
    )


def _receiver_coordinates(
    model: ResistorCapacitorLadderModelConfig,
    trajectory: ResistorCapacitorLadderTrajectory,
    bin_count: int,
) -> tuple[FloatArray, FloatArray]:
    capacitances = np.asarray(model.capacitances_farads, dtype=np.float64)
    spec = equal_width_receiver_spec(
        receiver_id=f"r{bin_count}.n{model.scale_cells}.mean",
        scale_cells=model.scale_cells,
        bin_count=bin_count,
    )
    matrix = receiver_matrix(spec, capacitances)
    bin_capacitances = np.asarray(
        [
            np.sum(capacitances[np.asarray(receiver_bin.node_indices, dtype=np.int64)])
            for receiver_bin in spec.bins
        ],
        dtype=np.float64,
    )
    return matrix @ trajectory.voltages_volts[-1], bin_capacitances


def _boundary_cells(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    labels: tuple[str, ...],
) -> tuple[str, ...]:
    n_duration = len(design.action_durations_t_star)
    cells: list[str] = []
    for amplitude_index in range(len(design.action_amplitudes_u_star) - 1):
        for duration_index in range(n_duration - 1):
            indices = (
                amplitude_index * n_duration + duration_index,
                amplitude_index * n_duration + duration_index + 1,
                (amplitude_index + 1) * n_duration + duration_index,
                (amplitude_index + 1) * n_duration + duration_index + 1,
            )
            if len({labels[index] for index in indices}) > 1:
                cells.append(f"cell.a{amplitude_index:02d}.d{duration_index:02d}")
    return tuple(cells)


def _boundary_panel(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig,
    *,
    scale_cells: int,
    receiver: str,
) -> _BoundaryPanel:
    labels: list[str] = []
    for amplitude in design.action_amplitudes_u_star:
        for duration in design.action_durations_t_star:
            model = _model_config(
                design,
                scale_cells=scale_cells,
                amplitude_u_star=amplitude,
                duration_t_star=duration,
                output_times_t_star=(Decimal(0), duration),
                suffix=f"boundary.a{amplitude}.d{duration}",
            )
            trajectory = solve_matrix_exponential(model)
            state = trajectory.voltages_volts[-1]
            capacitances = np.asarray(model.capacitances_farads, dtype=np.float64)
            q_star = float(
                np.sum(capacitances * state)
                / (
                    scale_cells
                    * float(design.capacitance_bar_farads)
                    * float(design.voltage_reference_volts)
                )
            )
            if receiver == "fine" or receiver == "guard":
                vmax_star = float(np.max(state) / float(design.voltage_reference_volts))
            elif receiver == "mean":
                coordinates, _ = _receiver_coordinates(model, trajectory, 8)
                vmax_star = float(np.max(coordinates) / float(design.voltage_reference_volts))
            else:
                raise ValueError("unknown boundary receiver")
            target_pass = q_star >= float(design.target_charge_minimum_q_star)
            sink_pass = vmax_star <= float(design.sink_voltage_maximum_v_star)
            labels.append("admit" if target_pass and sink_pass else "hold")
    label_tuple = tuple(labels)
    return _BoundaryPanel(
        labels=label_tuple,
        boundary_cells=_boundary_cells(design, label_tuple),
    )


def _boundary_result(
    *,
    result_id: str,
    map_family: PhysicalScaleMorphismMapFamily,
    map_id: str,
    source: _BoundaryPanel,
    target: _BoundaryPanel,
) -> PhysicalScaleMorphismBoundaryMapResult:
    reasons: tuple[str, ...]
    mismatches = sum(
        left != right for left, right in zip(source.labels, target.labels, strict=True)
    )
    false_safe = sum(
        left == "hold" and right == "admit"
        for left, right in zip(source.labels, target.labels, strict=True)
    )
    false_hold = sum(
        left == "admit" and right == "hold"
        for left, right in zip(source.labels, target.labels, strict=True)
    )
    symmetric = tuple(sorted(set(source.boundary_cells) ^ set(target.boundary_cells)))
    if not source.boundary_cells or not target.boundary_cells:
        state = PhysicalScaleMorphismEvidenceState.UNEVALUABLE
        reasons = ("boundary-absent-on-entered-grid",)
    elif not symmetric and not mismatches and not false_safe:
        state = PhysicalScaleMorphismEvidenceState.SUPPORTED
        reasons = ()
    else:
        state = PhysicalScaleMorphismEvidenceState.OPPOSED
        reasons = tuple(
            sorted(
                {
                    *(() if not symmetric else ("boundary-cell-set-mismatch",)),
                    *(() if not mismatches else ("category-mismatch",)),
                    *(() if not false_safe else ("false-safe-enlargement",)),
                }
            )
        )
    return PhysicalScaleMorphismBoundaryMapResult(
        boundary_result_id=result_id,
        map_family=map_family,
        map_id=map_id,
        source_boundary_cell_ids=source.boundary_cells,
        target_boundary_cell_ids=target.boundary_cells,
        symmetric_difference_cell_ids=symmetric,
        category_mismatch_count=mismatches,
        false_safe_count=false_safe,
        false_hold_count=false_hold,
        state=state,
        reason_codes=reasons,
    )


def _run_numerical_morphism_family(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig | None = None,
) -> PhysicalScaleMorphismNumericalMorphismExecution:
    """Execute the bounded N=16/32/64 numerical morphism family."""

    config = design or default_numerical_morphism_family_config()
    summaries: list[ResistorCapacitorLadderNumericalSummary] = []
    properties: list[PhysicalScaleMorphismMorphismPropertyResult] = []
    boundaries: list[PhysicalScaleMorphismBoundaryMapResult] = []
    arrays: dict[str, FloatArray] = {}
    models: dict[int, ResistorCapacitorLadderModelConfig] = {}
    exact: dict[int, ResistorCapacitorLadderTrajectory] = {}
    dimensionless: dict[int, dict[str, float]] = {}
    native: dict[int, dict[str, float]] = {}
    receiver_profiles: dict[tuple[int, int], FloatArray] = {}

    for scale_cells in config.scale_cells:
        model = _model_config(
            config,
            scale_cells=scale_cells,
            amplitude_u_star=Decimal(1),
            duration_t_star=config.probe_time_t_star,
            output_times_t_star=config.output_times_t_star,
            suffix="probe",
        )
        models[scale_cells] = model
        trajectory = solve_matrix_exponential(model)
        exact[scale_cells] = trajectory
        record = _endpoint_dimensionless(
            config,
            model,
            trajectory,
            amplitude_u_star=Decimal(1),
        )
        dimensionless[scale_cells], native[scale_cells] = _values(record)
        arrays[f"n{scale_cells}_times_seconds"] = trajectory.times_seconds
        arrays[f"n{scale_cells}_exact_voltages_volts"] = trajectory.voltages_volts
        arrays[f"n{scale_cells}_dimensionless_values"] = np.asarray(
            [dimensionless[scale_cells][key] for key in sorted(dimensionless[scale_cells])],
            dtype=np.float64,
        )

        chain = refinement_chain(
            model,
            h0_seconds=float(config.refinement_base_step_t_star) * _time_scale(config, scale_cells),
            convergence_tolerance_volts=float(config.numerical_voltage_margin_volts),
        )
        for view_id, numerical_trajectory, summary in chain:
            summaries.append(summary)
            arrays[f"n{scale_cells}_{view_id}_voltages_volts"] = numerical_trajectory.voltages_volts
            predicted = (
                PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR
                if view_id == "h0"
                else PhysicalScaleMorphismSurvivalPrediction.SURVIVOR
            )
            properties.append(
                _property_result(
                    result_id=f"numerical.n{scale_cells}.{view_id}.trajectory-voltage",
                    map_family=PhysicalScaleMorphismMapFamily.NUMERICAL,
                    map_id=f"j.n{scale_cells}.{view_id}.to-exact",
                    property_id="trajectory-voltage",
                    prediction=predicted,
                    source=0.0,
                    target=float(summary.maximum_voltage_defect_volts),
                    margin=float(config.numerical_voltage_margin_volts),
                    use_relative_margin=False,
                    reason_codes=("matrix-exponential-comparator",),
                )
            )

        capacitances = np.asarray(model.capacitances_farads, dtype=np.float64)
        state = trajectory.voltages_volts[-1]
        fine_charge = float(np.sum(capacitances * state))
        fine_energy = float(0.5 * np.sum(capacitances * state**2))
        fine_vmax = float(np.max(state))
        applied = float(config.voltage_reference_volts)
        fine_current = (applied - state[0]) / float(model.left_source_resistance_ohms)
        for bin_count in config.receiver_bin_counts:
            coordinates, bin_capacitances = _receiver_coordinates(model, trajectory, bin_count)
            receiver_profiles[(scale_cells, bin_count)] = coordinates
            arrays[f"n{scale_cells}_r{bin_count}_probe_volts"] = coordinates
            reconstructed = {
                "q-star": float(np.sum(bin_capacitances * coordinates)),
                "e-star": float(0.5 * np.sum(bin_capacitances * coordinates**2)),
                "v-max-star": float(np.max(coordinates)),
                "i-star": fine_current,
            }
            fine = {
                "q-star": fine_charge,
                "e-star": fine_energy,
                "v-max-star": fine_vmax,
                "i-star": fine_current,
            }
            for property_id in ("q-star", "e-star", "i-star", "v-max-star"):
                lossless_at_this_resolution = property_id in {"q-star", "i-star"} or (
                    bin_count == scale_cells
                )
                margin = (
                    float(config.receiver_exact_margin)
                    if lossless_at_this_resolution
                    else float(config.receiver_loss_margin)
                )
                properties.append(
                    _property_result(
                        result_id=(f"receiver.n{scale_cells}.r{bin_count}.mean.{property_id}"),
                        map_family=PhysicalScaleMorphismMapFamily.RECEIVER,
                        map_id=f"c.n{scale_cells}.full-to-r{bin_count}-mean",
                        property_id=property_id,
                        prediction=(
                            PhysicalScaleMorphismSurvivalPrediction.SURVIVOR
                            if lossless_at_this_resolution
                            else PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR
                        ),
                        source=fine[property_id],
                        target=reconstructed[property_id],
                        margin=margin,
                        use_relative_margin=not lossless_at_this_resolution,
                        reason_codes=("capacitance-weighted-mean",),
                    )
                )
            for property_id, value in fine.items():
                properties.append(
                    _property_result(
                        result_id=(f"receiver.n{scale_cells}.r{bin_count}.guard.{property_id}"),
                        map_family=PhysicalScaleMorphismMapFamily.RECEIVER,
                        map_id=f"c.n{scale_cells}.full-to-r{bin_count}-guard",
                        property_id=property_id,
                        prediction=PhysicalScaleMorphismSurvivalPrediction.SURVIVOR,
                        source=value,
                        target=value,
                        margin=float(config.receiver_exact_margin),
                        use_relative_margin=False,
                        reason_codes=("fine-derived-guard-carried",),
                    )
                )

        direct = equal_width_receiver_spec(
            receiver_id=f"r4.n{scale_cells}.direct",
            scale_cells=scale_cells,
            bin_count=4,
        )
        first = equal_width_receiver_spec(
            receiver_id=f"r8.n{scale_cells}.first",
            scale_cells=scale_cells,
            bin_count=8,
        )
        second = equal_width_receiver_spec(
            receiver_id=f"r4.n{scale_cells}.second",
            scale_cells=8,
            bin_count=4,
            source_receiver_id=first.receiver_id,
        )
        defect = composition_defect(
            direct=direct,
            first=first,
            second=second,
            capacitances=capacitances,
        )
        properties.append(
            _property_result(
                result_id=f"composition.receiver.n{scale_cells}.r8-r4",
                map_family=PhysicalScaleMorphismMapFamily.COMPOSITION,
                map_id=f"composition.receiver.n{scale_cells}.r8-r4",
                property_id="receiver-definition",
                prediction=PhysicalScaleMorphismSurvivalPrediction.SURVIVOR,
                source=0.0,
                target=defect,
                margin=float(config.receiver_exact_margin),
                use_relative_margin=False,
                reason_codes=("direct-versus-composed-map",),
            )
        )

    scale_pairs = ((16, 32), (32, 64), (16, 64))
    scale_survivors = ("e-star", "i-star", "lambda-star", "q-star", "t-star", "v-max-star")
    for source_scale, target_scale in scale_pairs:
        map_id = f"s{target_scale // source_scale}.n{source_scale}-to-n{target_scale}"
        for property_id in scale_survivors:
            properties.append(
                _property_result(
                    result_id=f"scale.n{source_scale}.n{target_scale}.{property_id}",
                    map_family=PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE,
                    map_id=map_id,
                    property_id=property_id,
                    prediction=PhysicalScaleMorphismSurvivalPrediction.SURVIVOR,
                    source=dimensionless[source_scale][property_id],
                    target=dimensionless[target_scale][property_id],
                    margin=float(config.scale_relative_margin),
                    use_relative_margin=True,
                    reason_codes=("smooth-step-numerical-stratum",),
                )
            )
        for property_id in (
            "charge-native",
            "decay-rate-native",
            "energy-native",
            "terminal-current-native",
            "time-native",
        ):
            properties.append(
                _property_result(
                    result_id=f"scale.n{source_scale}.n{target_scale}.{property_id}",
                    map_family=PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE,
                    map_id=map_id,
                    property_id=property_id,
                    prediction=PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR,
                    source=native[source_scale][property_id],
                    target=native[target_scale][property_id],
                    margin=float(config.scale_relative_margin),
                    use_relative_margin=True,
                    reason_codes=("native-nonfixed-control",),
                )
            )

    direct_profile_defect = float(
        np.max(np.abs(receiver_profiles[(16, 8)] - receiver_profiles[(64, 8)]))
    )
    step_one_defect = float(np.max(np.abs(receiver_profiles[(16, 8)] - receiver_profiles[(32, 8)])))
    step_two_defect = float(np.max(np.abs(receiver_profiles[(32, 8)] - receiver_profiles[(64, 8)])))
    propagated_budget = step_one_defect + step_two_defect
    composition_excess = max(0.0, direct_profile_defect - propagated_budget)
    properties.append(
        _property_result(
            result_id="composition.scale.n16-n32-n64.r8-profile",
            map_family=PhysicalScaleMorphismMapFamily.COMPOSITION,
            map_id="composition.scale.n16-n32-n64",
            property_id="r8-profile-metric-budget",
            prediction=PhysicalScaleMorphismSurvivalPrediction.SURVIVOR,
            source=0.0,
            target=composition_excess,
            margin=float(config.receiver_exact_margin),
            use_relative_margin=False,
            reason_codes=("metric-triangle-budget-only",),
        )
    )

    wrong_target = _model_config(
        config,
        scale_cells=32,
        amplitude_u_star=Decimal(1),
        duration_t_star=Decimal("0.10"),
        output_times_t_star=(Decimal(0), Decimal("0.10")),
        suffix="wrong-linear-time-control",
    )
    wrong_trajectory = solve_matrix_exponential(wrong_target)
    wrong_profile, _ = _receiver_coordinates(wrong_target, wrong_trajectory, 8)
    correct_step_defect = step_one_defect
    wrong_step_defect = float(np.max(np.abs(receiver_profiles[(16, 8)] - wrong_profile)))
    properties.append(
        _property_result(
            result_id="control.scale.n16-n32.wrong-linear-time",
            map_family=PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE,
            map_id="control.scale.n16-n32.wrong-linear-time",
            property_id="r8-profile",
            prediction=PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR,
            source=correct_step_defect,
            target=wrong_step_defect,
            margin=float(config.negative_control_minimum_defect),
            use_relative_margin=False,
            reason_codes=("wrong-n-time-scaling",),
        )
    )

    omitted = solve_matrix_exponential(models[16], omit_source_impedance=True)
    omitted_defect = float(np.max(np.abs(exact[16].voltages_volts - omitted.voltages_volts)))
    properties.append(
        _property_result(
            result_id="control.denominator.n16.omitted-source-impedance",
            map_family=PhysicalScaleMorphismMapFamily.NUMERICAL,
            map_id="control.denominator.n16.omitted-source-impedance",
            property_id="trajectory-voltage",
            prediction=PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR,
            source=0.0,
            target=omitted_defect,
            margin=float(config.negative_control_minimum_defect),
            use_relative_margin=False,
            reason_codes=("denominator-corruption",),
        )
    )
    averaged = solve_matrix_exponential(models[16], average_components=True)
    averaged_defect = float(np.max(np.abs(exact[16].voltages_volts - averaged.voltages_volts)))
    properties.append(
        _property_result(
            result_id="control.denominator.n16.averaged-components",
            map_family=PhysicalScaleMorphismMapFamily.NUMERICAL,
            map_id="control.denominator.n16.averaged-components",
            property_id="trajectory-voltage",
            prediction=PhysicalScaleMorphismSurvivalPrediction.NONSURVIVOR,
            source=0.0,
            target=averaged_defect,
            margin=float(config.negative_control_minimum_defect),
            use_relative_margin=False,
            reason_codes=("realized-component-basis-deletion",),
        )
    )

    boundary_by_scale: dict[int, _BoundaryPanel] = {}
    for scale_cells in config.scale_cells:
        fine_boundary = _boundary_panel(config, scale_cells=scale_cells, receiver="fine")
        mean_boundary = _boundary_panel(config, scale_cells=scale_cells, receiver="mean")
        guard_boundary = _boundary_panel(config, scale_cells=scale_cells, receiver="guard")
        boundary_by_scale[scale_cells] = fine_boundary
        boundaries.append(
            _boundary_result(
                result_id=f"boundary.receiver.n{scale_cells}.r8-mean",
                map_family=PhysicalScaleMorphismMapFamily.RECEIVER,
                map_id=f"c.n{scale_cells}.full-to-r8-mean",
                source=fine_boundary,
                target=mean_boundary,
            )
        )
        boundaries.append(
            _boundary_result(
                result_id=f"boundary.receiver.n{scale_cells}.r8-guard",
                map_family=PhysicalScaleMorphismMapFamily.RECEIVER,
                map_id=f"c.n{scale_cells}.full-to-r8-guard",
                source=fine_boundary,
                target=guard_boundary,
            )
        )
        arrays[f"n{scale_cells}_boundary_fine_labels"] = np.asarray(
            [1.0 if value == "admit" else 0.0 for value in fine_boundary.labels],
            dtype=np.float64,
        )
        arrays[f"n{scale_cells}_boundary_mean_labels"] = np.asarray(
            [1.0 if value == "admit" else 0.0 for value in mean_boundary.labels],
            dtype=np.float64,
        )

    for source_scale, target_scale in scale_pairs:
        boundaries.append(
            _boundary_result(
                result_id=f"boundary.scale.n{source_scale}.n{target_scale}",
                map_family=PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE,
                map_id=f"s{target_scale // source_scale}.n{source_scale}-to-n{target_scale}",
                source=boundary_by_scale[source_scale],
                target=boundary_by_scale[target_scale],
            )
        )

    result = PhysicalScaleMorphismNumericalMorphismFamilyResult(
        result_id="physical-scale-morphism-numerical-family-result",
        config=ObjectIdentity.from_record(config.config_id, config),
        evidence_world=PhysicalScaleMorphismEvidenceWorld.NUMERICAL_TWIN,
        numerical_summaries=tuple(sorted(summaries, key=lambda value: value.summary_id)),
        property_results=tuple(sorted(properties, key=lambda value: value.property_result_id)),
        boundary_results=tuple(sorted(boundaries, key=lambda value: value.boundary_result_id)),
        scale_ids=("n16", "n32", "n64"),
        result_wording_id="bounded-numerical-morphism-family-evaluated",
        evidence_ceiling=config.evidence_ceiling,
        outcome_access=config.outcome_access,
        visibility_ceiling=config.visibility_ceiling,
        physical_claim_allowed=False,
        independent_recurrence_claim_allowed=False,
    )
    return PhysicalScaleMorphismNumericalMorphismExecution(result=result, arrays=arrays)


def run_numerical_morphism_family(
    design: PhysicalScaleMorphismNumericalMorphismFamilyConfig | None = None,
) -> PhysicalScaleMorphismNumericalMorphismExecution:
    """Execute the panel under an explicit deterministic linear-algebra budget."""

    with threadpool_limits(limits=PHYSICAL_SCALE_MORPHISM_REFERENCE_BLAS_THREAD_COUNT):
        return _run_numerical_morphism_family(design)


__all__ = [
    'PhysicalScaleMorphismBoundaryMapResult',
    'PhysicalScaleMorphismMorphismPropertyResult',
    'PhysicalScaleMorphismNumericalMorphismExecution',
    'PhysicalScaleMorphismNumericalMorphismFamilyConfig',
    'PhysicalScaleMorphismNumericalMorphismFamilyResult',
    'PhysicalScaleMorphismSurvivalPrediction',
    "default_numerical_morphism_family_config",
    "run_numerical_morphism_family",
]
