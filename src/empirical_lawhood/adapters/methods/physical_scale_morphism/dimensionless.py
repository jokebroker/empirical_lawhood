"""Frozen RC-ladder nondimensionalization with native-value retention."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNormalizationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-spec'

    normalization_id: str
    scale_cells: int
    component_basis_sha256: str
    resistance_summary_rule_id: str
    capacitance_summary_rule_id: str
    resistance_bar_ohms: Decimal
    capacitance_bar_farads: Decimal
    voltage_reference_volts: Decimal
    realized_component_weights: bool
    diffusive_time_exponent: int

    def __post_init__(self) -> None:
        validate_stable_id(self.normalization_id, field_name="normalization_id")
        validate_sha256(self.component_basis_sha256, field_name="component_basis_sha256")
        validate_stable_id(self.resistance_summary_rule_id, field_name="resistance_summary_rule_id")
        validate_stable_id(
            self.capacitance_summary_rule_id, field_name="capacitance_summary_rule_id"
        )
        if self.scale_cells <= 0:
            raise ValueError("normalization scale must be positive")
        for name in (
            "resistance_bar_ohms",
            "capacitance_bar_farads",
            "voltage_reference_volts",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if not self.realized_component_weights:
            raise ValueError("primary physical scale morphism normalization requires realized components")
        if self.diffusive_time_exponent != 2:
            raise ValueError("primary physical scale morphism time normalization must use N squared")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNormalizationUncertainty(CanonicalRecord):
    """Rectangular simultaneous input bounds for one normalization act."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-normalization-uncertainty'

    uncertainty_id: str
    normalization_id: str
    voltage_uncertainty_volts: tuple[Decimal, ...]
    capacitance_uncertainty_farads: tuple[Decimal, ...]
    resistance_bar_uncertainty_ohms: Decimal
    capacitance_bar_uncertainty_farads: Decimal
    voltage_reference_uncertainty_volts: Decimal
    terminal_current_uncertainty_amperes: Decimal
    native_time_uncertainty_seconds: Decimal
    decay_rate_uncertainty_per_second: Decimal
    dissipated_energy_uncertainty_joules: Decimal | None
    penetration_depth_uncertainty_cells: Decimal | None
    first_passage_time_uncertainty_seconds: Decimal | None
    simultaneous_rectangular_bound: bool

    def __post_init__(self) -> None:
        for name in ("uncertainty_id", "normalization_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("voltage_uncertainty_volts", "capacitance_uncertainty_farads"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{name}[{index}]", minimum=Decimal(0))
        for name in (
            "resistance_bar_uncertainty_ohms",
            "capacitance_bar_uncertainty_farads",
            "voltage_reference_uncertainty_volts",
            "terminal_current_uncertainty_amperes",
            "native_time_uncertainty_seconds",
            "decay_rate_uncertainty_per_second",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in (
            "dissipated_energy_uncertainty_joules",
            "penetration_depth_uncertainty_cells",
            "first_passage_time_uncertainty_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not self.simultaneous_rectangular_bound:
            raise ValueError("physical scale morphism normalization uncertainty must be simultaneous")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismDimensionlessRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-dimensionless-record'

    record_id: str
    normalization_id: str
    normalization: ObjectIdentity
    normalization_uncertainty: ObjectIdentity
    native_values: tuple[NamedDecimal, ...]
    native_uncertainty_values: tuple[NamedDecimal, ...]
    dimensionless_values: tuple[NamedDecimal, ...]
    dimensionless_uncertainty_values: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        validate_stable_id(self.normalization_id, field_name="normalization_id")
        if self.normalization.object_schema != PhysicalScaleMorphismNormalizationSpec.SCHEMA:
            raise ValueError("dimensionless record binds the wrong normalization schema")
        if self.normalization.object_id != self.normalization_id:
            raise ValueError("dimensionless normalization ID and identity differ")
        if self.normalization_uncertainty.object_schema != PhysicalScaleMorphismNormalizationUncertainty.SCHEMA:
            raise ValueError("dimensionless record binds the wrong uncertainty schema")
        require_sorted_unique_ids(
            self.native_values, attribute="value_id", field_name="native_values"
        )
        require_sorted_unique_ids(
            self.native_uncertainty_values,
            attribute="value_id",
            field_name="native_uncertainty_values",
        )
        require_sorted_unique_ids(
            self.dimensionless_values,
            attribute="value_id",
            field_name="dimensionless_values",
        )
        require_sorted_unique_ids(
            self.dimensionless_uncertainty_values,
            attribute="value_id",
            field_name="dimensionless_uncertainty_values",
        )
        for values, uncertainties, name in (
            (self.native_values, self.native_uncertainty_values, "native"),
            (
                self.dimensionless_values,
                self.dimensionless_uncertainty_values,
                "dimensionless",
            ),
        ):
            if tuple(value.value_id for value in values) != tuple(
                value.value_id for value in uncertainties
            ):
                raise ValueError(f"{name} value/uncertainty rosters differ")
            if any(
                value.unit != uncertainty.unit
                for value, uncertainty in zip(values, uncertainties, strict=True)
            ):
                raise ValueError(f"{name} value/uncertainty units differ")
            if any(value.value < 0 for value in uncertainties):
                raise ValueError(f"{name} uncertainty must be nonnegative")
        required = {
            "e-star",
            "i-star",
            "lambda-star",
            "q-star",
            "t-star",
            "v-max-star",
        }
        if not required.issubset({value.value_id for value in self.dimensionless_values}):
            raise ValueError("dimensionless record lacks its primary fixed-section values")
        if any(value.unit != "1" for value in self.dimensionless_values):
            raise ValueError("dimensionless values must carry unit one")
        if any(value.unit != "1" for value in self.dimensionless_uncertainty_values):
            raise ValueError("dimensionless uncertainties must carry unit one")


def compute_dimensionless_record(
    *,
    record_id: str,
    spec: PhysicalScaleMorphismNormalizationSpec,
    uncertainty: PhysicalScaleMorphismNormalizationUncertainty,
    voltages: FloatArray,
    capacitances_farads: FloatArray,
    terminal_current_amperes: float,
    native_time_seconds: float,
    slowest_decay_rate_per_second: float,
    dissipated_energy_joules: float | None = None,
    penetration_depth_cells: float | None = None,
    first_passage_time_seconds: float | None = None,
) -> PhysicalScaleMorphismDimensionlessRecord:
    voltages_array = np.asarray(voltages, dtype=np.float64)
    capacitances = np.asarray(capacitances_farads, dtype=np.float64)
    if voltages_array.shape != (spec.scale_cells,) or capacitances.shape != (spec.scale_cells,):
        raise ValueError("dimensionless transform vector shape differs from scale")
    if uncertainty.normalization_id != spec.normalization_id:
        raise ValueError("normalization value and uncertainty records differ")
    voltage_uncertainty = np.asarray(uncertainty.voltage_uncertainty_volts, dtype=np.float64)
    capacitance_uncertainty = np.asarray(
        uncertainty.capacitance_uncertainty_farads, dtype=np.float64
    )
    if (
        voltage_uncertainty.shape != voltages_array.shape
        or capacitance_uncertainty.shape != capacitances.shape
        or np.any(capacitance_uncertainty >= capacitances)
    ):
        raise ValueError("normalization uncertainty vectors differ or cross zero capacitance")
    scalars = np.asarray(
        [terminal_current_amperes, native_time_seconds, slowest_decay_rate_per_second],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(voltages_array)) or not np.all(np.isfinite(capacitances)):
        raise ValueError("dimensionless transform received nonfinite vectors")
    if np.any(capacitances <= 0) or not np.all(np.isfinite(scalars)):
        raise ValueError("dimensionless transform received invalid components/scalars")
    if native_time_seconds < 0 or slowest_decay_rate_per_second < 0:
        raise ValueError("time and decay rate must be nonnegative")
    n_cells = float(spec.scale_cells)
    r_bar = float(spec.resistance_bar_ohms)
    c_bar = float(spec.capacitance_bar_farads)
    v_ref = float(spec.voltage_reference_volts)
    r_uncertainty = float(uncertainty.resistance_bar_uncertainty_ohms)
    c_uncertainty = float(uncertainty.capacitance_bar_uncertainty_farads)
    v_ref_uncertainty = float(uncertainty.voltage_reference_uncertainty_volts)
    if r_uncertainty >= r_bar or c_uncertainty >= c_bar or v_ref_uncertainty >= v_ref:
        raise ValueError("normalization reference uncertainty crosses zero")
    time_scale = r_bar * c_bar * n_cells**2
    charge = float(np.sum(capacitances * voltages_array))
    energy_functional = float(np.sum(capacitances * voltages_array**2))
    stored_energy = 0.5 * energy_functional
    dimensionless: dict[str, float] = {
        "e-star": energy_functional / (n_cells * c_bar * v_ref**2),
        "i-star": n_cells * r_bar * terminal_current_amperes / v_ref,
        "lambda-star": slowest_decay_rate_per_second * time_scale,
        "q-star": charge / (n_cells * c_bar * v_ref),
        "t-star": native_time_seconds / time_scale,
        "v-max-star": float(np.max(voltages_array)) / v_ref,
    }
    native = {
        "charge-native": (charge, "C"),
        "energy-native": (stored_energy, "J"),
        "terminal-current-native": (terminal_current_amperes, "A"),
        "time-native": (native_time_seconds, "s"),
        "decay-rate-native": (slowest_decay_rate_per_second, "1/s"),
    }

    def multiply(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
        values = (
            left[0] * right[0],
            left[0] * right[1],
            left[1] * right[0],
            left[1] * right[1],
        )
        return min(values), max(values)

    def divide(
        numerator: tuple[float, float], denominator: tuple[float, float]
    ) -> tuple[float, float]:
        if denominator[0] <= 0:
            raise ValueError("normalization uncertainty permits a nonpositive denominator")
        return multiply(numerator, (1 / denominator[1], 1 / denominator[0]))

    def half_width(value: float, bounds: tuple[float, float]) -> float:
        return max(abs(value - bounds[0]), abs(bounds[1] - value))

    voltage_bounds = tuple(
        (value - delta, value + delta)
        for value, delta in zip(voltages_array, voltage_uncertainty, strict=True)
    )
    capacitance_bounds = tuple(
        (value - delta, value + delta)
        for value, delta in zip(capacitances, capacitance_uncertainty, strict=True)
    )
    charge_terms = tuple(
        multiply(capacitance_bound, voltage_bound)
        for capacitance_bound, voltage_bound in zip(capacitance_bounds, voltage_bounds, strict=True)
    )
    charge_bounds = (
        sum(value[0] for value in charge_terms),
        sum(value[1] for value in charge_terms),
    )
    energy_terms = []
    for capacitance_bound, voltage_bound in zip(capacitance_bounds, voltage_bounds, strict=True):
        square_bound = (
            0.0
            if voltage_bound[0] <= 0 <= voltage_bound[1]
            else min(voltage_bound[0] ** 2, voltage_bound[1] ** 2),
            max(voltage_bound[0] ** 2, voltage_bound[1] ** 2),
        )
        energy_terms.append(multiply(capacitance_bound, square_bound))
    energy_functional_bounds = (
        sum(value[0] for value in energy_terms),
        sum(value[1] for value in energy_terms),
    )
    r_bounds = (r_bar - r_uncertainty, r_bar + r_uncertainty)
    c_bounds = (c_bar - c_uncertainty, c_bar + c_uncertainty)
    v_ref_bounds = (v_ref - v_ref_uncertainty, v_ref + v_ref_uncertainty)
    time_scale_bounds = multiply(r_bounds, c_bounds)
    time_scale_bounds = (
        time_scale_bounds[0] * n_cells**2,
        time_scale_bounds[1] * n_cells**2,
    )
    charge_denominator = multiply(c_bounds, v_ref_bounds)
    charge_denominator = (
        charge_denominator[0] * n_cells,
        charge_denominator[1] * n_cells,
    )
    voltage_square_reference = (v_ref_bounds[0] ** 2, v_ref_bounds[1] ** 2)
    energy_denominator = multiply(c_bounds, voltage_square_reference)
    energy_denominator = (
        energy_denominator[0] * n_cells,
        energy_denominator[1] * n_cells,
    )
    current_bounds = (
        terminal_current_amperes - float(uncertainty.terminal_current_uncertainty_amperes),
        terminal_current_amperes + float(uncertainty.terminal_current_uncertainty_amperes),
    )
    time_bounds = (
        native_time_seconds - float(uncertainty.native_time_uncertainty_seconds),
        native_time_seconds + float(uncertainty.native_time_uncertainty_seconds),
    )
    decay_bounds = (
        slowest_decay_rate_per_second - float(uncertainty.decay_rate_uncertainty_per_second),
        slowest_decay_rate_per_second + float(uncertainty.decay_rate_uncertainty_per_second),
    )
    if time_bounds[0] < 0 or decay_bounds[0] < 0:
        raise ValueError("normalization uncertainty crosses a nonnegative physical operand")
    resistance_current_bounds = multiply(r_bounds, current_bounds)
    scaled_resistance_current_bounds = (
        resistance_current_bounds[0] * n_cells,
        resistance_current_bounds[1] * n_cells,
    )
    dimensionless_bounds: dict[str, tuple[float, float]] = {
        "e-star": divide(energy_functional_bounds, energy_denominator),
        "i-star": divide(scaled_resistance_current_bounds, v_ref_bounds),
        "lambda-star": multiply(decay_bounds, time_scale_bounds),
        "q-star": divide(charge_bounds, charge_denominator),
        "t-star": divide(time_bounds, time_scale_bounds),
        "v-max-star": divide(
            (
                max(value[0] for value in voltage_bounds),
                max(value[1] for value in voltage_bounds),
            ),
            v_ref_bounds,
        ),
    }
    native_bounds: dict[str, tuple[float, float]] = {
        "charge-native": charge_bounds,
        "energy-native": (
            0.5 * energy_functional_bounds[0],
            0.5 * energy_functional_bounds[1],
        ),
        "terminal-current-native": current_bounds,
        "time-native": time_bounds,
        "decay-rate-native": decay_bounds,
    }
    if dissipated_energy_joules is not None:
        if not np.isfinite(dissipated_energy_joules) or dissipated_energy_joules < 0:
            raise ValueError("dissipated energy must be finite and nonnegative")
        native["dissipated-energy-native"] = (dissipated_energy_joules, "J")
        dimensionless["w-diss-star"] = dissipated_energy_joules / (n_cells * c_bar * v_ref**2)
        if uncertainty.dissipated_energy_uncertainty_joules is None:
            raise ValueError("dissipated-energy value lacks its uncertainty operand")
        dissipated_bounds = (
            dissipated_energy_joules - float(uncertainty.dissipated_energy_uncertainty_joules),
            dissipated_energy_joules + float(uncertainty.dissipated_energy_uncertainty_joules),
        )
        if dissipated_bounds[0] < 0:
            raise ValueError("dissipated-energy uncertainty crosses zero")
        native_bounds["dissipated-energy-native"] = dissipated_bounds
        dimensionless_bounds["w-diss-star"] = divide(dissipated_bounds, energy_denominator)
    elif uncertainty.dissipated_energy_uncertainty_joules is not None:
        raise ValueError("dissipated-energy uncertainty lacks its value operand")
    if penetration_depth_cells is not None:
        if not np.isfinite(penetration_depth_cells) or not 0 <= penetration_depth_cells <= n_cells:
            raise ValueError("penetration depth lies outside the ladder")
        native["penetration-depth-native"] = (penetration_depth_cells, "cell")
        dimensionless["delta-star"] = penetration_depth_cells / n_cells
        if uncertainty.penetration_depth_uncertainty_cells is None:
            raise ValueError("penetration-depth value lacks its uncertainty operand")
        penetration_bounds = (
            penetration_depth_cells - float(uncertainty.penetration_depth_uncertainty_cells),
            penetration_depth_cells + float(uncertainty.penetration_depth_uncertainty_cells),
        )
        if penetration_bounds[0] < 0 or penetration_bounds[1] > n_cells:
            raise ValueError("penetration-depth uncertainty lies outside the ladder")
        native_bounds["penetration-depth-native"] = penetration_bounds
        dimensionless_bounds["delta-star"] = (
            penetration_bounds[0] / n_cells,
            penetration_bounds[1] / n_cells,
        )
    elif uncertainty.penetration_depth_uncertainty_cells is not None:
        raise ValueError("penetration-depth uncertainty lacks its value operand")
    if first_passage_time_seconds is not None:
        if not np.isfinite(first_passage_time_seconds) or first_passage_time_seconds < 0:
            raise ValueError("first passage time must be finite and nonnegative")
        native["first-passage-time-native"] = (first_passage_time_seconds, "s")
        dimensionless["t-hit-star"] = first_passage_time_seconds / time_scale
        if uncertainty.first_passage_time_uncertainty_seconds is None:
            raise ValueError("first-passage value lacks its uncertainty operand")
        hit_bounds = (
            first_passage_time_seconds - float(uncertainty.first_passage_time_uncertainty_seconds),
            first_passage_time_seconds + float(uncertainty.first_passage_time_uncertainty_seconds),
        )
        if hit_bounds[0] < 0:
            raise ValueError("first-passage uncertainty crosses zero")
        native_bounds["first-passage-time-native"] = hit_bounds
        dimensionless_bounds["t-hit-star"] = divide(hit_bounds, time_scale_bounds)
    elif uncertainty.first_passage_time_uncertainty_seconds is not None:
        raise ValueError("first-passage uncertainty lacks its value operand")
    native_uncertainties = {
        value_id: (half_width(value, native_bounds[value_id]), unit)
        for value_id, (value, unit) in native.items()
    }
    dimensionless_uncertainties = {
        value_id: half_width(value, dimensionless_bounds[value_id])
        for value_id, value in dimensionless.items()
    }
    return PhysicalScaleMorphismDimensionlessRecord(
        record_id=record_id,
        normalization_id=spec.normalization_id,
        normalization=ObjectIdentity.from_record(spec.normalization_id, spec),
        normalization_uncertainty=ObjectIdentity.from_record(
            uncertainty.uncertainty_id, uncertainty
        ),
        native_values=tuple(
            NamedDecimal(value_id, Decimal(str(value)), unit)
            for value_id, (value, unit) in sorted(native.items())
        ),
        native_uncertainty_values=tuple(
            NamedDecimal(value_id, Decimal(str(value)), unit)
            for value_id, (value, unit) in sorted(native_uncertainties.items())
        ),
        dimensionless_values=tuple(
            NamedDecimal(value_id, Decimal(str(value)), "1")
            for value_id, value in sorted(dimensionless.items())
        ),
        dimensionless_uncertainty_values=tuple(
            NamedDecimal(value_id, Decimal(str(value)), "1")
            for value_id, value in sorted(dimensionless_uncertainties.items())
        ),
    )


__all__ = [
    'PhysicalScaleMorphismDimensionlessRecord',
    'PhysicalScaleMorphismNormalizationSpec',
    'PhysicalScaleMorphismNormalizationUncertainty',
    "compute_dimensionless_record",
]
