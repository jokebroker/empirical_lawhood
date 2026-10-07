"""Capacitance-weighted receiver lattice and exact composition checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]


class PhysicalScaleMorphismReceiverKind(StrEnum):
    FULL = "FULL"
    GUARDED_MEAN = "GUARDED_MEAN"
    SPARSE = "SPARSE"
    WEIGHTED_MEAN = "WEIGHTED_MEAN"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismReceiverBin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-receiver-bin'

    bin_id: str
    node_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bin_id, field_name="bin_id")
        if tuple(sorted(set(self.node_indices))) != self.node_indices:
            raise ValueError("receiver bin node indices must be sorted and unique")
        if not self.node_indices or self.node_indices[0] < 0:
            raise ValueError("receiver bin must contain nonnegative node indices")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismReceiverSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-receiver-spec'

    receiver_id: str
    scale_cells: int
    kind: PhysicalScaleMorphismReceiverKind
    bins: tuple[PhysicalScaleMorphismReceiverBin, ...]
    sparse_node_indices: tuple[int, ...]
    carried_scalar_ids: tuple[str, ...]
    includes_terminal_currents: bool
    source_receiver_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        validate_stable_id(self.source_receiver_id, field_name="source_receiver_id")
        if self.scale_cells <= 0:
            raise ValueError("receiver scale must be positive")
        require_sorted_unique_ids(self.bins, attribute="bin_id", field_name="bins")
        if tuple(sorted(set(self.sparse_node_indices))) != self.sparse_node_indices:
            raise ValueError("sparse receiver indices must be sorted and unique")
        if any(value < 0 or value >= self.scale_cells for value in self.sparse_node_indices):
            raise ValueError("sparse receiver index lies outside the board")
        require_sorted_unique_strings(self.carried_scalar_ids, field_name="carried_scalar_ids")
        occupied = tuple(sorted(index for item in self.bins for index in item.node_indices))
        if self.kind in {PhysicalScaleMorphismReceiverKind.WEIGHTED_MEAN, PhysicalScaleMorphismReceiverKind.GUARDED_MEAN}:
            if occupied != tuple(range(self.scale_cells)):
                raise ValueError("weighted receiver bins must partition every board node exactly")
            if self.sparse_node_indices:
                raise ValueError("weighted receiver cannot also be sparse")
        elif self.kind is PhysicalScaleMorphismReceiverKind.SPARSE:
            if self.bins or not self.sparse_node_indices:
                raise ValueError("sparse receiver uses only its exact sparse node roster")
        elif self.kind is PhysicalScaleMorphismReceiverKind.FULL:
            if self.bins or self.sparse_node_indices:
                raise ValueError("full receiver needs no reduction membership")
            if self.source_receiver_id != self.receiver_id:
                raise ValueError("full receiver must be its own source")
        if self.kind is PhysicalScaleMorphismReceiverKind.GUARDED_MEAN:
            required = (
                "charge",
                "energy",
                "maximum-voltage",
                "minimum-voltage",
                "sink-current",
            )
            if self.carried_scalar_ids != required:
                raise ValueError("guarded receiver must carry the exact frozen guard roster")
            if not self.includes_terminal_currents:
                raise ValueError("guarded receiver needs terminal currents for its sink guard")
        elif self.carried_scalar_ids:
            raise ValueError("unguarded receiver cannot claim scalars it does not carry")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismReceiverObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-receiver-observation'

    observation_id: str
    receiver_id: str
    coordinate_values_volts: tuple[Decimal, ...]
    coordinate_uncertainty_volts: tuple[Decimal, ...]
    terminal_currents_amperes: tuple[Decimal, ...]
    terminal_current_uncertainty_amperes: tuple[Decimal, ...]
    carried_scalars: tuple[NamedDecimal, ...]
    carried_scalar_uncertainties: tuple[NamedDecimal, ...]
    uncertainty_method_id: str
    derived_from_fine_receiver: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        validate_stable_id(self.uncertainty_method_id, field_name="uncertainty_method_id")
        if not self.coordinate_values_volts:
            raise ValueError("receiver observation requires coordinates")
        for values_name, uncertainty_name in (
            ("coordinate_values_volts", "coordinate_uncertainty_volts"),
            ("terminal_currents_amperes", "terminal_current_uncertainty_amperes"),
        ):
            values = getattr(self, values_name)
            uncertainties = getattr(self, uncertainty_name)
            if len(values) != len(uncertainties):
                raise ValueError(f"{values_name} value/uncertainty rosters differ")
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{values_name}[{index}]")
                validate_decimal(
                    uncertainties[index],
                    field_name=f"{uncertainty_name}[{index}]",
                    minimum=Decimal(0),
                )
        require_sorted_unique_ids(
            self.carried_scalars, attribute="value_id", field_name="carried_scalars"
        )
        require_sorted_unique_ids(
            self.carried_scalar_uncertainties,
            attribute="value_id",
            field_name="carried_scalar_uncertainties",
        )
        if tuple(value.value_id for value in self.carried_scalars) != tuple(
            value.value_id for value in self.carried_scalar_uncertainties
        ):
            raise ValueError("carried scalar value/uncertainty rosters differ")
        if any(
            value.unit != uncertainty.unit
            for value, uncertainty in zip(
                self.carried_scalars,
                self.carried_scalar_uncertainties,
                strict=True,
            )
        ):
            raise ValueError("carried scalar value/uncertainty units differ")
        if any(value.value < 0 for value in self.carried_scalar_uncertainties):
            raise ValueError("carried scalar uncertainty must be nonnegative")
        if not self.derived_from_fine_receiver:
            raise ValueError("physical scale morphism receiver reductions must retain fine-source provenance")


def equal_width_receiver_spec(
    *,
    receiver_id: str,
    scale_cells: int,
    bin_count: int,
    guarded: bool = False,
    source_receiver_id: str | None = None,
) -> PhysicalScaleMorphismReceiverSpec:
    if scale_cells <= 0 or bin_count <= 0 or scale_cells % bin_count:
        raise ValueError("equal-width receiver requires an exact positive partition")
    width = scale_cells // bin_count
    bins = tuple(
        PhysicalScaleMorphismReceiverBin(
            bin_id=f"{receiver_id}.bin-{index:02d}",
            node_indices=tuple(range(index * width, (index + 1) * width)),
        )
        for index in range(bin_count)
    )
    return PhysicalScaleMorphismReceiverSpec(
        receiver_id=receiver_id,
        scale_cells=scale_cells,
        kind=PhysicalScaleMorphismReceiverKind.GUARDED_MEAN if guarded else PhysicalScaleMorphismReceiverKind.WEIGHTED_MEAN,
        bins=bins,
        sparse_node_indices=(),
        carried_scalar_ids=(
            (
                "charge",
                "energy",
                "maximum-voltage",
                "minimum-voltage",
                "sink-current",
            )
            if guarded
            else ()
        ),
        includes_terminal_currents=True,
        source_receiver_id=source_receiver_id or f"r-full-{scale_cells}",
    )


def receiver_matrix(spec: PhysicalScaleMorphismReceiverSpec, capacitances: FloatArray) -> FloatArray:
    """Return the exact linear voltage-reduction matrix for one receiver."""

    values = np.asarray(capacitances, dtype=np.float64)
    if values.shape != (spec.scale_cells,) or not np.all(np.isfinite(values)):
        raise ValueError("capacitance vector differs from receiver scale")
    if np.any(values <= 0):
        raise ValueError("capacitances must be positive")
    if spec.kind is PhysicalScaleMorphismReceiverKind.FULL:
        return np.eye(spec.scale_cells, dtype=np.float64)
    if spec.kind is PhysicalScaleMorphismReceiverKind.SPARSE:
        matrix = np.zeros((len(spec.sparse_node_indices), spec.scale_cells), dtype=np.float64)
        matrix[np.arange(len(spec.sparse_node_indices)), spec.sparse_node_indices] = 1.0
        return matrix
    matrix = np.zeros((len(spec.bins), spec.scale_cells), dtype=np.float64)
    for row, receiver_bin in enumerate(spec.bins):
        indices = np.asarray(receiver_bin.node_indices, dtype=np.int64)
        weights = values[indices]
        matrix[row, indices] = weights / np.sum(weights)
    return matrix


def reduce_receiver(
    *,
    observation_id: str,
    spec: PhysicalScaleMorphismReceiverSpec,
    voltages: FloatArray,
    voltage_uncertainties: FloatArray,
    capacitances: FloatArray,
    capacitance_uncertainties: FloatArray,
    terminal_currents: FloatArray,
    terminal_current_uncertainties: FloatArray,
    uncertainty_method_id: str,
) -> PhysicalScaleMorphismReceiverObservation:
    values = np.asarray(voltages, dtype=np.float64)
    voltage_uncertainty = np.asarray(voltage_uncertainties, dtype=np.float64)
    currents = np.asarray(terminal_currents, dtype=np.float64)
    current_uncertainty = np.asarray(terminal_current_uncertainties, dtype=np.float64)
    caps = np.asarray(capacitances, dtype=np.float64)
    cap_uncertainty = np.asarray(capacitance_uncertainties, dtype=np.float64)
    if values.shape != (spec.scale_cells,) or not np.all(np.isfinite(values)):
        raise ValueError("fine voltage vector differs from receiver scale")
    if (
        voltage_uncertainty.shape != values.shape
        or cap_uncertainty.shape != caps.shape
        or not np.all(np.isfinite(voltage_uncertainty))
        or not np.all(np.isfinite(cap_uncertainty))
        or np.any(voltage_uncertainty < 0)
        or np.any(cap_uncertainty < 0)
        or np.any(caps - cap_uncertainty <= 0)
    ):
        raise ValueError("receiver uncertainty vectors differ or permit nonpositive capacitance")
    if (
        currents.ndim != 1
        or not np.all(np.isfinite(currents))
        or current_uncertainty.shape != currents.shape
        or not np.all(np.isfinite(current_uncertainty))
        or np.any(current_uncertainty < 0)
        or (spec.includes_terminal_currents and currents.shape != (2,))
        or (not spec.includes_terminal_currents and currents.size)
    ):
        raise ValueError("terminal-current vector differs")
    reduced = receiver_matrix(spec, caps) @ values
    voltage_lower = values - voltage_uncertainty
    voltage_upper = values + voltage_uncertainty
    cap_lower = caps - cap_uncertainty
    cap_upper = caps + cap_uncertainty

    def product_bounds(
        left_lower: np.ndarray,
        left_upper: np.ndarray,
        right_lower: np.ndarray,
        right_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        products = np.stack(
            (
                left_lower * right_lower,
                left_lower * right_upper,
                left_upper * right_lower,
                left_upper * right_upper,
            )
        )
        return np.min(products, axis=0), np.max(products, axis=0)

    def positive_denominator_ratio_bounds(
        numerator_lower: float,
        numerator_upper: float,
        denominator_lower: float,
        denominator_upper: float,
    ) -> tuple[float, float]:
        if denominator_lower <= 0:
            raise ValueError("receiver uncertainty permits a nonpositive denominator")
        ratios = (
            numerator_lower / denominator_lower,
            numerator_lower / denominator_upper,
            numerator_upper / denominator_lower,
            numerator_upper / denominator_upper,
        )
        return min(ratios), max(ratios)

    product_lower, product_upper = product_bounds(
        cap_lower, cap_upper, voltage_lower, voltage_upper
    )
    if spec.kind is PhysicalScaleMorphismReceiverKind.FULL:
        reduced_lower = voltage_lower
        reduced_upper = voltage_upper
    elif spec.kind is PhysicalScaleMorphismReceiverKind.SPARSE:
        reduced_lower = voltage_lower[np.asarray(spec.sparse_node_indices)]
        reduced_upper = voltage_upper[np.asarray(spec.sparse_node_indices)]
    else:
        lower_values = []
        upper_values = []
        for receiver_bin in spec.bins:
            indices = np.asarray(receiver_bin.node_indices, dtype=np.int64)
            lower, upper = positive_denominator_ratio_bounds(
                float(np.sum(product_lower[indices])),
                float(np.sum(product_upper[indices])),
                float(np.sum(cap_lower[indices])),
                float(np.sum(cap_upper[indices])),
            )
            lower_values.append(lower)
            upper_values.append(upper)
        reduced_lower = np.asarray(lower_values)
        reduced_upper = np.asarray(upper_values)
    reduced_uncertainty = np.maximum(
        np.abs(reduced - reduced_lower), np.abs(reduced_upper - reduced)
    )
    scalars: tuple[NamedDecimal, ...] = ()
    scalar_uncertainties: tuple[NamedDecimal, ...] = ()
    if spec.kind is PhysicalScaleMorphismReceiverKind.GUARDED_MEAN:
        charge = float(np.sum(caps * values))
        charge_lower = float(np.sum(product_lower))
        charge_upper = float(np.sum(product_upper))
        voltage_square_lower = np.where(
            (voltage_lower <= 0) & (voltage_upper >= 0),
            0,
            np.minimum(voltage_lower**2, voltage_upper**2),
        )
        voltage_square_upper = np.maximum(voltage_lower**2, voltage_upper**2)
        energy_term_lower, energy_term_upper = product_bounds(
            cap_lower,
            cap_upper,
            voltage_square_lower,
            voltage_square_upper,
        )
        energy = float(0.5 * np.sum(caps * values**2))
        energy_lower = float(0.5 * np.sum(energy_term_lower))
        energy_upper = float(0.5 * np.sum(energy_term_upper))
        maximum_voltage = float(np.max(values))
        maximum_lower = float(np.max(voltage_lower))
        maximum_upper = float(np.max(voltage_upper))
        minimum_voltage = float(np.min(values))
        minimum_lower = float(np.min(voltage_lower))
        minimum_upper = float(np.min(voltage_upper))
        current_lower = currents - current_uncertainty
        current_upper = currents + current_uncertainty
        abs_current_lower = np.where(
            (current_lower <= 0) & (current_upper >= 0),
            0,
            np.minimum(np.abs(current_lower), np.abs(current_upper)),
        )
        abs_current_upper = np.maximum(np.abs(current_lower), np.abs(current_upper))
        sink_current = float(np.max(np.abs(currents)))
        sink_lower = float(np.max(abs_current_lower))
        sink_upper = float(np.max(abs_current_upper))
        scalar_values = {
            "charge": (charge, charge_lower, charge_upper, "C"),
            "energy": (energy, energy_lower, energy_upper, "J"),
            "maximum-voltage": (
                maximum_voltage,
                maximum_lower,
                maximum_upper,
                "V",
            ),
            "minimum-voltage": (
                minimum_voltage,
                minimum_lower,
                minimum_upper,
                "V",
            ),
            "sink-current": (sink_current, sink_lower, sink_upper, "A"),
        }
        scalars = tuple(
            NamedDecimal(value_id, Decimal(str(value)), unit)
            for value_id, (value, _lower, _upper, unit) in scalar_values.items()
        )
        scalar_uncertainties = tuple(
            NamedDecimal(
                value_id,
                Decimal(str(max(abs(value - lower), abs(upper - value)))),
                unit,
            )
            for value_id, (value, lower, upper, unit) in scalar_values.items()
        )
    return PhysicalScaleMorphismReceiverObservation(
        observation_id=observation_id,
        receiver_id=spec.receiver_id,
        coordinate_values_volts=tuple(Decimal(str(float(value))) for value in reduced),
        coordinate_uncertainty_volts=tuple(
            Decimal(str(float(value))) for value in reduced_uncertainty
        ),
        terminal_currents_amperes=tuple(Decimal(str(float(value))) for value in currents),
        terminal_current_uncertainty_amperes=tuple(
            Decimal(str(float(value))) for value in current_uncertainty
        ),
        carried_scalars=scalars,
        carried_scalar_uncertainties=scalar_uncertainties,
        uncertainty_method_id=uncertainty_method_id,
        derived_from_fine_receiver=True,
    )


def composition_defect(
    *,
    direct: PhysicalScaleMorphismReceiverSpec,
    first: PhysicalScaleMorphismReceiverSpec,
    second: PhysicalScaleMorphismReceiverSpec,
    capacitances: FloatArray,
) -> float:
    """Return max-norm definition defect for a nested weighted-mean composition."""

    weighted_kinds = {
        PhysicalScaleMorphismReceiverKind.GUARDED_MEAN,
        PhysicalScaleMorphismReceiverKind.WEIGHTED_MEAN,
    }
    if any(value.kind not in weighted_kinds for value in (direct, first)):
        raise ValueError("nested receiver composition requires weighted direct/first maps")
    if direct.scale_cells != first.scale_cells:
        raise ValueError("direct and first receiver scales differ")
    if (
        direct.source_receiver_id != first.source_receiver_id
        or second.source_receiver_id != first.receiver_id
    ):
        raise ValueError("receiver composition source/target identities do not form a chain")
    direct_matrix = receiver_matrix(direct, capacitances)
    first_matrix = receiver_matrix(first, capacitances)
    if second.scale_cells != first_matrix.shape[0]:
        raise ValueError("second receiver scale differs from first receiver output")
    coarse_capacitances = np.asarray(
        [
            np.sum(np.asarray(capacitances)[np.asarray(value.node_indices, dtype=np.int64)])
            for value in first.bins
        ],
        dtype=np.float64,
    )
    composed = receiver_matrix(second, coarse_capacitances) @ first_matrix
    if composed.shape != direct_matrix.shape:
        raise ValueError("direct and composed receiver matrices differ in shape")
    return float(np.max(np.abs(direct_matrix - composed)))


__all__ = [
    'PhysicalScaleMorphismReceiverBin',
    'PhysicalScaleMorphismReceiverKind',
    'PhysicalScaleMorphismReceiverObservation',
    'PhysicalScaleMorphismReceiverSpec',
    "composition_defect",
    "equal_width_receiver_spec",
    "receiver_matrix",
    "reduce_receiver",
]
