"""Strict numerical RC-ladder contracts for physical scale morphism."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderActionSegment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-action-segment'

    segment_id: str
    start_seconds: Decimal
    end_seconds: Decimal
    left_voltage_volts: Decimal
    right_voltage_volts: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.segment_id, field_name="segment_id")
        for name in (
            "start_seconds",
            "end_seconds",
            "left_voltage_volts",
            "right_voltage_volts",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("RC action segment interval is invalid")


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderModelConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-model-config'

    config_id: str
    scale_cells: int
    capacitances_farads: tuple[Decimal, ...]
    interior_resistances_ohms: tuple[Decimal, ...]
    left_source_resistance_ohms: Decimal
    right_termination_resistance_ohms: Decimal
    initial_voltages_volts: tuple[Decimal, ...]
    action_segments: tuple[ResistorCapacitorLadderActionSegment, ...]
    output_times_seconds: tuple[Decimal, ...]
    component_metrology_frozen_before_response: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.scale_cells <= 0:
            raise ValueError("RC ladder must contain cells")
        if len(self.capacitances_farads) != self.scale_cells:
            raise ValueError("capacitance roster differs from ladder scale")
        if len(self.interior_resistances_ohms) != self.scale_cells - 1:
            raise ValueError("interior resistance roster differs from ladder topology")
        if len(self.initial_voltages_volts) != self.scale_cells:
            raise ValueError("initial voltage roster differs from ladder scale")
        for name, values in (
            ("capacitances_farads", self.capacitances_farads),
            ("interior_resistances_ohms", self.interior_resistances_ohms),
        ):
            for index, value in enumerate(values):
                validate_decimal(
                    value, field_name=f"{name}[{index}]", minimum=Decimal(0)
                )
                if value == 0:
                    raise ValueError(f"{name} must be positive")
        for name in (
            "left_source_resistance_ohms",
            "right_termination_resistance_ohms",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for index, value in enumerate(self.initial_voltages_volts):
            validate_decimal(value, field_name=f"initial_voltages_volts[{index}]")
        require_sorted_unique_ids(
            self.action_segments, attribute="segment_id", field_name="action_segments"
        )
        chronological = tuple(
            sorted(self.action_segments, key=lambda value: value.start_seconds)
        )
        if chronological != self.action_segments:
            raise ValueError("action segments must be chronological")
        if any(
            left.end_seconds > right.start_seconds
            for left, right in pairwise(chronological)
        ):
            raise ValueError("action segments overlap")
        if tuple(sorted(set(self.output_times_seconds))) != self.output_times_seconds:
            raise ValueError("output times must be sorted and unique")
        if len(self.output_times_seconds) < 2 or self.output_times_seconds[0] != 0:
            raise ValueError(
                "output time roster must start at zero and contain a horizon"
            )
        for index, value in enumerate(self.output_times_seconds):
            validate_decimal(
                value, field_name=f"output_times_seconds[{index}]", minimum=Decimal(0)
            )
        if (
            chronological
            and chronological[-1].end_seconds > self.output_times_seconds[-1]
        ):
            raise ValueError("action extends beyond output horizon")
        if not self.component_metrology_frozen_before_response:
            raise ValueError("board-specific twin requires pre-response metrology")


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderStudyConfig(CanonicalRecord):
    """A prospective model unit and two numerical views, without board claims."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/rc-ladder-response/resistor-capacitor-ladder-study-config'

    study_id: str
    unit_id: str
    model: ResistorCapacitorLadderModelConfig
    backward_euler_maximum_step_seconds: Decimal
    maximum_view_defect_volts: Decimal
    maximum_charge_rate_defect_amperes: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.study_id, field_name="study_id")
        validate_stable_id(self.unit_id, field_name="unit_id")
        for name in (
            "backward_euler_maximum_step_seconds",
            "maximum_view_defect_volts",
            "maximum_charge_rate_defect_amperes",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.backward_euler_maximum_step_seconds > min(
            right - left
            for left, right in zip(
                self.model.output_times_seconds,
                self.model.output_times_seconds[1:],
                strict=False,
            )
        ):
            raise ValueError("RC backward-Euler step exceeds an output interval")


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderNumericalSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-numerical-summary'

    summary_id: str
    config_id: str
    view_id: str
    trajectory_sha256: str
    maximum_voltage_defect_volts: Decimal
    maximum_output_grid_charge_rate_defect_amperes: Decimal
    converged: bool
    warning_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.view_id, field_name="view_id")
        if len(self.trajectory_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.trajectory_sha256
        ):
            raise ValueError("trajectory digest must be lowercase SHA-256")
        for name in (
            "maximum_voltage_defect_volts",
            "maximum_output_grid_charge_rate_defect_amperes",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if tuple(sorted(set(self.warning_codes))) != self.warning_codes:
            raise ValueError("numerical warning codes must be sorted and unique")
        if self.converged and self.warning_codes:
            raise ValueError(
                "converged numerical summary cannot retain failure warnings"
            )


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderNativePanel(CanonicalRecord):
    """One prepared circuit in one nested numerical view, with native SI data."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/rc-ladder-response/resistor-capacitor-ladder-native-panel'

    panel_id: str
    config_id: str
    unit_id: str
    view_id: str
    times_seconds: tuple[Decimal, ...]
    node_voltages_volts: tuple[tuple[Decimal, ...], ...]
    left_boundary_currents_amperes: tuple[Decimal, ...]
    right_boundary_currents_amperes: tuple[Decimal, ...]
    numerical_summary: ResistorCapacitorLadderNumericalSummary

    def __post_init__(self) -> None:
        for name in ("panel_id", "config_id", "unit_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        n = len(self.times_seconds)
        if (
            n < 2
            or self.times_seconds[0] != 0
            or self.times_seconds != tuple(sorted(set(self.times_seconds)))
            or len(self.node_voltages_volts) != n
            or len(self.left_boundary_currents_amperes) != n
            or len(self.right_boundary_currents_amperes) != n
            or not self.node_voltages_volts[0]
            or any(
                len(row) != len(self.node_voltages_volts[0])
                for row in self.node_voltages_volts
            )
        ):
            raise ValueError("RC native panel grid or node roster differs")
        for time in self.times_seconds:
            validate_decimal(time, field_name="times_seconds", minimum=Decimal(0))
        for row in self.node_voltages_volts:
            for voltage in row:
                validate_decimal(voltage, field_name="node_voltages_volts")
        for current in (
            *self.left_boundary_currents_amperes,
            *self.right_boundary_currents_amperes,
        ):
            validate_decimal(current, field_name="boundary_currents_amperes")
        if (
            self.numerical_summary.config_id != self.config_id
            or self.numerical_summary.view_id != self.view_id
        ):
            raise ValueError("RC native panel summary belongs to another config/view")


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderNumericalCheck(CanonicalRecord):
    """A bounded numerical conformance result, never a physical-board result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/rc-ladder-response/resistor-capacitor-ladder-numerical-check'

    check_id: str
    config_id: str
    unit_id: str
    view_ids: tuple[str, ...]
    maximum_view_defect_volts: Decimal
    maximum_charge_rate_defect_amperes: Decimal
    independent_unit_count: int
    nested_view_count: int
    converged: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("check_id", "config_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.view_ids != tuple(sorted(set(self.view_ids)))
            or len(self.view_ids) != 2
        ):
            raise ValueError("RC check requires exactly two nested numerical views")
        for name in ("maximum_view_defect_volts", "maximum_charge_rate_defect_amperes"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.independent_unit_count != 1 or self.nested_view_count != 2:
            raise ValueError("RC check cannot count views as independent units")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("RC check reasons must be sorted and unique")
        if self.converged and self.reason_codes:
            raise ValueError("converged RC check cannot carry refusal reasons")


__all__ = [
    'ResistorCapacitorLadderActionSegment',
    'ResistorCapacitorLadderModelConfig',
    'ResistorCapacitorLadderNativePanel',
    'ResistorCapacitorLadderNumericalCheck',
    'ResistorCapacitorLadderNumericalSummary',
    'ResistorCapacitorLadderStudyConfig',
]
