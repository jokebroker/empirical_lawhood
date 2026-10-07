"""Pure synthetic numerical qualification for the physical scale morphism RC-ladder provider."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import ResistorCapacitorLadderActionSegment, ResistorCapacitorLadderModelConfig, ResistorCapacitorLadderNumericalSummary
from .matrix_exponential import solve_matrix_exponential
from .refinement import refinement_chain, solve_backward_euler, summarize_against_matrix_exponential


@dataclass(frozen=True, slots=True)
class RcLadderResponseNumericalQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/rc-ladder-response/rc-ladder-response-numerical-qualification'

    qualification_id: str
    model_config_id: str
    summaries: tuple[ResistorCapacitorLadderNumericalSummary, ...]
    fixed_step_defects_strictly_decrease: bool
    fixed_step_charge_rate_defects_strictly_decrease: bool
    adaptive_comparator_converged: bool
    omitted_source_impedance_defect_volts: Decimal
    component_average_defect_volts: Decimal
    deliberate_nonconvergence_detected: bool
    numerical_method_qualified: bool
    reason_codes: tuple[str, ...]
    scientific_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.model_config_id, field_name="model_config_id")
        require_sorted_unique_ids(self.summaries, attribute="summary_id", field_name="summaries")
        for name in (
            "omitted_source_impedance_defect_volts",
            "component_average_defect_volts",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        expected = (
            self.fixed_step_defects_strictly_decrease
            and self.fixed_step_charge_rate_defects_strictly_decrease
            and self.adaptive_comparator_converged
            and self.omitted_source_impedance_defect_volts > Decimal("0.05")
            and self.component_average_defect_volts > Decimal("0.01")
            and self.deliberate_nonconvergence_detected
        )
        if self.numerical_method_qualified != expected:
            raise ValueError("numerical qualification is not derived from frozen checks")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("synthetic numerical qualification must remain nonpromotable")


def default_numerical_model_config() -> ResistorCapacitorLadderModelConfig:
    return ResistorCapacitorLadderModelConfig(
        config_id="rc-ladder-response-synthetic-n4",
        scale_cells=4,
        capacitances_farads=tuple(Decimal("0.001") for _ in range(4)),
        interior_resistances_ohms=tuple(Decimal("100") for _ in range(3)),
        left_source_resistance_ohms=Decimal("50"),
        right_termination_resistance_ohms=Decimal("100"),
        initial_voltages_volts=tuple(Decimal(0) for _ in range(4)),
        action_segments=(
            ResistorCapacitorLadderActionSegment(
                segment_id="segment-step",
                start_seconds=Decimal(0),
                end_seconds=Decimal(1),
                left_voltage_volts=Decimal(1),
                right_voltage_volts=Decimal(0),
            ),
        ),
        output_times_seconds=tuple(Decimal(index) / Decimal(20) for index in range(21)),
        component_metrology_frozen_before_response=True,
    )


def qualify_numerical_provider() -> RcLadderResponseNumericalQualification:
    config = default_numerical_model_config()
    chain = refinement_chain(
        config,
        h0_seconds=0.04,
        convergence_tolerance_volts=0.03,
    )
    fixed = tuple(
        float(summary.maximum_voltage_defect_volts)
        for view_id, _trajectory, summary in chain
        if view_id in {"h0", "h0-half", "h0-quarter"}
    )
    fixed_charge_rate = tuple(
        float(summary.maximum_output_grid_charge_rate_defect_amperes)
        for view_id, _trajectory, summary in chain
        if view_id in {"h0", "h0-half", "h0-quarter"}
    )
    exact = solve_matrix_exponential(config)
    omitted = solve_matrix_exponential(config, omit_source_impedance=True)
    nonuniform = ResistorCapacitorLadderModelConfig(
        config_id="rc-ladder-response-synthetic-nonuniform-n4",
        scale_cells=4,
        capacitances_farads=(
            Decimal("0.0005"),
            Decimal("0.001"),
            Decimal("0.0015"),
            Decimal("0.002"),
        ),
        interior_resistances_ohms=(Decimal("50"), Decimal("100"), Decimal("200")),
        left_source_resistance_ohms=config.left_source_resistance_ohms,
        right_termination_resistance_ohms=config.right_termination_resistance_ohms,
        initial_voltages_volts=config.initial_voltages_volts,
        action_segments=config.action_segments,
        output_times_seconds=config.output_times_seconds,
        component_metrology_frozen_before_response=True,
    )
    realized = solve_matrix_exponential(nonuniform)
    averaged = solve_matrix_exponential(nonuniform, average_components=True)
    coarse = solve_backward_euler(config, maximum_step_seconds=0.5)
    nonconverged = summarize_against_matrix_exponential(
        summary_id="summary.physical-scale-morphism-deliberate-nonconvergence",
        config=config,
        view_id="deliberate-nonconvergence",
        trajectory=coarse,
        convergence_tolerance_volts=1e-6,
    )
    omitted_defect = Decimal(
        str(float(np.max(np.abs(exact.voltages_volts - omitted.voltages_volts))))
    )
    average_defect = Decimal(
        str(float(np.max(np.abs(realized.voltages_volts - averaged.voltages_volts))))
    )
    decreasing = fixed[2] < fixed[1] < fixed[0]
    charge_rate_decreasing = fixed_charge_rate[2] < fixed_charge_rate[1] < fixed_charge_rate[0]
    adaptive = next(summary for view, _trajectory, summary in chain if view == "adaptive-bdf")
    qualified = (
        decreasing
        and charge_rate_decreasing
        and adaptive.converged
        and omitted_defect > Decimal("0.05")
        and average_defect > Decimal("0.01")
        and not nonconverged.converged
    )
    return RcLadderResponseNumericalQualification(
        qualification_id="qualification.physical-scale-morphism-numerical-provider",
        model_config_id=config.config_id,
        summaries=tuple(
            sorted(
                (*(value[2] for value in chain), nonconverged), key=lambda value: value.summary_id
            )
        ),
        fixed_step_defects_strictly_decrease=decreasing,
        fixed_step_charge_rate_defects_strictly_decrease=charge_rate_decreasing,
        adaptive_comparator_converged=adaptive.converged,
        omitted_source_impedance_defect_volts=omitted_defect,
        component_average_defect_volts=average_defect,
        deliberate_nonconvergence_detected=not nonconverged.converged,
        numerical_method_qualified=qualified,
        reason_codes=(
            "SYNTHETIC_NUMERICAL_METHOD_QUALIFIED"
            if qualified
            else "SYNTHETIC_NUMERICAL_METHOD_NOT_QUALIFIED",
        ),
        scientific_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'RcLadderResponseNumericalQualification',
    "default_numerical_model_config",
    "qualify_numerical_provider",
]
