"""Native four-cell RC output and numerical falsifier for one prepared model.

These functions are pure.  The two solver views share one independent model
unit; they do not represent two boards or independent trials.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from .contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderNumericalCheck, ResistorCapacitorLadderStudyConfig
from .matrix_exponential import solve_matrix_exponential
from .refinement import solve_backward_euler, summarize_against_matrix_exponential

VIEWS = ("backward-euler", "matrix-exponential")


def _decimals(values: np.ndarray) -> tuple[Decimal, ...]:
    if not np.all(np.isfinite(values)):
        raise ValueError("RC numerical output contains a non-finite value")
    return tuple(Decimal(str(float(value))) for value in values)


def run_native_view(
    study: ResistorCapacitorLadderStudyConfig, view_id: str
) -> ResistorCapacitorLadderNativePanel:
    """Execute one frozen solver view without changing the model or action."""

    if view_id == "matrix-exponential":
        trajectory = solve_matrix_exponential(study.model)
    elif view_id == "backward-euler":
        trajectory = solve_backward_euler(
            study.model,
            maximum_step_seconds=float(study.backward_euler_maximum_step_seconds),
        )
    else:
        raise ValueError("unsupported RC numerical view")
    summary = summarize_against_matrix_exponential(
        summary_id=f"{study.study_id}.summary.{view_id}",
        config=study.model,
        view_id=view_id,
        trajectory=trajectory,
        convergence_tolerance_volts=float(study.maximum_view_defect_volts),
    )
    return ResistorCapacitorLadderNativePanel(
        panel_id=f"{study.study_id}.panel.{view_id}",
        config_id=study.model.config_id,
        unit_id=study.unit_id,
        view_id=view_id,
        times_seconds=_decimals(trajectory.times_seconds),
        node_voltages_volts=tuple(_decimals(row) for row in trajectory.voltages_volts),
        left_boundary_currents_amperes=_decimals(trajectory.left_current_amperes),
        right_boundary_currents_amperes=_decimals(trajectory.right_current_amperes),
        numerical_summary=summary,
    )


def check_native_views(
    study: ResistorCapacitorLadderStudyConfig,
    panels: tuple[ResistorCapacitorLadderNativePanel, ...],
) -> ResistorCapacitorLadderNumericalCheck:
    """Compare emitted native rows and charge residuals, including refusal cases."""

    if len(panels) != 2 or tuple(sorted(panel.view_id for panel in panels)) != VIEWS:
        raise ValueError("RC numerical check requires its two exact solver views")
    by_view = {panel.view_id: panel for panel in panels}
    if any(
        panel.config_id != study.model.config_id
        or panel.unit_id != study.unit_id
        or panel.times_seconds != study.model.output_times_seconds
        or any(len(row) != study.model.scale_cells for row in panel.node_voltages_volts)
        for panel in panels
    ):
        raise ValueError("RC native panel changes its model, unit, grid or topology")
    exact = by_view["matrix-exponential"]
    approximate = by_view["backward-euler"]
    defect = max(
        abs(left - right)
        for exact_row, approximate_row in zip(
            exact.node_voltages_volts,
            approximate.node_voltages_volts,
            strict=True,
        )
        for left, right in zip(exact_row, approximate_row, strict=True)
    )
    charge_defect = max(
        panel.numerical_summary.maximum_output_grid_charge_rate_defect_amperes
        for panel in panels
    )
    reasons = []
    if defect > study.maximum_view_defect_volts or not all(
        panel.numerical_summary.converged for panel in panels
    ):
        reasons.append("VOLTAGE_VIEW_DEFECT_ABOVE_FROZEN_LIMIT")
    if charge_defect > study.maximum_charge_rate_defect_amperes:
        reasons.append("CHARGE_RATE_DEFECT_ABOVE_FROZEN_LIMIT")
    return ResistorCapacitorLadderNumericalCheck(
        check_id=f"{study.study_id}.numerical-check",
        config_id=study.model.config_id,
        unit_id=study.unit_id,
        view_ids=VIEWS,
        maximum_view_defect_volts=defect,
        maximum_charge_rate_defect_amperes=charge_defect,
        independent_unit_count=1,
        nested_view_count=2,
        converged=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = ["VIEWS", "check_native_views", "run_native_view"]
