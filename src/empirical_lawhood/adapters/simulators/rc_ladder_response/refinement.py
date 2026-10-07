"""Fixed/adaptive refinement views and numerical qualification summaries."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import ResistorCapacitorLadderModelConfig, ResistorCapacitorLadderNumericalSummary
from .matrix_exponential import ResistorCapacitorLadderTrajectory, solve_matrix_exponential
from .mna import action_breakpoints, boundary_input, build_operator


def _currents(
    config: ResistorCapacitorLadderModelConfig,
    times: np.ndarray,
    voltages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray([boundary_input(config, float(value)) for value in times])
    left = (inputs[:, 0] - voltages[:, 0]) / float(config.left_source_resistance_ohms)
    right = (inputs[:, 1] - voltages[:, -1]) / float(config.right_termination_resistance_ohms)
    return left, right


def solve_backward_euler(
    config: ResistorCapacitorLadderModelConfig,
    *,
    maximum_step_seconds: float,
) -> ResistorCapacitorLadderTrajectory:
    if not np.isfinite(maximum_step_seconds) or maximum_step_seconds <= 0:
        raise ValueError("backward-Euler step must be positive and finite")
    operator = build_operator(config)
    output_times = np.asarray(config.output_times_seconds, dtype=np.float64)
    breakpoints = action_breakpoints(config)
    state = np.asarray(config.initial_voltages_volts, dtype=np.float64)
    states = [state.copy()]
    identity = np.eye(config.scale_cells, dtype=np.float64)
    for start, end in zip(output_times, output_times[1:], strict=False):
        cuts = (start, *(value for value in breakpoints if start < value < end), end)
        for left, right in zip(cuts, cuts[1:], strict=False):
            remaining = right - left
            time = left
            while remaining > 0:
                step = min(maximum_step_seconds, remaining)
                input_value = boundary_input(config, time + step / 2)
                state = np.linalg.solve(
                    identity - step * operator.state_matrix,
                    state + step * (operator.input_matrix @ input_value),
                )
                time += step
                remaining = max(0.0, right - time)
        states.append(state.copy())
    voltages = np.asarray(states, dtype=np.float64)
    left_current, right_current = _currents(config, output_times, voltages)
    return ResistorCapacitorLadderTrajectory(
        times_seconds=output_times,
        voltages_volts=voltages,
        left_current_amperes=left_current,
        right_current_amperes=right_current,
    )


def solve_adaptive_bdf(
    config: ResistorCapacitorLadderModelConfig,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> ResistorCapacitorLadderTrajectory:
    if not 0 < relative_tolerance < 1 or not 0 < absolute_tolerance < 1:
        raise ValueError("adaptive tolerances must lie inside (0, 1)")
    operator = build_operator(config)
    output_times = np.asarray(config.output_times_seconds, dtype=np.float64)
    state = np.asarray(config.initial_voltages_volts, dtype=np.float64)
    states = [state.copy()]
    breakpoints = action_breakpoints(config)
    for start, end in zip(output_times, output_times[1:], strict=False):
        cuts = (start, *(value for value in breakpoints if start < value < end), end)
        for left, right in zip(cuts, cuts[1:], strict=False):
            input_value = boundary_input(config, (left + right) / 2)

            def rhs(_: float, value: np.ndarray) -> np.ndarray:
                return np.asarray(
                    operator.state_matrix @ value + operator.input_matrix @ input_value,
                    dtype=np.float64,
                )

            solution = solve_ivp(
                rhs,
                (left, right),
                state,
                method="BDF",
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                t_eval=(right,),
            )
            if not solution.success or solution.y.shape != (config.scale_cells, 1):
                raise RuntimeError(f"adaptive RC integration failed: {solution.message}")
            state = np.asarray(solution.y[:, -1], dtype=np.float64)
        states.append(state.copy())
    voltages = np.asarray(states, dtype=np.float64)
    left_current, right_current = _currents(config, output_times, voltages)
    return ResistorCapacitorLadderTrajectory(
        times_seconds=output_times,
        voltages_volts=voltages,
        left_current_amperes=left_current,
        right_current_amperes=right_current,
    )


def _trajectory_digest(trajectory: ResistorCapacitorLadderTrajectory) -> str:
    digest = sha256()
    for value in (
        trajectory.times_seconds,
        trajectory.voltages_volts,
        trajectory.left_current_amperes,
        trajectory.right_current_amperes,
    ):
        digest.update(np.ascontiguousarray(value.astype("<f8", copy=False)).tobytes())
    return digest.hexdigest()


def _charge_rate_balance_defect(
    config: ResistorCapacitorLadderModelConfig,
    trajectory: ResistorCapacitorLadderTrajectory,
) -> float:
    """Return an output-grid charge/current residual in amperes.

    This deliberately differentiates the emitted stored-charge trajectory and
    compares it with a trapezoidal boundary-current integral.  Re-evaluating
    the state-space right-hand side would make Kirchhoff balance an algebraic
    identity and could not detect a defective trajectory.  Endpoint inputs are
    taken from inside each interval so an action breakpoint at the right edge
    is not assigned to the preceding interval.
    """

    operator = build_operator(config)
    defects = []
    left_resistance = float(config.left_source_resistance_ohms)
    right_resistance = float(config.right_termination_resistance_ohms)
    for left_time, right_time, left_state, right_state in zip(
        trajectory.times_seconds[:-1],
        trajectory.times_seconds[1:],
        trajectory.voltages_volts[:-1],
        trajectory.voltages_volts[1:],
        strict=True,
    ):
        duration = float(right_time - left_time)
        left_input = boundary_input(config, float(np.nextafter(left_time, right_time)))
        right_input = boundary_input(config, float(np.nextafter(right_time, left_time)))
        left_injected = (left_input[0] - left_state[0]) / left_resistance
        left_injected += (left_input[1] - left_state[-1]) / right_resistance
        right_injected = (right_input[0] - right_state[0]) / left_resistance
        right_injected += (right_input[1] - right_state[-1]) / right_resistance
        stored_charge_change = float(operator.capacitances @ (right_state - left_state))
        integrated_boundary_current = Decimal(str(duration)) * Decimal(
            str((left_injected + right_injected) / 2)
        )
        charge_residual = abs(Decimal(str(stored_charge_change)) - integrated_boundary_current)
        defects.append(float(charge_residual / Decimal(str(duration))))
    return max(defects, default=0.0)


def summarize_against_matrix_exponential(
    *,
    summary_id: str,
    config: ResistorCapacitorLadderModelConfig,
    view_id: str,
    trajectory: ResistorCapacitorLadderTrajectory,
    convergence_tolerance_volts: float,
) -> ResistorCapacitorLadderNumericalSummary:
    if convergence_tolerance_volts <= 0:
        raise ValueError("convergence tolerance must be positive")
    exact = solve_matrix_exponential(config)
    if trajectory.voltages_volts.shape != exact.voltages_volts.shape:
        raise ValueError("numerical trajectory output grid differs from comparator")
    defect = float(np.max(np.abs(trajectory.voltages_volts - exact.voltages_volts)))
    balance = _charge_rate_balance_defect(config, trajectory)
    converged = defect <= convergence_tolerance_volts
    return ResistorCapacitorLadderNumericalSummary(
        summary_id=summary_id,
        config_id=config.config_id,
        view_id=view_id,
        trajectory_sha256=_trajectory_digest(trajectory),
        maximum_voltage_defect_volts=Decimal(str(defect)),
        maximum_output_grid_charge_rate_defect_amperes=Decimal(str(balance)),
        converged=converged,
        warning_codes=() if converged else ("voltage-defect-above-frozen-tolerance",),
    )


def refinement_chain(
    config: ResistorCapacitorLadderModelConfig,
    *,
    h0_seconds: float,
    convergence_tolerance_volts: float,
) -> tuple[tuple[str, ResistorCapacitorLadderTrajectory, ResistorCapacitorLadderNumericalSummary], ...]:
    if h0_seconds <= 0:
        raise ValueError("refinement base step must be positive")
    outputs = []
    for view_id, step in (
        ("h0", h0_seconds),
        ("h0-half", h0_seconds / 2),
        ("h0-quarter", h0_seconds / 4),
    ):
        trajectory = solve_backward_euler(config, maximum_step_seconds=step)
        outputs.append(
            (
                view_id,
                trajectory,
                summarize_against_matrix_exponential(
                    summary_id=f"summary.{config.config_id}.{view_id}",
                    config=config,
                    view_id=view_id,
                    trajectory=trajectory,
                    convergence_tolerance_volts=convergence_tolerance_volts,
                ),
            )
        )
    adaptive = solve_adaptive_bdf(
        config,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-12,
    )
    outputs.append(
        (
            "adaptive-bdf",
            adaptive,
            summarize_against_matrix_exponential(
                summary_id=f"summary.{config.config_id}.adaptive-bdf",
                config=config,
                view_id="adaptive-bdf",
                trajectory=adaptive,
                convergence_tolerance_volts=convergence_tolerance_volts,
            ),
        )
    )
    return tuple(outputs)


__all__ = [
    "refinement_chain",
    "solve_adaptive_bdf",
    "solve_backward_euler",
    "summarize_against_matrix_exponential",
]
