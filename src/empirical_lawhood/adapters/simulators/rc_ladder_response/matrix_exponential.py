"""Independent piecewise-constant matrix-exponential RC solution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from .contracts import ResistorCapacitorLadderModelConfig
from .mna import action_breakpoints, boundary_input, build_operator


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderTrajectory:
    times_seconds: FloatArray
    voltages_volts: FloatArray
    left_current_amperes: FloatArray
    right_current_amperes: FloatArray


def _propagate_constant(
    state: FloatArray,
    *,
    state_matrix: FloatArray,
    input_matrix: FloatArray,
    input_value: FloatArray,
    duration: float,
) -> FloatArray:
    if duration <= 0:
        return state
    n_state = state_matrix.shape[0]
    augmented = np.zeros((n_state + 1, n_state + 1), dtype=np.float64)
    augmented[:n_state, :n_state] = state_matrix
    augmented[:n_state, n_state] = input_matrix @ input_value
    propagated = expm(augmented * duration) @ np.concatenate((state, np.ones(1)))
    return np.asarray(propagated[:n_state], dtype=np.float64)


def solve_matrix_exponential(
    config: ResistorCapacitorLadderModelConfig,
    *,
    omit_source_impedance: bool = False,
    average_components: bool = False,
) -> ResistorCapacitorLadderTrajectory:
    operator = build_operator(
        config,
        omit_source_impedance=omit_source_impedance,
        average_components=average_components,
    )
    output_times = np.asarray(config.output_times_seconds, dtype=np.float64)
    breakpoints = action_breakpoints(config)
    state = np.asarray(config.initial_voltages_volts, dtype=np.float64)
    states = [state.copy()]
    for start, end in zip(output_times, output_times[1:], strict=False):
        cuts = (start, *(value for value in breakpoints if start < value < end), end)
        for left, right in zip(cuts, cuts[1:], strict=False):
            input_value = boundary_input(config, (left + right) / 2)
            state = _propagate_constant(
                state,
                state_matrix=operator.state_matrix,
                input_matrix=operator.input_matrix,
                input_value=input_value,
                duration=right - left,
            )
        states.append(state.copy())
    voltage_matrix = np.asarray(states, dtype=np.float64)
    left_resistance = float(config.left_source_resistance_ohms)
    if omit_source_impedance:
        resistances = np.asarray(config.interior_resistances_ohms, dtype=np.float64)
        left_resistance = min(left_resistance, max(np.min(resistances) * 1e-6, 1e-12))
    right_resistance = float(config.right_termination_resistance_ohms)
    inputs = np.asarray([boundary_input(config, value) for value in output_times], dtype=np.float64)
    left_current = (inputs[:, 0] - voltage_matrix[:, 0]) / left_resistance
    right_current = (inputs[:, 1] - voltage_matrix[:, -1]) / right_resistance
    return ResistorCapacitorLadderTrajectory(
        times_seconds=output_times,
        voltages_volts=voltage_matrix,
        left_current_amperes=left_current,
        right_current_amperes=right_current,
    )


__all__ = ['ResistorCapacitorLadderTrajectory', "solve_matrix_exponential"]
