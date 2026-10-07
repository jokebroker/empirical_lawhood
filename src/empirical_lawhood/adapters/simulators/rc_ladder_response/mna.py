"""Modified-nodal linear state-space construction for the RC ladder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contracts import ResistorCapacitorLadderModelConfig


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderOperator:
    capacitances: FloatArray
    conductance: FloatArray
    state_matrix: FloatArray
    input_matrix: FloatArray


def build_operator(
    config: ResistorCapacitorLadderModelConfig,
    *,
    omit_source_impedance: bool = False,
    average_components: bool = False,
) -> ResistorCapacitorLadderOperator:
    n_cells = config.scale_cells
    capacitances = np.asarray(config.capacitances_farads, dtype=np.float64)
    resistances = np.asarray(config.interior_resistances_ohms, dtype=np.float64)
    left_resistance = float(config.left_source_resistance_ohms)
    right_resistance = float(config.right_termination_resistance_ohms)
    if average_components:
        capacitances = np.full(n_cells, np.mean(capacitances), dtype=np.float64)
        if resistances.size:
            resistances = np.full(n_cells - 1, np.mean(resistances), dtype=np.float64)
    if omit_source_impedance:
        left_resistance = min(left_resistance, max(np.min(resistances) * 1e-6, 1e-12))
    conductance = np.zeros((n_cells, n_cells), dtype=np.float64)
    for index, resistance in enumerate(resistances):
        value = 1.0 / resistance
        conductance[index, index] += value
        conductance[index + 1, index + 1] += value
        conductance[index, index + 1] -= value
        conductance[index + 1, index] -= value
    left_g = 1.0 / left_resistance
    right_g = 1.0 / right_resistance
    conductance[0, 0] += left_g
    conductance[-1, -1] += right_g
    input_matrix = np.zeros((n_cells, 2), dtype=np.float64)
    input_matrix[0, 0] = left_g / capacitances[0]
    input_matrix[-1, 1] = right_g / capacitances[-1]
    state_matrix = -(conductance / capacitances[:, None])
    if not np.all(np.linalg.eigvalsh(conductance) > 0):
        raise ValueError("terminated RC conductance matrix must be positive definite")
    return ResistorCapacitorLadderOperator(
        capacitances=capacitances,
        conductance=conductance,
        state_matrix=state_matrix,
        input_matrix=input_matrix,
    )


def boundary_input(config: ResistorCapacitorLadderModelConfig, time_seconds: float) -> FloatArray:
    for segment in config.action_segments:
        if float(segment.start_seconds) <= time_seconds < float(segment.end_seconds):
            return np.asarray(
                [float(segment.left_voltage_volts), float(segment.right_voltage_volts)],
                dtype=np.float64,
            )
    return np.zeros(2, dtype=np.float64)


def action_breakpoints(config: ResistorCapacitorLadderModelConfig) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                float(value)
                for segment in config.action_segments
                for value in (segment.start_seconds, segment.end_seconds)
            }
        )
    )


def slowest_decay_rate(operator: ResistorCapacitorLadderOperator) -> float:
    eigenvalues = np.linalg.eigvals(operator.state_matrix)
    if np.any(np.real(eigenvalues) >= 0):
        raise ValueError("RC operator is not asymptotically stable")
    return float(np.min(-np.real(eigenvalues)))


__all__ = [
    'ResistorCapacitorLadderOperator',
    "action_breakpoints",
    "boundary_input",
    "build_operator",
    "slowest_decay_rate",
]
