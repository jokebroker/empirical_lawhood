"""Absolute-resolution conditioning diagnostics for history budget phase diagram histories."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramConditioningStep, HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramCoordinateLabel, HistoryBudgetPhaseDiagramDenominatorDescriptor

from .history import sampled_history_matrix


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("history budget phase diagram conditioning value is nonfinite")
    return Decimal(str(float(value)))


def dimensionless_history_operator(
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Apply the frozen capacitance-energy and RMS/resolution normalization."""

    if coordinate.scale_cells != descriptor.scale_cells:
        raise ValueError("history budget phase diagram conditioning coordinate/descriptor scale differs")
    matrix = sampled_history_matrix(config, descriptor, coordinate.depth)
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    c_bar = float(descriptor.capacitance_bar_farads)
    v_ref = float(descriptor.voltage_reference_volts)
    diagonal = v_ref * np.sqrt(descriptor.scale_cells * c_bar / capacitances)
    denominator = (
        float(coordinate.resolution_epsilon) * v_ref * np.sqrt(float(coordinate.depth + 1))
    )
    normalized = matrix * diagonal[None, :] / denominator
    if not np.all(np.isfinite(normalized)):
        raise ValueError("history budget phase diagram dimensionless history operator is nonfinite")
    return np.asarray(normalized, dtype=np.float64)


def conditioning_step(
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
    *,
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
) -> HistoryBudgetPhaseDiagramConditioningStep:
    normalized = (
        dimensionless_history_operator(config, descriptor, coordinate)
        if matrix is None
        else _normalize_matrix(matrix, descriptor, coordinate)
    )
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("history budget phase diagram conditioning spectrum is empty")
    smallest = float(singular_values[-1])
    condition = None if smallest == 0.0 else float(singular_values[0] / smallest)
    frobenius_squared = float(np.sum(singular_values**2))
    stable_rank = frobenius_squared / float(singular_values[0] ** 2)
    effective_rank = int(np.sum(singular_values >= 1.0))
    row_ceiling = min(normalized.shape)
    return HistoryBudgetPhaseDiagramConditioningStep(
        coordinate_id=coordinate.coordinate_id,
        resolution_epsilon=coordinate.resolution_epsilon,
        largest_singular_value=_decimal(float(singular_values[0])),
        smallest_singular_value=_decimal(smallest),
        condition_number=None if condition is None else _decimal(condition),
        stable_rank=_decimal(stable_rank),
        effective_rank=effective_rank,
        right_censored=effective_rank < row_ceiling
        and coordinate.depth == config.history_max_depth,
        algorithm_id=config.conditioning_algorithm_id,
    )


def _normalize_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    if matrix.shape != (8 * (coordinate.depth + 1), descriptor.scale_cells):
        raise ValueError("history budget phase diagram conditioning matrix shape differs")
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    c_bar = float(descriptor.capacitance_bar_farads)
    v_ref = float(descriptor.voltage_reference_volts)
    diagonal = v_ref * np.sqrt(descriptor.scale_cells * c_bar / capacitances)
    denominator = (
        float(coordinate.resolution_epsilon) * v_ref * np.sqrt(float(coordinate.depth + 1))
    )
    return np.asarray(matrix * diagonal[None, :] / denominator, dtype=np.float64)


__all__ = ["conditioning_step", "dimensionless_history_operator"]
