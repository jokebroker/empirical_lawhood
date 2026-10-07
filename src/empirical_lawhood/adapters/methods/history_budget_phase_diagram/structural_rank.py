"""Exact graph-support observability-jet term rank for history budget phase diagram."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramCoordinateLabel, HistoryBudgetPhaseDiagramDenominatorDescriptor, HistoryBudgetPhaseDiagramStructuralRankStep
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .history import assemble_dense_operator


BoolArray = npt.NDArray[np.bool_]


def _boolean_product(left: BoolArray, right: BoolArray) -> BoolArray:
    return np.asarray((left.astype(np.uint16) @ right.astype(np.uint16)) > 0, dtype=np.bool_)


def maximum_bipartite_matching_rank(support: BoolArray) -> int:
    """Return exact maximum matching size of a Boolean matrix support."""

    if support.ndim != 2:
        raise ValueError("history budget phase diagram support matrix must be two-dimensional")
    row_count, column_count = support.shape
    matched_row_by_column = [-1] * column_count

    def augment(row: int, seen: list[bool]) -> bool:
        for column in np.flatnonzero(support[row]):
            index = int(column)
            if seen[index]:
                continue
            seen[index] = True
            previous = matched_row_by_column[index]
            if previous == -1 or augment(previous, seen):
                matched_row_by_column[index] = row
                return True
        return False

    rank = 0
    for row in range(row_count):
        if augment(row, [False] * column_count):
            rank += 1
    return rank


def observability_jet_support(
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    depth: int,
) -> BoolArray:
    """Construct support of [C; CA; ...; CA^k] without numerical thresholds."""

    if depth < 0 or depth > 31:
        raise ValueError("history budget phase diagram structural depth is outside the frozen roster")
    operator = assemble_dense_operator(descriptor)
    state_support = np.asarray(operator.state_matrix != 0.0, dtype=np.bool_)
    current = np.asarray(operator.receiver != 0.0, dtype=np.bool_)
    blocks = [current.copy()]
    for _ in range(depth):
        current = _boolean_product(current, state_support)
        blocks.append(current.copy())
    result = np.vstack(blocks)
    if result.shape != (8 * (depth + 1), descriptor.scale_cells):
        raise ValueError("history budget phase diagram observability-jet support shape differs")
    return result


def structural_rank_step(
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
) -> HistoryBudgetPhaseDiagramStructuralRankStep:
    if coordinate.scale_cells != descriptor.scale_cells:
        raise ValueError("history budget phase diagram structural coordinate/descriptor scale differs")
    support = observability_jet_support(descriptor, coordinate.depth)
    support_payload = canonical_json_bytes(
        {
            "shape": support.shape,
            "true_indices": tuple((int(row), int(column)) for row, column in np.argwhere(support)),
        }
    )
    return HistoryBudgetPhaseDiagramStructuralRankStep(
        coordinate_id=coordinate.coordinate_id,
        structural_rank=maximum_bipartite_matching_rank(support),
        row_count=support.shape[0],
        state_count=support.shape[1],
        support_sha256=sha256(support_payload).hexdigest(),
        algorithm_id="history-budget-phase-diagram-observability-jet-matching",
    )


__all__ = [
    "maximum_bipartite_matching_rank",
    "observability_jet_support",
    "structural_rank_step",
]
