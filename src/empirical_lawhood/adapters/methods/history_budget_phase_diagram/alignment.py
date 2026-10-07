"""Whole-unit paired K/B cross-scale alignment for history budget phase diagram."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramScientificState


IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PairedAlignmentEstimate:
    unit_deltas: tuple[int, ...]
    mean_delta: Decimal
    lower_95: Decimal
    upper_95: Decimal
    disposition: HistoryBudgetPhaseDiagramScientificState


def _discordance(states: IntArray) -> IntArray:
    if states.ndim != 3 or states.shape[1:] != (3, 2):
        raise ValueError("history budget phase diagram alignment states require unit x scale x contrast")
    return np.asarray(
        [
            sum(
                int(row[left, contrast] != row[right, contrast])
                for contrast in range(2)
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            for row in states
        ],
        dtype=np.int64,
    )


def paired_alignment(
    *,
    absolute_states: IntArray,
    normalized_states: IntArray,
    family_codes: IntArray,
    resamples: int,
    seed: int,
) -> PairedAlignmentEstimate:
    """Estimate Delta_align=d_B-d_K with stratified whole-unit resampling."""

    if absolute_states.shape != normalized_states.shape:
        raise ValueError("history budget phase diagram K/B alignment rosters differ")
    if family_codes.shape != (absolute_states.shape[0],) or resamples != 10_000:
        raise ValueError("history budget phase diagram alignment family/bootstrap roster differs")
    unique_codes = tuple(int(value) for value in np.unique(family_codes))
    if not unique_codes or not set(unique_codes) <= {0, 1, 2}:
        raise ValueError("history budget phase diagram alignment family codes differ")
    deltas = _discordance(normalized_states) - _discordance(absolute_states)
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty(resamples, dtype=np.float64)
    family_indices = tuple(np.flatnonzero(family_codes == code) for code in unique_codes)
    if any(values.size == 0 for values in family_indices):
        raise ValueError("history budget phase diagram alignment has an empty family stratum")
    for index in range(resamples):
        sample = np.concatenate(
            tuple(rng.choice(values, size=values.size, replace=True) for values in family_indices)
        )
        bootstrap[index] = float(np.mean(deltas[sample]))
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    disposition = (
        HistoryBudgetPhaseDiagramScientificState.ABSOLUTE_DEPTH_ALIGNED
        if lower > 0.0
        else HistoryBudgetPhaseDiagramScientificState.NORMALIZED_BUDGET_ALIGNED
        if upper < 0.0
        else HistoryBudgetPhaseDiagramScientificState.NOT_DISTINGUISHED
    )
    return PairedAlignmentEstimate(
        unit_deltas=tuple(int(value) for value in deltas),
        mean_delta=Decimal(str(float(np.mean(deltas)))),
        lower_95=Decimal(str(float(lower))),
        upper_95=Decimal(str(float(upper))),
        disposition=disposition,
    )


__all__ = ["PairedAlignmentEstimate", "paired_alignment"]
