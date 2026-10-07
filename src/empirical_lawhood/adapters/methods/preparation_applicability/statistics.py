"""Exact whole-root paired binary comparison, independent of policy fitting."""

from math import comb

import numpy as np
from numpy.typing import NDArray


def paired_binary_test(
    selected: NDArray[np.bool_], fixed: NDArray[np.bool_]
) -> tuple[float, int, int, float]:
    if (
        selected.shape != fixed.shape
        or selected.ndim != 1
        or selected.dtype != bool
        or fixed.dtype != bool
        or not len(selected)
    ):
        raise ValueError("comparison requires paired independent-root binary outcomes")
    wins = int(np.sum(selected & ~fixed))
    losses = int(np.sum(fixed & ~selected))
    discordant = wins + losses
    p = sum(comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant)
    return (wins - losses) / len(selected), wins, losses, p
