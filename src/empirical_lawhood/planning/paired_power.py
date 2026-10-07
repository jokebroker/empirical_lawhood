"""Exact paired Bernoulli power calculation for prospective design."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def paired_binary_power(
    n: int,
    delta: float,
    discordance: float,
    alpha: float,
    materiality: float = 0.0,
) -> float:
    """Power of a one-sided conditional paired test under independent root pairs.

    The discordant count is Binomial(n, discordance); conditional wins are
    Binomial(M, (discordance + delta) / (2 * discordance)). This calculation
    does not model task effects, calibration, unknowns, or active selection.
    """

    if not (
        isinstance(n, int)
        and n > 0
        and math.isfinite(delta)
        and math.isfinite(discordance)
        and abs(delta) <= discordance <= 1
        and math.isfinite(alpha)
        and 0 < alpha < 1
        and math.isfinite(materiality)
        and materiality >= 0
    ):
        raise ValueError("invalid paired Bernoulli design parameters")
    if discordance == 0:
        return 0.0
    total = 0.0
    for count in range(n + 1):
        wins = np.arange(count + 1)
        qualifies = binom.sf(wins - 1, count, 0.5) <= alpha
        qualifies &= (2 * wins - count) >= math.ceil(n * materiality - 1e-12)
        conditional = binom.pmf(
            wins, count, (discordance + delta) / (2 * discordance)
        )
        total += float(
            binom.pmf(count, n, discordance) * conditional[qualifies].sum()
        )
    return total
