"""Fixed whole-independent-root resampling and nearest-rank intervals."""

from decimal import Decimal
import numpy as np


def root_indices(*, roots: int, draws: int, seed: int) -> np.ndarray:
    if roots < 1 or draws < 1:
        raise ValueError("bootstrap requires a nonempty fixed independent-root census")
    return np.random.Generator(np.random.PCG64(seed)).integers(0, roots, size=(draws, roots))


def nearest_rank_interval(values: np.ndarray, tail: float) -> tuple[Decimal, Decimal]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < tail < 0.5:
        raise ValueError("interval requires every fixed finite resample and declared tail")
    ordered = np.sort(values)
    endpoints = tuple(
        Decimal(repr(float(ordered[max(0, int(np.ceil(p * len(ordered))) - 1)])))
        for p in (tail, 1 - tail)
    )
    return endpoints[0], endpoints[1]
