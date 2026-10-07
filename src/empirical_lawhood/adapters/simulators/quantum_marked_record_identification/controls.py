"""Frozen adverse, thinning, shuffle, timing, and symmetry controls."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .contracts import Panel
from .receiver import crossfit_site_probabilities, site_counts
from .source import EventRecord


def _control_rng(scientific_seed: int) -> np.random.Generator:
    if type(scientific_seed) is not int or not 0 <= scientific_seed < 2**128:
        raise ValueError("control scientific seed must be an explicit 128-bit unsigned integer")
    return np.random.Generator(np.random.Philox(scientific_seed))


def validate_control_seed_census(
    unit_ids: Sequence[str],
    scientific_control_seeds: tuple[tuple[str, int, int], ...],
    *,
    allow_resampling: bool = False,
) -> None:
    """Bind mark-reassignment and shuffle seeds in the exact parent order.

    Caller labels never choose draws. Preserving an existing allocation requires
    reviewed numeric inputs exported from its original source and its own custody.
    A new census grants no historical qualification, receipt or authority.
    """
    rows = scientific_control_seeds
    if (
        not isinstance(rows, tuple)
        or len(rows) != len(unit_ids)
        or any(not isinstance(row, tuple) or len(row) != 3 for row in rows)
        or tuple(unit for unit, _, _ in rows) != tuple(unit_ids)
        or any(not isinstance(unit, str) or not unit for unit, _, _ in rows)
        or any(type(seed) is not int or not 0 <= seed < 2**128 for _, mark, shuffle in rows for seed in (mark, shuffle))
    ):
        raise ValueError("control scientific seed census differs from the ordered parent roster")
    unique: dict[str, tuple[int, int]] = {}
    for unit, mark, shuffle in rows:
        if unit in unique and (not allow_resampling or unique[unit] != (mark, shuffle)):
            raise ValueError("control scientific seed census repeats or changes a parent allocation")
        unique[unit] = (mark, shuffle)
    if len({seed for seeds in unique.values() for seed in seeds}) != 2 * len(unique):
        raise ValueError("control scientific seed census reuses independent purpose allocations")


def burst_baseline(
    events: Sequence[EventRecord],
    *,
    endpoint: float = 200.0,
    gamma: float = 1.0,
    particles: int = 6,
) -> float:
    """Consecutive same-half short-lag statistic with the fixed cutoff."""

    selected = [event for event in events if endpoint - 4.0 <= event.event_time < endpoint]
    threshold = math.log(2.0) / (gamma * particles)
    return float(
        sum(
            left.site < 6 and right.site < 6 and right.event_time - left.event_time <= threshold
            for left, right in zip(selected, selected[1:], strict=False)
        )
    )


def prehistory_site_probabilities(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
    folds: Sequence[int],
) -> NDArray[np.float64]:
    counts = np.asarray(
        [site_counts(events, start_time=180.0, end_time=200.0) for events in records],
        dtype=np.float64,
    )
    return crossfit_site_probabilities(counts, k_values, folds)


def thin_history(
    events: Sequence[EventRecord],
    probabilities: Sequence[float],
    *,
    scientific_seed: int,
) -> tuple[EventRecord, ...]:
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (12,) or np.any(probability < 0):
        raise ValueError("history-thinning site law is invalid")
    probability = probability / float(probability.sum())
    rng = _control_rng(scientific_seed)
    sites = rng.choice(12, size=len(events), p=probability)
    return tuple(
        EventRecord(event.event_index, event.event_time, int(site))
        for event, site in zip(events, sites, strict=True)
    )


def shuffle_history(
    events: Sequence[EventRecord],
    *,
    scientific_seed: int,
) -> tuple[EventRecord, ...]:
    rng = _control_rng(scientific_seed)
    sites = np.asarray([event.site for event in events], dtype=np.int64)
    sites = sites[rng.permutation(len(sites))]
    return tuple(
        EventRecord(event.event_index, event.event_time, int(site))
        for event, site in zip(events, sites, strict=True)
    )


def reflect_boundaries(events: Sequence[EventRecord]) -> tuple[EventRecord, ...]:
    """Reflect within both halves, swapping the two receiver boundaries."""

    return tuple(
        EventRecord(event.event_index, event.event_time, (5 - event.site) % 12) for event in events
    )


def active_controls(panel: Panel) -> tuple[str, ...]:
    controls = ["prehistory_thinning", "wrong_time"]
    if "USES_ORDER" in panel.semantic_flags:
        controls.append("history_shuffle")
    if "USES_BOUNDARY_DIRECTION" in panel.semantic_flags:
        controls.append("boundary_reflection")
    if "USES_FITTED_INTENSITY" in panel.semantic_flags:
        controls.append("intensity_calibration")
    return tuple(controls)


def symmetry_feature_map(values: Sequence[float]) -> NDArray[np.float64]:
    """Map the frozen 36-vector under within-half boundary reflection.

    This map is used as an independent recomputation sentinel.  Fourier
    quadratic coordinates are not asserted individually even/odd because
    reflection mixes sine and cosine in the chosen real basis.
    """

    vector = np.asarray(values, dtype=np.float64).copy()
    if vector.shape != (36,):
        raise ValueError("symmetry map requires the complete feature vector")
    # q_boundary_56 <-> q_boundary_110
    vector[18], vector[19] = vector[19], vector[18]
    # pair_56_oriented <-> pair_110_oriented; directed imbalance is odd.
    vector[28], vector[29] = vector[29], vector[28]
    vector[30] *= -1.0
    return vector


__all__ = [
    "validate_control_seed_census",
    "active_controls",
    "burst_baseline",
    "prehistory_site_probabilities",
    "reflect_boundaries",
    "shuffle_history",
    "symmetry_feature_map",
    "thin_history",
]
