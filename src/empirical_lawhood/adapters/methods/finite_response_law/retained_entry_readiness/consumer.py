"""Finite scalar-entry service, preserving the inherited native receiver."""

from typing import Any

import numpy as np
from ..fitting import DELTA
from ..science import FiniteResponseLawScienceSpec
from ..intervals import oriented_interval


def decisions(mean: Any, width: Any, entry: Any, direction: Any, requirement: Any) -> Any:
    n, schedules = mean.shape[:2]
    if width.shape != mean.shape or entry.shape != (n, schedules):
        raise ValueError("Service axes differ")
    lo, hi = mean - width, mean + width
    lo[..., 2:] = np.maximum(lo[..., 2:], 0)
    hi[..., 2:] = np.maximum(hi[..., 2:], 0)
    spec = FiniteResponseLawScienceSpec()
    caps = np.tile(np.asarray(spec.preservation, dtype=float), 2)
    eligible = np.zeros((n, schedules, 256, 2, 8), dtype=bool)
    precise = np.isfinite(width).all(axis=-1) & (width <= DELTA).all(axis=-1)
    for word in range(8):
        low, high = oriented_interval(lo, hi, word)
        for c in range(2):
            axis = direction[:, :, c][:, None] // 2
            positive = direction[:, :, c][:, None] % 2 == 0
            a = np.where(axis == 0, low[:, :, None, 0], low[:, :, None, 1])
            b = np.where(axis == 0, high[:, :, None, 0], high[:, :, None, 1])
            longitudinal_low = np.where(positive, a, -b)
            longitudinal_high = np.where(positive, b, -a)
            transverse = np.where(
                axis == 0,
                np.maximum(abs(low[:, :, None, 1]), abs(high[:, :, None, 1])),
                np.maximum(abs(low[:, :, None, 0]), abs(high[:, :, None, 0])),
            )
            eligible[..., c, word] = (
                entry[:, :, None]
                & precise[:, :, word // 2, None]
                & np.asarray((high[..., 2:] <= caps).all(axis=-1))[:, :, None]
                & (longitudinal_low >= requirement[:, None, :, c])
                & (longitudinal_high <= float(spec.upper[c]))
                & (transverse <= float(spec.transverse[c]))
            )
    choice = np.where(eligible.any(axis=-1), eligible.argmax(axis=-1), -1).astype(np.int64)
    return choice, eligible


def actual_success(y: Any, work: Any, direction: Any, requirement: Any) -> Any:
    """Invariant realized feasibility, computed once for every finite request/word."""
    spec = FiniteResponseLawScienceSpec()
    caps = np.tile(np.asarray(spec.preservation, dtype=float), 2)
    n, schedules = y.shape[:2]
    success = np.zeros((n, schedules, 256, 2, 8), dtype=bool)
    for word in range(8):
        values = y[:, :, word // 2]
        good = (
            np.isfinite(values).all(axis=(2, 3, 4))
            & (abs(values[..., 0] - values[..., 1]) <= (DELTA / 8)[None, None, :, None]).all(
                axis=(2, 3)
            )
            & (values[:, :, 2:] <= caps[None, None, :, None, None]).all(axis=(2, 3, 4))
            & np.isfinite(work).all(axis=-1)
            & (work <= float(spec.parent_work_maximum)).all(axis=-1)
        )
        response = values[:, :, :2] * (-1 if word % 2 == 0 else 1)
        for c in range(2):
            axis = direction[:, :, c][:, None, :, None, None] // 2
            polarity = np.where(direction[:, :, c] % 2 == 0, 1, -1)[:, None, :, None, None]
            longitudinal = polarity * np.where(
                axis == 0, response[:, :, None, 0], response[:, :, None, 1]
            )
            transverse = np.where(axis == 0, response[:, :, None, 1], response[:, :, None, 0])
            success[..., c, word] = (
                good[:, :, None]
                & (longitudinal >= requirement[:, None, :, c, None, None]).all(axis=(-1, -2))
                & (longitudinal <= float(spec.upper[c])).all(axis=(-1, -2))
                & (abs(transverse) <= float(spec.transverse[c])).all(axis=(-1, -2))
            )
    return success


def joint_success(choice: Any, actual: Any) -> Any:
    selected = np.take_along_axis(actual, np.maximum(choice, 0)[..., None], axis=-1)[..., 0]
    return ((choice >= 0) & selected).all(axis=-1)
