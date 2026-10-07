"""Exposed finite-menu oracle. It never supplies a causal consumer forecast."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .retained import FiniteResponseLawObservedPanel
from .science import FiniteResponseLawScienceSpec

REASONS = (
    "measurement_missing_or_invalid",
    "matched_hold_missing",
    "numerical_discrepancy",
    "parent_work",
    "branch_preservation",
    "below_requirement",
    "above_upper",
    "transverse",
)


@dataclass(frozen=True, slots=True)
class OracleResult:
    # root,parent,request,consumer,oriented word; word order magnitude,direction,sign(-,+).
    failures: NDArray[np.bool_]
    feasible: NDArray[np.bool_]
    selected: NDArray[np.int64]
    joint: NDArray[np.bool_]
    aligned_low: NDArray[np.bool_]
    aligned_high: NDArray[np.bool_]


def evaluate_oracle(
    panel: FiniteResponseLawObservedPanel,
    directions: NDArray[np.int64],
    lower: NDArray[np.float64],
    spec: FiniteResponseLawScienceSpec,
    *,
    futures: int = 1,
) -> OracleResult:
    if futures not in (1, 2) or directions.shape != (16, 256, 2) or lower.shape != directions.shape:
        raise ValueError("Oracle requires exactly 16 roots and the frozen request roster")
    if (
        directions.dtype != np.int64
        or np.any((directions < 0) | (directions > 3))
        or not np.isfinite(lower).all()
    ):
        raise ValueError("Invalid request operands")
    for c, (lo, hi) in enumerate(spec.lower_ranges):
        if np.any((lower[:, :, c] < float(lo)) | (lower[:, :, c] > float(hi))):
            raise ValueError("Request outside frozen native-unit range")
    # Compare all eight outputs for the chosen pair, not the entire unused menu.
    y = panel.y[:16, :, :, :, :futures, :]
    valid = panel.valid[:16, :, :, :, :futures, :].all(axis=(3, 4, 5))
    numerical = (
        np.abs(y[..., 0] - y[..., 1])
        <= np.asarray(spec.delta, dtype=float)[None, None, None, :, None] / 8
    ).all(axis=(3, 4))
    hold = panel.hold_observed[:16, :, :futures, :].all(axis=(2, 3))
    work = (
        np.isfinite(panel.parent_work[:16])
        & (panel.parent_work[:16] <= float(spec.parent_work_maximum))
    ).all(axis=2)
    preservation = (
        y[:, :, :, 2:]
        <= np.tile(np.asarray(spec.preservation, dtype=float), 2)[None, None, None, :, None, None]
    ).all(axis=(3, 4, 5))
    failures = np.empty((16, 5, 256, 2, 8, len(REASONS)), dtype=bool)
    for word in range(8):
        pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
        response = sign * y[:, :, pair, :2]
        for c in (0, 1):
            axis = directions[:, :, c] // 2
            polarity = np.where(directions[:, :, c] % 2 == 0, 1, -1)
            longitudinal = (
                np.where(
                    axis[:, None, :, None, None] == 0,
                    response[:, :, None, 0],
                    response[:, :, None, 1],
                )
                * polarity[:, None, :, None, None]
            )
            transverse = (
                np.where(
                    axis[:, None, :, None, None] == 0,
                    response[:, :, None, 1],
                    -response[:, :, None, 0],
                )
                * polarity[:, None, :, None, None]
            )
            target = failures[:, :, :, c, word]
            target[..., 0] = ~np.asarray(valid)[:, :, pair, None]
            target[..., 1] = ~np.asarray(hold)[:, :, None]
            target[..., 2] = ~np.asarray(numerical)[:, :, pair, None]
            target[..., 3] = ~np.asarray(work)[:, :, None]
            target[..., 4] = ~np.asarray(preservation)[:, :, pair, None]
            target[..., 5] = (longitudinal < lower[:, None, :, c, None, None]).any(axis=(3, 4))
            target[..., 6] = (longitudinal > float(spec.upper[c])).any(axis=(3, 4))
            target[..., 7] = (np.abs(transverse) > float(spec.transverse[c])).any(axis=(3, 4))
    feasible = np.asarray(~failures.any(axis=-1))
    # Existing word IDs sort negative before positive within each direction.
    selected = np.where(feasible.any(axis=-1), feasible.argmax(axis=-1), -1).astype(np.int64)
    joint = np.asarray((selected >= 0).all(axis=-1))
    # Only the same request-aligned direction/sign may supply discrimination.
    aligned_word = (directions // 2) * 2 + np.where(directions % 2 == 0, 1, 0)
    aligned_low = np.take_along_axis(
        feasible, np.broadcast_to(aligned_word[:, None, :, :, None], (16, 5, 256, 2, 1)), axis=4
    )[..., 0]
    aligned_high = np.take_along_axis(
        feasible,
        np.broadcast_to((aligned_word + 4)[:, None, :, :, None], (16, 5, 256, 2, 1)),
        axis=4,
    )[..., 0]
    return OracleResult(failures, feasible, selected, joint, aligned_low, aligned_high)
