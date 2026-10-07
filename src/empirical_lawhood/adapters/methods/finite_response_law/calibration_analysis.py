"Frozen finite response-law qualification score and request-census arithmetic, not law qualification.\n\nAll 32 assigned roots remain in the denominator. No fitted coefficient, scale,\nsupport threshold, action word or response requirement is selected here.\n"

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .assigned_prediction import AssignedPrediction
from .calibration_panel import FIXED_CALIBRATION_ROOT_IDS, Array, CalibrationPanel, valid_calibration_root_ids
from .intervals import Decisions, choose, quantile, scores, NU
from .science import FiniteResponseLawScienceSpec, seed_for


def calibration_requests(
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    *, request_seeds: tuple[int, ...] = (),
) -> tuple[NDArray[np.int64], Array]:
    """Outcome-independent fixed synthetic benchmark, 256 A/B pairs per root."""
    if not valid_calibration_root_ids(root_ids):
        raise ValueError("Calibration requests require one exact physical root cohort")
    if request_seeds and len(request_seeds) != len(root_ids):
        raise ValueError("Calibration requests require every committed scientific seed")
    spec = FiniteResponseLawScienceSpec()
    direction = np.empty((32, 256, 2), dtype=np.int64)
    requirement = np.empty(direction.shape, dtype=np.float64)
    for r, root_id in enumerate(root_ids):
        rng = np.random.Generator(np.random.PCG64(seed_for("calibration-request", root_id, committed_seed=request_seeds[r] if request_seeds else None)))
        direction[r] = rng.integers(0, 4, size=(256, 2))
        for consumer, (low, high) in enumerate(spec.lower_ranges):
            requirement[r, :, consumer] = rng.uniform(float(low), float(high), 256)
    return direction, requirement


def _validate_prediction(prediction: AssignedPrediction) -> None:
    if (
        prediction.mean.shape != (32, 4, 8)
        or prediction.mean.dtype != np.float64
        or prediction.sigma.shape != prediction.mean.shape
        or prediction.sigma.dtype != np.float64
        or prediction.supported.shape != (32,)
        or prediction.supported.dtype != np.bool_
        or np.any(np.isfinite(prediction.sigma) & (prediction.sigma <= 0))
    ):
        raise ValueError(
            "Calibration prediction changes its frozen assigned-parent operands"
        )


@dataclass(frozen=True, slots=True)
class CalibrationScores:
    root_scores: Array
    measured: NDArray[np.bool_]
    numerical: NDArray[np.bool_]
    rank: int
    q: float


def calibration_scores(
    panel: CalibrationPanel, prediction: AssignedPrediction
) -> CalibrationScores:
    """Apply the existing residual max to one acquired parent, including infinity."""
    _validate_prediction(prediction)
    values, measured, numerical = scores(
        panel,
        np.arange(32, dtype=np.int64),
        prediction.mean[:, None],
        prediction.sigma[:, None],
        prediction.supported[:, None],
    )
    roots = values[:, 0]
    rank, q = quantile(roots)
    return CalibrationScores(roots, measured[:, 0], numerical[:, 0], rank, q)


def calibration_decisions(
    prediction: AssignedPrediction,
    q: float,
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    *, request_seeds: tuple[int, ...] = (),
) -> Decisions:
    """Reuse frozen finite selection; unavailable predictions remain nonadmissions.

    Computational copies preserve the selector's five-slot layout. Only the
    single acquired-parent slot is returned; copies are never observations.
    """
    _validate_prediction(prediction)
    if np.isnan(q) or q < 0:
        raise ValueError(
            "Calibration multiplier must be nonnegative, possibly infinite"
        )
    direction, requirement = calibration_requests(root_ids, request_seeds=request_seeds)
    finite = np.isfinite(prediction.mean).all(axis=(1, 2)) & np.isfinite(
        prediction.sigma
    ).all(axis=(1, 2))
    selected = np.full((32, 1, 256, 2), -1, dtype=np.int64)
    feasible = np.zeros((*selected.shape, 8), dtype=bool)
    failures = np.zeros((*feasible.shape, 6), dtype=bool)
    failures[..., 0] = True
    halfwidth = q * prediction.sigma[:, None] + NU
    if finite.any():
        n = int(finite.sum())
        result = choose(
            np.broadcast_to(prediction.mean[finite, None], (n, 5, 4, 8)),
            np.broadcast_to(prediction.sigma[finite, None], (n, 5, 4, 8)),
            q,
            np.broadcast_to(prediction.supported[finite, None], (n, 5)),
            direction[finite],
            requirement[finite],
        )
        selected[finite] = result.selected[:, :1]
        feasible[finite] = result.feasible[:, :1]
        failures[finite] = result.failures[:, :1]
    return Decisions(selected, feasible, failures, halfwidth)
