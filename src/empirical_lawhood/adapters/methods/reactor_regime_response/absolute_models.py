"""Separate empirical zero-feed peak model with whole-root weighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Fit, LAMBDAS, RBF_MULTIPLIERS


@dataclass(frozen=True)
class AbsoluteRows:
    features: np.ndarray
    peaks_K: np.ndarray
    roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.features.ndim != 2
            or self.features.shape[0] != len(self.roots)
            or self.peaks_K.shape != (len(self.roots),)
            or not np.isfinite(self.features).all()
            or not np.isfinite(self.peaks_K).all()
        ):
            raise ValueError("absolute baseline rows are incomplete or nonfinite")

    @property
    def weights(self) -> np.ndarray:
        counts = {root: self.roots.count(root) for root in set(self.roots)}
        return np.asarray([1 / counts[root] for root in self.roots], dtype=np.float64)

    def root_losses(self, prediction: np.ndarray) -> dict[str, float]:
        if prediction.shape != self.peaks_K.shape or not np.isfinite(prediction).all():
            raise ValueError("absolute prediction shape or finiteness differs")
        return {
            root: float(np.mean((prediction[[i for i, r in enumerate(self.roots) if r == root]]
                                  - self.peaks_K[[i for i, r in enumerate(self.roots) if r == root]]) ** 2))
            for root in sorted(set(self.roots))
        }


def fit_absolute(rows: AbsoluteRows, kind: str, penalty: float, multiplier: float | None) -> Fit:
    'Same source roster as coefficient fitting, with independent absolute-observable targets.'
    if kind not in ("affine", "rbf") or penalty not in LAMBDAS:
        raise ValueError("absolute baseline requires a declared S specification")
    if not len(rows.roots):
        raise ValueError("absolute baseline has no fit roots")
    weights = rows.weights
    normalized = weights / np.sum(weights)
    mean = np.sum(rows.features * normalized[:, None], axis=0)
    scale = np.maximum(
        np.sqrt(np.sum((rows.features - mean) ** 2 * normalized[:, None], axis=0)), 1e-12
    )
    z = (rows.features - mean) / scale
    if kind == "affine":
        if multiplier is not None:
            raise ValueError("affine baseline has no RBF length multiplier")
        matrix = np.column_stack((np.ones(len(z)), z))
        regularizer = np.eye(matrix.shape[1]) * penalty
        regularizer[0, 0] = 0
        operator = np.linalg.solve(
            matrix.T @ (matrix * weights[:, None]) + regularizer,
            matrix.T @ (weights * rows.peaks_K),
        )
        return Fit("affine", mean, scale, operator, penalty)
    if multiplier not in RBF_MULTIPLIERS:
        raise ValueError("undeclared baseline RBF length multiplier")
    distance2 = np.maximum(
        np.sum(z * z, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2 * z @ z.T,
        0,
    )
    distances = np.sqrt(distance2[np.triu_indices(len(z), 1)])
    positive = distances[distances > 1e-12]
    if not len(positive):
        raise ValueError("absolute RBF has no positive fit-pair distance")
    length = float(multiplier * np.median(positive))
    kernel = np.exp(-distance2 / (2 * length**2))
    alpha = np.linalg.solve(kernel + np.diag(penalty / weights), rows.peaks_K)
    return Fit("rbf", mean, scale, alpha, penalty, length, z)
