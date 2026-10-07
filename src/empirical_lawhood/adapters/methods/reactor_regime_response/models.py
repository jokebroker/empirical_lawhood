"Finite empirical coefficient models for the reactor regime-response study.\n\nRows carry one root-balanced weight and two actual delivered masses.  The\nthrough-origin coefficient target is a summary of those same two contrasts;\nordinary ranking always returns to action-contrast loss in kelvin squared.\n"

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


LAMBDAS = (1e-6, 1e-3, 1e-1)
RBF_MULTIPLIERS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CoefficientRows:
    features: np.ndarray
    masses: np.ndarray
    contrasts: np.ndarray
    roots: tuple[str, ...]
    root_multiplicity: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.roots)
        if (
            self.features.ndim != 2
            or self.features.shape[0] != n
            or self.masses.shape != (n, 2)
            or self.contrasts.shape != (n, 2)
            or not np.isfinite(self.features).all()
            or not np.isfinite(self.masses).all()
            or not np.isfinite(self.contrasts).all()
            or np.any(self.masses <= 0)
            or np.any(self.masses[:, 1] <= self.masses[:, 0])
            or (self.root_multiplicity and (
                len(self.root_multiplicity) != n
                or any(type(v) is not int or v < 1 for v in self.root_multiplicity)
                or any(len({v for r, v in zip(self.roots, self.root_multiplicity, strict=True)
                            if r == root}) != 1 for root in set(self.roots))
            ))
        ):
            raise ValueError("coefficient rows lack a finite distinct action chart")

    @property
    def coefficient(self) -> np.ndarray:
        return np.asarray(
            np.sum(self.masses * self.contrasts, axis=1) / np.sum(self.masses**2, axis=1),
            dtype=np.float64,
        )

    @property
    def context_weight(self) -> np.ndarray:
        """Equal physical-root weight, with explicit bootstrap multiplicity."""
        counts = {root: self.roots.count(root) for root in set(self.roots)}
        return np.asarray(
            np.asarray(self.root_multiplicity or (1,) * len(self.roots))
            / np.asarray([counts[r] for r in self.roots]),
            dtype=np.float64,
        )

    @property
    def fit_weight(self) -> np.ndarray:
        return np.asarray(self.context_weight * np.sum(self.masses**2, axis=1) / 2,
                          dtype=np.float64)

    def average_root_loss(self, losses: dict[str, float]) -> float:
        multiplicity = dict(zip(self.roots, self.root_multiplicity or (1,) * len(self.roots), strict=True))
        return float(np.average(tuple(losses.values()),
                                weights=[multiplicity[root] for root in losses]))

    def resample_roots(self, drawn: tuple[str, ...]) -> CoefficientRows:
        """Reweight drawn roots without inflating distinct physical contact."""
        if not drawn or not set(drawn) <= set(self.roots):
            raise ValueError("bootstrap draw contains no roots or unknown roots")
        indices = np.asarray([i for i, root in enumerate(self.roots) if root in drawn])
        selected = self.subset(indices)
        return CoefficientRows(selected.features, selected.masses, selected.contrasts,
                               selected.roots, tuple(drawn.count(root) for root in selected.roots))

    def subset(self, indices: np.ndarray) -> CoefficientRows:
        return CoefficientRows(
            self.features[indices],
            self.masses[indices],
            self.contrasts[indices],
            tuple(self.roots[int(i)] for i in indices),
            tuple(self.root_multiplicity[int(i)] for i in indices) if self.root_multiplicity else (),
        )

    def root_losses(self, predictions: np.ndarray) -> dict[str, float]:
        if predictions.shape != (len(self.roots),) or not np.isfinite(predictions).all():
            raise ValueError("coefficient prediction shape or finiteness differs")
        loss = np.mean((predictions[:, None] * self.masses - self.contrasts) ** 2, axis=1)
        return {
            root: float(np.mean(loss[[i for i, r in enumerate(self.roots) if r == root]]))
            for root in sorted(set(self.roots))
        }


@dataclass(frozen=True)
class LocalLeaf:
    path: tuple[tuple[int, float, bool], ...]
    operator: np.ndarray


@dataclass(frozen=True)
class Fit:
    kind: Literal["K", "affine", "rbf", "local"]
    mean: np.ndarray
    scale: np.ndarray
    operator: np.ndarray
    penalty: float
    length: float | None = None
    training: np.ndarray | None = None
    leaves: tuple[LocalLeaf, ...] = ()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("coefficient feature shape differs")
        z = (features - self.mean) / self.scale
        if self.kind == "K":
            return np.asarray(np.full(len(z), self.operator[0]), dtype=np.float64)
        if self.kind == "affine":
            return np.asarray(
                np.column_stack((np.ones(len(z)), z)) @ self.operator, dtype=np.float64
            )
        if self.kind == "rbf":
            if self.training is None or self.length is None:
                raise ValueError("RBF fit is incomplete")
            return np.asarray(
                np.exp(-_squared_distances(z, self.training) / (2 * self.length**2))
                @ self.operator,
                dtype=np.float64,
            )
        values = np.empty(len(z), dtype=np.float64)
        for index, row in enumerate(features):
            matching = [
                leaf
                for leaf in self.leaves
                if all(
                    (row[coordinate] <= threshold) == left
                    for coordinate, threshold, left in leaf.path
                )
            ]
            if len(matching) != 1:
                raise ValueError("local partition does not uniquely cover the feature row")
            values[index] = np.dot(np.r_[1.0, z[index]], matching[0].operator)
        return values


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.maximum(
            np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T,
            0,
        ),
        dtype=np.float64,
    )


def _standardize(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = weights / np.sum(weights)
    mean = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * w[:, None], axis=0))
    scale = np.maximum(scale, 1e-12)
    return mean, scale, (x - mean) / scale


def _ridge(z: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float) -> np.ndarray:
    matrix = np.column_stack((np.ones(len(z)), z))
    regularizer = np.eye(matrix.shape[1]) * penalty
    regularizer[0, 0] = 0
    gram = matrix.T @ (matrix * weights[:, None]) + regularizer
    return np.linalg.solve(gram, matrix.T @ (weights * y))


def fit_constant(rows: CoefficientRows) -> Fit:
    weights = rows.fit_weight
    value = float(np.average(rows.coefficient, weights=weights))
    return Fit(
        "K",
        np.zeros(rows.features.shape[1]),
        np.ones(rows.features.shape[1]),
        np.asarray((value,)),
        0,
    )


def fit_affine(rows: CoefficientRows, penalty: float) -> Fit:
    if penalty not in LAMBDAS:
        raise ValueError("undeclared affine ridge penalty")
    mean, scale, z = _standardize(rows.features, rows.context_weight)
    return Fit(
        "affine", mean, scale, _ridge(z, rows.coefficient, rows.fit_weight, penalty), penalty
    )


def fit_rbf(rows: CoefficientRows, penalty: float, multiplier: float) -> Fit:
    if penalty not in LAMBDAS or multiplier not in RBF_MULTIPLIERS:
        raise ValueError("undeclared RBF hyperparameter")
    mean, scale, z = _standardize(rows.features, rows.context_weight)
    dist = np.sqrt(_squared_distances(z, z))
    positive = dist[np.triu_indices(len(z), 1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        raise ValueError("RBF has no positive fit-pair distance")
    length = float(multiplier * np.median(positive))
    kernel = np.exp(-(dist**2) / (2 * length**2))
    weights = rows.fit_weight
    alpha = np.linalg.solve(kernel + np.diag(penalty / weights), rows.coefficient)
    return Fit("rbf", mean, scale, alpha, penalty, length, z)


def _thresholds(column: np.ndarray) -> tuple[float, ...]:
    values = np.unique(column)
    if len(values) < 2:
        return ()
    result = set()
    for quantile in (0.25, 0.5, 0.75):
        index = min(len(values) - 2, max(0, int(np.floor(quantile * (len(values) - 1)))))
        result.add(float((values[index] + values[index + 1]) / 2))
    return tuple(sorted(result))


def split_stable(features: np.ndarray, coordinate: int, threshold: float) -> bool:
    column = features[:, coordinate]
    base = column <= threshold
    return all(
        float(np.mean(base != ((column + perturbation) <= threshold))) <= 0.01
        for perturbation in (-1e-8, 1e-8)
    )


def fit_local(rows: CoefficientRows, penalty: float) -> Fit:
    """Greedy root-contacted, stable tree with at most three affine leaves."""
    if penalty not in LAMBDAS:
        raise ValueError("undeclared local ridge penalty")
    mean, scale, z = _standardize(rows.features, rows.context_weight)
    y, w = rows.coefficient, rows.fit_weight
    matrix = np.column_stack((np.ones(len(z)), z))
    root_ids = np.asarray(rows.roots)

    def leaf_fit(mask: np.ndarray) -> tuple[np.ndarray, float]:
        operator = _ridge(z[mask], y[mask], w[mask], penalty)
        prediction = matrix[mask] @ operator
        action_error = prediction[:, None] * rows.masses[mask] - rows.contrasts[mask]
        # Contribution to the complete root-balanced action loss, including
        # non-scalar residuals. Absolute gain retains K² units.
        loss = float(np.sum(rows.context_weight[mask] * np.mean(action_error ** 2, axis=1))
                     / np.sum(rows.context_weight))
        return operator, loss

    whole = np.ones(len(y), dtype=bool)
    parent_operator, parent_loss = leaf_fit(whole)
    leaves: dict[tuple[tuple[int, float, bool], ...], tuple[np.ndarray, np.ndarray, float]] = {
        (): (whole, parent_operator, parent_loss)
    }
    for _ in range(2):
        best: (
            tuple[
                float,
                int,
                float,
                tuple[tuple[int, float, bool], ...],
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                float,
            ]
            | None
        ) = None
        for path, (mask, _, old_loss) in leaves.items():
            for coordinate in range(rows.features.shape[1]):
                for threshold in _thresholds(rows.features[mask, coordinate]):
                    if not split_stable(rows.features[mask], coordinate, threshold):
                        continue
                    left = mask & (rows.features[:, coordinate] <= threshold)
                    right = mask & ~left
                    if len(set(root_ids[left])) < 12 or len(set(root_ids[right])) < 12:
                        continue
                    a, loss_a = leaf_fit(left)
                    b, loss_b = leaf_fit(right)
                    gain = old_loss - loss_a - loss_b
                    parent_rows = rows.subset(np.where(mask)[0])
                    old_operator = leaves[path][1]
                    old_parent = parent_rows.average_root_loss(parent_rows.root_losses(matrix[mask] @ old_operator))
                    prediction = np.empty(int(mask.sum()))
                    inside_left = rows.features[mask, coordinate] <= threshold
                    prediction[inside_left] = matrix[left] @ a
                    prediction[~inside_left] = matrix[right] @ b
                    parent_loss = parent_rows.average_root_loss(parent_rows.root_losses(prediction))
                    if gain <= 0 or old_parent - parent_loss <= 1e-12 or old_parent - parent_loss < .1 * old_parent:
                        continue
                    candidate = (
                        gain,
                        coordinate,
                        threshold,
                        path,
                        left,
                        right,
                        a,
                        b,
                        loss_a,
                        loss_b,
                    )
                    if best is None or (-candidate[0], candidate[1], candidate[2], candidate[3]) < (
                        -best[0],
                        best[1],
                        best[2],
                        best[3],
                    ):
                        best = candidate
        if best is None:
            break
        _, coordinate, threshold, path, left, right, a, b, loss_a, loss_b = best
        del leaves[path]
        leaves[path + ((coordinate, threshold, True),)] = (left, a, loss_a)
        leaves[path + ((coordinate, threshold, False),)] = (right, b, loss_b)
    if len(leaves) == 1:
        raise ValueError("no stable local split reaches the declared gain")
    return Fit(
        "local",
        mean,
        scale,
        np.empty(0),
        penalty,
        leaves=tuple(
            LocalLeaf(path, operator) for path, (_, operator, _) in sorted(leaves.items())
        ),
    )
