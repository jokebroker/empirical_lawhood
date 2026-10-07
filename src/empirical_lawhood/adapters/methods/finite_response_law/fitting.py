"Frozen primary-view, masked whole-root affine fitting recipe.\n\nPure exposed-development calculations. These objects are not qualified laws.\n"

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator
from .retained import FiniteResponseLawObservedPanel
from .science import FiniteResponseLawScienceSpec

Array = NDArray[np.float64]
DELTA = np.asarray(FiniteResponseLawScienceSpec().delta, dtype=float)
FAMILIES = (("SNAPSHOT", 8), ("SNAPSHOT_AND_REFERENCE_SKETCH", 16), ("SNAPSHOT_REFERENCE_SKETCH_AND_RATE", 24))
BOUNDARIES = ("lower", "composed", "cached", "direct")


@dataclass(frozen=True)
class Features:
    x: Array  # root, 24 native channels, view
    z: Array  # root, parent, 24 native channels, view
    root_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.root_ids)
        if self.x.shape != (n, 24, 2) or self.z.shape != (n, 5, 24, 2):
            raise ValueError("Feature axes differ from the declared native instrument")
        if (
            len(set(self.root_ids)) != n
            or not np.isfinite(self.x).all()
            or not np.isfinite(self.z).all()
        ):
            raise ValueError("Missing/invalid authenticated feature input; do not impute")


@dataclass(frozen=True)
class Normalizer:
    center: Array
    scale: Array

    @classmethod
    def fit(cls, values: Array) -> "Normalizer":
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Empty/nonfinite normalizer population")
        center = values.mean(axis=0)
        return cls(center, np.maximum(np.sqrt(np.mean((values - center) ** 2, axis=0)), 1e-8))

    def apply(self, values: Array) -> Array:
        return (values - self.center) / self.scale


@dataclass(frozen=True)
class Targets:
    mean: Array  # root,parent,32; unavailable cells remain nan
    eligible: NDArray[np.bool_]  # root,32; complete five-parent primary futures


def targets(panel: FiniteResponseLawObservedPanel) -> Targets:
    valid = panel.observed[..., 0] & panel.valid[..., 0] & np.isfinite(panel.y[..., 0])
    eligible = valid.all(axis=(1, 4)).reshape(len(panel.root_ids), 32)
    mean = panel.y[..., 0].mean(axis=-1).reshape(len(panel.root_ids), 5, 32)
    return Targets(np.where(eligible[:, None, :], mean, np.nan), eligible)


def clamp(mean: Array) -> Array:
    result = mean.copy()
    result[..., 2:] = np.maximum(result[..., 2:], 0)
    return result


@dataclass(frozen=True)
class Recipe:
    family: str
    dimension: int
    ridge: float
    gamma: float

    def __post_init__(self) -> None:
        if (
            (self.family, self.dimension) not in FAMILIES
            or self.ridge not in (10.0, 1.0, 0.1)
            or self.gamma not in (0.5, 1.0)
        ):
            raise ValueError("Unknown feature version or unbound fitting recipe")


@dataclass(frozen=True)
class PointFit:
    roots: tuple[int, ...]
    recipe: Recipe
    direct_recipe: Recipe
    n0: Normalizer
    nh: Normalizer
    lower: Array
    lower_mean: Array
    upper: Array
    cached: Array
    direct: Array

    def predict(
        self, features: Features, roots: NDArray[np.int64]
    ) -> tuple[dict[str, Array], dict[str, NDArray[np.bool_]], Array]:
        d = self.recipe.dimension
        x = self.n0.apply(features.x[roots, :d, 0])
        z = features.z[roots, :, :d, 0]
        uz = np.stack([affine_prediction(self.upper[p], x) for p in range(5)], axis=1)
        nz, nu = self.nh.apply(z), self.nh.apply(uz)
        lower = affine_prediction(self.lower, nz.reshape(-1, d)).reshape(-1, 5, 4, 8)
        composed = affine_prediction(self.lower, nu.reshape(-1, d)).reshape(-1, 5, 4, 8)
        direct = np.stack([affine_prediction(self.direct[p], x) for p in range(5)], axis=1).reshape(
            -1, 5, 4, 8
        )
        cached = np.broadcast_to(self.cached.reshape(5, 4, 8), lower.shape).copy()
        prefix_support = (np.isfinite(x).all(axis=1) & (np.abs(x).max(axis=1) <= 6))[:, None]
        supports = {
            "lower": np.isfinite(nz).all(axis=2) & (np.abs(nz).max(axis=2) <= 6),
            "composed": prefix_support
            & np.isfinite(nu).all(axis=2)
            & (np.abs(nu).max(axis=2) <= 6),
            "direct": np.broadcast_to(prefix_support, (len(roots), 5)).copy(),
            "cached": np.ones((len(roots), 5), dtype=bool),
        }
        return (
            {
                name: clamp(a)
                for name, a in zip(BOUNDARIES, (lower, composed, cached, direct), strict=True)
            },
            supports,
            uz,
        )

    def arrays(self) -> dict[str, Array]:
        return {
            "n0_center": self.n0.center,
            "n0_scale": self.n0.scale,
            "nh_center": self.nh.center,
            "nh_scale": self.nh.scale,
            "lower": self.lower,
            "lower_mean": self.lower_mean,
            "upper": self.upper,
            "cached": self.cached,
            "direct": self.direct,
        }


def shrunk_fit(x: Array, y: Array, ridge: float, gamma: float) -> tuple[Array, Array]:
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("Required output fit has no valid native targets")
    mean = y.mean(axis=0)
    operator = fit_affine_operator(x, y, ridge=ridge) * gamma
    operator[-1] += (1 - gamma) * mean
    return operator, mean


def fit_points(
    features: Features,
    target: Targets,
    roots: NDArray[np.int64],
    recipe: Recipe,
    direct_recipe: Recipe,
) -> PointFit:
    if recipe.dimension != direct_recipe.dimension or recipe.family != direct_recipe.family:
        raise ValueError("Direct comparator must use the selected lower family")
    d = recipe.dimension
    x, z = features.x[roots, :d, 0], features.z[roots, :, :d, 0]
    n0, nh = Normalizer.fit(x), Normalizer.fit(z.reshape(-1, d))
    nx, nz = n0.apply(x), nh.apply(z)
    lower, lower_mean = np.empty((d + 1, 32)), np.empty(32)
    direct, cached = np.empty((5, d + 1, 32)), np.empty((5, 32))
    for j in range(32):
        good = target.eligible[roots, j]
        y = target.mean[roots[good], :, j]
        op, mean = shrunk_fit(nz[good].reshape(-1, d), y.reshape(-1, 1), recipe.ridge, recipe.gamma)
        lower[:, j], lower_mean[j] = op[:, 0], mean[0]
        for p in range(5):
            op, mean = shrunk_fit(nx[good], y[:, p, None], direct_recipe.ridge, direct_recipe.gamma)
            direct[p, :, j], cached[p, j] = op[:, 0], mean[0]
    upper = np.stack([fit_affine_operator(nx, z[:, p], ridge=10) for p in range(5)])
    return PointFit(
        tuple(map(int, roots)),
        recipe,
        direct_recipe,
        n0,
        nh,
        lower,
        lower_mean,
        upper,
        cached,
        direct,
    )


def cohort_folds(roots: NDArray[np.int64], count: int = 3) -> NDArray[np.int64]:
    folds = np.empty(len(roots), dtype=np.int64)
    for cohort in (roots < 16, roots >= 16):
        positions = np.flatnonzero(cohort)
        ranked = positions[np.argsort(roots[positions])]
        folds[ranked] = np.arange(len(ranked)) % count
    return folds


def selection_loss(
    prediction: Array, target: Targets, roots: NDArray[np.int64]
) -> tuple[float, Array]:
    error = ((prediction.reshape(-1, 5, 32) - target.mean[roots]) / np.tile(DELTA, 4)) ** 2
    losses = np.empty(len(roots))
    for i, root in enumerate(roots):
        mask = target.eligible[root]
        if not mask.any():
            raise ValueError("Declared comparison root has no eligible measured outputs")
        values = error[i][:, mask]
        losses[i] = float(values.mean()) if np.isfinite(values).all() else np.inf
    return float(losses.mean()), losses


def coefficient_oof(
    features: Features,
    target: Targets,
    roots: NDArray[np.int64],
    recipe: Recipe,
    direct_recipe: Recipe,
) -> tuple[dict[str, Array], dict[str, NDArray[np.bool_]], list[PointFit]]:
    folds = cohort_folds(roots)
    predictions = {b: np.full((len(roots), 5, 4, 8), np.nan) for b in BOUNDARIES}
    support = {b: np.zeros((len(roots), 5), dtype=bool) for b in BOUNDARIES}
    fits = []
    for fold in range(3):
        model = fit_points(features, target, roots[folds != fold], recipe, direct_recipe)
        pred, sup, _ = model.predict(features, roots[folds == fold])
        fits.append(model)
        for b in BOUNDARIES:
            predictions[b][folds == fold], support[b][folds == fold] = pred[b], sup[b]
    return predictions, support, fits


def nominate(
    features: Features, target: Targets, roots: NDArray[np.int64]
) -> tuple[Recipe, Recipe, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    winner: Recipe | None = None
    best = float("inf")
    # This iteration order is exactly the predeclared tolerance tie order.
    for name, d in FAMILIES:
        for ridge in (10.0, 1.0, 0.1):
            for gamma in (0.5, 1.0):
                recipe = Recipe(name, d, ridge, gamma)
                prediction, _, _ = coefficient_oof(features, target, roots, recipe, recipe)
                loss, per_root = selection_loss(prediction["lower"], target, roots)
                reports.append(
                    {
                        "boundary": "lower",
                        "recipe": recipe.__dict__,
                        "loss": loss,
                        "per_root": per_root.tolist(),
                    }
                )
                if winner is None or loss < best - 1e-12:
                    winner, best = recipe, loss
    assert winner is not None
    direct: Recipe | None = None
    best = float("inf")
    for ridge in (10.0, 1.0, 0.1):
        for gamma in (0.5, 1.0):
            recipe = Recipe(winner.family, winner.dimension, ridge, gamma)
            prediction, _, _ = coefficient_oof(features, target, roots, winner, recipe)
            loss, per_root = selection_loss(prediction["direct"], target, roots)
            reports.append(
                {
                    "boundary": "direct",
                    "recipe": recipe.__dict__,
                    "loss": loss,
                    "per_root": per_root.tolist(),
                }
            )
            if direct is None or loss < best - 1e-12:
                direct, best = recipe, loss
    assert direct is not None
    if not all(np.isfinite(r["loss"]) for r in reports):
        # Missing required fits are not a manufactured finite zero competitor.
        raise ValueError("Unavailable/nonfinite declared development candidate")
    return winner, direct, reports
