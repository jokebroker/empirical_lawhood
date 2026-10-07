"""Frozen coefficient-OOF widths, support scores and native finite decisions."""

from dataclasses import dataclass
from math import ceil
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator
from .fitting import Array, DELTA, Features, PointFit
from .retained import FiniteResponseLawObservedPanel
from .science import FiniteResponseLawScienceSpec

NU = DELTA / 8


class ScoreObservations(Protocol):
    """Measured array operands; the score does not assume five acquired parents."""

    @property
    def y(self) -> Array: ...

    @property
    def observed(self) -> NDArray[np.bool_]: ...

    @property
    def valid(self) -> NDArray[np.bool_]: ...


def quantile(scores: Array) -> tuple[int, float]:
    if scores.ndim != 1 or np.isnan(scores).any() or (scores < 0).any():
        raise ValueError("Invalid root score census")
    rank = ceil((len(scores) + 1) * 0.9)
    return rank, float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


@dataclass(frozen=True)
class WidthFit:
    boundary: str
    base: Array
    multiplier: Array
    targets: Array
    eligible_roots: NDArray[np.bool_]
    base_counts: NDArray[np.int64]

    def sigma(
        self, features: Features, roots: NDArray[np.int64], points: PointFit
    ) -> tuple[Array, Array]:
        d = points.recipe.dimension
        if self.boundary == "lower":
            z = points.nh.apply(features.z[roots, :, :d, 0])
            logg = affine_prediction(self.multiplier, z.reshape(-1, d)).reshape(-1, 5)
        elif self.boundary == "cached":
            logg = np.broadcast_to(self.multiplier, (len(roots), 5))
        else:
            x = points.n0.apply(features.x[roots, :d, 0])
            logg = np.column_stack(
                [affine_prediction(self.multiplier[p], x)[:, 0] for p in range(5)]
            )
        g = np.exp(np.clip(logg, np.log(0.25), np.log(4)))
        return self.base[None, None] * g[:, :, None, None], g


def fit_widths(
    boundary: str,
    panel: FiniteResponseLawObservedPanel,
    features: Features,
    roots: NDArray[np.int64],
    points: PointFit,
    prediction: Array,
) -> WidthFit:
    y = panel.y[roots]
    measured = panel.observed[roots] & panel.valid[roots] & np.isfinite(y)
    residual = y - prediction[..., None, None]
    eligible = np.asarray(measured[..., 0].all(axis=(1, 4)))
    base = np.empty((4, 8))
    for k in range(4):
        for j in range(8):
            good = eligible[:, k, j]
            if not good.any():
                raise ValueError("Empty required coefficient-OOF scale population")
            base[k, j] = max(
                DELTA[j] / 20, float(np.sqrt(np.mean(residual[good, :, k, j, :, 0] ** 2)))
            )
    complete = np.asarray(measured.all(axis=(1, 2, 3, 4, 5)))
    if not complete.any():
        raise ValueError("No complete five-parent panels for multiplier fit")
    # Unfavorable measured work/preservation/numerical outcomes remain targets.
    t = np.full((len(roots), 5), np.nan)
    t[complete] = np.log(
        np.maximum(
            0.25,
            (np.abs(residual[complete]) / base[None, None, :, :, None, None]).max(
                axis=(2, 3, 4, 5)
            ),
        )
    )
    d = points.recipe.dimension
    if boundary == "lower":
        z = points.nh.apply(features.z[roots[complete], :, :d, 0])
        multiplier = fit_affine_operator(z.reshape(-1, d), t[complete].reshape(-1, 1), ridge=10)
    elif boundary == "cached":
        multiplier = t[complete].mean(axis=0)
    elif boundary in ("composed", "direct"):
        x = points.n0.apply(features.x[roots[complete], :d, 0])
        multiplier = np.stack(
            [fit_affine_operator(x, t[complete, p, None], ridge=10) for p in range(5)]
        )
    else:
        raise ValueError("Undeclared boundary")
    return WidthFit(boundary, base, multiplier, t, complete, eligible.sum(axis=0))


def scores(
    panel: ScoreObservations,
    roots: NDArray[np.int64],
    prediction: Array,
    sigma: Array,
    support: NDArray[np.bool_],
) -> tuple[Array, NDArray[np.bool_], NDArray[np.bool_]]:
    y = panel.y[roots]
    measured = (panel.observed[roots] & panel.valid[roots] & np.isfinite(y)).all(axis=(2, 3, 4, 5))
    numerical = (np.abs(y[..., 0] - y[..., 1]) <= NU[None, None, None, :, None]).all(axis=(2, 3, 4))
    scaled = (
        np.maximum(np.abs(y - prediction[..., None, None]) - NU[None, None, None, :, None, None], 0)
        / sigma[..., None, None]
    )
    parent = scaled.max(axis=(2, 3, 4, 5))
    parent = np.where(measured & numerical & support & np.isfinite(parent), parent, np.inf)
    return parent, np.asarray(measured), np.asarray(numerical)


@dataclass(frozen=True)
class Decisions:
    selected: NDArray[np.int64]
    feasible: NDArray[np.bool_]
    failures: NDArray[np.bool_]
    halfwidth: Array


DECISION_REASONS = (
    "unsupported_or_unavailable",
    "selected_pair_precision",
    "response_lower",
    "response_upper",
    "transverse",
    "preservation",
)


def oriented_interval(low: Array, high: Array, word: int) -> tuple[Array, Array]:
    if (
        type(word) is not int
        or word not in range(8)
        or low.shape[-2:] != (4, 8)
        or high.shape != low.shape
    ):
        raise ValueError("Unknown native word or response chart")
    k = word // 2
    if word % 2:
        return low[..., k, :].copy(), high[..., k, :].copy()
    order = [0, 1, 5, 6, 7, 2, 3, 4]
    lo, hi = low[..., k, order].copy(), high[..., k, order].copy()
    lo[..., :2], hi[..., :2] = -high[..., k, :2], -low[..., k, :2]
    return lo, hi


def choose(
    prediction: Array,
    sigma: Array,
    q: float,
    support: NDArray[np.bool_],
    direction: NDArray[np.int64],
    requirement: Array,
) -> Decisions:
    """Truth-blind primary decisions. Later work/source failures cannot reselect."""
    n = len(prediction)
    if (
        prediction.shape != (n, 5, 4, 8)
        or sigma.shape != prediction.shape
        or support.shape != (n, 5)
    ):
        raise ValueError("Invalid finite prediction table")
    if direction.shape != (n, 256, 2) or requirement.shape != direction.shape:
        raise ValueError("Missing frozen A/B requests")
    if (
        direction.dtype != np.int64
        or np.any((direction < 0) | (direction > 3))
        or not np.isfinite(requirement).all()
    ):
        raise ValueError("Invalid native request operands")
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(sigma).all()
        or (sigma <= 0).any()
        or np.isnan(q)
        or q < 0
    ):
        raise ValueError("Invalid point/scale/q package")
    spec = FiniteResponseLawScienceSpec()
    h = q * sigma + NU
    low, high = prediction - h, prediction + h
    low[..., 2:] = np.maximum(low[..., 2:], 0)
    high[..., 2:] = np.maximum(high[..., 2:], 0)
    precise = np.asarray((h <= DELTA).all(axis=-1))
    preservation = (high[..., 2:] <= np.tile(np.asarray(spec.preservation, dtype=float), 2)).all(
        axis=-1
    )
    fail = np.empty((n, 5, 256, 2, 8, 6), dtype=bool)
    for w in range(8):
        k = w // 2
        lo, hi = oriented_interval(low, high, w)
        for c in (0, 1):
            axis = direction[:, :, c] // 2
            positive = direction[:, :, c] % 2 == 0
            a = np.where(axis[:, None] == 0, lo[:, :, None, 0], lo[:, :, None, 1])
            b = np.where(axis[:, None] == 0, hi[:, :, None, 0], hi[:, :, None, 1])
            longitudinal_lo = np.where(positive[:, None], a, -b)
            longitudinal_hi = np.where(positive[:, None], b, -a)
            transverse_max = np.where(
                axis[:, None] == 0,
                np.maximum(np.abs(lo[:, :, None, 1]), np.abs(hi[:, :, None, 1])),
                np.maximum(np.abs(lo[:, :, None, 0]), np.abs(hi[:, :, None, 0])),
            )
            f = fail[:, :, :, c, w]
            f[..., 0] = ~support[:, :, None] | ~np.isfinite(q)
            f[..., 1] = ~precise[:, :, k, None]
            f[..., 2] = longitudinal_lo < requirement[:, None, :, c]
            f[..., 3] = longitudinal_hi > float(spec.upper[c])
            f[..., 4] = transverse_max > float(spec.transverse[c])
            f[..., 5] = ~np.asarray(preservation)[:, :, k, None]
    feasible = np.asarray(~fail.any(axis=-1))
    selected = np.where(feasible.any(axis=-1), feasible.argmax(axis=-1), -1).astype(np.int64)
    return Decisions(selected, feasible, fail, h)


def actual_events(
    selected: NDArray[np.int64], oracle: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    admitted = selected >= 0
    succeeds = (
        admitted & np.take_along_axis(oracle, np.maximum(selected, 0)[..., None], axis=-1)[..., 0]
    )
    joint = succeeds.all(axis=-1)
    false = (admitted & ~succeeds).any(axis=-1)
    return np.asarray(joint), np.asarray(false), admitted
