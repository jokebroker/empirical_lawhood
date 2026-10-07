"""Frozen prediction arithmetic for one assigned parent per fresh root.

No fitting, quantile estimation, qualification, admission or native source I/O.
Pre-parent prediction cannot accept measured handoff features. The separate
lower evaluator is for calibration/evaluator use after actual preparation.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction
from .fitting import Array, PointFit, clamp
from .intervals import WidthFit


@dataclass(frozen=True)
class AssignedPrediction:
    mean: Array  # root, canonical action pair, response/preservation coordinate
    sigma: Array
    supported: NDArray[np.bool_]


@dataclass(frozen=True)
class PreParentPredictions:
    predicted_interface: Array  # native feature units, root, feature
    composed: AssignedPrediction
    cached: AssignedPrediction
    direct: AssignedPrediction


def _inputs(prefix: Array, parents: NDArray[np.int64]) -> None:
    if (
        prefix.ndim != 3
        or prefix.shape[1:] != (24, 2)
        or prefix.dtype != np.float64
        or parents.shape != (len(prefix),)
        or parents.dtype != np.int64
        or ((parents < 0) | (parents >= 5)).any()
    ):
        raise ValueError("Expected native features and one declared parent index per root")


def _sigma(width: WidthFit, logg: Array) -> Array:
    g = np.exp(np.clip(logg, np.log(0.25), np.log(4)))
    return width.base[None, None] * g[:, :, None, None]


def before_parent(
    points: PointFit,
    widths: Mapping[str, WidthFit],
    prefix: Array,
    parents: NDArray[np.int64],
) -> PreParentPredictions:
    """Apply unchanged U, F, cached and direct operators using prefix inputs only."""
    _inputs(prefix, parents)
    if set(widths) != {"composed", "cached", "direct"} or any(
        width.boundary != name for name, width in widths.items()
    ):
        raise ValueError("Pre-parent prediction requires its three frozen scale boundaries")
    d = points.recipe.dimension
    x = points.n0.apply(prefix[:, :d, 0])
    # Retain the frozen fit's full five-parent arithmetic layout; select the assigned parent
    # afterward. These are model evaluations, not additional acquisitions.
    uz = np.stack([affine_prediction(points.upper[p], x) for p in range(5)], axis=1)
    nu = points.nh.apply(uz)
    composed = clamp(affine_prediction(points.lower, nu.reshape(-1, d)).reshape(-1, 5, 4, 8))
    direct = clamp(
        np.stack(
            [affine_prediction(points.direct[p], x) for p in range(5)],
            axis=1,
        ).reshape(-1, 5, 4, 8)
    )
    cached = clamp(np.broadcast_to(points.cached.reshape(5, 4, 8), composed.shape).copy())
    prefix_support = (np.isfinite(x).all(axis=1) & (np.abs(x).max(axis=1) <= 6))[:, None]
    supports = {
        "composed": prefix_support & np.isfinite(nu).all(axis=2) & (np.abs(nu).max(axis=2) <= 6),
        "direct": np.broadcast_to(prefix_support, (len(prefix), 5)),
        "cached": np.ones((len(prefix), 5), dtype=bool),
    }
    outputs = {}
    rows = np.arange(len(prefix))
    for name, mean in (("composed", composed), ("cached", cached), ("direct", direct)):
        width = widths[name]
        logg = (
            np.broadcast_to(width.multiplier, (len(prefix), 5))
            if name == "cached"
            else np.column_stack(
                [affine_prediction(width.multiplier[p], x)[:, 0] for p in range(5)]
            )
        )
        outputs[name] = AssignedPrediction(
            mean[rows, parents],
            _sigma(width, logg)[rows, parents],
            supports[name][rows, parents],
        )
    return PreParentPredictions(
        uz[rows, parents], outputs["composed"], outputs["cached"], outputs["direct"]
    )


def after_parent(
    points: PointFit,
    width: WidthFit,
    handoff: Array,
    parents: NDArray[np.int64],
) -> AssignedPrediction:
    """Evaluate lower F at the measured handoff, with no predicted-interface substitution."""
    _inputs(handoff, parents)
    if width.boundary != "lower":
        raise ValueError("Measured-handoff prediction requires the frozen lower scale")
    d = points.recipe.dimension
    z = points.nh.apply(handoff[:, :d, 0])
    # Match the frozen five-slot BLAS layout, including single-root evaluation.
    # Padding consists only of computational copies of this measured handoff;
    # it is never emitted as features or as observations of unacquired parents.
    padded = np.repeat(z[:, None, :], 5, axis=1)
    mean = clamp(affine_prediction(points.lower, padded.reshape(-1, d)).reshape(-1, 5, 4, 8))
    logg = affine_prediction(width.multiplier, padded.reshape(-1, d)).reshape(-1, 5)
    supported = np.isfinite(z).all(axis=1) & (np.abs(z).max(axis=1) <= 6)
    rows = np.arange(len(handoff))
    return AssignedPrediction(mean[rows, parents], _sigma(width, logg)[rows, parents], supported)
