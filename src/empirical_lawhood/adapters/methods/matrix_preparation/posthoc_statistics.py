# SPDX-License-Identifier: MPL-2.0
"""Pure descriptive arithmetic for the finite exposed matrix panel."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


def components(values: Array) -> Array:
    if values.ndim < 2 or values.shape[-2] != 3:
        raise ValueError("response must retain the NEG/HOLD/POS axis before readout")
    return np.stack(
        (
            values[..., 1, :],
            (values[..., 2, :] - values[..., 0, :]) / 2,
            (values[..., 2, :] + values[..., 0, :]) / 2 - values[..., 1, :],
        ),
        axis=-2,
    )


def error_decomposition(predicted: Array, observed: Array) -> dict[str, Array]:
    """Exact three-action identity, including the signed baseline/even cross term."""
    if predicted.ndim != 5 or observed.shape != (*predicted.shape[:3], 4, 3, predicted.shape[-1]):
        raise ValueError("expected root/parent/view/[four futures]/three actions/readout")
    error = predicted[:, :, :, None] - observed
    e = components(predicted)[:, :, :, None] - components(observed)
    b, o, v = e[..., 0, :], e[..., 1, :], e[..., 2, :]
    terms = {
        "baseline_squared": b**2,
        "odd_squared_weighted": (2 / 3) * o**2,
        "even_squared_weighted": (2 / 3) * v**2,
        "baseline_even_cross": (4 / 3) * b * v,
    }
    total = np.mean(error**2, axis=-2)
    np.testing.assert_allclose(sum(terms.values()), total, rtol=2e-12, atol=2e-15)
    mean_future = observed.mean(axis=3)
    mean_error = (predicted - mean_future) ** 2
    within = np.mean((observed - mean_future[:, :, :, None]) ** 2, axis=3)
    np.testing.assert_allclose(
        mean_error + within, np.mean(error**2, axis=3), rtol=2e-12, atol=2e-15
    )
    return {
        **terms,
        "total": total,
        "four_future_mean_error_squared": mean_error,
        "within_four_future_variation": within,
    }


def distribution(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": len(array),
        "finite": len(finite),
        "nan": int(np.isnan(array).sum()),
        "infinite": int(np.isinf(array).sum()),
    }
    if len(finite):
        result.update(
            dict(
                zip(
                    ("min", "p01", "p05", "p25", "median", "p75", "p90", "p95", "p99", "max"),
                    map(
                        float,
                        np.quantile(finite, (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)),
                    ),
                    strict=True,
                )
            )
        )
        result.update(
            mean=float(finite.mean()),
            sd=float(finite.std()),
            rmse=float(np.sqrt(np.mean(finite**2))),
            mean_absolute=float(np.abs(finite).mean()),
            positive=int(np.sum(finite > 0)),
            negative=int(np.sum(finite < 0)),
            zero=int(np.sum(finite == 0)),
        )
    return result


def paired_roots(
    improvement: Array, root_indices: npt.NDArray[np.int64], bootstrap: npt.NDArray[np.int64]
) -> dict[str, Any]:
    """The caller supplies one paired loss difference per independent root."""
    if (
        improvement.ndim != 1
        or root_indices.shape != improvement.shape
        or len(improvement) < 2
        or bootstrap.ndim != 2
        or bootstrap.shape[1] != len(improvement)
        or bootstrap.min() < 0
        or bootstrap.max() >= len(improvement)
    ):
        raise ValueError("paired uncertainty must resample whole independent roots")
    result = distribution(improvement)
    if not np.isfinite(improvement).all():
        return {**result, "status": "MISSING_ROOT_OPERAND", "mean_improvement": None}
    loo = (improvement.sum() - improvement) / (len(improvement) - 1)
    intervals = np.quantile(improvement[bootstrap].mean(axis=1), (0.025, 0.975))
    return {
        **result,
        "status": "DESCRIPTIVE_EXPOSED_ONLY",
        "mean_improvement": float(improvement.mean()),
        "bootstrap_lower": float(intervals[0]),
        "bootstrap_upper": float(intervals[1]),
        "loo_mean_min": float(loo.min()),
        "loo_mean_max": float(loo.max()),
        "loo_sign_changes": int(np.sum(np.sign(loo) != np.sign(improvement.mean()))),
        "most_helpful_root": int(root_indices[np.argmax(improvement)]),
        "most_harmful_root": int(root_indices[np.argmin(improvement)]),
    }


def target_distance(endpoint: Array, targets: Array) -> Array:
    """Worst-view absolute distance for every root/parent/action."""
    if endpoint.ndim != 4 or endpoint.shape[-2:] != (2, 3) or targets.shape != (len(endpoint),):
        raise ValueError("target distance requires paired views, actions and root targets")
    return cast(Array, np.max(np.abs(endpoint - targets[:, None, None, None]), axis=2))


def replay_commands(
    prediction: Array,
    halfwidth: Array,
    targets: Array,
    support: npt.NDArray[np.bool_],
    preservation: Array,
    *,
    qualified: bool,
    ignore_qualification: bool = False,
    ignore_support: bool = False,
    ignore_width: bool = False,
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    """Hypothetical replay only; changing a gate never creates a qualified action."""
    if prediction.ndim != 4 or prediction.shape[-2:] != (2, 3):
        raise ValueError("command chart must retain root, parent, both views and three actions")
    if (
        halfwidth.shape != prediction.shape
        or targets.shape != (len(prediction),)
        or support.shape != prediction.shape[:2]
        or preservation.shape != support.shape
    ):
        raise ValueError("hypothetical replay changed its locked operand axes")
    width = np.zeros_like(halfwidth) if ignore_width else halfwidth
    known = np.all(np.isfinite(prediction) & np.isfinite(width) & (width >= 0), axis=2)
    margin = 0.125 - np.max(np.abs(prediction - targets[:, None, None, None]) + width, axis=2)
    inside = known & (margin >= 0)
    preservation_ok = np.isfinite(preservation) & (preservation >= 0.9) & (preservation <= 1)
    allowed = inside & preservation_ok[..., None]
    if not ignore_support:
        allowed &= support[..., None]
    if not (qualified or ignore_qualification):
        allowed[:] = False
    commands = np.full(support.shape, -1, dtype=np.int64)
    for action in (2, 0, 1):
        commands[allowed[..., action]] = action
    obstacles = {
        "unqualified": np.full(support.shape, not qualified),
        "unsupported": ~support,
        "preservation_probability_failed": ~preservation_ok,
        "no_finite_action_interval": ~known.any(axis=2),
        "no_contained_action_interval": ~inside.any(axis=2),
        "no_point_center_in_target": np.all(target_distance(prediction, targets) > 0.125, axis=2),
        "every_action_too_wide_even_if_centered": np.all(np.max(width, axis=2) > 0.125, axis=2),
        "best_interval_margin": np.max(np.where(known, margin, -np.inf), axis=2),
    }
    return commands, obstacles


def score_commands(commands: npt.NDArray[np.int64], success: npt.NDArray[np.bool_]) -> Any:
    if success.shape != (*commands.shape, 3) or np.any((commands < -1) | (commands > 2)):
        raise ValueError("task scoring changes action/refusal semantics")
    chosen = np.take_along_axis(success, np.maximum(commands, 0)[..., None], axis=2)[..., 0]
    return chosen & (commands >= 0)
