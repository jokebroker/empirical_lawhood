"Finite response-law qualification descriptive operands, with no qualification verdict or new entry gate."

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assigned_prediction import AssignedPrediction
from .calibration_analysis import _validate_prediction, calibration_decisions
from .calibration_panel import FIXED_CALIBRATION_ROOT_IDS, Array, CalibrationPanel
from .fitting import DELTA
from .intervals import DECISION_REASONS


def _mean(values: Array, available: NDArray[np.bool_]) -> float | None:
    if not available.any():
        return None
    selected = values[available]
    scale = float(selected.max())
    return float((selected / scale).mean()) * scale if scale else 0.0


def two_future_diagnostics(
    panel: CalibrationPanel, predictions: Mapping[str, AssignedPrediction]
) -> dict[str, Any]:
    """The frozen finite-future error identity; not a latent uncertainty estimate.

    Counts below are descriptive root/pair/view cells, never independent n.
    Missing or invalid observations remain explicit. Unsupported predictions
    may be described but cannot become admissions through this diagnostic.
    """
    if set(predictions) != {"lower", "composed", "cached", "direct"}:
        raise ValueError(
            "Finite response-law qualification diagnostics require the four declared prediction boundaries"
        )
    for prediction in predictions.values():
        _validate_prediction(prediction)
    y = panel.y[:, 0]
    measured = panel.observed[:, 0] & panel.valid[:, 0] & np.isfinite(y)
    paired = np.asarray(measured.all(axis=-2))
    complete = np.asarray(paired.all(axis=-1))
    with np.errstate(invalid="ignore", over="ignore"):
        center = (y[..., 0, :] + y[..., 1, :]) / 2
        gap = y[..., 0, :] - y[..., 1, :]
        spread = (gap / 2) ** 2
        variance = 2 * spread
        joint_range = np.max(y, axis=(-1, -2)) - np.min(y, axis=(-1, -2))
    outputs = []
    for j in range(8):
        mask = paired[:, :, j]
        spread_finite = mask & np.isfinite(spread[:, :, j])
        variance_finite = mask & np.isfinite(variance[:, :, j])
        joint_mask = complete[:, :, j]
        outputs.append(
            {
                "output_index": j,
                "planned_root_pair_views": 32 * 4 * 2,
                "measured_root_pair_views": int(mask.sum()),
                "unavailable_root_pair_views": int((~mask).sum()),
                "finite_spread_root_pair_views": int(spread_finite.sum()),
                "within_future_spread_mse": _mean(spread[:, :, j], spread_finite),
                "finite_variance_root_pair_views": int(variance_finite.sum()),
                "conditional_variance_estimate": _mean(
                    variance[:, :, j], variance_finite
                ),
                "future_gap_exceeds_2delta_count": int(
                    (mask & (np.abs(gap[:, :, j]) > 2 * DELTA[j])).sum()
                ),
                "planned_root_pairs": 32 * 4,
                "measured_complete_future_view_root_pairs": int(joint_mask.sum()),
                "unavailable_complete_future_view_root_pairs": int((~joint_mask).sum()),
                "all_future_view_range_exceeds_2delta_count": int(
                    (joint_mask & (joint_range[:, :, j] > 2 * DELTA[j])).sum()
                ),
            }
        )
    boundaries = {}
    for name, prediction in sorted(predictions.items()):
        with np.errstate(invalid="ignore", over="ignore"):
            mean_error = (center - prediction.mean[..., None]) ** 2
            mse = ((y - prediction.mean[..., None, None]) ** 2).mean(axis=-2)
            reconstructed = mean_error + spread
        finite = (
            paired
            & np.isfinite(mse)
            & np.isfinite(mean_error)
            & np.isfinite(reconstructed)
        )
        # This verifies the diagnostic itself. It never qualifies a predictor.
        np.testing.assert_allclose(
            mse[finite], reconstructed[finite], rtol=1e-12, atol=1e-15
        )
        boundaries[name] = [
            {
                "output_index": j,
                "finite_prediction_and_measured_root_pair_views": int(
                    finite[:, :, j].sum()
                ),
                "unavailable_root_pair_views": int((~finite[:, :, j]).sum()),
                "two_future_mse": _mean(mse[:, :, j], finite[:, :, j]),
                "squared_error_of_future_mean": _mean(
                    mean_error[:, :, j], finite[:, :, j]
                ),
                "within_future_spread_mse": _mean(spread[:, :, j], finite[:, :, j]),
            }
            for j in range(8)
        ]
    return {
        "independent_root_count": 32,
        "root_ids": list(panel.root_ids),
        "diagnostic_only": True,
        "conditional_variance_interpretation": "requires-conditionally-iid-futures; two-future-estimate-only",
        "outputs": outputs,
        "prediction_error_decomposition": boundaries,
    }


def calibration_usability(
    prediction: AssignedPrediction,
    q: float,
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    *, request_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Complete fixed request census; qualification eligibility remains separate."""
    decisions = calibration_decisions(prediction, q, root_ids, request_seeds=request_seeds)
    selected = decisions.selected[:, 0]
    admitted = selected >= 0
    joint = np.asarray(admitted.all(axis=-1))
    halfwidth = decisions.halfwidth[:, 0]
    finite = np.isfinite(halfwidth)
    width_rows = []
    for j in range(8):
        values = halfwidth[:, :, j]
        mask = finite[:, :, j]
        width_rows.append(
            {
                "output_index": j,
                "planned_root_pairs": 128,
                "finite_count": int(mask.sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "unavailable_count": int(np.isnan(values).sum()),
                "minimum": float(values[mask].min()) if mask.any() else None,
                "median": float(np.median(values[mask])) if mask.any() else None,
                "maximum": float(values[mask].max()) if mask.any() else None,
                "within_delta_count": int((mask & (values <= DELTA[j])).sum()),
            }
        )
    return {
        "independent_root_count": 32,
        "request_pairs_per_root": 256,
        "request_pair_count": 8192,
        "jointly_admissible_request_pairs": int(joint.sum()),
        "jointly_admissible_request_pairs_per_root": joint.sum(axis=1).tolist(),
        "roots_with_jointly_admissible_pair": int(joint.any(axis=1).sum()),
        "joint_decision_opportunity": bool(joint.any()),
        "requires_separate_boundary_qualification": True,
        "consumer_admission_counts": admitted.sum(axis=(0, 1)).tolist(),
        "consumer_selection_counts": [
            {str(word): int((selected[:, :, c] == word).sum()) for word in range(-1, 8)}
            for c in range(2)
        ],
        "word_refusal_counts_by_consumer": {
            name: decisions.failures[:, 0, :, :, :, j].sum(axis=(0, 1, 3)).tolist()
            for j, name in enumerate(DECISION_REASONS)
        },
        "full_menu_precise_roots": int(
            (finite & (halfwidth <= DELTA)).all(axis=(1, 2)).sum()
        ),
        "nominal_halfwidth_by_output": width_rows,
    }
