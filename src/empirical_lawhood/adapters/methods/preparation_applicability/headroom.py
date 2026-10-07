"""Outcome-visible finite-menu opportunity and causal-prefix LOO nomination.

These diagnostics are not prospective learned-policy qualification, do not
contact a native simulator, and do not change the frozen lower law.
"""

import numpy as np

def opportunity(maxima, success):
    """Whole-root, finite-menu ceilings; nested requests never become n."""
    if (
        maxima.ndim != 3 or maxima.shape[1:] != (3, 7)
        or success.shape != (len(maxima), 3, 256, 2)
        or success.dtype != np.bool_ or not np.isfinite(maxima).all()
    ):
        raise ValueError("opportunity requires the complete root/menu/conjunct census")
    valid = (maxima <= 1).all(axis=2)
    joint = success.all(axis=3).sum(axis=2)
    covered = joint * valid
    return {
        "n_independent_roots": len(maxima),
        "full_valid_counts": valid.sum(axis=0).tolist(),
        "support_counts": (maxima[:, :, 0] <= 1).sum(axis=0).tolist(),
        "negative_preparation_support_rescues_vs_wait": np.flatnonzero(
            (maxima[:, 0, 0] > 1) & (maxima[:, 1, 0] <= 1)
        ).tolist(),
        "full_valid_oracle_count": int(valid.any(axis=1).sum()),
        "full_valid_oracle_gain_over_best_fixed": int(
            valid.any(axis=1).sum() - valid.sum(axis=0).max()
        ),
        "joint_service_counts": joint.sum(axis=0).tolist(),
        "covered_service_counts": covered.sum(axis=0).tolist(),
        "joint_service_oracle_gain_over_best_fixed": int(
            joint.max(axis=1).sum() - joint.sum(axis=0).max()
        ),
        "covered_service_oracle_gain_over_best_fixed": int(
            covered.max(axis=1).sum() - covered.sum(axis=0).max()
        ),
        "oracle_uses_actual_outcomes_and_is_not_an_available_policy": True,
    }


def prefix_forecast(prefix, handoff, boundary):
    """Finite exploratory ridge diagnostic with genuine whole-root LOO fits.

    Fit scales on each training fold. No privileged handoff enters a held-root
    input; handoffs are supervised labels in this exposed nomination analysis.
    This diagnostic is not the strongest possible causal forecasting method.
    """
    if prefix.shape != (96, 24) or handoff.shape != (96,):
        raise ValueError("nomination forecast requires the exact exposed census")
    if not np.isfinite(prefix).all() or not np.isfinite(handoff).all():
        raise ValueError("nomination forecast loses missingness")
    forecasts = np.empty(96)
    for held in range(96):
        training = np.arange(96) != held
        x = prefix[training]
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        if (scale <= 0).any():
            raise ValueError("nomination forecast has an unmeasured constant input")
        design = np.column_stack((np.ones(95), (x - center) / scale))
        penalty = np.eye(25) * 10
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ handoff[training]
        )
        forecasts[held] = np.r_[1, (prefix[held] - center) / scale] @ coefficients
    actual_failure = handoff > boundary
    selected = forecasts > boundary
    return {
        "ridge": 10,
        "training_only_scales": True,
        "rmse_native_force_per_time": float(np.sqrt(np.mean((forecasts - handoff) ** 2))),
        "forecast_range_native_force_per_time": [float(forecasts.min()), float(forecasts.max())],
        "support_boundary_native_force_per_time": float(boundary),
        "actual_failed_roots": np.flatnonzero(actual_failure).tolist(),
        "forecast_failed_roots": np.flatnonzero(selected).tolist(),
        "actual_failure_root_forecasts": forecasts[actual_failure].tolist(),
        "ceiling": "EXPLORATORY_NOT_PROSPECTIVE_NOT_IMPOSSIBILITY",
    }
