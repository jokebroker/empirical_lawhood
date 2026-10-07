# SPDX-License-Identifier: MPL-2.0
"Pure descriptive arithmetic for the bounded reactor post-hoc audit.\n\nNo acquisition, fitting, qualification, admission or measurement through controller use status is produced.\nNumerical arrays use float64; exact threshold decisions use retained Decimals.\n"

from __future__ import annotations

from decimal import Decimal as D
from typing import Any

import numpy as np

TOLERANCES = (D(".035"), D(".0359125"), D(".0025"))
PHASE_EDGES = (0, 600, 9000, 18000, 25200, 28800)
MULTIPLIERS = ("0", ".25", ".5", "1", "1.5", "2", "2.25", "3", "4", "8")


def unpack(value: Any) -> Any:
    """Decode canonical value wrappers after custody authentication."""
    if isinstance(value, dict):
        if set(value) == {"decimal"}:
            return D(value["decimal"])
        if set(value) == {"schema", "version", "value"}:
            return unpack(value["value"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def interval_truth(grid: np.ndarray, sub: int) -> np.ndarray:
    """Include both endpoints in the peak, retaining the final native interval."""
    if sub < 1 or (len(grid) - 1) % sub:
        raise ValueError("incomplete interval grid")
    starts = grid[:-1:sub, 1]
    rest = grid[1:, 1].reshape(-1, sub).max(axis=1)
    return np.column_stack((np.maximum(starts, rest), grid[sub::sub, 3:5]))


def first_time(mask: np.ndarray, times: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def error_summary(residual: np.ndarray) -> dict[str, Any]:
    return {
        "max_absolute": np.abs(residual).max(axis=0).tolist(),
        "rmse": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "mean_truth_minus_prediction": residual.mean(axis=0).tolist(),
        "max_underprediction": residual.max(axis=0).tolist(),
        "max_overprediction": (-residual).max(axis=0).tolist(),
    }


def summarize_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    states, forecasts = trace["states"], trace["forecasts"]
    dt = trace["plant_dt_s"]
    grid = np.asarray(trace["native_grid"], dtype=float)
    if trace["failure_code"] is not None or len(states) != 2880 or len(forecasts) != 2880:
        raise ValueError("post-hoc census differs from complete retained experiment")
    if dt not in (D(1), D(".5")) or grid.shape != (int(D(28800) / dt) + 1, 8):
        raise ValueError("post-hoc native grid differs")
    if not np.array_equal(grid[:, 0], np.arange(len(grid)) * float(dt)):
        raise ValueError("native clock differs")
    sub = int(D(10) / dt)
    point = np.asarray(
        [
            [f[k] for k in ("peak_temperature_k", "endpoint_dosed_kg", "endpoint_conversion_b")]
            for f in forecasts
        ],
        dtype=float,
    )
    persistence = np.asarray(
        [
            [s["t_estimate_k"], s["measurement"]["dosed_kg"], D(1) - s["n_b_mol"] / D(3170)]
            for s in states
        ],
        dtype=float,
    )
    truth = interval_truth(grid, sub)
    residual = truth - point
    baseline = truth - persistence
    clocks = np.arange(2880) * 10
    if any(
        s["measurement"]["time_s"] != i * 10
        or f["command"]["time_s"] != i * 10
        or f["command"]["feed_kg_s"] != s["reference_feed_kg_s"]
        or f["command"]["jacket_k"] != s["reference_jacket_k"]
        for i, (s, f) in enumerate(zip(states, forecasts, strict=True))
    ):
        raise ValueError("causal command clock or reference action differs")
    stages = np.asarray(trace["observed_stages"], dtype=float)
    projected = np.asarray(
        [[f["predicted_feed_kg_s"], f["predicted_jacket_k"]] for f in forecasts], dtype=float
    )
    if stages.shape != (2880, 4):
        raise ValueError("delivery census differs")
    requested = np.asarray(
        [[f["command"]["feed_kg_s"], f["command"]["jacket_k"]] for f in forecasts], dtype=float
    )
    # Reconstruct exact maximum residuals at the float-selected maxima.
    exact_maxima = []
    locations = []
    for k in range(3):
        i = int(np.argmax(np.abs(residual[:, k])))
        interval = trace["native_grid"][i * sub : (i + 1) * sub + 1]
        actual = (max(row[1] for row in interval), interval[-1][3], interval[-1][4])[k]
        exact_point = forecasts[i][
            ("peak_temperature_k", "endpoint_dosed_kg", "endpoint_conversion_b")[k]
        ]
        exact_maxima.append(abs(actual - exact_point))
        locations.append(
            {
                "decision_time_s": int(clocks[i]),
                "truth": actual,
                "prediction": exact_point,
                "signed_error": actual - exact_point,
            }
        )
    physical_noise = np.asarray(
        [[s["measurement"]["t_reactor_k"], s["measurement"]["t_jacket_k"]] for s in states], float
    )
    delayed_indices = np.maximum(clocks - 10, 0) / float(dt)
    physical_noise -= grid[delayed_indices.astype(int)][:, 1:3]
    applied_integral = np.cumsum(grid[1:, 6] * float(dt))
    summary = {
        "plant_dt_s": dt,
        "forecast": error_summary(residual),
        "persistence": error_summary(baseline),
        "exact_maximum_errors": exact_maxima,
        "maximum_error_locations": locations,
        "phases": [
            {
                "start_s": lo,
                "end_s": hi,
                "forecast": error_summary(residual[(clocks >= lo) & (clocks < hi)]),
                "persistence": error_summary(baseline[(clocks >= lo) & (clocks < hi)]),
            }
            for lo, hi in zip(PHASE_EDGES, PHASE_EDGES[1:])
        ],
        "predicted_applied_stage_max_discrepancy": np.abs(stages[:, 2:] - projected)
        .max(axis=0)
        .tolist(),
        "requested_vs_applied_different_callbacks": np.sum(
            np.abs(requested - stages[:, 2:]) > 1e-12, axis=0
        ).tolist(),
        "dose_integral_max_discrepancy_kg": float(np.max(np.abs(applied_integral - grid[1:, 3]))),
        "outcomes": {
            "maximum_temperature_k": float(grid[:, 1].max()),
            "temperature_violation_grid_points": int(np.sum(grid[:, 1] > 356.2)),
            "dose_at_window_close_kg": float(grid[int(25200 / float(dt)), 3]),
            "final_dose_kg": float(grid[-1, 3]),
            "final_conversion": float(grid[-1, 4]),
            "completion_time_s": first_time(
                (grid[:, 3] >= 0.999 * 287.3) & (grid[:, 4] >= 0.98), grid[:, 0]
            ),
            "first_dose_completion_s": first_time(grid[:, 3] >= 0.999 * 287.3, grid[:, 0]),
            "first_conversion_completion_s": first_time(grid[:, 4] >= 0.98, grid[:, 0]),
            "dose_after_window_kg": float(grid[-1, 3] - grid[int(25200 / float(dt)), 3]),
        },
    }
    arrays = {
        "grid": grid,
        "point": point,
        "residual": residual,
        "baseline": baseline,
        "requested": requested,
        "stages": stages,
        "noise": physical_noise,
        "observer": np.asarray(
            [
                [s["ua_estimate_w_per_k"], s["kinetic_multiplier"], s["t_estimate_k"]]
                for s in states
            ],
            float,
        ),
    }
    return summary, arrays


def sensitivity(
    units: list[dict[str, Any]], tolerance: tuple[D, ...], padding: tuple[D, ...]
) -> dict[str, Any]:
    """Descriptive alternative arithmetic only; retain every assigned unit."""
    if len(units) != 42 or sum(u["split"] == "calibration" for u in units) != 32:
        raise ValueError("counterfactual cannot drop assigned units")
    invalid = [
        u["unit_id"]
        for u in units
        if any(d > t for d, t in zip(u["numerical"], tolerance, strict=True))
    ]
    calibration = [u for u in units if u["split"] == "calibration"]
    finite = not any(u["unit_id"] in invalid for u in calibration)
    bounds = (
        [max(u["raw_maximum_errors"][k] for u in calibration) + padding[k] for k in range(3)]
        if finite
        else None
    )
    heldout = [u for u in units if u["split"] == "heldout"]
    # None means undefined due to missing calibration, never observed zero coverage.
    covered = (
        None
        if bounds is None
        else [
            u["unit_id"]
            for u in heldout
            if u["unit_id"] not in invalid
            and all(e <= b for e, b in zip(u["raw_maximum_errors"], bounds, strict=True))
        ]
    )
    receiver_counts = (
        None
        if bounds is None
        else [
            sum(
                u["unit_id"] not in invalid and u["raw_maximum_errors"][k] <= bounds[k]
                for u in heldout
            )
            for k in range(3)
        ]
    )
    return {
        "tolerance": tolerance,
        "padding": padding,
        "invalid_units": invalid,
        "bounds": bounds,
        "heldout_covered_units": covered,
        "heldout_receiver_coverage_counts": receiver_counts,
        "nine_of_ten_arithmetic": None if covered is None else len(covered) >= 9,
        "classification": "POSTHOC_COUNTERFACTUAL_NO_QUALIFICATION",
    }
