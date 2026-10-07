# SPDX-License-Identifier: MPL-2.0

"""Whole-root sensitivity on outcome-visible retained reactor statistics.

The original 5,000 resamples and descriptive quantiles do not provide fresh
power, source selection, delayed-probe labels or control admission.

Authenticate retained inputs and obtain separate analysis authority first.
"""

from typing import Any

from empirical_lawhood.kernel.serialization import validate_sha256

import numpy as np


def _bootstrap(
    values: np.ndarray, size: int, rng: np.random.Generator
) -> dict[str, float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(5000, size))], axis=1)
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(draws, 0.05)),
        "p50": float(np.quantile(draws, 0.5)),
        "p95": float(np.quantile(draws, 0.95)),
    }


CONTEXT = (0, 1, 3, 4, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20)


def analyze_retained_margins(
    domain: dict[str, Any],
    summary: dict[str, Any],
    contexts: list[dict[str, Any]],
    opportunity: list[dict[str, Any]],
    features_by_root: dict[str, np.ndarray],
    reports_by_root: dict[str, dict[str, Any]],
    *,
    parent_complete_sha256: str,
    old_design_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reduce authenticated retained root reports and supplied feature charts.

    Temperature is in K, delivered mass in kg, and features end at the actual
    callback. The 64-root support margins and 5,000-resample sensitivity remain
    exposed development diagnostics. Supplied hashes describe separately verified
    inputs; this function performs no custody, source or publication act.
    """
    validate_sha256(parent_complete_sha256, field_name="parent_complete_sha256")
    validate_sha256(old_design_sha256, field_name="old_design_sha256")
    design_sha = old_design_sha256
    prepared_rows = {r["root"]: r for r in contexts if r["source"] == "prepared_feed"}
    if len(prepared_rows) != 64 or len(opportunity) != 256:
        raise ValueError("retained prepared root/request denominator differs")
    if set(features_by_root) != set(prepared_rows) or set(reports_by_root) != set(
        prepared_rows
    ):
        raise ValueError("Retained support inputs differ from the 64-root census")
    by_root: dict[str, list[dict[str, Any]]] = {}
    for item in opportunity:
        by_root.setdefault(item["root"], []).append(item)
    lower = np.asarray([float(v["decimal"]) for v in domain["support_lower"]])
    upper = np.asarray([float(v["decimal"]) for v in domain["support_upper"]])
    widths = np.maximum(upper - lower, 1e-12)
    path = tuple(
        ((int(i), float(t["decimal"]), bool(left)) for i, t, left in domain["path"])
    )
    old_q = summary["old_law_calibration_q"]
    q_t, q_c = (float(old_q["temperature"]), float(old_q["cooling"]))
    margins = []
    for root, row in sorted(prepared_rows.items()):
        feature = features_by_root[root]
        if feature.shape != (3, 23):
            raise ValueError("old prepared native feature chart differs")
        faces = np.minimum(
            (feature[:, CONTEXT] - lower) / widths,
            (upper - feature[:, CONTEXT]) / widths,
        )
        split_faces = [
            threshold - feature[:, coordinate]
            if left
            else feature[:, coordinate] - threshold
            for coordinate, threshold, left in path
        ]
        report = reports_by_root[root]
        if report["root"] != root or report["contact"] is not True:
            raise ValueError("old prepared diagnostic root changed")
        word = report["words"]
        if tuple((w["action"] for w in word)) != (1, 4, 7):
            raise ValueError("old prepared word order differs")
        response_width = [
            2 * (q_c * (5e-05 + 0.5 * abs(float(w["predicted_cooling_K"]))) + 1e-06)
            for w in word
        ]
        root_opportunity = sorted(by_root[root], key=lambda x: float(x["request_K"]))
        margins.append(
            {
                "root": root,
                "native_support_face_min_fraction": float(np.min(faces)),
                "native_split_face_min_coordinate": float(
                    min((np.min(v) for v in split_faces))
                ),
                "native_support_inside": bool(
                    np.all(faces >= -1e-12)
                    and all((np.all(v >= -1e-12) for v in split_faces))
                ),
                "temperature_limit_headroom_K": float(
                    356.2
                    - max((float(v) for w in word for v in w["measured_temperature_K"]))
                ),
                "old_temperature_interval_width_K": 2 * (q_t * 0.25 + 0.01),
                "old_cooling_interval_width_K_by_word": response_width,
                "requests": [
                    {
                        "request_K": item["request_K"],
                        "measured_best_cooling_margin_K": item["best_margin_K"],
                        "provisional_best_interval_margin_K": max(
                            item["provisional_interval_margins_K"]
                        ),
                        "measured_feasible_word": item["feasible_word"],
                        "provisional_old_law_word": item["provisional_old_law_word"],
                    }
                    for item in root_opportunity
                ],
            }
        )
    if len(margins) != 64:
        raise ValueError("prepared support margins lost an independent root")
    rng = np.random.default_rng(20260923)
    service = np.asarray(
        [
            all(
                (
                    request["provisional_old_law_word"] is not None
                    for request in row["requests"]
                )
            )
            for row in margins
        ],
        dtype=np.float64,
    )
    best = summary["cv_best"]
    smooth = min(
        (best["history:affine"], best["history:rbf"]),
        key=lambda item: item["mean_root_loss_K2"] or float("inf"),
    )
    local = best["history:local"]
    common = sorted(set(smooth["root_losses_K2"]) & set(local["root_losses_K2"]))
    if len(common) < 64:
        raise ValueError("retained model comparison lost independent roots")
    advantage = np.asarray(
        [
            0.8 * smooth["root_losses_K2"][r] - local["root_losses_K2"][r]
            for r in common
        ],
        dtype=np.float64,
    )
    stats: dict[str, Any] = {
        "kind": "REACTOR_REGIME_RESPONSE_RETAINED_MARGIN_AND_ROOT_SENSITIVITY",
        "parent_complete_sha256": parent_complete_sha256,
        "old_design_sha256": design_sha,
        "native_calls": 0,
        "claim_ceiling": "EXPOSED_DEVELOPMENT_ONLY",
        "prepared_roots": 64,
        "support_inside_roots": sum((row["native_support_inside"] for row in margins)),
        "support_face_min_fraction": min(
            (row["native_support_face_min_fraction"] for row in margins)
        ),
        "split_face_min_coordinate": min(
            (row["native_split_face_min_coordinate"] for row in margins)
        ),
        "temperature_headroom_min_K": min(
            (row["temperature_limit_headroom_K"] for row in margins)
        ),
        "old_temperature_interval_width_K": margins[0][
            "old_temperature_interval_width_K"
        ],
        "old_cooling_interval_width_max_K": max(
            (max(row["old_cooling_interval_width_K_by_word"]) for row in margins)
        ),
        "near_misses": [
            {
                "root": row["root"],
                "request_K": req["request_K"],
                "interval_margin_K": req["provisional_best_interval_margin_K"],
            }
            for row in margins
            for req in row["requests"]
            if abs(req["provisional_best_interval_margin_K"]) <= 0.0001
        ],
        "whole_root_bootstrap_draws": 5000,
        "whole_root_resampling": {
            str(n): {
                "old_provisional_four_request_fraction": _bootstrap(service, n, rng),
                "local_vs_smooth_20pct_advantage_K2": _bootstrap(advantage, n, rng),
            }
            for n in (16, 32, 64)
        },
        "resampling_limit": "Exposed roots and old unqualified intervals; these are sensitivity diagnostics, not fresh power or controller admission admission.",
        "probe_signal_limit": "Retained ten-second action labels do not measure the proposed delayed 300-second online probe readout.",
    }
    return (stats, margins)
