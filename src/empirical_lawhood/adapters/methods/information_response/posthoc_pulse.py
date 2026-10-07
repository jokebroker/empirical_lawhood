# SPDX-License-Identifier: MPL-2.0
"""Outcome-visible pulse, coordinate and future sensitivity reductions.

Consumes the retained information-response analysis; no new fits or native work.
"""

from itertools import combinations
import numpy as np
from .models import MODEL_IDS
from .posthoc_statistics import CEILING, PRIMARY
from .posthoc_reduction import CONTEXTS

PARENTS = (
    "hold",
    "y-negative-128",
    "y-positive-128",
    "y-negative-256",
    "y-positive-256",
)
HORIZONS = (64, 128, 192, 256, 320)


def norm(matrix):
    return np.linalg.norm(matrix, axis=(-2, -1))


def cosine(first, second):
    return (first * second).sum(axis=(-2, -1)) / np.maximum(
        norm(first) * norm(second), 1e-30
    )


def pulse_supplement(data, analysis):
    """Return original post-hoc supplement and derived arrays; authorize first."""
    result, arrays = {}, {}
    for ci, context in enumerate(CONTEXTS):
        g = data["gains"][ci].swapaxes(2, 3)
        f = data["forecasts"][ci].swapaxes(2, 3)
        matrix = g.mean(2)
        hold = matrix[:, :, 0]
        odd, even, durations = ([], [], [])
        for negative, positive, ticks in ((1, 2, 128), (3, 4, 256)):
            dn, dp = (matrix[:, :, negative] - hold, matrix[:, :, positive] - hold)
            o, e = ((dp - dn) / 2, (dp + dn) / 2)
            np.testing.assert_allclose(dp, e + o, rtol=1e-12, atol=1e-18)
            np.testing.assert_allclose(dn, e - o, rtol=1e-12, atol=1e-18)
            odd.append(o)
            even.append(e)
            durations.append(
                {
                    "duration_ticks": ticks,
                    "odd_norm_over_HOLD_median_view_horizon": np.median(
                        norm(o) / norm(hold), 0
                    ),
                    "even_norm_over_HOLD_median_view_horizon": np.median(
                        norm(e) / norm(hold), 0
                    ),
                    "even_to_odd_norm_ratio_median_view_horizon": np.median(
                        norm(e) / norm(o), 0
                    ),
                    "negative_positive_cosine_median_view_horizon": np.median(
                        cosine(dn, dp), 0
                    ),
                }
            )
        arrays[f"{context}__parent_odd"] = np.stack(odd, axis=2)
        arrays[f"{context}__parent_even"] = np.stack(even, axis=2)
        c = {
            "parent_pulse": {
                "matrix_uses_both_future_mean": True,
                "duration_rows": durations,
                "long_short_odd_norm_ratio_median_view_horizon": np.median(
                    norm(odd[1]) / norm(odd[0]), 0
                ),
                "long_short_odd_cosine_median_view_horizon": np.median(
                    cosine(odd[1], odd[0]), 0
                ),
                "long_short_even_norm_ratio_median_view_horizon": np.median(
                    norm(even[1]) / norm(even[0]), 0
                ),
                "doubling_odd_residual_relative_median_view_horizon": np.median(
                    norm(odd[1] - 2 * odd[0]) / norm(odd[1]), 0
                ),
                "mean_parent_work_view_parent": data["parent_work"][ci].mean(0),
                "work_unit": "native accumulated absolute parameter-change energy density",
                "long_short_work_ratio_median_view_sign": np.median(
                    data["parent_work"][ci, :, :, 3:]
                    / data["parent_work"][ci, :, :, 1:3],
                    0,
                ),
                "limit": "Longer pulse means different schedule and relaxation, not twice the measured work",
            },
            "targets": {},
        }
        for target in ("G", "C"):
            gg = g if target == "G" else g[:, :, :, 1:] - g[:, :, :, :1]
            ff = f if target == "G" else f[:, :, :, 1:] - f[:, :, :, :1]
            e = ff[:, :, :, None] - gg[:, :, None]
            pl = np.take(e**2, PRIMARY, axis=5).mean(axis=(4, 5, 6, 7))
            rows = []
            for a, b in combinations(range(7), 2):
                delta = pl[:, :, a] - pl[:, :, b]
                first_wins = (delta < 0).all(1)
                second_wins = (delta > 0).all(1)
                rows.append(
                    {
                        "first": MODEL_IDS[a],
                        "second": MODEL_IDS[b],
                        "first_minus_second_mse_view_future": delta.mean(0),
                        "first_concordant_wins_future": first_wins.sum(0),
                        "second_concordant_wins_future": second_wins.sum(0),
                        "same_root_opposite_winner_across_futures": int(
                            (
                                first_wins[:, 0] & second_wins[:, 1]
                                | second_wins[:, 0] & first_wins[:, 1]
                            ).sum()
                        ),
                        "same_root_opposite_winner_across_views_future": (
                            delta[:, 0] * delta[:, 1] < 0
                        ).sum(0),
                    }
                )
            numerical = np.sqrt(np.mean((gg[:, 0] - gg[:, 1]) ** 2, axis=1))
            signal = np.sqrt(np.mean(gg[:, 0] ** 2, axis=1))
            diagonal = np.eye(2)
            diag_error = (e**2 * diagonal).sum(axis=(-2, -1)) / 2
            off_error = (e**2 * (1 - diagonal)).sum(axis=(-2, -1)) / 2
            np.testing.assert_allclose(
                (diag_error + off_error) / 2, (e**2).mean(axis=(-2, -1))
            )
            mean_loss = pl.mean(axis=(0, 3))
            c["targets"][target] = {
                "all_pairs_future_sensitivity": rows,
                "diagonal_mse_view_model_horizon": diag_error.mean(axis=(0, 3, 4)),
                "off_diagonal_mse_view_model_horizon": off_error.mean(axis=(0, 3, 4)),
                "numerical_rms_mean_parent_horizon_receiver_direction": numerical.mean(
                    0
                ),
                "signal_rms_mean_parent_horizon_receiver_direction": signal.mean(0),
                "roots_above_eightfold_coordinate_floor_parent_horizon_receiver_direction": (
                    signal > 8 * np.maximum(1e-06, numerical)
                ).sum(0),
                "relative_view_change_primary_mse_model": mean_loss[1] / mean_loss[0]
                - 1,
            }
        prep = analysis["contexts"][context]["preparation_error_opportunity"]
        c["complete_preparation_grid"] = [
            {
                "model": row["model"],
                "threshold": t["max_signed_response_error"],
                "nomination_future": split["selection_future"],
                "evaluation_future": split["evaluation_future"],
                "hold_roots": round(32 * split["hold_success"]["mean"]),
                "selected_roots": round(32 * split["selected_success"]["mean"]),
                "hindsight_any_parent_roots": round(
                    32 * split["hindsight_any_parent_success"]
                ),
            }
            for row in prep
            for t in row["thresholds"]
            for split in t["splits"]
        ]
        result[context] = c
    return {
        "ceiling": CEILING,
        "contexts": result,
        "model_order": MODEL_IDS,
        "parent_order": PARENTS,
        "horizons": HORIZONS,
        "native_updates": 0,
        "new_fits": 0,
    }, arrays
