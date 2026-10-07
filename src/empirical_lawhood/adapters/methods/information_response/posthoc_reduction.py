# SPDX-License-Identifier: MPL-2.0
"""Retained information-response post-hoc reduction with explicit array inputs.

The two contexts each contain 32 independent retained roots and 16 separate
original development roots. Features end at their measured parent handoff;
pre-parent features remain a separate comparison. All reports are outcome-visible
and nonpromotable. This module performs no storage, native or authority act.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from .models import MODEL_IDS, FITTED_IDS, central_gain
from .posthoc_statistics import CEILING, PRIMARY, RIDGES, SEED, root_loss, interval, pair_table, fit_predict, energy_test, matrix_metrics, decision_arrays, model_inputs, crossfit, prefix_crossfit

CONTEXTS = ("assembling", "prepared")
BOOTSTRAPS = 5000


def analyse_arrays(data, development, development_predictions, bank, *, root_ids):
    """Return report, arrays, root atlas and casebook; authenticate inputs first.

    ``data`` uses the original retained gains/forecasts/history/sketch/prefix and
    endpoint array axes. ``development`` contains the separate original 32-root
    two-context panel. The bank must reproduce its frozen forecasts exactly.
    Supplied root IDs retain caller identity without inventing scientific units.
    """
    if len(root_ids) != 2 or any(len(ids) != 32 for ids in root_ids):
        raise ValueError("Information post-hoc reduction requires two 32-root rosters")
    if len({value for ids in root_ids for value in ids}) != 64:
        raise ValueError("Information post-hoc roots must be independently identified")
    dev, dev_predictions = development, development_predictions
    g_all = data["gains"].swapaxes(3, 4)
    f_all = data["forecasts"].swapaxes(3, 4)
    summary, arrays, atlas, cases = {}, {}, [], []
    for ci, context in enumerate(CONTEXTS):
        rng = np.random.default_rng(SEED + ci)
        boot = rng.integers(0, 32, (BOOTSTRAPS, 32))
        g, f = (g_all[ci], f_all[ci])
        e = f[:, :, :, None] - g[:, :, None]
        primary_e = np.take(e, PRIMARY, axis=5)
        total = np.mean(primary_e**2, axis=(0, 3, 4, 5, 6, 7))
        common = primary_e.mean(4, keepdims=True)
        hold = primary_e[:, :, :, :, :1]
        delta = primary_e - hold

        def average(value):
            return np.mean(value, axis=(0, 3, 4, 5, 6, 7))

        decomposition = {
            "common_parent_error": average(common**2),
            "parent_residual_error": average((primary_e - common) ** 2),
            "hold_error": average(hold**2),
            "weighted_contrast_error": average(delta**2),
            "hold_contrast_cross_term": 2 * average(hold * delta),
            "total": total,
        }
        np.testing.assert_allclose(
            total,
            decomposition["common_parent_error"]
            + decomposition["parent_residual_error"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            total,
            decomposition["hold_error"]
            + decomposition["weighted_contrast_error"]
            + decomposition["hold_contrast_cross_term"],
            rtol=1e-12,
        )
        correction = np.take(f - f[:, :, :1], PRIMARY, axis=4)
        target_residual = np.take(g.mean(2) - f[:, :, 0], PRIMARY, axis=3)
        correction_energy = np.mean(correction**2, axis=(0, 3, 4, 5, 6))
        alignment = 2 * np.mean(
            correction * target_residual[:, :, None], axis=(0, 3, 4, 5, 6)
        )
        np.testing.assert_allclose(
            total - total[:, :1], correction_energy - alignment, rtol=1e-10, atol=1e-18
        )
        c = {
            "ceiling": CEILING,
            "assigned_roots": 32,
            "error_decomposition": decomposition,
            "correction_relative_to_L": {
                "energy": correction_energy,
                "useful_alignment_twice": alignment,
            },
            "targets": {},
            "shrinkage_grid": [],
        }
        for target in ("G", "C"):
            losses = root_loss(f, g, target)
            gg = g[:, :, :, 1:] - g[:, :, :, :1] if target == "C" else g
            ff = f[:, :, :, 1:] - f[:, :, :, :1] if target == "C" else f
            target_e = ff[:, :, :, None] - gg[:, :, None]
            variance = (gg[:, :, 0] - gg[:, :, 1]) ** 2 / 2
            cross = target_e[:, :, :, 0] * target_e[:, :, :, 1]
            np.testing.assert_allclose(
                target_e.var(axis=3, ddof=0),
                np.broadcast_to(
                    variance[:, :, None] / 2, target_e.shape[:3] + target_e.shape[4:]
                ),
                rtol=1e-10,
                atol=1e-18,
            )
            root_variance = np.take(variance, PRIMARY, axis=3).mean(axis=(2, 3, 4, 5))
            root_cross = np.take(cross, PRIMARY, axis=4).mean(axis=(3, 4, 5, 6))
            np.testing.assert_allclose(
                losses, root_variance[:, :, None] + root_cross, rtol=1e-10, atol=1e-18
            )
            coordinate_loss = np.mean(target_e**2, axis=(0, 3))
            numerical = np.sqrt(np.mean((gg[:, 0] - gg[:, 1]) ** 2, axis=(1, 2, 4, 5)))
            signal = np.sqrt(np.mean(gg**2, axis=(2, 3, 5, 6)))
            model_rows = []
            for mi, model in enumerate(MODEL_IDS):
                root = losses[:, :, mi]
                model_rows.append(
                    {
                        "model": model,
                        "mse": interval(root, boot),
                        "rmse": np.sqrt(root.mean(0)),
                        "root_loss_quantiles": np.quantile(
                            root, (0, 0.25, 0.5, 0.75, 0.9, 1), axis=0
                        ),
                        "top_one_loss_share": np.sort(root, axis=0)[-1] / root.sum(0),
                        "top_three_loss_share": np.sort(root, axis=0)[-3:].sum(0)
                        / root.sum(0),
                        "conditional_mean_error_cross_product": interval(
                            root_cross[:, :, mi], boot
                        ),
                    }
                )
                for alpha in (0, 0.25, 0.5, 0.75, 1):
                    blended = f[:, :, :1] + alpha * (f[:, :, mi : mi + 1] - f[:, :, :1])
                    c["shrinkage_grid"].append(
                        {
                            "target": target,
                            "model": model,
                            "alpha": alpha,
                            "mse": root_loss(blended, g, target).mean(0)[:, 0],
                        }
                    )
            purpose_loss = np.mean(
                np.take(target_e**2, PRIMARY, axis=5), axis=(4, 5, 6, 7)
            )
            c["targets"][target] = {
                "models": model_rows,
                "all_pairs": pair_table(losses, boot),
                "coordinate_mse_view_model_parent_horizon_receiver_direction": coordinate_loss,
                "conditional_future_variance": interval(root_variance, boot),
                "conditional_mean_oracle": "UNIDENTIFIED_FROM_TWO_FUTURES",
                "hindsight_two_future_mean_self_score": root_variance.mean(0) / 2,
                "signal_rms_by_view_horizon": signal.mean(0),
                "numerical_rms_by_horizon": numerical.mean(0),
                "purpose_model_mse_view_model_future": purpose_loss.mean(0),
                "roots_above_eightfold_floor_by_horizon": (
                    signal[:, 0] > 8 * np.maximum(1e-06, numerical)
                ).sum(0),
            }
            arrays[f"{context}__{target}__root_loss"] = losses
            arrays[f"{context}__{target}__root_variance"] = root_variance
        arrays[f"{context}__orthogonal_error_components"] = np.stack(
            (
                decomposition["common_parent_error"],
                decomposition["parent_residual_error"],
            )
        )
        sl = slice(ci * 16, (ci + 1) * 16)
        old_history, old_sketch = (dev["history"][sl], dev["sketch"][sl])
        old_g = np.stack(
            [central_gain(dev["observed"][sl, :, vi]) for vi in range(2)], axis=1
        )
        old_f = np.stack(
            [
                bank.contexts[ci]
                .predict(
                    old_history[:, :, vi].reshape(-1, 16, 12),
                    old_sketch[:, :, vi].reshape(-1, 8),
                    tuple(range(5)) * 16,
                )
                .reshape(16, 5, 7, 5, 2, 2)
                .swapaxes(1, 2)
                for vi in range(2)
            ],
            axis=1,
        )
        xo = model_inputs(old_history, old_sketch)
        xn = model_inputs(data["history"][ci], data["sketch"][ci])
        old_prior, prior = (old_f[:, :, 5], f[:, :, 5])
        training = np.arange(16)
        c["cohort_loss"] = {}
        for target in ("G", "C"):
            c["cohort_loss"][target] = {
                "training": root_loss(old_f, old_g[:, :, None], target).mean(0),
                "nested_development": dev_predictions[
                    f"{context}__outer_{('full' if target == 'G' else 'contrast')}_mse"
                ].mean(0),
                "fresh": root_loss(f, g, target).mean(0),
            }
        (
            c["ridge_sensitivity"],
            c["development_delete_one_root"],
            c["feature_diagnostics"],
        ) = ([], [], [])
        delete_predictions = np.empty((16, 32, 2, 7, 5, 5, 2, 2))
        for mi, arm in enumerate(MODEL_IDS):
            frozen = bank.contexts[ci].models[mi]
            ridge = None if frozen.ridge is None else float(frozen.ridge)
            for vi in range(2):
                reproduced = fit_predict(
                    xo[arm][:, 0],
                    old_g[:, 0],
                    old_prior[:, 0],
                    training,
                    xn[arm][:, vi],
                    prior[:, vi],
                    arm,
                    ridge,
                )
                np.testing.assert_allclose(
                    reproduced, f[:, vi, mi], rtol=1e-10, atol=1e-13
                )
            for omitted in range(16):
                tr = training[training != omitted]
                for vi in range(2):
                    delete_predictions[omitted, :, vi, mi] = fit_predict(
                        xo[arm][:, 0],
                        old_g[:, 0],
                        old_prior[:, 0],
                        tr,
                        xn[arm][:, vi],
                        prior[:, vi],
                        arm,
                        ridge,
                    )
            if arm not in FITTED_IDS:
                continue
            for r in RIDGES:
                prediction = np.stack(
                    [
                        fit_predict(
                            xo[arm][:, 0],
                            old_g[:, 0],
                            old_prior[:, 0],
                            training,
                            xn[arm][:, vi],
                            prior[:, vi],
                            arm,
                            r,
                        )
                        for vi in range(2)
                    ],
                    axis=1,
                )[:, :, None]
                c["ridge_sensitivity"].append(
                    {
                        "arm": arm,
                        "ridge": r,
                        "selected_originally": ridge == r,
                        "G_mse": root_loss(prediction, g, "G").mean(0)[:, 0],
                        "C_mse": root_loss(prediction, g, "C").mean(0)[:, 0],
                    }
                )
            mean = xo[arm][:, 0].mean(0)
            scale = np.maximum(
                (xo[arm][:, 0] - mean).reshape(-1, mean.shape[-1]).std(0), 1e-12
            )
            standardized_old = (xo[arm][:, 0] - mean) / scale
            standardized_new = (xn[arm][:, 0] - mean) / scale
            singular = np.linalg.svd(
                standardized_old.reshape(-1, mean.shape[-1]), compute_uv=False
            )
            energy = singular**2
            root_old, root_new = (
                standardized_old.reshape(16, -1),
                standardized_new.reshape(32, -1),
            )
            distances = cdist(root_new, root_old) / np.sqrt(root_old.shape[1])
            old_distance = cdist(root_old, root_old) / np.sqrt(root_old.shape[1])
            np.fill_diagonal(old_distance, np.inf)
            outside = (standardized_new < standardized_old.min(0)) | (
                standardized_new > standardized_old.max(0)
            )
            frozen_loss = root_loss(f, g, "G")[:, 0, mi]
            operator = frozen.arrays()["output_operator"][:-1]
            temporal = (
                np.sum(operator[:128].reshape(16, 8, 20) ** 2, axis=(1, 2))
                if arm in ("history", "history-with-mechanism", "mechanism-with-residual")
                else None
            )
            c["feature_diagnostics"].append(
                {
                    "arm": arm,
                    "scalar_features": mean.shape[-1],
                    "training_roots": 16,
                    "correlated_parent_rows": 80,
                    "numerical_rank": int(
                        np.linalg.matrix_rank(standardized_old.reshape(80, -1))
                    ),
                    "effective_rank_participation": energy.sum() ** 2
                    / np.sum(energy**2),
                    "effective_regression_dof_including_intercept": 1
                    + np.sum(energy / (energy + ridge)),
                    "singular_values": singular,
                    "test_nearest_training_distance": distances.min(1),
                    "training_leave_self_out_nearest_distance": old_distance.min(1),
                    "test_fraction_coordinates_outside_training_range": outside.mean(
                        (1, 2)
                    ),
                    "distance_error_spearman_descriptive": float(
                        spearmanr(distances.min(1), frozen_loss).statistic
                    ),
                    "root_level_two_sample": energy_test(root_old, root_new),
                    "coefficient_energy_by_history_sample": temporal,
                    "coefficient_energy_is_not_unique_information_due_to_correlated_features": True,
                }
            )
        for omitted in range(16):
            c["development_delete_one_root"].append(
                {
                    "omitted_development_root": omitted,
                    "G_mse": root_loss(delete_predictions[omitted], g, "G").mean(0),
                    "C_mse": root_loss(delete_predictions[omitted], g, "C").mean(0),
                }
            )
        arrays[f"{context}__development_delete_one_predictions"] = delete_predictions
        c["fresh_crossfit"], c["learning_curves"] = ({}, [])
        for selection_target in ("G", "C"):
            cv, selected = crossfit(xn, g[:, 0, 0], prior, target=selection_target)
            c["fresh_crossfit"][selection_target] = {
                "selections": selected,
                "G_mse": interval(root_loss(cv, g, "G"), boot),
                "C_mse": interval(root_loss(cv, g, "C"), boot),
            }
            arrays[f"{context}__fresh_crossfit_{selection_target}"] = cv
        for size in (8, 16, 24):
            for order in (0, 1):
                cv, selected = crossfit(xn, g[:, 0, 0], prior, size=size, order=order)
                c["learning_curves"].append(
                    {
                        "training_roots_per_fold": size,
                        "subset_order": order,
                        "selections": selected,
                        "G_mse": root_loss(cv, g, "G").mean(0),
                        "C_mse": root_loss(cv, g, "C").mean(0),
                    }
                )
                arrays[f"{context}__learning_{size}_{order}"] = cv
        prefix, selected = prefix_crossfit(
            data["prefix_history"][ci], data["prefix_sketch"][ci], g[:, 0, 0]
        )
        c["preparent_crossfit"] = {
            "model_order": ("training-mean", "snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism"),
            "selections": selected,
            "G_mse": interval(root_loss(prefix, g, "G"), boot),
            "C_mse": interval(root_loss(prefix, g, "C"), boot),
            "comparison_limit": "Joint 100-output prefix regression differs in parameter count from shared 20-output handoff regression",
        }
        arrays[f"{context}__preparent_crossfit"] = prefix
        matrix = g.mean(2)
        metrics = matrix_metrics(matrix)
        parent_change = np.linalg.norm(
            matrix[:, :, 1:] - matrix[:, :, :1], axis=(-2, -1)
        )
        parent_change_ratio = parent_change / np.maximum(
            np.linalg.norm(matrix[:, :, :1], axis=(-2, -1)), 1e-15
        )
        ends = data["endpoints"][ci]
        mid0, mid1 = (
            (ends[..., 0, :, :] + ends[..., 1, :, :]) / 2,
            (ends[..., 2, :, :] + ends[..., 3, :, :]) / 2,
        )
        midpoint = (mid0 + mid1) / 2
        midpoint_rms = np.sqrt(np.mean(midpoint**2, axis=(3, 5)))
        gain_rms = np.sqrt(np.mean(g**2, axis=(2, 5, 6)))
        midpoint_variance = np.mean(
            (midpoint[:, :, :, 0] - midpoint[:, :, :, 1]) ** 2 / 2, axis=-1
        )
        gain_variance = np.mean((g[:, :, 0] - g[:, :, 1]) ** 2 / 2, axis=(-2, -1))
        diameter = np.max(
            np.linalg.norm(
                ends[..., :, None, :, :] - ends[..., None, :, :, :], axis=-1
            ),
            axis=(-3, -2),
        )
        work = data["parent_work"][ci]
        c["response_structure"] = {
            key: {
                "root_median": np.median(value, axis=0),
                "root_min": value.min(0),
                "root_max": value.max(0),
            }
            for key, value in metrics.items()
        }
        c["response_structure"].update(
            {
                "parent_change_norm_median": np.median(parent_change, axis=0),
                "parent_change_relative_to_HOLD_median": np.median(
                    parent_change_ratio, axis=0
                ),
                "parent_change_relative_to_HOLD_max": parent_change_ratio.max(0),
                "midpoint_rms_median": np.median(midpoint_rms, axis=0),
                "gain_rms_median": np.median(gain_rms, axis=0),
                "midpoint_to_gain_rms_ratio_median": np.median(
                    midpoint_rms / np.maximum(gain_rms, 1e-15), axis=0
                ),
                "axis_midpoint_disagreement_rms": np.sqrt(
                    np.mean((mid0 - mid1) ** 2, axis=(0, 3, 5))
                ),
                "midpoint_conditional_future_variance": midpoint_variance.mean(0),
                "gain_conditional_future_variance": gain_variance.mean(0),
                "action_endpoint_diameter_median": np.median(diameter, axis=(0, 3)),
                "parent_work_mean": work.mean(0),
                "parent_work_max": work.max(0),
                "force_work_mean": data["force_work"][ci].mean(axis=(0, 2, 3, 4)),
                "work_response_correlations": [
                    float(
                        spearmanr(
                            work[:, 0, p], parent_change[:, 0, p - 1, -1]
                        ).statistic
                    )
                    for p in range(1, 5)
                ],
            }
        )
        for key, value in metrics.items():
            arrays[f"{context}__matrix_{key}"] = value
        arrays[f"{context}__parent_change_ratio"] = parent_change_ratio
        arrays[f"{context}__midpoint_rms"] = midpoint_rms
        arrays[f"{context}__gain_rms"] = gain_rms
        maximum_error = abs(primary_e).max(axis=(5, 6, 7))
        parent_loss = np.mean(primary_e**2, axis=(5, 6, 7))
        selectors = parent_loss.mean(1).argmin(-1)
        c["preparation_error_opportunity"] = []
        for mi, arm in enumerate(MODEL_IDS):
            per_parent = parent_loss[:, :, mi].mean(2)
            c["preparation_error_opportunity"].append(
                {
                    "model": arm,
                    "hold_root_mse": per_parent[:, :, 0],
                    "hindsight_best_parent_root_mse": per_parent.min(-1),
                    "mean_relative_hindsight_reduction": np.mean(
                        1 - per_parent.min(-1) / per_parent[:, :, 0], axis=0
                    ),
                    "future_selected_parent_agreement": np.mean(
                        selectors[:, mi, 0] == selectors[:, mi, 1]
                    ),
                    "thresholds": [],
                }
            )
            row = c["preparation_error_opportunity"][-1]
            for threshold in (0.001, 0.002, 0.004, 0.008):
                good = (maximum_error[:, :, mi] <= threshold).all(1)
                splits = []
                for source, target in ((0, 1), (1, 0)):
                    chosen = good[np.arange(32), target, selectors[:, mi, source]]
                    splits.append(
                        {
                            "selection_future": source,
                            "evaluation_future": target,
                            "selected_success": interval(chosen.astype(float), boot),
                            "hold_success": interval(
                                good[:, target, 0].astype(float), boot
                            ),
                            "selected_minus_hold": interval(
                                chosen.astype(float) - good[:, target, 0], boot
                            ),
                            "hindsight_any_parent_success": good[:, target]
                            .any(-1)
                            .mean(),
                        }
                    )
                row["thresholds"].append(
                    {"max_signed_response_error": threshold, "splits": splits}
                )
        arrays[f"{context}__parent_max_error"] = maximum_error
        arrays[f"{context}__future_error_selected_parent"] = selectors
        choice, regret, margin, oracle = decision_arrays(f, g)
        agreement = choice[:, :, :, None] == oracle[:, :, None]
        decision_difference = choice != choice[:, :, :1]
        gap_floor = 16 * np.maximum(
            1e-06, np.linalg.norm(g[:, 0] - g[:, 1], axis=(-2, -1))
        )
        resolved = margin[:, 0] > gap_floor[..., None]
        c["finite_decision"] = {
            "direction_degrees": np.arange(16) * 22.5,
            "mean_regret_view_model_parent_horizon": regret.mean(axis=(0, 3, 6)),
            "oracle_agreement_view_model_parent_horizon": agreement.mean(
                axis=(0, 3, 6)
            ),
            "command_difference_from_L_view_model_parent_horizon": decision_difference.mean(
                axis=(0, 5)
            ),
            "root_mean_regret": interval(regret.mean(axis=(3, 4, 5, 6)), boot),
            "resolved_oracle_gap_fraction_parent_horizon": resolved.mean(
                axis=(0, 1, 4)
            ),
            "minimum_oracle_margin": margin.min(),
            "scope": "Signed-response functional only; no absolute endpoint or generic-controller claim",
        }
        arrays[f"{context}__decision_choice"] = choice
        arrays[f"{context}__decision_regret"] = regret
        arrays[f"{context}__decision_margin"] = margin
        arrays[f"{context}__decision_oracle"] = oracle
        l_g, l_c = (root_loss(f, g, "G"), root_loss(f, g, "C"))
        feature_row = next((r for r in c["feature_diagnostics"] if r["arm"] == "history-with-mechanism"))
        for ri in range(32):
            atlas.append(
                {
                    "context": context,
                    "root_index": ri,
                    "root_id": root_ids[ci][ri],
                    "G_mse_by_view_model": l_g[ri],
                    "C_mse_by_view_model": l_c[ri],
                    "HK_nearest_development_distance": feature_row[
                        "test_nearest_training_distance"
                    ][ri],
                    "parent_change_fraction_primary_horizons_max": parent_change_ratio[
                        ri, 0, :, PRIMARY
                    ].max(),
                    "gain_condition_ratio_max": metrics["condition_ratio"][ri].max(),
                    "decision_mean_regret_by_view_model": regret[ri].mean(
                        axis=(2, 3, 4, 5)
                    ),
                    "parent_work_by_view": work[ri],
                }
            )
        difference = l_g[:, 0, 6] - l_g[:, 0, 0]
        nominations = {
            "largest_MR_gain": int(np.argmin(difference)),
            "largest_MR_harm": int(np.argmax(difference)),
            "largest_HK_extrapolation": int(
                np.argmax(feature_row["test_nearest_training_distance"])
            ),
            "largest_parent_response_change": int(
                np.argmax(parent_change_ratio[:, 0].max(axis=(1, 2)))
            ),
            "largest_history_contrast_gain": int(
                np.argmin(l_c[:, 0, 2] - l_c[:, 0, 1])
            ),
            "largest_history_contrast_harm": int(
                np.argmax(l_c[:, 0, 2] - l_c[:, 0, 1])
            ),
        }
        cases.extend(
            (
                {
                    "context": context,
                    "selection_rule": rule,
                    "root_index": ri,
                    "diagnostics": atlas[-32 + ri],
                }
                for rule, ri in nominations.items()
            )
        )
        summary[context] = c
    report = {
        "ceiling": CEILING,
        "contexts": summary,
        "native_updates": 0,
        "independent_roots": {"assembling": 32, "prepared": 32},
        "development_roots_separate": {"assembling": 16, "prepared": 16},
    }
    return report, arrays, atlas, cases
