"""Bounded informative-composition nested development evidence, without prospective qualification."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction

from .fitting import Array, BOUNDARIES, DELTA, Features, Targets, cohort_folds, coefficient_oof, fit_points, nominate
from .intervals import DECISION_REASONS, actual_events, choose, fit_widths, quantile, scores
from .retained import FiniteResponseLawObservedPanel


def event_reduction(event: ArrayLike) -> dict[str, Any]:
    values = np.asarray(event, dtype=float)
    per_request = values.mean(axis=1)
    n, requests = per_request.shape
    return {
        "frequency": float(values.mean()),
        "per_root": per_request.mean(axis=1).tolist(),
        "per_parent": values.mean(axis=(0, 2)).tolist(),
        "conditional_request_mc_variance": float(
            per_request.var(axis=1, ddof=1).sum() / (n * n * requests)
        ),
    }


def develop_fold(
    panel: FiniteResponseLawObservedPanel,
    features: Features,
    target: Targets,
    train: NDArray[np.int64],
    held: NDArray[np.int64],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(train) & set(held) or len(set(train)) != len(train):
        raise ValueError("Root leakage/duplication in development fold")
    recipe, direct_recipe, candidates = nominate(features, target, train)
    points = fit_points(features, target, train, recipe, direct_recipe)
    oof, oof_support, coefficient_fits = coefficient_oof(
        features, target, train, recipe, direct_recipe
    )
    prediction, support, uz = points.predict(features, held)
    arrays: dict[str, Any] = {
        "train_roots": train,
        "held_roots": held,
        "inner_folds": cohort_folds(train),
        "predicted_interface": uz,
        **{"point_" + k: v for k, v in points.arrays().items()},
    }
    coefficient_dependencies = []
    for i, model in enumerate(coefficient_fits):
        arrays.update({f"coefficient_fold_{i}_{k}": v for k, v in model.arrays().items()})
        coefficient_dependencies.append(
            {
                "fold": i,
                "training_roots": list(model.roots),
                "held_roots": train[cohort_folds(train) == i].tolist(),
                "normalizer_prefix_rows": len(model.roots),
                "normalizer_handoff_rows": 5 * len(model.roots),
            }
        )
    # Exact affine response-mean collapse in standardized prefix coordinates.
    native_lower = points.lower.copy()
    native_lower[:-1] /= points.nh.scale[:, None]
    native_lower[-1] -= points.nh.center @ native_lower[:-1]
    products = []
    response_columns = np.asarray([8 * k + j for k in range(4) for j in (0, 1)])
    for p in range(5):
        upper_augmented = np.column_stack((points.upper[p], np.r_[np.zeros(recipe.dimension), 1]))
        products.append(upper_augmented @ native_lower[:, response_columns])
    arrays["response_coefficient_products"] = np.stack(products)
    if len(held):
        nx = points.n0.apply(features.x[held, : recipe.dimension, 0])
        collapsed = np.stack([affine_prediction(product, nx) for product in products], axis=1)
        np.testing.assert_allclose(
            collapsed.reshape(-1, 5, 4, 2), prediction["composed"][..., :2], rtol=1e-10, atol=1e-12
        )
    boundaries = {}
    for b in BOUNDARIES:
        width = fit_widths(b, panel, features, train, points, oof[b])
        sigma_train, g_train = width.sigma(features, train, points)
        parent_scores, measured, numerical = scores(
            panel, train, oof[b], sigma_train, oof_support[b]
        )
        complete_prepared_response = train < 16
        root_scores = parent_scores[complete_prepared_response].max(axis=1)
        rank, q = quantile(root_scores)
        sigma, g = width.sigma(features, held, points)
        for name, value in {
            "prediction": prediction[b],
            "support": support[b],
            "sigma": sigma,
            "g": g,
            "oof_prediction": oof[b],
            "oof_support": oof_support[b],
            "oof_sigma": sigma_train,
            "oof_g": g_train,
            "base_scale": width.base,
            "log_multiplier": width.multiplier,
            "log_multiplier_target": width.targets,
            "log_multiplier_eligible": width.eligible_roots,
            "base_scale_root_counts": width.base_counts,
            "parent_scores": parent_scores,
            "q2_root_scores": root_scores,
            "measured_complete": measured,
            "numerical_valid": numerical,
            "q": np.asarray(q),
        }.items():
            arrays[b + "_" + name] = value
        boundaries[b] = {
            "q_rank": rank,
            "q": q,
            "q_root_ids": train[complete_prepared_response].tolist(),
            "infinite_root_scores": int(np.isinf(root_scores).sum()),
            "held_supported_parent_rows": int(support[b].sum()),
            "base_scale_root_counts": width.base_counts.tolist(),
            "g_training_range": [float(g_train.min()), float(g_train.max())],
            "g_held_range": [float(g.min()), float(g.max())] if g.size else [],
            "g_constant": bool(np.ptp(g_train) == 0),
            "g_parent_constant": [bool(np.ptp(g_train[:, p]) == 0) for p in range(5)],
        }
    report = {
        "training_roots": train.tolist(),
        "held_roots": held.tolist(),
        "recipe": recipe.__dict__,
        "direct_recipe": direct_recipe.__dict__,
        "candidates": candidates,
        "coefficient_dependencies": coefficient_dependencies,
        "normalizer_prefix_root_ids": train.tolist(),
        "normalizer_handoff_rows": [[int(r), p] for r in train for p in range(5)],
        "target_eligible_root_counts": target.eligible[train].sum(axis=0).reshape(4, 8).tolist(),
        "boundaries": boundaries,
        "response_product_ranks": [int(np.linalg.matrix_rank(p[:-1])) for p in products],
        "dependency_scope": "Recipe, s, g and q use training outcomes; coefficient OOF excludes scored roots. Every held root is excluded from all training operands. Provisional, not independent split-conformal calibration.",
    }
    return report, arrays


def summarize(
    panel: FiniteResponseLawObservedPanel,
    features: Features,
    target: Targets,
    folds: list[tuple[dict[str, Any], dict[str, Any]]],
    direction: NDArray[np.int64],
    requirement: Array,
    oracle: NDArray[np.bool_],
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for b in BOUNDARIES:
        prediction, sigma = np.empty((48, 5, 4, 8)), np.empty((48, 5, 4, 8))
        support, q = np.empty((48, 5), dtype=bool), np.empty(48)
        failures = np.empty((16, 5, 256, 2, 8, 6), dtype=bool)
        selected = np.empty((16, 5, 256, 2), dtype=np.int64)
        h = np.empty((16, 5, 4, 8))
        for report, a in folds:
            held = a["held_roots"]
            prediction[held], sigma[held], support[held] = (
                a[b + "_prediction"],
                a[b + "_sigma"],
                a[b + "_support"],
            )
            q[held] = float(a[b + "_q"])
            complete = held < 16
            roots = held[complete]
            decisions = choose(
                prediction[roots],
                sigma[roots],
                float(a[b + "_q"]),
                support[roots],
                direction[roots],
                requirement[roots],
            )
            selected[roots], failures[roots], h[roots] = (
                decisions.selected,
                decisions.failures,
                decisions.halfwidth,
            )
        joint, false, admitted = actual_events(selected, oracle)
        squared = (prediction[:16, :, :, :2, None, None] - panel.y[:16, :, :, :2]) ** 2
        response_loss = squared.mean(axis=(1, 2, 3, 4, 5)) / 0.01**2
        pres_loss = (
            (
                (prediction[:16, :, :, 2:, None, None] - panel.y[:16, :, :, 2:])
                / DELTA[None, None, None, 2:, None, None]
            )
            ** 2
        ).mean(axis=(1, 2, 3, 4, 5))
        reports[b] = {
            "joint_success": event_reduction(joint),
            "false_admission": event_reduction(false),
            "any_admission": event_reduction(admitted.any(axis=-1)),
            "both_admitted": event_reduction(admitted.all(axis=-1)),
            "conditional_on_any_admission_false_frequency": float(
                false.sum() / admitted.any(axis=-1).sum()
            )
            if admitted.any()
            else None,
            "per_consumer_refusal": [event_reduction(~admitted[..., c]) for c in (0, 1)],
            "response_loss": {
                "mean": float(response_loss.mean()),
                "per_root": response_loss.tolist(),
            },
            "preservation_normalized_loss": {
                "mean": float(pres_loss.mean()),
                "per_root": pres_loss.tolist(),
            },
            "noncompensating_word_request_failure_counts": {
                name: int(failures[..., i].sum()) for i, name in enumerate(DECISION_REASONS)
            },
            "supported_q2_parent_rows": int(support[:16].sum()),
            "precise_q2_pairs": int((h <= DELTA).all(axis=-1).sum()),
            "selected_word_counts": [int((selected == w).sum()) for w in range(-1, 8)],
            "successful_episode_magnitude_counts": {
                str(m): int(((selected // 4 == m // 8 - 1) & joint[..., None]).sum())
                for m in (8, 16)
            },
        }
        for name, value in {
            "prediction": prediction,
            "sigma": sigma,
            "support": support,
            "q": q,
            "selected": selected,
            "decision_failures": failures,
            "halfwidth": h,
            "joint": joint,
            "false": false,
            "admitted": admitted,
            "response_loss_per_root": response_loss,
            "preservation_loss_per_root": pres_loss,
        }.items():
            merged[b + "_" + name] = value
    lower = merged["lower_prediction"][:16]
    composed = merged["composed_prediction"][:16]
    el = lower[..., None, None] - panel.y[:16]
    ei = (composed - lower)[..., None, None]
    ec = composed[..., None, None] - panel.y[:16]
    np.testing.assert_allclose(el + ei, ec, rtol=1e-12, atol=1e-14)
    decomposition = {}
    for name, selection in (("response", slice(0, 2)), ("preservation", slice(2, 8))):
        a, b, c = el[:, :, :, selection], ei[:, :, :, selection], ec[:, :, :, selection]
        decomposition[name] = {
            "lower_mse": float((a * a).mean()),
            "interface_mse": float((b * b).mean()),
            "cross_term": float((2 * a * b).mean()),
            "composed_mse": float((c * c).mean()),
        }
        np.testing.assert_allclose(
            decomposition[name]["lower_mse"]
            + decomposition[name]["interface_mse"]
            + decomposition[name]["cross_term"],
            decomposition[name]["composed_mse"],
            rtol=1e-12,
            atol=1e-15,
        )
    interface_errors = []
    for report, a in folds:
        d = report["recipe"]["dimension"]
        error = a["predicted_interface"] - features.z[a["held_roots"], :, :d, 0]
        a["interface_native_error"] = error
        interface_errors.append(
            {
                "held_roots": a["held_roots"].tolist(),
                "family": report["recipe"]["family"],
                "native_rmse_per_feature": np.sqrt((error**2).mean(axis=(0, 1))).tolist(),
                "native_rmse_per_root_feature": np.sqrt((error**2).mean(axis=1)).tolist(),
            }
        )
    contrasts = {}
    for comparator in ("cached", "direct", "lower"):
        delta_j = merged["composed_joint"].astype(float) - merged[comparator + "_joint"].astype(
            float
        )
        contrasts[comparator] = {
            "paired_success_difference": event_reduction(delta_j),
            "action_disagreement": float(
                (merged["composed_selected"] != merged[comparator + "_selected"]).mean()
            ),
            "per_root_response_loss_difference": (
                merged["composed_response_loss_per_root"]
                - merged[comparator + "_response_loss_per_root"]
            ).tolist(),
        }
    cached_loss = reports["cached"]["response_loss"]["mean"]
    composed_loss = reports["composed"]["response_loss"]["mean"]
    reduction = 1 - composed_loss / cached_loss if cached_loss > 0 else None
    gates = {
        "task_opportunity": True,
        "composed_joint_success": reports["composed"]["joint_success"]["frequency"] >= 0.70,
        "composed_false_admission": reports["composed"]["false_admission"]["frequency"] <= 0.10,
        "information": cached_loss > 0 and composed_loss <= 0.9 * cached_loss,
    }
    weights = np.zeros((4, 8))
    for mask in target.eligible.reshape(48, 4, 8):
        if not mask.any():
            raise ValueError("No outputs for required root")
        weights += mask / mask.sum() / 48
    y = panel.y[:16]
    mean_future, spread = y.mean(axis=4), (y[..., 0, :] - y[..., 1, :]) ** 2 / 4
    two_future = {
        "within_future_spread_mse_by_output": spread.mean(axis=(0, 1, 2, 4)).tolist(),
        "conditional_variance_estimate_by_output": (2 * spread).mean(axis=(0, 1, 2, 4)).tolist(),
        "future_difference_exceeds_2delta_by_output": (
            np.abs(y[..., 0, :] - y[..., 1, :]) > 2 * DELTA[None, None, None, :, None]
        )
        .mean(axis=(0, 1, 2, 4))
        .tolist(),
        "all_future_view_range_exceeds_2delta_by_output": (
            np.ptp(y.reshape(16, 5, 4, 8, 4), axis=-1) > 2 * DELTA
        )
        .mean(axis=(0, 1, 2))
        .tolist(),
    }
    for b in BOUNDARIES:
        mu = merged[b + "_prediction"][:16]
        lhs = ((y - mu[..., None, None]) ** 2).mean(axis=4)
        rhs = (mean_future - mu[..., None]) ** 2 + spread
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-15)
    report = {
        "schema": "finite-response-law-informative-composition-readout",
        "operation": "informative-composition",
        "evaluability": "EVALUABLE",
        "scientific_status": "INFORMATIVE_COMPOSITION_NOMINATED"
        if all(gates.values())
        else "DEVELOPMENT_COMPOSITION_NOT_NOMINATED",
        "independent_roots": {"complete_prepared_response": 16, "partial_information_response_prediction": 32},
        "evidence_ceiling": "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
        "gates": gates,
        "fresh_and_preparation_policy_condition": all(gates.values()),
        "information_relative_loss_reduction": reduction,
        "boundaries": reports,
        "paired_contrasts": contrasts,
        "composition_error_decomposition": decomposition,
        "interface_error": interface_errors,
        "two_future_diagnostics": two_future,
        "selection_mask_weights": {
            "nominal": {"high_response": 17 / 24, "low_response": 1 / 24, "preservation": 1 / 4},
            "actual": {
                "high_response": float(weights[2:, :2].sum()),
                "low_response": float(weights[:2, :2].sum()),
                "preservation": float(weights[:, 2:].sum()),
            },
            "per_pair_output": weights.tolist(),
        },
        "interpretation": "Affine-in-feature response composition tests reusable factorization and information, not nonlinear geometric necessity; paired differential assays, not single-run physical outcomes. Development interval q is provisional and has no fresh coverage claim.",
    }
    return report, merged
