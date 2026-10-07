# SPDX-License-Identifier: MPL-2.0

"""Outcome-visible feasibility reductions for retained reactor action charts.

Native delivered mass in kg and temperature in K are measured in both views.
Features end at the actual callback. Four whole-root folds and the original
finite model bank remain development diagnostics, not a new qualification.

Authenticate retained inputs and obtain separate analysis authority first.
"""

from __future__ import annotations
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import EXPLORATION_ARRAY_STEMS
from collections import Counter
from hashlib import sha256
import statistics
from typing import Any
import numpy as np
from .features import causal_features
from .models import CoefficientRows, LAMBDAS, RBF_MULTIPLIERS, fit_affine, fit_constant, fit_local, fit_rbf


def _chart(
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    domain: str,
    policy: int,
    callback: int,
    actions: list[int],
) -> tuple[dict[str, Any] | None, str]:
    if not {1, 4, 7}.issubset(actions):
        return None, "MISSING_THREE_WORD_CHART"
    donors = tuple(arrays.get(f"{EXPLORATION_ARRAY_STEMS[policy]}_v{v}_observations") for v in (0, 1))
    stages = arrays.get(f"{EXPLORATION_ARRAY_STEMS[policy]}_v0_stages")
    if any(d is None or len(d) <= callback for d in donors) or stages is None:
        return None, "MISSING_DONOR_PREFIX"
    donor_views = (
        arrays[f"{EXPLORATION_ARRAY_STEMS[policy]}_v0_observations"],
        arrays[f"{EXPLORATION_ARRAY_STEMS[policy]}_v1_observations"],
    )
    rows: list[dict[str, Any]] = []
    expected_requests = (0.0, 0.016, 0.032)
    previous_jacket = float(stages[callback - 1, 3]) if callback else 316.0
    for view in (0, 1):
        measured = []
        masses = []
        temperatures = []
        for action, requested in zip((1, 4, 7), expected_requests, strict=True):
            prefix = f"{domain}_a{action}_v{view}"
            needed = ("grid", "exposure", "stages", "request", "valid", "prefix_sha256")
            if any(f"{prefix}_{field}" not in arrays for field in needed):
                return None, "MISSING_ACTION_OR_VIEW"
            grid, exposure = arrays[f"{prefix}_grid"], arrays[f"{prefix}_exposure"]
            stage, request = arrays[f"{prefix}_stages"], arrays[f"{prefix}_request"]
            if (
                grid.shape != ((11 if view == 0 else 21), 8)
                or exposure.shape != ((10 if view == 0 else 20), 4)
                or arrays[f"{prefix}_valid"].tolist() != [1]
                or abs(float(request[0]) - requested) > 1e-12
                or abs(float(request[1]) - previous_jacket) > 1e-9
                or not np.allclose(exposure[:, 3], previous_jacket, atol=1e-9, rtol=0)
                or not np.isfinite(grid).all()
                or not np.isfinite(exposure).all()
                or not np.isfinite(stage).all()
                or not np.array_equal(
                    arrays[f"{prefix}_prefix_sha256"],
                    np.frombuffer(
                        sha256(donor_views[view][: callback + 1].tobytes()).digest(),
                        dtype=np.uint8,
                    ),
                )
            ):
                return None, "INVALID_DELIVERY_NUMERICS_OR_PREFIX"
            measured.append(float(np.max(grid[:, 1])))
            masses.append(float(np.sum(exposure[:, 1] * exposure[:, 2])))
            temperatures.append((float(np.min(grid[:, 1])), float(np.max(grid[:, 1]))))
        if abs(masses[0]) > 1e-12 or not 0 < masses[1] < masses[2]:
            return None, "ALIASED_OR_NONZERO_REFERENCE"
        rows.append(
            {
                "mass_kg": masses,
                "peak_K": measured,
                "cooling_K": [
                    0.0,
                    measured[0] - measured[1],
                    measured[0] - measured[2],
                ],
                "temperature_range_K": temperatures,
            }
        )
    if (
        max(
            abs(a - b)
            for a, b in zip(rows[0]["peak_K"], rows[1]["peak_K"], strict=True)
        )
        > 0.01
        or max(
            abs(a - b)
            for a, b in zip(rows[0]["cooling_K"], rows[1]["cooling_K"], strict=True)
        )
        > 1e-6
        or max(
            abs(a - b)
            for a, b in zip(rows[0]["mass_kg"], rows[1]["mass_kg"], strict=True)
        )
        > 1e-6
    ):
        return None, "NUMERICAL_VIEW_DISAGREEMENT"
    masses_np = np.asarray(rows[0]["mass_kg"][1:])
    contrasts = np.asarray(rows[0]["cooling_K"][1:])
    coefficient = float(np.dot(masses_np, contrasts) / np.dot(masses_np, masses_np))
    residual = float(np.max(np.abs(contrasts - coefficient * masses_np)))
    feature = causal_features(donor_views[0], stages, callback)
    return {
        "root": record["root"],
        "seed": record["seed"],
        "role": record["role"],
        "domain": domain,
        "policy": policy,
        "clock_s": callback * 10,
        "features": feature.tolist(),
        "views": rows,
        "coefficient_K_per_kg": coefficient,
        "scalar_residual_K": residual,
        "scalar_status": "SCALAR_COEFFICIENT_INADEQUATE"
        if residual > 1e-5
        else "WITHIN_RESOLUTION",
        "previous_actual_feed_kg_s": float(stages[callback - 1, 2])
        if callback
        else 0.0,
        "previous_actual_jacket_K": previous_jacket,
    }, "COMPATIBLE_CHART"


def _select_eight(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (r["clock_s"], r["domain"]))
    chosen: list[dict[str, Any]] = []
    for low, high in ((600, 1200), (7200, 9000), (14400, 16200)):
        candidate = next(
            (r for r in ordered if low <= r["clock_s"] <= high and r not in chosen),
            None,
        )
        if candidate is not None:
            chosen.append(candidate)
    prepared = next(
        (r for r in ordered if r["domain"] == "d11010" and r not in chosen), None
    )
    if prepared is not None:
        chosen.append(prepared)
    for row in ordered:
        if len(chosen) >= 8:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def _cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root_names = sorted({row["root"] for row in rows})
    fold = {root: int(root[-3:]) % 4 for root in root_names}
    results = []
    for mask_name, mask in (("current", slice(0, 6)), ("history", slice(0, 18))):
        data = CoefficientRows(
            np.asarray([row["features"][mask] for row in rows]),
            np.asarray([row["views"][0]["mass_kg"][1:] for row in rows]),
            np.asarray([row["views"][0]["cooling_K"][1:] for row in rows]),
            tuple(row["root"] for row in rows),
        )
        roster: list[tuple[str, float, float | None]] = [("K", 0.0, None)]
        roster += [("affine", lam, None) for lam in LAMBDAS]
        roster += [("rbf", lam, mult) for lam in LAMBDAS for mult in RBF_MULTIPLIERS]
        roster += [("local", lam, None) for lam in LAMBDAS]
        for family, penalty, multiplier in roster:
            losses: dict[str, float] = {}
            errors: list[str] = []
            folds: list[dict[str, Any]] = []
            for held in range(4):
                training = np.asarray(
                    [i for i, root in enumerate(data.roots) if fold[root] != held]
                )
                validation = np.asarray(
                    [i for i, root in enumerate(data.roots) if fold[root] == held]
                )
                if not len(training) or not len(validation):
                    errors.append("EMPTY_WHOLE_ROOT_FOLD")
                    continue
                fitted = data.subset(training)
                try:
                    model = (
                        fit_constant(fitted)
                        if family == "K"
                        else fit_affine(fitted, penalty)
                        if family == "affine"
                        else fit_rbf(
                            fitted,
                            penalty,
                            float(multiplier if multiplier is not None else 0),
                        )
                        if family == "rbf"
                        else fit_local(fitted, penalty)
                    )
                    held_rows = data.subset(validation)
                    losses.update(
                        held_rows.root_losses(model.predict(held_rows.features))
                    )
                    if family == "local":
                        leaves = []
                        for leaf in model.leaves:
                            leaf_mask = np.asarray(
                                [
                                    all(
                                        (row[coordinate] <= threshold) == left
                                        for coordinate, threshold, left in leaf.path
                                    )
                                    for row in fitted.features
                                ]
                            )
                            leaves.append(
                                {
                                    "path": leaf.path,
                                    "fit_root_count": len(
                                        set(np.asarray(fitted.roots)[leaf_mask])
                                    ),
                                }
                            )
                        crossings = []
                        for coordinate, threshold in {
                            (branch[0], branch[1])
                            for leaf in model.leaves
                            for branch in leaf.path
                        }:
                            column = fitted.features[:, coordinate]
                            crossings.append(
                                {
                                    "coordinate": coordinate,
                                    "threshold": threshold,
                                    "negative_1e8_crossings": int(
                                        np.sum(
                                            (column <= threshold)
                                            != (column - 1e-8 <= threshold)
                                        )
                                    ),
                                    "positive_1e8_crossings": int(
                                        np.sum(
                                            (column <= threshold)
                                            != (column + 1e-8 <= threshold)
                                        )
                                    ),
                                    "assigned_contexts": len(column),
                                }
                            )
                        folds.append(
                            {
                                "held_fold": held,
                                "leaves": leaves,
                                "split_stability": crossings,
                            }
                        )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    errors.append(f"fold-{held}:{type(exc).__name__}:{exc}")
            results.append(
                {
                    "mask": mask_name,
                    "family": family,
                    "penalty": penalty,
                    "rbf_multiplier": multiplier,
                    "root_losses_K2": losses,
                    "mean_root_loss_K2": statistics.mean(losses.values())
                    if len(losses) == len(root_names)
                    else None,
                    "fit_failures": errors,
                    "fold_models": folds,
                }
            )
    return results


REQUESTS = (0.0004, 0.0008, 0.0012, 0.0016)


def analyze_retained_inputs(
    prepared_analysis: dict[str, Any],
    global_data: dict[str, Any],
    local_data: dict[str, Any],
    feed_inputs: tuple[
        tuple[dict[str, Any], str, dict[str, np.ndarray], dict[str, Any]], ...
    ],
    local_inputs: tuple[tuple[dict[str, Any], str, dict[str, np.ndarray]], ...],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reduce authenticated feed/local records and their supplied native arrays.

    Feed inputs are (record, raw digest, arrays, derived report); local inputs are
    (record, raw digest, arrays). Retain the original 24/64/64 root census and
    ten-second charts. This outcome-visible calculation grants no qualification.
    """
    if len(feed_inputs) != 64 or len(local_inputs) != 64:
        raise ValueError("Retained feasibility requires 64 feed and 64 local roots")
    if (
        len({record["root"] for record, _, _, _ in feed_inputs}) != 64
        or len({record["root"] for record, _, _ in local_inputs}) != 64
    ):
        raise ValueError("Retained feasibility root identities are duplicated")
    q_temperature = float(prepared_analysis["calibration"][0]["value"]["q"]["decimal"])
    q_cooling = float(prepared_analysis["calibration"][1]["value"]["q"]["decimal"])
    global_roots = sorted(
        {item["root"] for item in global_data["raw_audit"]["branches"]}
    )
    if len(global_roots) != 24:
        raise ValueError("retained global development root census differs")
    census = [
        {
            "root": root,
            "source": "global_empirical",
            "chart": "INCOMPATIBLE",
            "reason": "NO_ZERO_REFERENCE_THREE_WORD_FIXED_JACKET_10S_ASSAY",
        }
        for root in global_roots
    ]
    rows: list[dict[str, Any]] = []
    boundary_entries: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for record, digest, arrays, report in feed_inputs:
        root = report["root"]
        if record["root"] != root or record["seed"] in used_seeds:
            raise ValueError("retained feed root/seed reused")
        used_seeds.add(record["seed"])
        if len(record["assays"]) != 1:
            raise ValueError("prepared-feed assay census differs")
        domain, policy, callback, actions = record["assays"][0]
        row, reason = (
            _chart(record, arrays, domain, policy, callback, actions)
            if callback is not None
            else (None, "NO_CONTACT")
        )
        census.append(
            {
                "root": root,
                "seed": record["seed"],
                "source": "prepared_feed",
                "raw_sha256": digest,
                "chart": reason,
                "context_count": int(row is not None),
                "preparation": "UNSHIFTED_EXPLORATION_PHASE_ZERO",
            }
        )
        if row is not None:
            row["source"] = "prepared_feed"
            rows.append(row)
    for record, digest, arrays in local_inputs:
        ordinal = int(record["root"][-3:])
        if record["seed"] in used_seeds:
            raise ValueError("retained local root/seed reused")
        used_seeds.add(record["seed"])
        reasons: Counter[str] = Counter()
        compatible: list[dict[str, Any]] = []
        for domain, policy, callback, actions in record["assays"]:
            if policy is None or callback is None:
                reasons["NO_CONTACT"] += 1
                continue
            if domain in ("d11010", "d11011"):
                donor_stage = arrays[f"{EXPLORATION_ARRAY_STEMS[policy]}_v0_stages"]
                boundary_entries.append(
                    {
                        "root": record["root"],
                        "domain": domain,
                        "clock_s": callback * 10,
                        "policy": policy,
                        "previous_actual_feed_kg_s": float(donor_stage[callback - 1, 2])
                        if callback
                        else 0.0,
                        "preparation_phase": ordinal % 3,
                    }
                )
            if ordinal % 3 != 0 or policy != 0:
                reasons["DIFFERENT_CAUSAL_PREPARATION"] += 1
                continue
            row, reason = _chart(record, arrays, domain, policy, callback, actions)
            reasons[reason] += 1
            if row is not None:
                row["source"] = "local_exploration_unshifted"
                compatible.append(row)
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in compatible:
            key = (
                row["clock_s"],
                tuple(row["views"][0]["mass_kg"]),
                row["previous_actual_jacket_K"],
            )
            existing = unique.get(key)
            if existing is None:
                unique[key] = row
            else:
                existing.setdefault("alias_domains", []).append(row["domain"])
        chosen = _select_eight(list(unique.values()))
        rows.extend(chosen)
        census.append(
            {
                "root": record["root"],
                "seed": record["seed"],
                "source": "local_law",
                "raw_sha256": digest,
                "preparation": "UNSHIFTED_EXPLORATION_PHASE_ZERO" if ordinal % 3 == 0 else "OTHER_PHASE",
                "compatible_contexts": len(unique),
                "selected_contexts": len(chosen),
                "omitted_contexts": len(unique) - len(chosen),
                "reasons": dict(reasons),
            }
        )
    if len(census) != 152 or len(used_seeds) != 128:
        raise ValueError("retained acquisition units or seeds differ")
    fits = _cv(rows)
    best = {}
    for mask in ("current", "history"):
        for family in ("K", "affine", "rbf", "local"):
            candidates = [
                r
                for r in fits
                if r["mask"] == mask
                and r["family"] == family
                and (r["mean_root_loss_K2"] is not None)
            ]
            best[f"{mask}:{family}"] = (
                min(
                    candidates,
                    key=lambda r: (
                        r["mean_root_loss_K2"],
                        -r["penalty"],
                        r["rbf_multiplier"] or 0,
                    ),
                )
                if candidates
                else None
            )
    smooth_candidates = [
        candidate
        for candidate in (best["history:affine"], best["history:rbf"])
        if candidate is not None
    ]
    smooth_loss = (
        min((r["mean_root_loss_K2"] for r in smooth_candidates))
        if smooth_candidates
        else None
    )
    local_best = best["history:local"]
    local_loss = local_best["mean_root_loss_K2"] if local_best is not None else None
    description = (
        "CONTINUOUS_DESCRIPTION_COMPETITIVE"
        if smooth_loss is not None
        and (local_loss is None or local_loss >= 0.8 * smooth_loss)
        else "LOCAL_OPPORTUNITY_REQUIRES_FRESH_CONFIRMATION"
        if local_loss is not None
        else "NO_STABLE_PARTITION"
    )
    prepared_rows = [r for r in rows if r["source"] == "prepared_feed"]
    residuals = [r["scalar_residual_K"] for r in rows]
    prepared_coefficients = [r["coefficient_K_per_kg"] for r in prepared_rows]
    opportunities = []
    report_lookup = {
        record["root"]: report for record, digest, arrays, report in feed_inputs
    }
    old_baseline_errors = [
        abs(
            float(report_lookup[r["root"]]["words"][0]["predicted_temperature_K"])
            - float(r["views"][view]["peak_K"][0])
        )
        for r in prepared_rows
        for view in (0, 1)
    ]
    for row in prepared_rows:
        old_words = report_lookup[row["root"]]["words"]
        if tuple((w["action"] for w in old_words)) != (1, 4, 7):
            raise ValueError("retained old-law word order differs")
        for target in REQUESTS:
            feasible = [
                i
                for i in (1, 2)
                if all(
                    (
                        view["cooling_K"][i] >= target and view["peak_K"][i] <= 356.2
                        for view in row["views"]
                    )
                )
            ]
            provisional = []
            interval_margins = []
            for i in (1, 2):
                predicted_cooling = float(old_words[i]["predicted_cooling_K"])
                predicted_temperature = float(old_words[i]["predicted_temperature_K"])
                lower_cooling = (
                    predicted_cooling
                    - q_cooling * (5e-05 + 0.5 * abs(predicted_cooling))
                    - 1e-06
                )
                upper_temperature = predicted_temperature + q_temperature * 0.25 + 0.01
                margin = min(lower_cooling - target, 356.2 - upper_temperature)
                interval_margins.append(margin)
                if margin >= 0 and abs(predicted_cooling) <= 0.01:
                    provisional.append(i)
            opportunities.append(
                {
                    "root": row["root"],
                    "request_K": target,
                    "feasible_word": min(feasible) if feasible else None,
                    "provisional_old_law_word": min(provisional)
                    if provisional
                    else None,
                    "provisional_interval_margins_K": interval_margins,
                    "old_law_qualified": False,
                    "best_margin_K": max(
                        (
                            min(
                                (view["cooling_K"][i] - target for view in row["views"])
                            )
                            for i in (1, 2)
                        )
                    ),
                }
            )
    summary = {
        "kind": "REACTOR_REGIME_RESPONSE_RETAINED_REGIME_OPPORTUNITY",
        "claim_ceiling": "EXPOSED_DEVELOPMENT_ONLY",
        "native_calls": 0,
        "assigned_retained_roots": 152,
        "root_sources": {"global_empirical": 24, "local_law": 64, "prepared_feed": 64},
        "compatible_selected_contexts": len(rows),
        "compatible_selected_roots": len({r["root"] for r in rows}),
        "prepared_contact": len(prepared_rows),
        "scalar_inadequate_contexts": sum((x > 1e-05 for x in residuals)),
        "maximum_scalar_residual_K": max(residuals),
        "prepared_coefficient_min_median_max_K_per_kg": [
            min(prepared_coefficients),
            statistics.median(prepared_coefficients),
            max(prepared_coefficients),
        ],
        "prepared_cooling_maxword_min_median_max_K": [
            min((r["views"][0]["cooling_K"][2] for r in prepared_rows)),
            statistics.median((r["views"][0]["cooling_K"][2] for r in prepared_rows)),
            max((r["views"][0]["cooling_K"][2] for r in prepared_rows)),
        ],
        "request_feasible_slots": {
            str(target): sum(
                (
                    o["request_K"] == target and o["feasible_word"] is not None
                    for o in opportunities
                )
            )
            for target in REQUESTS
        },
        "provisional_old_law_admissible_slots": {
            str(target): sum(
                (
                    o["request_K"] == target
                    and o["provisional_old_law_word"] is not None
                    for o in opportunities
                )
            )
            for target in REQUESTS
        },
        "provisional_old_law_choice_changes_from_hindsight": sum(
            (o["provisional_old_law_word"] != o["feasible_word"] for o in opportunities)
        ),
        "observation_noise_K": 0.07,
        "median_prepared_maxword_signal_over_observation_noise": statistics.median(
            (r["views"][0]["cooling_K"][2] for r in prepared_rows)
        )
        / 0.07,
        "old_law_calibration_q": {"temperature": q_temperature, "cooling": q_cooling},
        "old_baseline_absolute_error_median_max_K": [
            statistics.median(old_baseline_errors),
            max(old_baseline_errors),
        ],
        "historical_boundary": {
            domain: {
                str(feed): sum(
                    (
                        item["domain"] == domain
                        and item["previous_actual_feed_kg_s"] == feed
                        for item in boundary_entries
                    )
                )
                for feed in sorted(
                    {
                        item["previous_actual_feed_kg_s"]
                        for item in boundary_entries
                        if item["domain"] == domain
                    }
                )
            }
            for domain in ("d11010", "d11011")
        },
        "historical_boundary_entries": boundary_entries,
        "retained_description": description,
        "history_smooth_cv_loss_K2": smooth_loss,
        "history_local_cv_loss_K2": local_loss,
        "local_retained_initial_supported_callbacks": local_data["coverage"][
            "qualification"
        ]["initial_supported"],
        "local_retained_d0_residence_callbacks": local_data["all_domains"][
            "qualification"
        ]["d0"]["residence"],
        "cv_best": best,
        "limits": [
            "Retained roots are exposed; no fresh qualification or controller admission/prospective controller evaluation status.",
            "Local chart contexts with other preparation phases are censused separately.",
            "Provisional interval decisions use the old, unqualified feed law and are descriptive opportunity only.",
        ],
    }
    return (summary, census, rows, opportunities, fits)
