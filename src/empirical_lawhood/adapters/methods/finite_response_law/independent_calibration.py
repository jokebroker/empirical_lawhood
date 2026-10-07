# SPDX-License-Identifier: MPL-2.0
"Independent frozen prediction, root-score and finite-menu arithmetic for finite response-law qualification.\n\nThis checker consumes authenticated panel/receipt data. It does not import the\nproducer's predictor, score, quantile, selector or qualification verdict code.\n"

from io import BytesIO
from typing import Any

import numpy as np
from .seed_commitments import FIXED_SEEDS

BOUNDARIES = ("lower", "composed", "cached", "direct")
DELTA = np.asarray([0.01, 0.01, 0.0125, 0.005, 0.005, 0.0125, 0.005, 0.005])
NU = DELTA / 8
FIXED_CALIBRATION_ROOT_IDS = tuple(f"calibration.r{r:03d}" for r in range(32))


def independent_predictions(panel: Any, coefficients: bytes) -> dict[str, Any]:
    with np.load(BytesIO(coefficients), allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    x = (panel.prefix[:, :, 0] - a["point_n0_center"]) / a["point_n0_scale"]
    z = (panel.handoff[:, :, 0] - a["point_nh_center"]) / a["point_nh_scale"]
    parents = panel.parent_indices
    xd = np.column_stack((x, np.ones(32)))
    zd = np.column_stack((z, np.ones(32)))
    handoff = np.asarray([xd[r] @ a["point_upper"][p] for r, p in enumerate(parents)])
    nz = (handoff - a["point_nh_center"]) / a["point_nh_scale"]
    means = {
        "lower": zd @ a["point_lower"],
        "composed": np.column_stack((nz, np.ones(32))) @ a["point_lower"],
        "cached": a["point_cached"][parents],
        "direct": np.asarray(
            [xd[r] @ a["point_direct"][p] for r, p in enumerate(parents)]
        ),
    }
    prefix_support = np.isfinite(x).all(axis=1) & (np.abs(x).max(axis=1) <= 6)
    supports = {
        "lower": np.isfinite(z).all(axis=1) & (np.abs(z).max(axis=1) <= 6),
        "composed": prefix_support
        & np.isfinite(nz).all(axis=1)
        & (np.abs(nz).max(axis=1) <= 6),
        "cached": np.ones(32, dtype=bool),
        "direct": prefix_support,
    }
    result: dict[str, Any] = {"predicted_handoff": handoff}
    for b in BOUNDARIES:
        mu = means[b].reshape(32, 4, 8).copy()
        mu[..., 2:] = np.maximum(mu[..., 2:], 0)
        multiplier = a[b + "_log_multiplier"]
        if b == "lower":
            logg = (zd @ multiplier)[:, 0]
        elif b == "cached":
            logg = multiplier[parents]
        else:
            logg = np.asarray(
                [(xd[r] @ multiplier[p])[0] for r, p in enumerate(parents)]
            )
        sigma = (
            np.exp(np.clip(logg, np.log(0.25), np.log(4)))[:, None, None]
            * a[b + "_base_scale"]
        )
        result[b] = (mu, sigma, supports[b])
    return result


def independent_scores(
    panel: Any, mean: Any, sigma: Any, support: Any
) -> tuple[Any, float, list[list[str]]]:
    y = panel.y[:, 0]
    measured = (panel.observed[:, 0] & panel.valid[:, 0] & np.isfinite(y)).all(
        axis=(1, 2, 3, 4)
    )
    numerical = (np.abs(y[..., 0] - y[..., 1]) <= NU[None, None, :, None]).all(
        axis=(1, 2, 3)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        residual = np.maximum(
            np.abs(y - mean[..., None, None]) - NU[None, None, :, None, None], 0
        )
        maxima = (residual / sigma[..., None, None]).reshape(32, -1).max(axis=1)
    scores, reasons = [], []
    for r in range(32):
        tags = []
        if not measured[r]:
            tags.append("MEASUREMENT_UNAVAILABLE")
        if not numerical[r]:
            tags.append("NUMERICAL_DISCREPANCY")
        if not support[r]:
            tags.append("OUTSIDE_SUPPORT")
        if not np.isfinite(maxima[r]) and not tags:
            tags.append("NONFINITE_RESIDUAL")
        scores.append(float("inf") if tags else float(maxima[r]))
        reasons.append(sorted(tags))
    return np.asarray(scores), sorted(scores)[29], reasons


def independent_requests(
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    *, request_seeds: tuple[int, ...] = (),
) -> tuple[Any, Any]:
    if len(root_ids) != 32 or len(set(root_ids)) != 32:
        raise ValueError("Independent finite response-law qualification requests require all 32 physical roots")
    namespace = root_ids[0].rsplit(".r", 1)[0]
    if root_ids != FIXED_CALIBRATION_ROOT_IDS and (
        not namespace.startswith("empirical-lawhood.finite-response-law.calibration.")
        or root_ids != tuple(f"{namespace}.r{r:03d}" for r in range(32))
    ):
        raise ValueError("Independent finite response-law qualification requests change their assigned root order")
    direction = np.empty((32, 256, 2), dtype=np.int64)
    lower = np.empty(direction.shape)
    for r, root_id in enumerate(root_ids):
        if request_seeds:
            if len(request_seeds) != 32 or any(type(s) is not int or not 0 <= s < 2**128 for s in request_seeds):
                raise ValueError("Independent requests require all committed scientific seeds")
            seed = request_seeds[r]
        elif root_id in FIXED_SEEDS:
            seed = FIXED_SEEDS[root_id]["calibration-request"]
        else:
            raise ValueError("Independent assigned requests require explicit scientific seeds")
        rng = np.random.Generator(np.random.PCG64(seed))
        direction[r] = rng.integers(0, 4, size=(256, 2))
        lower[r, :, 0] = rng.uniform(0.02, 0.12, 256)
        lower[r, :, 1] = rng.uniform(0.02, 0.06, 256)
    return direction, lower


def independent_selections(
    mean: Any,
    sigma: Any,
    support: Any,
    q: float,
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    *, request_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    direction, required = independent_requests(root_ids, request_seeds=request_seeds)
    half = q * sigma + NU
    selected = np.full((32, 256, 2), -1, dtype=np.int64)
    failures = np.zeros((32, 256, 2, 8, 6), dtype=bool)
    finite = np.isfinite(mean).all(axis=(1, 2)) & np.isfinite(sigma).all(axis=(1, 2))
    for r in range(32):
        if not finite[r]:
            failures[r, ..., 0] = True
            continue
        for consumer, (upper, transverse) in enumerate(((0.16, 0.02), (0.10, 0.01))):
            axes = direction[r, :, consumer] // 2
            sense = np.where(direction[r, :, consumer] % 2 == 0, 1, -1)
            for word in range(8):
                pair = word // 2
                sign = -1 if word % 2 == 0 else 1
                m, h = mean[r, pair], half[r, pair]
                center = sense * sign * m[axes]
                preservation = np.maximum(m[2:] + h[2:], 0)
                f = failures[r, :, consumer, word]
                f[:, 0] = not support[r] or not np.isfinite(q)
                f[:, 1] = not (h <= DELTA).all()
                f[:, 2] = center - h[axes] < required[r, :, consumer]
                f[:, 3] = center + h[axes] > upper
                f[:, 4] = np.abs(m[1 - axes]) + h[1 - axes] > transverse
                f[:, 5] = not (
                    preservation <= [0.125, 0.05, 0.05, 0.125, 0.05, 0.05]
                ).all()
                eligible = ~f.any(axis=1) & (selected[r, :, consumer] < 0)
                selected[r, eligible, consumer] = word
    admitted = selected >= 0
    joint: Any = admitted.all(axis=-1)
    return {
        "selected": selected,
        "failures": failures,
        "halfwidth": half,
        "jointly_admissible_request_pairs": int(joint.sum()),
        "jointly_admissible_request_pairs_per_root": joint.sum(axis=1).tolist(),
        "roots_with_jointly_admissible_pair": int(joint.any(axis=1).sum()),
        "joint_decision_opportunity": bool(joint.any()),
        "consumer_admission_counts": admitted.sum(axis=(0, 1)).tolist(),
        "consumer_selection_counts": [
            {str(w): int((selected[:, :, c] == w).sum()) for w in range(-1, 8)}
            for c in range(2)
        ],
        "word_refusal_counts_by_consumer": {
            name: failures[..., i].sum(axis=(0, 1, 3)).tolist()
            for i, name in enumerate(
                (
                    "unsupported_or_unavailable",
                    "selected_pair_precision",
                    "response_lower",
                    "response_upper",
                    "transverse",
                    "preservation",
                )
            )
        },
        "full_menu_precise_roots": int(
            (np.isfinite(half) & (half <= DELTA)).all(axis=(1, 2)).sum()
        ),
    }


def verify_calibration_readout(
    panel: Any,
    predictions: dict[str, Any],
    coefficients: bytes,
    calibration: Any,
    qualification: Any,
    readout: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Check the original whole-root rank and complete finite-menu readout.

    Supply separately authenticated panel, coefficient and sole qualification
    records after reveal/analysis authorization. This independent check grants
    no source contact, candidate publication or qualification authority.
    """
    from collections import Counter

    from empirical_lawhood.kernel.status import ScientificStatus

    if qualification.calibration != calibration:
        raise ValueError("Qualification changed the frozen calibration result")
    independent = independent_predictions(panel, coefficients)
    np.testing.assert_allclose(
        predictions["predicted_handoff"],
        independent["predicted_handoff"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        predictions["assigned_parent_indices"], panel.parent_indices
    )
    boundaries = {}
    decision_arrays = {}
    for b, record, law in zip(
        BOUNDARIES, calibration.boundaries, qualification.qualifications, strict=True
    ):
        mean, sigma, support = (
            predictions[f"{b}_{field}"] for field in ("mean", "sigma", "supported")
        )
        for observed, expected in zip(
            (mean, sigma, support), independent[b], strict=True
        ):
            np.testing.assert_allclose(
                observed, expected, rtol=1e-10, atol=1e-12, equal_nan=True
            )
        scores, q, reasons = independent_scores(panel, mean, sigma, support)
        recorded_scores = np.asarray(
            [float("inf") if v is None else float(v) for v in record.scores]
        )
        np.testing.assert_allclose(scores, recorded_scores, rtol=1e-12, atol=1e-12)
        if (
            reasons != [list(v) for v in record.score_reasons]
            or record.order_index != 30
            or q != (float("inf") if record.q is None else float(record.q))
        ):
            raise ValueError(
                "Finite response-law qualification root reasons/order-statistic differ from independent arithmetic"
            )
        expected_status = (
            ScientificStatus.SUPPORTED
            if np.isfinite(q)
            else ScientificStatus.NOT_SUPPORTED
        )
        if law.scientific_status is not expected_status:
            raise ValueError(
                "Sole qualification result differs from the frozen finite-q gate"
            )
        decisions = independent_selections(mean, sigma, support, q, panel.root_ids, request_seeds=panel.request_seeds)
        decision_arrays[f"{b}_selected"] = decisions["selected"]
        usability = readout["usability"][b]
        for key, value in decisions.items():
            if (
                key not in ("selected", "failures", "halfwidth")
                and value != usability[key]
            ):
                raise ValueError(f'Finite response-law qualification complete finite-menu census differs: {b}/{key}')
        if (
            dict(calibration.joint_opportunities)[b]
            != decisions["joint_decision_opportunity"]
        ):
            raise ValueError(
                "Finite response-law qualification opportunity summary differs from the fixed request census"
            )
        boundaries[b] = {
            "q": None if not np.isfinite(q) else q,
            "qualification": law.scientific_status.value,
            "finite_score_roots": int(np.isfinite(scores).sum()),
            "infinite_score_root_ids": [
                panel.root_ids[i] for i in range(32) if not np.isfinite(scores[i])
            ],
            "score_reason_counts": dict(
                Counter((reason for row in reasons for reason in row))
            ),
            "jointly_admissible_request_pairs": decisions[
                "jointly_admissible_request_pairs"
            ],
            "roots_with_jointly_admissible_pair": decisions[
                "roots_with_jointly_admissible_pair"
            ],
            "consumer_admission_counts": decisions["consumer_admission_counts"],
            "consumer_selection_counts": decisions["consumer_selection_counts"],
            "word_refusal_counts_by_consumer": decisions[
                "word_refusal_counts_by_consumer"
            ],
            "nominal_halfwidth_by_output": usability["nominal_halfwidth_by_output"],
        }
    eligible = (
        all(
            (
                boundaries[b]["qualification"] == "SUPPORTED"
                for b in ("lower", "composed")
            )
        )
        and boundaries["composed"]["jointly_admissible_request_pairs"] > 0
    )
    if eligible != qualification.eligible_for_prospective_evaluation:
        raise ValueError("Finite response-law qualification terminal eligibility differs from its declared gates")
    return (boundaries, decision_arrays, eligible)
