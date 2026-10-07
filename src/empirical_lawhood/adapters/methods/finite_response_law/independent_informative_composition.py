# SPDX-License-Identifier: MPL-2.0
"""Independent retained-development coefficient, scale, score and menu audit.

Callers authenticate the 48 original feature/panel roots and all saved fold
operands before entry. This outcome-visible verifier neither acquires native
observations nor publishes a qualified result or grants authority. Its explicit
linear solves reproduce the original independent falsifier, not the producer.
"""

import json
import math
from hashlib import sha256
from typing import Any

import numpy as np

from .retained import FiniteResponseLawObservedPanel


def verify_retained_informative_composition(
    panel: FiniteResponseLawObservedPanel,
    feature: dict[str, Any],
    requests: dict[str, Any],
    truth: Any,
    output: dict[str, Any],
    result_raw: bytes,
    fold_reports: tuple[dict[str, Any], ...],
    fold_arrays: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Recompute the frozen four-fold development readout from supplied operands.

    The result bytes and all arrays must already be authenticated under the
    caller's reveal/analysis authority. A returned gate is a consistency check
    on exposed development data; it grants no fresh entry or qualification.
    """
    if len(fold_reports) != 4 or len(fold_arrays) != 4:
        raise ValueError("Independent informative-composition verifier requires four saved outer folds")
    result = json.loads(result_raw)
    delta = np.asarray([0.01, 0.01, 0.0125, 0.005, 0.005, 0.0125, 0.005, 0.005])
    nu = delta / 8
    y = panel.y
    eligible = (
        (panel.observed[..., 0] & panel.valid[..., 0] & np.isfinite(y[..., 0]))
        .all(axis=(1, 4))
        .reshape(48, 32)
    )
    target = y[..., 0].mean(axis=-1).reshape(48, 5, 32)

    def solve(x: Any, response: Any, ridge: float, gamma: float = 1.0) -> Any:
        design = np.column_stack((x, np.ones(len(x))))
        penalty = np.diag(np.r_[np.full(x.shape[1], ridge), 0.0])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ response)
        coefficients *= gamma
        coefficients[-1] += (1 - gamma) * response.mean(axis=0)
        return coefficients

    def standard(values: Any) -> tuple[Any, Any]:
        center = values.mean(axis=0)
        return (center, np.maximum(np.std(values, axis=0, ddof=0), 1e-08))

    def point_check(
        a: dict[str, Any],
        prefix: str,
        roots: Any,
        recipe: dict[str, Any],
        direct: dict[str, Any],
        held: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        d = recipe["dimension"]
        x, z = (feature["x"][roots, :d, 0], feature["z"][roots, :, :d, 0])
        c0, s0 = standard(x)
        ch, sh = standard(z.reshape(-1, d))
        for name, value in (
            ("n0_center", c0),
            ("n0_scale", s0),
            ("nh_center", ch),
            ("nh_scale", sh),
        ):
            np.testing.assert_allclose(a[prefix + name], value, rtol=1e-12, atol=1e-12)
        nx, nz = ((x - c0) / s0, (z - ch) / sh)
        lower = np.empty((d + 1, 32))
        direct_op = np.empty((5, d + 1, 32))
        mean = np.empty((5, 32))
        for j in range(32):
            good = eligible[roots, j]
            response = target[roots[good], :, j]
            lower[:, j] = solve(
                nz[good].reshape(-1, d),
                response.reshape(-1, 1),
                recipe["ridge"],
                recipe["gamma"],
            )[:, 0]
            for p in range(5):
                mean[p, j] = response[:, p].mean()
                direct_op[p, :, j] = solve(
                    nx[good], response[:, p, None], direct["ridge"], direct["gamma"]
                )[:, 0]
        upper = np.stack([solve(nx, z[:, p], 10) for p in range(5)])
        for name, value in (
            ("lower", lower),
            ("direct", direct_op),
            ("cached", mean),
            ("upper", upper),
        ):
            np.testing.assert_allclose(a[prefix + name], value, rtol=1e-09, atol=1e-11)
        xh = (feature["x"][held, :d, 0] - c0) / s0
        xdesign = np.column_stack((xh, np.ones(len(held))))
        actual = (feature["z"][held, :, :d, 0] - ch) / sh
        uz = np.stack([xdesign @ upper[p] for p in range(5)], axis=1)
        predicted = (uz - ch) / sh
        means = {}
        for b, chart in (("lower", actual), ("composed", predicted)):
            means[b] = (
                np.column_stack((chart.reshape(-1, d), np.ones(len(held) * 5))) @ lower
            ).reshape(-1, 5, 4, 8)
        means["direct"] = np.stack(
            [xdesign @ direct_op[p] for p in range(5)], axis=1
        ).reshape(-1, 5, 4, 8)
        means["cached"] = np.broadcast_to(
            mean.reshape(5, 4, 8), means["lower"].shape
        ).copy()
        for value in means.values():
            value[..., 2:] = np.maximum(value[..., 2:], 0)
        ps = (np.abs(xh).max(axis=1) <= 6)[:, None]
        support = {
            "lower": np.abs(actual).max(axis=2) <= 6,
            "composed": ps & (np.abs(predicted).max(axis=2) <= 6),
            "direct": np.broadcast_to(ps, (len(held), 5)),
            "cached": np.ones((len(held), 5), dtype=bool),
        }
        return (means, support)

    folds = []
    for fold in range(4):
        report = fold_reports[fold]
        a = fold_arrays[fold]
        train, held = (a["train_roots"], a["held_roots"])
        np.testing.assert_array_equal(held, np.arange(48)[np.arange(48) % 4 == fold])
        np.testing.assert_array_equal(train, np.arange(48)[np.arange(48) % 4 != fold])
        assert len(held[held < 16]) == 4 and len(train[train < 16]) == 12
        assert report["normalizer_prefix_root_ids"] == train.tolist()
        recipe, direct = (report["recipe"], report["direct_recipe"])
        for boundary, chosen in (("lower", recipe), ("direct", direct)):
            entries = [c for c in report["candidates"] if c["boundary"] == boundary]
            best = entries[0]
            for candidate in entries:
                assert candidate["loss"] == float(np.mean(candidate["per_root"]))
                if candidate["loss"] < best["loss"] - 1e-12:
                    best = candidate
            assert chosen == best["recipe"]
        means, support = point_check(a, "point_", train, recipe, direct, held)
        for b in means:
            np.testing.assert_allclose(
                a[b + "_prediction"], means[b], rtol=1e-09, atol=1e-11
            )
            np.testing.assert_array_equal(a[b + "_support"], support[b])
            np.testing.assert_array_equal(
                output[b + "_prediction"][held], a[b + "_prediction"]
            )
        inner = np.empty(len(train), dtype=int)
        for cohort in (train < 16, train >= 16):
            inner[cohort] = np.arange(cohort.sum()) % 3
        np.testing.assert_array_equal(a["inner_folds"], inner)
        for i in range(3):
            t, h = (train[inner != i], train[inner == i])
            dependency = report["coefficient_dependencies"][i]
            assert (
                dependency["training_roots"] == t.tolist()
                and dependency["held_roots"] == h.tolist()
            )
            means, support = point_check(
                a, f"coefficient_fold_{i}_", t, recipe, direct, h
            )
            for b in means:
                np.testing.assert_allclose(
                    a[b + "_oof_prediction"][inner == i],
                    means[b],
                    rtol=1e-09,
                    atol=1e-11,
                )
                np.testing.assert_array_equal(
                    a[b + "_oof_support"][inner == i], support[b]
                )
        for b in ("lower", "composed", "cached", "direct"):
            residual = y[train] - a[b + "_oof_prediction"][..., None, None]
            scale = np.empty((4, 8))
            for k in range(4):
                for j in range(8):
                    good = eligible[train, 8 * k + j]
                    scale[k, j] = max(
                        delta[j] / 20,
                        np.sqrt(np.mean(residual[good, :, k, j, :, 0] ** 2)),
                    )
            np.testing.assert_allclose(
                a[b + "_base_scale"], scale, rtol=1e-12, atol=1e-14
            )
            full = (
                panel.observed[train] & panel.valid[train] & np.isfinite(y[train])
            ).all(axis=(1, 2, 3, 4, 5))
            logtarget = np.log(
                np.maximum(
                    0.25,
                    (np.abs(residual[full]) / scale[None, None, :, :, None, None]).max(
                        axis=(2, 3, 4, 5)
                    ),
                )
            )
            np.testing.assert_allclose(
                a[b + "_log_multiplier_target"][full], logtarget, rtol=1e-12, atol=1e-12
            )
            d = recipe["dimension"]
            x = (feature["x"][train[full], :d, 0] - a["point_n0_center"]) / a[
                "point_n0_scale"
            ]
            z = (feature["z"][train[full], :, :d, 0] - a["point_nh_center"]) / a[
                "point_nh_scale"
            ]
            if b == "cached":
                op = logtarget.mean(axis=0)
            elif b == "lower":
                op = solve(z.reshape(-1, d), logtarget.reshape(-1, 1), 10)
            else:
                op = np.stack([solve(x, logtarget[:, p, None], 10) for p in range(5)])
            np.testing.assert_allclose(
                a[b + "_log_multiplier"], op, rtol=1e-09, atol=1e-11
            )
            for population, name in ((train, "oof_"), (held, "")):
                x = (feature["x"][population, :d, 0] - a["point_n0_center"]) / a[
                    "point_n0_scale"
                ]
                z = (feature["z"][population, :, :d, 0] - a["point_nh_center"]) / a[
                    "point_nh_scale"
                ]
                if b == "cached":
                    logg = np.broadcast_to(op, (len(population), 5))
                elif b == "lower":
                    logg = (
                        np.column_stack(
                            (z.reshape(-1, d), np.ones(len(population) * 5))
                        )
                        @ op
                    ).reshape(-1, 5)
                else:
                    xd = np.column_stack((x, np.ones(len(population))))
                    logg = np.column_stack([(xd @ op[p])[:, 0] for p in range(5)])
                g = np.exp(np.clip(logg, np.log(0.25), np.log(4)))
                np.testing.assert_allclose(
                    a[b + "_" + name + "g"], g, rtol=1e-10, atol=1e-11
                )
                np.testing.assert_allclose(
                    a[b + "_" + name + "sigma"],
                    g[:, :, None, None] * scale,
                    rtol=1e-10,
                    atol=1e-12,
                )
            measured = (
                panel.observed[train] & panel.valid[train] & np.isfinite(y[train])
            ).all(axis=(2, 3, 4, 5))
            numerical = (
                np.abs(y[train, ..., 0] - y[train, ..., 1])
                <= nu[None, None, None, :, None]
            ).all(axis=(2, 3, 4))
            value = (
                np.maximum(np.abs(residual) - nu[None, None, None, :, None, None], 0)
                / a[b + "_oof_sigma"][..., None, None]
            ).max(axis=(2, 3, 4, 5))
            value = np.where(
                measured & numerical & a[b + "_oof_support"], value, np.inf
            )
            root_scores = value[train < 16].max(axis=1)
            np.testing.assert_allclose(
                root_scores, a[b + "_q2_root_scores"], rtol=1e-12, atol=1e-12
            )
            rank = math.ceil((len(root_scores) + 1) * 0.9)
            assert rank == 12 and a[b + "_q"] == sorted(root_scores)[rank - 1]
        folds.append(report)
    results = {}
    for b in ("lower", "composed", "cached", "direct"):
        mu, sigma, q = (
            output[b + "_prediction"][:16],
            output[b + "_sigma"][:16],
            output[b + "_q"][:16],
        )
        half = q[:, None, None, None] * sigma + nu
        np.testing.assert_array_equal(output[b + "_halfwidth"], half)
        selected = np.full((16, 5, 256, 2), -1, dtype=np.int64)
        failures = np.empty((16, 5, 256, 2, 8, 6), dtype=bool)
        for r in range(16):
            for p in range(5):
                for c in range(2):
                    upper, transverse = ((0.16, 0.02), (0.1, 0.01))[c]
                    for request in range(256):
                        direction = int(requests["direction"][r, request, c])
                        axis = direction // 2
                        polarity = 1 if direction % 2 == 0 else -1
                        requirement = requests["lower"][r, request, c]
                        for word in range(8):
                            pair, sign = (word // 2, -1 if word % 2 == 0 else 1)
                            m, h = (mu[r, p, pair], half[r, p, pair])
                            projected_mean = polarity * sign * m[axis]
                            bounds = np.maximum(m[2:] + h[2:], 0)
                            fail = [
                                not output[b + "_support"][r, p]
                                or not np.isfinite(q[r]),
                                bool(np.any(h > delta)),
                                projected_mean - h[axis] < requirement,
                                projected_mean + h[axis] > upper,
                                abs(m[1 - axis]) + h[1 - axis] > transverse,
                                bool(
                                    np.any(
                                        bounds
                                        > np.asarray(
                                            [0.125, 0.05, 0.05, 0.125, 0.05, 0.05]
                                        )
                                    )
                                ),
                            ]
                            failures[r, p, request, c, word] = fail
                            if not any(fail) and selected[r, p, request, c] < 0:
                                selected[r, p, request, c] = word
        np.testing.assert_array_equal(failures, output[b + "_decision_failures"])
        np.testing.assert_array_equal(selected, output[b + "_selected"])
        success = np.zeros_like(selected, dtype=bool)
        bad = np.zeros_like(selected, dtype=bool)
        for index in np.ndindex(selected.shape):
            w = selected[index]
            if w >= 0:
                success[index] = truth[*index, w]
                bad[index] = not truth[*index, w]
        joint, false = (success.all(axis=-1), bad.any(axis=-1))
        np.testing.assert_array_equal(joint, output[b + "_joint"])
        np.testing.assert_array_equal(false, output[b + "_false"])
        loss = ((mu[:, :, :, :2, None, None] - y[:16, :, :, :2]) ** 2).mean(
            axis=(1, 2, 3, 4, 5)
        ) / 0.0001
        np.testing.assert_allclose(
            loss, output[b + "_response_loss_per_root"], rtol=1e-13
        )
        expected = result["boundaries"][b]
        assert float(joint.mean()) == expected["joint_success"]["frequency"]
        assert float(false.mean()) == expected["false_admission"]["frequency"]
        assert float(loss.mean()) == expected["response_loss"]["mean"]
        results[b] = {
            "joint_success": float(joint.mean()),
            "false_admission": float(false.mean()),
            "response_loss": float(loss.mean()),
        }
    composed, cached = (results["composed"], results["cached"])
    gates = {
        "task_opportunity": True,
        "composed_joint_success": composed["joint_success"] >= 0.7,
        "composed_false_admission": composed["false_admission"] <= 0.1,
        "information": cached["response_loss"] > 0
        and composed["response_loss"] <= 0.9 * cached["response_loss"],
    }
    assert gates == result["gates"]
    report = {
        "schema": 'finite-response-law-independent-informative-composition-readout',
        "verified": True,
        "result_sha256": sha256(result_raw).hexdigest(),
        "independent_roots": 16,
        "checked_outer_folds": 4,
        "checked_coefficient_folds": 12,
        "checked_word_request_cells": 4 * 16 * 5 * 256 * 2 * 8,
        "checks": "Recomputed selected affine coefficients, training-only normalizers, native U composition, coefficient-OOF predictions/support, s/t/g/sigma, all-root q, full finite interval failures/choices, actual two-future events and response loss. Consumer enumeration does not call producer acceptance.",
        "boundaries": results,
        "gates": gates,
        "fresh_and_preparation_policy_condition": all(gates.values()),
        "new_native_updates": 0,
    }
    return report
