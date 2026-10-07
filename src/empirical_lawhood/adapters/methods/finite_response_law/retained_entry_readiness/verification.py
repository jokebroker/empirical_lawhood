# SPDX-License-Identifier: MPL-2.0

"""Independent saved-operand checks for the retained tier-two closure.

The fixed 24-root, nine-schedule, two-view, two-future contract remains distinct
from tier-one calibration and qualification. This verifies supplied arrays and
reports without model solves, native work, issue or publication.

Authenticate retained inputs and obtain separate analysis authority first.
"""

from collections import Counter
import numpy as np
from ..fitting import DELTA
from ..science import FiniteResponseLawScienceSpec, development_requests


def equal(actual, expected, *, tolerance=1e-10):
    if np.asarray(expected).dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def independent_decisions(mean, width, offer, directions, requirements, y, work):
    n, ns = mean.shape[:2]
    feasible = np.zeros((n, ns, 256, 2, 8), dtype=bool)
    successes = np.zeros_like(feasible)
    spec = FiniteResponseLawScienceSpec()
    caps = np.tile(np.array(spec.preservation, dtype=float), 2)
    for w in range(8):
        pair = w // 2
        lo = mean[:, :, pair] - width[:, :, pair]
        hi = mean[:, :, pair] + width[:, :, pair]
        lo[..., 2:] = np.maximum(lo[..., 2:], 0)
        hi[..., 2:] = np.maximum(hi[..., 2:], 0)
        if w % 2 == 0:
            permutation = [0, 1, 5, 6, 7, 2, 3, 4]
            oldlo, oldhi = lo.copy(), hi.copy()
            lo, hi = lo[..., permutation], hi[..., permutation]
            lo[..., :2], hi[..., :2] = -oldhi[..., :2], -oldlo[..., :2]
        source = np.isfinite(y[:, :, pair]).all(axis=(2, 3, 4))
        numerical = (
            abs(y[:, :, pair, ..., 0] - y[:, :, pair, ..., 1])
            <= (DELTA / 8)[None, None, :, None]
        ).all(axis=(2, 3))
        physical = (
            source
            & numerical
            & np.isfinite(work).all(axis=-1)
            & (work <= float(spec.parent_work_maximum)).all(axis=-1)
            & (y[:, :, pair, 2:] <= caps[None, None, :, None, None]).all(axis=(2, 3, 4))
        )
        precise = np.isfinite(width[:, :, pair]).all(axis=-1) & (
            width[:, :, pair] <= DELTA
        ).all(axis=-1)
        for c in range(2):
            for d in range(4):
                axis = d // 2
                sign = 1 if d % 2 == 0 else -1
                low = lo[..., axis] if sign == 1 else -hi[..., axis]
                high = hi[..., axis] if sign == 1 else -lo[..., axis]
                good = (
                    offer
                    & precise
                    & (hi[..., 2:] <= caps).all(axis=-1)
                    & (high <= float(spec.upper[c]))
                    & (
                        np.maximum(abs(lo[..., 1 - axis]), abs(hi[..., 1 - axis]))
                        <= float(spec.transverse[c])
                    )
                )
                eligible = good[..., None] & (
                    low[..., None] >= requirements[:, None, :, c]
                )
                mask = directions[:, None, :, c] == d
                feasible[..., c, w] |= eligible & mask
                response = y[:, :, pair, :2] * (-1 if w % 2 == 0 else 1)
                long = sign * response[:, :, axis]
                observed = (
                    physical[..., None]
                    & (long[:, :, None] >= requirements[:, None, :, c, None, None]).all(
                        axis=(-1, -2)
                    )
                    & (long <= float(spec.upper[c])).all(axis=(-1, -2))[..., None]
                    & (abs(response[:, :, 1 - axis]) <= float(spec.transverse[c])).all(
                        axis=(-1, -2)
                    )[..., None]
                )
                successes[..., c, w] |= observed & mask
    choice = np.full((n, ns, 256, 2), -1, dtype=np.int64)
    for w in reversed(range(8)):
        choice = np.where(feasible[..., w], w, choice)
    picked = np.take_along_axis(successes, np.maximum(choice, 0)[..., None], axis=-1)[
        ..., 0
    ]
    joint = ((choice >= 0) & picked).all(axis=-1)
    return feasible, choice, joint, successes


def validate_retained_inputs(m, trajectory, lower):
    """Check authenticated retained arrays against the original native cache recipe."""
    for key, shape in {
        "x": (24, 24, 2),
        "z": (24, 9, 24, 2),
        "y": (24, 9, 4, 8, 2, 2),
    }.items():
        if (
            m[key].shape != shape
            or m[key].dtype != np.float64
            or (not np.isfinite(m[key]).all())
        ):
            raise ValueError(f"UNEVALUABLE retained {key} axes/dtype/missingness")
    np.testing.assert_allclose(
        trajectory["features"][:, :, :, -1].transpose(0, 1, 3, 2),
        m["z"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lower.predict(m["z"][..., 0].reshape(-1, 24)).mean.reshape(24, 9, 4, 8),
        m["mean"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lower.predict(m["upper_z"].reshape(-1, 24)).mean.reshape(24, 9, 4, 8),
        m["upper_mean"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lower.normalize(m["z"][..., 0].reshape(-1, 24)),
        ((m["z"][..., 0] - m["center"]) / m["scale"]).reshape(-1, 24),
        rtol=0,
        atol=1e-12,
    )
    for key in ("center", "scale", "operator"):
        np.testing.assert_array_equal(trajectory[key], m[key])


def check_arrays(
    a, report, m, t, bridge, saved, lower,
    *, expected_known_failure_count: int | None = 10,
    current_requests=None,
):
    folds = np.arange(24) % 4
    equal(a["folds"], folds)
    x = m["x"][..., 0]
    z = m["z"]
    y = m["y"]
    scale = m["scale"]
    center = m["center"]
    contractions = np.real(
        (t["positions"][:, 0, 0, 0].conj() * t["momenta"][:, 0, 0, 0]).sum(
            axis=(2, 3, 4)
        )
    )
    enriched_inputs = np.column_stack((x, contractions))
    equal(a["enriched_inputs"], enriched_inputs)
    if current_requests is None:
        requests = development_requests(FiniteResponseLawScienceSpec())
        rows = np.r_[np.arange(8), 16 + np.arange(16)]
        direction = requests["direction"][rows]
        requirement = requests["lower"][rows]
    else:
        direction, requirement = current_requests
    equal(a["direction"], direction)
    equal(a["requirement"], requirement)
    views = (
        6 - abs((z - center[None, None, :, None]) / scale[None, None, :, None])
    ).min(axis=2)
    margin = views.min(axis=-1)
    equal(a["margin_views"], views)
    equal(a["margin"], margin)
    actual = lower.predict(z[..., 0].reshape(-1, 24))
    actual_mean = actual.mean.reshape(24, 9, 4, 8)
    actual_width = float(lower.q) * actual.sigma.reshape(24, 9, 4, 8) + DELTA / 8
    error = abs(y - actual_mean[..., None, None])
    spec = FiniteResponseLawScienceSpec()
    caps = np.tile(np.asarray(spec.preservation, dtype=float), 2)
    conjuncts = np.stack(
        (
            views[..., 0] >= 0,
            np.isfinite(y).all(axis=(2, 3, 4, 5))
            & (
                abs(y[..., 0] - y[..., 1]) <= (DELTA / 8)[None, None, None, :, None]
            ).all(axis=(2, 3, 4)),
            (error <= DELTA[None, None, None, :, None, None]).all(axis=(2, 3, 4, 5)),
            (actual_width <= DELTA).all(axis=(2, 3)),
            (y[:, :, :, 2:] <= caps[None, None, None, :, None, None]).all(
                axis=(2, 3, 4, 5)
            ),
            np.isfinite(m["work"]).all(axis=-1)
            & (m["work"] <= float(spec.parent_work_maximum)).all(axis=-1),
        ),
        axis=-1,
    )
    continuous = {
        "support_faces": 6
        - abs((z - center[None, None, :, None]) / scale[None, None, :, None]),
        "error_ratio": error / DELTA[None, None, None, :, None, None],
        "coverage_ratio": error / actual_width[..., None, None],
        "numerical_ratio": abs(y[..., 0] - y[..., 1])
        / (DELTA / 8)[None, None, None, :, None],
        "preservation_ratio": y[:, :, :, 2:] / caps[None, None, None, :, None, None],
        "width": actual_width,
        "work": m["work"],
    }
    for key, value in continuous.items():
        if "actual." + key in a:
            equal(a["actual." + key], value, tolerance=1e-12)
    equal(a["conjuncts"], conjuncts)
    equal(m["conjuncts"], conjuncts)
    a0 = conjuncts.all(axis=-1)
    actual_q = (error <= actual_width[..., None, None]).all(axis=(2, 3, 4, 5))
    equal(a["a0"], a0)
    equal(a["actual_q"], actual_q)
    equal(a["known"], ~conjuncts[..., 0])
    if expected_known_failure_count is not None:
        assert int(a["known"].sum()) == expected_known_failure_count
    modal_delta = (t["features"][:, :, 0] - t["features"][:, :1, 0]) / scale
    selected = bridge["radial.kernel"][1:, 2:]
    ks = np.maximum(np.sqrt((selected**2).mean(axis=(0, 1))), 1e-12)
    md = (selected / ks).reshape(-1, 4)
    mt = modal_delta[:, 1:, 2:].transpose(1, 2, 0, 3).reshape(-1, 576)
    equal(a["kernel_scale"], ks)
    equal(a["modal_design"], md)
    equal(a["modal_target"], mt)
    weights = a["modal"].reshape(24, 4, 24).transpose(1, 0, 2).reshape(4, 576)
    equal(md.T @ (md @ weights), md.T @ mt)
    predictions = {}
    counts = Counter()
    for record in report["fits"]:
        key = record["key"]
        counts[record["kind"]] += 1
        prefix = "fit." + key + "."
        train = a[prefix + "train"]
        excluded = a[prefix + "excluded"]
        assert len(train) > 0 and not np.intersect1d(train, excluded).size
        fields = key.split(".")
        f = int(fields[1])
        outer_train = np.flatnonzero(folds != f)
        np.flatnonzero(folds == f)
        if fields[0] == "inner":
            labels = np.empty(18, dtype=np.int64)
            for mask in (outer_train < 8, outer_train >= 8):
                pos = np.flatnonzero(mask)
                labels[pos] = np.arange(len(pos)) % 3
            expected_train = outer_train[labels != int(fields[2])]
        else:
            expected_train = outer_train
        equal(train, expected_train)
        equal(np.sort(excluded), np.setdiff1d(np.arange(24), train))
        xx = enriched_inputs if record["inputs"] == 26 else x
        nc = xx[train].mean(axis=0)
        ns = np.maximum(np.sqrt(((xx[train] - nc) ** 2).mean(axis=0)), 1e-8)
        equal(a[prefix + "center"], nc)
        equal(a[prefix + "scale"], ns)
        design = np.column_stack(((xx[train] - nc) / ns, np.ones(len(train))))
        op = a[prefix + "operator"]
        target = a[prefix + "target"]
        kind = record["kind"]
        if kind.endswith("radial"):
            equal(target, a["modal"][train])
        elif kind.startswith("baseline"):
            equal(target, z[train, 0, :, 0])
        elif kind == "nine-schedule-map":
            equal(target, z[train, :, :, 0].reshape(len(train), -1))
        penalty = np.eye(len(op)) * 10
        penalty[-1, -1] = 0
        equal((design.T @ design + penalty) @ op, design.T @ target)
        predictions[key] = np.column_stack(((xx - nc) / ns, np.ones(24))) @ op
    assert dict(counts) == report["fit_counts"]
    assert counts == Counter(
        {
            "baseline-causal-features": 16,
            "baseline-position-momentum-enriched": 16,
            "inner-radial": 12,
            "saved-radial": 4,
            "scale": 16,
            "nine-schedule-map": 12,
        }
    )
    for f in range(4):
        train = np.flatnonzero(folds != f)
        test = np.flatnonzero(folds == f)
        labels = np.empty(18, dtype=np.int64)
        for mask in (train < 8, train >= 8):
            pos = np.flatnonzero(mask)
            labels[pos] = np.arange(len(pos)) % 3
        equal(a[f"fold.{f}.inner_labels"], labels)

        def radial(w):
            return (
                np.einsum(
                    "sk,rkj->rsj",
                    bridge["radial.kernel"][:, -1] / ks,
                    w.reshape(-1, 4, 24),
                )
                * scale
            )

        shift = radial(predictions[f"outer.{f}.radial"][test])
        equal(shift, a[f"fold.{f}.outer_shift"])
        equal(shift / scale, bridge["radial.all.forecast_delta"][test, :, -1])
        equal(a["native"][0, test], predictions[f"outer.{f}.fitted-baseline-radial-shift"][test, None] + shift)
        equal(a["native"][1, test], x[test, None] + shift)
        equal(a["native"][2, test], predictions[f"outer.{f}.position-momentum-enriched-baseline-radial-shift"][test, None] + shift)
        equal(a["native"][3, test], m["upper_z"][test])
        inner = a[f"fold.{f}.inner_native"]
        for i in range(3):
            itest = train[labels == i]
            shift = radial(predictions[f"inner.{f}.{i}.radial"][itest])
            equal(
                inner[0, labels == i],
                predictions[f"inner.{f}.{i}.fitted-baseline-radial-shift"][itest, None] + shift,
            )
            equal(inner[1, labels == i], x[itest, None] + shift)
            equal(
                inner[2, labels == i],
                predictions[f"inner.{f}.{i}.position-momentum-enriched-baseline-radial-shift"][itest, None] + shift,
            )
            equal(
                inner[3, labels == i],
                predictions[f"inner.{f}.{i}.schedule-map-comparator"][itest].reshape(-1, 9, 24),
            )
        for c, name in enumerate(("fitted-baseline-radial-shift", "observed-features-radial-shift", "position-momentum-enriched-baseline-radial-shift", "schedule-map-comparator")):
            key = f"scale.{f}.{name}"
            pre = f"fold.{f}.{name}."
            mean = lower.predict(inner[c].reshape(-1, 24)).mean.reshape(18, 9, 4, 8)
            mh = (6 - abs((inner[c] - center) / scale)).min(axis=-1)
            residual = y[train] - mean[..., None, None]
            base = np.maximum(
                np.sqrt((residual[..., 0] ** 2).mean(axis=(0, 1, 4))), DELTA / 20
            )
            mb = max(float(np.sqrt(((mh - margin[train]) ** 2).mean())), 1e-8)
            target_y = np.log(
                np.maximum(
                    0.25,
                    (abs(residual) / base[None, None, :, :, None, None]).max(
                        axis=(2, 3, 4, 5)
                    ),
                )
            )
            target_m = np.log(np.maximum(0.25, np.maximum(mh - margin[train], 0) / mb))
            equal(a["fit." + key + ".target"], np.column_stack((target_y, target_m)))
            equal(a[key + ".response_base"], base)
            equal(a[key + ".margin_base"], mb)
            mult = np.exp(np.clip(predictions[key], np.log(0.25), np.log(4)))
            sigma = base[None, None] * mult[:, :9, None, None]
            sm = mb * mult[:, 9:]
            equal(a[pre + "mean"], mean)
            equal(a[pre + "mhat"], mh)
            equal(a[pre + "sigma"], sigma[train])
            equal(a[pre + "sigma_m"], sm[train])
            score_y = (
                np.maximum(
                    abs(residual) - (DELTA / 8)[None, None, None, :, None, None], 0
                )
                / sigma[train, ..., None, None]
            ).max(axis=(2, 3, 4, 5))
            score = np.maximum(score_y, np.maximum(mh - margin[train], 0) / sm[train])
            score = np.where(conjuncts[train, :, 1] & np.isfinite(score), score, np.inf)
            equal(a[pre + "score"], score)
            q = np.sort(score.max(axis=1))[17]
            equal(a["q"][c, test], np.full(6, q))
            offer = (
                np.isfinite(q)
                & (mh - q * sm[train] >= 0)
                & np.isfinite(x[train]).all(axis=-1)[:, None]
            )
            eligible, choice, joint, _ = independent_decisions(
                mean,
                q * sigma[train] + DELTA / 8,
                offer,
                direction[train],
                requirement[train],
                y[train],
                m["work"][train],
            )
            cs = joint & a0[train, :, None] & actual_q[train, :, None]
            for suffix, v in [
                ("entry", offer),
                ("eligible", eligible),
                ("choice", choice),
                ("joint", joint),
                ("cstar", cs),
            ]:
                equal(a[pre + suffix], v)
            best = min(
                range(9),
                key=lambda s: (
                    -int(cs[:, s].sum()),
                    -int(a0[train, s].sum()),
                    -int(joint[:, s].sum()),
                    float(m["work"][train, s].mean()),
                    s,
                ),
            )
            equal(a["fixed"][c, test], np.full(6, best, dtype=np.int64))
            equal(a["sigma"][c, test], sigma[test])
            equal(a["sigma_m"][c, test], sm[test])
    equal(a["rank"], np.full((4, 4), 18, dtype=np.int64))
    mean = lower.predict(a["native"].reshape(-1, 24)).mean.reshape(4, 24, 9, 4, 8)
    mh = (6 - abs((a["native"] - center) / scale)).min(axis=-1)
    lm = mh - a["q"][:, :, None] * a["sigma_m"]
    width = a["q"][:, :, None, None, None] * a["sigma"] + DELTA / 8
    offer = (
        np.isfinite(a["q"][:, :, None])
        & (lm >= 0)
        & np.isfinite(x).all(axis=-1)[None, :, None]
    )
    for key, v in [
        ("mean", mean),
        ("mhat", mh),
        ("lm", lm),
        ("width", width),
        ("entry", offer),
    ]:
        equal(a[key], v)
    for c in range(4):
        eligible, choice, joint, success = independent_decisions(
            mean[c], width[c], offer[c], direction, requirement, y, m["work"]
        )
        equal(a["eligible"][c], eligible)
        equal(a["choice"][c], choice)
        equal(a["joint"][c], joint)
        equal(a["actual_success"], success)
        equal(a["cstar"][c], joint & a0[..., None] & actual_q[..., None])
    mse = ((mean[..., None, None] - y[None])[:, :, 1:, :, :2] ** 2).mean(
        axis=(2, 3, 4, 5, 6)
    )
    equal(a["mse_by_root"], mse)
    equal(
        a["response_error_terms"].sum(axis=0),
        y[None, ..., :2, :, :] - mean[..., :2, None, None],
    )
    equal(
        a["response_squared_cross_terms"].sum(axis=0),
        (y[None, ..., :2, :, :] - mean[..., :2, None, None]) ** 2,
    )
    counts = np.asarray(
        [a["cstar"][c, np.arange(24), a["fixed"][c]].sum(axis=-1) for c in range(4)]
    )
    equal(a["fixed_cstar_counts"], counts)
    gates = []
    for c, name in enumerate(("fitted-baseline-radial-shift", "observed-features-radial-shift", "position-momentum-enriched-baseline-radial-shift", "schedule-map-comparator")):
        candidate = report["result"]["candidates"][name]
        equal(candidate["active_response_mse"], mse[c].mean())
        equal(candidate["fixed_cstar"], counts[c].sum() / (24 * 256), tolerance=1e-12)
        equal(candidate["fixed_cstar_counts_by_root"], counts[c])
        rejected = not offer[c][a["known"]].any()
        assert candidate["known_rejected"] == int((~offer[c][a["known"]]).sum())
        if c < 3:
            gate = [
                bool(mse[c].mean() < mse[0].mean()) and rejected,
                bool(rejected),
                bool(counts[c].mean() / 256 >= 0.85),
                True,
            ]
            assert candidate["gates"] == gate
            gates.append(gate)
    passed = [c for c in range(3) if all(gates[c])]
    nominated = (
        min(passed, key=lambda c: (-counts[c].sum(), mse[c].mean(), c))
        if passed
        else None
    )
    nomination = None if nominated is None else ("fitted-baseline-radial-shift", "observed-features-radial-shift", "position-momentum-enriched-baseline-radial-shift")[nominated]
    assert report["result"]["nomination"] == nomination
    assert report["result"]["model_nomination_eligible"] == (nomination is not None)
    return {
        "verified": True,
        "phase": "retained-preparation-screen",
        "result": report["result"],
        "fit_normal_equations_checked": len(report["fits"]),
        "model_solves": 0,
        "native_calls": 0,
        "issue_calls": 0,
        "scheduler_calls": 0,
        "tolerances": {
            "prediction_absolute_relative": 1e-10,
            "ratio_absolute_relative": 1e-12,
            "masks_counts_ranks_verdicts": "EXACT",
        },
    }
