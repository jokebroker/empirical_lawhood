"Exposed retained preparation-policy development arithmetic; no calibration or prospective claims."

from typing import Any

import numpy as np
from ..fitting import DELTA, Normalizer
from ..preparation_policy_fitting import provisional_quantile
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction
from .science import CANDIDATES, FOLDS, inner_folds
from .fitting import FitStore, modal_operands, radial_prediction, scales
from .consumer import decisions, actual_success, joint_success


def joint_scores(
    mean: Any, sigma: Any, mhat: Any, margin: Any, sigma_m: Any, y: Any, valid: Any
) -> Any:
    response = (
        np.maximum(abs(y - mean[..., None, None]) - (DELTA / 8)[None, None, None, :, None, None], 0)
        / sigma[..., None, None]
    ).max(axis=(2, 3, 4, 5))
    entry = np.maximum(mhat - margin, 0) / sigma_m
    scores = np.maximum(response, entry)
    return np.where(valid & np.isfinite(scores), scores, np.inf)


def fixed_schedule(cstar: Any, adequacy: Any, joint: Any, work: Any) -> Any:
    counts = cstar.sum(axis=(0, 2))
    acounts = adequacy.sum(axis=0)
    jcounts = joint.sum(axis=(0, 2))
    return min(
        range(9),
        key=lambda s: (
            -int(counts[s]),
            -int(acounts[s]),
            -int(jcounts[s]),
            float(work[:, s].mean()),
            s,
        ),
    )


def verdicts(mse: Any, cstar: Any, entry: Any, known: Any, mapping: Any) -> Any:
    gates = []
    for c in range(3):
        rejected = not bool(entry[c][known].any())
        gates.append(
            (bool(mse[c] < mse[0]) and rejected, rejected, bool(cstar[c] >= 0.85), bool(mapping))
        )
    passing = [i for i, g in enumerate(gates) if all(g)]
    nominated = min(passing, key=lambda i: (-cstar[i], mse[i], i)) if passing else None
    return gates, None if nominated is None else CANDIDATES[nominated]


def compute(
    m: Any,
    trajectory: Any,
    bridge: Any,
    fits: Any,
    lower: Any,
    enriched_inputs: Any,
    direction: Any,
    requirement: Any,
    *,
    expected_known_failure_count: int | None = 10,
) -> Any:
    store = FitStore()
    out = store.arrays
    x, z, y = m["x"][..., 0], m["z"], m["y"]
    center, scale = m["center"], m["scale"]
    kernel = bridge["radial.kernel"]
    delta = (trajectory["features"][:, :, 0] - trajectory["features"][:, :1, 0]) / scale
    modal, kernel_scale, modal_design, modal_target = modal_operands(delta, kernel)
    out.update(
        modal=modal,
        kernel=kernel,
        kernel_scale=kernel_scale,
        modal_design=modal_design,
        modal_target=modal_target,
        enriched_inputs=enriched_inputs,
        folds=FOLDS,
        direction=direction,
        requirement=requirement,
    )
    margin_views = (6 - abs((z - center[None, None, :, None]) / scale[None, None, :, None])).min(
        axis=2
    )
    margin = margin_views.min(axis=-1)
    a0 = m["conjuncts"].all(axis=-1)
    known = ~m["conjuncts"][..., 0]
    if expected_known_failure_count is not None and known.sum() != expected_known_failure_count:
        raise ValueError("Known support failure census differs from ten")
    actual_q = (abs(y - m["mean"][..., None, None]) <= m["width"][..., None, None]).all(
        axis=(2, 3, 4, 5)
    )
    valid = np.isfinite(y).all(axis=(2, 3, 4, 5)) & m["conjuncts"][..., 1]
    for key in (
        "work",
        "support_faces",
        "error_ratio",
        "coverage_ratio",
        "numerical_ratio",
        "preservation_ratio",
        "maxima",
        "width",
    ):
        if key in m:
            out["actual." + key] = m[key]
    out["known_failure_indices"] = np.argwhere(known)
    actual = actual_success(y, m["work"], direction, requirement)
    out.update(
        margin_views=margin_views,
        margin=margin,
        a0=a0,
        known=known,
        actual_q=actual_q,
        actual_success=actual,
        score_valid=valid,
        conjuncts=m["conjuncts"],
    )
    outer_native = np.empty((4, 24, 9, 24))
    outer_sigma = np.empty((4, 24, 9, 4, 8))
    outer_sm = np.empty((4, 24, 9))
    qs = np.empty((4, 24))
    fixed = np.empty((4, 24), dtype=np.int64)
    for f in range(4):
        train, test = np.flatnonzero(FOLDS != f), np.flatnonzero(FOLDS == f)
        labels = inner_folds(train)
        out[f"fold.{f}.inner_labels"] = labels
        inner_native = np.empty((4, 18, 9, 24))
        saved = next(row for row in fits if row["model"] == "radial.all" and row["fold"] == f)
        if not np.array_equal(saved["training_roots"], train) or not np.array_equal(
            saved["test_roots"], test
        ):
            raise ValueError("Saved radial dependencies differ")
        norm = Normalizer(
            np.asarray(saved["normalizer_center"]), np.asarray(saved["normalizer_scale"])
        )
        operator = np.asarray(saved["operator"])
        store.save(f"outer.{f}.radial", norm, operator, train, test, "saved-radial", modal[train])
        weights = affine_prediction(operator, norm.apply(x))
        shift = radial_prediction(weights, kernel, np.asarray(saved["kernel_scale"]), scale)
        full_shift = np.einsum(
            "stk,rkj->rstj",
            kernel / np.asarray(saved["kernel_scale"]),
            weights[test].reshape(-1, 4, 24),
        )
        np.testing.assert_allclose(
            full_shift, bridge["radial.all.forecast_delta"][test], rtol=1e-10, atol=1e-10
        )
        archived = bridge["radial.all.native"]
        if archived is not None:
            np.testing.assert_allclose(
                m["upper_z"][test, :1] + shift[test], archived[test], rtol=1e-10, atol=1e-10
            )
        fitted_baseline = store.fit(
            f"outer.{f}.fitted-baseline-radial-shift", x, z[train, 0, :, 0], train, excluded=test, kind="baseline-causal-features"
        )
        np.testing.assert_allclose(fitted_baseline[test], m["upper_z"][test, 0], rtol=1e-10, atol=1e-10)
        enriched_baseline = store.fit(
            f"outer.{f}.position-momentum-enriched-baseline-radial-shift", enriched_inputs, z[train, 0, :, 0], train, excluded=test, kind="baseline-position-momentum-enriched"
        )
        outer_native[:, test] = np.stack(
            (
                fitted_baseline[test, None] + shift[test],
                x[test, None] + shift[test],
                enriched_baseline[test, None] + shift[test],
                m["upper_z"][test],
            )
        )
        out[f"fold.{f}.outer_shift"] = shift[test]
        for inner in range(3):
            itrain, itest = train[labels != inner], train[labels == inner]
            excluded = np.sort(np.r_[test, itest])
            key = f"inner.{f}.{inner}"
            weights = store.fit(
                f"{key}.radial", x, modal[itrain], itrain, excluded=excluded, kind="inner-radial"
            )
            shift = radial_prediction(weights[itest], kernel, kernel_scale, scale)
            fitted_baseline = store.fit(
                f"{key}.fitted-baseline-radial-shift", x, z[itrain, 0, :, 0], itrain, excluded=excluded, kind="baseline-causal-features"
            )[itest]
            enriched_baseline = store.fit(
                f"{key}.position-momentum-enriched-baseline-radial-shift", enriched_inputs, z[itrain, 0, :, 0], itrain, excluded=excluded, kind="baseline-position-momentum-enriched"
            )[itest]
            schedule_map = store.fit(
                f"{key}.schedule-map-comparator", x, z[itrain, :, :, 0], itrain, excluded=excluded, kind="nine-schedule-map"
            )[itest].reshape(-1, 9, 24)
            inner_native[:, labels == inner] = np.stack(
                (fitted_baseline[:, None] + shift, x[itest, None] + shift, enriched_baseline[:, None] + shift, schedule_map)
            )
        out[f"fold.{f}.inner_native"] = inner_native
        for c, name in enumerate(CANDIDATES):
            native = inner_native[c]
            mean = lower.predict(native.reshape(-1, 24)).mean.reshape(18, 9, 4, 8)
            mhat = (6 - abs((native - center) / scale)).min(axis=-1)
            sigma, sm = scales(
                store, f"scale.{f}.{name}", x, train, test, mean, mhat, y[train], margin[train]
            )
            score = joint_scores(
                mean, sigma[train], mhat, margin[train], sm[train], y[train], valid[train]
            )
            rank, q = provisional_quantile(score.max(axis=1), expected_n=18)
            if rank != 18:
                raise ValueError("Provisional rank differs")
            entry = (
                np.isfinite(q)
                & (mhat - q * sm[train] >= 0)
                & np.isfinite(x[train]).all(axis=-1)[:, None]
            )
            width = q * sigma[train] + DELTA / 8
            choice, eligible = decisions(mean, width, entry, direction[train], requirement[train])
            joint = joint_success(choice, actual[train])
            cstar = a0[train, :, None] & actual_q[train, :, None] & joint
            selected = fixed_schedule(cstar, a0[train], joint, m["work"][train])
            fixed[c, test] = selected
            qs[c, test] = q
            outer_sigma[c, test], outer_sm[c, test] = sigma[test], sm[test]
            for suffix, value in (
                ("score", score),
                ("mhat", mhat),
                ("mean", mean),
                ("sigma", sigma[train]),
                ("sigma_m", sm[train]),
                ("entry", entry),
                ("choice", choice),
                ("eligible", eligible),
                ("joint", joint),
                ("cstar", cstar),
                ("fixed", np.asarray(selected)),
            ):
                out[f"fold.{f}.{name}.{suffix}"] = value
    mean = lower.predict(outer_native.reshape(-1, 24)).mean.reshape(4, 24, 9, 4, 8)
    mhat = (6 - abs((outer_native - center) / scale)).min(axis=-1)
    lm = mhat - qs[:, :, None] * outer_sm
    entry = np.isfinite(qs[:, :, None]) & (lm >= 0) & np.isfinite(x).all(axis=-1)[None, :, None]
    width = qs[:, :, None, None, None] * outer_sigma + DELTA / 8
    choices, eligibility, joints, cstars, scores = [], [], [], [], []
    for c in range(4):
        choice, eligible = decisions(mean[c], width[c], entry[c], direction, requirement)
        joint = joint_success(choice, actual)
        choices.append(choice)
        eligibility.append(eligible)
        joints.append(joint)
        cstars.append(a0[..., None] & actual_q[..., None] & joint)
        scores.append(joint_scores(mean[c], outer_sigma[c], mhat[c], margin, outer_sm[c], y, valid))
    out.update(
        native=outer_native,
        mean=mean,
        mhat=mhat,
        lm=lm,
        entry=entry,
        width=width,
        sigma=outer_sigma,
        sigma_m=outer_sm,
        q=qs,
        rank=np.full((4, 4), 18),
        fixed=fixed,
        choice=np.asarray(choices),
        eligible=np.asarray(eligibility),
        joint=np.asarray(joints),
        cstar=np.asarray(cstars),
        scores=np.asarray(scores),
    )
    return store
