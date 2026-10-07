# SPDX-License-Identifier: MPL-2.0

"""Retained tier-two closure summaries and exact affine error decomposition.

Consumes the original 24-root, nine-schedule arrays. Leaves all covariance
cross terms, failed gates and outcome-visible evidence intact. No new fit,
native run, law publication or authority act is performed.
"""

import numpy as np


def summarize(a, m, mapping):
    from empirical_lawhood.adapters.methods.finite_response_law.retained_entry_readiness.screen import verdicts
    from empirical_lawhood.adapters.methods.finite_response_law.retained_entry_readiness.science import CANDIDATES, SCHEDULES

    errors = a["mean"][..., None, None] - m["y"][None]
    mse_root = np.mean(errors[:, :, 1:, :, :2] ** 2, axis=(2, 3, 4, 5, 6))
    rows = np.arange(24)
    fixed_counts = np.asarray(
        [a["cstar"][c, rows, a["fixed"][c]].sum(axis=-1) for c in range(4)]
    )
    rates = fixed_counts.mean(axis=-1) / 256
    gates, nomination = verdicts(
        mse_root.mean(axis=-1), rates, a["entry"], a["known"], mapping
    )
    result = {
        "phase": "retained-preparation-screen",
        "evaluability": "EVALUABLE" if mapping else "UNEVALUABLE",
        "scientific_status": ("SUPPORTED" if nomination else "NOT_SUPPORTED")
        if mapping
        else "UNASSESSED",
        "disposition": (
            "BASELINE_ENTRY_READINESS_ESTABLISHED"
            if nomination
            else "BASELINE_ENTRY_READINESS_NOT_ESTABLISHED"
        )
        if mapping
        else "FINITE_DECISION_COMPATIBILITY_OBSTRUCTION",
        "nomination": nomination,
        "model_nomination_eligible": nomination is not None and mapping,
        "follow_on_status": "UNENTERED",
        "candidates": {},
    }
    for c, name in enumerate(CANDIDATES):
        result["candidates"][name] = {
            "active_response_mse": mse_root[c].mean(),
            "mse_by_root": mse_root[c],
            "mse_by_cohort": [mse_root[c, :8].mean(), mse_root[c, 8:].mean()],
            "mse_leave_one_root_out": [
                (mse_root[c].sum() - v) / 23 for v in mse_root[c]
            ],
            "fixed_cstar": rates[c],
            "fixed_cstar_counts_by_root": fixed_counts[c],
            "cstar_by_cohort": [
                fixed_counts[c, :8].mean() / 256,
                fixed_counts[c, 8:].mean() / 256,
            ],
            "cstar_leave_one_root_out": [
                (fixed_counts[c].sum() - v) / (23 * 256) for v in fixed_counts[c]
            ],
            "fixed_a0_roots": int(a["a0"][rows, a["fixed"][c]].sum()),
            "fixed_q_roots": int(a["actual_q"][rows, a["fixed"][c]].sum()),
            "additional_worst_view_failures": int(
                ((a["margin"] < 0) & ~a["known"]).sum()
            ),
            "fixed_schedules": [SCHEDULES[int(s)] for s in a["fixed"][c]],
            "fixed_joint_counts_by_root": a["joint"][c, rows, a["fixed"][c]].sum(
                axis=-1
            ),
            "entry_cells": a["entry"][c].sum(),
            "known_rejected": int((~a["entry"][c][a["known"]]).sum()),
            "false_entry_cells": int((a["entry"][c] & (a["margin"] < 0)).sum()),
            "refused_consumer_requests": int((a["choice"][c] < 0).sum()),
            "gates": gates[c] if c < 3 else None,
            "q_by_fold": [a["q"][c, f] for f in range(4)],
        }
    a["mse_by_root"] = mse_root
    a["fixed_cstar_counts"] = fixed_counts
    # Exact first-two-output affine error decomposition, including all cross terms.
    b = np.stack(
        (m["upper_z"][:, 0], m["x"][..., 0], a["native"][2, :, 0], m["upper_z"][:, 0])
    )
    actual_error = m["y"][..., :2, :, :] - m["mean"][..., :2, None, None]
    op = m["operator"][:-1].reshape(24, 4, 8)[:, :, :2]
    baseline = np.einsum(
        "crj,jko->crko", (m["z"][None, :, 0, :, 0] - b) / m["scale"], op
    )
    shift = a["native"] - b[:, :, None]
    increment = np.einsum(
        "crsj,jko->crsko",
        ((m["z"][..., 0] - m["z"][:, :1, :, 0])[None] - shift) / m["scale"],
        op,
    )
    terms = np.broadcast_arrays(
        actual_error[None],
        baseline[:, :, None, :, :, None, None],
        increment[..., None, None],
    )
    a["response_error_terms"] = np.stack(terms)
    a["response_squared_cross_terms"] = np.stack(
        (
            terms[0] ** 2,
            terms[1] ** 2,
            terms[2] ** 2,
            2 * terms[0] * terms[1],
            2 * terms[0] * terms[2],
            2 * terms[1] * terms[2],
        )
    )
    np.testing.assert_allclose(
        sum(terms), -errors[..., :2, :, :], rtol=1e-10, atol=1e-10
    )
    return result
