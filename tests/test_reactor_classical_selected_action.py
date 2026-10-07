"""Selected-action receiver and root denominator, with independent mass arithmetic."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from scipy.stats import binom

from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import ROOTS, ClassicalDesign
from empirical_lawhood.adapters.methods.reactor_selected_action_response.measurement import measure_root, qualify_operands
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalAssay, ClassicalCausal, ClassicalDecision
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _paired_selected_action():
    design = ClassicalDesign()
    recipe = ObjectIdentity.from_record(design.config_id, design)
    root, role, _, seed = ROOTS[0]
    callback = 267
    causal_arrays = {}
    assay_arrays = {}
    for view, dt in ((0, 1.0), (1, 0.5)):
        causal_arrays[f"exploration_unshifted_v{view}_observations"] = np.column_stack(
            (
                np.arange(callback + 1) * 10,
                np.full(callback + 1, 350.0),
                np.full(callback + 1, 350.0),
                np.arange(callback + 1) * 0.16,
            )
        )
        causal_arrays[f"exploration_unshifted_v{view}_requests"] = np.tile((0.016, 350.0), (callback, 1))
        causal_arrays[f"exploration_unshifted_v{view}_stages"] = np.tile(
            (0.016, 350.0, 0.016, 350.0), (callback, 1)
        )
        causal_arrays[f"exploration_unshifted_v{view}_exposure"] = np.zeros((callback, int(10 / dt), 4))
        assay_arrays[f"v{view}_prefix_peak_K"] = np.array((350.0,))
        for word, feed in ((0, 0.0), (1, 0.016)):
            name = f"v{view}_a{word}"
            times = callback * 10 + np.arange(int(10 / dt) + 1) * dt
            grid = np.zeros((len(times), 8))
            grid[:, 0] = times
            grid[:, 1] = 350.0 + np.linspace(0, 0.01 - 0.002 * word, len(times))
            assay_arrays[f"{name}_grid"] = grid
            assay_arrays[f"{name}_requests"] = np.array((feed, 350.0))
            assay_arrays[f"{name}_stages"] = np.array((feed, 350.0) * 2)
            assay_arrays[f"{name}_exposure"] = np.column_stack(
                (
                    times[:-1],
                    np.full(len(times) - 1, dt),
                    np.full(len(times) - 1, feed),
                    np.full(len(times) - 1, 350.0),
                )
            )
            assay_arrays[f"{name}_valid"] = np.ones(1, dtype=np.uint8)
            for digest in ("prefix_sha256", "full_grid_sha256"):
                assay_arrays[f"{name}_{digest}"] = np.ones(32, dtype=np.uint8)
    causal = ClassicalCausal(
        root,
        role,
        seed,
        recipe,
        "0" * 64,
        callback,
        LocalArrayPayload.pack(causal_arrays),
        2,
        (),
    )
    parent = ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
    decision = ClassicalDecision(
        root,
        parent,
        recipe,
        None,
        callback,
        True,
        Decimal(350),
        ObjectIdentity(
            "synthetic-test-execution",
            'empirical-lawhood/planning/durable-authorization-record',
            "1.0.0",
            "1" * 64,
        ),
        (),
    )
    assay = ClassicalAssay(
        root,
        parent,
        ObjectIdentity.from_record(decision.decision_id, decision),
        LocalArrayPayload.pack(assay_arrays),
        4,
        (),
    )
    return causal, decision, assay


def test_selected_action_mass_effect_two_views_and_stage_refusal() -> None:
    causal, decision, assay = _paired_selected_action()
    score = measure_root(causal, decision, assay)
    assert score.evaluable and score.adequate and score.safe
    assert tuple(float(v) for v in score.cooling_K) == pytest.approx((0.002, 0.002))
    # Independent integral: 0.016 kg/s for ten seconds in either view.
    assert tuple(float(v) for v in score.mass_kg) == pytest.approx((0.16, 0.16))
    data = assay.arrays.unpack()
    data["v1_a1_stages"][2] = 0.032
    refused = measure_root(
        causal, decision, replace(assay, arrays=LocalArrayPayload.pack(data))
    )
    assert refused.evaluable and not refused.adequate
    assert "WRONG_REQUESTED_ACCEPTED_APPLIED_REALIZED_WORD" in refused.reasons
    data = assay.arrays.unpack()
    data["v0_a1_exposure"][:, 2] *= 0.5
    refused_mass = measure_root(
        causal, decision, replace(assay, arrays=LocalArrayPayload.pack(data))
    )
    assert refused_mass.evaluable and not refused_mass.adequate
    assert "WRONG_REQUESTED_ACCEPTED_APPLIED_REALIZED_WORD" in refused_mass.reasons


def test_qualification_counts_roots_and_exact_one_sided_binomial_tail() -> None:
    causal, decision, assay = _paired_selected_action()
    positive = measure_root(causal, decision, assay)
    roots = [root for root, role, _, _ in ROOTS if role == "qualification"]
    assays = tuple(replace(assay, root=root) for root in roots)
    scores = tuple(
        replace(
            positive,
            root=root,
            adequate=i < 62,
            reasons=() if i < 62 else ("SELECTED_RESPONSE_BOUND_FAILED",),
        )
        for i, root in enumerate(roots)
    )
    qualified = qualify_operands(assays, scores)
    assert qualified.successes == 62 and len(qualified.roots) == 64
    assert qualified.qualifies and qualified.evaluable
    lower = float(qualified.lower_95)
    assert binom.sf(61, 64, lower) == pytest.approx(0.05, abs=1e-12)
    near = qualify_operands(
        assays,
        tuple(
            replace(row, adequate=False, reasons=("SELECTED_RESPONSE_BOUND_FAILED",))
            if i == 61
            else row
            for i, row in enumerate(scores)
        ),
    )
    assert near.successes == 61 and not near.qualifies
    with pytest.raises(ValueError, match="denominator"):
        qualify_operands(assays, scores[:-1])
