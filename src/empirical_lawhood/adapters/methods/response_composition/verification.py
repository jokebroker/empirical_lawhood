# SPDX-License-Identifier: MPL-2.0

"""Independent root/parent maximum response-composition error reduction.

Both numerical views and the two declared primary horizons remain nested
within 16 assigned roots. Inputs must already be authenticated and authorized.
"""

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS


def parent_max(values: np.ndarray) -> np.ndarray:
    """Explicit per-root/per-parent reduction, independent of production scoring."""
    return np.array(
        [
            [
                float(np.abs(np.take(values[r, :, p], (2, 4), axis=-2)).max())
                for p in range(5)
            ]
            for r in range(16)
        ]
    )


NAMES = ("lower", "composed", "direct", "constant_gain", "parent_context_mean")
CANDIDATES = tuple(
    (
        (family, ridge)
        for family in ("direct-i0", "direct-i1", "mechanism-i1")
        for ridge in (10.0, 1.0, 0.1)
    )
)


def audit_retained_arrays(
    old, data, predictions, identities, result, *, expected_root_ids
):
    """Verify authenticated retained response-composition development arrays.

    This is the distinct original negative 32-root development audit. It verifies
    HOLD subtraction, preservation, both numerical views, nine selected candidates,
    all whole-root folds and the two recorded stopping gates. Supplied root IDs
    must come from the authenticated source census. No input, fit, native,
    publication or qualification act is performed here.
    """
    if len(expected_root_ids) != 32 or len(set(expected_root_ids)) != 32:
        raise ValueError(
            "Response-composition audit requires the original 32-root census"
        )
    expected = old["values"][:, :, :, [0, *range(9, 17)], :, :]
    np.testing.assert_array_equal(data["original_channels"], expected)
    np.testing.assert_array_equal(
        data["response"], expected[..., :2] - expected[..., :1, :, :2]
    )
    preserved = (
        (expected[..., 2] <= 0.125)
        & (expected[..., 3] <= 0.05)
        & (expected[..., 4] <= 0.05)
    ).all(axis=(1, 3, 4)) & (old["parent_work"] <= 32).all(axis=1)
    np.testing.assert_array_equal(preserved, data["preservation"])
    assert len({entry["root_id"] for entry in identities}) == 32
    assert result["native_updates"] == 0 and result["independent_roots"] == 32
    assert result["scientific_status"] == "NO_FRESH_CONFIRMATION_OR_PROMOTION"
    root_rows, parent_rows, candidate_rows, contexts = ([], [], [], {})
    for ci, context in enumerate(("assembling", "prepared")):
        start = ci * 16
        truth = data["response"][start : start + 16]
        contrast = truth - truth[:, :, :1]
        summary = result["contexts"][context]
        widths = predictions[context + "__nominal_widths"]
        errors, contrast_errors = ({}, {})
        for name in NAMES:
            pred = predictions[context + "__" + name]
            assert pred.shape == truth.shape and np.isfinite(pred).all()
            errors[name] = parent_max(pred - truth).max(1)
            contrast_errors[name] = parent_max(pred - pred[:, :, :1] - contrast).max(1)
            np.testing.assert_array_equal(
                errors[name], summary["predictors"][name]["root_errors"]
            )
            np.testing.assert_array_equal(
                contrast_errors[name],
                summary["predictors"][name]["root_parent_contrast_errors"],
            )
        numeric = parent_max((truth[:, :1] - truth[:, 1:]).repeat(2, axis=1))
        contrast_numeric = parent_max(
            (contrast[:, :1] - contrast[:, 1:]).repeat(2, axis=1)
        )
        qualification = (
            (numeric.max(1) <= 1 / 8192)
            & (contrast_numeric.max(1) <= 1 / 8192)
            & preserved[start : start + 16].all(1)
        )
        passed = qualification & (errors["lower"] <= 1 / 256)
        passed &= (errors["composed"] <= 1 / 1024) & (
            contrast_errors["composed"] <= 1 / 1024
        )
        np.testing.assert_array_equal(passed, summary["design1_root_pass"])
        assert int(qualification.sum()) == summary["qualified_roots"]
        parent_sizes = parent_max(contrast)
        counts = (parent_sizes[:, 1:] > 1 / 512).sum(0)
        np.testing.assert_array_equal(
            counts, summary["nontrivial_contrast_counts_by_parent"]
        )
        losses = parent_max(predictions[context + "__lower"] - truth)
        point_events = (
            (losses <= 1 / 256) & (numeric <= 1 / 2048) & preserved[start : start + 16]
        )
        events = point_events & (widths[:, None] <= 1 / 256)
        np.testing.assert_array_equal(
            events, predictions[context + "__adequacy_events"]
        )
        assert int(events[:, 0].sum()) == summary["design2_hold_adequate"]
        assert int(events.any(1).sum()) == summary["design2_oracle_adequate"]
        seen: list[int] = []
        for fold, entry in enumerate(summary["selection_log"]):
            held, train = (entry["outer_held"], entry["training_roots"])
            assert held == list(range(fold, 16, 4))
            assert sorted(held + train) == list(range(16))
            assert entry["scenario_donors"] == train
            assert sorted(sum(entry["inner_folds"], [])) == sorted(train)
            selected = int(np.argmin(entry["lower_candidate_inner_loss"]))
            assert list(CANDIDATES[selected]) == entry["selected_lower"]
            np.testing.assert_array_equal(
                predictions[context + "__lower"][held],
                predictions[context + "__candidate_predictions"][selected, held],
            )
            np.testing.assert_array_equal(
                widths[held], np.full(len(held), entry["nominal_width"])
            )
            seen.extend(held)
        assert sorted(seen) == list(range(16))
        for r in range(16):
            root_id = identities[start + r]["root_id"]
            assert root_id == expected_root_ids[start + r]
            root_rows.append(
                {
                    "context": context,
                    "original_root_id": root_id,
                    "outer_fold": r % 4,
                    "response_magnitude": float(parent_max(truth)[r].max()),
                    **{name + "_error": float(errors[name][r]) for name in NAMES},
                    **{
                        name + "_parent_contrast_error": float(contrast_errors[name][r])
                        for name in NAMES
                    },
                    "nominal_lower_width": float(widths[r]),
                    "numerical_response": float(numeric[r].max()),
                    "numerical_parent_contrast": float(contrast_numeric[r].max()),
                    "qualified": bool(qualification[r]),
                    "design1_pass": bool(passed[r]),
                }
            )
            for p, parent in enumerate(PARENTS):
                parent_rows.append(
                    {
                        "context": context,
                        "original_root_id": root_id,
                        "parent": parent,
                        "lower_error": float(losses[r, p]),
                        "contrast_magnitude": float(parent_sizes[r, p]),
                        "nominal_width": float(widths[r]),
                        "preserved": bool(preserved[start + r, p]),
                        "point_accurate_and_qualified": bool(point_events[r, p]),
                        "adequate": bool(events[r, p]),
                    }
                )
            for j, (family, ridge) in enumerate(CANDIDATES):
                candidate_rows.append(
                    {
                        "context": context,
                        "original_root_id": root_id,
                        "family": family,
                        "ridge": ridge,
                        "root_error": float(
                            parent_max(
                                predictions[context + "__candidate_predictions"][j]
                                - truth
                            )[r].max()
                        ),
                    }
                )
        contexts[context] = {
            "design1_pass": int(passed.sum()),
            "qualified_roots": int(qualification.sum()),
            "point_only_adequate_by_parent": point_events.sum(0).tolist(),
            "point_only_oracle_adequate": int(point_events.any(1).sum()),
            "minimum_parent_point_error": float(losses.min()),
            "nominal_width_range": [float(widths.min()), float(widths.max())],
            "median_response_magnitude": float(np.median(parent_max(truth).max(1))),
            "parent_contrast_counts_above_twice_epsilon": counts.tolist(),
            "median_parent_contrast_by_parent": np.median(
                parent_sizes, axis=0
            ).tolist(),
            "composed_beats_direct_roots": int(
                (errors["composed"] < errors["direct"]).sum()
            ),
            "composed_beats_constant_roots": int(
                (errors["composed"] < errors["constant_gain"]).sum()
            ),
            "lower_beats_constant_roots": int(
                (errors["lower"] < errors["constant_gain"]).sum()
            ),
            "median_best_parent_point_error_reduction_from_hold": float(
                np.median(losses[:, 0] - losses.min(1))
            ),
            "maximum_best_parent_point_error_reduction_from_hold": float(
                (losses[:, 0] - losses.min(1)).max()
            ),
        }
    assert all(
        (
            c["design1_pass"] < 12 and c["point_only_oracle_adequate"] == 0
            for c in contexts.values()
        )
    )
    assert result["design_results"] == {
        "design1": "D0_STOP_COMPOSITION_NOT_READY",
        "design2": "D0_STOP_NO_ADEQUACY_OPPORTUNITY",
    }
    return {
        "contexts": contexts,
        "root_rows": root_rows,
        "parent_rows": parent_rows,
        "candidate_rows": candidate_rows,
        "design_results": result["design_results"],
        "ceiling": "POST_RUN_ARITHMETIC_NONPROMOTABLE",
        "native_updates": 0,
        "additional_fits": 0,
    }
