"""Outcome-visible finite-word diagnostics; no acquisition or policy promotion.

SPDX-License-Identifier: MPL-2.0

The source's signed chart and separate consumer receiver limits are preserved.
Current requests and bootstrap allocations are explicit numerical operands.
"""

import numpy as np
from scipy.stats import beta

from empirical_lawhood.adapters.methods.preparation_applicability.science import DELTA, CAPS

THRESHOLDS = (5.999, 6.0, 6.0025, 6.01, 6.2, 6.5)

def feasible_words(y, work, direction, requirement):
    """Outcome-visible availability of the *declared paired contrast* receiver.

    This is neither an admission decision nor simultaneous physical delivery.
    Each consumer/word is assessed on its retained separate paired branches.
    """
    if y.shape != (3, 4, 8, 2, 2) or work.shape != (3, 2):
        raise ValueError("counterfactual requires the complete native paired chart")
    if direction.shape != (256, 2) or requirement.shape != (256, 2):
        raise ValueError("counterfactual changes the assigned consumer census")
    delta, caps = np.asarray(DELTA), np.asarray(CAPS)
    feasible = np.zeros((3, 256, 2, 8), dtype=bool)
    for s in range(3):
        for consumer in (0, 1):
            axis = direction[:, consumer] // 2
            polarity = np.where(direction[:, consumer] % 2 == 0, 1, -1)
            upper, transverse = ((0.16, 0.02), (0.10, 0.01))[consumer]
            for word in range(8):
                pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
                values = y[s, pair]
                source_valid = (
                    np.isfinite(values).all()
                    and (abs(values[..., 0] - values[..., 1]) <= (delta / 8)[:, None]).all()
                    and (values[2:] <= caps[:, None, None]).all()
                    and np.isfinite(work[s]).all()
                    and (work[s] <= 32).all()
                )
                response = sign * values[:2]
                longitudinal = polarity[:, None, None] * response[axis]
                feasible[s, :, consumer, word] = (
                    source_valid
                    & (longitudinal >= requirement[:, consumer, None, None]).all(axis=(1, 2))
                    & (longitudinal <= upper).all(axis=(1, 2))
                    & (abs(response[1 - axis]) <= transverse).all(axis=(1, 2))
                )
    return feasible


def selected_success(feasible, choices):
    if choices.shape != feasible.shape[:-1] or np.any((choices < -1) | (choices > 7)):
        raise ValueError("choice escapes word chart or NONATTEMPT")
    return np.take_along_axis(feasible, np.maximum(choices, 0)[..., None], axis=3)[..., 0] & (
        choices >= 0
    )


def aligned_words(direction, magnitude=8):
    if magnitude not in (8, 16) or np.any((direction < 0) | (direction > 3)):
        raise ValueError("fixed-word diagnostic changes native direction/magnitude")
    # Requests order +x,-x,+y,-y; native signed words order -x,+x,-y,+y.
    return np.broadcast_to(direction ^ 1, (3, 256, 2)).copy() + (4 if magnitude == 16 else 0)


def counts(success, choices=None):
    result = {"joint_pairs": success.all(axis=2).sum(axis=1).tolist()}
    if choices is not None:
        result.update(
            admitted_individual=(choices >= 0).sum(axis=(1, 2)).tolist(),
            admitted_pairs=(choices >= 0).all(axis=2).sum(axis=1).tolist(),
            false_admissions=((choices >= 0) & ~success).sum(axis=(1, 2)).tolist(),
        )
    return result


def whole_root_interval(differences, *, seed):
    """Resample all independent roots; requests/views remain nested."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap needs complete independent roots")
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ValueError("bootstrap requires an explicit128-bit PCG64 allocation")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = rng.integers(0, len(values), (20000, len(values)))
    return np.quantile(values[samples].mean(axis=1), (.025, .975)).tolist()


def root_probability_bounds(*, assigned, complete, false_admission_roots, full_valid_roots):
    """One-sided95 Clopper–Pearson bounds at actual independent-root counts."""
    values = (assigned, complete, false_admission_roots, full_valid_roots)
    if any(type(value) is not int for value in values) or not (
        0 <= complete <= assigned <= 32
        and 0 <= false_admission_roots <= complete
        and 0 <= full_valid_roots <= complete
    ):
        raise ValueError("probability bounds require valid complete-root counts")
    if not assigned:
        return {"status": "NOT_ESTIMATED", "n": 0, "upper_false_admission": None,
                "lower_full_valid": None}
    if complete != assigned:
        return {"status": "UNEVALUABLE", "n": assigned, "complete": complete,
                "upper_false_admission": None, "lower_full_valid": None}
    upper = 1.0 if false_admission_roots == assigned else float(
        beta.ppf(.95, false_admission_roots + 1, assigned - false_admission_roots))
    lower = 0.0 if full_valid_roots == 0 else float(
        beta.ppf(.05, full_valid_roots, assigned - full_valid_roots + 1))
    return {"status": "ESTIMATED_CONDITIONAL_IID_ROOTS", "n": assigned,
            "false_admission_roots": false_admission_roots,
            "full_valid_roots": full_valid_roots,
            "upper_false_admission": upper, "lower_full_valid": lower}


def numerical_masks(lower, z, conjuncts):
    """Keep three support definitions separate and preserve six other conjuncts."""
    if z.shape != (3, 24, 2) or conjuncts.shape != (3, 7):
        raise ValueError("diagnostic changes the complete handoff/conjunct chart")
    normalized = np.stack([lower.normalize(z[:, :, view]) for view in (0, 1)], axis=2)
    rest = conjuncts[:, 1:].all(axis=1)
    masks = {
        "primary": conjuncts.all(axis=1),
        "both_views": (abs(normalized) <= 6).all(axis=(1, 2)) & rest,
        "view_envelope": (
            abs(normalized).max(axis=2) + abs(normalized[:, :, 0] - normalized[:, :, 1]) <= 6
        ).all(axis=1) & rest,
    }
    return normalized, masks


def diagnose_root(*, lower, z, y, work, direction, requirement, choices):
    """Evaluate fixed-F counterfactuals without changing actual sealed choices."""
    from empirical_lawhood.adapters.methods.preparation_applicability.measurement import (
        lower_choices, service, validity,
    )
    _, conjuncts, _, mean, width, support = validity(lower, z, y, work)
    actual_choices, success = service(mean, width, support, y, work, direction, requirement)
    if not np.array_equal(actual_choices, choices):
        raise ValueError("diagnostic changes the actual sealed word choices")
    feasible = feasible_words(y, work, direction, requirement)
    if not np.array_equal(selected_success(feasible, choices), success):
        raise ValueError("independent word availability disagrees with the owned receiver")
    normalized, masks = numerical_masks(lower, z, conjuncts)
    bypass = lower_choices(mean, width, np.ones(3, dtype=bool), direction, requirement)
    fixed8, fixed16 = aligned_words(direction), aligned_words(direction, 16)
    row = {
        "actual": counts(success, choices),
        "support_bypassed_F": counts(selected_success(feasible, bypass), bypass),
        "fixed8_unlicensed": counts(selected_success(feasible, fixed8), fixed8),
        "fixed16_unlicensed": counts(selected_success(feasible, fixed16), fixed16),
        "native_word_oracle": counts(feasible.any(axis=3)),
        "mask_H_N_P": {key: mask.tolist() for key, mask in masks.items()},
        "primary_normalized_inward_margin_H_N_P": (6 - abs(normalized[:, :, 0]).max(axis=1)).tolist(),
        "support_threshold_sweep": {},
        "N_selected_word_histogram": np.bincount(choices[1].ravel() + 1, minlength=9).tolist(),
        "N_choices_equal_fixed8_where_admitted": bool((choices[1][choices[1] >= 0] == fixed8[1][choices[1] >= 0]).all()),
    }
    for threshold in THRESHOLDS:
        hypothetical = abs(normalized[:, :, 0]).max(axis=1) <= threshold
        selected = lower_choices(mean, width, hypothetical, direction, requirement)
        row["support_threshold_sweep"][str(threshold)] = counts(selected_success(feasible, selected), selected)
    return row, (mean, width)


def aggregate_rows(rows, *, bootstrap_seed):
    """Never drop adverse episodes; excluded root masks contribute zero."""
    if not rows or len(rows) not in (8, 32):
        raise ValueError("diagnostic requires the entire assigned Q8 or E32 census")
    covered = np.asarray([row["actual"]["joint_pairs"] for row in rows])
    result = {"n_independent_roots": len(rows), "n_distinct_nominal_waveform_recipes": 1,
              "counterfactuals": {}, "numerical_mask_diagnostics": {}}
    for key in ("support_bypassed_F", "fixed8_unlicensed", "fixed16_unlicensed", "native_word_oracle"):
        values = np.asarray([row[key]["joint_pairs"] for row in rows])
        diff = (values[:, 1] - values[:, 0]) / 256
        result["counterfactuals"][key] = {"joint_pairs_H_N_P": values.sum(axis=0).tolist(),
            "N_minus_H_fraction": float(diff.mean()),
            "whole_root_bootstrap_interval": whole_root_interval(diff, seed=bootstrap_seed),
            "ceiling": "OUTCOME_VISIBLE_DIAGNOSTIC_NOT_A_PROSPECTIVE_POLICY_OR_ADMISSION"}
    for key in ("primary", "both_views", "view_envelope"):
        mask = np.asarray([row["mask_H_N_P"][key] for row in rows])
        masked = covered * mask
        diff = (masked[:, 1] - masked[:, 0]) / 256
        result["numerical_mask_diagnostics"][key] = {"full_valid_counts_H_N_P": mask.sum(axis=0).tolist(),
            "covered_pairs_H_N_P": masked.sum(axis=0).tolist(), "N_minus_H_fraction": float(diff.mean()),
            "whole_root_bootstrap_interval": whole_root_interval(diff, seed=bootstrap_seed)}
    return result
