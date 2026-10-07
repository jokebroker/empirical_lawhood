"""All-assigned native selected-response predicates, separate from law verdicts."""

from decimal import Decimal as D

import numpy as np
from scipy.stats import beta, t

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.controller_evaluation import controller_prospective_student_summary
from .config import ClassicalDesign
from .records import ClassicalAssay, ClassicalCausal, ClassicalDecision, ClassicalQualification, ClassicalRootScore


def measure_root(
    causal: ClassicalCausal, decision: ClassicalDecision, assay: ClassicalAssay
) -> ClassicalRootScore:
    if (
        assay.causal_preparation
        != ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
        or assay.sealed_decision != ObjectIdentity.from_record(decision.decision_id, decision)
        or assay.root != decision.root
        or causal.root != decision.root
    ):
        raise ValueError("classical measurement changed its source/seal custody")
    design = ClassicalDesign()
    data = assay.arrays.unpack()
    reasons = set(assay.reasons)
    if not decision.causal_preparation_valid or len(data) != 30:
        return ClassicalRootScore(
            causal.root,
            False,
            False,
            False,
            (),
            (),
            (),
            (),
            tuple(sorted(reasons | {"INCOMPLETE_ASSIGNED_LOCAL_MEASUREMENT"})),
        )
    effects, peaks, masses, errors = [], [], [], []
    ca = causal.arrays.unpack()
    callback = decision.callback
    assert callback is not None and decision.observed_temperature_K is not None
    observed = float(decision.observed_temperature_K)
    safe = True
    required = {
        f"v{view}_a{word}_{field}"
        for view in (0, 1)
        for word in (0, 1)
        for field in (
            "grid",
            "requests",
            "stages",
            "exposure",
            "valid",
            "prefix_sha256",
            "full_grid_sha256",
        )
    }
    required |= {f"v{view}_prefix_peak_K" for view in (0, 1)}
    if set(data) != required or any(not np.isfinite(value).all() for value in data.values()):
        return ClassicalRootScore(
            causal.root,
            False,
            False,
            False,
            (),
            (),
            (),
            (),
            ("INVALID_OR_NONFINITE_LOCAL_MEASUREMENT",),
        )
    for view in (0, 1):
        zero, positive = (data[f"v{view}_a{word}_grid"] for word in (0, 1))
        expected_time = np.arange(
            callback * 10, callback * 10 + 10 + (0.5 if view else 1), 0.5 if view else 1
        )
        prefix_peak = data[f"v{view}_prefix_peak_K"]
        if prefix_peak.shape != (1,):
            return ClassicalRootScore(
                causal.root,
                False,
                False,
                False,
                (),
                (),
                (),
                (),
                ("INVALID_PREPARATION_SAFETY_MEASUREMENT",),
            )
        if prefix_peak[0] > float(design.temperature_limit_K):
            safe = False
            reasons.add("UNSAFE_PREPARATION_PREFIX")
        for word, grid in enumerate((zero, positive)):
            prefix = f"v{view}_a{word}"
            if (
                grid.shape != (len(expected_time), 8)
                or not np.isfinite(grid).all()
                or not np.array_equal(grid[:, 0], expected_time)
                or not data[f"{prefix}_valid"].all()
            ):
                return ClassicalRootScore(
                    causal.root,
                    False,
                    False,
                    safe,
                    (),
                    (),
                    (),
                    (),
                    tuple(sorted(reasons | {"INVALID_LOCAL_GRID_OR_CAUSAL_DELIVERY"})),
                )
            exposure = data[f"{prefix}_exposure"]
            requested = data[f"{prefix}_requests"]
            stages = data[f"{prefix}_stages"]
            if (
                exposure.shape != (len(expected_time) - 1, 4)
                or requested.shape != (2,)
                or stages.shape != (4,)
                or not np.array_equal(exposure[:, 0], expected_time[:-1])
                or not np.all(exposure[:, 1] == (0.5 if view else 1))
            ):
                return ClassicalRootScore(
                    causal.root,
                    False,
                    False,
                    safe,
                    (),
                    (),
                    (),
                    (),
                    tuple(sorted(reasons | {"INVALID_LOCAL_DELIVERY_MEASUREMENT"})),
                )
            jacket = ca[f"exploration_unshifted_v{view}_stages"][callback - 1, 3]
            mass = float(np.sum(exposure[:, 1] * exposure[:, 2]))
            if (
                not np.array_equal(requested, np.asarray((0.016 if word else 0.0, jacket)))
                or not np.array_equal(stages, np.asarray((0.016 if word else 0.0, jacket) * 2))
                or not np.all(exposure[:, 3] == jacket)
                or abs(mass - (0.16 if word else 0.0)) > 1e-12
            ):
                reasons.add("WRONG_REQUESTED_ACCEPTED_APPLIED_REALIZED_WORD")
            if word:
                masses.append(D(repr(mass)))
        if not np.array_equal(data[f"v{view}_a0_prefix_sha256"], data[f"v{view}_a1_prefix_sha256"]):
            reasons.add("UNMATCHED_CAUSAL_PREFIX")
        peak = float(np.max(positive[:, 1]))
        effect = float(np.max(zero[:, 1])) - peak
        effects.append(D(repr(effect)))
        peaks.append(D(repr(peak)))
        errors.append(D(repr(peak - observed)))
        if peak > float(design.temperature_limit_K):
            safe = False
            reasons.add("UNSAFE_POSITIVE_DELIVERY")
        if not float(design.response_lower_K) <= effect <= float(design.response_upper_K):
            reasons.add("SELECTED_RESPONSE_BOUND_FAILED")
        if abs(peak - observed) > float(design.causal_temperature_halfwidth_K):
            reasons.add("CAUSAL_TEMPERATURE_ENVELOPE_FAILED")
    if (
        abs(peaks[0] - peaks[1]) > design.numerical_peak_tolerance_K
        or abs(effects[0] - effects[1]) > design.numerical_cooling_tolerance_K
    ):
        reasons.add("NUMERICAL_VIEW_AGREEMENT_FAILED")
    return ClassicalRootScore(
        causal.root,
        True,
        not reasons,
        safe,
        tuple(effects),
        tuple(peaks),
        tuple(masses),
        tuple(errors),
        tuple(sorted(reasons)),
    )


def qualify_operands(
    assays: tuple[ClassicalAssay, ...], scores: tuple[ClassicalRootScore, ...]
) -> ClassicalQualification:
    n = 64
    successes = sum(row.adequate for row in scores)
    lower = D(0) if not successes else D(repr(float(beta.ppf(0.05, successes, n - successes + 1))))
    evaluable = all(row.evaluable for row in scores)
    mean = mean_lower = None
    if evaluable:
        mean, _, mean_lower = controller_prospective_student_summary(
            tuple(min(row.cooling_K) for row in scores),
            one_sided_critical_value=D(repr(float(t.ppf(0.95, n - 1)))),
        )
    design = ClassicalDesign()
    return ClassicalQualification(
        ObjectIdentity.from_record(design.config_id, design),
        tuple(ObjectIdentity.from_record(f"{row.root}.assay", row) for row in assays),
        scores,
        successes,
        lower,
        mean,
        mean_lower,
        evaluable,
        evaluable
        and lower >= design.qualification_probability_floor
        and all(row.safe for row in scores),
    )
