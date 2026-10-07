# SPDX-License-Identifier: MPL-2.0
"""Independent original 64-root evaluation and receipted-root arithmetic.

Authenticate the exact frozen source, qualification, successful receipts and
revealed native/control operands under separate analysis authority before entry.
These audits preserve every assigned root and unfavorable/missing observation;
they neither repair a campaign nor fabricate absent terminal adjudication.
"""

import json
import math
from collections import Counter
from random import Random
from typing import Any, cast

import numpy as np
from scipy.stats import beta, binom

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord

from .control_closeout import FiniteResponseLawRootSealedControl
from .control_records import FiniteResponseLawControlConfig
from .control_reveal import FiniteResponseLawProspectiveRootEvaluation
from .evaluation_native_records import FiniteResponseLawEvaluationNativeCompletion
from .evaluation_panel import EvaluationPanel, evaluation_panel
from .evaluation_readout import FiniteResponseLawRootInferenceOperands, cohort_inference
from .evaluation_results import FiniteResponseLawEvaluationCohort
from .law_binding import NATIVE_WORDS
from .method_records import FiniteResponseLawQualificationReport
from .science import FiniteResponseLawScienceSpec, seed_for


def _audit_events(
    config: FiniteResponseLawControlConfig,
    panel: EvaluationPanel,
    all_operands: tuple[FiniteResponseLawRootInferenceOperands, ...],
    sealed_records: tuple[FiniteResponseLawRootSealedControl, ...],
    stats: dict[str, Any],
    lower_qualified: bool,
    cached_qualified: bool,
) -> tuple[Any, ...]:
    if len(all_operands) != 64 or len(sealed_records) != 64:
        raise ValueError(
            "Independent evaluation requires every assigned root operand and seal"
        )
    bootstrap_seed = next((seed for p, i, seed in getattr(config.source, "scientific_seeds", ()) if p == "bootstrap" and i == -1), None)
    namespace = getattr(config.source, "cohort_namespace", "prospective-evaluation")
    committed = seed_for("bootstrap", namespace, committed_seed=bootstrap_seed)
    if any(seed_for("bootstrap", namespace, committed_seed=row.bootstrap_seed) != committed for row in all_operands):
        raise ValueError("Independent evaluation changes its committed bootstrap allocation")
    spec = FiniteResponseLawScienceSpec()
    delta = np.asarray([float(v) for v in spec.delta])
    nu = delta / 8
    events: list[dict[str, Any]] = []
    losses: dict[str, list[float | None]] = {
        b: [] for b in ("cached", "composed", "direct")
    }
    precision: dict[str, Counter[str]] = {b: Counter() for b in losses}
    for r, root in enumerate(config.source.roots):
        sealed = sealed_records[r]
        operands = all_operands[r]
        if (
            operands.root_id != root.stage_unit
            or sealed.join.lock.forecast.root != root
        ):
            raise ValueError("finite response-law audit changes assigned independent-root order")
        tables = {t.boundary: t for t in sealed.join.lock.forecast.tables}
        for b in losses:
            table = tables.get(b)
            value = None
            if (
                table is not None
                and table.point_mean is not None
                and panel.response_valid[r].all()
            ):
                prediction = np.asarray([float(v) for v in table.point_mean]).reshape(
                    4, 8
                )[:, :2]
                value = float(
                    np.mean(
                        ((panel.y[r, :, :2] - prediction[:, :, None, None]) / 0.01) ** 2
                    )
                )
            reported = dict(operands.response_losses)[b]
            if (value is None) != (reported is None) or (
                value is not None
                and (
                    reported is None
                    or not math.isclose(
                        value, float(reported), rel_tol=1e-10, abs_tol=1e-10
                    )
                )
            ):
                raise ValueError(
                    "finite response-law independent panel loss differs from direct endpoint arithmetic"
                )
            losses[b].append(value)
            widths = dict(operands.prediction_halfwidths)[b]
            precision[b]["all_root_denominator"] += 1
            precision[b]["complete_finite_width_roots"] += int(
                all((v is not None for v in widths))
            )
            precision[b]["all_menu_precision_pass_roots"] += int(
                all(
                    (
                        v is not None and v <= spec.delta[j % 8]
                        for j, v in enumerate(widths)
                    )
                )
            )
        for e in operands.events:
            b, c = e.policy_id.split(".consumer-")
            request = sealed.join.lock.requests.requests[int(c)]
            word = e.selected_word
            success = False
            if word >= 0:
                k = word // 2
                actual = panel.y[r, k].copy()
                if word % 2 == 0:
                    actual[:2] *= -1
                    actual[2:] = actual[[5, 6, 7, 2, 3, 4]]
                axis = request.direction // 2
                sign = 1 if request.direction % 2 == 0 else -1
                success = bool(
                    panel.use_valid[r, k].all()
                    and np.isfinite(actual).all()
                    and sealed.join.use_allowed
                    and (np.abs(actual[:, :, 0] - actual[:, :, 1]) <= nu[:, None]).all()
                    and (sign * actual[axis] >= float(request.lower)).all()
                    and (sign * actual[axis] <= float(spec.upper[int(c)])).all()
                    and (
                        np.abs(actual[1 - axis]) <= float(spec.transverse[int(c)])
                    ).all()
                    and (
                        actual[2:]
                        <= np.asarray([float(v) for v in spec.preservation] * 2)[
                            :, None, None
                        ]
                    ).all()
                )
            if (
                success != e.success
                or (word >= 0 and (not success)) != e.false_admission
            ):
                raise ValueError(
                    "finite response-law independent paired panel differs from scalar/controller use use event"
                )
            events.append(
                {
                    "root": root.stage_unit,
                    "policy": e.policy_id,
                    "selected_word": word,
                    "magnitude": None
                    if word < 0
                    else str(NATIVE_WORDS[word].magnitude),
                    "success": success,
                    "false_admission": word >= 0 and (not success),
                }
            )
    joint = {
        b: [
            all(
                (
                    e["success"]
                    for e in events
                    if e["root"] == r and e["policy"].startswith(b + ".")
                )
            )
            for r in panel.root_ids
        ]
        for b in losses
    }
    false = [
        any(
            (
                e["false_admission"]
                for e in events
                if e["root"] == r and e["policy"].startswith("composed.")
            )
        )
        for r in panel.root_ids
    ]
    good, bad = (sum(joint["composed"]), sum(false))
    lo = 0.0 if good == 0 else float(beta.ppf(0.05, good, 65 - good))
    hi = 1.0 if bad == 64 else float(beta.ppf(0.95, bad + 1, 64 - bad))
    use = good >= 52 and bad <= 2 and (lo >= 0.7) and (hi <= 0.1)
    np.testing.assert_allclose(
        [lo, hi], [stats["use_cp_lower"], stats["false_cp_upper"]], rtol=0, atol=1e-12
    )
    information = None
    bootstrap_upper = None
    if all((v is not None for b in ("cached", "composed") for v in losses[b])):
        primary_losses = np.asarray(losses["composed"], dtype=float)
        cached_losses = np.asarray(losses["cached"], dtype=float)
        difference = primary_losses - cached_losses
        rng = Random(committed)
        samples = sorted(
            (
                sum((float(difference[rng.randrange(64)]) for _ in range(64))) / 64
                for _ in range(20000)
            )
        )
        bootstrap_upper = samples[18999]
        information = bool(
            cached_losses.mean() > 0
            and primary_losses.mean() <= 0.9 * cached_losses.mean()
            and (bootstrap_upper < 0)
        )
        np.testing.assert_allclose(
            bootstrap_upper,
            float(stats["paired_root_bootstrap"]["upper"]),
            rtol=1e-10,
            atol=1e-10,
        )
    improved = sum(
        (a and (not b) for a, b in zip(joint["composed"], joint["cached"], strict=True))
    )
    deteriorated = sum(
        (b and (not a) for a, b in zip(joint["composed"], joint["cached"], strict=True))
    )
    p = (
        1.0
        if improved + deteriorated == 0
        else float(binom.sf(improved - 1, improved + deteriorated, 0.5))
    )
    added = (
        use
        and information is True
        and cached_qualified
        and (improved > deteriorated)
        and (p <= 0.05)
    )
    if (use, information, added, lower_qualified and use) != (
        stats["tier1_use_supported"],
        stats["tier1_information_supported"],
        stats["tier1_added_use_supported"],
        stats["preparation_policy_eligible"],
    ):
        raise ValueError(
            "finite response-law independent inference disagrees with canonical adjudication operands"
        )
    return (events, losses, precision, bootstrap_upper, p, use, information)


def verify_evaluation(
    config: FiniteResponseLawControlConfig,
    native: FiniteResponseLawEvaluationNativeCompletion,
    cohort: FiniteResponseLawEvaluationCohort,
    adjudication: ScientificAdjudicationRecord,
    sealed_records: tuple[FiniteResponseLawRootSealedControl, ...],
) -> dict[str, Any]:
    """Check the complete original cohort against revealed native measurements.

    Supplied records must already have authenticated receipt, source and reveal
    custody. Returned gates check exposed outcomes and grant no fresh authority.
    """
    if (
        native.config.projection.native_spec != config.source
        or cohort.qualification.object_fingerprint != config.qualification.sha256
    ):
        raise ValueError("finite response-law audit changes the source or frozen law qualification")
    panel = evaluation_panel(native)
    stats = json.loads(cohort.statistics_json)
    events, losses, precision, bootstrap_upper, p, use, information = _audit_events(
        config,
        panel,
        cohort.operands,
        sealed_records,
        stats,
        cohort.lower_qualified,
        cohort.cached_qualified,
    )
    if adjudication.scientific_status != cohort.scientific_status:
        raise ValueError("finite response-law terminal adjudication differs from its canonical cohort")
    result = {
        "schema": "finite-response-law-independent-prospective-readout",
        "verified": True,
        "independent_roots": 64,
        "physical_native_tasks": 1280,
        "logical_consumer_slots": 384,
        "scientific_status": adjudication.scientific_status.value,
        "statistics": stats,
        "precision": {b: dict(v) for b, v in precision.items()},
        "failure_reason_counts": dict(
            Counter((code for u in cohort.generic.units for code in u.reason_codes))
        ),
        "all_root_events": events,
        "independent_losses": losses,
        "independent_bootstrap_upper": bootstrap_upper,
        "independent_mcnemar_p": p,
        "world": "NUMERICAL_SIMULATOR",
        "limits": "FIXED_SYNTHETIC_NATIVE_MENU_REQUIREMENTS;EXPLICITLY_PAIRED_DIFFERENTIAL_RECEIVER;NO_PHYSICAL_OR_UNIVERSAL_LAW_PROMOTION",
    }
    return result


def verify_receipted_evaluation(
    config: FiniteResponseLawControlConfig,
    native: FiniteResponseLawEvaluationNativeCompletion,
    qualification: FiniteResponseLawQualificationReport,
    all_operands: tuple[FiniteResponseLawRootInferenceOperands, ...],
    sealed_records: tuple[FiniteResponseLawRootSealedControl, ...],
    prospective_evaluation_records: tuple[FiniteResponseLawProspectiveRootEvaluation, ...],
    reveal_authority: ObjectIdentity,
) -> dict[str, Any]:
    """Audit all revealed roots without supplying absent campaign adjudication.

    The caller authenticates reveal authority and every receipt before entry.
    This separate analysis cannot repair execution or stand in for its terminal.
    """
    if len(all_operands) != 64 or len(sealed_records) != 64 or len(prospective_evaluation_records) != 64:
        raise ValueError(
            "Receipted evaluation requires the complete assigned root census"
        )
    adjudication = None
    if qualification.fingerprint() != config.qualification.sha256:
        raise ValueError("finite response-law analysis substitutes the frozen law qualification")
    qualified = {
        b.boundary: q.scientific_status.value == "SUPPORTED"
        for b, q in zip(
            qualification.calibration.boundaries,
            qualification.qualifications,
            strict=True,
        )
    }
    lower_qualified, cached_qualified = (qualified["lower"], qualified["cached"])
    stats = cast(
        dict[str, Any],
        cohort_inference(
            all_operands,
            lower_qualified=lower_qualified,
            cached_qualified=cached_qualified,
        ),
    )
    failure_counts: Counter[str] = Counter()
    if native.config.projection.native_spec != config.source:
        raise ValueError("finite response-law audit changes the source or frozen law qualification")
    panel = evaluation_panel(native)
    for r, root in enumerate(config.source.roots):
        sealed = sealed_records[r]
        operands = all_operands[r]
        if (
            operands.root_id != root.stage_unit
            or sealed.join.lock.forecast.root != root
        ):
            raise ValueError("finite response-law audit changes assigned independent-root order")
        prospective_evaluation_record = prospective_evaluation_records[r]
        if (
            prospective_evaluation_record.sealed.object_fingerprint != sealed.fingerprint()
            or any((v.reveal_authorization != reveal_authority for v in prospective_evaluation_record.revealed))
            or tuple(
                (
                    (
                        u.root_id,
                        u.policy_id,
                        u.admitted,
                        u.task_success,
                        u.admitted_failure,
                    )
                    for u in prospective_evaluation_record.units
                )
            )
            != tuple(
                (
                    (
                        root.stage_unit,
                        e.policy_id,
                        e.selected_word >= 0,
                        e.success,
                        e.false_admission,
                    )
                    for e in operands.events
                )
            )
        ):
            raise ValueError(
                "finite response-law analysis loses generic controller use, reveal authority or scalar parity"
            )
        failure_counts.update((code for u in prospective_evaluation_record.units for code in u.reason_codes))
    events, losses, precision, bootstrap_upper, p, use, information = _audit_events(
        config,
        panel,
        all_operands,
        sealed_records,
        stats,
        lower_qualified,
        cached_qualified,
    )
    analysis_status = (
        "UNEVALUABLE"
        if information is None
        else "SUPPORTED"
        if use and information
        else "MIXED"
        if use or information
        else "NOT_SUPPORTED"
    )
    result = {
        "schema": "finite-response-law-receipted-prospective-analysis",
        "verified": True,
        "independent_roots": 64,
        "physical_native_tasks": 1280,
        "logical_consumer_slots": 384,
        "scientific_status": None
        if adjudication is None
        else adjudication.scientific_status.value,
        "independent_analysis_status": analysis_status,
        "statistics": stats,
        "precision": {b: dict(v) for b, v in precision.items()},
        "failure_reason_counts": dict(failure_counts),
        "all_root_events": events,
        "independent_losses": losses,
        "independent_bootstrap_upper": bootstrap_upper,
        "independent_mcnemar_p": p,
        "world": "NUMERICAL_SIMULATOR",
        "limits": "FIXED_SYNTHETIC_NATIVE_MENU_REQUIREMENTS;EXPLICITLY_PAIRED_DIFFERENTIAL_RECEIVER;NO_PHYSICAL_OR_UNIVERSAL_LAW_PROMOTION",
    }
    return result


def verify_continued_evaluation(
    config: FiniteResponseLawControlConfig,
    native: FiniteResponseLawEvaluationNativeCompletion,
    cohort: FiniteResponseLawEvaluationCohort,
    adjudication: ScientificAdjudicationRecord,
    sealed_records: tuple[FiniteResponseLawRootSealedControl, ...],
) -> dict[str, Any]:
    """Retain the continuation's separate analysis status on a complete cohort.

    Any retry amendment, repaired-source proof and cumulative elapsed accounting
    are authenticated by the caller beforehand. This function performs no retry.
    """
    if cohort.qualification.object_fingerprint != config.qualification.sha256:
        raise ValueError("finite response-law audit substitutes the frozen law qualification")
    if adjudication.scientific_status != cohort.scientific_status:
        raise ValueError("finite response-law terminal adjudication differs from its canonical cohort")
    result = verify_evaluation(config, native, cohort, adjudication, sealed_records)
    use = result["statistics"]["tier1_use_supported"]
    information = result["statistics"]["tier1_information_supported"]
    analysis_status = (
        "UNEVALUABLE"
        if information is None
        else "SUPPORTED"
        if use and information
        else "MIXED"
        if use or information
        else "NOT_SUPPORTED"
    )
    result["independent_analysis_status"] = analysis_status
    return result
