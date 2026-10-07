"""Leaf/contact/error and bounded-return evidence from the frozen B candidates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .config import ROOTS
from .discovery_records import RegimeContextMeasurement, RegimeDiscoveryDevelopment
from .measured_panel import CONTEXTS, measured_contexts
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .qualification_comparisons import _nominated_model_id, _scored_candidates
from .qualification_pipeline import RegimeQualificationPackage
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal
from .regime_stability import membership


def _d(value: float | None) -> D | None:
    return None if value is None else D(repr(float(value)))


@dataclass(frozen=True, slots=True)
class RegimeContextError(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-context-error'
    root: str
    context: str
    model_id: str
    receiver: str
    leaf: int | None
    observation_valid: bool
    supported: bool
    prediction: D | None
    target: D | None
    error: D | None
    action_errors_K: tuple[D, ...]
    refined_action_errors_K: tuple[D, ...]
    nominal_action_loss_K2: D | None
    support_face_margins: tuple[D, ...]


@dataclass(frozen=True, slots=True)
class RegimeLeafContact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-leaf-contact'
    model_id: str
    leaf: int
    observed_roots: tuple[str, ...]
    valid_roots: tuple[str, ...]
    supported_roots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegimePersistence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-persistence'
    root: str
    path: str
    model_id: str
    measured_coefficient_change_K_per_kg: D | None
    predicted_coefficient_change_K_per_kg: D | None
    action_loss_change_K2: D | None
    leaf_at_q: int | None
    leaf_at_q600: int | None
    endpoint_pair_valid: bool
    common_actual_return_valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegimeRootSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-root-summary'
    metric: str
    root_values: tuple[tuple[str, D], ...]
    mean: D | None
    descriptive_95: tuple[D, ...]
    leave_one_root_out: tuple[tuple[str, D], ...]
    tukey_outliers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegimePhaseDiagnostics(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-phase-diagnostics'
    role: str
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    contexts: tuple[RegimeContextMeasurement, ...]
    errors: tuple[RegimeContextError, ...]
    leaf_contacts: tuple[RegimeLeafContact, ...]
    persistence: tuple[RegimePersistence, ...]
    summaries: tuple[RegimeRootSummary, ...]

    def __post_init__(self) -> None:
        expected = tuple((root, context) for root, role, _, _ in ROOTS if role == self.role
                         for context in CONTEXTS)
        if (self.role not in ("nomination", "calibration", "qualification")
                or tuple((row.root, row.context) for row in self.contexts) != expected):
            raise ValueError("diagnostics changed the assigned root/context census")


def summarize_roots(metric: str, values: dict[str, float]) -> RegimeRootSummary:
    roots = tuple(sorted(values))
    if not roots:
        return RegimeRootSummary(metric, (), None, (), (), ())
    data = np.asarray([values[root] for root in roots])
    if not np.isfinite(data).all():
        raise ValueError("diagnostic cannot silently discard nonfinite model results")
    draws = np.random.default_rng(20260925).integers(0, len(roots), size=(5000, len(roots)))
    bounds = np.quantile(np.mean(data[draws], axis=1), (.025, .975))
    q1, q3 = np.quantile(data, (.25, .75))
    outliers = tuple(root for root in roots if values[root] < q1 - 1.5 * (q3 - q1)
                     or values[root] > q3 + 1.5 * (q3 - q1))
    return RegimeRootSummary(
        metric, tuple((root, D(repr(values[root]))) for root in roots), _d(float(np.mean(data))),
        tuple(D(repr(float(value))) for value in bounds),
        () if len(roots) == 1 else tuple((root, D(repr(float(np.mean(np.delete(data, i))))))
                                       for i, root in enumerate(roots)), outliers,
    )


def diagnostic_model_ids(fit: RegimeFitPackage, nomination: RegimeNominationPackage) -> tuple[str, ...]:
    scored = _scored_candidates(fit, nomination)
    ids = {value.coefficient_model_id for value in nomination.routes if value.coefficient_model_id is not None}
    for mask in ("current", "history"):
        for families in (("K",), ("affine", "rbf"), ("local",)):
            selected = _nominated_model_id(tuple(row for row in scored if row.mask == mask
                                                 and row.family in families))
            if selected is not None:
                ids.add(selected)
    absolute_ids = {row.absolute_model_id for row in nomination.routes if row.absolute_model_id is not None}
    return tuple(sorted(ids | absolute_ids))


def common_return_valid(causal: RegimeCausalPreparation) -> bool:
    """Check actual feed/jacket, never subtraction of cumulative observed dose."""
    t0 = next(callback for name, callback, _ in causal.anchors if name == "prepared_t0")
    if t0 is None:
        return False
    arrays = causal.arrays.unpack()
    for view in (0, 1):
        c = arrays.get(f"c_v{view}_exposure")
        p = arrays.get(f"p_v{view}_exposure")
        donor = arrays.get(f"exploration_unshifted_v{view}_stages")
        if (c is None or p is None or donor is None or len(donor) < t0
                or len(c) < t0 + 94 or len(p) < t0 + 94):
            return False
        for exposure in (c, p):
            returned = exposure[t0 + 30:t0 + 94]
            if (not np.isfinite(returned).all()
                    or not np.all(np.abs(returned[:, :, 2] - .016) <= 1e-12)
                    or not np.all(returned[:, :, 3] == donor[t0 - 1, 3])):
                return False
        if not np.array_equal(c[t0 + 30:t0 + 94], p[t0 + 30:t0 + 94]):
            return False
    return True


def build_phase_diagnostics(
    role: str,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    rows: tuple[tuple[RegimeCausalPreparation, RegimePredictionSeal, RegimeAssayPanel], ...],
) -> RegimePhaseDiagnostics:
    expected = tuple(root for root, assigned, _, _ in ROOTS if assigned == role)
    if tuple(causal.root for causal, _, _ in rows) != expected:
        raise ValueError("diagnostic input root order differs")
    ids = diagnostic_model_ids(fit, nomination)
    candidates = {item.model_id: item for item in (*fit.coefficients, *fit.absolute)}
    contexts: list[RegimeContextMeasurement] = []
    errors = []
    persistence = []
    for causal, seal, assay in rows:
        charts = measured_contexts(causal, assay)
        contexts.extend(RegimeContextMeasurement.from_context(chart) for chart in charts)
        arrays = seal.predictions.unpack()
        row_errors = {}
        for model_id in ids:
            candidate = candidates[model_id]
            assert candidate.fitted is not None
            model = candidate.fitted.to_fit()
            absolute = model_id.startswith('absolute-observable.')
            seal_id = f"{'absolute' if absolute else 'coefficient'}.{model_id}".lower()
            model_index = seal.candidate_ids.index(seal_id)
            for column, chart in enumerate(charts):
                x = chart.causal_current if candidate.mask == "current" else chart.causal_history
                value = float(arrays["forecast"][model_index, column]) if arrays["present"][model_index, column] else None
                target = (None if chart.nominal_peak_K is None else chart.nominal_peak_K[0]) if absolute else chart.coefficient_K_per_kg
                leaf = None if x is None or candidate.family != "local" else membership(model, np.asarray(x)[None, :])[0]
                margins = () if x is None else tuple(
                    D(repr(float(margin))) for margin in np.r_[
                        np.asarray(x) - np.asarray(candidate.fitted.support_lower, dtype=float),
                        np.asarray(candidate.fitted.support_upper, dtype=float) - np.asarray(x)]
                )
                action_error: tuple[D, ...] = ()
                refined_error: tuple[D, ...] = ()
                loss = None
                if not absolute and value is not None and chart.nominal_mass_kg is not None:
                    if chart.nominal_cooling_K is not None:
                        action_error = tuple(D(repr(value * m - c)) for m, c in
                                             zip(chart.nominal_mass_kg, chart.nominal_cooling_K, strict=True))
                        loss = float(np.mean(np.asarray(action_error[1:], dtype=float) ** 2))
                    if chart.refined_cooling_K is not None and chart.refined_mass_kg is not None:
                        refined_error = tuple(D(repr(value * m - c)) for m, c in
                                              zip(chart.refined_mass_kg, chart.refined_cooling_K, strict=True))
                error = RegimeContextError(
                    causal.root, chart.context, model_id, "p0_K" if absolute else "kappa_K_per_kg",
                    leaf, chart.valid and target is not None,
                    bool(arrays["support"][model_index, column]), _d(value), _d(target),
                    None if value is None or target is None else _d(value - target),
                    action_error, refined_error, _d(loss), margins,
                )
                errors.append(error)
                row_errors[model_id, chart.context] = error
        returned = common_return_valid(causal)
        for model_id in ids:
            if model_id.startswith('absolute-observable.'):
                continue
            for path in ("c", "p"):
                first, last = (row_errors[model_id, f"{path}_{suffix}"] for suffix in ("q", "q600"))
                valid = first.observation_valid and last.observation_valid
                def delta(a: D | None, b: D | None) -> D | None:
                    return None if a is None or b is None else b - a
                persistence.append(RegimePersistence(
                    causal.root, path, model_id, delta(first.target, last.target),
                    delta(first.prediction, last.prediction),
                    delta(first.nominal_action_loss_K2, last.nominal_action_loss_K2),
                    first.leaf, last.leaf, valid, returned,
                    tuple(reason for condition, reason in ((valid, "ENDPOINT_PAIR_INVALID"),
                                                           (returned, "ACTUAL_COMMON_RETURN_INVALID")) if not condition),
                ))
    contacts = []
    for model_id in ids:
        candidate = candidates[model_id]
        if candidate.family != "local" or candidate.fitted is None:
            continue
        for leaf in range(len(candidate.fitted.to_fit().leaves)):
            selected = [row for row in errors if row.model_id == model_id and row.leaf == leaf]
            contacts.append(RegimeLeafContact(
                model_id, leaf, tuple(sorted({row.root for row in selected})),
                tuple(sorted({row.root for row in selected if row.observation_valid})),
                tuple(sorted({row.root for row in selected if row.observation_valid and row.supported})),
            ))
    summaries = []
    for model_id in ids:
        for context in (*CONTEXTS, "all_contexts"):
            selected = [row for row in errors if row.model_id == model_id and row.observation_valid
                        and (context == "all_contexts" or row.context == context)]
            values = {}
            for root in expected:
                losses = [float(row.nominal_action_loss_K2) if row.nominal_action_loss_K2 is not None
                          else float(row.error or D(0)) ** 2 for row in selected if row.root == root
                          and (row.nominal_action_loss_K2 is not None or row.error is not None)]
                if losses:
                    values[root] = float(np.mean(losses))
            summaries.append(summarize_roots(f"{model_id}.{context}.loss_K2", values))
        if not model_id.startswith('absolute-observable.'):
            for path in ("c", "p"):
                for field in ("measured_coefficient_change_K_per_kg", "predicted_coefficient_change_K_per_kg", "action_loss_change_K2"):
                    values = {row.root: float(getattr(row, field)) for row in persistence
                              if row.model_id == model_id and row.path == path and row.endpoint_pair_valid
                              and row.common_actual_return_valid and getattr(row, field) is not None}
                    summaries.append(summarize_roots(f"{model_id}.{path}.return.{field}", values))
    return RegimePhaseDiagnostics(
        role, ObjectIdentity.from_record(fit.package_id, fit),
        ObjectIdentity.from_record(nomination.package_id, nomination), tuple(contexts),
        tuple(errors), tuple(contacts), tuple(persistence), tuple(summaries),
    )


@dataclass(frozen=True, slots=True)
class RegimeLocalConfirmation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-local-confirmation'
    model_id: str
    stability_pass: bool
    calibration_leaf_roots: tuple[int, ...]
    qualification_leaf_roots: tuple[int, ...]
    leaf_contact_pass: bool
    is_R_local: bool
    R_verdict: str
    useful_local_regime: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegimeDiscoveryConfirmation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-discovery-confirmation'
    development: ObjectIdentity
    calibration_diagnostics: ObjectIdentity
    qualification_diagnostics: ObjectIdentity
    qualification: ObjectIdentity
    locals: tuple[RegimeLocalConfirmation, ...]
    verdict: str
    reasons: tuple[str, ...]
    description_preference: str


def confirm_regimes(
    development: RegimeDiscoveryDevelopment,
    calibration: RegimePhaseDiagnostics,
    diagnostics: RegimePhaseDiagnostics,
    qualification: RegimeQualificationPackage,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
) -> RegimeDiscoveryConfirmation:
    if (calibration.role != "calibration" or diagnostics.role != "qualification"
            or development.fit_package != qualification.fit_package
            or development.nomination != qualification.nomination
            or calibration.fit_package != development.fit_package
            or diagnostics.fit_package != development.fit_package):
        raise ValueError("regime confirmation changed its frozen discovery parents")
    local_id = _nominated_model_id(tuple(row for row in _scored_candidates(fit, nomination)
                                         if row.mask == "history" and row.family == "local"))
    contrast = next(value for value in qualification.contrasts if value.contrast_id == "R")
    local_results = []
    for item in development.local_stability:
        cal_counts = tuple(len(row.valid_roots) for row in calibration.leaf_contacts if row.model_id == item.model_id)
        qual_counts = tuple(len(row.valid_roots) for row in diagnostics.leaf_contacts if row.model_id == item.model_id)
        contact = (len(cal_counts) == len(qual_counts) == item.original_leaf_count
                   and min(cal_counts, default=0) >= 8 and min(qual_counts, default=0) >= 16)
        is_local = item.model_id == local_id
        advantage = contrast.verdict == "SUPPORTED_20_PERCENT_ADVANTAGE"
        useful = is_local and advantage and contact and item.stability_pass
        reasons = tuple(reason for good, reason in (
            (item.stability_pass, "LOCAL_PARTITION_STABILITY_NOT_ESTABLISHED"),
            (contact, "INSUFFICIENT_FRESH_LEAF_CONTACT"),
            (is_local, "NOT_THE_PREREGISTERED_R_LOCAL_MODEL"),
            (advantage, "FRESH_LOCAL_ADDED_VALUE_NOT_ESTABLISHED"),
        ) if not good)
        local_results.append(RegimeLocalConfirmation(
            item.model_id, item.stability_pass, cal_counts, qual_counts, contact,
            is_local, contrast.verdict, useful, reasons,
        ))
    verdict = ("USEFUL_LOCAL_REGIME_SUPPORTED" if any(row.useful_local_regime for row in local_results)
               else "UNEVALUABLE" if local_id is None or contrast.verdict == "UNEVALUABLE"
               else "REGIME_ADDED_VALUE_NOT_ESTABLISHED")
    rival = _nominated_model_id(tuple(row for row in _scored_candidates(fit, nomination)
                                     if row.mask == "history" and row.family != "local"))
    preference = "FINITE_ROSTER_DESCRIPTION_UNRESOLVED"
    if verdict == "USEFUL_LOCAL_REGIME_SUPPORTED":
        preference = "USEFUL_LOCAL_DESCRIPTION_PREFERRED"
    elif (rival is not None and contrast.verdict != "UNEVALUABLE" and len(contrast.arm_rmse_K) == 2
          and contrast.arm_rmse_K[0] <= contrast.arm_rmse_K[1]):
        family = next(row.family for row in fit.coefficients if row.model_id == rival)
        preference = "CONSTANT_DESCRIPTION_PREFERRED" if family == "K" else "CONTINUOUS_DESCRIPTION_PREFERRED"
    return RegimeDiscoveryConfirmation(
        ObjectIdentity.from_record("regime.discovery-development", development),
        ObjectIdentity.from_record("regime.calibration-diagnostics", calibration),
        ObjectIdentity.from_record("regime.qualification-diagnostics", diagnostics),
        ObjectIdentity.from_record(qualification.package_id, qualification), tuple(local_results),
        verdict, () if verdict == "USEFUL_LOCAL_REGIME_SUPPORTED" else (verdict,), preference,
    )
