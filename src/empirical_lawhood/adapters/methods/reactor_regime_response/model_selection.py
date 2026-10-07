"""Role-separated finite K/S/L and absolute-baseline development roster."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .absolute_models import AbsoluteRows, fit_absolute
from .config import ROOTS
from .cv_local import fit_local_cv
from .measured_panel import MeasuredContext
from .models import CoefficientRows, Fit, LAMBDAS, RBF_MULTIPLIERS, fit_affine, fit_constant, fit_rbf


@dataclass(frozen=True)
class Candidate:
    model_id: str
    mask: str
    family: str
    penalty: float
    multiplier: float | None
    fitted: Fit | None
    support_lower: np.ndarray | None
    support_upper: np.ndarray | None
    nomination_root_losses_K2: dict[str, float]
    nomination_mean_loss_K2: float | None
    nomination_leaf_contacts: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.fitted is not None and self.nomination_mean_loss_K2 is not None and not self.reasons

    @property
    def parameter_count(self) -> int:
        if self.fitted is None:
            return 2**31
        if self.family == "local":
            return sum(len(leaf.operator) for leaf in self.fitted.leaves)
        return len(self.fitted.operator)

    @property
    def leaf_count(self) -> int:
        return len(self.fitted.leaves) if self.fitted is not None and self.family == "local" else 1

    def contains(self, x: np.ndarray) -> np.ndarray:
        if self.support_lower is None or self.support_upper is None:
            return np.zeros(len(x), dtype=bool)
        if x.ndim != 2 or x.shape[1] != len(self.support_lower):
            raise ValueError("candidate support feature shape differs")
        return np.asarray(
            ((x >= self.support_lower) & (x <= self.support_upper)).all(axis=1), dtype=bool
        )


def _features(context: MeasuredContext, mask: str) -> tuple[float, ...] | None:
    if mask == "current":
        return context.causal_current
    if mask == "history":
        return context.causal_history
    raise ValueError("undeclared coefficient feature mask")


def _coefficient_rows(assigned: tuple[str, ...], contexts: tuple[MeasuredContext, ...], mask: str) -> CoefficientRows:
    if any(row.root not in assigned for row in contexts):
        raise ValueError("whole-root role leakage into coefficient fitting")
    usable = tuple(
        row
        for row in contexts
        if row.valid
        and row.nominal_mass_kg is not None
        and row.nominal_cooling_K is not None
        and _features(row, mask) is not None
    )
    if not usable:
        raise ValueError("no valid source charts in this assigned role")
    return CoefficientRows(
        np.asarray([_features(row, mask) for row in usable], dtype=np.float64),
        np.asarray([row.nominal_mass_kg[1:] for row in usable if row.nominal_mass_kg is not None]),
        np.asarray([row.nominal_cooling_K[1:] for row in usable if row.nominal_cooling_K is not None]),
        tuple(row.root for row in usable),
    )


def _absolute_rows(assigned: tuple[str, ...], contexts: tuple[MeasuredContext, ...], mask: str) -> AbsoluteRows:
    if any(row.root not in assigned for row in contexts):
        raise ValueError("whole-root role leakage into absolute fitting")
    usable = tuple(
        row
        for row in contexts
        if row.valid and row.nominal_peak_K is not None and _features(row, mask) is not None
    )
    if not usable:
        raise ValueError("no valid absolute zero-reference labels in this role")
    return AbsoluteRows(
        np.asarray([_features(row, mask) for row in usable], dtype=np.float64),
        np.asarray([row.nominal_peak_K[0] for row in usable if row.nominal_peak_K is not None]),
        tuple(row.root for row in usable),
    )


def _roster() -> tuple[tuple[str, float, float | None], ...]:
    return (
        ("K", 0.0, None),
        *(("affine", p, None) for p in LAMBDAS),
        *(("rbf", p, m) for p in LAMBDAS for m in RBF_MULTIPLIERS),
        *(("local", p, None) for p in LAMBDAS),
    )


def _fit(rows: CoefficientRows, family: str, penalty: float, multiplier: float | None) -> Fit:
    if family == "K":
        return fit_constant(rows)
    if family == "affine":
        return fit_affine(rows, penalty)
    if family == "rbf" and multiplier is not None:
        return fit_rbf(rows, penalty, multiplier)
    if family == "local":
        return fit_local_cv(rows, penalty)
    raise ValueError("undeclared or incomplete model specification")


def _leaf_contacts(fitted: Fit, rows: CoefficientRows) -> tuple[int, ...]:
    return tuple(
        len(
            {
                root
                for root, row in zip(rows.roots, rows.features, strict=True)
                if all((row[i] <= t) == left for i, t, left in leaf.path)
            }
        )
        for leaf in fitted.leaves
    )


def fit_candidate_roster(fit_contexts: tuple[MeasuredContext, ...]) -> tuple[Candidate, ...]:
    """Freeze every K/S/L coefficient fit before nomination assays exist."""
    fit_roots = tuple(r for r, role, _, _ in ROOTS if role == "fit")
    if any(row.root not in fit_roots for row in fit_contexts):
        raise ValueError("whole-root role leak into coefficient fit")
    candidates = []
    for mask in ("current", "history"):
        try:
            fit_rows = _coefficient_rows(fit_roots, fit_contexts, mask)
        except ValueError:
            fit_rows = None
        if fit_rows is None:
            support_low = support_high = None
        else:
            low = np.min(fit_rows.features, axis=0)
            high = np.max(fit_rows.features, axis=0)
            support_low = low - 0.05 * (high - low)
            support_high = high + 0.05 * (high - low)
        for family, penalty, multiplier in _roster():
            model_id = f"{mask}.{family}.l{penalty:.6g}.r{multiplier if multiplier is not None else 'none'}"
            fitted = None
            reasons: set[str] = set()
            if fit_rows is None:
                reasons.add("NO_VALID_FIT_CONTACT")
            else:
                try:
                    fitted = _fit(fit_rows, family, penalty, multiplier)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    reasons.add(f"FIT_FAILED:{type(error).__name__}:{error}")
            candidates.append(
                Candidate(
                    model_id,
                    mask,
                    family,
                    penalty,
                    multiplier,
                    fitted,
                    support_low.copy() if fitted is not None and support_low is not None else None,
                    support_high.copy() if fitted is not None and support_high is not None else None,
                    {},
                    None,
                    (),
                    tuple(sorted(reasons)),
                )
            )
    return tuple(candidates)


def score_nomination_roster(
    fitted_roster: tuple[Candidate, ...],
    nomination_contexts: tuple[MeasuredContext, ...],
) -> tuple[Candidate, ...]:
    nomination_roots = tuple(r for r, role, _, _ in ROOTS if role == "nomination")
    if any(row.root not in nomination_roots for row in nomination_contexts):
        raise ValueError("whole-root role leak into nomination")
    if len(fitted_roster) != 2 * len(_roster()):
        raise ValueError("coefficient fit roster is incomplete")
    scored = []
    rows_by_mask: dict[str, CoefficientRows | None] = {}
    for mask in ("current", "history"):
        try:
            rows_by_mask[mask] = _coefficient_rows(nomination_roots, nomination_contexts, mask)
        except ValueError:
            rows_by_mask[mask] = None
    for candidate in fitted_roster:
        reasons = set(candidate.reasons)
        rows = rows_by_mask[candidate.mask]
        losses: dict[str, float] = {}
        mean_loss = None
        contacts: tuple[int, ...] = ()
        if rows is None:
            reasons.add("NO_VALID_NOMINATION_CONTACT")
        elif candidate.fitted is not None:
            try:
                losses = rows.root_losses(candidate.fitted.predict(rows.features))
                if not losses or not all(isfinite(value) for value in losses.values()):
                    reasons.add("NONFINITE_NOMINATION_LOSS")
                else:
                    mean_loss = float(np.mean(tuple(losses.values())))
                if candidate.family == "local":
                    contacts = _leaf_contacts(candidate.fitted, rows)
                    if any(count < 8 for count in contacts):
                        reasons.add("LOCAL_LEAF_HAS_FEWER_THAN_EIGHT_NOMINATION_ROOTS")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                reasons.add(f"NOMINATION_PREDICTION_FAILED:{type(error).__name__}:{error}")
        scored.append(
            Candidate(
                candidate.model_id,
                candidate.mask,
                candidate.family,
                candidate.penalty,
                candidate.multiplier,
                candidate.fitted,
                candidate.support_lower,
                candidate.support_upper,
                losses,
                mean_loss,
                contacts,
                tuple(sorted(reasons)),
            )
        )
    return tuple(scored)


def fit_roster(
    fit_contexts: tuple[MeasuredContext, ...],
    nomination_contexts: tuple[MeasuredContext, ...],
) -> tuple[Candidate, ...]:
    """Convenience path; production persists the fit stage before nomination."""
    if {row.root for row in fit_contexts} & {row.root for row in nomination_contexts}:
        raise ValueError("root appears in both fit and nomination")
    return score_nomination_roster(fit_candidate_roster(fit_contexts), nomination_contexts)


def nominate(candidates: tuple[Candidate, ...]) -> Candidate:
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ValueError("no finite nominated coefficient model")
    minimum = min(float(candidate.nomination_mean_loss_K2) for candidate in eligible if candidate.nomination_mean_loss_K2 is not None)
    ties = [
        candidate
        for candidate in eligible
        if candidate.nomination_mean_loss_K2 is not None
        and candidate.nomination_mean_loss_K2 <= 1.01 * minimum
    ]
    return min(
        ties,
        key=lambda candidate: (
            candidate.leaf_count,
            candidate.parameter_count,
            -candidate.penalty,
            candidate.model_id,
        ),
    )


def fit_absolute_candidates(fit_contexts: tuple[MeasuredContext, ...]) -> tuple[Candidate, ...]:
    fit_roots = tuple(r for r, role, _, _ in ROOTS if role == "fit")
    if any(row.root not in fit_roots for row in fit_contexts):
        raise ValueError("whole-root role leak into absolute fit")
    candidates = []
    for mask in ("current", "history"):
        try:
            fit_rows = _absolute_rows(fit_roots, fit_contexts, mask)
        except ValueError:
            fit_rows = None
        if fit_rows is None:
            low = high = None
        else:
            low = np.min(fit_rows.features, axis=0)
            high = np.max(fit_rows.features, axis=0)
        for family, penalty, multiplier in _roster():
            if family not in ("affine", "rbf"):
                continue
            model_id = f"absolute-observable.{mask}.{family}.l{penalty:.6g}.r{multiplier if multiplier is not None else 'none'}"
            fitted = None
            reasons: set[str] = set()
            if fit_rows is None:
                reasons.add("NO_VALID_ABSOLUTE_FIT_CONTACT")
            else:
                try:
                    fitted = fit_absolute(fit_rows, family, penalty, multiplier)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    reasons.add(f"ABSOLUTE_FIT_FAILED:{type(error).__name__}:{error}")
            candidates.append(
                Candidate(
                    model_id,
                    mask,
                    family,
                    penalty,
                    multiplier,
                    fitted,
                    low - 0.05 * (high - low) if fitted is not None and low is not None and high is not None else None,
                    high + 0.05 * (high - low) if fitted is not None and low is not None and high is not None else None,
                    {},
                    None,
                    (),
                    tuple(sorted(reasons)),
                )
            )
    return tuple(candidates)


def score_absolute_nomination(
    fitted_roster: tuple[Candidate, ...],
    nomination_contexts: tuple[MeasuredContext, ...],
) -> tuple[Candidate, ...]:
    nomination_roots = tuple(r for r, role, _, _ in ROOTS if role == "nomination")
    if any(row.root not in nomination_roots for row in nomination_contexts):
        raise ValueError("whole-root role leak into absolute nomination")
    if len(fitted_roster) != 24:
        raise ValueError("absolute fit roster is incomplete")
    scored = []
    rows_by_mask: dict[str, AbsoluteRows | None] = {}
    for mask in ("current", "history"):
        try:
            rows_by_mask[mask] = _absolute_rows(nomination_roots, nomination_contexts, mask)
        except ValueError:
            rows_by_mask[mask] = None
    for candidate in fitted_roster:
        reasons = set(candidate.reasons)
        rows = rows_by_mask[candidate.mask]
        losses: dict[str, float] = {}
        mean_loss = None
        if rows is None:
            reasons.add("NO_VALID_ABSOLUTE_NOMINATION_CONTACT")
        elif candidate.fitted is not None:
            try:
                losses = rows.root_losses(candidate.fitted.predict(rows.features))
                mean_loss = float(np.mean(tuple(losses.values())))
                if not isfinite(mean_loss):
                    reasons.add("NONFINITE_ABSOLUTE_NOMINATION_LOSS")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                reasons.add(f"ABSOLUTE_NOMINATION_FAILED:{type(error).__name__}:{error}")
        scored.append(
            Candidate(
                candidate.model_id,
                candidate.mask,
                candidate.family,
                candidate.penalty,
                candidate.multiplier,
                candidate.fitted,
                candidate.support_lower,
                candidate.support_upper,
                losses,
                mean_loss,
                (),
                tuple(sorted(reasons)),
            )
        )
    return tuple(scored)


def fit_absolute_roster(
    fit_contexts: tuple[MeasuredContext, ...],
    nomination_contexts: tuple[MeasuredContext, ...],
) -> tuple[Candidate, ...]:
    if {row.root for row in fit_contexts} & {row.root for row in nomination_contexts}:
        raise ValueError("root appears in both fit and nomination")
    return score_absolute_nomination(fit_absolute_candidates(fit_contexts), nomination_contexts)
