"""Frozen preparation-policy development fitting and gate arithmetic.

The unchanged lower package is supplied as an evaluator.  This module owns
only the new prefix-to-handoff maps, their provisional uncertainty, and the
target-blind adequacy provider.
"""

from dataclasses import dataclass
from math import ceil
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.response_formalization import (
    affine_prediction,
    fit_affine_operator,
)

from .fitting import DELTA, Normalizer, clamp
from .assigned_prediction import AssignedPrediction

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
NU = DELTA / 8


class LowerEvaluator(Protocol):
    """Exact qualified lower-package boundary used by composition."""

    def predict(self, native_handoff: Array) -> AssignedPrediction:
        """Return the unchanged lower package's assigned prediction."""


@dataclass(frozen=True)
class PreparationPolicyFeatures:
    root_ids: tuple[str, ...]
    cohorts: tuple[str, ...]
    cohort_indices: IntArray
    x: Array  # root, feature, view
    z: Array  # root, schedule, feature, view

    def __post_init__(self) -> None:
        n, d = len(self.root_ids), self.x.shape[1]
        if (
            n == 0
            or len(set(self.root_ids)) != n
            or self.cohorts != tuple("prepared-response" if i < 8 else "information-response-prediction" for i in range(n))
            or self.cohort_indices.shape != (n,)
            or self.x.shape != (n, d, 2)
            or self.z.shape != (n, 9, d, 2)
            or not np.isfinite(self.x).all()
            or not np.isfinite(self.z).all()
        ):
            raise ValueError("preparation-policy features change the authenticated root/chart geometry")
        expected = np.r_[np.arange(8), np.arange(n - 8)].astype(np.int64)
        if self.cohort_indices.dtype != np.int64 or not np.array_equal(
            self.cohort_indices, expected
        ):
            raise ValueError("preparation-policy cohort indices are not the declared within-cohort indices")


@dataclass(frozen=True)
class PreparationPolicyPanel:
    y: Array  # root,schedule,pair,output,future,view
    observed: BoolArray
    valid: BoolArray

    def __post_init__(self) -> None:
        if (
            self.y.ndim != 6
            or self.y.shape[1:] != (9, 4, 8, 2, 2)
            or self.observed.shape != self.y.shape
            or self.valid.shape != self.y.shape
            or self.observed.dtype != np.bool_
            or self.valid.dtype != np.bool_
            or np.isinf(self.y).any()
            or not np.isfinite(self.y[self.observed & self.valid]).all()
        ):
            raise ValueError("preparation-policy panel loses its explicit measured/valid geometry")


def outer_folds(features: PreparationPolicyFeatures) -> IntArray:
    """Four folds, each holding two prepared-response and four information-response-prediction roots for the preparation-policy development census."""
    if len(features.root_ids) != 24:
        raise ValueError("preparation-policy development outer folds require exactly 24 exposed roots")
    result = features.cohort_indices % 4
    if tuple(np.bincount(result, minlength=4)) != (6, 6, 6, 6):
        raise ValueError("preparation-policy development outer folds changed their 2-prepared-response/4-information-response-prediction allocation")
    return result.astype(np.int64)


def coefficient_folds(features: PreparationPolicyFeatures, roots: IntArray) -> IntArray:
    """Three cohort-ranked folds within a particular outer training set."""
    result = np.empty(len(roots), dtype=np.int64)
    for cohort in ("prepared-response", "information-response-prediction"):
        positions = np.flatnonzero(np.asarray([features.cohorts[i] == cohort for i in roots]))
        ranked = positions[np.argsort(features.cohort_indices[roots[positions]])]
        result[ranked] = np.arange(len(ranked)) % 3
    return result


@dataclass(frozen=True)
class UpperPointFit:
    roots: tuple[int, ...]
    normalizer: Normalizer
    operators: Array  # schedule, standardized-prefix+intercept, native feature

    def predict_native(self, features: PreparationPolicyFeatures, roots: IntArray) -> Array:
        x = self.normalizer.apply(features.x[roots, :, 0])
        return np.stack([affine_prediction(op, x) for op in self.operators], axis=1)


def fit_upper_points(features: PreparationPolicyFeatures, roots: IntArray) -> UpperPointFit:
    if len(roots) == 0 or len(set(map(int, roots))) != len(roots):
        raise ValueError("Upper fit requires a nonempty unique root population")
    normalizer = Normalizer.fit(features.x[roots, :, 0])
    x = normalizer.apply(features.x[roots, :, 0])
    operators = np.stack(
        [fit_affine_operator(x, features.z[roots, p, :, 0], ridge=10) for p in range(9)]
    )
    return UpperPointFit(tuple(map(int, roots)), normalizer, operators)


def composed_prediction(
    fit: UpperPointFit,
    features: PreparationPolicyFeatures,
    roots: IntArray,
    lower: LowerEvaluator,
) -> tuple[Array, BoolArray, Array]:
    native = fit.predict_native(features, roots)
    lower_result = lower.predict(native.reshape(-1, native.shape[-1]))
    prediction = lower_result.mean.reshape(len(roots), 9, 4, 8)
    lower_support = lower_result.supported.reshape(len(roots), 9)
    n = len(roots)
    if prediction.shape != (n, 9, 4, 8) or lower_support.shape != (n, 9):
        raise ValueError("Lower evaluator changes the frozen four-pair/eight-output chart")
    standardized = fit.normalizer.apply(features.x[roots, :, 0])
    prefix_support = np.isfinite(standardized).all(axis=1) & (
        np.abs(standardized).max(axis=1) <= 6
    )
    support = prefix_support[:, None] & lower_support
    return clamp(prediction), support, native


def coefficient_oof(
    features: PreparationPolicyFeatures,
    roots: IntArray,
    lower: LowerEvaluator,
) -> tuple[Array, BoolArray, tuple[UpperPointFit, ...]]:
    folds = coefficient_folds(features, roots)
    prediction = np.full((len(roots), 9, 4, 8), np.nan)
    support = np.zeros((len(roots), 9), dtype=bool)
    fits: list[UpperPointFit] = []
    for fold in range(3):
        fit = fit_upper_points(features, roots[folds != fold])
        pred, sup, _ = composed_prediction(fit, features, roots[folds == fold], lower)
        prediction[folds == fold] = pred
        support[folds == fold] = sup
        fits.append(fit)
    return prediction, support, tuple(fits)


@dataclass(frozen=True)
class UpperWidthFit:
    roots: tuple[int, ...]
    base: Array  # pair,output
    multipliers: Array  # schedule, standardized-prefix+intercept, 1
    complete_targets: Array  # root,schedule; nan means absent
    rms_eligible: BoolArray  # root,pair,output

    def sigma(self, normalizer: Normalizer, x: Array) -> tuple[Array, Array]:
        nx = normalizer.apply(x)
        logg = np.column_stack(
            [affine_prediction(self.multipliers[p], nx)[:, 0] for p in range(9)]
        )
        g = np.exp(np.clip(logg, np.log(0.25), np.log(4)))
        return self.base[None, None] * g[:, :, None, None], g


def fit_upper_widths(
    panel: PreparationPolicyPanel,
    features: PreparationPolicyFeatures,
    roots: IntArray,
    oof_prediction: Array,
    full_normalizer: Normalizer,
) -> UpperWidthFit:
    y = panel.y[roots]
    measured = panel.observed[roots] & panel.valid[roots] & np.isfinite(y)
    residual = y - oof_prediction[..., None, None]
    primary = measured[..., 0]
    eligible = np.asarray(
        primary.all(axis=(1, 4)), dtype=np.bool_
    )  # root,pair,output; complete 9x2 block
    base = np.empty((4, 8))
    for k in range(4):
        for j in range(8):
            good = eligible[:, k, j]
            if not good.any():
                raise ValueError("Empty required preparation-policy nine-schedule RMS population")
            values = residual[good, :, k, j, :, 0]
            base[k, j] = max(DELTA[j] / 20, float(np.sqrt(np.mean(values**2))))
    complete = np.asarray(measured.all(axis=(2, 3, 4, 5)), dtype=np.bool_)
    target = np.full((len(roots), 9), np.nan)
    scaled = np.abs(residual) / base[None, None, :, :, None, None]
    for i in range(len(roots)):
        for p in range(9):
            if complete[i, p]:
                target[i, p] = np.log(max(0.25, float(scaled[i, p].max())))
    nx = full_normalizer.apply(features.x[roots, :, 0])
    multipliers = []
    for p in range(9):
        good = np.isfinite(target[:, p])
        if not good.any():
            raise ValueError("Empty preparation-policy schedule-specific multiplier population")
        multipliers.append(fit_affine_operator(nx[good], target[good, p, None], ridge=10))
    return UpperWidthFit(
        tuple(map(int, roots)), base, np.stack(multipliers), target, eligible
    )


def provisional_scores(
    panel: PreparationPolicyPanel,
    roots: IntArray,
    prediction: Array,
    sigma: Array,
    support: BoolArray,
) -> Array:
    y = panel.y[roots]
    measured = panel.observed[roots] & panel.valid[roots] & np.isfinite(y)
    numerical = (np.abs(y[..., 0] - y[..., 1]) <= NU[None, None, None, :, None]).all(
        axis=(2, 3, 4)
    )
    scaled = np.maximum(
        np.abs(y - prediction[..., None, None]) - NU[None, None, None, :, None, None], 0
    ) / sigma[..., None, None]
    parent = scaled.max(axis=(2, 3, 4, 5))
    parent = np.where(
        measured.all(axis=(2, 3, 4, 5)) & numerical & support & np.isfinite(parent),
        parent,
        np.inf,
    )
    return np.asarray(parent.max(axis=1), dtype=np.float64)


def provisional_quantile(scores: Array, *, expected_n: int) -> tuple[int, float]:
    if scores.shape != (expected_n,) or np.isnan(scores).any() or (scores < 0).any():
        raise ValueError("Invalid preparation-policy root-score census")
    rank = ceil((expected_n + 1) * 0.9)
    return rank, float(np.sort(scores)[rank - 1]) if rank <= expected_n else float("inf")


@dataclass(frozen=True)
class AdequacyProvider:
    roots: tuple[int, ...]
    normalizer: Normalizer
    operators: Array
    schedule_ids: tuple[str, ...]
    best_fixed: int

    def choose(self, x: Array) -> tuple[IntArray, Array]:
        nx = self.normalizer.apply(x)
        scores = np.clip(
            np.column_stack([affine_prediction(op, nx)[:, 0] for op in self.operators]),
            0,
            1,
        )
        # Columns already follow HOLD then ascending ID, so argmax supplies exact ties.
        return scores.argmax(axis=1).astype(np.int64), scores


def fit_adequacy_provider(
    features: PreparationPolicyFeatures,
    roots: IntArray,
    adequacy: BoolArray,
    known: BoolArray,
    schedule_ids: tuple[str, ...],
) -> AdequacyProvider:
    if (
        adequacy.shape != (len(features.root_ids), 9)
        or known.shape != adequacy.shape
        or schedule_ids[0] != "hold"
        or schedule_ids[1:] != tuple(sorted(schedule_ids[1:]))
    ):
        raise ValueError("Adequacy labels or exact tie order differ from the frozen provider")
    normalizer = Normalizer.fit(features.x[roots, :, 0])
    nx = normalizer.apply(features.x[roots, :, 0])
    operators = []
    means = []
    for p in range(9):
        good = known[roots, p]
        if not good.any():
            raise ValueError("Unknown lineage cannot be manufactured as provider label zero")
        labels = adequacy[roots[good], p].astype(float)
        operators.append(fit_affine_operator(nx[good], labels[:, None], ridge=10))
        means.append(float(labels.mean()))
    best_fixed = int(np.asarray(means).argmax())
    return AdequacyProvider(
        tuple(map(int, roots)), normalizer, np.stack(operators), schedule_ids, best_fixed
    )


@dataclass(frozen=True)
class DevelopmentGates:
    headroom_roots: int
    adequacy_net_roots: int
    joint_improvement: float

    @property
    def passed(self) -> bool:
        return (
            self.headroom_roots >= 6
            and self.adequacy_net_roots >= 4
            and self.joint_improvement >= 0.05
        )


def development_gates(
    adequacy: BoolArray,
    selected: IntArray,
    fixed: IntArray,
    selected_joint: Array,
    fixed_joint: Array,
) -> DevelopmentGates:
    n = len(adequacy)
    if (
        adequacy.shape != (24, 9)
        or selected.shape != (n,)
        or fixed.shape != (n,)
        or selected_joint.shape != fixed_joint.shape
        or selected_joint.shape[0] != n
        or selected_joint.dtype != np.bool_
        or fixed_joint.dtype != np.bool_
    ):
        raise ValueError("preparation-policy development gates require the complete declared root/request denominator")
    chosen = adequacy[np.arange(n), selected]
    comparator = adequacy[np.arange(n), fixed]
    joint_difference = int(selected_joint.sum()) - int(fixed_joint.sum())
    return DevelopmentGates(
        int((~adequacy[:, 0] & adequacy.any(axis=1)).sum()),
        int(chosen.sum() - comparator.sum()),
        joint_difference / selected_joint.size,
    )
