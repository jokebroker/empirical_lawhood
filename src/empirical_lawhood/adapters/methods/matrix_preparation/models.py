"""Frozen finite-chart observable fitting, with all resampling at root level.

These producers emit models and residual operands. Scientific qualification is
performed separately by the registered assessment owner. Predictors accept only
the nine measured features; future signed outcomes enter fitting targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import PreparationRoot
from empirical_lawhood.adapters.simulators.matrix_preparation.crossfit_inputs import preparation_crossfit_rank

from .contracts import CANDIDATES, FEATURES, PENALTIES


Array = npt.NDArray[np.float64]
SIGNS = np.array([-1.0, 0.0, 1.0])
BENCHMARKS = (
    "scalar.raw",
    "scalar.affine",
    "full_hessian.raw",
    "full_hessian.affine",
    "microscopic.raw",
    "microscopic.affine",
)
MODEL_ROSTER = (*CANDIDATES, *BENCHMARKS)
ROLE = {"fit": 0, "interval": 1, "screen": 2}


def chart_components(response: Array) -> Array:
    """Final axes NEG,HOLD,POS by readout -> baseline,odd,even by readout."""
    if response.shape[-2:] != (3, 5):
        raise ValueError("finite response chart requires three signs and five readouts")
    negative, hold, positive = (response[..., i, :] for i in range(3))
    return np.stack((hold, (positive - negative) / 2, (positive + negative) / 2 - hold), axis=-2)


def chart_predictions(components: Array) -> Array:
    if components.shape[-2:] != (3, 5):
        raise ValueError("finite response coefficients require baseline,odd,even by readout")
    baseline, odd, even = (components[..., i, :] for i in range(3))
    return np.stack(tuple(baseline + sign * odd + sign**2 * even for sign in SIGNS), axis=-2)


def root_roles(roots: tuple[PreparationRoot, ...]) -> npt.NDArray[np.int64]:
    return np.array([ROLE[r.development_role] for r in roots], dtype=np.int64)


def root_folds(roots: tuple[PreparationRoot, ...]) -> npt.NDArray[np.int64]:
    """No observations enter the fold assignment; all nested rows move together."""
    if not roots or len({r.context for r in roots}) != 1 or len(set(roots)) != len(roots):
        raise ValueError("root folds need one context and distinct roots")
    order = sorted(
        range(len(roots)),
        key=lambda i: preparation_crossfit_rank(roots[i].context, roots[i].index),
    )
    folds = np.empty(len(roots), dtype=np.int64)
    folds[order] = np.arange(len(roots)) % 4
    return folds


def _features(features: Array, mean: Array, scale: Array, candidate: str) -> Array:
    if features.shape[-1] != len(FEATURES) or mean.shape != (9,) or scale.shape != (9,):
        raise ValueError("observable model accepts exactly its nine measured features")
    matrix = (np.asarray(features, dtype=np.float64).reshape(-1, 9) - mean) / scale
    if candidate == "direct_quadratic":
        matrix = np.column_stack(
            (matrix, *(matrix[:, i] * matrix[:, j] for i in range(9) for j in range(i, 9)))
        )
    elif candidate not in ("constant_gain", "direct_linear"):
        raise ValueError("observable candidate is outside its predeclared library")
    return matrix


@dataclass(frozen=True)
class ObservablePredictor:
    candidate: str
    penalty: float
    mean: Array
    scale: Array
    operator: Array
    constant_odd_even: Array | None
    training_roots: int

    def predict(self, features: Array) -> Array:
        matrix = _features(features, self.mean, self.scale, self.candidate)
        predicted = np.asarray(affine_prediction(self.operator, matrix), dtype=np.float64)
        if self.candidate == "constant_gain":
            if self.constant_odd_even is None or self.constant_odd_even.shape != (2, 5):
                raise ValueError("constant-gain model omitted its complete finite chart")
            components = np.empty((len(matrix), 3, 5))
            components[:, 0] = predicted
            components[:, 1:] = self.constant_odd_even
        else:
            components = predicted.reshape(-1, 3, 5)
        return chart_predictions(components).reshape(*features.shape[:-1], 3, 5)


def fit_observable(
    features: Array, response: Array, candidate: str, penalty: float
) -> ObservablePredictor:
    """Equal roots, parents and views; K=4 targets remain nested noisy futures.

    Inputs: root,parent,view,feature and root,parent,view,audit,sign,readout.
    The shared affine scientific owner fits the complete b/g/e target. K-mean
    targets give the same least-squares minimizer as equal-weight repeated rows.
    """
    if (
        features.ndim != 4
        or features.shape[1:] != (5, 2, 9)
        or response.shape != (*features.shape[:3], 4, 3, 5)
        or len(features) < 8
        or candidate not in CANDIDATES
        or penalty not in PENALTIES
        or not np.isfinite(features).all()
        or not np.isfinite(response).all()
    ):
        raise ValueError("observable fit changes its complete training-root measurement chart")
    raw = features.reshape(-1, 9)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    matrix = _features(features, mean, scale, candidate)
    target = chart_components(response.mean(axis=3)).reshape(-1, 3, 5)
    constant = target[:, 1:].mean(axis=0) if candidate == "constant_gain" else None
    values = target[:, 0] if constant is not None else target.reshape(-1, 15)
    # The fixed penalties apply to mean squared loss. Repeating views/futures
    # cannot weaken regularization or enlarge the independent-root census.
    operator = fit_affine_operator(matrix, values, ridge=penalty * len(matrix))
    return ObservablePredictor(candidate, penalty, mean, scale, operator, constant, len(features))


def normalized_loss(prediction: Array, response: Array) -> float:
    if response.shape != (*prediction.shape[:-2], 4, 3, 5):
        raise ValueError("validation loss must preserve independent audit futures")
    errors = prediction[..., None, :, :] - response
    odd_errors = (errors[..., 2, :] - errors[..., 0, :]) / 2
    return float(np.mean((errors / 0.125) ** 2) + np.mean((odd_errors / 0.015625) ** 2))


@dataclass(frozen=True)
class ResidualScaleModel:
    """Positive observable scales, fit to whole-root out-of-fold residuals.

    Separate numerical-view fits use the same nine features, fixed ridge 1.0
    and log mean-square targets over the four audit futures. These scales are
    frozen before any interval-calibration outcome is used.
    """

    means: Array
    scales: Array
    operators: Array
    reference_rms: Array
    training_roots: int

    def predict(self, features: Array) -> Array:
        if features.ndim != 4 or features.shape[1:] != (5, 2, 9):
            raise ValueError("residual scale model requires the causal observation chart")
        values = np.full((*features.shape[:3], 3, 5), np.nan)
        for view in range(2):
            matrix = (features[:, :, view].reshape(-1, 9) - self.means[view]) / self.scales[view]
            log_variance = affine_prediction(self.operators[view], matrix)
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = np.maximum(np.exp(0.5 * log_variance), 1 / 128)
            values[:, :, view] = predicted.reshape(len(features), 5, 3, 5)
        return values


def fit_residual_scales(features: Array, out_of_fold_residuals: Array) -> ResidualScaleModel:
    if (
        features.shape != (32, 5, 2, 9)
        or out_of_fold_residuals.shape != (32, 5, 2, 4, 3, 5)
        or not np.isfinite(features).all()
        or not np.isfinite(out_of_fold_residuals).all()
    ):
        raise ValueError("scale fit requires all 32 training roots and their out-of-fold residuals")
    means, scales, operators = np.empty((2, 9)), np.empty((2, 9)), np.empty((2, 10, 15))
    for view in range(2):
        raw = features[:, :, view].reshape(-1, 9)
        means[view], scales[view] = raw.mean(axis=0), raw.std(axis=0)
        scales[view, scales[view] <= np.finfo(float).eps] = 1.0
        variance = np.mean(out_of_fold_residuals[:, :, view] ** 2, axis=2)
        target = np.log(np.maximum(variance, (1 / 128) ** 2)).reshape(-1, 15)
        operators[view] = fit_affine_operator(
            (raw - means[view]) / scales[view], target, ridge=float(len(raw))
        )
    reference = np.maximum(np.sqrt(np.mean(out_of_fold_residuals**2, axis=(0, 1, 3))), 1 / 128)
    return ResidualScaleModel(means, scales, operators, reference, len(features))


@dataclass(frozen=True)
class ObservableFit:
    model: ObservablePredictor
    penalty_losses: Array
    fit_fold_predictions: Array
    all_root_fold_predictions: Array
    predictions: Array
    scale_model: ResidualScaleModel
    residual_scales: Array
    provisional_quantile: float
    development_quantile: float
    provisional_scores: Array
    development_scores: Array


def menu_scores(response: Array, prediction: Array, residual_scales: Array) -> Array:
    """One innovation bundle per root; max over the whole parent/view/chart.

    A second audit axis is deliberately rejected. Infinity retains unresolved
    roots as worst cases; an invalid root is never removed from calibration.
    """
    if (
        response.shape != prediction.shape
        or response.ndim != 5
        or response.shape[1:] != (5, 2, 3, 5)
        or residual_scales.shape != prediction.shape
        or np.any(np.isfinite(residual_scales) & (residual_scales < 1 / 128))
    ):
        raise ValueError(
            "calibration requires one full-menu score per root and fixed finite scales"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        errors = np.abs(response - prediction) / residual_scales
    scores = np.max(errors, axis=(1, 2, 3, 4))
    known = np.all(np.isfinite(residual_scales), axis=(1, 2, 3, 4))
    return np.where(np.isfinite(scores) & known, scores, np.inf)


def conformal_quantile(scores: Array, level: float = 0.95) -> float:
    if (
        scores.ndim != 1
        or not len(scores)
        or np.any(np.isnan(scores))
        or np.any(scores < 0)
        or level != 0.95
    ):
        raise ValueError("calibration must retain every nonnegative root score at the frozen level")
    rank = ceil((len(scores) + 1) * level)
    return np.inf if rank > len(scores) else float(np.sort(scores)[rank - 1])


def fit_candidate(
    roots: tuple[PreparationRoot, ...],
    features: Array,
    response: Array,
    candidate: str,
) -> ObservableFit:
    if len(roots) != 64 or features.shape != (64, 5, 2, 9) or response.shape != (64, 5, 2, 4, 3, 5):
        raise ValueError("development fit must retain its full 64-root context census")
    roles = root_roles(roots)
    train, interval = roles == 0, roles == 1
    fit_roots = tuple(r for r, chosen in zip(roots, train, strict=True) if chosen)
    fit_folds, all_folds = root_folds(fit_roots), root_folds(roots)
    losses, crossfit = [], []
    for penalty in PENALTIES:
        predicted = np.empty((32, 5, 2, 3, 5))
        for fold in range(4):
            fitted = fit_observable(
                features[train][fit_folds != fold],
                response[train][fit_folds != fold],
                candidate,
                penalty,
            )
            predicted[fit_folds == fold] = fitted.predict(features[train][fit_folds == fold])
        losses.append(normalized_loss(predicted, response[train]))
        crossfit.append(predicted)
    chosen = min(range(len(PENALTIES)), key=lambda i: (losses[i], -PENALTIES[i]))
    fitted = fit_observable(features[train], response[train], candidate, PENALTIES[chosen])
    fit_predictions = crossfit[chosen]
    residuals = fit_predictions[..., None, :, :] - response[train]
    scale_model = fit_residual_scales(features[train], residuals)
    scales = scale_model.predict(features)
    predicted = fitted.predict(features)
    provisional = menu_scores(response[interval, :, :, 0], predicted[interval], scales[interval])
    all_predictions = np.full_like(predicted, np.nan)
    for fold in range(4):
        if (
            not np.isfinite(features[all_folds != fold]).all()
            or not np.isfinite(response[all_folds != fold]).all()
        ):
            continue
        alternate = fit_observable(
            features[all_folds != fold], response[all_folds != fold], candidate, PENALTIES[chosen]
        )
        all_predictions[all_folds == fold] = alternate.predict(features[all_folds == fold])
    development = menu_scores(response[:, :, :, 0], all_predictions, scales)
    return ObservableFit(
        fitted,
        np.asarray(losses),
        fit_predictions,
        all_predictions,
        predicted,
        scale_model,
        scales,
        conformal_quantile(provisional),
        conformal_quantile(development),
        provisional,
        development,
    )


@dataclass(frozen=True)
class GainCalibration:
    intercept: Array
    slope: Array
    minimum: Array
    maximum: Array
    eligible: npt.NDArray[np.bool_]

    def predict(self, gain: Array) -> Array:
        if gain.shape[-1] != 5:
            raise ValueError("gain calibration changes its fixed readout chart")
        return np.asarray(self.intercept + self.slope * gain, dtype=np.float64)


def calibrate_gain(gain: Array, observed: Array) -> GainCalibration:
    """Imported positive-slope, training-only affine recipe, applied to every j.

    Parents and numerical views are nested equally within each training root.
    Coefficient rank failure is retained as ineligible, with no fallback retune.
    """
    if (
        gain.shape != observed.shape
        or gain.ndim != 4
        or gain.shape[1:] != (5, 2, 5)
        or len(gain) < 8
    ):
        raise ValueError("gain calibration requires root,parent,view,readout training arrays")
    if not np.isfinite(gain).all() or not np.isfinite(observed).all():
        raise ValueError("gain calibration requires the complete finite training roster")
    x, y = gain.reshape(-1, 5), observed.reshape(-1, 5)
    intercept, slope = np.full(5, np.nan), np.full(5, np.nan)
    eligible = np.zeros(5, dtype=bool)
    for j in range(5):
        mean, scale = float(x[:, j].mean()), float(x[:, j].std())
        if scale <= np.finfo(float).eps:
            continue
        z = ((x[:, j] - mean) / scale)[:, None]
        if np.linalg.matrix_rank(np.column_stack((z, np.ones(len(x))))) != 2:
            continue
        operator = fit_affine_operator(z, y[:, j, None], ridge=0.0)
        slope[j] = operator[0, 0] / scale
        intercept[j] = operator[1, 0] - slope[j] * mean
        eligible[j] = np.isfinite([intercept[j], slope[j]]).all() and slope[j] > 0
    return GainCalibration(intercept, slope, x.min(axis=0), x.max(axis=0), eligible)


def privileged_chart(observable_baseline_even: Array, gain: Array) -> Array:
    """Absolute benchmark has the SAME causal observable baseline/even parts.

    It cannot use the realized future HOLD to repair a privileged gain forecast.
    """
    if gain.shape != observable_baseline_even.shape[:-2] + (5,):
        raise ValueError("privileged gain differs from its companion observable chart")
    components = chart_components(observable_baseline_even).copy()
    components[..., 1, :] = gain
    return chart_predictions(components)


@dataclass(frozen=True)
class AdequacyEvents:
    absolute_maximum: Array
    odd_maximum: Array
    point_adequacy: npt.NDArray[np.bool_]
    simultaneous_coverage: npt.NDArray[np.bool_]
    width_pass: npt.NDArray[np.bool_]
    numerical: npt.NDArray[np.bool_]
    contact: npt.NDArray[np.bool_]
    preservation: npt.NDArray[np.bool_]
    delivery: npt.NDArray[np.bool_]
    usable: npt.NDArray[np.bool_]


def adequacy_events(
    prediction: Array,
    response: Array,
    halfwidth: Array,
    delivered: Array,
    contact: Array,
    preservation: Array,
) -> AdequacyEvents:
    """A and U are separate; every event has root,parent,bundle axes.

    Native arrays retain both views. Unknown physical/delivery/numerical flags
    fail U, while raw finite point errors still define the separate A readout.
    """
    if (
        prediction.ndim != 5
        or prediction.shape[1:] != (5, 2, 3, 5)
        or response.ndim != 6
        or response.shape[3] not in (1, 4, 5)
        or response.shape != (*prediction.shape[:3], response.shape[3], 3, 5)
        or halfwidth.shape != prediction.shape
        or delivered.shape != response.shape[:-1]
        or preservation.shape != delivered.shape
        or contact.shape != response.shape[:-2]
    ):
        raise ValueError(
            "adequacy events change the complete paired native view/action/bundle chart"
        )
    errors = np.abs(prediction[..., None, :, :] - response)
    odd_prediction = (prediction[..., 2, :] - prediction[..., 0, :]) / 2
    odd_response = (response[..., 2, :] - response[..., 0, :]) / 2
    absolute = np.max(errors, axis=(2, 4, 5)) / 0.125
    odd = np.max(np.abs(odd_prediction[..., None, :] - odd_response), axis=(2, 4)) / 0.015625
    point = (absolute <= 1) & (odd <= 1)
    width = np.all(np.isfinite(halfwidth) & (halfwidth >= 0) & (halfwidth <= 0.125), axis=(2, 3, 4))
    coverage = np.all(np.isfinite(errors) & (errors <= halfwidth[..., None, :, :]), axis=(2, 4, 5))
    numerical = (np.max(np.abs(response[:, :, 0] - response[:, :, 1]), axis=(3, 4)) <= 1 / 128) & (
        np.max(np.abs(odd_response[:, :, 0] - odd_response[:, :, 1]), axis=3) <= 1 / 256
    )
    native_contact = np.all(contact == 1, axis=2)
    preserved = np.all(preservation == 1, axis=(2, 4))
    delivery = np.all(delivered == 1, axis=(2, 4))
    usable = point & width[..., None] & numerical & native_contact & preserved & delivery
    return AdequacyEvents(
        absolute,
        odd,
        point,
        coverage,
        width,
        numerical,
        native_contact,
        preserved,
        delivery,
        usable,
    )
