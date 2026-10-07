"""Joint-event forecasts from the permitted preparent and handoff observers."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import expit

from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import PreparationRoot
from .contracts import PENALTIES
from .models import root_folds, root_roles


Array = npt.NDArray[np.float64]
EVENTS = ("point_adequacy", "usable", "preservation")


@dataclass(frozen=True)
class EventPredictor:
    stage: str
    penalty: float
    mean: Array
    scale: Array
    coefficients: Array
    constant: float | None
    lower: Array
    upper: Array
    training_roots: int

    def predict(self, features: Array) -> Array:
        if features.ndim != 3 or features.shape[1:] != (5, len(self.mean)):
            raise ValueError("joint-event predictor received another information chart")
        matrix = features.reshape(-1, len(self.mean))
        known = np.isfinite(matrix).all(axis=1)
        result = np.full(len(matrix), np.nan)
        result[known] = (
            self.constant
            if self.constant is not None
            else expit(
                ((matrix[known] - self.mean) / self.scale) @ self.coefficients[:-1]
                + self.coefficients[-1]
            )
        )
        return result.reshape(features.shape[:2])

    def supported(self, features: Array) -> npt.NDArray[np.bool_]:
        if features.ndim != 3 or features.shape[1:] != (5, len(self.mean)):
            raise ValueError("support check received another information chart")
        return np.asarray(
            np.all(
                np.isfinite(features) & (features >= self.lower) & (features <= self.upper), axis=-1
            ),
            dtype=bool,
        )


def forecast_features(preparent: Array, handoff: Array) -> tuple[Array, Array]:
    """Selectors use the primary observer, plus the declared parent identity.

    Numerical twins are error checks, never extra independent feature rows.
    Handoff features carry the parent's actual observed consequence causally.
    """
    if (
        preparent.ndim != 2
        or preparent.shape[1:] != (9,)
        or handoff.shape != (len(preparent), 5, 2, 9)
    ):
        raise ValueError("forecast observation chart changes its causal scalar inputs")
    before = np.concatenate(
        (
            np.repeat(preparent[:, None], 5, axis=1),
            np.broadcast_to(np.eye(5), (len(preparent), 5, 5)),
        ),
        axis=-1,
    )
    return before, handoff[:, :, 0]


def fit_event_predictor(
    features: Array, labels: Array, *, stage: str, penalty: float
) -> EventPredictor:
    expected_features = 14 if stage == "preparent" else 9 if stage == "handoff" else 0
    if (
        features.ndim != 3
        or features.shape[1:] != (5, expected_features)
        or len(features) < 8
        or labels.shape != (len(features), 5, 4)
        or penalty not in PENALTIES
        or not np.isfinite(features).all()
        or not np.isfinite(labels).all()
        or np.any((labels != 0) & (labels != 1))
    ):
        raise ValueError("joint-event fit changes its complete root/event/information chart")
    matrix = features.reshape(-1, expected_features)
    y = labels.mean(axis=2).ravel()
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    lower, upper = features.min(axis=0), features.max(axis=0)
    z = (matrix - mean) / scale
    constant = float(y[0]) if np.all(y == y[0]) and y[0] in (0, 1) else None
    theta = np.zeros(expected_features + 1)
    if constant is None:

        def objective(parameters: Array) -> tuple[float, Array]:
            linear = z @ parameters[:-1] + parameters[-1]
            error = expit(linear) - y
            value = float(
                np.mean(np.logaddexp(0.0, linear) - y * linear)
                + penalty * np.dot(parameters[:-1], parameters[:-1]) / 2
            )
            gradient = np.r_[z.T @ error / len(y) + penalty * parameters[:-1], error.mean()]
            return value, np.asarray(gradient, dtype=np.float64)

        fitted = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
        )
        if not fitted.success and np.linalg.norm(objective(fitted.x)[1], ord=np.inf) > 1e-6:
            raise FloatingPointError("preparation joint-event logistic fit did not converge")
        theta = np.asarray(fitted.x, dtype=np.float64)
    return EventPredictor(
        stage, penalty, mean, scale, theta, constant, lower, upper, len(features)
    )


@dataclass(frozen=True)
class ForecastFit:
    preparent_model: EventPredictor
    handoff_model: EventPredictor
    preparent_predictions: Array
    handoff_predictions: Array
    context_intercept: float
    parent_intercepts: Array
    penalty_losses: Array
    preparent_support: npt.NDArray[np.bool_]
    handoff_support: npt.NDArray[np.bool_]


def fit_forecast(
    roots: tuple[PreparationRoot, ...], preparent: Array, handoff: Array, labels: Array
) -> ForecastFit:
    if len(roots) != 64 or labels.shape != (64, 5, 4):
        raise ValueError("forecast development must retain all 64 root slots")
    before, after = forecast_features(preparent, handoff)
    training = root_roles(roots) == 0
    folds = root_folds(tuple(r for r, use in zip(roots, training, strict=True) if use))
    models, scores = [], []
    for stage, features in (("preparent", before), ("handoff", after)):
        losses = []
        for penalty in PENALTIES:
            predictions = np.empty((32, 5))
            for fold in range(4):
                fitted = fit_event_predictor(
                    features[training][folds != fold],
                    labels[training][folds != fold],
                    stage=stage,
                    penalty=penalty,
                )
                predictions[folds == fold] = fitted.predict(features[training][folds == fold])
            losses.append(float(np.mean((predictions[..., None] - labels[training]) ** 2)))
        chosen = min(range(len(PENALTIES)), key=lambda i: (losses[i], -PENALTIES[i]))
        models.append(
            fit_event_predictor(
                features[training], labels[training], stage=stage, penalty=PENALTIES[chosen]
            )
        )
        scores.append(losses)
    return ForecastFit(
        models[0],
        models[1],
        models[0].predict(before),
        models[1].predict(after),
        float(labels[training].mean()),
        labels[training].mean(axis=(0, 2)),
        np.asarray(scores),
        models[0].supported(before),
        models[1].supported(after),
    )


def brier_roots(probability: Array, events: Array) -> Array:
    if events.shape != (*probability.shape, 4) or probability.ndim not in (1, 2):
        raise ValueError("Brier evaluation changes its root/parent/independent-audit axes")
    squared = (probability[..., None] - events) ** 2
    # An unavailable forecast earns the worst bounded score; denominator stays.
    squared = np.where(np.isfinite(squared), squared, 1.0)
    return np.asarray(squared.mean(axis=tuple(range(1, squared.ndim))), dtype=np.float64)


def forecast_contact(
    probability: Array, events: Array, constant: float, parent_intercepts: Array
) -> dict[str, float | int | bool]:
    if (
        probability.shape != (16, 5)
        or events.shape != (16, 5, 4)
        or parent_intercepts.shape != (5,)
    ):
        raise ValueError("forecast contact requires the whole held development-screen roster")
    finite = np.isfinite(probability)
    spread = (
        float(np.max(probability[finite]) - np.min(probability[finite])) if finite.any() else 0.0
    )
    events_roots = int(np.any(events == 1, axis=(1, 2)).sum())
    nonevents_roots = int(np.any(events == 0, axis=(1, 2)).sum())
    candidate = brier_roots(probability, events)
    constant_difference = float(
        np.mean(brier_roots(np.full_like(probability, constant), events) - candidate)
    )
    parent_difference = float(
        np.mean(
            brier_roots(np.broadcast_to(parent_intercepts, probability.shape), events)
            - candidate
        )
    )
    return {
        "event_roots": events_roots,
        "nonevent_roots": nonevents_roots,
        "forecast_spread": spread,
        "constant_brier_improvement": constant_difference,
        "parent_brier_improvement": parent_difference,
        "passes": events_roots >= 8
        and nonevents_roots >= 8
        and spread >= 0.10
        and constant_difference > 0
        and parent_difference > 0,
    }
