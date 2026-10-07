"""Outcome-visible response composition prediction and opportunity screens over authenticated scalars.

The numerical regression and frozen mechanism remain owned by existing methods.
This module owns the new paired receiver, root splits and development estimands.
It provides no acquisition, source paths, authority, law finalization or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator
from empirical_lawhood.adapters.methods.prepared_response.models import prepared_mechanism_mean

Array = npt.NDArray[np.float64]
RIDGES = (10.0, 1.0, 0.1)
FAMILIES = ("direct-i0", "direct-i1", "mechanism-i1")
PRIMARY = (2, 4)
LOWER_EPSILON = 1 / 256
COMPOSITION_EPSILON = 1 / 1024


@dataclass(frozen=True, slots=True)
class ResponseCompositionDevelopmentSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-composition/response-composition-development-spec'
    plan_sha256: str
    source_manifest_sha256: str
    q2_terminal_sha256: str
    audit_input_manifest_sha256: str
    study_id: str = "response-composition-adequacy"
    evidence_role: str = "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
    roots_per_context: int = 16
    amplitude: int = 16
    lower_epsilon: Decimal = Decimal(1) / 256
    composition_epsilon: Decimal = Decimal(1) / 1024
    native_updates: int = 0

    def __post_init__(self) -> None:
        for key in (
            "plan_sha256",
            "source_manifest_sha256",
            "q2_terminal_sha256",
            "audit_input_manifest_sha256",
        ):
            validate_sha256(getattr(self, key), field_name=key)
        if (
            self.study_id != "response-composition-adequacy"
            or self.evidence_role != "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
            or type(self.roots_per_context) is not int
            or self.roots_per_context != 16
            or type(self.amplitude) is not int
            or self.amplitude != 16
            or type(self.native_updates) is not int
            or self.native_updates != 0
            or type(self.lower_epsilon) is not Decimal
            or self.lower_epsilon != Decimal(1) / 256
            or type(self.composition_epsilon) is not Decimal
            or self.composition_epsilon != Decimal(1) / 1024
        ):
            raise ValueError("response composition development changes its exposed receiver, roster or precision")


def paired_response(values: Array) -> Array:
    if values.ndim < 3 or values.shape[-3:] != (9, 5, 2) or not np.isfinite(values).all():
        raise ValueError("paired response requires the complete nine-word finite receiver")
    return values - values[..., :1, :, :]


def parent_contrast(response: Array) -> Array:
    if response.shape[-4:] != (5, 9, 5, 2) or not np.isfinite(response).all():
        raise ValueError("parent contrast requires all five parents")
    return response - response[..., :1, :, :, :]


def root_error(prediction: Array, truth: Array) -> Array:
    if prediction.shape != truth.shape or truth.ndim != 6 or truth.shape[1:] != (2, 5, 9, 5, 2):
        raise ValueError("root scoring requires both views and the entire finite panel")
    return cast(Array, np.max(np.abs(prediction - truth)[..., PRIMARY, :], axis=(1, 2, 3, 4, 5)))


def folds(indices: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    if len(indices) != len(set(indices)) or count < 2 or len(indices) < count:
        raise ValueError("folds require distinct whole-root indices")
    return tuple(
        tuple(index for j, index in enumerate(indices) if j % count == k) for k in range(count)
    )


def split_training(indices: tuple[int, ...], held: tuple[int, ...]) -> tuple[int, ...]:
    if not set(held) <= set(indices):
        raise ValueError("held-out root is outside the declared population")
    return tuple(i for i in indices if i not in held)


@dataclass(frozen=True)
class Regression:
    mean: Array
    scale: Array
    operator: Array
    shape: tuple[int, ...]

    def predict(self, x: Array) -> Array:
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("regression inputs change their finite feature contract")
        result = affine_prediction(self.operator, (x - self.mean) / self.scale)
        return result.reshape(len(x), *self.shape)


def regress(x: Array, y: Array, ridge: float) -> Regression:
    if (
        x.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or ridge not in RIDGES
    ):
        raise ValueError("regression requires finite training-only arrays and a declared ridge")
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    operator = fit_affine_operator((x - mean) / scale, y.reshape(len(x), -1), ridge=ridge)
    return Regression(mean, scale, operator, y.shape[1:])


def mechanism(features: Array) -> Array:
    raw = prepared_mechanism_mean(
        features[:, :192].reshape(-1, 16, 12), features[:, 192:], Decimal(16)
    )
    return paired_response(raw)


@dataclass(frozen=True)
class Lower:
    family: str
    regression: Regression

    def predict(self, features: Array, prior: Array | None = None) -> Array:
        x = features[:, :192] if self.family == "direct-i0" else features
        y = self.regression.predict(x)
        if self.family == "mechanism-i1":
            y = y + (mechanism(features) if prior is None else prior)
        return paired_response(y)


def fit_lower(features: Array, response: Array, prior: Array, family: str, ridge: float) -> Lower:
    if family not in FAMILIES:
        raise ValueError("undeclared lower family")
    x = features[:, :192] if family == "direct-i0" else features
    y = response - prior if family == "mechanism-i1" else response
    return Lower(family, regress(x, y, ridge))


def constant_gain(response: Array) -> Array:
    """One training-only axial central chart, with no even correction."""
    mean = response.mean(axis=(0, 1))
    k = np.stack(((mean[2] - mean[1]) / 32, (mean[4] - mean[3]) / 32), axis=-1)
    controls = np.array(
        ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)), dtype=float
    )
    controls[5:] /= np.sqrt(2)
    return cast(Array, np.einsum("wij,dj->dwi", k, controls * 16))


def opportunity_events(
    prediction: Array, truth: Array, widths: Array, preservation: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    if widths.shape != (len(truth),) or preservation.shape != (len(truth), 5):
        raise ValueError("adequacy requires assigned roots and parents")
    losses = np.max(np.abs(prediction - truth)[..., PRIMARY, :], axis=(1, 3, 4, 5))
    numeric = np.max(np.abs(truth[:, 0] - truth[:, 1])[..., PRIMARY, :], axis=(2, 3, 4))
    events = (
        (losses <= LOWER_EPSILON)
        & (widths[:, None] <= LOWER_EPSILON)
        & (numeric <= LOWER_EPSILON / 8)
        & preservation
    )
    hold, oracle = events[:, 0], events.any(axis=1)
    return {
        "events": events,
        "losses": losses,
        "hold": int(hold.sum()),
        "oracle": int(oracle.sum()),
        "maximum_improvement": float(np.mean(oracle.astype(float) - hold.astype(float))),
    }


def develop_context(
    features: Array, response: Array, preservation: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    """Nested whole-root development. No returned status is fresh adjudication."""
    if (
        features.shape != (16, 2, 6, 200)
        or response.shape != (16, 2, 5, 9, 5, 2)
        or not np.isfinite(features).all()
        or not np.isfinite(response).all()
    ):
        raise ValueError("development requires all sixteen assigned context roots and finite views")
    if np.max(np.abs(response[..., 0, :, :])) > 1e-12:
        raise ValueError("response labels must be paired HOLD contrasts")
    pre, handoff = features[:, 0, 0], features[:, :, 1:]
    prior = mechanism(handoff.reshape(-1, 200)).reshape(16, 2, 5, 9, 5, 2)
    candidates = tuple((f, r) for f in FAMILIES for r in RIDGES)
    lower = np.empty_like(response)
    composed = np.empty_like(response)
    direct = np.empty_like(response)
    constant = np.empty_like(response)
    context_mean = np.empty_like(response)
    candidate_predictions = np.empty((len(candidates), *response.shape))
    widths = np.empty(16)
    selection_log: list[dict[str, Any]] = []
    all_indices = tuple(range(16))
    outer_folds = folds(all_indices, 4)
    for outer_held in outer_folds:
        train = split_training(all_indices, outer_held)
        inner_folds = folds(train, 3)
        lower_scores = []
        inner_candidate_predictions = []
        for family, ridge in candidates:
            inner = np.empty((len(train), 2, 5, 9, 5, 2))
            for held in inner_folds:
                fitting = split_training(train, held)
                model = fit_lower(
                    handoff[list(fitting), 0].reshape(-1, 200),
                    response[list(fitting), 0].reshape(-1, 9, 5, 2),
                    prior[list(fitting), 0].reshape(-1, 9, 5, 2),
                    family,
                    ridge,
                )
                rows = [train.index(i) for i in held]
                inner[rows] = model.predict(
                    handoff[list(held)].reshape(-1, 200), prior[list(held)].reshape(-1, 9, 5, 2)
                ).reshape(len(held), 2, 5, 9, 5, 2)
            lower_scores.append(float(np.mean(root_error(inner, response[list(train)]))))
            inner_candidate_predictions.append(inner)
        selected = int(np.argmin(lower_scores))
        family, ridge = candidates[selected]
        widths[list(outer_held)] = root_error(
            inner_candidate_predictions[selected], response[list(train)]
        ).max()
        selected_model = None
        for ci, (candidate_family, candidate_ridge) in enumerate(candidates):
            model = fit_lower(
                handoff[list(train), 0].reshape(-1, 200),
                response[list(train), 0].reshape(-1, 9, 5, 2),
                prior[list(train), 0].reshape(-1, 9, 5, 2),
                candidate_family,
                candidate_ridge,
            )
            candidate_predictions[ci, list(outer_held)] = model.predict(
                handoff[list(outer_held)].reshape(-1, 200),
                prior[list(outer_held)].reshape(-1, 9, 5, 2),
            ).reshape(len(outer_held), 2, 5, 9, 5, 2)
            if ci == selected:
                selected_model = model
        assert selected_model is not None
        lower[list(outer_held)] = candidate_predictions[selected, list(outer_held)]
        upper_scores, direct_scores, upper_residuals = [], [], []
        target_scale = np.maximum(handoff[list(train), 0].std(axis=0), 1e-8)
        for upper_ridge in RIDGES:
            upper_oof = np.empty((len(train), 5, 200))
            direct_oof = np.empty((len(train), 2, 5, 9, 5, 2))
            for held in inner_folds:
                fitting = split_training(train, held)
                rows = [train.index(i) for i in held]
                upper_model = regress(pre[list(fitting)], handoff[list(fitting), 0], upper_ridge)
                upper_oof[rows] = upper_model.predict(pre[list(held)])
                direct_model = regress(pre[list(fitting)], response[list(fitting), 0], upper_ridge)
                direct_oof[rows] = direct_model.predict(pre[list(held)])[:, None]
            upper_scores.append(
                float(
                    np.mean(
                        np.max(
                            np.abs(upper_oof - handoff[list(train), 0]) / target_scale, axis=(1, 2)
                        )
                    )
                )
            )
            direct_scores.append(float(np.mean(root_error(direct_oof, response[list(train)]))))
            upper_residuals.append(handoff[list(train), 0] - upper_oof)
        ui, di = int(np.argmin(upper_scores)), int(np.argmin(direct_scores))
        upper_model = regress(pre[list(train)], handoff[list(train), 0], RIDGES[ui])
        mean_handoff = upper_model.predict(pre[list(outer_held)])
        scenarios = mean_handoff[:, None] + upper_residuals[ui][None]
        forecast = (
            selected_model.predict(scenarios.reshape(-1, 200))
            .reshape(len(outer_held), len(train), 5, 9, 5, 2)
            .mean(axis=1)
        )
        composed[list(outer_held)] = forecast[:, None]
        direct_model = regress(pre[list(train)], response[list(train), 0], RIDGES[di])
        direct[list(outer_held)] = direct_model.predict(pre[list(outer_held)])[:, None]
        constant[list(outer_held)] = constant_gain(response[list(train), 0])[None, None, None]
        context_mean[list(outer_held)] = response[list(train), 0].mean(axis=0)[None, None]
        selection_log.append(
            {
                "outer_held": outer_held,
                "training_roots": train,
                "inner_folds": inner_folds,
                "selected_lower": [family, ridge],
                "lower_candidate_inner_loss": lower_scores,
                "upper_ridge": RIDGES[ui],
                "upper_inner_loss": upper_scores,
                "direct_ridge": RIDGES[di],
                "direct_inner_loss": direct_scores,
                "scenario_donors": train,
                "nominal_width": float(widths[outer_held[0]]),
            }
        )
    contrast = parent_contrast(response)
    lower_error = root_error(lower, response)
    composed_error = root_error(composed, response)
    contrast_error = root_error(parent_contrast(composed), contrast)
    numerical_response = np.max(
        np.abs(response[:, 0] - response[:, 1])[..., PRIMARY, :], axis=(1, 2, 3, 4)
    )
    numerical_contrast = np.max(
        np.abs(contrast[:, 0] - contrast[:, 1])[..., PRIMARY, :], axis=(1, 2, 3, 4)
    )
    qualified = (
        (numerical_response <= COMPOSITION_EPSILON / 8)
        & (numerical_contrast <= COMPOSITION_EPSILON / 8)
        & preservation.all(axis=1)
    )
    passes = (
        qualified
        & (lower_error <= LOWER_EPSILON)
        & (composed_error <= COMPOSITION_EPSILON)
        & (contrast_error <= COMPOSITION_EPSILON)
    )
    parent_sizes = np.max(np.abs(contrast[..., PRIMARY, :]), axis=(1, 3, 4, 5))
    informative_counts = (parent_sizes[:, 1:] > 2 * COMPOSITION_EPSILON).sum(axis=0)
    opportunities = opportunity_events(lower, response, widths, preservation)
    summaries = {}
    for name, value in (
        ("lower", lower),
        ("composed", composed),
        ("direct", direct),
        ("constant_gain", constant),
        ("parent_context_mean", context_mean),
    ):
        errors = root_error(value, response)
        contrast_errors = root_error(parent_contrast(value), contrast)
        summaries[name] = {
            "root_errors": errors.tolist(),
            "median_error": float(np.median(errors)),
            "maximum_error": float(errors.max()),
            "root_parent_contrast_errors": contrast_errors.tolist(),
            "median_parent_contrast_error": float(np.median(contrast_errors)),
            "maximum_parent_contrast_error": float(contrast_errors.max()),
        }
    return {
        "lower": lower,
        "composed": composed,
        "direct": direct,
        "constant_gain": constant,
        "parent_context_mean": context_mean,
        "candidate_predictions": candidate_predictions,
        "nominal_widths": widths,
        "adequacy_events": opportunities["events"],
        "summary": {
            "selection_log": selection_log,
            "predictors": summaries,
            "qualified_roots": int(qualified.sum()),
            "design1_passing_roots": int(passes.sum()),
            "design1_root_pass": passes.tolist(),
            "nontrivial_contrast_counts_by_parent": informative_counts.tolist(),
            "design1_composition_ready": bool(passes.sum() >= 12),
            "design1_contrast_informative": bool(informative_counts.max() >= 8),
            "design2_hold_adequate": opportunities["hold"],
            "design2_oracle_adequate": opportunities["oracle"],
            "design2_maximum_improvement": opportunities["maximum_improvement"],
            "design2_oracle_opportunity": bool(opportunities["maximum_improvement"] >= 0.15),
            "numerical_response_discrepancies": numerical_response.tolist(),
            "numerical_contrast_discrepancies": numerical_contrast.tolist(),
            "nominal_widths": widths.tolist(),
        },
    }
