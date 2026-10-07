"""development calibration, validation operands and causal support fitting.

These finite operands feed the registered qualification owner. They do not
construct ResponseLaw objects, authorize actions or establish population bounds.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
from scipy.stats import t as student_t

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.adapters.methods.causal_access_tournament import (
    FeatureMap,
    StandardizedRidge,
    fit_standardized_ridge,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS

from .development_models import DEVELOPMENT_DELTA, REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentModelGroup, ResponseGeometryDevelopmentRootSeries, FloatArray, Representation, development_series, endpoint_predictions, organize_response_geometry_development_views, root_series, select_response_geometry_development_models
from .projection import _decimal


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-calibration'
    context: str
    representations: tuple[str, ...]
    roots: tuple[int, ...]
    root_scores: tuple[Decimal | None, ...]
    quantile: Decimal | None

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.roots != tuple(range(16, 32))
            or len(self.root_scores) != 16
        ):
            raise ValueError("development calibration requires the exact 16 calibration roots")
        if self.representations != tuple(r for r in REPRESENTATIONS if r in self.representations):
            raise ValueError("development calibration changes the selected representation roster")
        if any(
            value is not None and (not value.is_finite() or value < 0)
            for value in (*self.root_scores, self.quantile)
        ):
            raise ValueError("development calibration scores must be finite nonnegative or unavailable")
        expected = (
            None
            if not self.representations or any(v is None for v in self.root_scores)
            else max(v for v in self.root_scores if v is not None)
        )
        if self.quantile != expected:
            raise ValueError(
                "development 90% finite-split rank is ceil(.9*17)=16, including unavailable roots"
            )


def _standardized(errors: FloatArray, deviations: FloatArray) -> FloatArray:
    # Exact deterministic predictions can have zero scale. An unexplained
    # nonzero error at zero scale remains unbounded, not an epsilon rescue.
    return np.divide(
        np.abs(errors), deviations, out=np.where(errors == 0, 0.0, np.inf), where=deviations > 0
    )


def calibrate_response_geometry_development_context(
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    *,
    context: str,
) -> ResponseGeometryDevelopmentCalibration:
    if tuple(g.representation for g in groups) != REPRESENTATIONS or any(
        g.context != context for g in groups
    ):
        raise ValueError("development calibration model roster/context differs")
    pairs = organize_response_geometry_development_views(views, context=context, role="calibration")
    selected = tuple(g for g in groups if g.models)
    scores = []
    for pair in pairs.values():
        worst = 0.0 if selected else np.inf
        for group in selected:
            for model in group.models:
                series = root_series(pair, model.parent, group.representation)
                if series is None:
                    worst = np.inf
                    continue
                try:
                    prediction, deviation, actual = endpoint_predictions(model, series)
                    worst = max(worst, float(np.max(_standardized(prediction - actual, deviation))))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    worst = np.inf
        scores.append(None if not np.isfinite(worst) else _decimal(worst))
    quantile = (
        None
        if any(value is None for value in scores)
        else max(value for value in scores if value is not None)
    )
    return ResponseGeometryDevelopmentCalibration(
        context, tuple(g.representation for g in selected), tuple(pairs), tuple(scores), quantile
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentModelScreen(CanonicalRecord):
    """Raw finite validation operands; qualification applies the frozen thresholds."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-model-screen'
    context: str
    representation: str
    usable_roots: tuple[int, ...]
    refused_roots: tuple[int, ...]
    mean_error: Decimal | None
    rmse: Decimal | None
    joint_coverage: Decimal | None
    median_root_maximum_halfwidth: Decimal | None
    root_mean_errors: tuple[Decimal, ...]
    root_mean_squared_errors: tuple[Decimal, ...]
    root_joint_covered: tuple[bool | None, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.representation not in REPRESENTATIONS
        ):
            raise ValueError("development model screen changes its context/representation")
        if tuple(sorted((*self.usable_roots, *self.refused_roots))) != tuple(range(32, 64)) or any(
            values != tuple(sorted(set(values)))
            for values in (self.usable_roots, self.refused_roots)
        ):
            raise ValueError("development screen must retain every untouched validation root")
        if any(
            len(v) != len(self.usable_roots)
            for v in (self.root_mean_errors, self.root_mean_squared_errors, self.root_joint_covered)
        ):
            raise ValueError("development screen root metrics lose their independent denominator")
        values = (
            self.mean_error,
            self.rmse,
            self.joint_coverage,
            self.median_root_maximum_halfwidth,
            *self.root_mean_errors,
            *self.root_mean_squared_errors,
        )
        if any(value is not None and not value.is_finite() for value in values):
            raise ValueError("development finite screen cannot serialize nonfinite operands")

    @property
    def finite_screen_passes(self) -> bool:
        return (
            len(self.usable_roots) >= 16
            and self.mean_error is not None
            and self.rmse is not None
            and self.joint_coverage is not None
            and self.median_root_maximum_halfwidth is not None
            and abs(self.mean_error) <= Decimal("0.03125")
            and self.rmse <= Decimal("0.125")
            and self.joint_coverage >= Decimal("0.90")
            and self.median_root_maximum_halfwidth <= Decimal("0.25")
        )


@dataclass(frozen=True)
class ResponseGeometryDevelopmentValidationOperands:
    screen: ResponseGeometryDevelopmentModelScreen
    predictions: FloatArray
    errors: FloatArray
    halfwidths: FloatArray


def validate_response_geometry_development_context(
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
    calibration: ResponseGeometryDevelopmentCalibration,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    *,
    context: str,
) -> tuple[ResponseGeometryDevelopmentValidationOperands, ...]:
    if (
        tuple(g.representation for g in groups) != REPRESENTATIONS
        or any(g.context != context for g in groups)
        or calibration.context != context
    ):
        raise ValueError("development validation model/calibration/context differs")
    pairs = organize_response_geometry_development_views(views, context=context, role="validation")
    quantile = None if calibration.quantile is None else float(calibration.quantile)
    output = []
    for group in groups:
        predictions = np.full((32, 5, 2, 3), np.nan)
        errors, widths = np.full_like(predictions, np.nan), np.full_like(predictions, np.nan)
        for index, pair in pairs.items():
            for parent_index, model in enumerate(group.models):
                series = root_series(pair, model.parent, group.representation)
                if series is None:
                    continue
                try:
                    prediction, deviation, actual = endpoint_predictions(model, series)
                    predictions[index - 32, parent_index] = prediction
                    errors[index - 32, parent_index] = prediction - actual
                    if quantile is not None:
                        widths[index - 32, parent_index] = quantile * deviation
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    pass
        complete = np.isfinite(errors).all(axis=(1, 2, 3))
        finite_errors = errors[complete]
        root_errors = finite_errors.mean(axis=(1, 2, 3)) if len(finite_errors) else np.empty(0)
        root_losses = (
            np.square(finite_errors).mean(axis=(1, 2, 3)) if len(finite_errors) else np.empty(0)
        )
        bands_known = bool(quantile is not None and np.isfinite(widths[complete]).all())
        covered = np.asarray((np.abs(finite_errors) <= widths[complete]).all(axis=(1, 2, 3)))
        summary = ResponseGeometryDevelopmentModelScreen(
            context,
            group.representation,
            tuple(int(i + 32) for i in np.flatnonzero(complete)),
            tuple(int(i + 32) for i in np.flatnonzero(~complete)),
            _decimal(float(root_errors.mean())) if len(root_errors) else None,
            _decimal(float(np.sqrt(root_losses.mean()))) if len(root_losses) else None,
            _decimal(float(covered.mean())) if len(covered) and bands_known else None,
            _decimal(float(np.median(np.max(widths[complete], axis=(1, 2, 3)))))
            if len(root_errors) and bands_known
            else None,
            tuple(_decimal(float(v)) for v in root_errors),
            tuple(_decimal(float(v)) for v in root_losses),
            tuple(bool(v) if bands_known else None for v in covered),
        )
        output.append(ResponseGeometryDevelopmentValidationOperands(summary, predictions, errors, widths))
    return tuple(output)


def response_geometry_development_qualification_metrics(
    operands: ResponseGeometryDevelopmentValidationOperands,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
) -> tuple[NamedDecimal, ...]:
    """Raw finite law-screen operands, retaining whole-root inference and missing cells.

    These metrics carry no generic qualification status. The development profile applies
    its frozen criteria; the common qualification service owns the terminal.
    """
    screen = operands.screen
    pairs = organize_response_geometry_development_views(views, context=screen.context, role="validation")
    finite = np.isfinite(operands.errors).all(axis=(1, 2, 3))
    indices = np.flatnonzero(finite)
    if tuple(int(i + 32) for i in indices) != screen.usable_roots:
        raise ValueError("development law operands change their validation-root denominator")
    numbers: dict[str, tuple[float, str]] = {
        "usable-roots": (float(len(indices)), "1"),
        "minimum-invocation-repeats": (
            float(
                min(
                    sum(
                        pairs[int(i) + 32][0].report.root.invocation_offset == offset
                        for i in indices
                    )
                    for offset in range(384, 513, 16)
                )
            ),
            "1",
        ),
    }
    for name, value, unit in (
        ("mean-error", screen.mean_error, "hilbert-schmidt-native"),
        ("rmse", screen.rmse, "hilbert-schmidt-native"),
        ("joint-coverage", screen.joint_coverage, "1"),
        (
            "median-root-maximum-halfwidth",
            screen.median_root_maximum_halfwidth,
            "hilbert-schmidt-native",
        ),
    ):
        if value is not None:
            numbers[name] = (float(value), unit)
    contacts, numerical = [], []
    for parent_index, parent in enumerate(PARENTS):
        count = 0
        for index in indices:
            packets = tuple(
                next(p for p in view.report.packets if p.parent == parent)
                for view in pairs[int(index) + 32]
            )
            count += int(all(packet.contact is True for packet in packets))
            first, second = (np.asarray(packet.signed_responses, dtype=float) for packet in packets)
            if len(first) != 3 or len(second) != 3:
                raise ValueError(
                    "development predicted validation root lacks the complete measured action triplet"
                )
            numerical.append(float(np.max(np.abs(first - second))))
        contacts.append(count)
    numbers["minimum-parent-contact-roots"] = (float(min(contacts)), "1")
    if numerical:
        numbers["maximum-paired-numerical-error"] = (max(numerical), "hilbert-schmidt-native")
    if len(indices):
        prediction = operands.predictions[finite]
        errors = operands.errors[finite]
        actual = prediction - errors
        swapped_errors = prediction[..., ::-1] - actual
        active = (0, 2)
        improvement = (
            np.square(swapped_errors[..., active]) - np.square(errors[..., active])
        ).mean(axis=(1, 2, 3)) / DEVELOPMENT_DELTA**2
        informative = (
            np.max(np.abs(prediction[..., 2] - prediction[..., 0]), axis=(1, 2)) >= DEVELOPMENT_DELTA / 64
        )
        numbers["wrong-sign-loss-increase"] = (float(improvement.mean()), "1")
        numbers["informative-action-roots"] = (float(informative.sum()), "1")
        # These intervals describe variability of whole-root means under a
        # working independent-root t approximation; they are not pass gates.
        if len(indices) >= 2:
            root_bias = np.asarray(screen.root_mean_errors, dtype=float)
            root_mse = np.asarray(screen.root_mean_squared_errors, dtype=float)
            critical = float(student_t.ppf(0.975, len(indices) - 1))
            for name, values, unit in (
                ("mean-error", root_bias, "hilbert-schmidt-native"),
                ("mean-square-error", root_mse, "hilbert-schmidt-native-squared"),
            ):
                width = critical * float(values.std(ddof=1)) / np.sqrt(len(values))
                numbers[f"{name}-t-lower"] = (float(values.mean() - width), unit)
                numbers[f"{name}-t-upper"] = (float(values.mean() + width), unit)
    return tuple(
        NamedDecimal(name, _decimal(number), unit)
        for name, (number, unit) in sorted(numbers.items())
    )


@dataclass(frozen=True)
class ResponseGeometryDevelopmentSupportFit:
    context: str
    representation: Representation
    parent: str
    roots: tuple[int, ...]
    labels: tuple[int | None, ...]
    predictor: StandardizedRidge | None
    training_prevalence: float | None
    reason: str | None


def _support_features(model: ResponseGeometryDevelopmentAffineModel, series: ResponseGeometryDevelopmentRootSeries) -> FloatArray:
    states = model.feature_map.transform(
        series.scalar[:, 0, 0], series.features[:, 0, 0], model.dimension
    )
    # Both numerical views remain one observation; retain the exact invocation
    # clock alongside the compressed causal state/buffer coordinates.
    return np.concatenate((states.ravel(), series.features[0, 0, 0, -3:]))


def _conformance_label(
    pair: tuple[ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentMeasuredView],
    parent: str,
    errors: FloatArray,
    deviations: FloatArray,
    quantile: float | None,
) -> int | None:
    if quantile is None or not np.isfinite(errors).all() or not np.isfinite(deviations).all():
        return None
    packets = [next(p for p in v.report.packets if p.parent == parent) for v in pair]
    conditions = [v for p in packets for v in (p.contact, p.preservation, p.parent_work_admissible)]
    if any(value is None for value in conditions):
        return None
    return int(
        all(conditions)
        and bool(np.all(np.abs(errors) <= DEVELOPMENT_DELTA))
        and bool(np.all(np.abs(errors) <= quantile * deviations))
        and float(np.max(quantile * deviations)) <= 2 * DEVELOPMENT_DELTA
    )


def fit_response_geometry_development_support(
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
    calibration: ResponseGeometryDevelopmentCalibration,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    *,
    calibration_views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    context: str,
) -> tuple[tuple[ResponseGeometryDevelopmentSupportFit, ...], tuple[ResponseGeometryDevelopmentCalibration | None, ...]]:
    """Exclude scored roots from both fitted response and its calibration scale."""
    if calibration.context != context or tuple(g.representation for g in groups) != REPRESENTATIONS:
        raise ValueError("development support context/model roster differs")
    pairs = organize_response_geometry_development_views(views, context=context, role="fit")
    organize_response_geometry_development_views(calibration_views, context=context, role="calibration")
    all_series = {g.representation: development_series(pairs, g.representation) for g in groups}
    all_labels: dict[str, dict[str, dict[int, int | None]]] = {
        r: {p: {i: None for i in pairs} for p in PARENTS} for r in REPRESENTATIONS
    }
    outer_calibrations: list[ResponseGeometryDevelopmentCalibration | None] = [None] * 4
    if calibration.quantile is not None:
        for fold in range(4):
            outer_groups = tuple(
                select_response_geometry_development_models(
                    {
                        p: {i: s for i, s in values.items() if i % 4 != fold}
                        for p, values in all_series[group.representation].items()
                    },
                    context=context,
                    representation=group.representation,
                )
                for group in groups
            )
            outer_calibration = calibrate_response_geometry_development_context(
                outer_groups, calibration_views, context=context
            )
            outer_calibrations[fold] = outer_calibration
            if outer_calibration.quantile is None:
                continue
            quantile = float(outer_calibration.quantile)
            for group in outer_groups:
                series = all_series[group.representation]
                for model in group.models:
                    for index, pair in pairs.items():
                        if index % 4 != fold or index not in series[model.parent]:
                            continue
                        try:
                            prediction, deviation, actual = endpoint_predictions(
                                model, series[model.parent][index]
                            )
                            all_labels[group.representation][model.parent][index] = (
                                _conformance_label(
                                    pair, model.parent, prediction - actual, deviation, quantile
                                )
                            )
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                            pass
    fitted = []
    for group in groups:
        series, labels = all_series[group.representation], all_labels[group.representation]
        models = {m.parent: m for m in group.models}
        for parent in PARENTS:
            known = tuple(
                i
                for i, label in labels[parent].items()
                if label is not None and i in series[parent]
            )
            predictor = None
            prevalence = None
            reason: str | None = "SUPPORT_LABELS_UNAVAILABLE"
            if parent in models and len(known) >= 12:
                x = np.stack([_support_features(models[parent], series[parent][i]) for i in known])
                y = np.asarray([labels[parent][i] for i in known], dtype=float)
                predictor = fit_standardized_ridge(
                    x, y, feature_map=FeatureMap.LINEAR, ridge_alpha=1.0
                )
                prevalence = float(y.mean())
                reason = "FAILURE_STRATUM_ABSENT" if len(set(y)) < 2 else None
            fitted.append(
                ResponseGeometryDevelopmentSupportFit(
                    context,
                    group.representation,
                    parent,
                    tuple(pairs),
                    tuple(labels[parent].values()),
                    predictor,
                    prevalence,
                    reason,
                )
            )
    return tuple(fitted), tuple(outer_calibrations)


@dataclass(frozen=True)
class ResponseGeometryDevelopmentSupportValidation:
    representation: Representation
    parent: str
    probabilities: FloatArray
    labels: FloatArray
    brier_score: float | None
    prevalence_baseline_score: float | None
    bins: FloatArray
    failure_detection_evaluable: bool


def validate_response_geometry_development_support(
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
    support: tuple[ResponseGeometryDevelopmentSupportFit, ...],
    calibration: ResponseGeometryDevelopmentCalibration,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    *,
    context: str,
) -> tuple[ResponseGeometryDevelopmentSupportValidation, ...]:
    pairs = organize_response_geometry_development_views(views, context=context, role="validation")
    quantile = None if calibration.quantile is None else float(calibration.quantile)
    models = {(g.representation, m.parent): m for g in groups for m in g.models}
    output = []
    for fitted in support:
        model = models.get((fitted.representation, fitted.parent))
        probabilities, labels = np.full(32, np.nan), np.full(32, np.nan)
        for index, pair in pairs.items():
            if model is None or fitted.predictor is None:
                continue
            series = root_series(pair, fitted.parent, fitted.representation)
            if series is None:
                continue
            try:
                # Commit the forecast from the invocation whitelist before scoring.
                probabilities[index - 32] = np.clip(
                    fitted.predictor.predict(_support_features(model, series)[None])[0], 0, 1
                )
                prediction, deviation, actual = endpoint_predictions(model, series)
                label = _conformance_label(
                    pair, fitted.parent, prediction - actual, deviation, quantile
                )
                if label is not None:
                    labels[index - 32] = label
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
        known = np.isfinite(probabilities) & np.isfinite(labels)
        brier, baseline = None, None
        if known.any() and fitted.training_prevalence is not None:
            brier = float(np.mean(np.square(probabilities[known] - labels[known])))
            baseline = float(np.mean(np.square(fitted.training_prevalence - labels[known])))
        bins = []
        for low, high in zip((0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1), strict=True):
            take = (
                known
                & (probabilities >= low)
                & ((probabilities < high) if high < 1 else (probabilities <= high))
            )
            bins.append(
                [
                    low,
                    high,
                    float(take.sum()),
                    float(probabilities[take].mean()) if take.any() else np.nan,
                    float(labels[take].mean()) if take.any() else np.nan,
                ]
            )
        output.append(
            ResponseGeometryDevelopmentSupportValidation(
                fitted.representation,
                fitted.parent,
                probabilities,
                labels,
                brier,
                baseline,
                np.asarray(bins),
                bool(np.sum(labels[known] == 0) >= 8 and np.sum(labels[known] == 1) >= 8),
            )
        )
    return tuple(output)
