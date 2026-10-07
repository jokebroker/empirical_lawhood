"""Whole-root development representation, support and passive diagnostic summaries."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
from scipy.stats import t as student_t

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
from .development_assessment import validate_response_geometry_development_context, validate_response_geometry_development_support
from .development_models import DEVELOPMENT_DELTA, REPRESENTATIONS, ResponseGeometryDevelopmentMeasuredView, organize_response_geometry_development_views
from .development_records import ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult, read_response_geometry_development_fit
from .development_terminal import ResponseGeometryDevelopmentValidationReport


def _finite(value: float) -> Decimal | None:
    return Decimal(str(float(value))) if np.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentRootStatistic(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-root-statistic'
    statistic_id: str
    unit: str
    root_values: tuple[Decimal | None, ...]
    mean: Decimal | None
    descriptive_t95: tuple[Decimal, Decimal] | None

    def __post_init__(self) -> None:
        if len(self.root_values) != 32 or any(
            v is not None and not v.is_finite() for v in self.root_values
        ):
            raise ValueError(
                "development statistic must retain all 32 finite or unavailable validation roots"
            )
        known = tuple(v for v in self.root_values if v is not None)
        if (self.mean is None) != (not known) or (self.descriptive_t95 is None) != (len(known) < 2):
            raise ValueError("development statistic loses its available independent-root denominator")
        if self.descriptive_t95 is not None and (
            not all(v.is_finite() for v in self.descriptive_t95)
            or not self.descriptive_t95[0] <= self.mean <= self.descriptive_t95[1]  # type: ignore[operator]
        ):
            raise ValueError("development descriptive interval is inconsistent")

    @classmethod
    def from_values(cls, name: str, unit: str, values: np.ndarray) -> 'ResponseGeometryDevelopmentRootStatistic':
        known = values[np.isfinite(values)]
        mean = float(known.mean()) if len(known) else float("nan")
        interval = None
        if len(known) >= 2:
            width = float(
                student_t.ppf(0.975, len(known) - 1) * known.std(ddof=1) / np.sqrt(len(known))
            )
            interval = (Decimal(str(mean - width)), Decimal(str(mean + width)))
        return cls(name, unit, tuple(_finite(v) for v in values), _finite(mean), interval)


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentRepresentationContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-representation-contrast'
    comparator: str
    typed_minus_comparator_loss: ResponseGeometryDevelopmentRootStatistic
    informative_roots: tuple[int, ...]
    matched_information: bool

    def __post_init__(self) -> None:
        if self.comparator not in REPRESENTATIONS[1:] or self.matched_information != (
            self.comparator != "typed_microstate"
        ):
            raise ValueError("development contrast changes its matched versus additional-information role")
        if self.informative_roots != tuple(sorted(set(self.informative_roots))) or any(
            i not in range(32, 64) or self.typed_minus_comparator_loss.root_values[i - 32] is None
            for i in self.informative_roots
        ):
            raise ValueError("development informative contrast loses its observed root denominator")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentSupportEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-support-evaluation'
    representation: str
    parent: str
    probabilities: tuple[Decimal | None, ...]
    labels: tuple[int | None, ...]
    brier_score: Decimal | None
    prevalence_baseline_score: Decimal | None
    bins: tuple[tuple[Decimal | None, ...], ...]
    failure_detection_evaluable: bool

    def __post_init__(self) -> None:
        if (
            self.representation not in REPRESENTATIONS
            or self.parent not in PARENTS
            or len(self.probabilities) != 32
            or len(self.labels) != 32
        ):
            raise ValueError("development support evaluation changes its chart/root roster")
        if any(
            p is not None and (not p.is_finite() or not 0 <= p <= 1) for p in self.probabilities
        ) or any(v not in (None, 0, 1) for v in self.labels):
            raise ValueError("development support requires bounded point forecasts and binary/unknown labels")
        known = tuple(
            label
            for p, label in zip(self.probabilities, self.labels, strict=True)
            if p is not None and label is not None
        )
        if (
            self.failure_detection_evaluable != (known.count(0) >= 8 and known.count(1) >= 8)
            or len(self.bins) != 5
            or any(len(b) != 5 for b in self.bins)
        ):
            raise ValueError("development failure detection changes its natural-stratum criterion")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAnalysisReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-analysis-report'
    context: str
    validation_report: ObjectIdentity
    support_result: ObjectIdentity
    contrasts: tuple[ResponseGeometryDevelopmentRepresentationContrast, ...]
    support_evaluations: tuple[ResponseGeometryDevelopmentSupportEvaluation, ...]
    passive_statistics: tuple[ResponseGeometryDevelopmentRootStatistic, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.validation_report.object_schema != ResponseGeometryDevelopmentValidationReport.SCHEMA
            or self.support_result.object_schema != ResponseGeometryDevelopmentSupportResult.SCHEMA
        ):
            raise ValueError("development analysis changes its context/input roles")
        if tuple(c.comparator for c in self.contrasts) != REPRESENTATIONS[1:] or tuple(
            (s.representation, s.parent) for s in self.support_evaluations
        ) != tuple((r, p) for r in REPRESENTATIONS for p in PARENTS):
            raise ValueError("development analysis must retain every representation/support comparison")
        expected = tuple(
            sorted(
                f"{p}.{direction}.h{h}.{method}"
                for p in PARENTS
                for direction in ("input", "slow")
                for h in (64, 320)
                for method in ("no-change", "frozen-y", "causal-y-momentum", "escape")
            )
        )
        if tuple(s.statistic_id for s in self.passive_statistics) != expected:
            raise ValueError("development passive summary loses a declared direction/horizon/forecast")

    @property
    def report_id(self) -> str:
        return f"response-geometry-development.analysis.{self.context}"


def analyze_response_geometry_development_validation(
    *,
    fit: ResponseGeometryDevelopmentFitResult,
    fit_payload: bytes,
    calibration: ResponseGeometryDevelopmentCalibrationResult,
    support: ResponseGeometryDevelopmentSupportResult,
    validation: ResponseGeometryDevelopmentValidationReport,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
) -> ResponseGeometryDevelopmentAnalysisReport:
    fit_id = ObjectIdentity.from_record(fit.result_id, fit)
    cal_id = ObjectIdentity.from_record(calibration.result_id, calibration)
    if (
        any(
            v != fit_id for v in (calibration.fit_result, support.fit_result, validation.fit_result)
        )
        or support.calibration_result != cal_id
        or validation.calibration_result != cal_id
    ):
        raise ValueError("development analysis changes frozen fit/calibration lineage")
    groups = read_response_geometry_development_fit(fit, fit_payload)
    operands = validate_response_geometry_development_context(groups, calibration.calibration, views, context=fit.context)
    if tuple(v.screen for v in operands) != validation.screens:
        raise ValueError("development analysis changes previously evaluated validation operands")
    typed = operands[0]
    contrasts = []
    for comparator in operands[1:]:
        complete = np.isfinite(typed.errors).all(axis=(1, 2, 3)) & np.isfinite(
            comparator.errors
        ).all(axis=(1, 2, 3))
        losses = np.full(32, np.nan)
        losses[complete] = np.mean(
            (typed.errors[complete] ** 2 - comparator.errors[complete] ** 2) / DEVELOPMENT_DELTA**2,
            axis=(1, 2, 3),
        )
        informative = complete & (
            np.max(np.abs(typed.predictions - comparator.predictions), axis=(1, 2, 3))
            >= DEVELOPMENT_DELTA / 64
        )
        contrasts.append(
            ResponseGeometryDevelopmentRepresentationContrast(
                comparator.screen.representation,
                ResponseGeometryDevelopmentRootStatistic.from_values(
                    f"typed-minus-{comparator.screen.representation}",
                    "delta-squared-normalized-loss",
                    losses,
                ),
                tuple(int(i + 32) for i in np.flatnonzero(informative)),
                comparator.screen.representation != "typed_microstate",
            )
        )
    support_rows = validate_response_geometry_development_support(
        groups,
        tuple(s.as_fit(fit.context) for s in support.models),
        calibration.calibration,
        views,
        context=fit.context,
    )
    support_evaluations = tuple(
        ResponseGeometryDevelopmentSupportEvaluation(
            s.representation,
            s.parent,
            tuple(_finite(v) for v in s.probabilities),
            tuple(int(v) if np.isfinite(v) else None for v in s.labels),
            None if s.brier_score is None else _finite(s.brier_score),
            None if s.prevalence_baseline_score is None else _finite(s.prevalence_baseline_score),
            tuple(tuple(_finite(v) for v in row) for row in s.bins),
            s.failure_detection_evaluable,
        )
        for s in support_rows
    )
    pairs = organize_response_geometry_development_views(views, context=fit.context, role="validation")
    statistics = []
    for parent in PARENTS:
        for direction_index, direction in enumerate(("input", "slow")):
            for horizon_index, horizon in enumerate((64, 320)):
                values = np.full((4, 32), np.nan)
                for index, pair in pairs.items():
                    observed = []
                    for view in pair:
                        arrays = view.parents.get(parent)
                        if arrays is None:
                            break
                        directions = arrays["passive_directions"][
                            :, direction_index * 3 : (direction_index + 1) * 3
                        ]
                        native = arrays["passive_native"][:, horizon_index]
                        forecasts = arrays["passive_forecasts"][:, horizon_index]
                        if (
                            not np.isfinite(directions).all()
                            or not np.isfinite(native).all()
                            or np.linalg.norm(directions) <= 1e-12
                        ):
                            break
                        denominator = float(np.linalg.norm(directions))
                        errors = [
                            float(
                                np.max(np.linalg.norm((native - f) @ directions, axis=(1, 2)))
                                / denominator
                            )
                            if np.isfinite(f).all()
                            else float("nan")
                            for f in forecasts
                        ]
                        basis, singular, _ = np.linalg.svd(directions, full_matrices=False)
                        basis = basis[:, singular > singular[0] * 1e-10]
                        outside = np.eye(15) - basis @ basis.T
                        errors.append(
                            float(
                                np.max(np.linalg.norm(outside @ native @ directions, axis=(1, 2)))
                                / denominator
                            )
                        )
                        observed.append(errors)
                    if len(observed) == 2:
                        values[:, index - 32] = np.max(observed, axis=0)
                for name, row in zip(
                    ("no-change", "frozen-y", "causal-y-momentum", "escape"), values, strict=True
                ):
                    statistics.append(
                        ResponseGeometryDevelopmentRootStatistic.from_values(
                            f"{parent}.{direction}.h{horizon}.{name}",
                            "native-map-norm-per-input-norm",
                            row,
                        )
                    )
    return ResponseGeometryDevelopmentAnalysisReport(
        fit.context,
        ObjectIdentity.from_record(validation.report_id, validation),
        ObjectIdentity.from_record(support.result_id, support),
        tuple(contrasts),
        support_evaluations,
        tuple(sorted(statistics, key=lambda s: s.statistic_id)),
        (
            "DESCRIPTIVE_ROOT_T_INTERVALS_NOT_SIMULTANEOUS_SUPERIORITY",
            "PASSIVE_SAMPLED_DIRECTIONS_NOT_X_CONTROLLED_LAW",
            "POINT_SUPPORT_NOT_ACTION_ADMISSION",
        ),
    )
