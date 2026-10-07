"""Typed preprocessing, stable ridge fitting, and deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd

from ..quantum_scientific_order import ScientificOrder, scientific_order_ranks
from .contracts import Panel, stable_json_bytes
from .features import CATEGORICAL_FEATURES, FeatureContext, FeatureRow, feature_matrix


CONDITION_CEILING = 1.0e8
COEFFICIENT_NORM_CEILING = 25.0
SCORE_VARIANCE_FLOOR = 1.0e-6
LATENCY_DEADLINE_SECONDS = 0.01
NUMERIC_CAP = 50.0
REGULARIZATIONS = (0.0, 0.01, 0.1, 1.0, 10.0)


class CompilerFitError(ValueError):
    """Typed terminal fit failure used to keep candidate ledgers complete."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        telemetry: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.telemetry = dict(telemetry or {})


@dataclass(frozen=True, slots=True)
class FeatureTransform:
    feature_id: str
    coordinate_type: str
    center: float
    mad: float
    iqr: float
    scale: float
    native_floor: float
    lower_cap: float | None
    upper_cap: float | None


@dataclass(frozen=True, slots=True)
class Preprocessor:
    feature_order: tuple[str, ...]
    transforms: tuple[FeatureTransform, ...]

    def __post_init__(self) -> None:
        if (
            len(self.feature_order) != len(self.transforms)
            or tuple(transform.feature_id for transform in self.transforms) != self.feature_order
        ):
            raise ValueError("preprocessor manifest order differs")

    def transform(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.transforms):
            raise ValueError("preprocessor matrix shape differs")
        result = np.empty_like(values)
        for index, transform in enumerate(self.transforms):
            column = values[:, index]
            if transform.coordinate_type == "indicator":
                if np.any((column != 0.0) & (column != 1.0)):
                    raise ValueError("categorical indicator leaves native 0/1 support")
                result[:, index] = column
            else:
                if transform.lower_cap is None or transform.upper_cap is None:
                    raise ValueError("numeric transform lacks clipping bounds")
                scaled = (column - transform.center) / transform.scale
                result[:, index] = np.clip(
                    scaled,
                    float(transform.lower_cap),
                    float(transform.upper_cap),
                )
        if not np.all(np.isfinite(result)):
            raise ValueError("preprocessor produced a nonfinite coordinate")
        return result

    def clipped_weight(
        self,
        matrix: NDArray[np.float64],
        weights: Sequence[float],
    ) -> float:
        values = np.asarray(matrix, dtype=np.float64)
        weight_array = _normalized_weights(weights, len(values))
        clipped = self.clipped_mask(values)
        return float(weight_array[clipped].sum())

    def clipped_mask(self, matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.transforms):
            raise ValueError("preprocessor matrix shape differs")
        clipped = np.zeros(len(values), dtype=np.bool_)
        for index, transform in enumerate(self.transforms):
            if transform.coordinate_type == "indicator":
                continue
            if transform.lower_cap is None or transform.upper_cap is None:
                raise ValueError("numeric transform lacks clipping bounds")
            scaled = (values[:, index] - transform.center) / transform.scale
            clipped |= (scaled < float(transform.lower_cap)) | (scaled > float(transform.upper_cap))
        return clipped

    def document(self) -> dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "transforms": [
                {
                    "feature_id": row.feature_id,
                    "coordinate_type": row.coordinate_type,
                    "center": repr(row.center),
                    "mad": repr(row.mad),
                    "iqr": repr(row.iqr),
                    "scale": repr(row.scale),
                    "native_floor": repr(row.native_floor),
                    "lower_cap": None if row.lower_cap is None else repr(row.lower_cap),
                    "upper_cap": None if row.upper_cap is None else repr(row.upper_cap),
                }
                for row in self.transforms
            ],
        }


@dataclass(frozen=True, slots=True)
class ScoreCompiler:
    candidate_id: str
    panel: Panel
    regularization: float
    numeric_dimension: int
    feature_order: tuple[str, ...]
    semantic_flags: tuple[str, ...]
    preprocessor: Preprocessor
    intercept: float
    coefficients: tuple[float, ...]
    context: FeatureContext
    rank: int
    rank_tolerance: float
    singular_values: tuple[float, ...]
    condition_number: float
    coefficient_norm: float
    p99_latency_seconds: float

    def score_rows(self, rows: Sequence[FeatureRow]) -> NDArray[np.float64]:
        matrix = feature_matrix(rows, self.feature_order)
        transformed = self.preprocessor.transform(matrix)
        score = self.intercept + transformed @ np.asarray(self.coefficients, dtype=np.float64)
        if not np.all(np.isfinite(score)):
            raise ValueError("compiler emitted a nonfinite score")
        return np.asarray(score, dtype=np.float64)

    def document(self) -> dict[str, object]:
        return {
            "schema": 'empirical-lawhood/simulators/quantum-finite-grammar-identification/score-compiler',
            "version": '1.0.0',
            "value": {
                "candidate_id": self.candidate_id,
                "panel": self.panel.value,
                "regularization": repr(self.regularization),
                "numeric_dimension": self.numeric_dimension,
                "feature_order": list(self.feature_order),
                "semantic_flags": list(self.semantic_flags),
                "preprocessor": self.preprocessor.document(),
                "intercept": repr(self.intercept),
                "coefficients": [repr(value) for value in self.coefficients],
                "context": {
                    "site_center": [repr(value) for value in self.context.site_center],
                },
                "rank": self.rank,
                "rank_tolerance": repr(self.rank_tolerance),
                "singular_values": [repr(value) for value in self.singular_values],
                "condition_number": repr(self.condition_number),
                "coefficient_norm": repr(self.coefficient_norm),
                "p99_latency_seconds": repr(self.p99_latency_seconds),
            },
        }

    def canonical_bytes(self) -> bytes:
        return stable_json_bytes(self.document())


@dataclass(frozen=True, slots=True)
class CandidateResult:
    compiler: ScoreCompiler
    rho_matched: float
    rho_burst_advantage: float
    active_control_minimum: float
    valid_weight: float
    score_variance: float
    clipped_weight: float
    viable: bool
    rejection_reasons: tuple[str, ...]
    control_associations: Mapping[str, float]


def _normalized_weights(weights: Sequence[float], size: int) -> NDArray[np.float64]:
    values = np.asarray(weights, dtype=np.float64)
    if (
        values.shape != (size,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or float(values.sum()) <= 0
    ):
        raise ValueError("weights are invalid")
    return values / float(values.sum())


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile leaves [0,1]")
    order = np.lexsort((np.arange(len(values)), values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(sorted_values[index])


def _coordinate_type(feature_id: str) -> str:
    if feature_id in CATEGORICAL_FEATURES or feature_id == "empty_recent":
        return "indicator"
    if feature_id.startswith("age_"):
        return "age"
    if feature_id.startswith("q_"):
        return "quadratic"
    if feature_id.startswith("pair_") or feature_id.startswith("boundary_"):
        return "pair"
    return "count"


def _native_floor(coordinate_type: str) -> float:
    return {
        "age": 1.0 / 6.0,
        "quadratic": 1.0,
        "pair": 1.0,
        "count": 1.0,
        "indicator": 1.0,
    }[coordinate_type]


def fit_preprocessor(
    matrix: NDArray[np.float64],
    feature_order: Sequence[str],
    weights: Sequence[float],
) -> Preprocessor:
    values = np.asarray(matrix, dtype=np.float64)
    order = tuple(feature_order)
    if values.ndim != 2 or values.shape[1] != len(order) or not np.all(np.isfinite(values)):
        raise ValueError("preprocessing fit requires a finite manifested matrix")
    weight_array = _normalized_weights(weights, len(values))
    transforms: list[FeatureTransform] = []
    for index, feature_id in enumerate(order):
        coordinate_type = _coordinate_type(feature_id)
        column = values[:, index]
        if coordinate_type == "indicator":
            if np.any((column != 0.0) & (column != 1.0)):
                raise ValueError("indicator training column leaves native 0/1 support")
            transforms.append(
                FeatureTransform(
                    feature_id=feature_id,
                    coordinate_type=coordinate_type,
                    center=0.0,
                    mad=0.0,
                    iqr=0.0,
                    scale=1.0,
                    native_floor=1.0,
                    lower_cap=None,
                    upper_cap=None,
                )
            )
            continue
        center = _weighted_quantile(column, weight_array, 0.5)
        mad = _weighted_quantile(np.abs(column - center), weight_array, 0.5)
        q25 = _weighted_quantile(column, weight_array, 0.25)
        q75 = _weighted_quantile(column, weight_array, 0.75)
        iqr = q75 - q25
        floor = _native_floor(coordinate_type)
        scale = max(mad, iqr / 1.349, floor)
        transforms.append(
            FeatureTransform(
                feature_id=feature_id,
                coordinate_type=coordinate_type,
                center=center,
                mad=mad,
                iqr=iqr,
                scale=scale,
                native_floor=floor,
                lower_cap=-NUMERIC_CAP,
                upper_cap=NUMERIC_CAP,
            )
        )
    return Preprocessor(feature_order=order, transforms=tuple(transforms))


def fit_compiler(
    *,
    panel: Panel,
    regularization: float,
    feature_order: Sequence[str],
    training_rows: Sequence[FeatureRow],
    target: Sequence[float],
    weights: Sequence[float],
    context: FeatureContext,
    preprocessor: Preprocessor | None = None,
) -> ScoreCompiler:
    if regularization not in REGULARIZATIONS:
        raise CompilerFitError("INVALID_INPUT", "regularization leaves the frozen roster")
    order = tuple(feature_order)
    matrix = feature_matrix(training_rows, order)
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != (len(matrix),) or not np.all(np.isfinite(target_array)):
        raise CompilerFitError("INVALID_INPUT", "compiler target rows differ")
    weight_array = _normalized_weights(weights, len(matrix))
    if preprocessor is None:
        preprocessor = fit_preprocessor(matrix, order, weight_array.tolist())
    elif preprocessor.feature_order != order:
        raise CompilerFitError(
            "INVALID_INPUT",
            "committed preprocessor feature order differs",
        )
    transformed = preprocessor.transform(matrix)
    weighted_mean_x = weight_array @ transformed
    weighted_mean_y = float(weight_array @ target_array)
    centered_x = transformed - weighted_mean_x
    centered_y = target_array - weighted_mean_y
    sqrt_weight = np.sqrt(weight_array)
    weighted_x = centered_x * sqrt_weight[:, None]
    weighted_y = centered_y * sqrt_weight
    try:
        u_matrix, singular, vt_matrix = svd(
            weighted_x,
            full_matrices=False,
            lapack_driver="gesvd",
            check_finite=True,
        )
    except Exception as error:  # pragma: no cover - platform LAPACK failure
        raise CompilerFitError("SOLVE_FAILED", str(error)) from error
    # Canonicalize each singular-vector pair at the largest-magnitude right
    # coordinate. This leaves the fitted beta invariant while making the
    # intermediate solver convention deterministic across conformant LAPACK
    # builds.
    for component in range(len(singular)):
        pivot = int(np.argmax(np.abs(vt_matrix[component])))
        if vt_matrix[component, pivot] < 0.0:
            vt_matrix[component] *= -1.0
            u_matrix[:, component] *= -1.0
    sigma_max = float(singular[0]) if len(singular) else 0.0
    rank_tolerance = np.finfo(np.float64).eps * max(weighted_x.shape) * sigma_max
    rank = int(np.sum(singular > rank_tolerance))
    if regularization == 0.0 and rank < weighted_x.shape[1]:
        raise CompilerFitError(
            "RANK_REJECTED",
            "unregularized candidate is rank deficient",
            telemetry={
                "rank": rank,
                "dimension": weighted_x.shape[1],
                "rank_tolerance": rank_tolerance,
                "singular_values": [float(value) for value in singular],
            },
        )
    sigma_min = float(singular[-1]) if len(singular) and rank == weighted_x.shape[1] else 0.0
    denominator = sigma_min**2 + regularization
    condition = (sigma_max**2 + regularization) / denominator if denominator > 0.0 else math.inf
    try:
        factors = singular / (singular**2 + regularization)
        beta = vt_matrix.T @ (factors * (u_matrix.T @ weighted_y))
    except Exception as error:  # pragma: no cover - guarded numerical failure
        raise CompilerFitError(
            "SOLVE_FAILED",
            str(error),
            telemetry={
                "rank": rank,
                "condition_number": condition,
            },
        ) from error
    if not np.all(np.isfinite(beta)):
        raise CompilerFitError(
            "SOLVE_FAILED",
            "ridge coefficients are nonfinite",
            telemetry={"rank": rank, "condition_number": condition},
        )
    intercept = weighted_mean_y - float(weighted_mean_x @ beta)
    provisional = ScoreCompiler(
        candidate_id=f"{panel.value}.ridge-{regularization:g}",
        panel=panel,
        regularization=regularization,
        numeric_dimension=sum(name not in CATEGORICAL_FEATURES for name in order),
        feature_order=order,
        semantic_flags=panel.semantic_flags,
        preprocessor=preprocessor,
        intercept=intercept,
        coefficients=tuple(float(value) for value in beta),
        context=context,
        rank=rank,
        rank_tolerance=float(rank_tolerance),
        singular_values=tuple(float(value) for value in singular),
        condition_number=float(condition),
        coefficient_norm=float(np.linalg.norm(beta)),
        p99_latency_seconds=0.0,
    )
    timings: list[float] = []
    sample = tuple(training_rows[: min(64, len(training_rows))])
    for _ in range(8):
        started = time.perf_counter()
        provisional.score_rows(sample)
        timings.append((time.perf_counter() - started) / max(1, len(sample)))
    return ScoreCompiler(
        candidate_id=provisional.candidate_id,
        panel=provisional.panel,
        regularization=provisional.regularization,
        numeric_dimension=provisional.numeric_dimension,
        feature_order=provisional.feature_order,
        semantic_flags=provisional.semantic_flags,
        preprocessor=provisional.preprocessor,
        intercept=provisional.intercept,
        coefficients=provisional.coefficients,
        context=provisional.context,
        rank=provisional.rank,
        rank_tolerance=provisional.rank_tolerance,
        singular_values=provisional.singular_values,
        condition_number=provisional.condition_number,
        coefficient_norm=provisional.coefficient_norm,
        p99_latency_seconds=float(np.quantile(timings, 0.99)),
    )


def compiler_from_document(document: Mapping[str, object]) -> ScoreCompiler:
    if (
        set(document) != {"schema", "version", "value"}
        or document.get("schema") != 'empirical-lawhood/simulators/quantum-finite-grammar-identification/score-compiler'
        or document.get("version") != "3.1.0"
        or not isinstance(document.get("value"), Mapping)
    ):
        raise ValueError("compiler document identity differs")
    value = document["value"]
    assert isinstance(value, Mapping)
    expected = {
        "candidate_id",
        "panel",
        "regularization",
        "numeric_dimension",
        "feature_order",
        "semantic_flags",
        "preprocessor",
        "intercept",
        "coefficients",
        "context",
        "rank",
        "rank_tolerance",
        "singular_values",
        "condition_number",
        "coefficient_norm",
        "p99_latency_seconds",
    }
    if set(value) != expected:
        raise ValueError("compiler document keys differ")
    preprocessing = value["preprocessor"]
    context = value["context"]
    if not isinstance(preprocessing, Mapping) or not isinstance(context, Mapping):
        raise ValueError("compiler nested objects differ")
    raw_transforms = preprocessing.get("transforms")
    raw_order = preprocessing.get("feature_order")
    if not isinstance(raw_transforms, list) or not isinstance(raw_order, list):
        raise ValueError("preprocessor document differs")
    transforms: list[FeatureTransform] = []
    for raw in raw_transforms:
        if not isinstance(raw, Mapping):
            raise ValueError("preprocessor transform differs")
        transforms.append(
            FeatureTransform(
                feature_id=str(raw["feature_id"]),
                coordinate_type=str(raw["coordinate_type"]),
                center=float(str(raw["center"])),
                mad=float(str(raw["mad"])),
                iqr=float(str(raw["iqr"])),
                scale=float(str(raw["scale"])),
                native_floor=float(str(raw["native_floor"])),
                lower_cap=(None if raw["lower_cap"] is None else float(str(raw["lower_cap"]))),
                upper_cap=(None if raw["upper_cap"] is None else float(str(raw["upper_cap"]))),
            )
        )
    feature_order = tuple(str(item) for item in value["feature_order"])
    compiler = ScoreCompiler(
        candidate_id=str(value["candidate_id"]),
        panel=Panel(str(value["panel"])),
        regularization=float(str(value["regularization"])),
        numeric_dimension=int(value["numeric_dimension"]),
        feature_order=feature_order,
        semantic_flags=tuple(str(item) for item in value["semantic_flags"]),
        preprocessor=Preprocessor(
            feature_order=tuple(str(item) for item in raw_order),
            transforms=tuple(transforms),
        ),
        intercept=float(str(value["intercept"])),
        coefficients=tuple(float(item) for item in value["coefficients"]),
        context=FeatureContext(
            site_center=tuple(float(item) for item in context["site_center"]),
        ),
        rank=int(value["rank"]),
        rank_tolerance=float(str(value["rank_tolerance"])),
        singular_values=tuple(float(item) for item in value["singular_values"]),
        condition_number=float(str(value["condition_number"])),
        coefficient_norm=float(str(value["coefficient_norm"])),
        p99_latency_seconds=float(str(value["p99_latency_seconds"])),
    )
    if (
        compiler.preprocessor.feature_order != compiler.feature_order
        or compiler.semantic_flags != compiler.panel.semantic_flags
        or len(compiler.coefficients) != len(compiler.feature_order)
        or compiler.numeric_dimension
        != sum(name not in CATEGORICAL_FEATURES for name in compiler.feature_order)
    ):
        raise ValueError("compiler payload violates its feature manifest")
    return compiler


def select_candidate(candidates: Sequence[CandidateResult], *, scientific_candidate_order: ScientificOrder) -> CandidateResult | None:
    ranks = scientific_order_ranks(scientific_candidate_order, identifiers=tuple(candidate.compiler.candidate_id for candidate in candidates))
    viable = [candidate for candidate in candidates if candidate.viable]
    if not viable:
        return None
    best_rho = max(candidate.rho_matched for candidate in viable)
    retained = [candidate for candidate in viable if candidate.rho_matched >= best_rho - 0.02]
    best_control = max(candidate.active_control_minimum for candidate in retained)
    retained = [
        candidate
        for candidate in retained
        if candidate.active_control_minimum >= best_control - 0.02
    ]
    return min(
        retained,
        key=lambda candidate: (
            candidate.compiler.numeric_dimension,
            len(candidate.compiler.semantic_flags),
            -candidate.compiler.regularization,
            candidate.compiler.coefficient_norm,
            ranks[candidate.compiler.candidate_id],
        ),
    )


__all__ = [
    "COEFFICIENT_NORM_CEILING",
    "CONDITION_CEILING",
    "CandidateResult",
    "CompilerFitError",
    "FeatureTransform",
    "LATENCY_DEADLINE_SECONDS",
    "NUMERIC_CAP",
    "Preprocessor",
    "REGULARIZATIONS",
    "SCORE_VARIANCE_FLOOR",
    "ScoreCompiler",
    "compiler_from_document",
    "fit_compiler",
    "fit_preprocessor",
    "select_candidate",
]
