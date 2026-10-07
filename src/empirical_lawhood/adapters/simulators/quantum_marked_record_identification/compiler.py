"""Transparent ridge compiler and deterministic compiler-development selection for quantum marked record identification."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..quantum_scientific_order import ScientificOrder, scientific_order_ranks
from .contracts import Panel, stable_json_bytes
from .features import FeatureContext, FeatureRow, IntensityModel, feature_matrix, feature_names


CONDITION_CEILING = 1.0e8
SCALE_FLOOR = 1.0e-8
LATENCY_DEADLINE_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class Preprocessor:
    centers: tuple[float, ...]
    scales: tuple[float, ...]

    def transform(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(matrix, dtype=np.float64)
        centers = np.asarray(self.centers, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(centers):
            raise ValueError("preprocessor matrix shape differs")
        transformed = (values - centers) / scales
        if not np.all(np.isfinite(transformed)):
            raise ValueError("preprocessor produced a nonfinite coordinate")
        return transformed


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
    condition_number: float
    coefficient_norm: float
    p99_latency_seconds: float

    def score_rows(self, rows: Sequence[FeatureRow]) -> NDArray[np.float64]:
        matrix = feature_matrix(rows, self.panel)
        transformed = self.preprocessor.transform(matrix)
        score = self.intercept + transformed @ np.asarray(self.coefficients, dtype=np.float64)
        if not np.all(np.isfinite(score)):
            raise ValueError("compiler emitted a nonfinite score")
        return np.asarray(score)

    def document(self) -> dict[str, object]:
        return {
            "schema": 'empirical-lawhood/simulators/quantum-marked-record-identification/score-compiler',
            "version": '1.0.0',
            "value": {
                "candidate_id": self.candidate_id,
                "panel": self.panel.value,
                "regularization": repr(self.regularization),
                "numeric_dimension": self.numeric_dimension,
                "feature_order": list(self.feature_order),
                "semantic_flags": list(self.semantic_flags),
                "preprocessor": {
                    "centers": [repr(value) for value in self.preprocessor.centers],
                    "scales": [repr(value) for value in self.preprocessor.scales],
                },
                "intercept": repr(self.intercept),
                "coefficients": [repr(value) for value in self.coefficients],
                "context": {
                    "site_center": [repr(value) for value in self.context.site_center],
                    "intensity": {
                        "p_a_by_cell": {
                            key: repr(value)
                            for key, value in sorted(self.context.intensity.p_a_by_cell.items())
                        },
                        "p_boundary_by_cell": {
                            key: repr(value)
                            for key, value in sorted(
                                self.context.intensity.p_boundary_by_cell.items()
                            )
                        },
                        "global_p_a": repr(self.context.intensity.global_p_a),
                        "global_p_boundary": repr(self.context.intensity.global_p_boundary),
                    },
                },
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
    viable: bool
    rejection_reasons: tuple[str, ...]
    control_associations: Mapping[str, float]


def fit_preprocessor(matrix: NDArray[np.float64]) -> Preprocessor:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("preprocessing fit requires a finite matrix")
    centers = np.median(values, axis=0)
    mad = np.median(np.abs(values - centers), axis=0)
    scales = np.maximum(mad, SCALE_FLOOR)
    return Preprocessor(
        centers=tuple(float(value) for value in centers),
        scales=tuple(float(value) for value in scales),
    )


def fit_compiler(
    *,
    panel: Panel,
    regularization: float,
    training_rows: Sequence[FeatureRow],
    target: Sequence[float],
    weights: Sequence[float],
    context: FeatureContext,
) -> ScoreCompiler:
    if regularization not in {0.0, 0.01, 0.1, 1.0, 10.0}:
        raise ValueError("regularization leaves the frozen roster")
    matrix = feature_matrix(training_rows, panel)
    target_array = np.asarray(target, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if target_array.shape != (len(matrix),) or weight_array.shape != target_array.shape:
        raise ValueError("compiler fit rows differ")
    weight_array = weight_array / float(weight_array.sum())
    preprocessor = fit_preprocessor(matrix)
    transformed = preprocessor.transform(matrix)
    weighted_mean_x = weight_array @ transformed
    weighted_mean_y = float(weight_array @ target_array)
    centered_x = transformed - weighted_mean_x
    centered_y = target_array - weighted_mean_y
    gram = (centered_x * weight_array[:, None]).T @ centered_x
    rhs = (centered_x * weight_array[:, None]).T @ centered_y
    active = np.ptp(transformed, axis=0) > 1.0e-12
    singular = np.linalg.svd(centered_x[:, active], compute_uv=False)
    raw_condition = (
        float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else math.inf
    )
    if regularization == 0.0 and (
        not np.all(active)
        or np.linalg.matrix_rank(centered_x) < centered_x.shape[1]
        or raw_condition > CONDITION_CEILING
    ):
        raise ValueError("unregularized candidate violates the rank/condition gate")
    regularized_gram = gram + regularization * np.eye(gram.shape[0], dtype=np.float64)
    condition = float(np.linalg.cond(regularized_gram))
    beta = np.linalg.solve(regularized_gram, rhs)
    intercept = weighted_mean_y - float(weighted_mean_x @ beta)
    candidate_id = f"{panel.value}.ridge-{regularization:g}"
    provisional = ScoreCompiler(
        candidate_id=candidate_id,
        panel=panel,
        regularization=regularization,
        numeric_dimension=panel.dimension,
        feature_order=feature_names(panel),
        semantic_flags=panel.semantic_flags,
        preprocessor=preprocessor,
        intercept=float(intercept),
        coefficients=tuple(float(value) for value in beta),
        context=context,
        condition_number=condition,
        coefficient_norm=float(np.linalg.norm(beta)),
        p99_latency_seconds=0.0,
    )
    timings = []
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
        condition_number=provisional.condition_number,
        coefficient_norm=provisional.coefficient_norm,
        p99_latency_seconds=float(np.quantile(timings, 0.99)),
    )


def compiler_from_document(document: Mapping[str, object]) -> ScoreCompiler:
    if (
        set(document) != {"schema", "version", "value"}
        or document.get("schema") != 'empirical-lawhood/simulators/quantum-marked-record-identification/score-compiler'
        or document.get("version") != "1.0.0"
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
    intensity = context.get("intensity")
    if not isinstance(intensity, Mapping):
        raise ValueError("compiler intensity object differs")
    compiler = ScoreCompiler(
        candidate_id=str(value["candidate_id"]),
        panel=Panel(str(value["panel"])),
        regularization=float(str(value["regularization"])),
        numeric_dimension=int(value["numeric_dimension"]),
        feature_order=tuple(str(item) for item in value["feature_order"]),
        semantic_flags=tuple(str(item) for item in value["semantic_flags"]),
        preprocessor=Preprocessor(
            centers=tuple(float(item) for item in preprocessing["centers"]),
            scales=tuple(float(item) for item in preprocessing["scales"]),
        ),
        intercept=float(str(value["intercept"])),
        coefficients=tuple(float(item) for item in value["coefficients"]),
        context=FeatureContext(
            site_center=tuple(float(item) for item in context["site_center"]),
            intensity=IntensityModel(
                p_a_by_cell={
                    str(key): float(item) for key, item in intensity["p_a_by_cell"].items()
                },
                p_boundary_by_cell={
                    str(key): float(item) for key, item in intensity["p_boundary_by_cell"].items()
                },
                global_p_a=float(str(intensity["global_p_a"])),
                global_p_boundary=float(str(intensity["global_p_boundary"])),
            ),
        ),
        condition_number=float(str(value["condition_number"])),
        coefficient_norm=float(str(value["coefficient_norm"])),
        p99_latency_seconds=float(str(value["p99_latency_seconds"])),
    )
    if (
        compiler.numeric_dimension != compiler.panel.dimension
        or compiler.feature_order != feature_names(compiler.panel)
        or compiler.semantic_flags != compiler.panel.semantic_flags
        or len(compiler.coefficients) != compiler.panel.dimension + 2
    ):
        raise ValueError("compiler payload violates its panel contract")
    return compiler


def select_candidate(
    candidates: Sequence[CandidateResult],
    *,
    scientific_candidate_order: ScientificOrder,
) -> CandidateResult | None:
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
            candidate.compiler.p99_latency_seconds,
            ranks[candidate.compiler.candidate_id],
        ),
    )


__all__ = [
    "CONDITION_CEILING",
    "CandidateResult",
    "LATENCY_DEADLINE_SECONDS",
    "Preprocessor",
    "SCALE_FLOOR",
    "ScoreCompiler",
    "compiler_from_document",
    "fit_compiler",
    "fit_preprocessor",
    "select_candidate",
]
