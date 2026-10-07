"""Pure deterministic analysis for the Virtual Cell Challenge programme.

This module implements the Tier-L0 target-level model ladder.  It has no file,
network, registry or runtime dependencies: scientific arrays and frozen
contracts enter explicitly, and safe array artifacts are produced by runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .contracts import (
    LeaderboardSnapshot,
    MetricContract,
    PlacementAdjudication,
    PlacementClass,
)
from .dataset import ResponseSummaryArrays


class VirtualCellAnalysisError(ValueError):
    """Raised when a scientific array or frozen decision fails closed."""


class ModelFamily(StrEnum):
    NO_CHANGE_BASELINE = "no-change-baseline"
    WEIGHTED_COMMON_RESPONSE_BASELINE = "weighted-common-response-baseline"
    TARGET_FEATURE_RIDGE_RESPONSE = "target-feature-ridge-response"
    REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE = "reduced-rank-target-feature-ridge-response"
    ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE = "admission-conditioned-reduced-rank-ridge-response"


@dataclass(frozen=True, slots=True)
class TargetResponseMatrix:
    """One row per action/target; nested cells have already been aggregated."""

    target_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: NDArray[np.float64]
    target_weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        target_count = len(self.target_ids)
        gene_count = len(self.gene_ids)
        if target_count == 0 or gene_count == 0:
            raise ValueError("target response matrix cannot be empty")
        if len(set(self.target_ids)) != target_count:
            raise ValueError("target IDs must be unique")
        if len(set(self.gene_ids)) != gene_count:
            raise ValueError("gene IDs must be unique")
        if self.values.shape != (target_count, gene_count):
            raise ValueError("response array shape differs from target/gene registries")
        if self.target_weights.shape != (target_count,):
            raise ValueError("target weights have the wrong shape")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("response array contains nonfinite values")
        if not np.all(np.isfinite(self.target_weights)) or np.any(self.target_weights <= 0):
            raise ValueError("target weights must be finite and positive")


@dataclass(frozen=True, slots=True)
class TargetFeatureMatrix:
    """Outcome-blind features aligned to an explicit target roster."""

    target_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: NDArray[np.float64]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("feature target IDs must be nonempty and unique")
        if not self.feature_ids or len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("feature IDs must be nonempty and unique")
        if self.values.shape != (len(self.target_ids), len(self.feature_ids)):
            raise ValueError("feature array shape differs from its registries")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("feature array contains nonfinite values")
        validate_sha256(self.provenance_sha256, field_name="provenance_sha256")


@dataclass(frozen=True, slots=True)
class Standardization:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("standardization arrays must be aligned vectors")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("standardization contains nonfinite values")
        if np.any(self.scale <= 0):
            raise ValueError("standardization scales must be positive")


@dataclass(frozen=True, slots=True)
class PrefixAdmissionModel:
    """Outcome-blind-at-use predictor trained from development-only labels."""

    model_id: str
    feature_ids: tuple[str, ...]
    standardization: Standardization
    coefficients: NDArray[np.float64]
    intercept: float
    alpha: float

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        if not self.feature_ids or len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("prefix-admission feature IDs must be nonempty and unique")
        if self.coefficients.shape != (len(self.feature_ids),):
            raise ValueError("prefix-admission coefficients have the wrong shape")
        if self.standardization.mean.shape != self.coefficients.shape:
            raise ValueError("prefix-admission standardization has the wrong shape")
        if not np.all(np.isfinite(self.coefficients)) or not math.isfinite(self.intercept):
            raise ValueError("prefix-admission model contains nonfinite values")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("prefix-admission alpha must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class LinearResponseModel:
    """Safe in-memory model whose arrays can be serialized without pickle."""

    model_id: str
    family: ModelFamily
    feature_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    standardization: Standardization | None
    coefficients: NDArray[np.float64]
    intercept: NDArray[np.float64]
    response_basis: NDArray[np.float64] | None
    admission_required: bool
    admission_feature_count: int
    prefix_admission_model: PrefixAdmissionModel | None
    residual_scale: NDArray[np.float64]

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        require_sorted_unique_strings(tuple(sorted(self.feature_ids)), field_name="feature_ids")
        if not self.gene_ids or len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("model gene IDs must be nonempty and unique")
        gene_count = len(self.gene_ids)
        if self.intercept.shape != (gene_count,) or self.residual_scale.shape != (gene_count,):
            raise ValueError("model output vectors have the wrong shape")
        if self.response_basis is None:
            if self.coefficients.ndim != 2 or self.coefficients.shape[1] != gene_count:
                raise ValueError("full-space coefficients have the wrong shape")
        else:
            if self.response_basis.ndim != 2 or self.response_basis.shape[1] != gene_count:
                raise ValueError("response basis has the wrong gene dimension")
            if self.coefficients.ndim != 2 or (
                self.coefficients.shape[1] != self.response_basis.shape[0]
            ):
                raise ValueError("reduced coefficients and response basis are inconsistent")
        if self.coefficients.shape[0] != len(self.feature_ids):
            raise ValueError("model coefficient rows differ from feature IDs")
        if self.standardization is not None and (
            self.standardization.mean.shape != (len(self.feature_ids),)
        ):
            raise ValueError("model standardization differs from its feature roster")
        if self.admission_feature_count < 0:
            raise ValueError("admission_feature_count cannot be negative")
        if self.admission_required != (self.admission_feature_count > 0):
            raise ValueError("admission requirement and feature count disagree")
        if self.admission_required != (self.prefix_admission_model is not None):
            raise ValueError("Admission-conditioned reduced-rank ridge alone must carry its fitted prefix-admission predictor")
        for array in (self.coefficients, self.intercept, self.residual_scale):
            if not np.all(np.isfinite(array)):
                raise ValueError("model contains nonfinite values")
        if self.response_basis is not None and not np.all(np.isfinite(self.response_basis)):
            raise ValueError("model response basis contains nonfinite values")
        if np.any(self.residual_scale < 0):
            raise ValueError("residual scale cannot be negative")


@dataclass(frozen=True, slots=True)
class ModelFitRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/model-fit-record'

    fit_id: str
    model_id: str
    family: ModelFamily
    training_target_ids_sha256: str
    feature_provenance_sha256: str
    response_sha256: str
    alpha: Decimal
    reduced_rank: int | None
    parameter_count: int
    residual_mae: Decimal
    residual_cosine: Decimal
    deterministic: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.fit_id, field_name="fit_id")
        validate_stable_id(self.model_id, field_name="model_id")
        for name, value in (
            ("training_target_ids_sha256", self.training_target_ids_sha256),
            ("feature_provenance_sha256", self.feature_provenance_sha256),
            ("response_sha256", self.response_sha256),
        ):
            validate_sha256(value, field_name=name)
        validate_decimal(self.alpha, field_name="alpha", minimum=Decimal("0"))
        if self.reduced_rank is not None and self.reduced_rank <= 0:
            raise ValueError("reduced rank must be positive when present")
        if isinstance(self.parameter_count, bool) or self.parameter_count < 0:
            raise ValueError("parameter_count must be nonnegative")
        validate_decimal(self.residual_mae, field_name="residual_mae", minimum=Decimal("0"))
        validate_decimal(
            self.residual_cosine,
            field_name="residual_cosine",
            minimum=Decimal("-1"),
        )
        if self.residual_cosine > 1:
            raise ValueError("residual cosine cannot exceed one")
        if not self.deterministic:
            raise ValueError("Tier-L0 model fitting must be deterministic")


@dataclass(frozen=True, slots=True)
class FalsifierResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/falsifier-result'

    falsifier_id: str
    passed: bool
    observed: Decimal
    threshold: Decimal
    comparison: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.falsifier_id, field_name="falsifier_id")
        validate_decimal(self.observed, field_name="observed")
        validate_decimal(self.threshold, field_name="threshold")
        if self.comparison not in {"GE", "GT", "LE", "LT", "EQ"}:
            raise ValueError("comparison is not supported")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        comparisons = {
            "GE": self.observed >= self.threshold,
            "GT": self.observed > self.threshold,
            "LE": self.observed <= self.threshold,
            "LT": self.observed < self.threshold,
            "EQ": self.observed == self.threshold,
        }
        if self.passed is not comparisons[self.comparison]:
            raise ValueError("falsifier status is not derived from its comparison")
        if not self.passed and not self.reason_codes:
            raise ValueError("failed falsifier requires a typed reason")


@dataclass(frozen=True, slots=True)
class TournamentEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/tournament-entry'

    candidate_id: str
    validation_des: Decimal
    validation_pds: Decimal
    validation_mae: Decimal
    official_score: Decimal
    official_score_standard_error: Decimal
    parameter_count: int
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_decimal(self.validation_des, field_name="validation_des", minimum=Decimal("0"))
        validate_decimal(self.validation_pds, field_name="validation_pds", minimum=Decimal("0"))
        validate_decimal(self.validation_mae, field_name="validation_mae", minimum=Decimal("0"))
        validate_decimal(self.official_score, field_name="official_score", minimum=Decimal("0"))
        validate_decimal(
            self.official_score_standard_error,
            field_name="official_score_standard_error",
            minimum=Decimal("0"),
        )
        if isinstance(self.parameter_count, bool) or self.parameter_count < 0:
            raise ValueError("parameter_count must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class TournamentRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/tournament-record'

    tournament_id: str
    entries: tuple[TournamentEntry, ...]
    selected_candidate_id: str
    selection_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.tournament_id, field_name="tournament_id")
        require_sorted_unique_ids(self.entries, attribute="candidate_id", field_name="entries")
        validate_stable_id(self.selected_candidate_id, field_name="selected_candidate_id")
        eligible = {entry.candidate_id for entry in self.entries if entry.eligible}
        if self.selected_candidate_id not in eligible:
            raise ValueError("selected candidate is not eligible")
        validate_nonempty(self.selection_rule, field_name="selection_rule")


@dataclass(frozen=True, slots=True)
class OfficialAggregateScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/official-aggregate-score'

    des: Decimal
    pds: Decimal
    mae: Decimal
    normalized_des: Decimal
    normalized_pds: Decimal
    normalized_mae: Decimal
    average_score: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("des", self.des),
            ("pds", self.pds),
            ("mae", self.mae),
            ("normalized_des", self.normalized_des),
            ("normalized_pds", self.normalized_pds),
            ("normalized_mae", self.normalized_mae),
            ("average_score", self.average_score),
        ):
            validate_decimal(value, field_name=name)
        for value in (
            self.normalized_des,
            self.normalized_pds,
            self.normalized_mae,
            self.average_score,
        ):
            if value < 0:
                raise ValueError("official normalized metrics are clipped at zero")


@dataclass(frozen=True, slots=True)
class CompiledPrediction:
    target_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    cell_target_ids: tuple[str, ...]
    values: NDArray[np.float64]
    requested_means: NDArray[np.float64]
    maximum_mean_error: float

    def __post_init__(self) -> None:
        if len(set(self.target_ids)) != len(self.target_ids) or not self.target_ids:
            raise ValueError("compiled target IDs must be nonempty and unique")
        if len(set(self.gene_ids)) != len(self.gene_ids) or not self.gene_ids:
            raise ValueError("compiled gene IDs must be nonempty and unique")
        if self.values.shape != (len(self.cell_target_ids), len(self.gene_ids)):
            raise ValueError("compiled cell array has the wrong shape")
        if self.requested_means.shape != (len(self.target_ids), len(self.gene_ids)):
            raise ValueError("requested mean array has the wrong shape")
        if set(self.cell_target_ids) != set(self.target_ids):
            raise ValueError("each and only each requested target must be compiled")
        if np.any(self.values < 0) or not np.all(np.isfinite(self.values)):
            raise ValueError("compiled prediction must be finite and nonnegative")
        if not math.isfinite(self.maximum_mean_error) or self.maximum_mean_error < 0:
            raise ValueError("maximum mean error is invalid")


def _registry_digest(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _array_digest(array: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _align_features(
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix,
) -> NDArray[np.float64]:
    positions = {target_id: index for index, target_id in enumerate(features.target_ids)}
    missing = set(target_ids) - set(positions)
    if missing:
        raise VirtualCellAnalysisError(
            f"features lack requested targets: {', '.join(sorted(missing))}"
        )
    return np.asarray([features.values[positions[target_id]] for target_id in target_ids])


def augment_basal_target_expression(
    features: TargetFeatureMatrix,
    *,
    gene_ids: tuple[str, ...],
    control_mean: NDArray[np.float64],
) -> TargetFeatureMatrix:
    """Append permitted comparator expression for each target before prediction."""

    if control_mean.shape != (len(gene_ids),) or not np.all(np.isfinite(control_mean)):
        raise ValueError("control mean is not aligned to the gene registry")
    positions = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    missing = tuple(sorted(set(features.target_ids) - set(positions)))
    if missing:
        raise VirtualCellAnalysisError(
            "target features cannot map basal expression for: " + ", ".join(missing)
        )
    basal = np.asarray(
        [control_mean[positions[target_id]] for target_id in features.target_ids],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(features.provenance_sha256.encode("ascii"))
    digest.update(_array_digest(basal[:, None]).encode("ascii"))
    digest.update(_registry_digest(gene_ids).encode("ascii"))
    return TargetFeatureMatrix(
        target_ids=features.target_ids,
        feature_ids=(*features.feature_ids, "basal_control_target_log1p"),
        values=np.concatenate((features.values, basal[:, None]), axis=1),
        provenance_sha256=digest.hexdigest(),
    )


def realized_admission_labels(
    responses: TargetResponseMatrix,
    *,
    repression_scale: float,
) -> NDArray[np.float64]:
    """Development-only realized repression label; never a held-target input."""

    if not math.isfinite(repression_scale) or repression_scale <= 0:
        raise ValueError("realized-admission repression scale must be positive")
    positions = {gene_id: index for index, gene_id in enumerate(responses.gene_ids)}
    missing = tuple(sorted(set(responses.target_ids) - set(positions)))
    if missing:
        raise VirtualCellAnalysisError(
            "realized admission cannot map targets to receiver genes: " + ", ".join(missing)
        )
    diagonal = np.asarray(
        [
            responses.values[index, positions[target_id]]
            for index, target_id in enumerate(responses.target_ids)
        ],
        dtype=np.float64,
    )
    return np.asarray(np.clip(-diagonal / repression_scale, 0.0, 1.0), dtype=np.float64)


def fit_prefix_admission_model(
    *,
    model_id: str,
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix,
    realized_labels: NDArray[np.float64],
    alpha: float,
) -> PrefixAdmissionModel:
    """Fit a bounded-use ridge predictor from development-only realized labels."""

    if realized_labels.shape != (len(target_ids),):
        raise ValueError("realized-admission labels are not target aligned")
    if (
        not np.all(np.isfinite(realized_labels))
        or np.any(realized_labels < 0)
        or np.any(realized_labels > 1)
    ):
        raise ValueError("realized-admission labels must lie in [0, 1]")
    design, standardization = _standardize(_align_features(target_ids, features))
    coefficients, intercept = _ridge(
        design,
        realized_labels[:, None],
        alpha=alpha,
        weights=np.ones(len(target_ids), dtype=np.float64),
    )
    return PrefixAdmissionModel(
        model_id=model_id,
        feature_ids=features.feature_ids,
        standardization=standardization,
        coefficients=np.asarray(coefficients[:, 0], dtype=np.float64),
        intercept=float(intercept[0]),
        alpha=alpha,
    )


def predict_prefix_admission(
    model: PrefixAdmissionModel,
    *,
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix,
) -> NDArray[np.float64]:
    if features.feature_ids != model.feature_ids:
        raise VirtualCellAnalysisError("prefix-admission feature registry differs")
    raw = _align_features(target_ids, features)
    design = (raw - model.standardization.mean) / model.standardization.scale
    values = design @ model.coefficients + model.intercept
    if not np.all(np.isfinite(values)):
        raise VirtualCellAnalysisError("prefix-admission prediction is nonfinite")
    return np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float64)


def cross_fitted_prefix_admission(
    *,
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix,
    realized_labels: NDArray[np.float64],
    alpha: float,
    fold_count: int,
    seed: int,
) -> NDArray[np.float64]:
    """Produce training design values without in-fold realized-label reuse."""

    if realized_labels.shape != (len(target_ids),):
        raise ValueError("realized-admission labels are not target aligned")
    result = np.empty(len(target_ids), dtype=np.float64)
    all_indices = np.arange(len(target_ids))
    for fold_index, held_indices in enumerate(
        deterministic_target_folds(target_ids, fold_count=fold_count, seed=seed)
    ):
        held = np.asarray(held_indices, dtype=np.int64)
        training = np.setdiff1d(all_indices, held, assume_unique=True)
        training_ids = tuple(target_ids[index] for index in training)
        model = fit_prefix_admission_model(
            model_id=f"prefix-admission.crossfit-{seed}-{fold_index}",
            target_ids=training_ids,
            features=features,
            realized_labels=realized_labels[training],
            alpha=alpha,
        )
        held_ids = tuple(target_ids[index] for index in held)
        result[held] = predict_prefix_admission(
            model,
            target_ids=held_ids,
            features=features,
        )
    return result


def collapse_target_batch_responses(
    summary: ResponseSummaryArrays,
    *,
    control_label: str,
) -> TargetResponseMatrix:
    """Form equal-batch target displacements against exact matched controls."""

    validate_nonempty(control_label, field_name="control_label")
    controls = {
        batch_id: summary.means[index]
        for index, (target_id, batch_id) in enumerate(
            zip(summary.target_ids, summary.batch_ids, strict=True)
        )
        if target_id == control_label
    }
    if not controls:
        raise VirtualCellAnalysisError("response summary has no declared comparator cells")
    deltas: dict[str, list[NDArray[np.float64]]] = {}
    for index, (target_id, batch_id) in enumerate(
        zip(summary.target_ids, summary.batch_ids, strict=True)
    ):
        if target_id == control_label:
            continue
        control = controls.get(batch_id)
        if control is None:
            raise VirtualCellAnalysisError(
                f"target {target_id} has no comparator in batch {batch_id}"
            )
        deltas.setdefault(target_id, []).append(summary.means[index] - control)
    target_ids = tuple(sorted(deltas))
    if not target_ids:
        raise VirtualCellAnalysisError("response summary has no perturbed targets")
    values = np.stack([np.mean(deltas[target_id], axis=0) for target_id in target_ids])
    return TargetResponseMatrix(
        target_ids=target_ids,
        gene_ids=summary.gene_ids,
        values=np.asarray(values, dtype=np.float64),
        target_weights=np.ones(len(target_ids), dtype=np.float64),
    )


def _standardize(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], Standardization]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    standardization = Standardization(mean=mean, scale=scale)
    return (values - mean) / scale, standardization


def _ridge(
    design: NDArray[np.float64],
    response: NDArray[np.float64],
    *,
    alpha: float,
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha must be finite and nonnegative")
    if design.ndim != 2 or response.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("ridge arrays have inconsistent shapes")
    normalized_weights = weights / np.mean(weights)
    intercept = np.average(response, axis=0, weights=normalized_weights)
    centered_response = response - intercept
    weighted_design = design * np.sqrt(normalized_weights[:, None])
    weighted_response = centered_response * np.sqrt(normalized_weights[:, None])
    gram = weighted_design.T @ weighted_design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * alpha
    try:
        coefficients = np.linalg.solve(gram + regularizer, weighted_design.T @ weighted_response)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + regularizer) @ weighted_design.T @ weighted_response
    return (
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(intercept, dtype=np.float64),
    )


def _metrics(truth: NDArray[np.float64], prediction: NDArray[np.float64]) -> tuple[float, float]:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("metric arrays must be aligned matrices")
    mae = float(np.mean(np.abs(truth - prediction)))
    truth_norm = np.linalg.norm(truth, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    denominator = truth_norm * pred_norm
    dot = np.sum(truth * prediction, axis=1)
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    both_zero = (truth_norm == 0) & (pred_norm == 0)
    cosine[both_zero] = 1.0
    return mae, float(np.mean(cosine))


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise VirtualCellAnalysisError("cannot record a nonfinite metric")
    return Decimal(str(value))


def fit_tier_l0_model(
    *,
    model_id: str,
    family: ModelFamily,
    responses: TargetResponseMatrix,
    features: TargetFeatureMatrix | None,
    alpha: float = 1.0,
    reduced_rank: int | None = None,
    prefix_admission: NDArray[np.float64] | None = None,
    prefix_admission_model: PrefixAdmissionModel | None = None,
    response_basis_override: NDArray[np.float64] | None = None,
) -> tuple[LinearResponseModel, ModelFitRecord]:
    "Fit the no-change baseline, weighted common-response baseline, target-feature ridge, reduced-rank target-feature ridge or admission-conditioned reduced-rank ridge model."

    validate_stable_id(model_id, field_name="model_id")
    n_targets, n_genes = responses.values.shape
    admission_count = 0
    feature_ids: tuple[str, ...]
    if family in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}:
        if (
            features is not None
            or prefix_admission is not None
            or prefix_admission_model is not None
            or response_basis_override is not None
        ):
            raise VirtualCellAnalysisError("No-change and weighted common-response baselines do not accept target features")
        feature_ids = ("constant",)
        coefficients = np.zeros((1, n_genes), dtype=np.float64)
        intercept = (
            np.zeros(n_genes, dtype=np.float64)
            if family is ModelFamily.NO_CHANGE_BASELINE
            else np.average(responses.values, axis=0, weights=responses.target_weights)
        )
        standardization = None
        basis = None
    else:
        if features is None:
            raise VirtualCellAnalysisError(f"{family.value} requires outcome-blind target features")
        raw_design = _align_features(responses.target_ids, features)
        feature_ids = features.feature_ids
        if family is ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE:
            if prefix_admission is None or prefix_admission.shape != (n_targets,):
                raise VirtualCellAnalysisError("Admission-conditioned reduced-rank ridge requires one prefix admission value per target")
            if prefix_admission_model is None:
                raise VirtualCellAnalysisError("Admission-conditioned reduced-rank ridge requires its fitted prefix-admission predictor")
            if (
                not np.all(np.isfinite(prefix_admission))
                or np.any(prefix_admission < 0)
                or np.any(prefix_admission > 1)
            ):
                raise VirtualCellAnalysisError("prefix admission must lie in [0, 1]")
            raw_design = np.concatenate(
                [
                    raw_design,
                    prefix_admission[:, None],
                    raw_design * prefix_admission[:, None],
                ],
                axis=1,
            )
            feature_ids = (
                *feature_ids,
                "expected_admission_prefix",
                *(f"admission_x_{value}" for value in features.feature_ids),
            )
            admission_count = 1 + len(features.feature_ids)
        elif prefix_admission is not None or prefix_admission_model is not None:
            raise VirtualCellAnalysisError("only admission-conditioned reduced-rank ridge accepts prefix-admission inputs")
        design, standardization = _standardize(raw_design)
        if family in {ModelFamily.REDUCED_RANK_TARGET_FEATURE_RIDGE_RESPONSE, ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE}:
            if reduced_rank is None or reduced_rank <= 0:
                raise VirtualCellAnalysisError(f"{family.value} requires a positive reduced rank")
            maximum_rank = min(n_targets - 1, n_genes)
            if reduced_rank > maximum_rank:
                raise VirtualCellAnalysisError(
                    f"reduced rank {reduced_rank} exceeds training bound {maximum_rank}"
                )
            response_center = np.average(
                responses.values,
                axis=0,
                weights=responses.target_weights,
            )
            centered = responses.values - response_center
            if response_basis_override is None:
                _, _, right = np.linalg.svd(centered, full_matrices=False)
                basis = np.asarray(right[:reduced_rank], dtype=np.float64)
            else:
                if (
                    response_basis_override.ndim != 2
                    or response_basis_override.shape[0] < reduced_rank
                    or response_basis_override.shape[1] != n_genes
                    or not np.all(np.isfinite(response_basis_override))
                ):
                    raise VirtualCellAnalysisError("response-basis override is invalid")
                basis = np.asarray(response_basis_override[:reduced_rank], dtype=np.float64)
            scores = centered @ basis.T
            coefficients, score_intercept = _ridge(
                design,
                scores,
                alpha=alpha,
                weights=responses.target_weights,
            )
            intercept = response_center + score_intercept @ basis
        else:
            if response_basis_override is not None:
                raise VirtualCellAnalysisError("only reduced-rank families accept a basis")
            basis = None
            coefficients, intercept = _ridge(
                design,
                responses.values,
                alpha=alpha,
                weights=responses.target_weights,
            )
    provisional = LinearResponseModel(
        model_id=model_id,
        family=family,
        feature_ids=tuple(feature_ids),
        gene_ids=responses.gene_ids,
        standardization=standardization,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=np.asarray(intercept, dtype=np.float64),
        response_basis=basis,
        admission_required=family is ModelFamily.ADMISSION_CONDITIONED_REDUCED_RANK_RIDGE_RESPONSE,
        admission_feature_count=admission_count,
        prefix_admission_model=prefix_admission_model,
        residual_scale=np.zeros(n_genes, dtype=np.float64),
    )
    fitted = predict_response(
        provisional,
        target_ids=responses.target_ids,
        features=features,
        prefix_admission=prefix_admission,
    )
    residual = responses.values - fitted
    residual_scale = np.sqrt(
        np.average(residual * residual, axis=0, weights=responses.target_weights)
    )
    model = LinearResponseModel(
        model_id=provisional.model_id,
        family=provisional.family,
        feature_ids=provisional.feature_ids,
        gene_ids=provisional.gene_ids,
        standardization=provisional.standardization,
        coefficients=provisional.coefficients,
        intercept=provisional.intercept,
        response_basis=provisional.response_basis,
        admission_required=provisional.admission_required,
        admission_feature_count=provisional.admission_feature_count,
        prefix_admission_model=provisional.prefix_admission_model,
        residual_scale=residual_scale,
    )
    mae, cosine = _metrics(responses.values, fitted)
    if family is ModelFamily.NO_CHANGE_BASELINE:
        parameter_count = 0
    elif family is ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE:
        parameter_count = int(model.intercept.size)
    else:
        parameter_count = int(
            model.coefficients.size
            + model.intercept.size
            + (0 if model.response_basis is None else model.response_basis.size)
            + (
                0
                if model.prefix_admission_model is None
                else (model.prefix_admission_model.coefficients.size + 1)
            )
        )
    fit = ModelFitRecord(
        fit_id=f"fit.{model_id}",
        model_id=model_id,
        family=family,
        training_target_ids_sha256=_registry_digest(responses.target_ids),
        feature_provenance_sha256=(
            features.provenance_sha256 if features is not None else "0" * 64
        ),
        response_sha256=_array_digest(responses.values),
        alpha=_decimal(alpha),
        reduced_rank=reduced_rank,
        parameter_count=parameter_count,
        residual_mae=_decimal(mae),
        residual_cosine=_decimal(cosine),
        deterministic=True,
    )
    return model, fit


def _build_prediction_design(
    model: LinearResponseModel,
    *,
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix | None,
    prefix_admission: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    if model.family in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}:
        if features is not None or prefix_admission is not None:
            raise VirtualCellAnalysisError("No-change and weighted common-response baseline prediction does not accept features")
        return np.zeros((len(target_ids), 1), dtype=np.float64)
    if features is None:
        raise VirtualCellAnalysisError("model prediction requires target features")
    design = _align_features(target_ids, features)
    if model.admission_required:
        if prefix_admission is None:
            if model.prefix_admission_model is None:
                raise VirtualCellAnalysisError("Admission-conditioned reduced-rank ridge prediction lacks a prefix-admission model")
            prefix_admission = predict_prefix_admission(
                model.prefix_admission_model,
                target_ids=target_ids,
                features=features,
            )
        if prefix_admission.shape != (len(target_ids),):
            raise VirtualCellAnalysisError("Admission-conditioned reduced-rank ridge prediction requires aligned prefix admission")
        if np.any(prefix_admission < 0) or np.any(prefix_admission > 1):
            raise VirtualCellAnalysisError("prefix admission must lie in [0, 1]")
        design = np.concatenate(
            [design, prefix_admission[:, None], design * prefix_admission[:, None]],
            axis=1,
        )
    if model.standardization is None:
        raise VirtualCellAnalysisError("feature model lacks training standardization")
    return (design - model.standardization.mean) / model.standardization.scale


def predict_response(
    model: LinearResponseModel,
    *,
    target_ids: tuple[str, ...],
    features: TargetFeatureMatrix | None,
    prefix_admission: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    design = _build_prediction_design(
        model,
        target_ids=target_ids,
        features=features,
        prefix_admission=prefix_admission,
    )
    coordinates = design @ model.coefficients
    if model.response_basis is not None:
        prediction = coordinates @ model.response_basis + model.intercept
    else:
        prediction = coordinates + model.intercept
    if not np.all(np.isfinite(prediction)):
        raise VirtualCellAnalysisError("model prediction produced nonfinite values")
    return np.asarray(prediction, dtype=np.float64)


def deterministic_target_folds(
    target_ids: tuple[str, ...],
    *,
    fold_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Hash targets into reproducible folds; rows from one target cannot split."""

    if fold_count < 2 or fold_count > len(target_ids):
        raise ValueError("fold_count must be between two and the number of targets")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target IDs must be unique before grouped splitting")
    ordered = sorted(
        range(len(target_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{target_ids[index]}".encode("utf-8")).digest(),
    )
    folds: list[list[int]] = [list() for _ in range(fold_count)]
    for position, index in enumerate(ordered):
        folds[position % fold_count].append(index)
    return tuple(tuple(sorted(fold)) for fold in folds)


def evaluate_prediction(
    truth: TargetResponseMatrix,
    prediction: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    mae, cosine = _metrics(truth.values, prediction)
    per_target = np.mean(np.abs(truth.values - prediction), axis=1)
    return mae, cosine, per_target


def select_tournament(
    *,
    tournament_id: str,
    entries: tuple[TournamentEntry, ...],
    one_standard_error_margin: Decimal | None = None,
) -> TournamentRecord:
    """Choose the best exact validation score, favoring lower capacity within one SE."""

    eligible = [entry for entry in entries if entry.eligible]
    if not eligible:
        raise VirtualCellAnalysisError("tournament has no eligible candidates")
    best = min(
        eligible,
        key=lambda entry: (-entry.official_score, entry.parameter_count, entry.candidate_id),
    )
    margin = (
        best.official_score_standard_error
        if one_standard_error_margin is None
        else one_standard_error_margin
    )
    validate_decimal(
        margin,
        field_name="one_standard_error_margin",
        minimum=Decimal("0"),
    )
    competitive = [
        entry for entry in eligible if entry.official_score >= best.official_score - margin
    ]
    selected = min(competitive, key=lambda entry: (entry.parameter_count, entry.candidate_id))
    return TournamentRecord(
        tournament_id=tournament_id,
        entries=tuple(sorted(entries, key=lambda entry: entry.candidate_id)),
        selected_candidate_id=selected.candidate_id,
        selection_rule=(
            "maximum exact official validation score; best-candidate target-bootstrap "
            "one-standard-error capacity/ID tie-break"
        ),
    )


def official_aggregate_score(
    *,
    des: Decimal,
    pds: Decimal,
    mae: Decimal,
    metric_contract: MetricContract,
) -> OfficialAggregateScore:
    """Reproduce the exact 2025 scalar normalization and clipping formula."""

    for name, value in (("des", des), ("pds", pds), ("mae", mae)):
        validate_decimal(value, field_name=name)
    baseline = metric_contract.baseline
    normalized_des = max(Decimal("0"), (des - baseline.des) / (Decimal("1") - baseline.des))
    normalized_pds = max(Decimal("0"), (pds - baseline.pds) / (Decimal("1") - baseline.pds))
    normalized_mae = max(Decimal("0"), Decimal("1") - mae / baseline.mae)
    average = (normalized_des + normalized_pds + normalized_mae) / Decimal("3")
    return OfficialAggregateScore(
        des=des,
        pds=pds,
        mae=mae,
        normalized_des=normalized_des,
        normalized_pds=normalized_pds,
        normalized_mae=normalized_mae,
        average_score=average,
    )


def verify_official_evaluator_versions(metric_contract: MetricContract) -> tuple[str, str]:
    """Fail unless the installed evaluator distributions exactly match the freeze."""

    try:
        cell_eval_version = version(metric_contract.scorer_distribution)
        pdex_version = version("pdex")
    except PackageNotFoundError as error:
        raise VirtualCellAnalysisError(
            "the exact official evaluator stack is not installed"
        ) from error
    if cell_eval_version != metric_contract.scorer_version:
        raise VirtualCellAnalysisError(
            f"cell-eval version mismatch: {cell_eval_version} != {metric_contract.scorer_version}"
        )
    if pdex_version != metric_contract.pdex_version:
        raise VirtualCellAnalysisError(
            f"pdex version mismatch: {pdex_version} != {metric_contract.pdex_version}"
        )
    return cell_eval_version, pdex_version


def adjudicate_placement(
    *,
    adjudication_id: str,
    prediction_commitment_id: str,
    metric_contract_sha256: str,
    leaderboard_snapshot_sha256: str,
    candidate_score: Decimal,
    snapshot: LeaderboardSnapshot,
) -> PlacementAdjudication:
    """Return exact/tied placement or honest bounds for a partial/rounded field."""

    validate_decimal(candidate_score, field_name="candidate_score")
    reasons: tuple[str, ...]
    if snapshot.complete and snapshot.full_precision:
        strictly_better = sum(entry.score > candidate_score for entry in snapshot.entries)
        equal = sum(entry.score == candidate_score for entry in snapshot.entries)
        best_rank = strictly_better + 1
        worst_rank = strictly_better + equal + 1
        placement_class = (
            PlacementClass.EXACT_VALIDATED_RANK if equal == 0 else PlacementClass.TIED_RANK_INTERVAL
        )
        reasons = (
            ("complete-full-precision-field",)
            if equal == 0
            else ("complete-full-precision-field", "candidate-score-tie")
        )
    elif snapshot.entries:
        displayed_scores = [entry.score for entry in snapshot.entries]
        if not snapshot.full_precision:
            precision = max(
                -cast(int, entry.score.as_tuple().exponent) for entry in snapshot.entries
            )
            half_unit = Decimal(5).scaleb(-(precision + 1))
        else:
            half_unit = Decimal("0")
        definitely_better = sum(score - half_unit > candidate_score for score in displayed_scores)
        possibly_better = sum(score + half_unit >= candidate_score for score in displayed_scores)
        best_rank = definitely_better + 1
        if snapshot.complete:
            worst_rank = possibly_better + 1
            placement_class = PlacementClass.ROUNDED_SCORE_RANK_INTERVAL
            reasons = ("complete-rounded-field", "rounding-overlap-bounded")
        elif candidate_score > displayed_scores[-1] + half_unit:
            worst_rank = possibly_better + 1
            if not snapshot.full_precision:
                placement_class = PlacementClass.ROUNDED_SCORE_RANK_INTERVAL
                reasons = (
                    "candidate-above-partial-cutoff",
                    "partial-field-below-candidate",
                    "rounding-overlap-bounded",
                )
            elif possibly_better > definitely_better:
                placement_class = PlacementClass.TIED_RANK_INTERVAL
                reasons = (
                    "candidate-above-partial-cutoff",
                    "candidate-score-tie",
                    "partial-field-below-candidate",
                )
            else:
                placement_class = PlacementClass.EXACT_VALIDATED_RANK
                reasons = (
                    "candidate-above-partial-cutoff",
                    "full-precision-known-rank",
                    "partial-field-below-candidate",
                )
        else:
            # A candidate equal to the published cutoff can tie an observed
            # row even though the lower field is unavailable.  Preserve that
            # best case while allowing every unseen row in the worst case.
            best_rank = definitely_better + 1
            worst_rank = snapshot.total_ranked_entries + 1
            placement_class = PlacementClass.PARTIAL_FIELD_RANK_BOUND
            reasons = ("candidate-at-or-below-partial-cutoff", "unobserved-lower-field")
    else:
        best_rank = None
        worst_rank = None
        placement_class = PlacementClass.NOT_COMPARABLE
        reasons = ("leaderboard-has-no-ranked-rows",)
    return PlacementAdjudication(
        adjudication_id=adjudication_id,
        prediction_commitment_id=prediction_commitment_id,
        metric_contract_sha256=metric_contract_sha256,
        leaderboard_snapshot_sha256=leaderboard_snapshot_sha256,
        candidate_score=candidate_score,
        placement_class=placement_class,
        best_rank=best_rank,
        worst_rank=worst_rank,
        field_size=snapshot.total_ranked_entries,
        reason_codes=tuple(sorted(reasons)),
    )


def _project_to_nonnegative_sum(vector: NDArray[np.float64], total: float) -> NDArray[np.float64]:
    """Euclidean projection onto the nonnegative simplex with exact sum."""

    if not math.isfinite(total) or total < 0:
        raise VirtualCellAnalysisError("compiler requested an invalid nonnegative total")
    if total == 0:
        return np.zeros_like(vector)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - total
    candidates = np.nonzero(ordered - cumulative / np.arange(1, len(vector) + 1) > 0)[0]
    if candidates.size == 0:
        raise VirtualCellAnalysisError("nonnegative projection failed")
    rho = int(candidates[-1])
    theta = cumulative[rho] / (rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    residual = total - float(np.sum(projected))
    projected[int(np.argmax(projected))] += residual
    if np.any(projected < -1e-12):
        raise VirtualCellAnalysisError("nonnegative projection produced a negative value")
    return np.asarray(np.maximum(projected, 0.0), dtype=np.float64)


def compile_prediction_cells(
    *,
    target_ids: tuple[str, ...],
    gene_ids: tuple[str, ...],
    predicted_means: NDArray[np.float64],
    control_cells: NDArray[np.float64],
    cells_per_target: int,
    seed: int,
    mean_tolerance: float,
) -> CompiledPrediction:
    """Shift control residuals and project each gene to the requested mean."""

    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("target IDs must be nonempty and unique")
    if not gene_ids or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("gene IDs must be nonempty and unique")
    if predicted_means.shape != (len(target_ids), len(gene_ids)):
        raise ValueError("predicted means have the wrong shape")
    if control_cells.ndim != 2 or control_cells.shape[1] != len(gene_ids):
        raise ValueError("control-cell matrix has the wrong gene dimension")
    if control_cells.shape[0] == 0 or cells_per_target <= 0:
        raise ValueError("compiler requires controls and a positive cell count")
    if not np.all(np.isfinite(predicted_means)) or not np.all(np.isfinite(control_cells)):
        raise ValueError("compiler inputs must be finite")
    if np.any(predicted_means < 0):
        raise VirtualCellAnalysisError("requested count-compatible means cannot be negative")
    if not math.isfinite(mean_tolerance) or mean_tolerance < 0:
        raise ValueError("mean tolerance must be finite and nonnegative")
    control_mean = np.mean(control_cells, axis=0)
    residuals = control_cells - control_mean
    rng = np.random.default_rng(seed)
    compiled: list[NDArray[np.float64]] = []
    cell_targets: list[str] = []
    for target_index, target_id in enumerate(target_ids):
        sampled = residuals[
            rng.integers(0, residuals.shape[0], size=cells_per_target, endpoint=False)
        ]
        cells = sampled + predicted_means[target_index]
        for gene_index in range(len(gene_ids)):
            cells[:, gene_index] = _project_to_nonnegative_sum(
                cells[:, gene_index],
                float(cells_per_target * predicted_means[target_index, gene_index]),
            )
        compiled.append(cells)
        cell_targets.extend([target_id] * cells_per_target)
    values = np.concatenate(compiled, axis=0)
    observed_means = np.stack(
        [np.mean(values[np.asarray(cell_targets) == target_id], axis=0) for target_id in target_ids]
    )
    maximum_error = float(np.max(np.abs(observed_means - predicted_means)))
    if maximum_error > mean_tolerance:
        raise VirtualCellAnalysisError(
            f"compiled prediction mean error {maximum_error} exceeds {mean_tolerance}"
        )
    return CompiledPrediction(
        target_ids=target_ids,
        gene_ids=gene_ids,
        cell_target_ids=tuple(cell_targets),
        values=values,
        requested_means=np.asarray(predicted_means, dtype=np.float64),
        maximum_mean_error=maximum_error,
    )


__all__ = [
    "CompiledPrediction",
    "FalsifierResult",
    "LinearResponseModel",
    "ModelFamily",
    "ModelFitRecord",
    "OfficialAggregateScore",
    "PrefixAdmissionModel",
    "Standardization",
    "TargetFeatureMatrix",
    "TargetResponseMatrix",
    "TournamentEntry",
    "TournamentRecord",
    "VirtualCellAnalysisError",
    "adjudicate_placement",
    "augment_basal_target_expression",
    "collapse_target_batch_responses",
    "compile_prediction_cells",
    "deterministic_target_folds",
    "evaluate_prediction",
    "fit_tier_l0_model",
    "fit_prefix_admission_model",
    "official_aggregate_score",
    "predict_response",
    "predict_prefix_admission",
    "cross_fitted_prefix_admission",
    "realized_admission_labels",
    "select_tournament",
    "verify_official_evaluator_versions",
]
