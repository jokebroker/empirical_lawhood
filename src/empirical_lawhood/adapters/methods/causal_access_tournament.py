"""Substrate-neutral five-arm access-versus-scale tournament.

The method deliberately stays small and transparent.  It compares a linear
baseline, the same model with twice as many independent development units, a
quadratic same-interface model, a linear model with a nominated access view,
and a dimension-matched sham view.  Nested observations contribute to a
unit-level loss but never become resampling units.

This module does not decide whether an added coordinate is causal.  The caller
must supply a truth-known or intervention-backed rival disposition separately;
prediction alone can therefore produce at most ``PREDICTIVE_ACCESS_ONLY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


class TournamentArm(StrEnum):
    BASE = "BASE"
    MORE_DATA = "MORE_DATA"
    RICHER_MODEL = "RICHER_MODEL"
    BETTER_ACCESS = "BETTER_ACCESS"
    SHAM_ACCESS = "SHAM_ACCESS"


class TournamentDisposition(StrEnum):
    ACCESS_DOMINANT = "ACCESS_DOMINANT"
    PREDICTIVE_ACCESS_ONLY = "PREDICTIVE_ACCESS_ONLY"
    SCALE_DOMINANT = "SCALE_DOMINANT"
    ACCESS_AND_SCALE_EQUIVALENT = "ACCESS_AND_SCALE_EQUIVALENT"
    WRONG_ACCESS = "WRONG_ACCESS"
    NO_RESPONSE_TO_SEPARATE = "NO_RESPONSE_TO_SEPARATE"
    ACCESS_OPERAND_UNAVAILABLE = "ACCESS_OPERAND_UNAVAILABLE"
    UNEVALUABLE = "UNEVALUABLE"


class FeatureMap(StrEnum):
    LINEAR = "LINEAR"
    QUADRATIC = "QUADRATIC"


@dataclass(frozen=True, slots=True)
class TournamentObservation:
    """One nested observation belonging to one independent unit."""

    observation_id: str
    unit_id: str
    baseline_features: tuple[float, ...]
    access_features: tuple[float, ...]
    sham_features: tuple[float, ...]
    target: float

    def __post_init__(self) -> None:
        if not self.observation_id or not self.unit_id:
            raise ValueError("tournament observation identifiers cannot be empty")
        if not self.baseline_features:
            raise ValueError("tournament baseline view cannot be empty")
        if not self.access_features:
            raise ValueError("tournament access view cannot be empty")
        if len(self.access_features) != len(self.sham_features):
            raise ValueError("access and sham views must have equal dimension")
        values = (*self.baseline_features, *self.access_features, *self.sham_features, self.target)
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("tournament observation contains a non-finite value")


@dataclass(frozen=True, slots=True)
class TournamentPredictorInput:
    """Outcome-blind feature row used for prospective action commits."""

    observation_id: str
    unit_id: str
    action_id: str
    baseline_features: tuple[float, ...]
    access_features: tuple[float, ...]
    sham_features: tuple[float, ...]
    effort: float
    supported: bool

    def __post_init__(self) -> None:
        if not self.observation_id or not self.unit_id or not self.action_id:
            raise ValueError("action predictor identifiers cannot be empty")
        if not self.baseline_features or not self.access_features:
            raise ValueError("action predictor views cannot be empty")
        if len(self.access_features) != len(self.sham_features):
            raise ValueError("action access and sham views must have equal dimension")
        values = (*self.baseline_features, *self.access_features, *self.sham_features, self.effort)
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("action predictor contains a non-finite value")
        if self.effort < 0:
            raise ValueError("action effort cannot be negative")


def _quadratic_features(matrix: FloatArray) -> FloatArray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("quadratic feature input must be a nonempty matrix")
    columns: list[FloatArray] = [matrix]
    products = [
        (matrix[:, left] * matrix[:, right])[:, None]
        for left in range(matrix.shape[1])
        for right in range(left, matrix.shape[1])
    ]
    columns.extend(products)
    return np.concatenate(columns, axis=1)


@dataclass(frozen=True, slots=True)
class StandardizedRidge:
    """A deterministic scalar ridge model with a frozen feature map."""

    feature_map: FeatureMap
    input_dimension: int
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge_alpha: float

    def __post_init__(self) -> None:
        if self.input_dimension <= 0:
            raise ValueError("ridge model input dimension must be positive")
        expected = (
            self.input_dimension
            if self.feature_map is FeatureMap.LINEAR
            else self.input_dimension + self.input_dimension * (self.input_dimension + 1) // 2
        )
        if not (len(self.center) == len(self.scale) == len(self.coefficients) == expected):
            raise ValueError("ridge model coefficient dimensions are inconsistent")
        values = (*self.center, *self.scale, *self.coefficients, self.intercept, self.ridge_alpha)
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("ridge model contains a non-finite value")
        if any(value <= 0 for value in self.scale) or self.ridge_alpha <= 0:
            raise ValueError("ridge scales and penalty must be positive")

    def _mapped(self, matrix: FloatArray) -> FloatArray:
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dimension:
            raise ValueError("ridge prediction input dimension changed")
        return matrix if self.feature_map is FeatureMap.LINEAR else _quadratic_features(matrix)

    def predict(self, matrix: FloatArray) -> FloatArray:
        mapped = self._mapped(np.asarray(matrix, dtype=np.float64))
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        values = self.intercept + ((mapped - center) / scale) @ coefficients
        if not np.all(np.isfinite(values)):
            raise ValueError("ridge prediction produced a non-finite value")
        return np.asarray(values, dtype=np.float64)

    def to_mapping(self) -> dict[str, object]:
        return {
            "feature_map": self.feature_map.value,
            "input_dimension": self.input_dimension,
            "center": list(self.center),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "ridge_alpha": self.ridge_alpha,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StandardizedRidge:
        expected = {
            "feature_map",
            "input_dimension",
            "center",
            "scale",
            "coefficients",
            "intercept",
            "ridge_alpha",
        }
        if set(value) != expected:
            raise ValueError("ridge model mapping has unexpected fields")
        return cls(
            feature_map=FeatureMap(str(value["feature_map"])),
            input_dimension=_integer(value["input_dimension"]),
            center=tuple(_number(item) for item in _sequence(value["center"])),
            scale=tuple(_number(item) for item in _sequence(value["scale"])),
            coefficients=tuple(_number(item) for item in _sequence(value["coefficients"])),
            intercept=_number(value["intercept"]),
            ridge_alpha=_number(value["ridge_alpha"]),
        )


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("expected a numeric sequence")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("expected a numeric value")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("expected an integer value")
    return value


def fit_standardized_ridge(
    features: FloatArray,
    targets: FloatArray,
    *,
    feature_map: FeatureMap,
    ridge_alpha: float,
) -> StandardizedRidge:
    """Fit a deterministic transparent ridge predictor."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge fit requires aligned nontrivial feature and target arrays")
    if x.shape[1] == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge fit received an empty or non-finite array")
    if ridge_alpha <= 0 or not np.isfinite(ridge_alpha):
        raise ValueError("ridge penalty must be finite and positive")
    mapped = x if feature_map is FeatureMap.LINEAR else _quadratic_features(x)
    center = np.mean(mapped, axis=0)
    scale = np.std(mapped, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (mapped - center) / scale
    intercept = float(np.mean(y))
    centered_target = y - intercept
    gram = standardized.T @ standardized
    penalized = gram + ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(penalized, standardized.T @ centered_target)
    return StandardizedRidge(
        feature_map=feature_map,
        input_dimension=x.shape[1],
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
        ridge_alpha=ridge_alpha,
    )


@dataclass(frozen=True, slots=True)
class TournamentModelSet:
    models: Mapping[TournamentArm, StandardizedRidge]
    core_unit_ids: tuple[str, ...]
    additional_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.models) != set(TournamentArm):
            raise ValueError("tournament model set must contain exactly five arms")
        if not self.core_unit_ids or not self.additional_unit_ids:
            raise ValueError("tournament requires both core and additional units")
        if set(self.core_unit_ids) & set(self.additional_unit_ids):
            raise ValueError("core and additional development units overlap")
        if tuple(sorted(self.core_unit_ids)) != self.core_unit_ids:
            raise ValueError("core unit identifiers must be sorted")
        if tuple(sorted(self.additional_unit_ids)) != self.additional_unit_ids:
            raise ValueError("additional unit identifiers must be sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "models": {
                arm.value: self.models[arm].to_mapping()
                for arm in sorted(TournamentArm, key=lambda item: item.value)
            },
            "core_unit_ids": list(self.core_unit_ids),
            "additional_unit_ids": list(self.additional_unit_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TournamentModelSet:
        if set(value) != {"models", "core_unit_ids", "additional_unit_ids"}:
            raise ValueError("tournament model mapping has unexpected fields")
        raw_models = value["models"]
        if not isinstance(raw_models, Mapping):
            raise ValueError("tournament models must be a mapping")
        models: dict[TournamentArm, StandardizedRidge] = {}
        for key, raw_model in raw_models.items():
            if not isinstance(raw_model, Mapping):
                raise ValueError("tournament arm model must be a mapping")
            models[TournamentArm(str(key))] = StandardizedRidge.from_mapping(raw_model)
        return cls(
            models=models,
            core_unit_ids=tuple(str(item) for item in _sequence(value["core_unit_ids"])),
            additional_unit_ids=tuple(
                str(item) for item in _sequence(value["additional_unit_ids"])
            ),
        )


def _validate_observations(rows: Sequence[TournamentObservation]) -> None:
    if not rows:
        raise ValueError("tournament observation collection cannot be empty")
    if len({row.observation_id for row in rows}) != len(rows):
        raise ValueError("tournament observation identifiers are not unique")
    base_dimensions = {len(row.baseline_features) for row in rows}
    access_dimensions = {len(row.access_features) for row in rows}
    if len(base_dimensions) != 1 or len(access_dimensions) != 1:
        raise ValueError("tournament feature dimensions vary across observations")


def _feature_matrix(rows: Sequence[TournamentObservation], arm: TournamentArm) -> FloatArray:
    if arm in {TournamentArm.BASE, TournamentArm.MORE_DATA, TournamentArm.RICHER_MODEL}:
        values = [row.baseline_features for row in rows]
    elif arm is TournamentArm.BETTER_ACCESS:
        values = [(*row.baseline_features, *row.access_features) for row in rows]
    else:
        values = [(*row.baseline_features, *row.sham_features) for row in rows]
    return np.asarray(values, dtype=np.float64)


def fit_tournament_models(
    rows: Sequence[TournamentObservation],
    *,
    core_unit_ids: Sequence[str],
    ridge_alpha: float = 1e-3,
) -> TournamentModelSet:
    """Fit all arms with unit-count matching fixed by the caller."""

    _validate_observations(rows)
    available = {row.unit_id for row in rows}
    core = tuple(sorted(set(core_unit_ids)))
    if len(core) != len(tuple(core_unit_ids)) or not set(core) <= available:
        raise ValueError("core development units are duplicated or unavailable")
    additional = tuple(sorted(available - set(core)))
    if len(core) < 2 or len(additional) < 2:
        raise ValueError("tournament needs at least two core and two additional units")
    core_rows = [row for row in rows if row.unit_id in set(core)]
    all_rows = list(rows)
    targets_core = np.asarray([row.target for row in core_rows], dtype=np.float64)
    targets_all = np.asarray([row.target for row in all_rows], dtype=np.float64)
    models = {
        TournamentArm.BASE: fit_standardized_ridge(
            _feature_matrix(core_rows, TournamentArm.BASE),
            targets_core,
            feature_map=FeatureMap.LINEAR,
            ridge_alpha=ridge_alpha,
        ),
        TournamentArm.MORE_DATA: fit_standardized_ridge(
            _feature_matrix(all_rows, TournamentArm.MORE_DATA),
            targets_all,
            feature_map=FeatureMap.LINEAR,
            ridge_alpha=ridge_alpha,
        ),
        TournamentArm.RICHER_MODEL: fit_standardized_ridge(
            _feature_matrix(core_rows, TournamentArm.RICHER_MODEL),
            targets_core,
            feature_map=FeatureMap.QUADRATIC,
            ridge_alpha=ridge_alpha,
        ),
        TournamentArm.BETTER_ACCESS: fit_standardized_ridge(
            _feature_matrix(core_rows, TournamentArm.BETTER_ACCESS),
            targets_core,
            feature_map=FeatureMap.LINEAR,
            ridge_alpha=ridge_alpha,
        ),
        TournamentArm.SHAM_ACCESS: fit_standardized_ridge(
            _feature_matrix(core_rows, TournamentArm.SHAM_ACCESS),
            targets_core,
            feature_map=FeatureMap.LINEAR,
            ridge_alpha=ridge_alpha,
        ),
    }
    return TournamentModelSet(models=models, core_unit_ids=core, additional_unit_ids=additional)


@dataclass(frozen=True, slots=True)
class PairedInterval:
    comparator: TournamentArm
    mean_comparator_minus_access: float
    lower: float
    upper: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "comparator": self.comparator.value,
            "mean_comparator_minus_access": self.mean_comparator_minus_access,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class TournamentEvaluation:
    disposition: TournamentDisposition
    unit_count: int
    observation_count: int
    unit_rmse: Mapping[TournamentArm, Mapping[str, float]]
    mean_unit_rmse: Mapping[TournamentArm, float]
    paired_intervals: tuple[PairedInterval, ...]
    rival_eliminated: bool
    response_present: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "unit_count": self.unit_count,
            "observation_count": self.observation_count,
            "unit_rmse": {
                arm.value: dict(sorted(self.unit_rmse[arm].items()))
                for arm in sorted(TournamentArm, key=lambda item: item.value)
            },
            "mean_unit_rmse": {
                arm.value: self.mean_unit_rmse[arm]
                for arm in sorted(TournamentArm, key=lambda item: item.value)
            },
            "paired_intervals": [value.to_mapping() for value in self.paired_intervals],
            "rival_eliminated": self.rival_eliminated,
            "response_present": self.response_present,
            "nested_observations_not_resampled": True,
        }


def predict_tournament(
    model_set: TournamentModelSet, rows: Sequence[TournamentObservation]
) -> Mapping[TournamentArm, FloatArray]:
    _validate_observations(rows)
    return {arm: model_set.models[arm].predict(_feature_matrix(rows, arm)) for arm in TournamentArm}


def _unit_rmse(rows: Sequence[TournamentObservation], predictions: FloatArray) -> dict[str, float]:
    if predictions.shape != (len(rows),):
        raise ValueError("prediction vector and tournament rows differ")
    grouped: dict[str, list[float]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        grouped.setdefault(row.unit_id, []).append((row.target - float(prediction)) ** 2)
    return {
        unit_id: float(np.sqrt(np.mean(np.asarray(errors, dtype=np.float64))))
        for unit_id, errors in sorted(grouped.items())
    }


def _paired_interval(
    comparator: TournamentArm,
    comparator_loss: Mapping[str, float],
    access_loss: Mapping[str, float],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> PairedInterval:
    if set(comparator_loss) != set(access_loss):
        raise ValueError("paired arm losses use different independent units")
    if not 0 < confidence_level < 1 or bootstrap_resamples < 100:
        raise ValueError("paired interval configuration is invalid")
    units = tuple(sorted(access_loss))
    differences = np.asarray(
        [comparator_loss[unit] - access_loss[unit] for unit in units], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(units), size=(bootstrap_resamples, len(units)))
    samples = np.mean(differences[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return PairedInterval(
        comparator=comparator,
        mean_comparator_minus_access=float(np.mean(differences)),
        lower=float(np.quantile(samples, tail)),
        upper=float(np.quantile(samples, 1.0 - tail)),
    )


def evaluate_tournament(
    model_set: TournamentModelSet,
    rows: Sequence[TournamentObservation],
    *,
    rival_eliminated: bool,
    response_present: bool,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 20_000,
    seed: int = 1,
) -> TournamentEvaluation:
    """Evaluate all arms with paired independent-unit resampling."""

    _validate_observations(rows)
    predictions = predict_tournament(model_set, rows)
    unit_losses = {arm: _unit_rmse(rows, predictions[arm]) for arm in TournamentArm}
    unit_ids = set(unit_losses[TournamentArm.BASE])
    if len(unit_ids) < 4:
        disposition = TournamentDisposition.UNEVALUABLE
        intervals: tuple[PairedInterval, ...] = ()
    else:
        comparators = tuple(arm for arm in TournamentArm if arm is not TournamentArm.BETTER_ACCESS)
        intervals = tuple(
            _paired_interval(
                arm,
                unit_losses[arm],
                unit_losses[TournamentArm.BETTER_ACCESS],
                confidence_level=confidence_level,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + index,
            )
            for index, arm in enumerate(comparators)
        )
        by_arm = {value.comparator: value for value in intervals}
        access_wins_required = all(
            by_arm[arm].lower > 0
            for arm in (
                TournamentArm.MORE_DATA,
                TournamentArm.RICHER_MODEL,
                TournamentArm.SHAM_ACCESS,
            )
        )
        access_mean = float(np.mean(list(unit_losses[TournamentArm.BETTER_ACCESS].values())))
        mean_losses = {
            arm: float(np.mean(list(unit_losses[arm].values()))) for arm in TournamentArm
        }
        scale_wins = any(
            by_arm[arm].upper < 0 for arm in (TournamentArm.MORE_DATA, TournamentArm.RICHER_MODEL)
        )
        sham_wins = by_arm[TournamentArm.SHAM_ACCESS].upper < 0
        if not response_present:
            disposition = TournamentDisposition.NO_RESPONSE_TO_SEPARATE
        elif access_wins_required and rival_eliminated:
            disposition = TournamentDisposition.ACCESS_DOMINANT
        elif scale_wins:
            disposition = TournamentDisposition.SCALE_DOMINANT
        elif sham_wins or mean_losses[TournamentArm.SHAM_ACCESS] < access_mean:
            disposition = TournamentDisposition.WRONG_ACCESS
        elif access_mean < min(
            mean_losses[TournamentArm.BASE],
            mean_losses[TournamentArm.MORE_DATA],
            mean_losses[TournamentArm.RICHER_MODEL],
            mean_losses[TournamentArm.SHAM_ACCESS],
        ):
            disposition = TournamentDisposition.PREDICTIVE_ACCESS_ONLY
        else:
            disposition = TournamentDisposition.ACCESS_AND_SCALE_EQUIVALENT
    means = {arm: float(np.mean(list(unit_losses[arm].values()))) for arm in TournamentArm}
    return TournamentEvaluation(
        disposition=disposition,
        unit_count=len(unit_ids),
        observation_count=len(rows),
        unit_rmse=unit_losses,
        mean_unit_rmse=means,
        paired_intervals=intervals,
        rival_eliminated=rival_eliminated,
        response_present=response_present,
    )


__all__ = [
    "FeatureMap",
    "PairedInterval",
    "StandardizedRidge",
    "TournamentArm",
    "TournamentDisposition",
    "TournamentEvaluation",
    "TournamentModelSet",
    "TournamentObservation",
    "TournamentPredictorInput",
    "evaluate_tournament",
    "fit_standardized_ridge",
    "fit_tournament_models",
    "predict_tournament",
]
