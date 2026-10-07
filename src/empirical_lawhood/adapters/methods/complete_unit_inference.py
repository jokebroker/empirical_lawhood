"""Prospective, complete-unit simultaneous inference for independent substrate grounding targets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class CompleteUnitInferenceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/complete-unit-inference-config'

    config_id: str
    family_id: str
    confidence_level: Decimal
    bootstrap_replicates: int
    seed: int
    chunk_size: int
    minimum_complete_units: int
    resampling_unit: str
    stratification_frozen: bool
    simultaneous_family_frozen: bool
    frozen_before_outcomes: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.family_id, field_name="family_id")
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal("0.5") < self.confidence_level < Decimal(1):
            raise ValueError("complete-unit confidence level differs")
        if self.bootstrap_replicates < 200:
            raise ValueError("complete-unit inference requires at least 200 replicates")
        if self.seed < 0 or self.chunk_size <= 0 or self.minimum_complete_units < 2:
            raise ValueError("complete-unit inference resource/count config differs")
        if self.resampling_unit != "COMPLETE_UNIT":
            raise ValueError("nested observations cannot become resampling units")
        if (
            not all(
                (
                    self.stratification_frozen,
                    self.simultaneous_family_frozen,
                    self.frozen_before_outcomes,
                )
            )
            or self.protected_outcome_access_count
        ):
            raise ValueError("complete-unit inference crossed its prospective freeze")


@dataclass(frozen=True, slots=True)
class CompleteUnitVector(CanonicalRecord):
    """One physical/numerical preparation and its fixed nested estimand vector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/complete-unit-vector'

    unit_id: str
    stratum_id: str
    estimands: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        require_sorted_unique_ids(
            self.estimands,
            attribute="value_id",
            field_name="estimands",
        )
        if not self.estimands:
            raise ValueError("complete-unit vector requires estimands")


@dataclass(frozen=True, slots=True)
class CompleteUnitStratumCount(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/complete-unit-stratum-count'

    stratum_id: str
    complete_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        if self.complete_unit_count <= 0:
            raise ValueError("complete-unit stratum count must be positive")


@dataclass(frozen=True, slots=True)
class CompleteUnitSimultaneousInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/complete-unit-simultaneous-interval'

    interval_id: str
    estimand_id: str
    native_unit: str
    point: Decimal
    lower: Decimal
    upper: Decimal
    standard_error: Decimal
    critical_value: Decimal
    complete_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_id, field_name="interval_id")
        validate_stable_id(self.estimand_id, field_name="estimand_id")
        for name in (
            "point",
            "lower",
            "upper",
            "standard_error",
            "critical_value",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.lower > self.point or self.point > self.upper:
            raise ValueError("complete-unit interval does not contain its point")
        if self.standard_error < 0 or self.critical_value < 0:
            raise ValueError("complete-unit uncertainty values must be nonnegative")
        if self.complete_unit_count < 2:
            raise ValueError("complete-unit interval requires at least two units")


@dataclass(frozen=True, slots=True)
class CompleteUnitInferenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/complete-unit-inference-result'

    result_id: str
    config: ObjectIdentity
    complete_unit_ids_sha256: str
    stratification_sha256: str
    complete_unit_count: int
    stratum_counts: tuple[CompleteUnitStratumCount, ...]
    intervals: tuple[CompleteUnitSimultaneousInterval, ...]
    rng_family: str
    draw_index_sha256: str
    nested_observations_resampled: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != CompleteUnitInferenceConfig.SCHEMA:
            raise ValueError("complete-unit inference config identity differs")
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        validate_sha256(
            self.stratification_sha256,
            field_name="stratification_sha256",
        )
        validate_sha256(self.draw_index_sha256, field_name="draw_index_sha256")
        require_sorted_unique_ids(
            self.stratum_counts,
            attribute="stratum_id",
            field_name="stratum_counts",
        )
        require_sorted_unique_ids(
            self.intervals,
            attribute="interval_id",
            field_name="intervals",
        )
        if self.complete_unit_count != sum(
            value.complete_unit_count for value in self.stratum_counts
        ):
            raise ValueError("complete-unit count differs from frozen strata")
        if any(value.complete_unit_count != self.complete_unit_count for value in self.intervals):
            raise ValueError("simultaneous intervals differ on complete-unit count")
        if self.rng_family != "numpy.random.PCG64":
            raise ValueError("complete-unit inference RNG family differs")
        if self.nested_observations_resampled:
            raise ValueError("nested observations cannot inflate inference")


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("complete-unit inference produced a non-finite value")
    return Decimal(format(float(value), ".17g"))


def complete_unit_simultaneous_inference(
    *,
    result_id: str,
    config: CompleteUnitInferenceConfig,
    units: tuple[CompleteUnitVector, ...],
) -> CompleteUnitInferenceResult:
    """Resample whole units within frozen strata and form one max-t family."""

    ordered = tuple(sorted(units, key=lambda value: value.unit_id))
    if len(ordered) < config.minimum_complete_units:
        raise ValueError("complete-unit panel is below its frozen minimum")
    if len({value.unit_id for value in ordered}) != len(ordered):
        raise ValueError("complete-unit identities repeat")
    coordinates = tuple((value.value_id, value.unit) for value in ordered[0].estimands)
    if any(
        tuple((value.value_id, value.unit) for value in unit.estimands) != coordinates
        for unit in ordered
    ):
        raise ValueError("complete-unit estimand coordinates differ")
    matrix = np.asarray(
        [[float(value.value) for value in unit.estimands] for unit in ordered],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("complete-unit estimand matrix differs")
    strata = {
        stratum_id: np.asarray(
            [index for index, unit in enumerate(ordered) if unit.stratum_id == stratum_id],
            dtype=np.int64,
        )
        for stratum_id in sorted({value.stratum_id for value in ordered})
    }
    generator = np.random.Generator(np.random.PCG64(config.seed))
    bootstrap_means = np.empty(
        (config.bootstrap_replicates, matrix.shape[1]),
        dtype=np.float64,
    )
    digest = sha256()
    offset = 0
    while offset < config.bootstrap_replicates:
        count = min(config.chunk_size, config.bootstrap_replicates - offset)
        sampled_chunks = tuple(
            generator.choice(indices, size=(count, len(indices)), replace=True)
            for indices in strata.values()
        )
        sampled = np.concatenate(sampled_chunks, axis=1)
        digest.update(np.ascontiguousarray(sampled.astype("<i8", copy=False)).tobytes())
        bootstrap_means[offset : offset + count] = np.mean(matrix[sampled], axis=1)
        offset += count
    center = np.mean(matrix, axis=0)
    standard_error = np.std(bootstrap_means, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, 1.0)
    standardized = np.abs((bootstrap_means - center) / safe)
    standardized[:, standard_error == 0] = 0.0
    maxima = np.max(standardized, axis=1)
    critical = float(
        np.quantile(
            maxima,
            float(config.confidence_level),
            method="higher",
        )
    )
    half_width = critical * standard_error
    intervals = tuple(
        CompleteUnitSimultaneousInterval(
            interval_id=f"{result_id}.interval.{estimand_id}",
            estimand_id=estimand_id,
            native_unit=native_unit,
            point=_decimal(center[index]),
            lower=_decimal(center[index] - half_width[index]),
            upper=_decimal(center[index] + half_width[index]),
            standard_error=_decimal(standard_error[index]),
            critical_value=_decimal(critical),
            complete_unit_count=len(ordered),
        )
        for index, (estimand_id, native_unit) in enumerate(coordinates)
    )
    unit_digest = sha256(
        ("\n".join(value.unit_id for value in ordered) + "\n").encode("ascii")
    ).hexdigest()
    stratification_digest = sha256(
        ("\n".join(f"{value.unit_id}\0{value.stratum_id}" for value in ordered) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    return CompleteUnitInferenceResult(
        result_id=result_id,
        config=ObjectIdentity.from_record(config.config_id, config),
        complete_unit_ids_sha256=unit_digest,
        stratification_sha256=stratification_digest,
        complete_unit_count=len(ordered),
        stratum_counts=tuple(
            CompleteUnitStratumCount(
                stratum_id=stratum_id,
                complete_unit_count=len(indices),
            )
            for stratum_id, indices in strata.items()
        ),
        intervals=intervals,
        rng_family="numpy.random.PCG64",
        draw_index_sha256=digest.hexdigest(),
        nested_observations_resampled=False,
    )


__all__ = [
    'CompleteUnitInferenceConfig',
    'CompleteUnitInferenceResult',
    'CompleteUnitSimultaneousInterval',
    'CompleteUnitStratumCount',
    'CompleteUnitVector',
    "complete_unit_simultaneous_inference",
]
