"""Parent-preparation inference for stationarity and preparation memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .basis import target_weight
from .receivers import CHART_NUMERIC_COORDINATES, CHART_ONE_HOT_COORDINATES, CONTINUOUS_COORDINATES, RECEIVER_COORDINATES, STATE_COORDINATES
from .schemas import QuantumTrajectoryReferenceValidationConfig
from .seed_commitments import memory_seed_census, stage_bootstrap_seed
from .types import CoordinateInterval, Denominator, MemoryResult, Stage, Validity


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    unit_index: int
    unit_id: str
    roster_id: str
    k_left: int
    boundary_occupation: int
    translation_orbit: int
    denominator: Denominator
    endpoint: float
    values: Mapping[str, float]
    energy: float
    event_rate: float
    state_sha256: str
    valid: bool = True
    reason_code: str = "OK"

    def __post_init__(self) -> None:
        if type(self.unit_index) is not int or self.unit_index < 0:
            raise ValueError("prepared observation requires its explicit nonnegative numerical parent order")


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    denominator: Denominator
    comparison_id: str
    earlier: float
    later: float


def _ordered_records(
    records: Sequence[PreparedObservation],
    denominator: Denominator,
    endpoint: float,
) -> tuple[PreparedObservation, ...]:
    selected = tuple(
        sorted(
            (row for row in records if row.denominator == denominator and row.endpoint == endpoint),
            key=lambda row: row.unit_index,
        )
    )
    if not selected:
        raise ValueError(
            f"no observations for denominator={denominator.value}, endpoint={endpoint}"
        )
    if len({row.unit_id for row in selected}) != len(selected):
        raise ValueError("observation unit identities are duplicated")
    if len({row.unit_index for row in selected}) != len(selected):
        raise ValueError("observation numerical parent orders are duplicated")
    return selected


def _matrix(rows: Sequence[PreparedObservation]) -> np.ndarray:
    if any(tuple(row.values) != CONTINUOUS_COORDINATES for row in rows):
        raise ValueError("observation coordinate order differs from quantum trajectory reference validation")
    result = np.asarray(
        [[row.values[name] for name in CONTINUOUS_COORDINATES] for row in rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(result)):
        raise ValueError("observation matrix contains non-finite values")
    return result


def _unit_weights(rows: Sequence[PreparedObservation]) -> np.ndarray:
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.k_left] = counts.get(row.k_left, 0) + 1
    l_sites = 12
    particles = 6
    weights = np.asarray(
        [target_weight(l_sites, particles, row.k_left) / counts[row.k_left] for row in rows],
        dtype=np.float64,
    )
    if not np.isclose(weights.sum(), 1.0, atol=1e-14):
        raise AssertionError("target preparation weights do not sum to one")
    return weights


def _moments(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ matrix
    variance = weights @ ((matrix - mean) ** 2)
    return np.asarray(mean), np.asarray(variance)


def _floors(config: QuantumTrajectoryReferenceValidationConfig, denominator: Denominator) -> np.ndarray:
    result: list[float] = []
    elapsed_names = {
        "chart_time_since_last_event",
        "chart_time_since_last_boundary_event",
    }
    trig_names = {"chart_sin_last_site", "chart_cos_last_site"}
    half_name = "state_half_occupation"
    for name in CONTINUOUS_COORDINATES:
        if name in RECEIVER_COORDINATES:
            result.append(float(config.count_floor))
        elif name in CHART_ONE_HOT_COORDINATES:
            result.append(float(config.one_hot_floor))
        elif name in elapsed_names:
            result.append(
                float(config.elapsed_floor_factor) / (denominator.gamma * config.n_primary)
            )
        elif name in trig_names:
            result.append(float(config.trig_floor))
        elif name in CHART_NUMERIC_COORDINATES:
            result.append(float(config.count_floor))
        elif name == half_name:
            result.append(float(config.half_occupation_floor))
        elif name in STATE_COORDINATES:
            result.append(float(config.local_observable_floor))
        else:
            raise AssertionError(f"no scale floor for {name}")
    return np.asarray(result, dtype=np.float64)


def _resample_positions(
    rows: Sequence[PreparedObservation],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    by_k: dict[int, np.ndarray] = {}
    for k_left in sorted({row.k_left for row in rows}):
        by_k[k_left] = np.asarray(
            [index for index, row in enumerate(rows) if row.k_left == k_left],
            dtype=np.int64,
        )
    rng = np.random.Generator(np.random.Philox(seed))
    sampled = np.empty((draws, len(rows)), dtype=np.int64)
    offset = 0
    for k_left in sorted(by_k):
        positions = by_k[k_left]
        sampled[:, offset : offset + len(positions)] = rng.choice(
            positions,
            size=(draws, len(positions)),
            replace=True,
        )
        offset += len(positions)
    return sampled


def _bootstrap_comparisons(
    records: Sequence[PreparedObservation],
    comparisons: Sequence[ComparisonSpec],
    *,
    config: QuantumTrajectoryReferenceValidationConfig,
    stage: Stage,
) -> tuple[
    list[tuple[ComparisonSpec, tuple[PreparedObservation, ...], np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    prepared: list[
        tuple[ComparisonSpec, tuple[PreparedObservation, ...], np.ndarray, np.ndarray]
    ] = []
    reference_units: tuple[str, ...] | None = None
    for spec in comparisons:
        earlier_rows = _ordered_records(records, spec.denominator, spec.earlier)
        later_rows = _ordered_records(records, spec.denominator, spec.later)
        earlier_ids = tuple(row.unit_id for row in earlier_rows)
        later_ids = tuple(row.unit_id for row in later_rows)
        if earlier_ids != later_ids:
            raise ValueError("paired clock observations differ in parent units")
        if reference_units is None:
            reference_units = earlier_ids
        elif earlier_ids != reference_units:
            raise ValueError("simultaneous comparison families differ in parent rosters")
        prepared.append((spec, earlier_rows, _matrix(earlier_rows), _matrix(later_rows)))
    if not prepared:
        raise ValueError("comparison family is empty")
    base_rows = prepared[0][1]
    sampled = _resample_positions(
        base_rows,
        draws=config.bootstrap_draws,
        seed=stage_bootstrap_seed(stage),
    )
    row_count = len(prepared)
    coordinate_count = len(CONTINUOUS_COORDINATES)
    boot_delta = np.empty(
        (config.bootstrap_draws, row_count, coordinate_count),
        dtype=np.float64,
    )
    boot_variance_delta = np.empty_like(boot_delta)
    boot_energy_delta = np.empty((config.bootstrap_draws, row_count), dtype=np.float64)
    boot_rate_delta = np.empty_like(boot_energy_delta)
    batch_size = 128
    strata = sorted({row.k_left for row in base_rows})
    stratum_positions = {
        k_left: np.asarray(
            [index for index, row in enumerate(base_rows) if row.k_left == k_left],
            dtype=np.int64,
        )
        for k_left in strata
    }
    stratum_offsets: dict[int, slice] = {}
    offset = 0
    for k_left in strata:
        count = len(stratum_positions[k_left])
        stratum_offsets[k_left] = slice(offset, offset + count)
        offset += count
    for start in range(0, config.bootstrap_draws, batch_size):
        stop = min(start + batch_size, config.bootstrap_draws)
        selected = sampled[start:stop]
        for row_index, (_, earlier_rows, earlier, later) in enumerate(prepared):
            earlier_mean = np.zeros((stop - start, coordinate_count))
            later_mean = np.zeros_like(earlier_mean)
            earlier_second = np.zeros_like(earlier_mean)
            later_second = np.zeros_like(earlier_mean)
            earlier_energy = np.asarray([row.energy for row in earlier_rows])
            later_energy = np.asarray(
                [
                    row.energy
                    for row in _ordered_records(
                        records,
                        prepared[row_index][0].denominator,
                        prepared[row_index][0].later,
                    )
                ]
            )
            earlier_rate = np.asarray([row.event_rate for row in earlier_rows])
            later_rate = np.asarray(
                [
                    row.event_rate
                    for row in _ordered_records(
                        records,
                        prepared[row_index][0].denominator,
                        prepared[row_index][0].later,
                    )
                ]
            )
            energy_early_mean = np.zeros(stop - start)
            energy_late_mean = np.zeros(stop - start)
            rate_early_mean = np.zeros(stop - start)
            rate_late_mean = np.zeros(stop - start)
            for k_left in strata:
                choice = selected[:, stratum_offsets[k_left]]
                target = target_weight(12, 6, k_left)
                early_selected = earlier[choice]
                late_selected = later[choice]
                earlier_mean += target * early_selected.mean(axis=1)
                later_mean += target * late_selected.mean(axis=1)
                earlier_second += target * (early_selected**2).mean(axis=1)
                later_second += target * (late_selected**2).mean(axis=1)
                energy_early_mean += target * earlier_energy[choice].mean(axis=1)
                energy_late_mean += target * later_energy[choice].mean(axis=1)
                rate_early_mean += target * earlier_rate[choice].mean(axis=1)
                rate_late_mean += target * later_rate[choice].mean(axis=1)
            boot_delta[start:stop, row_index] = later_mean - earlier_mean
            boot_variance_delta[start:stop, row_index] = (
                later_second - later_mean**2 - (earlier_second - earlier_mean**2)
            )
            boot_energy_delta[start:stop, row_index] = energy_late_mean - energy_early_mean
            boot_rate_delta[start:stop, row_index] = rate_late_mean - rate_early_mean
    return prepared, boot_delta, boot_variance_delta, boot_energy_delta, boot_rate_delta


def stationarity_intervals(
    records: Sequence[PreparedObservation],
    comparisons: Sequence[ComparisonSpec],
    *,
    config: QuantumTrajectoryReferenceValidationConfig,
    stage: Stage,
) -> tuple[CoordinateInterval, ...]:
    (
        prepared,
        boot_delta,
        boot_variance_delta,
        boot_energy_delta,
        boot_rate_delta,
    ) = _bootstrap_comparisons(records, comparisons, config=config, stage=stage)
    row_count = len(prepared)
    coordinate_count = len(CONTINUOUS_COORDINATES)
    observed_delta = np.empty((row_count, coordinate_count))
    observed_variance_delta = np.empty_like(observed_delta)
    observed_scale = np.empty_like(observed_delta)
    observed_variance_scale = np.empty_like(observed_delta)
    observed_energy = np.empty(row_count)
    observed_rate = np.empty(row_count)
    observed_rate_scale = np.empty(row_count)
    for row_index, (spec, earlier_rows, earlier, later) in enumerate(prepared):
        weights = _unit_weights(earlier_rows)
        earlier_mean, earlier_variance = _moments(earlier, weights)
        later_mean, later_variance = _moments(later, weights)
        floors = _floors(config, spec.denominator)
        observed_delta[row_index] = later_mean - earlier_mean
        observed_variance_delta[row_index] = later_variance - earlier_variance
        observed_scale[row_index] = np.maximum(
            np.sqrt(0.5 * (earlier_variance + later_variance)),
            floors,
        )
        observed_variance_scale[row_index] = np.maximum(
            0.5 * (earlier_variance + later_variance),
            floors**2,
        )
        later_rows = _ordered_records(records, spec.denominator, spec.later)
        earlier_energy = np.asarray([row.energy for row in earlier_rows])
        later_energy = np.asarray([row.energy for row in later_rows])
        earlier_rate = np.asarray([row.event_rate for row in earlier_rows])
        later_rate = np.asarray([row.event_rate for row in later_rows])
        observed_energy[row_index] = weights @ (later_energy - earlier_energy)
        observed_rate[row_index] = weights @ (later_rate - earlier_rate)
        observed_rate_scale[row_index] = max(
            0.5 * (weights @ earlier_rate + weights @ later_rate),
            1e-12,
        )
    mean_se = boot_delta.std(axis=0, ddof=1)
    variance_se = boot_variance_delta.std(axis=0, ddof=1)
    energy_se = boot_energy_delta.std(axis=0, ddof=1)
    rate_se = boot_rate_delta.std(axis=0, ddof=1)

    def critical(
        boot: np.ndarray,
        observed: np.ndarray,
        se: np.ndarray,
    ) -> float:
        denominator = np.maximum(se, 1e-15)
        t_values = np.abs((boot - observed) / denominator)
        max_t = np.max(t_values.reshape((len(t_values), -1)), axis=1)
        return float(
            np.quantile(
                max_t,
                1.0 - float(config.family_alpha),
                method="higher",
            )
        )

    mean_critical = critical(boot_delta, observed_delta, mean_se)
    variance_critical = critical(
        boot_variance_delta,
        observed_variance_delta,
        variance_se,
    )
    energy_critical = critical(boot_energy_delta, observed_energy, energy_se)
    rate_critical = critical(boot_rate_delta, observed_rate, rate_se)
    results: list[CoordinateInterval] = []
    for row_index, (spec, earlier_rows, _, _) in enumerate(prepared):
        valid = all(row.valid for row in earlier_rows) and all(
            row.valid for row in _ordered_records(records, spec.denominator, spec.later)
        )
        for coordinate_index, coordinate in enumerate(CONTINUOUS_COORDINATES):
            upper = (
                abs(observed_delta[row_index, coordinate_index])
                + mean_critical * mean_se[row_index, coordinate_index]
            ) / observed_scale[row_index, coordinate_index]
            results.append(
                CoordinateInterval(
                    stage=stage,
                    denominator=spec.denominator,
                    comparison_id=spec.comparison_id,
                    family="mean_standardized",
                    coordinate=coordinate,
                    estimate=observed_delta[row_index, coordinate_index],
                    standard_error=mean_se[row_index, coordinate_index],
                    scale=observed_scale[row_index, coordinate_index],
                    critical_value=mean_critical,
                    upper_bound=upper,
                    margin=float(config.mean_margin),
                    passed=bool(valid and upper <= float(config.mean_margin)),
                    validity=Validity.VALID if valid else Validity.UNEVALUABLE,
                    reason_code="OK" if valid else "INVALID_PARENT_OBSERVATION",
                )
            )
            variance_upper = (
                abs(observed_variance_delta[row_index, coordinate_index])
                + variance_critical * variance_se[row_index, coordinate_index]
            ) / observed_variance_scale[row_index, coordinate_index]
            results.append(
                CoordinateInterval(
                    stage=stage,
                    denominator=spec.denominator,
                    comparison_id=spec.comparison_id,
                    family="variance_standardized",
                    coordinate=coordinate,
                    estimate=observed_variance_delta[row_index, coordinate_index],
                    standard_error=variance_se[row_index, coordinate_index],
                    scale=observed_variance_scale[row_index, coordinate_index],
                    critical_value=variance_critical,
                    upper_bound=variance_upper,
                    margin=float(config.variance_margin),
                    passed=bool(valid and variance_upper <= float(config.variance_margin)),
                    validity=Validity.VALID if valid else Validity.UNEVALUABLE,
                    reason_code="OK" if valid else "INVALID_PARENT_OBSERVATION",
                )
            )
        energy_upper = abs(observed_energy[row_index]) + energy_critical * energy_se[row_index]
        results.append(
            CoordinateInterval(
                stage=stage,
                denominator=spec.denominator,
                comparison_id=spec.comparison_id,
                family="energy_native",
                coordinate="terminal_energy",
                estimate=observed_energy[row_index],
                standard_error=energy_se[row_index],
                scale=1.0,
                critical_value=energy_critical,
                upper_bound=energy_upper,
                margin=float(config.energy_margin),
                passed=bool(valid and energy_upper <= float(config.energy_margin)),
                validity=Validity.VALID if valid else Validity.UNEVALUABLE,
                reason_code="OK" if valid else "INVALID_PARENT_OBSERVATION",
            )
        )
        rate_upper = (
            abs(observed_rate[row_index]) + rate_critical * rate_se[row_index]
        ) / observed_rate_scale[row_index]
        results.append(
            CoordinateInterval(
                stage=stage,
                denominator=spec.denominator,
                comparison_id=spec.comparison_id,
                family="event_rate_relative",
                coordinate="event_rate",
                estimate=observed_rate[row_index],
                standard_error=rate_se[row_index],
                scale=observed_rate_scale[row_index],
                critical_value=rate_critical,
                upper_bound=rate_upper,
                margin=float(config.event_rate_margin),
                passed=bool(valid and rate_upper <= float(config.event_rate_margin)),
                validity=Validity.VALID if valid else Validity.UNEVALUABLE,
                reason_code="OK" if valid else "INVALID_PARENT_OBSERVATION",
            )
        )
    return tuple(results)


def _eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    overall = weights @ values
    total = weights @ ((values - overall) ** 2)
    between = np.zeros(values.shape[1])
    for label in np.unique(labels):
        selected = labels == label
        group_weight = float(weights[selected].sum())
        if group_weight <= 0.0:
            continue
        group_mean = (weights[selected] @ values[selected]) / group_weight
        between += group_weight * (group_mean - overall) ** 2
    return np.asarray(
        np.divide(
            between,
            total,
            out=np.zeros_like(between),
            where=total > 1e-15,
        ),
        dtype=np.float64,
    )


def preparation_memory(
    records: Sequence[PreparedObservation],
    *,
    denominator: Denominator,
    endpoint: float,
    clock_id: str,
    config: QuantumTrajectoryReferenceValidationConfig,
    stage: Stage,
    scientific_seeds: tuple[tuple[str, int], ...] | None = None,
) -> tuple[MemoryResult, ...]:
    seeds = dict(memory_seed_census(stage, denominator, clock_id, scientific_seeds))
    rows = _ordered_records(records, denominator, endpoint)
    values = _matrix(rows)
    weights = _unit_weights(rows)
    factors = {
        "initial_k": np.asarray([row.k_left for row in rows], dtype=np.int64),
        "boundary_occupation": np.asarray(
            [row.boundary_occupation for row in rows],
            dtype=np.int64,
        ),
        "translation_orbit": np.asarray(
            [row.translation_orbit for row in rows],
            dtype=np.int64,
        ),
    }
    results: list[MemoryResult] = []
    for factor, labels in factors.items():
        observed = _eta_squared(values, labels, weights)
        invariant = len(np.unique(labels)) == len(labels)
        if invariant:
            p_values = np.ones(values.shape[1])
        else:
            rng = np.random.Generator(
                np.random.Philox(
                    seeds[factor]
                )
            )
            permuted_max = np.empty(config.permutation_draws)
            for draw in range(config.permutation_draws):
                permuted = rng.permutation(labels)
                permuted_max[draw] = float(np.max(_eta_squared(values, permuted, weights)))
            p_values = np.asarray(
                [
                    (1.0 + np.sum(permuted_max >= value)) / (config.permutation_draws + 1.0)
                    for value in observed
                ]
            )
        for coordinate, eta, p_value in zip(
            CONTINUOUS_COORDINATES,
            observed,
            p_values,
            strict=True,
        ):
            material = bool(
                eta >= float(config.memory_eta_squared)
                and p_value <= float(config.family_alpha) / 3.0
            )
            results.append(
                MemoryResult(
                    stage=stage,
                    denominator=denominator,
                    clock_id=clock_id,
                    factor=factor,
                    coordinate=coordinate,
                    eta_squared=float(eta),
                    adjusted_p_value=float(p_value),
                    material=material,
                    invariant=invariant,
                    validity=(
                        Validity.UNEVALUABLE
                        if invariant and factor == "translation_orbit"
                        else Validity.VALID
                    ),
                )
            )
    return tuple(results)


def intervals_by_denominator(
    intervals: Sequence[CoordinateInterval],
) -> dict[Denominator, tuple[CoordinateInterval, ...]]:
    return {
        denominator: tuple(row for row in intervals if row.denominator == denominator)
        for denominator in Denominator
    }


__all__ = [
    "ComparisonSpec",
    "PreparedObservation",
    "intervals_by_denominator",
    "preparation_memory",
    "stationarity_intervals",
]
