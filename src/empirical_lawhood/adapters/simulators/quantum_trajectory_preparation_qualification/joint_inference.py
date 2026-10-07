"""Joint nested-prefix preparation recurrence inference for quantum trajectory preparation qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .basis import target_weight
from .receivers import CHART_NUMERIC_COORDINATES, CHART_ONE_HOT_COORDINATES, CONTINUOUS_COORDINATES, RECEIVER_COORDINATES, STATE_COORDINATES
from .schemas import QuantumTrajectoryPreparationQualificationConfig
from .seed_commitments import stage_inference_seed
from .types import CoordinateInterval, Denominator, MemoryResult, Stage, Validity


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    unit_id: str
    unit_index: int
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


@dataclass(frozen=True, slots=True)
class LookSpec:
    candidate: float
    sample_size: int
    comparison_id: str
    earlier: float
    later: float


def comparison_specs(
    candidates: Sequence[float],
    sample_sizes: Sequence[int],
    extension: float,
) -> tuple[LookSpec, ...]:
    return tuple(
        spec
        for candidate in candidates
        for sample_size in sample_sizes
        for spec in (
            LookSpec(
                candidate,
                sample_size,
                f"candidate-{candidate:g}-n{sample_size}",
                candidate,
                candidate + extension,
            ),
            LookSpec(
                candidate,
                sample_size,
                f"sentinel-{2.0 * candidate:g}-n{sample_size}",
                2.0 * candidate,
                2.0 * candidate + extension,
            ),
        )
    )


def _ordered_records(
    records: Sequence[PreparedObservation],
    endpoint: float,
) -> tuple[PreparedObservation, ...]:
    selected = tuple(
        sorted(
            (
                row
                for row in records
                if row.denominator == Denominator.STRONG and row.endpoint == endpoint
            ),
            key=lambda row: row.unit_index,
        )
    )
    if not selected:
        raise ValueError(f"no strong observations at endpoint {endpoint}")
    if [row.unit_index for row in selected] != list(range(len(selected))):
        raise ValueError("observations do not form one contiguous parent prefix")
    if len({row.unit_id for row in selected}) != len(selected):
        raise ValueError("observation parent identities are duplicated")
    return selected


def _arrays(
    records: Sequence[PreparedObservation],
    endpoints: Sequence[float],
) -> tuple[
    tuple[PreparedObservation, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows_by_endpoint = [_ordered_records(records, endpoint) for endpoint in endpoints]
    base = rows_by_endpoint[0]
    base_ids = tuple(row.unit_id for row in base)
    if any(tuple(row.unit_id for row in rows) != base_ids for rows in rows_by_endpoint):
        raise ValueError("paired endpoint observations differ in parent units")
    if any(
        tuple(row.values) != CONTINUOUS_COORDINATES for rows in rows_by_endpoint for row in rows
    ):
        raise ValueError("observation coordinate order differs from the frozen 49-vector")
    values = np.asarray(
        [
            [[row.values[name] for name in CONTINUOUS_COORDINATES] for row in rows]
            for rows in rows_by_endpoint
        ],
        dtype=np.float64,
    ).transpose(1, 0, 2)
    energy = np.asarray(
        [[row.energy for row in rows] for rows in rows_by_endpoint],
        dtype=np.float64,
    ).T
    rate = np.asarray(
        [[row.event_rate for row in rows] for rows in rows_by_endpoint],
        dtype=np.float64,
    ).T
    valid = np.asarray(
        [[row.valid for row in rows] for rows in rows_by_endpoint],
        dtype=bool,
    ).T
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(energy)) or np.any(~np.isfinite(rate)):
        raise ValueError("observation arrays contain non-finite values")
    return base, values, energy, rate, valid


def _floors(config: QuantumTrajectoryPreparationQualificationConfig) -> np.ndarray:
    result: list[float] = []
    elapsed_names = {
        "chart_time_since_last_event",
        "chart_time_since_last_boundary_event",
    }
    trig_names = {"chart_sin_last_site", "chart_cos_last_site"}
    for name in CONTINUOUS_COORDINATES:
        if name in RECEIVER_COORDINATES:
            result.append(float(config.count_floor))
        elif name in CHART_ONE_HOT_COORDINATES:
            result.append(float(config.one_hot_floor))
        elif name in elapsed_names:
            result.append(
                float(config.elapsed_floor_factor) / (Denominator.STRONG.gamma * config.n_primary)
            )
        elif name in trig_names:
            result.append(float(config.trig_floor))
        elif name in CHART_NUMERIC_COORDINATES:
            result.append(float(config.count_floor))
        elif name == "state_half_occupation":
            result.append(float(config.half_occupation_floor))
        elif name in STATE_COORDINATES:
            result.append(float(config.local_observable_floor))
        else:
            raise AssertionError(f"no frozen scale floor for {name}")
    return np.asarray(result)


def _weights(rows: Sequence[PreparedObservation]) -> np.ndarray:
    counts = {k_left: sum(row.k_left == k_left for row in rows) for k_left in range(7)}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("a required preparation stratum is empty")
    result = np.asarray(
        [target_weight(12, 6, row.k_left) / counts[row.k_left] for row in rows],
        dtype=np.float64,
    )
    if not np.isclose(result.sum(), 1.0, atol=1e-14):
        raise AssertionError("target preparation weights do not sum to one")
    return result


def _counts_from_uniforms(uniforms: np.ndarray, size: int) -> np.ndarray:
    batch = uniforms.shape[0]
    choices = np.floor(uniforms[:, :size] * size).astype(np.int64)
    combined = np.arange(batch, dtype=np.int64)[:, None] * size + choices
    return np.bincount(combined.ravel(), minlength=batch * size).reshape(batch, size)


def _joint_bootstrap(
    base: Sequence[PreparedObservation],
    values: np.ndarray,
    energy: np.ndarray,
    rate: np.ndarray,
    endpoints: Sequence[float],
    looks: Sequence[LookSpec],
    *,
    config: QuantumTrajectoryPreparationQualificationConfig,
    stage: Stage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    endpoint_index = {endpoint: index for index, endpoint in enumerate(endpoints)}
    draws = config.bootstrap_draws
    coordinate_count = len(CONTINUOUS_COORDINATES)
    boot_mean = np.empty((draws, len(looks), coordinate_count), dtype=np.float64)
    boot_variance = np.empty_like(boot_mean)
    boot_energy = np.empty((draws, len(looks)), dtype=np.float64)
    boot_rate = np.empty_like(boot_energy)
    prefix_positions = {
        sample_size: {
            k_left: np.asarray(
                [index for index, row in enumerate(base[:sample_size]) if row.k_left == k_left],
                dtype=np.int64,
            )
            for k_left in range(7)
        }
        for sample_size in sorted({look.sample_size for look in looks})
    }
    maximum_counts = {
        k_left: max(len(prefix_positions[sample_size][k_left]) for sample_size in prefix_positions)
        for k_left in range(7)
    }
    rng = np.random.Generator(np.random.Philox(stage_inference_seed(stage, "joint-grid-bootstrap")))
    batch_size = 32
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        batch = stop - start
        common_uniforms = {
            k_left: rng.random((batch, maximum_counts[k_left])) for k_left in range(7)
        }
        moments: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for sample_size, by_k in prefix_positions.items():
            mean = np.zeros((batch, len(endpoints), coordinate_count))
            second = np.zeros_like(mean)
            energy_mean = np.zeros((batch, len(endpoints)))
            rate_mean = np.zeros_like(energy_mean)
            for k_left, positions in by_k.items():
                counts = _counts_from_uniforms(
                    common_uniforms[k_left],
                    len(positions),
                )
                normalized = counts / len(positions)
                target = target_weight(12, 6, k_left)
                panel = values[positions]
                mean += target * np.einsum("bn,nec->bec", normalized, panel)
                second += target * np.einsum(
                    "bn,nec->bec",
                    normalized,
                    panel**2,
                )
                energy_mean += target * (normalized @ energy[positions])
                rate_mean += target * (normalized @ rate[positions])
            moments[sample_size] = (
                mean,
                second - mean**2,
                energy_mean,
                rate_mean,
            )
        for look_index, look in enumerate(looks):
            mean, variance, energy_mean, rate_mean = moments[look.sample_size]
            earlier = endpoint_index[look.earlier]
            later = endpoint_index[look.later]
            boot_mean[start:stop, look_index] = mean[:, later] - mean[:, earlier]
            boot_variance[start:stop, look_index] = variance[:, later] - variance[:, earlier]
            boot_energy[start:stop, look_index] = energy_mean[:, later] - energy_mean[:, earlier]
            boot_rate[start:stop, look_index] = rate_mean[:, later] - rate_mean[:, earlier]
    return boot_mean, boot_variance, boot_energy, boot_rate


def _critical(
    bootstrap: np.ndarray,
    observed: np.ndarray,
    standard_error: np.ndarray,
    alpha: float,
) -> float:
    denominator = np.maximum(standard_error, 1e-15)
    maximum = np.max(
        np.abs((bootstrap - observed) / denominator).reshape((len(bootstrap), -1)),
        axis=1,
    )
    return float(np.quantile(maximum, 1.0 - alpha, method="higher"))


def stationarity_intervals(
    records: Sequence[PreparedObservation],
    *,
    candidates: Sequence[float],
    sample_sizes: Sequence[int],
    config: QuantumTrajectoryPreparationQualificationConfig,
    stage: Stage,
) -> tuple[CoordinateInterval, ...]:
    looks = comparison_specs(candidates, sample_sizes, float(config.burnin_extension))
    endpoints = tuple(
        sorted({endpoint for look in looks for endpoint in (look.earlier, look.later)})
    )
    base, values, energy, rate, valid = _arrays(records, endpoints)
    maximum_n = max(sample_sizes)
    if len(base) != maximum_n:
        raise ValueError("observation roster size differs from the requested look family")
    endpoint_index = {endpoint: index for index, endpoint in enumerate(endpoints)}
    coordinate_count = len(CONTINUOUS_COORDINATES)
    observed_mean = np.empty((len(looks), coordinate_count))
    observed_variance = np.empty_like(observed_mean)
    mean_scale = np.empty_like(observed_mean)
    variance_scale = np.empty_like(observed_mean)
    observed_energy = np.empty(len(looks))
    observed_rate = np.empty(len(looks))
    rate_scale = np.empty(len(looks))
    look_valid = np.empty(len(looks), dtype=bool)
    floors = _floors(config)
    for index, look in enumerate(looks):
        weights = _weights(base[: look.sample_size])
        early = endpoint_index[look.earlier]
        late = endpoint_index[look.later]
        early_values = values[: look.sample_size, early]
        late_values = values[: look.sample_size, late]
        early_mean = weights @ early_values
        late_mean = weights @ late_values
        early_variance = weights @ ((early_values - early_mean) ** 2)
        late_variance = weights @ ((late_values - late_mean) ** 2)
        observed_mean[index] = late_mean - early_mean
        observed_variance[index] = late_variance - early_variance
        mean_scale[index] = np.maximum(
            np.sqrt(0.5 * (early_variance + late_variance)),
            floors,
        )
        variance_scale[index] = np.maximum(
            0.5 * (early_variance + late_variance),
            floors**2,
        )
        observed_energy[index] = weights @ (
            energy[: look.sample_size, late] - energy[: look.sample_size, early]
        )
        early_rate = weights @ rate[: look.sample_size, early]
        late_rate = weights @ rate[: look.sample_size, late]
        observed_rate[index] = late_rate - early_rate
        rate_scale[index] = max(0.5 * (early_rate + late_rate), 1e-12)
        look_valid[index] = bool(
            np.all(valid[: look.sample_size, early]) and np.all(valid[: look.sample_size, late])
        )

    boot_mean, boot_variance, boot_energy, boot_rate = _joint_bootstrap(
        base,
        values,
        energy,
        rate,
        endpoints,
        looks,
        config=config,
        stage=stage,
    )
    mean_se = boot_mean.std(axis=0, ddof=1)
    variance_se = boot_variance.std(axis=0, ddof=1)
    energy_se = boot_energy.std(axis=0, ddof=1)
    rate_se = boot_rate.std(axis=0, ddof=1)
    alpha = float(config.family_alpha)
    criticals = {
        "mean_standardized": _critical(boot_mean, observed_mean, mean_se, alpha),
        "variance_standardized": _critical(
            boot_variance,
            observed_variance,
            variance_se,
            alpha,
        ),
        "energy_native": _critical(
            boot_energy,
            observed_energy,
            energy_se,
            alpha,
        ),
        "event_rate_relative": _critical(
            boot_rate,
            observed_rate,
            rate_se,
            alpha,
        ),
    }
    results: list[CoordinateInterval] = []
    for look_index, look in enumerate(looks):
        validity = Validity.VALID if look_valid[look_index] else Validity.UNEVALUABLE
        reason = "OK" if look_valid[look_index] else "INVALID_PARENT_OBSERVATION"
        for coordinate_index, coordinate in enumerate(CONTINUOUS_COORDINATES):
            upper = (
                abs(observed_mean[look_index, coordinate_index])
                + criticals["mean_standardized"] * mean_se[look_index, coordinate_index]
            ) / mean_scale[look_index, coordinate_index]
            results.append(
                CoordinateInterval(
                    stage,
                    Denominator.STRONG,
                    look.comparison_id,
                    look.sample_size,
                    "mean_standardized",
                    coordinate,
                    float(observed_mean[look_index, coordinate_index]),
                    float(mean_se[look_index, coordinate_index]),
                    float(mean_scale[look_index, coordinate_index]),
                    criticals["mean_standardized"],
                    float(upper),
                    float(config.mean_margin),
                    bool(look_valid[look_index] and upper <= float(config.mean_margin)),
                    validity,
                    reason,
                )
            )
            variance_upper = (
                abs(observed_variance[look_index, coordinate_index])
                + criticals["variance_standardized"] * variance_se[look_index, coordinate_index]
            ) / variance_scale[look_index, coordinate_index]
            results.append(
                CoordinateInterval(
                    stage,
                    Denominator.STRONG,
                    look.comparison_id,
                    look.sample_size,
                    "variance_standardized",
                    coordinate,
                    float(observed_variance[look_index, coordinate_index]),
                    float(variance_se[look_index, coordinate_index]),
                    float(variance_scale[look_index, coordinate_index]),
                    criticals["variance_standardized"],
                    float(variance_upper),
                    float(config.variance_margin),
                    bool(
                        look_valid[look_index] and variance_upper <= float(config.variance_margin)
                    ),
                    validity,
                    reason,
                )
            )
        energy_upper = (
            abs(observed_energy[look_index]) + criticals["energy_native"] * energy_se[look_index]
        )
        results.append(
            CoordinateInterval(
                stage,
                Denominator.STRONG,
                look.comparison_id,
                look.sample_size,
                "energy_native",
                "terminal_energy",
                float(observed_energy[look_index]),
                float(energy_se[look_index]),
                1.0,
                criticals["energy_native"],
                float(energy_upper),
                float(config.energy_margin),
                bool(look_valid[look_index] and energy_upper <= float(config.energy_margin)),
                validity,
                reason,
            )
        )
        rate_upper = (
            abs(observed_rate[look_index]) + criticals["event_rate_relative"] * rate_se[look_index]
        ) / rate_scale[look_index]
        results.append(
            CoordinateInterval(
                stage,
                Denominator.STRONG,
                look.comparison_id,
                look.sample_size,
                "event_rate_relative",
                "event_rate",
                float(observed_rate[look_index]),
                float(rate_se[look_index]),
                float(rate_scale[look_index]),
                criticals["event_rate_relative"],
                float(rate_upper),
                float(config.event_rate_margin),
                bool(look_valid[look_index] and rate_upper <= float(config.event_rate_margin)),
                validity,
                reason,
            )
        )
    return tuple(results)


def _eta_squared(values: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
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
    candidates: Sequence[float],
    sample_sizes: Sequence[int],
    config: QuantumTrajectoryPreparationQualificationConfig,
    stage: Stage,
) -> tuple[MemoryResult, ...]:
    clock_specs = tuple(
        spec
        for candidate in candidates
        for spec in (
            (f"candidate-{candidate:g}", candidate),
            (f"sentinel-{2 * candidate:g}", 2 * candidate),
        )
    )
    endpoints = tuple(sorted({endpoint for _, endpoint in clock_specs}))
    endpoint_index = {endpoint: index for index, endpoint in enumerate(endpoints)}
    base, values, _, _, _ = _arrays(records, endpoints)
    factors = {
        "initial_k": np.asarray([row.k_left for row in base], dtype=np.int64),
        "boundary_occupation": np.asarray(
            [row.boundary_occupation for row in base],
            dtype=np.int64,
        ),
        "translation_orbit": np.asarray(
            [row.translation_orbit for row in base],
            dtype=np.int64,
        ),
    }
    result: list[MemoryResult] = []
    threshold = float(config.memory_eta_squared)
    for factor, labels in factors.items():
        pending: list[tuple[int, int, int, float]] = []
        for sample_size in sample_sizes:
            weights = _weights(base[:sample_size])
            for clock_index, (_clock_id, endpoint) in enumerate(clock_specs):
                observed = _eta_squared(
                    values[:sample_size, endpoint_index[endpoint]],
                    labels[:sample_size],
                    weights,
                )
                pending.extend(
                    (sample_size, clock_index, coordinate_index, float(eta))
                    for coordinate_index, eta in enumerate(observed)
                )
        adverse = [row for row in pending if row[3] >= threshold]
        maximum_null: np.ndarray | None = None
        if adverse:
            rng = np.random.Generator(
                np.random.Philox(stage_inference_seed(stage, "joint-memory", factor=factor))
            )
            maximum_null = np.zeros(config.permutation_draws)
            # Exact permutations are only needed when the effect-size half of
            # the conjunction can pass. This is a decision-exact shortcut.
            for draw in range(config.permutation_draws):
                common_keys = rng.random(len(labels))
                draw_maximum = 0.0
                for sample_size in sample_sizes:
                    order = np.argsort(common_keys[:sample_size], kind="stable")
                    permuted = labels[:sample_size][order]
                    weights = _weights(base[:sample_size])
                    for clock_index, (_clock_id, endpoint) in enumerate(clock_specs):
                        eta = _eta_squared(
                            values[:sample_size, endpoint_index[endpoint]],
                            permuted,
                            weights,
                        )
                        draw_maximum = max(draw_maximum, float(np.max(eta)))
                maximum_null[draw] = draw_maximum
        for sample_size, clock_index, coordinate_index, eta_value in pending:
            if eta_value < threshold:
                p_value = 1.0
            else:
                assert maximum_null is not None
                p_value = float(
                    (1 + np.sum(maximum_null >= eta_value)) / (config.permutation_draws + 1)
                )
            material = bool(eta_value >= threshold and p_value <= float(config.family_alpha) / 3.0)
            result.append(
                MemoryResult(
                    stage,
                    Denominator.STRONG,
                    f"{clock_specs[clock_index][0]}-n{sample_size}",
                    sample_size,
                    factor,
                    CONTINUOUS_COORDINATES[coordinate_index],
                    eta_value,
                    p_value,
                    material,
                    False,
                    Validity.VALID,
                )
            )
    return tuple(result)


def analyze_grid(
    records: Sequence[PreparedObservation],
    *,
    candidates: Sequence[float],
    sample_sizes: Sequence[int],
    config: QuantumTrajectoryPreparationQualificationConfig,
    stage: Stage,
) -> dict[str, object]:
    intervals = stationarity_intervals(
        records,
        candidates=candidates,
        sample_sizes=sample_sizes,
        config=config,
        stage=stage,
    )
    memory = preparation_memory(
        records,
        candidates=candidates,
        sample_sizes=sample_sizes,
        config=config,
        stage=stage,
    )
    look_pass: dict[tuple[float, int], bool] = {}
    for candidate in candidates:
        for sample_size in sample_sizes:
            token = f"-n{sample_size}"
            relevant_intervals = [
                row
                for row in intervals
                if row.sample_size == sample_size
                and row.comparison_id.endswith(token)
                and (
                    row.comparison_id.startswith(f"candidate-{candidate:g}-")
                    or row.comparison_id.startswith(f"sentinel-{2 * candidate:g}-")
                )
            ]
            relevant_memory = [
                row
                for row in memory
                if row.sample_size == sample_size
                and row.clock_id
                in {
                    f"candidate-{candidate:g}-n{sample_size}",
                    f"sentinel-{2 * candidate:g}-n{sample_size}",
                }
            ]
            look_pass[(candidate, sample_size)] = bool(
                len(relevant_intervals) == 200
                and len(relevant_memory) == 294
                and all(row.passed for row in relevant_intervals)
                and all(row.validity == Validity.VALID for row in relevant_memory)
                and not any(row.material for row in relevant_memory)
            )
    suffix_start: dict[float, int | None] = {}
    ordered_sizes = tuple(sorted(sample_sizes))
    for candidate in candidates:
        suffix_start[candidate] = next(
            (
                sample_size
                for index, sample_size in enumerate(ordered_sizes)
                if all(look_pass[(candidate, later)] for later in ordered_sizes[index:])
            ),
            None,
        )
    selected = next(
        (
            candidate
            for candidate in sorted(candidates)
            if suffix_start[candidate] is not None and look_pass[(candidate, ordered_sizes[-1])]
        ),
        None,
    )
    return {
        "stage": stage.value,
        "candidate_order": list(candidates),
        "sample_sizes": list(sample_sizes),
        "look_pass": [
            {
                "candidate": candidate,
                "sample_size": sample_size,
                "passed": look_pass[(candidate, sample_size)],
            }
            for candidate in candidates
            for sample_size in sample_sizes
        ],
        "stable_suffix_start": {
            f"{candidate:g}": suffix_start[candidate] for candidate in candidates
        },
        "selected_burnin": selected,
        "intervals": [asdict(row) for row in intervals],
        "memory": [asdict(row) for row in memory],
    }


__all__ = [
    "LookSpec",
    "PreparedObservation",
    "analyze_grid",
    "comparison_specs",
    "preparation_memory",
    "stationarity_intervals",
]
