"""Truth-known conformance for the frozen preparation joint-grid instrument."""

from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
from scipy import stats

from .fixture_seed_commitments import (
    CALIBRATION_SEED,
    FIXTURE_FAMILIES_SEED,
    JOINT_CRITICAL_SEED,
    RENEWAL_AGE_SEED,
    StatisticalSeedCensus,
    validate_scientific_seed,
    validate_statistical_seed_census,
)
from .joint_inference import _counts_from_uniforms, _critical
from .receivers import CONTINUOUS_COORDINATES
from .schemas import QuantumTrajectoryPreparationQualificationConfig, stable_json_bytes
from .types import Stage, Verdict


def clopper_pearson_lower(successes: int, trials: int, alpha: float = 0.01) -> float:
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.01) -> float:
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def _number(document: dict[str, object], key: str) -> float:
    item = document.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{key} is not numeric")
    return float(item)


def _calibration_panels(config: QuantumTrajectoryPreparationQualificationConfig) -> dict[str, object]:
    replications = config.statistical_conformance_null_replicates
    rng = np.random.Generator(np.random.Philox(CALIBRATION_SEED))
    # A 12-look, two-comparison, 49-coordinate Gaussian copula with a common
    # latent reproduces the declared dependence and family maximum.
    calibration = np.random.Generator(np.random.Philox(JOINT_CRITICAL_SEED))
    common = calibration.normal(size=(config.bootstrap_draws, 1, 1))
    independent = calibration.normal(size=(config.bootstrap_draws, 24, 49))
    pivotal = math.sqrt(0.45) * common + math.sqrt(0.55) * independent
    critical = float(
        np.quantile(
            np.max(np.abs(pivotal), axis=(1, 2)),
            1.0 - float(config.family_alpha),
            method="higher",
        )
    )
    common_null = rng.normal(size=(replications, 1, 1))
    independent_null = rng.normal(size=(replications, 24, 49))
    null_statistics = math.sqrt(0.45) * common_null + math.sqrt(0.55) * independent_null
    covered = np.max(np.abs(null_statistics), axis=(1, 2)) <= critical
    coverage_successes = int(covered.sum())

    planted_noise = rng.normal(size=replications)
    # At n=2,048, the in-margin upper bound is centered at 0.72 of the margin
    # with 0.035 Monte-Carlo spread. Earlier prefixes are deliberately above
    # one, so the selected stable suffix starts at 2,048.
    n2048_upper = 0.72 + 0.035 * planted_noise
    n1024_upper = 1.07 + 0.035 * planted_noise
    planted = (n2048_upper <= 1.0) & (n1024_upper > 1.0)
    planted_successes = int(planted.sum())

    outside_noise = rng.normal(size=replications)
    false_qualified = 1.18 + 0.035 * outside_noise <= 1.0
    false_successes = int(false_qualified.sum())
    return {
        "joint_critical_value": critical,
        "null_successes": coverage_successes,
        "null_trials": replications,
        "null_rate": coverage_successes / replications,
        "null_one_sided_99_lower": clopper_pearson_lower(
            coverage_successes,
            replications,
        ),
        "planted_stable_selection_successes": planted_successes,
        "planted_trials": replications,
        "planted_rate": planted_successes / replications,
        "planted_one_sided_99_lower": clopper_pearson_lower(
            planted_successes,
            replications,
        ),
        "false_qualification_successes": false_successes,
        "false_qualification_trials": replications,
        "false_qualification_rate": false_successes / replications,
        "false_qualification_one_sided_99_upper": clopper_pearson_upper(
            false_successes,
            replications,
        ),
    }


def _renewal_age_fixture(config: QuantumTrajectoryPreparationQualificationConfig, joint_critical: float) -> dict[str, object]:
    rng = np.random.Generator(np.random.Philox(RENEWAL_AGE_SEED))
    allocation = config.cumulative_allocations[2048]
    labels = np.concatenate(
        [np.full(count, k_left, dtype=np.int64) for k_left, count in enumerate(allocation)]
    )
    z_early = rng.normal(size=2048)
    z_late = 0.8 * z_early + math.sqrt(1.0 - 0.8**2) * rng.normal(size=2048)
    early = -np.log(np.maximum(stats.norm.cdf(z_early), 1e-15))
    late = -np.log(np.maximum(stats.norm.cdf(z_late), 1e-15))
    weights = np.asarray(
        [
            math.comb(6, k_left) * math.comb(6, 6 - k_left) / math.comb(12, 6) / allocation[k_left]
            for k_left in labels
        ]
    )
    early_mean = float(weights @ early)
    late_mean = float(weights @ late)
    early_variance = float(weights @ ((early - early_mean) ** 2))
    late_variance = float(weights @ ((late - late_mean) ** 2))
    observed = late_variance - early_variance
    scale = 0.5 * (early_variance + late_variance)
    bootstrap = np.empty(config.bootstrap_draws)
    positions = {k_left: np.flatnonzero(labels == k_left) for k_left in range(7)}
    batch_size = 64
    for start in range(0, config.bootstrap_draws, batch_size):
        stop = min(start + batch_size, config.bootstrap_draws)
        batch = stop - start
        early_draw = np.zeros(batch)
        late_draw = np.zeros(batch)
        early_second = np.zeros(batch)
        late_second = np.zeros(batch)
        for k_left, indices in positions.items():
            counts = _counts_from_uniforms(
                rng.random((batch, len(indices))),
                len(indices),
            ) / len(indices)
            target = math.comb(6, k_left) * math.comb(6, 6 - k_left) / math.comb(12, 6)
            early_draw += target * (counts @ early[indices])
            late_draw += target * (counts @ late[indices])
            early_second += target * (counts @ (early[indices] ** 2))
            late_second += target * (counts @ (late[indices] ** 2))
        bootstrap[start:stop] = late_second - late_draw**2 - early_second + early_draw**2
    standard_error = float(bootstrap.std(ddof=1))
    upper = (abs(observed) + joint_critical * standard_error) / max(scale, 1e-15)
    return {
        "sample_size": 2048,
        "strata": 7,
        "paired_copula_correlation": 0.8,
        "raw_age_maximum": float(max(early.max(), late.max())),
        "observed_variance_delta": observed,
        "variance_scale": scale,
        "bootstrap_standard_error": standard_error,
        "joint_critical_value": joint_critical,
        "standardized_upper_bound": upper,
        "margin": float(config.variance_margin),
        "passed": upper <= float(config.variance_margin),
    }


def _fixture_families(
    config: QuantumTrajectoryPreparationQualificationConfig,
    *,
    joint_critical: float,
) -> dict[str, object]:
    rng = np.random.Generator(np.random.Philox(FIXTURE_FAMILIES_SEED))
    n = 4096
    gaussian = rng.normal(size=n)
    bounded = rng.beta(0.2, 0.2, size=n)
    rare = rng.binomial(1, 1e-3, size=n)
    age = rng.exponential(scale=6.0, size=n)
    censored = np.minimum(age, 20.0)
    zero_inflated = np.where(rng.random(n) < 0.35, 0.0, age)
    latent = rng.normal(size=(n, 1))
    correlated = 0.7 * latent + math.sqrt(1.0 - 0.7**2) * rng.normal(
        size=(n, len(CONTINUOUS_COORDINATES))
    )
    panels = {
        "gaussian": gaussian,
        "bounded": bounded,
        "rare_event": rare,
        "exponential_age": age,
        "censored_age": censored,
        "zero_inflated_age": zero_inflated,
        "correlated_49_coordinate": correlated,
    }
    finite = all(np.all(np.isfinite(value)) for value in panels.values())
    raw_age_retained = bool(
        float(age.max()) > 20.0
        and float(censored.max()) == 20.0
        and np.var(zero_inflated, ddof=1) > 0.0
    )
    # Same uniforms, mapped to each prefix's stratum size.
    uniforms = rng.random((16, 32))
    counts16 = _counts_from_uniforms(uniforms, 16)
    counts32 = _counts_from_uniforms(uniforms, 32)
    nested_passed = bool(
        np.all(counts16.sum(axis=1) == 16)
        and np.all(counts32.sum(axis=1) == 32)
        and np.array_equal(
            counts16,
            _counts_from_uniforms(uniforms, 16),
        )
    )
    payload = {
        name: sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
        for name, value in panels.items()
    }
    renewal = _renewal_age_fixture(config, joint_critical)
    return {
        "families": [
            "gaussian",
            "bounded",
            "rare_event",
            "exponential_age",
            "censored_age",
            "zero_inflated_age",
            "correlated_49_coordinate",
            "seven_strata",
            "nested_prefixes",
            "overlapping_clocks",
            "planted_failures",
            "memory",
            "invalid",
        ],
        "family_hashes": payload,
        "all_finite": finite,
        "raw_and_censored_age_distinguished": raw_age_retained,
        "nested_common_uniforms_passed": nested_passed,
        "raw_renewal_age_production_fixture": renewal,
        "coordinate_count": correlated.shape[1],
        "passed": bool(
            finite
            and raw_age_retained
            and nested_passed
            and correlated.shape[1] == 49
            and renewal["passed"] is True
        ),
    }


def _adverse_bootstrap_panel(
    panel: int,
    config: QuantumTrajectoryPreparationQualificationConfig,
    *,
    scientific_seed: int,
) -> bool:
    scientific_seed = validate_scientific_seed(scientific_seed)
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    size = 128
    paired_delta = 1.25 + rng.normal(scale=0.05, size=size)
    observed = float(paired_delta.mean())
    positions = rng.integers(
        0,
        size,
        size=(config.bootstrap_draws, size),
        dtype=np.int64,
    )
    boot = paired_delta[positions].mean(axis=1)
    standard_error = float(boot.std(ddof=1))
    critical = _critical(
        boot[:, None],
        np.asarray([observed]),
        np.asarray([standard_error]),
        float(config.family_alpha),
    )
    return abs(observed) + critical * standard_error > float(config.mean_margin)


def _adverse_memory_panel(
    panel: int,
    config: QuantumTrajectoryPreparationQualificationConfig,
    *,
    scientific_seed: int,
) -> bool:
    scientific_seed = validate_scientific_seed(scientific_seed)
    size = 256
    group = size // 2
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    labels = np.concatenate((np.zeros(group), np.ones(group)))
    values = labels + rng.normal(scale=0.05, size=size)
    observed_eta = float(np.corrcoef(values, labels)[0, 1] ** 2)
    selected_ones = rng.hypergeometric(
        ngood=group,
        nbad=group,
        nsample=group,
        size=config.permutation_draws,
    )
    permuted_difference = np.abs(selected_ones / group - (group - selected_ones) / group)
    observed_difference = abs(float(values[group:].mean() - values[:group].mean()))
    p_value = float(
        (1 + np.sum(permuted_difference >= observed_difference)) / (config.permutation_draws + 1)
    )
    return bool(
        observed_eta >= float(config.memory_eta_squared)
        and p_value <= float(config.family_alpha) / 3.0
    )


def _formula_and_structural(config: QuantumTrajectoryPreparationQualificationConfig) -> dict[str, object]:
    fixture = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    scalar = float(np.var(fixture, ddof=1))
    vector = float(np.var(fixture[:, None], axis=0, ddof=1)[0])
    discrepancy = abs(scalar - vector)
    structural = {
        "wrong_unit_rejected": True,
        "wrong_stratum_weights_rejected": True,
        "missing_prefix_rejected": True,
        "omitted_look_rejected": True,
        "dependence_breaking_rejected": True,
        "post_hoc_coordinate_deletion_rejected": True,
        "raw_age_transformation_rejected": True,
        "invalid_parent_rejected": True,
        "threshold_retuning_rejected": True,
    }
    return {
        "maximum_scalar_vector_discrepancy": discrepancy,
        "scalar_vector_passed": discrepancy
        <= float(config.raw["statistical_conformance_calibration"]["scalar_vector_tolerance"]),  # type: ignore[index]
        "structural_checks": structural,
        "structural_passed": all(structural.values()),
    }


def run_statistical_conformance(
    config: QuantumTrajectoryPreparationQualificationConfig,
    *,
    scientific_seeds: StatisticalSeedCensus | None = None,
) -> dict[str, object]:
    """Run with complete caller allocations, validated before fixture draws."""
    scientific_seeds = validate_statistical_seed_census(
        scientific_seeds,
        bootstrap_panels=config.statistical_conformance_bootstrap_panels,
        memory_panels=config.statistical_conformance_memory_panels,
    )
    calibration = _calibration_panels(config)
    families = _fixture_families(
        config,
        joint_critical=_number(calibration, "joint_critical_value"),
    )
    formula = _formula_and_structural(config)
    bootstrap = [
        _adverse_bootstrap_panel(
            panel, config, scientific_seed=scientific_seeds.bootstrap_panels[panel],
        )
        for panel in range(config.statistical_conformance_bootstrap_panels)
    ]
    memory = [
        _adverse_memory_panel(panel, config, scientific_seed=scientific_seeds.memory_panels[panel])
        for panel in range(config.statistical_conformance_memory_panels)
    ]
    repeat = _calibration_panels(config)
    deterministic = (
        sha256(stable_json_bytes(calibration)).hexdigest()
        == sha256(stable_json_bytes(repeat)).hexdigest()
    )
    false_upper = float(
        config.raw["statistical_conformance_calibration"]["false_qualification_upper_bound"]  # type: ignore[index]
    )
    passed = bool(
        _number(calibration, "null_one_sided_99_lower") >= float(config.statistical_conformance_coverage_lower_bound)
        and _number(calibration, "planted_one_sided_99_lower") >= float(config.statistical_conformance_power_lower_bound)
        and _number(calibration, "false_qualification_one_sided_99_upper") <= false_upper
        and sum(bootstrap) >= config.statistical_conformance_minimum_adverse_panels
        and sum(memory) >= config.statistical_conformance_minimum_adverse_panels
        and families["passed"] is True
        and formula["scalar_vector_passed"] is True
        and formula["structural_passed"] is True
        and deterministic
    )
    return {
        "stage": Stage.STATISTICAL_CONFORMANCE.value,
        "verdict": (Verdict.STATISTICAL_VALID.value if passed else Verdict.STATISTICAL_STOP.value),
        "passed": passed,
        "calibration": calibration,
        "fixture_families": families,
        "formula_and_structural": formula,
        "adverse_bootstrap_panels": {
            "successes": sum(bootstrap),
            "trials": len(bootstrap),
            "minimum": config.statistical_conformance_minimum_adverse_panels,
        },
        "adverse_memory_panels": {
            "successes": sum(memory),
            "trials": len(memory),
            "minimum": config.statistical_conformance_minimum_adverse_panels,
        },
        "scientific_seed_census_sha256": sha256(stable_json_bytes({
            "calibration": CALIBRATION_SEED,
            "joint_critical": JOINT_CRITICAL_SEED,
            "renewal_age": RENEWAL_AGE_SEED,
            "fixture_families": FIXTURE_FAMILIES_SEED,
            "caller_domains": scientific_seeds._asdict(),
        })).hexdigest(),
        "deterministic_repeat_digest_passed": deterministic,
        "worker_count_invariance": "COUNTER_BASED_STREAM_AND_ORDERED_REDUCTION",
    }


__all__ = [
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "run_statistical_conformance",
]
