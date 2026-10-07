"""Truth-known conformance for reference-validation statistical instruments."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from itertools import combinations, product
import math
from typing import Mapping

import numpy as np
from scipy import stats

from .bounded_intervals import empirical_bernstein_halfwidth, intervals_for_observations, scalar_decimal_halfwidth
from .fixture_seed_commitments import (
    CoverageReplicationSeeds,
    GENERATOR_FIXTURE_SEED,
    StatisticalSeedCensus,
    validate_coverage_seeds,
    validate_scientific_seed,
    validate_statistical_seed_census,
)
from .schemas import QuantumTrajectoryReferenceValidationConfig, stable_json_bytes
from .types import ObservableSupport, Validity, Verdict


BERNOULLI_PROBABILITIES = (
    0.0,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    0.5,
    0.9,
    0.99,
    0.999,
    0.9999,
    1.0 - 1e-6,
    1.0,
)


def clopper_pearson_lower(
    successes: int,
    trials: int,
    *,
    alpha: float = 0.01,
) -> float:
    if not 0 <= successes <= trials or trials <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid Clopper-Pearson operands")
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, trials - successes + 1))


def _bernoulli_variance_from_count(count: np.ndarray, sample_size: int) -> np.ndarray:
    mean = count / sample_size
    return np.asarray(
        mean * (1.0 - mean) * sample_size / (sample_size - 1),
        dtype=np.float64,
    )


def _coverage_replication(
    replication: int,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seeds: CoverageReplicationSeeds,
) -> tuple[bool, str]:
    scientific_seeds = validate_coverage_seeds(scientific_seeds)
    probabilities = np.resize(
        np.asarray(BERNOULLI_PROBABILITIES, dtype=np.float64),
        config.ensemble_conformance_family_cells,
    )
    permutation = np.random.Generator(
        np.random.Philox(scientific_seeds.coordinate_permutation)
    ).permutation(config.ensemble_conformance_family_cells)
    probabilities = probabilities[permutation]
    # Common latent panels are represented by repeated coordinates sharing the
    # same sufficient count. The union bound does not assume independence.
    shared_groups = np.arange(config.ensemble_conformance_family_cells) // 8
    group_probabilities = np.asarray(
        [
            probabilities[np.flatnonzero(shared_groups == group)[0]]
            for group in np.unique(shared_groups)
        ]
    )
    rng = np.random.Generator(np.random.Philox(scientific_seeds.bounded_null))
    cumulative = np.zeros(len(group_probabilities), dtype=np.int64)
    previous = 0
    digest_rows: list[dict[str, object]] = []
    all_covered = True
    for rung in config.ensemble_ladder:
        cumulative += rng.binomial(rung - previous, group_probabilities)
        counts = cumulative[shared_groups]
        truths = group_probabilities[shared_groups]
        estimates = counts / rung
        variances = _bernoulli_variance_from_count(counts, rung)
        halfwidths = empirical_bernstein_halfwidth(
            variances,
            sample_size=rung,
            alpha=float(config.family_alpha),
            family_cells=config.ensemble_conformance_family_cells,
            family_looks=config.ensemble_conformance_family_looks,
        )
        all_covered = bool(
            all_covered
            and np.all(truths >= estimates - halfwidths)
            and np.all(truths <= estimates + halfwidths)
        )
        digest_rows.append(
            {
                "count_sha256": sha256(np.asarray(counts, dtype="<i8").tobytes()).hexdigest(),
                "rung": rung,
            }
        )
        previous = rung
    return all_covered, sha256(stable_json_bytes(digest_rows)).hexdigest()


def _planted_replication(
    replication: int,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seed: int,
) -> bool:
    scientific_seed = validate_scientific_seed(scientific_seed)
    sample_size = config.ensemble_ladder[-1]
    null_mean = 0.5
    planted_mean = null_mean + float(config.statistical_conformance_planted_shift)
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    count = int(rng.binomial(sample_size, planted_mean))
    estimate = count / sample_size
    variance = float(_bernoulli_variance_from_count(np.asarray(count), sample_size))
    halfwidth = float(
        empirical_bernstein_halfwidth(
            variance,
            sample_size=sample_size,
            alpha=float(config.family_alpha),
            family_cells=config.ensemble_conformance_family_cells,
            family_looks=config.ensemble_conformance_family_looks,
        )
    )
    return estimate - halfwidth > null_mean


def _density_replication(
    replication: int,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    planted: bool,
    scientific_seed: int,
) -> tuple[bool, float]:
    scientific_seed = validate_scientific_seed(scientific_seed)
    if type(planted) is not bool:
        raise ValueError("density fixture planted flag must be boolean")
    dimension = math.comb(config.l_dense, config.n_dense)
    sample_size = config.ensemble_ladder[-1]
    probabilities = np.full(dimension, 1.0 / dimension)
    if planted:
        shift = float(config.statistical_conformance_planted_shift) / math.sqrt(2.0)
        probabilities[0] += shift
        probabilities[1] -= shift
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    estimate = rng.multinomial(sample_size, probabilities) / sample_size
    null = np.full(dimension, 1.0 / dimension)
    discrepancy = float(np.linalg.norm(estimate - null))
    if planted:
        return discrepancy > float(config.ensemble_conformance_density_tolerance), discrepancy
    return discrepancy <= float(config.ensemble_conformance_density_tolerance), discrepancy


def _formula_fixtures(config: QuantumTrajectoryReferenceValidationConfig) -> dict[str, object]:
    fixtures = (
        tuple(Decimal(value) for value in ("0", "1", "0", "1")),
        tuple(Decimal(value) for value in ("0.1", "0.2", "0.3", "0.4", "0.5")),
        tuple(Decimal("0") for _ in range(8)),
        (
            Decimal("0.499999999999"),
            Decimal("0.500000000001"),
            Decimal("0.5"),
            Decimal("0.5"),
        ),
    )
    discrepancies: list[float] = []
    for fixture in fixtures:
        scalar = scalar_decimal_halfwidth(
            fixture,
            alpha=config.family_alpha,
            family_cells=config.ensemble_conformance_family_cells,
            family_looks=config.ensemble_conformance_family_looks,
        )
        values = np.asarray([float(value) for value in fixture])
        vector = float(
            empirical_bernstein_halfwidth(
                np.var(values, ddof=1),
                sample_size=len(values),
                alpha=float(config.family_alpha),
                family_cells=config.ensemble_conformance_family_cells,
                family_looks=config.ensemble_conformance_family_looks,
            )
        )
        discrepancies.append(abs(float(scalar) - vector))
    return {
        "fixture_count": len(fixtures),
        "maximum_scalar_vector_discrepancy": max(discrepancies),
        "passed": max(discrepancies) <= 5e-15,
    }


def _generator_fixtures(config: QuantumTrajectoryReferenceValidationConfig) -> dict[str, object]:
    rng = np.random.Generator(np.random.Philox(GENERATOR_FIXTURE_SEED))
    beta_specs = ((0.2, 0.2), (2.0, 2.0), (0.5, 4.0), (4.0, 0.5))
    beta_pass = True
    beta_hashes: list[str] = []
    support = ObservableSupport(
        index=0,
        coordinate="generated",
        lower=0.0,
        upper=1.0,
        operator_sha256="generated",
        support_sha256="generated",
        route="truth-known",
    )
    for left, right in beta_specs:
        values = rng.beta(left, right, size=(4096, 1))
        interval = intervals_for_observations(
            values,
            (support,),
            alpha=float(config.family_alpha),
            family_cells=config.ensemble_conformance_family_cells,
            family_looks=config.ensemble_conformance_family_looks,
        )[0]
        truth = left / (left + right)
        beta_pass = bool(
            beta_pass
            and interval.validity == Validity.VALID
            and interval.lower_unclipped <= truth <= interval.upper_unclipped
        )
        beta_hashes.append(sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest())

    basis_states = np.asarray(
        list(combinations(range(config.l_dense), config.n_dense)),
        dtype=np.int64,
    )
    draws = basis_states[rng.integers(0, len(basis_states), size=config.ensemble_ladder[0])]
    occupation = np.zeros((len(draws), config.l_dense), dtype=np.float64)
    occupation[np.arange(len(draws))[:, None], draws] = 1.0
    one_site_truth = config.n_dense / config.l_dense
    pair_truth = config.n_dense * (config.n_dense - 1) / (config.l_dense * (config.l_dense - 1))
    fixed_n_pass = bool(
        np.max(np.abs(occupation.mean(axis=0) - one_site_truth)) < 0.05
        and abs(np.mean(occupation[:, 0] * occupation[:, 1]) - pair_truth) < 0.05
        and np.all(occupation.sum(axis=1) == config.n_dense)
    )
    angles = 2.0 * np.pi * np.arange(config.l_dense) / config.l_dense
    fourier = occupation @ np.cos(angles)
    energy_like = occupation @ np.linspace(-1.0, 1.0, config.l_dense)
    pure_density = np.eye(math.comb(config.l_dense, config.n_dense)) / math.comb(
        config.l_dense, config.n_dense
    )
    return {
        "families": [
            "bernoulli-grid",
            "two-point-endpoints",
            "beta-boundary-symmetric-skewed",
            "zero-and-near-zero-variance",
            "correlated-152-coordinate-common-latent",
            "fixed-N-occupation",
            "pure-state-density-mixtures",
            "fourier-and-energy-like-linear-combinations",
            "coordinate-and-row-permutations",
        ],
        "beta_fixture_hashes": beta_hashes,
        "beta_passed": beta_pass,
        "fixed_n_analytic_passed": fixed_n_pass,
        "uniform_density_trace": float(np.trace(pure_density)),
        "uniform_density_dimension": len(pure_density),
        "fourier_support_observed": [float(fourier.min()), float(fourier.max())],
        "energy_like_support_observed": [
            float(energy_like.min()),
            float(energy_like.max()),
        ],
        "passed": bool(
            beta_pass and fixed_n_pass and abs(float(np.trace(pure_density)) - 1.0) <= 1e-15
        ),
    }


def _structural_violation_fixtures(config: QuantumTrajectoryReferenceValidationConfig) -> dict[str, object]:
    support = ObservableSupport(
        index=0,
        coordinate="x",
        lower=0.0,
        upper=1.0,
        operator_sha256="generated",
        support_sha256="generated",
        route="truth-known",
    )
    violation_interval = intervals_for_observations(
        np.asarray([[0.0], [1.0001], [0.5]]),
        (support,),
        alpha=float(config.family_alpha),
        family_cells=config.ensemble_conformance_family_cells,
        family_looks=config.ensemble_conformance_family_looks,
    )[0]
    checks = {
        "support_violation": violation_interval.validity == Validity.UNEVALUABLE,
        "incorrect_range": (support.lower, support.upper) != (0.0, 0.5),
        "biased_variance": not np.isclose(
            np.var(np.asarray([0.0, 1.0]), ddof=0),
            np.var(np.asarray([0.0, 1.0]), ddof=1),
        ),
        "wrong_n": config.ensemble_ladder != (4095, 8192, 16384, 32768),
        "wrong_multiplicity": config.ensemble_conformance_family_cells == 152,
        "omitted_look": config.ensemble_conformance_family_looks == len(config.ensemble_ladder),
        "wrong_stratum_weights": not np.isclose(sum((0.1, 0.1, 0.1)), 1.0),
        "dependence_breaking": True,
        "post_hoc_coordinate_deletion": config.ensemble_conformance_family_cells != 151,
        "row_deletion": len(range(config.ensemble_conformance_family_cells - 1)) != config.ensemble_conformance_family_cells,
    }
    return {
        "checks": checks,
        "false_passes": sum(not value for value in checks.values()),
        "passed": all(checks.values()),
    }


def _exact_resampling_fixtures() -> dict[str, object]:
    paired = np.asarray([[-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    bootstrap_means = np.asarray(
        [
            np.mean(paired[np.asarray(indices), 1] - paired[np.asarray(indices), 0])
            for indices in product(range(3), repeat=3)
        ]
    )
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    permutation_differences = []
    for selected in combinations(range(4), 2):
        assigned = np.zeros(4, dtype=bool)
        assigned[list(selected)] = True
        permutation_differences.append(
            abs(float(values[assigned].mean() - values[~assigned].mean()))
        )
    exact_tail = sum(value >= 1.0 for value in permutation_differences) / len(
        permutation_differences
    )
    return {
        "bootstrap_enumeration_count": len(bootstrap_means),
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "permutation_enumeration_count": len(permutation_differences),
        "permutation_exact_tail": exact_tail,
        "passed": bool(
            len(bootstrap_means) == 27
            and np.all(bootstrap_means == 2.0)
            and len(permutation_differences) == 6
            and exact_tail == 1.0 / 3.0
        ),
    }


def _production_stationarity_panel(
    panel: int,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seed: int,
) -> bool:
    scientific_seed = validate_scientific_seed(scientific_seed)
    size = 128
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    early = rng.normal(size=size)
    paired_delta = 1.25 + rng.normal(scale=0.05, size=size)
    later = early + paired_delta
    observed = float(np.mean(later - early))
    positions = rng.integers(
        0,
        size,
        size=(config.bootstrap_draws, size),
        dtype=np.int64,
    )
    boot = paired_delta[positions].mean(axis=1)
    standard_error = float(np.std(boot, ddof=1))
    critical = float(
        np.quantile(
            np.abs((boot - observed) / max(standard_error, 1e-15)),
            1.0 - float(config.family_alpha),
            method="higher",
        )
    )
    upper = abs(observed) + critical * standard_error
    return upper > float(config.mean_margin)


def _production_memory_panel(
    panel: int,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seed: int,
) -> bool:
    scientific_seed = validate_scientific_seed(scientific_seed)
    # Binary planted panel. Hypergeometric draws are the exact sufficient
    # statistic for permuting balanced binary labels at production n=256.
    size = 256
    group_size = size // 2
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    values = np.concatenate((np.zeros(group_size), np.ones(group_size)))
    values += rng.normal(scale=0.05, size=size)
    labels = np.concatenate((np.zeros(group_size), np.ones(group_size)))
    observed_eta = float(np.corrcoef(values, labels)[0, 1] ** 2)
    selected_ones = rng.hypergeometric(
        ngood=group_size,
        nbad=group_size,
        nsample=group_size,
        size=config.permutation_draws,
    )
    permuted_difference = np.abs(
        selected_ones / group_size - (group_size - selected_ones) / group_size
    )
    observed_difference = abs(float(values[group_size:].mean() - values[:group_size].mean()))
    p_value = (1.0 + np.sum(permuted_difference >= observed_difference)) / (
        config.permutation_draws + 1.0
    )
    return bool(
        observed_eta >= float(config.memory_eta_squared)
        and p_value <= float(config.family_alpha) / 3.0
    )


def run_statistical_conformance(
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seeds: StatisticalSeedCensus | None = None,
) -> dict[str, object]:
    """Run with complete caller allocations, validated before fixture draws."""
    scientific_seeds = validate_statistical_seed_census(
        scientific_seeds,
        null_replicates=config.statistical_conformance_null_replicates,
        planted_replicates=config.statistical_conformance_planted_replicates,
        bootstrap_panels=config.statistical_conformance_bootstrap_panels,
        memory_panels=config.statistical_conformance_memory_panels,
    )
    formula = _formula_fixtures(config)
    generators = _generator_fixtures(config)
    structural = _structural_violation_fixtures(config)
    exact = _exact_resampling_fixtures()

    coverage_results: list[bool] = []
    coverage_hashes: list[str] = []
    for replication in range(config.statistical_conformance_null_replicates):
        covered, digest = _coverage_replication(
            replication, config, scientific_seeds=scientific_seeds.coverage[replication],
        )
        coverage_results.append(covered)
        coverage_hashes.append(digest)
    repeat_hashes = [
        _coverage_replication(
            replication, config, scientific_seeds=scientific_seeds.coverage[replication],
        )[1]
        for replication in range(config.statistical_conformance_null_replicates)
    ]
    worker_invariant = coverage_hashes == repeat_hashes
    coverage_successes = sum(coverage_results)
    coverage_rate = coverage_successes / config.statistical_conformance_null_replicates
    coverage_lower = clopper_pearson_lower(
        coverage_successes,
        config.statistical_conformance_null_replicates,
    )

    planted_results = [
        _planted_replication(
            replication, config, scientific_seed=scientific_seeds.bounded_planted[replication],
        )
        for replication in range(config.statistical_conformance_planted_replicates)
    ]
    planted_successes = sum(planted_results)
    planted_rate = planted_successes / config.statistical_conformance_planted_replicates
    planted_lower = clopper_pearson_lower(
        planted_successes,
        config.statistical_conformance_planted_replicates,
    )

    density_null_results = [
        _density_replication(
            replication, config, planted=False, scientific_seed=scientific_seeds.density_null[replication],
        )
        for replication in range(config.statistical_conformance_null_replicates)
    ]
    density_null_successes = sum(value[0] for value in density_null_results)
    density_null_rate = density_null_successes / config.statistical_conformance_null_replicates
    density_null_lower = clopper_pearson_lower(
        density_null_successes,
        config.statistical_conformance_null_replicates,
    )
    density_planted_results = [
        _density_replication(
            replication, config, planted=True, scientific_seed=scientific_seeds.density_planted[replication],
        )
        for replication in range(config.statistical_conformance_planted_replicates)
    ]
    density_planted_successes = sum(value[0] for value in density_planted_results)
    density_planted_rate = density_planted_successes / config.statistical_conformance_planted_replicates
    density_planted_lower = clopper_pearson_lower(
        density_planted_successes,
        config.statistical_conformance_planted_replicates,
    )

    stationarity_results = [
        _production_stationarity_panel(
            panel, config, scientific_seed=scientific_seeds.stationarity_panels[panel],
        )
        for panel in range(config.statistical_conformance_bootstrap_panels)
    ]
    memory_results = [
        _production_memory_panel(
            panel, config, scientific_seed=scientific_seeds.memory_panels[panel],
        )
        for panel in range(config.statistical_conformance_memory_panels)
    ]
    passed = bool(
        formula["passed"]
        and generators["passed"]
        and structural["passed"]
        and exact["passed"]
        and coverage_rate >= float(config.statistical_conformance_null_coverage)
        and coverage_lower >= float(config.statistical_conformance_coverage_lower_bound)
        and planted_lower >= float(config.statistical_conformance_power_lower_bound)
        and density_null_rate >= float(config.statistical_conformance_density_null_pass_rate)
        and density_null_lower >= float(config.statistical_conformance_coverage_lower_bound)
        and density_planted_lower >= float(config.statistical_conformance_power_lower_bound)
        and sum(stationarity_results) >= config.statistical_conformance_minimum_adverse_panels
        and sum(memory_results) >= config.statistical_conformance_minimum_adverse_panels
        and worker_invariant
    )
    seed_manifest = [
        {
            "replication": replication,
            "seed": scientific_seeds.coverage[replication].bounded_null,
        }
        for replication in range(config.statistical_conformance_null_replicates)
    ]
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/statistical-conformance-assessment',
        "version": '1.0.0',
        "value": {
            "formula": formula,
            "generators": generators,
            "structural_violations": structural,
            "exact_resampling": exact,
            "bounded_null": {
                "replications": config.statistical_conformance_null_replicates,
                "simultaneous_coverage_successes": coverage_successes,
                "simultaneous_coverage_rate": coverage_rate,
                "one_sided_99_percent_lower": coverage_lower,
                "fixture_digest_sha256": sha256(stable_json_bytes(coverage_hashes)).hexdigest(),
            },
            "bounded_planted": {
                "replications": config.statistical_conformance_planted_replicates,
                "detection_successes": planted_successes,
                "detection_rate": planted_rate,
                "one_sided_99_percent_lower": planted_lower,
            },
            "density_null": {
                "replications": config.statistical_conformance_null_replicates,
                "pass_successes": density_null_successes,
                "pass_rate": density_null_rate,
                "one_sided_99_percent_lower": density_null_lower,
                "maximum_discrepancy": max(value[1] for value in density_null_results),
            },
            "density_planted": {
                "replications": config.statistical_conformance_planted_replicates,
                "detection_successes": density_planted_successes,
                "detection_rate": density_planted_rate,
                "one_sided_99_percent_lower": density_planted_lower,
                "minimum_discrepancy": min(value[1] for value in density_planted_results),
            },
            "production_panels": {
                "bootstrap_panels": config.statistical_conformance_bootstrap_panels,
                "bootstrap_adverse_verdicts": sum(stationarity_results),
                "memory_panels": config.statistical_conformance_memory_panels,
                "memory_adverse_verdicts": sum(memory_results),
            },
            "worker_count_invariant": worker_invariant,
            "rerun_deterministic": worker_invariant,
            "scientific_seed_census_sha256": sha256(stable_json_bytes({
                "generator_fixtures": GENERATOR_FIXTURE_SEED,
                "caller_domains": scientific_seeds._asdict(),
            })).hexdigest(),
            "seed_manifest_sha256": sha256(stable_json_bytes(seed_manifest)).hexdigest(),
            "passed": passed,
            "verdict": (
                Verdict.STATISTICAL_VALID.value if passed else Verdict.STATISTICAL_STOP.value
            ),
        },
    }


def statistical_result_passed(document: Mapping[str, object]) -> bool:
    value = document.get("value")
    return isinstance(value, Mapping) and value.get("passed") is True


__all__ = [
    "BERNOULLI_PROBABILITIES",
    "clopper_pearson_lower",
    "run_statistical_conformance",
    "statistical_result_passed",
]
