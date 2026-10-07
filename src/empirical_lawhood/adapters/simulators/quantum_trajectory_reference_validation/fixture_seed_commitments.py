"""Numeric fixture operands and complete caller seed censuses.

Public plan, stage, denominator and purpose labels do not allocate draws.
The constants bind the declared fixtures only; arbitrary replication/panel
allocations must be supplied as a complete ordered numeric census. This module
transfers no historical scientific qualification or receipt authority.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, NamedTuple

GENERATOR_FIXTURE_SEED: Final = 16568974647114125161


class CoverageReplicationSeeds(NamedTuple):
    coordinate_permutation: int
    bounded_null: int


class StatisticalSeedCensus(NamedTuple):
    coverage: tuple[CoverageReplicationSeeds, ...]
    bounded_planted: tuple[int, ...]
    density_null: tuple[int, ...]
    density_planted: tuple[int, ...]
    stationarity_panels: tuple[int, ...]
    memory_panels: tuple[int, ...]


GENERATOR_CONFORMANCE_SEEDS: Final = MappingProxyType({
    (8, 4, 1.0): 12256985506165680611,
    (8, 4, 0.1): 2303578514964584014,
})


def validate_scientific_seed(seed: int) -> int:
    """Require one explicit unsigned 64-bit scientific stream operand."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("scientific seed must be an unsigned 64-bit integer")
    return seed


def generator_fixture_seed(
    l_sites: int,
    particles: int,
    gamma: float,
    scientific_seed: int | None,
) -> int:
    """Resolve only the declared physical basis; other bases need an operand."""
    if (
        type(l_sites) is not int
        or type(particles) is not int
        or l_sites <= 1
        or not 0 < particles < l_sites
        or type(gamma) is not float
        or gamma not in (1.0, 0.1)
    ):
        raise ValueError("invalid generator fixture physical domain")
    fixed = GENERATOR_CONFORMANCE_SEEDS.get((l_sites, particles, gamma))
    if scientific_seed is None:
        if fixed is None:
            raise ValueError("caller basis requires an explicit scientific fixture seed")
        return fixed
    seed = validate_scientific_seed(scientific_seed)
    if fixed is not None and seed != fixed:
        raise ValueError("scientific fixture seed differs from the declared physical allocation")
    return seed


def validate_coverage_seeds(scientific_seeds: CoverageReplicationSeeds) -> CoverageReplicationSeeds:
    if not isinstance(scientific_seeds, CoverageReplicationSeeds):
        raise ValueError("coverage replication requires its two typed scientific seeds")
    for seed in scientific_seeds:
        validate_scientific_seed(seed)
    if scientific_seeds.coordinate_permutation == scientific_seeds.bounded_null:
        raise ValueError("coverage replication scientific seeds must be distinct")
    return scientific_seeds


def validate_statistical_seed_census(
    scientific_seeds: StatisticalSeedCensus | None,
    *,
    null_replicates: int,
    planted_replicates: int,
    bootstrap_panels: int,
    memory_panels: int,
) -> StatisticalSeedCensus:
    """Validate every caller stream before any fixture draws or outcome access."""
    if not isinstance(scientific_seeds, StatisticalSeedCensus):
        raise ValueError("statistical conformance requires a complete typed scientific seed census")
    counts = (null_replicates, planted_replicates, null_replicates, planted_replicates, bootstrap_panels, memory_panels)
    all_seeds: list[int] = []
    for index, (domain, count) in enumerate(zip(scientific_seeds, counts, strict=True)):
        if type(count) is not int or count <= 0 or type(domain) is not tuple or len(domain) != count:
            raise ValueError("scientific seed census does not cover the complete declared domain")
        if index == 0:
            for row in domain:
                all_seeds.extend(validate_coverage_seeds(row))
        else:
            all_seeds.extend(validate_scientific_seed(seed) for seed in domain)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("scientific seed census contains duplicate stream allocations")
    return scientific_seeds
