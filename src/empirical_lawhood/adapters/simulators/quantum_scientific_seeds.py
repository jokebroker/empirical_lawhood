"""Exact numerical RNG inputs, separate from labels and historical authority."""

from __future__ import annotations

from typing import TypeAlias

ScientificSeedCensus: TypeAlias = tuple[tuple[int, int], ...]


def validate_scientific_seed(seed: int, *, bits: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**bits:
        raise ValueError("scientific seed must be an explicit unsigned integer of the declared width")


def validate_scientific_seed_census(
    seeds: ScientificSeedCensus,
    *,
    indices: tuple[int, ...],
    bits: int,
) -> None:
    if (
        not isinstance(seeds, tuple)
        or len(seeds) != len(indices)
        or any(not isinstance(row, tuple) or len(row) != 2 for row in seeds)
        or any(type(index) is not int or type(seed) is not int for index, seed in seeds)
        or tuple(index for index, _ in seeds) != indices
        or len(set(indices)) != len(indices)
        or any(not 0 <= seed < 2**bits for _, seed in seeds)
        or len({seed for _, seed in seeds}) != len(seeds)
    ):
        raise ValueError("scientific seed census differs from the exact ordered draw roster")


__all__ = ["ScientificSeedCensus", "validate_scientific_seed", "validate_scientific_seed_census"]
