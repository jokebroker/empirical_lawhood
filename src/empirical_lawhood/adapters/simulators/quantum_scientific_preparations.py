"""Validate the numerical preparation commitment before source execution.

The ordered rows are (preparation sector, basis state, scientific stream seed).
Labels do not select states or random draws. An exported prior allocation needs
its original source and custody; these inputs confer no historical authority.
"""

from __future__ import annotations

from typing import TypeAlias

ScientificPreparations: TypeAlias = tuple[tuple[int, int, int], ...]


def validate_scientific_preparations(
    rows: ScientificPreparations,
    *,
    l_sites: int,
    particles: int,
    sector_order: tuple[int, ...],
    seed_bits: int,
) -> None:
    if (
        not isinstance(rows, tuple)
        or len(rows) != len(sector_order)
        or any(not isinstance(row, tuple) or len(row) != 3 for row in rows)
        or any(type(value) is not int for row in rows for value in row)
        or tuple(row[0] for row in rows) != sector_order
        or any(
            not 0 <= state < 2**l_sites
            or state.bit_count() != particles
            or (state & ((1 << (l_sites // 2)) - 1)).bit_count() != sector
            or not 0 <= seed < 2**seed_bits
            for sector, state, seed in rows
        )
        or len({row[2] for row in rows}) != len(rows)
    ):
        raise ValueError("scientific preparation census differs from the ordered native allocation")


__all__ = ["ScientificPreparations", "validate_scientific_preparations"]
