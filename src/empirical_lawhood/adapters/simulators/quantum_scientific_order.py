"""Explicit numerical tie orders bound to complete descriptive caller rosters."""

from __future__ import annotations

from typing import TypeAlias


ScientificOrder: TypeAlias = tuple[tuple[str, int], ...]


def scientific_order_ranks(order: ScientificOrder, *, identifiers: tuple[str, ...]) -> dict[str, int]:
    if (
        type(order) is not tuple
        or len(order) != len(identifiers)
        or any(type(row) is not tuple or len(row) != 2 for row in order)
        or any(type(identifier) is not str or type(rank) is not int or rank < 0 for identifier, rank in order)
        or tuple(identifier for identifier, _ in order) != identifiers
        or len(set(identifiers)) != len(identifiers)
        or len({rank for _, rank in order}) != len(order)
    ):
        raise ValueError("scientific order differs from the complete ordered identifier roster")
    return dict(order)


__all__ = ["ScientificOrder", "scientific_order_ranks"]
