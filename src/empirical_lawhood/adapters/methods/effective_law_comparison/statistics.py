"""Unit-safe deterministic statistics for bounded post-hoc analyses."""

from __future__ import annotations

from random import Random
from typing import Sequence


def ratio(numerator: int, denominator: int) -> dict[str, object]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("ratio counts are invalid")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else format(numerator / denominator, ".12g"),
    }


def complete_unit_bootstrap_mean(
    values: Sequence[float],
    *,
    seed: int,
    draws: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Percentile interval resampling only the supplied complete-unit values."""

    if len(values) < 2:
        raise ValueError("complete-unit bootstrap needs at least two units")
    if draws < 1_000:
        raise ValueError("complete-unit bootstrap is underresolved")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must lie inside (0, 1)")
    rng = Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws)
    )
    tail = (1 - confidence) / 2
    lower_index = max(0, int(tail * draws))
    upper_index = min(draws - 1, int((1 - tail) * draws) - 1)
    return {
        "block": "complete-independent-unit",
        "confidence": format(confidence, ".12g"),
        "draws": draws,
        "lower": format(samples[lower_index], ".12g"),
        "point": format(sum(values) / count, ".12g"),
        "seed": seed,
        "upper": format(samples[upper_index], ".12g"),
        "unit_count": count,
    }


def leave_one_unit_means(values: Sequence[float], unit_ids: Sequence[str]) -> list[dict[str, str]]:
    if len(values) != len(unit_ids) or len(values) < 2:
        raise ValueError("leave-one-unit operands are invalid")
    total = sum(values)
    return [
        {
            "deleted_unit_id": unit_id,
            "point": format((total - value) / (len(values) - 1), ".12g"),
        }
        for unit_id, value in sorted(zip(unit_ids, values, strict=True))
    ]


__all__ = ["complete_unit_bootstrap_mean", "leave_one_unit_means", "ratio"]
