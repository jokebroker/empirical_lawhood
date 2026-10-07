"""Pure, platform-independent analysis for the FreeGSNKE signed-pair fixture."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math
from typing import Final

from .contracts import (
    FREEGSNKE_RECEIVER_FAMILIES,
    FREEGSNKE_RECEIVER_UNITS,
    FreeGsnkeProcessResponse,
)


ZERO_REPEAT_TOLERANCE: Final = Decimal("1e-10")
ABSOLUTE_FLOOR: Final = Decimal("1e-12")
REQUIRED_ANALYSIS_BRANCHES: Final = frozenset(
    {
        "zero-01",
        "zero-02",
        "p4-plus",
        "p4-minus",
        "p5-plus",
        "p5-minus",
        "p4-plus-repeat",
        "p5-minus-repeat",
    }
)


def receiver_map(response: FreeGsnkeProcessResponse) -> dict[Decimal, dict[str, Decimal]]:
    """Project one canonical response onto receiver clock/value coordinates."""

    return {
        snapshot.receiver_clock_s: {value.value_id: value.value for value in snapshot.values}
        for snapshot in response.episode.receiver_snapshots
    }


def compute_signed_pair_metrics(
    branches: Mapping[str, FreeGsnkeProcessResponse],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compute the frozen zero/repeat and signed odd-response diagnostics.

    The function is intentionally free of storage, runtime, plotting and reveal
    concerns so the same adjudication primitive can be reused by historical
    reproduction and the independent substrate grounding target DAG.
    """

    if set(branches) != REQUIRED_ANALYSIS_BRANCHES:
        raise ValueError("analysis branch set is incomplete")
    maps = {key: receiver_map(value) for key, value in branches.items()}
    clocks = tuple(maps["zero-01"])
    if any(tuple(value) != clocks for value in maps.values()):
        raise ValueError("analysis receiver clocks differ")
    doses = {
        "p4": branches["p4-plus"].episode.action_ledger.requested_port_increments[0].value,
        "p5": branches["p5-plus"].episode.action_ledger.requested_port_increments[1].value,
    }
    rows: list[dict[str, object]] = []
    maximum_zero = Decimal(0)
    maximum_repeat = Decimal(0)
    primary_signal = {"p4": False, "p5": False}
    # Pulse end is the first frozen clock and pulse+tau the third. This is a
    # preregistered request coordinate, not an outcome-selected clock.
    primary_clocks = {clocks[0], clocks[2]}
    for clock in clocks:
        for receiver_id, unit in sorted(FREEGSNKE_RECEIVER_UNITS.items()):
            zero_1 = maps["zero-01"][clock][receiver_id]
            zero_2 = maps["zero-02"][clock][receiver_id]
            zero_difference = abs(zero_1 - zero_2)
            maximum_zero = max(maximum_zero, zero_difference)
            zero = (zero_1 + zero_2) / Decimal(2)
            floor = max(ABSOLUTE_FLOOR, Decimal(10) * zero_difference)
            row: dict[str, object] = {
                "clock_s": str(clock),
                "family": next(
                    family
                    for family, members in FREEGSNKE_RECEIVER_FAMILIES.items()
                    if receiver_id in members
                ),
                "receiver_id": receiver_id,
                "unit": unit,
                "zero_01": str(zero_1),
                "zero_02": str(zero_2),
                "zero_abs_difference": str(zero_difference),
                "numerical_floor": str(floor),
            }
            for port in ("p4", "p5"):
                plus = maps[f"{port}-plus"][clock][receiver_id]
                minus = maps[f"{port}-minus"][clock][receiver_id]
                delta_plus = plus - zero
                delta_minus = minus - zero
                slope = (plus - minus) / (Decimal(2) * doses[port])
                denominator = max(abs(delta_plus), abs(delta_minus), floor)
                oddness = abs(delta_plus + delta_minus) / denominator
                row.update(
                    {
                        f"{port}_plus_delta": str(delta_plus),
                        f"{port}_minus_delta": str(delta_minus),
                        f"{port}_signed_slope_per_v": str(slope),
                        f"{port}_oddness_residual": str(oddness),
                        f"{port}_above_floor_both_signs": abs(delta_plus) > floor
                        and abs(delta_minus) > floor,
                    }
                )
                if (
                    clock in primary_clocks
                    and math.isfinite(float(slope))
                    and abs(delta_plus) > floor
                    and abs(delta_minus) > floor
                ):
                    primary_signal[port] = True
            admission_evaluation_repeat = abs(
                maps["p4-plus"][clock][receiver_id] - maps["p4-plus-repeat"][clock][receiver_id]
            )
            prospective_validation_repeat = abs(
                maps["p5-minus"][clock][receiver_id] - maps["p5-minus-repeat"][clock][receiver_id]
            )
            maximum_repeat = max(maximum_repeat, admission_evaluation_repeat, prospective_validation_repeat)
            row["p4_plus_repeat_abs_difference"] = str(admission_evaluation_repeat)
            row["p5_minus_repeat_abs_difference"] = str(prospective_validation_repeat)
            rows.append(row)
    summary: dict[str, object] = {
        "maximum_zero_pair_abs_difference_native": str(maximum_zero),
        "maximum_exact_repeat_abs_difference_native": str(maximum_repeat),
        "zero_pair_pass": maximum_zero <= ZERO_REPEAT_TOLERANCE,
        "repeat_pass": maximum_repeat <= ZERO_REPEAT_TOLERANCE,
        "primary_signal_pass": primary_signal,
    }
    return rows, summary


__all__ = [
    "ABSOLUTE_FLOOR",
    "REQUIRED_ANALYSIS_BRANCHES",
    "ZERO_REPEAT_TOLERANCE",
    "compute_signed_pair_metrics",
    "receiver_map",
]
