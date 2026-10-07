# SPDX-License-Identifier: MPL-2.0
"""Stdlib-only operational admission for the two editable native canaries.

The byte allowances cover retained numeric arrays, monitor copies and output
buffers. They are conservative admission estimates, not process RSS bounds or
scientific accuracy criteria. Native compiler/runtime overhead is separate.
This file is also imported by the standalone Brian2 worker in its own env.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math


MAX_NATIVE_UPDATES_PER_ARM = 100_000
TORAX_MAX_HISTORY_CELLS = 1_000_000
TORAX_MAX_ESTIMATED_HISTORY_BYTES = 64 * 1024**2
TORAX_ESTIMATED_BYTES_PER_RADIAL_CELL = 512
TORAX_ESTIMATED_BYTES_PER_RETAINED_SAMPLE = 1024
BRIAN2_MAX_ESTIMATED_HISTORY_BYTES = 16 * 1024**2
BRIAN2_ESTIMATED_BYTES_PER_UPDATE = 128


@dataclass(frozen=True, slots=True)
class NativeTimeGridAdmission:
    update_count: int
    sample_count: int
    interval_updates: tuple[int, ...]
    estimated_history_bytes: int


def admit_native_time_grid(
    *,
    timestep: Decimal,
    intervals: tuple[Decimal, ...],
    label: str,
    require_integral_intervals: bool,
    sample_offset: int,
    estimated_bytes_per_sample: int,
    maximum_estimated_bytes: int,
) -> NativeTimeGridAdmission:
    """Admit a clock without rounded Decimal division or native allocation.

Units are supplied by each substrate: seconds for TORAX, milliseconds for
Brian2. Partial final TORAX steps count as full updates. Brian2 preserves the
exact integer tick count of each interval and its floating-point worker clock.
"""

    if not intervals:
        raise ValueError(f"{label} native time grid requires a duration")
    native_times: list[float] = []
    for value in (timestep, *intervals):
        if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
            raise ValueError(f"{label} native time grid requires finite positive times")
        native = float(value)
        if not math.isfinite(native) or native <= 0:
            raise ValueError(f"{label} native time grid is not representable in float64")
        native_times.append(native)
    native_dt, *native_intervals = native_times
    try:
        native_duration = math.fsum(native_intervals)
    except OverflowError as exc:
        raise ValueError(f"{label} native time grid is not representable in float64") from exc
    if not math.isfinite(native_duration) or native_duration + native_dt <= native_duration:
        raise ValueError(f"{label} native time grid cannot advance its float64 clock")

    dt = Fraction(timestep)
    ratios = tuple(Fraction(value) / dt for value in intervals)
    total = sum(ratios, Fraction())
    # Compare before constructing a potentially enormous update-count integer.
    if total > MAX_NATIVE_UPDATES_PER_ARM:
        raise ValueError(f"{label} native time grid exceeds {MAX_NATIVE_UPDATES_PER_ARM} updates per arm")
    if require_integral_intervals and any(value.denominator != 1 for value in ratios):
        raise ValueError(f"{label} native time grid requires exact integral intervals")
    interval_updates = tuple(-(-value.numerator // value.denominator) for value in ratios)
    exact_updates = -(-total.numerator // total.denominator)
    native_ratio = native_duration / native_dt
    if require_integral_intervals:
        if round(native_ratio) != exact_updates or any(
            round(value / native_dt) != count
            for value, count in zip(native_intervals, interval_updates, strict=True)
        ):
            raise ValueError(f"{label} native time grid changes tick counts in float64")
        updates = exact_updates
    else:
        updates = max(exact_updates, math.ceil(native_ratio))
    if updates > MAX_NATIVE_UPDATES_PER_ARM:
        raise ValueError(f"{label} native time grid exceeds {MAX_NATIVE_UPDATES_PER_ARM} updates per arm")
    samples = updates + sample_offset
    estimated_bytes = samples * estimated_bytes_per_sample
    if estimated_bytes > maximum_estimated_bytes:
        raise ValueError(f"{label} native history estimate exceeds {maximum_estimated_bytes} bytes per arm")
    return NativeTimeGridAdmission(updates, samples, interval_updates, estimated_bytes)
