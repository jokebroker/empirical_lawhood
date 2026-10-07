"""Substrate-owned four-family preparation schedules."""

from __future__ import annotations

from math import isfinite


FOUR_FAMILY_PREPARATION_IDS = (
    "matrix-history.joint-increasing-coupling",
    "matrix-history.joint-decreasing-coupling",
    "matrix-history.x-first-increasing-coupling",
    "matrix-history.y-first-increasing-coupling",
)


def four_family_preparation_couplings(
    family_id: str,
    native_step: int,
    *,
    ramp_steps: int,
    integrator_multiplier: int,
    target_x: float,
    target_y: float,
    decreasing_coupling_start: float = 8.0,
) -> tuple[float, float]:
    """Return the native X/Y couplings for one preparation tick.

    ``ramp_steps`` is expressed on the primary receiver clock.  A refined
    numerical view supplies its integer ``integrator_multiplier`` so that both
    views describe the same preparation history without becoming new units.
    """

    if (
        family_id not in FOUR_FAMILY_PREPARATION_IDS
        or not isinstance(native_step, int)
        or isinstance(native_step, bool)
        or native_step < 0
        or not isinstance(ramp_steps, int)
        or isinstance(ramp_steps, bool)
        or ramp_steps <= 0
        or not isinstance(integrator_multiplier, int)
        or isinstance(integrator_multiplier, bool)
        or integrator_multiplier <= 0
        or not all(isfinite(value) for value in (target_x, target_y, decreasing_coupling_start))
    ):
        raise ValueError("four-family preparation inputs differ")
    ramp = ramp_steps * integrator_multiplier
    fraction = min(native_step, ramp) / ramp
    if family_id == "matrix-history.joint-increasing-coupling":
        return target_x * fraction, target_y * fraction
    if family_id == "matrix-history.joint-decreasing-coupling":
        return (
            decreasing_coupling_start + (target_x - decreasing_coupling_start) * fraction,
            decreasing_coupling_start + (target_y - decreasing_coupling_start) * fraction,
        )
    first = min(1.0, 2.0 * fraction)
    second = max(0.0, min(1.0, 2.0 * fraction - 1.0))
    if family_id == "matrix-history.x-first-increasing-coupling":
        return target_x * first, target_y * second
    return target_x * second, target_y * first


__all__ = [
    "FOUR_FAMILY_PREPARATION_IDS",
    'four_family_preparation_couplings',
]
