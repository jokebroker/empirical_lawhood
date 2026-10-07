"""Typed projector-jump semantics that cannot conflate Born mass with norm error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import FixedNumberBasis
from .operators import normalize_state
from .types import ComplexVector


class ZeroProjectedMassError(ValueError):
    """Raised before normalization when the selected projector has zero mass."""


@dataclass(frozen=True, slots=True)
class JumpMasses:
    projected_masses: np.ndarray
    mark_probabilities: np.ndarray
    mass_sum_residual: float
    probability_sum_residual: float


@dataclass(frozen=True, slots=True)
class AppliedJump:
    state: ComplexVector
    projected_mass_expectation: float
    projected_vector_norm_squared: float
    projected_mass_identity_residual: float
    post_jump_norm_residual: float


def jump_masses(basis: FixedNumberBasis, state: ComplexVector) -> JumpMasses:
    weights = np.abs(state) ** 2
    masses = np.asarray(weights @ basis.occupations, dtype=np.float64)
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("projected mass vector is invalid")
    probabilities = np.asarray(masses / basis.particles, dtype=np.float64)
    return JumpMasses(
        projected_masses=masses,
        mark_probabilities=probabilities,
        mass_sum_residual=abs(float(masses.sum()) - basis.particles),
        probability_sum_residual=abs(float(probabilities.sum()) - 1.0),
    )


def sample_mark(probabilities: np.ndarray, uniform: float) -> int:
    """Use exactly one [0,1) variate and the frozen right-boundary rule."""

    if (
        probabilities.ndim != 1
        or len(probabilities) == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("mark probabilities are invalid")
    if not np.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("mark uniform must lie in [0,1)")
    total = float(probabilities.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError("mark probabilities are not normalized")
    cumulative = np.cumsum(probabilities)
    site = int(np.searchsorted(cumulative, uniform, side="right"))
    if site >= len(probabilities):
        site = len(probabilities) - 1
    if probabilities[site] <= 0.0:
        positive = np.flatnonzero(probabilities > 0.0)
        following = positive[positive > site]
        if len(following):
            site = int(following[0])
        elif len(positive):
            site = int(positive[-1])
        else:
            raise ValueError("mark distribution has no positive site")
    return site


def apply_jump(
    basis: FixedNumberBasis,
    state: ComplexVector,
    site: int,
    *,
    expected_mass: float | None = None,
) -> AppliedJump:
    if not 0 <= site < basis.l_sites:
        raise ValueError("jump site leaves the chain")
    expectation = (
        float((np.abs(state) ** 2) @ basis.occupations[:, site])
        if expected_mass is None
        else float(expected_mass)
    )
    projected = np.asarray(state * basis.occupations[:, site], dtype=np.complex128)
    vector_norm_squared = float(np.vdot(projected, projected).real)
    residual = abs(vector_norm_squared - expectation)
    if expectation <= 0.0 or vector_norm_squared <= 0.0:
        raise ZeroProjectedMassError("ZERO_PROJECTED_MASS")
    normalized, post_residual = normalize_state(projected / np.sqrt(expectation))
    return AppliedJump(
        state=normalized,
        projected_mass_expectation=expectation,
        projected_vector_norm_squared=vector_norm_squared,
        projected_mass_identity_residual=residual,
        post_jump_norm_residual=post_residual,
    )


__all__ = [
    "AppliedJump",
    "JumpMasses",
    "ZeroProjectedMassError",
    "apply_jump",
    "jump_masses",
    "sample_mark",
]
