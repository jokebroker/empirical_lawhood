"""Fixed branch response and shadow contact summaries; no superiority verdict."""

from dataclasses import dataclass
import numpy as np
from .numerical import Decision


@dataclass(frozen=True)
class BranchReadout:
    assigned_roots: int
    contacted_pairs: int
    invalid_pairs: int
    mean: tuple[float, ...] | None
    lower: tuple[float, ...] | None
    upper: tuple[float, ...] | None


def branch_readout(
    differences: np.ndarray, contacted: np.ndarray, valid: np.ndarray
) -> BranchReadout:
    # root, intervention (F/J), elapsed horizon (10/60/300), receiver (T/C)
    if differences.shape != (24, 2, 3, 2) or contacted.shape != (24, 2) or valid.shape != (24, 2):
        raise ValueError("complete 24-root paired branch census required")
    if not np.isfinite(differences).all() or not valid.all():
        return BranchReadout(24, int(contacted.sum()), int((~valid).sum()), None, None, None)
    rng = np.random.Generator(np.random.PCG64(86101))
    draws = np.empty((10000, 24), dtype=int)
    for stratum in range(4):
        roots = np.arange(stratum, 24, 4)
        draws[:, stratum * 6 : (stratum + 1) * 6] = rng.choice(roots, (10000, 6), replace=True)
    sampled = differences[draws].mean(axis=1)
    lo, hi = np.quantile(sampled, (0.025, 0.975), axis=0)
    return BranchReadout(
        24,
        int(contacted.sum()),
        0,
        tuple(differences.mean(axis=0).ravel()),
        tuple(lo.ravel()),
        tuple(hi.ravel()),
    )


@dataclass(frozen=True)
class ShadowContact:
    queries: int
    refused: int
    distinct_alternative_queries: int
    alternative_fraction: float
    rival_choice_differences: tuple[int, ...]


def shadow_contact(
    primary: tuple[Decision, ...], rivals: tuple[tuple[Decision, ...], ...]
) -> ShadowContact:
    if not primary or any(len(r) != len(primary) for r in rivals):
        raise ValueError("shadow query census differs")
    alternatives = sum(
        len({c.projection.delivery_key for c in d.candidates if not c.reasons}) >= 2
        for d in primary
    )
    return ShadowContact(
        len(primary),
        sum(d.selected is None for d in primary),
        alternatives,
        alternatives / len(primary),
        tuple(
            sum(a.selected != b.selected for a, b in zip(primary, r, strict=True)) for r in rivals
        ),
    )
