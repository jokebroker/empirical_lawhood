"""Predeclared whole-root bounds and source-free power/resource qualification."""

from dataclasses import dataclass
from math import ceil, log, sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import beta, binom


Array = npt.NDArray[np.float64]
CLAIM_SLOTS = 64
FAMILY_ALPHA = 0.05
CLAIM_ALPHA = FAMILY_ALPHA / CLAIM_SLOTS
FAMILY_POWER = 0.80
MC_REPLICATES = 100_000
MC_SEED = 2026090917


def hoeffding_interval(
    values: Array, *, lower: float, upper: float, alpha: float = CLAIM_ALPHA
) -> tuple[float, float, float]:
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not lower < upper
        or np.any(values < lower)
        or np.any(values > upper)
        or not 0 < alpha < 1
    ):
        raise ValueError("bounded inference requires every independent root and its declared range")
    mean = float(values.mean())
    half = (upper - lower) * sqrt(log(1 / alpha) / (2 * len(values)))
    return mean, max(lower, mean - half), min(upper, mean + half)


def paired_adequacy_bounds(
    selected: Array,
    comparator: Array,
    selected_known: npt.NDArray[np.bool_],
    comparator_known: npt.NDArray[np.bool_],
) -> tuple[float, float, float, int]:
    if (
        selected.shape != comparator.shape
        or selected.ndim != 2
        or selected.shape[1] != 4
        or selected_known.shape != selected.shape
        or comparator_known.shape != selected.shape
        or np.any(np.isfinite(selected) & ((selected < 0) | (selected > 1)))
        or np.any(np.isfinite(comparator) & ((comparator < 0) | (comparator > 1)))
    ):
        raise ValueError(
            "paired adequacy requires root-by-four-audit operands and separate validity"
        )
    left = np.where(selected_known & np.isfinite(selected), selected, 0.0)
    right = np.where(comparator_known & np.isfinite(comparator), comparator, 1.0)
    result = hoeffding_interval(np.mean(left - right, axis=1), lower=-1, upper=1)
    unresolved = int(
        np.any(
            ~selected_known | ~comparator_known | ~np.isfinite(selected) | ~np.isfinite(comparator),
            axis=1,
        ).sum()
    )
    return *result, unresolved


def exact_binomial_upper(count: int, total: int, alpha: float = CLAIM_ALPHA) -> float:
    if (
        type(count) is not int
        or type(total) is not int
        or not 0 <= count <= total
        or total <= 0
        or not 0 < alpha < 1
    ):
        raise ValueError("exact risk bound requires independent binary root counts")
    return 1.0 if count == total else float(beta.ppf(1 - alpha, count + 1, total - count))


def preservation_noninferiority(
    selected: npt.NDArray[np.bool_], comparator: npt.NDArray[np.bool_], known: npt.NDArray[np.bool_]
) -> tuple[int, int, float, bool]:
    """Conservative paired harm-only bound; improvements cannot offset harms.

    A root is harmful if any required audit loses preservation that its paired
    comparator retains. Unresolved pairs count as harmful. Its upper risk
    bounds the excess failure probability without independent-arm subtraction.
    """
    if (
        selected.shape != comparator.shape
        or selected.shape != known.shape
        or selected.ndim != 2
        or selected.shape[1] != 4
    ):
        raise ValueError("preservation comparison changes its root/audit denominator")
    harms = np.any((~selected & comparator) | ~known, axis=1)
    count = int(harms.sum())
    upper = exact_binomial_upper(count, len(harms))
    return count, len(harms), upper, upper < 0.01


def coarse_reliability(
    probability: Array, events: Array
) -> tuple[dict[str, float | int | None], ...]:
    """Coarse-bin means with a whole-root bootstrap, including empty bins.

    This descriptive development uncertainty does not promote the exposed
    panel. Repeated parents/audits never appear as independently sampled rows.
    """
    if probability.ndim not in (1, 2) or events.shape != (*probability.shape, 4):
        raise ValueError("reliability chart changes root/parent/audit axes")
    p = probability.reshape(len(probability), -1)
    y = events.mean(axis=-1).reshape(p.shape)
    rng = np.random.default_rng(2026090918)
    bootstrap = rng.integers(len(p), size=(2048, len(p)))
    rows = []
    for index in range(4):
        low, high = index / 4, (index + 1) / 4
        chosen = np.isfinite(p) & (p >= low) & ((p < high) if index < 3 else (p <= high))
        totals, sum_p, sum_y = (
            chosen.sum(axis=1),
            np.where(chosen, p, 0).sum(axis=1),
            np.where(chosen, y, 0).sum(axis=1),
        )
        roots, cells = int((totals > 0).sum()), int(totals.sum())
        lower = upper = None
        if roots >= 8:
            denominator = totals[bootstrap].sum(axis=1)
            valid = denominator > 0
            differences = (
                sum_p[bootstrap].sum(axis=1)[valid] - sum_y[bootstrap].sum(axis=1)[valid]
            ) / denominator[valid]
            lower, upper = (float(v) for v in np.quantile(differences, [0.025, 0.975]))
        rows.append(
            {
                "bin": index,
                "lower_edge": low,
                "upper_edge": high,
                "root_count": roots,
                "parent_cells": cells,
                "mean_probability": float(sum_p.sum() / cells) if cells else None,
                "event_fraction": float(sum_y.sum() / cells) if cells else None,
                "calibration_error_lower": lower,
                "calibration_error_upper": upper,
            }
        )
    return tuple(rows)


@dataclass(frozen=True)
class PowerCell:
    roots: int
    operand: str
    design_value: float
    support_radius: float
    monte_carlo_power: float
    monte_carlo_lower: float
    analytic_power: float
    passes: bool


def _mc_lower(successes: int, total: int, alpha: float) -> float:
    return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, total - successes + 1))


def power_qualification() -> tuple[int | None, tuple[PowerCell, ...]]:
    """Validate the actual frozen bounds, not only a normal approximation.

    A union bound allocates the family type-II budget to all 64 possible
    claim slots. Variance envelopes include worst bounded paired dispersion;
    nested audit outcomes may be perfectly dependent within a root. Risk
    precision uses explicit 95% coverage, 2.5% false certification, 60%
    admission, 25% useful windows and 0.1% paired preservation-harm designs.
    """
    counts = tuple(range(3072, 8193, 512))
    rng = np.random.default_rng(MC_SEED)
    rows: list[PowerCell] = []
    required_power = 1 - (1 - FAMILY_POWER) / CLAIM_SLOTS
    # All cells are predeclared; this conservative Monte Carlo confidence
    # allocation also covers choosing N from this fixed grid.
    mc_alpha = 0.01 / (len(counts) * 11)
    for n in counts:
        at_n: list[PowerCell] = []
        for operand, effect, minimum in (
            ("adequacy-or-task", 0.15, 0.05),
            ("brier-improvement", 0.10, 0.01),
        ):
            for radius in (0.5, sqrt(0.5), 1.0):
                chance = (1 + effect / radius) / 2
                # The analysis always retains the declared [-1,1] bound, even
                # for a lower-variance planning scenario.
                cutoff = minimum + sqrt(2 * log(1 / CLAIM_ALPHA) / n)
                critical = int(np.floor(n * (1 + cutoff / radius) / 2))
                successes = rng.binomial(n, chance, size=MC_REPLICATES) > critical
                analytic = float(binom.sf(critical, n, chance))
                lower = _mc_lower(int(successes.sum()), MC_REPLICATES, mc_alpha)
                at_n.append(
                    PowerCell(
                        n,
                        operand,
                        effect,
                        radius,
                        float(successes.mean()),
                        lower,
                        analytic,
                        lower >= required_power,
                    )
                )
        for operand, total, chance, cutoff, direction in (
            (
                "simultaneous-coverage",
                n,
                0.95,
                0.90 + sqrt(log(1 / CLAIM_ALPHA) / (2 * n)),
                "above",
            ),
            ("false-certification", n // 2, 0.025, 0.10 - sqrt(log(1 / CLAIM_ALPHA) / n), "below"),
            ("minimum-admission", n, 0.60, 0.5 - 1 / n, "above"),
            ("useful-window-rate", n, 0.25, 0.05 + sqrt(log(1 / CLAIM_ALPHA) / (2 * n)), "above"),
        ):
            draws = rng.binomial(total, chance, size=MC_REPLICATES)
            if direction == "above":
                critical = int(np.floor(total * cutoff))
                successes, analytic = draws > critical, float(binom.sf(critical, total, chance))
            else:
                critical = ceil(total * cutoff) - 1
                successes, analytic = draws <= critical, float(binom.cdf(critical, total, chance))
            lower = _mc_lower(int(successes.sum()), MC_REPLICATES, mc_alpha)
            at_n.append(
                PowerCell(
                    n,
                    operand,
                    chance,
                    1.0,
                    float(successes.mean()),
                    lower,
                    analytic,
                    lower >= required_power,
                )
            )
        possible = np.arange(n + 1)
        upper = np.r_[beta.ppf(1 - CLAIM_ALPHA, possible[:-1] + 1, n - possible[:-1]), 1.0]
        acceptable = possible[upper < 0.01]
        critical = int(acceptable[-1]) if len(acceptable) else -1
        draws = rng.binomial(n, 0.001, size=MC_REPLICATES)
        successes, analytic = draws <= critical, float(binom.cdf(critical, n, 0.001))
        lower = _mc_lower(int(successes.sum()), MC_REPLICATES, mc_alpha)
        at_n.append(
            PowerCell(
                n,
                "paired-preservation-harm",
                0.001,
                1.0,
                float(successes.mean()),
                lower,
                analytic,
                lower >= required_power,
            )
        )
        rows.extend(at_n)
    selected = next((n for n in counts if all(row.passes for row in rows if row.roots == n)), None)
    return selected, tuple(rows)
