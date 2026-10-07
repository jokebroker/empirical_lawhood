"""Frozen root-level inference for the prepared-interface tranche.

CP bounds apply only to the declared iid context populations. Conditional
forecast panels and R's equally weighted fixed problem mixture use independent
Bernoulli Chernoff bounds, allowing heterogeneous success probabilities.
"""

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import ClassVar

from scipy.stats import beta

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id


@dataclass(frozen=True, slots=True)
class PreparedStatisticalSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-statistical-spec'
    spec_id: str
    alpha: Decimal = Decimal("0.05")
    calibration_roots_per_context: int = 96
    evaluation_roots_per_context: int = 128
    reuse_problems: int = 12
    reuse_roots_per_problem: int = 16
    minimum_success: Decimal = Decimal("0.75")
    maximum_conditional_risk: Decimal = Decimal("0.05")
    minimum_paired_increase: Decimal = Decimal("0.15")
    constant_command_increase: Decimal = Decimal("0.20")
    coverage_noninferiority: Decimal = Decimal("-0.05")
    calibration_error: Decimal = Decimal("0.10")
    minimum_planning_power: Decimal = Decimal("0.80")
    probability_clip: Decimal = Decimal("0.000001")
    threshold_order: tuple[Decimal, ...] = (Decimal("0.975"), Decimal("0.95"), Decimal("0.90"))
    cp_tail: str = "EXACT_ONE_SIDED_INCLUSIVE_BINOMIAL"
    paired_bound: str = "DISCORDANT_CP_LOWER_MINUS_UPPER_ALPHA_HALVES"
    heterogeneous_bound: str = "INDEPENDENT_BERNOULLI_KL_CHERNOFF_INVERSION"
    missingness: str = "ALL_ASSIGNED_ROOTS_UNKNOWN_NOT_SUCCESS_ADMITTED_UNKNOWN_IS_FAILURE"
    core_multiplicity: str = "INTERSECTION_UNION_BOTH_CONTEXTS"
    secondary_multiplicity: str = "DEVELOPMENT_FROZEN_HOLM_OR_DESCRIPTIVE"
    reuse_population: str = "EQUAL_WEIGHT_FIXED_TWELVE_PROBLEM_MIXTURE"

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        expected_decimals = {
            "alpha": "0.05",
            "minimum_success": "0.75",
            "maximum_conditional_risk": "0.05",
            "minimum_paired_increase": "0.15",
            "constant_command_increase": "0.20",
            "coverage_noninferiority": "-0.05",
            "calibration_error": "0.10",
            "minimum_planning_power": "0.80",
            "probability_clip": "0.000001",
        }
        for name, expected in expected_decimals.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value != Decimal(expected)
            ):
                raise ValueError(f"prepared statistical lock changes {name}")
        for name, expected_count in (
            ("calibration_roots_per_context", 96),
            ("evaluation_roots_per_context", 128),
            ("reuse_problems", 12),
            ("reuse_roots_per_problem", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != expected_count:
                raise ValueError(f"prepared statistical lock changes {name}")
        if (
            self.threshold_order != (Decimal("0.975"), Decimal("0.95"), Decimal("0.90"))
            or any(not isinstance(v, Decimal) for v in self.threshold_order)
            or self.cp_tail != "EXACT_ONE_SIDED_INCLUSIVE_BINOMIAL"
            or self.paired_bound != "DISCORDANT_CP_LOWER_MINUS_UPPER_ALPHA_HALVES"
            or self.heterogeneous_bound != "INDEPENDENT_BERNOULLI_KL_CHERNOFF_INVERSION"
            or self.missingness
            != "ALL_ASSIGNED_ROOTS_UNKNOWN_NOT_SUCCESS_ADMITTED_UNKNOWN_IS_FAILURE"
            or self.core_multiplicity != "INTERSECTION_UNION_BOTH_CONTEXTS"
            or self.secondary_multiplicity != "DEVELOPMENT_FROZEN_HOLM_OR_DESCRIPTIVE"
            or self.reuse_population != "EQUAL_WEIGHT_FIXED_TWELVE_PROBLEM_MIXTURE"
        ):
            raise ValueError("prepared statistical lock changes its inferential rules")


def _counts(k: int, n: int, alpha: float) -> None:
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n < 1:
        raise ValueError("binary bound requires a nonempty exact independent-root count")
    if not math.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("binary tail probability must lie strictly between zero and one half")


def cp_bounds(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Each endpoint is a one-sided (1-alpha) exact iid binomial bound."""
    _counts(k, n, alpha)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.isf(alpha, k + 1, n - k))
    return lower, upper


def _binary_kl(q: float, p: float) -> float:
    if q == 0:
        return -math.log1p(-p) if p < 1 else math.inf
    if q == 1:
        return -math.log(p) if p > 0 else math.inf
    if p <= 0 or p >= 1:
        return math.inf
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def independent_bernoulli_mean_bounds(
    k: int, n: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Chernoff/Jensen inversion; no common Bernoulli parameter is assumed.

    Equal unit weights are required. At fixed n each endpoint has one-sided
    error at most alpha. There is no sequential stopping guarantee.
    """
    _counts(k, n, alpha)
    q, limit = k / n, -math.log(alpha) / n
    if k == 0:
        return 0.0, -math.expm1(math.log(alpha) / n)
    if k == n:
        return math.exp(math.log(alpha) / n), 1.0
    left, right = 0.0, q
    for _ in range(80):
        midpoint = (left + right) / 2
        if _binary_kl(q, midpoint) > limit:
            left = midpoint
        else:
            right = midpoint
    lower = left  # round outwards, preserving conservatism
    left, right = q, 1.0
    for _ in range(80):
        midpoint = (left + right) / 2
        if _binary_kl(q, midpoint) > limit:
            right = midpoint
        else:
            left = midpoint
    return lower, right


def paired_difference_bounds(
    law_only: int,
    comparator_only: int,
    n: int,
    *,
    alpha: float = 0.05,
    heterogeneous: bool = False,
) -> tuple[float, float, float]:
    _counts(law_only, n, alpha)
    _counts(comparator_only, n, alpha)
    if law_only + comparator_only > n or type(heterogeneous) is not bool:
        raise ValueError("paired discordant cells change the common-root denominator")
    bound = independent_bernoulli_mean_bounds if heterogeneous else cp_bounds
    b_lower, b_upper = bound(law_only, n, alpha=alpha / 2)
    c_lower, c_upper = bound(comparator_only, n, alpha=alpha / 2)
    return (law_only - comparator_only) / n, b_lower - c_upper, b_upper - c_lower


def heterogeneous_conditional_risk_upper(
    admitted_failures: int,
    admissions: int,
    assigned_roots: int,
    *,
    alpha: float = 0.05,
) -> float | None:
    _counts(admissions, assigned_roots, alpha)
    if type(admitted_failures) is not int or not 0 <= admitted_failures <= admissions:
        raise ValueError("admitted-failure numerator lies outside the admission denominator")
    if admissions == 0:
        return None
    failure_upper = independent_bernoulli_mean_bounds(
        admitted_failures, assigned_roots, alpha=alpha / 2
    )[1]
    admission_lower = independent_bernoulli_mean_bounds(
        admissions, assigned_roots, alpha=alpha / 2
    )[0]
    return min(1.0, failure_upper / admission_lower) if admission_lower > 0 else None


def simultaneous_constant_word_contrast(
    task_successes: int,
    roots: int,
    audit_word_counts: tuple[tuple[int, int], ...],
) -> tuple[float, float] | None:
    """Nine preassigned audit-word rates versus the primary task population.

    Cross-word independence is unnecessary for Bonferroni. Counts refer to the
    independently assigned direction subsets, including all failed handoffs.
    A missing assigned subset supplies no superiority claim.
    """
    _counts(task_successes, roots, 0.025)
    if len(audit_word_counts) != 9:
        raise ValueError("constant-command comparison requires all nine declared words")
    for k, n in audit_word_counts:
        if type(k) is not int or type(n) is not int or not 0 <= k <= n <= roots:
            raise ValueError("constant-command audit subset count is invalid")
    if (
        audit_word_counts[0][1] != roots
        or any(audit_word_counts[i][1] != audit_word_counts[i + 1][1] for i in (1, 3, 5, 7))
        or sum(audit_word_counts[i][1] for i in (1, 3, 5, 7)) != roots
    ):
        raise ValueError(
            "constant-command counts change the four-direction randomized triplet census"
        )
    if any(n == 0 for _, n in audit_word_counts):
        return None
    point = task_successes / roots - max(k / n for k, n in audit_word_counts)
    lower = cp_bounds(task_successes, roots, alpha=0.025)[0] - max(
        cp_bounds(k, n, alpha=0.025 / 9)[1] for k, n in audit_word_counts
    )
    return point, lower


def conformal_order_index(n: int, *, level: Decimal = Decimal("0.95")) -> int:
    """One-based order including the conventional extra infinite score."""
    if (
        type(n) is not int
        or n < 1
        or not isinstance(level, Decimal)
        or not level.is_finite()
        or not 0 < level < 1
    ):
        raise ValueError("conformal order requires a finite calibration census and level")
    return int((level * (n + 1)).to_integral_value(rounding="ROUND_CEILING"))
