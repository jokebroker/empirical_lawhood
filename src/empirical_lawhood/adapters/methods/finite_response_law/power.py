# SPDX-License-Identifier: MPL-2.0
"""Fixed-sample sensitivity for response composition under the frozen design.

This diagnostic draws no roots and reads no observed responses. Exact discrete
probabilities assume independent, identically distributed root categories.
The separate prediction comparison uses a normal approximation.
"""

from __future__ import annotations

import math

from scipy.stats import binom, norm

from empirical_lawhood.planning.paired_power import paired_binary_power

from .science import PLAN_SHA256


EVALUATION_ROOTS = 64
REQUIRED_SUCCESSES = 52
MAXIMUM_FALSE_ADMISSIONS = 2
SUCCESS_PROBABILITIES = (0.80, 0.85, 0.90)
FALSE_ADMISSION_PROBABILITIES = (0.0, 0.01, 0.02, 0.05)
PAIRED_PROBABILITIES = ((0.10, 0.0), (0.15, 0.05), (0.20, 0.05), (0.25, 0.05), (0.30, 0.10))
PREDICTION_EFFECTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def joint_success_risk_power(
    n: int,
    required_successes: int,
    maximum_false_admissions: int,
    p_success: float,
    p_false: float,
) -> float:
    """Probability of both count gates for mutually exclusive root categories.

Condition on the binomial success count. False admissions among the remaining
roots have probability ``p_false / (1 - p_success)``. This sums the same
multinomial event without treating success and risk as independent gates.
"""

    if (
        any(not isinstance(v, int) or isinstance(v, bool)
            for v in (n, required_successes, maximum_false_admissions))
        or not 1 <= n <= EVALUATION_ROOTS
        or not 0 <= required_successes <= n
        or not 0 <= maximum_false_admissions <= n
        or not math.isfinite(p_success)
        or not math.isfinite(p_false)
        or not 0 <= p_success <= 1
        or not 0 <= p_false <= 1 - p_success
    ):
        raise ValueError("invalid bounded success/risk design parameters")
    if p_success == 1:
        return 1.0
    conditional_false = p_false / (1 - p_success)
    return math.fsum(
        float(binom.pmf(successes, n, p_success))
        * float(binom.cdf(maximum_false_admissions, n - successes, conditional_false))
        for successes in range(required_successes, n + 1)
    )


def fixed_sample_power_report() -> dict[str, object]:
    """Assemble the required pre-entry report without changing design operands."""

    cases = [
        {
            "p_success": success,
            "p_false": false,
            "joint_use_risk_power": joint_success_risk_power(
                EVALUATION_ROOTS, REQUIRED_SUCCESSES, MAXIMUM_FALSE_ADMISSIONS, success, false,
            ),
            "success_marginal_power": float(binom.sf(REQUIRED_SUCCESSES - 1, EVALUATION_ROOTS, success)),
            "risk_marginal_power": float(binom.cdf(MAXIMUM_FALSE_ADMISSIONS, EVALUATION_ROOTS, false)),
        }
        for success in SUCCESS_PROBABILITIES
        for false in FALSE_ADMISSION_PROBABILITIES
    ]
    return {
        "schema": 'empirical-lawhood/response-composition/fixed-sample-power',
        "specification_sha256": PLAN_SHA256,
        "n": EVALUATION_ROOTS,
        "required_successes": REQUIRED_SUCCESSES,
        "maximum_false_admissions": MAXIMUM_FALSE_ADMISSIONS,
        "multinomial_cases": cases,
        "paired_use_sensitivity": [
            {
                "p_improve": better,
                "p_deteriorate": worse,
                "exact_paired_added_use_power": paired_binary_power(
                    n=EVALUATION_ROOTS, delta=better - worse,
                    discordance=better + worse, alpha=0.05, materiality=0.0,
                ),
            }
            for better, worse in PAIRED_PROBABILITIES
        ],
        "prediction_comparison_sensitivity": [
            {
                "mean_paired_improvement_over_root_sd": effect,
                "normal_approximation_one_sided_power": float(norm.cdf(
                    effect * math.sqrt(EVALUATION_ROOTS) - norm.ppf(0.95),
                )),
            }
            for effect in PREDICTION_EFFECTS
        ],
        "assumptions": (
            "Independent, identically distributed roots; success, false admission and other failure "
            "are mutually exclusive categories. Paired sensitivity uses improve minus deteriorate "
            "as delta and their sum as discordance, at one-sided alpha=0.05 without a materiality gate."
        ),
        "limits": (
            "Prediction approximation assumes finite root variance and normal mean behavior; "
            "it does not simulate the frozen percentile bootstrap or its additional 10% relevance gate. "
            "Full-conjunction power is at most the smallest marginal power. With use/risk power p "
            "and information power i, unconstrained dependence gives bounds max(0,p+i-1) to min(p,i). "
            "No identified joint information-power distribution or 80% full-conjunction guarantee; "
            "sample sizes remain fixed."
        ),
        "evidence_ceiling": (
            "Design sensitivity only; no nomination, fitted-package qualification, authority, "
            "new cohort or prospective support. Fresh entry also requires the declared development "
            "gates, final package freeze and technical prerequisites."
        ),
        "native_contact": False,
        "outcome_reads": False,
        "campaign_issued": False,
    }


__all__ = ["fixed_sample_power_report", "joint_success_risk_power"]
