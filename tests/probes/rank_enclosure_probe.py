# SPDX-License-Identifier: MPL-2.0
"""Independent enclosure audit; exit nonzero while any tested bound is inward.

This synthetic mathematical-property probe is also exercised by pytest.
It does not execute or qualify a scientific campaign.
"""

from __future__ import annotations

from fractions import Fraction
import json

from flint import arb, ctx, fmpq
import mpmath as mp
import numpy as np

from empirical_lawhood.adapters.methods._arb import endpoint_decimal
from empirical_lawhood.adapters.methods.receiver_history_closure import discrete_rank as receiver
from empirical_lawhood.adapters.methods.split_cohort_history_budget import discrete_rank as split_cohort
from empirical_lawhood.adapters.methods.history_budget_phase_diagram import discrete_rank as phase_diagram


FAMILIES = {"receiver-history": receiver, "split-cohort": split_cohort, "phase-diagram": phase_diagram}


def exact_endpoint(ball: arb, *, upper: bool) -> Fraction:
    # Independent reference: exact midpoint +/- exact stored radius, without
    # using the producer's lower()/upper() or dyadic-to-Decimal conversion.
    midpoint, radius = ball.mid().fmpq(), ball.rad().fmpq()
    mid = Fraction(int(midpoint.p), int(midpoint.q))
    rad = Fraction(int(radius.p), int(radius.q))
    return mid + rad if upper else mid - rad


def observe_endpoints(family: str, digits: int, *, threshold: bool = False) -> dict[str, object]:
    with ctx.workprec(384), mp.workdps(digits):
        # The second ball is centered at the exact rank-threshold factor 2^-160.
        ball = arb(fmpq(1, 2 ** 160)) if threshold else arb(1)
        ball += arb("0 +/- 1e-100")
        lower = Fraction(endpoint_decimal(ball, upper=False))
        upper = Fraction(endpoint_decimal(ball, upper=True))
        exact_lower = exact_endpoint(ball, upper=False)
        exact_upper = exact_endpoint(ball, upper=True)
        return {
            "family": family, "arb_bits": 384, "mpmath_digits": digits,
            "ball": "2^-160 +/- 1e-100" if threshold else "1 +/- 1e-100",
            "lower_outward": lower <= exact_lower,
            "upper_outward": upper >= exact_upper,
            "reference_width": str(exact_upper - exact_lower),
            "converted_width": str(upper - lower),
        }


def observe_residuals(family: str, digits: int) -> dict[str, object]:
    module = FAMILIES[family]
    with ctx.workprec(384), mp.workdps(digits):
        matrix = np.array([[0.1, 1e-90], [0.3, 0.5]], dtype=np.float64)
        lower, upper, residual, separation = module.residual_certified_rank_bracket(matrix)
        return {"family": family, "arb_initial_bits": 384, "requested_rank_bits": 256,
                "mpmath_digits": digits, "rank": [lower, upper],
                "residual": str(residual), "separation": None if separation is None else str(separation)}


def main() -> int:
    endpoints = [observe_endpoints(family, digits, threshold=threshold)
                 for family in FAMILIES for digits in (15, 180) for threshold in (False, True)]
    residuals = [observe_residuals(family, digits) for family in FAMILIES for digits in (15, 180)]
    failed = [row for row in endpoints if not row["lower_outward"] or not row["upper_outward"]]
    print(json.dumps({"evidence_ceiling": "synthetic enclosure acceptance; no qualification",
                      "endpoints": endpoints, "residual_observations": residuals,
                      "certificate_containment_failed": bool(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
