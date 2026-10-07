# SPDX-License-Identifier: MPL-2.0
"""Lossless boundary from rigorous Arb intervals to stored Decimal bounds."""

from decimal import Decimal

from flint import arb


def endpoint_decimal(value: arb, *, upper: bool) -> Decimal:
    """Return an outward endpoint exactly, without decimal or mpmath arithmetic.

    Arb lower()/upper() round in the indicated direction. Their finite point
    results are dyadics m*2**e. For e < 0, m*5**(-e)*10**e is the same number;
    the integer and tuple Decimal constructors preserve it at any context.
    """
    if not value.is_finite():
        raise ValueError("a finite Arb enclosure is required")
    endpoint = value.upper() if upper else value.lower()
    mantissa, exponent = (int(part) for part in endpoint.man_exp())
    if exponent >= 0:
        return Decimal(mantissa << exponent)
    coefficient = Decimal(mantissa * 5 ** (-exponent)).as_tuple()
    return Decimal((coefficient.sign, coefficient.digits, exponent))
