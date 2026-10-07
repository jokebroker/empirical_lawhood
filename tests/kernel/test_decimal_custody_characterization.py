# SPDX-License-Identifier: MPL-2.0
"""Regression acceptance for lossless, context-independent canonical Decimal."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, getcontext, localcontext
from fractions import Fraction
from hashlib import sha256
import json
from typing import ClassVar

import pytest

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalizationError, CanonicalRecord, canonical_json_bytes


@dataclass(frozen=True, slots=True)
class DecimalOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/decimal-custody-characterization"
    operand: Decimal


@dataclass(frozen=True, slots=True)
class NestedOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/nested-decimals"
    operands: tuple[DecimalOperand, ...]
    values: frozenset[Decimal]


LONG_OPERAND = Decimal("1.23456789012345678901234567890123456789")
VALUES = (
    LONG_OPERAND, Decimal("-12345678901234567890.9876543210987654321"),
    Decimal("1e300"), Decimal("1e-1000"), Decimal("-0.000"), Decimal("1.23000"),
)


def _context_state() -> tuple[object, ...]:
    c = getcontext()
    return (c.prec, c.rounding, c.Emin, c.Emax, c.capitals, c.clamp, dict(c.traps), dict(c.flags))


@pytest.mark.parametrize("precision", (1, 10, 28, 80))
@pytest.mark.parametrize("rounding", (ROUND_HALF_EVEN, ROUND_FLOOR, ROUND_CEILING))
def test_values_bytes_and_fingerprints_ignore_arithmetic_context(precision, rounding):
    record = NestedOperands(tuple(DecimalOperand(value) for value in VALUES), frozenset(VALUES))
    # Independently stipulated fixed-point spellings, with exact rational values.
    spellings = (
        "1.23456789012345678901234567890123456789", "-12345678901234567890.9876543210987654321",
        "1" + "0" * 300, "0." + "0" * 999 + "1", "0", "1.23",
    )
    expected = record.canonical_bytes()
    before = _context_state()
    with localcontext() as context:
        context.prec, context.rounding = precision, rounding
        context.Emin, context.Emax, context.clamp = -2, 2, 1
        for signal in context.traps:
            context.traps[signal] = True
        settings = _context_state()
        assert record.canonical_bytes() == expected
        assert record.fingerprint() == sha256(expected).hexdigest()
        assert record.to_document() == json.loads(expected)
        assert canonical_json_bytes(record) == expected
        decoded = decode_canonical_bytes(expected, NestedOperands, maximum_bytes=len(expected))
        assert decoded == record
        for value, text in zip(VALUES, spellings, strict=True):
            fresh = DecimalOperand(value)
            payload = fresh.canonical_bytes()
            assert json.loads(payload)["value"]["operand"] == {"decimal": text}
            assert canonical_json_bytes({"x": (value,)}) == (f'{{"x":[{{"decimal":"{text}"}}]}}\n').encode()
            assert Fraction(text) == Fraction(value)
            assert decode_canonical_bytes(payload, DecimalOperand, maximum_bytes=len(payload)).operand == value
        assert _context_state() == settings
    assert _context_state() == before


@pytest.mark.parametrize("spellings", (("-0", "0.0", "0e10000000"), ("1.23", "1.230000", "123e-2")))
def test_equal_numerical_values_have_identical_bytes(spellings):
    records = [DecimalOperand(Decimal(text)) for text in spellings]
    assert len({record.canonical_bytes() for record in records}) == 1
    assert len({record.fingerprint() for record in records}) == 1


@pytest.mark.parametrize("value", ("NaN", "sNaN", "Infinity", "-Infinity"))
def test_nonfinite_values_refuse_without_signalling_or_changing_context(value):
    before = _context_state()
    with pytest.raises(CanonicalizationError, match="non-finite"):
        DecimalOperand(Decimal(value)).canonical_bytes()
    assert _context_state() == before
