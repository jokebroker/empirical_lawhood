# SPDX-License-Identifier: MPL-2.0
"""Bound malformed canonical input while preserving broader document decoding."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, Inexact, Overflow, localcontext
import json
import subprocess
import sys
from typing import ClassVar

import pytest

from empirical_lawhood.kernel import decoding, serialization
from empirical_lawhood.kernel.decoding import decode_canonical_bytes, decode_canonical_record
from empirical_lawhood.kernel.serialization import CanonicalRecord, CanonicalizationError


@dataclass(frozen=True, slots=True)
class DecimalEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/decimal-decoding-bounds"
    amount: Decimal
    enabled: bool = True
    count: int = 1


def _document(text: str) -> dict[str, object]:
    return {"schema": DecimalEnvelope.SCHEMA, "version": "1.0.0",
            "value": {"amount": {"decimal": text}, "count": 1, "enabled": True}}


def _payload(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize("text", ("0", "1", "-1", "1000", "-1000", "0.01", "-0.01", "123.0001",
                                   "1.23456789012345678901234567890123456789", "0." + "0" * 99 + "1"))
def test_canonical_fixed_point_operands_keep_exact_bytes_and_values(text: str) -> None:
    expected = _payload(_document(text))
    with localcontext() as context:
        context.prec = 120
        record = decode_canonical_bytes(expected, DecimalEnvelope, maximum_bytes=len(expected))
        assert record.amount == Decimal(text)
        assert record.canonical_bytes() == expected
        assert decode_canonical_record(_document(text), DecimalEnvelope) == record


@pytest.mark.parametrize("text", ("1e100000", "1e10000000", "1e-10000000", "1E+2", "+1", "01", "00.1",
                                   "-0", "-0.0", "1.0", "1.2300", ".1", "1.", " 1", "1 ", "١", "NaN", "Infinity"))
def test_noncanonical_decimal_refuses_before_reserialization(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_expansion(_value: Decimal) -> str:
        pytest.fail("noncanonical token reached fixed-point serialization")

    with monkeypatch.context() as patch:
        patch.setattr(serialization, "_decimal_text", forbidden_expansion)
        with pytest.raises(CanonicalizationError, match="spelling is not canonical"):
            decode_canonical_bytes(_payload(_document(text)), DecimalEnvelope, maximum_bytes=4096)
    good = _payload(_document("1"))
    assert decode_canonical_bytes(good, DecimalEnvelope, maximum_bytes=4096).amount == 1


@pytest.mark.parametrize("text", ("1e2", "+1", "01", "1.2300", "-0", " 1 "))
def test_document_acceptance_is_preserved_after_exact_refusal(text: str) -> None:
    document = _document(text)
    assert decode_canonical_record(document, DecimalEnvelope).amount == Decimal(text)
    with pytest.raises(CanonicalizationError):
        decode_canonical_bytes(_payload(document), DecimalEnvelope, maximum_bytes=4096)
    assert decode_canonical_record(document, DecimalEnvelope).amount == Decimal(text)


@pytest.mark.parametrize("signal", (Overflow, Inexact))
def test_exact_decode_ignores_traps_but_constructor_arithmetic_failure_is_chained(signal: type[Exception]) -> None:
    @dataclass(frozen=True, slots=True)
    class ArithmeticEnvelope(DecimalEnvelope):
        def __post_init__(self):
            # Some scientific constructors perform arithmetic independently of
            # canonical storage. Their failure must still become a local refusal.
            +self.amount

    with localcontext() as context:
        context.prec = 10
        context.Emax = 2
        context.traps[signal] = True
        text = "1000" if signal is Overflow else "1.234567890123456789"
        payload = _payload(_document(text))
        assert decode_canonical_bytes(payload, DecimalEnvelope, maximum_bytes=4096).amount == Decimal(text)
        with pytest.raises(CanonicalizationError, match="Decimal arithmetic") as caught:
            decode_canonical_bytes(payload, ArithmeticEnvelope, maximum_bytes=4096)
        assert isinstance(caught.value.__cause__, signal)
    assert decode_canonical_record(_document("1e2"), DecimalEnvelope).amount == 100
    assert decode_canonical_bytes(_payload(_document("1")), DecimalEnvelope, maximum_bytes=4096).amount == 1


@pytest.mark.parametrize("stage", ("json-hook", "typed", "serialization"))
def test_recursion_refusal_covers_the_entire_exact_path(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def overflow(*_args: object, **_kwargs: object) -> object:
        raise RecursionError("synthetic recursion boundary")

    payload = _payload(_document("1"))
    with monkeypatch.context() as patch:
        if stage == "json-hook":
            # JSON parses these arrays, then its object hook recursively keys them.
            payload = b'{"nested":' + b'[' * 700 + b'0' + b']' * 700 + b'}'
        elif stage == "typed":
            patch.setattr(decoding, "_decode", overflow)
        else:
            patch.setattr(DecimalEnvelope, "canonical_bytes", overflow)
        with pytest.raises(CanonicalizationError, match="nesting") as caught:
            decode_canonical_bytes(payload, DecimalEnvelope, maximum_bytes=4096)
        assert isinstance(caught.value.__cause__, RecursionError)
    assert decode_canonical_bytes(_payload(_document("1")), DecimalEnvelope, maximum_bytes=4096).amount == 1
    assert decode_canonical_record(_document("1e2"), DecimalEnvelope).amount == 100


@pytest.mark.parametrize("malformation", ("duplicate", "unknown", "revision", "float", "constant", "bool-as-int", "int-as-bool"))
def test_existing_semantic_and_json_refusals_remain(malformation: str) -> None:
    document = _document("1")
    values = document["value"]
    if malformation == "duplicate":
        payload = _payload(document).replace(b'"count":1', b'"count":1,"count":1')
    else:
        if malformation == "unknown":
            values["extra"] = True
        elif malformation == "revision":
            document["version"] = "2.0.0"
        elif malformation == "float":
            values["count"] = 1.0
        elif malformation == "constant":
            values["count"] = float("nan")
        elif malformation == "bool-as-int":
            values["count"] = True
        else:
            values["enabled"] = 1
        payload = _payload(document)
    with pytest.raises(CanonicalizationError):
        decode_canonical_bytes(payload, DecimalEnvelope, maximum_bytes=4096)
    assert decode_canonical_bytes(_payload(_document("1")), DecimalEnvelope, maximum_bytes=4096).amount == 1


_SUBPROCESS = '''
import json, resource, sys, tracemalloc
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalizationError
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
payload = sys.stdin.buffer.read(8193)
original_digit_limit = sys.get_int_max_str_digits()
tracemalloc.start()
try:
    decode_canonical_bytes(payload, NamedDecimal, maximum_bytes=8192)
except CanonicalizationError as error:
    _, peak = tracemalloc.get_traced_memory()
    print(json.dumps({"exception": type(error).__name__, "cause": type(error.__cause__).__name__, "peak": peak}))
else:
    raise AssertionError("malformed payload accepted")
assert sys.get_int_max_str_digits() == original_digit_limit
print(decode_canonical_bytes(NamedDecimal("good", __import__("decimal").Decimal(1), "1").canonical_bytes(), NamedDecimal, maximum_bytes=8192).value)
'''


@pytest.mark.parametrize("malformation", ("exponent", "nested", "integer"))
def test_small_malformed_payload_has_bounded_subprocess_work(malformation: str) -> None:
    if malformation == "exponent":
        payload = _payload({"schema": "empirical-lawhood/kernel/named-decimal", "version": "1.0.0",
                            "value": {"value_id": "bad", "value": {"decimal": "1e10000000"}, "unit": "1"}})
    elif malformation == "nested":
        payload = b'[' * 1100 + b'0' + b']' * 1100
    else:
        payload = b'{"value":' + b'1' * 5000 + b'}'
    result = subprocess.run([sys.executable, "-c", _SUBPROCESS], input=payload, capture_output=True, timeout=5, check=True)
    lines = result.stdout.decode().splitlines()
    diagnostic = json.loads(lines[0])
    assert diagnostic["exception"] == "CanonicalizationError"
    assert diagnostic["peak"] < 2 * 1024 * 1024
    assert lines[1] == "1"
