# SPDX-License-Identifier: MPL-2.0
"""Check original Decimal operands at explicitly selected new-output boundaries.

Canonical encoding is lossless globally. These two history-bundle writers also
check their proposed handoff against the original typed operands.
"""

from dataclasses import fields, is_dataclass
from decimal import Decimal

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord


def require_decimal_operands_preserved(
    original: CanonicalRecord,
    payload: bytes,
    *,
    maximum_bytes: int,
) -> None:
    """Compare typed original leaves with the one proposed canonical payload."""
    if len(payload) > maximum_bytes:
        raise ValueError("new canonical payload exceeds its output byte budget")
    decoded = decode_canonical_bytes(payload, type(original), maximum_bytes=maximum_bytes)
    pending = [(original, decoded, "$")]
    while pending:
        expected, stored, path = pending.pop()
        if isinstance(expected, Decimal):
            if not isinstance(stored, Decimal) or expected != stored:
                raise ValueError(f"new custody changes Decimal operand at {path}")
        elif is_dataclass(expected):
            if type(stored) is not type(expected):
                raise ValueError(f"new custody changes record shape at {path}")
            pending.extend((getattr(expected, field.name), getattr(stored, field.name),
                            f"{path}.{field.name}") for field in fields(expected))
        elif isinstance(expected, tuple):
            if not isinstance(stored, tuple) or len(expected) != len(stored):
                raise ValueError(f"new custody changes tuple shape at {path}")
            pending.extend((left, right, f"{path}[{index}]")
                           for index, (left, right) in enumerate(zip(expected, stored, strict=True)))
