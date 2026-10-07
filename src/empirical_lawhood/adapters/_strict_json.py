# SPDX-License-Identifier: MPL-2.0
"""Unambiguous JSON decoding for external adapter import envelopes.

Historical namespaces can be adapted by their scientific owner after parsing;
duplicate keys and non-finite JSON constants cannot be interpreted safely.
The duplicate-key rule follows the existing canonical and authoring decoders.
"""

from __future__ import annotations

import json


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON mapping key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def loads_external_json(payload: bytes) -> object:
    """Decode bounded external bytes without discarding conflicting fields."""

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
