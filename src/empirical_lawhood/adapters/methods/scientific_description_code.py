"""Evaluate declared scientific description costs independently of wire names.

Public canonical bytes and fingerprints identify the emitted record. Scientific
description costs use fixed numeric format, field and symbol charges instead.
An unbound symbol stops scoring; matching words in other fields do not bind it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from empirical_lawhood.kernel.serialization import CanonicalRecord

from ._scientific_description_codebook import (
    OWNED_FORMAT_TOKEN_OCTETS,
    RECORD_PROFILES,
    SOURCE_SYMBOL_TOKEN_OCTETS,
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


_PROFILES = _freeze(RECORD_PROFILES)
_FORMATS = _freeze(OWNED_FORMAT_TOKEN_OCTETS)
_SYMBOLS = _freeze(SOURCE_SYMBOL_TOKEN_OCTETS)


def _operand_octets(
    value: object,
    grammar: Mapping[str, Any],
    schema: str,
    field: str,
    symbol_codes: Mapping[str, Any],
) -> int:
    kind = grammar["kind"]
    if kind in ("boolean", "integer", "null"):
        expected_type = {"boolean": bool, "integer": int, "null": type(None)}[kind]
        if type(value) is not expected_type:
            raise ValueError(f"Scientific operand type differs at {schema}.{field}")
        return len(json.dumps(value, allow_nan=False, separators=(",", ":")))
    if kind == "nominal-union":
        alternative = next(
            arm for arm in grammar["alternatives"] if (arm["kind"] == "null") == (value is None)
        )
        return _operand_octets(value, alternative, schema, field, symbol_codes)
    if kind == "immutable-sequence":
        if type(value) is not list:
            raise ValueError(f"Scientific sequence differs at {schema}.{field}")
        return 2 + max(0, len(value) - 1) + sum(
            _operand_octets(item, grammar["element"], schema, field, symbol_codes) for item in value
        )
    if kind == "nested-record":
        if not isinstance(value, dict) or value.get("schema") != grammar["schema"]:
            raise ValueError(f"Scientific nested format differs at {schema}.{field}")
        return _record_octets(value, symbol_codes)
    if type(value) is not str:
        raise ValueError(f"Scientific symbol type differs at {schema}.{field}")
    if kind == "source-literal-string":
        # Translations use only exact record/field bindings. Other values retain
        # the source scientific literal code, including Unicode and escaping.
        explicit = symbol_codes.get(schema, {}).get(field, {})
        if value in explicit:
            return explicit[value]
        return len(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    if kind == "declared-string-symbol" and "source_generated_id" in grammar:
        explicit = symbol_codes.get(schema, {}).get(field, {})
        if value in explicit:
            return explicit[value]
        template = grammar["source_generated_id"]
        matched = re.fullmatch(template["pattern"], value)
        if matched is not None:
            # This code covers the exact producer's unchanged namespace/index
            # domain. It never grants source identity, custody or authority.
            index_group = template["index_group"]
            return (
                template["fixed_token_octets"]
                + len(matched[1])
                + template["slug_token_octets"][matched[2]]
                + (len(matched[index_group]) if index_group is not None else 0)
            )
    if kind == "owned-or-declared-source-format-symbol" and value in _FORMATS:
        return _FORMATS[value]
    if kind in ("declared-string-symbol", "owned-or-declared-source-format-symbol") and grammar.get("source_literal_fallback", False):
        explicit = symbol_codes.get(schema, {}).get(field, {})
        if value in explicit:
            return explicit[value]
        # Exact generated templates and explicit bindings take precedence.
        # Other caller-provided IDs retain their source literal string cost.
        return len(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    if kind == "sha256-hex-symbol":
        if re.fullmatch(grammar["exact_pattern"], value) is None:
            raise ValueError(f"Scientific fingerprint domain differs at {schema}.{field}")
        # Equal width is a coding property, never evidence of equal identities.
        return grammar["token_octets"]
    try:
        if kind in ("nominal-enum-symbol", "finite-scientific-symbol"):
            return grammar["symbol_token_octets"][value]
        if kind == "owned-or-declared-source-format-symbol" and value in _FORMATS:
            return _FORMATS[value]
        if kind in ("declared-string-symbol", "owned-or-declared-source-format-symbol"):
            return symbol_codes[schema][field][value]
    except KeyError:
        raise ValueError(f"Scientific symbol code is not bound at {schema}.{field}") from None
    raise ValueError(f"Scientific operand rule is not bound at {schema}.{field}")


def _record_octets(document: Mapping[str, Any], symbol_codes: Mapping[str, Any]) -> int:
    if set(document) != {"schema", "version", "value"}:
        raise ValueError("Scientific record envelope differs from its declared profile")
    try:
        profile = _PROFILES[document["schema"]]
    except (KeyError, TypeError):
        raise ValueError("Scientific record format has no declared code profile") from None
    values = document["value"]
    if (
        document["version"] != profile["expected_public_revision"]
        or not isinstance(values, dict)
        or set(values) != set(profile["fields"])
    ):
        raise ValueError("Scientific record revision or fields differ from their declared profile")
    count = len(profile["fields"])
    body = 2 + max(0, count - 1) + count
    body += sum(
        field["field_token_octets"]
        + _operand_octets(values[name], field["operand"], document["schema"], name, symbol_codes)
        for name, field in profile["fields"].items()
    )
    # Envelope punctuation and keys have 31 fixed octets. Nested records have
    # no terminal; exactly one terminal belongs to the top-level description.
    return 31 + profile["format_token_octets"] + profile["revision_token_octets"] + body


def scientific_description_octets(record: CanonicalRecord) -> int:
    """Return the nominal code cost, stopping on unbound input interpretations."""
    return _record_octets(record.to_document(), _SYMBOLS) + 1
