"""Strict, bounded decoding for canonical scientific records."""

from __future__ import annotations

import json
import re
import types
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .parsed_nodes import ParsedNodeInterner
from .serialization import CanonicalizationError, CanonicalRecord, validate_document_shape

RecordT = TypeVar("RecordT", bound=CanonicalRecord)
_decoded_records: ContextVar[dict[tuple[type[CanonicalRecord], int], CanonicalRecord] | None] = (
    ContextVar("decoded_records", default=None)
)

_exact_decimal_tokens: ContextVar[bool] = ContextVar("exact_decimal_tokens", default=False)
_CANONICAL_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?")


@lru_cache(maxsize=1024)
def record_annotations(record_type: type[CanonicalRecord]) -> Mapping[str, object]:
    """Cache installed type metadata; decode and validate every payload afresh."""

    return MappingProxyType(get_type_hints(record_type))


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CanonicalizationError(f"{where} must be a string-keyed mapping")
    return value


def _decode_decimal(value: object, *, where: str) -> Decimal:
    raw = _mapping(value, where=where)
    if set(raw) != {"decimal"} or not isinstance(raw["decimal"], str):
        raise CanonicalizationError(f"{where} must be an exact tagged decimal")
    text = raw["decimal"]
    if _exact_decimal_tokens.get() and (text == "-0" or _CANONICAL_DECIMAL.fullmatch(text) is None):
        # Fixed-point canonical spellings cannot hide an expanding exponent.
        # The broader document decoder deliberately retains its old acceptance.
        raise CanonicalizationError(f"{where} decimal spelling is not canonical")
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise CanonicalizationError(f"{where} contains an invalid decimal") from error
    if not result.is_finite():
        raise CanonicalizationError(f"{where} decimal must be finite")
    return result


def _decode(value: object, annotation: object, *, where: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any or annotation is object:
        raise CanonicalizationError(f"{where} has an unbounded canonical type")
    if annotation is type(None):
        if value is not None:
            raise CanonicalizationError(f"{where} must be null")
        return None
    if origin in {Union, types.UnionType}:
        errors: list[str] = []
        for option in arguments:
            try:
                return _decode(value, option, where=where)
            except (CanonicalizationError, TypeError, ValueError) as error:
                errors.append(str(error))
        raise CanonicalizationError(f"{where} does not match its closed union: {errors}")
    if origin is tuple:
        if not isinstance(value, list):
            raise CanonicalizationError(f"{where} must be a sequence")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode(item, arguments[0], where=f"{where}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise CanonicalizationError(f"{where} has the wrong tuple length")
        return tuple(
            _decode(item, expected, where=f"{where}[{index}]")
            for index, (item, expected) in enumerate(zip(value, arguments, strict=True))
        )
    if origin is frozenset:
        if not isinstance(value, list):
            raise CanonicalizationError(f"{where} must be a sequence")
        return frozenset(
            _decode(item, arguments[0], where=f"{where}[{index}]")
            for index, item in enumerate(value)
        )
    if annotation is Decimal:
        return _decode_decimal(value, where=where)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not isinstance(value, str):
            raise CanonicalizationError(f"{where} enum value must be a string")
        try:
            return annotation(value)
        except ValueError as error:
            raise CanonicalizationError(f"{where} has an unknown enum value") from error
    if isinstance(annotation, type) and issubclass(annotation, CanonicalRecord):
        return decode_canonical_record(value, annotation, where=where)
    if annotation is bool:
        if type(value) is not bool:
            raise CanonicalizationError(f"{where} must be boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise CanonicalizationError(f"{where} must be an integer")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise CanonicalizationError(f"{where} must be a string")
        return value
    raise CanonicalizationError(f"{where} uses unsupported canonical type {annotation!r}")


def decode_canonical_record(
    document: object,
    record_type: type[RecordT],
    *,
    where: str = "document",
) -> RecordT:
    """Decode one closed canonical dataclass with exact schema and fields."""

    if not is_dataclass(record_type):
        raise TypeError("canonical record types must be dataclasses")
    cache = _decoded_records.get()
    cache_key = (record_type, id(document))
    if cache is not None and cache_key in cache:
        from typing import cast

        return cast(RecordT, cache[cache_key])
    raw = _mapping(document, where=where)
    names = frozenset(field.name for field in fields(record_type))
    values = validate_document_shape(
        raw,
        expected_schema=record_type.SCHEMA,
        expected_version=record_type.VERSION,
        field_names=names,
    )
    annotations = record_annotations(record_type)
    decoded = {
        field.name: _decode(
            values[field.name],
            annotations[field.name],
            where=f"{where}.value.{field.name}",
        )
        for field in fields(record_type)
    }
    try:
        result = record_type(**decoded)
        if cache is not None:
            cache[cache_key] = result
        return result
    except (TypeError, ValueError) as error:
        raise CanonicalizationError(f"{where} failed semantic validation: {error}") from error


def decode_canonical_bytes(
    payload: bytes,
    record_type: type[RecordT],
    *,
    maximum_bytes: int,
) -> RecordT:
    """Decode exact canonical JSON without accepting duplicates or binary numbers."""

    if not isinstance(payload, bytes):
        raise CanonicalizationError("canonical payload must be immutable bytes")
    if maximum_bytes <= 0:
        raise ValueError("canonical decode byte limit must be positive")
    if len(payload) > maximum_bytes:
        raise CanonicalizationError("canonical payload exceeds its bounded contract")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CanonicalizationError("canonical payload must be UTF-8") from error

    # Hash-cons only within this parse. Every occurrence still undergoes JSON
    # duplicate-key/number checks. Equal immutable subrecords can then undergo
    # the same typed semantic validation once, without changing canonical bytes.
    interner = ParsedNodeInterner()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(f"duplicate canonical mapping key: {key}")
            result[key] = value
        return interner.mapping(result)

    try:
        document = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                CanonicalizationError("canonical binary numbers are forbidden; use tagged decimals")
            ),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CanonicalizationError("non-finite canonical numbers are forbidden")
            ),
        )
    except CanonicalizationError:
        raise
    except ValueError as error:
        # Includes the interpreter's integer-token digit refusal. Keep that
        # process-level safeguard enabled and translate only this parse's error.
        raise CanonicalizationError("canonical payload is not valid JSON") from error
    except RecursionError as error:
        raise CanonicalizationError("canonical payload nesting exceeds decoder capacity") from error
    token = _decoded_records.set({})
    decimal_token = _exact_decimal_tokens.set(True)
    try:
        record = decode_canonical_record(document, record_type)
        if record.canonical_bytes() != payload:
            raise CanonicalizationError("canonical payload bytes are not canonical")
        return record
    except DecimalException as error:
        raise CanonicalizationError("canonical payload failed Decimal arithmetic") from error
    except RecursionError as error:
        raise CanonicalizationError("canonical payload nesting exceeds decoder capacity") from error
    finally:
        _exact_decimal_tokens.reset(decimal_token)
        _decoded_records.reset(token)
