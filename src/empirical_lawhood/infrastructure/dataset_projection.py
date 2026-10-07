"""Strict canonical codecs for dataset and unified catalog projections."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TypeVar, Union, get_args, get_origin

from empirical_lawhood.kernel.decoding import record_annotations
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    validate_document_shape,
)
from empirical_lawhood.planning.dataset_limits import MAX_DATASET_RECORD_CANONICAL_BYTES
from empirical_lawhood.runtime.datasets import (
    MAX_DATASET_CATALOG_CANONICAL_BYTES,
    DatasetCatalogSnapshot,
    UnifiedCatalogSnapshot,
)


MAX_DATASET_RECORD_JSON_BYTES = MAX_DATASET_RECORD_CANONICAL_BYTES
MAX_UNIFIED_CATALOG_PROJECTION_BYTES = MAX_DATASET_CATALOG_CANONICAL_BYTES + 65 * 1024 * 1024
MAX_CANONICAL_JSON_NESTING = 128

_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CanonicalizationError(f"{where} must be a string-keyed mapping")
    return value


def _decimal(value: object, *, where: str) -> Decimal:
    mapping = _mapping(value, where=where)
    if set(mapping) != {"decimal"} or not isinstance(mapping["decimal"], str):
        raise CanonicalizationError(f"{where} must be an exact tagged decimal")
    try:
        result = Decimal(mapping["decimal"])
    except InvalidOperation as error:
        raise CanonicalizationError(f"{where} contains an invalid decimal") from error
    if not result.is_finite():
        raise CanonicalizationError(f"{where} decimal must be finite")
    return result


def _decode(value: object, annotation: object, *, where: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
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
            raise CanonicalizationError(f"{where} must be an array")
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
            raise CanonicalizationError(f"{where} must be an array")
        return frozenset(
            _decode(item, arguments[0], where=f"{where}[{index}]")
            for index, item in enumerate(value)
        )
    if annotation is Decimal:
        return _decimal(value, where=where)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not isinstance(value, str):
            raise CanonicalizationError(f"{where} enum value must be a string")
        try:
            return annotation(value)
        except ValueError as error:
            raise CanonicalizationError(f"{where} has an unsupported enum value") from error
    if isinstance(annotation, type) and issubclass(annotation, CanonicalRecord):
        return _decode_record(value, annotation, where=where)
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
    raise CanonicalizationError(f"{where} uses an unsupported canonical type")


def _decode_record(
    document: object,
    record_type: type[_RecordT],
    *,
    where: str,
) -> _RecordT:
    if not is_dataclass(record_type):
        raise TypeError("canonical projection roots must be dataclasses")
    names = frozenset(field.name for field in fields(record_type))
    values = validate_document_shape(
        _mapping(document, where=where),
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
        return record_type(**decoded)
    except (TypeError, ValueError) as error:
        raise CanonicalizationError(f"{where} failed semantic validation: {error}") from error


def _parse_json(payload: bytes, *, maximum_bytes: int) -> object:
    if len(payload) > maximum_bytes:
        raise CanonicalizationError("canonical projection exceeds its byte limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CanonicalizationError("canonical projection must be UTF-8") from error

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(f"duplicate JSON mapping key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                CanonicalizationError("binary JSON numbers are forbidden")
            ),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CanonicalizationError("non-finite JSON numbers are forbidden")
            ),
        )
    except json.JSONDecodeError as error:
        raise CanonicalizationError(f"invalid canonical JSON: {error.msg}") from error
    except RecursionError as error:
        raise CanonicalizationError("canonical JSON exceeds its nesting limit") from error

    stack: list[tuple[object, int]] = [(document, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_CANONICAL_JSON_NESTING:
            raise CanonicalizationError("canonical JSON exceeds its nesting limit")
        if isinstance(value, Mapping):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
    return document


def decode_canonical_record(
    payload: bytes,
    record_type: type[_RecordT],
    *,
    maximum_bytes: int,
) -> _RecordT:
    record = _decode_record(
        _parse_json(payload, maximum_bytes=maximum_bytes), record_type, where="document"
    )
    if record.canonical_bytes() != payload:
        raise CanonicalizationError("document is not exact canonical JSON")
    return record


def decode_dataset_record_json(
    record_json: str,
    record_type: type[_RecordT],
) -> _RecordT:
    if not isinstance(record_json, str):
        raise CanonicalizationError("record_json must be text")
    try:
        payload = record_json.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CanonicalizationError("record_json must be valid UTF-8") from error
    return decode_canonical_record(
        payload,
        record_type,
        maximum_bytes=MAX_DATASET_RECORD_JSON_BYTES,
    )


def encode_dataset_catalog_snapshot(snapshot: DatasetCatalogSnapshot) -> bytes:
    if not isinstance(snapshot, DatasetCatalogSnapshot):
        raise TypeError("snapshot must be DatasetCatalogSnapshot")
    payload = snapshot.canonical_bytes()
    if len(payload) > MAX_DATASET_CATALOG_CANONICAL_BYTES:
        raise ValueError("dataset catalog projection exceeds its byte limit")
    return payload


def decode_dataset_catalog_snapshot(payload: bytes) -> DatasetCatalogSnapshot:
    return decode_canonical_record(
        payload,
        DatasetCatalogSnapshot,
        maximum_bytes=MAX_DATASET_CATALOG_CANONICAL_BYTES,
    )


def encode_unified_catalog_snapshot(snapshot: UnifiedCatalogSnapshot) -> bytes:
    if not isinstance(snapshot, UnifiedCatalogSnapshot):
        raise TypeError("snapshot must be UnifiedCatalogSnapshot")
    payload = snapshot.canonical_bytes()
    if len(payload) > MAX_UNIFIED_CATALOG_PROJECTION_BYTES:
        raise ValueError("unified catalog projection exceeds its byte limit")
    return payload


def decode_unified_catalog_snapshot(payload: bytes) -> UnifiedCatalogSnapshot:
    return decode_canonical_record(
        payload,
        UnifiedCatalogSnapshot,
        maximum_bytes=MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
    )


__all__ = [
    "MAX_CANONICAL_JSON_NESTING",
    "MAX_DATASET_RECORD_JSON_BYTES",
    "MAX_UNIFIED_CATALOG_PROJECTION_BYTES",
    "decode_canonical_record",
    "decode_dataset_catalog_snapshot",
    "decode_dataset_record_json",
    "decode_unified_catalog_snapshot",
    "encode_dataset_catalog_snapshot",
    "encode_unified_catalog_snapshot",
]
