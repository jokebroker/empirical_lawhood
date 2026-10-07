"""Small closed registries for bounded canonical scientific-record codecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import types
from typing import ClassVar, TypeVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


RecordT = TypeVar("RecordT", bound=CanonicalRecord)
MAX_STATIC_CODEC_REGISTRATIONS = 256


@dataclass(frozen=True, slots=True)
class CanonicalRecordCodecRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/canonical-record-codec-registration'

    codec_id: str
    record_schema: str
    record_version: str
    maximum_bytes: int
    decoder_key: str
    decoder_version: str
    decoder_implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.codec_id, field_name="codec_id")
        validate_schema(self.record_schema)
        validate_semantic_version(self.record_version)
        if self.maximum_bytes < 1:
            raise ValueError("canonical codec byte ceiling must be positive")
        validate_stable_id(self.decoder_key, field_name="decoder_key")
        validate_semantic_version(self.decoder_version)
        validate_sha256(
            self.decoder_implementation_sha256,
            field_name="decoder_implementation_sha256",
        )


@dataclass(frozen=True, slots=True)
class CanonicalRecordCodecRegistry:
    """Runtime-only type closure paired with canonical registration evidence."""

    registry_id: str
    registrations: tuple[CanonicalRecordCodecRegistration, ...]
    record_types: Mapping[str, type[CanonicalRecord]]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(
            self.registrations,
            attribute="record_schema",
            field_name="registrations",
        )
        if not self.registrations or len(self.registrations) > MAX_STATIC_CODEC_REGISTRATIONS:
            raise ValueError("static codec registry has an invalid closed size")
        snapshot = dict(self.record_types)
        if set(snapshot) != {value.record_schema for value in self.registrations}:
            raise ValueError("static codec types differ from canonical registrations")
        for registration in self.registrations:
            record_type = snapshot[registration.record_schema]
            if (
                not isinstance(record_type, type)
                or not issubclass(record_type, CanonicalRecord)
                or record_type.SCHEMA != registration.record_schema
                or record_type.VERSION != registration.record_version
            ):
                raise ValueError("static codec registration differs from its record type")
        object.__setattr__(self, "record_types", types.MappingProxyType(snapshot))

    def registration(self, record_schema: str) -> CanonicalRecordCodecRegistration:
        validate_schema(record_schema)
        for value in self.registrations:
            if value.record_schema == record_schema:
                return value
        raise KeyError(f"unregistered canonical record schema: {record_schema}")

    def decode(self, payload: bytes, *, expected_schema: str) -> CanonicalRecord:
        registration = self.registration(expected_schema)
        return decode_canonical_bytes(
            payload,
            self.record_types[expected_schema],
            maximum_bytes=registration.maximum_bytes,
        )


def build_canonical_record_codec_registry(
    *,
    registry_id: str,
    record_types: tuple[type[CanonicalRecord], ...],
    maximum_bytes_by_schema: Mapping[str, int],
    decoder_implementation_sha256: str,
) -> CanonicalRecordCodecRegistry:
    types_by_schema = {value.SCHEMA: value for value in record_types}
    if len(types_by_schema) != len(record_types):
        raise ValueError("static codec roster duplicates a canonical schema")
    if set(maximum_bytes_by_schema) != set(types_by_schema):
        raise ValueError("static codec byte ceilings differ from the type roster")
    registrations = tuple(
        sorted(
            (
                CanonicalRecordCodecRegistration(
                    codec_id=f"canonical-json.{schema.replace('/', '.').replace('_', '-')}",
                    record_schema=schema,
                    record_version=record_type.VERSION,
                    maximum_bytes=maximum_bytes_by_schema[schema],
                    decoder_key="kernel.canonical-json-decoder",
                    decoder_version="1.0.0",
                    decoder_implementation_sha256=decoder_implementation_sha256,
                )
                for schema, record_type in types_by_schema.items()
            ),
            key=lambda value: value.record_schema,
        )
    )
    return CanonicalRecordCodecRegistry(
        registry_id=registry_id,
        registrations=registrations,
        record_types=types_by_schema,
    )


__all__ = [
    'CanonicalRecordCodecRegistration',
    "CanonicalRecordCodecRegistry",
    "build_canonical_record_codec_registry",
]
