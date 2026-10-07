"""Immutable structural validator registrations and bounded artifact decoders."""

from __future__ import annotations

import codecs
import functools
import hashlib
import io
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from types import MappingProxyType
from enum import StrEnum
from importlib.metadata import distribution
from pathlib import Path
from typing import BinaryIO, Iterator, TypeVar

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import onnx
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load as load_safetensors

from empirical_lawhood.kernel.serialization import (
    canonical_json_bytes,
    validate_nonempty,
    validate_schema,
    validate_sha256,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactGenericValidation,
    ArtifactProfile,
    ArtifactStreamWriteRequest,
    ArtifactWriteRequest,
)

from .bounded_io import (
    BoundedFileIOError,
    read_bounded_bytes,
)


class ArtifactPlaneError(RuntimeError):
    "Base error for guarded artifact operations."


class ExternalRootUnavailable(ArtifactPlaneError):
    pass


class ArtifactIdentityConflict(ArtifactPlaneError):
    pass


class ArtifactInputLimitExceeded(ArtifactPlaneError):
    pass


MAX_TEXT_PARAMETER_BYTES = 64 * 1024**2
MAX_MODEL_PAYLOAD_BYTES = 256 * 1024**2
MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024
MAX_IN_MEMORY_ARTIFACT_BYTES = 32 * 1024**2
MAX_JSONL_RECORD_BYTES = 1024 * 1024
MAX_CANONICAL_JSON_TOKEN_BYTES = 1024 * 1024
MAX_CANONICAL_JSON_DEPTH = 128
MAX_VALIDATOR_IMPLEMENTATION_SOURCE_BYTES = 2 * 1024**2
MAX_GENERIC_VALIDATION_COMPATIBILITIES = 64
GenericValidationCompatibility = tuple[
    ArtifactGenericValidation,
    ArtifactGenericValidation,
]


@dataclass(frozen=True, slots=True)
class GenericValidatorImplementationCompatibility:
    """One exact predecessor implementation admitted for schema-generic replay."""

    validator_key: str
    validator_version: str
    validator_implementation_sha256: str
    profile: ArtifactProfile

    def __post_init__(self) -> None:
        validate_nonempty(self.validator_key, field_name="validator_key")
        validate_nonempty(self.validator_version, field_name="validator_version")
        validate_sha256(
            self.validator_implementation_sha256,
            field_name="validator_implementation_sha256",
        )

    @property
    def key(self) -> tuple[str, str, str, ArtifactProfile]:
        return (
            self.validator_key,
            self.validator_version,
            self.validator_implementation_sha256,
            self.profile,
        )


@dataclass(frozen=True, slots=True)
class NumpyArrayContract:
    """Exact safe-array semantics registered by infrastructure composition."""

    payload_schema: str
    dtype: str
    shape: tuple[int | None, ...]

    def __post_init__(self) -> None:
        validate_schema(self.payload_schema)
        dtype = np.dtype(self.dtype)
        if dtype.hasobject:
            raise ValueError("NumPy contracts cannot allow object dtypes")
        if not self.shape or any(
            dimension is not None and dimension < 0 for dimension in self.shape
        ):
            raise ValueError("NumPy contracts require a nonnegative shape")

    def accepts_shape(self, observed: tuple[int, ...]) -> bool:
        """Match exact dimensions while allowing explicitly registered wildcards."""

        return len(observed) == len(self.shape) and all(
            expected is None or expected == actual
            for expected, actual in zip(self.shape, observed, strict=True)
        )


def _validate_metadata_pairs(
    values: tuple[tuple[str, str], ...],
    *,
    field_name: str,
    reserved: frozenset[str] = frozenset(),
) -> None:
    keys = tuple(key for key, _value in values)
    if tuple(sorted(set(keys))) != keys:
        raise ValueError(f"{field_name} must have sorted unique keys")
    for key, value in values:
        if not key or not value:
            raise ValueError(f"{field_name} keys and values must be nonempty")
        if key in reserved:
            raise ValueError(f"{field_name} cannot override reserved metadata")


def _metadata_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


_TABLE_RESERVED_METADATA = frozenset(
    {
        "empirical_lawhood_payload_schema",
        "empirical_lawhood_logical_types",
        "empirical_lawhood_units",
        "empirical_lawhood_frames",
        "empirical_lawhood_clocks",
        "empirical_lawhood_keys",
    }
)
_ContractT = TypeVar("_ContractT")


@dataclass(frozen=True, slots=True)
class ArrowFieldContract:
    """One exact ordered Arrow field and its scientific coordinate semantics."""

    name: str
    physical_type: str
    logical_type: str
    nullable: bool
    native_unit: str
    coordinate_frame: str
    clock_id: str | None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("physical_type", self.physical_type),
            ("logical_type", self.logical_type),
            ("native_unit", self.native_unit),
            ("coordinate_frame", self.coordinate_frame),
        ):
            if not value:
                raise ValueError(f"Arrow field {field_name} must be nonempty")
        if self.clock_id is not None and not self.clock_id:
            raise ValueError("Arrow field clock_id must be nonempty when present")
        _validate_metadata_pairs(self.metadata, field_name="Arrow field metadata")


@dataclass(frozen=True, slots=True)
class ArrowTableContract:
    """Exact Arrow schema, metadata, units, frames, clocks and primary key."""

    payload_schema: str
    fields: tuple[ArrowFieldContract, ...]
    primary_key_fields: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        validate_schema(self.payload_schema)
        if not self.fields:
            raise ValueError("Arrow table contract requires fields")
        field_names = tuple(field.name for field in self.fields)
        if len(set(field_names)) != len(field_names):
            raise ValueError("Arrow table contract field names must be unique")
        if not self.primary_key_fields or len(set(self.primary_key_fields)) != len(
            self.primary_key_fields
        ):
            raise ValueError("Arrow table primary key must be ordered and unique")
        if not set(self.primary_key_fields).issubset(field_names):
            raise ValueError("Arrow table primary key references an unknown field")
        _validate_metadata_pairs(
            self.metadata,
            field_name="Arrow schema metadata",
            reserved=_TABLE_RESERVED_METADATA,
        )

    def expected_metadata(self) -> dict[bytes, bytes]:
        """Return the exact serialized schema metadata required on the wire."""

        metadata = {
            b"empirical_lawhood_payload_schema": self.payload_schema.encode("utf-8"),
            b"empirical_lawhood_logical_types": _metadata_json(
                {field.name: field.logical_type for field in self.fields}
            ),
            b"empirical_lawhood_units": _metadata_json(
                {field.name: field.native_unit for field in self.fields}
            ),
            b"empirical_lawhood_frames": _metadata_json(
                {field.name: field.coordinate_frame for field in self.fields}
            ),
            b"empirical_lawhood_clocks": _metadata_json(
                {field.name: field.clock_id for field in self.fields if field.clock_id is not None}
            ),
            b"empirical_lawhood_keys": _metadata_json(list(self.primary_key_fields)),
        }
        metadata.update(
            (key.encode("utf-8"), value.encode("utf-8")) for key, value in self.metadata
        )
        return metadata


@dataclass(frozen=True, slots=True)
class ParquetColumnContract:
    """One exact physical Parquet leaf column."""

    path: str
    physical_type: str
    logical_type: str
    max_definition_level: int
    max_repetition_level: int

    def __post_init__(self) -> None:
        if not self.path or not self.physical_type or not self.logical_type:
            raise ValueError("Parquet column identity must be nonempty")
        if self.max_definition_level < 0 or self.max_repetition_level < 0:
            raise ValueError("Parquet column levels must be nonnegative")


@dataclass(frozen=True, slots=True)
class ParquetTableContract:
    """Exact Arrow-level and encoded Parquet-level table structure."""

    arrow: ArrowTableContract
    columns: tuple[ParquetColumnContract, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("Parquet table contract requires physical columns")
        paths = tuple(column.path for column in self.columns)
        if len(set(paths)) != len(paths):
            raise ValueError("Parquet physical column paths must be unique")

    @property
    def payload_schema(self) -> str:
        return self.arrow.payload_schema


class HDF5ObjectKind(StrEnum):
    GROUP = "GROUP"
    DATASET = "DATASET"


class HDF5AttributeKind(StrEnum):
    UTF8 = "UTF8"
    BYTES = "BYTES"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"


@dataclass(frozen=True, slots=True)
class HDF5AttributeContract:
    """One bounded scalar HDF5 attribute.

    ``value=None`` retains an exact attribute name and scalar kind while
    allowing identity/content values that are fixed only when a task is
    issued.  The registry-wide decoded-byte ceiling remains mandatory.
    """

    name: str
    kind: HDF5AttributeKind
    value: str | None
    required: bool = True

    def __post_init__(self) -> None:
        if type(self.required) is not bool:
            raise ValueError("HDF5 attribute required policy must be boolean")
        if not self.name or self.value == "":
            raise ValueError("HDF5 attribute name and any exact value must be nonempty")
        if self.value is None:
            return
        if self.kind is HDF5AttributeKind.BOOLEAN and self.value not in {"false", "true"}:
            raise ValueError("HDF5 boolean attributes use canonical true/false values")
        if self.kind is HDF5AttributeKind.INTEGER:
            try:
                if str(int(self.value)) != self.value:
                    raise ValueError
            except ValueError as error:
                raise ValueError("HDF5 integer attribute is not canonical") from error
        if self.kind is HDF5AttributeKind.FLOAT:
            try:
                decoded = float.fromhex(self.value)
            except ValueError as error:
                raise ValueError("HDF5 float attribute is not canonical") from error
            if not math.isfinite(decoded) or decoded.hex() != self.value:
                raise ValueError("HDF5 float attribute is not canonical")


@dataclass(frozen=True, slots=True)
class HDF5DatasetStorageContract:
    """Exact byte-observable dataset creation/storage properties.

    Presence of this record opts a dataset into layout validation.  Its
    absence preserves the pre-existing audited-HDF5 behavior, which validates
    inventory, type, shape, attributes and decoded-work bounds without making
    a storage-layout claim.
    """

    chunk_shape: tuple[int, ...]
    compression_filter: str | None
    compression_options: str | None
    shuffle_filter_enabled: bool
    fletcher32_checksum_enabled: bool
    scale_offset: int | None
    zero_fill_value_required: bool
    external_storage_forbidden: bool

    def __post_init__(self) -> None:
        if not self.chunk_shape or any(value < 1 for value in self.chunk_shape):
            raise ValueError("HDF5 storage chunk geometry must be positive")
        if self.compression_filter is None and self.compression_options is not None:
            raise ValueError("HDF5 compression options require a compression filter")
        if self.compression_filter == "" or self.compression_options == "":
            raise ValueError("HDF5 compression identities must be nonempty when present")
        if self.scale_offset is not None and self.scale_offset < 0:
            raise ValueError("HDF5 scale-offset precision must be nonnegative")


@dataclass(frozen=True, slots=True)
class HDF5ObjectContract:
    """One closed path; required means present whenever its parent is present."""

    path: str
    kind: HDF5ObjectKind
    dtype: str | None
    shape: tuple[int, ...] | None
    attributes: tuple[HDF5AttributeContract, ...] = ()
    maximum_shape: tuple[int, ...] | None = None
    storage: HDF5DatasetStorageContract | None = None
    track_link_creation_order: bool | None = None
    index_link_creation_order: bool | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if type(self.required) is not bool:
            raise ValueError("HDF5 object required policy must be boolean")
        if self.path != "/" and (
            not self.path.startswith("/")
            or self.path.endswith("/")
            or "//" in self.path
            or "/../" in f"{self.path}/"
        ):
            raise ValueError("HDF5 object paths must be normalized absolute paths")
        attribute_names = tuple(attribute.name for attribute in self.attributes)
        if tuple(sorted(set(attribute_names))) != attribute_names:
            raise ValueError("HDF5 attributes must have sorted unique names")
        if self.kind is HDF5ObjectKind.GROUP:
            if (
                self.dtype is not None
                or self.shape is not None
                or self.maximum_shape is not None
                or self.storage is not None
            ):
                raise ValueError("HDF5 groups cannot declare dtype or shape bounds")
            if self.index_link_creation_order and not self.track_link_creation_order:
                raise ValueError("HDF5 indexed link creation order requires tracked creation order")
            return
        if self.track_link_creation_order is not None or self.index_link_creation_order is not None:
            raise ValueError("HDF5 datasets cannot declare group link-creation policy")
        if self.dtype is None or self.shape is None:
            raise ValueError("HDF5 datasets require exact dtype and shape")
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.fields is not None:
            raise ValueError("HDF5 contracts forbid object and structured dtypes")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("HDF5 dataset shapes must be nonnegative")
        if self.maximum_shape is not None and (
            len(self.maximum_shape) != len(self.shape)
            or any(dimension < 0 for dimension in self.maximum_shape)
            or any(
                lower > upper
                for lower, upper in zip(
                    self.shape,
                    self.maximum_shape,
                    strict=True,
                )
            )
        ):
            raise ValueError("HDF5 dataset maximum shape does not bound its minimum shape")
        if self.storage is not None and (
            len(self.storage.chunk_shape) != len(self.shape)
            or any(
                chunk > minimum
                for chunk, minimum in zip(
                    self.storage.chunk_shape,
                    self.shape,
                    strict=True,
                )
            )
        ):
            raise ValueError("HDF5 storage chunks must fit the minimum dataset geometry")


@dataclass(frozen=True, slots=True)
class HDF5InventoryContract:
    """Closed required/optional HDF5 inventory with decoded-work ceilings."""

    payload_schema: str
    objects: tuple[HDF5ObjectContract, ...]
    maximum_objects: int
    maximum_dataset_elements: int
    maximum_total_elements: int
    maximum_decoded_bytes: int
    maximum_name_bytes: int
    maximum_attribute_bytes: int

    def __post_init__(self) -> None:
        validate_schema(self.payload_schema)
        paths = tuple(item.path for item in self.objects)
        if not paths or tuple(sorted(set(paths))) != paths or paths[0] != "/":
            raise ValueError("HDF5 inventory paths must be sorted, unique and include root")
        if self.objects[0].kind is not HDF5ObjectKind.GROUP or not self.objects[0].required:
            raise ValueError("HDF5 inventory root must be a required group")
        by_path = {item.path: item for item in self.objects}
        for item in self.objects[1:]:
            parent = by_path.get(item.path.rsplit("/", 1)[0] or "/")
            if parent is None or parent.kind is not HDF5ObjectKind.GROUP:
                raise ValueError("HDF5 object parent must be a registered group")
        if (
            min(
                self.maximum_objects,
                self.maximum_dataset_elements,
                self.maximum_total_elements,
                self.maximum_decoded_bytes,
                self.maximum_name_bytes,
                self.maximum_attribute_bytes,
            )
            <= 0
        ):
            raise ValueError("HDF5 decoded-work limits must be positive")
        if len(self.objects) > self.maximum_objects:
            raise ValueError("HDF5 registered inventory exceeds its object limit")
        contract_name_bytes = sum(
            len(item.path.encode("utf-8")) for item in self.objects if item.path != "/"
        )
        if contract_name_bytes > self.maximum_name_bytes:
            raise ValueError("HDF5 registered paths exceed their decoded-work limit")
        contract_attribute_bytes = sum(
            len(attribute.name.encode("utf-8"))
            + (0 if attribute.value is None else len(attribute.value.encode("utf-8")))
            for item in self.objects
            for attribute in item.attributes
        )
        if contract_attribute_bytes > self.maximum_attribute_bytes:
            raise ValueError("HDF5 registered attributes exceed their decoded-work limit")
        root_attributes = {attribute.name: attribute for attribute in self.objects[0].attributes}
        required = {
            "empirical_lawhood_payload_schema",
            "empirical_lawhood_units",
            "empirical_lawhood_frames",
            "empirical_lawhood_clocks",
            "empirical_lawhood_keys",
        }
        if not required.issubset(root_attributes) or any(
            not root_attributes[key].required for key in required if key in root_attributes
        ):
            raise ValueError("HDF5 root metadata contract is incomplete")
        schema_attribute = root_attributes["empirical_lawhood_payload_schema"]
        if (
            schema_attribute.kind is not HDF5AttributeKind.UTF8
            or schema_attribute.value != self.payload_schema
        ):
            raise ValueError("HDF5 root schema attribute differs from its contract")
        for key in (
            "empirical_lawhood_units",
            "empirical_lawhood_frames",
            "empirical_lawhood_clocks",
            "empirical_lawhood_keys",
        ):
            attribute = root_attributes[key]
            if attribute.kind is not HDF5AttributeKind.UTF8 or attribute.value is None:
                raise ValueError("HDF5 semantic metadata attributes must be UTF-8")
            decoded = _strict_json(
                attribute.value.encode("utf-8"),
                label="HDF5 contract semantic metadata",
            )
            if key == "empirical_lawhood_keys":
                if not isinstance(decoded, list) or not decoded:
                    raise ValueError("HDF5 key metadata must be a nonempty list")
            elif not isinstance(decoded, dict) or not decoded:
                raise ValueError("HDF5 unit/frame/clock metadata must be nonempty objects")


def _strict_json(payload: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactIdentityConflict(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIdentityConflict(f"{label} validator rejected the payload") from error


def _decoded_json_bytes(value: object) -> bytes:
    """Re-encode already-decoded JSON with the repository canonical byte form."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactIdentityConflict("decoded JSON value is not canonicalizable") from error


_CONTAINER_VALUE = object()
_CANONICAL_READ_CHUNK_BYTES = 64 * 1024


class _CanonicalJsonFileInspector:
    """Incrementally validate canonical JSON and its registered document shape."""

    def __init__(
        self,
        handle: BinaryIO,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        self._handle = handle
        self._pending: int | None = None
        self._buffer = b""
        self._buffer_offset = 0
        self._string_special = re.compile(rb'["\\\x00-\x1f\x7f-\xff]')
        self._request = request
        self._semantic = request.semantic_validation

    def validate(self) -> None:
        expected_top_level = None if self._semantic is None else self._semantic.top_level_keys
        token, _value = self._next_token()
        if token != "{":
            raise ArtifactIdentityConflict("canonical JSON payload must be an object")
        self._parse_object(
            depth=1,
            expected_keys=expected_top_level,
            is_top_level=True,
            is_value_object=False,
        )
        token, _value = self._next_token()
        if token != "newline":
            raise ArtifactIdentityConflict(
                "canonical JSON payload requires exactly one terminal newline"
            )
        token, _value = self._next_token()
        if token != "eof":
            raise ArtifactIdentityConflict("canonical JSON payload contains trailing bytes")

    def _parse_value(
        self,
        token: str,
        value: object,
        *,
        depth: int,
        expected_object_keys: tuple[str, ...] | None = None,
        is_value_object: bool = False,
    ) -> object:
        if depth > MAX_CANONICAL_JSON_DEPTH:
            raise ArtifactIdentityConflict("canonical JSON nesting exceeds its depth bound")
        if token == "{":
            self._parse_object(
                depth=depth,
                expected_keys=expected_object_keys,
                is_top_level=False,
                is_value_object=is_value_object,
            )
            return _CONTAINER_VALUE
        if expected_object_keys is not None:
            raise ArtifactIdentityConflict(
                "artifact value shape differs from its semantic contract"
            )
        if token == "[":
            self._parse_array(depth=depth)
            return _CONTAINER_VALUE
        if token in {"string", "scalar"}:
            return value
        raise ArtifactIdentityConflict("canonical JSON value has invalid syntax")

    def _parse_object(
        self,
        *,
        depth: int,
        expected_keys: tuple[str, ...] | None,
        is_top_level: bool,
        is_value_object: bool,
    ) -> None:
        previous_key: str | None = None
        key_index = 0
        observed_schema = False
        observed_version = False
        observed_bindings: set[str] = set()
        bindings = (
            {}
            if not is_top_level or self._semantic is None
            else dict(self._semantic.field_bindings)
        )
        while True:
            token, value = self._next_token()
            if token == "}":
                if key_index:
                    raise ArtifactIdentityConflict("canonical JSON object has a trailing comma")
                break
            if token != "string" or not isinstance(value, str):
                raise ArtifactIdentityConflict("canonical JSON object key is invalid")
            key = value
            if previous_key is not None and key <= previous_key:
                raise ArtifactIdentityConflict(
                    "canonical JSON object keys are not sorted and unique"
                )
            previous_key = key
            if expected_keys is not None and (
                key_index >= len(expected_keys) or key != expected_keys[key_index]
            ):
                label = "value" if is_value_object else "document"
                raise ArtifactIdentityConflict(
                    f"artifact {label} shape differs from its semantic contract"
                )
            key_index += 1
            token, _separator = self._next_token()
            if token != ":":
                raise ArtifactIdentityConflict("canonical JSON object lacks a key separator")
            token, child = self._next_token(
                retain_string=is_top_level and (key in {"schema", "version"} or key in bindings)
            )
            expected_value_keys = (
                self._semantic.value_keys
                if is_top_level
                and key == "value"
                and self._semantic is not None
                and self._semantic.value_keys
                else None
            )
            decoded = self._parse_value(
                token,
                child,
                depth=depth + 1,
                expected_object_keys=expected_value_keys,
                is_value_object=expected_value_keys is not None,
            )
            if is_top_level and key == "schema":
                observed_schema = True
                if decoded != self._request.payload_schema:
                    raise ArtifactIdentityConflict(
                        "artifact payload schema differs from its semantic contract"
                    )
            if is_top_level and key == "version" and self._semantic is not None:
                observed_version = True
                expected_version = bindings.get("version", "1.0.0")
                if self._semantic.value_keys and decoded != expected_version:
                    raise ArtifactIdentityConflict(
                        "artifact record version differs from its semantic contract"
                    )
            if is_top_level and key in bindings:
                observed_bindings.add(key)
                if decoded != bindings[key]:
                    raise ArtifactIdentityConflict(
                        "artifact context binding differs from its semantic contract"
                    )
            token, _separator = self._next_token()
            if token == "}":
                break
            if token != ",":
                raise ArtifactIdentityConflict("canonical JSON object separator is invalid")
        if expected_keys is not None and key_index != len(expected_keys):
            label = "value" if is_value_object else "document"
            raise ArtifactIdentityConflict(
                f"artifact {label} shape differs from its semantic contract"
            )
        if is_top_level:
            if not observed_schema:
                raise ArtifactIdentityConflict("canonical JSON payload schema is absent")
            if self._semantic is not None:
                if self._semantic.value_keys and not observed_version:
                    raise ArtifactIdentityConflict(
                        "artifact record version differs from its semantic contract"
                    )
                if observed_bindings != set(bindings):
                    raise ArtifactIdentityConflict(
                        "artifact context binding differs from its semantic contract"
                    )

    def _parse_array(self, *, depth: int) -> None:
        token, value = self._next_token(retain_string=False)
        if token == "]":
            return
        while True:
            self._parse_value(token, value, depth=depth + 1)
            token, value = self._next_token(retain_string=False)
            if token == "]":
                return
            if token != ",":
                raise ArtifactIdentityConflict("canonical JSON array separator is invalid")
            token, value = self._next_token(retain_string=False)

    def _read_byte(self) -> int | None:
        if self._pending is not None:
            value = self._pending
            self._pending = None
            return value
        if not self._fill_buffer():
            return None
        value = self._buffer[self._buffer_offset]
        self._buffer_offset += 1
        return value

    def _fill_buffer(self) -> bool:
        if self._buffer_offset == len(self._buffer):
            self._buffer = self._handle.read(_CANONICAL_READ_CHUNK_BYTES)
            self._buffer_offset = 0
        return bool(self._buffer)

    def _read_token_bytes(self, count: int) -> bytes:
        result = bytearray()
        for _ in range(count):
            value = self._read_byte()
            if value is None:
                break
            result.append(value)
        return bytes(result)

    def _unread_byte(self, value: int) -> None:
        if self._pending is not None:
            raise AssertionError("canonical JSON token reader has multiple pending bytes")
        self._pending = value

    def _next_token(self, *, retain_string: bool = True) -> tuple[str, object]:
        value = self._read_byte()
        if value is None:
            return "eof", None
        punctuation = {
            ord("{"): "{",
            ord("}"): "}",
            ord("["): "[",
            ord("]"): "]",
            ord(":"): ":",
            ord(","): ",",
        }
        if value in punctuation:
            return punctuation[value], None
        if value == ord("\n"):
            return "newline", None
        if value == ord('"'):
            return "string", self._string_token() if retain_string else self._skip_string_token()
        if value in b"-0123456789":
            self._unread_byte(value)
            return "scalar", self._number_token()
        for literal, decoded in (
            (b"true", True),
            (b"false", False),
            (b"null", None),
        ):
            if value == literal[0]:
                tail = self._read_token_bytes(len(literal) - 1)
                if bytes((value,)) + tail != literal:
                    raise ArtifactIdentityConflict("canonical JSON literal is invalid")
                return "scalar", decoded
        raise ArtifactIdentityConflict("canonical JSON bytes are not canonical")

    def _skip_string_token(self) -> object:
        """Validate a nonsemantic value in bounded memory, within the file budget.

        Large encoded arrays and embedded JSON are legitimate string values.
        Keys and compared semantic fields still use the bounded token decoder.
        Every escape is checked for exact canonical spelling, including controls
        and Unicode; no content is silently accepted merely because it is large.
        """
        while True:
            if self._pending is None and self._fill_buffer():
                special = self._string_special.search(self._buffer, self._buffer_offset)
                self._buffer_offset = len(self._buffer) if special is None else special.start()
                if special is None:
                    continue
            value = self._read_byte()
            if value is None:
                raise ArtifactIdentityConflict("canonical JSON string is unterminated")
            if value == ord('"'):
                return _CONTAINER_VALUE
            if value >= 0x7F or value < 0x20:
                raise ArtifactIdentityConflict("canonical JSON string bytes are invalid")
            if value != ord("\\"):
                continue
            escape = self._read_byte()
            if escape is None:
                raise ArtifactIdentityConflict("canonical JSON escape is unterminated")
            raw = b'"\\' + bytes((escape,))
            if escape == ord("u"):
                digits = self._read_token_bytes(4)
                if len(digits) != 4 or any(digit not in b"0123456789abcdef" for digit in digits):
                    raise ArtifactIdentityConflict("canonical JSON unicode escape is invalid")
                raw += digits
            elif escape not in b'"\\bfnrt':
                raise ArtifactIdentityConflict("canonical JSON escape is invalid")
            raw += b'"'
            if json.dumps(json.loads(raw), ensure_ascii=True).encode("ascii") != raw:
                raise ArtifactIdentityConflict("canonical JSON string encoding is not canonical")

    def _string_token(self) -> str:
        raw = bytearray(b'"')
        escaped = False
        while True:
            # Scan ordinary ASCII spans in C instead of calling Python once
            # for every byte in full-size plans and issued manifest siblings.
            if self._pending is None and self._fill_buffer():
                special = self._string_special.search(self._buffer, self._buffer_offset)
                end = len(self._buffer) if special is None else special.start()
                if len(raw) + end - self._buffer_offset > MAX_CANONICAL_JSON_TOKEN_BYTES:
                    raise ArtifactIdentityConflict("canonical JSON token exceeds its byte bound")
                raw.extend(self._buffer[self._buffer_offset : end])
                self._buffer_offset = end
                if special is None:
                    continue
            value = self._read_byte()
            if value is None:
                raise ArtifactIdentityConflict("canonical JSON string is unterminated")
            raw.append(value)
            if len(raw) > MAX_CANONICAL_JSON_TOKEN_BYTES:
                raise ArtifactIdentityConflict("canonical JSON token exceeds its byte bound")
            if value >= 0x7F or value < 0x20:
                raise ArtifactIdentityConflict("canonical JSON string bytes are invalid")
            if value == ord('"'):
                break
            if value != ord("\\"):
                continue
            escaped = True
            escape = self._read_byte()
            if escape is None:
                raise ArtifactIdentityConflict("canonical JSON escape is unterminated")
            raw.append(escape)
            if escape == ord("u"):
                digits = self._read_token_bytes(4)
                raw.extend(digits)
                if len(digits) != 4 or any(
                    digit not in b"0123456789abcdefABCDEF" for digit in digits
                ):
                    raise ArtifactIdentityConflict("canonical JSON unicode escape is invalid")
            elif escape not in b'"\\/bfnrt':
                raise ArtifactIdentityConflict("canonical JSON escape is invalid")
        if not escaped:
            return raw[1:-1].decode("ascii")
        try:
            decoded = json.loads(bytes(raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIdentityConflict("canonical JSON string is invalid") from error
        if not isinstance(decoded, str):
            raise ArtifactIdentityConflict("canonical JSON string token is invalid")
        canonical = json.dumps(decoded, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        if canonical != raw:
            raise ArtifactIdentityConflict("canonical JSON string encoding is not canonical")
        return decoded

    def _number_token(self) -> int | float:
        raw = bytearray()
        while True:
            value = self._read_byte()
            if value is None:
                break
            if value not in b"+-0123456789.eE":
                self._unread_byte(value)
                break
            raw.append(value)
            if len(raw) > 128:
                raise ArtifactIdentityConflict("canonical JSON number exceeds its byte bound")
        try:
            decoded = json.loads(bytes(raw))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactIdentityConflict("canonical JSON number is invalid") from error
        if isinstance(decoded, bool):
            raise ArtifactIdentityConflict("canonical JSON number is invalid")
        if isinstance(decoded, int):
            number: int | float = decoded
        elif isinstance(decoded, float):
            number = decoded
        else:
            raise ArtifactIdentityConflict("canonical JSON number is invalid")
        try:
            canonical = json.dumps(number, allow_nan=False, separators=(",", ":")).encode("ascii")
        except ValueError as error:
            raise ArtifactIdentityConflict("canonical JSON number is non-finite") from error
        if canonical != raw:
            raise ArtifactIdentityConflict("canonical JSON number encoding is not canonical")
        return number


@functools.lru_cache(maxsize=1)
def _validator_source_sha256() -> str:
    """Hash the loaded validator module source once per process."""

    try:
        source = read_bounded_bytes(
            Path(__file__).resolve(strict=True),
            maximum_bytes=MAX_VALIDATOR_IMPLEMENTATION_SOURCE_BYTES,
        )
    except (BoundedFileIOError, OSError) as error:
        raise ArtifactIdentityConflict(
            "artifact validator implementation source cannot be fingerprinted"
        ) from error
    return hashlib.sha256(source).hexdigest()


@functools.lru_cache(maxsize=8)
def _distribution_implementation_descriptor(name: str) -> dict[str, str]:
    """Bind a dependency version to its installed wheel RECORD manifest."""

    package = distribution(name)
    record_entries = tuple(
        item for item in package.files or () if str(item).endswith(".dist-info/RECORD")
    )
    if len(record_entries) != 1:
        raise ArtifactIdentityConflict(
            f"validator dependency {name!r} lacks one installed RECORD manifest"
        )
    try:
        record = read_bounded_bytes(
            Path(record_entries[0].locate()).resolve(strict=True),
            maximum_bytes=MAX_VALIDATOR_IMPLEMENTATION_SOURCE_BYTES,
        )
    except (BoundedFileIOError, OSError) as error:
        raise ArtifactIdentityConflict(
            f"validator dependency {name!r} cannot be fingerprinted"
        ) from error
    return {
        "name": package.metadata["Name"] or name,
        "version": package.version,
        "record_sha256": hashlib.sha256(record).hexdigest(),
    }


@functools.lru_cache(maxsize=len(ArtifactProfile))
def _profile_implementation_sha256(profile: ArtifactProfile) -> str:
    dependencies = {
        ArtifactProfile.ARROW_IPC: ("pyarrow",),
        ArtifactProfile.PARQUET: ("pyarrow",),
        ArtifactProfile.AUDITED_HDF5: ("h5py", "numpy"),
        ArtifactProfile.NUMPY_NO_PICKLE: ("numpy",),
        ArtifactProfile.ONNX: ("onnx", "protobuf"),
        ArtifactProfile.SAFETENSORS: ("safetensors", "numpy"),
    }.get(profile, ())
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "source_sha256": _validator_source_sha256(),
                "runtime": sys.implementation.cache_tag,
                "dependencies": tuple(
                    _distribution_implementation_descriptor(name) for name in dependencies
                ),
            }
        )
    ).hexdigest()


def _registered_implementation_sha256(profile: ArtifactProfile, contract: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "validator_key": ArtifactProfileValidatorRegistry._generic_validator_key(profile),
                "validator_version": ArtifactProfileValidatorRegistry.REGISTRY_VERSION,
                "implementation_sha256": _profile_implementation_sha256(profile),
                "structural_contract": contract,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ArtifactProfileValidatorRegistry:
    """Closed decoder owner with immutable contracts and startup implementation identity.

    Runtime monkeypatching or subclass substitution is not a validator registration.
    New decoder behavior changes this code owner and its frozen source identity.
    """

    _arrow_contracts: Mapping[str, ArrowTableContract]
    _parquet_contracts: Mapping[str, ParquetTableContract]
    _hdf5_contracts: Mapping[str, HDF5InventoryContract]
    _numpy_contracts: Mapping[str, NumpyArrayContract]
    _generic_validation_compatibilities: Mapping[str, GenericValidationCompatibility]
    _generic_validator_implementation_compatibilities: Mapping[
        tuple[str, str, str, ArtifactProfile], GenericValidatorImplementationCompatibility
    ]
    _implementation_hashes: Mapping[tuple[ArtifactProfile, str], str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "artifact validators are closed registrations, not subclass extension points"
        )

    REGISTRY_VERSION = "1.0.0"
    _BASE_ENABLED = frozenset(
        {
            ArtifactProfile.CANONICAL_JSON,
            ArtifactProfile.JSONL_CHUNKS,
            ArtifactProfile.ONNX,
            ArtifactProfile.SAFETENSORS,
            ArtifactProfile.TEXT_PARAMETERS,
            ArtifactProfile.RAW_SOURCE_BYTES,
        }
    )

    def __init__(
        self,
        *,
        arrow_contracts: tuple[ArrowTableContract, ...] = (),
        parquet_contracts: tuple[ParquetTableContract, ...] = (),
        hdf5_contracts: tuple[HDF5InventoryContract, ...] = (),
        numpy_contracts: tuple[NumpyArrayContract, ...] = (),
        generic_validation_compatibilities: tuple[GenericValidationCompatibility, ...] = (),
        generic_validator_implementation_compatibilities: tuple[
            GenericValidatorImplementationCompatibility, ...
        ] = (),
    ) -> None:
        for label, contracts in (
            ("Arrow", arrow_contracts),
            ("Parquet", parquet_contracts),
            ("HDF5", hdf5_contracts),
            ("NumPy", numpy_contracts),
        ):
            schema_ids = tuple(contract.payload_schema for contract in contracts)
            if tuple(sorted(set(schema_ids))) != schema_ids:
                raise ValueError(f"{label} contracts must have sorted unique schema IDs")
        object.__setattr__(
            self,
            "_arrow_contracts",
            MappingProxyType({contract.payload_schema: contract for contract in arrow_contracts}),
        )
        object.__setattr__(
            self,
            "_parquet_contracts",
            MappingProxyType({contract.payload_schema: contract for contract in parquet_contracts}),
        )
        object.__setattr__(
            self,
            "_hdf5_contracts",
            MappingProxyType({contract.payload_schema: contract for contract in hdf5_contracts}),
        )
        object.__setattr__(
            self,
            "_numpy_contracts",
            MappingProxyType({contract.payload_schema: contract for contract in numpy_contracts}),
        )
        source_fingerprints = tuple(
            source.fingerprint() for source, _target in generic_validation_compatibilities
        )
        implementation_compatibility_keys = tuple(
            value.key for value in generic_validator_implementation_compatibilities
        )
        if (
            len(generic_validation_compatibilities)
            + len(generic_validator_implementation_compatibilities)
            > MAX_GENERIC_VALIDATION_COMPATIBILITIES
            or tuple(sorted(set(source_fingerprints))) != source_fingerprints
            or tuple(sorted(set(implementation_compatibility_keys)))
            != implementation_compatibility_keys
        ):
            raise ValueError(
                "artifact generic validation compatibility sources must be bounded, sorted, "
                "and unique"
            )
        target_fingerprints = {
            target.fingerprint() for _source, target in generic_validation_compatibilities
        }
        if set(source_fingerprints).intersection(target_fingerprints):
            raise ValueError("artifact generic validation compatibility chains are forbidden")
        object.__setattr__(
            self,
            "_generic_validation_compatibilities",
            MappingProxyType(
                {
                    source.fingerprint(): (source, target)
                    for source, target in generic_validation_compatibilities
                }
            ),
        )
        object.__setattr__(
            self,
            "_generic_validator_implementation_compatibilities",
            MappingProxyType(
                {value.key: value for value in generic_validator_implementation_compatibilities}
            ),
        )
        implementations = {
            (profile, ""): _registered_implementation_sha256(profile, None)
            for profile in self._BASE_ENABLED
        }
        for profile, contracts in (
            (ArtifactProfile.ARROW_IPC, arrow_contracts),
            (ArtifactProfile.PARQUET, parquet_contracts),
            (ArtifactProfile.AUDITED_HDF5, hdf5_contracts),
            (ArtifactProfile.NUMPY_NO_PICKLE, numpy_contracts),
        ):
            for contract in contracts:
                implementations[profile, contract.payload_schema] = (
                    _registered_implementation_sha256(profile, asdict(contract))
                )
        object.__setattr__(self, "_implementation_hashes", MappingProxyType(implementations))
        for source, target in generic_validation_compatibilities:
            if (
                source.validator_key != target.validator_key
                or source.profile is not target.profile
                or source.payload_schema != target.payload_schema
                or source == target
            ):
                raise ValueError(
                    "artifact generic validation compatibility changes identity domain"
                )
            expected_target = self.generic_validation(
                profile=source.profile,
                payload_schema=source.payload_schema,
            )
            if target != expected_target:
                raise ValueError(
                    "artifact generic validation compatibility target differs from "
                    "the current registration"
                )
        for value in generic_validator_implementation_compatibilities:
            if (
                value.validator_key != self._generic_validator_key(value.profile)
                or value.validator_version == self.REGISTRY_VERSION
            ):
                raise ValueError(
                    "artifact generic implementation compatibility changes identity domain"
                )

    @property
    def generic_validation_compatibilities(
        self,
    ) -> tuple[GenericValidationCompatibility, ...]:
        return tuple(
            self._generic_validation_compatibilities[key]
            for key in sorted(self._generic_validation_compatibilities)
        )

    @property
    def generic_validator_implementation_compatibilities(
        self,
    ) -> tuple[GenericValidatorImplementationCompatibility, ...]:
        return tuple(
            self._generic_validator_implementation_compatibilities[key]
            for key in sorted(self._generic_validator_implementation_compatibilities)
        )

    @property
    def enabled_profiles(self) -> tuple[ArtifactProfile, ...]:
        profiles = set(self._BASE_ENABLED)
        if self._arrow_contracts:
            profiles.add(ArtifactProfile.ARROW_IPC)
        if self._parquet_contracts:
            profiles.add(ArtifactProfile.PARQUET)
        if self._hdf5_contracts:
            profiles.add(ArtifactProfile.AUDITED_HDF5)
        if self._numpy_contracts:
            profiles.add(ArtifactProfile.NUMPY_NO_PICKLE)
        return tuple(sorted(profiles, key=lambda value: value.value))

    @staticmethod
    def _contract_for(
        contracts: Mapping[str, _ContractT],
        payload_schema: str,
        *,
        label: str,
    ) -> _ContractT:
        try:
            return contracts[payload_schema]
        except KeyError as error:
            raise ArtifactIdentityConflict(
                f"{label} payload schema has no registered structural contract"
            ) from error

    def arrow_contract(self, payload_schema: str) -> ArrowTableContract:
        return self._contract_for(
            self._arrow_contracts,
            payload_schema,
            label="Arrow",
        )

    def parquet_contract(self, payload_schema: str) -> ParquetTableContract:
        return self._contract_for(
            self._parquet_contracts,
            payload_schema,
            label="Parquet",
        )

    def hdf5_contract(self, payload_schema: str) -> HDF5InventoryContract:
        return self._contract_for(
            self._hdf5_contracts,
            payload_schema,
            label="HDF5",
        )

    def numpy_contract(self, payload_schema: str) -> NumpyArrayContract:
        return self._contract_for(
            self._numpy_contracts,
            payload_schema,
            label="NumPy",
        )

    @classmethod
    def _generic_validator_key(cls, profile: ArtifactProfile) -> str:
        suffix = profile.value.lower().replace("_", "-")
        return f"artifact-profile.{suffix}"

    def _generic_validator_implementation_sha256(
        self, profile: ArtifactProfile, payload_schema: str
    ) -> str:
        try:
            return self._implementation_hashes[
                profile, "" if profile in self._BASE_ENABLED else payload_schema
            ]
        except KeyError as error:
            label = {
                ArtifactProfile.ARROW_IPC: "Arrow",
                ArtifactProfile.PARQUET: "Parquet",
                ArtifactProfile.AUDITED_HDF5: "HDF5",
                ArtifactProfile.NUMPY_NO_PICKLE: "NumPy",
            }.get(profile, profile.value)
            raise ArtifactIdentityConflict(
                f"{label} payload schema has no registered structural contract"
            ) from error

    def generic_validation(
        self,
        *,
        profile: ArtifactProfile,
        payload_schema: str,
    ) -> ArtifactGenericValidation:
        """Resolve one exact generic validator from this immutable registry."""

        validate_schema(payload_schema)
        if profile not in self.enabled_profiles:
            raise ArtifactIdentityConflict(
                f"artifact profile validator is unavailable for {profile.value}"
            )
        return ArtifactGenericValidation(
            validator_key=self._generic_validator_key(profile),
            validator_version=self.REGISTRY_VERSION,
            validator_implementation_sha256=(
                self._generic_validator_implementation_sha256(profile, payload_schema)
            ),
            payload_schema=payload_schema,
            profile=profile,
        )

    def require_generic_validation(
        self,
        observed: ArtifactGenericValidation,
    ) -> ArtifactGenericValidation:
        """Reject manifest-owned generic validator substitution or drift."""

        expected = self.generic_validation(
            profile=observed.profile,
            payload_schema=observed.payload_schema,
        )
        if observed == expected:
            return expected
        compatibility = self._generic_validation_compatibilities.get(observed.fingerprint())
        if (
            compatibility is not None
            and compatibility[0] == observed
            and compatibility[1] == expected
        ):
            return expected
        implementation_compatibility = self._generic_validator_implementation_compatibilities.get(
            (
                observed.validator_key,
                observed.validator_version,
                observed.validator_implementation_sha256,
                observed.profile,
            )
        )
        if (
            implementation_compatibility is not None
            and observed.validator_key == expected.validator_key
            and observed.profile is expected.profile
            and observed.payload_schema == expected.payload_schema
        ):
            return expected
        if (
            observed.validator_key != expected.validator_key
            or observed.validator_version != expected.validator_version
        ):
            raise ArtifactIdentityConflict(
                "artifact generic validator identity differs from its registration"
            )
        if observed.validator_implementation_sha256 != expected.validator_implementation_sha256:
            raise ArtifactIdentityConflict(
                "artifact generic validator implementation differs from its registration"
            )
        if observed != expected:
            raise ArtifactIdentityConflict(
                "artifact generic validation differs from its registration"
            )
        return expected

    def validate(self, request: ArtifactWriteRequest) -> None:
        if request.profile not in self.enabled_profiles:
            raise ArtifactIdentityConflict(
                f"artifact profile validator is unavailable for {request.profile.value}"
            )
        if request.profile is ArtifactProfile.CANONICAL_JSON:
            with io.BytesIO(request.payload) as handle:
                _CanonicalJsonFileInspector(handle, request).validate()
            return
        validators = {
            ArtifactProfile.ARROW_IPC: self._arrow_ipc,
            ArtifactProfile.PARQUET: self._parquet,
            ArtifactProfile.AUDITED_HDF5: self._hdf5,
            ArtifactProfile.NUMPY_NO_PICKLE: self._numpy,
            ArtifactProfile.JSONL_CHUNKS: self._jsonl,
            ArtifactProfile.ONNX: self._onnx,
            ArtifactProfile.SAFETENSORS: self._safetensors,
            ArtifactProfile.TEXT_PARAMETERS: self._text,
            ArtifactProfile.RAW_SOURCE_BYTES: self._raw_source,
        }
        validator = validators.get(request.profile)
        if validator is None:
            raise ArtifactIdentityConflict(
                f"artifact profile validator is unavailable for {request.profile.value}"
            )
        validator(request)
        self._validate_semantic_payload(request, request.payload)

    def validate_file(
        self,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        """Validate a bounded on-disk payload without whole-file materialization."""

        size = path.stat().st_size
        if isinstance(request, ArtifactStreamWriteRequest) and size > request.maximum_bytes:
            raise ArtifactIdentityConflict("artifact exceeds its declared validation byte limit")
        if request.profile not in self.enabled_profiles:
            raise ArtifactIdentityConflict(
                f"artifact profile validator is unavailable for {request.profile.value}"
            )
        if request.profile is ArtifactProfile.CANONICAL_JSON:
            with path.open("rb", buffering=STREAM_CHUNK_BYTES) as handle:
                _CanonicalJsonFileInspector(handle, request).validate()
            return
        if request.profile is ArtifactProfile.RAW_SOURCE_BYTES:
            self._raw_source_contract(request, size)
            return
        self._validate_semantic_file(path, request)
        if request.profile is ArtifactProfile.JSONL_CHUNKS:
            self._jsonl_file(path, request)
            return
        if request.profile is ArtifactProfile.PARQUET:
            self._parquet_file(path, request)
            return
        if request.profile is ArtifactProfile.ARROW_IPC:
            self._arrow_file(path, request)
            return
        if request.profile is ArtifactProfile.AUDITED_HDF5:
            self._hdf5_file(path, request)
            return
        if request.profile is ArtifactProfile.NUMPY_NO_PICKLE:
            self._numpy_file(path, request)
            return
        if request.profile is ArtifactProfile.SAFETENSORS:
            self._safetensors_file(path, request)
            return
        if request.profile is ArtifactProfile.ONNX:
            if size > MAX_MODEL_PAYLOAD_BYTES:
                raise ArtifactIdentityConflict("ONNX payload exceeds its safe byte budget")
            self._onnx_file(path, request)
            return
        if request.profile is ArtifactProfile.TEXT_PARAMETERS:
            self._text_file(path)
            return
        raise ArtifactIdentityConflict("artifact profile lacks a bounded file validator")

    @staticmethod
    def _raw_source_contract(
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest, size: int
    ) -> None:
        """Custody-only bytes; no parser, executable deserializer or format claim."""

        if (
            request.payload_schema not in {
                'empirical-lawhood/source/checksummed-uninterpreted-public-bytes',
                'empirical-lawhood/source/capture-assured-uninterpreted-public-bytes',
            }
            or not 0 < size <= 64 * 1024**2
            or request.media_type not in {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "text/csv", "text/plain", "application/pdf", "application/zip",
                "chemical/x-mmcif", "application/gzip",
            }
            or (request.media_type == "application/gzip" and request.payload_schema != 'empirical-lawhood/source/capture-assured-uninterpreted-public-bytes')
            or request.semantic_validation is not None
        ):
            raise ArtifactIdentityConflict("raw source custody exceeds its uninterpreted contract")

    def _raw_source(self, request: ArtifactWriteRequest) -> None:
        self._raw_source_contract(request, len(request.payload))

    @staticmethod
    def _validate_semantic_payload(
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
        payload: bytes,
    ) -> None:
        contract = request.semantic_validation
        if contract is None or contract.profile is not ArtifactProfile.CANONICAL_JSON:
            return
        decoded = _strict_json(payload, label="registered artifact semantics")
        if not isinstance(decoded, dict) or tuple(sorted(decoded)) != contract.top_level_keys:
            raise ArtifactIdentityConflict(
                "artifact document shape differs from its semantic contract"
            )
        if decoded.get("schema") != contract.payload_schema:
            raise ArtifactIdentityConflict(
                "artifact payload schema differs from its semantic contract"
            )
        if contract.value_keys:
            value = decoded.get("value")
            if not isinstance(value, dict) or tuple(sorted(value)) != contract.value_keys:
                raise ArtifactIdentityConflict(
                    "artifact value shape differs from its semantic contract"
                )
            expected_version = dict(contract.field_bindings).get("version", "1.0.0")
            if decoded.get("version") != expected_version:
                raise ArtifactIdentityConflict(
                    "artifact record version differs from its semantic contract"
                )
        for field_name, expected in contract.field_bindings:
            if decoded.get(field_name) != expected:
                raise ArtifactIdentityConflict(
                    "artifact context binding differs from its semantic contract"
                )

    @classmethod
    def _validate_semantic_file(
        cls,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        contract = request.semantic_validation
        if contract is not None and contract.profile is ArtifactProfile.CANONICAL_JSON:
            raise AssertionError("canonical JSON file validation must use the streaming inspector")

    @staticmethod
    def _canonical_json(request: ArtifactWriteRequest) -> None:
        value = _strict_json(request.payload, label="canonical JSON profile")
        if _decoded_json_bytes(value) != request.payload:
            raise ArtifactIdentityConflict("canonical JSON bytes are not canonical")
        if not isinstance(value, dict) or value.get("schema") != request.payload_schema:
            raise ArtifactIdentityConflict("canonical JSON payload schema differs")

    @staticmethod
    def _require_arrow_contract(
        schema: pa.Schema,
        contract: ArrowTableContract,
    ) -> None:
        if len(schema) != len(contract.fields):
            raise ArtifactIdentityConflict("table field count differs from its contract")
        for observed, expected in zip(schema, contract.fields, strict=True):
            if observed.name != expected.name:
                raise ArtifactIdentityConflict("table field name or order differs")
            if str(observed.type) != expected.physical_type:
                raise ArtifactIdentityConflict("table field physical type differs")
            if observed.nullable is not expected.nullable:
                raise ArtifactIdentityConflict("table field nullability differs")
            expected_field_metadata = {
                key.encode("utf-8"): value.encode("utf-8") for key, value in expected.metadata
            }
            if (observed.metadata or {}) != expected_field_metadata:
                raise ArtifactIdentityConflict("table field metadata differs")
        if (schema.metadata or {}) != contract.expected_metadata():
            raise ArtifactIdentityConflict(
                "table schema metadata, logical types, units, frames, clocks or keys differ"
            )

    @staticmethod
    def _require_parquet_contract(
        schema: pq.ParquetSchema,
        contract: ParquetTableContract,
    ) -> None:
        if len(schema) != len(contract.columns):
            raise ArtifactIdentityConflict("Parquet physical column count differs")
        for index, expected in enumerate(contract.columns):
            observed = schema.column(index)
            if (
                observed.path != expected.path
                or observed.physical_type != expected.physical_type
                or str(observed.logical_type) != expected.logical_type
                or observed.max_definition_level != expected.max_definition_level
                or observed.max_repetition_level != expected.max_repetition_level
            ):
                raise ArtifactIdentityConflict(
                    "Parquet physical/logical column structure differs from its contract"
                )

    def validate_table_schema(
        self,
        profile: ArtifactProfile,
        payload_schema: str,
        schema: pa.Schema,
    ) -> None:
        """Validate an encoder input schema against its static profile contract."""

        if profile is ArtifactProfile.ARROW_IPC:
            contract = self.arrow_contract(payload_schema)
        elif profile is ArtifactProfile.PARQUET:
            contract = self.parquet_contract(payload_schema).arrow
        else:
            raise ArtifactIdentityConflict("table schema validation requires a table profile")
        self._require_arrow_contract(schema, contract)

    def _arrow_ipc(self, request: ArtifactWriteRequest) -> None:
        try:
            source = pa.BufferReader(request.payload)
            try:
                reader = pa.ipc.open_file(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_stream(source)
            schema = reader.schema
            reader.read_all()
        except (pa.ArrowException, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("Arrow IPC validator rejected the payload") from error
        self._require_arrow_contract(schema, self.arrow_contract(request.payload_schema))

    def _parquet(self, request: ArtifactWriteRequest) -> None:
        try:
            parquet = pq.ParquetFile(pa.BufferReader(request.payload))
            schema = parquet.schema_arrow
            if parquet.metadata is None or parquet.metadata.num_row_groups < 0:
                raise ValueError("Parquet metadata is unavailable")
            parquet.read()
        except (pa.ArrowException, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("Parquet validator rejected the payload") from error
        contract = self.parquet_contract(request.payload_schema)
        self._require_arrow_contract(schema, contract.arrow)
        self._require_parquet_contract(parquet.schema, contract)

    def _arrow_file(
        self,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        try:
            with pa.memory_map(str(path), "r") as source:
                try:
                    reader = pa.ipc.open_file(source)
                    schema = reader.schema
                    for index in range(reader.num_record_batches):
                        reader.get_batch(index)
                except pa.ArrowInvalid:
                    source.seek(0)
                    stream = pa.ipc.open_stream(source)
                    schema = stream.schema
                    for _batch in stream:
                        pass
        except (pa.ArrowException, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("Arrow IPC validator rejected the payload") from error
        self._require_arrow_contract(schema, self.arrow_contract(request.payload_schema))

    def _parquet_file(
        self,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        try:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            if parquet.metadata is None:
                raise ValueError("Parquet metadata is unavailable")
            for _batch in parquet.iter_batches(batch_size=65_536):
                pass
        except (pa.ArrowException, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("Parquet validator rejected the payload") from error
        contract = self.parquet_contract(request.payload_schema)
        self._require_arrow_contract(schema, contract.arrow)
        self._require_parquet_contract(parquet.schema, contract)

    @staticmethod
    def _hdf5_attribute_identity(value: object) -> tuple[HDF5AttributeKind, str]:
        if isinstance(value, (str, np.str_)):
            return HDF5AttributeKind.UTF8, str(value)
        if isinstance(value, (bytes, np.bytes_)):
            return HDF5AttributeKind.BYTES, bytes(value).hex()
        if isinstance(value, (bool, np.bool_)):
            return HDF5AttributeKind.BOOLEAN, "true" if bool(value) else "false"
        if isinstance(value, (int, np.integer)):
            return HDF5AttributeKind.INTEGER, str(int(value))
        if isinstance(value, (float, np.floating)):
            decoded = float(value)
            if not math.isfinite(decoded):
                raise ArtifactIdentityConflict("HDF5 non-finite attributes are forbidden")
            return HDF5AttributeKind.FLOAT, decoded.hex()
        raise ArtifactIdentityConflict("HDF5 attributes must be bounded scalar values")

    @classmethod
    def _bounded_hdf5_attribute_names(
        cls,
        value: h5py.Group | h5py.Dataset,
        expected: tuple[HDF5AttributeContract, ...],
        *,
        maximum_name_bytes: int,
    ) -> tuple[str, ...]:
        expected_names = frozenset(attribute.name for attribute in expected)
        observed: list[str] = []
        observed_bytes = 0
        failure: str | None = None

        def visit(raw_name: bytes) -> int | None:
            nonlocal failure, observed_bytes
            if len(observed) >= len(expected):
                failure = "HDF5 attributes exceed the registered attribute count"
                return 1
            observed_bytes += len(raw_name)
            if observed_bytes > maximum_name_bytes:
                failure = "HDF5 attributes exceed decoded-work limit"
                return 1
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError:
                failure = "HDF5 attribute name is not UTF-8"
                return 1
            if name not in expected_names:
                failure = "HDF5 object attributes or metadata differ from contract"
                return 1
            observed.append(name)
            return None

        h5py.h5a.iterate(
            value.id,
            visit,
            index_type=h5py.h5.INDEX_NAME,
            order=h5py.h5.ITER_INC,
        )
        if failure is not None:
            raise ArtifactIdentityConflict(failure)
        return tuple(observed)

    @classmethod
    def _require_hdf5_attributes(
        cls,
        value: h5py.Group | h5py.Dataset,
        expected: tuple[HDF5AttributeContract, ...],
        *,
        remaining_attribute_bytes: int,
    ) -> int:
        names = cls._bounded_hdf5_attribute_names(
            value,
            expected,
            maximum_name_bytes=remaining_attribute_bytes,
        )
        if not {attribute.name for attribute in expected if attribute.required}.issubset(names):
            raise ArtifactIdentityConflict(
                "HDF5 object attributes or metadata differ from contract"
            )
        consumed = 0
        for contract in expected:
            if contract.name not in names:
                continue
            attribute_id = None
            try:
                attribute_id = h5py.h5a.open(value.id, contract.name.encode("utf-8"))
                data_type = attribute_id.get_type()
                data_space = attribute_id.get_space()
                points = int(data_space.get_simple_extent_npoints())
                dimensions = int(data_space.get_simple_extent_ndims())
                type_class = data_type.get_class()
                encoded_size = int(data_type.get_size()) * points
                storage_size = int(attribute_id.get_storage_size())
                variable_string_check = getattr(data_type, "is_variable_str", None)
                variable_string = bool(
                    variable_string_check() if variable_string_check is not None else False
                )
            except (KeyError, RuntimeError, ValueError) as error:
                raise ArtifactIdentityConflict(
                    "HDF5 attribute metadata could not be bounded"
                ) from error
            finally:
                if attribute_id is not None:
                    attribute_id.close()
            if points != 1 or dimensions != 0:
                raise ArtifactIdentityConflict("HDF5 attributes must be scalar")
            if type_class not in {h5py.h5t.STRING, h5py.h5t.INTEGER, h5py.h5t.FLOAT, h5py.h5t.ENUM}:
                raise ArtifactIdentityConflict("HDF5 attribute type is outside the scalar contract")
            if variable_string:
                raise ArtifactIdentityConflict(
                    "HDF5 variable-length attributes are forbidden by the bounded profile"
                )
            if (
                encoded_size <= 0
                or encoded_size > remaining_attribute_bytes - consumed
                or storage_size > remaining_attribute_bytes - consumed
            ):
                raise ArtifactIdentityConflict("HDF5 attributes exceed decoded-work limit")
            kind, observed = cls._hdf5_attribute_identity(value.attrs[contract.name])
            if kind is HDF5AttributeKind.BYTES and contract.kind is HDF5AttributeKind.UTF8:
                try:
                    observed = bytes.fromhex(observed).rstrip(b"\0").decode("utf-8")
                except (UnicodeDecodeError, ValueError) as error:
                    raise ArtifactIdentityConflict(
                        "HDF5 UTF-8 attribute encoding is invalid"
                    ) from error
                kind = HDF5AttributeKind.UTF8
            consumed += len(contract.name.encode("utf-8")) + len(observed.encode("utf-8"))
            if consumed > remaining_attribute_bytes:
                raise ArtifactIdentityConflict("HDF5 attributes exceed decoded-work limit")
            if kind is not contract.kind or (
                contract.value is not None and observed != contract.value
            ):
                if contract.name == "empirical_lawhood_payload_schema":
                    raise ArtifactIdentityConflict(
                        "HDF5 payload schema metadata differs from contract"
                    )
                raise ArtifactIdentityConflict("HDF5 attribute value differs from contract")
        return consumed

    @staticmethod
    def _bounded_hdf5_link_names(
        value: h5py.Group,
        *,
        path: str,
        expected_paths: frozenset[str],
        maximum_names: int,
        maximum_name_bytes: int,
    ) -> tuple[tuple[str, ...], int]:
        observed: list[str] = []
        observed_bytes = 0
        failure: str | None = None

        def visit(raw_name: bytes) -> int | None:
            nonlocal failure, observed_bytes
            if len(observed) >= maximum_names:
                failure = "HDF5 inventory exceeds object limit"
                return 1
            observed_bytes += len(raw_name)
            if observed_bytes > maximum_name_bytes:
                failure = "HDF5 object names exceed decoded-work limit"
                return 1
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError:
                failure = "HDF5 object name is not UTF-8"
                return 1
            child_path = f"/{name}" if path == "/" else f"{path}/{name}"
            if child_path not in expected_paths:
                failure = "HDF5 inventory contains an unregistered path"
                return 1
            observed.append(name)
            return None

        value.id.links.iterate(
            visit,
            idx_type=h5py.h5.INDEX_NAME,
            order=h5py.h5.ITER_INC,
        )
        if failure is not None:
            raise ArtifactIdentityConflict(failure)
        return tuple(observed), observed_bytes

    @staticmethod
    def _require_hdf5_creation_contract(
        value: h5py.Group | h5py.Dataset,
        contract: HDF5ObjectContract,
    ) -> None:
        creation = value.id.get_create_plist()
        if isinstance(value, h5py.Group):
            flags = int(creation.get_link_creation_order())
            observed_track = bool(flags & h5py.h5p.CRT_ORDER_TRACKED)
            observed_index = bool(flags & h5py.h5p.CRT_ORDER_INDEXED)
            if (
                contract.track_link_creation_order is not None
                and observed_track is not contract.track_link_creation_order
            ):
                raise ArtifactIdentityConflict(
                    "HDF5 link creation-order tracking differs from contract"
                )
            if (
                contract.index_link_creation_order is not None
                and observed_index is not contract.index_link_creation_order
            ):
                raise ArtifactIdentityConflict(
                    "HDF5 link creation-order indexing differs from contract"
                )

    @staticmethod
    def _canonical_hdf5_filter_options(value: object) -> str:
        def normalize(item: object) -> object:
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, tuple):
                return [normalize(child) for child in item]
            if isinstance(item, list):
                return [normalize(child) for child in item]
            return item

        try:
            return json.dumps(
                normalize(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ArtifactIdentityConflict(
                "HDF5 compression options cannot be canonicalized"
            ) from error

    @classmethod
    def _require_hdf5_dataset_storage(
        cls,
        value: h5py.Dataset,
        contract: HDF5DatasetStorageContract,
    ) -> None:
        if value.chunks != contract.chunk_shape:
            raise ArtifactIdentityConflict("HDF5 dataset chunk shape differs from contract")
        if value.compression != contract.compression_filter:
            raise ArtifactIdentityConflict("HDF5 dataset compression filter differs from contract")
        observed_options = (
            None
            if value.compression_opts is None
            else cls._canonical_hdf5_filter_options(value.compression_opts)
        )
        if observed_options != contract.compression_options:
            raise ArtifactIdentityConflict("HDF5 dataset compression options differ from contract")
        if bool(value.shuffle) is not contract.shuffle_filter_enabled:
            raise ArtifactIdentityConflict("HDF5 dataset shuffle filter differs from contract")
        if bool(value.fletcher32) is not contract.fletcher32_checksum_enabled:
            raise ArtifactIdentityConflict("HDF5 dataset checksum filter differs from contract")
        if value.scaleoffset != contract.scale_offset:
            raise ArtifactIdentityConflict("HDF5 dataset scale-offset differs from contract")
        if contract.zero_fill_value_required:
            fill = np.asarray(value.fillvalue)
            if fill.shape != () or not bool(np.equal(fill, 0)):
                raise ArtifactIdentityConflict("HDF5 dataset fill value differs from contract")
        if contract.external_storage_forbidden and value.external:
            raise ArtifactIdentityConflict("HDF5 external dataset storage is forbidden")

    @staticmethod
    def _bounded_hdf5_dataset_slices(
        shape: tuple[int, ...],
        *,
        itemsize: int,
    ) -> Iterator[tuple[slice, ...]]:
        """Tile any dataset rank into hyperslabs no larger than one I/O chunk."""

        if itemsize < 1 or itemsize > STREAM_CHUNK_BYTES:
            raise ArtifactIdentityConflict("HDF5 element exceeds the bounded validator")
        if not shape or any(dimension == 0 for dimension in shape):
            return
        remaining_elements = STREAM_CHUNK_BYTES // itemsize
        block_shape = [1] * len(shape)
        for axis in range(len(shape) - 1, -1, -1):
            width = min(shape[axis], remaining_elements)
            block_shape[axis] = max(1, width)
            remaining_elements = max(1, remaining_elements // block_shape[axis])
        starts = tuple(
            range(0, dimension, block) for dimension, block in zip(shape, block_shape, strict=True)
        )
        for origin in itertools.product(*starts):
            yield tuple(
                slice(start, min(start + block, dimension))
                for start, block, dimension in zip(
                    origin,
                    block_shape,
                    shape,
                    strict=True,
                )
            )

    @classmethod
    def _require_hdf5_contract(
        cls,
        handle: h5py.File,
        contract: HDF5InventoryContract,
    ) -> None:
        expected = {item.path: item for item in contract.objects}
        observed_paths: list[str] = []
        visited: set[int] = set()
        total_elements = 0
        total_decoded_bytes = 0
        total_name_bytes = 0
        total_attribute_bytes = 0

        def inspect(value: h5py.Group | h5py.Dataset, path: str) -> None:
            nonlocal total_attribute_bytes, total_decoded_bytes, total_elements
            nonlocal total_name_bytes
            if len(observed_paths) >= contract.maximum_objects:
                raise ArtifactIdentityConflict("HDF5 inventory exceeds object limit")
            address = int(h5py.h5o.get_info(value.id).addr)
            if address in visited:
                raise ArtifactIdentityConflict("HDF5 hard-link aliases or cycles are forbidden")
            visited.add(address)
            observed_paths.append(path)
            object_contract = expected.get(path)
            if object_contract is None:
                raise ArtifactIdentityConflict("HDF5 inventory contains an unregistered path")
            observed_kind = (
                HDF5ObjectKind.GROUP if isinstance(value, h5py.Group) else HDF5ObjectKind.DATASET
            )
            if observed_kind is not object_contract.kind:
                raise ArtifactIdentityConflict("HDF5 object kind differs from contract")
            cls._require_hdf5_creation_contract(value, object_contract)
            total_attribute_bytes += cls._require_hdf5_attributes(
                value,
                object_contract.attributes,
                remaining_attribute_bytes=(
                    contract.maximum_attribute_bytes - total_attribute_bytes
                ),
            )
            if isinstance(value, h5py.Dataset):
                # Dataset storage is another I/O/decoder boundary even when
                # inventory, dimensions and dtype are all registered.
                if value.external:
                    raise ArtifactIdentityConflict("HDF5 external dataset storage is forbidden")
                if object_contract.storage is None and value.id.get_create_plist().get_nfilters():
                    raise ArtifactIdentityConflict("HDF5 dataset uses unregistered filters")
                if value.dtype.hasobject or value.dtype.fields is not None or value.is_virtual:
                    raise ArtifactIdentityConflict(
                        "HDF5 object, structured and virtual datasets are forbidden"
                    )
                if object_contract.dtype is None or object_contract.shape is None:
                    raise ArtifactIdentityConflict("HDF5 dataset contract is incomplete")
                if value.dtype.str != np.dtype(object_contract.dtype).str:
                    raise ArtifactIdentityConflict("HDF5 dataset dtype differs from contract")
                if object_contract.storage is not None:
                    cls._require_hdf5_dataset_storage(value, object_contract.storage)
                if object_contract.maximum_shape is None:
                    shape_valid = value.shape == object_contract.shape
                else:
                    shape_valid = len(value.shape) == len(object_contract.shape) and all(
                        lower <= observed <= upper
                        for observed, lower, upper in zip(
                            value.shape,
                            object_contract.shape,
                            object_contract.maximum_shape,
                            strict=True,
                        )
                    )
                if not shape_valid:
                    raise ArtifactIdentityConflict("HDF5 dataset shape differs from contract")
                elements = int(math.prod(value.shape)) if value.shape else 1
                decoded_bytes = elements * value.dtype.itemsize
                if elements > contract.maximum_dataset_elements:
                    raise ArtifactIdentityConflict("HDF5 dataset exceeds element limit")
                total_elements += elements
                total_decoded_bytes += decoded_bytes
                if (
                    total_elements > contract.maximum_total_elements
                    or total_decoded_bytes > contract.maximum_decoded_bytes
                ):
                    raise ArtifactIdentityConflict("HDF5 decoded work exceeds its contract")
                if value.dtype.itemsize > STREAM_CHUNK_BYTES:
                    raise ArtifactIdentityConflict("HDF5 element exceeds the bounded validator")
                if not value.shape:
                    value[()]
                    return
                for selection in cls._bounded_hdf5_dataset_slices(
                    value.shape,
                    itemsize=value.dtype.itemsize,
                ):
                    value[selection]
                return
            child_names, consumed_name_bytes = cls._bounded_hdf5_link_names(
                value,
                path=path,
                expected_paths=frozenset(expected),
                maximum_names=contract.maximum_objects - len(observed_paths),
                maximum_name_bytes=contract.maximum_name_bytes - total_name_bytes,
            )
            total_name_bytes += consumed_name_bytes
            # Required children are conditional on the containing group being
            # present. Optional groups still have a closed, typed inventory.
            required_children = {
                item.path.rsplit("/", 1)[1]
                for item in contract.objects
                if item.path != "/"
                and item.required
                and (item.path.rsplit("/", 1)[0] or "/") == path
            }
            if not required_children.issubset(child_names):
                raise ArtifactIdentityConflict("HDF5 inventory is missing a required object")
            for name in child_names:
                link = value.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ArtifactIdentityConflict("HDF5 external and symbolic links are forbidden")
                child = value[name]
                if not isinstance(child, (h5py.Group, h5py.Dataset)):
                    raise ArtifactIdentityConflict("HDF5 inventory contains an unsupported object")
                child_path = f"/{name}" if path == "/" else f"{path}/{name}"
                inspect(child, child_path)

        inspect(handle, "/")

    def _hdf5(self, request: ArtifactWriteRequest) -> None:
        try:
            with h5py.File(io.BytesIO(request.payload), "r") as handle:
                self._require_hdf5_contract(
                    handle,
                    self.hdf5_contract(request.payload_schema),
                )
        except ArtifactIdentityConflict:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ArtifactIdentityConflict("HDF5 validator rejected the payload") from error

    def _hdf5_file(
        self,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        try:
            # A sealed memfd has no filesystem realpath for HDF5's native
            # driver. Reopen the supplied descriptor path as a Python stream:
            # this owns an independent file description/offset, so validation
            # neither closes nor advances the caller's bounded input reader.
            with path.open("rb") as source, h5py.File(source, "r") as handle:
                self._require_hdf5_contract(
                    handle,
                    self.hdf5_contract(request.payload_schema),
                )
        except ArtifactIdentityConflict:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ArtifactIdentityConflict("HDF5 validator rejected the payload") from error

    def _numpy(self, request: ArtifactWriteRequest) -> None:
        contract = self._numpy_contracts.get(request.payload_schema)
        if contract is None:
            raise ArtifactIdentityConflict("NumPy payload schema has no registered contract")
        try:
            source = io.BytesIO(request.payload)
            value = np.load(source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ArtifactIdentityConflict("NumPy validator rejected the payload") from error
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ArtifactIdentityConflict("NumPy payload must be one non-object ndarray")
        if source.tell() != len(request.payload):
            raise ArtifactIdentityConflict("NumPy payload contains trailing bytes")
        if value.dtype != np.dtype(contract.dtype) or not contract.accepts_shape(value.shape):
            raise ArtifactIdentityConflict("NumPy dtype or shape differs from its contract")

    def _numpy_file(
        self,
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        contract = self._numpy_contracts.get(request.payload_schema)
        if contract is None:
            raise ArtifactIdentityConflict("NumPy payload schema has no registered contract")
        try:
            value = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ArtifactIdentityConflict("NumPy validator rejected the payload") from error
        if not isinstance(value, np.memmap) or value.dtype.hasobject:
            raise ArtifactIdentityConflict("NumPy payload must be one non-object ndarray")
        if value.dtype != np.dtype(contract.dtype) or not contract.accepts_shape(value.shape):
            raise ArtifactIdentityConflict("NumPy dtype or shape differs from its contract")
        if value.offset + value.nbytes != path.stat().st_size:
            raise ArtifactIdentityConflict("NumPy payload contains trailing or truncated bytes")

    @staticmethod
    def _jsonl(request: ArtifactWriteRequest) -> None:
        if not request.payload.endswith(b"\n") or b"\r" in request.payload:
            raise ArtifactIdentityConflict("JSONL bytes require canonical LF termination")
        lines = request.payload.splitlines()
        if not lines:
            raise ArtifactIdentityConflict("JSONL payload must contain records")
        for index, line in enumerate(lines):
            value = _strict_json(line, label="JSONL profile")
            if _decoded_json_bytes(value).rstrip(b"\n") != line:
                raise ArtifactIdentityConflict("JSONL record bytes are not canonical")
            if not isinstance(value, dict) or value.get("schema") != request.payload_schema:
                raise ArtifactIdentityConflict("JSONL payload schema differs")
            if value.get("sequence") != index:
                raise ArtifactIdentityConflict("JSONL sequence is not contiguous")

    @staticmethod
    def _jsonl_file(
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        count = 0
        with path.open("rb") as handle:
            while True:
                line = handle.readline(MAX_JSONL_RECORD_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_JSONL_RECORD_BYTES:
                    raise ArtifactIdentityConflict("JSONL record exceeds its byte bound")
                if not line.endswith(b"\n") or b"\r" in line:
                    raise ArtifactIdentityConflict("JSONL bytes require canonical LF termination")
                record = line.removesuffix(b"\n")
                value = _strict_json(record, label="JSONL profile")
                if _decoded_json_bytes(value).rstrip(b"\n") != record:
                    raise ArtifactIdentityConflict("JSONL record bytes are not canonical")
                if not isinstance(value, dict) or value.get("schema") != request.payload_schema:
                    raise ArtifactIdentityConflict("JSONL payload schema differs")
                if value.get("sequence") != count:
                    raise ArtifactIdentityConflict("JSONL sequence is not contiguous")
                count += 1
        if count == 0:
            raise ArtifactIdentityConflict("JSONL payload must contain records")

    @staticmethod
    def _onnx_graph_tensors(graph: onnx.GraphProto) -> tuple[onnx.TensorProto, ...]:
        tensors: list[onnx.TensorProto] = list(graph.initializer)
        for sparse in graph.sparse_initializer:
            tensors.extend((sparse.values, sparse.indices))
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    tensors.append(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    tensors.extend(attribute.tensors)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    tensors.extend(
                        (attribute.sparse_tensor.values, attribute.sparse_tensor.indices)
                    )
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse in attribute.sparse_tensors:
                        tensors.extend((sparse.values, sparse.indices))
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    tensors.extend(
                        ArtifactProfileValidatorRegistry._onnx_graph_tensors(attribute.g)
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        tensors.extend(ArtifactProfileValidatorRegistry._onnx_graph_tensors(nested))
        return tuple(tensors)

    @staticmethod
    def _onnx(request: ArtifactWriteRequest) -> None:
        if len(request.payload) > MAX_MODEL_PAYLOAD_BYTES:
            raise ArtifactIdentityConflict("ONNX payload exceeds its safe byte budget")
        try:
            model = onnx.load_model_from_string(request.payload)
            onnx.checker.check_model(model, full_check=True)
        except Exception as error:
            raise ArtifactIdentityConflict("ONNX validator rejected the payload") from error
        ArtifactProfileValidatorRegistry._validate_onnx_model(
            model,
            request.payload_schema,
        )

    @staticmethod
    def _onnx_file(
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        try:
            model = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(model, full_check=True)
        except Exception as error:
            raise ArtifactIdentityConflict("ONNX validator rejected the payload") from error
        ArtifactProfileValidatorRegistry._validate_onnx_model(
            model,
            request.payload_schema,
        )

    @staticmethod
    def _validate_onnx_model(model: onnx.ModelProto, payload_schema: str) -> None:
        metadata: dict[str, str] = {}
        for item in model.metadata_props:
            if item.key in metadata:
                raise ArtifactIdentityConflict("ONNX metadata contains a duplicate key")
            metadata[item.key] = item.value
        if metadata.get("empirical_lawhood_payload_schema") != payload_schema:
            raise ArtifactIdentityConflict("ONNX payload schema metadata differs")
        if any(
            tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
            for tensor in ArtifactProfileValidatorRegistry._onnx_graph_tensors(model.graph)
        ):
            raise ArtifactIdentityConflict("ONNX external tensor data is forbidden")

    @staticmethod
    def _safetensors(request: ArtifactWriteRequest) -> None:
        if len(request.payload) > MAX_MODEL_PAYLOAD_BYTES:
            raise ArtifactIdentityConflict("SafeTensors payload exceeds its safe byte budget")
        if len(request.payload) < 8:
            raise ArtifactIdentityConflict("SafeTensors payload is truncated")
        header_size = int.from_bytes(request.payload[:8], byteorder="little")
        if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
            raise ArtifactIdentityConflict("SafeTensors header exceeds its safe byte budget")
        header_end = 8 + header_size
        if header_end > len(request.payload):
            raise ArtifactIdentityConflict("SafeTensors header is truncated")
        header = _strict_json(
            request.payload[8:header_end],
            label="SafeTensors header",
        )
        if not isinstance(header, dict):
            raise ArtifactIdentityConflict("SafeTensors header is not an object")
        metadata = header.get("__metadata__")
        if (
            not isinstance(metadata, dict)
            or metadata.get("empirical_lawhood_payload_schema") != request.payload_schema
        ):
            raise ArtifactIdentityConflict("SafeTensors payload schema metadata differs")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ArtifactIdentityConflict("SafeTensors metadata must contain strings")
        try:
            tensors = load_safetensors(request.payload)
        except (SafetensorError, ValueError) as error:
            raise ArtifactIdentityConflict("SafeTensors validator rejected the payload") from error
        if not tensors:
            raise ArtifactIdentityConflict("SafeTensors payload must contain a tensor")
        if any(value.dtype.hasobject for value in tensors.values()):
            raise ArtifactIdentityConflict("SafeTensors object arrays are forbidden")

    @staticmethod
    def _safetensors_file(
        path: Path,
        request: ArtifactWriteRequest | ArtifactStreamWriteRequest,
    ) -> None:
        if path.stat().st_size > MAX_MODEL_PAYLOAD_BYTES:
            raise ArtifactIdentityConflict("SafeTensors payload exceeds its safe byte budget")
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ArtifactIdentityConflict("SafeTensors payload is truncated")
            header_size = int.from_bytes(prefix, byteorder="little")
            if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ArtifactIdentityConflict("SafeTensors header exceeds its safe byte budget")
            header_payload = handle.read(header_size)
            if len(header_payload) != header_size:
                raise ArtifactIdentityConflict("SafeTensors header is truncated")
        header = _strict_json(header_payload, label="SafeTensors header")
        if not isinstance(header, dict):
            raise ArtifactIdentityConflict("SafeTensors header is not an object")
        metadata = header.get("__metadata__")
        if (
            not isinstance(metadata, dict)
            or metadata.get("empirical_lawhood_payload_schema") != request.payload_schema
        ):
            raise ArtifactIdentityConflict("SafeTensors payload schema metadata differs")
        try:
            with safe_open(path, framework="numpy") as tensors:
                keys = tuple(tensors.keys())
                if not keys:
                    raise ArtifactIdentityConflict("SafeTensors payload must contain a tensor")
                for key in keys:
                    tensors.get_slice(key).get_shape()
        except ArtifactIdentityConflict:
            raise
        except (SafetensorError, OSError, ValueError) as error:
            raise ArtifactIdentityConflict("SafeTensors validator rejected the payload") from error

    @staticmethod
    def _text(request: ArtifactWriteRequest) -> None:
        if len(request.payload) > MAX_TEXT_PARAMETER_BYTES:
            raise ArtifactIdentityConflict("bounded text profile exceeds its byte budget")
        try:
            decoded = request.payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactIdentityConflict(
                "bounded text profile validator rejected payload"
            ) from error
        if "\x00" in decoded:
            raise ArtifactIdentityConflict(
                "bounded text profile validator rejected a NUL character"
            )

    @staticmethod
    def _text_file(path: Path) -> None:
        if path.stat().st_size > MAX_TEXT_PARAMETER_BYTES:
            raise ArtifactIdentityConflict("bounded text profile exceeds its byte budget")
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    if "\x00" in decoder.decode(chunk, final=False):
                        raise ArtifactIdentityConflict(
                            "bounded text profile validator rejected a NUL character"
                        )
            if "\x00" in decoder.decode(b"", final=True):
                raise ArtifactIdentityConflict(
                    "bounded text profile validator rejected a NUL character"
                )
        except UnicodeDecodeError as error:
            raise ArtifactIdentityConflict(
                "bounded text profile validator rejected payload"
            ) from error


def parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )
    return bytes(sink.getvalue().to_pybytes())


def numpy_no_pickle_bytes(array: npt.NDArray[np.generic]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def validate_numpy_no_pickle(payload: bytes) -> npt.NDArray[np.generic]:
    buffer = io.BytesIO(payload)
    value = np.load(buffer, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("safe NumPy payload did not decode to an array")
    return value
