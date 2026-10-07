"""Strict normalized comparison with the frozen historical 20-column CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from io import StringIO
import math
import re
from typing import BinaryIO, ClassVar

import pyarrow as pa  # type: ignore[import-untyped]

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_sha256,
)

from .contracts import CANONICAL_COLUMNS, GlennAdapterError
from .transform import canonical_table_logical_sha256, glenn_arrow_schema


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_HISTORICAL_CSV_BYTES = 64 * 1024**2
_MAXIMUM_RECORDED_MISMATCHES = 64
_NULLABLE_FLOATS = frozenset({"prior_fitness", "prior_focal_r50_um"})
_BOOLEAN_FIELDS = frozenset({"prior_fitness_missing", "prior_focal_r50_missing"})
_INTEGER_FIELDS = frozenset({"burst", "acquisition_order"})


class GlennEquivalenceClassification(StrEnum):
    """An operational table comparison, never a scientific status."""

    LOGICALLY_EQUIVALENT = "LOGICALLY_EQUIVALENT"
    LOGICALLY_DIFFERENT = "LOGICALLY_DIFFERENT"


@dataclass(slots=True)
class GlennHistoricalCSVInput:
    """Exact small comparison artifact opened behind the caller's storage guard."""

    stream: BinaryIO
    locator: str
    guard_evidence_id: str
    expected_size_bytes: int
    expected_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("locator", self.locator),
            ("guard_evidence_id", self.guard_evidence_id),
        ):
            if not value or value != value.strip() or "\x00" in value or len(value) > 4096:
                raise ValueError(f"{name} must be a bounded nonblank exact string")
        if not 0 < self.expected_size_bytes <= _MAXIMUM_HISTORICAL_CSV_BYTES:
            raise ValueError("historical CSV size is outside its absolute byte ceiling")
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise ValueError("historical CSV SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class GlennMismatch(CanonicalRecord):
    """One bounded mismatch location; source values are deliberately omitted."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-mismatch'

    row_index: int
    column_name: str
    reason_code: str

    @property
    def mismatch_id(self) -> str:
        return f"mismatch.{self.column_name.replace('_', '-')}.{self.row_index}"

    def __post_init__(self) -> None:
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int):
            raise ValueError("equivalence mismatch row index must be an integer")
        if self.row_index < 0:
            raise ValueError("equivalence mismatch row index cannot be negative")
        if not self.column_name or self.column_name != self.column_name.strip():
            raise ValueError("equivalence mismatch column name must be exact")
        if self.reason_code not in {"NULL_MISMATCH", "VALUE_MISMATCH"}:
            raise ValueError("equivalence mismatch has an unknown reason code")


@dataclass(frozen=True, slots=True)
class GlennLogicalEquivalence(CanonicalRecord):
    """Operational equivalence evidence with an explicit non-lineage boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-logical-equivalence'

    classification: GlennEquivalenceClassification
    equivalent: bool
    normalization_profile_id: str
    transformed_logical_sha256: str
    historical_normalized_logical_sha256: str
    historical_physical_sha256: str
    historical_size_bytes: int
    historical_locator: str
    historical_guard_evidence_id: str
    row_count: int
    column_count: int
    mismatch_count: int
    recorded_mismatches: tuple[GlennMismatch, ...]
    mismatches_truncated: bool
    historical_artifact_role: str = "COMPARISON_ONLY_NON_LINEAGE_PARENT"
    scientific_verdict_issued: bool = False
    lineage_parent_created: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.classification, GlennEquivalenceClassification):
            raise ValueError("equivalence classification is invalid")
        expected_classification = (
            GlennEquivalenceClassification.LOGICALLY_EQUIVALENT
            if self.equivalent
            else GlennEquivalenceClassification.LOGICALLY_DIFFERENT
        )
        if self.classification is not expected_classification:
            raise ValueError("equivalence flag and classification differ")
        for field_name, digest in (
            ("transformed_logical_sha256", self.transformed_logical_sha256),
            (
                "historical_normalized_logical_sha256",
                self.historical_normalized_logical_sha256,
            ),
            ("historical_physical_sha256", self.historical_physical_sha256),
        ):
            validate_sha256(digest, field_name=field_name)
        for field_name, numeric_value in (
            ("historical_size_bytes", self.historical_size_bytes),
            ("row_count", self.row_count),
            ("column_count", self.column_count),
            ("mismatch_count", self.mismatch_count),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value < 0
            ):
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.row_count <= 0 or self.column_count <= 0 or self.historical_size_bytes <= 0:
            raise ValueError("equivalence dimensions and historical bytes must be positive")
        require_sorted_unique_ids(
            self.recorded_mismatches,
            attribute="mismatch_id",
            field_name="recorded_mismatches",
        )
        if len(self.recorded_mismatches) > self.mismatch_count:
            raise ValueError("recorded mismatches exceed the mismatch count")
        if self.mismatches_truncated != (self.mismatch_count > len(self.recorded_mismatches)):
            raise ValueError("mismatch truncation flag differs from its counts")
        if self.equivalent != (
            self.mismatch_count == 0
            and self.transformed_logical_sha256 == self.historical_normalized_logical_sha256
        ):
            raise ValueError("equivalence status differs from logical identities/mismatches")
        if self.historical_artifact_role != "COMPARISON_ONLY_NON_LINEAGE_PARENT":
            raise ValueError("historical comparison cannot become a transformation parent")
        if self.scientific_verdict_issued or self.lineage_parent_created:
            raise ValueError("operational equivalence cannot issue science or lineage")


def _read_historical_payload(source: GlennHistoricalCSVInput) -> bytes:
    try:
        if not source.stream.readable() or not source.stream.seekable():
            raise GlennAdapterError("historical CSV input must be readable and seekable")
        source.stream.seek(0)
        payload = source.stream.read(source.expected_size_bytes + 1)
        source.stream.seek(0)
    except GlennAdapterError:
        raise
    except (AttributeError, OSError, ValueError) as error:
        raise GlennAdapterError("historical CSV input cannot be read safely") from error
    if not isinstance(payload, bytes) or len(payload) != source.expected_size_bytes:
        raise GlennAdapterError("historical CSV byte count differs from its exact expectation")
    if sha256(payload).hexdigest() != source.expected_sha256:
        raise GlennAdapterError("historical CSV SHA-256 differs from its exact expectation")
    if b"\x00" in payload:
        raise GlennAdapterError("historical CSV contains forbidden NUL bytes")
    return payload


def _finite_float(token: str, *, column: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise GlennAdapterError(f"historical column {column!r} is not numeric") from error
    if not math.isfinite(value):
        raise GlennAdapterError(f"historical column {column!r} must be finite")
    return value


def _normalize_cell(token: str, *, column: str) -> object:
    if column == "run":
        if not token or token != token.strip():
            raise GlennAdapterError("historical run identity must be nonblank and exact")
        return token
    if column in _NULLABLE_FLOATS:
        return None if token == "" else _finite_float(token, column=column)
    if token == "":
        raise GlennAdapterError(f"historical nonnullable column {column!r} contains an empty token")
    if column in _BOOLEAN_FIELDS:
        numeric = _finite_float(token, column=column)
        if numeric not in {0.0, 1.0}:
            raise GlennAdapterError(f"historical missing flag {column!r} must normalize from 0/1")
        return numeric == 1.0
    if column in _INTEGER_FIELDS:
        numeric = _finite_float(token, column=column)
        if not numeric.is_integer():
            raise GlennAdapterError(f"historical integer column {column!r} is not integral")
        return int(numeric)
    return _finite_float(token, column=column)


def _normalize_historical_csv(payload: bytes, *, expected_rows: int) -> pa.Table:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GlennAdapterError("historical CSV is not exact UTF-8") from error
    try:
        rows = list(csv.reader(StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise GlennAdapterError("historical CSV violates strict RFC 4180 parsing") from error
    if not rows or tuple(rows[0]) != CANONICAL_COLUMNS:
        raise GlennAdapterError(
            "historical CSV column order differs from the exact 20-column schema"
        )
    data_rows = rows[1:]
    if len(data_rows) != expected_rows:
        raise GlennAdapterError("historical CSV row count differs from the transformed_table table")
    columns: dict[str, list[object]] = {name: [] for name in CANONICAL_COLUMNS}
    for row_index, row in enumerate(data_rows, start=1):
        if len(row) != len(CANONICAL_COLUMNS):
            raise GlennAdapterError(f"historical CSV row {row_index} has the wrong field count")
        for column, token in zip(CANONICAL_COLUMNS, row, strict=True):
            columns[column].append(_normalize_cell(token, column=column))
    schema = glenn_arrow_schema()
    arrays = [pa.array(columns[name], type=schema.field(name).type) for name in CANONICAL_COLUMNS]
    table = pa.Table.from_arrays(arrays, schema=schema)
    # Computing the digest also applies the full native-null/order/finite contract.
    canonical_table_logical_sha256(table)
    return table


def _cell_token(value: object) -> object:
    if isinstance(value, float):
        return ("float64", value.hex())
    return value


def _mismatches(transformed_table: pa.Table, historical: pa.Table) -> tuple[int, tuple[GlennMismatch, ...]]:
    total = 0
    recorded: list[GlennMismatch] = []
    for column in CANONICAL_COLUMNS:
        transformed_values = transformed_table[column].to_pylist()
        historical_values = historical[column].to_pylist()
        for row_index, (current, prior) in enumerate(
            zip(transformed_values, historical_values, strict=True)
        ):
            if _cell_token(current) == _cell_token(prior):
                continue
            total += 1
            if len(recorded) < _MAXIMUM_RECORDED_MISMATCHES:
                reason = (
                    "NULL_MISMATCH" if (current is None) != (prior is None) else "VALUE_MISMATCH"
                )
                recorded.append(
                    GlennMismatch(
                        row_index=row_index,
                        column_name=column,
                        reason_code=reason,
                    )
                )
    return total, tuple(recorded)


def compare_historical_observations_csv(
    transformed_table: pa.Table,
    historical: GlennHistoricalCSVInput,
) -> GlennLogicalEquivalence:
    """Compare after the declared CSV-float/int/bool/native-null normalization.

    This function does not construct a dataset lineage edge, issue a scientific
    verdict, or treat the retrospective historical artifact as a transform
    parent.  It only records deterministic operational equivalence.
    """

    transformed_sha256 = canonical_table_logical_sha256(transformed_table)
    payload = _read_historical_payload(historical)
    normalized = _normalize_historical_csv(payload, expected_rows=transformed_table.num_rows)
    historical_sha256 = canonical_table_logical_sha256(normalized)
    mismatch_count, recorded = _mismatches(transformed_table, normalized)
    equivalent = mismatch_count == 0 and transformed_sha256 == historical_sha256
    classification = (
        GlennEquivalenceClassification.LOGICALLY_EQUIVALENT
        if equivalent
        else GlennEquivalenceClassification.LOGICALLY_DIFFERENT
    )
    recorded = tuple(sorted(recorded, key=lambda value: value.mismatch_id))
    return GlennLogicalEquivalence(
        classification=classification,
        equivalent=equivalent,
        normalization_profile_id="glenn-historical-csv-type-null-normalization",
        transformed_logical_sha256=transformed_sha256,
        historical_normalized_logical_sha256=historical_sha256,
        historical_physical_sha256=historical.expected_sha256,
        historical_size_bytes=len(payload),
        historical_locator=historical.locator,
        historical_guard_evidence_id=historical.guard_evidence_id,
        row_count=transformed_table.num_rows,
        column_count=transformed_table.num_columns,
        mismatch_count=mismatch_count,
        recorded_mismatches=recorded,
        mismatches_truncated=mismatch_count > len(recorded),
    )


__all__ = [
    "GlennEquivalenceClassification",
    "GlennHistoricalCSVInput",
    "GlennLogicalEquivalence",
    "GlennMismatch",
    "compare_historical_observations_csv",
]
