"""Exact, non-executing Glenn member decode and canonical table transform."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from types import MappingProxyType
from typing import ClassVar, Final, Mapping

import numpy as np
from numpy.lib import format as npy_format
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import (
    ACTION_COLUMNS,
    CANONICAL_COLUMNS,
    COLUMN_ROLES,
    MODEL_COLUMNS,
    NUISANCE_COLUMNS,
    OBSERVATION_COLUMNS,
    GlennAdapterError,
    GlennArchiveInput,
    GlennColumnRole,
    GlennExceptionCode,
    GlennTransformProfile,
)
from .source import GlennArchiveAudit, inspect_glenn_archive


GLENN_OUTPUT_SCHEMA_ID = 'empirical-lawhood/physical/glenn/glenn-observations'
GLENN_LOGICAL_PROFILE_ID = 'empirical-lawhood/physical/glenn/glenn-logical-table-sha256'
GLENN_PARQUET_PROFILE_ID = "glenn-parquet-pyarrow"
GLENN_ARROW_PRIMARY_KEY: Final = ("run", "burst")
ACTION_LABEL_CAVEAT = (
    "Deposited action headers are retained verbatim: the source notebook says astig0 "
    "and astig45 were swapped during the experiment and spherical represented "
    "45-degree second-order astigmatism. No column is renamed, swapped, or sign-changed."
)
_GENERATION_FUNCTION_MARKER = "def get_pct_enclosed_radius_burst"
_MAXIMUM_NPY_HEADER_BYTES = 4096
_MAXIMUM_NOTEBOOK_JSON_DEPTH = 64


def _column_unit(name: str) -> str:
    if name in ACTION_COLUMNS:
        return "source-native-command"
    if name in {"focal_r50_um", "prior_focal_r50_um"}:
        return "um"
    if name == "acquisition_order":
        return "burst-index"
    if COLUMN_ROLES[name] in {
        GlennColumnRole.RUN_IDENTITY,
        GlennColumnRole.INDEPENDENT_UNIT,
        GlennColumnRole.HISTORY_MISSINGNESS,
    }:
        return "1"
    return "source-native"


def _column_frame(name: str) -> str:
    if name == "run":
        return "glenn-run-identity"
    if name == "burst":
        return "glenn-burst-identity"
    if COLUMN_ROLES[name] is GlennColumnRole.HISTORY_MISSINGNESS:
        return "glenn-validity-flag"
    return "glenn-source-frame"


def _column_clock(name: str) -> str | None:
    if name in {"run", "burst", "prior_fitness_missing", "prior_focal_r50_missing"}:
        return None
    return "glenn-burst-clock"


GLENN_ARROW_NATIVE_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {name: _column_unit(name) for name in CANONICAL_COLUMNS}
)
GLENN_ARROW_COORDINATE_FRAMES: Final[Mapping[str, str]] = MappingProxyType(
    {name: _column_frame(name) for name in CANONICAL_COLUMNS}
)
GLENN_ARROW_CLOCKS: Final[Mapping[str, str | None]] = MappingProxyType(
    {name: _column_clock(name) for name in CANONICAL_COLUMNS}
)
GLENN_ARROW_SCHEMA_METADATA: Final[tuple[tuple[str, str], ...]] = (
    ("empirical-lawhood-action-columns", ",".join(ACTION_COLUMNS)),
    ("empirical-lawhood-evidence-boundary", "retrospective-outcome-visible-nonpromotable"),
    ("empirical-lawhood-focal-status", "post-command-source-reconstruction"),
    ("empirical-lawhood-independent-unit", "burst"),
    ("empirical-lawhood-nested-shots-per-burst", "10"),
    ("empirical-lawhood-nuisance-columns", ",".join(NUISANCE_COLUMNS)),
    ("empirical-lawhood-output-schema", GLENN_OUTPUT_SCHEMA_ID),
)


@dataclass(frozen=True, slots=True)
class _ParsedLog:
    runs: tuple[str, ...]
    numeric: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class GlennParquetProfile(CanonicalRecord):
    """Fully declared deterministic writer settings and runtime identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-parquet-profile'

    profile_id: str
    writer_library: str
    writer_library_version: str
    parquet_version: str
    compression: str
    compression_level: int
    use_dictionary: bool
    write_statistics: bool
    data_page_version: str
    row_group_policy: str
    write_page_checksum: bool


@dataclass(frozen=True, slots=True)
class GlennParquetArtifact:
    """Deterministic bytes plus both physical and logical identities."""

    payload: bytes
    physical_sha256: str
    size_bytes: int
    logical_sha256: str
    row_count: int
    column_count: int
    media_type: str
    profile: GlennParquetProfile


@dataclass(frozen=True, slots=True)
class GlennTransformResult:
    """Operational transform output; it intentionally carries no scientific verdict."""

    table: pa.Table
    archive_audit: GlennArchiveAudit
    logical_sha256: str
    output_schema_id: str
    exception_codes: tuple[GlennExceptionCode, ...]
    action_columns: tuple[str, ...]
    nuisance_covariate_columns: tuple[str, ...]
    action_label_caveat: str
    focal_native_to_um: float
    focal_is_post_command_mediator: bool
    focal_generation_method_available: bool
    focal_uncertainty_available: bool
    deposited_model_log_role: str


def _decode_utf8(payload: bytes, *, label: str) -> str:
    if b"\x00" in payload:
        raise GlennAdapterError(f"{label} contains forbidden NUL bytes")
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GlennAdapterError(f"{label} is not exact UTF-8") from error


def _parse_log(
    payload: bytes,
    *,
    expected_columns: tuple[str, ...],
    profile: GlennTransformProfile,
    is_model_log: bool,
) -> _ParsedLog:
    label = "deposited model log" if is_model_log else "observation log"
    text = _decode_utf8(payload, label=label)
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or any(not line.strip() for line in lines):
        raise GlennAdapterError(f"{label} contains an empty interior row")
    if tuple(lines[0].split()) != expected_columns:
        raise GlennAdapterError(f"{label} columns differ from the exact deposited schema")
    data_lines = lines[1:]
    if len(data_lines) != profile.expected_rows:
        raise GlennAdapterError(f"{label} row count differs from the exact profile")

    runs: list[str] = []
    numeric_lists: dict[str, list[float]] = {
        column: [] for column in expected_columns if column != "run"
    }
    for row_index, line in enumerate(data_lines, start=1):
        tokens = line.split()
        if len(tokens) != len(expected_columns):
            raise GlennAdapterError(f"{label} row {row_index} has the wrong field count")
        if tokens[0] != profile.expected_run:
            raise GlennAdapterError(f"{label} run identity differs from the exact profile")
        runs.append(tokens[0])
        for column, token in zip(expected_columns[1:], tokens[1:], strict=True):
            try:
                value = float(token)
            except ValueError as error:
                raise GlennAdapterError(
                    f"{label} column {column!r} contains a nonnumeric token"
                ) from error
            if not math.isfinite(value):
                raise GlennAdapterError(f"{label} column {column!r} must be finite")
            numeric_lists[column].append(value)

    numeric: dict[str, np.ndarray] = {}
    for column, values in numeric_lists.items():
        array = np.asarray(values, dtype=np.float64)
        array.setflags(write=False)
        numeric[column] = array
    bursts = numeric["burst"]
    expected_bursts = np.arange(1, profile.expected_rows + 1, dtype=np.float64)
    if not np.array_equal(bursts, expected_bursts):
        raise GlennAdapterError(f"{label} bursts must be ordered exactly 1 through N")
    for column in NUISANCE_COLUMNS:
        if not np.equal(numeric[column], numeric[column][0]).all():
            raise GlennAdapterError(f"{label} nuisance covariate {column!r} must be constant")
    error_column = "error" if is_model_log else "fitness_error"
    if np.less(numeric[error_column], 0.0).any():
        raise GlennAdapterError(f"{label} uncertainty/error column cannot be negative")
    if not is_model_log:
        for column in ACTION_COLUMNS:
            if np.unique(numeric[column]).size <= 1:
                raise GlennAdapterError(f"executed action {column!r} must vary")
            if np.unique(numeric[column][: profile.initial_exploration_rows]).size <= 1:
                raise GlennAdapterError(f"initial exploration must vary executed action {column!r}")
        first_action = np.asarray([numeric[column][0] for column in ACTION_COLUMNS])
        if not np.allclose(first_action, 0.0, rtol=0.0, atol=1e-12):
            raise GlennAdapterError("burst 1 must retain the deposited zero command")
    return _ParsedLog(runs=tuple(runs), numeric=MappingProxyType(numeric))


def _validate_model_alignment(observations: _ParsedLog, model: _ParsedLog) -> None:
    if observations.runs != model.runs:
        raise GlennAdapterError("observation and deposited-model run keys do not align")
    if not np.array_equal(observations.numeric["burst"], model.numeric["burst"]):
        raise GlennAdapterError("observation and deposited-model burst keys do not align")
    for column in NUISANCE_COLUMNS:
        if not np.array_equal(observations.numeric[column], model.numeric[column]):
            raise GlennAdapterError(
                f"observation and deposited-model nuisance field {column!r} disagree"
            )


def _parse_focal_radius(payload: bytes, *, profile: GlennTransformProfile) -> np.ndarray:
    header_stream = BytesIO(payload)
    try:
        version = npy_format.read_magic(header_stream)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(
                header_stream,
                max_header_size=_MAXIMUM_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(
                header_stream,
                max_header_size=_MAXIMUM_NPY_HEADER_BYTES,
            )
        else:
            raise GlennAdapterError("focal-radius NPY version is not allowlisted")
    except (OSError, ValueError, TypeError) as error:
        raise GlennAdapterError("focal-radius NPY header is not safe bounded metadata") from error
    if dtype.hasobject or dtype.fields is not None:
        raise GlennAdapterError("object or structured focal-radius arrays are forbidden")
    if (
        shape != (profile.expected_rows,)
        or dtype.kind not in {"f", "i", "u"}
        or dtype.itemsize > 8
        or fortran_order
    ):
        raise GlennAdapterError(
            "focal-radius NPY header must declare one C-order bounded numeric vector"
        )
    expected_payload_bytes = profile.expected_rows * dtype.itemsize
    if len(payload) - header_stream.tell() != expected_payload_bytes:
        raise GlennAdapterError("focal-radius NPY payload size disagrees with its safe header")
    try:
        value = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise GlennAdapterError("focal-radius NPY is not safe non-pickle numeric data") from error
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise GlennAdapterError("focal-radius NPY decode disagrees with its preflight header")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or not np.greater(result, 0.0).all():
        raise GlennAdapterError("focal-radius values must be finite and positive")
    result = result * profile.focal_native_to_um
    result.setflags(write=False)
    return result


def _reject_json_constant(value: str) -> object:
    raise GlennAdapterError(f"notebook JSON contains a nonstandard constant: {value}")


def _validate_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAXIMUM_NOTEBOOK_JSON_DEPTH:
                raise GlennAdapterError("figure notebook exceeds its JSON nesting ceiling")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise GlennAdapterError("figure notebook JSON delimiters are unbalanced")
    if in_string or depth != 0:
        raise GlennAdapterError("figure notebook JSON delimiters are unbalanced")


def _parse_notebook(payload: bytes, *, profile: GlennTransformProfile) -> str:
    text = _decode_utf8(payload, label="figure notebook")
    _validate_json_nesting(text)
    try:
        decoded = json.loads(text, parse_constant=_reject_json_constant)
    except GlennAdapterError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise GlennAdapterError("figure notebook is not bounded strict JSON") from error
    if not isinstance(decoded, dict):
        raise GlennAdapterError("figure notebook root must be an object")
    cells = decoded.get("cells")
    if not isinstance(cells, list) or len(cells) > 4096:
        raise GlennAdapterError("figure notebook must contain a bounded cell list")
    pieces: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise GlennAdapterError("figure notebook cell must be an object")
        source = cell.get("source", [])
        if isinstance(source, str):
            pieces.append(source)
        elif isinstance(source, list) and all(isinstance(item, str) for item in source):
            pieces.extend(source)
        else:
            raise GlennAdapterError("figure notebook cell source must be text only")
    joined = "".join(pieces)
    missing = [marker for marker in profile.required_notebook_markers if marker not in joined]
    if missing:
        raise GlennAdapterError(f"figure notebook lacks exact method markers: {missing}")
    if _GENERATION_FUNCTION_MARKER in joined:
        raise GlennAdapterError(
            "figure notebook now contains a generation function; exception metadata must be revised"
        )
    return joined


def _field_metadata(name: str) -> dict[bytes, bytes]:
    metadata = {
        b"empirical-lawhood-column-role": COLUMN_ROLES[name].value.encode("ascii"),
    }
    if name in ACTION_COLUMNS:
        metadata[b"empirical-lawhood-action-label-policy"] = b"deposited-verbatim"
        metadata[b"empirical-lawhood-units"] = b"source-native-command"
    elif name in NUISANCE_COLUMNS:
        metadata[b"empirical-lawhood-covariate-policy"] = b"nuisance-not-action"
        metadata[b"empirical-lawhood-units"] = b"source-native"
    elif name in {"focal_r50_um", "prior_focal_r50_um"}:
        metadata[b"empirical-lawhood-units"] = b"um"
    return metadata


def glenn_arrow_field_metadata(name: str) -> tuple[tuple[str, str], ...]:
    """Return the exact adapter-owned field metadata for structural admission."""

    if name not in CANONICAL_COLUMNS:
        raise KeyError(name)
    return tuple(
        sorted(
            (key.decode("ascii"), value.decode("ascii"))
            for key, value in _field_metadata(name).items()
        )
    )


def _standard_table_metadata() -> dict[bytes, bytes]:
    def encoded(value: object) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    return {
        b"empirical_lawhood_payload_schema": GLENN_OUTPUT_SCHEMA_ID.encode("ascii"),
        b"empirical_lawhood_logical_types": encoded(
            {name: COLUMN_ROLES[name].value for name in CANONICAL_COLUMNS}
        ),
        b"empirical_lawhood_units": encoded(dict(GLENN_ARROW_NATIVE_UNITS)),
        b"empirical_lawhood_frames": encoded(dict(GLENN_ARROW_COORDINATE_FRAMES)),
        b"empirical_lawhood_clocks": encoded(
            {name: clock for name, clock in GLENN_ARROW_CLOCKS.items() if clock is not None}
        ),
        b"empirical_lawhood_keys": encoded(list(GLENN_ARROW_PRIMARY_KEY)),
    }


def glenn_arrow_schema() -> pa.Schema:
    """Return the exact ordered Arrow schema, including action/nuisance roles."""

    string_fields = {"run"}
    integer_fields = {"burst", "acquisition_order"}
    boolean_fields = {"prior_fitness_missing", "prior_focal_r50_missing"}
    nullable_fields = {"prior_fitness", "prior_focal_r50_um"}
    fields: list[pa.Field] = []
    for name in CANONICAL_COLUMNS:
        data_type = (
            pa.string()
            if name in string_fields
            else pa.int64()
            if name in integer_fields
            else pa.bool_()
            if name in boolean_fields
            else pa.float64()
        )
        fields.append(
            pa.field(
                name,
                data_type,
                nullable=name in nullable_fields,
                metadata=_field_metadata(name),
            )
        )
    metadata = _standard_table_metadata()
    metadata.update(
        {key.encode("ascii"): value.encode("ascii") for key, value in GLENN_ARROW_SCHEMA_METADATA}
    )
    return pa.schema(fields, metadata=metadata)


def _build_table(
    observations: _ParsedLog,
    focal_r50_um: np.ndarray,
    *,
    expected_rows: int,
) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "run": pa.array(observations.runs, type=pa.string()),
        "burst": pa.array(observations.numeric["burst"].astype(np.int64), type=pa.int64()),
    }
    for name in OBSERVATION_COLUMNS[2:]:
        columns[name] = pa.array(observations.numeric[name], type=pa.float64())
    columns["focal_r50_um"] = pa.array(focal_r50_um, type=pa.float64())
    columns["acquisition_order"] = pa.array(
        np.arange(1, expected_rows + 1, dtype=np.int64), type=pa.int64()
    )
    columns["prior_fitness"] = pa.array(
        [None, *observations.numeric["fitness"][:-1].tolist()], type=pa.float64()
    )
    columns["prior_fitness_missing"] = pa.array(
        [True, *([False] * (expected_rows - 1))], type=pa.bool_()
    )
    columns["prior_focal_r50_um"] = pa.array([None, *focal_r50_um[:-1].tolist()], type=pa.float64())
    columns["prior_focal_r50_missing"] = pa.array(
        [True, *([False] * (expected_rows - 1))], type=pa.bool_()
    )
    table = pa.Table.from_arrays(
        [columns[name] for name in CANONICAL_COLUMNS],
        schema=glenn_arrow_schema(),
    )
    _validate_glenn_table(table, expected_rows=expected_rows)
    return table


def _validate_glenn_table(table: pa.Table, *, expected_rows: int | None = None) -> None:
    try:
        table.validate(full=True)
    except pa.ArrowInvalid as error:
        raise GlennAdapterError("canonical Arrow table is structurally invalid") from error
    if not table.schema.equals(glenn_arrow_schema(), check_metadata=True):
        raise GlennAdapterError("table differs from the exact Glenn Arrow schema")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise GlennAdapterError("table row count differs from the transform profile")
    if table.num_rows <= 0:
        raise GlennAdapterError("canonical Glenn table cannot be empty")
    nullable = {"prior_fitness", "prior_focal_r50_um"}
    for name in CANONICAL_COLUMNS:
        expected_nulls = 1 if name in nullable else 0
        if table[name].null_count != expected_nulls:
            raise GlennAdapterError(f"column {name!r} violates its exact null contract")
    prior_fitness = table["prior_fitness"].to_pylist()
    prior_focal = table["prior_focal_r50_um"].to_pylist()
    fitness_missing = table["prior_fitness_missing"].to_pylist()
    focal_missing = table["prior_focal_r50_missing"].to_pylist()
    if fitness_missing != [value is None for value in prior_fitness]:
        raise GlennAdapterError("prior-fitness missing flags disagree with native nulls")
    if focal_missing != [value is None for value in prior_focal]:
        raise GlennAdapterError("prior-focal missing flags disagree with native nulls")
    expected_order = list(range(1, table.num_rows + 1))
    if table["burst"].to_pylist() != expected_order:
        raise GlennAdapterError("burst order is not exactly 1 through N")
    if table["acquisition_order"].to_pylist() != expected_order:
        raise GlennAdapterError("acquisition order is not exactly 1 through N")
    for name in CANONICAL_COLUMNS:
        if pa.types.is_floating(table.schema.field(name).type):
            if any(
                value is not None and not math.isfinite(value) for value in table[name].to_pylist()
            ):
                raise GlennAdapterError(f"column {name!r} contains a non-finite value")


def _logical_value(value: object, *, role: GlennColumnRole) -> object:
    del role  # role is included in the surrounding field contract
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GlennAdapterError("logical table digest refuses non-finite floats")
        return {"float64_hex": value.hex()}
    raise GlennAdapterError("logical table contains an unsupported scalar type")


def canonical_table_logical_sha256(table: pa.Table) -> str:
    """Hash exact typed values independently of Parquet writer bytes."""

    _validate_glenn_table(table)
    fields = []
    for name in CANONICAL_COLUMNS:
        field = table.schema.field(name)
        role = COLUMN_ROLES[name]
        fields.append(
            {
                "data_type": str(field.type),
                "name": name,
                "nullable": field.nullable,
                "role": role.value,
                "values": [_logical_value(value, role=role) for value in table[name].to_pylist()],
            }
        )
    payload = json.dumps(
        {
            "fields": fields,
            "logical_profile_id": GLENN_LOGICAL_PROFILE_ID,
            "output_schema_id": GLENN_OUTPUT_SCHEMA_ID,
            "row_count": table.num_rows,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def transform_glenn_archive(
    source: GlennArchiveInput,
    *,
    profile: GlennTransformProfile,
) -> GlennTransformResult:
    """Produce the deterministic 20-column table from the safe four-member read set."""

    decoded = inspect_glenn_archive(source, profile=profile)
    observations = _parse_log(
        decoded.observation_log,
        expected_columns=OBSERVATION_COLUMNS,
        profile=profile,
        is_model_log=False,
    )
    deposited_model = _parse_log(
        decoded.model_log,
        expected_columns=MODEL_COLUMNS,
        profile=profile,
        is_model_log=True,
    )
    _validate_model_alignment(observations, deposited_model)
    focal_r50_um = _parse_focal_radius(decoded.focal_radius_npy, profile=profile)
    _parse_notebook(decoded.figure_notebook_json, profile=profile)
    table = _build_table(observations, focal_r50_um, expected_rows=profile.expected_rows)
    logical_sha256 = canonical_table_logical_sha256(table)
    return GlennTransformResult(
        table=table,
        archive_audit=decoded.audit,
        logical_sha256=logical_sha256,
        output_schema_id=GLENN_OUTPUT_SCHEMA_ID,
        exception_codes=(
            GlennExceptionCode.FOCAL_R50_GENERATION_METHOD_UNAVAILABLE,
            GlennExceptionCode.FOCAL_R50_UNCERTAINTY_UNAVAILABLE,
            GlennExceptionCode.SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT,
        ),
        action_columns=ACTION_COLUMNS,
        nuisance_covariate_columns=NUISANCE_COLUMNS,
        action_label_caveat=ACTION_LABEL_CAVEAT,
        focal_native_to_um=profile.focal_native_to_um,
        focal_is_post_command_mediator=True,
        focal_generation_method_available=False,
        focal_uncertainty_available=False,
        deposited_model_log_role="SOURCE_VALIDATION_ONLY_NOT_PREBURST_PREDICTION",
    )


def serialize_glenn_parquet(table: pa.Table) -> GlennParquetArtifact:
    """Serialize with one explicit deterministic same-runtime Parquet profile."""

    _validate_glenn_table(table)
    sink = BytesIO()
    profile = GlennParquetProfile(
        profile_id=GLENN_PARQUET_PROFILE_ID,
        writer_library="pyarrow",
        writer_library_version=pa.__version__,
        parquet_version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        row_group_policy="exactly-one-row-group",
        write_page_checksum=True,
    )
    try:
        pq.write_table(
            table,
            sink,
            version=profile.parquet_version,
            compression=profile.compression,
            compression_level=profile.compression_level,
            use_dictionary=profile.use_dictionary,
            write_statistics=profile.write_statistics,
            data_page_version=profile.data_page_version,
            row_group_size=table.num_rows,
            use_compliant_nested_type=True,
            store_schema=True,
            write_page_index=False,
            write_page_checksum=profile.write_page_checksum,
        )
    except (pa.ArrowException, ValueError) as error:
        raise GlennAdapterError("canonical Glenn table cannot be serialized as Parquet") from error
    payload = sink.getvalue()
    try:
        replay = pq.read_table(BytesIO(payload))
    except pa.ArrowException as error:
        raise GlennAdapterError("serialized Glenn Parquet failed immediate replay") from error
    if not replay.equals(table, check_metadata=True):
        raise GlennAdapterError("serialized Glenn Parquet changed canonical table semantics")
    return GlennParquetArtifact(
        payload=payload,
        physical_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        logical_sha256=canonical_table_logical_sha256(table),
        row_count=table.num_rows,
        column_count=table.num_columns,
        media_type="application/vnd.apache.parquet",
        profile=profile,
    )


__all__ = [
    "GLENN_PARQUET_PROFILE_ID",
    "ACTION_LABEL_CAVEAT",
    "GLENN_ARROW_CLOCKS",
    "GLENN_ARROW_COORDINATE_FRAMES",
    "GLENN_ARROW_NATIVE_UNITS",
    "GLENN_ARROW_PRIMARY_KEY",
    "GLENN_ARROW_SCHEMA_METADATA",
    "GLENN_LOGICAL_PROFILE_ID",
    "GLENN_OUTPUT_SCHEMA_ID",
    "GlennParquetArtifact",
    "GlennParquetProfile",
    "GlennTransformResult",
    "canonical_table_logical_sha256",
    "glenn_arrow_field_metadata",
    "glenn_arrow_schema",
    "serialize_glenn_parquet",
    "transform_glenn_archive",
]
