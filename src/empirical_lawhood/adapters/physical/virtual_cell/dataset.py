"""Bounded, path-free source handling for Virtual Cell Challenge artifacts.

The scientific source is supplied by runtime input ports.  This module only
accepts already-open streams, so source selection, custody, mount containment
and authorization remain responsibilities of the existing dataset/runtime
layers.  In particular, ordinary inspection refuses the real test H5AD.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
from typing import BinaryIO, ClassVar, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .contracts import (
    MatrixEncoding,
    SplitCardinality,
    VirtualCellContractError,
    VirtualCellSourceManifest,
    VirtualCellSourceObject,
    VirtualCellSourcePart,
    VirtualCellSplit,
)


_COPY_CHUNK_BYTES = 8 * 1024**2
_MAX_CUSTODY_RECEIPT_BYTES = 1024**2

_SOURCE_OBJECT_IDENTITIES = {
    "virtual-cell-challenge/2025/gene_names.csv": (
        "source.virtual-cell-2025-gene-names",
        VirtualCellSplit.SHARED,
        "feature-registry",
        "text/csv",
    ),
    "virtual-cell-challenge/2025/test/adata_Test.h5ad": (
        "source.virtual-cell-2025-test-h5ad",
        VirtualCellSplit.TEST,
        "response/evaluation-sealed",
        "application/x-hdf5",
    ),
    "virtual-cell-challenge/2025/test/pert_counts_Test.csv": (
        "source.virtual-cell-2025-test-pert-counts",
        VirtualCellSplit.TEST,
        "prefix/target-roster",
        "text/csv",
    ),
    "virtual-cell-challenge/2025/train/adata_Training.h5ad": (
        "source.virtual-cell-2025-train-h5ad",
        VirtualCellSplit.TRAIN,
        "response/development",
        "application/x-hdf5",
    ),
    "virtual-cell-challenge/2025/train/pert_counts_Training.csv": (
        "source.virtual-cell-2025-train-pert-counts",
        VirtualCellSplit.TRAIN,
        "prefix/target-roster",
        "text/csv",
    ),
    "virtual-cell-challenge/2025/validation/adata_Validation.h5ad": (
        "source.virtual-cell-2025-validation-h5ad",
        VirtualCellSplit.VALIDATION,
        "response/development",
        "application/x-hdf5",
    ),
    "virtual-cell-challenge/2025/validation/pert_counts_Validation.csv": (
        "source.virtual-cell-2025-validation-pert-counts",
        VirtualCellSplit.VALIDATION,
        "prefix/target-roster",
        "text/csv",
    ),
}


def _require_positive_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _decode_text(value: object, *, field_name: str) -> str:
    if isinstance(value, bytes):
        try:
            result = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise VirtualCellContractError(f"{field_name} is not UTF-8") from error
    elif isinstance(value, str):
        result = value
    else:
        result = str(value)
    if not result:
        raise VirtualCellContractError(f"{field_name} is empty")
    return result


class SegmentedBinaryReader(io.RawIOBase):
    """Expose ordered immutable source parts as one seekable binary stream.

    HDF5 performs random reads, while runtime input ports are intentionally
    sequential.  A runner may first spool each verified port into guarded
    external scratch and then pass those already-open files here.  No source
    pathname is accepted or retained by this class.
    """

    def __init__(self, parts: Sequence[BinaryIO], sizes: Sequence[int]) -> None:
        super().__init__()
        if not parts or len(parts) != len(sizes):
            raise ValueError("parts and sizes must be nonempty and have equal length")
        normalized_sizes: list[int] = []
        normalized_parts: list[BinaryIO] = []
        offsets = [0]
        for index, (part, size) in enumerate(zip(parts, sizes, strict=True)):
            _require_positive_integer(size, field_name=f"sizes[{index}]")
            try:
                if not part.readable() or not part.seekable():
                    raise ValueError("each source part must be readable and seekable")
                initial = part.tell()
                part.seek(0, io.SEEK_END)
                observed = part.tell()
                part.seek(initial, io.SEEK_SET)
            except (AttributeError, OSError, ValueError) as error:
                raise ValueError(f"source part {index} cannot be positioned") from error
            if observed != size:
                raise ValueError(
                    f"source part {index} byte count differs: expected {size}, got {observed}"
                )
            normalized_parts.append(part)
            normalized_sizes.append(size)
            offsets.append(offsets[-1] + size)
        self._parts = tuple(normalized_parts)
        self._sizes = tuple(normalized_sizes)
        self._offsets = tuple(offsets)
        self._position = 0

    @property
    def size(self) -> int:
        return self._offsets[-1]

    def readable(self) -> bool:
        return not self.closed

    def seekable(self) -> bool:
        return not self.closed

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("offset must be an integer")
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def readinto(self, buffer: object) -> int:
        self._checkClosed()
        view = memoryview(buffer).cast("B")  # type: ignore[arg-type]
        if not view:
            return 0
        if self._position >= self.size:
            return 0
        requested = min(len(view), self.size - self._position)
        copied = 0
        while copied < requested:
            part_index = bisect_right(self._offsets, self._position) - 1
            if part_index >= len(self._parts):
                break
            local_offset = self._position - self._offsets[part_index]
            available = self._sizes[part_index] - local_offset
            take = min(requested - copied, available)
            part = self._parts[part_index]
            try:
                part.seek(local_offset)
                payload = part.read(take)
            except (OSError, ValueError) as error:
                raise VirtualCellContractError(
                    f"source part {part_index} failed during segmented read"
                ) from error
            if not isinstance(payload, bytes) or len(payload) != take:
                raise VirtualCellContractError(
                    f"source part {part_index} changed or truncated during segmented read"
                )
            view[copied : copied + take] = payload
            copied += take
            self._position += take
        return copied

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if size is None or size < 0:
            size = max(0, self.size - self._position)
        if size == 0:
            return b""
        payload = bytearray(size)
        count = self.readinto(payload)
        return bytes(payload[:count])

    def close(self) -> None:
        """Close every owned part handle exactly once."""

        if self.closed:
            return
        first_error: OSError | None = None
        for part in self._parts:
            try:
                part.close()
            except OSError as error:
                if first_error is None:
                    first_error = error
        super().close()
        if first_error is not None:
            raise first_error


@dataclass(frozen=True, slots=True)
class StreamIdentity(CanonicalRecord):
    """Identity computed while copying one sequential runtime input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/stream-identity'

    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_positive_integer(self.byte_count, field_name="byte_count")
        validate_sha256(self.sha256, field_name="sha256")


def copy_and_hash_stream(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    expected_size: int,
    expected_sha256: str,
    chunk_bytes: int = _COPY_CHUNK_BYTES,
) -> StreamIdentity:
    """Copy a sequential source once and prove exact byte/hash closure."""

    _require_positive_integer(expected_size, field_name="expected_size")
    validate_sha256(expected_sha256, field_name="expected_sha256")
    _require_positive_integer(chunk_bytes, field_name="chunk_bytes")
    digest = hashlib.sha256()
    observed = 0
    while observed < expected_size:
        payload = source.read(min(chunk_bytes, expected_size - observed))
        if not isinstance(payload, bytes) or not payload:
            raise VirtualCellContractError("source stream ended before its declared byte count")
        destination.write(payload)
        digest.update(payload)
        observed += len(payload)
    if source.read(1) != b"":
        raise VirtualCellContractError("source stream exceeds its declared byte count")
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise VirtualCellContractError("source stream SHA-256 differs from its manifest")
    return StreamIdentity(byte_count=observed, sha256=observed_sha256)


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VirtualCellContractError(f"{field_name} must be a string-keyed object")
    return cast(dict[str, object], value)


def _string(mapping: dict[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise VirtualCellContractError(f"custody receipt {field} must be a nonempty string")
    return value


def _integer(mapping: dict[str, object], field: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VirtualCellContractError(f"custody receipt {field} must be a positive integer")
    return value


def decode_custody_receipt(
    payload: bytes,
    *,
    expected_sha256: str,
    storage_root_id: str,
    license_locator: str,
    gene_order_sha256: str,
) -> VirtualCellSourceManifest:
    """Decode the exact seven-object external receipt into a strict manifest."""

    validate_sha256(expected_sha256, field_name="expected_sha256")
    validate_stable_id(storage_root_id, field_name="storage_root_id")
    validate_nonempty(license_locator, field_name="license_locator")
    validate_sha256(gene_order_sha256, field_name="gene_order_sha256")
    if not payload or len(payload) > _MAX_CUSTODY_RECEIPT_BYTES:
        raise VirtualCellContractError("custody receipt is empty or exceeds its byte bound")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise VirtualCellContractError("custody receipt SHA-256 differs from the frozen config")
    try:
        document = _mapping(json.loads(payload), field_name="custody receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VirtualCellContractError("custody receipt is not canonicalizable JSON") from error
    if document.get("schema") != "icf-yolo/vcc-2025-processed-source-custody-receipt/v1":
        raise VirtualCellContractError("custody receipt has the wrong schema")
    source_values = document.get("source_objects")
    if not isinstance(source_values, list) or len(source_values) != len(_SOURCE_OBJECT_IDENTITIES):
        raise VirtualCellContractError("custody receipt lacks the exact seven-object release")
    objects: list[VirtualCellSourceObject] = []
    observed_names: set[str] = set()
    for source_index, raw_source in enumerate(source_values):
        source = _mapping(raw_source, field_name=f"source_objects[{source_index}]")
        source_name = _string(source, "source_name")
        if source_name in observed_names or source_name not in _SOURCE_OBJECT_IDENTITIES:
            raise VirtualCellContractError("custody receipt source roster is duplicate or unknown")
        observed_names.add(source_name)
        object_id, split, role, media_type = _SOURCE_OBJECT_IDENTITIES[source_name]
        if source.get("verification") != "PASS":
            raise VirtualCellContractError(f"source object {source_name} is not byte-qualified")
        sealed = source.get("sealed")
        if not isinstance(sealed, bool):
            raise VirtualCellContractError("custody receipt sealed flag must be boolean")
        expected_sealed = source_name.endswith("test/adata_Test.h5ad")
        if sealed is not expected_sealed:
            raise VirtualCellContractError("custody receipt test sealing differs from source role")
        raw_parts = source.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise VirtualCellContractError("custody receipt source object has no parts")
        parts: list[VirtualCellSourcePart] = []
        for part_index, raw_part in enumerate(raw_parts):
            part = _mapping(raw_part, field_name=f"parts[{part_index}]")
            parts.append(
                VirtualCellSourcePart(
                    part_id=f"{object_id}.part-{part_index:04d}",
                    relative_locator=_string(part, "relative_path"),
                    size_bytes=_integer(part, "size_bytes"),
                    sha256=_string(part, "sha256"),
                )
            )
        objects.append(
            VirtualCellSourceObject(
                object_id=object_id,
                split=split,
                role=role,
                source_name=source_name,
                generation=_string(source, "generation"),
                size_bytes=_integer(source, "size"),
                sha256=_string(source, "local_sha256"),
                crc32c_base64=_string(source, "verified_crc32c_base64"),
                media_type=media_type,
                parts=tuple(parts),
                sealed=sealed,
            )
        )
    if observed_names != set(_SOURCE_OBJECT_IDENTITIES):
        raise VirtualCellContractError("custody receipt is missing an exact source object")
    return VirtualCellSourceManifest(
        manifest_id="source-manifest.virtual-cell-2025-processed",
        release_id="virtual-cell-2025-gcs-release-2025-12-16",
        upstream_prefix="gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/",
        storage_root_id=storage_root_id,
        license_locator=license_locator,
        custody_receipt_artifact_id="artifact.virtual-cell-2025-custody-receipt",
        custody_receipt_sha256=expected_sha256,
        gene_count=18_080,
        gene_order_sha256=gene_order_sha256,
        cardinalities=(
            SplitCardinality(
                split=VirtualCellSplit.TEST,
                target_count=100,
                reported_perturbed_cell_count=132_670,
            ),
            SplitCardinality(
                split=VirtualCellSplit.TRAIN,
                target_count=150,
                reported_perturbed_cell_count=183_097,
            ),
            SplitCardinality(
                split=VirtualCellSplit.VALIDATION,
                target_count=50,
                reported_perturbed_cell_count=60_751,
            ),
        ),
        objects=tuple(sorted(objects, key=lambda value: value.object_id)),
        expected_object_count=7,
        complete_processed_release=True,
    )


@dataclass(frozen=True, slots=True)
class H5ADStructuralInspection(CanonicalRecord):
    """Bounded structural facts; never a substitute for outcome access."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/virtual-cell/h5-ad-structural-inspection'

    inspection_id: str
    split: VirtualCellSplit
    row_count: int
    gene_count: int
    matrix_encoding: MatrixEncoding
    matrix_data_dtype: str
    matrix_indices_dtype: str | None
    obs_fields: tuple[str, ...]
    var_index_field: str
    layer_ids: tuple[str, ...]
    gene_order_sha256: str | None
    value_datasets_opened: tuple[str, ...]
    structure_only: bool
    synthetic_test_fixture: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        _require_positive_integer(self.row_count, field_name="row_count")
        _require_positive_integer(self.gene_count, field_name="gene_count")
        validate_nonempty(self.matrix_data_dtype, field_name="matrix_data_dtype")
        if self.matrix_indices_dtype is not None:
            validate_nonempty(self.matrix_indices_dtype, field_name="matrix_indices_dtype")
        require_sorted_unique_strings(self.obs_fields, field_name="obs_fields")
        validate_nonempty(self.var_index_field, field_name="var_index_field")
        require_sorted_unique_strings(self.layer_ids, field_name="layer_ids")
        require_sorted_unique_strings(
            self.value_datasets_opened,
            field_name="value_datasets_opened",
        )
        if self.gene_order_sha256 is not None:
            validate_sha256(self.gene_order_sha256, field_name="gene_order_sha256")
        if self.split is VirtualCellSplit.TEST:
            if not self.structure_only or not self.synthetic_test_fixture:
                raise ValueError("only an explicit synthetic test fixture may be inspected")
            if self.value_datasets_opened:
                raise ValueError("test structural inspection cannot open value datasets")


@dataclass(frozen=True, slots=True)
class CSVStructuralInspection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/csv-structural-inspection'

    inspection_id: str
    byte_count: int
    sha256: str
    header: tuple[str, ...]
    row_count: int
    maximum_line_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        _require_positive_integer(self.byte_count, field_name="byte_count")
        validate_sha256(self.sha256, field_name="sha256")
        if not self.header or len(set(self.header)) != len(self.header):
            raise ValueError("CSV header must be nonempty and unique")
        for value in self.header:
            validate_nonempty(value, field_name="header")
        if isinstance(self.row_count, bool) or self.row_count < 0:
            raise ValueError("row_count must be a nonnegative integer")
        _require_positive_integer(self.maximum_line_bytes, field_name="maximum_line_bytes")


def inspect_csv(
    stream: BinaryIO,
    *,
    inspection_id: str,
    expected_header: tuple[str, ...],
    maximum_bytes: int,
    maximum_line_bytes: int = 1024**2,
) -> CSVStructuralInspection:
    """Inspect a simple UTF-8 CSV without accepting dialect ambiguity."""

    _require_positive_integer(maximum_bytes, field_name="maximum_bytes")
    _require_positive_integer(maximum_line_bytes, field_name="maximum_line_bytes")
    digest = hashlib.sha256()
    byte_count = 0
    row_count = -1
    observed_header: tuple[str, ...] | None = None
    for raw_line in stream:
        if not isinstance(raw_line, bytes):
            raise VirtualCellContractError("CSV input must yield bytes")
        if len(raw_line) > maximum_line_bytes:
            raise VirtualCellContractError("CSV line exceeds its bounded limit")
        byte_count += len(raw_line)
        if byte_count > maximum_bytes:
            raise VirtualCellContractError("CSV input exceeds its bounded byte limit")
        digest.update(raw_line)
        if row_count == -1:
            try:
                header_text = raw_line.decode("utf-8-sig").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise VirtualCellContractError("CSV header is not UTF-8") from error
            observed_header = tuple(header_text.split(","))
            if observed_header != expected_header:
                raise VirtualCellContractError("CSV header differs from its frozen contract")
            row_count = 0
        else:
            row_count += 1
    if observed_header is None or byte_count == 0:
        raise VirtualCellContractError("CSV input is empty")
    return CSVStructuralInspection(
        inspection_id=inspection_id,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
        header=observed_header,
        row_count=row_count,
        maximum_line_bytes=maximum_line_bytes,
    )


def _encoding(group: h5py.Group | h5py.Dataset) -> str:
    return _decode_text(group.attrs.get("encoding-type", ""), field_name="encoding-type")


def _matrix_shape(matrix: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    if isinstance(matrix, h5py.Dataset):
        if matrix.ndim != 2:
            raise VirtualCellContractError("dense X must be two-dimensional")
        return int(matrix.shape[0]), int(matrix.shape[1])
    shape = matrix.attrs.get("shape")
    if shape is None or len(shape) != 2:
        raise VirtualCellContractError("sparse X lacks its two-dimensional shape")
    return int(shape[0]), int(shape[1])


def _frame_columns(group: h5py.Group) -> tuple[str, ...]:
    raw = group.attrs.get("column-order")
    if raw is None:
        return tuple(sorted(key for key in group.keys() if key != "_index"))
    return tuple(sorted(_decode_text(value, field_name="column-order") for value in raw))


def _frame_index_field(group: h5py.Group) -> str:
    return _decode_text(group.attrs.get("_index", "_index"), field_name="_index")


def _hash_string_dataset(dataset: h5py.Dataset, *, maximum_values: int) -> str:
    if dataset.ndim != 1 or dataset.shape[0] > maximum_values:
        raise VirtualCellContractError("string registry exceeds its bounded cardinality")
    digest = hashlib.sha256()
    for value in dataset.asstr()[:]:
        encoded = _decode_text(value, field_name="registry value").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def inspect_h5ad(
    stream: BinaryIO,
    *,
    inspection_id: str,
    split: VirtualCellSplit,
    maximum_rows: int,
    maximum_genes: int,
    expected_obs_fields: tuple[str, ...],
    synthetic_test_fixture: bool = False,
) -> H5ADStructuralInspection:
    """Read bounded H5AD structure and, for development only, gene ordering.

    Real test bytes are deliberately rejected.  Synthetic test fixtures may be
    used to prove sealed topology, but even then this function opens no value
    datasets and does not derive a gene-order digest.
    """

    _require_positive_integer(maximum_rows, field_name="maximum_rows")
    _require_positive_integer(maximum_genes, field_name="maximum_genes")
    if split is VirtualCellSplit.TEST and not synthetic_test_fixture:
        raise VirtualCellContractError("real test H5AD inspection is evaluator-sealed")
    if split is VirtualCellSplit.SHARED:
        raise ValueError("a shared registry is not an H5AD split")
    try:
        with h5py.File(stream, "r") as handle:
            if "X" not in handle or "obs" not in handle or "var" not in handle:
                raise VirtualCellContractError("H5AD lacks X, obs or var")
            matrix = handle["X"]
            if not isinstance(matrix, (h5py.Group, h5py.Dataset)):
                raise VirtualCellContractError("H5AD X has an unsupported object kind")
            rows, genes = _matrix_shape(matrix)
            if rows <= 0 or rows > maximum_rows or genes <= 0 or genes > maximum_genes:
                raise VirtualCellContractError("H5AD shape exceeds its frozen bounds")
            encoding_name = _encoding(matrix)
            if isinstance(matrix, h5py.Dataset):
                matrix_encoding = MatrixEncoding.DENSE
                data_dtype = str(matrix.dtype)
                indices_dtype = None
            else:
                encoding_map = {
                    "csr_matrix": MatrixEncoding.CSR,
                    "csc_matrix": MatrixEncoding.CSC,
                }
                try:
                    matrix_encoding = encoding_map[encoding_name]
                except KeyError as error:
                    raise VirtualCellContractError(
                        f"unsupported sparse X encoding {encoding_name!r}"
                    ) from error
                required = {"data", "indices", "indptr"}
                if not required.issubset(matrix.keys()):
                    raise VirtualCellContractError("sparse X lacks data/indices/indptr")
                data_dtype = str(matrix["data"].dtype)
                indices_dtype = str(matrix["indices"].dtype)
                expected_indptr = rows + 1 if matrix_encoding is MatrixEncoding.CSR else genes + 1
                if matrix["indptr"].shape != (expected_indptr,):
                    raise VirtualCellContractError("sparse X indptr shape is inconsistent")
                if matrix["data"].shape != matrix["indices"].shape:
                    raise VirtualCellContractError("sparse X data/indices lengths differ")
            obs = handle["obs"]
            var = handle["var"]
            if not isinstance(obs, h5py.Group) or not isinstance(var, h5py.Group):
                raise VirtualCellContractError("H5AD obs/var must be dataframe groups")
            obs_fields = _frame_columns(obs)
            missing = set(expected_obs_fields) - set(obs_fields)
            if missing:
                raise VirtualCellContractError(
                    f"H5AD obs lacks frozen fields: {', '.join(sorted(missing))}"
                )
            var_index_field = _frame_index_field(var)
            if var_index_field not in var:
                raise VirtualCellContractError("H5AD var index dataset is absent")
            if var[var_index_field].shape != (genes,):
                raise VirtualCellContractError("H5AD var index length differs from X")
            layer_ids = tuple(sorted(handle.get("layers", {}).keys()))
            structure_only = split is VirtualCellSplit.TEST
            gene_order_sha256 = None
            value_datasets_opened: tuple[str, ...] = ()
            if not structure_only:
                gene_order_sha256 = _hash_string_dataset(
                    var[var_index_field],
                    maximum_values=maximum_genes,
                )
                value_datasets_opened = (f"var/{var_index_field}",)
    except VirtualCellContractError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise VirtualCellContractError("H5AD structural inspection failed closed") from error
    return H5ADStructuralInspection(
        inspection_id=inspection_id,
        split=split,
        row_count=rows,
        gene_count=genes,
        matrix_encoding=matrix_encoding,
        matrix_data_dtype=data_dtype,
        matrix_indices_dtype=indices_dtype,
        obs_fields=obs_fields,
        var_index_field=var_index_field,
        layer_ids=layer_ids,
        gene_order_sha256=gene_order_sha256,
        value_datasets_opened=value_datasets_opened,
        structure_only=structure_only,
        synthetic_test_fixture=synthetic_test_fixture,
    )


def _read_h5ad_column(group: h5py.Group, field: str, *, expected_rows: int) -> NDArray[np.str_]:
    if field not in group:
        raise VirtualCellContractError(f"obs field {field!r} is absent")
    node = group[field]
    if isinstance(node, h5py.Dataset):
        if node.shape != (expected_rows,):
            raise VirtualCellContractError(f"obs field {field!r} has the wrong length")
        return cast(NDArray[np.str_], np.asarray(node.asstr()[:], dtype=np.str_))
    if not isinstance(node, h5py.Group) or _encoding(node) != "categorical":
        raise VirtualCellContractError(f"obs field {field!r} has an unsupported encoding")
    if "codes" not in node or "categories" not in node:
        raise VirtualCellContractError(f"categorical obs field {field!r} is incomplete")
    codes = np.asarray(node["codes"][:])
    categories = np.asarray(node["categories"].asstr()[:], dtype=np.str_)
    if codes.shape != (expected_rows,):
        raise VirtualCellContractError(f"obs field {field!r} has the wrong length")
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise VirtualCellContractError(f"obs field {field!r} contains missing/invalid codes")
    return cast(NDArray[np.str_], categories[codes])


@dataclass(frozen=True, slots=True)
class ResponseSummaryArrays:
    """In-memory target/batch sufficient statistics for pure analysis code."""

    group_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    normalization: str
    normalization_target_sum: float | None
    cell_counts: NDArray[np.int64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]

    def __post_init__(self) -> None:
        group_count = len(self.group_ids)
        gene_count = len(self.gene_ids)
        if group_count == 0 or gene_count == 0:
            raise ValueError("response summary cannot be empty")
        if self.normalization not in {"source", "log1p-cp10000", "log1p-fixed-total"}:
            raise ValueError("response summary normalization is unknown")
        if self.normalization == "log1p-fixed-total":
            if (
                self.normalization_target_sum is None
                or not math.isfinite(self.normalization_target_sum)
                or self.normalization_target_sum <= 0
            ):
                raise ValueError("fixed-total response summary lacks a positive target sum")
        elif self.normalization_target_sum is not None:
            raise ValueError("only fixed-total summaries may carry a normalization target")
        if not (
            len(self.target_ids) == len(self.batch_ids) == group_count
            and self.cell_counts.shape == (group_count,)
            and self.means.shape == (group_count, gene_count)
            and self.variances.shape == (group_count, gene_count)
        ):
            raise ValueError("response summary array shapes are inconsistent")
        if (
            len(set(self.group_ids)) != group_count
            or tuple(sorted(self.group_ids)) != self.group_ids
        ):
            raise ValueError("group IDs must be sorted and unique")
        if len(set(self.gene_ids)) != gene_count:
            raise ValueError("gene IDs must be unique")
        if np.any(self.cell_counts <= 0):
            raise ValueError("each response group must contain at least one cell")
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.variances)):
            raise ValueError("response summary contains nonfinite values")
        if np.any(self.variances < 0):
            raise ValueError("response variances cannot be negative")


def require_response_summary_support(
    summary: ResponseSummaryArrays,
    *,
    control_label: str,
    minimum_cells_per_target: int,
) -> None:
    """Require frozen cell support and a matched comparator for every target batch."""

    validate_nonempty(control_label, field_name="control_label")
    _require_positive_integer(
        minimum_cells_per_target,
        field_name="minimum_cells_per_target",
    )
    counts: dict[str, int] = {}
    control_batches = set()
    target_batches = set()
    for target_id, batch_id, count in zip(
        summary.target_ids,
        summary.batch_ids,
        summary.cell_counts,
        strict=True,
    ):
        counts[target_id] = counts.get(target_id, 0) + int(count)
        if target_id == control_label:
            control_batches.add(batch_id)
        else:
            target_batches.add(batch_id)
    if control_label not in counts:
        raise VirtualCellContractError("response summary lacks the frozen comparator")
    unsupported = tuple(
        sorted(target_id for target_id, count in counts.items() if count < minimum_cells_per_target)
    )
    if unsupported:
        raise VirtualCellContractError(
            "response summary target cell support is below the frozen minimum: "
            + ",".join(unsupported)
        )
    missing_control_batches = tuple(sorted(target_batches - control_batches))
    if missing_control_batches:
        raise VirtualCellContractError(
            "response summary lacks matched control batches: " + ",".join(missing_control_batches)
        )


@dataclass(frozen=True, slots=True)
class ControlReservoirArrays:
    """Deterministic nested-cell reservoir for prediction-shape compilation.

    Reservoir rows retain their source/batch identities but are never promoted
    to independent preparations.  They supply only an empirical residual shape
    after development-visible normalization.
    """

    row_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    normalization: str
    normalization_target_sum: float
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not self.row_ids or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("control reservoir row IDs must be nonempty and unique")
        if len(self.batch_ids) != len(self.row_ids):
            raise ValueError("control reservoir batch IDs are not row aligned")
        if not self.gene_ids or len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("control reservoir gene IDs must be nonempty and unique")
        if self.normalization != "log1p-fixed-total":
            raise ValueError("control reservoir requires fixed-total log1p normalization")
        if not math.isfinite(self.normalization_target_sum) or self.normalization_target_sum <= 0:
            raise ValueError("control reservoir normalization target is invalid")
        if self.values.shape != (len(self.row_ids), len(self.gene_ids)):
            raise ValueError("control reservoir matrix has the wrong shape")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("control reservoir values must be finite and nonnegative")
        if float(np.max(self.values)) >= 15.0:
            raise ValueError("control reservoir exceeds cell-eval's log1p range")


def _selected_control_rows(
    *,
    targets: NDArray[np.str_],
    batches: NDArray[np.str_],
    control_label: str,
    maximum_cells: int,
    seed: int,
) -> NDArray[np.int64]:
    control_indices = np.flatnonzero(targets == control_label)
    if control_indices.size == 0:
        raise VirtualCellContractError("H5AD lacks declared comparator cells")
    batch_values = tuple(sorted(set(batches[control_indices].tolist())))
    if maximum_cells < len(batch_values):
        raise VirtualCellContractError("control reservoir is too small to cover observed batches")

    def priority(row: int) -> bytes:
        return hashlib.sha256(f"{seed}:{row}".encode("ascii")).digest()

    chosen: set[int] = set()
    for batch in batch_values:
        candidates = control_indices[batches[control_indices] == batch]
        chosen.add(min((int(value) for value in candidates), key=priority))
    remaining = maximum_cells - len(chosen)
    if remaining > 0:
        candidates = (int(value) for value in control_indices if int(value) not in chosen)
        chosen.update(sorted(candidates, key=priority)[:remaining])
    return np.asarray(sorted(chosen), dtype=np.int64)


def extract_control_reservoir_h5ad(
    stream: BinaryIO,
    *,
    split: VirtualCellSplit,
    target_field: str,
    batch_field: str,
    control_label: str,
    maximum_rows: int,
    maximum_genes: int,
    maximum_cells: int,
    normalization_target_sum: float,
    seed: int,
) -> ControlReservoirArrays:
    """Read a bounded deterministic control reservoir from a development split."""

    if split not in {VirtualCellSplit.TRAIN, VirtualCellSplit.VALIDATION}:
        raise VirtualCellContractError("control reservoirs are development-only")
    for name, value in (
        ("maximum_rows", maximum_rows),
        ("maximum_genes", maximum_genes),
        ("maximum_cells", maximum_cells),
    ):
        _require_positive_integer(value, field_name=name)
    if not math.isfinite(normalization_target_sum) or normalization_target_sum <= 0:
        raise ValueError("normalization target sum must be finite and positive")
    try:
        with h5py.File(stream, "r") as handle:
            matrix = handle["X"]
            if not isinstance(matrix, (h5py.Group, h5py.Dataset)):
                raise VirtualCellContractError("H5AD X has an unsupported object kind")
            rows, genes = _matrix_shape(matrix)
            if rows > maximum_rows or genes > maximum_genes:
                raise VirtualCellContractError("H5AD exceeds control-reservoir bounds")
            obs = handle["obs"]
            var = handle["var"]
            if not isinstance(obs, h5py.Group) or not isinstance(var, h5py.Group):
                raise VirtualCellContractError("H5AD obs/var must be dataframe groups")
            targets = _read_h5ad_column(obs, target_field, expected_rows=rows)
            batches = _read_h5ad_column(obs, batch_field, expected_rows=rows)
            selected = _selected_control_rows(
                targets=targets,
                batches=batches,
                control_label=control_label,
                maximum_cells=maximum_cells,
                seed=seed,
            )
            values = np.zeros((len(selected), genes), dtype=np.float32)
            if isinstance(matrix, h5py.Dataset):
                for output_index, row_index in enumerate(selected):
                    row = np.asarray(matrix[int(row_index), :], dtype=np.float64)
                    total = float(np.sum(row))
                    if total <= 0 or not np.all(np.isfinite(row)):
                        raise VirtualCellContractError("control row is invalid for normalization")
                    values[output_index] = np.log1p(
                        row * (normalization_target_sum / total)
                    ).astype(np.float32)
            else:
                if _encoding(matrix) != "csr_matrix":
                    raise VirtualCellContractError("control reservoir requires CSR or dense X")
                indptr_ds = matrix["indptr"]
                indices_ds = matrix["indices"]
                data_ds = matrix["data"]
                for output_index, row_index in enumerate(selected):
                    bounds = np.asarray(
                        indptr_ds[int(row_index) : int(row_index) + 2],
                        dtype=np.int64,
                    )
                    left, right = int(bounds[0]), int(bounds[1])
                    indices = np.asarray(indices_ds[left:right], dtype=np.int64)
                    data = np.asarray(data_ds[left:right], dtype=np.float64)
                    if np.any(indices < 0) or np.any(indices >= genes):
                        raise VirtualCellContractError("control CSR indices are invalid")
                    total = float(np.sum(data))
                    if total <= 0 or not np.all(np.isfinite(data)):
                        raise VirtualCellContractError("control row is invalid for normalization")
                    values[output_index, indices] = np.log1p(
                        data * (normalization_target_sum / total)
                    ).astype(np.float32)
            index_field = _frame_index_field(var)
            gene_ids = tuple(np.asarray(var[index_field].asstr()[:], dtype=np.str_).tolist())
            if len(gene_ids) != genes or len(set(gene_ids)) != genes:
                raise VirtualCellContractError("control reservoir gene registry is invalid")
    except VirtualCellContractError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise VirtualCellContractError("H5AD control-reservoir extraction failed closed") from error
    return ControlReservoirArrays(
        row_ids=tuple(f"{split.value.lower()}-row-{int(value):09d}" for value in selected),
        batch_ids=tuple(str(batches[int(value)]) for value in selected),
        gene_ids=gene_ids,
        normalization="log1p-fixed-total",
        normalization_target_sum=normalization_target_sum,
        values=values,
    )


def summarize_h5ad(
    stream: BinaryIO,
    *,
    split: VirtualCellSplit,
    target_field: str,
    batch_field: str,
    maximum_rows: int,
    maximum_genes: int,
    row_chunk_size: int = 4096,
    normalization: str = "source",
    normalization_target_sum: float | None = None,
    outcome_access: OutcomeAccess | None = None,
) -> ResponseSummaryArrays:
    """Aggregate CSR/dense development matrices by target and observed batch.

    Cells are nested views only.  The returned groups are target-by-batch
    summaries; downstream uncertainty therefore cannot silently treat cells as
    independent preparation replicates.
    """

    if split in {VirtualCellSplit.TRAIN, VirtualCellSplit.VALIDATION}:
        if outcome_access not in {None, OutcomeAccess.DEVELOPMENT_VISIBLE}:
            raise VirtualCellContractError("development summary has the wrong outcome access")
    elif split is VirtualCellSplit.TEST:
        if outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise VirtualCellContractError("test summaries are evaluator-reveal only")
    else:  # pragma: no cover - closed enum guard
        raise VirtualCellContractError("shared sources cannot be response summarized")
    _require_positive_integer(maximum_rows, field_name="maximum_rows")
    _require_positive_integer(maximum_genes, field_name="maximum_genes")
    _require_positive_integer(row_chunk_size, field_name="row_chunk_size")
    if normalization not in {"source", "log1p-cp10000", "log1p-fixed-total"}:
        raise ValueError("normalization is not supported")
    if normalization == "log1p-fixed-total":
        if (
            normalization_target_sum is None
            or not math.isfinite(normalization_target_sum)
            or normalization_target_sum <= 0
        ):
            raise ValueError("fixed-total normalization requires a positive target sum")
        target_sum = normalization_target_sum
    else:
        if normalization_target_sum is not None:
            raise ValueError("normalization target sum is only valid for fixed-total mode")
        target_sum = 10_000.0
    try:
        with h5py.File(stream, "r") as handle:
            matrix = handle["X"]
            if not isinstance(matrix, (h5py.Group, h5py.Dataset)):
                raise VirtualCellContractError("H5AD X has an unsupported object kind")
            rows, genes = _matrix_shape(matrix)
            if rows <= 0 or rows > maximum_rows or genes <= 0 or genes > maximum_genes:
                raise VirtualCellContractError("H5AD shape exceeds aggregation bounds")
            obs = handle["obs"]
            var = handle["var"]
            if not isinstance(obs, h5py.Group) or not isinstance(var, h5py.Group):
                raise VirtualCellContractError("H5AD obs/var must be dataframe groups")
            targets = _read_h5ad_column(obs, target_field, expected_rows=rows)
            batches = _read_h5ad_column(obs, batch_field, expected_rows=rows)
            index_field = _frame_index_field(var)
            gene_ids_array = np.asarray(var[index_field].asstr()[:], dtype=np.str_)
            if gene_ids_array.shape != (genes,) or len(set(gene_ids_array.tolist())) != genes:
                raise VirtualCellContractError("gene registry is missing, duplicated or mis-sized")
            pairs = sorted(set(zip(targets.tolist(), batches.tolist(), strict=True)))
            group_ids = tuple(f"target={target}|batch={batch}" for target, batch in pairs)
            pair_to_index = {pair: index for index, pair in enumerate(pairs)}
            row_groups = np.fromiter(
                (
                    pair_to_index[(target, batch)]
                    for target, batch in zip(targets.tolist(), batches.tolist(), strict=True)
                ),
                dtype=np.int64,
                count=rows,
            )
            counts = np.bincount(row_groups, minlength=len(pairs)).astype(np.int64, copy=False)
            sums = np.zeros((len(pairs), genes), dtype=np.float64)
            squared_sums = np.zeros_like(sums)
            if isinstance(matrix, h5py.Dataset):
                for start in range(0, rows, row_chunk_size):
                    stop = min(rows, start + row_chunk_size)
                    dense = np.asarray(matrix[start:stop, :], dtype=np.float64)
                    if not np.all(np.isfinite(dense)):
                        raise VirtualCellContractError("X contains nonfinite development values")
                    if normalization in {"log1p-cp10000", "log1p-fixed-total"}:
                        totals = np.sum(dense, axis=1)
                        if np.any(totals <= 0):
                            raise VirtualCellContractError(
                                "count normalization encountered a nonpositive cell total"
                            )
                        dense = np.log1p(dense * (target_sum / totals[:, None]))
                    np.add.at(sums, row_groups[start:stop], dense)
                    np.add.at(squared_sums, row_groups[start:stop], dense * dense)
            else:
                if _encoding(matrix) != "csr_matrix":
                    raise VirtualCellContractError(
                        "streaming summary currently requires CSR or dense X"
                    )
                indptr_ds = matrix["indptr"]
                indices_ds = matrix["indices"]
                data_ds = matrix["data"]
                for start in range(0, rows, row_chunk_size):
                    stop = min(rows, start + row_chunk_size)
                    indptr = np.asarray(indptr_ds[start : stop + 1], dtype=np.int64)
                    first = int(indptr[0])
                    last = int(indptr[-1])
                    indices = np.asarray(indices_ds[first:last], dtype=np.int64)
                    data = np.asarray(data_ds[first:last], dtype=np.float64)
                    if np.any(indices < 0) or np.any(indices >= genes):
                        raise VirtualCellContractError("CSR X contains out-of-range indices")
                    if not np.all(np.isfinite(data)):
                        raise VirtualCellContractError("X contains nonfinite development values")
                    local_indptr = indptr - first
                    for local_row, group_index in enumerate(row_groups[start:stop]):
                        left = int(local_indptr[local_row])
                        right = int(local_indptr[local_row + 1])
                        row_indices = indices[left:right]
                        row_data = data[left:right]
                        if normalization in {"log1p-cp10000", "log1p-fixed-total"}:
                            total = float(np.sum(row_data))
                            if total <= 0:
                                raise VirtualCellContractError(
                                    "count normalization encountered a nonpositive cell total"
                                )
                            row_data = np.log1p(row_data * (target_sum / total))
                        np.add.at(sums[group_index], row_indices, row_data)
                        np.add.at(squared_sums[group_index], row_indices, row_data * row_data)
            means = sums / counts[:, None]
            variances = np.maximum(squared_sums / counts[:, None] - means * means, 0.0)
    except VirtualCellContractError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise VirtualCellContractError("H5AD response summarization failed closed") from error
    return ResponseSummaryArrays(
        group_ids=group_ids,
        target_ids=tuple(target for target, _ in pairs),
        batch_ids=tuple(batch for _, batch in pairs),
        gene_ids=tuple(gene_ids_array.tolist()),
        normalization=normalization,
        normalization_target_sum=normalization_target_sum,
        cell_counts=counts,
        means=means,
        variances=variances,
    )


__all__ = [
    "CSVStructuralInspection",
    "ControlReservoirArrays",
    "H5ADStructuralInspection",
    "ResponseSummaryArrays",
    "SegmentedBinaryReader",
    "StreamIdentity",
    "copy_and_hash_stream",
    "decode_custody_receipt",
    "extract_control_reservoir_h5ad",
    "inspect_csv",
    "inspect_h5ad",
    "require_response_summary_support",
    "summarize_h5ad",
]
