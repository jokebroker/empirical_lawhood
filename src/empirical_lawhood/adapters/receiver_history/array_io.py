"""Deterministic no-pickle array packing for receiver-history task outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import prod
from typing import Mapping, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .runtime_contracts import (
    FLOAT64_ARRAY_PAYLOAD_SCHEMA,
    INT64_ARRAY_PAYLOAD_SCHEMA,
    ReceiverHistoryArrayEntry,
    ReceiverHistoryArrayManifest,
)


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
TypedArray = FloatArray | IntArray


@dataclass(frozen=True, slots=True)
class ArraySemantics:
    native_unit: str
    coordinate_frame: str
    clock_id: str
    row_key_schema: str


@dataclass(frozen=True, slots=True)
class PackedArrays:
    payload: bytes
    manifest: ReceiverHistoryArrayManifest


def _dtype_id(arrays: Mapping[str, TypedArray]) -> str:
    dtypes = {np.asarray(value).dtype for value in arrays.values()}
    if dtypes == {np.dtype(np.float64)}:
        return "float64"
    if dtypes == {np.dtype(np.int64)}:
        return "int64"
    raise ValueError("receiver-history payload must contain exactly one supported dtype")


def logical_arrays_digest(arrays: Mapping[str, TypedArray]) -> str:
    """Hash typed logical arrays independently of the NPY container."""

    digest = sha256()
    dtype_id = _dtype_id(arrays)
    dtype = np.float64 if dtype_id == "float64" else np.int64
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=dtype)
        digest.update(key.encode("utf-8"))
        digest.update(f"\0{dtype_id}\0".encode("ascii"))
        digest.update(canonical_json_bytes(tuple(int(value) for value in values.shape)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def pack_arrays(
    *,
    manifest_id: str,
    unit_id: str,
    arrays: Mapping[str, TypedArray],
    semantics: dict[str, ArraySemantics],
) -> PackedArrays:
    if not arrays:
        raise ValueError("receiver-history array payload cannot be empty")
    if set(semantics) != set(arrays):
        raise ValueError("receiver-history array semantics must cover the exact array roster")
    entries: list[ReceiverHistoryArrayEntry] = []
    dtype_id = _dtype_id(arrays)
    dtype = np.float64 if dtype_id == "float64" else np.int64
    flattened: list[TypedArray] = []
    offset = 0
    for key in sorted(arrays):
        values = cast(TypedArray, np.ascontiguousarray(arrays[key], dtype=dtype))
        if dtype_id == "float64" and not np.all(np.isfinite(values)):
            raise ValueError("receiver-history array payload contains nonfinite values")
        flat = cast(TypedArray, values.reshape(-1))
        entries.append(
            ReceiverHistoryArrayEntry(
                array_id=key,
                offset=offset,
                element_count=flat.size,
                shape=tuple(int(value) for value in values.shape),
                dtype_id=dtype_id,
                native_unit=semantics[key].native_unit,
                coordinate_frame=semantics[key].coordinate_frame,
                clock_id=semantics[key].clock_id,
                independent_unit_id=unit_id,
                row_key_schema=semantics[key].row_key_schema,
                content_sha256=sha256(values.tobytes(order="C")).hexdigest(),
            )
        )
        flattened.append(flat)
        offset += flat.size
    payload_array = np.concatenate(flattened).astype(dtype, copy=False)
    stream = BytesIO()
    np.save(stream, payload_array, allow_pickle=False)
    payload = stream.getvalue()
    manifest = ReceiverHistoryArrayManifest(
        manifest_id=manifest_id,
        unit_id=unit_id,
        payload_schema=(
            FLOAT64_ARRAY_PAYLOAD_SCHEMA if dtype_id == "float64" else INT64_ARRAY_PAYLOAD_SCHEMA
        ),
        payload_sha256=sha256(payload).hexdigest(),
        logical_arrays_sha256=logical_arrays_digest(arrays),
        entries=tuple(entries),
    )
    return PackedArrays(payload=payload, manifest=manifest)


def receiver_history_array_semantics(array_id: str) -> ArraySemantics:
    """Return the closed native semantics for one known receiver-history logical array."""

    leaf = array_id.split(".", 1)[-1]
    if leaf.startswith("untouched-"):
        leaf = leaf.removeprefix("untouched-")
    if leaf in {"singular-spectra", "row-equilibrated-singular-spectra"}:
        return ArraySemantics(
            native_unit="dimensionless",
            coordinate_frame="history-depth-by-singular-index",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/history-depth',
        )
    if leaf == "inverse-lag-propagator":
        return ArraySemantics(
            native_unit="dimensionless",
            coordinate_frame="rc-cell-to-rc-cell",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/resistor-capacitor-cell',
        )
    if leaf in {"nomination-modes", "targeted-directions"}:
        return ArraySemantics(
            native_unit="dimensionless-mode",
            coordinate_frame="nomination-by-rc-cell",
            clock_id="decision-time",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/nomination',
        )
    if leaf == "certificate-closure-digest":
        return ArraySemantics(
            native_unit="byte-code",
            coordinate_frame="certificate-closure-digest",
            clock_id="pre-outcome-freeze",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/digest-byte',
        )
    if leaf in {
        "present-states",
        "present-states-n64",
        "present-states-n128",
        "present-states-n256",
        "initial-states-n64",
        "initial-states-n128",
        "initial-states-n256",
    }:
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="decision-time",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/nomination',
        )
    if leaf.startswith("diagnostic-"):
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="future-diagnostic-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/nomination',
        )
    if leaf.startswith("action.a"):
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="action-termination-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/nomination',
        )
    if leaf.startswith(("state-timeline-n", "receiver-history-n")) or leaf == "receiver-history":
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="preparation-by-history-lag-by-coordinate",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/preparation',
        )
    if leaf == "action-metrics":
        return ArraySemantics(
            native_unit="dimensionless",
            coordinate_frame="preparation-by-action-by-metric",
            clock_id="common-panel-endpoint-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/preparation-action',
        )
    if leaf == "action-receiver":
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="preparation-by-action-by-receiver-bin",
            clock_id="common-panel-endpoint-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/preparation-action',
        )
    if leaf.startswith("collision-edges-n"):
        return ArraySemantics(
            native_unit="index",
            coordinate_frame="coordinate-left-right",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/collision-edge',
        )
    if leaf.startswith("raw-bits-n"):
        return ArraySemantics(
            native_unit="ieee754-binary64-bit-pattern",
            coordinate_frame="preparation-by-rc-cell",
            clock_id="earliest-history-time",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/preparation',
        )
    if leaf.startswith("validity-mask-n"):
        return ArraySemantics(
            native_unit="binary-validity",
            coordinate_frame="preparation",
            clock_id="pre-outcome-freeze",
            row_key_schema='empirical-lawhood/receiver-history/array-rows/preparation',
        )
    raise ValueError(f"unknown receiver-history logical array semantics: {array_id}")


def unpack_arrays(
    payload: bytes,
    manifest: ReceiverHistoryArrayManifest,
    *,
    maximum_bytes: int,
) -> dict[str, TypedArray]:
    if len(payload) > maximum_bytes or sha256(payload).hexdigest() != manifest.payload_sha256:
        raise ValueError("receiver-history NPY payload exceeds or differs from its manifest")
    stream = BytesIO(payload)
    values = np.load(stream, allow_pickle=False)
    expected_dtype = (
        np.dtype(np.float64)
        if manifest.payload_schema == FLOAT64_ARRAY_PAYLOAD_SCHEMA
        else np.dtype(np.int64)
    )
    if stream.read(1) or values.dtype != expected_dtype or values.ndim != 1:
        raise ValueError("receiver-history NPY payload dtype/vector contract differs")
    expected_size = sum(value.element_count for value in manifest.entries)
    if values.size != expected_size:
        raise ValueError("receiver-history NPY vector size differs from its manifest")
    arrays: dict[str, TypedArray] = {}
    for entry in manifest.entries:
        if prod(entry.shape) != entry.element_count:
            raise ValueError("receiver-history array manifest shape differs")
        stop = entry.offset + entry.element_count
        array = np.ascontiguousarray(values[entry.offset : stop].reshape(entry.shape))
        if sha256(array.tobytes(order="C")).hexdigest() != entry.content_sha256:
            raise ValueError("receiver-history array content differs from its entry digest")
        arrays[entry.array_id] = array
    if logical_arrays_digest(arrays) != manifest.logical_arrays_sha256:
        raise ValueError("receiver-history logical array digest differs")
    return arrays


__all__ = [
    "ArraySemantics",
    "PackedArrays",
    "receiver_history_array_semantics",
    "logical_arrays_digest",
    "pack_arrays",
    "unpack_arrays",
]
