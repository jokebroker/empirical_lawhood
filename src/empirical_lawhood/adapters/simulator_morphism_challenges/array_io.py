"""Deterministic no-pickle array packing for simulator morphism challenges task outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import prod

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .runtime_contracts import SimulatorMorphismChallengeArrayEntry, SimulatorMorphismChallengeArrayManifest


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ArraySemantics:
    native_unit: str
    coordinate_frame: str
    clock_id: str
    row_key_schema: str


@dataclass(frozen=True, slots=True)
class PackedArrays:
    payload: bytes
    manifest: SimulatorMorphismChallengeArrayManifest


def logical_arrays_digest(arrays: dict[str, FloatArray]) -> str:
    """Hash typed logical arrays independently of the NPY container."""

    digest = sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=np.float64)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(canonical_json_bytes(tuple(int(value) for value in values.shape)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def pack_arrays(
    *,
    manifest_id: str,
    unit_id: str,
    arrays: dict[str, FloatArray],
    semantics: dict[str, ArraySemantics],
) -> PackedArrays:
    if not arrays:
        raise ValueError("simulator morphism challenges array payload cannot be empty")
    if set(semantics) != set(arrays):
        raise ValueError("simulator morphism challenges array semantics must cover the exact array roster")
    entries: list[SimulatorMorphismChallengeArrayEntry] = []
    flattened: list[FloatArray] = []
    offset = 0
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("simulator morphism challenges array payload contains nonfinite values")
        flat = values.reshape(-1)
        entries.append(
            SimulatorMorphismChallengeArrayEntry(
                array_id=key,
                offset=offset,
                element_count=flat.size,
                shape=tuple(int(value) for value in values.shape),
                dtype_id="float64",
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
    payload_array = np.concatenate(flattened).astype(np.float64, copy=False)
    stream = BytesIO()
    np.save(stream, payload_array, allow_pickle=False)
    payload = stream.getvalue()
    manifest = SimulatorMorphismChallengeArrayManifest(
        manifest_id=manifest_id,
        unit_id=unit_id,
        payload_sha256=sha256(payload).hexdigest(),
        logical_arrays_sha256=logical_arrays_digest(arrays),
        entries=tuple(entries),
    )
    return PackedArrays(payload=payload, manifest=manifest)


def simulator_morphism_challenges_array_semantics(array_id: str) -> ArraySemantics:
    """Return the closed native semantics for one known simulator morphism challenge logical array."""

    leaf = array_id.split(".", 1)[-1]
    if leaf in {"singular-spectra", "row-equilibrated-singular-spectra"}:
        return ArraySemantics(
            native_unit="dimensionless",
            coordinate_frame="history-depth-by-singular-index",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/history-depth',
        )
    if leaf == "inverse-lag-propagator":
        return ArraySemantics(
            native_unit="dimensionless",
            coordinate_frame="rc-cell-to-rc-cell",
            clock_id="pre-action-history-lag-t-star",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/resistor-capacitor-cell',
        )
    if leaf == "nomination-modes":
        return ArraySemantics(
            native_unit="dimensionless-mode",
            coordinate_frame="nomination-by-rc-cell",
            clock_id="decision-time",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/nomination',
        )
    if leaf == "present-states":
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="decision-time",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/nomination',
        )
    if leaf.startswith("diagnostic-"):
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="future-diagnostic-t-star",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/nomination',
        )
    if leaf.startswith("action.a"):
        return ArraySemantics(
            native_unit="volt",
            coordinate_frame="nomination-member-by-rc-cell",
            clock_id="action-termination-t-star",
            row_key_schema='empirical-lawhood/simulator-morphism-challenges/array-rows/nomination',
        )
    raise ValueError(f"unknown simulator morphism challenges logical array semantics: {array_id}")


def unpack_arrays(
    payload: bytes,
    manifest: SimulatorMorphismChallengeArrayManifest,
    *,
    maximum_bytes: int,
) -> dict[str, FloatArray]:
    if len(payload) > maximum_bytes or sha256(payload).hexdigest() != manifest.payload_sha256:
        raise ValueError("simulator morphism challenges NPY payload exceeds or differs from its manifest")
    stream = BytesIO(payload)
    values = np.load(stream, allow_pickle=False)
    if stream.read(1) or values.dtype != np.float64 or values.ndim != 1:
        raise ValueError("simulator morphism challenges NPY payload is not one exact float64 vector")
    expected_size = sum(value.element_count for value in manifest.entries)
    if values.size != expected_size:
        raise ValueError("simulator morphism challenges NPY vector size differs from its manifest")
    arrays: dict[str, FloatArray] = {}
    for entry in manifest.entries:
        if prod(entry.shape) != entry.element_count:
            raise ValueError("simulator morphism challenges array manifest shape differs")
        stop = entry.offset + entry.element_count
        array = np.ascontiguousarray(values[entry.offset : stop].reshape(entry.shape))
        if sha256(array.tobytes(order="C")).hexdigest() != entry.content_sha256:
            raise ValueError("simulator morphism challenges array content differs from its entry digest")
        arrays[entry.array_id] = array
    if logical_arrays_digest(arrays) != manifest.logical_arrays_sha256:
        raise ValueError("simulator morphism challenges logical array digest differs")
    return arrays


__all__ = [
    "ArraySemantics",
    "PackedArrays",
    'simulator_morphism_challenges_array_semantics',
    "logical_arrays_digest",
    "pack_arrays",
    "unpack_arrays",
]
