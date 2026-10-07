'Auditable HDF5 envelope for one bounded material control solver-output archive.\n\nThe inner deterministic tar stream preserves the solver filenames and bytes.\nThe outer fixed-inventory HDF5 envelope lets the existing artifact plane apply\nits bounded, no-links/no-object-dtype validator.  The tar SHA-256 is the logical\nscientific content identity; HDF5 file bytes retain their separate physical\npublication identity.\n'

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import h5py  # type: ignore[import-untyped]
import numpy as np

RAW_ARCHIVE_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-control/material-control-raw-workflow-archive'
RAW_ARCHIVE_VERSION: Final = "1.0.0"
RAW_ARCHIVE_DATASET: Final = "archive_bytes"
RAW_ARCHIVE_MEDIA_TYPE: Final = "application/x-hdf5"
COPY_CHUNK_BYTES: Final = 8 * 1024**2

ROOT_ATTRIBUTES: Final = {
    "archive_media_type": "application/gzip+tar",
    "empirical_lawhood_clocks": '{"archive_bytes":"not-applicable"}',
    "empirical_lawhood_frames": '{"archive_bytes":"deterministic-tar-byte-stream"}',
    "empirical_lawhood_keys": '["archive_sha256","profile_id"]',
    "empirical_lawhood_payload_schema": RAW_ARCHIVE_SCHEMA,
    "empirical_lawhood_units": '{"archive_bytes":"byte"}',
    "schema": RAW_ARCHIVE_SCHEMA,
    "version": RAW_ARCHIVE_VERSION,
}


def encode_raw_archive_hdf5(
    *, archive_path: Path, output_path: Path, profile_id: str, archive_sha256: str
) -> tuple[str, int]:
    """Wrap an already bounded archive without loading it wholly into memory."""

    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError('material control raw archive is not a regular file')
    if output_path.exists() or output_path.is_symlink():
        raise ValueError('material control raw HDF5 output path is not fresh')
    size = archive_path.stat().st_size
    if size <= 0:
        raise ValueError('material control raw archive cannot be empty')
    observed = sha256()
    with archive_path.open("rb") as source:
        while block := source.read(COPY_CHUNK_BYTES):
            observed.update(block)
    if observed.hexdigest() != archive_sha256:
        raise ValueError('material control raw archive logical identity differs')

    with h5py.File(output_path, "x") as handle:
        attributes = {
            **ROOT_ATTRIBUTES,
            "archive_sha256": archive_sha256,
            "profile_id": profile_id,
        }
        for name, value in sorted(attributes.items()):
            handle.attrs[name] = np.bytes_(value.encode("utf-8"))
        dataset = handle.create_dataset(
            RAW_ARCHIVE_DATASET,
            shape=(size,),
            dtype=np.dtype("u1"),
            chunks=(min(COPY_CHUNK_BYTES, size),),
            track_times=False,
        )
        with archive_path.open("rb") as source:
            offset = 0
            while block := source.read(COPY_CHUNK_BYTES):
                stop = offset + len(block)
                dataset[offset:stop] = np.frombuffer(block, dtype=np.uint8)
                offset = stop
        if offset != size:
            raise ValueError('material control raw archive changed while being wrapped')
        handle.flush()

    physical = sha256()
    with output_path.open("rb") as source:
        while block := source.read(COPY_CHUNK_BYTES):
            physical.update(block)
    return physical.hexdigest(), output_path.stat().st_size


__all__ = [
    "RAW_ARCHIVE_DATASET",
    "RAW_ARCHIVE_MEDIA_TYPE",
    "RAW_ARCHIVE_SCHEMA",
    "RAW_ARCHIVE_VERSION",
    "ROOT_ATTRIBUTES",
    "encode_raw_archive_hdf5",
]
