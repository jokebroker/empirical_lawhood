"""Trusted, read-only composition for the local dataset catalog projection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import stat

from sqlalchemy import Engine

from empirical_lawhood.runtime.dataset_catalog_service import DatasetCatalogReadService
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogProjectionAnchor,
    DatasetCatalogSnapshot,
)
from empirical_lawhood.planning.datasets import EvidenceReference, EvidenceReferenceKind
from empirical_lawhood.runtime.datasets import DatasetCatalogProjectionReceipt

from .dataset_projection import decode_canonical_record
from .sql import SQLiteDatasetRepository, create_read_only_catalog_engine


_ReadOnlyCatalogEngineFactory = Callable[[Path], Engine]
_MAX_LOCAL_TRUST_REFERENCE_BYTES = 1024 * 1024


def load_local_dataset_projection_receipt_reference(path: Path) -> EvidenceReference:
    """Load one owner-only local pointer to authoritative external receipt bytes."""

    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise ValueError("dataset projection trust path must name one file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(path.parent, directory_flags)
    descriptor = -1
    try:
        directory_state = os.fstat(directory)
        if (
            not stat.S_ISDIR(directory_state.st_mode)
            or directory_state.st_uid != os.geteuid()
            or stat.S_IMODE(directory_state.st_mode) & 0o077
        ):
            raise PermissionError("dataset projection trust directory must be owner-only")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _MAX_LOCAL_TRUST_REFERENCE_BYTES
        ):
            raise PermissionError(
                "dataset projection trust reference must be one owner-only 0600 file"
            )
        chunks: list[bytes] = []
        remaining = _MAX_LOCAL_TRUST_REFERENCE_BYTES
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("dataset projection trust reference exceeds its byte limit")
        after = os.fstat(descriptor)
        if _trust_file_identity(before) != _trust_file_identity(after):
            raise PermissionError("dataset projection trust reference changed while loading")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)
    reference = decode_canonical_record(
        b"".join(chunks),
        EvidenceReference,
        maximum_bytes=_MAX_LOCAL_TRUST_REFERENCE_BYTES,
    )
    if (
        reference.kind is not EvidenceReferenceKind.RECEIPT
        or reference.evidence_schema != DatasetCatalogProjectionReceipt.SCHEMA
    ):
        raise ValueError("local dataset trust must pin one projection receipt")
    return reference


def _trust_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class DatasetCatalogReadSessionProvider:
    """Compose one authenticated dataset view over an existing SQLite projection.

    The snapshot and anchor are trust inputs, not artifacts authenticated by this
    provider.  A production composition root must obtain both from an external
    authority before constructing this object.  The local SQLite database is only
    a disposable read projection and can never supply or repair that authority.
    """

    __slots__ = (
        "database_path",
        "trusted_anchor",
        "trusted_snapshot",
        "_engine_factory",
    )

    def __init__(
        self,
        database_path: Path,
        trusted_snapshot: DatasetCatalogSnapshot,
        trusted_anchor: DatasetCatalogProjectionAnchor,
        *,
        engine_factory: _ReadOnlyCatalogEngineFactory = create_read_only_catalog_engine,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path")
        if not isinstance(trusted_snapshot, DatasetCatalogSnapshot):
            raise TypeError("trusted_snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(trusted_anchor, DatasetCatalogProjectionAnchor):
            raise TypeError("trusted_anchor must be a DatasetCatalogProjectionAnchor")
        if not trusted_anchor.validates_snapshot(trusted_snapshot):
            raise ValueError("trusted projection anchor differs from its exact snapshot")
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")

        # Retain the exact caller-supplied path and trust records.  In particular,
        # construction performs no filesystem access and derives nothing from SQLite.
        self.database_path = database_path
        self.trusted_snapshot = trusted_snapshot
        self.trusted_anchor = trusted_anchor
        # This keyword-only override is a bounded test seam.  Production
        # composition must retain the fixed read-only factory above.
        self._engine_factory = engine_factory

    @contextmanager
    def open(self) -> Iterator[DatasetCatalogReadService]:
        """Yield a trusted read service and dispose its read-only engine on exit."""

        engine = self._engine_factory(self.database_path)
        if not isinstance(engine, Engine):
            raise TypeError("engine_factory must return a SQLAlchemy Engine")
        try:
            repository = SQLiteDatasetRepository(engine)
            yield DatasetCatalogReadService(
                repository,
                trusted_snapshot=self.trusted_snapshot,
                trusted_anchor=self.trusted_anchor,
            )
        finally:
            engine.dispose()


__all__ = [
    "DatasetCatalogReadSessionProvider",
    "load_local_dataset_projection_receipt_reference",
]
