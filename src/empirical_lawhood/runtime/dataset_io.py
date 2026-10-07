"""Path-free runtime contracts for opening exact, already-held dataset bytes."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from collections.abc import Callable
from typing import BinaryIO, ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import (
    DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA,
    DatasetStorageScope,
)
from empirical_lawhood.planning.dataset_manifests import DatasetDirectorySelectorManifest


@dataclass(frozen=True, slots=True)
class DatasetSourceGuardReceipt(CanonicalRecord):
    """Read-only proof for one exact regular file behind a registered root.

    This receipt is deliberately about storage/custody mechanics only.  It does
    not claim that the bytes satisfy a dataset schema or that a materialization
    is verified; a registered inspector must establish those facts separately.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-source-guard-receipt'

    guard_receipt_id: str
    storage_root: ObjectIdentity
    source_scope: DatasetStorageScope
    relative_locator: str
    observed_size_bytes: int
    observed_sha256: str
    verified_at_utc: str
    regular_file: bool
    opened_read_only: bool
    nofollow_enforced: bool
    complete_eof: bool
    file_identity_stable: bool
    network_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.guard_receipt_id, field_name="guard_receipt_id")
        if not isinstance(self.storage_root, ObjectIdentity):
            raise ValueError("storage_root must be an exact ObjectIdentity")
        if self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA:
            raise ValueError("storage_root must identify an ExternalRootContract")
        if not isinstance(self.source_scope, DatasetStorageScope):
            raise ValueError("source_scope must be a DatasetStorageScope")
        if self.source_scope.storage_root != self.storage_root:
            raise ValueError("source scope and guarded storage root differ")
        validate_relative_locator(self.relative_locator)
        if self.relative_locator != self.source_scope.relative_prefix:
            raise ValueError("guard receipt must bind the exact source-scope locator")
        if (
            not isinstance(self.observed_size_bytes, int)
            or isinstance(self.observed_size_bytes, bool)
            or self.observed_size_bytes < 0
        ):
            raise ValueError("observed_size_bytes must be a nonnegative integer")
        validate_sha256(self.observed_sha256, field_name="observed_sha256")
        parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        for field_name in (
            "regular_file",
            "opened_read_only",
            "nofollow_enforced",
            "complete_eof",
            "file_identity_stable",
        ):
            if getattr(self, field_name) is not True:
                raise ValueError(f"{field_name} must be true for a guard receipt")
        if (
            not isinstance(self.network_bytes, int)
            or isinstance(self.network_bytes, bool)
            or self.network_bytes != 0
        ):
            raise ValueError("guarded held-source reads must use zero network bytes")

    @property
    def observed_file_count(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class DatasetDirectoryGuardReceipt(CanonicalRecord):
    """Aggregate proof for an exact allowlist of regular directory members."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-directory-guard-receipt'

    guard_receipt_id: str
    storage_root: ObjectIdentity
    source_scope: DatasetStorageScope
    selector: ObjectIdentity
    observed_content_sha256: str
    observed_size_bytes: int
    observed_file_count: int
    verified_at_utc: str
    regular_files: bool
    opened_read_only: bool
    nofollow_enforced: bool
    complete_eof: bool
    file_identities_stable: bool
    literal_allowlist_only: bool
    network_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.guard_receipt_id, field_name="guard_receipt_id")
        if not isinstance(self.storage_root, ObjectIdentity):
            raise ValueError("storage_root must be an exact ObjectIdentity")
        if self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA:
            raise ValueError("storage_root must identify an ExternalRootContract")
        if not isinstance(self.source_scope, DatasetStorageScope):
            raise ValueError("source_scope must be a DatasetStorageScope")
        if self.source_scope.storage_root != self.storage_root:
            raise ValueError("source scope and guarded storage root differ")
        if (
            not isinstance(self.selector, ObjectIdentity)
            or self.selector.object_schema != DatasetDirectorySelectorManifest.SCHEMA
        ):
            raise ValueError("selector must identify a directory selector manifest")
        validate_sha256(
            self.observed_content_sha256,
            field_name="observed_content_sha256",
        )
        if (
            isinstance(self.observed_size_bytes, bool)
            or not isinstance(self.observed_size_bytes, int)
            or self.observed_size_bytes <= 0
        ):
            raise ValueError("observed_size_bytes must be a positive integer")
        if (
            isinstance(self.observed_file_count, bool)
            or not isinstance(self.observed_file_count, int)
            or self.observed_file_count <= 0
        ):
            raise ValueError("observed_file_count must be a positive integer")
        parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        for field_name in (
            "regular_files",
            "opened_read_only",
            "nofollow_enforced",
            "complete_eof",
            "file_identities_stable",
            "literal_allowlist_only",
        ):
            if getattr(self, field_name) is not True:
                raise ValueError(f"{field_name} must be true for a directory guard")
        if (
            isinstance(self.network_bytes, bool)
            or not isinstance(self.network_bytes, int)
            or self.network_bytes != 0
        ):
            raise ValueError("guarded directory reads must use zero network bytes")


@dataclass(frozen=True, slots=True)
class GuardedDatasetSource:
    """A seekable stream whose custody receipt appears only after clean close.

    The stream remains valid only inside the context returned by a
    :class:`DatasetSourceOpener`.  Callers receive no filesystem path and
    cannot consume a success receipt before the opener's post-use mutation
    check has passed.
    """

    stream: BinaryIO
    guard_receipt_id: str
    _completed_receipt_reader: Callable[[], DatasetSourceGuardReceipt | None]

    def __post_init__(self) -> None:
        validate_stable_id(self.guard_receipt_id, field_name="guard_receipt_id")
        if not callable(self._completed_receipt_reader):
            raise ValueError("completed receipt reader must be callable")

    @property
    def guard_receipt(self) -> DatasetSourceGuardReceipt:
        """Return the receipt only after the guarded context exits cleanly."""

        receipt = self._completed_receipt_reader()
        if receipt is None:
            raise RuntimeError("dataset source guard receipt is not complete")
        if receipt.guard_receipt_id != self.guard_receipt_id:
            raise RuntimeError("completed dataset source guard receipt identity differs")
        return receipt


class DatasetSourceOpener(Protocol):
    """Open one exact held source without accepting an arbitrary path."""

    def open_exact(
        self,
        *,
        source_scope: DatasetStorageScope,
        expected_size_bytes: int,
        expected_sha256: str,
        maximum_bytes: int,
        trusted_at_utc: str,
    ) -> AbstractContextManager[GuardedDatasetSource]: ...


class DatasetDirectorySourceInspector(Protocol):
    """Inspect one exact literal directory allowlist without enumerating a root."""

    def inspect_exact(
        self,
        *,
        source_scope: DatasetStorageScope,
        selector: DatasetDirectorySelectorManifest,
        maximum_bytes: int,
        maximum_files: int,
        maximum_single_file_bytes: int,
        trusted_at_utc: str,
    ) -> DatasetDirectoryGuardReceipt: ...


__all__ = [
    "DatasetSourceGuardReceipt",
    "DatasetDirectoryGuardReceipt",
    "DatasetDirectorySourceInspector",
    "DatasetSourceOpener",
    "GuardedDatasetSource",
]
