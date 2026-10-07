"""Load one statically pinned dataset projection authority without discovery.

This module authenticates bytes only through an exact, configuration-pinned
receipt :class:`EvidenceReference`.  It never scans for a latest receipt, takes
a caller path, initializes SQLite, or treats a local projection as authority.

Decoding ``DatasetProjectionRebuildAuthorization`` checks the record's self-contained
signature against the key embedded in that same record.  That is useful for
detecting byte corruption, but it is not trusted-issuer authentication and is
not complete policy/request replay.  Any later write must still perform the
normal write-time authorization replay against a statically trusted issuer and
the exact request.  Production composition must remain absent until an exact
trusted receipt reference has been configured independently of SQLite.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
import stat
from types import MappingProxyType
from typing import Final, TypeVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_limits import MAX_DATASET_RECORD_CANONICAL_BYTES
from empirical_lawhood.planning.dataset_rebuild import DatasetProjectionRebuildAuthorization
from empirical_lawhood.planning.datasets import EvidenceReference, EvidenceReferenceKind
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogProjectionAnchor,
    DatasetCatalogProjectionReceipt,
    DatasetCatalogProjectionState,
    DatasetCatalogSnapshot,
    UnifiedCatalogSnapshot,
)

from .artifacts import ArtifactPlaneError, GuardedExternalRoot
from .dataset_io import _descriptor_identity, _secure_open_relative
from .dataset_projection import (
    MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
    decode_canonical_record,
)


_READ_CHUNK_BYTES: Final[int] = 1024 * 1024
_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


class DatasetCatalogAuthorityError(RuntimeError):
    """Pinned external projection authority is absent, invalid, or unstable."""


@dataclass(frozen=True, slots=True)
class DatasetCatalogAuthorityBundle:
    """Immutable, byte-authenticated inputs for one read-only catalog session.

    ``authorization_record`` is retained for later complete replay.  Its
    presence and internally valid signature do not by themselves establish a
    trusted issuer or authorize another write.
    """

    unified_snapshot: UnifiedCatalogSnapshot
    dataset_snapshot: DatasetCatalogSnapshot
    projection_state: DatasetCatalogProjectionState
    receipt: DatasetCatalogProjectionReceipt
    authorization_record: DatasetProjectionRebuildAuthorization
    anchor: DatasetCatalogProjectionAnchor


def _validate_byte_ceiling(
    value: int,
    *,
    field_name: str,
    absolute_maximum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= absolute_maximum:
        raise ValueError(f"{field_name} must be a positive integer within its absolute maximum")


def _read_descriptor_bounded(descriptor: int, *, maximum_bytes: int) -> bytes:
    """Read to EOF once, failing before retaining bytes beyond the ceiling."""

    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum_bytes
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise DatasetCatalogAuthorityError(
            "external authority record exceeds its configured byte ceiling"
        )
    return b"".join(chunks)


def _load_exact_reference(
    reference: EvidenceReference,
    *,
    roots_by_id: Mapping[str, GuardedExternalRoot],
    expected_kind: EvidenceReferenceKind,
    record_type: type[_RecordT],
    maximum_bytes: int,
    label: str,
) -> _RecordT:
    if reference.kind is not expected_kind:
        raise DatasetCatalogAuthorityError(f"{label} has the wrong evidence kind")
    if reference.evidence_schema != record_type.SCHEMA:
        raise DatasetCatalogAuthorityError(f"{label} has the wrong evidence schema")
    root = roots_by_id.get(reference.storage_root_id)
    if root is None:
        raise DatasetCatalogAuthorityError(f"{label} names an unregistered storage root")

    descriptor = -1
    rebound_descriptor = -1
    try:
        # Retain the root's ordinary mount and lexical-containment checks, but
        # never open the returned Path.  Every component is opened relative to
        # retained descriptors with O_NOFOLLOW.
        root.resolve(reference.relative_locator, for_write=False)
        descriptor = _secure_open_relative(
            root.contract.canonical_path,
            reference.relative_locator,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DatasetCatalogAuthorityError(f"{label} is not a regular file")
        if before.st_size > maximum_bytes:
            raise DatasetCatalogAuthorityError(f"{label} exceeds its configured byte ceiling")

        payload = _read_descriptor_bounded(descriptor, maximum_bytes=maximum_bytes)
        after_read = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after_read):
            raise DatasetCatalogAuthorityError(f"{label} changed during bounded read")

        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != reference.evidence_sha256:
            raise DatasetCatalogAuthorityError(f"{label} digest differs from its reference")
        record = decode_canonical_record(
            payload,
            record_type,
            maximum_bytes=maximum_bytes,
        )
        if record.canonical_bytes() != payload or record.fingerprint() != observed_sha256:
            raise DatasetCatalogAuthorityError(f"{label} is not exact canonical evidence")

        after_decode = os.fstat(descriptor)
        if _descriptor_identity(after_read) != _descriptor_identity(after_decode):
            raise DatasetCatalogAuthorityError(f"{label} changed during canonical decode")
        replayed_payload = _read_descriptor_bounded(
            descriptor,
            maximum_bytes=maximum_bytes,
        )
        after_replay = os.fstat(descriptor)
        if replayed_payload != payload or _descriptor_identity(
            after_decode
        ) != _descriptor_identity(after_replay):
            raise DatasetCatalogAuthorityError(f"{label} changed during stable replay")

        rebound_descriptor = _secure_open_relative(
            root.contract.canonical_path,
            reference.relative_locator,
        )
        rebound = os.fstat(rebound_descriptor)
        if not stat.S_ISREG(rebound.st_mode) or _descriptor_identity(
            after_replay
        ) != _descriptor_identity(rebound):
            raise DatasetCatalogAuthorityError(f"{label} locator changed during replay")
    except DatasetCatalogAuthorityError:
        raise
    except (ArtifactPlaneError, CanonicalizationError, OSError, TypeError, ValueError) as error:
        raise DatasetCatalogAuthorityError(f"{label} guarded canonical read failed") from error
    finally:
        if rebound_descriptor >= 0:
            os.close(rebound_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
    return record


@dataclass(frozen=True, slots=True)
class ExternalDatasetCatalogAuthorityLoader:
    """Load only explicitly pinned receipt authorities from registered roots."""

    trusted_receipt_references: tuple[EvidenceReference, ...]
    roots: tuple[GuardedExternalRoot, ...]
    maximum_receipt_bytes: int
    maximum_projection_bytes: int
    maximum_authorization_bytes: int
    _trusted_receipts_by_id: Mapping[str, EvidenceReference] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _roots_by_id: Mapping[str, GuardedExternalRoot] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.roots, tuple)
            or not self.roots
            or any(not isinstance(root, GuardedExternalRoot) for root in self.roots)
        ):
            raise ValueError("dataset authority roots must be a nonempty tuple of guarded roots")
        root_ids = tuple(root.contract.storage_root_id for root in self.roots)
        if tuple(sorted(set(root_ids))) != root_ids:
            raise ValueError("dataset authority roots must be sorted and unique by ID")
        canonical_paths = tuple(root.contract.canonical_path for root in self.roots)
        if len(set(canonical_paths)) != len(canonical_paths):
            raise ValueError("dataset authority roots cannot alias one canonical path")
        roots_by_id = MappingProxyType(dict(zip(root_ids, self.roots, strict=True)))

        if (
            not isinstance(self.trusted_receipt_references, tuple)
            or not self.trusted_receipt_references
            or any(
                not isinstance(reference, EvidenceReference)
                for reference in self.trusted_receipt_references
            )
        ):
            raise ValueError(
                "dataset authority requires a nonempty tuple of trusted receipt references"
            )
        receipt_ids = tuple(reference.evidence_id for reference in self.trusted_receipt_references)
        if tuple(sorted(set(receipt_ids))) != receipt_ids:
            raise ValueError("trusted receipt references must be sorted and unique by ID")
        locator_keys = tuple(
            (reference.storage_root_id, reference.relative_locator)
            for reference in self.trusted_receipt_references
        )
        if len(set(locator_keys)) != len(locator_keys):
            raise ValueError("trusted receipt references cannot repeat a locator")
        for reference in self.trusted_receipt_references:
            if (
                reference.kind is not EvidenceReferenceKind.RECEIPT
                or reference.evidence_schema != DatasetCatalogProjectionReceipt.SCHEMA
            ):
                raise ValueError("trusted receipt references must name dataset projection receipts")
            if reference.storage_root_id not in roots_by_id:
                raise ValueError("trusted receipt reference names an unregistered root")

        _validate_byte_ceiling(
            self.maximum_receipt_bytes,
            field_name="maximum_receipt_bytes",
            absolute_maximum=MAX_DATASET_RECORD_CANONICAL_BYTES,
        )
        _validate_byte_ceiling(
            self.maximum_projection_bytes,
            field_name="maximum_projection_bytes",
            absolute_maximum=MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
        )
        _validate_byte_ceiling(
            self.maximum_authorization_bytes,
            field_name="maximum_authorization_bytes",
            absolute_maximum=MAX_DATASET_RECORD_CANONICAL_BYTES,
        )
        object.__setattr__(self, "_roots_by_id", roots_by_id)
        object.__setattr__(
            self,
            "_trusted_receipts_by_id",
            MappingProxyType(
                dict(
                    zip(
                        receipt_ids,
                        self.trusted_receipt_references,
                        strict=True,
                    )
                )
            ),
        )

    def load(self, receipt_id: str) -> DatasetCatalogAuthorityBundle:
        """Load one exact pinned receipt; IDs never resolve to caller locators."""

        validate_stable_id(receipt_id, field_name="receipt_id")
        receipt_reference = self._trusted_receipts_by_id.get(receipt_id)
        if receipt_reference is None:
            raise DatasetCatalogAuthorityError("dataset projection receipt is not trusted")

        receipt = _load_exact_reference(
            receipt_reference,
            roots_by_id=self._roots_by_id,
            expected_kind=EvidenceReferenceKind.RECEIPT,
            record_type=DatasetCatalogProjectionReceipt,
            maximum_bytes=self.maximum_receipt_bytes,
            label="dataset projection receipt",
        )
        if receipt.receipt_id != receipt_id:
            raise DatasetCatalogAuthorityError(
                "trusted receipt ID differs from the decoded projection receipt"
            )

        unified_snapshot = _load_exact_reference(
            receipt.unified_projection_evidence,
            roots_by_id=self._roots_by_id,
            expected_kind=EvidenceReferenceKind.MANIFEST,
            record_type=UnifiedCatalogSnapshot,
            maximum_bytes=self.maximum_projection_bytes,
            label="unified catalog projection",
        )
        authorization = _load_exact_reference(
            receipt.authorization_evidence,
            roots_by_id=self._roots_by_id,
            expected_kind=EvidenceReferenceKind.AUTHORIZATION,
            record_type=DatasetProjectionRebuildAuthorization,
            maximum_bytes=self.maximum_authorization_bytes,
            label="dataset operation authorization",
        )
        if receipt.authorization_evidence.evidence_id != authorization.decision.authorization_id:
            raise DatasetCatalogAuthorityError(
                "authorization evidence ID differs from the authorization record"
            )

        dataset_snapshot = unified_snapshot.datasets
        projection_state = receipt.projection_state
        try:
            build_valid = receipt.validates_build(
                unified_snapshot,
                dataset_snapshot,
                projection_state,
            )
        except (TypeError, ValueError) as error:
            raise DatasetCatalogAuthorityError(
                "projection receipt differs from its exact nested dataset build"
            ) from error
        if not build_valid:
            raise DatasetCatalogAuthorityError(
                "projection receipt differs from its exact nested dataset build"
            )
        try:
            anchor = receipt.derive_anchor(
                receipt_reference,
                unified_snapshot=unified_snapshot,
                dataset_snapshot=dataset_snapshot,
                projection_state=projection_state,
            )
        except (TypeError, ValueError) as error:
            raise DatasetCatalogAuthorityError(
                "projection anchor cannot be derived from the pinned receipt"
            ) from error
        return DatasetCatalogAuthorityBundle(
            unified_snapshot=unified_snapshot,
            dataset_snapshot=dataset_snapshot,
            projection_state=projection_state,
            receipt=receipt,
            authorization_record=authorization,
            anchor=anchor,
        )


__all__ = [
    "DatasetCatalogAuthorityBundle",
    "DatasetCatalogAuthorityError",
    "ExternalDatasetCatalogAuthorityLoader",
]
