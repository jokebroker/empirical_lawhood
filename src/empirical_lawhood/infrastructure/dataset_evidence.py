"""Guarded replay of exact canonical dataset evidence from external roots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
import stat
from types import MappingProxyType
from typing import Final

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityKind,
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
)
from empirical_lawhood.planning.datasets import DatasetMaterializationVerificationReceipt
from empirical_lawhood.runtime.datasets import (
    DatasetEvidenceVerification,
    DatasetEvidenceVerificationRequest,
    DatasetEvidenceVerifierRegistration,
)

from .artifacts import ArtifactIdentityConflict, GuardedExternalRoot
from .dataset_io import _descriptor_identity, _secure_open_relative


_READ_CHUNK_BYTES: Final[int] = 1024 * 1024
EXTERNAL_DATASET_EVIDENCE_RECORD_TYPES: Final[Mapping[str, type[CanonicalRecord]]] = (
    MappingProxyType(
        {
            DatasetMaterializationVerificationReceipt.SCHEMA: (
                DatasetMaterializationVerificationReceipt
            ),
            DatasetRegistrationManifest.SCHEMA: DatasetRegistrationManifest,
            DatasetTransformationManifest.SCHEMA: DatasetTransformationManifest,
        }
    )
)
EXTERNAL_DATASET_EVIDENCE_SCHEMA_IDS: Final[tuple[str, ...]] = tuple(
    sorted(EXTERNAL_DATASET_EVIDENCE_RECORD_TYPES)
)


def _require_verifier_authority_binding(
    record: CanonicalRecord,
    registration: DatasetEvidenceVerifierRegistration,
) -> None:
    if not isinstance(record, (DatasetRegistrationManifest, DatasetTransformationManifest)):
        return
    matches = tuple(
        binding
        for binding in record.capabilities
        if binding.kind is DatasetCapabilityKind.EVIDENCE_VERIFIER
    )
    expected_identity = ObjectIdentity.from_record(
        registration.registry_id,
        registration,
    )
    if (
        len(matches) != 1
        or matches[0].registry_key != registration.registry_id
        or matches[0].implementation != expected_identity
    ):
        raise ArtifactIdentityConflict("dataset evidence manifest binds another evidence verifier")


def _read_descriptor_bounded(descriptor: int, *, maximum_bytes: int) -> bytes:
    """Read one complete descriptor without crossing the registered byte bound."""

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
        raise ArtifactIdentityConflict("dataset evidence exceeds its registered byte bound")
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class ExternalDatasetEvidenceVerifier:
    """Verify registered evidence through stable descriptor-relative reads only."""

    registration: DatasetEvidenceVerifierRegistration
    roots: tuple[GuardedExternalRoot, ...]
    _roots_by_id: Mapping[str, GuardedExternalRoot] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.registration, DatasetEvidenceVerifierRegistration):
            raise ValueError("registration must be a DatasetEvidenceVerifierRegistration")
        if self.registration.supported_evidence_schema_ids != (
            EXTERNAL_DATASET_EVIDENCE_SCHEMA_IDS
        ):
            raise ValueError(
                "external evidence verifier registration must bind the exact closed schema set"
            )
        if not isinstance(self.roots, tuple) or not self.roots:
            raise ValueError("external evidence verifier requires registered storage roots")
        if any(not isinstance(root, GuardedExternalRoot) for root in self.roots):
            raise ValueError("external evidence verifier roots must be GuardedExternalRoot values")
        root_ids = tuple(root.contract.storage_root_id for root in self.roots)
        if tuple(sorted(set(root_ids))) != root_ids:
            raise ValueError("external evidence verifier roots must be sorted and unique")
        object.__setattr__(
            self,
            "_roots_by_id",
            MappingProxyType(dict(zip(root_ids, self.roots, strict=True))),
        )

    @property
    def capability_key(self) -> str:
        return self.registration.capability_key

    @property
    def capability_version(self) -> str:
        return self.registration.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.registration.implementation_sha256

    @property
    def capability_manifest_fingerprint(self) -> str:
        return self.registration.capability.fingerprint()

    @property
    def registration_fingerprint(self) -> str:
        return self.registration.fingerprint()

    def verify(
        self,
        request: DatasetEvidenceVerificationRequest,
        *,
        maximum_bytes: int,
    ) -> DatasetEvidenceVerification:
        if not isinstance(request, DatasetEvidenceVerificationRequest):
            raise TypeError("request must be a DatasetEvidenceVerificationRequest")
        if (
            not isinstance(maximum_bytes, int)
            or isinstance(maximum_bytes, bool)
            or maximum_bytes <= 0
        ):
            raise ValueError("maximum_bytes must be a positive integer")
        reference = request.reference
        if reference.evidence_schema not in self.registration.supported_evidence_schema_ids:
            raise ArtifactIdentityConflict("dataset evidence schema is not registered")
        record_type = EXTERNAL_DATASET_EVIDENCE_RECORD_TYPES.get(reference.evidence_schema)
        if record_type is None:
            raise ArtifactIdentityConflict("dataset evidence schema has no closed decoder")
        root = self._roots_by_id.get(reference.storage_root_id)
        if root is None:
            raise ArtifactIdentityConflict("dataset evidence storage root is not registered")
        effective_maximum = min(
            maximum_bytes,
            self.registration.limits.max_input_bytes,
            self.registration.capability.resource_ceiling.source_scan_bytes // 2,
        )
        if effective_maximum <= 0:
            raise ArtifactIdentityConflict(
                "dataset evidence registration admits no stable two-pass input bytes"
            )

        # Retain the ordinary mount/containment diagnostic, but never open the
        # returned Path. Every component is opened relative to retained
        # descriptors with O_NOFOLLOW below.
        root.resolve(reference.relative_locator, for_write=False)
        descriptor = -1
        rebound_descriptor = -1
        try:
            descriptor = _secure_open_relative(
                root.contract.canonical_path,
                reference.relative_locator,
            )
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIdentityConflict("dataset evidence is not a regular file")
            payload = _read_descriptor_bounded(
                descriptor,
                maximum_bytes=effective_maximum,
            )
            after_read = os.fstat(descriptor)
            if _descriptor_identity(before) != _descriptor_identity(after_read):
                # Renaming an open file changes its ctime, so a locator swap can
                # first look like in-place mutation on the retained descriptor.
                # Rebind the registered locator before choosing the diagnostic;
                # either branch remains fail-closed.
                rebound_descriptor = _secure_open_relative(
                    root.contract.canonical_path,
                    reference.relative_locator,
                )
                rebound_after_read = os.fstat(rebound_descriptor)
                if _descriptor_identity(after_read) != _descriptor_identity(rebound_after_read):
                    raise ArtifactIdentityConflict("dataset evidence locator changed during replay")
                raise ArtifactIdentityConflict("dataset evidence changed during bounded read")
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if observed_sha256 != reference.evidence_sha256:
                raise ArtifactIdentityConflict("dataset evidence digest differs from reference")
            record = decode_canonical_bytes(
                payload,
                record_type,
                maximum_bytes=effective_maximum,
            )
            if record.canonical_bytes() != payload:
                raise ArtifactIdentityConflict("dataset evidence bytes are not exactly canonical")
            _require_verifier_authority_binding(record, self.registration)
            after_decode = os.fstat(descriptor)
            if _descriptor_identity(after_read) != _descriptor_identity(after_decode):
                raise ArtifactIdentityConflict("dataset evidence changed during canonical decode")
            replayed_payload = _read_descriptor_bounded(
                descriptor,
                maximum_bytes=effective_maximum,
            )
            after_replay = os.fstat(descriptor)
            if replayed_payload != payload or _descriptor_identity(
                after_decode
            ) != _descriptor_identity(after_replay):
                raise ArtifactIdentityConflict("dataset evidence changed during stable replay")
            rebound_descriptor = _secure_open_relative(
                root.contract.canonical_path,
                reference.relative_locator,
            )
            rebound = os.fstat(rebound_descriptor)
            if _descriptor_identity(after_replay) != _descriptor_identity(rebound):
                raise ArtifactIdentityConflict("dataset evidence locator changed during replay")
        except OSError as error:
            raise ArtifactIdentityConflict("dataset evidence guarded read failed") from error
        finally:
            if rebound_descriptor >= 0:
                os.close(rebound_descriptor)
            if descriptor >= 0:
                os.close(descriptor)

        receipt = record if isinstance(record, DatasetMaterializationVerificationReceipt) else None
        result = DatasetEvidenceVerification(
            request_fingerprint=request.fingerprint(),
            owner_kind=request.owner_kind,
            owner_id=request.owner_id,
            evidence_reference_fingerprint=reference.fingerprint(),
            evidence_id=reference.evidence_id,
            storage_root_id=reference.storage_root_id,
            relative_locator=reference.relative_locator,
            evidence_schema=reference.evidence_schema,
            declared_sha256=reference.evidence_sha256,
            observed_sha256=observed_sha256,
            observed_size_bytes=len(payload),
            complete_eof=True,
            trusted_root=True,
            locator_contained=True,
            read_only=True,
            schema_validated=True,
            materialization_subject_fingerprint=(
                request.materialization_subject.fingerprint()
                if request.materialization_subject is not None
                else None
            ),
            materialization_verification_receipt=receipt,
        )
        if not result.validates(request):
            raise ArtifactIdentityConflict(
                "dataset evidence record differs from its exact owner-bound request"
            )
        return result


__all__ = [
    "EXTERNAL_DATASET_EVIDENCE_RECORD_TYPES",
    "EXTERNAL_DATASET_EVIDENCE_SCHEMA_IDS",
    "ExternalDatasetEvidenceVerifier",
]
