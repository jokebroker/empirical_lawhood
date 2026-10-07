"""Runtime contracts for authorized registration of already-held datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import (
    DatasetOperationAuthorization,
    DatasetOperationPolicy,
    DatasetOperationRequest,
)
from empirical_lawhood.planning.dataset_manifests import DatasetRegistrationManifest
from empirical_lawhood.planning.datasets import (
    DatasetMaterialization,
    DatasetMaterializationVerificationObservations,
    DatasetMaterializationVerificationReceipt,
    DatasetMaterializationVerificationSubject,
    EvidenceReference,
    EvidenceReferenceKind,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.dataset_io import (
    DatasetDirectoryGuardReceipt,
    DatasetSourceGuardReceipt,
)
from empirical_lawhood.runtime.dataset_operations import DatasetOperationAuthorityBundle
from empirical_lawhood.runtime.datasets import DatasetCatalogSnapshot


@dataclass(frozen=True, slots=True)
class DatasetRegistrationSupportingRecord:
    """One process-local canonical record prepared for the publication batch."""

    record_id: str
    record: CanonicalRecord

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        if not isinstance(self.record, CanonicalRecord):
            raise TypeError("supporting record must be canonical")
        ObjectIdentity.from_record(self.record_id, self.record)


@dataclass(frozen=True, slots=True)
class PreparedDatasetRegistration:
    """Noncanonical in-process preparation; it contains no persistence claim."""

    inspection_id: str
    inspection_record: CanonicalRecord
    manifest: ObjectIdentity
    inspector: RegisteredImplementationIdentity
    verification_policy_id: str
    verification_policy_record: CanonicalRecord
    source_guard_receipts: tuple[
        DatasetSourceGuardReceipt | DatasetDirectoryGuardReceipt,
        ...,
    ]
    supporting_records: tuple[DatasetRegistrationSupportingRecord, ...]
    subject: DatasetMaterializationVerificationSubject
    observations: DatasetMaterializationVerificationObservations
    distinct_source_bytes: int
    source_full_hash_passes: int
    source_full_hash_read_ceiling_bytes: int
    network_bytes: int
    source_mutated: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        if not isinstance(self.inspection_record, CanonicalRecord):
            raise TypeError("inspection_record must be canonical")
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetRegistrationManifest.SCHEMA
        ):
            raise ValueError("manifest must identify a DatasetRegistrationManifest")
        if not isinstance(self.inspector, RegisteredImplementationIdentity):
            raise TypeError("inspector must be a RegisteredImplementationIdentity")
        validate_stable_id(self.verification_policy_id, field_name="verification_policy_id")
        if not isinstance(self.verification_policy_record, CanonicalRecord):
            raise TypeError("verification_policy_record must be canonical")
        if getattr(self.verification_policy_record, "policy_id", None) != (
            self.verification_policy_id
        ):
            raise ValueError("verification policy record has another identity")
        if not self.source_guard_receipts or any(
            not isinstance(value, (DatasetSourceGuardReceipt, DatasetDirectoryGuardReceipt))
            for value in self.source_guard_receipts
        ):
            raise ValueError("prepared registration requires completed source guards")
        require_sorted_unique_ids(
            self.source_guard_receipts,
            attribute="guard_receipt_id",
            field_name="source_guard_receipts",
        )
        if any(
            not isinstance(value, DatasetRegistrationSupportingRecord)
            for value in self.supporting_records
        ):
            raise TypeError("supporting_records contains another record type")
        require_sorted_unique_ids(
            self.supporting_records,
            attribute="record_id",
            field_name="supporting_records",
        )
        reserved_ids = {
            self.inspection_id,
            self.verification_policy_id,
            *(value.guard_receipt_id for value in self.source_guard_receipts),
        }
        if reserved_ids.intersection(value.record_id for value in self.supporting_records):
            raise ValueError("prepared registration record identities overlap")
        if not isinstance(self.subject, DatasetMaterializationVerificationSubject):
            raise TypeError("subject must be a DatasetMaterializationVerificationSubject")
        if not isinstance(self.observations, DatasetMaterializationVerificationObservations):
            raise TypeError("observations must be verification observations")
        if not self.observations.validates(self.subject):
            raise ValueError("prepared observations differ from their exact subject")
        for field_name, value in (
            ("distinct_source_bytes", self.distinct_source_bytes),
            ("source_full_hash_passes", self.source_full_hash_passes),
            (
                "source_full_hash_read_ceiling_bytes",
                self.source_full_hash_read_ceiling_bytes,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.source_full_hash_read_ceiling_bytes != (
            self.distinct_source_bytes * self.source_full_hash_passes
        ):
            raise ValueError("prepared full-hash read ceiling is inconsistent")
        if self.network_bytes != 0 or self.source_mutated is not False:
            raise ValueError("held registration preparation must be zero-network and read-only")


class DatasetRegistrationInspector(Protocol):
    def prepare_registration(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str,
    ) -> PreparedDatasetRegistration: ...


@dataclass(frozen=True, slots=True)
class DatasetRegistrationPublicationReceipt(CanonicalRecord):
    """External commit record for one exact held-source registration batch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-registration-publication-receipt'

    receipt_id: str
    authority_bundle_id: str
    manifest: ObjectIdentity
    policy: ObjectIdentity
    request: ObjectIdentity
    authorization: ObjectIdentity
    inspection_evidence: EvidenceReference
    verification_policy_evidence: EvidenceReference
    source_guard_evidence: tuple[EvidenceReference, ...]
    supporting_evidence: tuple[EvidenceReference, ...]
    manifest_evidence: EvidenceReference
    materialization_verification_evidence: EvidenceReference
    dataset_snapshot_evidence: EvidenceReference
    materialization: ObjectIdentity
    registered_at_utc: str
    distinct_source_bytes: int
    source_full_hash_passes: int
    source_full_hash_read_ceiling_bytes: int
    network_bytes: int
    source_mutated: bool
    download_performed: bool
    dataset_bytes_written: bool
    control_batch_published: bool
    catalog_written: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, text_value in (
            ("receipt_id", self.receipt_id),
            ("authority_bundle_id", self.authority_bundle_id),
        ):
            validate_stable_id(text_value, field_name=field_name)
        expected_schemas = (
            ("manifest", self.manifest, DatasetRegistrationManifest.SCHEMA),
            ("policy", self.policy, DatasetOperationPolicy.SCHEMA),
            ("request", self.request, DatasetOperationRequest.SCHEMA),
            ("authorization", self.authorization, DatasetOperationAuthorization.SCHEMA),
            ("materialization", self.materialization, DatasetMaterialization.SCHEMA),
        )
        for field_name, identity, expected_schema in expected_schemas:
            if not isinstance(identity, ObjectIdentity):
                raise TypeError(f"{field_name} must be an ObjectIdentity")
            if expected_schema is not None and identity.object_schema != expected_schema:
                raise ValueError(f"{field_name} has another object schema")
        for field_name, reference, expected_kind in (
            ("inspection_evidence", self.inspection_evidence, EvidenceReferenceKind.VERIFICATION),
            (
                "verification_policy_evidence",
                self.verification_policy_evidence,
                EvidenceReferenceKind.MANIFEST,
            ),
            ("manifest_evidence", self.manifest_evidence, EvidenceReferenceKind.MANIFEST),
            (
                "materialization_verification_evidence",
                self.materialization_verification_evidence,
                EvidenceReferenceKind.VERIFICATION,
            ),
            (
                "dataset_snapshot_evidence",
                self.dataset_snapshot_evidence,
                EvidenceReferenceKind.MANIFEST,
            ),
        ):
            if not isinstance(reference, EvidenceReference) or reference.kind is not expected_kind:
                raise ValueError(f"{field_name} has another evidence kind")
        if self.manifest_evidence.evidence_schema != DatasetRegistrationManifest.SCHEMA:
            raise ValueError("manifest_evidence has another schema")
        if (
            self.manifest_evidence.evidence_id != self.manifest.object_id
            or self.manifest_evidence.evidence_sha256 != self.manifest.object_fingerprint
        ):
            raise ValueError("manifest_evidence differs from the exact manifest identity")
        if (
            self.materialization_verification_evidence.evidence_schema
            != DatasetMaterializationVerificationReceipt.SCHEMA
        ):
            raise ValueError("materialization verification evidence has another schema")
        if self.dataset_snapshot_evidence.evidence_schema != DatasetCatalogSnapshot.SCHEMA:
            raise ValueError("dataset snapshot evidence has another schema")
        for field_name, values, expected_kind in (
            ("source_guard_evidence", self.source_guard_evidence, EvidenceReferenceKind.RECEIPT),
            ("supporting_evidence", self.supporting_evidence, EvidenceReferenceKind.RECEIPT),
        ):
            if any(
                not isinstance(value, EvidenceReference) or value.kind is not expected_kind
                for value in values
            ):
                raise ValueError(f"{field_name} contains another evidence kind")
            require_sorted_unique_ids(values, attribute="evidence_id", field_name=field_name)
        if not self.source_guard_evidence:
            raise ValueError("registration receipt requires source guard evidence")
        all_references = (
            self.inspection_evidence,
            self.verification_policy_evidence,
            *self.source_guard_evidence,
            *self.supporting_evidence,
            self.manifest_evidence,
            self.materialization_verification_evidence,
            self.dataset_snapshot_evidence,
        )
        reference_ids = tuple(value.evidence_id for value in all_references)
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("registration receipt evidence identities overlap")
        if len({value.storage_root_id for value in all_references}) != 1:
            raise ValueError("registration receipt evidence crosses storage roots")
        parse_utc_timestamp(self.registered_at_utc, field_name="registered_at_utc")
        for field_name, numeric_value in (
            ("distinct_source_bytes", self.distinct_source_bytes),
            ("source_full_hash_passes", self.source_full_hash_passes),
            (
                "source_full_hash_read_ceiling_bytes",
                self.source_full_hash_read_ceiling_bytes,
            ),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.source_full_hash_read_ceiling_bytes != (
            self.distinct_source_bytes * self.source_full_hash_passes
        ):
            raise ValueError("registration receipt full-hash read ceiling is inconsistent")
        if self.network_bytes != 0:
            raise ValueError("held-source registration must use zero network bytes")
        for field_name in (
            "source_mutated",
            "download_performed",
            "dataset_bytes_written",
            "catalog_written",
        ):
            if getattr(self, field_name) is not False:
                raise ValueError(f"held-source registration requires {field_name}=false")
        if self.control_batch_published is not True:
            raise ValueError("successful registration must publish its control batch")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetRegistrationInstallation(CanonicalRecord):
    """Trusted result returned after new publication or exact restart replay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-registration-installation'

    receipt: DatasetRegistrationPublicationReceipt
    receipt_evidence: EvidenceReference
    snapshot: DatasetCatalogSnapshot
    materialization: DatasetMaterialization
    replayed_existing: bool
    new_external_artifacts_created: bool
    catalog_written: bool
    network_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DatasetRegistrationPublicationReceipt):
            raise TypeError("receipt must be a DatasetRegistrationPublicationReceipt")
        if (
            not isinstance(self.receipt_evidence, EvidenceReference)
            or self.receipt_evidence.kind is not EvidenceReferenceKind.RECEIPT
            or self.receipt_evidence.evidence_id != self.receipt.receipt_id
            or self.receipt_evidence.evidence_schema != self.receipt.SCHEMA
            or self.receipt_evidence.evidence_sha256 != self.receipt.fingerprint()
        ):
            raise ValueError("receipt_evidence differs from the registration receipt")
        if not isinstance(self.snapshot, DatasetCatalogSnapshot):
            raise TypeError("snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(self.materialization, DatasetMaterialization):
            raise TypeError("materialization must be a DatasetMaterialization")
        matches = tuple(
            value
            for value in self.snapshot.materializations
            if value.materialization_id == self.materialization.materialization_id
        )
        if len(matches) != 1 or matches[0] != self.materialization:
            raise ValueError("installation materialization differs from its snapshot")
        if (
            ObjectIdentity.from_record(
                self.materialization.materialization_id,
                self.materialization,
            )
            != self.receipt.materialization
        ):
            raise ValueError("installation materialization differs from its receipt")
        if self.receipt.dataset_snapshot_evidence.evidence_sha256 != self.snapshot.fingerprint():
            raise ValueError("installation snapshot differs from its receipt")
        if not isinstance(self.replayed_existing, bool) or not isinstance(
            self.new_external_artifacts_created,
            bool,
        ):
            raise ValueError("installation replay/publication flags must be boolean")
        if self.replayed_existing and self.new_external_artifacts_created:
            raise ValueError("replayed registration cannot report newly created artifacts")
        if self.catalog_written is not False or self.network_bytes != 0:
            raise ValueError("registration installation must not write catalog or network bytes")


class DatasetRegistrationInstallerPort(Protocol):
    def install(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        manifest: DatasetRegistrationManifest,
        receipt_id: str,
    ) -> DatasetRegistrationInstallation: ...


__all__ = [
    "DatasetRegistrationInspector",
    "DatasetRegistrationInstallation",
    "DatasetRegistrationInstallerPort",
    "DatasetRegistrationPublicationReceipt",
    "DatasetRegistrationSupportingRecord",
    "PreparedDatasetRegistration",
]
