"""Runtime contracts for authorized publication of dataset transformations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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
from empirical_lawhood.planning.dataset_manifests import DatasetTransformationManifest
from empirical_lawhood.planning.datasets import (
    DatasetMaterialization,
    DatasetMaterializationVerificationSubject,
    EvidenceReference,
    EvidenceReferenceKind,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactMaterialization,
    ArtifactProfile,
    LogicalArtifactIdentity,
)
from empirical_lawhood.runtime.dataset_io import DatasetSourceGuardReceipt
from empirical_lawhood.runtime.dataset_operations import DatasetOperationAuthorityBundle
from empirical_lawhood.runtime.datasets import DatasetCatalogSnapshot


@dataclass(frozen=True, slots=True)
class DatasetTransformationSupportingRecord:
    """One canonical, typed supporting record prepared with an output."""

    record_id: str
    record: CanonicalRecord
    evidence_kind: EvidenceReferenceKind

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        if not isinstance(self.record, CanonicalRecord):
            raise TypeError("transformation supporting record must be canonical")
        if not isinstance(self.evidence_kind, EvidenceReferenceKind):
            raise TypeError("transformation supporting evidence kind is invalid")
        ObjectIdentity.from_record(self.record_id, self.record)


@dataclass(frozen=True, slots=True)
class PreparedDatasetTransformation:
    """Bounded process-local output; it contains no persistence claim."""

    manifest: ObjectIdentity
    transformer: RegisteredImplementationIdentity
    verification_policy_id: str
    verification_policy_record: CanonicalRecord
    output_id: str
    artifact_profile: ArtifactProfile
    payload: bytes
    subject: DatasetMaterializationVerificationSubject
    row_count: int
    column_count: int
    source_guard_receipts: tuple[DatasetSourceGuardReceipt, ...]
    supporting_records: tuple[DatasetTransformationSupportingRecord, ...]
    network_bytes: int
    source_mutated: bool
    scientific_verdict_issued: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetTransformationManifest.SCHEMA
        ):
            raise ValueError("manifest must identify a DatasetTransformationManifest")
        if not isinstance(self.transformer, RegisteredImplementationIdentity):
            raise TypeError("transformer must be a RegisteredImplementationIdentity")
        validate_stable_id(self.verification_policy_id, field_name="verification_policy_id")
        if not isinstance(self.verification_policy_record, CanonicalRecord):
            raise TypeError("verification policy record must be canonical")
        if getattr(self.verification_policy_record, "policy_id", None) != (
            self.verification_policy_id
        ):
            raise ValueError("verification policy record has another identity")
        validate_stable_id(self.output_id, field_name="output_id")
        if not isinstance(self.artifact_profile, ArtifactProfile):
            raise TypeError("artifact_profile must be an ArtifactProfile")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("prepared transformation payload must be nonempty immutable bytes")
        if not isinstance(self.subject, DatasetMaterializationVerificationSubject):
            raise TypeError("subject must be a DatasetMaterializationVerificationSubject")
        if (
            self.subject.physical_sha256 != sha256(self.payload).hexdigest()
            or self.subject.byte_size != len(self.payload)
            or self.subject.file_count != 1
            or self.subject.logical_identity is None
        ):
            raise ValueError("prepared payload differs from its exact verification subject")
        for field_name, numeric_value in (
            ("row_count", self.row_count),
            ("column_count", self.column_count),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if not self.source_guard_receipts or any(
            not isinstance(value, DatasetSourceGuardReceipt) for value in self.source_guard_receipts
        ):
            raise ValueError("prepared transformation requires completed source guards")
        require_sorted_unique_ids(
            self.source_guard_receipts,
            attribute="guard_receipt_id",
            field_name="source_guard_receipts",
        )
        if any(
            not isinstance(value, DatasetTransformationSupportingRecord)
            for value in self.supporting_records
        ):
            raise TypeError("supporting_records contains another record type")
        require_sorted_unique_ids(
            self.supporting_records,
            attribute="record_id",
            field_name="supporting_records",
        )
        reserved = {
            self.verification_policy_id,
            *(value.guard_receipt_id for value in self.source_guard_receipts),
        }
        if reserved.intersection(value.record_id for value in self.supporting_records):
            raise ValueError("prepared transformation evidence identities overlap")
        if self.network_bytes != 0 or self.source_mutated is not False:
            raise ValueError("held-source transformation must be zero-network and read-only")
        if self.scientific_verdict_issued is not False:
            raise ValueError("dataset transformation cannot issue a scientific verdict")


class DatasetTransformationPreparer(Protocol):
    def prepare_transformation(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str,
    ) -> PreparedDatasetTransformation: ...


@dataclass(frozen=True, slots=True)
class DatasetTransformationPublicationReceipt(CanonicalRecord):
    """Commit record for one exact derivative and its control closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-transformation-publication-receipt'

    receipt_id: str
    authority_bundle_id: str
    manifest: ObjectIdentity
    policy: ObjectIdentity
    request: ObjectIdentity
    authorization: ObjectIdentity
    output_logical: LogicalArtifactIdentity
    output_artifact: ArtifactMaterialization
    manifest_evidence: EvidenceReference
    verification_policy_evidence: EvidenceReference
    source_guard_evidence: tuple[EvidenceReference, ...]
    supporting_evidence: tuple[EvidenceReference, ...]
    materialization_verification_evidence: EvidenceReference
    dataset_snapshot_evidence: EvidenceReference
    materialization: ObjectIdentity
    parent_materializations: tuple[ObjectIdentity, ...]
    transformed_at_utc: str
    row_count: int
    column_count: int
    distinct_source_bytes: int
    output_bytes_written: int
    network_bytes: int
    source_mutated: bool
    download_performed: bool
    scientific_verdict_issued: bool
    data_artifact_published: bool
    control_batch_published: bool
    catalog_written: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, stable_id in (
            ("receipt_id", self.receipt_id),
            ("authority_bundle_id", self.authority_bundle_id),
        ):
            validate_stable_id(stable_id, field_name=field_name)
        expected = (
            ("manifest", self.manifest, DatasetTransformationManifest.SCHEMA),
            ("policy", self.policy, DatasetOperationPolicy.SCHEMA),
            ("request", self.request, DatasetOperationRequest.SCHEMA),
            ("authorization", self.authorization, DatasetOperationAuthorization.SCHEMA),
            ("materialization", self.materialization, DatasetMaterialization.SCHEMA),
        )
        for field_name, identity, schema in expected:
            if not isinstance(identity, ObjectIdentity) or identity.object_schema != schema:
                raise ValueError(f"{field_name} has another object schema")
        if not isinstance(self.output_logical, LogicalArtifactIdentity):
            raise TypeError("output_logical must be a LogicalArtifactIdentity")
        if not isinstance(self.output_artifact, ArtifactMaterialization):
            raise TypeError("output_artifact must be an ArtifactMaterialization")
        if self.output_artifact.logical_artifact_id != self.output_logical.logical_artifact_id:
            raise ValueError("output logical and physical artifact identities differ")
        if self.output_logical.content_sha256 != self.output_artifact.physical_sha256:
            raise ValueError("output artifact byte identity differs from its materialization")
        references = (
            (self.manifest_evidence, EvidenceReferenceKind.MANIFEST),
            (self.verification_policy_evidence, EvidenceReferenceKind.MANIFEST),
            (self.materialization_verification_evidence, EvidenceReferenceKind.VERIFICATION),
            (self.dataset_snapshot_evidence, EvidenceReferenceKind.MANIFEST),
        )
        for reference, kind in references:
            if not isinstance(reference, EvidenceReference) or reference.kind is not kind:
                raise ValueError("transformation receipt evidence kind differs")
        if self.manifest_evidence.evidence_schema != DatasetTransformationManifest.SCHEMA:
            raise ValueError("manifest evidence has another schema")
        if (
            self.manifest_evidence.evidence_id != self.manifest.object_id
            or self.manifest_evidence.evidence_sha256 != self.manifest.object_fingerprint
        ):
            raise ValueError("manifest evidence differs from the exact manifest")
        for field_name, values in (
            ("source_guard_evidence", self.source_guard_evidence),
            ("supporting_evidence", self.supporting_evidence),
        ):
            if any(not isinstance(value, EvidenceReference) for value in values):
                raise ValueError(f"{field_name} contains another record type")
            require_sorted_unique_ids(values, attribute="evidence_id", field_name=field_name)
        if not self.source_guard_evidence:
            raise ValueError("transformation receipt requires source guard evidence")
        if any(
            value.kind is not EvidenceReferenceKind.RECEIPT for value in self.source_guard_evidence
        ):
            raise ValueError("transformation source guards must be receipt evidence")
        all_references = (
            self.manifest_evidence,
            self.verification_policy_evidence,
            *self.source_guard_evidence,
            *self.supporting_evidence,
            self.materialization_verification_evidence,
            self.dataset_snapshot_evidence,
        )
        if len({value.evidence_id for value in all_references}) != len(all_references):
            raise ValueError("transformation receipt evidence identities overlap")
        if len({value.storage_root_id for value in all_references}) != 1:
            raise ValueError("transformation receipt evidence crosses storage roots")
        require_sorted_unique_ids(
            self.parent_materializations,
            attribute="object_id",
            field_name="parent_materializations",
        )
        if not self.parent_materializations or any(
            value.object_schema != DatasetMaterialization.SCHEMA
            for value in self.parent_materializations
        ):
            raise ValueError("transformation receipt requires exact materialization parents")
        parse_utc_timestamp(self.transformed_at_utc, field_name="transformed_at_utc")
        for field_name, numeric_value in (
            ("row_count", self.row_count),
            ("column_count", self.column_count),
            ("distinct_source_bytes", self.distinct_source_bytes),
            ("output_bytes_written", self.output_bytes_written),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.output_bytes_written != self.output_artifact.size_bytes:
            raise ValueError("output byte accounting differs from the artifact")
        if self.network_bytes != 0:
            raise ValueError("held-source transformation must use zero network bytes")
        for field_name in (
            "source_mutated",
            "download_performed",
            "scientific_verdict_issued",
            "catalog_written",
        ):
            if getattr(self, field_name) is not False:
                raise ValueError(f"transformation requires {field_name}=false")
        if self.data_artifact_published is not True or self.control_batch_published is not True:
            raise ValueError("successful transformation must publish data and control evidence")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetTransformationInstallation(CanonicalRecord):
    """Trusted result returned after publication or exact restart replay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-transformation-installation'

    receipt: DatasetTransformationPublicationReceipt
    receipt_evidence: EvidenceReference
    snapshot: DatasetCatalogSnapshot
    materialization: DatasetMaterialization
    replayed_existing: bool
    new_external_artifacts_created: bool
    catalog_written: bool
    network_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DatasetTransformationPublicationReceipt):
            raise TypeError("receipt must be a DatasetTransformationPublicationReceipt")
        if (
            not isinstance(self.receipt_evidence, EvidenceReference)
            or self.receipt_evidence.kind is not EvidenceReferenceKind.RECEIPT
            or self.receipt_evidence.evidence_id != self.receipt.receipt_id
            or self.receipt_evidence.evidence_schema != self.receipt.SCHEMA
            or self.receipt_evidence.evidence_sha256 != self.receipt.fingerprint()
        ):
            raise ValueError("receipt evidence differs from the transformation receipt")
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
            raise ValueError("replayed transformation cannot create new artifacts")
        if self.catalog_written is not False or self.network_bytes != 0:
            raise ValueError("transformation installation must not write catalog or network")


class DatasetTransformationInstallerPort(Protocol):
    def install(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        manifest: DatasetTransformationManifest,
        base_snapshot: DatasetCatalogSnapshot,
        receipt_id: str,
    ) -> DatasetTransformationInstallation: ...


__all__ = [
    "DatasetTransformationInstallation",
    "DatasetTransformationInstallerPort",
    "DatasetTransformationPreparer",
    "DatasetTransformationPublicationReceipt",
    "DatasetTransformationSupportingRecord",
    "PreparedDatasetTransformation",
]
