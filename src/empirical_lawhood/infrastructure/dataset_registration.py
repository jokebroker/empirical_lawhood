"""Authorized atomic publication of already-held dataset registrations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final

from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError,
    MAX_ARTIFACT_MANIFEST_BYTES,
    read_bounded_bytes,
)
from empirical_lawhood.infrastructure.dataset_catalog_authority import (
    DatasetCatalogAuthorityError,
    _load_exact_reference,
)
from empirical_lawhood.infrastructure.dataset_operations import (
    ExternalDatasetOperationAuthorityStore,
)
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.dataset_authority import DatasetDecisionClock
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityKind,
    DatasetRegistrationManifest,
)
from empirical_lawhood.planning.datasets import (
    CustodyState,
    DatasetFamily,
    DatasetMaterialization,
    DatasetMaterializationVerificationReceipt,
    DatasetRelease,
    EvidenceReference,
    EvidenceReferenceKind,
    IdentityState,
    ReleaseResolutionClass,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.dataset_operations import (
    DatasetOperationAuthorityBundle,
    DatasetOperationPreviewService,
    DatasetOperationPreviewState,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationInstallation,
    DatasetRegistrationInspector,
    DatasetRegistrationPublicationReceipt,
    PreparedDatasetRegistration,
)
from empirical_lawhood.runtime.datasets import DatasetCatalogSnapshot


_PUBLICATION_SCOPE_ID: Final = "dataset-registration-publication"
_MAX_REGISTRATION_RECORD_BYTES: Final = 10_000_000


def _relative_path(prefix: str, category: str, record_id: str) -> str:
    validate_stable_id(record_id, field_name="record_id")
    return PurePosixPath(prefix, category, f"{record_id}.json").as_posix()


def _reference(
    *,
    record_id: str,
    record: CanonicalRecord,
    kind: EvidenceReferenceKind,
    storage_root_id: str,
    relative_locator: str,
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=record_id,
        kind=kind,
        evidence_schema=record.SCHEMA,
        evidence_sha256=record.fingerprint(),
        storage_root_id=storage_root_id,
        relative_locator=relative_locator,
    )


def _lineage_parent(
    identity: ObjectIdentity,
    *,
    visibility_ceiling: VisibilityCeiling,
    outcome_access: OutcomeAccess,
) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=identity,
        visibility_ceiling=visibility_ceiling,
        outcome_access=outcome_access,
    )


def _parents(
    values: tuple[ArtifactLineageParent, ...],
) -> tuple[ArtifactLineageParent, ...]:
    return tuple(sorted(values, key=lineage_parent_sort_key))


def _release_identity_state(resolution: ReleaseResolutionClass) -> IdentityState:
    return (
        IdentityState.UNRESOLVED
        if resolution is ReleaseResolutionClass.UNRESOLVED_LOCAL_CUSTODY
        else IdentityState.RELEASE_RESOLVED
    )


def _catalog_records(
    *,
    manifest: DatasetRegistrationManifest,
    manifest_evidence: EvidenceReference,
    verification_receipt: DatasetMaterializationVerificationReceipt,
    verification_evidence: EvidenceReference,
) -> tuple[DatasetCatalogSnapshot, DatasetMaterialization]:
    family_input = manifest.family
    release_input = manifest.release
    reason_codes = tuple(value.value for value in manifest.exception_codes)
    family = DatasetFamily(
        family_id=family_input.family_id,
        canonical_name=family_input.canonical_name,
        provider_id=family_input.provider_id,
        external_identifiers=family_input.external_identifiers,
        description=family_input.description,
        keywords=family_input.keywords,
        first_observation_id=None,
        latest_observation_id=None,
        identity_state=IdentityState.FAMILY_RESOLVED,
        reason_codes=(),
    )
    release = DatasetRelease(
        release_id=release_input.release_id,
        family_id=release_input.family_id,
        identity_state=_release_identity_state(release_input.expected_resolution_class),
        resolution_class=release_input.expected_resolution_class,
        access_state=release_input.expected_access_state,
        provider_id=release_input.provider_id,
        external_identifiers=release_input.external_identifiers,
        local_selector=manifest.selector,
        publication_at_utc=release_input.expected_publication_at_utc,
        observed_at_utc=release_input.expected_retrieved_at_utc,
        mutable_snapshot=release_input.expected_mutable_snapshot,
        expected_format_profile_ids=release_input.expected_format_profile_ids,
        expected_file_count_minimum=release_input.expected_file_count_minimum,
        expected_file_count_maximum=release_input.expected_file_count_maximum,
        expected_byte_count_minimum=release_input.expected_byte_count_minimum,
        expected_byte_count_maximum=release_input.expected_byte_count_maximum,
        expected_physical_sha256=release_input.expected_physical_sha256,
        manifest_sha256=(release_input.expected_manifest_sha256 or manifest.fingerprint()),
        licence_evidence=None,
        access_evidence=None,
        evidence_refs=(manifest_evidence,),
        reason_codes=reason_codes,
    )
    subject = verification_receipt.subject
    materialization = DatasetMaterialization(
        materialization_id=subject.materialization_id,
        release_id=subject.release_id,
        materialization_class=subject.materialization_class,
        evidence_class=subject.evidence_class,
        outcome_access=subject.outcome_access,
        storage_root_id=subject.storage_root_id,
        relative_locator=subject.relative_locator,
        selector=subject.selector,
        physical_sha256=subject.physical_sha256,
        byte_size=subject.byte_size,
        file_count=subject.file_count,
        media_type=subject.media_type,
        format_profile_id=subject.format_profile_id,
        logical_identity=subject.logical_identity,
        custody_state=CustodyState.VERIFIED,
        verifier=verification_receipt.verifier,
        verification_policy_id=verification_receipt.verification_policy_id,
        verification_policy_sha256=verification_receipt.verification_policy_sha256,
        verified_at_utc=verification_receipt.verified_at_utc,
        verification_evidence_refs=tuple(
            sorted(
                (manifest_evidence, verification_evidence),
                key=lambda value: value.evidence_id,
            )
        ),
        manifest_relative_locator=manifest_evidence.relative_locator,
        receipt_relative_locator=verification_evidence.relative_locator,
        reason_codes=(),
    )
    snapshot = DatasetCatalogSnapshot(
        families=(family,),
        releases=(release,),
        observations=(),
        materializations=(materialization,),
        acquisition_attempts=(),
        bindings=(),
    )
    return snapshot, materialization


def _artifact_request(
    *,
    record_id: str,
    record: CanonicalRecord,
    relative_path: str,
    publication_root: str,
    visibility_ceiling: VisibilityCeiling,
    outcome_access: OutcomeAccess,
    lineage_parents: tuple[ArtifactLineageParent, ...],
) -> ArtifactWriteRequest:
    parents = _parents(lineage_parents)
    return ArtifactWriteRequest(
        logical_artifact_id=record_id,
        relative_path=relative_path,
        payload_schema=record.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        publication_scope_id=_PUBLICATION_SCOPE_ID,
        publication_scope_relative_root=publication_root,
        payload=record.canonical_bytes(),
        visibility_ceiling=visibility_ceiling,
        parent_visibility_ceilings=tuple(value.visibility_ceiling for value in parents),
        outcome_access=outcome_access,
        logical_content_sha256=record.fingerprint(),
        lineage_parents=parents,
    )


class ExternalDatasetRegistrationPublisher:
    """Replay authority, inspect exact held bytes, and publish one control batch."""

    def __init__(
        self,
        *,
        artifact_plane: ExternalArtifactPlane,
        authority_store: ExternalDatasetOperationAuthorityStore,
        preview_service: DatasetOperationPreviewService,
        clock: DatasetDecisionClock,
        inspector: DatasetRegistrationInspector,
    ) -> None:
        self._artifact_plane = artifact_plane
        self._authority_store = authority_store
        self._preview_service = preview_service
        self._clock = clock
        self._inspector = inspector
        root = artifact_plane.root
        self._roots_by_id: Mapping[str, GuardedExternalRoot] = MappingProxyType(
            {root.contract.storage_root_id: root}
        )

    def install(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        manifest: DatasetRegistrationManifest,
        receipt_id: str,
    ) -> DatasetRegistrationInstallation:
        if not isinstance(bundle, DatasetOperationAuthorityBundle):
            raise TypeError("bundle must be a DatasetOperationAuthorityBundle")
        if not isinstance(manifest, DatasetRegistrationManifest):
            raise TypeError("manifest must be a DatasetRegistrationManifest")
        validate_stable_id(receipt_id, field_name="receipt_id")
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        if (
            bundle.request.action is not AuthorityAction.DATASET_REGISTRATION
            or bundle.request.manifest != manifest_identity
            or bundle.policy.manifest != manifest_identity
        ):
            raise ValueError("registration publisher authority binds another operation")
        durable = self._authority_store.load_bundle(
            bundle_id=bundle.bundle_id,
            policy_id=bundle.policy.policy_id,
            request_id=bundle.request.request_id,
            authorization_id=bundle.authorization.decision.authorization_id,
            clock=self._clock,
        )
        if durable != bundle:
            raise ValueError("durable registration authority differs from the requested bundle")
        preview = self._preview_service.preview(
            manifest=manifest,
            policy=bundle.policy,
            request=bundle.request,
            authorization_id=bundle.authorization.decision.authorization_id,
            authorization=bundle.authorization,
        )
        if preview.state is not DatasetOperationPreviewState.READY:
            raise PermissionError("dataset registration did not pass its exact preview")
        root_identity = ObjectIdentity.from_record(
            self._artifact_plane.root.contract.storage_root_id,
            self._artifact_plane.root.contract,
        )
        if (
            manifest.control_write_scope.storage_root != root_identity
            or manifest.source_scope.storage_root != root_identity
        ):
            raise ValueError("registration source/control scope differs from its artifact plane")

        existing = self._load_existing(
            receipt_id=receipt_id,
            manifest=manifest,
            bundle=bundle,
        )
        if existing is not None:
            return existing

        prepared = self._inspector.prepare_registration(
            manifest_identity,
            trusted_at_utc=self._clock.now_utc(),
        )
        self._validate_prepared(prepared, manifest=manifest)
        return self._publish(
            bundle=bundle,
            manifest=manifest,
            prepared=prepared,
            receipt_id=receipt_id,
        )

    def _validate_prepared(
        self,
        prepared: PreparedDatasetRegistration,
        *,
        manifest: DatasetRegistrationManifest,
    ) -> None:
        if not isinstance(prepared, PreparedDatasetRegistration):
            raise TypeError("registration inspector returned another preparation type")
        if prepared.manifest != ObjectIdentity.from_record(manifest.manifest_id, manifest):
            raise ValueError("registration preparation binds another manifest")
        inspector_bindings = tuple(
            value
            for value in manifest.capabilities
            if value.kind is DatasetCapabilityKind.INSPECTOR
        )
        if (
            len(inspector_bindings) != 1
            or prepared.inspector.implementation_key != inspector_bindings[0].registry_key
            or inspector_bindings[0].implementation.object_id
            != prepared.inspector.implementation_key
        ):
            raise ValueError("registration preparation used another inspector capability")
        subject = prepared.subject
        if (
            subject.materialization_id != manifest.proposed_materialization_id
            or subject.release_id != manifest.release.release_id
            or subject.materialization_class is not manifest.materialization_class
            or subject.evidence_class is not manifest.evidence_class
            or subject.outcome_access is not manifest.outcome_access
            or subject.storage_root_id != manifest.source_scope.storage_root.object_id
            or subject.relative_locator != manifest.source_scope.relative_prefix
            or subject.selector != manifest.selector
            or subject.physical_sha256 != manifest.expected_physical_sha256
            or subject.byte_size != manifest.expected_byte_size
            or subject.file_count != manifest.expected_file_count
            or subject.media_type != manifest.expected_media_type
            or subject.format_profile_id not in manifest.expected_format_profile_ids
            or subject.logical_identity is not None
        ):
            raise ValueError("registration preparation differs from the manifest subject")
        if (
            sum(receipt.observed_file_count for receipt in prepared.source_guard_receipts)
            != manifest.expected_file_count
        ):
            raise ValueError("registration preparation guard count differs from the manifest")
        if any(
            receipt.source_scope != manifest.source_scope
            or receipt.storage_root != manifest.source_scope.storage_root
            for receipt in prepared.source_guard_receipts
        ):
            raise ValueError("registration preparation guards another source scope")
        if sum(value.observed_size_bytes for value in prepared.source_guard_receipts) != (
            prepared.distinct_source_bytes
        ):
            raise ValueError("registration preparation distinct-byte accounting differs")
        if prepared.distinct_source_bytes != manifest.expected_byte_size:
            raise ValueError("registration preparation scanned another byte envelope")
        verified_times = {value.verified_at_utc for value in prepared.source_guard_receipts}
        if len(verified_times) != 1:
            raise ValueError("registration preparation guard times differ")

    def _publish(
        self,
        *,
        bundle: DatasetOperationAuthorityBundle,
        manifest: DatasetRegistrationManifest,
        prepared: PreparedDatasetRegistration,
        receipt_id: str,
    ) -> DatasetRegistrationInstallation:
        storage_root_id = manifest.control_write_scope.storage_root.object_id
        prefix = manifest.control_write_scope.relative_prefix
        manifest_path = _relative_path(prefix, "manifests", manifest.manifest_id)
        inspection_path = _relative_path(prefix, "inspections", prepared.inspection_id)
        policy_path = _relative_path(
            prefix,
            "verification-policies",
            prepared.verification_policy_id,
        )
        guard_paths = {
            value.guard_receipt_id: _relative_path(
                prefix,
                "source-guards",
                value.guard_receipt_id,
            )
            for value in prepared.source_guard_receipts
        }
        supporting_paths = {
            value.record_id: _relative_path(prefix, "supporting-evidence", value.record_id)
            for value in prepared.supporting_records
        }
        verification_receipt_id = f"verification.{prepared.subject.materialization_id}"
        verification_path = _relative_path(
            prefix,
            "materialization-verifications",
            verification_receipt_id,
        )
        snapshot_id = f"snapshot.{manifest.manifest_id}"
        snapshot_path = _relative_path(prefix, "dataset-snapshots", snapshot_id)
        publication_receipt_path = _relative_path(
            prefix,
            "registration-receipts",
            receipt_id,
        )

        manifest_evidence = _reference(
            record_id=manifest.manifest_id,
            record=manifest,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=manifest_path,
        )
        verification_receipt = DatasetMaterializationVerificationReceipt(
            receipt_id=verification_receipt_id,
            subject=prepared.subject,
            subject_fingerprint=prepared.subject.fingerprint(),
            verifier=prepared.inspector,
            verification_policy_id=prepared.verification_policy_id,
            verification_policy_sha256=prepared.verification_policy_record.fingerprint(),
            verified_at_utc=prepared.source_guard_receipts[0].verified_at_utc,
            manifest_evidence=manifest_evidence,
            observations=prepared.observations,
        )
        verification_evidence = _reference(
            record_id=verification_receipt_id,
            record=verification_receipt,
            kind=EvidenceReferenceKind.VERIFICATION,
            storage_root_id=storage_root_id,
            relative_locator=verification_path,
        )
        snapshot, materialization = _catalog_records(
            manifest=manifest,
            manifest_evidence=manifest_evidence,
            verification_receipt=verification_receipt,
            verification_evidence=verification_evidence,
        )
        snapshot_evidence = _reference(
            record_id=snapshot_id,
            record=snapshot,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=snapshot_path,
        )
        inspection_evidence = _reference(
            record_id=prepared.inspection_id,
            record=prepared.inspection_record,
            kind=EvidenceReferenceKind.VERIFICATION,
            storage_root_id=storage_root_id,
            relative_locator=inspection_path,
        )
        policy_evidence = _reference(
            record_id=prepared.verification_policy_id,
            record=prepared.verification_policy_record,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=policy_path,
        )
        guard_evidence = tuple(
            sorted(
                (
                    _reference(
                        record_id=value.guard_receipt_id,
                        record=value,
                        kind=EvidenceReferenceKind.RECEIPT,
                        storage_root_id=storage_root_id,
                        relative_locator=guard_paths[value.guard_receipt_id],
                    )
                    for value in prepared.source_guard_receipts
                ),
                key=lambda value: value.evidence_id,
            )
        )
        supporting_evidence = tuple(
            sorted(
                (
                    _reference(
                        record_id=value.record_id,
                        record=value.record,
                        kind=EvidenceReferenceKind.RECEIPT,
                        storage_root_id=storage_root_id,
                        relative_locator=supporting_paths[value.record_id],
                    )
                    for value in prepared.supporting_records
                ),
                key=lambda value: value.evidence_id,
            )
        )
        policy_identity = ObjectIdentity.from_record(bundle.policy.policy_id, bundle.policy)
        request_identity = ObjectIdentity.from_record(bundle.request.request_id, bundle.request)
        authorization_identity = ObjectIdentity.from_record(
            bundle.authorization.decision.authorization_id,
            bundle.authorization,
        )
        materialization_identity = ObjectIdentity.from_record(
            materialization.materialization_id,
            materialization,
        )
        publication_receipt = DatasetRegistrationPublicationReceipt(
            receipt_id=receipt_id,
            authority_bundle_id=bundle.bundle_id,
            manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
            policy=policy_identity,
            request=request_identity,
            authorization=authorization_identity,
            inspection_evidence=inspection_evidence,
            verification_policy_evidence=policy_evidence,
            source_guard_evidence=guard_evidence,
            supporting_evidence=supporting_evidence,
            manifest_evidence=manifest_evidence,
            materialization_verification_evidence=verification_evidence,
            dataset_snapshot_evidence=snapshot_evidence,
            materialization=materialization_identity,
            registered_at_utc=verification_receipt.verified_at_utc,
            distinct_source_bytes=prepared.distinct_source_bytes,
            source_full_hash_passes=prepared.source_full_hash_passes,
            source_full_hash_read_ceiling_bytes=(prepared.source_full_hash_read_ceiling_bytes),
            network_bytes=0,
            source_mutated=False,
            download_performed=False,
            dataset_bytes_written=False,
            control_batch_published=True,
            catalog_written=False,
            reason_codes=tuple(value.value for value in manifest.exception_codes),
        )
        publication_receipt_evidence = _reference(
            record_id=receipt_id,
            record=publication_receipt,
            kind=EvidenceReferenceKind.RECEIPT,
            storage_root_id=storage_root_id,
            relative_locator=publication_receipt_path,
        )

        output_visibility = manifest.visibility_ceiling
        output_access = manifest.outcome_access
        authority_parents = _parents(
            (
                _lineage_parent(
                    policy_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
                _lineage_parent(
                    request_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
                _lineage_parent(
                    authorization_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
            )
        )

        def output_parent(record_id: str, record: CanonicalRecord) -> ArtifactLineageParent:
            return _lineage_parent(
                ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=output_visibility,
                outcome_access=output_access,
            )

        manifest_parent = output_parent(manifest.manifest_id, manifest)
        policy_parent = output_parent(
            prepared.verification_policy_id,
            prepared.verification_policy_record,
        )
        guard_parents = tuple(
            output_parent(value.guard_receipt_id, value) for value in prepared.source_guard_receipts
        )
        supporting_parents = tuple(
            output_parent(value.record_id, value.record) for value in prepared.supporting_records
        )
        inspection_parent = output_parent(prepared.inspection_id, prepared.inspection_record)
        verification_parent = output_parent(verification_receipt_id, verification_receipt)
        snapshot_parent = output_parent(snapshot_id, snapshot)
        records: tuple[tuple[str, CanonicalRecord, str, tuple[ArtifactLineageParent, ...]], ...] = (
            (manifest.manifest_id, manifest, manifest_path, authority_parents),
            (
                prepared.verification_policy_id,
                prepared.verification_policy_record,
                policy_path,
                (*authority_parents, manifest_parent),
            ),
            *tuple(
                (
                    value.guard_receipt_id,
                    value,
                    guard_paths[value.guard_receipt_id],
                    (*authority_parents, manifest_parent),
                )
                for value in prepared.source_guard_receipts
            ),
            *tuple(
                (
                    value.record_id,
                    value.record,
                    supporting_paths[value.record_id],
                    (*authority_parents, manifest_parent, *guard_parents),
                )
                for value in prepared.supporting_records
            ),
            (
                prepared.inspection_id,
                prepared.inspection_record,
                inspection_path,
                (
                    *authority_parents,
                    manifest_parent,
                    policy_parent,
                    *guard_parents,
                    *supporting_parents,
                ),
            ),
            (
                verification_receipt_id,
                verification_receipt,
                verification_path,
                (
                    *authority_parents,
                    manifest_parent,
                    policy_parent,
                    *guard_parents,
                    *supporting_parents,
                    inspection_parent,
                ),
            ),
            (
                snapshot_id,
                snapshot,
                snapshot_path,
                (*authority_parents, manifest_parent, verification_parent),
            ),
            (
                receipt_id,
                publication_receipt,
                publication_receipt_path,
                (
                    *authority_parents,
                    manifest_parent,
                    policy_parent,
                    *guard_parents,
                    *supporting_parents,
                    inspection_parent,
                    verification_parent,
                    snapshot_parent,
                ),
            ),
        )
        if len(records) > manifest.work_envelope.control.max_metadata_records:
            raise ValueError("registration publication exceeds its metadata-record ceiling")
        for record_id, record, _path, _lineage in records:
            maximum = (
                manifest.work_envelope.control.max_manifest_bytes
                if record_id == manifest.manifest_id
                else manifest.work_envelope.control.max_receipt_bytes
            )
            if len(record.canonical_bytes()) > min(maximum, _MAX_REGISTRATION_RECORD_BYTES):
                raise ValueError("registration publication record exceeds its byte ceiling")
        requests = tuple(
            _artifact_request(
                record_id=record_id,
                record=record,
                relative_path=path,
                publication_root=prefix,
                visibility_ceiling=output_visibility,
                outcome_access=output_access,
                lineage_parents=lineage,
            )
            for record_id, record, path, lineage in records
        )
        results = self._artifact_plane.write_batch(requests)
        return DatasetRegistrationInstallation(
            receipt=publication_receipt,
            receipt_evidence=publication_receipt_evidence,
            snapshot=snapshot,
            materialization=materialization,
            replayed_existing=False,
            new_external_artifacts_created=any(value.created for value in results),
            catalog_written=False,
            network_bytes=0,
        )

    def _load_existing(
        self,
        *,
        receipt_id: str,
        manifest: DatasetRegistrationManifest,
        bundle: DatasetOperationAuthorityBundle,
    ) -> DatasetRegistrationInstallation | None:
        receipt_path = _relative_path(
            manifest.control_write_scope.relative_prefix,
            "registration-receipts",
            receipt_id,
        )
        payload_path = self._artifact_plane.root.resolve(receipt_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{receipt_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.exists() and not sidecar_path.exists():
            return None
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise ValueError("registration receipt publication is incomplete")
        try:
            sidecar = decode_artifact_manifest(
                read_bounded_bytes(
                    sidecar_path,
                    maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
                )
            )
            if (
                sidecar.logical.logical_artifact_id != receipt_id
                or sidecar.logical.payload_schema != DatasetRegistrationPublicationReceipt.SCHEMA
                or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
                or sidecar.materialization.relative_path != receipt_path
                or sidecar.materialization.size_bytes > _MAX_REGISTRATION_RECORD_BYTES
            ):
                raise ValueError("registration receipt artifact manifest differs")
            self._artifact_plane.verify_manifest(sidecar)
            payload = read_bounded_bytes(
                payload_path,
                maximum_bytes=_MAX_REGISTRATION_RECORD_BYTES,
            )
            receipt = decode_canonical_bytes(
                payload,
                DatasetRegistrationPublicationReceipt,
                maximum_bytes=_MAX_REGISTRATION_RECORD_BYTES,
            )
        except (
            ArtifactIdentityConflict,
            BoundedFileIOError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError("published registration receipt cannot be replayed") from error
        if (
            receipt.receipt_id != receipt_id
            or receipt.canonical_bytes() != payload
            or receipt.fingerprint() != sidecar.logical.content_sha256
            or receipt.authority_bundle_id != bundle.bundle_id
            or receipt.manifest != ObjectIdentity.from_record(manifest.manifest_id, manifest)
            or receipt.policy != ObjectIdentity.from_record(bundle.policy.policy_id, bundle.policy)
            or receipt.request
            != ObjectIdentity.from_record(
                bundle.request.request_id,
                bundle.request,
            )
            or receipt.authorization
            != ObjectIdentity.from_record(
                bundle.authorization.decision.authorization_id,
                bundle.authorization,
            )
        ):
            raise ValueError("published registration receipt binds another operation")
        try:
            snapshot = _load_exact_reference(
                receipt.dataset_snapshot_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.MANIFEST,
                record_type=DatasetCatalogSnapshot,
                maximum_bytes=_MAX_REGISTRATION_RECORD_BYTES,
                label="registration dataset snapshot",
            )
            verification_receipt = _load_exact_reference(
                receipt.materialization_verification_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.VERIFICATION,
                record_type=DatasetMaterializationVerificationReceipt,
                maximum_bytes=_MAX_REGISTRATION_RECORD_BYTES,
                label="registration materialization verification",
            )
        except DatasetCatalogAuthorityError as error:
            raise ValueError("published registration closure cannot be replayed") from error
        matches = tuple(
            value
            for value in snapshot.materializations
            if ObjectIdentity.from_record(value.materialization_id, value)
            == receipt.materialization
        )
        if (
            len(matches) != 1
            or verification_receipt.subject != matches[0].verification_subject()
            or not verification_receipt.validates(
                verification_receipt.subject,
                expected_receipt_id=receipt.materialization_verification_evidence.evidence_id,
                expected_verifier=verification_receipt.verifier,
                expected_verification_policy_id=verification_receipt.verification_policy_id,
                expected_verification_policy_sha256=(
                    verification_receipt.verification_policy_sha256
                ),
                expected_verified_at_utc=verification_receipt.verified_at_utc,
                expected_manifest_evidence=receipt.manifest_evidence,
            )
        ):
            raise ValueError("published registration snapshot differs from its receipt closure")
        receipt_evidence = _reference(
            record_id=receipt_id,
            record=receipt,
            kind=EvidenceReferenceKind.RECEIPT,
            storage_root_id=manifest.control_write_scope.storage_root.object_id,
            relative_locator=receipt_path,
        )
        return DatasetRegistrationInstallation(
            receipt=receipt,
            receipt_evidence=receipt_evidence,
            snapshot=snapshot,
            materialization=matches[0],
            replayed_existing=True,
            new_external_artifacts_created=False,
            catalog_written=False,
            network_bytes=0,
        )


__all__ = ["ExternalDatasetRegistrationPublisher"]
