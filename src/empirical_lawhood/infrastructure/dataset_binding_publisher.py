"""Authorized atomic publication of metadata-only experiment dataset bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
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
from empirical_lawhood.infrastructure.dataset_operations import ExternalDatasetOperationAuthorityStore
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.dataset_authority import DatasetDecisionClock
from empirical_lawhood.planning.dataset_manifests import ProposedExperimentDatasetBindingManifest
from empirical_lawhood.planning.datasets import (
    BindingState,
    EvidenceReference,
    EvidenceReferenceKind,
    ExperimentDatasetBinding,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.dataset_binding import (
    DatasetBindingInstallation,
    DatasetBindingPublicationReceipt,
    DatasetBindingSourceAuthority,
    DatasetBindingVerificationReceipt,
    RegisteredDatasetBindingCompiler,
)
from empirical_lawhood.runtime.dataset_operations import (
    DatasetOperationAuthorityBundle,
    DatasetOperationPreviewService,
    DatasetOperationPreviewState,
)
from empirical_lawhood.runtime.datasets import DatasetCatalogSnapshot


_PUBLICATION_SCOPE_ID: Final = "dataset-binding-publication"
_MAX_BINDING_RECORD_BYTES: Final = 10_000_000


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


def _parent(
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


def _parents(values: tuple[ArtifactLineageParent, ...]) -> tuple[ArtifactLineageParent, ...]:
    return tuple(sorted(values, key=lineage_parent_sort_key))


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


def _append_binding(
    snapshot: DatasetCatalogSnapshot,
    binding: ExperimentDatasetBinding,
) -> DatasetCatalogSnapshot:
    by_id = {value.binding_id: value for value in snapshot.bindings}
    existing = by_id.get(binding.binding_id)
    if existing is not None and existing != binding:
        raise ValueError("append-only dataset binding ID already has different content")
    by_id[binding.binding_id] = binding
    return DatasetCatalogSnapshot(
        families=snapshot.families,
        releases=snapshot.releases,
        observations=snapshot.observations,
        materializations=snapshot.materializations,
        acquisition_attempts=snapshot.acquisition_attempts,
        bindings=tuple(sorted(by_id.values(), key=lambda value: value.binding_id)),
    )


class ExternalDatasetBindingPublisher:
    """Replay exact authority, compile metadata, and atomically publish its closure."""

    def __init__(
        self,
        *,
        artifact_plane: ExternalArtifactPlane,
        authority_store: ExternalDatasetOperationAuthorityStore,
        preview_service: DatasetOperationPreviewService,
        clock: DatasetDecisionClock,
        compiler: RegisteredDatasetBindingCompiler,
    ) -> None:
        self._artifact_plane = artifact_plane
        self._authority_store = authority_store
        self._preview_service = preview_service
        self._clock = clock
        self._compiler = compiler
        root = artifact_plane.root
        self._roots_by_id: Mapping[str, GuardedExternalRoot] = MappingProxyType(
            {root.contract.storage_root_id: root}
        )

    def install(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        manifest: ProposedExperimentDatasetBindingManifest,
        base_snapshot: DatasetCatalogSnapshot,
        source_authority: DatasetBindingSourceAuthority,
        binding_id: str,
        receipt_id: str,
        run_id: str | None = None,
    ) -> DatasetBindingInstallation:
        if not isinstance(bundle, DatasetOperationAuthorityBundle):
            raise TypeError("bundle must be a DatasetOperationAuthorityBundle")
        if not isinstance(manifest, ProposedExperimentDatasetBindingManifest):
            raise TypeError("manifest must be a proposed experiment binding")
        if not isinstance(base_snapshot, DatasetCatalogSnapshot):
            raise TypeError("base_snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(source_authority, DatasetBindingSourceAuthority):
            raise TypeError("source_authority must be a DatasetBindingSourceAuthority")
        validate_stable_id(binding_id, field_name="binding_id")
        validate_stable_id(receipt_id, field_name="receipt_id")
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        if (
            bundle.request.action is not AuthorityAction.DATASET_BINDING
            or bundle.request.manifest != manifest_identity
            or bundle.policy.manifest != manifest_identity
        ):
            raise ValueError("binding publisher authority binds another operation")
        durable = self._authority_store.load_bundle(
            bundle_id=bundle.bundle_id,
            policy_id=bundle.policy.policy_id,
            request_id=bundle.request.request_id,
            authorization_id=bundle.authorization.decision.authorization_id,
            clock=self._clock,
        )
        if durable != bundle:
            raise ValueError("durable binding authority differs from requested bundle")
        root_identity = ObjectIdentity.from_record(
            self._artifact_plane.root.contract.storage_root_id,
            self._artifact_plane.root.contract,
        )
        if manifest.control_write_scope.storage_root != root_identity:
            raise ValueError("binding control scope differs from its artifact plane")

        # Exact replay is a read-only authentication of an already committed
        # publication.  It must remain available after the worktree advances;
        # repository and writable-capacity preflights govern only a new act.
        existing = self._load_existing(
            receipt_id=receipt_id,
            manifest=manifest,
            bundle=bundle,
            source_authority=source_authority,
        )
        if existing is not None:
            return existing
        preview = self._preview_service.preview(
            manifest=manifest,
            policy=bundle.policy,
            request=bundle.request,
            authorization_id=bundle.authorization.decision.authorization_id,
            authorization=bundle.authorization,
        )
        if preview.state is not DatasetOperationPreviewState.READY:
            raise PermissionError("dataset binding did not pass its exact preview")
        proposed = self._compiler.compile(
            manifest,
            snapshot=base_snapshot,
            source_authority=source_authority,
            binding_id=binding_id,
            run_id=run_id,
        )
        return self._publish(
            bundle=bundle,
            manifest=manifest,
            base_snapshot=base_snapshot,
            source_authority=source_authority,
            proposed=proposed,
            receipt_id=receipt_id,
        )

    def _publish(
        self,
        *,
        bundle: DatasetOperationAuthorityBundle,
        manifest: ProposedExperimentDatasetBindingManifest,
        base_snapshot: DatasetCatalogSnapshot,
        source_authority: DatasetBindingSourceAuthority,
        proposed: ExperimentDatasetBinding,
        receipt_id: str,
    ) -> DatasetBindingInstallation:
        storage_root_id = manifest.control_write_scope.storage_root.object_id
        prefix = manifest.control_write_scope.relative_prefix
        manifest_path = _relative_path(prefix, "manifests", manifest.manifest_id)
        proposed_id = f"proposed.{proposed.binding_id}"
        proposed_path = _relative_path(prefix, "proposed-bindings", proposed_id)
        verification_id = f"verification.{proposed.binding_id}"
        verification_path = _relative_path(
            prefix,
            "binding-verifications",
            verification_id,
        )
        snapshot_id = f"snapshot.{manifest.manifest_id}"
        snapshot_path = _relative_path(prefix, "dataset-snapshots", snapshot_id)
        publication_path = _relative_path(prefix, "binding-receipts", receipt_id)

        manifest_evidence = _reference(
            record_id=manifest.manifest_id,
            record=manifest,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=manifest_path,
        )
        proposed_evidence = _reference(
            record_id=proposed_id,
            record=proposed,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=proposed_path,
        )
        bound_at_utc = self._clock.now_utc()
        verification = DatasetBindingVerificationReceipt(
            receipt_id=verification_id,
            manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
            proposed_binding=ObjectIdentity.from_record(proposed.binding_id, proposed),
            compiler=self._compiler.implementation,
            source_authority=source_authority.publication_evidence,
            verified_at_utc=bound_at_utc,
        )
        verification_evidence = _reference(
            record_id=verification_id,
            record=verification,
            kind=EvidenceReferenceKind.BINDING,
            storage_root_id=storage_root_id,
            relative_locator=verification_path,
        )
        final_binding = replace(
            proposed,
            binding_state=BindingState.VERIFIED,
            binding_receipt=verification_evidence,
            evidence_refs=tuple(
                sorted(
                    (*proposed.evidence_refs, verification_evidence),
                    key=lambda value: value.evidence_id,
                )
            ),
        )
        snapshot = _append_binding(base_snapshot, final_binding)
        snapshot_evidence = _reference(
            record_id=snapshot_id,
            record=snapshot,
            kind=EvidenceReferenceKind.MANIFEST,
            storage_root_id=storage_root_id,
            relative_locator=snapshot_path,
        )
        policy_identity = ObjectIdentity.from_record(bundle.policy.policy_id, bundle.policy)
        request_identity = ObjectIdentity.from_record(bundle.request.request_id, bundle.request)
        authorization_identity = ObjectIdentity.from_record(
            bundle.authorization.decision.authorization_id,
            bundle.authorization,
        )
        publication = DatasetBindingPublicationReceipt(
            receipt_id=receipt_id,
            authority_bundle_id=bundle.bundle_id,
            manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
            policy=policy_identity,
            request=request_identity,
            authorization=authorization_identity,
            source_authority_evidence=source_authority.publication_evidence,
            manifest_evidence=manifest_evidence,
            proposed_binding_evidence=proposed_evidence,
            binding_verification_evidence=verification_evidence,
            dataset_snapshot_evidence=snapshot_evidence,
            binding=ObjectIdentity.from_record(final_binding.binding_id, final_binding),
            bound_at_utc=bound_at_utc,
            network_bytes=0,
            dataset_bytes_written=False,
            scientific_verdict_issued=False,
            control_batch_published=True,
            catalog_written=False,
            reason_codes=tuple(value.value for value in manifest.exception_codes),
        )
        publication_evidence = _reference(
            record_id=receipt_id,
            record=publication,
            kind=EvidenceReferenceKind.RECEIPT,
            storage_root_id=storage_root_id,
            relative_locator=publication_path,
        )

        output_visibility = manifest.visibility_ceiling
        output_access = manifest.outcome_access
        authority_parents = (
            _parent(
                policy_identity,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
            _parent(
                request_identity,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
            _parent(
                authorization_identity,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
        )

        def output_parent(record_id: str, record: CanonicalRecord) -> ArtifactLineageParent:
            return _parent(
                ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=output_visibility,
                outcome_access=output_access,
            )

        source_parent = _parent(
            ObjectIdentity.from_record(
                source_authority.publication_receipt.receipt_id,
                source_authority.publication_receipt,
            ),
            visibility_ceiling=source_authority.source_manifest.visibility_ceiling,
            outcome_access=source_authority.source_manifest.outcome_access,
        )
        manifest_parent = output_parent(manifest.manifest_id, manifest)
        proposed_parent = output_parent(proposed_id, proposed)
        verification_parent = output_parent(verification_id, verification)
        snapshot_parent = output_parent(snapshot_id, snapshot)
        records: tuple[tuple[str, CanonicalRecord, str, tuple[ArtifactLineageParent, ...]], ...] = (
            (
                manifest.manifest_id,
                manifest,
                manifest_path,
                (*authority_parents, source_parent),
            ),
            (
                proposed_id,
                proposed,
                proposed_path,
                (*authority_parents, source_parent, manifest_parent),
            ),
            (
                verification_id,
                verification,
                verification_path,
                (*authority_parents, source_parent, manifest_parent, proposed_parent),
            ),
            (
                snapshot_id,
                snapshot,
                snapshot_path,
                (*authority_parents, source_parent, manifest_parent, verification_parent),
            ),
            (
                receipt_id,
                publication,
                publication_path,
                (
                    *authority_parents,
                    source_parent,
                    manifest_parent,
                    proposed_parent,
                    verification_parent,
                    snapshot_parent,
                ),
            ),
        )
        if len(records) > manifest.work_envelope.control.max_metadata_records:
            raise ValueError("binding publication exceeds its metadata-record ceiling")
        requests = []
        for record_id, record, path, lineage in records:
            maximum = (
                manifest.work_envelope.control.max_manifest_bytes
                if record_id == manifest.manifest_id
                else manifest.work_envelope.control.max_receipt_bytes
            )
            if len(record.canonical_bytes()) > min(maximum, _MAX_BINDING_RECORD_BYTES):
                raise ValueError("binding publication record exceeds its byte ceiling")
            requests.append(
                _artifact_request(
                    record_id=record_id,
                    record=record,
                    relative_path=path,
                    publication_root=prefix,
                    visibility_ceiling=output_visibility,
                    outcome_access=output_access,
                    lineage_parents=lineage,
                )
            )
        results = self._artifact_plane.write_batch(tuple(requests))
        return DatasetBindingInstallation(
            receipt=publication,
            receipt_evidence=publication_evidence,
            snapshot=snapshot,
            binding=final_binding,
            replayed_existing=False,
            new_external_artifacts_created=any(value.created for value in results),
            catalog_written=False,
            network_bytes=0,
        )

    def _load_existing(
        self,
        *,
        receipt_id: str,
        manifest: ProposedExperimentDatasetBindingManifest,
        bundle: DatasetOperationAuthorityBundle,
        source_authority: DatasetBindingSourceAuthority,
    ) -> DatasetBindingInstallation | None:
        receipt_path = _relative_path(
            manifest.control_write_scope.relative_prefix,
            "binding-receipts",
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
            raise ValueError("binding receipt publication is incomplete")
        try:
            sidecar = decode_artifact_manifest(
                read_bounded_bytes(
                    sidecar_path,
                    maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
                )
            )
            if (
                sidecar.logical.logical_artifact_id != receipt_id
                or sidecar.logical.payload_schema != DatasetBindingPublicationReceipt.SCHEMA
                or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
                or sidecar.materialization.relative_path != receipt_path
                or sidecar.materialization.size_bytes > _MAX_BINDING_RECORD_BYTES
            ):
                raise ValueError("binding receipt artifact manifest differs")
            self._artifact_plane.verify_manifest(sidecar)
            payload = read_bounded_bytes(
                payload_path,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
            )
            receipt = decode_canonical_bytes(
                payload,
                DatasetBindingPublicationReceipt,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
            )
        except (
            ArtifactIdentityConflict,
            BoundedFileIOError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError("published binding receipt cannot be replayed") from error
        if (
            receipt.receipt_id != receipt_id
            or receipt.canonical_bytes() != payload
            or receipt.fingerprint() != sidecar.logical.content_sha256
            or receipt.authority_bundle_id != bundle.bundle_id
            or receipt.manifest != ObjectIdentity.from_record(manifest.manifest_id, manifest)
            or receipt.policy != ObjectIdentity.from_record(bundle.policy.policy_id, bundle.policy)
            or receipt.request
            != ObjectIdentity.from_record(bundle.request.request_id, bundle.request)
            or receipt.authorization
            != ObjectIdentity.from_record(
                bundle.authorization.decision.authorization_id,
                bundle.authorization,
            )
            or receipt.source_authority_evidence != source_authority.publication_evidence
        ):
            raise ValueError("published binding receipt binds another operation")
        try:
            committed_manifest = _load_exact_reference(
                receipt.manifest_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.MANIFEST,
                record_type=ProposedExperimentDatasetBindingManifest,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
                label="dataset binding manifest",
            )
            proposed = _load_exact_reference(
                receipt.proposed_binding_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.MANIFEST,
                record_type=ExperimentDatasetBinding,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
                label="proposed dataset binding",
            )
            snapshot = _load_exact_reference(
                receipt.dataset_snapshot_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.MANIFEST,
                record_type=DatasetCatalogSnapshot,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
                label="binding dataset snapshot",
            )
            verification = _load_exact_reference(
                receipt.binding_verification_evidence,
                roots_by_id=self._roots_by_id,
                expected_kind=EvidenceReferenceKind.BINDING,
                record_type=DatasetBindingVerificationReceipt,
                maximum_bytes=_MAX_BINDING_RECORD_BYTES,
                label="dataset binding verification",
            )
        except DatasetCatalogAuthorityError as error:
            raise ValueError("published binding closure cannot be replayed") from error
        matches = tuple(
            value
            for value in snapshot.bindings
            if ObjectIdentity.from_record(value.binding_id, value) == receipt.binding
        )
        expected_final = replace(
            proposed,
            binding_state=BindingState.VERIFIED,
            binding_receipt=receipt.binding_verification_evidence,
            evidence_refs=tuple(
                sorted(
                    (*proposed.evidence_refs, receipt.binding_verification_evidence),
                    key=lambda value: value.evidence_id,
                )
            ),
        )
        if (
            committed_manifest != manifest
            or proposed.binding_state is not BindingState.PROPOSED
            or verification.manifest != receipt.manifest
            or verification.proposed_binding
            != ObjectIdentity.from_record(proposed.binding_id, proposed)
            or verification.compiler != self._compiler.implementation
            or verification.source_authority != receipt.source_authority_evidence
            or matches != (expected_final,)
        ):
            raise ValueError("published binding snapshot differs from receipt closure")
        receipt_evidence = _reference(
            record_id=receipt_id,
            record=receipt,
            kind=EvidenceReferenceKind.RECEIPT,
            storage_root_id=manifest.control_write_scope.storage_root.object_id,
            relative_locator=receipt_path,
        )
        return DatasetBindingInstallation(
            receipt=receipt,
            receipt_evidence=receipt_evidence,
            snapshot=snapshot,
            binding=matches[0],
            replayed_existing=True,
            new_external_artifacts_created=False,
            catalog_written=False,
            network_bytes=0,
        )


__all__ = ["ExternalDatasetBindingPublisher"]
