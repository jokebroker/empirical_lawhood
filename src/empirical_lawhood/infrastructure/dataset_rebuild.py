"""Infrastructure boundary for authorized dataset projection rebuilds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Final

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError,
    MAX_CATALOG_DATABASE_BYTES,
    bounded_file_sha256,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.planning.dataset_authority import DatasetDecisionClock
from empirical_lawhood.planning.dataset_rebuild import (
    DatasetProjectionRebuildAuthorization,
    DatasetProjectionRebuildManifest,
    DatasetProjectionRebuildPolicy,
    DatasetProjectionRebuildRequest,
    LocalCatalogTargetState,
    replay_dataset_projection_rebuild_authorization,
)
from empirical_lawhood.planning.datasets import EvidenceReference, EvidenceReferenceKind
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.dataset_operations import DatasetAuthorizationIssuerRegistry
from empirical_lawhood.runtime.dataset_rebuild import (
    DatasetProjectionRebuildAuthorityBundle,
    DatasetProjectionRebuildInstallation,
    DatasetProjectionRebuildPolicyRegistry,
    DatasetProjectionRebuildPreviewService,
    DatasetProjectionRebuildPreviewState,
    LocalCatalogTargetPreflight,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogProjectionReceipt,
    DatasetImplementationRegistry,
    UnifiedCatalogSnapshot,
)

from .catalog_rebuild import CatalogRebuildPrepared, rebuild_production_catalog
from .dataset_catalog_authority import _load_exact_reference
from .dataset_operations import _ExternalDatasetAuthorityRecordStore
from .dataset_projection import MAX_UNIFIED_CATALOG_PROJECTION_BYTES


_PROJECTION_READ_AMPLIFICATION: Final[int] = 2


class LocalCatalogTargetInspector:
    """Observe the exact fixed local target without creating repository state."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve(strict=True)

    def inspect(self, expected: LocalCatalogTargetState) -> LocalCatalogTargetPreflight:
        if not isinstance(expected, LocalCatalogTargetState):
            raise TypeError("expected must be a LocalCatalogTargetState")
        target = self._repo_root / expected.relative_path
        reasons: set[str] = set()
        observed: LocalCatalogTargetState | None = None
        try:
            state_directory = target.parent
            if state_directory.is_symlink() or target.is_symlink():
                reasons.add("LOCAL_CATALOG_TARGET_SYMLINK_FORBIDDEN")
            elif not target.exists():
                observed = LocalCatalogTargetState(
                    target_id=expected.target_id,
                    relative_path=expected.relative_path,
                    exists=False,
                    size_bytes=None,
                    sha256=None,
                )
            elif not target.is_file():
                reasons.add("LOCAL_CATALOG_TARGET_NOT_REGULAR")
            else:
                size_bytes, sha256 = bounded_file_sha256(
                    target,
                    maximum_bytes=MAX_CATALOG_DATABASE_BYTES,
                )
                observed = LocalCatalogTargetState(
                    target_id=expected.target_id,
                    relative_path=expected.relative_path,
                    exists=True,
                    size_bytes=size_bytes,
                    sha256=sha256,
                )
        except (BoundedFileIOError, OSError, ValueError):
            reasons.add("LOCAL_CATALOG_TARGET_INSPECTION_FAILED")
        if observed is not None and observed != expected:
            reasons.add("LOCAL_CATALOG_TARGET_PRIOR_STATE_MISMATCH")
        return LocalCatalogTargetPreflight(
            expected=expected,
            observed=observed,
            matched=observed == expected and not reasons,
            reason_codes=tuple(sorted(reasons)),
        )


def _parent(identity: ObjectIdentity) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=identity,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _parents(*identities: ObjectIdentity) -> tuple[ArtifactLineageParent, ...]:
    return tuple(sorted((_parent(value) for value in identities), key=lineage_parent_sort_key))


class ExternalDatasetProjectionRebuildAuthorityStore:
    """Immutable manifest-to-authorization closure for one rebuild authority."""

    _RELATIVE_ROOT = "dataset-operations/rebuild-authority"
    _PUBLICATION_SCOPE_ID = "dataset-projection-rebuild-authority"

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        manifest_decoder: Callable[[bytes], DatasetProjectionRebuildManifest],
        policy_decoder: Callable[[bytes], DatasetProjectionRebuildPolicy],
        request_decoder: Callable[[bytes], DatasetProjectionRebuildRequest],
        authorization_decoder: Callable[[bytes], DatasetProjectionRebuildAuthorization],
        policy_registry: DatasetProjectionRebuildPolicyRegistry,
        issuer_registry: DatasetAuthorizationIssuerRegistry,
    ) -> None:
        self._artifact_plane = artifact_plane
        self._policy_registry = policy_registry
        self._issuer_registry = issuer_registry
        self._manifests = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="manifests",
            record_type=DatasetProjectionRebuildManifest,
            decoder=manifest_decoder,
            relative_root=self._RELATIVE_ROOT,
            publication_scope_id=self._PUBLICATION_SCOPE_ID,
        )
        self._policies = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="policies",
            record_type=DatasetProjectionRebuildPolicy,
            decoder=policy_decoder,
            relative_root=self._RELATIVE_ROOT,
            publication_scope_id=self._PUBLICATION_SCOPE_ID,
        )
        self._requests = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="requests",
            record_type=DatasetProjectionRebuildRequest,
            decoder=request_decoder,
            relative_root=self._RELATIVE_ROOT,
            publication_scope_id=self._PUBLICATION_SCOPE_ID,
        )
        self._authorizations = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="authorizations",
            record_type=DatasetProjectionRebuildAuthorization,
            decoder=authorization_decoder,
            relative_root=self._RELATIVE_ROOT,
            publication_scope_id=self._PUBLICATION_SCOPE_ID,
        )

    def authorization_evidence(
        self,
        authorization: DatasetProjectionRebuildAuthorization,
    ) -> EvidenceReference:
        if not isinstance(authorization, DatasetProjectionRebuildAuthorization):
            raise TypeError("authorization must be dedicated rebuild authority")
        authorization_id = authorization.decision.authorization_id
        return EvidenceReference(
            evidence_id=authorization_id,
            kind=EvidenceReferenceKind.AUTHORIZATION,
            evidence_schema=authorization.SCHEMA,
            evidence_sha256=authorization.fingerprint(),
            storage_root_id=self._artifact_plane.root.contract.storage_root_id,
            relative_locator=(f"{self._RELATIVE_ROOT}/authorizations/{authorization_id}.json"),
        )

    def persist_bundle(
        self,
        bundle: DatasetProjectionRebuildAuthorityBundle,
        *,
        clock: DatasetDecisionClock,
    ) -> tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity, ObjectIdentity]:
        if not isinstance(bundle, DatasetProjectionRebuildAuthorityBundle):
            raise TypeError("bundle must be a DatasetProjectionRebuildAuthorityBundle")
        self._policy_registry.validate(bundle.policy)
        issuer = self._issuer_registry.resolve_identity(bundle.authorization.authenticated_issuer)
        replay_dataset_projection_rebuild_authorization(
            policy=bundle.policy,
            request=bundle.request,
            authorization=bundle.authorization,
            issuer_registration=issuer,
            manifest=bundle.manifest,
            clock=clock,
        )
        manifest_identity = self._manifests.persist(bundle.manifest)
        policy_identity = self._policies.persist(
            bundle.policy,
            lineage_parents=_parents(manifest_identity),
        )
        request_identity = self._requests.persist(
            bundle.request,
            lineage_parents=_parents(manifest_identity, policy_identity),
        )
        authorization_identity = self._authorizations.persist(
            bundle.authorization,
            lineage_parents=_parents(
                manifest_identity,
                policy_identity,
                request_identity,
                bundle.authorization.authenticated_issuer,
            ),
        )
        return (
            manifest_identity,
            policy_identity,
            request_identity,
            authorization_identity,
        )

    def load_bundle(
        self,
        *,
        bundle_id: str,
        manifest_id: str,
        policy_id: str,
        request_id: str,
        authorization_id: str,
        clock: DatasetDecisionClock,
    ) -> DatasetProjectionRebuildAuthorityBundle:
        manifest = self._manifests.load(manifest_id)
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        policy = self._policies.load_with_lineage(
            policy_id,
            expected_lineage_parents=_parents(manifest_identity),
        )
        policy_identity = ObjectIdentity.from_record(policy.policy_id, policy)
        request_parents = _parents(manifest_identity, policy_identity)
        request = self._requests.load_with_lineage(
            request_id,
            expected_lineage_parents=request_parents,
        )
        request_identity = ObjectIdentity.from_record(request.request_id, request)
        untrusted_authorization = self._authorizations.peek_untrusted(authorization_id)
        authorization_parents = _parents(
            manifest_identity,
            policy_identity,
            request_identity,
            untrusted_authorization.authenticated_issuer,
        )
        authorization = self._authorizations.load_with_lineage(
            authorization_id,
            expected_lineage_parents=authorization_parents,
        )
        bundle = DatasetProjectionRebuildAuthorityBundle(
            bundle_id=bundle_id,
            manifest=manifest,
            policy=policy,
            request=request,
            authorization=authorization,
        )
        self._policy_registry.validate(policy)
        issuer = self._issuer_registry.resolve_identity(authorization.authenticated_issuer)
        replay_dataset_projection_rebuild_authorization(
            policy=policy,
            request=request,
            authorization=authorization,
            issuer_registration=issuer,
            manifest=manifest,
            clock=clock,
        )
        return bundle


class DatasetProjectionRebuildInstaller:
    """Replay durable authority, install one exact projection, then publish its receipt."""

    def __init__(
        self,
        *,
        repo_root: Path,
        roots: tuple[GuardedExternalRoot, ...],
        receipt_artifact_plane: ExternalArtifactPlane,
        catalog_artifact_plane: ExternalArtifactPlane,
        implementation_registry: DatasetImplementationRegistry | None,
        evidence_verifier_registry_id: str | tuple[str, ...] | None,
        authority_store: ExternalDatasetProjectionRebuildAuthorityStore,
        preview_service: DatasetProjectionRebuildPreviewService,
        clock: DatasetDecisionClock,
    ) -> None:
        self._repo_root = repo_root.resolve(strict=True)
        if (
            not isinstance(roots, tuple)
            or not roots
            or any(not isinstance(value, GuardedExternalRoot) for value in roots)
        ):
            raise TypeError("rebuild installer roots must be a nonempty guarded tuple")
        root_ids = tuple(value.contract.storage_root_id for value in roots)
        if tuple(sorted(set(root_ids))) != root_ids:
            raise ValueError("rebuild installer roots must be sorted and unique")
        self._roots_by_id = MappingProxyType(dict(zip(root_ids, roots, strict=True)))
        self._receipt_artifact_plane = receipt_artifact_plane
        self._catalog_artifact_plane = catalog_artifact_plane
        self._implementation_registry = implementation_registry
        self._evidence_verifier_registry_id = evidence_verifier_registry_id
        self._authority_store = authority_store
        self._preview_service = preview_service
        self._clock = clock

    def install(
        self,
        bundle: DatasetProjectionRebuildAuthorityBundle,
        *,
        receipt_id: str,
    ) -> DatasetProjectionRebuildInstallation:
        if not isinstance(bundle, DatasetProjectionRebuildAuthorityBundle):
            raise TypeError("bundle must be a DatasetProjectionRebuildAuthorityBundle")
        validate_stable_id(receipt_id, field_name="receipt_id")
        durable = self._authority_store.load_bundle(
            bundle_id=bundle.bundle_id,
            manifest_id=bundle.manifest.manifest_id,
            policy_id=bundle.policy.policy_id,
            request_id=bundle.request.request_id,
            authorization_id=bundle.authorization.decision.authorization_id,
            clock=self._clock,
        )
        if durable != bundle:
            raise ValueError("durable rebuild authority differs from the requested bundle")
        preview = self._preview_service.preview(
            manifest=bundle.manifest,
            policy=bundle.policy,
            request=bundle.request,
            authorization_id=bundle.authorization.decision.authorization_id,
            authorization=bundle.authorization,
        )
        if preview.state is not DatasetProjectionRebuildPreviewState.READY:
            raise PermissionError("dataset projection rebuild did not pass its exact preview")
        intent = bundle.manifest.intent
        receipt_root = self._receipt_artifact_plane.root
        receipt_root_identity = ObjectIdentity.from_record(
            receipt_root.contract.storage_root_id,
            receipt_root.contract,
        )
        if receipt_root_identity != intent.receipt_write_scope.storage_root:
            raise ValueError("receipt artifact plane differs from the authorized write scope")
        projection_reference = EvidenceReference(
            evidence_id=intent.projection.object_id,
            kind=EvidenceReferenceKind.MANIFEST,
            evidence_schema=UnifiedCatalogSnapshot.SCHEMA,
            evidence_sha256=intent.projection.object_fingerprint,
            storage_root_id=intent.projection_scope.storage_root.object_id,
            relative_locator=intent.projection_scope.relative_prefix,
        )
        maximum_projection_bytes = min(
            intent.work_envelope.files.max_single_file_bytes,
            intent.work_envelope.resources.source_scan_bytes,
            MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
        )
        unified = _load_exact_reference(
            projection_reference,
            roots_by_id=self._roots_by_id,
            expected_kind=EvidenceReferenceKind.MANIFEST,
            record_type=UnifiedCatalogSnapshot,
            maximum_bytes=maximum_projection_bytes,
            label="authorized unified catalog projection",
        )
        if ObjectIdentity.from_record(intent.projection.object_id, unified) != intent.projection:
            raise ValueError("loaded unified projection differs from its authorized identity")
        dataset_snapshot = unified.datasets
        if (
            ObjectIdentity.from_record(intent.dataset_snapshot.object_id, dataset_snapshot)
            != intent.dataset_snapshot
        ):
            raise ValueError("unified projection contains another dataset snapshot")
        # Re-run every non-writing gate after the stable source read.  The catalog
        # rebuild also compares the prior target under its mutation lock.
        final_preview = self._preview_service.preview(
            manifest=bundle.manifest,
            policy=bundle.policy,
            request=bundle.request,
            authorization_id=bundle.authorization.decision.authorization_id,
            authorization=bundle.authorization,
        )
        if final_preview.state is not DatasetProjectionRebuildPreviewState.READY:
            raise PermissionError("dataset projection rebuild readiness changed before install")
        projection_payload = unified.canonical_bytes()
        authorization_reference = self._authority_store.authorization_evidence(bundle.authorization)
        built_at_utc = self._clock.now_utc()
        published: tuple[DatasetCatalogProjectionReceipt, EvidenceReference] | None = None

        def publish_receipt(prepared: CatalogRebuildPrepared) -> None:
            nonlocal published
            projection_state = prepared.dataset_projection_state
            if projection_state is None:
                raise RuntimeError("unified rebuild stage lacks dataset projection state")
            receipt = DatasetCatalogProjectionReceipt(
                receipt_id=receipt_id,
                unified_projection_evidence=projection_reference,
                dataset_snapshot=intent.dataset_snapshot,
                projection_state=projection_state,
                rebuild_implementation=intent.rebuild_implementation,
                authorization_evidence=authorization_reference,
                built_at_utc=built_at_utc,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            if not receipt.validates_build(
                unified,
                dataset_snapshot,
                projection_state,
            ):
                raise RuntimeError("projection receipt does not validate the prepared build")
            receipt_relative_path = (
                f"{intent.receipt_write_scope.relative_prefix}/{receipt.receipt_id}.json"
            )
            lineage = _parents(
                intent.projection,
                intent.dataset_snapshot,
                ObjectIdentity.from_record(
                    bundle.authorization.decision.authorization_id,
                    bundle.authorization,
                ),
                ObjectIdentity.from_record(
                    intent.rebuild_implementation.implementation_key,
                    intent.rebuild_implementation,
                ),
            )
            written = self._receipt_artifact_plane.write(
                ArtifactWriteRequest(
                    logical_artifact_id=receipt.receipt_id,
                    relative_path=receipt_relative_path,
                    payload_schema=receipt.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    publication_scope_id=f"publication.{receipt.receipt_id}",
                    publication_scope_relative_root=(intent.receipt_write_scope.relative_prefix),
                    payload=receipt.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    parent_visibility_ceilings=tuple(value.visibility_ceiling for value in lineage),
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    logical_content_sha256=receipt.fingerprint(),
                    lineage_parents=lineage,
                )
            )
            if (
                written.logical.logical_artifact_id != receipt.receipt_id
                or written.logical.content_sha256 != receipt.fingerprint()
            ):
                raise RuntimeError("projection receipt publication returned another identity")
            published = (
                receipt,
                EvidenceReference(
                    evidence_id=receipt.receipt_id,
                    kind=EvidenceReferenceKind.RECEIPT,
                    evidence_schema=receipt.SCHEMA,
                    evidence_sha256=receipt.fingerprint(),
                    storage_root_id=receipt_root.contract.storage_root_id,
                    relative_locator=receipt_relative_path,
                ),
            )

        rebuild = rebuild_production_catalog(
            self._repo_root,
            projection_payload,
            artifact_plane=self._catalog_artifact_plane,
            implementation_registry=self._implementation_registry,
            evidence_verifier_registry_id=self._evidence_verifier_registry_id,
            expected_prior_database=intent.target_prior_state,
            before_install=publish_receipt,
        )
        if published is None:
            raise RuntimeError("catalog rebuild did not publish its external receipt")
        receipt, receipt_evidence = published
        projection_state = rebuild.dataset_projection_state
        if projection_state is None or projection_state != receipt.projection_state:
            raise RuntimeError("installed projection state differs from its external receipt")
        return DatasetProjectionRebuildInstallation(
            installation_id=f"installation.{receipt.receipt_id}",
            receipt=receipt,
            receipt_evidence=receipt_evidence,
            database_relative_path=rebuild.database_relative_path,
            database_sha256=rebuild.database_sha256,
            database_size_bytes=rebuild.database_size_bytes,
            projection_sha256=rebuild.projection_sha256,
            projection_state=projection_state,
            source_bytes_read=_PROJECTION_READ_AMPLIFICATION * len(projection_payload),
            network_bytes=0,
            external_receipt_written=True,
            catalog_written=True,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )


__all__ = [
    "ExternalDatasetProjectionRebuildAuthorityStore",
    "DatasetProjectionRebuildInstaller",
    "LocalCatalogTargetInspector",
]
