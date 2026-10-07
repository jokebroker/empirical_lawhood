"""Infrastructure for dataset-operation preflight and immutable authority replay."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Generic, TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.dataset_authority import (
    DatasetDecisionClock,
    DatasetOperationAuthorization,
    DatasetOperationPolicy,
    DatasetOperationRequest,
    DatasetStorageScope,
    replay_dataset_operation_authorization,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactProfile,
    ArtifactWriteRequest,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.dataset_operations import (
    DatasetAuthorizationIssuerRegistry,
    DatasetOperationAuthorityBundle,
    DatasetOperationPolicyRegistry,
    DatasetScopeAccess,
    DatasetStorageScopePreflight,
    RepositoryCommitPreflight,
)

from .artifacts import ArtifactPlaneError, ExternalArtifactPlane, GuardedExternalRoot
from .bounded_io import BoundedFileIOError, read_bounded_bytes
from .bounded_process import BoundedProcessError, run_bounded_command
from .task_receipts import decode_artifact_manifest


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)
MAX_DATASET_AUTHORITY_RECORD_BYTES = 10_000_000
_MAX_GIT_STATUS_BYTES = 1024 * 1024


class CleanRepositoryCommitPreflight:
    """Bounded, non-mutating Git HEAD and full-worktree cleanliness check."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve(strict=True)

    def inspect(self, expected_commit: str) -> RepositoryCommitPreflight:
        observed_commit: str | None = None
        clean = False
        reasons: set[str] = set()
        try:
            head = run_bounded_command(
                ("git", "rev-parse", "--verify", "HEAD"),
                cwd=self._repo_root,
                timeout_seconds=10,
                maximum_stdout_bytes=4096,
                maximum_stderr_bytes=64 * 1024,
            )
            if head.returncode != 0:
                reasons.add("IMPLEMENTATION_COMMIT_UNAVAILABLE")
            else:
                observed_commit = head.stdout.decode("ascii").strip()
            status = run_bounded_command(
                ("git", "status", "--porcelain=v1", "--untracked-files=all"),
                cwd=self._repo_root,
                timeout_seconds=10,
                maximum_stdout_bytes=_MAX_GIT_STATUS_BYTES,
                maximum_stderr_bytes=64 * 1024,
            )
            if status.returncode != 0:
                reasons.add("IMPLEMENTATION_WORKTREE_STATUS_UNAVAILABLE")
            else:
                clean = not status.stdout
                if not clean:
                    reasons.add("IMPLEMENTATION_WORKTREE_DIRTY")
        except (BoundedProcessError, OSError, UnicodeDecodeError, ValueError):
            reasons.add("IMPLEMENTATION_IDENTITY_INSPECTION_FAILED")
        if observed_commit != expected_commit:
            reasons.add("IMPLEMENTATION_COMMIT_MISMATCH")
        return RepositoryCommitPreflight(
            expected_commit=expected_commit,
            observed_commit=observed_commit,
            clean=clean,
            verified=clean and observed_commit == expected_commit and not reasons,
            reason_codes=tuple(sorted(reasons)),
        )


class DatasetStorageRootRegistry:
    """Process-local exact root bindings with alias/nesting rejection."""

    def __init__(self, roots: tuple[GuardedExternalRoot, ...]) -> None:
        if not isinstance(roots, tuple) or any(
            not isinstance(value, GuardedExternalRoot) for value in roots
        ):
            raise TypeError("dataset storage roots must be a tuple of guarded roots")
        root_ids = tuple(value.contract.storage_root_id for value in roots)
        if tuple(sorted(set(root_ids))) != root_ids:
            raise ValueError("dataset storage roots must be sorted and unique")
        canonical_paths = tuple(PurePosixPath(value.contract.canonical_path) for value in roots)
        if len(set(canonical_paths)) != len(canonical_paths):
            raise ValueError("dataset storage root aliases are forbidden")
        for index, path in enumerate(canonical_paths):
            for other in canonical_paths[index + 1 :]:
                if path in other.parents or other in path.parents:
                    raise ValueError("nested dataset storage roots are forbidden")
        self._roots = roots

    def _resolve(self, scope: DatasetStorageScope) -> GuardedExternalRoot | None:
        for value in self._roots:
            identity = ObjectIdentity.from_record(
                value.contract.storage_root_id,
                value.contract,
            )
            if identity == scope.storage_root:
                return value
        return None

    def inspect(
        self,
        scope: DatasetStorageScope,
        *,
        access: DatasetScopeAccess,
        minimum_free_bytes: int,
    ) -> DatasetStorageScopePreflight:
        if not isinstance(scope, DatasetStorageScope):
            raise TypeError("scope must be a DatasetStorageScope")
        if not isinstance(access, DatasetScopeAccess):
            raise TypeError("access must be a DatasetScopeAccess")
        root = self._resolve(scope)
        if root is None:
            return DatasetStorageScopePreflight(
                scope=scope,
                access=access,
                root_registered=False,
                relative_locator_safe=False,
                mount_read_ready=False,
                mount_write_ready=False,
                observed_free_bytes=None,
                effective_write_floor_bytes=None,
                ready=False,
                reason_codes=("DATASET_STORAGE_ROOT_UNREGISTERED",),
            )
        try:
            diagnostic = root.diagnostic(operation_minimum_free_bytes=minimum_free_bytes)
        except (ArtifactPlaneError, OSError, ValueError):
            return DatasetStorageScopePreflight(
                scope=scope,
                access=access,
                root_registered=True,
                relative_locator_safe=False,
                mount_read_ready=False,
                mount_write_ready=False,
                observed_free_bytes=None,
                effective_write_floor_bytes=None,
                ready=False,
                reason_codes=("DATASET_STORAGE_INSPECTION_FAILED",),
            )
        reasons = set(diagnostic.reason_codes)
        if access is DatasetScopeAccess.READ:
            reasons.discard("EXTERNAL_STORAGE_NOT_WRITABLE")
            reasons.discard("EXTERNAL_FREE_SPACE_BELOW_FLOOR")
        locator_safe = True
        try:
            resolved = root.resolve(
                scope.relative_prefix,
                # The diagnostic above owns write readiness.  Resolve here in
                # read mode only to prove registered-root containment without
                # turning low space or read-only media into a false path error.
                for_write=False,
            )
            if access is DatasetScopeAccess.READ and not resolved.exists():
                reasons.add("DATASET_SOURCE_SCOPE_MISSING")
        except (ArtifactPlaneError, OSError, ValueError):
            locator_safe = False
            reasons.add("DATASET_SCOPE_CONTAINMENT_FAILED")
        ready = (
            locator_safe
            and diagnostic.read_ready
            and (access is DatasetScopeAccess.READ or diagnostic.write_ready)
            and not reasons
        )
        return DatasetStorageScopePreflight(
            scope=scope,
            access=access,
            root_registered=True,
            relative_locator_safe=locator_safe,
            mount_read_ready=diagnostic.read_ready,
            mount_write_ready=diagnostic.write_ready,
            observed_free_bytes=diagnostic.observed_free_bytes,
            effective_write_floor_bytes=diagnostic.effective_write_floor_bytes,
            ready=ready,
            reason_codes=tuple(sorted(reasons)),
        )


def _record_id(record: CanonicalRecord) -> str:
    for attribute in ("manifest_id", "policy_id", "request_id", "authorization_id"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return value
    decision = getattr(record, "decision", None)
    value = getattr(decision, "authorization_id", None)
    if isinstance(value, str):
        return value
    raise TypeError("dataset authority record lacks a stable identity")


def _read_bounded(path: Path) -> bytes:
    try:
        return read_bounded_bytes(
            path,
            maximum_bytes=MAX_DATASET_AUTHORITY_RECORD_BYTES,
        )
    except (BoundedFileIOError, OSError) as error:
        raise ValueError("dataset authority record violates its file bound") from error


class _ExternalDatasetAuthorityRecordStore(Generic[_RecordT]):
    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        namespace: str,
        record_type: type[_RecordT],
        decoder: Callable[[bytes], _RecordT],
        relative_root: str = "dataset-operations/authority",
        publication_scope_id: str = "dataset-operation-authority",
    ) -> None:
        self._artifact_plane = artifact_plane
        self._namespace = namespace
        self._record_type = record_type
        self._decoder = decoder
        self._relative_root = relative_root
        self._publication_scope_id = publication_scope_id

    def _relative_path(self, object_id: str) -> str:
        return f"{self._relative_root}/{self._namespace}/{object_id}.json"

    def persist(
        self,
        record: _RecordT,
        *,
        lineage_parents: tuple[ArtifactLineageParent, ...] = (),
    ) -> ObjectIdentity:
        if not isinstance(record, self._record_type):
            raise TypeError("dataset authority store received another record type")
        object_id = _record_id(record)
        parents = tuple(sorted(lineage_parents, key=lineage_parent_sort_key))
        result = self._artifact_plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=object_id,
                relative_path=self._relative_path(object_id),
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=self._publication_scope_id,
                publication_scope_relative_root=self._relative_root,
                payload=record.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=tuple(value.visibility_ceiling for value in parents),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=record.fingerprint(),
                lineage_parents=parents,
            )
        )
        expected = ObjectIdentity.from_record(object_id, record)
        if (
            result.logical.logical_artifact_id != expected.object_id
            or result.logical.content_sha256 != expected.object_fingerprint
        ):
            raise RuntimeError("dataset authority store persisted another identity")
        return expected

    def load(self, object_id: str) -> _RecordT:
        return self.load_with_lineage(object_id, expected_lineage_parents=())

    def peek_untrusted(self, object_id: str) -> _RecordT:
        """Boundedly decode a body only to discover its claimed parent identity.

        The caller must subsequently call :meth:`load_with_lineage`; this value
        is never returned as trusted state by the public store.
        """

        payload_path = self._artifact_plane.root.resolve(
            self._relative_path(object_id),
            for_write=False,
        )
        if not payload_path.is_file():
            raise KeyError(object_id)
        payload = _read_bounded(payload_path)
        try:
            value = self._decoder(payload)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("dataset authority payload is not canonical") from error
        if (
            not isinstance(value, self._record_type)
            or value.canonical_bytes() != payload
            or _record_id(value) != object_id
        ):
            raise ValueError("untrusted dataset authority payload is malformed")
        return value

    def load_with_lineage(
        self,
        object_id: str,
        *,
        expected_lineage_parents: tuple[ArtifactLineageParent, ...],
    ) -> _RecordT:
        relative_path = self._relative_path(object_id)
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        manifest_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not manifest_path.is_file():
            raise KeyError(object_id)
        manifest = decode_artifact_manifest(_read_bounded(manifest_path))
        expected_parents = tuple(sorted(expected_lineage_parents, key=lineage_parent_sort_key))
        if (
            manifest.logical.logical_artifact_id != object_id
            or manifest.logical.payload_schema != self._record_type.SCHEMA
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/json"
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or manifest.materialization.relative_path != relative_path
            or manifest.materialization.size_bytes > MAX_DATASET_AUTHORITY_RECORD_BYTES
            or manifest.logical.lineage_parents != expected_parents
        ):
            raise ValueError("dataset authority manifest differs from its contract")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=manifest.logical,
                materialization=manifest.materialization,
            )
        )
        payload = _read_bounded(payload_path)
        try:
            value = self._decoder(payload)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("dataset authority payload is not canonical") from error
        if not isinstance(value, self._record_type):
            raise ValueError("dataset authority payload has another record type")
        if (
            value.canonical_bytes() != payload
            or _record_id(value) != object_id
            or value.fingerprint() != manifest.logical.content_sha256
        ):
            raise ValueError("dataset authority payload identity differs from its manifest")
        return value


class ExternalDatasetOperationAuthorityStore:
    """Three immutable namespaces whose signed authorization is the commit point."""

    def __init__(
        self,
        artifact_plane: ExternalArtifactPlane,
        *,
        policy_decoder: Callable[[bytes], DatasetOperationPolicy],
        request_decoder: Callable[[bytes], DatasetOperationRequest],
        authorization_decoder: Callable[[bytes], DatasetOperationAuthorization],
        policy_registry: DatasetOperationPolicyRegistry,
        issuer_registry: DatasetAuthorizationIssuerRegistry,
    ) -> None:
        self._policy_registry = policy_registry
        self._issuer_registry = issuer_registry
        self._policies = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="policies",
            record_type=DatasetOperationPolicy,
            decoder=policy_decoder,
        )
        self._requests = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="requests",
            record_type=DatasetOperationRequest,
            decoder=request_decoder,
        )
        self._authorizations = _ExternalDatasetAuthorityRecordStore(
            artifact_plane,
            namespace="authorizations",
            record_type=DatasetOperationAuthorization,
            decoder=authorization_decoder,
        )

    def persist_bundle(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        clock: DatasetDecisionClock,
    ) -> tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity]:
        if not isinstance(bundle, DatasetOperationAuthorityBundle):
            raise TypeError("bundle must be a DatasetOperationAuthorityBundle")
        self._policy_registry.validate(bundle.policy)
        issuer = self._issuer_registry.resolve_identity(bundle.authorization.authenticated_issuer)
        replay_dataset_operation_authorization(
            policy=bundle.policy,
            request=bundle.request,
            authorization=bundle.authorization,
            issuer_registration=issuer,
            manifest=bundle.request.manifest,
            required_action=bundle.request.action,
            implementation_commit=bundle.request.implementation_commit,
            clock=clock,
        )
        policy_identity = self._policies.persist(bundle.policy)
        request_identity = self._requests.persist(
            bundle.request,
            lineage_parents=(
                ArtifactLineageParent(
                    identity=policy_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
            ),
        )
        authorization_identity = self._authorizations.persist(
            bundle.authorization,
            lineage_parents=(
                ArtifactLineageParent(
                    identity=policy_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
                ArtifactLineageParent(
                    identity=request_identity,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
                ArtifactLineageParent(
                    identity=bundle.authorization.authenticated_issuer,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
            ),
        )
        return policy_identity, request_identity, authorization_identity

    def load_bundle(
        self,
        *,
        bundle_id: str,
        policy_id: str,
        request_id: str,
        authorization_id: str,
        clock: DatasetDecisionClock,
    ) -> DatasetOperationAuthorityBundle:
        policy = self._policies.load(policy_id)
        policy_identity = ObjectIdentity.from_record(policy.policy_id, policy)
        request_parent = ArtifactLineageParent(
            identity=policy_identity,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        request = self._requests.load_with_lineage(
            request_id,
            expected_lineage_parents=(request_parent,),
        )
        request_identity = ObjectIdentity.from_record(request.request_id, request)
        # Decode the authorization only after the exact policy/request closure is
        # authenticated; its trusted issuer identity is then replay-checked below.
        authorization = self._authorizations.peek_untrusted(authorization_id)
        authorization_parents = (
            request_parent,
            ArtifactLineageParent(
                identity=request_identity,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
            ArtifactLineageParent(
                identity=authorization.authenticated_issuer,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
        )
        authorization = self._authorizations.load_with_lineage(
            authorization_id,
            expected_lineage_parents=authorization_parents,
        )
        bundle = DatasetOperationAuthorityBundle(
            bundle_id=bundle_id,
            policy=policy,
            request=request,
            authorization=authorization,
        )
        self._policy_registry.validate(policy)
        issuer = self._issuer_registry.resolve_identity(authorization.authenticated_issuer)
        replay_dataset_operation_authorization(
            policy=policy,
            request=request,
            authorization=authorization,
            issuer_registration=issuer,
            manifest=request.manifest,
            required_action=request.action,
            implementation_commit=request.implementation_commit,
            clock=clock,
        )
        return bundle


__all__ = [
    "CleanRepositoryCommitPreflight",
    "DatasetStorageRootRegistry",
    "ExternalDatasetOperationAuthorityStore",
    "MAX_DATASET_AUTHORITY_RECORD_BYTES",
]
