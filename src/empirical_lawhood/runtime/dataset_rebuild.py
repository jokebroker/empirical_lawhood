"""Trusted preview and authority closure for unified dataset projection rebuilds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_authority import (
    DatasetAuthorizationDecision,
    DatasetDecisionClock,
)
from empirical_lawhood.planning.dataset_rebuild import (
    DatasetProjectionRebuildAuthorization,
    DatasetProjectionRebuildDecision,
    DatasetProjectionRebuildManifest,
    DatasetProjectionRebuildPolicy,
    DatasetProjectionRebuildRequest,
    LocalCatalogTargetState,
    decide_dataset_projection_rebuild_authorization,
    replay_dataset_projection_rebuild_authorization,
)
from empirical_lawhood.planning.datasets import EvidenceReference, EvidenceReferenceKind

from .dataset_operations import (
    DatasetAuthorizationIssuerRegistry,
    DatasetScopeAccess,
    DatasetStoragePreflightProvider,
    DatasetStorageScopePreflight,
    RepositoryCommitPreflight,
    RepositoryCommitPreflightProvider,
)
from .datasets import DatasetCatalogProjectionReceipt, DatasetCatalogProjectionState


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildPolicyRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-projection-rebuild-policy-registry'

    registry_id: str
    policies: tuple[DatasetProjectionRebuildPolicy, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        if any(not isinstance(value, DatasetProjectionRebuildPolicy) for value in self.policies):
            raise TypeError("rebuild policy registry contains another record type")
        require_sorted_unique_ids(self.policies, attribute="policy_id", field_name="policies")

    def validate(self, policy: DatasetProjectionRebuildPolicy) -> None:
        for value in self.policies:
            if value.policy_id == policy.policy_id:
                if value != policy:
                    raise ValueError("rebuild policy differs from its trusted registry")
                return
        raise KeyError(f"unregistered dataset projection rebuild policy: {policy.policy_id}")


@dataclass(frozen=True, slots=True)
class LocalCatalogTargetPreflight(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/local-catalog-target-preflight'

    expected: LocalCatalogTargetState
    observed: LocalCatalogTargetState | None
    matched: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expected, LocalCatalogTargetState):
            raise TypeError("expected must be a LocalCatalogTargetState")
        if self.observed is not None and not isinstance(
            self.observed,
            LocalCatalogTargetState,
        ):
            raise TypeError("observed must be a LocalCatalogTargetState or null")
        if not isinstance(self.matched, bool):
            raise ValueError("matched must be boolean")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.matched != (self.observed == self.expected and not self.reason_codes):
            raise ValueError("local catalog target match differs from its evidence")
        if not self.matched and not self.reason_codes:
            raise ValueError("blocked local catalog target requires reason codes")


class LocalCatalogTargetPreflightProvider(Protocol):
    def inspect(self, expected: LocalCatalogTargetState) -> LocalCatalogTargetPreflight: ...


class DatasetProjectionRebuildPreviewState(StrEnum):
    READY = "READY"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    REFUSED = "REFUSED"
    IDENTITY_BLOCKED = "IDENTITY_BLOCKED"
    STORAGE_BLOCKED = "STORAGE_BLOCKED"
    TARGET_BLOCKED = "TARGET_BLOCKED"


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildPreview(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-projection-rebuild-preview'

    preview_id: str
    manifest: ObjectIdentity
    policy: ObjectIdentity
    request: ObjectIdentity
    authorization: ObjectIdentity | None
    decision: DatasetProjectionRebuildDecision
    repository: RepositoryCommitPreflight
    projection_preflight: DatasetStorageScopePreflight
    receipt_preflight: DatasetStorageScopePreflight
    target_preflight: LocalCatalogTargetPreflight
    state: DatasetProjectionRebuildPreviewState
    reason_codes: tuple[str, ...]
    source_bytes_read: bool
    external_bytes_written: bool
    catalog_written: bool
    network_bytes: int
    gate_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.preview_id, field_name="preview_id")
        for name in ("manifest", "policy", "request"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"{name} must be an ObjectIdentity")
        if self.authorization is not None and not isinstance(self.authorization, ObjectIdentity):
            raise TypeError("authorization must be an ObjectIdentity or null")
        if not isinstance(self.decision, DatasetProjectionRebuildDecision):
            raise TypeError("decision must be a DatasetProjectionRebuildDecision")
        if not isinstance(self.repository, RepositoryCommitPreflight):
            raise TypeError("repository must be a RepositoryCommitPreflight")
        if (
            not isinstance(self.projection_preflight, DatasetStorageScopePreflight)
            or self.projection_preflight.access is not DatasetScopeAccess.READ
        ):
            raise ValueError("projection_preflight must be a read preflight")
        if (
            not isinstance(self.receipt_preflight, DatasetStorageScopePreflight)
            or self.receipt_preflight.access is not DatasetScopeAccess.WRITE
        ):
            raise ValueError("receipt_preflight must be a write preflight")
        if not isinstance(self.target_preflight, LocalCatalogTargetPreflight):
            raise TypeError("target_preflight must be a LocalCatalogTargetPreflight")
        if not isinstance(self.state, DatasetProjectionRebuildPreviewState):
            raise TypeError("state must be a DatasetProjectionRebuildPreviewState")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name in ("source_bytes_read", "external_bytes_written", "catalog_written"):
            if getattr(self, name) is not False:
                raise ValueError(f"rebuild preview {name} must be false")
        if self.network_bytes != 0 or self.gate_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("rebuild preview must be zero-network and outcome-blind")
        expected_state = (
            DatasetProjectionRebuildPreviewState.REFUSED
            if self.decision.decision is DatasetAuthorizationDecision.REFUSED
            else DatasetProjectionRebuildPreviewState.IDENTITY_BLOCKED
            if not self.repository.verified
            else DatasetProjectionRebuildPreviewState.STORAGE_BLOCKED
            if not self.projection_preflight.ready or not self.receipt_preflight.ready
            else DatasetProjectionRebuildPreviewState.TARGET_BLOCKED
            if not self.target_preflight.matched
            else DatasetProjectionRebuildPreviewState.AUTHORITY_REQUIRED
            if self.authorization is None
            else DatasetProjectionRebuildPreviewState.READY
        )
        if self.state is not expected_state:
            raise ValueError("rebuild preview state differs from its evidence")
        if self.state is DatasetProjectionRebuildPreviewState.READY:
            if self.reason_codes:
                raise ValueError("ready rebuild preview cannot retain blockers")
        elif not self.reason_codes:
            raise ValueError("blocked rebuild preview requires reason codes")


class DatasetProjectionRebuildPreviewService:
    def __init__(
        self,
        *,
        policy_registry: DatasetProjectionRebuildPolicyRegistry,
        issuer_registry: DatasetAuthorizationIssuerRegistry,
        clock: DatasetDecisionClock,
        repository_preflight: RepositoryCommitPreflightProvider,
        storage_preflight: DatasetStoragePreflightProvider,
        target_preflight: LocalCatalogTargetPreflightProvider,
    ) -> None:
        self._policy_registry = policy_registry
        self._issuer_registry = issuer_registry
        self._clock = clock
        self._repository_preflight = repository_preflight
        self._storage_preflight = storage_preflight
        self._target_preflight = target_preflight

    def preview(
        self,
        *,
        manifest: DatasetProjectionRebuildManifest,
        policy: DatasetProjectionRebuildPolicy,
        request: DatasetProjectionRebuildRequest,
        authorization_id: str,
        authorization: DatasetProjectionRebuildAuthorization | None = None,
    ) -> DatasetProjectionRebuildPreview:
        validate_stable_id(authorization_id, field_name="authorization_id")
        self._policy_registry.validate(policy)
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        policy_identity = ObjectIdentity.from_record(policy.policy_id, policy)
        request_identity = ObjectIdentity.from_record(request.request_id, request)
        if (
            policy.manifest != manifest_identity
            or request.manifest != manifest_identity
            or request.policy != policy_identity
            or policy.intent != manifest.intent
            or request.intent != manifest.intent
        ):
            raise ValueError("rebuild preview inputs do not share one exact intent")
        intent = manifest.intent
        repository = self._repository_preflight.inspect(intent.implementation_commit)
        projection_preflight = self._storage_preflight.inspect(
            intent.projection_scope,
            access=DatasetScopeAccess.READ,
            minimum_free_bytes=0,
        )
        receipt_preflight = self._storage_preflight.inspect(
            intent.receipt_write_scope,
            access=DatasetScopeAccess.WRITE,
            minimum_free_bytes=intent.work_envelope.control.max_receipt_bytes,
        )
        target_preflight = self._target_preflight.inspect(intent.target_prior_state)
        authorization_identity = None
        if authorization is None:
            decision = decide_dataset_projection_rebuild_authorization(
                policy=policy,
                request=request,
                authorization_id=authorization_id,
                approved_by=policy.authorized_approver_id,
                clock=self._clock,
            )
        else:
            if authorization.decision.authorization_id != authorization_id:
                raise ValueError("supplied rebuild authorization ID differs from its decision")
            issuer = self._issuer_registry.resolve_identity(authorization.authenticated_issuer)
            replay_dataset_projection_rebuild_authorization(
                policy=policy,
                request=request,
                authorization=authorization,
                issuer_registration=issuer,
                manifest=manifest,
                clock=self._clock,
            )
            decision = authorization.decision
            authorization_identity = ObjectIdentity.from_record(authorization_id, authorization)
        if decision.decision is DatasetAuthorizationDecision.REFUSED:
            state = DatasetProjectionRebuildPreviewState.REFUSED
            reasons = decision.reason_codes
        elif not repository.verified:
            state = DatasetProjectionRebuildPreviewState.IDENTITY_BLOCKED
            reasons = repository.reason_codes
        elif not projection_preflight.ready or not receipt_preflight.ready:
            state = DatasetProjectionRebuildPreviewState.STORAGE_BLOCKED
            reasons = tuple(
                sorted(set(projection_preflight.reason_codes).union(receipt_preflight.reason_codes))
            )
        elif not target_preflight.matched:
            state = DatasetProjectionRebuildPreviewState.TARGET_BLOCKED
            reasons = target_preflight.reason_codes
        elif authorization_identity is None:
            state = DatasetProjectionRebuildPreviewState.AUTHORITY_REQUIRED
            reasons = ("DATASET_PROJECTION_REBUILD_AUTHORITY_REQUIRED",)
        else:
            state = DatasetProjectionRebuildPreviewState.READY
            reasons = ()
        return DatasetProjectionRebuildPreview(
            preview_id=f"preview.{request.request_id}",
            manifest=manifest_identity,
            policy=policy_identity,
            request=request_identity,
            authorization=authorization_identity,
            decision=decision,
            repository=repository,
            projection_preflight=projection_preflight,
            receipt_preflight=receipt_preflight,
            target_preflight=target_preflight,
            state=state,
            reason_codes=reasons,
            source_bytes_read=False,
            external_bytes_written=False,
            catalog_written=False,
            network_bytes=0,
            gate_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildAuthorityBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-projection-rebuild-authority-bundle'

    bundle_id: str
    manifest: DatasetProjectionRebuildManifest
    policy: DatasetProjectionRebuildPolicy
    request: DatasetProjectionRebuildRequest
    authorization: DatasetProjectionRebuildAuthorization

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        manifest_identity = ObjectIdentity.from_record(
            self.manifest.manifest_id,
            self.manifest,
        )
        if self.policy.manifest != manifest_identity or self.request.manifest != manifest_identity:
            raise ValueError("rebuild authority bundle binds another manifest")
        if self.request.policy != ObjectIdentity.from_record(self.policy.policy_id, self.policy):
            raise ValueError("rebuild authority bundle request binds another policy")
        decision = self.authorization.decision
        if decision.request != ObjectIdentity.from_record(self.request.request_id, self.request):
            raise ValueError("rebuild authority bundle decision binds another request")
        if decision.decision is not DatasetAuthorizationDecision.APPROVED:
            raise ValueError("rebuild authority bundle requires an approved decision")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildInstallation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-projection-rebuild-installation'

    installation_id: str
    receipt: DatasetCatalogProjectionReceipt
    receipt_evidence: EvidenceReference
    database_relative_path: str
    database_sha256: str
    database_size_bytes: int
    projection_sha256: str
    projection_state: DatasetCatalogProjectionState
    source_bytes_read: int
    network_bytes: int
    external_receipt_written: bool
    catalog_written: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.installation_id, field_name="installation_id")
        if not isinstance(self.receipt, DatasetCatalogProjectionReceipt):
            raise TypeError("receipt must be a DatasetCatalogProjectionReceipt")
        if not isinstance(self.receipt_evidence, EvidenceReference):
            raise TypeError("receipt_evidence must be an EvidenceReference")
        if (
            self.receipt_evidence.evidence_id != self.receipt.receipt_id
            or self.receipt_evidence.kind is not EvidenceReferenceKind.RECEIPT
            or self.receipt_evidence.evidence_schema != self.receipt.SCHEMA
            or self.receipt_evidence.evidence_sha256 != self.receipt.fingerprint()
        ):
            raise ValueError("installation receipt evidence binds another receipt")
        if not isinstance(self.projection_state, DatasetCatalogProjectionState):
            raise TypeError("projection_state must be a DatasetCatalogProjectionState")
        validate_sha256(self.database_sha256, field_name="database_sha256")
        validate_sha256(self.projection_sha256, field_name="projection_sha256")
        if self.database_relative_path != ".empirical-lawhood/experiment_catalog.sqlite3":
            raise ValueError("installation database path differs from the fixed target")
        for name in ("database_size_bytes", "source_bytes_read", "network_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.network_bytes != 0:
            raise ValueError("dataset projection rebuild must use zero network bytes")
        if self.source_bytes_read == 0:
            raise ValueError("successful rebuild must account for its source reads")
        if self.external_receipt_written is not True or self.catalog_written is not True:
            raise ValueError("successful rebuild installation requires both durable effects")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset projection rebuild installation must be outcome-blind")


class DatasetProjectionRebuildInstallerPort(Protocol):
    def install(
        self,
        bundle: DatasetProjectionRebuildAuthorityBundle,
        *,
        receipt_id: str,
    ) -> DatasetProjectionRebuildInstallation: ...


__all__ = [
    "DatasetProjectionRebuildAuthorityBundle",
    "DatasetProjectionRebuildInstallation",
    "DatasetProjectionRebuildInstallerPort",
    "DatasetProjectionRebuildPolicyRegistry",
    "DatasetProjectionRebuildPreview",
    "DatasetProjectionRebuildPreviewService",
    "DatasetProjectionRebuildPreviewState",
    "LocalCatalogTargetPreflight",
    "LocalCatalogTargetPreflightProvider",
]
