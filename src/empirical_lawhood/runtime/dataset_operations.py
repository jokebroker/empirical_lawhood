"""Trusted composition and preview of exact dataset-operation authority.

The records in this module remain outcome-blind.  A preview can prove that an
exact manifest, policy, request, clean implementation commit and storage
preflight agree, but it cannot manufacture a durable authorization or perform
the operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar, Protocol, TypeAlias

from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.dataset_authority import (
    DatasetAuthorizationDecision,
    DatasetAuthorizationIssuerRegistration,
    DatasetDecisionClock,
    DatasetOperationAuthorization,
    DatasetOperationDecision,
    DatasetOperationPolicy,
    DatasetOperationRequest,
    DatasetStorageScope,
    decide_dataset_operation_authorization,
    replay_dataset_operation_authorization,
)
from empirical_lawhood.planning.dataset_manifests import (
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
    ProposedExperimentDatasetBindingManifest,
)
from empirical_lawhood.planning.datasets import (
    DatasetEvidenceClass,
    DatasetMaterializationClass,
)


DatasetOperationManifest: TypeAlias = (
    DatasetRegistrationManifest
    | DatasetTransformationManifest
    | ProposedExperimentDatasetBindingManifest
)


def _manifest_identity(manifest: DatasetOperationManifest) -> ObjectIdentity:
    return ObjectIdentity.from_record(manifest.manifest_id, manifest)


def _manifest_action(manifest: DatasetOperationManifest) -> AuthorityAction:
    if isinstance(manifest, DatasetRegistrationManifest):
        return AuthorityAction.DATASET_REGISTRATION
    if isinstance(manifest, DatasetTransformationManifest):
        return AuthorityAction.DATASET_TRANSFORMATION
    if isinstance(manifest, ProposedExperimentDatasetBindingManifest):
        return AuthorityAction.DATASET_BINDING
    raise TypeError("unsupported dataset operation manifest")


def _manifest_scopes(
    manifest: DatasetOperationManifest,
) -> tuple[
    tuple[DatasetStorageScope, ...],
    DatasetStorageScope | None,
    DatasetStorageScope,
]:
    if isinstance(manifest, DatasetRegistrationManifest):
        return (manifest.source_scope,), None, manifest.control_write_scope
    if isinstance(manifest, DatasetTransformationManifest):
        source_by_id: dict[str, DatasetStorageScope] = {}

        def include(scope: DatasetStorageScope) -> None:
            existing = source_by_id.get(scope.scope_id)
            if existing is not None and existing != scope:
                raise ValueError("dataset manifest reuses a source scope ID")
            source_by_id[scope.scope_id] = scope

        for transform_input in manifest.inputs:
            include(transform_input.scope)
            if transform_input.selector_manifest_scope is not None:
                include(transform_input.selector_manifest_scope)
        for comparison_target in manifest.comparison_targets:
            include(comparison_target.historical_scope)
        sources = tuple(source_by_id[key] for key in sorted(source_by_id))
        return sources, manifest.destination_scope, manifest.control_write_scope
    if isinstance(manifest, ProposedExperimentDatasetBindingManifest):
        return (), None, manifest.control_write_scope
    raise TypeError("unsupported dataset operation manifest")


def compose_dataset_operation_policy(
    manifest: DatasetOperationManifest,
    *,
    policy_id: str,
    evidence_class: DatasetEvidenceClass,
    materialization_class: DatasetMaterializationClass,
    world_kind: WorldKind,
    implementation_commit: str,
    issued_by: str,
    authorized_approver_id: str,
    valid_from_utc: str,
    valid_until_utc: str,
    reason_codes: tuple[str, ...],
) -> DatasetOperationPolicy:
    """Derive an exact policy shape from one already validated manifest."""

    if isinstance(manifest, DatasetRegistrationManifest) and (
        evidence_class is not manifest.evidence_class
        or materialization_class is not manifest.materialization_class
    ):
        raise ValueError("registration policy classification differs from its manifest")
    if isinstance(manifest, ProposedExperimentDatasetBindingManifest) and (
        evidence_class is not manifest.evidence_class
        or materialization_class is not manifest.materialization_class
    ):
        raise ValueError("binding policy classification differs from its manifest")
    source_scopes, destination_scope, control_write_scope = _manifest_scopes(manifest)
    return DatasetOperationPolicy(
        policy_id=policy_id,
        manifest=_manifest_identity(manifest),
        action=_manifest_action(manifest),
        source_scopes=source_scopes,
        destination_scope=destination_scope,
        control_write_scope=control_write_scope,
        work_envelope_ceiling=manifest.work_envelope,
        evidence_class=evidence_class,
        materialization_class=materialization_class,
        world_kind=world_kind,
        outcome_access=manifest.outcome_access,
        visibility_ceiling=manifest.visibility_ceiling,
        implementation_commit=implementation_commit,
        issued_by=issued_by,
        authorized_approver_id=authorized_approver_id,
        valid_from_utc=valid_from_utc,
        valid_until_utc=valid_until_utc,
        reason_codes=reason_codes,
    )


def compose_dataset_operation_request(
    policy: DatasetOperationPolicy,
    manifest: DatasetOperationManifest,
    *,
    request_id: str,
    requested_by: str,
    valid_from_utc: str,
    valid_until_utc: str,
    reason_codes: tuple[str, ...],
) -> DatasetOperationRequest:
    """Derive the no-download/no-mutation request admitted by an exact policy."""

    if policy.manifest != _manifest_identity(manifest):
        raise ValueError("dataset request manifest differs from its exact policy")
    source_scopes, destination_scope, control_write_scope = _manifest_scopes(manifest)
    if (
        policy.action is not _manifest_action(manifest)
        or policy.source_scopes != source_scopes
        or policy.destination_scope != destination_scope
        or policy.control_write_scope != control_write_scope
        or policy.work_envelope_ceiling != manifest.work_envelope
        or policy.outcome_access is not manifest.outcome_access
        or policy.visibility_ceiling is not manifest.visibility_ceiling
    ):
        raise ValueError("dataset policy changes the manifest operation shape")
    return DatasetOperationRequest(
        request_id=request_id,
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        manifest=_manifest_identity(manifest),
        action=policy.action,
        source_scopes=policy.source_scopes,
        destination_scope=policy.destination_scope,
        control_write_scope=policy.control_write_scope,
        work_envelope=manifest.work_envelope,
        evidence_class=policy.evidence_class,
        materialization_class=policy.materialization_class,
        world_kind=policy.world_kind,
        outcome_access=policy.outcome_access,
        visibility_ceiling=policy.visibility_ceiling,
        implementation_commit=policy.implementation_commit,
        requested_by=requested_by,
        valid_from_utc=valid_from_utc,
        valid_until_utc=valid_until_utc,
        source_mutation_requested=False,
        download_requested=False,
        deletion_requested=False,
        reason_codes=reason_codes,
    )


@dataclass(frozen=True, slots=True)
class DatasetOperationPolicyRegistry(CanonicalRecord):
    """Closed set of exact policies trusted by one process composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-operation-policy-registry'

    registry_id: str
    policies: tuple[DatasetOperationPolicy, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        if any(not isinstance(value, DatasetOperationPolicy) for value in self.policies):
            raise ValueError("dataset policy registry contains another record type")
        require_sorted_unique_ids(
            self.policies,
            attribute="policy_id",
            field_name="policies",
        )

    def resolve(self, policy_id: str) -> DatasetOperationPolicy:
        validate_stable_id(policy_id, field_name="policy_id")
        for value in self.policies:
            if value.policy_id == policy_id:
                return value
        raise KeyError(f"unregistered dataset operation policy: {policy_id}")

    def validate(self, policy: DatasetOperationPolicy) -> None:
        if self.resolve(policy.policy_id) != policy:
            raise ValueError("dataset operation policy differs from its trusted registry")


@dataclass(frozen=True, slots=True)
class DatasetAuthorizationIssuerRegistry(CanonicalRecord):
    """Closed trust-anchor registry for detached-signed dataset authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-authorization-issuer-registry'

    registry_id: str
    issuers: tuple[DatasetAuthorizationIssuerRegistration, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        if any(
            not isinstance(value, DatasetAuthorizationIssuerRegistration) for value in self.issuers
        ):
            raise ValueError("dataset issuer registry contains another record type")
        require_sorted_unique_ids(
            self.issuers,
            attribute="issuer_registration_id",
            field_name="issuers",
        )

    def resolve_identity(
        self,
        identity: ObjectIdentity,
    ) -> DatasetAuthorizationIssuerRegistration:
        if not isinstance(identity, ObjectIdentity):
            raise TypeError("issuer identity must be an ObjectIdentity")
        for value in self.issuers:
            expected = ObjectIdentity.from_record(value.issuer_registration_id, value)
            if expected == identity:
                return value
        raise KeyError("dataset authorization issuer identity is not trusted")


@dataclass(frozen=True, slots=True)
class RepositoryCommitPreflight(CanonicalRecord):
    """Bounded observation of the exact implementation worktree identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/repository-commit-preflight'

    expected_commit: str
    observed_commit: str | None
    clean: bool
    verified: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", self.expected_commit) is None:
            raise ValueError("expected_commit must be a lowercase Git SHA-1")
        if self.observed_commit is not None:
            if re.fullmatch(r"[0-9a-f]{40}", self.observed_commit) is None:
                raise ValueError("observed_commit must be a lowercase Git SHA-1")
        if not isinstance(self.clean, bool) or not isinstance(self.verified, bool):
            raise ValueError("repository preflight flags must be boolean")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.verified != (
            self.clean and self.observed_commit == self.expected_commit and not self.reason_codes
        ):
            raise ValueError("repository preflight status differs from its observations")


class DatasetScopeAccess(StrEnum):
    READ = "READ"
    WRITE = "WRITE"


@dataclass(frozen=True, slots=True)
class DatasetStorageScopePreflight(CanonicalRecord):
    """One root/scope readiness observation without opening scientific bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-storage-scope-preflight'

    scope: DatasetStorageScope
    access: DatasetScopeAccess
    root_registered: bool
    relative_locator_safe: bool
    mount_read_ready: bool
    mount_write_ready: bool
    observed_free_bytes: int | None
    effective_write_floor_bytes: int | None
    ready: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, DatasetStorageScope):
            raise TypeError("scope must be a DatasetStorageScope")
        if not isinstance(self.access, DatasetScopeAccess):
            raise TypeError("access must be a DatasetScopeAccess")
        for name in (
            "root_registered",
            "relative_locator_safe",
            "mount_read_ready",
            "mount_write_ready",
            "ready",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("observed_free_bytes", "effective_write_floor_bytes"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_ready = (
            self.root_registered
            and self.relative_locator_safe
            and self.mount_read_ready
            and (self.access is DatasetScopeAccess.READ or self.mount_write_ready)
            and not self.reason_codes
        )
        if self.ready != expected_ready:
            raise ValueError("storage scope preflight status differs from its facts")
        if not self.ready and not self.reason_codes:
            raise ValueError("blocked storage scope preflight requires reason codes")

    @property
    def scope_id(self) -> str:
        return self.scope.scope_id


class DatasetOperationPreviewState(StrEnum):
    READY = "READY"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    REFUSED = "REFUSED"
    IDENTITY_BLOCKED = "IDENTITY_BLOCKED"
    STORAGE_BLOCKED = "STORAGE_BLOCKED"


@dataclass(frozen=True, slots=True)
class DatasetOperationPreview(CanonicalRecord):
    """Outcome-blind, non-writing decision and infrastructure preview."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-operation-preview'

    preview_id: str
    policy: ObjectIdentity
    request: ObjectIdentity
    manifest: ObjectIdentity
    decision: DatasetOperationDecision
    authorization: ObjectIdentity | None
    state: DatasetOperationPreviewState
    repository: RepositoryCommitPreflight
    source_preflights: tuple[DatasetStorageScopePreflight, ...]
    destination_preflight: DatasetStorageScopePreflight | None
    control_preflight: DatasetStorageScopePreflight
    reason_codes: tuple[str, ...]
    source_bytes_read: bool
    network_bytes: int
    external_bytes_written: bool
    catalog_written: bool
    gate_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.preview_id, field_name="preview_id")
        for name in ("policy", "request", "manifest"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"{name} must be an ObjectIdentity")
        if not isinstance(self.decision, DatasetOperationDecision):
            raise TypeError("decision must be a DatasetOperationDecision")
        if self.authorization is not None and not isinstance(
            self.authorization,
            ObjectIdentity,
        ):
            raise TypeError("authorization must be an ObjectIdentity or null")
        if not isinstance(self.state, DatasetOperationPreviewState):
            raise TypeError("state must be a DatasetOperationPreviewState")
        if not isinstance(self.repository, RepositoryCommitPreflight):
            raise TypeError("repository must be a RepositoryCommitPreflight")
        if any(
            not isinstance(value, DatasetStorageScopePreflight)
            or value.access is not DatasetScopeAccess.READ
            for value in self.source_preflights
        ):
            raise ValueError("source_preflights must contain read preflights")
        require_sorted_unique_ids(
            self.source_preflights,
            attribute="scope_id",
            field_name="source_preflights",
        )
        for value, name in (
            (self.destination_preflight, "destination_preflight"),
            (self.control_preflight, "control_preflight"),
        ):
            if value is not None and (
                not isinstance(value, DatasetStorageScopePreflight)
                or value.access is not DatasetScopeAccess.WRITE
            ):
                raise ValueError(f"{name} must be a write preflight")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name in ("source_bytes_read", "external_bytes_written", "catalog_written"):
            if getattr(self, name) is not False:
                raise ValueError(f"dataset operation preview {name} must be false")
        if self.network_bytes != 0:
            raise ValueError("dataset operation preview must use zero network bytes")
        if self.gate_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset operation preview must remain outcome-blind")
        if self.policy != self.decision.policy or self.request != self.decision.request:
            raise ValueError("preview identities differ from the decision")
        if self.manifest != self.decision.manifest:
            raise ValueError("preview manifest differs from the decision")
        storage_ready = (
            all(value.ready for value in self.source_preflights)
            and (self.destination_preflight is None or self.destination_preflight.ready)
            and self.control_preflight.ready
        )
        expected_state = (
            DatasetOperationPreviewState.REFUSED
            if self.decision.decision is DatasetAuthorizationDecision.REFUSED
            else DatasetOperationPreviewState.IDENTITY_BLOCKED
            if not self.repository.verified
            else DatasetOperationPreviewState.STORAGE_BLOCKED
            if not storage_ready
            else DatasetOperationPreviewState.AUTHORITY_REQUIRED
            if self.authorization is None
            else DatasetOperationPreviewState.READY
        )
        if self.state is not expected_state:
            raise ValueError("dataset operation preview state differs from its evidence")
        if self.state is DatasetOperationPreviewState.READY and self.reason_codes:
            raise ValueError("ready dataset operation preview cannot retain blockers")
        if self.state is not DatasetOperationPreviewState.READY and not self.reason_codes:
            raise ValueError("blocked/refused dataset preview requires reason codes")


class RepositoryCommitPreflightProvider(Protocol):
    def inspect(self, expected_commit: str) -> RepositoryCommitPreflight: ...


class DatasetStoragePreflightProvider(Protocol):
    def inspect(
        self,
        scope: DatasetStorageScope,
        *,
        access: DatasetScopeAccess,
        minimum_free_bytes: int,
    ) -> DatasetStorageScopePreflight: ...


class DatasetOperationPreviewService:
    """Compose one exact preview without source reads, writes, or signing."""

    def __init__(
        self,
        *,
        policy_registry: DatasetOperationPolicyRegistry,
        issuer_registry: DatasetAuthorizationIssuerRegistry,
        clock: DatasetDecisionClock,
        repository_preflight: RepositoryCommitPreflightProvider,
        storage_preflight: DatasetStoragePreflightProvider,
    ) -> None:
        self._policy_registry = policy_registry
        self._issuer_registry = issuer_registry
        self._clock = clock
        self._repository_preflight = repository_preflight
        self._storage_preflight = storage_preflight

    def preview(
        self,
        *,
        manifest: DatasetOperationManifest,
        policy: DatasetOperationPolicy,
        request: DatasetOperationRequest,
        authorization_id: str,
        authorization: DatasetOperationAuthorization | None = None,
    ) -> DatasetOperationPreview:
        validate_stable_id(authorization_id, field_name="authorization_id")
        self._policy_registry.validate(policy)
        manifest_identity = _manifest_identity(manifest)
        if request.policy != ObjectIdentity.from_record(policy.policy_id, policy):
            raise ValueError("dataset request does not bind its trusted policy")
        if request.manifest != manifest_identity or policy.manifest != manifest_identity:
            raise ValueError("dataset operation authority binds another manifest")
        repository = self._repository_preflight.inspect(request.implementation_commit)
        source_preflights = tuple(
            self._storage_preflight.inspect(
                scope,
                access=DatasetScopeAccess.READ,
                minimum_free_bytes=0,
            )
            for scope in request.source_scopes
        )
        destination_preflight = (
            None
            if request.destination_scope is None
            else self._storage_preflight.inspect(
                request.destination_scope,
                access=DatasetScopeAccess.WRITE,
                minimum_free_bytes=request.work_envelope.resources.output_bytes,
            )
        )
        control_preflight = self._storage_preflight.inspect(
            request.control_write_scope,
            access=DatasetScopeAccess.WRITE,
            minimum_free_bytes=(
                request.work_envelope.control.max_manifest_bytes
                + request.work_envelope.control.max_receipt_bytes
            ),
        )
        authorization_identity = None
        if authorization is None:
            decision = decide_dataset_operation_authorization(
                policy=policy,
                request=request,
                authorization_id=authorization_id,
                approved_by=policy.authorized_approver_id,
                clock=self._clock,
            )
        else:
            if authorization.decision.authorization_id != authorization_id:
                raise ValueError("supplied authorization ID differs from its decision")
            issuer = self._issuer_registry.resolve_identity(authorization.authenticated_issuer)
            replay_dataset_operation_authorization(
                policy=policy,
                request=request,
                authorization=authorization,
                issuer_registration=issuer,
                manifest=manifest_identity,
                required_action=request.action,
                implementation_commit=request.implementation_commit,
                clock=self._clock,
            )
            decision = authorization.decision
            authorization_identity = ObjectIdentity.from_record(
                decision.authorization_id,
                authorization,
            )
        storage_ready = (
            all(value.ready for value in source_preflights)
            and (destination_preflight is None or destination_preflight.ready)
            and control_preflight.ready
        )
        if decision.decision is DatasetAuthorizationDecision.REFUSED:
            state = DatasetOperationPreviewState.REFUSED
            reasons = decision.reason_codes
        elif not repository.verified:
            state = DatasetOperationPreviewState.IDENTITY_BLOCKED
            reasons = repository.reason_codes
        elif not storage_ready:
            state = DatasetOperationPreviewState.STORAGE_BLOCKED
            reasons = tuple(
                sorted(
                    {
                        reason
                        for value in (
                            *source_preflights,
                            *((destination_preflight,) if destination_preflight else ()),
                            control_preflight,
                        )
                        for reason in value.reason_codes
                    }
                )
            )
        elif authorization_identity is None:
            state = DatasetOperationPreviewState.AUTHORITY_REQUIRED
            reasons = ("DATASET_OPERATION_AUTHORITY_REQUIRED",)
        else:
            state = DatasetOperationPreviewState.READY
            reasons = ()
        return DatasetOperationPreview(
            preview_id=f"preview.{request.request_id}",
            policy=ObjectIdentity.from_record(policy.policy_id, policy),
            request=ObjectIdentity.from_record(request.request_id, request),
            manifest=manifest_identity,
            decision=decision,
            authorization=authorization_identity,
            state=state,
            repository=repository,
            source_preflights=source_preflights,
            destination_preflight=destination_preflight,
            control_preflight=control_preflight,
            reason_codes=reasons,
            source_bytes_read=False,
            network_bytes=0,
            external_bytes_written=False,
            catalog_written=False,
            gate_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )


@dataclass(frozen=True, slots=True)
class DatasetOperationAuthorityBundle(CanonicalRecord):
    """Complete durable policy/request/authorization closure for restart replay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-operation-authority-bundle'

    bundle_id: str
    policy: DatasetOperationPolicy
    request: DatasetOperationRequest
    authorization: DatasetOperationAuthorization

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if not isinstance(self.policy, DatasetOperationPolicy):
            raise TypeError("policy must be a DatasetOperationPolicy")
        if not isinstance(self.request, DatasetOperationRequest):
            raise TypeError("request must be a DatasetOperationRequest")
        if not isinstance(self.authorization, DatasetOperationAuthorization):
            raise TypeError("authorization must be a DatasetOperationAuthorization")
        decision = self.authorization.decision
        if decision.decision is not DatasetAuthorizationDecision.APPROVED:
            raise ValueError("authority bundle requires an approved durable decision")
        if self.request.policy != ObjectIdentity.from_record(
            self.policy.policy_id,
            self.policy,
        ):
            raise ValueError("authority bundle request binds another policy")
        if decision.policy != self.request.policy:
            raise ValueError("authority bundle decision binds another policy")
        if decision.request != ObjectIdentity.from_record(
            self.request.request_id,
            self.request,
        ):
            raise ValueError("authority bundle decision binds another request")
        if decision.manifest != self.request.manifest:
            raise ValueError("authority bundle decision binds another manifest")


__all__ = [
    "DatasetAuthorizationIssuerRegistry",
    "DatasetOperationAuthorityBundle",
    "DatasetOperationManifest",
    "DatasetOperationPolicyRegistry",
    "DatasetOperationPreview",
    "DatasetOperationPreviewService",
    "DatasetOperationPreviewState",
    "DatasetScopeAccess",
    "DatasetStoragePreflightProvider",
    "DatasetStorageScopePreflight",
    "RepositoryCommitPreflight",
    "RepositoryCommitPreflightProvider",
    "compose_dataset_operation_policy",
    "compose_dataset_operation_request",
]
