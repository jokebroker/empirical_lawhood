"""Pure authority contracts for one exact unified-catalog projection rebuild.

The rebuild action is deliberately distinct from dataset registration,
transformation and binding.  Its immutable intent binds the exact external
projection, nested dataset snapshot, fixed local target, prior target bytes,
rebuild implementation and external receipt scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp

from .dataset_authority import (
    DatasetAuthorizationDecision,
    DatasetAuthorizationIssuerRegistration,
    DatasetAuthorizationSignatureAlgorithm,
    DatasetAuthorizationSigner,
    DatasetDecisionClock,
    DatasetStorageScope,
    DatasetWorkEnvelope,
)
from .datasets import RegisteredImplementationIdentity


UNIFIED_CATALOG_SNAPSHOT_SCHEMA = 'empirical-lawhood/runtime/unified-catalog-snapshot'
DATASET_CATALOG_SNAPSHOT_SCHEMA = 'empirical-lawhood/runtime/dataset-catalog-snapshot'
PRODUCTION_CATALOG_RELATIVE_PATH = ".empirical-lawhood/experiment_catalog.sqlite3"
_SIGNATURE_VERSION = "1.0.0"
_MAX_REASON_CODES = 64
_MAX_REASON_CODE_BYTES = 128


def _validate_commit(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("implementation_commit must be a lowercase Git SHA-1")


def _validate_interval(valid_from_utc: str, valid_until_utc: str) -> None:
    start = parse_utc_timestamp(valid_from_utc, field_name="valid_from_utc")
    end = parse_utc_timestamp(valid_until_utc, field_name="valid_until_utc")
    if start >= end:
        raise ValueError("rebuild authority validity interval must be positive")


def _validate_reasons(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = True,
) -> None:
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    if len(values) > _MAX_REASON_CODES:
        raise ValueError(f"{field_name} exceeds its count limit")
    if any(len(value.encode("utf-8")) > _MAX_REASON_CODE_BYTES for value in values):
        raise ValueError(f"{field_name} contains an oversized reason code")


def _hex_bytes(value: str, *, byte_length: int, field_name: str) -> bytes:
    if re.fullmatch(rf"[0-9a-f]{{{byte_length * 2}}}", value) is None:
        raise ValueError(f"{field_name} has the wrong lowercase hexadecimal shape")
    return bytes.fromhex(value)


@dataclass(frozen=True, slots=True)
class LocalCatalogTargetState(CanonicalRecord):
    """Exact prior byte identity for the one fixed repository-local target."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/local-catalog-target-state'

    target_id: str
    relative_path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.target_id, field_name="target_id")
        if self.relative_path != PRODUCTION_CATALOG_RELATIVE_PATH:
            raise ValueError("dataset rebuild target must be the fixed production catalog")
        if not isinstance(self.exists, bool):
            raise ValueError("exists must be boolean")
        if self.exists:
            if (
                not isinstance(self.size_bytes, int)
                or isinstance(self.size_bytes, bool)
                or self.size_bytes < 0
                or self.sha256 is None
            ):
                raise ValueError("present catalog target requires exact size and digest")
            validate_sha256(self.sha256, field_name="sha256")
        elif self.size_bytes is not None or self.sha256 is not None:
            raise ValueError("absent catalog target cannot claim byte identity")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildIntent(CanonicalRecord):
    """Complete immutable effect set for one projection installation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-intent'

    projection: ObjectIdentity
    dataset_snapshot: ObjectIdentity
    projection_scope: DatasetStorageScope
    receipt_write_scope: DatasetStorageScope
    target_prior_state: LocalCatalogTargetState
    rebuild_implementation: RegisteredImplementationIdentity
    implementation_commit: str
    work_envelope: DatasetWorkEnvelope
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        if (
            not isinstance(self.projection, ObjectIdentity)
            or self.projection.object_schema != UNIFIED_CATALOG_SNAPSHOT_SCHEMA
        ):
            raise ValueError("projection must name one exact UnifiedCatalogSnapshot")
        if (
            not isinstance(self.dataset_snapshot, ObjectIdentity)
            or self.dataset_snapshot.object_schema != DATASET_CATALOG_SNAPSHOT_SCHEMA
        ):
            raise ValueError("dataset_snapshot must name one exact DatasetCatalogSnapshot")
        if not isinstance(self.projection_scope, DatasetStorageScope):
            raise TypeError("projection_scope must be a DatasetStorageScope")
        if not isinstance(self.receipt_write_scope, DatasetStorageScope):
            raise TypeError("receipt_write_scope must be a DatasetStorageScope")
        if self.projection_scope.overlaps(self.receipt_write_scope):
            raise ValueError("projection input and receipt output scopes must not overlap")
        if not isinstance(self.target_prior_state, LocalCatalogTargetState):
            raise TypeError("target_prior_state must be a LocalCatalogTargetState")
        if not isinstance(self.rebuild_implementation, RegisteredImplementationIdentity):
            raise TypeError("rebuild_implementation must be registered")
        _validate_commit(self.implementation_commit)
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise TypeError("work_envelope must be a DatasetWorkEnvelope")
        files = self.work_envelope.files
        resources = self.work_envelope.resources
        if (
            files.max_source_files != 1
            or files.max_destination_files < 3
            or files.max_archive_members != 0
            or files.max_single_file_bytes == 0
            or resources.source_scan_bytes == 0
            or resources.output_bytes == 0
        ):
            raise ValueError("rebuild work envelope does not cover its exact file effects")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset projection rebuild must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-manifest'

    manifest_id: str
    intent: DatasetProjectionRebuildIntent
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if not isinstance(self.intent, DatasetProjectionRebuildIntent):
            raise TypeError("intent must be a DatasetProjectionRebuildIntent")
        _validate_reasons(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildPolicy(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-policy'

    policy_id: str
    manifest: ObjectIdentity
    intent: DatasetProjectionRebuildIntent
    issued_by: str
    authorized_approver_id: str
    valid_from_utc: str
    valid_until_utc: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("policy_id", self.policy_id),
            ("issued_by", self.issued_by),
            ("authorized_approver_id", self.authorized_approver_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetProjectionRebuildManifest.SCHEMA
        ):
            raise ValueError("policy must bind a rebuild manifest")
        if not isinstance(self.intent, DatasetProjectionRebuildIntent):
            raise TypeError("intent must be a DatasetProjectionRebuildIntent")
        _validate_interval(self.valid_from_utc, self.valid_until_utc)
        _validate_reasons(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-request'

    request_id: str
    policy: ObjectIdentity
    manifest: ObjectIdentity
    intent: DatasetProjectionRebuildIntent
    requested_by: str
    valid_from_utc: str
    valid_until_utc: str
    source_mutation_requested: bool
    target_write_requested: bool
    receipt_write_requested: bool
    download_requested: bool
    deletion_requested: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.requested_by, field_name="requested_by")
        if not isinstance(self.policy, ObjectIdentity):
            raise TypeError("policy must be an ObjectIdentity")
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetProjectionRebuildManifest.SCHEMA
        ):
            raise ValueError("request must bind a rebuild manifest")
        if not isinstance(self.intent, DatasetProjectionRebuildIntent):
            raise TypeError("intent must be a DatasetProjectionRebuildIntent")
        _validate_interval(self.valid_from_utc, self.valid_until_utc)
        for name in (
            "source_mutation_requested",
            "target_write_requested",
            "receipt_write_requested",
            "download_requested",
            "deletion_requested",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        _validate_reasons(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-decision'

    authorization_id: str
    policy: ObjectIdentity
    request: ObjectIdentity
    manifest: ObjectIdentity
    intent: DatasetProjectionRebuildIntent
    decision: DatasetAuthorizationDecision
    requested_by: str
    approved_by: str
    decided_at_utc: str
    decision_clock_id: str
    valid_from_utc: str
    valid_until_utc: str
    source_mutation_requested: bool
    target_write_requested: bool
    receipt_write_requested: bool
    download_requested: bool
    deletion_requested: bool
    request_reason_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    gate_outcome_access: OutcomeAccess
    plan_mutated: bool
    grants_claim_promotion: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_id", self.authorization_id),
            ("requested_by", self.requested_by),
            ("approved_by", self.approved_by),
            ("decision_clock_id", self.decision_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        for name in ("policy", "request", "manifest"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"{name} must be an ObjectIdentity")
        if not isinstance(self.intent, DatasetProjectionRebuildIntent):
            raise TypeError("intent must be a DatasetProjectionRebuildIntent")
        if not isinstance(self.decision, DatasetAuthorizationDecision):
            raise TypeError("decision must be a DatasetAuthorizationDecision")
        parse_utc_timestamp(self.decided_at_utc, field_name="decided_at_utc")
        _validate_interval(self.valid_from_utc, self.valid_until_utc)
        for name in (
            "source_mutation_requested",
            "target_write_requested",
            "receipt_write_requested",
            "download_requested",
            "deletion_requested",
            "plan_mutated",
            "grants_claim_promotion",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        _validate_reasons(self.request_reason_codes, field_name="request_reason_codes")
        _validate_reasons(self.reason_codes, field_name="reason_codes", allow_empty=False)
        if self.decision is DatasetAuthorizationDecision.APPROVED:
            if self.reason_codes != ("DATASET_PROJECTION_REBUILD_APPROVED",):
                raise ValueError("approved rebuild decision has invalid reason codes")
            if self.requested_by == self.approved_by:
                raise ValueError("dataset projection rebuild cannot approve itself")
        elif "DATASET_PROJECTION_REBUILD_APPROVED" in self.reason_codes:
            raise ValueError("refused rebuild decision cannot carry approval")
        if self.gate_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("rebuild authority gate must remain outcome-blind")
        if self.plan_mutated or self.grants_claim_promotion:
            raise ValueError("rebuild authority cannot mutate plans or promote claims")


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildAuthorizationPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-authorization-payload'

    decision: DatasetProjectionRebuildDecision
    authenticated_issuer: ObjectIdentity
    issuer_id: str
    issuer_implementation_sha256: str
    issuer_implementation_version: str
    signature_algorithm: DatasetAuthorizationSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    issued_at_utc: str

    def __post_init__(self) -> None:
        _validate_signed_fields(self)


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildAuthorization(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-projection-rebuild-authorization'

    decision: DatasetProjectionRebuildDecision
    authenticated_issuer: ObjectIdentity
    issuer_id: str
    issuer_implementation_sha256: str
    issuer_implementation_version: str
    signature_algorithm: DatasetAuthorizationSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    issued_at_utc: str
    signature_hex: str

    def __post_init__(self) -> None:
        _validate_signed_fields(self)
        _hex_bytes(self.signature_hex, byte_length=64, field_name="signature_hex")
        _verify_signature(self, verification_key_hex=self.verification_key_hex)

    def unsigned_payload(self) -> DatasetProjectionRebuildAuthorizationPayload:
        return DatasetProjectionRebuildAuthorizationPayload(
            decision=self.decision,
            authenticated_issuer=self.authenticated_issuer,
            issuer_id=self.issuer_id,
            issuer_implementation_sha256=self.issuer_implementation_sha256,
            issuer_implementation_version=self.issuer_implementation_version,
            signature_algorithm=self.signature_algorithm,
            signature_version=self.signature_version,
            verification_key_hex=self.verification_key_hex,
            issued_at_utc=self.issued_at_utc,
        )


def _validate_signed_fields(
    value: DatasetProjectionRebuildAuthorization | DatasetProjectionRebuildAuthorizationPayload,
) -> None:
    if not isinstance(value.decision, DatasetProjectionRebuildDecision):
        raise TypeError("rebuild authorization requires a rebuild decision")
    if not isinstance(value.authenticated_issuer, ObjectIdentity):
        raise TypeError("authenticated_issuer must be an ObjectIdentity")
    validate_stable_id(value.issuer_id, field_name="issuer_id")
    validate_sha256(
        value.issuer_implementation_sha256,
        field_name="issuer_implementation_sha256",
    )
    validate_semantic_version(value.issuer_implementation_version)
    if value.signature_algorithm is not DatasetAuthorizationSignatureAlgorithm.ED25519:
        raise ValueError("unsupported rebuild authorization signature algorithm")
    if value.signature_version != _SIGNATURE_VERSION:
        raise ValueError("unsupported rebuild authorization signature version")
    _hex_bytes(value.verification_key_hex, byte_length=32, field_name="verification_key_hex")
    issued = parse_utc_timestamp(value.issued_at_utc, field_name="issued_at_utc")
    decided = parse_utc_timestamp(value.decision.decided_at_utc, field_name="decided_at_utc")
    valid_until = parse_utc_timestamp(
        value.decision.valid_until_utc,
        field_name="valid_until_utc",
    )
    if not decided <= issued <= valid_until:
        raise ValueError("rebuild authorization issue time is outside its decision interval")


def _verify_signature(
    authorization: DatasetProjectionRebuildAuthorization,
    *,
    verification_key_hex: str,
) -> None:
    public_key = _hex_bytes(
        verification_key_hex,
        byte_length=32,
        field_name="trusted_verification_key_hex",
    )
    signature = _hex_bytes(authorization.signature_hex, byte_length=64, field_name="signature_hex")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            authorization.unsigned_payload().canonical_bytes(),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("rebuild authorization signature verification failed") from error


def compose_dataset_projection_rebuild_policy(
    manifest: DatasetProjectionRebuildManifest,
    *,
    policy_id: str,
    issued_by: str,
    authorized_approver_id: str,
    valid_from_utc: str,
    valid_until_utc: str,
    reason_codes: tuple[str, ...],
) -> DatasetProjectionRebuildPolicy:
    return DatasetProjectionRebuildPolicy(
        policy_id=policy_id,
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        intent=manifest.intent,
        issued_by=issued_by,
        authorized_approver_id=authorized_approver_id,
        valid_from_utc=valid_from_utc,
        valid_until_utc=valid_until_utc,
        reason_codes=reason_codes,
    )


def compose_dataset_projection_rebuild_request(
    policy: DatasetProjectionRebuildPolicy,
    manifest: DatasetProjectionRebuildManifest,
    *,
    request_id: str,
    requested_by: str,
    valid_from_utc: str,
    valid_until_utc: str,
    reason_codes: tuple[str, ...],
) -> DatasetProjectionRebuildRequest:
    manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if policy.manifest != manifest_identity or policy.intent != manifest.intent:
        raise ValueError("rebuild policy differs from its exact manifest")
    return DatasetProjectionRebuildRequest(
        request_id=request_id,
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        manifest=manifest_identity,
        intent=manifest.intent,
        requested_by=requested_by,
        valid_from_utc=valid_from_utc,
        valid_until_utc=valid_until_utc,
        source_mutation_requested=False,
        target_write_requested=True,
        receipt_write_requested=True,
        download_requested=False,
        deletion_requested=False,
        reason_codes=reason_codes,
    )


def _refusal_reasons(
    *,
    policy: DatasetProjectionRebuildPolicy,
    request: DatasetProjectionRebuildRequest,
    approved_by: str,
    decided_at_utc: str,
) -> set[str]:
    reasons: set[str] = set()
    if request.policy != ObjectIdentity.from_record(policy.policy_id, policy):
        reasons.add("POLICY_IDENTITY_MISMATCH")
    if request.manifest != policy.manifest:
        reasons.add("MANIFEST_IDENTITY_MISMATCH")
    if request.intent != policy.intent:
        reasons.add("REBUILD_INTENT_MISMATCH")
    start = parse_utc_timestamp(policy.valid_from_utc, field_name="valid_from_utc")
    end = parse_utc_timestamp(policy.valid_until_utc, field_name="valid_until_utc")
    request_start = parse_utc_timestamp(request.valid_from_utc, field_name="valid_from_utc")
    request_end = parse_utc_timestamp(request.valid_until_utc, field_name="valid_until_utc")
    decided = parse_utc_timestamp(decided_at_utc, field_name="decided_at_utc")
    if request_start < start or request_end > end:
        reasons.add("REQUEST_VALIDITY_OUTSIDE_POLICY")
    if not start <= decided <= end or not request_start <= decided <= request_end:
        reasons.add("DECISION_OUTSIDE_VALIDITY")
    if approved_by != policy.authorized_approver_id:
        reasons.add("APPROVER_NOT_AUTHORIZED")
    if request.requested_by == approved_by:
        reasons.add("SELF_APPROVAL_FORBIDDEN")
    if request.source_mutation_requested:
        reasons.add("SOURCE_MUTATION_FORBIDDEN")
    if not request.target_write_requested:
        reasons.add("TARGET_WRITE_REQUIRED")
    if not request.receipt_write_requested:
        reasons.add("RECEIPT_WRITE_REQUIRED")
    if request.download_requested:
        reasons.add("DOWNLOAD_FORBIDDEN")
    if request.deletion_requested:
        reasons.add("DELETION_FORBIDDEN")
    return reasons


def _decide_at(
    *,
    policy: DatasetProjectionRebuildPolicy,
    request: DatasetProjectionRebuildRequest,
    authorization_id: str,
    approved_by: str,
    decided_at_utc: str,
    decision_clock_id: str,
) -> DatasetProjectionRebuildDecision:
    validate_stable_id(authorization_id, field_name="authorization_id")
    validate_stable_id(approved_by, field_name="approved_by")
    validate_stable_id(decision_clock_id, field_name="decision_clock_id")
    reasons = _refusal_reasons(
        policy=policy,
        request=request,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
    )
    if reasons:
        decision = DatasetAuthorizationDecision.REFUSED
        reasons.add("DATASET_PROJECTION_REBUILD_REFUSED")
    else:
        decision = DatasetAuthorizationDecision.APPROVED
        reasons.add("DATASET_PROJECTION_REBUILD_APPROVED")
    return DatasetProjectionRebuildDecision(
        authorization_id=authorization_id,
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        request=ObjectIdentity.from_record(request.request_id, request),
        manifest=request.manifest,
        intent=request.intent,
        decision=decision,
        requested_by=request.requested_by,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
        decision_clock_id=decision_clock_id,
        valid_from_utc=request.valid_from_utc,
        valid_until_utc=request.valid_until_utc,
        source_mutation_requested=request.source_mutation_requested,
        target_write_requested=request.target_write_requested,
        receipt_write_requested=request.receipt_write_requested,
        download_requested=request.download_requested,
        deletion_requested=request.deletion_requested,
        request_reason_codes=request.reason_codes,
        reason_codes=tuple(sorted(reasons)),
        gate_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        plan_mutated=False,
        grants_claim_promotion=False,
    )


def decide_dataset_projection_rebuild_authorization(
    *,
    policy: DatasetProjectionRebuildPolicy,
    request: DatasetProjectionRebuildRequest,
    authorization_id: str,
    approved_by: str,
    clock: DatasetDecisionClock,
) -> DatasetProjectionRebuildDecision:
    return _decide_at(
        policy=policy,
        request=request,
        authorization_id=authorization_id,
        approved_by=approved_by,
        decided_at_utc=clock.now_utc(),
        decision_clock_id=clock.clock_id,
    )


def issue_dataset_projection_rebuild_authorization(
    *,
    policy: DatasetProjectionRebuildPolicy,
    request: DatasetProjectionRebuildRequest,
    authorization_id: str,
    approved_by: str,
    issuer_registration: DatasetAuthorizationIssuerRegistration,
    signer: DatasetAuthorizationSigner,
    clock: DatasetDecisionClock,
) -> DatasetProjectionRebuildAuthorization:
    decision = decide_dataset_projection_rebuild_authorization(
        policy=policy,
        request=request,
        authorization_id=authorization_id,
        approved_by=approved_by,
        clock=clock,
    )
    if AuthorityAction.DATASET_PROJECTION_REBUILD not in issuer_registration.allowed_actions:
        raise ValueError("issuer is not registered for dataset projection rebuild")
    if (
        signer.signature_algorithm is not issuer_registration.signature_algorithm
        or signer.signature_version != issuer_registration.signature_version
        or signer.verification_key_hex != issuer_registration.verification_key_hex
    ):
        raise ValueError("rebuild signer differs from its issuer registration")
    issuer_identity = ObjectIdentity.from_record(
        issuer_registration.issuer_registration_id,
        issuer_registration,
    )
    payload = DatasetProjectionRebuildAuthorizationPayload(
        decision=decision,
        authenticated_issuer=issuer_identity,
        issuer_id=issuer_registration.issuer_id,
        issuer_implementation_sha256=issuer_registration.implementation_sha256,
        issuer_implementation_version=issuer_registration.implementation_version,
        signature_algorithm=issuer_registration.signature_algorithm,
        signature_version=issuer_registration.signature_version,
        verification_key_hex=issuer_registration.verification_key_hex,
        issued_at_utc=decision.decided_at_utc,
    )
    signature = signer.sign(payload.canonical_bytes())
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("rebuild signer returned a non-Ed25519 signature")
    return DatasetProjectionRebuildAuthorization(
        decision=payload.decision,
        authenticated_issuer=payload.authenticated_issuer,
        issuer_id=payload.issuer_id,
        issuer_implementation_sha256=payload.issuer_implementation_sha256,
        issuer_implementation_version=payload.issuer_implementation_version,
        signature_algorithm=payload.signature_algorithm,
        signature_version=payload.signature_version,
        verification_key_hex=payload.verification_key_hex,
        issued_at_utc=payload.issued_at_utc,
        signature_hex=signature.hex(),
    )


def replay_dataset_projection_rebuild_authorization(
    *,
    policy: DatasetProjectionRebuildPolicy,
    request: DatasetProjectionRebuildRequest,
    authorization: DatasetProjectionRebuildAuthorization,
    issuer_registration: DatasetAuthorizationIssuerRegistration,
    manifest: DatasetProjectionRebuildManifest,
    clock: DatasetDecisionClock,
) -> DatasetProjectionRebuildRequest:
    if not isinstance(authorization, DatasetProjectionRebuildAuthorization):
        raise TypeError("rebuild replay requires dedicated rebuild authorization")
    decision = authorization.decision
    expected_manifest = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    expected_policy = ObjectIdentity.from_record(policy.policy_id, policy)
    expected_request = ObjectIdentity.from_record(request.request_id, request)
    expected_issuer = ObjectIdentity.from_record(
        issuer_registration.issuer_registration_id,
        issuer_registration,
    )
    if authorization.authenticated_issuer != expected_issuer:
        raise ValueError("rebuild authorization binds an untrusted issuer")
    for name, value in (
        ("issuer_id", issuer_registration.issuer_id),
        ("issuer_implementation_sha256", issuer_registration.implementation_sha256),
        ("issuer_implementation_version", issuer_registration.implementation_version),
        ("signature_algorithm", issuer_registration.signature_algorithm),
        ("signature_version", issuer_registration.signature_version),
        ("verification_key_hex", issuer_registration.verification_key_hex),
    ):
        if getattr(authorization, name) != value:
            raise ValueError(f"rebuild authorization changes trusted {name}")
    if AuthorityAction.DATASET_PROJECTION_REBUILD not in issuer_registration.allowed_actions:
        raise ValueError("issuer is not trusted for dataset projection rebuild")
    _verify_signature(
        authorization,
        verification_key_hex=issuer_registration.verification_key_hex,
    )
    if request.policy != expected_policy or decision.policy != expected_policy:
        raise ValueError("rebuild authorization binds another policy")
    if decision.request != expected_request:
        raise ValueError("rebuild authorization binds another request")
    if request.manifest != expected_manifest or decision.manifest != expected_manifest:
        raise ValueError("rebuild authorization binds another manifest")
    if request.intent != manifest.intent or policy.intent != manifest.intent:
        raise ValueError("rebuild authority changes the manifest intent")
    expected_decision = _decide_at(
        policy=policy,
        request=request,
        authorization_id=decision.authorization_id,
        approved_by=decision.approved_by,
        decided_at_utc=decision.decided_at_utc,
        decision_clock_id=decision.decision_clock_id,
    )
    if decision != expected_decision:
        raise ValueError("rebuild authorization differs from complete replay")
    if decision.decision is not DatasetAuthorizationDecision.APPROVED:
        raise PermissionError("dataset projection rebuild authorization was refused")
    if decision.decision_clock_id != clock.clock_id:
        raise ValueError("rebuild authorization binds an untrusted decision clock")
    use_time = parse_utc_timestamp(clock.now_utc(), field_name="trusted_use_time")
    issued = parse_utc_timestamp(authorization.issued_at_utc, field_name="issued_at_utc")
    valid_from = parse_utc_timestamp(decision.valid_from_utc, field_name="valid_from_utc")
    valid_until = parse_utc_timestamp(decision.valid_until_utc, field_name="valid_until_utc")
    if use_time < issued or use_time < valid_from:
        raise PermissionError("dataset projection rebuild authority is not yet valid")
    if use_time > valid_until:
        raise PermissionError("dataset projection rebuild authority has expired")
    return request


__all__ = [
    "DATASET_CATALOG_SNAPSHOT_SCHEMA",
    "DatasetProjectionRebuildAuthorization",
    "DatasetProjectionRebuildAuthorizationPayload",
    "DatasetProjectionRebuildDecision",
    "DatasetProjectionRebuildIntent",
    "DatasetProjectionRebuildManifest",
    "DatasetProjectionRebuildPolicy",
    "DatasetProjectionRebuildRequest",
    "LocalCatalogTargetState",
    "PRODUCTION_CATALOG_RELATIVE_PATH",
    "UNIFIED_CATALOG_SNAPSHOT_SCHEMA",
    "compose_dataset_projection_rebuild_policy",
    "compose_dataset_projection_rebuild_request",
    "decide_dataset_projection_rebuild_authorization",
    "issue_dataset_projection_rebuild_authorization",
    "replay_dataset_projection_rebuild_authorization",
]
