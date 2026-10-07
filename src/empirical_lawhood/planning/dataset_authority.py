"""Pure, operation-specific authority contracts for dataset writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind

from .datasets import DatasetEvidenceClass, DatasetMaterializationClass


DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA = 'empirical-lawhood/runtime/external-root-contract'
DATASET_REGISTRATION_MANIFEST_SCHEMA = 'empirical-lawhood/planning/dataset-registration-manifest'
DATASET_TRANSFORMATION_MANIFEST_SCHEMA = 'empirical-lawhood/planning/dataset-transformation-manifest'
DATASET_BINDING_MANIFEST_SCHEMA = 'empirical-lawhood/planning/proposed-experiment-dataset-binding-manifest'


_DATASET_ACTIONS = frozenset(
    {
        AuthorityAction.DATASET_REGISTRATION,
        AuthorityAction.DATASET_TRANSFORMATION,
        AuthorityAction.DATASET_BINDING,
    }
)
_DATASET_ISSUER_ACTIONS = _DATASET_ACTIONS | {
    AuthorityAction.DATASET_PROJECTION_REBUILD,
}
_DATASET_AUTHORIZATION_SIGNATURE_VERSION = "1.0.0"
_MAX_DATASET_STORAGE_SCOPES = 64
_MAX_DATASET_REASON_CODES = 64
_MAX_DATASET_REASON_CODE_BYTES = 128
_MAX_DATASET_REASON_CODES_TOTAL_BYTES = 4096
_MAX_DATASET_FILES = 10_000_000
_MAX_DATASET_ARCHIVE_MEMBERS = 10_000_000
_MAX_DATASET_SINGLE_FILE_BYTES = 16 * 1024**4
_MAX_DATASET_CONTROL_BYTES = 1024**3
_MAX_DATASET_METADATA_RECORDS = 10_000_000
_MAX_DATASET_CPU_CORES = 4096
_MAX_DATASET_MEMORY_BYTES = 16 * 1024**4
_MAX_DATASET_GPU_DEVICES = 1024
_MAX_DATASET_WALL_TIME_SECONDS = 365 * 24 * 60 * 60
_MAX_DATASET_AGGREGATE_BYTES = 1024**5


class DatasetDecisionClock(Protocol):
    """Trusted clock composed by the application, never supplied by a request."""

    clock_id: str

    def now_utc(self) -> str: ...


def _validate_lowercase_hex(value: str, *, byte_length: int, field_name: str) -> bytes:
    if re.fullmatch(rf"[0-9a-f]{{{byte_length * 2}}}", value) is None:
        raise ValueError(
            f"{field_name} must be exactly {byte_length} bytes of lowercase hexadecimal"
        )
    return bytes.fromhex(value)


def _validate_reason_codes(
    reason_codes: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
    envelope_limit: int | None = None,
) -> None:
    require_sorted_unique_strings(
        reason_codes,
        field_name=field_name,
        allow_empty=allow_empty,
    )
    if len(reason_codes) > _MAX_DATASET_REASON_CODES:
        raise ValueError(f"{field_name} exceeds the global reason-code count limit")
    encoded_lengths: list[int] = []
    for reason_code in reason_codes:
        try:
            encoded_length = len(reason_code.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError(f"{field_name} must contain valid UTF-8") from error
        if encoded_length > _MAX_DATASET_REASON_CODE_BYTES:
            raise ValueError(f"{field_name} contains an oversized reason code")
        encoded_lengths.append(encoded_length)
    if sum(encoded_lengths) > _MAX_DATASET_REASON_CODES_TOTAL_BYTES:
        raise ValueError(f"{field_name} exceeds the global reason-code byte limit")
    if envelope_limit is not None and len(reason_codes) > envelope_limit:
        raise ValueError(f"{field_name} exceeds the work-envelope reason-code limit")


def _validate_dataset_resource_budget(resources: ResourceBudget) -> None:
    if not isinstance(resources, ResourceBudget):
        raise ValueError("resources must be a ResourceBudget")
    limits = (
        ("cpu_cores", resources.cpu_cores, 1, _MAX_DATASET_CPU_CORES),
        ("memory_bytes", resources.memory_bytes, 1, _MAX_DATASET_MEMORY_BYTES),
        ("gpu_devices", resources.gpu_devices, 0, _MAX_DATASET_GPU_DEVICES),
        (
            "wall_time_seconds",
            resources.wall_time_seconds,
            1,
            _MAX_DATASET_WALL_TIME_SECONDS,
        ),
        (
            "source_scan_bytes",
            resources.source_scan_bytes,
            0,
            _MAX_DATASET_AGGREGATE_BYTES,
        ),
        ("output_bytes", resources.output_bytes, 0, _MAX_DATASET_AGGREGATE_BYTES),
    )
    for field_name, requested_value, minimum, maximum in limits:
        if not isinstance(requested_value, int) or isinstance(requested_value, bool):
            raise ValueError(f"dataset resource {field_name} must be an integer")
        if not minimum <= requested_value <= maximum:
            raise ValueError(f"dataset resource {field_name} exceeds its absolute bound")


def _validate_dataset_action(value: AuthorityAction) -> None:
    if not isinstance(value, AuthorityAction) or value not in _DATASET_ACTIONS:
        raise ValueError("dataset authority requires a dataset operation action")


def _validate_dataset_manifest_identity(
    manifest: ObjectIdentity,
    action: AuthorityAction,
) -> None:
    "Bind each authority action to its one current authoring-root schema.\n\n    The schema literals live here because ``dataset_manifests`` imports the\n    storage/work contracts from this module. Importing the record classes back\n    into this module would create an inward planning cycle.\n    "

    if not isinstance(manifest, ObjectIdentity):
        raise ValueError("manifest must be an exact ObjectIdentity")
    _validate_dataset_action(action)
    expected_schema = {
        AuthorityAction.DATASET_REGISTRATION: DATASET_REGISTRATION_MANIFEST_SCHEMA,
        AuthorityAction.DATASET_TRANSFORMATION: DATASET_TRANSFORMATION_MANIFEST_SCHEMA,
        AuthorityAction.DATASET_BINDING: DATASET_BINDING_MANIFEST_SCHEMA,
    }[action]
    if manifest.object_schema != expected_schema:
        raise ValueError("dataset action requires its exact current manifest schema")


def _validate_implementation_commit(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("implementation_commit must be a lowercase Git SHA-1")


def _validate_validity_interval(valid_from_utc: str, valid_until_utc: str) -> None:
    valid_from = parse_utc_timestamp(valid_from_utc, field_name="valid_from_utc")
    valid_until = parse_utc_timestamp(valid_until_utc, field_name="valid_until_utc")
    if valid_from >= valid_until:
        raise ValueError("dataset authority validity interval must be positive")


def _validate_context(
    *,
    evidence_class: DatasetEvidenceClass,
    materialization_class: DatasetMaterializationClass,
    world_kind: WorldKind,
    outcome_access: OutcomeAccess,
    visibility_ceiling: VisibilityCeiling,
) -> None:
    if not isinstance(evidence_class, DatasetEvidenceClass):
        raise ValueError("evidence_class must be a DatasetEvidenceClass")
    if not isinstance(materialization_class, DatasetMaterializationClass):
        raise ValueError("materialization_class must be a DatasetMaterializationClass")
    if not isinstance(world_kind, WorldKind):
        raise ValueError("world_kind must be a WorldKind")
    if not isinstance(outcome_access, OutcomeAccess):
        raise ValueError("outcome_access must be an OutcomeAccess")
    if not isinstance(visibility_ceiling, VisibilityCeiling):
        raise ValueError("visibility_ceiling must be a VisibilityCeiling")
    inherited = inherited_visibility((), outcome_access)
    if not visibility_ceiling.is_at_least_as_restrictive_as(inherited):
        raise ValueError("dataset operation visibility cannot be lower than outcome access")


@dataclass(frozen=True, slots=True)
class DatasetStorageScope(CanonicalRecord):
    """One exact registered storage root and root-relative prefix."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-storage-scope'

    scope_id: str
    storage_root: ObjectIdentity
    relative_prefix: str

    def __post_init__(self) -> None:
        validate_stable_id(self.scope_id, field_name="scope_id")
        if not isinstance(self.storage_root, ObjectIdentity):
            raise ValueError("storage_root must be an exact ObjectIdentity")
        if self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA:
            raise ValueError(
                "storage_root object schema must be the current ExternalRootContract schema"
            )
        validate_relative_locator(self.relative_prefix)

    def overlaps(self, other: DatasetStorageScope) -> bool:
        """Conservatively overlap prefixes sharing one registered root ID.

        The exact root fingerprint is separately policy-bound. A trusted root
        registry must reject distinct root IDs that resolve to the same
        canonical path; this pure layer neither resolves nor compares paths.
        """

        if self.storage_root.object_id != other.storage_root.object_id:
            return False
        return (
            self.relative_prefix == other.relative_prefix
            or self.relative_prefix.startswith(f"{other.relative_prefix}/")
            or other.relative_prefix.startswith(f"{self.relative_prefix}/")
        )


def _validate_scopes(scopes: tuple[DatasetStorageScope, ...], *, field_name: str) -> None:
    if not isinstance(scopes, tuple) or not all(
        isinstance(scope, DatasetStorageScope) for scope in scopes
    ):
        raise ValueError(f"{field_name} must contain DatasetStorageScope records")
    if len(scopes) > _MAX_DATASET_STORAGE_SCOPES:
        raise ValueError(f"{field_name} exceeds the storage-scope count limit")
    require_sorted_unique_ids(scopes, attribute="scope_id", field_name=field_name)
    root_prefixes = tuple((scope.storage_root.object_id, scope.relative_prefix) for scope in scopes)
    if len(set(root_prefixes)) != len(root_prefixes):
        raise ValueError(f"{field_name} contains duplicate storage root/prefix scopes")


def _validate_disjoint_payload_scopes(
    source_scopes: tuple[DatasetStorageScope, ...],
    destination_scope: DatasetStorageScope | None,
    control_write_scope: DatasetStorageScope,
) -> None:
    if destination_scope is not None and not isinstance(
        destination_scope,
        DatasetStorageScope,
    ):
        raise ValueError("destination_scope must be a DatasetStorageScope")
    if not isinstance(control_write_scope, DatasetStorageScope):
        raise ValueError("control_write_scope must be a DatasetStorageScope")
    for index, scope in enumerate(source_scopes):
        if any(scope.overlaps(other) for other in source_scopes[index + 1 :]):
            raise ValueError("dataset source scopes must not overlap")
        if scope.overlaps(control_write_scope):
            raise ValueError("dataset source and control-write scopes must not overlap")
    if destination_scope is None:
        return
    if destination_scope.overlaps(control_write_scope):
        raise ValueError("dataset destination and control-write scopes must not overlap")
    if any(destination_scope.overlaps(source) for source in source_scopes):
        raise ValueError("dataset destination scope must not overlap a source scope")


@dataclass(frozen=True, slots=True)
class DatasetFileLimits(CanonicalRecord):
    """Bounded payload enumeration and per-file work."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-file-limits'

    max_source_files: int
    max_destination_files: int
    max_archive_members: int
    max_single_file_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_source_files", self.max_source_files),
            ("max_destination_files", self.max_destination_files),
            ("max_archive_members", self.max_archive_members),
            ("max_single_file_bytes", self.max_single_file_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_source_files > _MAX_DATASET_FILES:
            raise ValueError("max_source_files exceeds the absolute dataset-file bound")
        if self.max_destination_files > _MAX_DATASET_FILES:
            raise ValueError("max_destination_files exceeds the absolute dataset-file bound")
        if self.max_archive_members > _MAX_DATASET_ARCHIVE_MEMBERS:
            raise ValueError("max_archive_members exceeds the absolute archive-member bound")
        if self.max_single_file_bytes > _MAX_DATASET_SINGLE_FILE_BYTES:
            raise ValueError("max_single_file_bytes exceeds the absolute single-file bound")
        file_count = self.max_source_files + self.max_destination_files
        if file_count == 0 and self.max_single_file_bytes != 0:
            raise ValueError("max_single_file_bytes must be zero when no files are admitted")
        if file_count > 0 and self.max_single_file_bytes == 0:
            raise ValueError("file work requires a positive max_single_file_bytes")

    def contains(self, requested: DatasetFileLimits) -> bool:
        return all(
            requested_value <= ceiling_value
            for requested_value, ceiling_value in (
                (requested.max_source_files, self.max_source_files),
                (requested.max_destination_files, self.max_destination_files),
                (requested.max_archive_members, self.max_archive_members),
                (requested.max_single_file_bytes, self.max_single_file_bytes),
            )
        )


@dataclass(frozen=True, slots=True)
class DatasetControlLimits(CanonicalRecord):
    """Bounded manifest and control-plane output work."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-control-limits'

    max_manifest_bytes: int
    max_metadata_records: int
    max_receipt_bytes: int
    max_reason_codes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_metadata_records", self.max_metadata_records),
            ("max_receipt_bytes", self.max_receipt_bytes),
            ("max_reason_codes", self.max_reason_codes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_manifest_bytes > _MAX_DATASET_CONTROL_BYTES:
            raise ValueError("max_manifest_bytes exceeds the absolute control-byte bound")
        if self.max_receipt_bytes > _MAX_DATASET_CONTROL_BYTES:
            raise ValueError("max_receipt_bytes exceeds the absolute control-byte bound")
        if self.max_metadata_records > _MAX_DATASET_METADATA_RECORDS:
            raise ValueError("max_metadata_records exceeds the absolute metadata-record bound")
        if self.max_reason_codes > _MAX_DATASET_REASON_CODES:
            raise ValueError("max_reason_codes exceeds the global reason-code count limit")

    def contains(self, requested: DatasetControlLimits) -> bool:
        return all(
            requested_value <= ceiling_value
            for requested_value, ceiling_value in (
                (requested.max_manifest_bytes, self.max_manifest_bytes),
                (requested.max_metadata_records, self.max_metadata_records),
                (requested.max_receipt_bytes, self.max_receipt_bytes),
                (requested.max_reason_codes, self.max_reason_codes),
            )
        )


@dataclass(frozen=True, slots=True)
class DatasetWorkEnvelope(CanonicalRecord):
    """Complete bounded resource, file and control-plane request."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-work-envelope'

    resources: ResourceBudget
    files: DatasetFileLimits
    control: DatasetControlLimits

    def __post_init__(self) -> None:
        _validate_dataset_resource_budget(self.resources)
        if not isinstance(self.files, DatasetFileLimits):
            raise ValueError("files must be DatasetFileLimits")
        if not isinstance(self.control, DatasetControlLimits):
            raise ValueError("control must be DatasetControlLimits")

    def contains(self, requested: DatasetWorkEnvelope) -> bool:
        return (
            self.resources.contains(requested.resources)
            and self.files.contains(requested.files)
            and self.control.contains(requested.control)
        )


def _validate_policy_operation_shape(policy: DatasetOperationPolicy) -> None:
    files = policy.work_envelope_ceiling.files
    resources = policy.work_envelope_ceiling.resources
    if policy.action is AuthorityAction.DATASET_REGISTRATION:
        if not policy.source_scopes:
            raise ValueError("registration policy requires exact source scopes")
        if policy.destination_scope is not None or files.max_destination_files != 0:
            raise ValueError("registration policy cannot admit dataset destination writes")
    elif policy.action is AuthorityAction.DATASET_TRANSFORMATION:
        if not policy.source_scopes:
            raise ValueError("transformation policy requires exact source scopes")
        if policy.destination_scope is None or files.max_destination_files == 0:
            raise ValueError("transformation policy requires one exact destination scope")
    else:
        if policy.source_scopes or policy.destination_scope is not None:
            raise ValueError("binding policy may write only its control scope")
        if any(
            (
                files.max_source_files,
                files.max_destination_files,
                files.max_archive_members,
                files.max_single_file_bytes,
                resources.source_scan_bytes,
            )
        ):
            raise ValueError("binding policy cannot admit dataset payload I/O")


@dataclass(frozen=True, slots=True)
class DatasetOperationPolicy(CanonicalRecord):
    """Exact operation policy; one policy cannot float across manifests or roots."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-operation-policy'

    policy_id: str
    manifest: ObjectIdentity
    action: AuthorityAction
    source_scopes: tuple[DatasetStorageScope, ...]
    destination_scope: DatasetStorageScope | None
    control_write_scope: DatasetStorageScope
    work_envelope_ceiling: DatasetWorkEnvelope
    evidence_class: DatasetEvidenceClass
    materialization_class: DatasetMaterializationClass
    world_kind: WorldKind
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    implementation_commit: str
    issued_by: str
    authorized_approver_id: str
    valid_from_utc: str
    valid_until_utc: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, identifier in (
            ("policy_id", self.policy_id),
            ("issued_by", self.issued_by),
            ("authorized_approver_id", self.authorized_approver_id),
        ):
            validate_stable_id(identifier, field_name=field_name)
        if not isinstance(self.work_envelope_ceiling, DatasetWorkEnvelope):
            raise ValueError("work_envelope_ceiling must be a DatasetWorkEnvelope")
        _validate_dataset_manifest_identity(self.manifest, self.action)
        _validate_scopes(self.source_scopes, field_name="source_scopes")
        _validate_disjoint_payload_scopes(
            self.source_scopes,
            self.destination_scope,
            self.control_write_scope,
        )
        _validate_context(
            evidence_class=self.evidence_class,
            materialization_class=self.materialization_class,
            world_kind=self.world_kind,
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
        )
        _validate_implementation_commit(self.implementation_commit)
        _validate_validity_interval(self.valid_from_utc, self.valid_until_utc)
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            envelope_limit=self.work_envelope_ceiling.control.max_reason_codes,
        )
        _validate_policy_operation_shape(self)


@dataclass(frozen=True, slots=True)
class DatasetOperationRequest(CanonicalRecord):
    """Frozen request containing every effect the dataset gate must adjudicate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-operation-request'

    request_id: str
    policy: ObjectIdentity
    manifest: ObjectIdentity
    action: AuthorityAction
    source_scopes: tuple[DatasetStorageScope, ...]
    destination_scope: DatasetStorageScope | None
    control_write_scope: DatasetStorageScope
    work_envelope: DatasetWorkEnvelope
    evidence_class: DatasetEvidenceClass
    materialization_class: DatasetMaterializationClass
    world_kind: WorldKind
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    implementation_commit: str
    requested_by: str
    valid_from_utc: str
    valid_until_utc: str
    source_mutation_requested: bool
    download_requested: bool
    deletion_requested: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.requested_by, field_name="requested_by")
        if not isinstance(self.policy, ObjectIdentity):
            raise ValueError("policy must be an exact ObjectIdentity")
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise ValueError("work_envelope must be a DatasetWorkEnvelope")
        _validate_dataset_manifest_identity(self.manifest, self.action)
        _validate_scopes(self.source_scopes, field_name="source_scopes")
        _validate_disjoint_payload_scopes(
            self.source_scopes,
            self.destination_scope,
            self.control_write_scope,
        )
        _validate_context(
            evidence_class=self.evidence_class,
            materialization_class=self.materialization_class,
            world_kind=self.world_kind,
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
        )
        _validate_implementation_commit(self.implementation_commit)
        _validate_validity_interval(self.valid_from_utc, self.valid_until_utc)
        for flag_name, flag_value in (
            ("source_mutation_requested", self.source_mutation_requested),
            ("download_requested", self.download_requested),
            ("deletion_requested", self.deletion_requested),
        ):
            if not isinstance(flag_value, bool):
                raise ValueError(f"{flag_name} must be boolean")
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            envelope_limit=self.work_envelope.control.max_reason_codes,
        )


class DatasetAuthorizationDecision(StrEnum):
    APPROVED = "APPROVED"
    REFUSED = "REFUSED"


@dataclass(frozen=True, slots=True)
class DatasetOperationDecision(CanonicalRecord):
    """Unsigned outcome-blind decision; it grants no authority by itself."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-operation-decision'

    authorization_id: str
    policy: ObjectIdentity
    request: ObjectIdentity
    manifest: ObjectIdentity
    decision: DatasetAuthorizationDecision
    action: AuthorityAction
    source_scopes: tuple[DatasetStorageScope, ...]
    destination_scope: DatasetStorageScope | None
    control_write_scope: DatasetStorageScope
    work_envelope: DatasetWorkEnvelope
    evidence_class: DatasetEvidenceClass
    materialization_class: DatasetMaterializationClass
    world_kind: WorldKind
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    implementation_commit: str
    requested_by: str
    approved_by: str
    decided_at_utc: str
    decision_clock_id: str
    valid_from_utc: str
    valid_until_utc: str
    source_mutation_requested: bool
    download_requested: bool
    deletion_requested: bool
    request_reason_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    gate_outcome_access: OutcomeAccess
    plan_mutated: bool
    grants_claim_promotion: bool

    def __post_init__(self) -> None:
        for field_name, identifier in (
            ("authorization_id", self.authorization_id),
            ("requested_by", self.requested_by),
            ("approved_by", self.approved_by),
            ("decision_clock_id", self.decision_clock_id),
        ):
            validate_stable_id(identifier, field_name=field_name)
        for field_name, identity in (
            ("policy", self.policy),
            ("request", self.request),
            ("manifest", self.manifest),
        ):
            if not isinstance(identity, ObjectIdentity):
                raise ValueError(f"{field_name} must be an exact ObjectIdentity")
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise ValueError("work_envelope must be a DatasetWorkEnvelope")
        if not isinstance(self.decision, DatasetAuthorizationDecision):
            raise ValueError("decision must be a DatasetAuthorizationDecision")
        _validate_dataset_manifest_identity(self.manifest, self.action)
        _validate_scopes(self.source_scopes, field_name="source_scopes")
        _validate_disjoint_payload_scopes(
            self.source_scopes,
            self.destination_scope,
            self.control_write_scope,
        )
        _validate_context(
            evidence_class=self.evidence_class,
            materialization_class=self.materialization_class,
            world_kind=self.world_kind,
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
        )
        _validate_implementation_commit(self.implementation_commit)
        parse_utc_timestamp(self.decided_at_utc, field_name="decided_at_utc")
        _validate_validity_interval(self.valid_from_utc, self.valid_until_utc)
        _validate_reason_codes(
            self.request_reason_codes,
            field_name="request_reason_codes",
            envelope_limit=self.work_envelope.control.max_reason_codes,
        )
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
        )
        for flag_name, flag_value in (
            ("source_mutation_requested", self.source_mutation_requested),
            ("download_requested", self.download_requested),
            ("deletion_requested", self.deletion_requested),
            ("plan_mutated", self.plan_mutated),
            ("grants_claim_promotion", self.grants_claim_promotion),
        ):
            if not isinstance(flag_value, bool):
                raise ValueError(f"{flag_name} must be boolean")
        if self.decision is DatasetAuthorizationDecision.APPROVED:
            if self.reason_codes != ("DATASET_OPERATION_APPROVED",):
                raise ValueError("approved dataset authorization has invalid reason codes")
            if self.requested_by == self.approved_by:
                raise ValueError("dataset operation cannot approve itself")
        elif "DATASET_OPERATION_APPROVED" in self.reason_codes:
            raise ValueError("refused dataset authorization cannot carry approval")
        if self.gate_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset authorization gate must be outcome-blind")
        if self.plan_mutated:
            raise ValueError("dataset authorization gate cannot mutate the manifest or plan")
        if self.grants_claim_promotion:
            raise ValueError("dataset authorization cannot promote scientific claims")


class DatasetAuthorizationSignatureAlgorithm(StrEnum):
    ED25519 = "ED25519"


@dataclass(frozen=True, slots=True)
class DatasetAuthorizationIssuerRegistration(CanonicalRecord):
    """Immutable trust anchor for an outcome-blind dataset decision issuer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-authorization-issuer-registration'

    issuer_registration_id: str
    issuer_id: str
    allowed_actions: frozenset[AuthorityAction]
    implementation_sha256: str
    implementation_version: str
    signature_algorithm: DatasetAuthorizationSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(
            self.issuer_registration_id,
            field_name="issuer_registration_id",
        )
        validate_stable_id(self.issuer_id, field_name="issuer_id")
        if not self.allowed_actions or not all(
            isinstance(action, AuthorityAction) and action in _DATASET_ISSUER_ACTIONS
            for action in self.allowed_actions
        ):
            raise ValueError("dataset authorization issuer has invalid allowed actions")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        validate_semantic_version(self.implementation_version)
        if self.signature_algorithm is not DatasetAuthorizationSignatureAlgorithm.ED25519:
            raise ValueError("unsupported dataset authorization signature algorithm")
        if self.signature_version != _DATASET_AUTHORIZATION_SIGNATURE_VERSION:
            raise ValueError("unsupported dataset authorization signature version")
        _validate_lowercase_hex(
            self.verification_key_hex,
            byte_length=32,
            field_name="verification_key_hex",
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset authorization issuer must be outcome-blind")


class DatasetAuthorizationSigner(Protocol):
    """Injected private signing boundary; no key material lives in this module."""

    signature_algorithm: DatasetAuthorizationSignatureAlgorithm
    signature_version: str
    verification_key_hex: str

    def sign(self, payload: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DatasetOperationAuthorizationPayload(CanonicalRecord):
    """Canonical detached-signature payload for one dataset decision."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-operation-authorization-payload'

    decision: DatasetOperationDecision
    authenticated_issuer: ObjectIdentity
    issuer_id: str
    issuer_implementation_sha256: str
    issuer_implementation_version: str
    signature_algorithm: DatasetAuthorizationSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    issued_at_utc: str

    def __post_init__(self) -> None:
        _validate_signed_authorization_material(self)


@dataclass(frozen=True, slots=True)
class DatasetOperationAuthorization(CanonicalRecord):
    """Detached-signed durable authority for one exact dataset decision."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-operation-authorization'

    decision: DatasetOperationDecision
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
        _validate_signed_authorization_material(self)
        # Reject bool/int substitution before replay equality can compare 0/1 as False/True.
        for name in (
            "source_mutation_requested",
            "download_requested",
            "deletion_requested",
            "plan_mutated",
            "grants_claim_promotion",
        ):
            if not isinstance(getattr(self.decision, name), bool):
                raise ValueError(f"authorization decision {name} must be boolean")
        _validate_lowercase_hex(
            self.signature_hex,
            byte_length=64,
            field_name="signature_hex",
        )
        _verify_dataset_authorization_signature(
            self,
            verification_key_hex=self.verification_key_hex,
        )

    def unsigned_payload(self) -> DatasetOperationAuthorizationPayload:
        return DatasetOperationAuthorizationPayload(
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


def _validate_signed_authorization_material(
    value: DatasetOperationAuthorization | DatasetOperationAuthorizationPayload,
) -> None:
    if not isinstance(value.decision, DatasetOperationDecision):
        raise TypeError("signed dataset authority requires a dataset operation decision")
    if not isinstance(value.authenticated_issuer, ObjectIdentity):
        raise ValueError("authenticated_issuer must be an exact ObjectIdentity")
    validate_stable_id(value.issuer_id, field_name="issuer_id")
    validate_sha256(
        value.issuer_implementation_sha256,
        field_name="issuer_implementation_sha256",
    )
    validate_semantic_version(value.issuer_implementation_version)
    if value.signature_algorithm is not DatasetAuthorizationSignatureAlgorithm.ED25519:
        raise ValueError("unsupported dataset authorization signature algorithm")
    if value.signature_version != _DATASET_AUTHORIZATION_SIGNATURE_VERSION:
        raise ValueError("unsupported dataset authorization signature version")
    _validate_lowercase_hex(
        value.verification_key_hex,
        byte_length=32,
        field_name="verification_key_hex",
    )
    issued_at = parse_utc_timestamp(value.issued_at_utc, field_name="issued_at_utc")
    decided_at = parse_utc_timestamp(
        value.decision.decided_at_utc,
        field_name="decided_at_utc",
    )
    valid_until = parse_utc_timestamp(
        value.decision.valid_until_utc,
        field_name="valid_until_utc",
    )
    if issued_at < decided_at:
        raise ValueError("dataset authorization cannot be issued before its decision")
    if issued_at > valid_until:
        raise ValueError("dataset authorization cannot be issued after its validity interval")


def _verify_dataset_authorization_signature(
    authorization: DatasetOperationAuthorization,
    *,
    verification_key_hex: str,
) -> None:
    public_key = _validate_lowercase_hex(
        verification_key_hex,
        byte_length=32,
        field_name="trusted_verification_key_hex",
    )
    signature = _validate_lowercase_hex(
        authorization.signature_hex,
        byte_length=64,
        field_name="signature_hex",
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            authorization.unsigned_payload().canonical_bytes(),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("dataset authorization signature verification failed") from error


def issue_dataset_operation_authorization(
    *,
    policy: DatasetOperationPolicy,
    request: DatasetOperationRequest,
    authorization_id: str,
    approved_by: str,
    issuer_registration: DatasetAuthorizationIssuerRegistration,
    signer: DatasetAuthorizationSigner,
    clock: DatasetDecisionClock,
) -> DatasetOperationAuthorization:
    """Derive and sign one decision inside the registered issuer boundary."""

    decided_at_utc = clock.now_utc()
    decision_clock_id = clock.clock_id
    decision = _decide_dataset_operation_authorization_at(
        policy=policy,
        request=request,
        authorization_id=authorization_id,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
        decision_clock_id=decision_clock_id,
    )
    if decision.action not in issuer_registration.allowed_actions:
        raise ValueError("dataset authorization issuer is not registered for this action")
    if signer.signature_algorithm is not issuer_registration.signature_algorithm:
        raise ValueError("dataset authorization signer algorithm differs from registration")
    if signer.signature_version != issuer_registration.signature_version:
        raise ValueError("dataset authorization signer version differs from registration")
    if signer.verification_key_hex != issuer_registration.verification_key_hex:
        raise ValueError("dataset authorization signer key differs from registration")
    issued_at_utc = decided_at_utc
    authenticated_issuer = ObjectIdentity.from_record(
        issuer_registration.issuer_registration_id,
        issuer_registration,
    )
    payload = DatasetOperationAuthorizationPayload(
        decision=decision,
        authenticated_issuer=authenticated_issuer,
        issuer_id=issuer_registration.issuer_id,
        issuer_implementation_sha256=issuer_registration.implementation_sha256,
        issuer_implementation_version=issuer_registration.implementation_version,
        signature_algorithm=issuer_registration.signature_algorithm,
        signature_version=issuer_registration.signature_version,
        verification_key_hex=issuer_registration.verification_key_hex,
        issued_at_utc=issued_at_utc,
    )
    signature = signer.sign(payload.canonical_bytes())
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("dataset authorization signer returned a non-Ed25519 signature")
    return DatasetOperationAuthorization(
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


def _operation_refusal_reasons(
    *,
    policy: DatasetOperationPolicy,
    request: DatasetOperationRequest,
    approved_by: str,
    decided_at_utc: str,
) -> set[str]:
    reasons: set[str] = set()
    if request.policy != ObjectIdentity.from_record(policy.policy_id, policy):
        reasons.add("POLICY_IDENTITY_MISMATCH")
    if request.manifest != policy.manifest:
        reasons.add("MANIFEST_IDENTITY_MISMATCH")
    if request.action is not policy.action:
        reasons.add("OPERATION_MISMATCH")
    if request.source_scopes != policy.source_scopes:
        reasons.add("SOURCE_SCOPE_MISMATCH")
    if request.destination_scope != policy.destination_scope:
        reasons.add("DESTINATION_SCOPE_MISMATCH")
    if request.control_write_scope != policy.control_write_scope:
        reasons.add("CONTROL_SCOPE_MISMATCH")
    if not policy.work_envelope_ceiling.contains(request.work_envelope):
        reasons.add("WORK_ENVELOPE_EXCEEDED")
    for field_name in (
        "evidence_class",
        "materialization_class",
        "world_kind",
        "outcome_access",
        "visibility_ceiling",
        "implementation_commit",
    ):
        if getattr(request, field_name) != getattr(policy, field_name):
            reasons.add(f"{field_name.upper()}_MISMATCH")
    policy_start = parse_utc_timestamp(policy.valid_from_utc, field_name="valid_from_utc")
    policy_end = parse_utc_timestamp(policy.valid_until_utc, field_name="valid_until_utc")
    request_start = parse_utc_timestamp(request.valid_from_utc, field_name="valid_from_utc")
    request_end = parse_utc_timestamp(request.valid_until_utc, field_name="valid_until_utc")
    decision_time = parse_utc_timestamp(decided_at_utc, field_name="decided_at_utc")
    if request_start < policy_start or request_end > policy_end:
        reasons.add("REQUEST_VALIDITY_OUTSIDE_POLICY")
    if not policy_start <= decision_time <= policy_end:
        reasons.add("DECISION_OUTSIDE_POLICY_VALIDITY")
    if not request_start <= decision_time <= request_end:
        reasons.add("DECISION_OUTSIDE_REQUEST_VALIDITY")
    if approved_by != policy.authorized_approver_id:
        reasons.add("APPROVER_NOT_AUTHORIZED")
    if request.requested_by == approved_by:
        reasons.add("SELF_APPROVAL_FORBIDDEN")
    if request.source_mutation_requested:
        reasons.add("SOURCE_MUTATION_FORBIDDEN")
    if request.download_requested:
        reasons.add("DOWNLOAD_FORBIDDEN")
    if request.deletion_requested:
        reasons.add("DELETION_FORBIDDEN")

    files = request.work_envelope.files
    resources = request.work_envelope.resources
    if request.action is AuthorityAction.DATASET_REGISTRATION:
        if not request.source_scopes:
            reasons.add("REGISTRATION_SOURCE_REQUIRED")
        if request.destination_scope is not None or files.max_destination_files != 0:
            reasons.add("REGISTRATION_DESTINATION_WRITE_FORBIDDEN")
    elif request.action is AuthorityAction.DATASET_TRANSFORMATION:
        if not request.source_scopes:
            reasons.add("TRANSFORMATION_SOURCE_REQUIRED")
        if request.destination_scope is None or files.max_destination_files == 0:
            reasons.add("TRANSFORMATION_DESTINATION_REQUIRED")
    else:
        if request.source_scopes:
            reasons.add("BINDING_SOURCE_SCOPE_FORBIDDEN")
        if request.destination_scope is not None:
            reasons.add("BINDING_DESTINATION_WRITE_FORBIDDEN")
        if any(
            (
                files.max_source_files,
                files.max_destination_files,
                files.max_archive_members,
                files.max_single_file_bytes,
                resources.source_scan_bytes,
            )
        ):
            reasons.add("BINDING_PAYLOAD_IO_FORBIDDEN")
    return reasons


def _decide_dataset_operation_authorization_at(
    *,
    policy: DatasetOperationPolicy,
    request: DatasetOperationRequest,
    authorization_id: str,
    approved_by: str,
    decided_at_utc: str,
    decision_clock_id: str,
) -> DatasetOperationDecision:
    validate_stable_id(authorization_id, field_name="authorization_id")
    validate_stable_id(approved_by, field_name="approved_by")
    validate_stable_id(decision_clock_id, field_name="decision_clock_id")
    parse_utc_timestamp(decided_at_utc, field_name="decided_at_utc")
    reasons = _operation_refusal_reasons(
        policy=policy,
        request=request,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
    )
    if reasons:
        decision = DatasetAuthorizationDecision.REFUSED
        reasons.add("DATASET_OPERATION_REFUSED")
    else:
        decision = DatasetAuthorizationDecision.APPROVED
        reasons.add("DATASET_OPERATION_APPROVED")
    return DatasetOperationDecision(
        authorization_id=authorization_id,
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        request=ObjectIdentity.from_record(request.request_id, request),
        manifest=request.manifest,
        decision=decision,
        action=request.action,
        source_scopes=request.source_scopes,
        destination_scope=request.destination_scope,
        control_write_scope=request.control_write_scope,
        work_envelope=request.work_envelope,
        evidence_class=request.evidence_class,
        materialization_class=request.materialization_class,
        world_kind=request.world_kind,
        outcome_access=request.outcome_access,
        visibility_ceiling=request.visibility_ceiling,
        implementation_commit=request.implementation_commit,
        requested_by=request.requested_by,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
        decision_clock_id=decision_clock_id,
        valid_from_utc=request.valid_from_utc,
        valid_until_utc=request.valid_until_utc,
        source_mutation_requested=request.source_mutation_requested,
        download_requested=request.download_requested,
        deletion_requested=request.deletion_requested,
        request_reason_codes=request.reason_codes,
        reason_codes=tuple(sorted(reasons)),
        gate_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        plan_mutated=False,
        grants_claim_promotion=False,
    )


def decide_dataset_operation_authorization(
    *,
    policy: DatasetOperationPolicy,
    request: DatasetOperationRequest,
    authorization_id: str,
    approved_by: str,
    clock: DatasetDecisionClock,
) -> DatasetOperationDecision:
    """Preview one unsigned decision using only the composed trusted clock."""

    decision_clock_id = clock.clock_id
    decided_at_utc = clock.now_utc()
    validate_stable_id(decision_clock_id, field_name="decision_clock_id")
    parse_utc_timestamp(decided_at_utc, field_name="decided_at_utc")
    return _decide_dataset_operation_authorization_at(
        policy=policy,
        request=request,
        authorization_id=authorization_id,
        approved_by=approved_by,
        decided_at_utc=decided_at_utc,
        decision_clock_id=decision_clock_id,
    )


def _expected_authorization_bindings(
    request: DatasetOperationRequest,
) -> tuple[tuple[str, object], ...]:
    return (
        ("manifest", request.manifest),
        ("action", request.action),
        ("source_scopes", request.source_scopes),
        ("destination_scope", request.destination_scope),
        ("control_write_scope", request.control_write_scope),
        ("work_envelope", request.work_envelope),
        ("evidence_class", request.evidence_class),
        ("materialization_class", request.materialization_class),
        ("world_kind", request.world_kind),
        ("outcome_access", request.outcome_access),
        ("visibility_ceiling", request.visibility_ceiling),
        ("implementation_commit", request.implementation_commit),
        ("requested_by", request.requested_by),
        ("valid_from_utc", request.valid_from_utc),
        ("valid_until_utc", request.valid_until_utc),
        ("source_mutation_requested", request.source_mutation_requested),
        ("download_requested", request.download_requested),
        ("deletion_requested", request.deletion_requested),
        ("request_reason_codes", request.reason_codes),
    )


def replay_dataset_operation_authorization(
    *,
    policy: DatasetOperationPolicy,
    request: DatasetOperationRequest,
    authorization: DatasetOperationAuthorization,
    issuer_registration: DatasetAuthorizationIssuerRegistration,
    manifest: ObjectIdentity,
    required_action: AuthorityAction,
    implementation_commit: str,
    clock: DatasetDecisionClock,
) -> DatasetOperationRequest:
    """Completely replay a durable dataset decision for one exact use context."""

    if not isinstance(authorization, DatasetOperationAuthorization):
        raise TypeError(
            "dataset operation replay refuses experiment or other authorization records"
        )
    if not isinstance(issuer_registration, DatasetAuthorizationIssuerRegistration):
        raise TypeError("dataset authorization replay requires a trusted issuer registration")
    decision = authorization.decision
    _validate_dataset_action(required_action)
    _validate_implementation_commit(implementation_commit)
    trusted_decision_clock_id = clock.clock_id
    validate_stable_id(trusted_decision_clock_id, field_name="trusted_decision_clock_id")
    use_time = parse_utc_timestamp(clock.now_utc(), field_name="trusted_use_time")
    expected_policy = ObjectIdentity.from_record(policy.policy_id, policy)
    expected_request = ObjectIdentity.from_record(request.request_id, request)
    expected_issuer = ObjectIdentity.from_record(
        issuer_registration.issuer_registration_id,
        issuer_registration,
    )
    if authorization.authenticated_issuer != expected_issuer:
        raise ValueError("dataset authorization replay binds an untrusted issuer registration")
    for trusted_field, trusted_value in (
        ("issuer_id", issuer_registration.issuer_id),
        ("issuer_implementation_sha256", issuer_registration.implementation_sha256),
        ("issuer_implementation_version", issuer_registration.implementation_version),
        ("signature_algorithm", issuer_registration.signature_algorithm),
        ("signature_version", issuer_registration.signature_version),
        ("verification_key_hex", issuer_registration.verification_key_hex),
    ):
        if getattr(authorization, trusted_field) != trusted_value:
            raise ValueError(f"dataset authorization replay changes trusted {trusted_field}")
    if required_action not in issuer_registration.allowed_actions:
        raise ValueError("dataset authorization issuer is not registered for this action")
    _verify_dataset_authorization_signature(
        authorization,
        verification_key_hex=issuer_registration.verification_key_hex,
    )
    if request.policy != expected_policy or decision.policy != expected_policy:
        raise ValueError("dataset authorization replay binds another policy")
    if decision.request != expected_request:
        raise ValueError("dataset authorization replay binds another request")
    if request.manifest != manifest or decision.manifest != manifest:
        raise ValueError("dataset authorization replay binds another manifest")
    if request.action is not required_action or decision.action is not required_action:
        raise ValueError("dataset authorization replay binds another operation")
    if implementation_commit != request.implementation_commit:
        raise ValueError("dataset authorization replay binds another implementation")
    if decision.decision_clock_id != trusted_decision_clock_id:
        raise ValueError("dataset authorization replay binds an untrusted decision clock")
    for binding_field, bound_value in _expected_authorization_bindings(request):
        if getattr(decision, binding_field) != bound_value:
            raise ValueError(f"dataset authorization replay changes {binding_field}")

    replayed = _decide_dataset_operation_authorization_at(
        policy=policy,
        request=request,
        authorization_id=decision.authorization_id,
        approved_by=decision.approved_by,
        decided_at_utc=decision.decided_at_utc,
        decision_clock_id=decision.decision_clock_id,
    )
    if replayed != decision:
        raise ValueError("dataset authorization replay differs from the durable decision")
    if decision.decision is not DatasetAuthorizationDecision.APPROVED:
        raise PermissionError("dataset operation authorization was refused")
    valid_from = parse_utc_timestamp(
        decision.valid_from_utc,
        field_name="valid_from_utc",
    )
    valid_until = parse_utc_timestamp(
        decision.valid_until_utc,
        field_name="valid_until_utc",
    )
    issued_at = parse_utc_timestamp(
        authorization.issued_at_utc,
        field_name="issued_at_utc",
    )
    if use_time < issued_at:
        raise PermissionError("dataset operation authorization has not been issued yet")
    if use_time < valid_from:
        raise PermissionError("dataset operation authorization is not yet valid")
    if use_time > valid_until:
        raise PermissionError("dataset operation authorization has expired")
    return request


__all__ = [
    "DatasetAuthorizationDecision",
    "DatasetAuthorizationIssuerRegistration",
    "DatasetAuthorizationSignatureAlgorithm",
    "DatasetAuthorizationSigner",
    "DatasetControlLimits",
    "DatasetDecisionClock",
    "DatasetFileLimits",
    "DatasetOperationAuthorization",
    "DatasetOperationAuthorizationPayload",
    "DatasetOperationDecision",
    "DatasetOperationPolicy",
    "DatasetOperationRequest",
    "DatasetStorageScope",
    "DatasetWorkEnvelope",
    "decide_dataset_operation_authorization",
    "issue_dataset_operation_authorization",
    "replay_dataset_operation_authorization",
]
