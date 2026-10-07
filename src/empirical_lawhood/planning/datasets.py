"Pure dataset identity, custody, observation, acquisition, and binding contracts.\n\nThis module deliberately contains no persistence, filesystem or provider,\nscientific-scoring, or experiment-execution behavior.  Records containing\nverification results are trusted current records produced by a verifier; they\nare not caller-authoring manifests.\n"

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final, TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp

from .dataset_limits import MAX_DATASET_INTEGER

MAX_DATASET_ID_BYTES: Final[int] = 128
MAX_DATASET_SCHEMA_BYTES: Final[int] = 300
MAX_DATASET_VERSION_BYTES: Final[int] = 40
MAX_DATASET_NAME_BYTES: Final[int] = 256
MAX_DATASET_TEXT_BYTES: Final[int] = 4096
MAX_EXTERNAL_IDENTIFIER_BYTES: Final[int] = 1024
MAX_MEDIA_TYPE_BYTES: Final[int] = 256
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_PROVIDER_OBJECTS: Final[int] = 4096
MAX_REASON_CODES: Final[int] = 128
_T = TypeVar("_T")


class IdentityState(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    FAMILY_RESOLVED = "FAMILY_RESOLVED"
    RELEASE_RESOLVED = "RELEASE_RESOLVED"
    CONFLICT = "CONFLICT"


class DiscoveryState(StrEnum):
    OBSERVED = "OBSERVED"
    CANDIDATE = "CANDIDATE"
    DISMISSED = "DISMISSED"
    WITHDRAWN = "WITHDRAWN"


class CustodyState(StrEnum):
    NOT_PRESENT = "NOT_PRESENT"
    UNVERIFIED = "UNVERIFIED"
    PARTIAL = "PARTIAL"
    VERIFIED = "VERIFIED"
    QUARANTINED = "QUARANTINED"
    INVALID = "INVALID"
    MISSING = "MISSING"


class AccessState(StrEnum):
    PUBLIC = "PUBLIC"
    REGISTRATION = "REGISTRATION"
    CREDENTIALLED = "CREDENTIALLED"
    AGREEMENT = "AGREEMENT"
    PAID = "PAID"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class AcquisitionState(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PREVIEWED = "PREVIEWED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    ALREADY_PRESENT = "ALREADY_PRESENT"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class BindingState(StrEnum):
    PROPOSED = "PROPOSED"
    VERIFIED = "VERIFIED"
    STALE = "STALE"
    INVALID = "INVALID"


class ReleaseResolutionClass(StrEnum):
    PROVIDER_IMMUTABLE_RESOLVED = "PROVIDER_IMMUTABLE_RESOLVED"
    CAPTURED_SNAPSHOT_RESOLVED = "CAPTURED_SNAPSHOT_RESOLVED"
    UNRESOLVED_LOCAL_CUSTODY = "UNRESOLVED_LOCAL_CUSTODY"
    NOT_APPLICABLE_GENERATED = "NOT_APPLICABLE_GENERATED"


class DatasetMaterializationClass(StrEnum):
    AUTHORITATIVE_SOURCE = "AUTHORITATIVE_SOURCE"
    AUXILIARY_SOURCE = "AUXILIARY_SOURCE"
    HISTORICAL_DERIVATIVE = "HISTORICAL_DERIVATIVE"
    TRANSFORMED_DERIVATIVE = "TRANSFORMED_DERIVATIVE"
    SEALED_EVALUATOR_ONLY = "SEALED_EVALUATOR_ONLY"
    SOFTWARE_RUNTIME_MEDIUM = "SOFTWARE_RUNTIME_MEDIUM"
    GENERATED_OBSERVATION = "GENERATED_OBSERVATION"
    METADATA_OBSERVATION = "METADATA_OBSERVATION"
    NON_AUTHORITATIVE_WORKING_COPY = "NON_AUTHORITATIVE_WORKING_COPY"


class DatasetEvidenceClass(StrEnum):
    EMPIRICAL_SOURCE = "EMPIRICAL_SOURCE"
    ANALYTIC_REFERENCE = "ANALYTIC_REFERENCE"
    SIMULATION = "SIMULATION"
    HIL = "HIL"
    PHYSICAL_EXPERIMENT = "PHYSICAL_EXPERIMENT"
    GENERATED_TRUTH = "GENERATED_TRUTH"
    DERIVED = "DERIVED"
    METADATA_ONLY = "METADATA_ONLY"
    SOFTWARE = "SOFTWARE"
    NON_AUTHORITATIVE = "NON_AUTHORITATIVE"


class ExternalIdentifierKind(StrEnum):
    PROVIDER_RELEASE = "PROVIDER_RELEASE"
    ACCESSION = "ACCESSION"
    DEPOSIT = "DEPOSIT"
    REPOSITORY_COMMIT = "REPOSITORY_COMMIT"
    IMMUTABLE_SNAPSHOT = "IMMUTABLE_SNAPSHOT"
    PROVIDER_COLLECTION = "PROVIDER_COLLECTION"
    PROVIDER_OBJECT = "PROVIDER_OBJECT"
    OTHER_REGISTERED = "OTHER_REGISTERED"


class DatasetSelectorKind(StrEnum):
    COMPLETE_RELEASE = "COMPLETE_RELEASE"
    MANIFEST = "MANIFEST"
    PARTITION = "PARTITION"
    SPECIMEN = "SPECIMEN"
    SWEEP = "SWEEP"
    TIME_RANGE = "TIME_RANGE"
    SCENARIO = "SCENARIO"
    COMPOSITE = "COMPOSITE"
    OTHER_REGISTERED = "OTHER_REGISTERED"


class EvidenceReferenceKind(StrEnum):
    MANIFEST = "MANIFEST"
    RECEIPT = "RECEIPT"
    OBSERVATION = "OBSERVATION"
    VERIFICATION = "VERIFICATION"
    AUTHORIZATION = "AUTHORIZATION"
    LICENCE_ASSERTION = "LICENCE_ASSERTION"
    ACCESS_ASSERTION = "ACCESS_ASSERTION"
    TRANSFORM = "TRANSFORM"
    BINDING = "BINDING"
    PREVIEW = "PREVIEW"
    OTHER_REGISTERED = "OTHER_REGISTERED"


class DatasetBindingRole(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    CALIBRATION = "CALIBRATION"
    EVALUATION_SEALED = "EVALUATION_SEALED"
    EVALUATOR_REVEAL = "EVALUATOR_REVEAL"
    AUXILIARY = "AUXILIARY"
    COMPARATOR = "COMPARATOR"
    GENERATED_TRUTH = "GENERATED_TRUTH"
    PARENT = "PARENT"


class ReleaseIdentityRelation(StrEnum):
    SAME_RELEASE = "SAME_RELEASE"
    DIFFERENT_RELEASE = "DIFFERENT_RELEASE"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class ReleaseIdentityBasis(StrEnum):
    INTERNAL_RELEASE_ID = "INTERNAL_RELEASE_ID"
    PROVIDER_IMMUTABLE_IDENTIFIER = "PROVIDER_IMMUTABLE_IDENTIFIER"
    CAPTURED_SNAPSHOT_IDENTITY = "CAPTURED_SNAPSHOT_IDENTITY"
    GENERATED_IDENTITY = "GENERATED_IDENTITY"
    FAMILY_IDENTITY = "FAMILY_IDENTITY"
    NONE = "NONE"


class PhysicalEquality(StrEnum):
    EQUAL = "EQUAL"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


class LogicalEquivalence(StrEnum):
    EQUIVALENT = "EQUIVALENT"
    DIFFERENT = "DIFFERENT"
    INCOMPARABLE = "INCOMPARABLE"
    UNKNOWN = "UNKNOWN"


class AcquisitionDisposition(StrEnum):
    ALREADY_PRESENT = "ALREADY_PRESENT"
    VERIFY_EXISTING = "VERIFY_EXISTING"
    TRANSFORMATION_REQUIRED = "TRANSFORMATION_REQUIRED"
    ACQUISITION_PREVIEW_REQUIRED = "ACQUISITION_PREVIEW_REQUIRED"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"


def _utf8_length(value: str, *, field_name: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8 text") from error


def _validate_id(value: str, *, field_name: str) -> str:
    validate_stable_id(value, field_name=field_name)
    if _utf8_length(value, field_name=field_name) > MAX_DATASET_ID_BYTES:
        raise ValueError(f"{field_name} exceeds its UTF-8 byte limit")
    return value


def _validate_optional_id(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _validate_id(value, field_name=field_name)


def _validate_dataset_schema(value: str, *, field_name: str) -> str:
    validate_schema(value)
    if _utf8_length(value, field_name=field_name) > MAX_DATASET_SCHEMA_BYTES:
        raise ValueError(f"{field_name} exceeds its UTF-8 byte limit")
    return value


def _validate_dataset_version(value: str, *, field_name: str) -> str:
    validate_semantic_version(value)
    if _utf8_length(value, field_name=field_name) > MAX_DATASET_VERSION_BYTES:
        raise ValueError(f"{field_name} exceeds its UTF-8 byte limit")
    return value


def _validate_text(
    value: str,
    *,
    field_name: str,
    maximum_bytes: int = MAX_DATASET_TEXT_BYTES,
    allow_empty: bool = False,
) -> str:
    if allow_empty and value == "":
        return value
    validate_nonempty(value, field_name=field_name)
    if _utf8_length(value, field_name=field_name) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds its UTF-8 byte limit")
    return value


def _require_bounded_strings(
    values: tuple[str, ...],
    *,
    field_name: str,
    maximum_items: int = MAX_COLLECTION_ITEMS,
    maximum_item_bytes: int = MAX_DATASET_ID_BYTES,
    stable_ids: bool = True,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if len(values) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item limit")
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    for value in values:
        if stable_ids:
            _validate_id(value, field_name=field_name)
        else:
            _validate_text(
                value,
                field_name=field_name,
                maximum_bytes=maximum_item_bytes,
            )


def _require_reason_codes(values: tuple[str, ...], *, allow_empty: bool = True) -> None:
    _require_bounded_strings(
        values,
        field_name="reason_codes",
        maximum_items=MAX_REASON_CODES,
        maximum_item_bytes=MAX_DATASET_ID_BYTES,
        stable_ids=False,
        allow_empty=allow_empty,
    )
    for value in values:
        if not value.replace("_", "").isalnum() or value != value.upper() or not value[0].isalpha():
            raise ValueError("reason_codes must use canonical UPPER_SNAKE_CASE")


def _require_sorted_records(
    values: tuple[_T, ...],
    *,
    field_name: str,
    expected_type: type[_T],
    key: Callable[[_T], tuple[str, ...]],
    maximum_items: int = MAX_COLLECTION_ITEMS,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if len(values) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item limit")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, expected_type) for value in values):
        raise ValueError(f"{field_name} contains an invalid record type")
    keys = tuple(key(value) for value in values)
    if tuple(sorted(set(keys))) != keys:
        raise ValueError(f"{field_name} must be sorted and unique")


def _validate_nonnegative_integer(value: int, *, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_DATASET_INTEGER
    ):
        raise ValueError(
            f"{field_name} must be a nonnegative integer no greater than {MAX_DATASET_INTEGER}"
        )
    return value


def _validate_positive_integer(value: int, *, field_name: str) -> int:
    _validate_nonnegative_integer(value, field_name=field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _validate_optional_range(
    minimum: int | None,
    maximum: int | None,
    *,
    field_name: str,
) -> None:
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{field_name} minimum and maximum must be supplied together")
    if minimum is None or maximum is None:
        return
    _validate_nonnegative_integer(minimum, field_name=f"{field_name}_minimum")
    _validate_nonnegative_integer(maximum, field_name=f"{field_name}_maximum")
    if minimum > maximum:
        raise ValueError(f"{field_name} minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class RegisteredImplementationIdentity(CanonicalRecord):
    """Static registered implementation identity; never a callable or import path."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/registered-implementation-identity'

    implementation_key: str
    implementation_version: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        _validate_id(self.implementation_key, field_name="implementation_key")
        _validate_dataset_version(
            self.implementation_version,
            field_name="implementation_version",
        )
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


@dataclass(frozen=True, slots=True)
class ExternalIdentifier(CanonicalRecord):
    """One bounded provider-native identifier with an explicit semantic kind."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/external-identifier'

    kind: ExternalIdentifierKind
    namespace: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ExternalIdentifierKind):
            raise ValueError("kind must be an ExternalIdentifierKind")
        _validate_id(self.namespace, field_name="namespace")
        _validate_text(
            self.value,
            field_name="value",
            maximum_bytes=MAX_EXTERNAL_IDENTIFIER_BYTES,
        )


@dataclass(frozen=True, slots=True)
class DatasetSelectorRef(CanonicalRecord):
    """Digest-bound reference to one registered, bounded selector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-selector-ref'

    selector_id: str
    kind: DatasetSelectorKind
    selector_schema: str
    selector_sha256: str
    complete_release: bool

    def __post_init__(self) -> None:
        _validate_id(self.selector_id, field_name="selector_id")
        if not isinstance(self.kind, DatasetSelectorKind):
            raise ValueError("kind must be a DatasetSelectorKind")
        _validate_dataset_schema(self.selector_schema, field_name="selector_schema")
        validate_sha256(self.selector_sha256, field_name="selector_sha256")
        if not isinstance(self.complete_release, bool):
            raise ValueError("complete_release must be boolean")
        if (self.kind is DatasetSelectorKind.COMPLETE_RELEASE) != self.complete_release:
            raise ValueError("COMPLETE_RELEASE kind and complete_release must agree")


@dataclass(frozen=True, slots=True)
class EvidenceReference(CanonicalRecord):
    """Exact external evidence reference; receipt bodies never enter this record."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-reference'

    evidence_id: str
    kind: EvidenceReferenceKind
    evidence_schema: str
    evidence_sha256: str
    storage_root_id: str
    relative_locator: str

    def __post_init__(self) -> None:
        _validate_id(self.evidence_id, field_name="evidence_id")
        if not isinstance(self.kind, EvidenceReferenceKind):
            raise ValueError("kind must be an EvidenceReferenceKind")
        _validate_dataset_schema(self.evidence_schema, field_name="evidence_schema")
        validate_sha256(self.evidence_sha256, field_name="evidence_sha256")
        _validate_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.relative_locator)


@dataclass(frozen=True, slots=True)
class LogicalContentIdentity(CanonicalRecord):
    """Logical digest valid only under one exact registered decoder and schema."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/logical-content-identity'

    decoder: RegisteredImplementationIdentity
    logical_schema: str
    logical_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.decoder, RegisteredImplementationIdentity):
            raise ValueError("decoder must be a RegisteredImplementationIdentity")
        _validate_dataset_schema(self.logical_schema, field_name="logical_schema")
        validate_sha256(self.logical_sha256, field_name="logical_sha256")


def _validate_external_identifiers(values: tuple[ExternalIdentifier, ...]) -> None:
    _require_sorted_records(
        values,
        field_name="external_identifiers",
        expected_type=ExternalIdentifier,
        key=lambda value: (value.kind.value, value.namespace, value.value),
    )


def _validate_evidence_refs(
    values: tuple[EvidenceReference, ...],
    *,
    field_name: str = "evidence_refs",
    allow_empty: bool = True,
) -> None:
    _require_sorted_records(
        values,
        field_name=field_name,
        expected_type=EvidenceReference,
        key=lambda value: (value.evidence_id,),
        allow_empty=allow_empty,
    )


@dataclass(frozen=True, slots=True)
class DatasetFamily(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-family'

    family_id: str
    canonical_name: str
    provider_id: str | None
    external_identifiers: tuple[ExternalIdentifier, ...]
    description: str
    keywords: tuple[str, ...]
    first_observation_id: str | None
    latest_observation_id: str | None
    identity_state: IdentityState
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.family_id, field_name="family_id")
        _validate_text(
            self.canonical_name,
            field_name="canonical_name",
            maximum_bytes=MAX_DATASET_NAME_BYTES,
        )
        _validate_optional_id(self.provider_id, field_name="provider_id")
        _validate_external_identifiers(self.external_identifiers)
        _validate_text(self.description, field_name="description", allow_empty=True)
        _require_bounded_strings(
            self.keywords,
            field_name="keywords",
            maximum_item_bytes=MAX_DATASET_NAME_BYTES,
            stable_ids=False,
        )
        _validate_optional_id(self.first_observation_id, field_name="first_observation_id")
        _validate_optional_id(self.latest_observation_id, field_name="latest_observation_id")
        if (self.first_observation_id is None) != (self.latest_observation_id is None):
            raise ValueError("first and latest observation IDs must be supplied together")
        if not isinstance(self.identity_state, IdentityState):
            raise ValueError("identity_state must be an IdentityState")
        if self.identity_state is IdentityState.RELEASE_RESOLVED:
            raise ValueError("a dataset family cannot claim release resolution")
        _require_reason_codes(self.reason_codes)
        if (
            self.identity_state in {IdentityState.UNRESOLVED, IdentityState.CONFLICT}
            and not self.reason_codes
        ):
            raise ValueError("unresolved or conflicting family identity requires reason codes")


_IMMUTABLE_RELEASE_KINDS: Final[frozenset[ExternalIdentifierKind]] = frozenset(
    {
        ExternalIdentifierKind.PROVIDER_RELEASE,
        ExternalIdentifierKind.ACCESSION,
        ExternalIdentifierKind.DEPOSIT,
        ExternalIdentifierKind.REPOSITORY_COMMIT,
        ExternalIdentifierKind.IMMUTABLE_SNAPSHOT,
    }
)
_PROVIDER_OBJECT_KINDS: Final[frozenset[ExternalIdentifierKind]] = frozenset(
    {
        ExternalIdentifierKind.PROVIDER_COLLECTION,
        ExternalIdentifierKind.PROVIDER_OBJECT,
    }
)


@dataclass(frozen=True, slots=True)
class DatasetRelease(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-release'

    release_id: str
    family_id: str
    identity_state: IdentityState
    resolution_class: ReleaseResolutionClass
    access_state: AccessState
    provider_id: str | None
    external_identifiers: tuple[ExternalIdentifier, ...]
    local_selector: DatasetSelectorRef | None
    publication_at_utc: str | None
    observed_at_utc: str | None
    mutable_snapshot: bool
    expected_format_profile_ids: tuple[str, ...]
    expected_file_count_minimum: int | None
    expected_file_count_maximum: int | None
    expected_byte_count_minimum: int | None
    expected_byte_count_maximum: int | None
    expected_physical_sha256: str | None
    manifest_sha256: str | None
    licence_evidence: EvidenceReference | None
    access_evidence: EvidenceReference | None
    evidence_refs: tuple[EvidenceReference, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.release_id, field_name="release_id")
        _validate_id(self.family_id, field_name="family_id")
        if not isinstance(self.identity_state, IdentityState):
            raise ValueError("identity_state must be an IdentityState")
        if not isinstance(self.resolution_class, ReleaseResolutionClass):
            raise ValueError("resolution_class must be a ReleaseResolutionClass")
        if not isinstance(self.access_state, AccessState):
            raise ValueError("access_state must be an AccessState")
        _validate_optional_id(self.provider_id, field_name="provider_id")
        _validate_external_identifiers(self.external_identifiers)
        if self.local_selector is not None and not isinstance(
            self.local_selector, DatasetSelectorRef
        ):
            raise ValueError("local_selector must be a DatasetSelectorRef")
        published = (
            parse_utc_timestamp(self.publication_at_utc, field_name="publication_at_utc")
            if self.publication_at_utc is not None
            else None
        )
        observed = (
            parse_utc_timestamp(self.observed_at_utc, field_name="observed_at_utc")
            if self.observed_at_utc is not None
            else None
        )
        if published is not None and observed is not None and published > observed:
            raise ValueError("publication time cannot follow observation time")
        if not isinstance(self.mutable_snapshot, bool):
            raise ValueError("mutable_snapshot must be boolean")
        _require_bounded_strings(
            self.expected_format_profile_ids,
            field_name="expected_format_profile_ids",
        )
        _validate_optional_range(
            self.expected_file_count_minimum,
            self.expected_file_count_maximum,
            field_name="expected_file_count",
        )
        _validate_optional_range(
            self.expected_byte_count_minimum,
            self.expected_byte_count_maximum,
            field_name="expected_byte_count",
        )
        if self.expected_physical_sha256 is not None:
            validate_sha256(
                self.expected_physical_sha256,
                field_name="expected_physical_sha256",
            )
        if self.manifest_sha256 is not None:
            validate_sha256(self.manifest_sha256, field_name="manifest_sha256")
        for field_name, reference in (
            ("licence_evidence", self.licence_evidence),
            ("access_evidence", self.access_evidence),
        ):
            if reference is not None and not isinstance(reference, EvidenceReference):
                raise ValueError(f"{field_name} must be an EvidenceReference")
        if (
            self.licence_evidence is not None
            and self.licence_evidence.kind is not EvidenceReferenceKind.LICENCE_ASSERTION
        ):
            raise ValueError("licence_evidence must be a LICENCE_ASSERTION")
        if (
            self.access_evidence is not None
            and self.access_evidence.kind is not EvidenceReferenceKind.ACCESS_ASSERTION
        ):
            raise ValueError("access_evidence must be an ACCESS_ASSERTION")
        _validate_evidence_refs(self.evidence_refs, allow_empty=False)
        evidence_ids = {reference.evidence_id for reference in self.evidence_refs}
        for field_name, reference in (
            ("licence_evidence", self.licence_evidence),
            ("access_evidence", self.access_evidence),
        ):
            if reference is not None and reference.evidence_id not in evidence_ids:
                raise ValueError(f"{field_name} must also appear in evidence_refs")
        _require_reason_codes(self.reason_codes)
        if self.identity_state is IdentityState.CONFLICT and not self.reason_codes:
            raise ValueError("conflicting release identity requires reason codes")
        if (
            self.access_state in {AccessState.UNAVAILABLE, AccessState.UNKNOWN}
            and not self.reason_codes
        ):
            raise ValueError("unavailable or unknown access requires reason codes")
        self._validate_resolution_class()

    @property
    def immutable_external_identifiers(self) -> tuple[ExternalIdentifier, ...]:
        return tuple(
            identifier
            for identifier in self.external_identifiers
            if identifier.kind in _IMMUTABLE_RELEASE_KINDS
        )

    def _validate_resolution_class(self) -> None:
        immutable_ids = self.immutable_external_identifiers
        immutable_keys = tuple(
            (identifier.kind, identifier.namespace) for identifier in immutable_ids
        )
        if len(set(immutable_keys)) != len(immutable_keys):
            raise ValueError(
                "a resolved release cannot contain conflicting immutable identifiers "
                "of the same kind and namespace"
            )
        provider_objects = tuple(
            identifier
            for identifier in self.external_identifiers
            if identifier.kind in _PROVIDER_OBJECT_KINDS
        )
        if self.resolution_class is ReleaseResolutionClass.PROVIDER_IMMUTABLE_RESOLVED:
            if self.identity_state is not IdentityState.RELEASE_RESOLVED:
                raise ValueError("provider immutable release must be release-resolved")
            if self.provider_id is None or not immutable_ids:
                raise ValueError("provider immutable release requires provider and immutable ID")
            if self.mutable_snapshot:
                raise ValueError("provider immutable release cannot be marked mutable")
        elif self.resolution_class is ReleaseResolutionClass.CAPTURED_SNAPSHOT_RESOLVED:
            if self.identity_state is not IdentityState.RELEASE_RESOLVED:
                raise ValueError("captured snapshot must be release-resolved")
            exact_bytes = (
                self.expected_byte_count_minimum is not None
                and self.expected_byte_count_minimum == self.expected_byte_count_maximum
            )
            if (
                self.provider_id is None
                or not provider_objects
                or self.observed_at_utc is None
                or self.expected_physical_sha256 is None
                or not exact_bytes
                or not self.mutable_snapshot
            ):
                raise ValueError(
                    "captured snapshot requires provider object, retrieval time, exact bytes, "
                    "physical digest, and mutable-snapshot declaration"
                )
            if immutable_ids:
                raise ValueError("captured snapshot cannot also claim a provider immutable release")
        elif self.resolution_class is ReleaseResolutionClass.UNRESOLVED_LOCAL_CUSTODY:
            if self.identity_state is IdentityState.RELEASE_RESOLVED:
                raise ValueError("unresolved local custody cannot claim release resolution")
            if immutable_ids:
                raise ValueError("unresolved local custody cannot retain an immutable release ID")
            if not self.reason_codes:
                raise ValueError("unresolved local custody requires reason codes")
        else:
            if self.identity_state is not IdentityState.RELEASE_RESOLVED:
                raise ValueError("generated release identity must be release-resolved")
            if self.provider_id is not None or self.external_identifiers:
                raise ValueError("generated release cannot claim provider release identity")
            if self.mutable_snapshot:
                raise ValueError("generated release cannot be a mutable provider snapshot")

    def has_complete_expectation_envelope(self) -> bool:
        """Return whether exact local custody can satisfy this release without acquisition.

        This is deliberately stricter than release resolution: it requires one
        exact selector/profile, physical digest, file count and byte count.
        ``manifest_sha256`` is not a payload digest and therefore is not part of
        the physical-satisfaction envelope.
        """

        return (
            self.identity_state is IdentityState.RELEASE_RESOLVED
            and self.local_selector is not None
            and self.local_selector.complete_release
            and self.expected_physical_sha256 is not None
            and len(self.expected_format_profile_ids) == 1
            and self.expected_file_count_minimum is not None
            and self.expected_file_count_minimum == self.expected_file_count_maximum
            and self.expected_byte_count_minimum is not None
            and self.expected_byte_count_minimum == self.expected_byte_count_maximum
        )


_DERIVATIVE_CLASSES: Final[frozenset[DatasetMaterializationClass]] = frozenset(
    {
        DatasetMaterializationClass.HISTORICAL_DERIVATIVE,
        DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
    }
)
_GENERATED_EVIDENCE: Final[frozenset[DatasetEvidenceClass]] = frozenset(
    {
        DatasetEvidenceClass.ANALYTIC_REFERENCE,
        DatasetEvidenceClass.SIMULATION,
        DatasetEvidenceClass.HIL,
        DatasetEvidenceClass.PHYSICAL_EXPERIMENT,
        DatasetEvidenceClass.GENERATED_TRUTH,
    }
)
_ACQUISITION_SATISFYING_CLASSES: Final[frozenset[DatasetMaterializationClass]] = frozenset(
    {
        DatasetMaterializationClass.AUTHORITATIVE_SOURCE,
        DatasetMaterializationClass.AUXILIARY_SOURCE,
        DatasetMaterializationClass.SEALED_EVALUATOR_ONLY,
        DatasetMaterializationClass.SOFTWARE_RUNTIME_MEDIUM,
    }
)


@dataclass(frozen=True, slots=True)
class DatasetMaterializationVerificationSubject(CanonicalRecord):
    """Proof-free subject of one materialization verification receipt.

    The subject deliberately excludes custody, verifier, time, evidence, and
    receipt/manifest locators.  Its fingerprint can therefore be embedded in a
    receipt that is itself referenced by the materialization without creating
    a digest cycle.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-materialization-verification-subject'

    materialization_id: str
    release_id: str
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    storage_root_id: str
    relative_locator: str
    selector: DatasetSelectorRef
    physical_sha256: str | None
    byte_size: int | None
    file_count: int | None
    media_type: str | None
    format_profile_id: str | None
    logical_identity: LogicalContentIdentity | None

    def __post_init__(self) -> None:
        _validate_id(self.materialization_id, field_name="materialization_id")
        _validate_id(self.release_id, field_name="release_id")
        if not isinstance(self.materialization_class, DatasetMaterializationClass):
            raise ValueError("materialization_class must be a DatasetMaterializationClass")
        if not isinstance(self.evidence_class, DatasetEvidenceClass):
            raise ValueError("evidence_class must be a DatasetEvidenceClass")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome_access must be an OutcomeAccess")
        _validate_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.relative_locator)
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        physical_parts = (
            self.physical_sha256 is not None,
            self.byte_size is not None,
            self.file_count is not None,
        )
        if any(physical_parts) and not all(physical_parts):
            raise ValueError("physical digest, byte size, and file count are indivisible")
        if self.physical_sha256 is not None:
            validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.byte_size is not None:
            _validate_nonnegative_integer(self.byte_size, field_name="byte_size")
        if self.file_count is not None:
            _validate_positive_integer(self.file_count, field_name="file_count")
        if self.media_type is not None:
            _validate_text(
                self.media_type,
                field_name="media_type",
                maximum_bytes=MAX_MEDIA_TYPE_BYTES,
            )
        _validate_optional_id(self.format_profile_id, field_name="format_profile_id")
        if self.logical_identity is not None and not isinstance(
            self.logical_identity, LogicalContentIdentity
        ):
            raise ValueError("logical_identity must be a LogicalContentIdentity")


@dataclass(frozen=True, slots=True)
class DatasetMaterializationVerificationObservations(CanonicalRecord):
    """Exact read-only observations asserted by a materialization receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-materialization-verification-observations'

    observed_storage_root_id: str
    observed_relative_locator: str
    observed_selector: DatasetSelectorRef
    observed_physical_sha256: str
    observed_byte_size: int
    observed_file_count: int
    observed_media_type: str | None
    observed_format_profile_id: str | None
    observed_logical_identity: LogicalContentIdentity | None
    complete_eof: bool
    trusted_root: bool
    locator_contained: bool
    read_only: bool

    def __post_init__(self) -> None:
        _validate_id(self.observed_storage_root_id, field_name="observed_storage_root_id")
        validate_relative_locator(self.observed_relative_locator)
        if not isinstance(self.observed_selector, DatasetSelectorRef):
            raise ValueError("observed_selector must be a DatasetSelectorRef")
        validate_sha256(
            self.observed_physical_sha256,
            field_name="observed_physical_sha256",
        )
        _validate_nonnegative_integer(
            self.observed_byte_size,
            field_name="observed_byte_size",
        )
        _validate_positive_integer(
            self.observed_file_count,
            field_name="observed_file_count",
        )
        if self.observed_media_type is not None:
            _validate_text(
                self.observed_media_type,
                field_name="observed_media_type",
                maximum_bytes=MAX_MEDIA_TYPE_BYTES,
            )
        _validate_optional_id(
            self.observed_format_profile_id,
            field_name="observed_format_profile_id",
        )
        if self.observed_logical_identity is not None and not isinstance(
            self.observed_logical_identity, LogicalContentIdentity
        ):
            raise ValueError("observed_logical_identity must be a LogicalContentIdentity")
        for field_name, value in (
            ("complete_eof", self.complete_eof),
            ("trusted_root", self.trusted_root),
            ("locator_contained", self.locator_contained),
            ("read_only", self.read_only),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")

    def validates(self, subject: DatasetMaterializationVerificationSubject) -> bool:
        return (
            isinstance(subject, DatasetMaterializationVerificationSubject)
            and subject.physical_sha256 is not None
            and subject.byte_size is not None
            and subject.file_count is not None
            and self.observed_storage_root_id == subject.storage_root_id
            and self.observed_relative_locator == subject.relative_locator
            and self.observed_selector == subject.selector
            and self.observed_physical_sha256 == subject.physical_sha256
            and self.observed_byte_size == subject.byte_size
            and self.observed_file_count == subject.file_count
            and self.observed_media_type == subject.media_type
            and self.observed_format_profile_id == subject.format_profile_id
            and self.observed_logical_identity == subject.logical_identity
            and self.complete_eof
            and self.trusted_root
            and self.locator_contained
            and self.read_only
        )


@dataclass(frozen=True, slots=True)
class DatasetMaterializationVerificationReceipt(CanonicalRecord):
    """Non-circular receipt proving observations of one exact subject."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-materialization-verification-receipt'

    receipt_id: str
    subject: DatasetMaterializationVerificationSubject
    subject_fingerprint: str
    verifier: RegisteredImplementationIdentity
    verification_policy_id: str
    verification_policy_sha256: str
    verified_at_utc: str
    manifest_evidence: EvidenceReference
    observations: DatasetMaterializationVerificationObservations

    def __post_init__(self) -> None:
        _validate_id(self.receipt_id, field_name="receipt_id")
        if not isinstance(self.subject, DatasetMaterializationVerificationSubject):
            raise ValueError("subject must be a DatasetMaterializationVerificationSubject")
        validate_sha256(self.subject_fingerprint, field_name="subject_fingerprint")
        if self.subject_fingerprint != self.subject.fingerprint():
            raise ValueError("subject_fingerprint must bind the exact verification subject")
        if not isinstance(self.verifier, RegisteredImplementationIdentity):
            raise ValueError("verifier must be a RegisteredImplementationIdentity")
        _validate_id(self.verification_policy_id, field_name="verification_policy_id")
        validate_sha256(
            self.verification_policy_sha256,
            field_name="verification_policy_sha256",
        )
        parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        if not isinstance(self.manifest_evidence, EvidenceReference):
            raise ValueError("manifest_evidence must be an EvidenceReference")
        if self.manifest_evidence.kind is not EvidenceReferenceKind.MANIFEST:
            raise ValueError("manifest_evidence must have MANIFEST kind")
        if self.manifest_evidence.storage_root_id != self.subject.storage_root_id:
            raise ValueError("manifest evidence must use the subject storage root")
        if not isinstance(
            self.observations,
            DatasetMaterializationVerificationObservations,
        ):
            raise ValueError("observations must be DatasetMaterializationVerificationObservations")
        if not self.observations.validates(self.subject):
            raise ValueError("receipt observations differ from the exact subject")

    def validates(
        self,
        subject: DatasetMaterializationVerificationSubject,
        *,
        expected_receipt_id: str,
        expected_verifier: RegisteredImplementationIdentity,
        expected_verification_policy_id: str,
        expected_verification_policy_sha256: str,
        expected_verified_at_utc: str,
        expected_manifest_evidence: EvidenceReference,
    ) -> bool:
        return (
            self.receipt_id == expected_receipt_id
            and self.subject == subject
            and self.subject_fingerprint == subject.fingerprint()
            and self.verifier == expected_verifier
            and self.verification_policy_id == expected_verification_policy_id
            and self.verification_policy_sha256 == expected_verification_policy_sha256
            and self.verified_at_utc == expected_verified_at_utc
            and self.manifest_evidence == expected_manifest_evidence
            and self.observations.validates(subject)
        )


@dataclass(frozen=True, slots=True)
class DatasetMaterialization(CanonicalRecord):
    """Trusted description of one physical materialization of one release."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-materialization'

    materialization_id: str
    release_id: str
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    storage_root_id: str
    relative_locator: str
    selector: DatasetSelectorRef
    physical_sha256: str | None
    byte_size: int | None
    file_count: int | None
    media_type: str | None
    format_profile_id: str | None
    logical_identity: LogicalContentIdentity | None
    custody_state: CustodyState
    verifier: RegisteredImplementationIdentity | None
    verification_policy_id: str | None
    verification_policy_sha256: str | None
    verified_at_utc: str | None
    verification_evidence_refs: tuple[EvidenceReference, ...]
    manifest_relative_locator: str | None
    receipt_relative_locator: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.materialization_id, field_name="materialization_id")
        _validate_id(self.release_id, field_name="release_id")
        if not isinstance(self.materialization_class, DatasetMaterializationClass):
            raise ValueError("materialization_class must be a DatasetMaterializationClass")
        if not isinstance(self.evidence_class, DatasetEvidenceClass):
            raise ValueError("evidence_class must be a DatasetEvidenceClass")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome_access must be an OutcomeAccess")
        _validate_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.relative_locator)
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        self._validate_physical_facts()
        if self.media_type is not None:
            _validate_text(
                self.media_type,
                field_name="media_type",
                maximum_bytes=MAX_MEDIA_TYPE_BYTES,
            )
        _validate_optional_id(self.format_profile_id, field_name="format_profile_id")
        if self.logical_identity is not None and not isinstance(
            self.logical_identity, LogicalContentIdentity
        ):
            raise ValueError("logical_identity must be a LogicalContentIdentity")
        if not isinstance(self.custody_state, CustodyState):
            raise ValueError("custody_state must be a CustodyState")
        if self.verifier is not None and not isinstance(
            self.verifier, RegisteredImplementationIdentity
        ):
            raise ValueError("verifier must be a RegisteredImplementationIdentity")
        _validate_optional_id(
            self.verification_policy_id,
            field_name="verification_policy_id",
        )
        if self.verification_policy_sha256 is not None:
            validate_sha256(
                self.verification_policy_sha256,
                field_name="verification_policy_sha256",
            )
        if self.verified_at_utc is not None:
            parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        _validate_evidence_refs(
            self.verification_evidence_refs,
            field_name="verification_evidence_refs",
        )
        proof_parts = (
            self.verifier is not None,
            self.verification_policy_id is not None,
            self.verification_policy_sha256 is not None,
            self.verified_at_utc is not None,
            bool(self.verification_evidence_refs),
        )
        if any(proof_parts) and not all(proof_parts):
            raise ValueError("verification implementation, time, and evidence are indivisible")
        if all(proof_parts) and not any(
            reference.kind is EvidenceReferenceKind.VERIFICATION
            for reference in self.verification_evidence_refs
        ):
            raise ValueError("verification proof requires typed VERIFICATION evidence")
        if self.logical_identity is not None and not all(proof_parts):
            raise ValueError("logical identity requires complete verifier-derived evidence")
        for field_name, locator in (
            ("manifest_relative_locator", self.manifest_relative_locator),
            ("receipt_relative_locator", self.receipt_relative_locator),
        ):
            if locator is not None:
                validate_relative_locator(locator)
        if (self.manifest_relative_locator is None) != (self.receipt_relative_locator is None):
            raise ValueError("external manifest and receipt locators must be supplied together")
        _require_reason_codes(self.reason_codes)
        self._validate_classification()
        self._validate_custody(proof_complete=all(proof_parts))

    def verification_subject(self) -> DatasetMaterializationVerificationSubject:
        """Return the proof-free, digest-stable subject of receipt verification."""

        return DatasetMaterializationVerificationSubject(
            materialization_id=self.materialization_id,
            release_id=self.release_id,
            materialization_class=self.materialization_class,
            evidence_class=self.evidence_class,
            outcome_access=self.outcome_access,
            storage_root_id=self.storage_root_id,
            relative_locator=self.relative_locator,
            selector=self.selector,
            physical_sha256=self.physical_sha256,
            byte_size=self.byte_size,
            file_count=self.file_count,
            media_type=self.media_type,
            format_profile_id=self.format_profile_id,
            logical_identity=self.logical_identity,
        )

    def _validate_physical_facts(self) -> None:
        physical_parts = (
            self.physical_sha256 is not None,
            self.byte_size is not None,
            self.file_count is not None,
        )
        if any(physical_parts) and not all(physical_parts):
            raise ValueError("physical digest, byte size, and file count are indivisible")
        if self.physical_sha256 is not None:
            validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.byte_size is not None:
            _validate_nonnegative_integer(self.byte_size, field_name="byte_size")
        if self.file_count is not None:
            _validate_positive_integer(self.file_count, field_name="file_count")

    def _validate_classification(self) -> None:
        if (
            self.materialization_class in _DERIVATIVE_CLASSES
            and self.evidence_class is not DatasetEvidenceClass.DERIVED
        ):
            raise ValueError("derivative materializations must retain DERIVED evidence class")
        if (
            self.materialization_class is DatasetMaterializationClass.SOFTWARE_RUNTIME_MEDIUM
            and self.evidence_class is not DatasetEvidenceClass.SOFTWARE
        ):
            raise ValueError("software materialization must retain SOFTWARE evidence class")
        if (
            self.materialization_class is DatasetMaterializationClass.METADATA_OBSERVATION
            and self.evidence_class is not DatasetEvidenceClass.METADATA_ONLY
        ):
            raise ValueError("metadata observation must retain METADATA_ONLY evidence class")
        if (
            self.materialization_class is DatasetMaterializationClass.NON_AUTHORITATIVE_WORKING_COPY
            and self.evidence_class is not DatasetEvidenceClass.NON_AUTHORITATIVE
        ):
            raise ValueError("working copy must retain NON_AUTHORITATIVE evidence class")
        if (
            self.materialization_class is DatasetMaterializationClass.GENERATED_OBSERVATION
            and self.evidence_class not in _GENERATED_EVIDENCE
        ):
            raise ValueError("generated observation cannot masquerade as source or derivative")
        if (
            self.materialization_class is DatasetMaterializationClass.AUTHORITATIVE_SOURCE
            and self.evidence_class
            in {
                DatasetEvidenceClass.DERIVED,
                DatasetEvidenceClass.METADATA_ONLY,
                DatasetEvidenceClass.SOFTWARE,
                DatasetEvidenceClass.NON_AUTHORITATIVE,
            }
        ):
            raise ValueError("authoritative source has an incompatible evidence class")

    def _validate_custody(self, *, proof_complete: bool) -> None:
        has_physical = self.physical_sha256 is not None
        has_external_pair = self.manifest_relative_locator is not None
        if self.custody_state in {CustodyState.VERIFIED, CustodyState.PARTIAL}:
            if not has_physical or not proof_complete or not has_external_pair:
                raise ValueError(
                    f"{self.custody_state.value} custody requires physical facts, proof, "
                    "manifest, and receipt"
                )
            receipt_references = tuple(
                reference
                for reference in self.verification_evidence_refs
                if reference.kind is EvidenceReferenceKind.VERIFICATION
            )
            if (
                len(receipt_references) != 1
                or receipt_references[0].evidence_schema
                != DatasetMaterializationVerificationReceipt.SCHEMA
                or receipt_references[0].storage_root_id != self.storage_root_id
                or receipt_references[0].relative_locator != self.receipt_relative_locator
            ):
                raise ValueError(
                    f"{self.custody_state.value} custody requires one exact typed "
                    "materialization verification receipt reference"
                )
            manifest_references = tuple(
                reference
                for reference in self.verification_evidence_refs
                if reference.kind is EvidenceReferenceKind.MANIFEST
            )
            if (
                len(manifest_references) != 1
                or manifest_references[0].storage_root_id != self.storage_root_id
                or manifest_references[0].relative_locator != self.manifest_relative_locator
            ):
                raise ValueError(
                    f"{self.custody_state.value} custody requires one exact manifest "
                    "evidence reference"
                )
        if self.custody_state is CustodyState.PARTIAL:
            if not has_physical or not proof_complete or not self.reason_codes:
                raise ValueError("PARTIAL custody requires verified scoped facts and reasons")
        if (
            self.custody_state in {CustodyState.QUARANTINED, CustodyState.INVALID}
            and not self.reason_codes
        ):
            raise ValueError("quarantined or invalid custody requires reason codes")
        if self.custody_state in {CustodyState.NOT_PRESENT, CustodyState.MISSING}:
            if (
                has_physical
                or proof_complete
                or self.logical_identity is not None
                or has_external_pair
            ):
                raise ValueError("absent custody cannot retain observed byte or verification facts")
            if self.custody_state is CustodyState.MISSING and not self.reason_codes:
                raise ValueError("MISSING custody requires reason codes")


@dataclass(frozen=True, slots=True)
class CurrentMaterializationVerification(CanonicalRecord):
    """Time-bounded verifier output; trusted only inside registered composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/current-materialization-verification'

    verification_id: str
    materialization_id: str
    materialization_fingerprint: str
    observed_physical_sha256: str
    verifier: RegisteredImplementationIdentity
    verification_policy_id: str
    verification_policy_sha256: str
    verified_at_utc: str
    valid_until_utc: str
    evidence: EvidenceReference
    accessible: bool

    def __post_init__(self) -> None:
        _validate_id(self.verification_id, field_name="verification_id")
        _validate_id(self.materialization_id, field_name="materialization_id")
        validate_sha256(
            self.materialization_fingerprint,
            field_name="materialization_fingerprint",
        )
        validate_sha256(
            self.observed_physical_sha256,
            field_name="observed_physical_sha256",
        )
        if not isinstance(self.verifier, RegisteredImplementationIdentity):
            raise ValueError("verifier must be a RegisteredImplementationIdentity")
        _validate_id(self.verification_policy_id, field_name="verification_policy_id")
        validate_sha256(
            self.verification_policy_sha256,
            field_name="verification_policy_sha256",
        )
        verified_at = parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        valid_until = parse_utc_timestamp(self.valid_until_utc, field_name="valid_until_utc")
        if verified_at >= valid_until:
            raise ValueError("materialization verification validity interval must be positive")
        if not isinstance(self.evidence, EvidenceReference):
            raise ValueError("evidence must be an EvidenceReference")
        if self.evidence.kind is not EvidenceReferenceKind.VERIFICATION:
            raise ValueError("current materialization verification requires verification evidence")
        if not isinstance(self.accessible, bool):
            raise ValueError("accessible must be boolean")

    def validates(self, materialization: DatasetMaterialization) -> bool:
        return (
            self.materialization_id == materialization.materialization_id
            and self.materialization_fingerprint == materialization.fingerprint()
            and self.observed_physical_sha256 == materialization.physical_sha256
        )

    def is_current_at(self, at_utc: str) -> bool:
        at = parse_utc_timestamp(at_utc, field_name="trusted_at_utc")
        verified_at = parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")
        valid_until = parse_utc_timestamp(self.valid_until_utc, field_name="valid_until_utc")
        return self.accessible and verified_at <= at <= valid_until


@dataclass(frozen=True, slots=True)
class DatasetObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-observation'

    observation_id: str
    provider_id: str
    adapter: RegisteredImplementationIdentity
    query_id: str
    canonical_request_sha256: str
    retrieved_at_utc: str
    response_sha256: str
    storage_root_id: str
    response_relative_locator: str
    provider_object_ids: tuple[ExternalIdentifier, ...]
    page_index: int
    predecessor_observation_id: str | None
    pagination_token_sha256: str | None
    truncated: bool
    complete: bool
    rate_limited: bool
    schema_drift: bool
    discovery_state: DiscoveryState
    candidate_release_ids: tuple[str, ...]
    unresolved_lead_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.observation_id, field_name="observation_id")
        _validate_id(self.provider_id, field_name="provider_id")
        if not isinstance(self.adapter, RegisteredImplementationIdentity):
            raise ValueError("adapter must be a RegisteredImplementationIdentity")
        _validate_id(self.query_id, field_name="query_id")
        validate_sha256(self.canonical_request_sha256, field_name="canonical_request_sha256")
        parse_utc_timestamp(self.retrieved_at_utc, field_name="retrieved_at_utc")
        validate_sha256(self.response_sha256, field_name="response_sha256")
        _validate_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.response_relative_locator)
        _require_sorted_records(
            self.provider_object_ids,
            field_name="provider_object_ids",
            expected_type=ExternalIdentifier,
            key=lambda value: (value.kind.value, value.namespace, value.value),
            maximum_items=MAX_PROVIDER_OBJECTS,
        )
        _validate_positive_integer(self.page_index, field_name="page_index")
        _validate_optional_id(
            self.predecessor_observation_id,
            field_name="predecessor_observation_id",
        )
        if self.predecessor_observation_id == self.observation_id:
            raise ValueError("observation cannot name itself as predecessor")
        if self.page_index == 1 and self.predecessor_observation_id is not None:
            raise ValueError("first page cannot name a predecessor")
        if self.page_index > 1 and self.predecessor_observation_id is None:
            raise ValueError("later page requires a predecessor observation")
        if self.pagination_token_sha256 is not None:
            validate_sha256(
                self.pagination_token_sha256,
                field_name="pagination_token_sha256",
            )
        for field_name, value in (
            ("truncated", self.truncated),
            ("complete", self.complete),
            ("rate_limited", self.rate_limited),
            ("schema_drift", self.schema_drift),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.complete and (self.truncated or self.rate_limited or self.schema_drift):
            raise ValueError("complete observation cannot be truncated, limited, or schema-drifted")
        if not isinstance(self.discovery_state, DiscoveryState):
            raise ValueError("discovery_state must be a DiscoveryState")
        _require_bounded_strings(
            self.candidate_release_ids,
            field_name="candidate_release_ids",
        )
        _require_bounded_strings(
            self.unresolved_lead_ids,
            field_name="unresolved_lead_ids",
        )
        if set(self.candidate_release_ids).intersection(self.unresolved_lead_ids):
            raise ValueError("candidate release IDs and unresolved leads must be disjoint")
        if self.discovery_state is DiscoveryState.CANDIDATE and not (
            self.candidate_release_ids or self.unresolved_lead_ids
        ):
            raise ValueError("candidate observation requires a release or unresolved lead")
        _validate_evidence_refs(self.evidence_refs)
        _require_reason_codes(self.reason_codes)
        if (
            not self.complete
            or self.discovery_state in {DiscoveryState.DISMISSED, DiscoveryState.WITHDRAWN}
        ) and not self.reason_codes:
            raise ValueError("incomplete, dismissed, or withdrawn observation requires reasons")


@dataclass(frozen=True, slots=True)
class ExperimentDatasetBinding(CanonicalRecord):
    "Verified, receipt-bound use of one exact materialization by one experiment.\n\n    This executable binding contains the exact identity, lineage, visibility\n    and source-authority inputs needed at the experiment compiler boundary.\n    Earlier incomplete placeholders cannot replace this binding.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-dataset-binding'

    binding_id: str
    experiment_spec_id: str
    experiment: ObjectIdentity
    run_id: str | None
    release_id: str
    materialization_id: str
    materialization: ObjectIdentity
    selector: DatasetSelectorRef
    role: DatasetBindingRole
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    dataset_contract: ObjectIdentity
    transformation_manifest: ObjectIdentity | None
    parent_materializations: tuple[ObjectIdentity, ...]
    transform_reference: EvidenceReference | None
    transform_implementation: RegisteredImplementationIdentity | None
    source_authority: EvidenceReference
    binding_implementation: RegisteredImplementationIdentity
    binding_state: BindingState
    reason_codes: tuple[str, ...]
    binding_receipt: EvidenceReference | None
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("binding_id", self.binding_id),
            ("experiment_spec_id", self.experiment_spec_id),
            ("release_id", self.release_id),
            ("materialization_id", self.materialization_id),
        ):
            _validate_id(value, field_name=field_name)
        if (
            not isinstance(self.experiment, ObjectIdentity)
            or self.experiment.object_id != self.experiment_spec_id
            or self.experiment.object_schema != 'empirical-lawhood/kernel/experiment-spec'
        ):
            raise ValueError("experiment must identify the exact bound ExperimentSpec")
        _validate_optional_id(self.run_id, field_name="run_id")
        if (
            not isinstance(self.materialization, ObjectIdentity)
            or self.materialization.object_id != self.materialization_id
            or self.materialization.object_schema != DatasetMaterialization.SCHEMA
        ):
            raise ValueError("materialization must identify the exact bound materialization")
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        if not isinstance(self.role, DatasetBindingRole):
            raise ValueError("role must be a DatasetBindingRole")
        if not isinstance(self.materialization_class, DatasetMaterializationClass):
            raise ValueError("materialization_class must be a DatasetMaterializationClass")
        if not isinstance(self.evidence_class, DatasetEvidenceClass):
            raise ValueError("evidence_class must be a DatasetEvidenceClass")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome_access must be an OutcomeAccess")
        if not isinstance(self.visibility_ceiling, VisibilityCeiling):
            raise ValueError("visibility_ceiling must be a VisibilityCeiling")
        if (
            not isinstance(self.dataset_contract, ObjectIdentity)
            or self.dataset_contract.object_schema != 'empirical-lawhood/planning/dataset-output-contract'
        ):
            raise ValueError("dataset_contract must identify a DatasetOutputContract")
        if self.transformation_manifest is not None and (
            not isinstance(self.transformation_manifest, ObjectIdentity)
            or self.transformation_manifest.object_schema
            != 'empirical-lawhood/planning/dataset-transformation-manifest'
        ):
            raise ValueError(
                "transformation_manifest must identify a DatasetTransformationManifest"
            )
        if any(
            not isinstance(value, ObjectIdentity)
            or value.object_schema != DatasetMaterialization.SCHEMA
            for value in self.parent_materializations
        ):
            raise ValueError("parent_materializations must identify dataset materializations")
        parent_ids = tuple(value.object_id for value in self.parent_materializations)
        if tuple(sorted(set(parent_ids))) != parent_ids:
            raise ValueError("parent_materializations must be sorted and unique")
        if self.materialization_id in parent_ids:
            raise ValueError("a binding materialization cannot parent itself")
        if (
            self.materialization_class
            in _DERIVATIVE_CLASSES
            | {
                DatasetMaterializationClass.GENERATED_OBSERVATION,
            }
            and not self.parent_materializations
        ):
            raise ValueError("derived or generated binding requires exact parent materializations")
        for field_name, reference in (
            ("transform_reference", self.transform_reference),
            ("binding_receipt", self.binding_receipt),
            ("source_authority", self.source_authority),
        ):
            if reference is not None and not isinstance(reference, EvidenceReference):
                raise ValueError(f"{field_name} must be an EvidenceReference")
        if self.transform_implementation is not None and not isinstance(
            self.transform_implementation, RegisteredImplementationIdentity
        ):
            raise ValueError("transform_implementation has the wrong type")
        if (self.transform_reference is None) != (self.transform_implementation is None):
            raise ValueError("transform reference and implementation must be supplied together")
        if (
            self.transform_reference is not None
            and self.transform_reference.kind is not EvidenceReferenceKind.TRANSFORM
        ):
            raise ValueError("transform_reference must be typed TRANSFORM evidence")
        if (
            self.binding_receipt is not None
            and self.binding_receipt.kind is not EvidenceReferenceKind.BINDING
        ):
            raise ValueError("binding_receipt must be typed BINDING evidence")
        if self.source_authority.kind is not EvidenceReferenceKind.RECEIPT:
            raise ValueError("source_authority must be typed RECEIPT evidence")
        if not isinstance(self.binding_implementation, RegisteredImplementationIdentity):
            raise ValueError("binding_implementation has the wrong type")
        if not isinstance(self.binding_state, BindingState):
            raise ValueError("binding_state must be a BindingState")
        _require_reason_codes(self.reason_codes)
        _validate_evidence_refs(self.evidence_refs)
        evidence_ids = {reference.evidence_id for reference in self.evidence_refs}
        for field_name, reference in (
            ("transform_reference", self.transform_reference),
            ("binding_receipt", self.binding_receipt),
            ("source_authority", self.source_authority),
        ):
            if reference is not None and reference.evidence_id not in evidence_ids:
                raise ValueError(f"{field_name} must also appear in evidence_refs")
        if self.binding_state is BindingState.VERIFIED:
            if self.binding_receipt is None or not self.evidence_refs:
                raise ValueError("VERIFIED binding requires receipt-bound evidence")
        elif self.binding_receipt is not None:
            raise ValueError("only VERIFIED binding may name a binding receipt")
        if (
            self.binding_state in {BindingState.STALE, BindingState.INVALID}
            and not self.reason_codes
        ):
            raise ValueError("stale or invalid binding requires reason codes")
        if (
            self.role is DatasetBindingRole.EVALUATION_SEALED
            and self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("sealed evaluation role requires sealed outcome access")
        if (
            self.role is DatasetBindingRole.EVALUATOR_REVEAL
            and self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("evaluator reveal role requires evaluator-reveal access")

    def validate_target(
        self,
        release: DatasetRelease,
        materialization: DatasetMaterialization,
    ) -> None:
        """Check exact catalog references without making a scientific-use claim."""

        if release.release_id != self.release_id:
            raise ValueError("binding names a different release")
        if materialization.materialization_id != self.materialization_id:
            raise ValueError("binding names a different materialization")
        if ObjectIdentity.from_record(materialization.materialization_id, materialization) != (
            self.materialization
        ):
            raise ValueError("binding materialization identity is stale or substituted")
        if materialization.release_id != release.release_id:
            raise ValueError("materialization belongs to a different release")
        if materialization.custody_state is not CustodyState.VERIFIED:
            raise ValueError("binding target lacks VERIFIED custody")
        if materialization.selector != self.selector:
            raise ValueError("binding selector differs from the verified materialization selector")
        if materialization.outcome_access is not self.outcome_access:
            raise ValueError("binding cannot restate or lower materialization outcome access")
        if materialization.materialization_class is not self.materialization_class:
            raise ValueError("binding materialization class differs from its target")
        if materialization.evidence_class is not self.evidence_class:
            raise ValueError("binding evidence class differs from its target")


@dataclass(frozen=True, slots=True)
class AcquisitionAttempt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/acquisition-attempt'

    attempt_id: str
    release_id: str
    provider_capability_key: str
    source_endpoint_id: str
    preview: EvidenceReference
    expected_physical_sha256: str | None
    expected_size_bytes: int
    expected_format_profile_ids: tuple[str, ...]
    authorization: EvidenceReference | None
    authorization_valid_from_utc: str | None
    authorization_valid_until_utc: str | None
    requested_bytes: int
    observed_bytes: int
    storage_root_id: str
    temporary_relative_locator: str | None
    final_relative_locator: str | None
    retry_count: int
    resumed_from_attempt_id: str | None
    acquisition_state: AcquisitionState
    result_materialization_id: str | None
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("attempt_id", self.attempt_id),
            ("release_id", self.release_id),
            ("provider_capability_key", self.provider_capability_key),
            ("source_endpoint_id", self.source_endpoint_id),
            ("storage_root_id", self.storage_root_id),
        ):
            _validate_id(value, field_name=field_name)
        if not isinstance(self.preview, EvidenceReference):
            raise ValueError("preview must be an EvidenceReference")
        if self.preview.kind is not EvidenceReferenceKind.PREVIEW:
            raise ValueError("preview must be typed PREVIEW evidence")
        if self.expected_physical_sha256 is not None:
            validate_sha256(
                self.expected_physical_sha256,
                field_name="expected_physical_sha256",
            )
        _validate_nonnegative_integer(self.expected_size_bytes, field_name="expected_size_bytes")
        _require_bounded_strings(
            self.expected_format_profile_ids,
            field_name="expected_format_profile_ids",
            allow_empty=False,
        )
        if self.authorization is not None and not isinstance(self.authorization, EvidenceReference):
            raise ValueError("authorization must be an EvidenceReference")
        if (
            self.authorization is not None
            and self.authorization.kind is not EvidenceReferenceKind.AUTHORIZATION
        ):
            raise ValueError("authorization must be typed AUTHORIZATION evidence")
        validity_values = (
            self.authorization_valid_from_utc,
            self.authorization_valid_until_utc,
        )
        if self.authorization is None:
            if any(value is not None for value in validity_values):
                raise ValueError("authorization validity requires an authorization")
        else:
            if any(value is None for value in validity_values):
                raise ValueError("authorization requires a complete validity interval")
            assert self.authorization_valid_from_utc is not None
            assert self.authorization_valid_until_utc is not None
            valid_from = parse_utc_timestamp(
                self.authorization_valid_from_utc,
                field_name="authorization_valid_from_utc",
            )
            valid_until = parse_utc_timestamp(
                self.authorization_valid_until_utc,
                field_name="authorization_valid_until_utc",
            )
            if valid_from >= valid_until:
                raise ValueError("authorization validity interval must be positive")
        _validate_nonnegative_integer(self.requested_bytes, field_name="requested_bytes")
        _validate_nonnegative_integer(self.observed_bytes, field_name="observed_bytes")
        for field_name, locator in (
            ("temporary_relative_locator", self.temporary_relative_locator),
            ("final_relative_locator", self.final_relative_locator),
        ):
            if locator is not None:
                validate_relative_locator(locator)
        _validate_nonnegative_integer(self.retry_count, field_name="retry_count")
        _validate_optional_id(
            self.resumed_from_attempt_id,
            field_name="resumed_from_attempt_id",
        )
        if (self.retry_count > 0) != (self.resumed_from_attempt_id is not None):
            raise ValueError("retry count and resumed-from attempt must agree")
        if self.resumed_from_attempt_id == self.attempt_id:
            raise ValueError("acquisition attempt cannot resume itself")
        if not isinstance(self.acquisition_state, AcquisitionState):
            raise ValueError("acquisition_state must be an AcquisitionState")
        if self.acquisition_state is AcquisitionState.NOT_REQUESTED:
            raise ValueError("NOT_REQUESTED does not create an acquisition attempt")
        _validate_optional_id(
            self.result_materialization_id,
            field_name="result_materialization_id",
        )
        _require_reason_codes(self.reason_codes)
        checksum_unavailable = "UPSTREAM_CHECKSUM_UNAVAILABLE" in self.reason_codes
        if (self.expected_physical_sha256 is None) != checksum_unavailable:
            raise ValueError(
                "missing expected checksum and UPSTREAM_CHECKSUM_UNAVAILABLE must agree"
            )
        _validate_evidence_refs(self.evidence_refs, allow_empty=False)
        evidence_ids = {reference.evidence_id for reference in self.evidence_refs}
        if self.preview.evidence_id not in evidence_ids:
            raise ValueError("preview must also appear in evidence_refs")
        if self.authorization is not None and self.authorization.evidence_id not in evidence_ids:
            raise ValueError("authorization must also appear in evidence_refs")
        self._validate_acquisition_state()

    def validate_result(self, materialization: DatasetMaterialization) -> None:
        """Validate a verifier-derived result without inventing an upstream checksum."""

        if self.acquisition_state not in {
            AcquisitionState.ALREADY_PRESENT,
            AcquisitionState.COMPLETED,
        }:
            raise ValueError("only a successful acquisition attempt can validate a result")
        if self.result_materialization_id != materialization.materialization_id:
            raise ValueError("acquisition result names a different materialization")
        if self.release_id != materialization.release_id:
            raise ValueError("acquisition result belongs to a different release")
        if materialization.custody_state is not CustodyState.VERIFIED:
            raise ValueError("acquisition result must have VERIFIED custody")
        if materialization.storage_root_id != self.storage_root_id:
            raise ValueError("acquisition result differs from the preview storage root")
        if materialization.byte_size != self.expected_size_bytes:
            raise ValueError("acquisition result differs from the expected size")
        if materialization.format_profile_id not in self.expected_format_profile_ids:
            raise ValueError("acquisition result differs from the expected format profile")
        if (
            self.expected_physical_sha256 is not None
            and materialization.physical_sha256 != self.expected_physical_sha256
        ):
            raise ValueError("acquisition result differs from the expected physical digest")
        if self.acquisition_state is AcquisitionState.COMPLETED:
            if materialization.relative_locator != self.final_relative_locator:
                raise ValueError("acquisition result differs from the committed final locator")
            if self.observed_bytes != self.expected_size_bytes:
                raise ValueError("completed transfer bytes differ from the preview expected size")

    def _validate_acquisition_state(self) -> None:
        if self.observed_bytes > self.requested_bytes:
            raise ValueError("observed bytes cannot exceed the requested byte ceiling")
        if self.acquisition_state is AcquisitionState.ALREADY_PRESENT:
            if (
                self.result_materialization_id is None
                or self.requested_bytes != 0
                or self.observed_bytes != 0
                or self.temporary_relative_locator is not None
                or self.final_relative_locator is not None
                or self.authorization is not None
            ):
                raise ValueError("ALREADY_PRESENT must name existing custody and write zero bytes")
            if not self.reason_codes:
                raise ValueError("ALREADY_PRESENT requires an exact-satisfaction reason")
            return
        if self.acquisition_state in {
            AcquisitionState.PREVIEWED,
            AcquisitionState.AUTHORITY_REQUIRED,
        }:
            if (
                self.observed_bytes != 0
                or self.authorization is not None
                or self.temporary_relative_locator is not None
                or self.final_relative_locator is not None
                or self.result_materialization_id is not None
            ):
                raise ValueError("preview/authority stop cannot retain transfer effects")
            if (
                self.acquisition_state is AcquisitionState.AUTHORITY_REQUIRED
                and not self.reason_codes
            ):
                raise ValueError("authority stop requires reason codes")
            return
        if (
            self.acquisition_state in {AcquisitionState.RUNNING, AcquisitionState.COMPLETED}
            and self.authorization is None
        ):
            raise ValueError("running or completed acquisition requires authorization")
        if self.acquisition_state is AcquisitionState.RUNNING:
            if (
                self.temporary_relative_locator is None
                or self.final_relative_locator is not None
                or self.result_materialization_id is not None
            ):
                raise ValueError("running acquisition requires scratch and no final result")
        if self.acquisition_state is AcquisitionState.COMPLETED:
            if (
                self.result_materialization_id is None
                or self.final_relative_locator is None
                or self.observed_bytes == 0
            ):
                raise ValueError("completed acquisition requires bytes, final locator, and result")
            if self.observed_bytes != self.expected_size_bytes:
                raise ValueError("completed acquisition bytes must equal the preview expected size")
        if self.acquisition_state in {AcquisitionState.FAILED, AcquisitionState.BLOCKED}:
            if not self.reason_codes:
                raise ValueError("failed or blocked acquisition requires reason codes")
            if (
                self.result_materialization_id is not None
                or self.final_relative_locator is not None
            ):
                raise ValueError("failed or blocked acquisition cannot retain a final result")


@dataclass(frozen=True, slots=True)
class ReleaseIdentityComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/release-identity-comparison'

    left_release_id: str
    right_release_id: str
    relation: ReleaseIdentityRelation
    basis: ReleaseIdentityBasis
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.left_release_id, field_name="left_release_id")
        _validate_id(self.right_release_id, field_name="right_release_id")
        if not isinstance(self.relation, ReleaseIdentityRelation):
            raise ValueError("relation must be a ReleaseIdentityRelation")
        if not isinstance(self.basis, ReleaseIdentityBasis):
            raise ValueError("basis must be a ReleaseIdentityBasis")
        _require_reason_codes(self.reason_codes, allow_empty=False)


@dataclass(frozen=True, slots=True)
class ReleaseResolution(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/release-resolution'

    requested_release_id: str
    relation: ReleaseIdentityRelation
    basis: ReleaseIdentityBasis
    compared_release_ids: tuple[str, ...]
    matching_release_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.requested_release_id, field_name="requested_release_id")
        if not isinstance(self.relation, ReleaseIdentityRelation):
            raise ValueError("relation must be a ReleaseIdentityRelation")
        if not isinstance(self.basis, ReleaseIdentityBasis):
            raise ValueError("basis must be a ReleaseIdentityBasis")
        _require_bounded_strings(
            self.compared_release_ids,
            field_name="compared_release_ids",
        )
        _require_bounded_strings(
            self.matching_release_ids,
            field_name="matching_release_ids",
        )
        if not set(self.matching_release_ids).issubset(self.compared_release_ids):
            raise ValueError("matching releases must be included in compared releases")
        if (
            self.relation is ReleaseIdentityRelation.SAME_RELEASE
            and len(self.matching_release_ids) != 1
        ):
            raise ValueError("same-release resolution requires exactly one match")
        if self.relation is not ReleaseIdentityRelation.SAME_RELEASE and self.matching_release_ids:
            raise ValueError("only same-release resolution may select a match")
        _require_reason_codes(self.reason_codes, allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterializationComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/materialization-comparison'

    left_materialization_id: str
    right_materialization_id: str
    release_identity: ReleaseIdentityComparison
    physical_equality: PhysicalEquality
    logical_equivalence: LogicalEquivalence

    def __post_init__(self) -> None:
        _validate_id(self.left_materialization_id, field_name="left_materialization_id")
        _validate_id(self.right_materialization_id, field_name="right_materialization_id")
        if not isinstance(self.release_identity, ReleaseIdentityComparison):
            raise ValueError("release_identity must be a ReleaseIdentityComparison")
        if not isinstance(self.physical_equality, PhysicalEquality):
            raise ValueError("physical_equality must be a PhysicalEquality")
        if not isinstance(self.logical_equivalence, LogicalEquivalence):
            raise ValueError("logical_equivalence must be a LogicalEquivalence")


@dataclass(frozen=True, slots=True)
class AcquisitionDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/acquisition-decision'

    requested_release_id: str
    disposition: AcquisitionDisposition
    selected_materialization_id: str | None
    considered_materialization_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    network_bytes: int = 0

    def __post_init__(self) -> None:
        _validate_id(self.requested_release_id, field_name="requested_release_id")
        if not isinstance(self.disposition, AcquisitionDisposition):
            raise ValueError("disposition must be an AcquisitionDisposition")
        _validate_optional_id(
            self.selected_materialization_id,
            field_name="selected_materialization_id",
        )
        _require_bounded_strings(
            self.considered_materialization_ids,
            field_name="considered_materialization_ids",
        )
        if (
            self.selected_materialization_id is not None
            and self.selected_materialization_id not in self.considered_materialization_ids
        ):
            raise ValueError("selected materialization must have been considered")
        if self.disposition is AcquisitionDisposition.ALREADY_PRESENT:
            if self.selected_materialization_id is None:
                raise ValueError("ALREADY_PRESENT requires one selected materialization")
        elif self.selected_materialization_id is not None:
            raise ValueError("only ALREADY_PRESENT may select a materialization")
        _require_reason_codes(self.reason_codes, allow_empty=False)
        _validate_nonnegative_integer(self.network_bytes, field_name="network_bytes")
        if self.network_bytes != 0:
            raise ValueError("pure acquisition decisions cannot perform network transfer")


def _immutable_identifier_keys(release: DatasetRelease) -> frozenset[tuple[str, str, str]]:
    return frozenset(
        (identifier.kind.value, identifier.namespace, identifier.value)
        for identifier in release.immutable_external_identifiers
    )


def _immutable_identifiers_by_namespace(
    release: DatasetRelease,
) -> dict[tuple[str, str], frozenset[str]]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for identifier in release.immutable_external_identifiers:
        key = (identifier.kind.value, identifier.namespace)
        grouped.setdefault(key, set()).add(identifier.value)
    return {key: frozenset(values) for key, values in grouped.items()}


def compare_release_identity(
    left: DatasetRelease,
    right: DatasetRelease,
) -> ReleaseIdentityComparison:
    """Compare release identity without using names, paths, bytes, or similarity."""

    if not isinstance(left, DatasetRelease) or not isinstance(right, DatasetRelease):
        raise TypeError("release comparison requires DatasetRelease records")
    if left.release_id == right.release_id:
        if left == right:
            return ReleaseIdentityComparison(
                left_release_id=left.release_id,
                right_release_id=right.release_id,
                relation=ReleaseIdentityRelation.SAME_RELEASE,
                basis=ReleaseIdentityBasis.INTERNAL_RELEASE_ID,
                reason_codes=("EXACT_INTERNAL_RELEASE_RECORD",),
            )
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.AMBIGUOUS,
            basis=ReleaseIdentityBasis.INTERNAL_RELEASE_ID,
            reason_codes=("CONFLICTING_RECORDS_FOR_RELEASE_ID",),
        )
    if left.family_id != right.family_id:
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.DIFFERENT_RELEASE,
            basis=ReleaseIdentityBasis.FAMILY_IDENTITY,
            reason_codes=("DIFFERENT_DATASET_FAMILY",),
        )
    if IdentityState.CONFLICT in {left.identity_state, right.identity_state}:
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.AMBIGUOUS,
            basis=ReleaseIdentityBasis.NONE,
            reason_codes=("RELEASE_IDENTITY_CONFLICT",),
        )
    if (
        left.resolution_class is ReleaseResolutionClass.PROVIDER_IMMUTABLE_RESOLVED
        and right.resolution_class is ReleaseResolutionClass.PROVIDER_IMMUTABLE_RESOLVED
    ):
        left_ids = _immutable_identifier_keys(left)
        right_ids = _immutable_identifier_keys(right)
        if left.provider_id != right.provider_id:
            return ReleaseIdentityComparison(
                left_release_id=left.release_id,
                right_release_id=right.release_id,
                relation=ReleaseIdentityRelation.AMBIGUOUS,
                basis=ReleaseIdentityBasis.PROVIDER_IMMUTABLE_IDENTIFIER,
                reason_codes=("PROVIDER_IDENTITY_CONFLICT",),
            )
        left_by_namespace = _immutable_identifiers_by_namespace(left)
        right_by_namespace = _immutable_identifiers_by_namespace(right)
        comparable_keys = set(left_by_namespace).intersection(right_by_namespace)
        conflicting_keys = {
            key for key in comparable_keys if left_by_namespace[key] != right_by_namespace[key]
        }
        if left_ids.intersection(right_ids) and conflicting_keys:
            return ReleaseIdentityComparison(
                left_release_id=left.release_id,
                right_release_id=right.release_id,
                relation=ReleaseIdentityRelation.AMBIGUOUS,
                basis=ReleaseIdentityBasis.PROVIDER_IMMUTABLE_IDENTIFIER,
                reason_codes=("CONFLICTING_IMMUTABLE_IDENTIFIER_SET",),
            )
        if left_ids.intersection(right_ids):
            return ReleaseIdentityComparison(
                left_release_id=left.release_id,
                right_release_id=right.release_id,
                relation=ReleaseIdentityRelation.SAME_RELEASE,
                basis=ReleaseIdentityBasis.PROVIDER_IMMUTABLE_IDENTIFIER,
                reason_codes=("MATCHING_PROVIDER_IMMUTABLE_IDENTIFIER",),
            )
        if not comparable_keys:
            return ReleaseIdentityComparison(
                left_release_id=left.release_id,
                right_release_id=right.release_id,
                relation=ReleaseIdentityRelation.AMBIGUOUS,
                basis=ReleaseIdentityBasis.PROVIDER_IMMUTABLE_IDENTIFIER,
                reason_codes=("INCOMPARABLE_IMMUTABLE_IDENTIFIER_KINDS",),
            )
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.DIFFERENT_RELEASE,
            basis=ReleaseIdentityBasis.PROVIDER_IMMUTABLE_IDENTIFIER,
            reason_codes=("DISTINCT_PROVIDER_IMMUTABLE_IDENTIFIERS",),
        )
    if (
        left.resolution_class is ReleaseResolutionClass.CAPTURED_SNAPSHOT_RESOLVED
        and right.resolution_class is ReleaseResolutionClass.CAPTURED_SNAPSHOT_RESOLVED
    ):
        same_capture = (
            left.provider_id == right.provider_id
            and left.external_identifiers == right.external_identifiers
            and left.observed_at_utc == right.observed_at_utc
            and left.expected_byte_count_minimum == right.expected_byte_count_minimum
            and left.expected_physical_sha256 == right.expected_physical_sha256
        )
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=(
                ReleaseIdentityRelation.SAME_RELEASE
                if same_capture
                else ReleaseIdentityRelation.DIFFERENT_RELEASE
            ),
            basis=ReleaseIdentityBasis.CAPTURED_SNAPSHOT_IDENTITY,
            reason_codes=(
                "MATCHING_CAPTURED_SNAPSHOT_IDENTITY"
                if same_capture
                else "DISTINCT_CAPTURED_SNAPSHOT_IDENTITY",
            ),
        )
    if (
        left.resolution_class is ReleaseResolutionClass.NOT_APPLICABLE_GENERATED
        and right.resolution_class is ReleaseResolutionClass.NOT_APPLICABLE_GENERATED
    ):
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.DIFFERENT_RELEASE,
            basis=ReleaseIdentityBasis.GENERATED_IDENTITY,
            reason_codes=("DISTINCT_GENERATED_RELEASE_IDENTITIES",),
        )
    if (
        left.identity_state is not IdentityState.RELEASE_RESOLVED
        or right.identity_state is not IdentityState.RELEASE_RESOLVED
    ):
        return ReleaseIdentityComparison(
            left_release_id=left.release_id,
            right_release_id=right.release_id,
            relation=ReleaseIdentityRelation.UNRESOLVED,
            basis=ReleaseIdentityBasis.NONE,
            reason_codes=("RELEASE_IDENTITY_UNRESOLVED",),
        )
    return ReleaseIdentityComparison(
        left_release_id=left.release_id,
        right_release_id=right.release_id,
        relation=ReleaseIdentityRelation.DIFFERENT_RELEASE,
        basis=ReleaseIdentityBasis.NONE,
        reason_codes=("DISTINCT_RESOLVED_RELEASE_CLASSES",),
    )


def resolve_release_identity(
    requested: DatasetRelease,
    candidates: tuple[DatasetRelease, ...],
) -> ReleaseResolution:
    """Resolve one request against bounded candidates without heuristic merging."""

    if not isinstance(requested, DatasetRelease):
        raise TypeError("requested must be a DatasetRelease")
    if not isinstance(candidates, tuple):
        raise ValueError("candidates must be a tuple")
    if len(candidates) > MAX_COLLECTION_ITEMS:
        raise ValueError("candidates exceeds its item limit")
    if any(not isinstance(candidate, DatasetRelease) for candidate in candidates):
        raise ValueError("candidates contains an invalid record type")
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.release_id))
    candidate_ids = tuple(candidate.release_id for candidate in ordered)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidates contains duplicate release IDs")
    if requested.identity_state is IdentityState.CONFLICT:
        return ReleaseResolution(
            requested_release_id=requested.release_id,
            relation=ReleaseIdentityRelation.AMBIGUOUS,
            basis=ReleaseIdentityBasis.NONE,
            compared_release_ids=candidate_ids,
            matching_release_ids=(),
            reason_codes=("REQUESTED_RELEASE_IDENTITY_CONFLICT",),
        )
    if requested.identity_state is not IdentityState.RELEASE_RESOLVED:
        return ReleaseResolution(
            requested_release_id=requested.release_id,
            relation=ReleaseIdentityRelation.UNRESOLVED,
            basis=ReleaseIdentityBasis.NONE,
            compared_release_ids=candidate_ids,
            matching_release_ids=(),
            reason_codes=("REQUESTED_RELEASE_IDENTITY_UNRESOLVED",),
        )
    comparisons = tuple(compare_release_identity(requested, candidate) for candidate in ordered)
    matches = tuple(
        comparison.right_release_id
        for comparison in comparisons
        if comparison.relation is ReleaseIdentityRelation.SAME_RELEASE
    )
    has_ambiguity = any(
        comparison.relation is ReleaseIdentityRelation.AMBIGUOUS for comparison in comparisons
    )
    if len(matches) > 1 or has_ambiguity:
        return ReleaseResolution(
            requested_release_id=requested.release_id,
            relation=ReleaseIdentityRelation.AMBIGUOUS,
            basis=ReleaseIdentityBasis.NONE,
            compared_release_ids=candidate_ids,
            matching_release_ids=(),
            reason_codes=("AMBIGUOUS_RELEASE_MATCH",),
        )
    if len(matches) == 1:
        match = next(
            comparison
            for comparison in comparisons
            if comparison.relation is ReleaseIdentityRelation.SAME_RELEASE
        )
        return ReleaseResolution(
            requested_release_id=requested.release_id,
            relation=ReleaseIdentityRelation.SAME_RELEASE,
            basis=match.basis,
            compared_release_ids=candidate_ids,
            matching_release_ids=matches,
            reason_codes=("EXACT_RELEASE_MATCH",),
        )
    return ReleaseResolution(
        requested_release_id=requested.release_id,
        relation=ReleaseIdentityRelation.DIFFERENT_RELEASE,
        basis=ReleaseIdentityBasis.NONE,
        compared_release_ids=candidate_ids,
        matching_release_ids=(),
        reason_codes=("NO_EXACT_RELEASE_MATCH",),
    )


def compare_materializations(
    left_release: DatasetRelease,
    left: DatasetMaterialization,
    right_release: DatasetRelease,
    right: DatasetMaterialization,
) -> MaterializationComparison:
    """Return separate release, physical-byte, and logical-content facts."""

    if left.release_id != left_release.release_id or right.release_id != right_release.release_id:
        raise ValueError("materialization comparison received a mismatched release")
    release_comparison = compare_release_identity(left_release, right_release)
    if left.physical_sha256 is None or right.physical_sha256 is None:
        physical = PhysicalEquality.UNKNOWN
    elif left.physical_sha256 == right.physical_sha256:
        physical = PhysicalEquality.EQUAL
    else:
        physical = PhysicalEquality.DIFFERENT
    if left.logical_identity is None or right.logical_identity is None:
        logical = LogicalEquivalence.UNKNOWN
    elif (
        left.logical_identity.decoder != right.logical_identity.decoder
        or left.logical_identity.logical_schema != right.logical_identity.logical_schema
    ):
        logical = LogicalEquivalence.INCOMPARABLE
    elif left.logical_identity.logical_sha256 == right.logical_identity.logical_sha256:
        logical = LogicalEquivalence.EQUIVALENT
    else:
        logical = LogicalEquivalence.DIFFERENT
    return MaterializationComparison(
        left_materialization_id=left.materialization_id,
        right_materialization_id=right.materialization_id,
        release_identity=release_comparison,
        physical_equality=physical,
        logical_equivalence=logical,
    )


def _release_expectation_mismatches(
    release: DatasetRelease,
    materialization: DatasetMaterialization,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        release.expected_physical_sha256 is not None
        and materialization.physical_sha256 != release.expected_physical_sha256
    ):
        reasons.append("EXPECTED_PHYSICAL_DIGEST_MISMATCH")
    byte_minimum = release.expected_byte_count_minimum
    byte_maximum = release.expected_byte_count_maximum
    if (
        byte_minimum is not None
        and byte_maximum is not None
        and materialization.byte_size is not None
        and not byte_minimum <= materialization.byte_size <= byte_maximum
    ):
        reasons.append("EXPECTED_BYTE_COUNT_MISMATCH")
    file_minimum = release.expected_file_count_minimum
    file_maximum = release.expected_file_count_maximum
    if (
        file_minimum is not None
        and file_maximum is not None
        and materialization.file_count is not None
        and not file_minimum <= materialization.file_count <= file_maximum
    ):
        reasons.append("EXPECTED_FILE_COUNT_MISMATCH")
    if (
        release.expected_format_profile_ids
        and materialization.format_profile_id not in release.expected_format_profile_ids
    ):
        reasons.append("EXPECTED_FORMAT_PROFILE_MISMATCH")
    return tuple(sorted(reasons))


def decide_acquisition(
    requested_release: DatasetRelease,
    materializations: tuple[DatasetMaterialization, ...],
    *,
    required_selector: DatasetSelectorRef | None = None,
    expected_physical_sha256: str | None = None,
    expected_logical_identity: LogicalContentIdentity | None = None,
    required_format_profile_id: str | None = None,
    current_verifications: tuple[CurrentMaterializationVerification, ...] = (),
) -> AcquisitionDecision:
    """Choose a pure zero-effect disposition for one exact release.

    This boundary compares only identity and recorded custody.  It can nominate
    ``VERIFY_EXISTING`` but can never authenticate ``ALREADY_PRESENT``.  A
    caller-supplied current-verification record is rejected even when it is
    structurally valid; only the registered runtime service may resolve and call
    an implementation, replay its external evidence, and promote the result.
    Physical or logical equality never substitutes for release identity.
    """

    if not isinstance(requested_release, DatasetRelease):
        raise TypeError("requested_release must be a DatasetRelease")
    if not isinstance(materializations, tuple):
        raise ValueError("materializations must be a tuple")
    if len(materializations) > MAX_COLLECTION_ITEMS:
        raise ValueError("materializations exceeds its item limit")
    if any(not isinstance(value, DatasetMaterialization) for value in materializations):
        raise ValueError("materializations contains an invalid record type")
    ordered = tuple(sorted(materializations, key=lambda value: value.materialization_id))
    materialization_ids = tuple(value.materialization_id for value in ordered)
    if len(set(materialization_ids)) != len(materialization_ids):
        raise ValueError("materializations contains duplicate materialization IDs")
    if required_selector is not None and not isinstance(required_selector, DatasetSelectorRef):
        raise ValueError("required_selector must be a DatasetSelectorRef")
    if expected_physical_sha256 is not None:
        validate_sha256(expected_physical_sha256, field_name="expected_physical_sha256")
    if expected_logical_identity is not None and not isinstance(
        expected_logical_identity, LogicalContentIdentity
    ):
        raise ValueError("expected_logical_identity must be a LogicalContentIdentity")
    if required_format_profile_id is not None:
        _validate_id(required_format_profile_id, field_name="required_format_profile_id")
    if not isinstance(current_verifications, tuple):
        raise ValueError("current_verifications must be a tuple")
    if current_verifications:
        raise ValueError("caller-supplied current verification cannot authorize acquisition")
    if requested_release.identity_state is IdentityState.CONFLICT:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.IDENTITY_CONFLICT,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("REQUESTED_RELEASE_IDENTITY_CONFLICT",),
        )
    if requested_release.identity_state is not IdentityState.RELEASE_RESOLVED:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.IDENTITY_UNRESOLVED,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("REQUESTED_RELEASE_IDENTITY_UNRESOLVED",),
        )
    if not requested_release.has_complete_expectation_envelope():
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.ACQUISITION_PREVIEW_REQUIRED,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("RELEASE_EXPECTATION_ENVELOPE_INCOMPLETE",),
        )
    exact_release = tuple(
        value for value in ordered if value.release_id == requested_release.release_id
    )
    if not exact_release:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.ACQUISITION_PREVIEW_REQUIRED,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=(
                "DIFFERENT_RELEASE_PRESERVED"
                if materialization_ids
                else "NO_LOCAL_MATERIALIZATION",
            ),
        )
    # The release envelope carries the exact selector identity that was
    # registered. Completeness alone cannot make a different selection satisfy
    # the no-redownload decision.
    target_selector = (
        required_selector if required_selector is not None else requested_release.local_selector
    )
    if target_selector is None:  # guarded by has_complete_expectation_envelope()
        raise RuntimeError("complete release expectation lost its exact selector")
    selector_matches = tuple(value for value in exact_release if value.selector == target_selector)
    if not selector_matches:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.ACQUISITION_PREVIEW_REQUIRED,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("EXACT_SELECTOR_NOT_PRESENT",),
        )
    verified = tuple(
        value
        for value in selector_matches
        if value.custody_state is CustodyState.VERIFIED
        and value.materialization_class in _ACQUISITION_SATISFYING_CLASSES
    )
    mismatch_reasons = tuple(
        sorted(
            {
                reason
                for value in verified
                for reason in _release_expectation_mismatches(requested_release, value)
            }
        )
    )
    if mismatch_reasons:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.IDENTITY_CONFLICT,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=mismatch_reasons,
        )
    expectation_matches = verified
    digest_matches = tuple(
        value
        for value in expectation_matches
        if expected_physical_sha256 is None or value.physical_sha256 == expected_physical_sha256
    )
    if (
        expected_physical_sha256 is not None
        and expectation_matches
        and len(digest_matches) != len(expectation_matches)
    ):
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.IDENTITY_CONFLICT,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("VERIFIED_PHYSICAL_DIGEST_MISMATCH",),
        )
    logical_matches = tuple(
        value
        for value in digest_matches
        if expected_logical_identity is None or value.logical_identity == expected_logical_identity
    )
    if (
        expected_logical_identity is not None
        and digest_matches
        and len(logical_matches) != len(digest_matches)
    ):
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.IDENTITY_CONFLICT,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("VERIFIED_LOGICAL_IDENTITY_MISMATCH",),
        )
    format_matches = tuple(
        value
        for value in logical_matches
        if required_format_profile_id is None
        or value.format_profile_id == required_format_profile_id
    )
    if required_format_profile_id is not None and logical_matches and not format_matches:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.TRANSFORMATION_REQUIRED,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("EXACT_RELEASE_REQUIRES_REQUESTED_FORMAT",),
        )
    if format_matches:
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.VERIFY_EXISTING,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("FRESH_REVALIDATION_REQUIRED",),
        )
    if any(value.custody_state is CustodyState.UNVERIFIED for value in selector_matches):
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.VERIFY_EXISTING,
            selected_materialization_id=None,
            considered_materialization_ids=materialization_ids,
            reason_codes=("EXACT_RELEASE_REQUIRES_VERIFICATION",),
        )
    return AcquisitionDecision(
        requested_release_id=requested_release.release_id,
        disposition=AcquisitionDisposition.ACQUISITION_PREVIEW_REQUIRED,
        selected_materialization_id=None,
        considered_materialization_ids=materialization_ids,
        reason_codes=("NO_EXACT_VERIFIED_SATISFACTION",),
    )


def acquisition_revalidation_candidates(
    requested_release: DatasetRelease,
    materializations: tuple[DatasetMaterialization, ...],
    *,
    required_selector: DatasetSelectorRef | None = None,
    expected_physical_sha256: str | None = None,
    expected_logical_identity: LogicalContentIdentity | None = None,
    required_format_profile_id: str | None = None,
) -> tuple[DatasetMaterialization, ...]:
    """Return deterministic candidates that require registered revalidation.

    The returned records are nominations, not verification results.  The helper
    deliberately returns nothing for unverified custody, conflicts, incomplete
    release expectations, or any disposition other than the exact
    ``FRESH_REVALIDATION_REQUIRED`` path.
    """

    decision = decide_acquisition(
        requested_release,
        materializations,
        required_selector=required_selector,
        expected_physical_sha256=expected_physical_sha256,
        expected_logical_identity=expected_logical_identity,
        required_format_profile_id=required_format_profile_id,
    )
    if (
        decision.disposition is not AcquisitionDisposition.VERIFY_EXISTING
        or decision.reason_codes != ("FRESH_REVALIDATION_REQUIRED",)
    ):
        return ()
    target_selector = (
        required_selector if required_selector is not None else requested_release.local_selector
    )
    if target_selector is None:
        return ()
    return tuple(
        value
        for value in sorted(
            materializations,
            key=lambda candidate: candidate.materialization_id,
        )
        if value.release_id == requested_release.release_id
        and value.selector == target_selector
        and value.custody_state is CustodyState.VERIFIED
        and value.materialization_class in _ACQUISITION_SATISFYING_CLASSES
        and not _release_expectation_mismatches(requested_release, value)
        and (expected_physical_sha256 is None or value.physical_sha256 == expected_physical_sha256)
        and (
            expected_logical_identity is None or value.logical_identity == expected_logical_identity
        )
        and (
            required_format_profile_id is None
            or value.format_profile_id == required_format_profile_id
        )
    )


__all__ = [
    "AccessState",
    "AcquisitionAttempt",
    "AcquisitionDecision",
    "AcquisitionDisposition",
    "AcquisitionState",
    "BindingState",
    "CurrentMaterializationVerification",
    "CustodyState",
    "DatasetBindingRole",
    "DatasetEvidenceClass",
    "DatasetFamily",
    "DatasetMaterialization",
    "DatasetMaterializationClass",
    "DatasetMaterializationVerificationObservations",
    "DatasetMaterializationVerificationReceipt",
    "DatasetMaterializationVerificationSubject",
    "DatasetObservation",
    "DatasetRelease",
    "DatasetSelectorKind",
    "DatasetSelectorRef",
    "DiscoveryState",
    "EvidenceReference",
    "EvidenceReferenceKind",
    "ExperimentDatasetBinding",
    "ExternalIdentifier",
    "ExternalIdentifierKind",
    "IdentityState",
    "LogicalContentIdentity",
    "LogicalEquivalence",
    "MAX_DATASET_INTEGER",
    "MAX_DATASET_SCHEMA_BYTES",
    "MAX_DATASET_VERSION_BYTES",
    "MaterializationComparison",
    "PhysicalEquality",
    "RegisteredImplementationIdentity",
    "ReleaseIdentityBasis",
    "ReleaseIdentityComparison",
    "ReleaseIdentityRelation",
    "ReleaseResolution",
    "ReleaseResolutionClass",
    "acquisition_revalidation_candidates",
    "compare_materializations",
    "compare_release_identity",
    "decide_acquisition",
    "resolve_release_identity",
]
