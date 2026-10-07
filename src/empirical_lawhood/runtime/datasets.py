"""Infrastructure-neutral dataset capabilities, catalog snapshots, and ports.

This module owns runtime composition contracts only.  It deliberately contains
no filesystem, network, SQL, provider-specific, or transform implementation.
Static registrations are canonical and fingerprinted; executable objects live
only in the separate, in-process :class:`DatasetImplementationRegistry`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, TypeAlias, TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import DatasetOperationAuthorization
from empirical_lawhood.planning.dataset_rebuild import DatasetProjectionRebuildAuthorization
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityBinding,
    DatasetCapabilityKind,
    ProposedExperimentDatasetBindingManifest,
)
from empirical_lawhood.planning.datasets import (
    AcquisitionAttempt,
    AcquisitionState,
    BindingState,
    CurrentMaterializationVerification,
    CustodyState,
    DatasetBindingRole,
    DatasetFamily,
    DatasetMaterialization,
    DatasetMaterializationVerificationReceipt,
    DatasetMaterializationVerificationSubject,
    DatasetObservation,
    DatasetRelease,
    EvidenceReference,
    EvidenceReferenceKind,
    ExperimentDatasetBinding,
    ExternalIdentifier,
    MAX_DATASET_INTEGER,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.catalog import CatalogSnapshot
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)


MAX_DATASET_CAPABILITY_REGISTRATIONS: Final[int] = 4096
MAX_DATASET_CAPABILITY_BYTES: Final[int] = 1024**5
MAX_DATASET_CAPABILITY_RECORDS: Final[int] = 10_000_000
MAX_DATASET_CAPABILITY_FILES: Final[int] = 10_000_000
MAX_DATASET_CAPABILITY_ARCHIVE_MEMBERS: Final[int] = 10_000_000
MAX_DATASET_CATALOG_RECORDS: Final[int] = 100_000
MAX_DATASET_CATALOG_CANONICAL_BYTES: Final[int] = 512 * 1024**2
MAX_DATASET_QUERY_FILTER_ITEMS: Final[int] = 256
MAX_DATASET_PAGE_RECORDS: Final[int] = 256
MAX_DATASET_QUERY_STEPS: Final[int] = 1_000_000
MAX_DATASET_RUNTIME_ID_BYTES: Final[int] = 128
MAX_DATASET_MEDIA_TYPE_BYTES: Final[int] = 256
_DATASET_BINDING_MANIFEST_SCHEMA: Final[str] = (
    'empirical-lawhood/planning/proposed-experiment-dataset-binding-manifest'
)
_DATASET_BINDING_RECORD_SCHEMA: Final[str] = 'empirical-lawhood/planning/experiment-dataset-binding'
_DATASET_EVIDENCE_VERIFICATION_SCHEMA: Final[str] = (
    'empirical-lawhood/runtime/dataset-evidence-verification'
)


def _utf8_length(value: str, *, field_name: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8 text") from error


def _validate_runtime_id(value: str, *, field_name: str) -> None:
    validate_stable_id(value, field_name=field_name)
    if _utf8_length(value, field_name=field_name) > MAX_DATASET_RUNTIME_ID_BYTES:
        raise ValueError(f"{field_name} exceeds its UTF-8 byte limit")


def _validate_media_types(values: tuple[str, ...], *, field_name: str) -> None:
    require_sorted_unique_strings(values, field_name=field_name)
    if len(values) > MAX_DATASET_QUERY_FILTER_ITEMS:
        raise ValueError(f"{field_name} exceeds its item limit")
    for value in values:
        validate_nonempty(value, field_name=field_name)
        if _utf8_length(value, field_name=field_name) > MAX_DATASET_MEDIA_TYPE_BYTES:
            raise ValueError(f"{field_name} contains an oversized media type")
        if "/" not in value or any(character.isspace() for character in value):
            raise ValueError(f"{field_name} must contain normalized media types")


def _validate_schemas(values: tuple[str, ...], *, field_name: str) -> None:
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
    if len(values) > MAX_DATASET_QUERY_FILTER_ITEMS:
        raise ValueError(f"{field_name} exceeds its item limit")
    for value in values:
        validate_schema(value)


def _validate_profile_ids(values: tuple[str, ...], *, field_name: str) -> None:
    require_sorted_unique_strings(values, field_name=field_name)
    if len(values) > MAX_DATASET_QUERY_FILTER_ITEMS:
        raise ValueError(f"{field_name} exceeds its item limit")
    for value in values:
        _validate_runtime_id(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class DatasetCapabilityLimits(CanonicalRecord):
    """Hard ceilings enforced by an implementation outside this module."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-capability-limits'

    max_input_bytes: int
    max_output_bytes: int
    max_records: int
    max_files: int
    max_archive_members: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_input_bytes", self.max_input_bytes),
            ("max_output_bytes", self.max_output_bytes),
            ("max_records", self.max_records),
            ("max_files", self.max_files),
            ("max_archive_members", self.max_archive_members),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        for field_name, value, maximum in (
            ("max_input_bytes", self.max_input_bytes, MAX_DATASET_CAPABILITY_BYTES),
            ("max_output_bytes", self.max_output_bytes, MAX_DATASET_CAPABILITY_BYTES),
            ("max_records", self.max_records, MAX_DATASET_CAPABILITY_RECORDS),
            ("max_files", self.max_files, MAX_DATASET_CAPABILITY_FILES),
            (
                "max_archive_members",
                self.max_archive_members,
                MAX_DATASET_CAPABILITY_ARCHIVE_MEMBERS,
            ),
        ):
            if value > maximum:
                raise ValueError(f"{field_name} exceeds its absolute maximum")
        if not any(
            (
                self.max_input_bytes,
                self.max_output_bytes,
                self.max_records,
                self.max_files,
                self.max_archive_members,
            )
        ):
            raise ValueError("a dataset capability must admit some bounded work")


def _validate_registration_common(
    *,
    capability: CapabilityManifest,
    expected_kind: CapabilityKind,
    limits: DatasetCapabilityLimits,
) -> None:
    if not isinstance(capability, CapabilityManifest):
        raise ValueError("capability must be CapabilityManifest")
    if capability.kind is not expected_kind:
        raise ValueError(f"dataset registration requires {expected_kind.value} capability kind")
    _validate_schemas(capability.input_schema_ids, field_name="input_schema_ids")
    _validate_schemas(capability.output_schema_ids, field_name="output_schema_ids")
    if not isinstance(limits, DatasetCapabilityLimits):
        raise ValueError("limits must be DatasetCapabilityLimits")
    if limits.max_input_bytes > capability.resource_ceiling.source_scan_bytes:
        raise ValueError("dataset input limit exceeds the capability resource ceiling")
    if limits.max_output_bytes > capability.resource_ceiling.output_bytes:
        raise ValueError("dataset output limit exceeds the capability resource ceiling")


@dataclass(frozen=True, slots=True)
class DatasetProviderRegistration(CanonicalRecord):
    """Static metadata for one bounded provider observation adapter."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-provider-registration'

    provider_id: str
    capability: CapabilityManifest
    accepted_media_types: tuple[str, ...]
    produced_media_types: tuple[str, ...]
    limits: DatasetCapabilityLimits

    @property
    def registry_id(self) -> str:
        return self.capability.registry_id

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_runtime_id(self.provider_id, field_name="provider_id")
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.SOURCE,
            limits=self.limits,
        )
        _validate_media_types(self.accepted_media_types, field_name="accepted_media_types")
        _validate_media_types(self.produced_media_types, field_name="produced_media_types")


@dataclass(frozen=True, slots=True)
class DatasetFormatInspectorRegistration(CanonicalRecord):
    """Static metadata for one safe inspector and current verifier pair."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-format-inspector-registration'

    capability: CapabilityManifest
    accepted_media_types: tuple[str, ...]
    supported_format_profile_ids: tuple[str, ...]
    limits: DatasetCapabilityLimits

    @property
    def registry_id(self) -> str:
        return self.capability.registry_id

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.SOURCE,
            limits=self.limits,
        )
        _validate_media_types(self.accepted_media_types, field_name="accepted_media_types")
        _validate_profile_ids(
            self.supported_format_profile_ids,
            field_name="supported_format_profile_ids",
        )


@dataclass(frozen=True, slots=True)
class DatasetTransformRegistration(CanonicalRecord):
    """Static metadata for one exact source-to-destination transform."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-transform-registration'

    capability: CapabilityManifest
    accepted_media_types: tuple[str, ...]
    produced_media_types: tuple[str, ...]
    source_format_profile_ids: tuple[str, ...]
    destination_format_profile_ids: tuple[str, ...]
    limits: DatasetCapabilityLimits

    @property
    def registry_id(self) -> str:
        return self.capability.registry_id

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.TRANSFORM,
            limits=self.limits,
        )
        _validate_media_types(self.accepted_media_types, field_name="accepted_media_types")
        _validate_media_types(self.produced_media_types, field_name="produced_media_types")
        _validate_profile_ids(
            self.source_format_profile_ids,
            field_name="source_format_profile_ids",
        )
        _validate_profile_ids(
            self.destination_format_profile_ids,
            field_name="destination_format_profile_ids",
        )
        if not self.source_format_profile_ids or not self.destination_format_profile_ids:
            raise ValueError("a transform must bind source and destination format profiles")


@dataclass(frozen=True, slots=True)
class DatasetBindingCompilerRegistration(CanonicalRecord):
    """Static metadata for one exact proposed-binding compiler role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-binding-compiler-registration'

    capability: CapabilityManifest
    limits: DatasetCapabilityLimits

    @property
    def registry_id(self) -> str:
        return self.capability.registry_id

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.TRANSFORM,
            limits=self.limits,
        )
        if _DATASET_BINDING_MANIFEST_SCHEMA not in self.capability.input_schema_ids:
            raise ValueError("binding compiler must accept the exact binding-manifest schema")
        if _DATASET_BINDING_RECORD_SCHEMA not in self.capability.output_schema_ids:
            raise ValueError("binding compiler must produce the exact dataset-binding schema")


@dataclass(frozen=True, slots=True)
class DatasetStorageVerificationRegistration(CanonicalRecord):
    """Static storage-root verification policy selected by a bounded key."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-storage-verification-registration'

    capability: CapabilityManifest
    mount_contract_schema_ids: tuple[str, ...]
    allowed_filesystem_types: tuple[str, ...]
    limits: DatasetCapabilityLimits

    @property
    def registry_id(self) -> str:
        return self.capability.registry_id

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.SOURCE,
            limits=self.limits,
        )
        _validate_schemas(
            self.mount_contract_schema_ids,
            field_name="mount_contract_schema_ids",
        )
        require_sorted_unique_strings(
            self.allowed_filesystem_types,
            field_name="allowed_filesystem_types",
            allow_empty=False,
        )
        if len(self.allowed_filesystem_types) > MAX_DATASET_QUERY_FILTER_ITEMS:
            raise ValueError("allowed_filesystem_types exceeds its item limit")
        for value in self.allowed_filesystem_types:
            _validate_runtime_id(value, field_name="allowed_filesystem_types")


@dataclass(frozen=True, slots=True)
class DatasetEvidenceVerifierRegistration(CanonicalRecord):
    """Static policy for bounded replay of canonical external evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-evidence-verifier-registration'

    registry_id: str
    capability: CapabilityManifest
    supported_evidence_schema_ids: tuple[str, ...]
    limits: DatasetCapabilityLimits

    @property
    def capability_key(self) -> str:
        return self.capability.capability_key

    @property
    def capability_version(self) -> str:
        return self.capability.capability_version

    @property
    def implementation_sha256(self) -> str:
        return self.capability.implementation_sha256

    def __post_init__(self) -> None:
        _validate_runtime_id(self.registry_id, field_name="registry_id")
        _validate_registration_common(
            capability=self.capability,
            expected_kind=CapabilityKind.SOURCE,
            limits=self.limits,
        )
        _validate_schemas(
            self.supported_evidence_schema_ids,
            field_name="supported_evidence_schema_ids",
        )
        if self.capability.input_schema_ids != self.supported_evidence_schema_ids:
            raise ValueError(
                "evidence verifier capability inputs must exactly reproduce supported schemas"
            )
        if self.capability.output_schema_ids != (_DATASET_EVIDENCE_VERIFICATION_SCHEMA,):
            raise ValueError(
                "evidence verifier must produce only the exact verification-result schema"
            )
        if self.capability.permissions != (CapabilityPermission.READ_EXTERNAL_ARTIFACTS,):
            raise ValueError("evidence verifier receives only external read permission")
        if self.capability.maximum_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("evidence verifier must remain outcome-blind")
        if (
            not self.capability.deterministic
            or self.capability.seed_required
            or not self.capability.requires_active_mount
            or self.capability.requires_network
        ):
            raise ValueError(
                "evidence verifier must be deterministic, mount-bound, and network-free"
            )
        if (
            self.limits.max_input_bytes <= 0
            or self.limits.max_output_bytes <= 0
            or self.limits.max_files <= 0
        ):
            raise ValueError("evidence verifier requires positive byte and file limits")
        if self.capability.resource_ceiling.source_scan_bytes < 2 * self.limits.max_input_bytes:
            raise ValueError(
                "evidence verifier source-scan ceiling must admit two stable input passes"
            )


@dataclass(frozen=True, slots=True)
class DatasetCapabilityRegistry(CanonicalRecord):
    """Canonical static registrations; executable objects are never serialized."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-capability-registry'

    registry_id: str
    providers: tuple[DatasetProviderRegistration, ...]
    inspectors: tuple[DatasetFormatInspectorRegistration, ...]
    transforms: tuple[DatasetTransformRegistration, ...]
    binding_compilers: tuple[DatasetBindingCompilerRegistration, ...]
    storage_verifiers: tuple[DatasetStorageVerificationRegistration, ...]
    evidence_verifiers: tuple[DatasetEvidenceVerifierRegistration, ...]

    def __post_init__(self) -> None:
        _validate_runtime_id(self.registry_id, field_name="registry_id")
        count = sum(
            len(values)
            for values in (
                self.providers,
                self.inspectors,
                self.transforms,
                self.binding_compilers,
                self.storage_verifiers,
                self.evidence_verifiers,
            )
        )
        if count == 0:
            raise ValueError("dataset capability registry must not be empty")
        if count > MAX_DATASET_CAPABILITY_REGISTRATIONS:
            raise ValueError("dataset capability registry exceeds its aggregate limit")
        for values, field_name in (
            (self.providers, "providers"),
            (self.inspectors, "inspectors"),
            (self.transforms, "transforms"),
            (self.binding_compilers, "binding_compilers"),
            (self.storage_verifiers, "storage_verifiers"),
            (self.evidence_verifiers, "evidence_verifiers"),
        ):
            require_sorted_unique_ids(
                values,
                attribute="registry_id",
                field_name=field_name,
            )
        all_registry_ids = tuple(
            value.registry_id
            for values in (
                self.providers,
                self.inspectors,
                self.transforms,
                self.binding_compilers,
                self.storage_verifiers,
                self.evidence_verifiers,
            )
            for value in values
        )
        if len(set(all_registry_ids)) != len(all_registry_ids):
            raise ValueError("dataset capability registry IDs must be globally unique")

    def provider(self, registry_id: str) -> DatasetProviderRegistration:
        return _resolve_registration(self.providers, registry_id, "provider")

    def inspector(self, registry_id: str) -> DatasetFormatInspectorRegistration:
        return _resolve_registration(self.inspectors, registry_id, "inspector")

    def transform(self, registry_id: str) -> DatasetTransformRegistration:
        return _resolve_registration(self.transforms, registry_id, "transform")

    def binding_compiler(
        self,
        registry_id: str,
    ) -> DatasetBindingCompilerRegistration:
        return _resolve_registration(
            self.binding_compilers,
            registry_id,
            "binding compiler",
        )

    def storage_verifier(
        self,
        registry_id: str,
    ) -> DatasetStorageVerificationRegistration:
        return _resolve_registration(self.storage_verifiers, registry_id, "storage verifier")

    def evidence_verifier(
        self,
        registry_id: str,
    ) -> DatasetEvidenceVerifierRegistration:
        return _resolve_registration(self.evidence_verifiers, registry_id, "evidence verifier")


@dataclass(frozen=True, slots=True)
class DatasetSupportingIdentityRegistry(CanonicalRecord):
    """Frozen non-executable config, runtime, and environment identities."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-supporting-identity-registry'

    registry_id: str
    configs: tuple[ObjectIdentity, ...]
    runtimes: tuple[ObjectIdentity, ...]
    environments: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _validate_runtime_id(self.registry_id, field_name="registry_id")
        expected_schemas = (
            (self.configs, "configs", 'empirical-lawhood/runtime/capability-config-ref'),
            (self.runtimes, "runtimes", 'empirical-lawhood/kernel/artifact-identity'),
            (
                self.environments,
                "environments",
                'empirical-lawhood/kernel/artifact-identity',
            ),
        )
        total = sum(len(values) for values, _, _ in expected_schemas)
        if total > MAX_DATASET_CAPABILITY_REGISTRATIONS:
            raise ValueError("dataset supporting identity registry exceeds its limit")
        for values, field_name, expected_schema in expected_schemas:
            require_sorted_unique_ids(values, attribute="object_id", field_name=field_name)
            if any(value.object_schema != expected_schema for value in values):
                raise ValueError(f"{field_name} contains an identity with the wrong schema")
        object_ids = tuple(value.object_id for values, _, _ in expected_schemas for value in values)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("dataset supporting identity keys must be globally unique")

    def resolve(self, kind: DatasetCapabilityKind, registry_key: str) -> ObjectIdentity:
        values = {
            DatasetCapabilityKind.CONFIG: self.configs,
            DatasetCapabilityKind.RUNTIME: self.runtimes,
            DatasetCapabilityKind.ENVIRONMENT: self.environments,
        }.get(kind)
        if values is None:
            raise ValueError("executable dataset capability cannot use supporting registry")
        matches = tuple(value for value in values if value.object_id == registry_key)
        if len(matches) != 1:
            raise ValueError("dataset supporting identity is missing or ambiguous")
        return matches[0]


_EXECUTABLE_DATASET_CAPABILITY_KINDS: Final[Mapping[DatasetCapabilityKind, CapabilityKind]] = (
    MappingProxyType(
        {
            DatasetCapabilityKind.INSPECTOR: CapabilityKind.SOURCE,
            DatasetCapabilityKind.EVIDENCE_VERIFIER: CapabilityKind.SOURCE,
            DatasetCapabilityKind.TRANSFORM: CapabilityKind.TRANSFORM,
            DatasetCapabilityKind.BINDING_COMPILER: CapabilityKind.TRANSFORM,
        }
    )
)


def validate_dataset_capability_bindings(
    bindings: tuple[DatasetCapabilityBinding, ...],
    *,
    capability_registry: CapabilityRegistry,
    dataset_registry: DatasetCapabilityRegistry,
    implementation_registry: DatasetImplementationRegistry,
    supporting_registry: DatasetSupportingIdentityRegistry,
) -> None:
    """Replay every authoring binding through its exact frozen registry category."""

    if not isinstance(bindings, tuple) or len(bindings) > MAX_DATASET_CAPABILITY_REGISTRATIONS:
        raise ValueError("dataset capability bindings must be a bounded tuple")
    if any(not isinstance(value, DatasetCapabilityBinding) for value in bindings):
        raise ValueError("dataset capability bindings contain an invalid record")
    require_sorted_unique_ids(bindings, attribute="binding_id", field_name="bindings")
    if not isinstance(capability_registry, CapabilityRegistry):
        raise ValueError("capability_registry must be a CapabilityRegistry")
    if not isinstance(dataset_registry, DatasetCapabilityRegistry):
        raise ValueError("dataset_registry must be a DatasetCapabilityRegistry")
    if not isinstance(implementation_registry, DatasetImplementationRegistry):
        raise ValueError("implementation_registry must be a DatasetImplementationRegistry")
    if implementation_registry.static_registry != dataset_registry:
        raise ValueError("dataset implementation registry reproduces another static registry")
    if not isinstance(supporting_registry, DatasetSupportingIdentityRegistry):
        raise ValueError("supporting_registry must be a DatasetSupportingIdentityRegistry")

    for binding in bindings:
        expected_kind = _EXECUTABLE_DATASET_CAPABILITY_KINDS.get(binding.kind)
        if expected_kind is None:
            expected_identity = supporting_registry.resolve(binding.kind, binding.registry_key)
            if binding.implementation != expected_identity:
                raise ValueError("dataset supporting identity reproduction differs")
            continue

        if binding.kind is DatasetCapabilityKind.EVIDENCE_VERIFIER:
            evidence_matches = tuple(
                value
                for value in dataset_registry.evidence_verifiers
                if value.registry_id == binding.registry_key
            )
            if len(evidence_matches) != 1:
                raise ValueError("dataset evidence-verifier registration is missing or ambiguous")
            evidence_registration = evidence_matches[0]
            expected_identity = ObjectIdentity.from_record(
                binding.registry_key,
                evidence_registration,
            )
            if binding.implementation != expected_identity:
                raise ValueError("dataset evidence-verifier registration reproduction differs")
            try:
                manifest = capability_registry.resolve(
                    evidence_registration.capability_key,
                    evidence_registration.capability_version,
                )
            except KeyError as error:
                raise ValueError(
                    "dataset evidence-verifier capability is not registered"
                ) from error
            if manifest != evidence_registration.capability or manifest.kind is not expected_kind:
                raise ValueError("dataset evidence-verifier capability reproduction differs")
            implementation_registry.evidence_verifier(evidence_registration.registry_id)
            continue

        matches = tuple(
            manifest
            for manifest in capability_registry.capabilities
            if manifest.capability_key == binding.registry_key
        )
        if len(matches) != 1:
            raise ValueError("dataset executable capability is missing or ambiguous")
        manifest = matches[0]
        if manifest.kind is not expected_kind:
            raise ValueError("dataset authoring capability kind is incompatible with registry role")
        expected_identity = ObjectIdentity.from_record(binding.registry_key, manifest)
        if binding.implementation != expected_identity:
            raise ValueError("dataset capability manifest reproduction differs")

        if binding.kind is DatasetCapabilityKind.INSPECTOR:
            dataset_matches: tuple[Registration, ...] = tuple(
                value for value in dataset_registry.inspectors if value.capability == manifest
            )
        elif binding.kind is DatasetCapabilityKind.TRANSFORM:
            dataset_matches = tuple(
                value for value in dataset_registry.transforms if value.capability == manifest
            )
        else:
            dataset_matches = tuple(
                value
                for value in dataset_registry.binding_compilers
                if value.capability == manifest
            )
        if len(dataset_matches) != 1:
            raise ValueError("dataset-specific capability registration is missing or ambiguous")
        registration = dataset_matches[0]
        if binding.kind is DatasetCapabilityKind.INSPECTOR:
            implementation_registry.inspector(registration.registry_id)
        elif binding.kind is DatasetCapabilityKind.TRANSFORM:
            implementation_registry.transform(registration.registry_id)
        else:
            implementation_registry.binding_compiler(registration.registry_id)


Registration = (
    DatasetProviderRegistration
    | DatasetFormatInspectorRegistration
    | DatasetTransformRegistration
    | DatasetBindingCompilerRegistration
    | DatasetStorageVerificationRegistration
    | DatasetEvidenceVerifierRegistration
)
_RegistrationT = TypeVar("_RegistrationT", bound=Registration)


def _resolve_registration(
    values: tuple[_RegistrationT, ...],
    registry_id: str,
    kind: str,
) -> _RegistrationT:
    validate_nonempty(registry_id, field_name="registry_id")
    for value in values:
        if value.registry_id == registry_id:
            return value
    raise KeyError(f"unregistered dataset {kind}: {registry_id}")


class DatasetProviderMetadataAdapter(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def observe(self, request: ObjectIdentity) -> DatasetObservation: ...


class SafeDatasetInspector(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def inspect(self, request: ObjectIdentity) -> DatasetMaterialization: ...


class CurrentDatasetMaterializationVerifier(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def revalidate(
        self,
        materialization: DatasetMaterialization,
        *,
        trusted_at_utc: str,
    ) -> CurrentMaterializationVerification: ...


class DatasetTransformRunner(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def transform(self, request: ObjectIdentity) -> DatasetMaterialization: ...


class DatasetBindingCompiler(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def compile(
        self,
        manifest: ProposedExperimentDatasetBindingManifest,
        *,
        snapshot: DatasetCatalogSnapshot,
        source_authority: object,
        binding_id: str,
        run_id: str | None = None,
    ) -> ExperimentDatasetBinding: ...


class DatasetStorageRootVerifier(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def verify(self, storage_root: ObjectIdentity) -> EvidenceReference: ...


class ImmutableDatasetAuthorizationStore(Protocol):
    def get(self, authorization_id: str) -> DatasetOperationAuthorization | None: ...


class DatasetEvidenceOwnerKind(StrEnum):
    """Closed owner kinds retained while external evidence is verified."""

    RELEASE = "RELEASE"
    OBSERVATION = "OBSERVATION"
    MATERIALIZATION = "MATERIALIZATION"
    ACQUISITION_ATTEMPT = "ACQUISITION_ATTEMPT"
    BINDING = "BINDING"


@dataclass(frozen=True, slots=True)
class DatasetEvidenceVerificationRequest(CanonicalRecord):
    """One exact evidence reference plus its claim-bearing owner context."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-evidence-verification-request'

    owner_kind: DatasetEvidenceOwnerKind
    owner_id: str
    reference: EvidenceReference
    materialization_subject: DatasetMaterializationVerificationSubject | None
    expected_verifier: RegisteredImplementationIdentity | None
    expected_verification_policy_id: str | None
    expected_verification_policy_sha256: str | None
    expected_verified_at_utc: str | None
    expected_manifest_evidence: EvidenceReference | None

    def __post_init__(self) -> None:
        if not isinstance(self.owner_kind, DatasetEvidenceOwnerKind):
            raise ValueError("owner_kind must be a DatasetEvidenceOwnerKind")
        _validate_runtime_id(self.owner_id, field_name="owner_id")
        if not isinstance(self.reference, EvidenceReference):
            raise ValueError("reference must be an EvidenceReference")
        if self.materialization_subject is not None and not isinstance(
            self.materialization_subject,
            DatasetMaterializationVerificationSubject,
        ):
            raise ValueError("materialization_subject must be a verification subject")
        if self.expected_verifier is not None and not isinstance(
            self.expected_verifier,
            RegisteredImplementationIdentity,
        ):
            raise ValueError("expected_verifier must be a RegisteredImplementationIdentity")
        materialization_context = (
            self.materialization_subject is not None,
            self.expected_verifier is not None,
            self.expected_verification_policy_id is not None,
            self.expected_verification_policy_sha256 is not None,
            self.expected_verified_at_utc is not None,
        )
        if self.owner_kind is DatasetEvidenceOwnerKind.MATERIALIZATION:
            if not all(materialization_context):
                raise ValueError(
                    "materialization evidence requires subject, verifier, policy, and "
                    "verification time"
                )
            assert self.materialization_subject is not None
            if self.owner_id != self.materialization_subject.materialization_id:
                raise ValueError("materialization evidence owner differs from its subject")
            assert self.expected_verified_at_utc is not None
            parse_utc_timestamp(
                self.expected_verified_at_utc,
                field_name="expected_verified_at_utc",
            )
            assert self.expected_verification_policy_id is not None
            _validate_runtime_id(
                self.expected_verification_policy_id,
                field_name="expected_verification_policy_id",
            )
            assert self.expected_verification_policy_sha256 is not None
            validate_sha256(
                self.expected_verification_policy_sha256,
                field_name="expected_verification_policy_sha256",
            )
            if self.expected_manifest_evidence is not None:
                if not isinstance(self.expected_manifest_evidence, EvidenceReference):
                    raise ValueError("expected_manifest_evidence must be an EvidenceReference")
                if self.expected_manifest_evidence.kind is not EvidenceReferenceKind.MANIFEST:
                    raise ValueError("expected manifest evidence must have MANIFEST kind")
            if self.reference.evidence_schema == DatasetMaterializationVerificationReceipt.SCHEMA:
                if self.reference.kind is not EvidenceReferenceKind.VERIFICATION:
                    raise ValueError("typed materialization receipt must have VERIFICATION kind")
                if self.expected_manifest_evidence is None:
                    raise ValueError(
                        "typed materialization receipt requires exact manifest evidence"
                    )
        else:
            if self.reference.evidence_schema == DatasetMaterializationVerificationReceipt.SCHEMA:
                raise ValueError("typed materialization receipts require a materialization owner")
            if any(materialization_context) or self.expected_manifest_evidence is not None:
                raise ValueError(
                    "non-materialization evidence cannot carry materialization proof context"
                )


@dataclass(frozen=True, slots=True)
class DatasetEvidenceVerification(CanonicalRecord):
    """Bounded read-only verification result for one exact owner-bound request."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-evidence-verification'

    request_fingerprint: str
    owner_kind: DatasetEvidenceOwnerKind
    owner_id: str
    evidence_reference_fingerprint: str
    evidence_id: str
    storage_root_id: str
    relative_locator: str
    evidence_schema: str
    declared_sha256: str
    observed_sha256: str
    observed_size_bytes: int
    complete_eof: bool
    trusted_root: bool
    locator_contained: bool
    read_only: bool
    schema_validated: bool
    materialization_subject_fingerprint: str | None
    materialization_verification_receipt: DatasetMaterializationVerificationReceipt | None

    def __post_init__(self) -> None:
        validate_sha256(self.request_fingerprint, field_name="request_fingerprint")
        if not isinstance(self.owner_kind, DatasetEvidenceOwnerKind):
            raise ValueError("owner_kind must be a DatasetEvidenceOwnerKind")
        _validate_runtime_id(self.owner_id, field_name="owner_id")
        validate_sha256(
            self.evidence_reference_fingerprint,
            field_name="evidence_reference_fingerprint",
        )
        _validate_runtime_id(self.evidence_id, field_name="evidence_id")
        _validate_runtime_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.relative_locator)
        validate_schema(self.evidence_schema)
        validate_sha256(self.declared_sha256, field_name="declared_sha256")
        validate_sha256(self.observed_sha256, field_name="observed_sha256")
        if (
            not isinstance(self.observed_size_bytes, int)
            or isinstance(self.observed_size_bytes, bool)
            or self.observed_size_bytes < 0
        ):
            raise ValueError("observed_size_bytes must be a nonnegative integer")
        for field_name, value in (
            ("complete_eof", self.complete_eof),
            ("trusted_root", self.trusted_root),
            ("locator_contained", self.locator_contained),
            ("read_only", self.read_only),
            ("schema_validated", self.schema_validated),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.materialization_subject_fingerprint is not None:
            validate_sha256(
                self.materialization_subject_fingerprint,
                field_name="materialization_subject_fingerprint",
            )
        if self.materialization_verification_receipt is not None and not isinstance(
            self.materialization_verification_receipt,
            DatasetMaterializationVerificationReceipt,
        ):
            raise ValueError("materialization_verification_receipt must be a typed receipt")

    def validates(self, request: DatasetEvidenceVerificationRequest) -> bool:
        if not isinstance(request, DatasetEvidenceVerificationRequest):
            return False
        reference = request.reference
        common_valid = (
            self.request_fingerprint == request.fingerprint()
            and self.owner_kind is request.owner_kind
            and self.owner_id == request.owner_id
            and self.evidence_reference_fingerprint == reference.fingerprint()
            and self.evidence_id == reference.evidence_id
            and self.storage_root_id == reference.storage_root_id
            and self.relative_locator == reference.relative_locator
            and self.evidence_schema == reference.evidence_schema
            and self.declared_sha256 == reference.evidence_sha256
            and self.observed_sha256 == reference.evidence_sha256
            and self.complete_eof
            and self.trusted_root
            and self.locator_contained
            and self.read_only
            and self.schema_validated
        )
        expected_subject_fingerprint = (
            request.materialization_subject.fingerprint()
            if request.materialization_subject is not None
            else None
        )
        if (
            not common_valid
            or self.materialization_subject_fingerprint != expected_subject_fingerprint
        ):
            return False
        typed_receipt_expected = (
            reference.evidence_schema == DatasetMaterializationVerificationReceipt.SCHEMA
        )
        if not typed_receipt_expected:
            return self.materialization_verification_receipt is None
        receipt = self.materialization_verification_receipt
        subject = request.materialization_subject
        expected_verifier = request.expected_verifier
        expected_verification_policy_id = request.expected_verification_policy_id
        expected_verification_policy_sha256 = request.expected_verification_policy_sha256
        expected_verified_at_utc = request.expected_verified_at_utc
        expected_manifest_evidence = request.expected_manifest_evidence
        return (
            receipt is not None
            and subject is not None
            and expected_verifier is not None
            and expected_verification_policy_id is not None
            and expected_verification_policy_sha256 is not None
            and expected_verified_at_utc is not None
            and expected_manifest_evidence is not None
            and receipt.fingerprint() == reference.evidence_sha256
            and receipt.validates(
                subject,
                expected_receipt_id=reference.evidence_id,
                expected_verifier=expected_verifier,
                expected_verification_policy_id=expected_verification_policy_id,
                expected_verification_policy_sha256=(expected_verification_policy_sha256),
                expected_verified_at_utc=expected_verified_at_utc,
                expected_manifest_evidence=expected_manifest_evidence,
            )
        )


class DatasetEvidenceVerifier(Protocol):
    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def verify(
        self,
        request: DatasetEvidenceVerificationRequest,
        *,
        maximum_bytes: int,
    ) -> DatasetEvidenceVerification: ...


@dataclass(frozen=True, slots=True)
class DatasetProviderImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    implementation: DatasetProviderMetadataAdapter


@dataclass(frozen=True, slots=True)
class DatasetInspectorImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    inspector: SafeDatasetInspector
    current_verifier: CurrentDatasetMaterializationVerifier


@dataclass(frozen=True, slots=True)
class DatasetTransformImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    implementation: DatasetTransformRunner


@dataclass(frozen=True, slots=True)
class DatasetBindingCompilerImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    implementation: DatasetBindingCompiler


@dataclass(frozen=True, slots=True)
class DatasetStorageVerifierImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    implementation: DatasetStorageRootVerifier


@dataclass(frozen=True, slots=True)
class DatasetEvidenceVerifierImplementationBinding:
    registry_id: str
    registration_fingerprint: str
    implementation: DatasetEvidenceVerifier


ImplementationBinding: TypeAlias = (
    DatasetProviderImplementationBinding
    | DatasetInspectorImplementationBinding
    | DatasetTransformImplementationBinding
    | DatasetBindingCompilerImplementationBinding
    | DatasetStorageVerifierImplementationBinding
    | DatasetEvidenceVerifierImplementationBinding
)
_ImplementationBindingT = TypeVar("_ImplementationBindingT", bound=ImplementationBinding)


def _implementation_objects(binding: ImplementationBinding) -> tuple[object, ...]:
    if isinstance(binding, DatasetInspectorImplementationBinding):
        return (binding.inspector, binding.current_verifier)
    return (binding.implementation,)


def _validate_implementation_binding(
    binding: ImplementationBinding,
    registration: Registration,
) -> None:
    if binding.registry_id != registration.registry_id:
        raise ValueError("implementation binding names a different registration")
    validate_sha256(
        binding.registration_fingerprint,
        field_name="registration_fingerprint",
    )
    if binding.registration_fingerprint != registration.fingerprint():
        raise ValueError("implementation binding registration fingerprint differs")
    for implementation in _implementation_objects(binding):
        identity = (
            getattr(implementation, "capability_key", None),
            getattr(implementation, "capability_version", None),
            getattr(implementation, "implementation_sha256", None),
        )
        expected = (
            registration.capability_key,
            registration.capability_version,
            registration.implementation_sha256,
        )
        if identity != expected:
            raise ValueError("injected dataset implementation identity differs")
        if (
            getattr(implementation, "capability_manifest_fingerprint", None)
            != registration.capability.fingerprint()
        ):
            raise ValueError("injected capability manifest reproduction differs")
        if getattr(implementation, "registration_fingerprint", None) != registration.fingerprint():
            raise ValueError("injected dataset registration reproduction differs")


@dataclass(frozen=True, slots=True)
class DatasetImplementationRegistry:
    """Frozen, noncanonical process-local implementations bound to static hashes."""

    static_registry: DatasetCapabilityRegistry
    providers: tuple[DatasetProviderImplementationBinding, ...]
    inspectors: tuple[DatasetInspectorImplementationBinding, ...]
    transforms: tuple[DatasetTransformImplementationBinding, ...]
    binding_compilers: tuple[DatasetBindingCompilerImplementationBinding, ...]
    storage_verifiers: tuple[DatasetStorageVerifierImplementationBinding, ...]
    evidence_verifiers: tuple[DatasetEvidenceVerifierImplementationBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.static_registry, DatasetCapabilityRegistry):
            raise ValueError("static_registry must be DatasetCapabilityRegistry")
        _validate_implementation_group(
            self.providers,
            self.static_registry.providers,
            "providers",
        )
        _validate_implementation_group(
            self.inspectors,
            self.static_registry.inspectors,
            "inspectors",
        )
        _validate_implementation_group(
            self.transforms,
            self.static_registry.transforms,
            "transforms",
        )
        _validate_implementation_group(
            self.binding_compilers,
            self.static_registry.binding_compilers,
            "binding_compilers",
        )
        _validate_implementation_group(
            self.storage_verifiers,
            self.static_registry.storage_verifiers,
            "storage_verifiers",
        )
        _validate_implementation_group(
            self.evidence_verifiers,
            self.static_registry.evidence_verifiers,
            "evidence_verifiers",
        )

    def provider(self, registry_id: str) -> DatasetProviderMetadataAdapter:
        binding = _resolve_implementation(self.providers, registry_id)
        _validate_implementation_binding(binding, self.static_registry.provider(registry_id))
        return binding.implementation

    def inspector(
        self,
        registry_id: str,
    ) -> tuple[SafeDatasetInspector, CurrentDatasetMaterializationVerifier]:
        binding = _resolve_implementation(self.inspectors, registry_id)
        _validate_implementation_binding(binding, self.static_registry.inspector(registry_id))
        return binding.inspector, binding.current_verifier

    def transform(self, registry_id: str) -> DatasetTransformRunner:
        binding = _resolve_implementation(self.transforms, registry_id)
        _validate_implementation_binding(binding, self.static_registry.transform(registry_id))
        return binding.implementation

    def binding_compiler(self, registry_id: str) -> DatasetBindingCompiler:
        binding = _resolve_implementation(self.binding_compilers, registry_id)
        _validate_implementation_binding(
            binding,
            self.static_registry.binding_compiler(registry_id),
        )
        return binding.implementation

    def storage_verifier(self, registry_id: str) -> DatasetStorageRootVerifier:
        binding = _resolve_implementation(self.storage_verifiers, registry_id)
        _validate_implementation_binding(
            binding,
            self.static_registry.storage_verifier(registry_id),
        )
        return binding.implementation

    def evidence_verifier(self, registry_id: str) -> DatasetEvidenceVerifier:
        binding = _resolve_implementation(self.evidence_verifiers, registry_id)
        _validate_implementation_binding(
            binding,
            self.static_registry.evidence_verifier(registry_id),
        )
        return binding.implementation


def _resolve_implementation(
    bindings: tuple[_ImplementationBindingT, ...],
    registry_id: str,
) -> _ImplementationBindingT:
    validate_nonempty(registry_id, field_name="registry_id")
    for binding in bindings:
        if binding.registry_id == registry_id:
            return binding
    raise KeyError(f"dataset implementation is not injected: {registry_id}")


def _validate_implementation_group(
    bindings: tuple[ImplementationBinding, ...],
    registrations: tuple[Registration, ...],
    field_name: str,
) -> None:
    binding_ids = tuple(binding.registry_id for binding in bindings)
    registration_ids = tuple(value.registry_id for value in registrations)
    if tuple(sorted(set(binding_ids))) != binding_ids:
        raise ValueError(f"{field_name} bindings must be sorted and unique")
    if binding_ids != registration_ids:
        raise ValueError(f"{field_name} implementations must exactly cover registrations")
    for binding, registration in zip(bindings, registrations, strict=True):
        _validate_implementation_binding(binding, registration)


def validate_dataset_snapshot_record_count(*counts: int) -> None:
    """Reject invalid or oversized aggregates before constructing lookup sets."""

    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError("dataset snapshot record counts must be nonnegative integers")
    if sum(counts) > MAX_DATASET_CATALOG_RECORDS:
        raise ValueError("dataset catalog snapshot exceeds its aggregate record limit")


def validate_dataset_snapshot_byte_budget(
    *record_groups: tuple[CanonicalRecord, ...],
) -> None:
    """Bound canonical projection bytes before building reference indexes."""

    total = 0
    for records in record_groups:
        for record in records:
            if not isinstance(record, CanonicalRecord):
                raise ValueError("dataset snapshot contains a noncanonical record")
            total += len(record.canonical_bytes())
            if total > MAX_DATASET_CATALOG_CANONICAL_BYTES:
                raise ValueError("dataset catalog snapshot exceeds its canonical byte limit")


@dataclass(frozen=True, slots=True)
class DatasetCatalogSnapshot(CanonicalRecord):
    """Complete six-record dataset projection with closed internal references."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-snapshot'

    families: tuple[DatasetFamily, ...]
    releases: tuple[DatasetRelease, ...]
    observations: tuple[DatasetObservation, ...]
    materializations: tuple[DatasetMaterialization, ...]
    acquisition_attempts: tuple[AcquisitionAttempt, ...]
    bindings: tuple[ExperimentDatasetBinding, ...]

    def __post_init__(self) -> None:
        validate_dataset_snapshot_record_count(
            len(self.families),
            len(self.releases),
            len(self.observations),
            len(self.materializations),
            len(self.acquisition_attempts),
            len(self.bindings),
        )
        validate_dataset_snapshot_byte_budget(
            self.families,
            self.releases,
            self.observations,
            self.materializations,
            self.acquisition_attempts,
            self.bindings,
        )
        for values, attribute, field_name in (
            (self.families, "family_id", "families"),
            (self.releases, "release_id", "releases"),
            (self.observations, "observation_id", "observations"),
            (self.materializations, "materialization_id", "materializations"),
            (self.acquisition_attempts, "attempt_id", "acquisition_attempts"),
            (self.bindings, "binding_id", "bindings"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=field_name)
        self._validate_references()

    def _validate_references(self) -> None:
        families = {record.family_id: record for record in self.families}
        releases = {record.release_id: record for record in self.releases}
        observations = {record.observation_id: record for record in self.observations}
        materializations = {record.materialization_id: record for record in self.materializations}
        attempts = {record.attempt_id: record for record in self.acquisition_attempts}

        for family in self.families:
            observation_ids = tuple(
                value
                for value in (family.first_observation_id, family.latest_observation_id)
                if value is not None
            )
            if any(value not in observations for value in observation_ids):
                raise ValueError("dataset family references an unknown observation")
        for release in self.releases:
            if release.family_id not in families:
                raise ValueError("dataset release references an unknown family")
        for observation in self.observations:
            if (
                observation.predecessor_observation_id is not None
                and observation.predecessor_observation_id not in observations
            ):
                raise ValueError("dataset observation references an unknown predecessor")
            if not set(observation.candidate_release_ids).issubset(releases):
                raise ValueError("dataset observation references an unknown candidate release")
        _validate_observation_chains(observations)
        for materialization in self.materializations:
            if materialization.release_id not in releases:
                raise ValueError("dataset materialization references an unknown release")
        for attempt in self.acquisition_attempts:
            if attempt.release_id not in releases:
                raise ValueError("dataset acquisition attempt references an unknown release")
            if (
                attempt.resumed_from_attempt_id is not None
                and attempt.resumed_from_attempt_id not in attempts
            ):
                raise ValueError("dataset acquisition attempt references an unknown predecessor")
            if attempt.result_materialization_id is not None:
                result = materializations.get(attempt.result_materialization_id)
                if result is None:
                    raise ValueError("dataset acquisition attempt references an unknown result")
                attempt.validate_result(result)
        _validate_acquisition_chains(attempts)
        for binding in self.bindings:
            target_release = releases.get(binding.release_id)
            target_materialization = materializations.get(binding.materialization_id)
            if target_release is None or target_materialization is None:
                raise ValueError("dataset binding references an unknown target")
            if target_materialization.release_id != target_release.release_id:
                raise ValueError("dataset binding target crosses releases")
            if target_materialization.selector != binding.selector:
                raise ValueError("dataset binding selector differs from its target")
            if target_materialization.outcome_access is not binding.outcome_access:
                raise ValueError("dataset binding outcome access differs from its target")
            for parent in binding.parent_materializations:
                observed_parent = materializations.get(parent.object_id)
                if observed_parent is None:
                    raise ValueError("dataset binding references an unknown parent materialization")
                if (
                    ObjectIdentity.from_record(
                        observed_parent.materialization_id,
                        observed_parent,
                    )
                    != parent
                ):
                    raise ValueError("dataset binding parent materialization identity is stale")
            if binding.binding_state is BindingState.VERIFIED:
                binding.validate_target(target_release, target_materialization)

    @classmethod
    def empty(cls) -> DatasetCatalogSnapshot:
        return cls((), (), (), (), (), ())


def _validate_observation_chains(
    observations: dict[str, DatasetObservation],
) -> None:
    for observation in observations.values():
        predecessor_id = observation.predecessor_observation_id
        if predecessor_id is not None:
            predecessor = observations[predecessor_id]
            if (
                predecessor.provider_id != observation.provider_id
                or predecessor.query_id != observation.query_id
                or predecessor.canonical_request_sha256 != observation.canonical_request_sha256
            ):
                raise ValueError("dataset observation predecessor crosses its exact request")
            if observation.page_index != predecessor.page_index + 1:
                raise ValueError("dataset observation pages must advance by exactly one")
        visited: set[str] = set()
        current: DatasetObservation | None = observation
        while current is not None:
            if current.observation_id in visited:
                raise ValueError("dataset observation predecessor chain contains a cycle")
            visited.add(current.observation_id)
            current = (
                observations[current.predecessor_observation_id]
                if current.predecessor_observation_id is not None
                else None
            )


def _validate_acquisition_chains(
    attempts: dict[str, AcquisitionAttempt],
) -> None:
    for attempt in attempts.values():
        predecessor_id = attempt.resumed_from_attempt_id
        if predecessor_id is not None:
            predecessor = attempts[predecessor_id]
            if predecessor.release_id != attempt.release_id:
                raise ValueError("dataset acquisition retry crosses releases")
            if attempt.retry_count != predecessor.retry_count + 1:
                raise ValueError("dataset acquisition retry count must advance by exactly one")
        visited: set[str] = set()
        current: AcquisitionAttempt | None = attempt
        while current is not None:
            if current.attempt_id in visited:
                raise ValueError("dataset acquisition retry chain contains a cycle")
            visited.add(current.attempt_id)
            current = (
                attempts[current.resumed_from_attempt_id]
                if current.resumed_from_attempt_id is not None
                else None
            )


@dataclass(frozen=True, slots=True)
class UnifiedCatalogSnapshot(CanonicalRecord):
    """One rebuild boundary containing both scientific and dataset projections."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/unified-catalog-snapshot'

    scientific: CatalogSnapshot
    datasets: DatasetCatalogSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.scientific, CatalogSnapshot):
            raise ValueError("scientific must be CatalogSnapshot")
        if not isinstance(self.datasets, DatasetCatalogSnapshot):
            raise ValueError("datasets must be DatasetCatalogSnapshot")
        scientific_roots = {record.storage_root_id for record in self.scientific.storage_roots}
        dataset_roots = _dataset_storage_root_ids(self.datasets)
        if not dataset_roots.issubset(scientific_roots):
            raise ValueError(
                "dataset catalog references storage roots absent from the scientific catalog"
            )

    @classmethod
    def empty(cls) -> UnifiedCatalogSnapshot:
        return cls(CatalogSnapshot.empty(), DatasetCatalogSnapshot.empty())


def _dataset_storage_root_ids(snapshot: DatasetCatalogSnapshot) -> set[str]:
    roots: set[str] = set()

    def include_evidence(values: tuple[EvidenceReference, ...]) -> None:
        roots.update(value.storage_root_id for value in values)

    for release in snapshot.releases:
        include_evidence(release.evidence_refs)
    for observation in snapshot.observations:
        roots.add(observation.storage_root_id)
        include_evidence(observation.evidence_refs)
    for materialization in snapshot.materializations:
        roots.add(materialization.storage_root_id)
        include_evidence(materialization.verification_evidence_refs)
    for attempt in snapshot.acquisition_attempts:
        roots.add(attempt.storage_root_id)
        include_evidence(attempt.evidence_refs)
    for binding in snapshot.bindings:
        include_evidence(binding.evidence_refs)
    return roots


class DatasetCatalogRecordKind(StrEnum):
    FAMILY = "FAMILY"
    RELEASE = "RELEASE"
    OBSERVATION = "OBSERVATION"
    MATERIALIZATION = "MATERIALIZATION"
    ACQUISITION_ATTEMPT = "ACQUISITION_ATTEMPT"
    BINDING = "BINDING"


DatasetCatalogRecord: TypeAlias = (
    DatasetFamily
    | DatasetRelease
    | DatasetObservation
    | DatasetMaterialization
    | AcquisitionAttempt
    | ExperimentDatasetBinding
)


def dataset_catalog_record_key(
    record: DatasetCatalogRecord,
) -> tuple[DatasetCatalogRecordKind, str]:
    if isinstance(record, DatasetFamily):
        return DatasetCatalogRecordKind.FAMILY, record.family_id
    if isinstance(record, DatasetRelease):
        return DatasetCatalogRecordKind.RELEASE, record.release_id
    if isinstance(record, DatasetObservation):
        return DatasetCatalogRecordKind.OBSERVATION, record.observation_id
    if isinstance(record, DatasetMaterialization):
        return DatasetCatalogRecordKind.MATERIALIZATION, record.materialization_id
    if isinstance(record, AcquisitionAttempt):
        return DatasetCatalogRecordKind.ACQUISITION_ATTEMPT, record.attempt_id
    if isinstance(record, ExperimentDatasetBinding):
        return DatasetCatalogRecordKind.BINDING, record.binding_id
    raise TypeError("unsupported dataset catalog record")


def _validate_filter_ids(values: tuple[str, ...], *, field_name: str) -> None:
    require_sorted_unique_strings(values, field_name=field_name)
    if len(values) > MAX_DATASET_QUERY_FILTER_ITEMS:
        raise ValueError(f"{field_name} exceeds its item limit")
    for value in values:
        _validate_runtime_id(value, field_name=field_name)


def _validate_enum_filter(
    values: tuple[StrEnum, ...],
    *,
    enum_type: type[StrEnum],
    field_name: str,
) -> None:
    if any(not isinstance(value, enum_type) for value in values):
        raise ValueError(f"{field_name} contains the wrong enum type")
    require_sorted_unique_strings(values, field_name=field_name)
    if len(values) > MAX_DATASET_QUERY_FILTER_ITEMS:
        raise ValueError(f"{field_name} exceeds its item limit")


@dataclass(frozen=True, slots=True)
class DatasetCatalogProjectionState(CanonicalRecord):
    """Authenticated clean dataset snapshot boundary without its record payloads."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-projection-state'

    snapshot_schema: str
    snapshot_fingerprint: str
    family_count: int
    release_count: int
    observation_count: int
    materialization_count: int
    acquisition_attempt_count: int
    binding_count: int
    canonical_byte_count: int
    projection_revision: int

    def __post_init__(self) -> None:
        if self.snapshot_schema != DatasetCatalogSnapshot.SCHEMA:
            raise ValueError("projection state names an unsupported snapshot schema")
        validate_sha256(self.snapshot_fingerprint, field_name="snapshot_fingerprint")
        counts = (
            self.family_count,
            self.release_count,
            self.observation_count,
            self.materialization_count,
            self.acquisition_attempt_count,
            self.binding_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
            raise ValueError("projection counts must be integers")
        validate_dataset_snapshot_record_count(*counts)
        if (
            not isinstance(self.canonical_byte_count, int)
            or isinstance(self.canonical_byte_count, bool)
            or not len(DatasetCatalogSnapshot.empty().canonical_bytes())
            <= self.canonical_byte_count
            <= MAX_DATASET_CATALOG_CANONICAL_BYTES
        ):
            raise ValueError("projection canonical byte count lies outside its bound")
        if (
            not isinstance(self.projection_revision, int)
            or isinstance(self.projection_revision, bool)
            or not 0 <= self.projection_revision <= MAX_DATASET_INTEGER
        ):
            raise ValueError("projection revision must be a nonnegative integer")
        empty = DatasetCatalogSnapshot.empty()
        if sum(counts) == 0 and (
            self.snapshot_fingerprint != empty.fingerprint()
            or self.canonical_byte_count != len(empty.canonical_bytes())
            or self.projection_revision != 0
        ):
            raise ValueError("empty projection state differs from the canonical snapshot")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: DatasetCatalogSnapshot,
        *,
        projection_revision: int,
    ) -> DatasetCatalogProjectionState:
        """Construct the exact bounded state for a separately observed revision."""

        if not isinstance(snapshot, DatasetCatalogSnapshot):
            raise TypeError("snapshot must be a DatasetCatalogSnapshot")
        return cls(
            snapshot_schema=snapshot.SCHEMA,
            snapshot_fingerprint=snapshot.fingerprint(),
            family_count=len(snapshot.families),
            release_count=len(snapshot.releases),
            observation_count=len(snapshot.observations),
            materialization_count=len(snapshot.materializations),
            acquisition_attempt_count=len(snapshot.acquisition_attempts),
            binding_count=len(snapshot.bindings),
            canonical_byte_count=len(snapshot.canonical_bytes()),
            projection_revision=projection_revision,
        )

    @property
    def record_count(self) -> int:
        return sum(
            (
                self.family_count,
                self.release_count,
                self.observation_count,
                self.materialization_count,
                self.acquisition_attempt_count,
                self.binding_count,
            )
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogProjectionAnchor(CanonicalRecord):
    """Externally authenticated receipt identity for one exact projection state."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-projection-anchor'

    receipt_id: str
    receipt_schema: str
    receipt_sha256: str
    projection_state: DatasetCatalogProjectionState

    def __post_init__(self) -> None:
        _validate_runtime_id(self.receipt_id, field_name="receipt_id")
        validate_schema(self.receipt_schema)
        validate_sha256(self.receipt_sha256, field_name="receipt_sha256")
        if not isinstance(self.projection_state, DatasetCatalogProjectionState):
            raise ValueError("projection_state must be a DatasetCatalogProjectionState")

    def validates_snapshot(self, snapshot: DatasetCatalogSnapshot) -> bool:
        if not isinstance(snapshot, DatasetCatalogSnapshot):
            return False
        state = self.projection_state
        return (
            state.snapshot_schema == snapshot.SCHEMA
            and state.snapshot_fingerprint == snapshot.fingerprint()
            and state.family_count == len(snapshot.families)
            and state.release_count == len(snapshot.releases)
            and state.observation_count == len(snapshot.observations)
            and state.materialization_count == len(snapshot.materializations)
            and state.acquisition_attempt_count == len(snapshot.acquisition_attempts)
            and state.binding_count == len(snapshot.bindings)
            and state.canonical_byte_count == len(snapshot.canonical_bytes())
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogProjectionReceipt(CanonicalRecord):
    """Durable external authority for one exact unified dataset projection build."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-projection-receipt'

    receipt_id: str
    unified_projection_evidence: EvidenceReference
    dataset_snapshot: ObjectIdentity
    projection_state: DatasetCatalogProjectionState
    rebuild_implementation: RegisteredImplementationIdentity
    authorization_evidence: EvidenceReference
    built_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        _validate_runtime_id(self.receipt_id, field_name="receipt_id")
        if not isinstance(self.unified_projection_evidence, EvidenceReference):
            raise TypeError("unified_projection_evidence must be an EvidenceReference")
        if self.unified_projection_evidence.kind is not EvidenceReferenceKind.MANIFEST:
            raise ValueError("unified projection evidence must have MANIFEST kind")
        if self.unified_projection_evidence.evidence_schema != UnifiedCatalogSnapshot.SCHEMA:
            raise ValueError("unified projection evidence must name UnifiedCatalogSnapshot")
        if not isinstance(self.dataset_snapshot, ObjectIdentity):
            raise TypeError("dataset_snapshot must be an exact ObjectIdentity")
        if (
            self.dataset_snapshot.object_schema != DatasetCatalogSnapshot.SCHEMA
            or self.dataset_snapshot.object_version != DatasetCatalogSnapshot.VERSION
        ):
            raise ValueError("dataset_snapshot must name DatasetCatalogSnapshot")
        if not isinstance(self.projection_state, DatasetCatalogProjectionState):
            raise TypeError("projection_state must be a DatasetCatalogProjectionState")
        if self.dataset_snapshot.object_fingerprint != self.projection_state.snapshot_fingerprint:
            raise ValueError("dataset snapshot identity differs from projection state")
        if not isinstance(self.rebuild_implementation, RegisteredImplementationIdentity):
            raise TypeError("rebuild_implementation must be a RegisteredImplementationIdentity")
        if not isinstance(self.authorization_evidence, EvidenceReference):
            raise TypeError("authorization_evidence must be an EvidenceReference")
        if self.authorization_evidence.kind is not EvidenceReferenceKind.AUTHORIZATION:
            raise ValueError("dataset rebuild authorization must have AUTHORIZATION kind")
        if (
            self.authorization_evidence.evidence_schema
            != DatasetProjectionRebuildAuthorization.SCHEMA
        ):
            raise ValueError(
                "dataset rebuild authorization must name DatasetProjectionRebuildAuthorization"
            )
        parse_utc_timestamp(self.built_at_utc, field_name="built_at_utc")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("dataset projection build must be outcome-blind")

    def validates_build(
        self,
        unified_snapshot: UnifiedCatalogSnapshot,
        dataset_snapshot: DatasetCatalogSnapshot,
        projection_state: DatasetCatalogProjectionState,
    ) -> bool:
        """Validate separately decoded projection, nested snapshot, and observed state."""

        if (
            not isinstance(unified_snapshot, UnifiedCatalogSnapshot)
            or not isinstance(dataset_snapshot, DatasetCatalogSnapshot)
            or not isinstance(projection_state, DatasetCatalogProjectionState)
        ):
            return False
        if unified_snapshot.datasets != dataset_snapshot:
            return False
        expected_identity = ObjectIdentity.from_record(
            self.dataset_snapshot.object_id,
            dataset_snapshot,
        )
        expected_state = DatasetCatalogProjectionState.from_snapshot(
            dataset_snapshot,
            projection_revision=projection_state.projection_revision,
        )
        return (
            self.unified_projection_evidence.evidence_sha256 == unified_snapshot.fingerprint()
            and self.dataset_snapshot == expected_identity
            and self.projection_state == projection_state
            and self.projection_state == expected_state
        )

    def derive_anchor(
        self,
        receipt_evidence: EvidenceReference,
        *,
        unified_snapshot: UnifiedCatalogSnapshot,
        dataset_snapshot: DatasetCatalogSnapshot,
        projection_state: DatasetCatalogProjectionState,
    ) -> DatasetCatalogProjectionAnchor:
        """Derive an anchor only from exact canonical build and receipt evidence."""

        if not isinstance(receipt_evidence, EvidenceReference):
            raise TypeError("receipt_evidence must be an EvidenceReference")
        if receipt_evidence.kind is not EvidenceReferenceKind.RECEIPT:
            raise ValueError("dataset projection receipt evidence must have RECEIPT kind")
        if (
            receipt_evidence.evidence_id != self.receipt_id
            or receipt_evidence.evidence_schema != self.SCHEMA
            or receipt_evidence.evidence_sha256 != self.fingerprint()
        ):
            raise ValueError("dataset projection receipt evidence differs from this receipt")
        if not self.validates_build(
            unified_snapshot,
            dataset_snapshot,
            projection_state,
        ):
            raise ValueError("dataset projection receipt differs from the exact build")
        return DatasetCatalogProjectionAnchor(
            receipt_id=receipt_evidence.evidence_id,
            receipt_schema=receipt_evidence.evidence_schema,
            receipt_sha256=receipt_evidence.evidence_sha256,
            projection_state=self.projection_state,
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogWorkLimit(CanonicalRecord):
    """Enforceable per-page lookup work, independent of returned row count."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-work-limit'

    max_records_examined: int
    max_query_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_records_examined, int)
            or isinstance(self.max_records_examined, bool)
            or not 1 <= self.max_records_examined <= MAX_DATASET_CATALOG_RECORDS
        ):
            raise ValueError("max_records_examined lies outside its absolute range")
        if (
            not isinstance(self.max_query_steps, int)
            or isinstance(self.max_query_steps, bool)
            or not 1 <= self.max_query_steps <= MAX_DATASET_QUERY_STEPS
        ):
            raise ValueError("max_query_steps lies outside its absolute range")


@dataclass(frozen=True, slots=True)
class DatasetCatalogQueryBoundary(CanonicalRecord):
    """All query predicates plus the exact immutable snapshot fingerprint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-query-boundary'

    snapshot_fingerprint: str
    projection_state_fingerprint: str
    projection_anchor_fingerprint: str
    work_limit: DatasetCatalogWorkLimit
    family_ids: tuple[str, ...]
    release_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    external_identifiers: tuple[ExternalIdentifier, ...]
    custody_states: tuple[CustodyState, ...]
    acquisition_states: tuple[AcquisitionState, ...]
    experiment_spec_ids: tuple[str, ...]
    roles: tuple[DatasetBindingRole, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.snapshot_fingerprint, field_name="snapshot_fingerprint")
        validate_sha256(
            self.projection_state_fingerprint,
            field_name="projection_state_fingerprint",
        )
        validate_sha256(
            self.projection_anchor_fingerprint,
            field_name="projection_anchor_fingerprint",
        )
        if not isinstance(self.work_limit, DatasetCatalogWorkLimit):
            raise ValueError("work_limit must be DatasetCatalogWorkLimit")
        for values, field_name in (
            (self.family_ids, "family_ids"),
            (self.release_ids, "release_ids"),
            (self.provider_ids, "provider_ids"),
            (self.experiment_spec_ids, "experiment_spec_ids"),
        ):
            _validate_filter_ids(values, field_name=field_name)
        if any(not isinstance(value, ExternalIdentifier) for value in self.external_identifiers):
            raise ValueError("external_identifiers contains the wrong record type")
        keys = tuple(
            (value.kind.value, value.namespace, value.value) for value in self.external_identifiers
        )
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("external_identifiers must be sorted and unique")
        if len(self.external_identifiers) > MAX_DATASET_QUERY_FILTER_ITEMS:
            raise ValueError("external_identifiers exceeds its item limit")
        _validate_enum_filter(
            self.custody_states,
            enum_type=CustodyState,
            field_name="custody_states",
        )
        _validate_enum_filter(
            self.acquisition_states,
            enum_type=AcquisitionState,
            field_name="acquisition_states",
        )
        _validate_enum_filter(
            self.roles,
            enum_type=DatasetBindingRole,
            field_name="roles",
        )
        aggregate = sum(
            len(values)
            for values in (
                self.family_ids,
                self.release_ids,
                self.provider_ids,
                self.external_identifiers,
                self.custody_states,
                self.acquisition_states,
                self.experiment_spec_ids,
                self.roles,
            )
        )
        if aggregate > MAX_DATASET_QUERY_FILTER_ITEMS:
            raise ValueError("dataset catalog query exceeds its aggregate filter limit")


@dataclass(frozen=True, slots=True)
class DatasetCatalogCursor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-cursor'

    snapshot_fingerprint: str
    projection_state_fingerprint: str
    projection_anchor_fingerprint: str
    query_boundary_fingerprint: str
    after_record_kind: DatasetCatalogRecordKind
    after_record_id: str

    def __post_init__(self) -> None:
        validate_sha256(self.snapshot_fingerprint, field_name="snapshot_fingerprint")
        validate_sha256(
            self.projection_state_fingerprint,
            field_name="projection_state_fingerprint",
        )
        validate_sha256(
            self.projection_anchor_fingerprint,
            field_name="projection_anchor_fingerprint",
        )
        validate_sha256(
            self.query_boundary_fingerprint,
            field_name="query_boundary_fingerprint",
        )
        if not isinstance(self.after_record_kind, DatasetCatalogRecordKind):
            raise ValueError("after_record_kind must be DatasetCatalogRecordKind")
        _validate_runtime_id(self.after_record_id, field_name="after_record_id")

    def validates(self, boundary: DatasetCatalogQueryBoundary) -> bool:
        return (
            self.snapshot_fingerprint == boundary.snapshot_fingerprint
            and self.projection_state_fingerprint == boundary.projection_state_fingerprint
            and self.projection_anchor_fingerprint == boundary.projection_anchor_fingerprint
            and self.query_boundary_fingerprint == boundary.fingerprint()
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogQuery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-query'

    boundary: DatasetCatalogQueryBoundary
    limit: int
    cursor: DatasetCatalogCursor | None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, DatasetCatalogQueryBoundary):
            raise ValueError("boundary must be DatasetCatalogQueryBoundary")
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= MAX_DATASET_PAGE_RECORDS
        ):
            raise ValueError("limit lies outside the bounded page range")
        if self.cursor is not None:
            if not isinstance(self.cursor, DatasetCatalogCursor):
                raise ValueError("cursor must be DatasetCatalogCursor")
            if not self.cursor.validates(self.boundary):
                raise ValueError("cursor crosses its exact query or snapshot boundary")


@dataclass(frozen=True, slots=True)
class DatasetCatalogPage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-catalog-page'

    boundary: DatasetCatalogQueryBoundary
    limit: int
    records: tuple[DatasetCatalogRecord, ...]
    records_examined: int
    query_steps: int
    has_more: bool
    next_cursor: DatasetCatalogCursor | None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, DatasetCatalogQueryBoundary):
            raise ValueError("boundary must be DatasetCatalogQueryBoundary")
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= MAX_DATASET_PAGE_RECORDS
        ):
            raise ValueError("limit lies outside the bounded page range")
        if len(self.records) > self.limit:
            raise ValueError("dataset catalog page exceeds its requested limit")
        if (
            not isinstance(self.records_examined, int)
            or isinstance(self.records_examined, bool)
            or not len(self.records)
            <= self.records_examined
            <= self.boundary.work_limit.max_records_examined
        ):
            raise ValueError("records_examined lies outside the enforceable work limit")
        if (
            not isinstance(self.query_steps, int)
            or isinstance(self.query_steps, bool)
            or not 0 <= self.query_steps <= self.boundary.work_limit.max_query_steps
        ):
            raise ValueError("query_steps lies outside the enforceable work limit")
        if (self.records or self.has_more) and self.query_steps == 0:
            raise ValueError("a nonempty query page must report positive query steps")
        keys = tuple(dataset_catalog_record_key(record) for record in self.records)
        sortable_keys = tuple((kind.value, record_id) for kind, record_id in keys)
        if tuple(sorted(set(sortable_keys))) != sortable_keys:
            raise ValueError("dataset catalog page records must be sorted and unique")
        if not isinstance(self.has_more, bool):
            raise ValueError("has_more must be boolean")
        if self.has_more:
            if not self.records or self.next_cursor is None:
                raise ValueError("a continued page requires records and a next cursor")
        elif self.next_cursor is not None:
            raise ValueError("a terminal page cannot carry a next cursor")
        if self.next_cursor is not None:
            if not isinstance(self.next_cursor, DatasetCatalogCursor):
                raise ValueError("next_cursor must be DatasetCatalogCursor")
            if not self.next_cursor.validates(self.boundary):
                raise ValueError("next cursor crosses its exact query or snapshot boundary")
            final_kind, final_id = keys[-1]
            if (
                self.next_cursor.after_record_kind is not final_kind
                or self.next_cursor.after_record_id != final_id
            ):
                raise ValueError("next cursor must continue after the final page record")


class DatasetRepository(Protocol):
    def append_snapshot(self, snapshot: DatasetCatalogSnapshot) -> None: ...

    def snapshot(self) -> DatasetCatalogSnapshot: ...

    def projection_state(
        self,
        work_limit: DatasetCatalogWorkLimit,
    ) -> DatasetCatalogProjectionState: ...

    def query(self, query: DatasetCatalogQuery) -> DatasetCatalogPage: ...

    def integrity_check(self) -> tuple[str, ...]: ...

    def family(self, family_id: str) -> DatasetFamily | None: ...

    def release(self, release_id: str) -> DatasetRelease | None: ...

    def materialization(self, materialization_id: str) -> DatasetMaterialization | None: ...


__all__ = [
    "CurrentDatasetMaterializationVerifier",
    "DatasetBindingCompiler",
    "DatasetBindingCompilerImplementationBinding",
    "DatasetBindingCompilerRegistration",
    "DatasetCapabilityLimits",
    "DatasetCapabilityRegistry",
    "DatasetCatalogCursor",
    "DatasetCatalogPage",
    "DatasetCatalogProjectionAnchor",
    "DatasetCatalogProjectionReceipt",
    "DatasetCatalogProjectionState",
    "DatasetCatalogQuery",
    "DatasetCatalogQueryBoundary",
    "DatasetCatalogRecord",
    "DatasetCatalogRecordKind",
    "DatasetCatalogSnapshot",
    "DatasetCatalogWorkLimit",
    "DatasetEvidenceOwnerKind",
    "DatasetEvidenceVerifierImplementationBinding",
    "DatasetEvidenceVerifierRegistration",
    "DatasetEvidenceVerification",
    "DatasetEvidenceVerificationRequest",
    "DatasetEvidenceVerifier",
    "DatasetFormatInspectorRegistration",
    "DatasetImplementationRegistry",
    "DatasetInspectorImplementationBinding",
    "DatasetProviderImplementationBinding",
    "DatasetProviderMetadataAdapter",
    "DatasetProviderRegistration",
    "DatasetRepository",
    "DatasetStorageRootVerifier",
    "DatasetStorageVerificationRegistration",
    "DatasetStorageVerifierImplementationBinding",
    "DatasetSupportingIdentityRegistry",
    "DatasetTransformImplementationBinding",
    "DatasetTransformRegistration",
    "DatasetTransformRunner",
    "ImmutableDatasetAuthorizationStore",
    "MAX_DATASET_CAPABILITY_REGISTRATIONS",
    "MAX_DATASET_CAPABILITY_ARCHIVE_MEMBERS",
    "MAX_DATASET_CAPABILITY_BYTES",
    "MAX_DATASET_CAPABILITY_FILES",
    "MAX_DATASET_CAPABILITY_RECORDS",
    "MAX_DATASET_CATALOG_CANONICAL_BYTES",
    "MAX_DATASET_CATALOG_RECORDS",
    "MAX_DATASET_PAGE_RECORDS",
    "MAX_DATASET_QUERY_FILTER_ITEMS",
    "MAX_DATASET_QUERY_STEPS",
    "SafeDatasetInspector",
    "UnifiedCatalogSnapshot",
    "dataset_catalog_record_key",
    "validate_dataset_snapshot_record_count",
    "validate_dataset_snapshot_byte_budget",
    "validate_dataset_capability_bindings",
]
