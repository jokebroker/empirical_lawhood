"""Closed executable bindings layered over static extension discovery.

The canonical records in this module contain only identities and bounded static
metadata.  Python factories and platform ports live in a separate, code-owned
registry and can never be selected by an import path or callable stored in
configuration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol, cast, runtime_checkable

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .adjudication import ScientificAdjudicationOutputContract
from .capabilities import CapabilityManifest, CapabilityRegistry
from .conditional_children import FrozenParentInputBinding
from .execution import TaskRunner
from .extension_bundles import ExtensionComponentKind, ExtensionComponentRegistration, GeneratedExtensionBundleAggregate
from .plans import ProtocolExecutionPlan
from .study_issue import StudyExtensionDecoderRegistration
from .providers import (
    CampaignRuntimeProvider,
    CampaignRuntimeProviderRegistry,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)


class ExecutableBindingRole(StrEnum):
    """Closed construction roles; only one role may own task runners."""

    CAMPAIGN_RUNTIME_PROVIDER = "CAMPAIGN_RUNTIME_PROVIDER"
    PROFILE_COMPILER = "PROFILE_COMPILER"
    LINKED_CAMPAIGN_COORDINATOR = "LINKED_CAMPAIGN_COORDINATOR"
    PROGRAMME_AUTHOR = "PROGRAMME_AUTHOR"
    SOURCE_PROVIDER = "SOURCE_PROVIDER"


class CapabilityExecutionStatus(StrEnum):
    """Discovery/readiness states; none of them grants operation authority."""

    STATIC_ONLY = "STATIC_ONLY"
    EXECUTABLE_INPUTS_REQUIRED = "EXECUTABLE_INPUTS_REQUIRED"
    EXECUTABLE_SOURCE_NOT_READY = "EXECUTABLE_SOURCE_NOT_READY"
    EXECUTABLE_AUTHORITY_REQUIRED = "EXECUTABLE_AUTHORITY_REQUIRED"
    EXECUTABLE_READY = "EXECUTABLE_READY"


class CapabilityExecutionReason(StrEnum):
    """Closed, machine-readable explanations for execution availability."""

    NO_EXECUTABLE_BINDING_INSTALLED = "NO_EXECUTABLE_BINDING_INSTALLED"
    EXECUTABLE_INPUTS_NOT_SUPPLIED = "EXECUTABLE_INPUTS_NOT_SUPPLIED"
    SOURCE_QUALIFICATION_INCOMPLETE = "SOURCE_QUALIFICATION_INCOMPLETE"
    STORAGE_NOT_READY = "STORAGE_NOT_READY"
    NETWORK_NOT_READY = "NETWORK_NOT_READY"
    OPERATION_AUTHORITY_REQUIRED = "OPERATION_AUTHORITY_REQUIRED"
    PROVIDER_CONSTRUCTED = "PROVIDER_CONSTRUCTED"


@dataclass(frozen=True, slots=True)
class ExecutableRecordTypeBinding(CanonicalRecord):
    """One accepted canonical record schema and semantic record version."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-record-type-binding'

    record_schema: str
    record_version: str

    def __post_init__(self) -> None:
        validate_schema(self.record_schema)
        validate_semantic_version(self.record_version)

    @property
    def type_id(self) -> str:
        return f"{self.record_schema}@{self.record_version}"


@dataclass(frozen=True, slots=True)
class ExecutableCapabilityBinding(CanonicalRecord):
    """Static association between one discovered subject and installed code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-capability-binding'

    binding_id: str
    capability_key: str
    capability_version: str
    capability_implementation_sha256: str
    role: ExecutableBindingRole
    provider_key: str
    provider_version: str
    provider_implementation_sha256: str
    capability_backed: bool
    discovery_components: tuple[ExtensionComponentRegistration, ...]
    accepted_profile_types: tuple[ExecutableRecordTypeBinding, ...]
    accepted_config_types: tuple[ExecutableRecordTypeBinding, ...]
    required_issued_payload_schemas: tuple[str, ...]
    codec_registration_identities: tuple[ObjectIdentity, ...]
    issued_decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    input_schema_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    artifact_validator_identities: tuple[ObjectIdentity, ...]
    required_platform_port_keys: tuple[str, ...]
    may_require_active_mount: bool
    may_require_source_qualification: bool
    may_require_network: bool
    may_require_authority: bool
    required_authenticated_record_schemas: tuple[str, ...] = ()
    grants_authority: bool = False
    embeds_scientific_payload: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("capability_key", self.capability_key),
            ("provider_key", self.provider_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        validate_semantic_version(self.provider_version)
        validate_sha256(
            self.capability_implementation_sha256,
            field_name="capability_implementation_sha256",
        )
        validate_sha256(
            self.provider_implementation_sha256,
            field_name="provider_implementation_sha256",
        )
        require_sorted_unique_ids(
            self.discovery_components,
            attribute="registration_id",
            field_name="discovery_components",
        )
        for name in ("accepted_profile_types", "accepted_config_types"):
            values = getattr(self, name)
            identifiers = tuple(value.type_id for value in values)
            if tuple(sorted(set(identifiers))) != identifiers:
                raise ValueError(f"{name} must have sorted, unique record types")
        for name in (
            "required_issued_payload_schemas",
            "required_authenticated_record_schemas",
            "input_schema_ids",
            "output_schema_ids",
        ):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_schema(value)
        if not self.output_schema_ids:
            raise ValueError("executable binding requires an output schema")
        for name in (
            "codec_registration_identities",
            "artifact_validator_identities",
        ):
            require_sorted_unique_ids(
                getattr(self, name),
                attribute="object_id",
                field_name=name,
            )
        require_sorted_unique_ids(
            self.issued_decoder_registrations,
            attribute="registration_id",
            field_name="issued_decoder_registrations",
        )
        decoder_keys = tuple(
            (value.decoder_key, value.decoder_version)
            for value in self.issued_decoder_registrations
        )
        if len(set(decoder_keys)) != len(decoder_keys):
            raise ValueError("issued decoder registrations overlap a decoder key/version")
        decoder_payloads = tuple(
            value.payload_schema for value in self.issued_decoder_registrations
        )
        if len(set(decoder_payloads)) != len(decoder_payloads):
            raise ValueError("issued decoder registrations overlap a payload schema")
        require_sorted_unique_strings(
            self.required_platform_port_keys,
            field_name="required_platform_port_keys",
        )
        for value in self.required_platform_port_keys:
            validate_stable_id(value, field_name="required_platform_port_key")
        if not self.capability_backed and not self.discovery_components:
            raise ValueError("component-backed executable binding requires a component")
        accepted_schemas = self.accepted_record_schema_ids
        if not set(self.required_issued_payload_schemas).issubset(accepted_schemas):
            raise ValueError("required issued payload schema is not accepted by the binding")
        if not set(self.required_authenticated_record_schemas).issubset(accepted_schemas):
            raise ValueError("required authenticated record schema is not accepted by the binding")
        if set(self.required_issued_payload_schemas) & set(
            self.required_authenticated_record_schemas
        ):
            raise ValueError("issued and separately authenticated record ownership overlaps")
        decoder_payload_schemas = {
            value.payload_schema for value in self.issued_decoder_registrations
        }
        if not set(self.required_issued_payload_schemas).issubset(decoder_payload_schemas):
            raise ValueError("required issued payloads lack an installed decoder registration")
        if not decoder_payload_schemas.issubset(accepted_schemas):
            raise ValueError("installed issued decoders are not accepted by the binding")
        codec_components = tuple(
            value
            for value in self.discovery_components
            if value.kind is ExtensionComponentKind.CONFIG_DECODER
        )
        codec_component_identities = {_component_identity(value) for value in codec_components}
        if codec_component_identities != set(self.codec_registration_identities):
            raise ValueError(
                "static codec identities differ from their discovery component records"
            )
        for registration in self.issued_decoder_registrations:
            if not any(
                component.component_key == registration.decoder_key
                and component.component_version == registration.decoder_version
                and component.implementation_sha256 == registration.implementation_sha256
                and registration.payload_schema
                in {*component.input_schema_ids, *component.output_schema_ids}
                for component in codec_components
            ):
                raise ValueError(
                    "issued decoder registration lacks an exact static codec compatibility"
                )
        if self.grants_authority or self.embeds_scientific_payload:
            raise ValueError(
                "executable binding cannot grant authority or embed scientific payload"
            )

    @property
    def capability_registry_id(self) -> str:
        return f"{self.capability_key}@{self.capability_version}"

    @property
    def accepted_record_schema_ids(self) -> frozenset[str]:
        return frozenset(
            (
                *(value.record_schema for value in self.accepted_profile_types),
                *(value.record_schema for value in self.accepted_config_types),
                *self.input_schema_ids,
            )
        )

    @property
    def construction_record_schema_ids(self) -> frozenset[str]:
        """Record schemas consumed while constructing this provider.

        Capability input schemas describe runtime task artifacts.  They may
        lawfully overlap between producer/consumer stages and therefore do not
        establish issued-payload ownership during provider reconstruction.
        """

        return frozenset(
            (
                *(value.record_schema for value in self.accepted_profile_types),
                *(value.record_schema for value in self.accepted_config_types),
            )
        )

    def accepts_issued_decoder_registration(
        self,
        registration: StudyExtensionDecoderRegistration,
    ) -> bool:
        """Require the full decoder/config/implementation record, never a name alone."""

        return registration in self.issued_decoder_registrations


@dataclass(frozen=True, slots=True)
class ExecutableBindingContribution(CanonicalRecord):
    """One import-safe package descriptor containing only static bindings."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-binding-contribution'

    contribution_id: str
    contribution_version: str
    bindings: tuple[ExecutableCapabilityBinding, ...]
    grants_authority: bool = False
    embeds_scientific_payload: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.contribution_id, field_name="contribution_id")
        validate_semantic_version(self.contribution_version)
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        if not self.bindings:
            raise ValueError("executable contribution requires a binding")
        if self.grants_authority or self.embeds_scientific_payload:
            raise ValueError(
                "executable contribution cannot grant authority or embed scientific payload"
            )


@dataclass(frozen=True, slots=True)
class CapabilityExecutionAvailability(CanonicalRecord):
    """Static executable availability for one exact discovery subject."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-execution-availability'

    availability_id: str
    discovery_subject: ObjectIdentity
    binding: ObjectIdentity | None
    role: ExecutableBindingRole | None
    status: CapabilityExecutionStatus
    reason_codes: tuple[CapabilityExecutionReason, ...]
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.availability_id, field_name="availability_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.status is CapabilityExecutionStatus.STATIC_ONLY:
            if self.binding is not None or self.role is not None:
                raise ValueError("static-only availability cannot name an executable binding")
            if CapabilityExecutionReason.NO_EXECUTABLE_BINDING_INSTALLED not in self.reason_codes:
                raise ValueError("static-only availability requires its closed reason code")
        elif self.status is CapabilityExecutionStatus.EXECUTABLE_INPUTS_REQUIRED:
            if self.binding is None or self.role is None:
                raise ValueError("executable availability requires a binding and role")
            if CapabilityExecutionReason.EXECUTABLE_INPUTS_NOT_SUPPLIED not in self.reason_codes:
                raise ValueError("static executable availability cannot claim resolved inputs")
        else:
            raise ValueError("dynamic readiness cannot be stored as static availability")
        if self.grants_authority:
            raise ValueError("execution availability cannot grant authority")


@dataclass(frozen=True, slots=True)
class CapabilityExecutionReadiness(CanonicalRecord):
    """Dynamic, non-authorizing readiness readout for exact supplied inputs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-execution-readiness'

    readiness_id: str
    availability: ObjectIdentity
    binding: ObjectIdentity | None
    status: CapabilityExecutionStatus
    reason_codes: tuple[CapabilityExecutionReason, ...]
    resolved_record_identities: tuple[ObjectIdentity, ...]
    qualification_identities: tuple[ObjectIdentity, ...]
    provider_registry_sha256: str | None
    grants_authority: bool = False
    executed: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.readiness_id, field_name="readiness_id")
        if self.availability.object_schema != CapabilityExecutionAvailability.SCHEMA:
            raise ValueError("readiness binds another availability schema")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        for name in ("resolved_record_identities", "qualification_identities"):
            require_sorted_unique_ids(
                getattr(self, name),
                attribute="object_id",
                field_name=name,
            )
        if self.provider_registry_sha256 is not None:
            validate_sha256(
                self.provider_registry_sha256,
                field_name="provider_registry_sha256",
            )
        if self.status is CapabilityExecutionStatus.STATIC_ONLY:
            if self.binding is not None or self.provider_registry_sha256 is not None:
                raise ValueError("static-only readiness cannot name executable products")
        elif self.binding is None:
            raise ValueError("non-static readiness requires an executable binding")
        if self.status is CapabilityExecutionStatus.EXECUTABLE_READY:
            if self.provider_registry_sha256 is None:
                raise ValueError("ready status requires an exact provider registry")
            if CapabilityExecutionReason.PROVIDER_CONSTRUCTED not in self.reason_codes:
                raise ValueError("ready status requires provider-construction evidence")
        if self.grants_authority or self.executed:
            raise ValueError("readiness cannot grant authority or claim execution")


@dataclass(frozen=True, slots=True)
class GeneratedExecutableBindingAggregate(CanonicalRecord):
    """Generated executable root cross-linked to one exact discovery aggregate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/generated-executable-binding-aggregate'

    aggregate_id: str
    aggregate_version: str
    discovery_aggregate: ObjectIdentity
    contributions: tuple[ObjectIdentity, ...]
    bindings: tuple[ExecutableCapabilityBinding, ...]
    availability: tuple[CapabilityExecutionAvailability, ...]
    grants_authority: bool = False
    embeds_scientific_payload: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        validate_semantic_version(self.aggregate_version)
        if self.discovery_aggregate.object_schema != GeneratedExtensionBundleAggregate.SCHEMA:
            raise ValueError("executable aggregate binds another discovery schema")
        require_sorted_unique_ids(
            self.contributions,
            attribute="object_id",
            field_name="contributions",
        )
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        require_sorted_unique_ids(
            self.availability,
            attribute="availability_id",
            field_name="availability",
        )
        if not self.contributions or not self.bindings or not self.availability:
            raise ValueError("generated executable aggregate omits a required roster")
        capability_keys = tuple(
            (value.capability_key, value.capability_version) for value in self.bindings
        )
        if len(set(capability_keys)) != len(capability_keys):
            raise ValueError("generated executable aggregate overlaps a capability binding")
        provider_keys = tuple(
            (
                value.provider_key,
                value.provider_version,
                value.provider_implementation_sha256,
            )
            for value in self.bindings
        )
        if len(set(provider_keys)) != len(provider_keys):
            raise ValueError("generated executable aggregate overlaps a provider binding")
        known_bindings = {
            value.binding_id: ObjectIdentity.from_record(value.binding_id, value)
            for value in self.bindings
        }
        represented: set[str] = set()
        subjects: set[tuple[str, str, str, str]] = set()
        for value in self.availability:
            subject_key = (
                value.discovery_subject.object_id,
                value.discovery_subject.object_schema,
                value.discovery_subject.object_version,
                value.discovery_subject.object_fingerprint,
            )
            if subject_key in subjects:
                raise ValueError("execution availability repeats a discovery subject")
            subjects.add(subject_key)
            if value.binding is None:
                continue
            expected = known_bindings.get(value.binding.object_id)
            if expected != value.binding:
                raise ValueError("execution availability binds an unknown executable record")
            if value.role is not next(
                binding.role
                for binding in self.bindings
                if binding.binding_id == value.binding.object_id
            ):
                raise ValueError("execution availability reports another binding role")
            represented.add(value.binding.object_id)
        if represented != set(known_bindings):
            raise ValueError("each executable binding must cross-link discovery availability")
        if self.grants_authority or self.embeds_scientific_payload:
            raise ValueError(
                "generated executable aggregate cannot grant authority or embed payload"
            )

    def binding(
        self,
        capability_key: str,
        capability_version: str,
    ) -> ExecutableCapabilityBinding | None:
        return next(
            (
                value
                for value in self.bindings
                if value.capability_key == capability_key
                and value.capability_version == capability_version
            ),
            None,
        )

    def availability_for_subject(
        self,
        object_id: str,
    ) -> CapabilityExecutionAvailability | None:
        validate_stable_id(object_id, field_name="object_id")
        return next(
            (
                value
                for value in self.availability
                if value.discovery_subject.object_id == object_id
            ),
            None,
        )

    def has_executable_binding(
        self,
        capability_key: str,
        capability_version: str,
        *,
        role: ExecutableBindingRole | None = None,
    ) -> bool:
        """Report an installed static binding without constructing its product."""

        value = self.binding(capability_key, capability_version)
        return value is not None and (role is None or value.role is role)


def _capability_manifest_object_id(manifest: CapabilityManifest) -> str:
    return (
        f"capability-manifest.{manifest.capability_key}."
        f"{manifest.capability_version.replace('.', '-')}"
    )


def _component_identity(value: ExtensionComponentRegistration) -> ObjectIdentity:
    return ObjectIdentity.from_record(value.registration_id, value)


def _discovery_component_groups(
    discovery: GeneratedExtensionBundleAggregate,
) -> dict[ExtensionComponentKind, frozenset[ObjectIdentity]]:
    return {
        ExtensionComponentKind.METHOD: frozenset(discovery.bundle.method_registrations),
        ExtensionComponentKind.CONFIG_DECODER: frozenset(discovery.bundle.config_decoders),
        ExtensionComponentKind.ARTIFACT_VALIDATOR: frozenset(
            discovery.bundle.artifact_validators
        ),
        ExtensionComponentKind.RUNTIME_PROVIDER: frozenset(discovery.bundle.runtime_providers),
        ExtensionComponentKind.PROGRAMME_AUTHOR: frozenset(discovery.bundle.study_authors),
    }


def _validate_binding_against_discovery(
    binding: ExecutableCapabilityBinding,
    discovery: GeneratedExtensionBundleAggregate,
) -> tuple[ObjectIdentity, ...]:
    groups = _discovery_component_groups(discovery)
    component_subjects: list[ObjectIdentity] = []
    for component in binding.discovery_components:
        identity = _component_identity(component)
        if identity not in groups[component.kind]:
            raise ValueError(
                f"executable binding {binding.binding_id} has an unknown discovery component"
            )
        component_subjects.append(identity)
    if any(
        value not in groups[ExtensionComponentKind.CONFIG_DECODER]
        for value in binding.codec_registration_identities
    ):
        raise ValueError("executable binding codec differs from generated discovery")
    if any(
        value not in groups[ExtensionComponentKind.ARTIFACT_VALIDATOR]
        for value in binding.artifact_validator_identities
    ):
        raise ValueError("executable binding artifact validator differs from discovery")

    manifest = next(
        (
            value
            for value in discovery.capability_registry.capabilities
            if value.capability_key == binding.capability_key
            and value.capability_version == binding.capability_version
        ),
        None,
    )
    subjects: list[ObjectIdentity] = []
    if binding.capability_backed:
        if manifest is None:
            raise ValueError("executable binding lacks its discovery capability")
        if manifest.implementation_sha256 != binding.capability_implementation_sha256:
            raise ValueError("executable binding capability implementation differs")
        if manifest.input_schema_ids != binding.input_schema_ids:
            raise ValueError("executable binding input schemas differ from discovery")
        if manifest.output_schema_ids != binding.output_schema_ids:
            raise ValueError("executable binding output schemas differ from discovery")
        if manifest.config_schema not in {
            value.record_schema for value in binding.accepted_config_types
        }:
            raise ValueError("executable binding omits the discovered config schema")
        if manifest.requires_active_mount and not binding.may_require_active_mount:
            raise ValueError("executable binding understates active-mount readiness")
        if manifest.requires_network and not binding.may_require_network:
            raise ValueError("executable binding understates network readiness")
        subjects.append(
            ObjectIdentity.from_record(_capability_manifest_object_id(manifest), manifest)
        )
    else:
        if not any(
            component.component_key == binding.capability_key
            and component.component_version == binding.capability_version
            and component.implementation_sha256 == binding.capability_implementation_sha256
            and component.input_schema_ids == binding.input_schema_ids
            and component.output_schema_ids == binding.output_schema_ids
            for component in binding.discovery_components
        ):
            raise ValueError("component-backed executable binding metadata differs")
    subjects.extend(component_subjects)
    return tuple(sorted(set(subjects), key=lambda value: value.object_id))


def compose_executable_binding_aggregate(
    contributions: tuple[ExecutableBindingContribution, ...],
    *,
    discovery_aggregate: GeneratedExtensionBundleAggregate,
    aggregate_id: str = "executable-binding-aggregate.consolidated-backbone",
    aggregate_version: str = "1.0.0",
) -> GeneratedExecutableBindingAggregate:
    """Compose and cross-check the deterministic executable/discovery join."""

    ordered = tuple(sorted(contributions, key=lambda value: value.contribution_id))
    require_sorted_unique_ids(
        ordered,
        attribute="contribution_id",
        field_name="contributions",
    )
    if not ordered:
        raise ValueError("generated executable aggregate requires contributions")
    bindings = tuple(
        sorted(
            (value for contribution in ordered for value in contribution.bindings),
            key=lambda value: value.binding_id,
        )
    )
    require_sorted_unique_ids(bindings, attribute="binding_id", field_name="bindings")
    capability_subjects = {
        ObjectIdentity.from_record(_capability_manifest_object_id(value), value)
        for value in discovery_aggregate.capability_registry.capabilities
    }
    component_subjects = set(
        (
            *discovery_aggregate.bundle.method_registrations,
            *discovery_aggregate.bundle.runtime_providers,
            *discovery_aggregate.bundle.study_authors,
        )
    )
    all_subjects = capability_subjects | component_subjects
    binding_by_subject: dict[ObjectIdentity, ExecutableCapabilityBinding] = {}
    decoder_subjects = set(discovery_aggregate.bundle.config_decoders)
    capability_keys: set[tuple[str, str]] = set()
    provider_bindings: set[tuple[str, str, str]] = set()
    for binding in bindings:
        capability_key = (binding.capability_key, binding.capability_version)
        if capability_key in capability_keys:
            raise ValueError("executable contributions overlap a capability binding")
        capability_keys.add(capability_key)
        provider_key = (
            binding.provider_key,
            binding.provider_version,
            binding.provider_implementation_sha256,
        )
        if provider_key in provider_bindings:
            raise ValueError("executable contributions overlap a provider binding")
        provider_bindings.add(provider_key)
        for subject in _validate_binding_against_discovery(binding, discovery_aggregate):
            previous = binding_by_subject.get(subject)
            if previous is not None:
                if subject not in decoder_subjects:
                    raise ValueError("executable bindings overlap a discovery subject")
                # Compile-time and runtime consumers may reference one exact
                # installed codec; selected runtime ownership stays exclusive.
                component = next(
                    c for c in binding.discovery_components if _component_identity(c) == subject
                )
                current = tuple(
                    r
                    for r in binding.issued_decoder_registrations
                    if r.decoder_key == component.component_key
                )
                prior = tuple(
                    r
                    for r in previous.issued_decoder_registrations
                    if r.decoder_key == component.component_key
                )
                if not current or current != prior:
                    raise ValueError("shared executable decoder registrations differ")
                # Availability names the first deterministic consuming binding.
                continue
            binding_by_subject[subject] = binding
            all_subjects.add(subject)

    availability: list[CapabilityExecutionAvailability] = []
    for subject in sorted(all_subjects, key=lambda value: value.object_id):
        subject_binding = binding_by_subject.get(subject)
        if subject_binding is None:
            binding_identity = None
            role = None
            status = CapabilityExecutionStatus.STATIC_ONLY
            reasons = (CapabilityExecutionReason.NO_EXECUTABLE_BINDING_INSTALLED,)
        else:
            binding_identity = ObjectIdentity.from_record(
                subject_binding.binding_id,
                subject_binding,
            )
            role = subject_binding.role
            status = CapabilityExecutionStatus.EXECUTABLE_INPUTS_REQUIRED
            reasons = (CapabilityExecutionReason.EXECUTABLE_INPUTS_NOT_SUPPLIED,)
        availability.append(
            CapabilityExecutionAvailability(
                availability_id=f"execution-availability.{subject.object_id}",
                discovery_subject=subject,
                binding=binding_identity,
                role=role,
                status=status,
                reason_codes=reasons,
            )
        )
    return GeneratedExecutableBindingAggregate(
        aggregate_id=aggregate_id,
        aggregate_version=aggregate_version,
        discovery_aggregate=ObjectIdentity.from_record(
            discovery_aggregate.aggregate_id,
            discovery_aggregate,
        ),
        contributions=tuple(
            ObjectIdentity.from_record(value.contribution_id, value) for value in ordered
        ),
        bindings=bindings,
        availability=tuple(availability),
    )


@dataclass(frozen=True, slots=True)
class ExecutablePlatformPort:
    """One explicitly named, code-injected platform port (never canonical config)."""

    port_key: str
    port: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_stable_id(self.port_key, field_name="port_key")


@runtime_checkable
class ExecutableCapabilityProviderFactory(Protocol):
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider: ...


@runtime_checkable
class ExecutableProfileCompilerFactory(Protocol):
    binding: ExecutableCapabilityBinding

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object: ...


@runtime_checkable
class ExecutableLinkedCampaignCoordinatorFactory(Protocol):
    binding: ExecutableCapabilityBinding

    def build_coordinator(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object: ...


@runtime_checkable
class ExecutableStudyAuthorFactory(Protocol):
    binding: ExecutableCapabilityBinding

    def build_study_author(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object: ...


@runtime_checkable
class ExecutableSourceProviderFactory(Protocol):
    binding: ExecutableCapabilityBinding

    def build_source_provider(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object: ...


ExecutableFactory = (
    ExecutableCapabilityProviderFactory
    | ExecutableProfileCompilerFactory
    | ExecutableLinkedCampaignCoordinatorFactory
    | ExecutableStudyAuthorFactory
    | ExecutableSourceProviderFactory
)


_FACTORY_METHOD_BY_ROLE = {
    ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER: "build_provider",
    ExecutableBindingRole.PROFILE_COMPILER: "build_compiler",
    ExecutableBindingRole.LINKED_CAMPAIGN_COORDINATOR: "build_coordinator",
    ExecutableBindingRole.PROGRAMME_AUTHOR: 'build_study_author',
    ExecutableBindingRole.SOURCE_PROVIDER: "build_source_provider",
}


class ExecutableCapabilityProviderFactoryRegistry:
    """Closed code-owned factory objects authenticated by the canonical aggregate."""

    def __init__(
        self,
        *,
        aggregate: GeneratedExecutableBindingAggregate,
        factories: tuple[ExecutableFactory, ...],
    ) -> None:
        identifiers = tuple(value.binding.binding_id for value in factories)
        if tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("executable factories must have sorted unique binding IDs")
        expected = tuple(value.binding_id for value in aggregate.bindings)
        if identifiers != expected:
            raise ValueError("factory roster differs from the canonical executable aggregate")
        by_id = {value.binding_id: value for value in aggregate.bindings}
        for factory in factories:
            binding = factory.binding
            if by_id.get(binding.binding_id) != binding:
                raise ValueError("factory metadata differs from its canonical binding")
            method = _FACTORY_METHOD_BY_ROLE[binding.role]
            if not callable(getattr(factory, method, None)):
                raise ValueError(f"factory lacks the construction method for {binding.role}")
        self.aggregate = aggregate
        self._factories = factories

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(value.binding.binding_id for value in self._factories)

    @property
    def capability_count(self) -> int:
        return len(self._factories)

    @property
    def fingerprint(self) -> str:
        return self.aggregate.fingerprint()

    def factory(self, binding_id: str) -> ExecutableFactory:
        validate_stable_id(binding_id, field_name="binding_id")
        for value in self._factories:
            if value.binding.binding_id == binding_id:
                return value
        raise KeyError(f"no executable factory is registered for {binding_id}")

    def provider_factory(
        self,
        binding_id: str,
    ) -> ExecutableCapabilityProviderFactory:
        value = self.factory(binding_id)
        if value.binding.role is not ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER:
            raise TypeError("binding is not a campaign runtime-provider factory")
        if not isinstance(value, ExecutableCapabilityProviderFactory):
            raise TypeError("campaign runtime-provider factory protocol is not satisfied")
        return value


class CampaignRuntimeProviderResolver(Protocol):
    """Resolve a complete immutable provider for one exact capability registry."""

    def resolve(
        self,
        registry: CapabilityRegistry,
        *,
        decoded_records: tuple[CanonicalRecord, ...] = (),
        platform_ports: tuple[ExecutablePlatformPort, ...] = (),
    ) -> CampaignRuntimeProvider: ...

    @property
    def resolver_fingerprint(self) -> str: ...


class CompositeCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Delegate one complete registry to exact, nonoverlapping provider products."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        bindings: tuple[ExecutableCapabilityBinding, ...],
        providers: tuple[CampaignRuntimeProvider, ...],
    ) -> None:
        if len(bindings) != len(providers) or len(bindings) != len(registry.capabilities):
            raise ValueError("composite provider roster does not cover the registry")
        expected_ids = tuple(value.registry_id for value in registry.capabilities)
        observed_ids = tuple(value.capability_registry_id for value in bindings)
        if observed_ids != expected_ids:
            raise ValueError("composite binding coverage differs from the registry")
        for manifest, binding, provider in zip(
            registry.capabilities,
            bindings,
            providers,
            strict=True,
        ):
            if binding.role is not ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER:
                raise ValueError("non-runtime executable role cannot enter a composite provider")
            if (
                manifest.capability_key != binding.capability_key
                or manifest.capability_version != binding.capability_version
                or manifest.implementation_sha256 != binding.capability_implementation_sha256
            ):
                raise ValueError("composite provider capability implementation differs")
            if provider.registry_sha256 != registry.fingerprint():
                raise ValueError("factory provider is bound to another registry fingerprint")
            if provider.capability_count <= 0:
                raise ValueError("factory provider has an invalid capability count")
        source_schema_owners: dict[str, int] = {}
        for index, provider in enumerate(providers):
            schemas = provider.issued_source_schema_ids
            require_sorted_unique_strings(schemas, field_name="issued_source_schema_ids")
            for schema in schemas:
                validate_schema(schema)
                if schema in source_schema_owners:
                    raise ValueError("composite providers overlap issued-source ownership")
                source_schema_owners[schema] = index
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self.bindings = bindings
        self._providers = providers
        self._source_schema_owners = source_schema_owners
        self.issued_source_schema_ids = tuple(sorted(source_schema_owners))
        self.composition_sha256 = hashlib.sha256(
            canonical_json_bytes(
                (
                    self.registry_sha256,
                    *(value.fingerprint() for value in self.bindings),
                )
            )
        ).hexdigest()

    def _require_registry(self, registry: CapabilityRegistry) -> None:
        if registry != self.registry or registry.fingerprint() != self.registry_sha256:
            raise ValueError("composite provider registry identity differs")

    def _source_records_by_provider(
        self,
        source_records: tuple[CanonicalRecord, ...],
    ) -> tuple[tuple[CanonicalRecord, ...], ...]:
        schemas = tuple(value.SCHEMA for value in source_records)
        if tuple(sorted(schemas)) != schemas:
            raise ValueError("issued source records must be sorted by schema")
        if set(schemas) != set(self.issued_source_schema_ids):
            raise ValueError("issued source record roster differs from provider ownership")
        routed: list[list[CanonicalRecord]] = [[] for _ in self._providers]
        for value in source_records:
            routed[self._source_schema_owners[value.SCHEMA]].append(value)
        return tuple(tuple(value) for value in routed)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._require_registry(registry)
        routed = self._source_records_by_provider(source_records)
        runners: list[TaskRunner] = []
        for binding, provider, records in zip(
            self.bindings,
            self._providers,
            routed,
            strict=True,
        ):
            values = provider.runners(registry, records)
            if len(values) != 1:
                raise ValueError("each executable factory must own exactly one task runner")
            runner = values[0]
            if (
                runner.manifest.capability_key != binding.capability_key
                or runner.manifest.capability_version != binding.capability_version
                or runner.manifest.implementation_sha256 != binding.capability_implementation_sha256
            ):
                raise ValueError("factory runner ownership differs from its binding")
            runners.append(runner)
        identifiers = tuple(value.manifest.registry_id for value in runners)
        expected = tuple(value.registry_id for value in registry.capabilities)
        if identifiers != expected:
            raise ValueError("composite runner coverage differs from the registry")
        return tuple(runners)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("composite provider plan registry differs")
        routed = self._source_records_by_provider(source_records)
        binding_by_capability = {
            (value.capability_key, value.capability_version): index
            for index, value in enumerate(self.bindings)
        }
        expected: list[set[str]] = [set() for _ in self._providers]
        seen_expected: dict[str, int] = {}
        exact_specs: dict[str, object] = {}
        for task in plan.tasks:
            key = (task.capability.capability_key, task.capability.capability_version)
            owner = binding_by_capability.get(key)
            if owner is None or (
                task.capability_implementation_sha256
                != self.bindings[owner].capability_implementation_sha256
            ):
                raise ValueError("execution task lacks exact composite-provider ownership")
            for spec in task.external_inputs:
                previous = seen_expected.get(spec.logical_artifact_id)
                if previous is not None:
                    if exact_specs[spec.logical_artifact_id] != spec:
                        raise ValueError("composite providers disagree on exact shared external input")
                    if previous != owner:
                        selected = min(previous, owner)
                        expected[previous].discard(spec.logical_artifact_id)
                        expected[selected].add(spec.logical_artifact_id)
                        seen_expected[spec.logical_artifact_id] = selected
                    continue
                exact_specs[spec.logical_artifact_id] = spec
                seen_expected[spec.logical_artifact_id] = owner
                expected[owner].add(spec.logical_artifact_id)
        values: list[ExternalInputPayload] = []
        for index, (provider, records) in enumerate(zip(self._providers, routed, strict=True)):
            provided = provider.external_inputs(plan, records)
            identifiers = tuple(value.logical_artifact_id for value in provided)
            if tuple(sorted(set(identifiers))) != identifiers:
                raise ValueError("provider external inputs must be sorted and unique")
            if set(identifiers) != expected[index]:
                raise ValueError("factory external-input ownership differs from its tasks")
            values.extend(provided)
        values.sort(key=lambda value: value.logical_artifact_id)
        identifiers = tuple(value.logical_artifact_id for value in values)
        if tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("composite providers conflict for an external input")
        if set(identifiers) != set(seen_expected):
            raise ValueError("composite provider does not cover every external input")
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._require_registry(registry)
        if execution_plan is not None and execution_plan.registry_sha256 != self.registry_sha256:
            raise ValueError("semantic execution plan registry differs")
        values: list[CapabilityOutputSemanticContract] = []
        for binding, provider in zip(self.bindings, self._providers, strict=True):
            provided = provider.output_semantic_contracts(registry, execution_plan)
            schemas: set[str] = set()
            for contract in provided:
                if (
                    contract.capability_key != binding.capability_key
                    or contract.capability_version != binding.capability_version
                    or contract.capability_implementation_sha256
                    != binding.capability_implementation_sha256
                ):
                    raise ValueError("factory semantic ownership differs from its binding")
                schemas.add(contract.payload_schema)
            if schemas != set(binding.output_schema_ids):
                raise ValueError("factory semantic contracts do not cover binding outputs")
            values.extend(provided)
        keys = tuple(value.key for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("composite providers conflict for output semantics")
        if execution_plan is not None:
            available = set(keys)
            selected = set()
            for task in execution_plan.tasks:
                for output in task.outputs:
                    key = (
                        task.capability.capability_key,
                        task.capability.capability_version,
                        output.payload_schema,
                        output.profile,
                    )
                    if key not in available:
                        raise ValueError("execution output lacks composite semantic ownership")
                    selected.add(key)
            # Factory inventory is complete above. A plan requests only its
            # selected products; unused declared outputs are not task outputs.
            values = [value for value in values if value.key in selected]
        return tuple(sorted(values, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        self._require_registry(registry)
        values: list[ScientificAdjudicationOutputContract] = []
        for binding, provider in zip(self.bindings, self._providers, strict=True):
            value = provider.scientific_adjudication_contract(registry, execution_plan)
            if value is None:
                continue
            if (
                value.capability_key != binding.capability_key
                or value.capability_version != binding.capability_version
                or value.payload_schema not in binding.output_schema_ids
            ):
                raise ValueError("factory adjudication ownership differs from its binding")
            values.append(value)
        if len(values) > 1:
            raise ValueError("composite provider has conflicting adjudication locators")
        return values[0] if values else None


class LayeredCampaignRuntimeProviderResolver:
    """Prefer exact precomposition, then construct a complete factory composite."""

    def __init__(
        self,
        *,
        precomposed: CampaignRuntimeProviderRegistry,
        factories: ExecutableCapabilityProviderFactoryRegistry,
    ) -> None:
        self.precomposed = precomposed
        self.factories = factories
        self.resolver_fingerprint = hashlib.sha256(
            canonical_json_bytes(
                (
                    factories.fingerprint,
                    *precomposed.fingerprints,
                )
            )
        ).hexdigest()

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return self.precomposed.fingerprints

    @property
    def capability_count(self) -> int:
        runtime_bindings = sum(
            value.role is ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER
            for value in self.factories.aggregate.bindings
        )
        return self.precomposed.capability_count + runtime_bindings

    def _binding_for_manifest(
        self,
        manifest: CapabilityManifest,
    ) -> ExecutableCapabilityBinding:
        named = tuple(
            value
            for value in self.factories.aggregate.bindings
            if value.capability_backed
            and value.capability_key == manifest.capability_key
            and value.capability_version == manifest.capability_version
        )
        if not named:
            raise KeyError(f"capability is static-only: {manifest.registry_id}")
        if len(named) != 1:  # protected by aggregate validation
            raise ValueError("executable aggregate overlaps a capability")
        binding = named[0]
        if binding.capability_implementation_sha256 != manifest.implementation_sha256:
            raise ValueError("executable binding implementation differs from the registry")
        if binding.role is not ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER:
            raise TypeError("capability binding is not a campaign runtime-provider role")
        return binding

    def authenticate_parent_inputs(
        self,
        *,
        registry: CapabilityRegistry,
        target_candidate: ObjectIdentity,
        scientific_graph_sha256: str,
        bindings: tuple[FrozenParentInputBinding, ...],
    ) -> object | None:
        """Invoke the one selected family authenticator before provider construction."""

        authenticators = []
        for manifest in registry.capabilities:
            binding = self._binding_for_manifest(manifest)
            factory = self.factories.provider_factory(binding.binding_id)
            authenticate = getattr(factory, "authenticate_parent_inputs", None)
            if callable(authenticate):
                authenticators.append(authenticate)
        if not authenticators:
            return None
        if len(authenticators) != 1:
            raise ValueError("selected provider roster has ambiguous parent authentication")
        return cast(
            object,
            authenticators[0](
                target_candidate=target_candidate,
                scientific_graph_sha256=scientific_graph_sha256,
                bindings=bindings,
            ),
        )

    def resolve(
        self,
        registry: CapabilityRegistry,
        *,
        decoded_records: tuple[CanonicalRecord, ...] = (),
        platform_ports: tuple[ExecutablePlatformPort, ...] = (),
    ) -> CampaignRuntimeProvider:
        registry_sha256 = registry.fingerprint()
        try:
            provider = self.precomposed.resolve(registry_sha256)
        except KeyError:
            provider = None
        if provider is not None:
            if decoded_records or platform_ports:
                raise ValueError("precomposed provider leaves construction inputs unconsumed")
            if provider.capability_count != len(registry.capabilities):
                raise ValueError("precomposed provider capability count differs")
            return provider

        record_schemas = tuple(value.SCHEMA for value in decoded_records)
        if tuple(sorted(set(record_schemas))) != record_schemas:
            raise ValueError("decoded executable records must have sorted unique schemas")
        port_keys = tuple(value.port_key for value in platform_ports)
        if tuple(sorted(set(port_keys))) != port_keys:
            raise ValueError("executable platform ports must have sorted unique keys")

        bindings = tuple(self._binding_for_manifest(value) for value in registry.capabilities)
        record_owner: dict[str, int] = {}
        port_owner: dict[str, int] = {}
        for index, binding in enumerate(bindings):
            for schema in binding.construction_record_schema_ids:
                if schema in record_owner:
                    raise ValueError("executable bindings overlap decoded-payload ownership")
                record_owner[schema] = index
            for port_key in binding.required_platform_port_keys:
                if port_key in port_owner:
                    raise ValueError("executable bindings overlap platform-port ownership")
                port_owner[port_key] = index
        routed_records: list[list[CanonicalRecord]] = [[] for _ in bindings]
        for value in decoded_records:
            owner = record_owner.get(value.SCHEMA)
            if owner is None:
                raise ValueError("decoded executable record is unconsumed")
            accepted_type = next(
                (
                    item
                    for item in (
                        *bindings[owner].accepted_profile_types,
                        *bindings[owner].accepted_config_types,
                    )
                    if item.record_schema == value.SCHEMA
                ),
                None,
            )
            if accepted_type is not None and accepted_type.record_version != value.VERSION:
                raise ValueError("decoded executable record version differs from its binding")
            routed_records[owner].append(value)
        routed_ports: list[list[ExecutablePlatformPort]] = [[] for _ in bindings]
        for platform_port in platform_ports:
            owner = port_owner.get(platform_port.port_key)
            if owner is None:
                raise ValueError("executable platform port is unconsumed")
            routed_ports[owner].append(platform_port)

        providers: list[CampaignRuntimeProvider] = []
        for index, binding in enumerate(bindings):
            supplied_schemas = {value.SCHEMA for value in routed_records[index]}
            if not set(binding.required_issued_payload_schemas).issubset(supplied_schemas):
                raise ValueError("required issued executable payload is missing")
            if not set(binding.required_authenticated_record_schemas).issubset(supplied_schemas):
                raise ValueError("required authenticated executable record is missing")
            supplied_ports = {value.port_key for value in routed_ports[index]}
            if supplied_ports != set(binding.required_platform_port_keys):
                raise ValueError("required executable platform port is missing")
            factory = self.factories.provider_factory(binding.binding_id)
            provider = factory.build_provider(
                registry=registry,
                records=tuple(routed_records[index]),
                platform_ports=tuple(routed_ports[index]),
            )
            if not isinstance(provider, CampaignRuntimeProvider):
                raise TypeError("runtime-provider factory returned another product role")
            providers.append(provider)
        return CompositeCampaignRuntimeProvider(
            registry=registry,
            bindings=bindings,
            providers=tuple(providers),
        )


__all__ = [
    "CampaignRuntimeProviderResolver",
    'CapabilityExecutionAvailability',
    'CapabilityExecutionReadiness',
    'CapabilityExecutionReason',
    'CapabilityExecutionStatus',
    "CompositeCampaignRuntimeProvider",
    'ExecutableBindingContribution',
    'ExecutableBindingRole',
    'ExecutableCapabilityBinding',
    "ExecutableCapabilityProviderFactory",
    "ExecutableCapabilityProviderFactoryRegistry",
    "ExecutableFactory",
    "ExecutableLinkedCampaignCoordinatorFactory",
    "ExecutablePlatformPort",
    "ExecutableProfileCompilerFactory",
    'ExecutableStudyAuthorFactory',
    'ExecutableRecordTypeBinding',
    "ExecutableSourceProviderFactory",
    'GeneratedExecutableBindingAggregate',
    "LayeredCampaignRuntimeProviderResolver",
    'compose_executable_binding_aggregate',
]
