"""Noncontacting executable bindings for the MAST-U query source context."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from empirical_lawhood.kernel.authority import ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.source_pipelines import SourcePipelineMode, SourcePipelineProfile
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutableFactory, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
from empirical_lawhood.runtime.profile_compilation import SourceProfileCompilerBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.sources import SourceCapabilityManifest, SourceMode
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)

from .extension_bundle import MASTU_QUERY_CONFIG_DECODERS, MASTU_QUERY_DATASET_REGISTRY, MASTU_QUERY_PROFILE_COMPILER, MASTU_QUERY_PROJECTION_SELECTION, MASTU_QUERY_PROJECTION_MANIFEST, MASTU_QUERY_SOURCE_MANIFEST, MASTU_QUERY_SOURCE_PROVIDER, MASTU_PARAMETERISED_ACQUISITION_CAPABILITY, MASTU_PARAMETERISED_ACQUISITION_PROVIDER, MASTU_PARAMETERISED_COMPACT_CAPABILITY, MASTU_PARAMETERISED_COMPACT_PROVIDER, MASTU_PARAMETERISED_PROJECTION_CAPABILITY, MASTU_PARAMETERISED_PROJECTION_PROVIDER
from .parameterised_provider import MASTUArchiveAcquisitionProvider, MASTUArchiveAcquisitionSessionPort, MASTUArchiveCompactProviderConfig, MASTUArchiveCompactProvider, MASTUArchiveParameterisedProviderConfig, MastArchiveReceiverProjectionProvider
from .archive_receiver_projection import MastArchiveReceiverProjectionConfig, MastArchiveReceiverProjectorPort, MastArchiveReceiverScientificProjection
from .query_source import MASTUArchiveAcquisitionResult, MASTUArchiveCompactManifestResult, MASTUArchiveQueryManifest, MASTUArchiveQuerySourceConfig


_MAXIMUM_QUERY_CONFIG_BYTES = 8 * 1024 * 1024
_DECODER_BY_SCHEMA = {value.input_schema_ids[0]: value for value in MASTU_QUERY_CONFIG_DECODERS}
MASTU_QUERY_DECODER_REGISTRATIONS = tuple(
    StudyExtensionDecoderRegistration(
        registration_id=f"decoder-registration.mastu-query-{label}",
        decoder_key=component.component_key,
        decoder_version=component.component_version,
        payload_schema=record_type.SCHEMA,
        payload_version=record_type.VERSION,
        config_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "decoder_key": component.component_key,
                    "decoder_version": component.component_version,
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest(),
        implementation_sha256=component.implementation_sha256,
        maximum_payload_bytes=_MAXIMUM_QUERY_CONFIG_BYTES,
    )
    for label, record_type, component in (
        (
            "manifest",
            MASTUArchiveQueryManifest,
            _DECODER_BY_SCHEMA[MASTUArchiveQueryManifest.SCHEMA],
        ),
        (
            "source-config",
            MASTUArchiveQuerySourceConfig,
            _DECODER_BY_SCHEMA[MASTUArchiveQuerySourceConfig.SCHEMA],
        ),
        (
            "parameterised-provider-config",
            MASTUArchiveParameterisedProviderConfig,
            _DECODER_BY_SCHEMA[MASTUArchiveParameterisedProviderConfig.SCHEMA],
        ),
        (
            "compact-provider-config",
            MASTUArchiveCompactProviderConfig,
            _DECODER_BY_SCHEMA[MASTUArchiveCompactProviderConfig.SCHEMA],
        ),
        (
            "pf-projection-config",
            MastArchiveReceiverProjectionConfig,
            _DECODER_BY_SCHEMA[MastArchiveReceiverProjectionConfig.SCHEMA],
        ),
    )
)

MASTU_QUERY_SOURCE_CAPABILITY = SourceCapabilityManifest(
    capability_key=MASTU_QUERY_SOURCE_MANIFEST.capability_key,
    capability_version=MASTU_QUERY_SOURCE_MANIFEST.capability_version,
    mode=SourceMode.ARCHIVAL,
    source_access=SourceAccessClass.PRIVATE_CREDENTIAL,
    output_schema_ids=MASTU_QUERY_SOURCE_MANIFEST.output_schema_ids,
    maximum_evidence=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    deterministic=False,
    implementation_sha256=MASTU_QUERY_SOURCE_MANIFEST.implementation_sha256,
    resource_budget=ResourceBudget(2, 2 * 1024**3, 0, 3600, 64 * 1024**3, 64 * 1024**3),
)

_MASTU_QUERY_CONFIG_TYPES: tuple[type[CanonicalRecord], ...] = (
    MASTUArchiveCompactProviderConfig,
    MASTUArchiveParameterisedProviderConfig,
    MASTUArchiveQueryManifest,
    MASTUArchiveQuerySourceConfig,
    MastArchiveReceiverProjectionConfig,
)


MASTU_QUERY_SOURCE_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.mastu-query-source",
    capability_key=MASTU_QUERY_SOURCE_MANIFEST.capability_key,
    capability_version=MASTU_QUERY_SOURCE_MANIFEST.capability_version,
    capability_implementation_sha256=MASTU_QUERY_SOURCE_MANIFEST.implementation_sha256,
    role=ExecutableBindingRole.SOURCE_PROVIDER,
    provider_key=MASTU_QUERY_SOURCE_PROVIDER.component_key,
    provider_version=MASTU_QUERY_SOURCE_PROVIDER.component_version,
    provider_implementation_sha256=MASTU_QUERY_SOURCE_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (MASTU_QUERY_SOURCE_PROVIDER,),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            SourcePipelineProfile.SCHEMA,
            SourcePipelineProfile.VERSION,
        ),
    ),
    accepted_config_types=tuple(
        ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
        for value in sorted(_MASTU_QUERY_CONFIG_TYPES, key=lambda value: value.SCHEMA)
    ),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=MASTU_QUERY_SOURCE_MANIFEST.input_schema_ids,
    output_schema_ids=MASTU_QUERY_SOURCE_MANIFEST.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=True,
    may_require_network=True,
    may_require_authority=True,
    required_authenticated_record_schemas=tuple(
        sorted(
            (
                SourcePipelineProfile.SCHEMA,
                MASTUArchiveQueryManifest.SCHEMA,
                MASTUArchiveQuerySourceConfig.SCHEMA,
            )
        )
    ),
)

MASTU_QUERY_PROFILE_COMPILER_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.mastu-query-profile-compiler",
    capability_key=MASTU_QUERY_PROFILE_COMPILER.component_key,
    capability_version=MASTU_QUERY_PROFILE_COMPILER.component_version,
    capability_implementation_sha256=MASTU_QUERY_PROFILE_COMPILER.implementation_sha256,
    role=ExecutableBindingRole.PROFILE_COMPILER,
    provider_key=MASTU_QUERY_PROFILE_COMPILER.component_key,
    provider_version=MASTU_QUERY_PROFILE_COMPILER.component_version,
    provider_implementation_sha256=MASTU_QUERY_PROFILE_COMPILER.implementation_sha256,
    capability_backed=False,
    discovery_components=(MASTU_QUERY_PROFILE_COMPILER,),
    accepted_profile_types=(),
    accepted_config_types=(),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=(),
    output_schema_ids=MASTU_QUERY_PROFILE_COMPILER.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
)

MASTU_QUERY_PROJECTION_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.mastu-query-projection-selection",
    capability_key=MASTU_QUERY_PROJECTION_MANIFEST.capability_key,
    capability_version=MASTU_QUERY_PROJECTION_MANIFEST.capability_version,
    capability_implementation_sha256=MASTU_QUERY_PROJECTION_MANIFEST.implementation_sha256,
    role=ExecutableBindingRole.SOURCE_PROVIDER,
    provider_key=MASTU_QUERY_PROJECTION_SELECTION.component_key,
    provider_version=MASTU_QUERY_PROJECTION_SELECTION.component_version,
    provider_implementation_sha256=MASTU_QUERY_PROJECTION_SELECTION.implementation_sha256,
    capability_backed=True,
    discovery_components=(MASTU_QUERY_PROJECTION_SELECTION,),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            SourcePipelineProfile.SCHEMA,
            SourcePipelineProfile.VERSION,
        ),
    ),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            MASTU_QUERY_PROJECTION_MANIFEST.config_schema,
            "1.0.0",
        ),
    ),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=MASTU_QUERY_PROJECTION_MANIFEST.input_schema_ids,
    output_schema_ids=MASTU_QUERY_PROJECTION_MANIFEST.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=True,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)


def _parameterised_executable_binding(
    *,
    binding_id: str,
    capability: CapabilityManifest,
    provider_component: ExtensionComponentRegistration,
    accepted_types: tuple[type[CanonicalRecord], ...],
    authenticated_types: tuple[type[CanonicalRecord], ...] = (),
    port_keys: tuple[str, ...] = (),
) -> ExecutableCapabilityBinding:
    issued_types = tuple(value for value in accepted_types if value not in authenticated_types)
    decoder_components = tuple(
        value
        for value in MASTU_QUERY_CONFIG_DECODERS
        if value.input_schema_ids[0] in {item.SCHEMA for item in issued_types}
    )
    decoder_registrations = tuple(
        value
        for value in MASTU_QUERY_DECODER_REGISTRATIONS
        if value.payload_schema in {item.SCHEMA for item in issued_types}
    )
    return ExecutableCapabilityBinding(
        binding_id=binding_id,
        capability_key=capability.capability_key,
        capability_version=capability.capability_version,
        capability_implementation_sha256=capability.implementation_sha256,
        role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
        provider_key=provider_component.component_key,
        provider_version=provider_component.component_version,
        provider_implementation_sha256=provider_component.implementation_sha256,
        capability_backed=True,
        discovery_components=tuple(
            sorted(
                (*decoder_components, provider_component),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=tuple(
            sorted(
                (
                    ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                    for value in accepted_types
                ),
                key=lambda value: value.record_schema,
            )
        ),
        required_issued_payload_schemas=tuple(sorted(value.SCHEMA for value in issued_types)),
        required_authenticated_record_schemas=tuple(
            sorted(value.SCHEMA for value in authenticated_types)
        ),
        codec_registration_identities=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in decoder_components
                ),
                key=lambda value: value.object_id,
            )
        ),
        issued_decoder_registrations=tuple(
            sorted(decoder_registrations, key=lambda value: value.registration_id)
        ),
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=(),
        required_platform_port_keys=port_keys,
        may_require_active_mount=True,
        may_require_source_qualification=True,
        may_require_network=bool(port_keys and "session" in port_keys[0]),
        may_require_authority=True,
    )


MASTU_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING = _parameterised_executable_binding(
    binding_id="binding.mastu-parameterised-acquisition",
    capability=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY,
    provider_component=MASTU_PARAMETERISED_ACQUISITION_PROVIDER,
    accepted_types=(
        ResponseExperimentExtensionSet,
        MASTUArchiveParameterisedProviderConfig,
        MASTUArchiveQueryManifest,
        MASTUArchiveQuerySourceConfig,
        ResponseSubstrateBinding,
    ),
    authenticated_types=(ResponseExperimentExtensionSet, ResponseSubstrateBinding),
    port_keys=("mastu-archive-acquisition-session",),
)
MASTU_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING = _parameterised_executable_binding(
    binding_id="binding.mastu-parameterised-projection",
    capability=MASTU_PARAMETERISED_PROJECTION_CAPABILITY,
    provider_component=MASTU_PARAMETERISED_PROJECTION_PROVIDER,
    accepted_types=(MastArchiveReceiverProjectionConfig,),
    port_keys=("mastu-pf-projector",),
)
MASTU_PARAMETERISED_COMPACT_EXECUTABLE_BINDING = _parameterised_executable_binding(
    binding_id="binding.mastu-parameterised-compact",
    capability=MASTU_PARAMETERISED_COMPACT_CAPABILITY,
    provider_component=MASTU_PARAMETERISED_COMPACT_PROVIDER,
    accepted_types=(MASTUArchiveCompactProviderConfig,),
)


@dataclass(frozen=True, slots=True)
class RegisteredMASTUQuerySourceSelection:
    profile: SourcePipelineProfile
    config: MASTUArchiveQuerySourceConfig
    query_manifest: MASTUArchiveQueryManifest
    source_contacted: bool = False
    grants_authority: bool = False

    def __post_init__(self) -> None:
        if self.source_contacted or self.grants_authority:
            raise ValueError("query-source selection cannot contact or authorize the source")


@dataclass(frozen=True, slots=True)
class MASTUQuerySourceFactory:
    binding: ExecutableCapabilityBinding = MASTU_QUERY_SOURCE_EXECUTABLE_BINDING

    def build_source_provider(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegisteredMASTUQuerySourceSelection:
        if platform_ports:
            raise ValueError("MAST-U query-source selection accepts no platform ports")
        by_schema = {value.SCHEMA: value for value in records}
        expected = {
            SourcePipelineProfile.SCHEMA,
            MASTUArchiveQuerySourceConfig.SCHEMA,
            MASTUArchiveQueryManifest.SCHEMA,
        }
        if len(by_schema) != len(records) or set(by_schema) != expected:
            raise ValueError("MAST-U query-source selection requires its exact record roster")
        profile = by_schema[SourcePipelineProfile.SCHEMA]
        config = by_schema[MASTUArchiveQuerySourceConfig.SCHEMA]
        manifest = by_schema[MASTUArchiveQueryManifest.SCHEMA]
        if not isinstance(profile, SourcePipelineProfile):
            raise TypeError("MAST-U query source profile has another type")
        if not isinstance(config, MASTUArchiveQuerySourceConfig):
            raise TypeError("MAST-U query source config has another type")
        if not isinstance(manifest, MASTUArchiveQueryManifest):
            raise TypeError("MAST-U query manifest has another type")
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        if (
            profile.mode is not SourcePipelineMode.ARCHIVAL
            or profile.source_config != ObjectIdentity.from_record(config.config_id, config)
            or profile.source_materialization != manifest_identity
            or config.query_manifest != manifest_identity
            or profile.expected_source_sha256 != manifest.fingerprint()
            or profile.expected_source_size_bytes != len(manifest.canonical_bytes())
            or profile.source_relative_locator != config.query_manifest_relative_locator
            or profile.source_schema != MASTUArchiveQueryManifest.SCHEMA
            or config.provider_key != manifest.provider_key
            or config.provider_version != manifest.provider_version
            or config.request_schema != manifest.request_schema
            or config.response_object_schema != manifest.response_object_schema
        ):
            raise ValueError("MAST-U source profile fabricates or substitutes query semantics")
        return RegisteredMASTUQuerySourceSelection(profile, config, manifest)


@dataclass(frozen=True, slots=True)
class MASTUQueryProfileCompilerFactory:
    binding: ExecutableCapabilityBinding = MASTU_QUERY_PROFILE_COMPILER_EXECUTABLE_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SourceProfileCompilerBinding:
        if records or platform_ports:
            raise ValueError("MAST-U profile compiler context is input-free")
        capabilities = CapabilityRegistry(
            registry_id="capability-registry.mastu-query-source",
            capabilities=tuple(
                sorted(
                    (MASTU_QUERY_PROJECTION_MANIFEST, MASTU_QUERY_SOURCE_MANIFEST),
                    key=lambda value: value.registry_id,
                )
            ),
        )
        return SourceProfileCompilerBinding(
            binding_id="source-profile-compiler.mastu-query-source",
            source_manifest=MASTU_QUERY_SOURCE_CAPABILITY,
            capability_registry=capabilities,
            dataset_registry=MASTU_QUERY_DATASET_REGISTRY,
        )


@dataclass(frozen=True, slots=True)
class RegisteredMASTUQueryProjectionSelection:
    profile: SourcePipelineProfile
    source_contacted: bool = False
    executable_runner_installed: bool = False

    def __post_init__(self) -> None:
        if self.source_contacted or self.executable_runner_installed:
            raise ValueError("query projection selection cannot claim runtime execution")


@dataclass(frozen=True, slots=True)
class MASTUQueryProjectionSelectionFactory:
    binding: ExecutableCapabilityBinding = MASTU_QUERY_PROJECTION_EXECUTABLE_BINDING

    def build_source_provider(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegisteredMASTUQueryProjectionSelection:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(
                records[0],
                SourcePipelineProfile,
            )
        ):
            raise ValueError("MAST-U projection selection requires one profile and no ports")
        profile = records[0]
        matching_edges = tuple(
            edge
            for edge in profile.transform_edges
            if edge.selection.capability_key == self.binding.capability_key
            and edge.selection.capability_version == self.binding.capability_version
            and edge.selection.implementation_sha256
            == self.binding.capability_implementation_sha256
            and edge.dataset_transform_registry_id == MASTU_QUERY_PROJECTION_MANIFEST.registry_id
            and edge.config.object_schema == MASTU_QUERY_PROJECTION_MANIFEST.config_schema
        )
        if not matching_edges:
            raise ValueError("MAST-U profile does not select the installed projection")
        return RegisteredMASTUQueryProjectionSelection(profile)


@dataclass(frozen=True, slots=True)
class MASTUArchiveAcquisitionProviderFactory:
    binding: ExecutableCapabilityBinding = MASTU_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING

    def expand_parameterised_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
    ) -> ProtocolTemplate:
        """Supply the complete archive source/view contracts to neutral fan-out."""

        from empirical_lawhood.runtime.linked_campaigns import expand_linked_study_acquisition_views

        extension = next(
            (value for value in records if isinstance(value, ResponseExperimentExtensionSet)),
            None,
        )
        substrate = next(
            (value for value in records if isinstance(value, ResponseSubstrateBinding)),
            None,
        )
        provider_config = next(
            (
                value
                for value in records
                if isinstance(value, MASTUArchiveParameterisedProviderConfig)
            ),
            None,
        )
        projection_config = next(
            (value for value in records if isinstance(value, MastArchiveReceiverProjectionConfig)),
            None,
        )
        compact_config = next(
            (value for value in records if isinstance(value, MASTUArchiveCompactProviderConfig)),
            None,
        )
        if not all(
            value is not None
            for value in (
                extension,
                substrate,
                provider_config,
                projection_config,
                compact_config,
            )
        ):
            raise ValueError("MAST-U protocol expansion lacks its exact adapter records")
        assert isinstance(extension, ResponseExperimentExtensionSet)
        assert isinstance(substrate, ResponseSubstrateBinding)
        assert isinstance(provider_config, MASTUArchiveParameterisedProviderConfig)
        assert isinstance(projection_config, MastArchiveReceiverProjectionConfig)
        assert isinstance(compact_config, MASTUArchiveCompactProviderConfig)
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("MAST-U protocol substrate selects another executable binding")

        def config_ref(
            record: CanonicalRecord,
            manifest: CapabilityManifest,
        ) -> CapabilityConfigRef:
            config_id = cast(str, getattr(record, "config_id"))
            return CapabilityConfigRef(
                config_id=config_id,
                config_schema=record.SCHEMA,
                config_schema_sha256=manifest.config_schema_sha256,
                content_sha256=record.fingerprint(),
                artifact_id=f"config-artifact.{config_id}",
            )

        def output(output_id: str, schema: str) -> OutputTemplate:
            return OutputTemplate(
                output_id=output_id,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".canonical.json",
            )

        source = ProtocolStepTemplate(
            step_id="mastu-source-prototype",
            stage=ScientificStage.PREPARE,
            capability_key=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY.capability_key,
            capability_version=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY.capability_version,
            config=config_ref(provider_config, MASTU_PARAMETERISED_ACQUISITION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output(
                            "acquisition-result",
                            MASTUArchiveAcquisitionResult.SCHEMA,
                        ),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY.resource_ceiling,
            resource_lock_ids=("lock.mastu-archive-session",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("mastu-archive-object-custody",),
        )
        projection = ProtocolStepTemplate(
            step_id="mastu-projection-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=MASTU_PARAMETERISED_PROJECTION_CAPABILITY.capability_key,
            capability_version=MASTU_PARAMETERISED_PROJECTION_CAPABILITY.capability_version,
            config=config_ref(projection_config, MASTU_PARAMETERISED_PROJECTION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output("projection", MastArchiveReceiverScientificProjection.SCHEMA),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=MASTU_PARAMETERISED_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=MASTU_PARAMETERISED_PROJECTION_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("mastu-pf-view-projection",),
        )
        aggregate = ProtocolStepTemplate(
            step_id="mastu-compact-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=MASTU_PARAMETERISED_COMPACT_CAPABILITY.capability_key,
            capability_version=MASTU_PARAMETERISED_COMPACT_CAPABILITY.capability_version,
            config=config_ref(compact_config, MASTU_PARAMETERISED_COMPACT_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output(
                            "compact-manifest",
                            MASTUArchiveCompactManifestResult.SCHEMA,
                        ),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=MASTU_PARAMETERISED_COMPACT_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=MASTU_PARAMETERISED_COMPACT_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("mastu-compact-unit-manifest",),
        )
        return expand_linked_study_acquisition_views(
            template=template,
            identification_config=extension.identification_config,
            source_task_prefix=provider_config.source_task_prefix,
            projection_task_prefix=projection_config.projection_task_prefix,
            source_step_template=source,
            projection_step_template=projection,
            acquisition_aggregate_step_template=aggregate,
            acquisition_aggregate_task_id=compact_config.task_id,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        by_type = {type(value): value for value in records}
        expected = {
            ResponseExperimentExtensionSet,
            MASTUArchiveParameterisedProviderConfig,
            MASTUArchiveQueryManifest,
            MASTUArchiveQuerySourceConfig,
            ResponseSubstrateBinding,
        }
        if len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("MAST-U acquisition factory requires its exact record roster")
        if (
            len(platform_ports) != 1
            or platform_ports[0].port_key != "mastu-archive-acquisition-session"
            or not isinstance(platform_ports[0].port, MASTUArchiveAcquisitionSessionPort)
        ):
            raise ValueError("MAST-U acquisition factory requires one bounded session port")
        binding = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
        if binding.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("MAST-U substrate binding differs from installed acquisition")
        return MASTUArchiveAcquisitionProvider(
            registry=registry,
            manifest=MASTU_PARAMETERISED_ACQUISITION_CAPABILITY,
            extension_set=cast(
                ResponseExperimentExtensionSet,
                by_type[ResponseExperimentExtensionSet],
            ),
            substrate_binding=binding,
            query_manifest=cast(
                MASTUArchiveQueryManifest,
                by_type[MASTUArchiveQueryManifest],
            ),
            query_config=cast(
                MASTUArchiveQuerySourceConfig,
                by_type[MASTUArchiveQuerySourceConfig],
            ),
            provider_config=cast(
                MASTUArchiveParameterisedProviderConfig,
                by_type[MASTUArchiveParameterisedProviderConfig],
            ),
            session=platform_ports[0].port,
        )


@dataclass(frozen=True, slots=True)
class MastArchiveReceiverProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = MASTU_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if len(records) != 1 or not isinstance(records[0], MastArchiveReceiverProjectionConfig):
            raise ValueError("MAST-U projection factory requires one exact config")
        if (
            len(platform_ports) != 1
            or platform_ports[0].port_key != "mastu-pf-projector"
            or not isinstance(platform_ports[0].port, MastArchiveReceiverProjectorPort)
        ):
            raise ValueError("MAST-U projection factory requires one custody projector")
        return MastArchiveReceiverProjectionProvider(
            registry=registry,
            manifest=MASTU_PARAMETERISED_PROJECTION_CAPABILITY,
            config=records[0],
            projector=platform_ports[0].port,
        )


@dataclass(frozen=True, slots=True)
class MASTUArchiveCompactProviderFactory:
    binding: ExecutableCapabilityBinding = MASTU_PARAMETERISED_COMPACT_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            len(records) != 1
            or not isinstance(records[0], MASTUArchiveCompactProviderConfig)
            or platform_ports
        ):
            raise ValueError("MAST-U compact factory requires one exact config and no ports")
        return MASTUArchiveCompactProvider(
            registry=registry,
            manifest=MASTU_PARAMETERISED_COMPACT_CAPABILITY,
            config=records[0],
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.mastu-query-source",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                MASTU_QUERY_PROFILE_COMPILER_EXECUTABLE_BINDING,
                MASTU_QUERY_PROJECTION_EXECUTABLE_BINDING,
                MASTU_QUERY_SOURCE_EXECUTABLE_BINDING,
                MASTU_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING,
                MASTU_PARAMETERISED_COMPACT_EXECUTABLE_BINDING,
                MASTU_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES: tuple[ExecutableFactory, ...] = tuple(
    sorted(
        cast(
            tuple[ExecutableFactory, ...],
            (
                MASTUQueryProfileCompilerFactory(),
                MASTUQueryProjectionSelectionFactory(),
                MASTUQuerySourceFactory(),
                MASTUArchiveAcquisitionProviderFactory(),
                MASTUArchiveCompactProviderFactory(),
                MastArchiveReceiverProjectionProviderFactory(),
            ),
        ),
        key=lambda value: value.binding.binding_id,
    )
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = _MASTU_QUERY_CONFIG_TYPES


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "MASTU_QUERY_PROFILE_COMPILER_EXECUTABLE_BINDING",
    "MASTU_QUERY_PROJECTION_EXECUTABLE_BINDING",
    "MASTU_QUERY_SOURCE_EXECUTABLE_BINDING",
    'MASTUQueryProfileCompilerFactory',
    'MASTUQueryProjectionSelectionFactory',
    'MASTUQuerySourceFactory',
    'MASTUArchiveAcquisitionProviderFactory',
    'MASTUArchiveCompactProviderFactory',
    'MastArchiveReceiverProjectionProviderFactory',
    "MASTU_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING",
    "MASTU_PARAMETERISED_COMPACT_EXECUTABLE_BINDING",
    "MASTU_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING",
    'RegisteredMASTUQuerySourceSelection',
    'RegisteredMASTUQueryProjectionSelection',
]
