"""Executable factories and issued codecs for structural transport methods."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from empirical_lawhood.adapters.methods.prospective_structural_recurrence import StructuralRecurrenceFrozenLawTransportForecasts, ProspectiveStructuralRecurrenceMethodSpec, ProspectiveStructuralRecurrencePlan
from empirical_lawhood.adapters.methods.interval_property_comparison import IntervalPropertyComparisonMethodSpec
from empirical_lawhood.adapters.methods.transformed_property_comparison import TransformedPropertyComparisonMethodSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
from empirical_lawhood.runtime.conditional_children import FrozenParentInputBinding
from empirical_lawhood.runtime.law_transport_handoff import AuthenticatedLawTransportHandoff, authenticate_law_transport_handoff
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider

from .campaign import STRUCTURAL_TARGET_SOURCE_PORT_KEY, StructuralDevelopmentBridgeConfig, StructuralDevelopmentInputs, StructuralRecurrenceTransportConfig, StructuralReporterConfig, StructuralTargetBridgeConfig
from .metatheory_conformance import METATHEORY_CONFORMANCE_EXECUTABLE_CONFIG_TYPES, MetatheoryConformanceAcquisitionConfig, MetatheoryConformanceDevelopmentConfig, MetatheoryConformanceEvaluatorConfig, MetatheoryConformancePredictionConfig, MetatheoryConformanceQualificationConfig, MetatheoryConformanceReporterConfig
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES
from .extension_bundle import (
    EXECUTABLE_METATHEORY_ACQUISITION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_ACQUISITION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST,
    EXECUTABLE_METATHEORY_ACQUISITION_METHOD,
    EXECUTABLE_METATHEORY_ACQUISITION_RUNTIME_PROVIDER,
    EXECUTABLE_METATHEORY_DEVELOPMENT_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_DEVELOPMENT_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST,
    EXECUTABLE_METATHEORY_DEVELOPMENT_METHOD,
    EXECUTABLE_METATHEORY_DEVELOPMENT_RUNTIME_PROVIDER,
    EXECUTABLE_METATHEORY_EVALUATOR_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_EVALUATOR_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST,
    EXECUTABLE_METATHEORY_EVALUATOR_METHOD,
    EXECUTABLE_METATHEORY_EVALUATOR_RUNTIME_PROVIDER,
    EXECUTABLE_METATHEORY_PREDICTION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_PREDICTION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_PREDICTION_MANIFEST,
    EXECUTABLE_METATHEORY_PREDICTION_METHOD,
    EXECUTABLE_METATHEORY_PREDICTION_RUNTIME_PROVIDER,
    EXECUTABLE_METATHEORY_QUALIFICATION_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_QUALIFICATION_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST,
    EXECUTABLE_METATHEORY_QUALIFICATION_METHOD,
    EXECUTABLE_METATHEORY_QUALIFICATION_RUNTIME_PROVIDER,
    EXECUTABLE_METATHEORY_REPORTER_ARTIFACT_VALIDATOR,
    EXECUTABLE_METATHEORY_REPORTER_CONFIG_DECODER,
    EXECUTABLE_METATHEORY_REPORTER_MANIFEST,
    EXECUTABLE_METATHEORY_REPORTER_METHOD,
    EXECUTABLE_METATHEORY_REPORTER_RUNTIME_PROVIDER,
    INTERVAL_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
    TRANSFORMED_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
    INTERVAL_PROPERTY_COMPARISON_CONFIG_DECODER,
    TRANSFORMED_PROPERTY_COMPARISON_CONFIG_DECODER,
    INTERVAL_PROPERTY_COMPARISON_MANIFEST,
    TRANSFORMED_PROPERTY_COMPARISON_MANIFEST,
    INTERVAL_PROPERTY_COMPARISON_METHOD,
    TRANSFORMED_PROPERTY_COMPARISON_METHOD,
    INTERVAL_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
    TRANSFORMED_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
    PROSPECTIVE_STRUCTURAL_RECURRENCE_ARTIFACT_VALIDATOR,
    PROSPECTIVE_STRUCTURAL_RECURRENCE_CONFIG_DECODER,
    PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST,
    PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD,
    PROSPECTIVE_STRUCTURAL_RECURRENCE_RUNTIME_PROVIDER,
    STRUCTURAL_DEVELOPMENT_BRIDGE_ARTIFACT_VALIDATOR,
    STRUCTURAL_DEVELOPMENT_BRIDGE_CONFIG_DECODER,
    STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
    STRUCTURAL_DEVELOPMENT_BRIDGE_METHOD,
    STRUCTURAL_DEVELOPMENT_BRIDGE_RUNTIME_PROVIDER,
    STRUCTURAL_RECURRENCE_TRANSPORT_ARTIFACT_VALIDATOR,
    STRUCTURAL_RECURRENCE_TRANSPORT_CONFIG_DECODER,
    STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
    STRUCTURAL_RECURRENCE_TRANSPORT_METHOD,
    STRUCTURAL_RECURRENCE_TRANSPORT_RUNTIME_PROVIDER,
    STRUCTURAL_REPORTER_ARTIFACT_VALIDATOR,
    STRUCTURAL_REPORTER_CONFIG_DECODER,
    STRUCTURAL_REPORTER_MANIFEST,
    STRUCTURAL_REPORTER_METHOD,
    STRUCTURAL_REPORTER_RUNTIME_PROVIDER,
    STRUCTURAL_TARGET_BRIDGE_ARTIFACT_VALIDATOR,
    STRUCTURAL_TARGET_BRIDGE_CONFIG_DECODER,
    STRUCTURAL_TARGET_BRIDGE_MANIFEST,
    STRUCTURAL_TARGET_BRIDGE_METHOD,
    STRUCTURAL_TARGET_BRIDGE_RUNTIME_PROVIDER,
    SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS,
)
from .provider import StructuralTargetSourcePort, interval_property_comparison_provider, transformed_property_comparison_provider, structural_recurrence_prospective_provider, structural_development_bridge_provider, structural_structural_recurrence_transport_provider, structural_reporter_provider, structural_target_bridge_provider
from .metatheory_provider import metatheory_conformance_provider
from .source_free_property_transport_provider import source_free_property_transport_provider


_MAXIMUM_CONFIG_BYTES = 4 * 1024 * 1024


def _decoder(
    component: object,
    record_type: type[CanonicalRecord],
    registration_id: str,
) -> StudyExtensionDecoderRegistration:
    decoder_key = getattr(component, "component_key")
    decoder_version = getattr(component, "component_version")
    implementation_sha256 = getattr(component, "implementation_sha256")
    return StudyExtensionDecoderRegistration(
        registration_id=registration_id,
        decoder_key=decoder_key,
        decoder_version=decoder_version,
        payload_schema=record_type.SCHEMA,
        payload_version=record_type.VERSION,
        config_sha256=sha256(
            canonical_json_bytes(
                {
                    "decoder_key": decoder_key,
                    "decoder_version": decoder_version,
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest(),
        implementation_sha256=implementation_sha256,
        maximum_payload_bytes=_MAXIMUM_CONFIG_BYTES,
    )


INTERVAL_PROPERTY_COMPARISON_DECODER_REGISTRATION = _decoder(
    INTERVAL_PROPERTY_COMPARISON_CONFIG_DECODER,
    IntervalPropertyComparisonMethodSpec,
    'decoder-registration.property-comparison-method-spec',
)
TRANSFORMED_PROPERTY_COMPARISON_DECODER_REGISTRATION = _decoder(
    TRANSFORMED_PROPERTY_COMPARISON_CONFIG_DECODER,
    TransformedPropertyComparisonMethodSpec,
    "decoder-registration.transformed-property-comparison-method-spec",
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_DECODER_REGISTRATION = _decoder(
    PROSPECTIVE_STRUCTURAL_RECURRENCE_CONFIG_DECODER,
    ProspectiveStructuralRecurrenceMethodSpec,
    'decoder-registration.structural-recurrence-prospective-method-spec',
)
STRUCTURAL_DEVELOPMENT_BRIDGE_DECODER_REGISTRATION = _decoder(
    STRUCTURAL_DEVELOPMENT_BRIDGE_CONFIG_DECODER,
    StructuralDevelopmentBridgeConfig,
    "decoder-registration.structural-development-bridge-config",
)
STRUCTURAL_RECURRENCE_TRANSPORT_DECODER_REGISTRATION = _decoder(
    STRUCTURAL_RECURRENCE_TRANSPORT_CONFIG_DECODER,
    StructuralRecurrenceTransportConfig,
    'decoder-registration.structural-recurrence-transport-config',
)
STRUCTURAL_REPORTER_DECODER_REGISTRATION = _decoder(
    STRUCTURAL_REPORTER_CONFIG_DECODER,
    StructuralReporterConfig,
    "decoder-registration.structural-reporter-config",
)
STRUCTURAL_TARGET_BRIDGE_DECODER_REGISTRATION = _decoder(
    STRUCTURAL_TARGET_BRIDGE_CONFIG_DECODER,
    StructuralTargetBridgeConfig,
    "decoder-registration.structural-target-bridge-config",
)
EXECUTABLE_METATHEORY_ACQUISITION_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_ACQUISITION_CONFIG_DECODER,
    MetatheoryConformanceAcquisitionConfig,
    "decoder-registration.executable-metatheory-acquisition-config",
)
EXECUTABLE_METATHEORY_DEVELOPMENT_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_DEVELOPMENT_CONFIG_DECODER,
    MetatheoryConformanceDevelopmentConfig,
    "decoder-registration.executable-metatheory-development-config",
)
EXECUTABLE_METATHEORY_EVALUATOR_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_EVALUATOR_CONFIG_DECODER,
    MetatheoryConformanceEvaluatorConfig,
    "decoder-registration.executable-metatheory-evaluator-config",
)
EXECUTABLE_METATHEORY_PREDICTION_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_PREDICTION_CONFIG_DECODER,
    MetatheoryConformancePredictionConfig,
    "decoder-registration.executable-metatheory-prediction-config",
)
EXECUTABLE_METATHEORY_QUALIFICATION_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_QUALIFICATION_CONFIG_DECODER,
    MetatheoryConformanceQualificationConfig,
    "decoder-registration.executable-metatheory-qualification-config",
)
EXECUTABLE_METATHEORY_REPORTER_DECODER_REGISTRATION = _decoder(
    EXECUTABLE_METATHEORY_REPORTER_CONFIG_DECODER,
    MetatheoryConformanceReporterConfig,
    "decoder-registration.executable-metatheory-reporter-config",
)


def _binding(
    *,
    binding_id: str,
    manifest: CapabilityManifest,
    method: ExtensionComponentRegistration,
    decoder_component: ExtensionComponentRegistration,
    runtime_provider: ExtensionComponentRegistration,
    validator: ExtensionComponentRegistration,
    record_type: type[CanonicalRecord],
    decoder: StudyExtensionDecoderRegistration,
    authenticated_record_types: tuple[type[CanonicalRecord], ...] = (),
    required_platform_port_keys: tuple[str, ...] = (),
    capability_backed: bool = True,
) -> ExecutableCapabilityBinding:
    return ExecutableCapabilityBinding(
        binding_id=binding_id,
        capability_key=manifest.capability_key,
        capability_version=manifest.capability_version,
        capability_implementation_sha256=manifest.implementation_sha256,
        role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
        provider_key=runtime_provider.component_key,
        provider_version=runtime_provider.component_version,
        provider_implementation_sha256=runtime_provider.implementation_sha256,
        capability_backed=capability_backed,
        discovery_components=tuple(
            sorted(
                (method, decoder_component, runtime_provider, validator),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=tuple(
            sorted(
                (
                    ExecutableRecordTypeBinding(record_type.SCHEMA, record_type.VERSION),
                    *(
                        ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                        for value in authenticated_record_types
                    ),
                ),
                key=lambda value: value.type_id,
            )
        ),
        required_issued_payload_schemas=(record_type.SCHEMA,),
        codec_registration_identities=(
            ObjectIdentity.from_record(
                decoder_component.registration_id,
                decoder_component,
            ),
        ),
        issued_decoder_registrations=(decoder,),
        input_schema_ids=manifest.input_schema_ids,
        output_schema_ids=manifest.output_schema_ids,
        artifact_validator_identities=(
            ObjectIdentity.from_record(validator.registration_id, validator),
        ),
        required_platform_port_keys=required_platform_port_keys,
        may_require_active_mount=False,
        may_require_source_qualification=False,
        may_require_network=False,
        may_require_authority=True,
        required_authenticated_record_schemas=tuple(
            sorted(value.SCHEMA for value in authenticated_record_types)
        ),
    )


SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_REGISTRATIONS = tuple(
    (
        role,
        _decoder(
            decoder_component,
            record_type,
            f"decoder-registration.source-free-property-transport-{role.value.lower().replace('_', '-')}",
        ),
    )
    for (
        role,
        _,
        record_type,
        _,
        decoder_component,
        _,
        _,
    ) in SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS
)
_SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_BY_ROLE = dict(SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_REGISTRATIONS)
SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_BINDINGS = tuple(
    sorted(
        (
            _binding(
                binding_id=(f"binding.source-free-property-transport-{role.value.lower().replace('_', '-')}"),
                manifest=manifest,
                method=method,
                decoder_component=decoder_component,
                runtime_provider=runtime_provider,
                validator=validator,
                record_type=record_type,
                decoder=_SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_BY_ROLE[role],
            )
            for (
                role,
                manifest,
                record_type,
                method,
                decoder_component,
                validator,
                runtime_provider,
            ) in SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS
        ),
        key=lambda value: value.binding_id,
    )
)


INTERVAL_PROPERTY_COMPARISON_EXECUTABLE_BINDING = _binding(
    binding_id='binding.property-comparison-method',
    manifest=INTERVAL_PROPERTY_COMPARISON_MANIFEST,
    method=INTERVAL_PROPERTY_COMPARISON_METHOD,
    decoder_component=INTERVAL_PROPERTY_COMPARISON_CONFIG_DECODER,
    runtime_provider=INTERVAL_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
    validator=INTERVAL_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
    record_type=IntervalPropertyComparisonMethodSpec,
    decoder=INTERVAL_PROPERTY_COMPARISON_DECODER_REGISTRATION,
)
TRANSFORMED_PROPERTY_COMPARISON_EXECUTABLE_BINDING = _binding(
    binding_id="binding.transformed-property-comparison-method",
    manifest=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST,
    method=TRANSFORMED_PROPERTY_COMPARISON_METHOD,
    decoder_component=TRANSFORMED_PROPERTY_COMPARISON_CONFIG_DECODER,
    runtime_provider=TRANSFORMED_PROPERTY_COMPARISON_RUNTIME_PROVIDER,
    validator=TRANSFORMED_PROPERTY_COMPARISON_ARTIFACT_VALIDATOR,
    record_type=TransformedPropertyComparisonMethodSpec,
    decoder=TRANSFORMED_PROPERTY_COMPARISON_DECODER_REGISTRATION,
)
PROSPECTIVE_STRUCTURAL_RECURRENCE_EXECUTABLE_BINDING = _binding(
    binding_id='binding.structural-recurrence-prospective-method',
    manifest=PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST,
    method=PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD,
    decoder_component=PROSPECTIVE_STRUCTURAL_RECURRENCE_CONFIG_DECODER,
    runtime_provider=PROSPECTIVE_STRUCTURAL_RECURRENCE_RUNTIME_PROVIDER,
    validator=PROSPECTIVE_STRUCTURAL_RECURRENCE_ARTIFACT_VALIDATOR,
    record_type=ProspectiveStructuralRecurrenceMethodSpec,
    decoder=PROSPECTIVE_STRUCTURAL_RECURRENCE_DECODER_REGISTRATION,
)
STRUCTURAL_DEVELOPMENT_BRIDGE_EXECUTABLE_BINDING = _binding(
    binding_id="binding.structural-development-bridge",
    manifest=STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
    method=STRUCTURAL_DEVELOPMENT_BRIDGE_METHOD,
    decoder_component=STRUCTURAL_DEVELOPMENT_BRIDGE_CONFIG_DECODER,
    runtime_provider=STRUCTURAL_DEVELOPMENT_BRIDGE_RUNTIME_PROVIDER,
    validator=STRUCTURAL_DEVELOPMENT_BRIDGE_ARTIFACT_VALIDATOR,
    record_type=StructuralDevelopmentBridgeConfig,
    decoder=STRUCTURAL_DEVELOPMENT_BRIDGE_DECODER_REGISTRATION,
    authenticated_record_types=(StructuralDevelopmentInputs,),
)
STRUCTURAL_RECURRENCE_TRANSPORT_EXECUTABLE_BINDING = _binding(
    binding_id='binding.structural-recurrence-transport',
    manifest=STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
    method=STRUCTURAL_RECURRENCE_TRANSPORT_METHOD,
    decoder_component=STRUCTURAL_RECURRENCE_TRANSPORT_CONFIG_DECODER,
    runtime_provider=STRUCTURAL_RECURRENCE_TRANSPORT_RUNTIME_PROVIDER,
    validator=STRUCTURAL_RECURRENCE_TRANSPORT_ARTIFACT_VALIDATOR,
    record_type=StructuralRecurrenceTransportConfig,
    decoder=STRUCTURAL_RECURRENCE_TRANSPORT_DECODER_REGISTRATION,
)
STRUCTURAL_REPORTER_EXECUTABLE_BINDING = _binding(
    binding_id="binding.structural-reporter",
    manifest=STRUCTURAL_REPORTER_MANIFEST,
    method=STRUCTURAL_REPORTER_METHOD,
    decoder_component=STRUCTURAL_REPORTER_CONFIG_DECODER,
    runtime_provider=STRUCTURAL_REPORTER_RUNTIME_PROVIDER,
    validator=STRUCTURAL_REPORTER_ARTIFACT_VALIDATOR,
    record_type=StructuralReporterConfig,
    decoder=STRUCTURAL_REPORTER_DECODER_REGISTRATION,
)
STRUCTURAL_TARGET_BRIDGE_EXECUTABLE_BINDING = _binding(
    binding_id="binding.structural-target-bridge",
    manifest=STRUCTURAL_TARGET_BRIDGE_MANIFEST,
    method=STRUCTURAL_TARGET_BRIDGE_METHOD,
    decoder_component=STRUCTURAL_TARGET_BRIDGE_CONFIG_DECODER,
    runtime_provider=STRUCTURAL_TARGET_BRIDGE_RUNTIME_PROVIDER,
    validator=STRUCTURAL_TARGET_BRIDGE_ARTIFACT_VALIDATOR,
    record_type=StructuralTargetBridgeConfig,
    decoder=STRUCTURAL_TARGET_BRIDGE_DECODER_REGISTRATION,
    required_platform_port_keys=(STRUCTURAL_TARGET_SOURCE_PORT_KEY,),
)
EXECUTABLE_METATHEORY_ACQUISITION_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-acquisition",
    manifest=EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST,
    method=EXECUTABLE_METATHEORY_ACQUISITION_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_ACQUISITION_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_ACQUISITION_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_ACQUISITION_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformanceAcquisitionConfig,
    decoder=EXECUTABLE_METATHEORY_ACQUISITION_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_DEVELOPMENT_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-development",
    manifest=EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST,
    method=EXECUTABLE_METATHEORY_DEVELOPMENT_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_DEVELOPMENT_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_DEVELOPMENT_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_DEVELOPMENT_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformanceDevelopmentConfig,
    decoder=EXECUTABLE_METATHEORY_DEVELOPMENT_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_EVALUATOR_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-evaluator",
    manifest=EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST,
    method=EXECUTABLE_METATHEORY_EVALUATOR_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_EVALUATOR_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_EVALUATOR_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_EVALUATOR_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformanceEvaluatorConfig,
    decoder=EXECUTABLE_METATHEORY_EVALUATOR_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_PREDICTION_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-prediction",
    manifest=EXECUTABLE_METATHEORY_PREDICTION_MANIFEST,
    method=EXECUTABLE_METATHEORY_PREDICTION_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_PREDICTION_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_PREDICTION_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_PREDICTION_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformancePredictionConfig,
    decoder=EXECUTABLE_METATHEORY_PREDICTION_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_QUALIFICATION_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-qualification",
    manifest=EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST,
    method=EXECUTABLE_METATHEORY_QUALIFICATION_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_QUALIFICATION_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_QUALIFICATION_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_QUALIFICATION_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformanceQualificationConfig,
    decoder=EXECUTABLE_METATHEORY_QUALIFICATION_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_REPORTER_EXECUTABLE_BINDING = _binding(
    binding_id="binding.executable-metatheory-reporter",
    manifest=EXECUTABLE_METATHEORY_REPORTER_MANIFEST,
    method=EXECUTABLE_METATHEORY_REPORTER_METHOD,
    decoder_component=EXECUTABLE_METATHEORY_REPORTER_CONFIG_DECODER,
    runtime_provider=EXECUTABLE_METATHEORY_REPORTER_RUNTIME_PROVIDER,
    validator=EXECUTABLE_METATHEORY_REPORTER_ARTIFACT_VALIDATOR,
    record_type=MetatheoryConformanceReporterConfig,
    decoder=EXECUTABLE_METATHEORY_REPORTER_DECODER_REGISTRATION,
)
EXECUTABLE_METATHEORY_CONFORMANCE_EXECUTABLE_BINDINGS = tuple(
    sorted(
        (
            EXECUTABLE_METATHEORY_ACQUISITION_EXECUTABLE_BINDING,
            EXECUTABLE_METATHEORY_DEVELOPMENT_EXECUTABLE_BINDING,
            EXECUTABLE_METATHEORY_EVALUATOR_EXECUTABLE_BINDING,
            EXECUTABLE_METATHEORY_PREDICTION_EXECUTABLE_BINDING,
            EXECUTABLE_METATHEORY_QUALIFICATION_EXECUTABLE_BINDING,
            EXECUTABLE_METATHEORY_REPORTER_EXECUTABLE_BINDING,
        ),
        key=lambda value: value.binding_id,
    )
)


@dataclass(frozen=True, slots=True)
class IntervalPropertyComparisonProviderFactory:
    binding: ExecutableCapabilityBinding = INTERVAL_PROPERTY_COMPARISON_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], IntervalPropertyComparisonMethodSpec)
        ):
            raise ValueError("property-comparison factory requires one exact issued method spec")
        return interval_property_comparison_provider(
            registry=registry,
            manifest=INTERVAL_PROPERTY_COMPARISON_MANIFEST,
            spec=records[0],
        )


@dataclass(frozen=True, slots=True)
class TransformedPropertyComparisonProviderFactory:
    binding: ExecutableCapabilityBinding = TRANSFORMED_PROPERTY_COMPARISON_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], TransformedPropertyComparisonMethodSpec)
        ):
            raise ValueError('transformed property comparison factory requires one exact issued spec')
        return transformed_property_comparison_provider(
            registry=registry,
            manifest=TRANSFORMED_PROPERTY_COMPARISON_MANIFEST,
            spec=records[0],
        )


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceProviderFactory:
    binding: ExecutableCapabilityBinding = PROSPECTIVE_STRUCTURAL_RECURRENCE_EXECUTABLE_BINDING

    def authenticate_parent_inputs(
        self,
        *,
        target_candidate: ObjectIdentity,
        scientific_graph_sha256: str,
        bindings: tuple[FrozenParentInputBinding, ...],
    ) -> AuthenticatedLawTransportHandoff:
        """Authenticate the complete donor-law handoff before provider construction."""

        return authenticate_law_transport_handoff(
            target_candidate=target_candidate,
            scientific_graph_sha256=scientific_graph_sha256,
            bindings=bindings,
            forecast_method_spec_type=ProspectiveStructuralRecurrenceMethodSpec,
            forecast_method_config_type=ProspectiveStructuralRecurrencePlan,
            frozen_forecasts_type=StructuralRecurrenceFrozenLawTransportForecasts,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], ProspectiveStructuralRecurrenceMethodSpec)
        ):
            raise ValueError('structural recurrence prospective factory requires one exact issued method spec')
        return structural_recurrence_prospective_provider(
            registry=registry,
            manifest=PROSPECTIVE_STRUCTURAL_RECURRENCE_MANIFEST,
            spec=records[0],
        )


@dataclass(frozen=True, slots=True)
class StructuralDevelopmentBridgeProviderFactory:
    binding: ExecutableCapabilityBinding = STRUCTURAL_DEVELOPMENT_BRIDGE_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        configs = tuple(
            value for value in records if isinstance(value, StructuralDevelopmentBridgeConfig)
        )
        inputs = tuple(
            value for value in records if isinstance(value, StructuralDevelopmentInputs)
        )
        if platform_ports or len(configs) != 1 or len(inputs) != 1 or len(records) != 2:
            raise ValueError(
                "structural development factory requires issued config and authenticated inputs"
            )
        return structural_development_bridge_provider(
            registry=registry,
            manifest=STRUCTURAL_DEVELOPMENT_BRIDGE_MANIFEST,
            config=configs[0],
            development_inputs=inputs[0],
        )


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceTransportProviderFactory:
    binding: ExecutableCapabilityBinding = STRUCTURAL_RECURRENCE_TRANSPORT_EXECUTABLE_BINDING

    def authenticate_parent_inputs(
        self,
        *,
        target_candidate: ObjectIdentity,
        scientific_graph_sha256: str,
        bindings: tuple[FrozenParentInputBinding, ...],
    ) -> AuthenticatedLawTransportHandoff:
        return authenticate_law_transport_handoff(
            target_candidate=target_candidate,
            scientific_graph_sha256=scientific_graph_sha256,
            bindings=bindings,
            forecast_method_spec_type=ProspectiveStructuralRecurrenceMethodSpec,
            forecast_method_config_type=ProspectiveStructuralRecurrencePlan,
            frozen_forecasts_type=StructuralRecurrenceFrozenLawTransportForecasts,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], StructuralRecurrenceTransportConfig)
        ):
            raise ValueError('structural structural recurrence factory requires one exact issued config')
        return structural_structural_recurrence_transport_provider(
            registry=registry,
            manifest=STRUCTURAL_RECURRENCE_TRANSPORT_MANIFEST,
            config=records[0],
        )


@dataclass(frozen=True, slots=True)
class StructuralReporterProviderFactory:
    binding: ExecutableCapabilityBinding = STRUCTURAL_REPORTER_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], StructuralReporterConfig)
        ):
            raise ValueError("structural reporter factory requires one exact issued config")
        return structural_reporter_provider(
            registry=registry,
            manifest=STRUCTURAL_REPORTER_MANIFEST,
            config=records[0],
        )


@dataclass(frozen=True, slots=True)
class StructuralTargetBridgeProviderFactory:
    binding: ExecutableCapabilityBinding = STRUCTURAL_TARGET_BRIDGE_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        source = platform_ports[0].port if len(platform_ports) == 1 else None
        if (
            len(records) != 1
            or not isinstance(records[0], StructuralTargetBridgeConfig)
            or len(platform_ports) != 1
            or platform_ports[0].port_key != STRUCTURAL_TARGET_SOURCE_PORT_KEY
            or not isinstance(source, StructuralTargetSourcePort)
        ):
            raise ValueError(
                "structural target factory requires issued config and exact target-source port"
            )
        return structural_target_bridge_provider(
            registry=registry,
            manifest=STRUCTURAL_TARGET_BRIDGE_MANIFEST,
            config=records[0],
            source=source,
        )


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportProviderFactory:
    binding: ExecutableCapabilityBinding
    manifest: CapabilityManifest
    role: MetatheoryCampaignStageRole
    record_type: type[CanonicalRecord]

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if platform_ports or len(records) != 1 or not isinstance(records[0], self.record_type):
            raise ValueError("metatheory R1 factory requires one exact issued stage config")
        return source_free_property_transport_provider(
            registry=registry,
            manifest=self.manifest,
            role=self.role,
            config=records[0],
        )


SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_FACTORIES = tuple(
    sorted(
        (
            SourceFreePropertyTransportProviderFactory(
                binding=next(
                    value
                    for value in SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_BINDINGS
                    if value.capability_key == manifest.capability_key
                ),
                manifest=manifest,
                role=role,
                record_type=record_type,
            )
            for role, manifest, record_type, *_ in SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_COMPONENTS
        ),
        key=lambda value: value.binding.binding_id,
    )
)


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryConformanceProviderFactory:
    binding: ExecutableCapabilityBinding
    manifest: CapabilityManifest
    record_type: type[CanonicalRecord]

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], self.record_type)
            or not isinstance(records[0], METATHEORY_CONFORMANCE_EXECUTABLE_CONFIG_TYPES)
        ):
            raise ValueError("metatheory conformance factory requires one exact issued config")
        return metatheory_conformance_provider(
            registry=registry,
            manifest=self.manifest,
            config=records[0],
        )


EXECUTABLE_METATHEORY_CONFORMANCE_PROVIDER_FACTORIES = tuple(
    sorted(
        (
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_ACQUISITION_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_ACQUISITION_MANIFEST,
                MetatheoryConformanceAcquisitionConfig,
            ),
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_DEVELOPMENT_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_DEVELOPMENT_MANIFEST,
                MetatheoryConformanceDevelopmentConfig,
            ),
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_EVALUATOR_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_EVALUATOR_MANIFEST,
                MetatheoryConformanceEvaluatorConfig,
            ),
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_PREDICTION_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_PREDICTION_MANIFEST,
                MetatheoryConformancePredictionConfig,
            ),
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_QUALIFICATION_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_QUALIFICATION_MANIFEST,
                MetatheoryConformanceQualificationConfig,
            ),
            ExecutableMetatheoryConformanceProviderFactory(
                EXECUTABLE_METATHEORY_REPORTER_EXECUTABLE_BINDING,
                EXECUTABLE_METATHEORY_REPORTER_MANIFEST,
                MetatheoryConformanceReporterConfig,
            ),
        ),
        key=lambda value: value.binding.binding_id,
    )
)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.structural-transport",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                *EXECUTABLE_METATHEORY_CONFORMANCE_EXECUTABLE_BINDINGS,
                *SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_BINDINGS,
                INTERVAL_PROPERTY_COMPARISON_EXECUTABLE_BINDING,
                TRANSFORMED_PROPERTY_COMPARISON_EXECUTABLE_BINDING,
                PROSPECTIVE_STRUCTURAL_RECURRENCE_EXECUTABLE_BINDING,
                STRUCTURAL_DEVELOPMENT_BRIDGE_EXECUTABLE_BINDING,
                STRUCTURAL_RECURRENCE_TRANSPORT_EXECUTABLE_BINDING,
                STRUCTURAL_REPORTER_EXECUTABLE_BINDING,
                STRUCTURAL_TARGET_BRIDGE_EXECUTABLE_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    *EXECUTABLE_METATHEORY_CONFORMANCE_PROVIDER_FACTORIES,
    *SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_FACTORIES,
    IntervalPropertyComparisonProviderFactory(),
    TransformedPropertyComparisonProviderFactory(),
    ProspectiveStructuralRecurrenceProviderFactory(),
    StructuralDevelopmentBridgeProviderFactory(),
    StructuralRecurrenceTransportProviderFactory(),
    StructuralReporterProviderFactory(),
    StructuralTargetBridgeProviderFactory(),
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    *METATHEORY_CONFORMANCE_EXECUTABLE_CONFIG_TYPES,
    *SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES,
    ProspectiveStructuralRecurrenceMethodSpec,
    IntervalPropertyComparisonMethodSpec,
    TransformedPropertyComparisonMethodSpec,
    StructuralDevelopmentBridgeConfig,
    StructuralRecurrenceTransportConfig,
    StructuralReporterConfig,
    StructuralTargetBridgeConfig,
)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "EXECUTABLE_METATHEORY_ACQUISITION_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_ACQUISITION_EXECUTABLE_BINDING",
    "EXECUTABLE_METATHEORY_CONFORMANCE_EXECUTABLE_BINDINGS",
    "EXECUTABLE_METATHEORY_CONFORMANCE_PROVIDER_FACTORIES",
    "EXECUTABLE_METATHEORY_DEVELOPMENT_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_DEVELOPMENT_EXECUTABLE_BINDING",
    "EXECUTABLE_METATHEORY_EVALUATOR_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_EVALUATOR_EXECUTABLE_BINDING",
    "EXECUTABLE_METATHEORY_PREDICTION_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_PREDICTION_EXECUTABLE_BINDING",
    "EXECUTABLE_METATHEORY_QUALIFICATION_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_QUALIFICATION_EXECUTABLE_BINDING",
    "EXECUTABLE_METATHEORY_REPORTER_DECODER_REGISTRATION",
    "EXECUTABLE_METATHEORY_REPORTER_EXECUTABLE_BINDING",
    "SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_REGISTRATIONS",
    "SOURCE_FREE_PROPERTY_TRANSPORT_EXECUTABLE_BINDINGS",
    "SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_FACTORIES",
    'SourceFreePropertyTransportProviderFactory',
    'ExecutableMetatheoryConformanceProviderFactory',
    'INTERVAL_PROPERTY_COMPARISON_DECODER_REGISTRATION',
    'TRANSFORMED_PROPERTY_COMPARISON_DECODER_REGISTRATION',
    'INTERVAL_PROPERTY_COMPARISON_EXECUTABLE_BINDING',
    'TRANSFORMED_PROPERTY_COMPARISON_EXECUTABLE_BINDING',
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_DECODER_REGISTRATION',
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_EXECUTABLE_BINDING',
    'ProspectiveStructuralRecurrenceProviderFactory',
    'IntervalPropertyComparisonProviderFactory',
    'TransformedPropertyComparisonProviderFactory',
    "STRUCTURAL_DEVELOPMENT_BRIDGE_DECODER_REGISTRATION",
    "STRUCTURAL_DEVELOPMENT_BRIDGE_EXECUTABLE_BINDING",
    'STRUCTURAL_RECURRENCE_TRANSPORT_DECODER_REGISTRATION',
    'STRUCTURAL_RECURRENCE_TRANSPORT_EXECUTABLE_BINDING',
    "STRUCTURAL_REPORTER_DECODER_REGISTRATION",
    "STRUCTURAL_REPORTER_EXECUTABLE_BINDING",
    "STRUCTURAL_TARGET_BRIDGE_DECODER_REGISTRATION",
    "STRUCTURAL_TARGET_BRIDGE_EXECUTABLE_BINDING",
    'StructuralDevelopmentBridgeProviderFactory',
    'StructuralRecurrenceTransportProviderFactory',
    'StructuralReporterProviderFactory',
    'StructuralTargetBridgeProviderFactory',
]
