"""Installed D native factory; source reconstruction remains nonexecuting."""

from dataclasses import dataclass, replace

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.planning.native_source import NativeSourceProfile, NativeLawQualificationConfig, NativeLawQualificationExperiment
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.adapters.control.backbone_linked_campaign.extension_bundle import (
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS,
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS,
    RESPONSE_SUBSTRATE_BINDING_DECODER,
)
from empirical_lawhood.adapters.control.backbone_linked_campaign.executable_binding import (
    RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import executable
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_provider import ResponseGeometryDevelopmentSourceProvider
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig
from .extension_bundle import DEVELOPMENT_SOURCE_CAPABILITY, DEVELOPMENT_SOURCE_COMPONENTS


DEVELOPMENT_SOURCE_BINDING = executable(DEVELOPMENT_SOURCE_CAPABILITY, DEVELOPMENT_SOURCE_COMPONENTS, (ResponseGeometryDevelopmentNativeConfig,))
_SOURCE_RECORDS = (
    ResponseGeometryDevelopmentNativeConfig,
    NativeSourceProfile,
    NativeLawQualificationConfig,
    NativeLawQualificationExperiment,
    ResponseSubstrateBinding,
)
_SHARED_DECODERS = (*NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS, RESPONSE_SUBSTRATE_BINDING_DECODER)
DEVELOPMENT_SOURCE_BINDING = replace(
    DEVELOPMENT_SOURCE_BINDING,
    discovery_components=tuple(
        sorted((*DEVELOPMENT_SOURCE_COMPONENTS, *_SHARED_DECODERS), key=lambda d: d.registration_id)
    ),
    accepted_config_types=tuple(
        sorted(
            (ExecutableRecordTypeBinding(t.SCHEMA, t.VERSION) for t in _SOURCE_RECORDS),
            key=lambda t: t.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(sorted(t.SCHEMA for t in _SOURCE_RECORDS)),
    codec_registration_identities=tuple(
        sorted(
            (
                *DEVELOPMENT_SOURCE_BINDING.codec_registration_identities,
                *(ObjectIdentity.from_record(d.registration_id, d) for d in _SHARED_DECODERS),
            ),
            key=lambda d: d.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *DEVELOPMENT_SOURCE_BINDING.issued_decoder_registrations,
                *NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS,
                RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION,
            ),
            key=lambda d: d.registration_id,
        )
    ),
)


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentSourceProviderFactory:
    binding: ExecutableCapabilityBinding = DEVELOPMENT_SOURCE_BINDING

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_protocol import expand_response_geometry_development_protocol

        return expand_response_geometry_development_protocol(
            records=records,
            template=template,
            binding=ObjectIdentity.from_record(self.binding.binding_id, self.binding),
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResponseGeometryDevelopmentSourceProvider:
        from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_protocol import validate_response_geometry_development_source_records

        if (
            platform_ports
            or len(records) != 5
            or {type(r) for r in records} != set(_SOURCE_RECORDS)
        ):
            raise ValueError("D native factory requires its five exact issued source records")
        config = validate_response_geometry_development_source_records(
            records, ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        )
        return ResponseGeometryDevelopmentSourceProvider(registry, DEVELOPMENT_SOURCE_CAPABILITY, config)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.response-geometry-development.source",
    "1.0.0",
    (DEVELOPMENT_SOURCE_BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (ResponseGeometryDevelopmentSourceProviderFactory(),)
EXECUTABLE_RECORD_TYPES = (ResponseGeometryDevelopmentNativeConfig,)
