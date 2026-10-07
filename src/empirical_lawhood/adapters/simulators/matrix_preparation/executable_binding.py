"""Installed preparation source factory with a bounded lazy retained-prefix port."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.native_source import NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeSourceProfile
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.adapters.control.backbone_linked_campaign.executable_binding import (
    RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION,
)
from empirical_lawhood.adapters.control.backbone_linked_campaign.extension_bundle import (
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS as NATIVE_LAW_DECODERS,
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS as NATIVE_LAW_DECODER_REGISTRATIONS,
    RESPONSE_SUBSTRATE_BINDING_DECODER,
)
from empirical_lawhood.adapters.composition.discovery import executable
from .contracts import DEVELOPMENT, PreparationSourceConfig
from .continuation import PreparationDevelopmentContinuation, preparation_continuation
from .extension_bundle import SOURCE_CAPABILITY, SOURCE_COMPONENTS
from .provider import PREFIX_SOURCE_PORT, PreparationPrefixSourceFactory, PreparationSourceProvider


_SOURCE_RECORDS = (
    PreparationSourceConfig,
    NativeSourceProfile,
    NativeLawQualificationConfig,
    NativeLawQualificationExperiment,
    ResponseSubstrateBinding,
)
_SHARED_DECODERS = (*NATIVE_LAW_DECODERS, RESPONSE_SUBSTRATE_BINDING_DECODER)
_BASE = executable(
    SOURCE_CAPABILITY,
    SOURCE_COMPONENTS,
    (PreparationSourceConfig, PreparationDevelopmentContinuation),
)
SOURCE_BINDING = replace(
    _BASE,
    discovery_components=tuple(
        sorted((*SOURCE_COMPONENTS, *_SHARED_DECODERS), key=lambda d: d.registration_id)
    ),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(t.SCHEMA, t.VERSION)
                for t in (*_SOURCE_RECORDS, PreparationDevelopmentContinuation)
            ),
            key=lambda t: t.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(sorted(t.SCHEMA for t in _SOURCE_RECORDS)),
    codec_registration_identities=tuple(
        sorted(
            (
                *_BASE.codec_registration_identities,
                *(ObjectIdentity.from_record(d.registration_id, d) for d in _SHARED_DECODERS),
            ),
            key=lambda r: r.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *_BASE.issued_decoder_registrations,
                *NATIVE_LAW_DECODER_REGISTRATIONS,
                RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION,
            ),
            key=lambda r: r.registration_id,
        )
    ),
    required_platform_port_keys=(PREFIX_SOURCE_PORT,),
)


@dataclass(frozen=True, slots=True)
class PreparationSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        from .protocol import expand_preparation_protocol

        return expand_preparation_protocol(
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
    ) -> PreparationSourceProvider:
        from .protocol import validate_preparation_source_records

        continuation = preparation_continuation(records)
        expected = (
            *_SOURCE_RECORDS,
            *((PreparationDevelopmentContinuation,) if continuation else ()),
        )
        if (
            len(records) != len(expected)
            or {type(r) for r in records} != set(expected)
            or tuple(p.port_key for p in platform_ports) != (PREFIX_SOURCE_PORT,)
        ):
            raise ValueError(
                "preparation source factory requires its exact typed records and retained-input port"
            )
        source = validate_preparation_source_records(
            records, ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        )
        port = platform_ports[0].port
        if (
            not hasattr(port, "prefixes")
            or not hasattr(port, "prefix_bundles")
            or not callable(getattr(port, "create_source", None))
        ):
            raise TypeError("preparation source requires bounded lazy prefix discovery")
        if continuation is not None:
            continuation.validate_source(source)
        return PreparationSourceProvider(
            registry,
            SOURCE_CAPABILITY,
            source,
            cast(PreparationPrefixSourceFactory, port),
            continuation=continuation,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{DEVELOPMENT}.source", "1.0.0", (SOURCE_BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (PreparationSourceFactory(),)
EXECUTABLE_RECORD_TYPES = (PreparationSourceConfig, PreparationDevelopmentContinuation)
