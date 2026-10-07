"""Reconstruct only the declared retained-input provider from issued records."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.adapters.methods.response_geometry_prospective.development_assessment_provider import DEVELOPMENT_CANDIDATE_PAYLOAD_PORT, ResponseGeometryDevelopmentAssessmentProvider
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentCloseoutConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_continuation import RETAINED_ANALYSIS_RUN, ResponseGeometryDevelopmentAnalysisContinuationConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_continuation_provider import DEVELOPMENT_RETAINED_PROJECTION_SOURCE_PORT, ResponseGeometryDevelopmentAnalysisAssessmentProvider, ResponseGeometryDevelopmentAnalysisContinuationProvider, ResponseGeometryDevelopmentRetainedProjectionSourceFactory
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import ResponseGeometryDevelopmentMethodConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import ResponseGeometryDevelopmentQualificationConfig
from empirical_lawhood.adapters.methods.response_geometry_development.executable_binding import DEVELOPMENT_BINDINGS
from empirical_lawhood.adapters.methods.response_geometry_development.extension_bundle import DEVELOPMENT_ROLES
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentKind
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider

from .extension_bundle import RETAINED_ANALYSIS_CAPABILITIES, RETAINED_ANALYSIS_COMPONENTS, RETAINED_ANALYSIS_CONFIG_TYPES, RETAINED_ANALYSIS_ROLES


def _binding(index: int) -> ExecutableCapabilityBinding:
    capability, components, kind = (
        RETAINED_ANALYSIS_CAPABILITIES[index],
        RETAINED_ANALYSIS_COMPONENTS[index],
        RETAINED_ANALYSIS_CONFIG_TYPES[index],
    )
    current = executable(capability, components, (ResponseGeometryDevelopmentAnalysisContinuationConfig,))
    original = DEVELOPMENT_BINDINGS[DEVELOPMENT_ROLES.index(RETAINED_ANALYSIS_ROLES[index])]
    if index != 0:
        return replace(
            current,
            discovery_components=tuple(
                sorted(
                    (
                        *(
                            value
                            for value in current.discovery_components
                            if value.kind is not ExtensionComponentKind.CONFIG_DECODER
                        ),
                        *(
                            value
                            for value in original.discovery_components
                            if value.kind is ExtensionComponentKind.CONFIG_DECODER
                        ),
                    ),
                    key=lambda value: value.registration_id,
                )
            ),
            accepted_config_types=original.accepted_config_types,
            required_issued_payload_schemas=original.required_issued_payload_schemas,
            codec_registration_identities=original.codec_registration_identities,
            issued_decoder_registrations=original.issued_decoder_registrations,
            required_platform_port_keys=tuple(
                sorted(
                    (
                        DEVELOPMENT_CANDIDATE_PAYLOAD_PORT,
                        f"{DEVELOPMENT_RETAINED_PROJECTION_SOURCE_PORT}.assessment",
                    )
                )
            )
            if kind is ResponseGeometryDevelopmentQualificationConfig
            else (),
        )
    return replace(
        current,
        discovery_components=tuple(
            sorted(
                (
                    *current.discovery_components,
                    *(
                        value
                        for value in original.discovery_components
                        if value.kind is ExtensionComponentKind.CONFIG_DECODER
                    ),
                ),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_config_types=tuple(
            sorted(
                (*current.accepted_config_types, *original.accepted_config_types),
                key=lambda value: value.type_id,
            )
        ),
        required_issued_payload_schemas=tuple(
            sorted(
                (
                    *current.required_issued_payload_schemas,
                    *original.required_issued_payload_schemas,
                )
            )
        ),
        codec_registration_identities=tuple(
            sorted(
                (
                    *current.codec_registration_identities,
                    *original.codec_registration_identities,
                ),
                key=lambda value: value.object_id,
            )
        ),
        issued_decoder_registrations=tuple(
            sorted(
                (
                    *current.issued_decoder_registrations,
                    *original.issued_decoder_registrations,
                ),
                key=lambda value: value.registration_id,
            )
        ),
        required_platform_port_keys=tuple(
            sorted(
                (
                    f"{DEVELOPMENT_RETAINED_PROJECTION_SOURCE_PORT}.development",
                    *(
                        (DEVELOPMENT_CANDIDATE_PAYLOAD_PORT,)
                        if kind is ResponseGeometryDevelopmentQualificationConfig
                        else ()
                    ),
                )
            )
        )
        if kind is not ResponseGeometryDevelopmentCloseoutConfig
        else (),
    )


RETAINED_ANALYSIS_BINDINGS = tuple(_binding(index) for index in range(3))


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAnalysisContinuationFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        index = RETAINED_ANALYSIS_BINDINGS.index(self.binding)
        kind = RETAINED_ANALYSIS_CONFIG_TYPES[index]
        expected = {kind, ResponseGeometryDevelopmentAnalysisContinuationConfig} if index == 0 else {kind}
        if (
            len(records) != len(expected)
            or {type(record) for record in records} != expected
            or tuple(port.port_key for port in platform_ports)
            != self.binding.required_platform_port_keys
        ):
            raise ValueError(
                "development analysis factory requires exact scientific/custody records and bounded ports"
            )
        config = next(record for record in records if type(record) is kind)
        sources = plane = None
        ports = {port.port_key: port.port for port in platform_ports}
        if index in (0, 1):
            source_port = ports[
                f"{DEVELOPMENT_RETAINED_PROJECTION_SOURCE_PORT}.{RETAINED_ANALYSIS_ROLES[index]}"
            ]
            if not callable(getattr(source_port, "create_source", None)):
                raise TypeError(
                    "development analysis requires its nonexecuting retained-artifact source factory"
                )
            sources = cast(ResponseGeometryDevelopmentRetainedProjectionSourceFactory, source_port)
        if kind is ResponseGeometryDevelopmentQualificationConfig:
            candidate_port = ports[DEVELOPMENT_CANDIDATE_PAYLOAD_PORT]
            if not all(
                callable(getattr(candidate_port, name, None))
                for name in ("publish_candidate_payload", "read_candidate_payload")
            ):
                raise TypeError(
                    "development analysis requires the existing candidate artifact port"
                )
            plane = cast(CandidatePayloadPlane, candidate_port)
        assert isinstance(
            config, (ResponseGeometryDevelopmentMethodConfig, ResponseGeometryDevelopmentQualificationConfig, ResponseGeometryDevelopmentCloseoutConfig)
        )
        if isinstance(config, ResponseGeometryDevelopmentQualificationConfig):
            assert plane is not None and sources is not None
            return ResponseGeometryDevelopmentAnalysisAssessmentProvider(
                registry, RETAINED_ANALYSIS_CAPABILITIES[index], config, plane, sources
            )
        if isinstance(config, ResponseGeometryDevelopmentCloseoutConfig):
            return ResponseGeometryDevelopmentAssessmentProvider(
                registry, RETAINED_ANALYSIS_CAPABILITIES[index], config
            )
        continuation = next(
            record
            for record in records
            if isinstance(record, ResponseGeometryDevelopmentAnalysisContinuationConfig)
        )
        return ResponseGeometryDevelopmentAnalysisContinuationProvider(
            registry,
            RETAINED_ANALYSIS_CAPABILITIES[index],
            config,
            continuation,
            sources,
            plane,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{RETAINED_ANALYSIS_RUN}.methods",
    "1.0.0",
    tuple(sorted(RETAINED_ANALYSIS_BINDINGS, key=lambda value: value.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = tuple(
    ResponseGeometryDevelopmentAnalysisContinuationFactory(binding)
    for binding in EXECUTABLE_BINDING_CONTRIBUTION.bindings
)
EXECUTABLE_RECORD_TYPES = (ResponseGeometryDevelopmentAnalysisContinuationConfig,)
