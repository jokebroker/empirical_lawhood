"""Installed D method factories consume only their exact issued configuration."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import executable
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_projection import ResponseGeometryDevelopmentProjectionConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import ResponseGeometryDevelopmentMethodConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_provider import ResponseGeometryDevelopmentDevelopmentProvider
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import ResponseGeometryDevelopmentQualificationConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentCloseoutConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_assessment_provider import ResponseGeometryDevelopmentAssessmentProvider, DEVELOPMENT_CANDIDATE_PAYLOAD_PORT
from .extension_bundle import DEVELOPMENT_CAPABILITIES, DEVELOPMENT_COMPONENTS, DEVELOPMENT_CONFIG_TYPES, DEVELOPMENT_ROLES

DEVELOPMENT_BINDINGS = tuple(
    replace(
        executable(capability, components, (kind,)),
        required_platform_port_keys=(DEVELOPMENT_CANDIDATE_PAYLOAD_PORT,)
        if kind is ResponseGeometryDevelopmentQualificationConfig
        else (),
    )
    for capability, components, kind in zip(
        DEVELOPMENT_CAPABILITIES, DEVELOPMENT_COMPONENTS, DEVELOPMENT_CONFIG_TYPES, strict=True
    )
)


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentDevelopmentFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        index = DEVELOPMENT_BINDINGS.index(self.binding)
        if (
            len(records) != 1
            or type(records[0]) is not DEVELOPMENT_CONFIG_TYPES[index]
            or tuple(p.port_key for p in platform_ports) != self.binding.required_platform_port_keys
        ):
            raise ValueError("D method factory requires its one exact issued config")
        config = records[0]
        if isinstance(config, ResponseGeometryDevelopmentQualificationConfig):
            plane = platform_ports[0].port
            if not all(
                callable(getattr(plane, method, None))
                for method in ("publish_candidate_payload", "read_candidate_payload")
            ):
                raise TypeError("D assessment requires its bounded candidate publication/read port")
            return ResponseGeometryDevelopmentAssessmentProvider(
                registry, DEVELOPMENT_CAPABILITIES[index], config, cast(CandidatePayloadPlane, plane)
            )
        if isinstance(config, ResponseGeometryDevelopmentCloseoutConfig):
            return ResponseGeometryDevelopmentAssessmentProvider(registry, DEVELOPMENT_CAPABILITIES[index], config)
        assert isinstance(config, (ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentMethodConfig))
        return ResponseGeometryDevelopmentDevelopmentProvider(registry, DEVELOPMENT_CAPABILITIES[index], config, DEVELOPMENT_ROLES[index])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.response-geometry-development.methods",
    "1.0.0",
    tuple(sorted(DEVELOPMENT_BINDINGS, key=lambda v: v.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = tuple(
    ResponseGeometryDevelopmentDevelopmentFactory(b) for b in EXECUTABLE_BINDING_CONTRIBUTION.bindings
)
EXECUTABLE_RECORD_TYPES = tuple(sorted(set(DEVELOPMENT_CONFIG_TYPES), key=lambda v: v.SCHEMA))
