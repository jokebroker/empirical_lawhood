"""Exact native-method factory/codec bindings through the common provider route."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.source_resolution import ContentAddressedInputResolver

from .extension_bundle import EVALUATION_CAPABILITY, EVALUATION_COMPONENTS, METHOD_CAPABILITY, METHOD_COMPONENTS, PROJECTION_CAPABILITY, PROJECTION_COMPONENTS
from .method_provider import CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT, FiniteResponseLawCalibrationMethodProvider
from .method_records import FiniteResponseLawAssignedCalibrationMethodConfig
from .native_provider import FiniteResponseLawNativeMethodProvider
from .native_records import FiniteResponseLawNativeEvaluationConfig, FiniteResponseLawProjectionConfig
from .science import PROGRAMME

PROJECTION_BINDING = executable(
    PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, (FiniteResponseLawProjectionConfig,)
)
EVALUATION_BINDING = executable(
    EVALUATION_CAPABILITY, EVALUATION_COMPONENTS, (FiniteResponseLawNativeEvaluationConfig,)
)
METHOD_BINDING = replace(
    executable(
        METHOD_CAPABILITY, METHOD_COMPONENTS, (FiniteResponseLawAssignedCalibrationMethodConfig,)
    ),
    required_platform_port_keys=tuple(
        sorted((CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT))
    ),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationQualificationFactory:
    binding: ExecutableCapabilityBinding = METHOD_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawCalibrationMethodProvider:
        ports = {p.port_key: p.port for p in platform_ports}
        if (
            self.binding != METHOD_BINDING
            or len(records) != 1
            or type(records[0]) is not FiniteResponseLawAssignedCalibrationMethodConfig
            or len(ports) != len(platform_ports)
            or tuple(sorted(ports)) != self.binding.required_platform_port_keys
        ):
            raise ValueError(
                "Finite response-law qualification requires its exact issued configuration and existing I/O ports"
            )
        if (
            not callable(getattr(ports[INPUT_RESOLVER_PORT], "resolve", None))
            or not callable(
                getattr(
                    ports[CANDIDATE_PAYLOAD_PORT], "publish_candidate_payload", None
                )
            )
            or not callable(
                getattr(ports[CANDIDATE_PAYLOAD_PORT], "read_candidate_payload", None)
            )
        ):
            raise TypeError(
                "Finite response-law qualification requires working custody and payload ports"
            )
        return FiniteResponseLawCalibrationMethodProvider(
            registry,
            METHOD_CAPABILITY,
            records[0],
            cast(ContentAddressedInputResolver, ports[INPUT_RESOLVER_PORT]),
            cast(CandidatePayloadPlane, ports[CANDIDATE_PAYLOAD_PORT]),
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawNativeMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError(
                "Finite response-law method factory requires one exact issued configuration"
            )
        config = records[0]
        if self.binding == PROJECTION_BINDING and type(config) is FiniteResponseLawProjectionConfig:
            return FiniteResponseLawNativeMethodProvider(registry, PROJECTION_CAPABILITY, config)
        if (
            self.binding == EVALUATION_BINDING
            and type(config) is FiniteResponseLawNativeEvaluationConfig
        ):
            return FiniteResponseLawNativeMethodProvider(registry, EVALUATION_CAPABILITY, config)
        raise ValueError("Finite response-law method factory changes its binding/configuration")


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{PROGRAMME}.methods",
    "1.0.0",
    tuple(
        sorted(
            (PROJECTION_BINDING, EVALUATION_BINDING, METHOD_BINDING),
            key=lambda b: b.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    FiniteResponseLawNativeMethodFactory(PROJECTION_BINDING),
    FiniteResponseLawNativeMethodFactory(EVALUATION_BINDING),
    FiniteResponseLawCalibrationQualificationFactory(),
)
_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawProjectionConfig,
    FiniteResponseLawNativeEvaluationConfig,
    FiniteResponseLawAssignedCalibrationMethodConfig,
)
EXECUTABLE_RECORD_TYPES = tuple(sorted(_RECORD_TYPES, key=lambda t: t.SCHEMA))
