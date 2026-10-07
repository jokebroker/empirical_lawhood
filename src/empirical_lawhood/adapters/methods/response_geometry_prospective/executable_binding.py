"""Installed factories for the finite assay scientific projections and evaluator."""

from dataclasses import dataclass

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import executable

from .extension_bundle import ASSAY_PROJECTION_CAPABILITY, ASSAY_EVALUATION_CAPABILITY, ASSAY_PROJECTION_COMPONENTS, ASSAY_EVALUATION_COMPONENTS
from .provider import ResponseGeometryAssayMethodProvider
from .qualification import ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig


ASSAY_PROJECTION_BINDING = executable(
    ASSAY_PROJECTION_CAPABILITY, ASSAY_PROJECTION_COMPONENTS, (ResponseGeometryAssayProjectionConfig,)
)
ASSAY_EVALUATION_BINDING = executable(
    ASSAY_EVALUATION_CAPABILITY, ASSAY_EVALUATION_COMPONENTS, (ResponseGeometryAssayEvaluationConfig,)
)


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = ASSAY_PROJECTION_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResponseGeometryAssayMethodProvider:
        if platform_ports or len(records) != 1 or not isinstance(records[0], ResponseGeometryAssayProjectionConfig):
            raise ValueError("assay projection factory requires its exact issued config")
        return ResponseGeometryAssayMethodProvider(registry, ASSAY_PROJECTION_CAPABILITY, records[0])


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayEvaluationProviderFactory:
    binding: ExecutableCapabilityBinding = ASSAY_EVALUATION_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResponseGeometryAssayMethodProvider:
        if platform_ports or len(records) != 1 or not isinstance(records[0], ResponseGeometryAssayEvaluationConfig):
            raise ValueError("assay evaluator factory requires its exact issued config")
        return ResponseGeometryAssayMethodProvider(registry, ASSAY_EVALUATION_CAPABILITY, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.response-geometry-assay.methods",
    "1.0.0",
    (ASSAY_EVALUATION_BINDING, ASSAY_PROJECTION_BINDING),
)
EXECUTABLE_BINDING_FACTORIES = (ResponseGeometryAssayEvaluationProviderFactory(), ResponseGeometryAssayProjectionProviderFactory())
EXECUTABLE_RECORD_TYPES = tuple(
    sorted((ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig), key=lambda value: value.SCHEMA)
)
