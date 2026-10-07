"""Code-owned reconstruction of the native full-batch provider; construction has no effects."""

from dataclasses import dataclass, replace

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .extension_bundle import CAPABILITY, COMPONENTS
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchConfig
from .provider import ReactorBatchProvider


BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorBatchConfig,)),
    required_platform_port_keys=("tbs-reactor-batch-source-input",),
)


@dataclass(frozen=True, slots=True)
class ReactorBatchFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ReactorBatchProvider:
        if len(records) != 1 or not isinstance(records[0], ReactorBatchConfig):
            raise ValueError("reactor factory requires its exact full-batch configuration")
        if tuple(
            p.port_key for p in platform_ports
        ) != self.binding.required_platform_port_keys or not isinstance(
            platform_ports[0].port, ExternalInputPayload
        ):
            raise ValueError("reactor factory requires its authenticated source input port")
        return ReactorBatchProvider(registry, records[0], platform_ports[0].port)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.terminal-bench-science-batch.reactor-batch", "1.0.0", (BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ReactorBatchFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorBatchConfig,)
