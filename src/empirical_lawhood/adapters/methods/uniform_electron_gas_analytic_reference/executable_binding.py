"""Selected uniform electron gas analytic evaluator with a closed issued config."""

from dataclasses import dataclass

from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_contracts import UniformElectronGasAnalyticEvaluationConfig
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import UniformElectronGasAnalyticEvaluationProvider

BINDING = executable(CAPABILITY, COMPONENTS, (UniformElectronGasAnalyticEvaluationConfig,))


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticEvaluationFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> UniformElectronGasAnalyticEvaluationProvider:
        if len(records) != 1 or not isinstance(
            records[0], UniformElectronGasAnalyticEvaluationConfig
        ):
            raise ValueError(
                "uniform electron gas analytic check needs its exact issued evaluation config"
            )
        if platform_ports:
            raise ValueError("uniform electron gas analytic check cannot accept a platform port")
        return UniformElectronGasAnalyticEvaluationProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.uniform-electron-gas-analytic-reference.finite-q-check",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (UniformElectronGasAnalyticEvaluationFactory(),)
EXECUTABLE_RECORD_TYPES = (UniformElectronGasAnalyticEvaluationConfig,)
