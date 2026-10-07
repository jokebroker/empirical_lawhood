'Selected ambient pressure superconductor gauge covariant response method evaluator with a closed issued config.'

from dataclasses import dataclass

from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_contracts import SyntheticMaterialResponseMethodEvaluationConfig
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import SyntheticMaterialResponseMethodEvaluationProvider

BINDING = executable(CAPABILITY, COMPONENTS, (SyntheticMaterialResponseMethodEvaluationConfig,))


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodEvaluationFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SyntheticMaterialResponseMethodEvaluationProvider:
        if len(records) != 1 or not isinstance(
            records[0], SyntheticMaterialResponseMethodEvaluationConfig
        ):
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method check needs its exact issued evaluation config'
            )
        if platform_ports:
            raise ValueError('ambient pressure superconductor gauge covariant response method check cannot accept a platform port')
        return SyntheticMaterialResponseMethodEvaluationProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.synthetic-material-response-method.finite-q-check",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (SyntheticMaterialResponseMethodEvaluationFactory(),)
EXECUTABLE_RECORD_TYPES = (SyntheticMaterialResponseMethodEvaluationConfig,)
