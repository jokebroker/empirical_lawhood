'Closed issued-config decoder and provider for the ambient pressure superconductor gauge covariant response method unit.'

from dataclasses import dataclass

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .extension_bundle import CAPABILITY, COMPONENTS
from .synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig
from .synthetic_material_method_provider import SyntheticMaterialResponseMethodProvider

BINDING = executable(CAPABILITY, COMPONENTS, (SyntheticMaterialResponseMethodConfig,))


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SyntheticMaterialResponseMethodProvider:
        if len(records) != 1 or not isinstance(records[0], SyntheticMaterialResponseMethodConfig):
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method native factory requires its exact issued config'
            )
        if platform_ports:
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method source cannot accept a physical platform port'
            )
        return SyntheticMaterialResponseMethodProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.synthetic-material-response-method.native-panel",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (SyntheticMaterialResponseMethodFactory(),)
EXECUTABLE_RECORD_TYPES = (SyntheticMaterialResponseMethodConfig,)
