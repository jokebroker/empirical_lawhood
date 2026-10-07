"""Code-owned native RC binding and closed issued-study decoder."""

from dataclasses import dataclass

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .contracts import ResistorCapacitorLadderStudyConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .runtime_provider import ResistorCapacitorLadderNativeProvider

BINDING = executable(CAPABILITY, COMPONENTS, (ResistorCapacitorLadderStudyConfig,))


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderNativeFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResistorCapacitorLadderNativeProvider:
        if len(records) != 1 or not isinstance(records[0], ResistorCapacitorLadderStudyConfig):
            raise ValueError("RC native factory requires its exact issued study")
        if platform_ports:
            raise ValueError("local RC model cannot accept an external platform port")
        return ResistorCapacitorLadderNativeProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.rc-ladder-response.native-view",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (ResistorCapacitorLadderNativeFactory(),)
EXECUTABLE_RECORD_TYPES = (ResistorCapacitorLadderStudyConfig,)
