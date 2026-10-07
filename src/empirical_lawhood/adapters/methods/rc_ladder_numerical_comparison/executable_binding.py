"""Closed issued evaluator config for the consuming RC numerical route."""

from dataclasses import dataclass

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort

from .contracts import ResistorCapacitorLadderEvaluationConfig
from .extension_bundle import CAPABILITY, COMPONENTS
from .provider import ResistorCapacitorLadderNumericalProvider

BINDING = executable(CAPABILITY, COMPONENTS, (ResistorCapacitorLadderEvaluationConfig,))


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderNumericalFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResistorCapacitorLadderNumericalProvider:
        if len(records) != 1 or not isinstance(records[0], ResistorCapacitorLadderEvaluationConfig):
            raise ValueError("RC evaluator factory requires its exact issued config")
        if platform_ports:
            raise ValueError(
                "RC numerical evaluator cannot accept external platform ports"
            )
        return ResistorCapacitorLadderNumericalProvider(registry, records[0])


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.rc-ladder-response.numerical-check",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (ResistorCapacitorLadderNumericalFactory(),)
EXECUTABLE_RECORD_TYPES = (ResistorCapacitorLadderEvaluationConfig,)
