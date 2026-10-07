"Exact information response prediction method factories and issued configuration codecs."

from dataclasses import dataclass, replace

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.adapters.composition.discovery import executable
from .extension_bundle import PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, EVALUATION_CAPABILITY, EVALUATION_COMPONENTS
from .models import PROGRAMME
from .records import InformationResponseProjectionConfig, InformationResponseEvaluationConfig
from .provider import InformationResponseMethodProvider

PROJECTION_BINDING, EVALUATION_BINDING = tuple(
    replace(
        binding,
        issued_decoder_registrations=tuple(
            replace(d, maximum_payload_bytes=16 * 1024**2)
            for d in binding.issued_decoder_registrations
        ),
    )
    for binding in (
        executable(PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, (InformationResponseProjectionConfig,)),
        executable(EVALUATION_CAPABILITY, EVALUATION_COMPONENTS, (InformationResponseEvaluationConfig,)),
    )
)


@dataclass(frozen=True, slots=True)
class InformationResponseMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> InformationResponseMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError("information response prediction method factory requires one exact issued configuration")
        config = records[0]
        if self.binding == PROJECTION_BINDING and type(config) is InformationResponseProjectionConfig:
            return InformationResponseMethodProvider(registry, PROJECTION_CAPABILITY, config)
        if self.binding == EVALUATION_BINDING and type(config) is InformationResponseEvaluationConfig:
            return InformationResponseMethodProvider(registry, EVALUATION_CAPABILITY, config)
        raise ValueError("information response prediction method factory changes its installed binding/configuration")


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{PROGRAMME}.methods",
    "1.0.0",
    tuple(sorted((PROJECTION_BINDING, EVALUATION_BINDING), key=lambda b: b.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = tuple(
    InformationResponseMethodFactory(b) for b in EXECUTABLE_BINDING_CONTRIBUTION.bindings
)
_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (InformationResponseProjectionConfig, InformationResponseEvaluationConfig)
EXECUTABLE_RECORD_TYPES = tuple(sorted(_RECORD_TYPES, key=lambda t: t.SCHEMA))
