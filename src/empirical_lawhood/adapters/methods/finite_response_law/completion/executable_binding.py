"""Installed retained-completion binding on the existing input resolver."""

from dataclasses import dataclass, replace
from typing import cast
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.source_resolution import ContentAddressedInputResolver
from empirical_lawhood.adapters.composition.discovery import executable
from .contracts import FiniteResponseLawAssignedRetainedCompletionConfig, FiniteResponseLawRetainedCompletionConfig, PREFIX
from .extension_bundle import ASSIGNED_CAPABILITY, ASSIGNED_COMPONENTS, CAPABILITY, COMPONENTS
from .provider import FiniteResponseLawRetainedCompletionProvider
from ..method_provider import INPUT_RESOLVER_PORT

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (FiniteResponseLawRetainedCompletionConfig,)),
    required_platform_port_keys=(INPUT_RESOLVER_PORT,),
)
ASSIGNED_BINDING = replace(
    executable(
        ASSIGNED_CAPABILITY,
        ASSIGNED_COMPONENTS,
        (FiniteResponseLawAssignedRetainedCompletionConfig,),
    ),
    required_platform_port_keys=(INPUT_RESOLVER_PORT,),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRetainedCompletionFactory:
    binding: ExecutableCapabilityBinding = BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawRetainedCompletionProvider:
        manifest = CAPABILITY if self.binding == BINDING else ASSIGNED_CAPABILITY
        config_type = (
            FiniteResponseLawRetainedCompletionConfig
            if self.binding == BINDING
            else FiniteResponseLawAssignedRetainedCompletionConfig
        )
        if (
            self.binding not in (BINDING, ASSIGNED_BINDING)
            or len(records) != 1
            or type(records[0]) is not config_type
            or tuple(p.port_key for p in platform_ports) != (INPUT_RESOLVER_PORT,)
        ):
            raise ValueError("Retained completion requires its exact config and resolver")
        return FiniteResponseLawRetainedCompletionProvider(
            registry,
            manifest,
            records[0],
            cast(ContentAddressedInputResolver, platform_ports[0].port),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{PREFIX}",
    "1.0.0",
    tuple(sorted((BINDING, ASSIGNED_BINDING), key=lambda value: value.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = (
    FiniteResponseLawRetainedCompletionFactory(),
    FiniteResponseLawRetainedCompletionFactory(ASSIGNED_BINDING),
)
EXECUTABLE_RECORD_TYPES = (
    FiniteResponseLawRetainedCompletionConfig,
    FiniteResponseLawAssignedRetainedCompletionConfig,
)
