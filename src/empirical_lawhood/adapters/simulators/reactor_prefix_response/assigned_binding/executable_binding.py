# SPDX-License-Identifier: MPL-2.0

"Fixed assignment factory; the assignment selects data, never executable code."

from dataclasses import dataclass, replace

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution
from empirical_lawhood.runtime.providers import ExternalInputPayload

from ..assigned import ReactorAssignedPrefixConfig, ReactorAssignedPrefixPanel
from ..provider import ReactorPrefixProvider, ReactorPrefixRunner
from .extension_bundle import CAPABILITY, COMPONENTS

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorAssignedPrefixConfig,)),
    required_platform_port_keys=("tbs-reactor-source-input",),
)


class ReactorAssignedPrefixRunner(ReactorPrefixRunner):
    manifest = CAPABILITY
    panel_type = ReactorAssignedPrefixPanel


class ReactorAssignedPrefixProvider(ReactorPrefixProvider):
    manifest = CAPABILITY
    panel_type = ReactorAssignedPrefixPanel
    runner_type = ReactorAssignedPrefixRunner


@dataclass(frozen=True, slots=True)
class ReactorAssignedPrefixFactory:
    binding = BINDING

    def build_provider(self, *, registry, records, platform_ports):
        if (
            len(records) != 1
            or type(records[0]) is not ReactorAssignedPrefixConfig
            or tuple(p.port_key for p in platform_ports)
            != BINDING.required_platform_port_keys
            or not isinstance(platform_ports[0].port, ExternalInputPayload)
        ):
            raise ValueError("REACTOR_ASSIGNED_FACTORY_EXACT_INPUTS_REQUIRED")
        return ReactorAssignedPrefixProvider(
            registry, records[0], platform_ports[0].port
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.assigned-reactor-prefix-response",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (ReactorAssignedPrefixFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorAssignedPrefixConfig,)
