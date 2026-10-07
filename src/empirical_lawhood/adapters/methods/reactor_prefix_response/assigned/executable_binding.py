# SPDX-License-Identifier: MPL-2.0

"Schema binding preserves the finite owner, cutoffs and terminal rules."

from dataclasses import dataclass, replace

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixPanel
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution

from ..provider import FiniteChainProvider, FiniteChainRunner
from .extension_bundle import CAPABILITY, COMPONENTS, INPUT_TYPES, OUTPUT_TYPES
from .records import ReactorAssignedNativeCustody, ReactorAssignedScienceDesign, ReactorAssignedScienceResult

BINDING = replace(
    executable(CAPABILITY, COMPONENTS, (ReactorAssignedScienceDesign,)),
    required_platform_port_keys=(
        "candidate-payload-publisher",
        "candidate-payload-reader",
        "dependency-custody-reader",
    ),
)


class AssignedFiniteChainRunner(FiniteChainRunner):
    manifest = CAPABILITY
    input_types = INPUT_TYPES
    output_types = OUTPUT_TYPES
    panel_type = ReactorAssignedPrefixPanel
    result_type = ReactorAssignedScienceResult

    def native_custody(self, receipts):
        return ReactorAssignedNativeCustody(receipts, self.config.native)


class AssignedFiniteChainProvider(FiniteChainProvider):
    manifest = CAPABILITY
    output_types = OUTPUT_TYPES
    runner_type = AssignedFiniteChainRunner


@dataclass(frozen=True, slots=True)
class AssignedFiniteChainFactory:
    binding = BINDING

    def build_provider(self, *, registry, records, platform_ports):
        ports = {p.port_key: p.port for p in platform_ports}
        if (
            len(records) != 1
            or type(records[0]) is not ReactorAssignedScienceDesign
            or len(platform_ports) != 3
            or tuple(sorted(ports)) != BINDING.required_platform_port_keys
        ):
            raise ValueError("REACTOR_ASSIGNED_FINITE_CHAIN_INPUTS_REQUIRED")
        for key, method in (
            ("candidate-payload-publisher", "publish_candidate_payload"),
            ("candidate-payload-reader", "read_candidate_payload"),
            ("dependency-custody-reader", "read_dependency"),
        ):
            if not callable(getattr(ports[key], method, None)):
                raise TypeError("REACTOR_ASSIGNED_FINITE_CHAIN_PORT_PROTOCOL")
        return AssignedFiniteChainProvider(
            registry,
            records[0],
            *(ports[key] for key in BINDING.required_platform_port_keys),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.assigned-reactor-finite-chain",
    "1.0.0",
    (BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (AssignedFiniteChainFactory(),)
EXECUTABLE_RECORD_TYPES = (ReactorAssignedScienceDesign,)
