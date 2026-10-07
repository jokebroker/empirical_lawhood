"""Closed installed RC factories reconstruct exact issued inputs and custody."""

from dataclasses import dataclass, replace

from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider

from .contracts import SimulatorMorphismChallengePhase
from .extension_bundle import CAPABILITIES, COMPONENTS_BY_CAPABILITY, INSTALLED_IMPLEMENTATION_SHA256
from .issued_inputs import CAPABILITY_INPUT_TYPES, RCChallengeRuntimeInputPort, capability_port_key
from .capability_configs import CAPABILITY_CONFIG_TYPES


BINDINGS = tuple(replace(executable(capability, COMPONENTS_BY_CAPABILITY[capability.capability_key],
                                  (CAPABILITY_CONFIG_TYPES[capability.capability_key], CAPABILITY_INPUT_TYPES[capability.capability_key])),
                         required_platform_port_keys=(capability_port_key(capability.capability_key),)) for capability in CAPABILITIES)


class RCChallengeCapabilityRuntimeProvider(CampaignRuntimeProvider):
    """One factory owns one runner; shared external operands have lexical ownership."""

    def __init__(self, provider, capability_key: str) -> None:
        self.provider = provider
        self.capability_key = capability_key
        self.registry_sha256 = provider.registry_sha256
        self.capability_count = 1

    def runners(self, registry, source_records=()):
        return tuple(value for value in self.provider.runners(registry, source_records)
                     if value.manifest.capability_key == self.capability_key)

    def external_inputs(self, plan, source_records=()):
        owners = {}
        for task in plan.tasks:
            for spec in task.external_inputs:
                owners[spec.logical_artifact_id] = min(owners.get(spec.logical_artifact_id, task.capability.capability_key), task.capability.capability_key)
        return tuple(value for value in self.provider.external_inputs(plan, source_records)
                     if owners[value.logical_artifact_id] == self.capability_key)

    def output_semantic_contracts(self, registry, execution_plan=None):
        # The installed binding covers every declared output schema. A phase
        # uses a subset of these schemas; its actual output roster is the plan.
        if execution_plan is not None and execution_plan.registry_sha256 != self.registry_sha256:
            raise ValueError("RC semantic execution plan registry differs")
        return tuple(value for value in self.provider.output_semantic_contracts(registry)
                     if value.capability_key == self.capability_key)

    def scientific_adjudication_contract(self, registry, execution_plan=None):
        value = self.provider.scientific_adjudication_contract(registry, execution_plan)
        return value if value.capability_key == self.capability_key else None


@dataclass(frozen=True, slots=True)
class RCChallengeRuntimeFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(self, *, registry: CapabilityRegistry,
                       records: tuple[CanonicalRecord, ...],
                       platform_ports: tuple[ExecutablePlatformPort, ...]):
        from .runtime_provider import SimulatorMorphismChallengeCampaignRuntimeProvider

        by_type = {type(record): record for record in records}
        config_type = CAPABILITY_CONFIG_TYPES[self.binding.capability_key]
        inputs_type = CAPABILITY_INPUT_TYPES[self.binding.capability_key]
        if len(records) != 2 or set(by_type) != {config_type, inputs_type}:
            raise ValueError("RC factory requires exact issued config and reconstruction inputs")
        if len(platform_ports) != 1 or platform_ports[0].port_key != capability_port_key(self.binding.capability_key) or not isinstance(platform_ports[0].port, RCChallengeRuntimeInputPort):
            raise ValueError("RC factory requires the current source/custody platform port")
        issued = by_type[inputs_type].inputs
        if by_type[config_type].config != issued.config:
            raise ValueError("RC capability config differs from its exact issued phase input")
        config = issued.config
        port = platform_ports[0].port
        port.authenticate(issued)
        packet = issued.numeric_input
        scientific = () if config.phase in (SimulatorMorphismChallengePhase.CANARY, SimulatorMorphismChallengePhase.NOMINATION) else packet.scientific_inputs(config) if packet is not None else None
        provider = SimulatorMorphismChallengeCampaignRuntimeProvider(
            registry=registry, config=config, external_records=issued.external_records(),
            source_files=port.source_files, scientific_inputs=scientific,
            injected_nomination_seeds=packet.source.seeds if packet is not None and config.phase is SimulatorMorphismChallengePhase.NOMINATION else None,
            manifest_implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256,
        )
        return RCChallengeCapabilityRuntimeProvider(provider, self.binding.capability_key)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.simulator-morphism-challenges", "1.0.0", BINDINGS)
EXECUTABLE_BINDING_FACTORIES = tuple(RCChallengeRuntimeFactory(binding) for binding in BINDINGS)
EXECUTABLE_RECORD_TYPES = tuple(kind for value in CAPABILITIES
    for kind in (CAPABILITY_CONFIG_TYPES[value.capability_key], CAPABILITY_INPUT_TYPES[value.capability_key]))
