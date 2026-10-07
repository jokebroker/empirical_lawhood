"""Static single-capability providers over the existing task/custody contracts."""

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskRunner
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.task_records import contracts, external_config, verify_registry


class RecordCampaignProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        capability: CapabilityManifest,
        runner: TaskRunner,
        config: CanonicalRecord,
        config_id: str,
        inputs: tuple[ExternalInputPayload, ...],
        outputs: tuple[type[CanonicalRecord], ...],
        *,
        adjudication: bool,
    ) -> None:
        self.registry_sha256 = verify_registry(registry, capability)
        self.capability_count = 1
        self.capability, self.runner, self.config, self.config_id = (
            capability,
            runner,
            config,
            config_id,
        )
        self.inputs, self.outputs, self.adjudication = inputs, outputs, adjudication

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("provider registry or source differs from its static binding")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("provider plan or source differs from its static binding")
        return tuple(
            sorted(
                (
                    *external_config(plan, self.capability, self.config, config_id=self.config_id),
                    *self.inputs,
                ),
                key=lambda p: p.logical_artifact_id,
            )
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return contracts(self.capability, self.outputs)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        self.runners(registry)
        if not self.adjudication:
            return None
        if execution_plan is None:
            raise ValueError("adjudication requires its exact compiled locator")
        outputs = [
            o
            for t in execution_plan.tasks
            for o in t.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError("provider changed the single terminal adjudication census")
        return ScientificAdjudicationOutputContract(
            self.capability.capability_key,
            self.capability.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )
