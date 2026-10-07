"""Selected synthetic uniform electron gas source provider; science stays in the adapter."""

from dataclasses import fields
from decimal import Decimal

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskProgressEmitter,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .analytic_contracts import UniformElectronGasAnalyticPanel
from .analytic_science import generate_analytic_panel
from .extension_bundle import CAPABILITY
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig


def native_task_id(config: UniformElectronGasAnalyticReferenceConfig) -> str:
    return f"{config.config_id}.native-panel"


class UniformElectronGasAnalyticSourceRunner:
    manifest = CAPABILITY

    def __init__(self, config: UniformElectronGasAnalyticReferenceConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id != native_task_id(self.config)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or len(context.input_ports) != 2
            or len({port.logical_artifact_id for port in context.input_ports}) != 2
            or any(
                port.payload_schema != UniformElectronGasAnalyticReferenceConfig.SCHEMA
                or port.outcome_access is not OutcomeAccess.OUTCOME_BLIND
                or port.size_bytes > 128 * 1024
                for port in context.input_ports
            )
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != UniformElectronGasAnalyticPanel.SCHEMA
        ):
            raise ValueError(
                "uniform electron gas analytic native task or ports differ from frozen plan"
            )
        inputs = tuple(
            decode_canonical_bytes(
                port.read(), UniformElectronGasAnalyticReferenceConfig, maximum_bytes=128 * 1024
            )
            for port in context.input_ports
        )
        if any(value != self.config for value in inputs):
            raise ValueError("uniform electron gas analytic source bytes differ from selected provider")
        panel = generate_analytic_panel(self.config)
        return RunnerResult(
            (
                TaskOutputPayload(
                    context.output_ports[0].output_id, panel.canonical_bytes()
                ),
            ),
            (
                ReceiptCheck("uniform-electron-gas-synthetic-source-disclosed", True, ()),
                ReceiptCheck("uniform-electron-gas-one-unit-twenty-nested-conditions", True, ()),
            ),
        )

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        result = self.execute(context)
        emitter.advance(Decimal(1))
        return result


class UniformElectronGasAnalyticSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, config: UniformElectronGasAnalyticReferenceConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError("uniform electron gas analytic native manifest differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config = config

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("uniform electron gas analytic native registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("uniform electron gas analytic provider needs its selected issued config")
        return (UniformElectronGasAnalyticSourceRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("uniform electron gas analytic native plan or source differs")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
        )
        if (
            len(tasks) != 1
            or tasks[0].task_id != native_task_id(self.config)
            or len(tasks[0].external_inputs) != 2
        ):
            raise ValueError(
                "uniform electron gas analytic provider requires one source task and two config locators"
            )
        specs = tasks[0].external_inputs
        if len({spec.logical_artifact_id for spec in specs}) != 2 or any(
            spec.expected_payload_schema != UniformElectronGasAnalyticReferenceConfig.SCHEMA
            or spec.expected_content_sha256 != self.config.fingerprint()
            for spec in specs
        ):
            raise ValueError("uniform electron gas analytic source identity differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=UniformElectronGasAnalyticReferenceConfig.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=self.config.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.fingerprint(),
            )
            for spec in specs
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return (
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=UniformElectronGasAnalyticPanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=UniformElectronGasAnalyticPanel.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(field.name for field in fields(UniformElectronGasAnalyticPanel))
                ),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self._check(registry)


__all__ = ['UniformElectronGasAnalyticSourceProvider', "native_task_id"]
