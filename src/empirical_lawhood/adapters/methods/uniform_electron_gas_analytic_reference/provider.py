"""Receipt-bound uniform electron gas analytic check attached to the one native panel."""

from dataclasses import fields

from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_contracts import UniformElectronGasAnalyticCheck, UniformElectronGasAnalyticEvaluationConfig, UniformElectronGasAnalyticPanel
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_science import check_analytic_panel
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
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
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .extension_bundle import CAPABILITY


def evaluation_task_id(config: UniformElectronGasAnalyticEvaluationConfig) -> str:
    suffix = ".evaluation-config"
    if not config.config_id.endswith(suffix):
        raise ValueError("uniform electron gas analytic evaluation config has no experiment binding")
    return f"{config.config_id.removesuffix(suffix)}.finite-q-evaluate"


class UniformElectronGasAnalyticEvaluationRunner:
    manifest = CAPABILITY

    def __init__(self, config: UniformElectronGasAnalyticEvaluationConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id != evaluation_task_id(self.config)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.input_ports) != 2
            or sum(
                port.payload_schema == UniformElectronGasAnalyticPanel.SCHEMA
                and port.outcome_access is OutcomeAccess.EVALUATION_SEALED
                and port.size_bytes <= 1024**2
                for port in context.input_ports
            )
            != 1
            or sum(
                port.payload_schema == UniformElectronGasAnalyticEvaluationConfig.SCHEMA
                and port.outcome_access is OutcomeAccess.OUTCOME_BLIND
                and port.size_bytes <= 1024**2
                for port in context.input_ports
            )
            != 1
            or len(context.output_ports) != 2
            or {port.payload_schema for port in context.output_ports}
            != {UniformElectronGasAnalyticCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA}
            or context.scientific_adjudication_context is None
            or not context.dependency_receipt_ids
        ):
            raise ValueError("uniform electron gas analytic evaluation task or sealed ports differ")
        supplied = decode_canonical_bytes(
            next(
                port
                for port in context.input_ports
                if port.payload_schema == UniformElectronGasAnalyticEvaluationConfig.SCHEMA
            ).read(),
            UniformElectronGasAnalyticEvaluationConfig,
            maximum_bytes=1024**2,
        )
        if supplied != self.config:
            raise ValueError(
                "uniform electron gas analytic evaluator config differs from selected provider"
            )
        panel = decode_canonical_bytes(
            next(
                port
                for port in context.input_ports
                if port.payload_schema == UniformElectronGasAnalyticPanel.SCHEMA
            ).read(),
            UniformElectronGasAnalyticPanel,
            maximum_bytes=1024**2,
        )
        result = check_analytic_panel(self.config.source, panel)
        adjudication = context.scientific_adjudication_context
        verdict = ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    port.logical_artifact_id
                    for port in context.output_ports
                    if port.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication.evidence_world_id,
            evidence_world_kind=adjudication.evidence_world_kind,
            relation=adjudication.relation,
            independent_unit_id=adjudication.independent_unit_id,
            information_cutoffs=adjudication.information_cutoffs,
            visibility_ceiling=adjudication.visibility_ceiling,
            outcome_access=adjudication.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=(
                ScientificStatus.SUPPORTED
                if result.passed
                else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=(
                ("UNIFORM_ELECTRON_GAS_SYNTHETIC_ANALYTIC_DEVELOPMENT_ONLY",)
                if result.passed
                else result.reason_codes
            ),
        )
        outputs = {record.SCHEMA: record for record in (result, verdict)}
        return RunnerResult(
            tuple(
                TaskOutputPayload(
                    port.output_id, outputs[port.payload_schema].canonical_bytes()
                )
                for port in context.output_ports
            ),
            (ReceiptCheck("uniform-electron-gas-analytic-finite-q-and-slab-falsifier", True, ()),),
        )


class UniformElectronGasAnalyticEvaluationProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, config: UniformElectronGasAnalyticEvaluationConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError("uniform electron gas analytic evaluation manifest differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config = config

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("uniform electron gas analytic evaluator registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("uniform electron gas analytic evaluator needs its selected issued config")
        return (UniformElectronGasAnalyticEvaluationRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("uniform electron gas analytic evaluator plan or source differs")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
        )
        if (
            len(tasks) != 1
            or tasks[0].task_id != evaluation_task_id(self.config)
            or len(tasks[0].external_inputs) != 1
        ):
            raise ValueError("uniform electron gas analytic evaluator requires one config input")
        spec = tasks[0].external_inputs[0]
        if (
            spec.expected_payload_schema != UniformElectronGasAnalyticEvaluationConfig.SCHEMA
            or spec.expected_content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("uniform electron gas analytic evaluator config identity differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return (
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=UniformElectronGasAnalyticEvaluationConfig.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=self.config.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.fingerprint(),
            ),
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=record_type.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(field.name for field in fields(record_type))),
            )
            for record_type in (UniformElectronGasAnalyticCheck, ScientificAdjudicationRecord)
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self._check(registry)
        if execution_plan is None:
            raise ValueError("uniform electron gas analytic adjudication needs exact output locator")
        outputs = tuple(
            output
            for task in execution_plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for output in task.outputs
            if output.payload_schema == ScientificAdjudicationRecord.SCHEMA
        )
        if len(outputs) != 1:
            raise ValueError("uniform electron gas analytic evaluator requires one adjudication output")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = ['UniformElectronGasAnalyticEvaluationProvider', "evaluation_task_id"]
