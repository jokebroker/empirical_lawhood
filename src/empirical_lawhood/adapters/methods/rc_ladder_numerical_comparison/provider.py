"""Receipt-bound RC numerical comparison attached to two native view tasks."""

from dataclasses import fields

from empirical_lawhood.adapters.simulators.rc_ladder_response.campaign import check_native_views
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderNumericalCheck
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

from .contracts import ResistorCapacitorLadderEvaluationConfig
from .extension_bundle import CAPABILITY


def _task_id(config: ResistorCapacitorLadderEvaluationConfig) -> str:
    suffix = ".evaluation-config"
    if not config.config_id.endswith(suffix):
        raise ValueError("RC evaluator config lacks its experiment binding")
    return f"{config.config_id.removesuffix(suffix)}.numerical-evaluate"


class ResistorCapacitorLadderNumericalRunner:
    manifest = CAPABILITY

    def __init__(self, config: ResistorCapacitorLadderEvaluationConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id != _task_id(self.config)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.input_ports) != 3
            or sum(
                port.payload_schema == ResistorCapacitorLadderNativePanel.SCHEMA
                and port.outcome_access is OutcomeAccess.EVALUATION_SEALED
                and port.size_bytes <= 512 * 1024
                for port in context.input_ports
            ) != 2
            or sum(
                port.payload_schema == ResistorCapacitorLadderEvaluationConfig.SCHEMA
                and port.outcome_access is OutcomeAccess.OUTCOME_BLIND
                and port.size_bytes <= 512 * 1024
                for port in context.input_ports
            ) != 1
            or len(context.output_ports) != 2
            or {port.payload_schema for port in context.output_ports}
            != {ResistorCapacitorLadderNumericalCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA}
            or context.scientific_adjudication_context is None
            or not context.dependency_receipt_ids
        ):
            raise ValueError(
                "RC evaluation task, sealed inputs or adjudication context differ"
            )
        config_port = next(
            port
            for port in context.input_ports
            if port.payload_schema == ResistorCapacitorLadderEvaluationConfig.SCHEMA
        )
        config = decode_canonical_bytes(
            config_port.read(), ResistorCapacitorLadderEvaluationConfig, maximum_bytes=512 * 1024
        )
        if config != self.config:
            raise ValueError("RC evaluator config bytes differ from selected provider")
        panels = tuple(
            decode_canonical_bytes(
                port.read(), ResistorCapacitorLadderNativePanel, maximum_bytes=512 * 1024
            )
            for port in context.input_ports
            if port.payload_schema == ResistorCapacitorLadderNativePanel.SCHEMA
        )
        result = check_native_views(self.config.study, panels)
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
                if result.converged
                else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=(
                ("RC_SYNTHETIC_NUMERICAL_CONFORMANCE_ONLY",)
                if result.converged
                else result.reason_codes
            ),
        )
        records = {record.SCHEMA: record for record in (result, verdict)}
        return RunnerResult(
            tuple(
                TaskOutputPayload(
                    port.output_id, records[port.payload_schema].canonical_bytes()
                )
                for port in context.output_ports
            ),
            (ReceiptCheck("rc-frozen-numerical-falsifier", True, ()),),
        )


class ResistorCapacitorLadderNumericalProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, config: ResistorCapacitorLadderEvaluationConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError("RC numerical provider installed manifest differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config = config

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("RC numerical provider registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("RC numerical evaluator requires its issued config")
        return (ResistorCapacitorLadderNumericalRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("RC numerical evaluator plan or source differs")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
        )
        if (
            len(tasks) != 1
            or tasks[0].task_id != _task_id(self.config)
            or len(tasks[0].external_inputs) != 1
        ):
            raise ValueError("RC numerical evaluator requires one exact config input")
        spec = tasks[0].external_inputs[0]
        if (
            spec.expected_payload_schema != ResistorCapacitorLadderEvaluationConfig.SCHEMA
            or spec.expected_content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("RC numerical evaluator config identity differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return (
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=ResistorCapacitorLadderEvaluationConfig.SCHEMA,
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
            for record_type in (ResistorCapacitorLadderNumericalCheck, ScientificAdjudicationRecord)
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self._check(registry)
        if execution_plan is None:
            raise ValueError("RC numerical adjudication needs its exact output locator")
        outputs = tuple(
            output
            for task in execution_plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for output in task.outputs
            if output.payload_schema == ScientificAdjudicationRecord.SCHEMA
        )
        if len(outputs) != 1:
            raise ValueError("RC numerical evaluator requires one adjudication output")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = ['ResistorCapacitorLadderNumericalProvider']
