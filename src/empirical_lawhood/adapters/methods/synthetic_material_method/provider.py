'Receipt-bound ambient pressure superconductor gauge covariant response method check attached to the one native panel.'

from dataclasses import fields

from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_contracts import SyntheticMaterialResponseMethodCheck, SyntheticMaterialResponseMethodEvaluationConfig, SyntheticMaterialResponseMethodPanel, check_synthetic_material_response_method_conformance
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


def evaluation_task_id(config: SyntheticMaterialResponseMethodEvaluationConfig) -> str:
    suffix = ".evaluation-config"
    if not config.config_id.endswith(suffix):
        raise ValueError('ambient pressure superconductor gauge covariant response method evaluation config has no experiment binding')
    return f'{config.config_id.removesuffix(suffix)}.gauge-covariant-response-method-evaluate'


class SyntheticMaterialResponseMethodEvaluationRunner:
    manifest = CAPABILITY

    def __init__(self, config: SyntheticMaterialResponseMethodEvaluationConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id != evaluation_task_id(self.config)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.input_ports) != 2
            or sum(
                port.payload_schema == SyntheticMaterialResponseMethodPanel.SCHEMA
                and port.outcome_access is OutcomeAccess.EVALUATION_SEALED
                and port.size_bytes <= 1024**2
                for port in context.input_ports
            )
            != 1
            or sum(
                port.payload_schema == SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA
                and port.outcome_access is OutcomeAccess.OUTCOME_BLIND
                and port.size_bytes <= 1024**2
                for port in context.input_ports
            )
            != 1
            or len(context.output_ports) != 2
            or {port.payload_schema for port in context.output_ports}
            != {SyntheticMaterialResponseMethodCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA}
            or context.scientific_adjudication_context is None
            or not context.dependency_receipt_ids
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluation task or sealed ports differ')
        supplied = decode_canonical_bytes(
            next(
                port
                for port in context.input_ports
                if port.payload_schema == SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA
            ).read(),
            SyntheticMaterialResponseMethodEvaluationConfig,
            maximum_bytes=1024**2,
        )
        if supplied != self.config:
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method evaluator config differs from selected provider'
            )
        panel = decode_canonical_bytes(
            next(
                port
                for port in context.input_ports
                if port.payload_schema == SyntheticMaterialResponseMethodPanel.SCHEMA
            ).read(),
            SyntheticMaterialResponseMethodPanel,
            maximum_bytes=1024**2,
        )
        result = check_synthetic_material_response_method_conformance(self.config.source, panel)
        adjudication = context.scientific_adjudication_context
        verdict = ScientificAdjudicationRecord(
            adjudication_id=f'adjudication.{context.run_id}.{context.task_id}',
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
                ("SYNTHETIC_MATERIAL_RESPONSE_METHOD_FIXTURE_ONLY",)
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
            (ReceiptCheck('ambient-pressure-superconductor-gauge-covariant-response-nine-fixture-falsifier', True, ()),),
        )


class SyntheticMaterialResponseMethodEvaluationProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, config: SyntheticMaterialResponseMethodEvaluationConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluation manifest differs')
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config = config

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluator registry differs')

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method evaluator needs its selected issued config'
            )
        return (SyntheticMaterialResponseMethodEvaluationRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluator plan or source differs')
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
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluator requires one config input')
        spec = tasks[0].external_inputs[0]
        if (
            spec.expected_payload_schema != SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA
            or spec.expected_content_sha256 != self.config.fingerprint()
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response method evaluator config identity differs')
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return (
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA,
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
            for record_type in (SyntheticMaterialResponseMethodCheck, ScientificAdjudicationRecord)
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self._check(registry)
        if execution_plan is None:
            raise ValueError('ambient pressure superconductor gauge covariant response method adjudication needs exact output locator')
        outputs = tuple(
            output
            for task in execution_plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for output in task.outputs
            if output.payload_schema == ScientificAdjudicationRecord.SCHEMA
        )
        if len(outputs) != 1:
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method evaluator requires one adjudication output'
            )
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = ['SyntheticMaterialResponseMethodEvaluationProvider', "evaluation_task_id"]
