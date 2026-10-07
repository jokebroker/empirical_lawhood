"source qualification scalar projection and separate reveal/adjudication runtime providers."

from dataclasses import replace
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskRunner,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.provider import prepared_native_stage_envelope
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from .qualification import PreparedResponseSourceQualificationEvaluationConfig, PreparedResponseSourceQualificationEvaluation, evaluate_prepared_response_source_qualification
from .qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation
from .qualification_projection import project_prepared_response_source_qualification_view


SOURCE_QUALIFICATION_EVALUATOR_TASK_ID = "prepared-response.source-qualification.evaluate"


def _evaluation_task_id(config: CanonicalRecord) -> str:
    config_id = config.config_id  # type: ignore[attr-defined]
    suffix = ".evaluation-config"
    if not config_id.endswith(suffix):
        raise ValueError("prepared response evaluation configuration changes its task identity")
    return f"{config_id[: -len(suffix)]}.evaluate"


def prepared_response_source_qualification_projection_envelope(
    report: PreparedResponseSourceQualificationViewObservation,
) -> LinkedCampaignStageEnvelope:
    complete = report.mode_disposition == "RESOLVED" and all(
        value.delivery_complete
        and all(x is not None for row in value.outputs for x in row)
        for value in report.words
    )
    return envelope(
        report.report_id,
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        complete,
        () if complete else ("INCOMPLETE_OR_UNAVAILABLE_NATIVE_SOURCE_QUALIFICATION_OBSERVATIONS",),
    )


def _read_config(
    context: TaskContext,
    config: PreparedResponseSourceQualificationProjectionConfig | PreparedResponseSourceQualificationEvaluationConfig,
    manifest: CapabilityManifest,
) -> None:
    external = tuple(
        p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL
    )
    if (
        len(external) != 1
        or external[0].artifact_id != f"config-artifact.{config.config_id}"
        or external[0].payload_schema != config.SCHEMA
        or context.config.config_id != config.config_id
        or context.config.config_schema != config.SCHEMA
        or context.config.config_schema_sha256 != manifest.config_schema_sha256
        or context.config.content_sha256 != config.fingerprint()
        or context.config.artifact_id != external[0].artifact_id
    ):
        raise ValueError(
            "prepared source qualification method changes its exact external configuration role"
        )
    if decode_port(external[0], type(config)) != config:
        raise ValueError("prepared source qualification method received another frozen configuration")


def _group(
    context: TaskContext,
    task_id: str,
    materializations: tuple[str, ...],
    expected_schemas: tuple[str, ...],
) -> dict[str, WorkerInputPort]:
    ports = tuple(
        p for p in context.input_ports if p.materialization_id in materializations
    )
    if len(ports) != len(expected_schemas) or {p.payload_schema for p in ports} != set(
        expected_schemas
    ):
        raise ValueError(
            f"prepared source qualification dependency {task_id} changes its complete output census"
        )
    return {p.payload_schema: p for p in ports}


class PreparedResponseSourceQualificationProjectionTask:
    def __init__(
        self, manifest: CapabilityManifest, config: PreparedResponseSourceQualificationProjectionConfig
    ) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"{root.root_id}.project.r{r}": (root, r)
                for root in self.config.native_spec.roots
                for r in (1, 2)
            }
            if context.task_id not in slots:
                raise ValueError(
                    "prepared source qualification projection task is outside its exact root/view census"
                )
            _read_config(context, self.config, self.manifest)
            inputs = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (
                        PreparedNativeTaskResult.SCHEMA,
                        NATIVE_PAIR_SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                    ),
                )
                result = decode_port(
                    group[PreparedNativeTaskResult.SCHEMA], PreparedNativeTaskResult
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if (
                    result.invocation.task_id != receipt.task_id
                    or stage != prepared_native_stage_envelope(result)
                ):
                    raise ValueError(
                        "prepared source qualification projection received a detached native result/envelope"
                    )
                inputs.append(
                    (result, read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES))
                )
            report = project_prepared_response_source_qualification_view(
                self.config, *slots[context.task_id], tuple(inputs)
            )
            stage = prepared_response_source_qualification_projection_envelope(report)
            return output_result(
                context,
                {
                    report.SCHEMA: report.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "complete-native-root-census",
                    "no-native-effect-in-q-projection",
                    "absolute-preparent-task-frame",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def prepared_response_source_qualification_adjudication(
    context: TaskContext, result: PreparedResponseSourceQualificationEvaluation
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError(
            "prepared source qualification evaluator lacks its authenticated adjudication context"
        )
    status = result.scientific_status
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=authority.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(
            sorted(
                p.logical_artifact_id
                for p in context.output_ports
                if p.logical_artifact_id is not None
            )
        ),
        required_receipt_ids=context.dependency_receipt_ids,
        evidence_world_id=authority.evidence_world_id,
        evidence_world_kind=authority.evidence_world_kind,
        relation=authority.relation,
        independent_unit_id=authority.independent_unit_id,
        information_cutoffs=authority.information_cutoffs,
        visibility_ceiling=authority.visibility_ceiling,
        outcome_access=authority.outcome_access,
        evaluability=AdjudicationEvaluability.UNEVALUABLE
        if status is ScientificStatus.UNEVALUABLE
        else AdjudicationEvaluability.EVALUABLE,
        scientific_status=status,
        admission_status=AdmissionStatus.UNEVALUABLE
        if status is ScientificStatus.UNEVALUABLE
        else AdmissionStatus.NOT_EVALUATED,
        reason_codes=result.reasons,
        fixture_scope_id=authority.fixture_scope_id,
        plumbing_only=authority.plumbing_only,
    )


class PreparedResponseSourceQualificationEvaluationTask:
    def __init__(
        self, manifest: CapabilityManifest, config: PreparedResponseSourceQualificationEvaluationConfig
    ) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if (
                context.task_id != _evaluation_task_id(self.config)
                or context.scientific_adjudication_context is None
            ):
                raise ValueError(
                    "prepared source qualification evaluation requires its declared task and authenticated adjudication context"
                )
            _read_config(context, self.config, self.manifest)
            reports = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (
                        PreparedResponseSourceQualificationViewObservation.SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                    ),
                )
                report = decode_port(
                    group[PreparedResponseSourceQualificationViewObservation.SCHEMA], PreparedResponseSourceQualificationViewObservation
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if (
                    report.report_id != receipt.task_id
                    or stage != prepared_response_source_qualification_projection_envelope(report)
                ):
                    raise ValueError(
                        "prepared source qualification evaluation received a detached projection/envelope"
                    )
                reports.append(report)
            result = evaluate_prepared_response_source_qualification(self.config, tuple(reports))
            stage = envelope(
                context.task_id,
                result,
                result.evaluation_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                result.selected_charter is not None,
                result.reasons,
            )
            if result.scientific_status is ScientificStatus.NOT_SUPPORTED:
                stage = replace(
                    stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
                )
            adjudication = prepared_response_source_qualification_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "all-assigned-source-qualification-roots-retained",
                    "fixed-finite-charter-order",
                    "separate-q-adjudication-owner",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedResponseSourceQualificationMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: PreparedResponseSourceQualificationProjectionConfig | PreparedResponseSourceQualificationEvaluationConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version)
            != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError(
                "prepared source qualification method changes its exact registry/configuration"
            )
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared source qualification method runner registry/records differ")
        runner = (
            PreparedResponseSourceQualificationProjectionTask(self.manifest, self.config)
            if isinstance(self.config, PreparedResponseSourceQualificationProjectionConfig)
            else PreparedResponseSourceQualificationEvaluationTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared source qualification method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.project.r{r}"
                for root in self.config.native_spec.roots
                for r in (1, 2)
            }
            if isinstance(self.config, PreparedResponseSourceQualificationProjectionConfig)
            else {_evaluation_task_id(self.config)}
        )
        return config_payloads(
            plan, self.manifest, self.config, self.config.config_id, expected
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("prepared source qualification method semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedResponseSourceQualificationViewObservation, LinkedCampaignStageEnvelope)
            if isinstance(self.config, PreparedResponseSourceQualificationProjectionConfig)
            else (
                PreparedResponseSourceQualificationEvaluation,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("prepared source qualification method adjudication registry differs")
        if isinstance(self.config, PreparedResponseSourceQualificationProjectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{_evaluation_task_id(self.config)}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )
