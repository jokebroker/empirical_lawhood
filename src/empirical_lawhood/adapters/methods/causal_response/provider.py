"causal response prediction scalar projection and separate reveal/adjudication runtime bindings."

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
from .records import CausalResponseProjectionConfig, CausalResponseEvaluationConfig, CausalResponseViewObservation, CausalResponseEvaluation, CausalResponseCommittedPrediction
from .projection import project_root, evaluate


EVALUATOR_TASK_ID = "causal-response-prediction.evaluate"


def _evaluation_task_id(config: CanonicalRecord) -> str:
    config_id = config.config_id  # type: ignore[attr-defined]
    suffix = ".evaluation-config"
    if not config_id.endswith(suffix):
        raise ValueError("prepared response evaluation configuration changes its task identity")
    return f"{config_id[: -len(suffix)]}.evaluate"


def projection_envelope(
    report: CausalResponseViewObservation,
) -> LinkedCampaignStageEnvelope:
    complete = not report.reasons
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
    config: CausalResponseProjectionConfig | CausalResponseEvaluationConfig,
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
        raise ValueError("causal response prediction method changes its exact external configuration role")
    if decode_port(external[0], type(config)) != config:
        raise ValueError("causal response prediction method received another frozen configuration")


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
            f"causal response prediction dependency {task_id} changes its complete output census"
        )
    return {p.payload_schema: p for p in ports}


class CausalResponseProjectionTask:
    def __init__(
        self, manifest: CapabilityManifest, config: CausalResponseProjectionConfig
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
                    "causal response prediction projection task is outside its exact root/view census"
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
                        CausalResponseCommittedPrediction.SCHEMA,
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
                        "causal response prediction projection received a detached native result/envelope"
                    )
                inputs.append(
                    (
                        result,
                        read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES),
                        decode_port(
                            group[CausalResponseCommittedPrediction.SCHEMA],
                            CausalResponseCommittedPrediction,
                        ),
                    )
                )
            report = project_root(self.config, *slots[context.task_id], tuple(inputs))
            stage = projection_envelope(report)
            return output_result(
                context,
                {
                    report.SCHEMA: report.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "complete-native-root-census",
                    "no-native-effect-in-q-projection",
                    "committed-handoff-predictions-and-independent-futures",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def make_adjudication(
    context: TaskContext, result: CausalResponseEvaluation
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("causal response prediction evaluator lacks its authenticated adjudication context")
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


class CausalResponseEvaluationTask:
    def __init__(
        self, manifest: CapabilityManifest, config: CausalResponseEvaluationConfig
    ) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if (
                context.task_id != _evaluation_task_id(self.config)
                or context.scientific_adjudication_context is None
            ):
                raise ValueError(
                    "causal response prediction evaluation requires its declared task and authenticated adjudication context"
                )
            _read_config(context, self.config, self.manifest)
            reports = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (CausalResponseViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                report = decode_port(
                    group[CausalResponseViewObservation.SCHEMA], CausalResponseViewObservation
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if report.report_id != receipt.task_id or stage != projection_envelope(
                    report
                ):
                    raise ValueError(
                        "causal response prediction evaluation received a detached projection/envelope"
                    )
                reports.append(report)
            result = evaluate(self.config, tuple(reports))
            stage = envelope(
                context.task_id,
                result,
                result.evaluation_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                result.scientific_status
                in (ScientificStatus.SUPPORTED, ScientificStatus.MIXED),
                result.reasons,
            )
            if result.scientific_status is ScientificStatus.NOT_SUPPORTED:
                stage = replace(
                    stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
                )
            adjudication = make_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "all-assigned-source-qualification-roots-retained",
                    "predeclared-root-sign-and-practical-effect-comparison",
                    "separate-q-adjudication-owner",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class CausalResponseMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: CausalResponseProjectionConfig | CausalResponseEvaluationConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version)
            != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("causal response prediction method changes its exact registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("causal response prediction method runner registry/records differ")
        runner = (
            CausalResponseProjectionTask(self.manifest, self.config)
            if isinstance(self.config, CausalResponseProjectionConfig)
            else CausalResponseEvaluationTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("causal response prediction method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.project.r{r}"
                for root in self.config.native_spec.roots
                for r in (1, 2)
            }
            if isinstance(self.config, CausalResponseProjectionConfig)
            else {_evaluation_task_id(self.config)}
        )
        return config_payloads(
            plan, self.manifest, self.config, self.config.config_id, expected
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("causal response prediction method semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (CausalResponseViewObservation, LinkedCampaignStageEnvelope)
            if isinstance(self.config, CausalResponseProjectionConfig)
            else (
                CausalResponseEvaluation,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("causal response prediction method adjudication registry differs")
        if isinstance(self.config, CausalResponseProjectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{_evaluation_task_id(self.config)}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )
