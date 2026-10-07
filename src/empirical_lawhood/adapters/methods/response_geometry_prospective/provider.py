"""assay projection/evaluation at existing sealed-input and adjudication ports."""

from hashlib import sha256
from dataclasses import replace
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import read_port, decode_port, envelope, output_result, config_payloads, semantic_contracts, source_stage_envelope
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeSegmentResult, assay_roots
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA

from .projection import ASSAY_DIAGNOSTICS_SCHEMA, project_response_geometry_assay_view
from .qualification import ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig, ResponseGeometryAssayViewReport, ResponseGeometryAssayEvaluation, evaluate_response_geometry_assay


EVALUATOR_TASK_ID = "response-geometry-assay.evaluate"


def projection_stage_envelope(report: ResponseGeometryAssayViewReport) -> LinkedCampaignStageEnvelope:
    return envelope(
        f"{report.root.root_id}.project.r{report.refinement}",
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        all(packet.delivered for packet in report.packets),
        tuple(
            sorted({reason for packet in report.packets for reason in packet.qualification_reasons})
        ),
    )


class ResponseGeometryAssayProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryAssayProjectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        slot = next(
            (
                (root, refinement)
                for root in assay_roots()
                for refinement in root.refinements
                if context.task_id == f"{root.root_id}.project.r{refinement}"
            ),
            None,
        )
        if slot is None:
            raise ValueError("assay projection task differs from its exact native root/view")
        configs, results, stages, payloads = [], [], [], {}
        for port in context.input_ports:
            if port.payload_schema == ResponseGeometryAssayProjectionConfig.SCHEMA:
                configs.append(decode_port(port, ResponseGeometryAssayProjectionConfig))
            elif port.payload_schema == ResponseGeometryAssayNativeSegmentResult.SCHEMA:
                results.append(decode_port(port, ResponseGeometryAssayNativeSegmentResult))
            elif port.payload_schema == LinkedCampaignStageEnvelope.SCHEMA:
                stages.append(decode_port(port, LinkedCampaignStageEnvelope))
            elif port.payload_schema == ASSAY_HDF5_SCHEMA:
                payload = read_port(port, 16 * 1024**2)
                digest = sha256(payload).hexdigest()
                if digest in payloads:
                    raise ValueError("assay projection repeats a native shard")
                payloads[digest] = payload
            else:
                port.close()
                raise ValueError("assay projection input schema differs")
        if (
            configs != [self.config]
            or len(stages) != len(results)
            or len(payloads) != len(results)
            or set(stages) != {source_stage_envelope(value) for value in results}
            or set(payloads) != {value.observations_sha256 for value in results}
        ):
            raise ValueError("assay projection input config/custody roster differs")
        report, data = project_response_geometry_assay_view(
            self.config,
            *slot,
            tuple((value, payloads[value.observations_sha256]) for value in results),
        )
        stage = projection_stage_envelope(report)
        return output_result(
            context,
            {
                report.SCHEMA: report.canonical_bytes(),
                ASSAY_DIAGNOSTICS_SCHEMA: data,
                stage.SCHEMA: stage.canonical_bytes(),
            },
            (
                "exact-root-view-source-custody",
                "causal-prefix-only-geometry",
                "no-native-effect-in-projection",
            ),
        )


def _assay_status(evaluation: ResponseGeometryAssayEvaluation) -> ScientificStatus:
    # The scientific owner already applies the common-assay/earliest-landmark
    # rule. Requiring every unselected cell here would create a second rule.
    return {
        "FIXED_ORIGIN_QUALIFIED": ScientificStatus.SUPPORTED,
        "INNER_ACTION_CHANNEL_UNQUALIFIED": ScientificStatus.NOT_SUPPORTED,
        "NUMERICAL_OR_SEMANTIC_SUPPORT_UNRESOLVED": ScientificStatus.UNEVALUABLE,
    }[evaluation.terminal]


def _adjudication(context: TaskContext, evaluation: ResponseGeometryAssayEvaluation) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("assay evaluator lacks its authenticated scientific adjudication context")
    status = _assay_status(evaluation)
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=authority.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
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
        evaluability=AdjudicationEvaluability.EVALUABLE
        if status is not ScientificStatus.UNEVALUABLE
        else AdjudicationEvaluability.UNEVALUABLE,
        scientific_status=status,
        admission_status=(
            AdmissionStatus.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else AdmissionStatus.NOT_EVALUATED
        ),
        reason_codes=evaluation.reasons,
        fixture_scope_id=authority.fixture_scope_id,
        plumbing_only=authority.plumbing_only,
    )


class ResponseGeometryAssayEvaluationTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryAssayEvaluationConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        if context.task_id != EVALUATOR_TASK_ID:
            raise ValueError("assay evaluator task identity differs")
        configs, reports, stages, diagnostics = [], [], [], set()
        for port in context.input_ports:
            if port.payload_schema == ResponseGeometryAssayEvaluationConfig.SCHEMA:
                configs.append(decode_port(port, ResponseGeometryAssayEvaluationConfig))
            elif port.payload_schema == ResponseGeometryAssayViewReport.SCHEMA:
                reports.append(decode_port(port, ResponseGeometryAssayViewReport))
            elif port.payload_schema == LinkedCampaignStageEnvelope.SCHEMA:
                stages.append(decode_port(port, LinkedCampaignStageEnvelope))
            elif port.payload_schema == ASSAY_DIAGNOSTICS_SCHEMA:
                digest = sha256(read_port(port, 16 * 1024**2)).hexdigest()
                if digest in diagnostics:
                    raise ValueError("assay evaluator repeats a diagnostic shard")
                diagnostics.add(digest)
            else:
                port.close()
                raise ValueError("assay evaluator input schema differs")
        if (
            configs != [self.config]
            or len(stages) != 72
            or len(diagnostics) != 72
            or set(stages) != {projection_stage_envelope(value) for value in reports}
            or diagnostics != {value.diagnostics_sha256 for value in reports}
        ):
            raise ValueError("assay evaluator input config/custody roster differs")
        evaluation = evaluate_response_geometry_assay(self.config, tuple(reports))
        stage = envelope(
            context.task_id,
            evaluation,
            evaluation.evaluation_id,
            LinkedCampaignStageRole.METHOD_IDENTIFICATION,
            evaluation.selected_assay is not None,
            evaluation.reasons,
        )
        if _assay_status(evaluation) is ScientificStatus.NOT_SUPPORTED:
            stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
        adjudication = _adjudication(context, evaluation)
        return output_result(
            context,
            {
                evaluation.SCHEMA: evaluation.canonical_bytes(),
                stage.SCHEMA: stage.canonical_bytes(),
                adjudication.SCHEMA: adjudication.canonical_bytes(),
            },
            (
                "all-declared-independent-roots-retained",
                "finite-frozen-assay-selection",
                "sole-runtime-adjudication-contract",
            ),
        )


class ResponseGeometryAssayMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: ResponseGeometryAssayProjectionConfig | ResponseGeometryAssayEvaluationConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("assay method registry/config differs")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("assay method runner registry/records differ")
        runner = (
            ResponseGeometryAssayProjectionTask(self.manifest, self.config)
            if isinstance(self.config, ResponseGeometryAssayProjectionConfig)
            else ResponseGeometryAssayEvaluationTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("assay method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.project.r{refinement}"
                for root in assay_roots()
                for refinement in root.refinements
            }
            if isinstance(self.config, ResponseGeometryAssayProjectionConfig)
            else {EVALUATOR_TASK_ID}
        )
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("assay method semantic registry differs")
        if isinstance(self.config, ResponseGeometryAssayProjectionConfig):
            return semantic_contracts(
                self.manifest, (ResponseGeometryAssayViewReport, LinkedCampaignStageEnvelope), ASSAY_DIAGNOSTICS_SCHEMA
            )
        return semantic_contracts(
            self.manifest,
            (ResponseGeometryAssayEvaluation, LinkedCampaignStageEnvelope, ScientificAdjudicationRecord),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("assay method adjudication registry differs")
        if isinstance(self.config, ResponseGeometryAssayProjectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{EVALUATOR_TASK_ID}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )
