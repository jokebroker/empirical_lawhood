"""Registered paired projection and separate native-completion adjudication."""

from typing import TYPE_CHECKING, cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.provider import native_stage, native_result_type
from .native_projection import project_native_view
from .native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeEvaluationConfig, FiniteResponseLawNativeViewObservation, FiniteResponseLawNativeEvaluation, native_method_types

if TYPE_CHECKING:
    from .evaluation_results import FiniteResponseLawEvaluationCohort

EVALUATOR_TASK_ID = "finite-response-law.native-evaluate"


def projection_stage(report: FiniteResponseLawNativeViewObservation) -> LinkedCampaignStageEnvelope:
    return envelope(
        report.report_id,
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        not report.reasons,
        report.reasons,
    )


def _read_config(
    context: TaskContext,
    config: FiniteResponseLawProjectionConfig | FiniteResponseLawNativeEvaluationConfig,
    manifest: CapabilityManifest,
    outputs_per_dependency: int | None,
) -> None:
    external = tuple(p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL)
    if (
        len(external) != 1
        or external[0].artifact_id != f"config-artifact.{config.config_id}"
        or external[0].payload_schema != config.SCHEMA
        or context.config.config_id != config.config_id
        or context.config.config_schema != config.SCHEMA
        or context.config.config_schema_sha256 != manifest.config_schema_sha256
        or context.config.content_sha256 != config.fingerprint()
        or context.config.artifact_id != external[0].artifact_id
        or len(context.input_ports)
        != 1
        + (
            sum(len(r.output_materialization_ids) for r in context.dependency_receipts)
            if outputs_per_dependency is None
            else outputs_per_dependency * len(context.dependency_receipts)
        )
    ):
        raise ValueError("Finite response-law method changes its exact config/dependency input roster")
    if decode_port(external[0], type(config)) != config:
        raise ValueError("Finite response-law method received another frozen configuration")


class FiniteResponseLawProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawProjectionConfig) -> None:
        self.manifest, self.config = manifest, config
        self.result_type = native_result_type(config.native_spec)

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"{root.root_id}.flh-project.r{r}": (root, r)
                for root in self.config.native_spec.roots
                for r in (1, 2)
            }
            if context.task_id not in slots:
                raise ValueError("Finite response-law projection task changes its assigned root/view")
            _read_config(context, self.config, self.manifest, 3)
            inputs = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (
                        self.result_type.SCHEMA,
                        NATIVE_PAIR_SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                    ),
                )
                record = decode_port(group[self.result_type.SCHEMA], self.result_type)
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                )
                if record.invocation.task_id != receipt.task_id or stage != native_stage(record):
                    raise ValueError(
                        "Finite response-law native projection input differs from its task/stage receipt"
                    )
                inputs.append((record, read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)))
            root, refinement = slots[context.task_id]
            report = project_native_view(
                self.config,
                root,
                refinement,
                tuple(sorted(inputs, key=lambda v: v[0].invocation.task_id)),
            )
            stage = projection_stage(report)
            return output_result(
                context,
                {report.SCHEMA: report.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()},
                (
                    "all-assigned-native-paths-projected",
                    "same-purpose-paired-hold",
                    "native-192-readout",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def native_adjudication(
    context: TaskContext, result: 'FiniteResponseLawNativeEvaluation | FiniteResponseLawEvaluationCohort'
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("Finite response-law native evaluator lacks its authenticated adjudication context")
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
        reason_codes=result.reasons or ("NATIVE_MEASUREMENT_CENSUS_COMPLETE",),
        fixture_scope_id=authority.fixture_scope_id,
        plumbing_only=authority.plumbing_only,
    )


class FiniteResponseLawNativeEvaluationTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawNativeEvaluationConfig) -> None:
        self.manifest, self.config = manifest, config
        _, _, _, self.view_type, self.evaluation_type = native_method_types(
            config.projection.native_spec
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if (
                context.task_id != EVALUATOR_TASK_ID
                or context.scientific_adjudication_context is None
            ):
                raise ValueError(
                    "Finite response-law native evaluator changes its task or lacks adjudication authority"
                )
            _read_config(context, self.config, self.manifest, 2)
            reports = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (self.view_type.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                report = decode_port(group[self.view_type.SCHEMA], self.view_type)
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                )
                if report.report_id != receipt.task_id or stage != projection_stage(report):
                    raise ValueError("Finite response-law native evaluator received a detached projection receipt")
                reports.append(report)
            result = self.evaluation_type(
                self.config, tuple(sorted(reports, key=lambda v: v.report_id))
            )
            stage = envelope(
                context.task_id,
                result,
                result.evaluation_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                not result.reasons,
                result.reasons,
            )
            adjudication = native_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "complete-native-measurement-census",
                    "independent-unit-accounting",
                    "separate-source-completion-adjudication",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawNativeMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawProjectionConfig | FiniteResponseLawNativeEvaluationConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("Finite response-law native method changes its registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Finite response-law native method runner registry/records differ")
        runner = (
            FiniteResponseLawProjectionTask(self.manifest, self.config)
            if isinstance(self.config, FiniteResponseLawProjectionConfig)
            else FiniteResponseLawNativeEvaluationTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Finite response-law native method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.flh-project.r{r}"
                for root in self.config.native_spec.roots
                for r in (1, 2)
            }
            if isinstance(self.config, FiniteResponseLawProjectionConfig)
            else {EVALUATOR_TASK_ID}
        )
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Finite response-law native method semantic registry differs")
        source = (
            self.config.native_spec
            if isinstance(self.config, FiniteResponseLawProjectionConfig)
            else self.config.projection.native_spec
        )
        _, _, _, view_type, evaluation_type = native_method_types(source)
        records = (
            (view_type, LinkedCampaignStageEnvelope)
            if isinstance(self.config, FiniteResponseLawProjectionConfig)
            else (
                evaluation_type,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            )
        )
        return semantic_contracts(self.manifest, records, None)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("Finite response-law native method adjudication registry differs")
        if isinstance(self.config, FiniteResponseLawProjectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{EVALUATOR_TASK_ID}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )
