"Finite response-law reveal/controller use/scalar inference through the ordinary registered evaluator."

from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationRecord,
    ScientificAdjudicationOutputContract,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedPolicyUnitEvaluation
from empirical_lawhood.runtime.execution import (
    TaskContext,
    TaskRunner,
    RunnerResult,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, output_result, semantic_contracts
from .consumer import FiniteResponseLawNativeUnitReadoutMap, SPEC, response_coordinates
from .control_closeout import FiniteResponseLawRootSealedControl
from .control_evaluation import evaluation_design
from .control_provider import FREEZE_PRIOR
from .method_records import FiniteResponseLawAssignedQualificationReport
from .control_ports import FiniteResponseLawControlRuntimePort
from .control_reveal import FiniteResponseLawProspectiveRootEvaluation, reveal_root
from .assigned_native_records import FiniteResponseLawAssignedEvaluationViewObservation, FiniteResponseLawAssignedEvaluationNativeCompletion
from .evaluation_readout import FiniteResponseLawRootInferenceOperands, root_operands, cohort_inference
from .evaluation_results import FiniteResponseLawEvaluationRevealConfig, FiniteResponseLawEvaluationCohort, COHORT, ADJUDICATE, reveal_tasks, encode_cohort_statistics
from .native_provider import EVALUATOR_TASK_ID, native_adjudication, projection_stage


def reveal_dependencies(config: FiniteResponseLawEvaluationRevealConfig, task_id: str) -> tuple[str, ...]:
    if task_id == ADJUDICATE:
        return (COHORT,)
    if task_id == COHORT:
        return tuple(
            sorted(
                (
                    FREEZE_PRIOR,
                    *(f"{r.stage_unit}.evaluate-use" for r in config.control.source.roots),
                )
            )
        )
    root = next(
        (r for r in config.control.source.roots if task_id == f"{r.stage_unit}.evaluate-use"), None
    )
    if root is None:
        raise ValueError("finite response-law reveal is outside its frozen task census")
    return tuple(
        sorted(
            (
                EVALUATOR_TASK_ID,
                f"{root.stage_unit}.seal-control",
                *(f"{root.root_id}.flh-project.r{v}" for v in (1, 2)),
            )
        )
    )


class FiniteResponseLawEvaluationRevealTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawEvaluationRevealConfig,
        runtime: FiniteResponseLawControlRuntimePort,
    ):
        self.manifest, self.config, self.runtime = manifest, config, runtime

    def _group(
        self, context: TaskContext, task: str, schemas: tuple[str, ...]
    ) -> dict[str, WorkerInputPort]:
        receipts = tuple(r for r in context.dependency_receipts if r.task_id == task)
        if len(receipts) != 1:
            raise ValueError("finite response-law reveal lacks exactly one declared dependency receipt")
        return _group(context, task, receipts[0].output_materialization_ids, schemas)

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            expected = reveal_dependencies(self.config, context.task_id)
            if tuple(sorted(r.task_id for r in context.dependency_receipts)) != expected:
                raise ValueError("finite response-law reveal changes recovery/reveal/inference ordering")
            external = {
                p.artifact_id: p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL
            }
            config_id = f"config-artifact.{self.config.config_id}"
            if (
                set(external) != {config_id}
                or context.config.content_sha256 != self.config.fingerprint()
                or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
                or decode_port(external[config_id], FiniteResponseLawEvaluationRevealConfig) != self.config
            ):
                raise ValueError("finite response-law reveal changes its frozen configuration or prior inputs")
            authority = self.runtime.outcome_authority()
            authority_id = ObjectIdentity.from_record(authority.authority_id, authority)
            if context.task_id == ADJUDICATE:
                group = self._group(context, COHORT, (FiniteResponseLawEvaluationCohort.SCHEMA,))
                cohort = decode_port(
                    group[FiniteResponseLawEvaluationCohort.SCHEMA], FiniteResponseLawEvaluationCohort, maximum=16 * 1024**2
                )
                if cohort.config != ObjectIdentity.from_record(self.config.config_id, self.config):
                    raise ValueError("finite response-law adjudication changes its frozen readout configuration")
                # Shared adjudication custody/denominator binding; scientific
                # status/reasons come solely from the complete prospective evaluation cohort.
                adjudication = native_adjudication(context, cohort)
                values = {adjudication.SCHEMA: adjudication.canonical_bytes()}
            elif context.task_id == COHORT:
                group = self._group(context, FREEZE_PRIOR, (FiniteResponseLawAssignedQualificationReport.SCHEMA,))
                report = decode_port(
                    group[FiniteResponseLawAssignedQualificationReport.SCHEMA],
                    FiniteResponseLawAssignedQualificationReport,
                    maximum=8 * 1024**2,
                )
                if (
                    report.fingerprint() != self.config.control.qualification.sha256
                    or not report.eligible_for_prospective_evaluation
                ):
                    raise ValueError(
                        "finite response-law inference substitutes the authenticated prior qualification"
                    )
                units: list[PreparedPolicyUnitEvaluation] = []
                operands: list[FiniteResponseLawRootInferenceOperands] = []
                plan, evaluator = None, None
                for task in expected:
                    if task == FREEZE_PRIOR:
                        continue
                    group = self._group(
                        context,
                        task,
                        (FiniteResponseLawProspectiveRootEvaluation.SCHEMA, FiniteResponseLawRootInferenceOperands.SCHEMA),
                    )
                    decoded_prospective_root_evaluation = decode_port(
                        group[FiniteResponseLawProspectiveRootEvaluation.SCHEMA],
                        FiniteResponseLawProspectiveRootEvaluation,
                        maximum=16 * 1024**2,
                    )
                    operand = decode_port(
                        group[FiniteResponseLawRootInferenceOperands.SCHEMA], FiniteResponseLawRootInferenceOperands
                    )
                    if task != f"{operand.root_id}.evaluate-use" or any(
                        u.root_id != operand.root_id for u in decoded_prospective_root_evaluation.units
                    ):
                        raise ValueError("finite response-law inference substitutes a root's receipted controller use operands")
                    if plan is None:
                        design = decoded_prospective_root_evaluation.revealed[0].sealed.design
                        scope = design.evaluation_plan
                        mapping = FiniteResponseLawNativeUnitReadoutMap(
                            response_coordinates(ObjectIdentity.from_record("flh-science", SPEC))
                        )
                        plan, _ = evaluation_design(
                            source=self.config.control.source,
                            evaluator=design.evaluator_binding,
                            readout_map=mapping,
                            view_ids=scope.numerical_view_ids,
                            hold_word=scope.matched_hold_word,
                        )
                        evaluator = NestedControllerUseEvaluator(
                            design.evaluator_binding, plan.reducer
                        )
                    units.extend(decoded_prospective_root_evaluation.units)
                    operands.append(operand)
                assert plan is not None and evaluator is not None
                generic = evaluator.reduce_prepared_cohort(plan=plan, units=tuple(units))
                qualifications = {
                    b.boundary: q.scientific_status is ScientificStatus.SUPPORTED
                    for b, q in zip(
                        report.calibration.boundaries, report.qualifications, strict=True
                    )
                }
                rows = tuple(sorted(operands, key=lambda r: r.root_id))
                stats = cohort_inference(
                    rows,
                    lower_qualified=qualifications["lower"],
                    cached_qualified=qualifications["cached"],
                )
                cohort = FiniteResponseLawEvaluationCohort(
                    ObjectIdentity.from_record(self.config.config_id, self.config),
                    ObjectIdentity.from_record(report.report_id, report),
                    generic,
                    rows,
                    qualifications["lower"],
                    qualifications["cached"],
                    encode_cohort_statistics(stats),
                )
                values = {cohort.SCHEMA: cohort.canonical_bytes()}
            else:
                root_id = context.task_id.removesuffix(".evaluate-use")
                group = self._group(
                    context, f"{root_id}.seal-control", (FiniteResponseLawRootSealedControl.SCHEMA,)
                )
                sealed = decode_port(
                    group[FiniteResponseLawRootSealedControl.SCHEMA],
                    FiniteResponseLawRootSealedControl,
                    maximum=16 * 1024**2,
                )
                if sealed.join.lock.forecast.root.stage_unit != root_id:
                    raise ValueError("finite response-law reveal substitutes its sealed root")
                # The terminal receipt authenticates the closed whole cohort.
                # Its full aggregate need not be decoded again for each root.
                self._group(context, EVALUATOR_TASK_ID, (FiniteResponseLawAssignedEvaluationNativeCompletion.SCHEMA,))
                views = []
                for v in (1, 2):
                    task = f"{sealed.join.lock.forecast.root.root_id}.flh-project.r{v}"
                    group = self._group(
                        context,
                        task,
                        (
                            FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
                            LinkedCampaignStageEnvelope.SCHEMA,
                        ),
                    )
                    view = decode_port(
                        group[FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA], FiniteResponseLawAssignedEvaluationViewObservation
                    )
                    if decode_port(
                        group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                    ) != projection_stage(view):
                        raise ValueError("finite response-law reveal substitutes a projected numerical view")
                    views.append(view)
                prospective_root_evaluation = reveal_root(
                    sealed=sealed,
                    views=tuple(views),
                    reveal_authority=authority_id,
                    store=self.runtime.prepared_store,
                )
                scalar = root_operands(sealed, tuple(views), prospective_root_evaluation)
                values = {prospective_root_evaluation.SCHEMA: prospective_root_evaluation.canonical_bytes(), scalar.SCHEMA: scalar.canonical_bytes()}
            return output_result(
                context, values, ('existing-prospective-evaluation-and-independent-all-root-inference',)
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawEvaluationRevealProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawEvaluationRevealConfig,
        runtime: FiniteResponseLawControlRuntimePort,
    ):
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("finite response-law reveal changes its installed registry/configuration")
        self.registry, self.manifest, self.config, self.runtime = (
            registry,
            manifest,
            config,
            runtime,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("finite response-law reveal changes its runner registry")
        return (
            cast(TaskRunner, FiniteResponseLawEvaluationRevealTask(self.manifest, self.config, self.runtime)),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("finite response-law reveal changes its execution registry")
        # Prior qualification arrives through its ordinary freeze dependency;
        # external payload ownership stays with the control provider.
        return config_payloads(
            plan, self.manifest, self.config, self.config.config_id, set(reveal_tasks(self.config))
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("finite response-law reveal changes its output registry")
        return semantic_contracts(
            self.manifest,
            (
                FiniteResponseLawProspectiveRootEvaluation,
                FiniteResponseLawRootInferenceOperands,
                FiniteResponseLawEvaluationCohort,
                ScientificAdjudicationRecord,
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        if registry != self.registry:
            raise ValueError("finite response-law reveal changes its adjudication registry")
        return ScientificAdjudicationOutputContract(
            self.manifest.capability_key,
            self.manifest.capability_version,
            f"{ADJUDICATE}.product",
            ScientificAdjudicationRecord.SCHEMA,
        )
