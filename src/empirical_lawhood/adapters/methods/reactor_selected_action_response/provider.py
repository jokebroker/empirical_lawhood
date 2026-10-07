"""Closed scientific task bindings; each claim delegates to the existing owner."""

from dataclasses import dataclass
from typing import Protocol

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.prospective_records import ClassicalNativeRoot
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationRecord,
    ScientificAdjudicationOutputContract,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult, TaskRunner
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from .causal import seal_decision
from .config import ROOTS, ClassicalDesign, assignment
from .control_owner import ControlRecordPublisher
from .control_prospective_plan import ReactorSelectedActionResponseProspectivePlan, build_prepared_prospective_plan
from .control_prospective_closeout import ClassicalSealResult, seal_native_root
from .control_prospective_reveal import ClassicalRevealResult, reveal_native_root
from .control_prospective_cohort import ReactorSelectedActionResponseProspectiveCohort, reduce_prospective
from .control_services import implementation_payloads
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS
from .law_terminal import ClassicalLaw, RESULT_ID, qualify_classical_law
from .measurement import measure_root, qualify_operands
from .provider_common import artifact, config_input, contracts, dependency, external_config, result, verify_registry
from .records import ClassicalCausal, ClassicalDecision, ClassicalPrivate, ClassicalAssay, ClassicalQualification
from .science import classical_system


class ClassicalControlCustody(Protocol):
    @property
    def authority(self) -> ObjectIdentity: ...
    @property
    def approval(self) -> ObjectIdentity: ...
    @property
    def reveal(self) -> ObjectIdentity: ...
    @property
    def resources(self) -> ObjectIdentity: ...
    @property
    def issued_study(self) -> ObjectIdentity: ...

    def open_control_store(self, root: str) -> ControlRecordPublisher: ...
    def open_prepared_store(self) -> DurablePreparedExecutionEventStore: ...
    def event_clock(self, root: str, stage: str) -> str: ...


@dataclass(frozen=True)
class ClassicalMethodRunner:
    design: ClassicalDesign
    native: ClassicalNativeConfig
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    publisher: CandidatePayloadPublisher
    reader: CandidatePayloadReader
    control: ClassicalControlCustody
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.design)
            records = self._execute(context)
            return result(context, records)

    def _execute(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        task = context.task_id
        if task.startswith("classical.decision."):
            root = task.removeprefix("classical.decision.")
            role, _ = assignment(root)
            causal, _, _ = dependency(
                context, self.custody, ClassicalCausal, f"classical.prepare.{root}"
            )
            law_id = None
            if role == "prospective":
                law, _, _ = dependency(context, self.custody, ClassicalLaw, "classical.law")
                if law.qualification.scientific_status is ScientificStatus.SUPPORTED:
                    law_id = ObjectIdentity.from_record(RESULT_ID, law)
            return (seal_decision(self.native, causal, self.control.approval, law_id),)
        if task == "classical.qualification":
            assays, scores = [], []
            for root, role, _, _ in ROOTS:
                if role != "qualification":
                    continue
                causal, _, _ = dependency(
                    context, self.custody, ClassicalCausal, f"classical.prepare.{root}"
                )
                decision, _, _ = dependency(
                    context, self.custody, ClassicalDecision, f"classical.decision.{root}"
                )
                assay, _, _ = dependency(
                    context, self.custody, ClassicalAssay, f"classical.assay.{root}"
                )
                assays.append(assay)
                scores.append(measure_root(causal, decision, assay))
            return (qualify_operands(tuple(assays), tuple(scores)),)
        if task == "classical.law":
            qualification, _, _ = dependency(
                context, self.custody, ClassicalQualification, "classical.qualification"
            )
            custody = tuple(
                dependency(context, self.custody, ClassicalAssay, f"classical.assay.{root}")
                for root, role, _, _ in ROOTS
                if role == "qualification"
            )
            raw = self.design.canonical_bytes()
            design_artifact = ArtifactIdentity(
                self.design.config_id,
                "classical-frozen-design",
                self.design.SCHEMA,
                self.design.fingerprint(),
                "application/json",
                len(raw),
            )
            return (
                qualify_classical_law(
                    self.design,
                    self.native,
                    qualification,
                    tuple(row[1] for row in custody),
                    tuple(row[2] for row in custody),
                    CAPABILITY,
                    self.publisher,
                    self.reader,
                    design_artifact,
                ),
            )
        if task == "classical.prospective-evaluation-plan":
            law, _, _ = dependency(context, self.custody, ClassicalLaw, "classical.law")
            prepared = tuple(
                (
                    dependency(
                        context, self.custody, ClassicalCausal, f"classical.prepare.{root}"
                    )[0],
                    dependency(
                        context, self.custody, ClassicalDecision, f"classical.decision.{root}"
                    )[0],
                )
                for root, role, _, _ in ROOTS
                if role == "prospective"
            )
            binding = implementation_payloads()[0][0]
            return (
                build_prepared_prospective_plan(
                    design=self.design,
                    report=law,
                    prepared=prepared,
                    producer=ObjectIdentity.from_record(binding.binding_id, binding),
                    resource=self.control.resources,
                    authority=self.control.approval,
                ),
            )
        if task.startswith(("classical.prospective-evaluation-seal.", "classical.prospective-evaluation-reveal.")):
            kind, root = task.split(".", 2)[1:]
            native, manifest, receipt = dependency(
                context, self.custody, ClassicalNativeRoot, f"classical.action.{root}"
            )
            plan, _, _ = dependency(context, self.custody, ReactorSelectedActionResponseProspectivePlan, "classical.prospective-evaluation-plan")
            causal, _, _ = dependency(
                context, self.custody, ClassicalCausal, f"classical.prepare.{root}"
            )
            private, _, _ = dependency(
                context, self.custody, ClassicalPrivate, f"classical.prepare.{root}"
            )
            decision, _, _ = dependency(
                context, self.custody, ClassicalDecision, f"classical.decision.{root}"
            )
            if kind == "prospective-evaluation-seal":
                sealed = seal_native_root(
                    native=native,
                    plan_bundle=plan,
                    assignment=decision,
                    causal=causal,
                    private=private,
                    receipt=receipt,
                    artifact=artifact(manifest),
                    store=self.control.open_prepared_store(),
                    occurred_at_utc=self.control.event_clock(root, "seal"),
                )
                return (
                    ClassicalSealResult(
                        root,
                        ObjectIdentity.from_record(f"{root}.native-action", native),
                        ObjectIdentity.from_record("classical.prospective-evaluation-plan", plan),
                        sealed,
                        ()
                        if sealed is not None
                        else native.reasons or ("LOCAL_LAW_OR_PREPARATION_NONENTRY",),
                    ),
                )
            if context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
                raise ValueError("classical controller-use reveal lacks the separate evaluator boundary")
            sealed_result, _, _ = dependency(
                context, self.custody, ClassicalSealResult, f"classical.prospective-evaluation-seal.{root}"
            )
            return (
                reveal_native_root(
                    sealed=sealed_result,
                    native=native,
                    plan_bundle=plan,
                    assignment=decision,
                    causal=causal,
                    private=private,
                    reveal_authority=self.control.reveal,
                ),
            )
        if task == "classical.prospective-evaluation-cohort":
            plan, _, _ = dependency(context, self.custody, ReactorSelectedActionResponseProspectivePlan, "classical.prospective-evaluation-plan")
            roots = tuple(
                dependency(
                    context, self.custody, ClassicalRevealResult, f"classical.prospective-evaluation-reveal.{root}"
                )[0]
                for root, role, _, _ in ROOTS
                if role == "prospective"
            )
            return (reduce_prospective(plan, roots),)
        if task == "classical.adjudication":
            law, _, _ = dependency(context, self.custody, ClassicalLaw, "classical.law")
            cohort, _, _ = dependency(
                context, self.custody, ReactorSelectedActionResponseProspectiveCohort, "classical.prospective-evaluation-cohort"
            )
            a = context.scientific_adjudication_context
            system = classical_system()
            if (
                a is None
                or a.relation
                != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
                or a.evidence_world_id != system.world.world_id
                or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            ):
                raise ValueError(
                    "classical adjudication lost its declared relation or reveal boundary"
                )
            status = (
                cohort.scientific_status
                if law.qualification.response_law is not None
                else law.qualification.scientific_status
            )
            return (
                ScientificAdjudicationRecord(
                    f"adjudication.{context.run_id}.{task}",
                    context.run_id,
                    task,
                    a.execution_plan,
                    context.input_materialization_ids,
                    tuple(
                        sorted(
                            p.logical_artifact_id
                            for p in context.output_ports
                            if p.logical_artifact_id is not None
                        )
                    ),
                    context.dependency_receipt_ids,
                    a.evidence_world_id,
                    a.evidence_world_kind,
                    a.relation,
                    a.independent_unit_id,
                    a.information_cutoffs,
                    a.visibility_ceiling,
                    a.outcome_access,
                    AdjudicationEvaluability.UNEVALUABLE
                    if status is ScientificStatus.UNEVALUABLE
                    else AdjudicationEvaluability.EVALUABLE,
                    status,
                    AdmissionStatus.ADMITTED
                    if status is ScientificStatus.SUPPORTED
                    else AdmissionStatus.UNEVALUABLE
                    if status is ScientificStatus.UNEVALUABLE
                    else AdmissionStatus.EMPTY,
                    (f"CLASSICAL_LOCAL_CONTROLLER_USE_{status.value}",)
                    if law.qualification.response_law is not None
                    else ("CLASSICAL_LOCAL_LAW_PREREQUISITE_NOT_SUPPORTED_CONTROLLER_USE_NONENTRY",),
                ),
            )
        raise ValueError("unregistered classical method task")


class ClassicalMethodProvider(CampaignRuntimeProvider):
    def __init__(self, registry: CapabilityRegistry, runner: ClassicalMethodRunner) -> None:
        self.registry_sha256 = verify_registry(registry, CAPABILITY)
        self.capability_count, self.runner = 1, runner

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("classical provider registry/source changed")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("classical provider plan/source changed")
        return external_config(plan, CAPABILITY, self.runner.design)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return contracts(CAPABILITY, OUTPUT_RECORDS)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self.runners(registry)
        if execution_plan is None:
            raise ValueError("classical adjudication requires its compiled output locator")
        outputs = [
            o
            for task in execution_plan.tasks
            for o in task.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError("classical adjudication output census changed")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )
