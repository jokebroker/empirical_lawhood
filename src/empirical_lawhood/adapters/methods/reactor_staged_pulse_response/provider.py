"Declared staged-pulse scientific tasks using shared qualification and control owners."

from dataclasses import dataclass
from typing import Any

from empirical_lawhood.adapters.composition.phase_inputs import (
    artifact,
    config_input,
    dependency,
    result,
    upstream,
)
from empirical_lawhood.adapters.composition.record_provider import RecordCampaignProvider
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot, measure_root as old_measure
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierAssay, FrontierPreparation
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.provider import source_input
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import AdjudicationEvaluability, ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .calibration import calibrate
from .comparison import comparison
from .config import CALIBRATION_ROOTS, ClassicalStage, retained_key
from .discovery import ClassicalNomination
from .law_terminal import ClassicalLaws, qualify_laws
from .measurement import ClassicalMeasuredRoot, measure_root
from .nomination import first_compatibility, menu_nomination, sequence_nomination
from .prospective_cohort import ReactorStagedPulseResponseProspectiveCohort, reduce_cohort
from .prospective_lock import ClassicalFrozenRoot, freeze_root
from .prospective_plan import ReactorStagedPulseResponseProspectivePlan, build_prospective_plan
from .prospective_reveal import ClassicalRevealedRoot, reveal_root
from .prospective_seal import ClassicalSealedRoot, seal_native_root
from .phase_records import ClassicalFirstScreen, ClassicalInducedBundle
from .prediction import ClassicalPredictionSeal, seal_predictions
from .qualification import ClassicalQualification, qualification_operands
from .records import ClassicalAssay, ClassicalPreparation
from .science import stage_system
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS


@dataclass(frozen=True)
class ClassicalMethodRunner:
    stage: ClassicalStage
    native: ClassicalNativeConfig
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    publisher: CandidatePayloadPublisher
    reader: CandidatePayloadReader
    control: Any
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.stage)
            return result(context, self._execute(context))

    def _execute(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        task = context.task_id

        def dep(kind: type[Any], producer: str) -> Any:
            return dependency(context, self.custody, kind, producer)[0]

        def old(key: str, kind: type[Any]) -> Any:
            return upstream(context, self.stage, key, kind)

        if task == "classical.calibration":
            return calibrate(
                old("historical-b", FrontierDevelopment),
                tuple(
                    old(retained_key(r, "measurement"), FrontierMeasuredRoot)
                    for r in CALIBRATION_ROOTS
                ),
                tuple(
                    old(retained_key(r, "prediction"), FrontierPredictionSeal)
                    for r in CALIBRATION_ROOTS
                ),
            )
        if task == "classical.first-screen":
            parent = old("historical-b", FrontierDevelopment)
            preparations = tuple(
                old(retained_key(r, "preparation"), FrontierPreparation)
                for r in self.stage.root_ids
            )
            measurements = tuple(
                old(retained_key(r, "measurement"), FrontierMeasuredRoot)
                for r in self.stage.root_ids
            )
            return (
                ClassicalFirstScreen(
                    ObjectIdentity.from_record(parent.record_id, parent),
                    first_compatibility(parent, preparations, measurements),
                    tuple(ObjectIdentity.from_record(p.record_id, p) for p in preparations),
                    tuple(
                        ObjectIdentity.from_record(f"{m.root}.measurement", m) for m in measurements
                    ),
                ),
            )
        if task.startswith("classical.predictions."):
            root = task.removeprefix("classical.predictions.")
            return (
                seal_predictions(
                    source_input(context).batch,
                    dep(ClassicalPreparation, f"classical.prepare.{root}"),
                    self.native.prepared_domain,
                    None if self.stage.role == "NOMINATION" else old("nomination", ClassicalNomination),
                ),
            )
        if task.startswith("classical.measure."):
            root = task.removeprefix("classical.measure.")
            retained = (
                old_measure(
                    old(retained_key(root, "preparation"), FrontierPreparation),
                    old(retained_key(root, "assay"), FrontierAssay),
                )
                if self.stage.stage == "expanded-menu-comparison-NOMINATION"
                else None
            )
            return (
                measure_root(
                    dep(ClassicalPreparation, f"classical.prepare.{root}"),
                    dep(ClassicalAssay, f"classical.assay.{root}"),
                    retained=retained,
                ),
            )
        if task in ("classical.nomination", "classical.qualification"):
            measurements = tuple(
                dep(ClassicalMeasuredRoot, f"classical.measure.{r}") for r in self.stage.root_ids
            )
            seals = tuple(
                dep(ClassicalPredictionSeal, f"classical.predictions.{r}")
                for r in self.stage.root_ids
            )
            induced = (
                tuple(
                    s
                    for r in self.stage.root_ids
                    for s in dep(ClassicalInducedBundle, f"classical.assay.{r}").seals
                )
                if self.stage.block == "staged-sequence-comparison"
                else ()
            )
            for r in self.stage.root_ids if self.stage.block == "staged-sequence-comparison" else ():
                assay = dep(ClassicalAssay, f"classical.assay.{r}")
                saved = next((s for s in induced if s.context.root == r), None)
                if assay.induced_seal != (
                    None if saved is None else ObjectIdentity.from_record(saved.record_id, saved)
                ) or assay.induced != (() if saved is None else (saved.context,)):
                    raise ValueError("measurement replaced its immutable pre-second-response seal")
            if task == "classical.qualification":
                return (
                    qualification_operands(
                        old("nomination", ClassicalNomination), measurements, seals, induced
                    ),
                )
            if self.stage.block == "expanded-menu-comparison":
                return (
                    menu_nomination(
                        old("historical-b", FrontierDevelopment), measurements, seals
                    ),
                )
            screen = dep(ClassicalFirstScreen, "classical.first-screen")
            return (
                sequence_nomination(
                    screen.bound,
                    measurements,
                    seals,
                    induced,
                    historical_parent=screen.historical_parent,
                ),
            )
        if task == "classical.laws":
            qualification = dep(ClassicalQualification, "classical.qualification")
            custody = tuple(
                dependency(context, self.custody, ClassicalAssay, f"classical.assay.{r}")
                for r in self.stage.root_ids
            )
            design = self.stage.design
            design_artifact = ArtifactIdentity(
                design.config_id,
                "classical-frozen-design",
                design.SCHEMA,
                design.fingerprint(),
                "application/json",
                len(design.canonical_bytes()),
            )
            return (
                qualify_laws(
                    native=self.native,
                    qualification=qualification,
                    manifests=tuple(c[1] for c in custody),
                    receipts=tuple(c[2] for c in custody),
                    capability=CAPABILITY,
                    publisher=self.publisher,
                    reader=self.reader,
                    design_artifact=design_artifact,
                ),
            )
        if task == "classical.prospective-evaluation-plan":
            return (
                build_prospective_plan(
                    old("laws", ClassicalLaws),
                    tuple(
                        dep(ClassicalPreparation, f"classical.prepare.{r}")
                        for r in self.stage.root_ids
                    ),
                    tuple(
                        dep(ClassicalPredictionSeal, f"classical.predictions.{r}")
                        for r in self.stage.root_ids
                    ),
                    source_input(context).batch,
                ),
            )
        if task.startswith("classical.freeze."):
            root = task.removeprefix("classical.freeze.")
            prep, manifest, receipt = dependency(
                context, self.custody, ClassicalPreparation, f"classical.prepare.{root}"
            )
            return (
                freeze_root(
                    laws=old("laws", ClassicalLaws),
                    preparation=prep,
                    seal=dep(ClassicalPredictionSeal, f"classical.predictions.{root}"),
                    plan=dep(ReactorStagedPulseResponseProspectivePlan, "classical.prospective-evaluation-plan"),
                    source=source_input(context).batch,
                    publisher=self.control.open_control_store(root),
                    reader=self.reader,
                    control=self.control,
                    causal_receipt=receipt,
                    causal_artifact=artifact(manifest),
                ),
            )
        if task.startswith(("classical.prospective-evaluation-seal.", "classical.prospective-evaluation-reveal.")):
            kind, root = task.split(".", 2)[1:]
            native, manifest, receipt = dependency(
                context, self.custody, ClassicalNativeRoot, f"classical.action.{root}"
            )
            plan = dep(ReactorStagedPulseResponseProspectivePlan, "classical.prospective-evaluation-plan")
            if kind == "prospective-evaluation-seal":
                return (
                    seal_native_root(
                        native=native,
                        frozen=dep(ClassicalFrozenRoot, f"classical.freeze.{root}"),
                        plan=plan,
                        preparation=dep(ClassicalPreparation, f"classical.prepare.{root}"),
                        receipt=receipt,
                        artifact=artifact(manifest),
                        store=self.control.open_prepared_store(),
                        occurred_at_utc=self.control.event_clock(root, "seal"),
                    ),
                )
            if context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
                raise ValueError("Controller-use reveal lacks its separately authorized outcome boundary")
            return (
                reveal_root(
                    sealed=dep(ClassicalSealedRoot, f"classical.prospective-evaluation-seal.{root}"),
                    native=native,
                    plan=plan,
                    reveal_authority=self.control.reveal,
                    publisher=self.control.open_control_store(root),
                ),
            )
        if task in ("classical.prospective-evaluation-cohort", "classical.comparison"):
            revealed = tuple(
                dep(ClassicalRevealedRoot, f"classical.prospective-evaluation-reveal.{r}")
                for r in self.stage.root_ids
            )
            if task == "classical.prospective-evaluation-cohort":
                return (reduce_cohort(dep(ReactorStagedPulseResponseProspectivePlan, "classical.prospective-evaluation-plan"), revealed),)
            return (comparison(dep(ReactorStagedPulseResponseProspectiveCohort, "classical.prospective-evaluation-cohort"), revealed),)
        if task != "classical.adjudication":
            raise ValueError("unregistered classical staged-pulse method task")
        if self.stage.role == "NOMINATION":
            nomination = dep(
                ClassicalNomination,
                "classical.calibration" if self.stage.block == "base-menu-comparison" else "classical.nomination",
            )
            status = ScientificStatus.UNEVALUABLE
            reasons = (
                "EXPLORATORY_NOMINATION_NOT_EMPIRICAL_QUALIFICATION",
                f"NOMINATED_RELATIONS_{sum(r.entered for r in nomination.recipes)}",
            )
        else:
            if self.stage.role == "QUALIFICATION":
                laws = dep(ClassicalLaws, "classical.laws")
                statuses = tuple(r.qualification.scientific_status for r in laws.rows)
                reasons = (
                    f"QUALIFIED_RELATIONS_{len(laws.qualified)}",
                    "INDIVIDUAL_RELATION_RESULTS_GOVERN_ADMISSION",
                )
            else:
                cohort = dep(ReactorStagedPulseResponseProspectiveCohort, "classical.prospective-evaluation-cohort")
                statuses = tuple(p.status for p in cohort.policies)
                reasons = (
                    f"SUPPORTED_CONTROLLER_USE_POLICIES_{sum(s is ScientificStatus.SUPPORTED for s in statuses)}",
                    "METHOD_CONTRIBUTION_HAS_SEPARATE_COMPARISON_ENDPOINTS",
                )
            status = (
                ScientificStatus.SUPPORTED
                if ScientificStatus.SUPPORTED in statuses
                else ScientificStatus.NOT_SUPPORTED
                if ScientificStatus.NOT_SUPPORTED in statuses
                else ScientificStatus.UNEVALUABLE
            )
        a = context.scientific_adjudication_context
        system = stage_system(self.stage)
        if (
            a is None
            or a.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or a.evidence_world_id != system.world.world_id
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("adjudication substitutes its relation or authorized reveal boundary")
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
                tuple(sorted(reasons)),
            ),
        )


class ClassicalMethodProvider(RecordCampaignProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        runner: ClassicalMethodRunner,
        inputs: tuple[ExternalInputPayload, ...],
    ) -> None:
        super().__init__(
            registry,
            CAPABILITY,
            runner,
            runner.stage,
            runner.stage.config_id,
            inputs,
            OUTPUT_RECORDS,
            adjudication=True,
        )
