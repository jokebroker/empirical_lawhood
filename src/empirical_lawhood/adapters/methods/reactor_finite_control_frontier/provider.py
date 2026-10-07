"""Phase-bound scientific tasks, delegating qualification and control to shared owners."""

from empirical_lawhood.adapters.composition.record_provider import RecordCampaignProvider
from dataclasses import dataclass
from typing import Any
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.provider import source_input
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult
from empirical_lawhood.runtime.providers import (
    ExternalInputPayload,
)
from .phase import FrontierPhase
from .records import FrontierPreparation, FrontierAssay
from .selection import FrontierPredictionSeal, seal_predictions, primary_requests
from .measurement import FrontierMeasuredRoot, measure_root
from .discovery import FrontierDevelopment, discover
from .qualification import FrontierQualification, qualification_operands
from .law_terminal import FrontierLaws, qualify_laws
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan, build_prospective_plan
from .prospective_lock import FrontierFrozenRoot, freeze_root
from .prospective_seal import FrontierSealedRoot, seal_native_root
from .prospective_reveal import FrontierRevealedRoot, reveal_root
from .prospective_cohort import ArchivedReactorFrontierCohort, reduce_prospective
from .comparison import comparison
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS
from .provider_common import artifact, config_input, dependency, result, upstream
from .science import frontier_system


@dataclass(frozen=True)
class FrontierMethodRunner:
    phase: FrontierPhase
    native: FrontierNativeConfig
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    publisher: CandidatePayloadPublisher
    reader: CandidatePayloadReader
    control: Any
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.phase)
            return result(context, self._execute(context))

    def _execute(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        task = context.task_id

        def dep(kind: type[Any], name: str) -> Any:
            return dependency(context, self.custody, kind, name)[0]

        if task.startswith("frontier.predictions."):
            root = task.removeprefix("frontier.predictions.")
            prep = dep(FrontierPreparation, f"frontier.prepare.{root}")
            development = (
                None
                if self.phase.phase == "B"
                else upstream(context, self.phase, "development", FrontierDevelopment)
            )
            laws = (
                upstream(context, self.phase, "laws", FrontierLaws)
                if self.phase.phase == "D"
                else None
            )
            qualified = (
                tuple(b.coordinate.coordinate_id for b in development.bounds if b.nominated)
                if development is not None and laws is None
                else ()
                if laws is None
                else laws.qualified
            )
            return (
                seal_predictions(
                    source=source_input(context).batch,
                    preparation=prep,
                    domain=self.native.prepared_domain,
                    development=development,
                    qualified=qualified,
                    parent_laws=None
                    if laws is None
                    else ObjectIdentity.from_record(laws.record_id, laws),
                    requests=()
                    if laws is None or development is None
                    else primary_requests(development, laws.qualified),
                ),
            )
        if task.startswith("frontier.measure."):
            root = task.removeprefix("frontier.measure.")
            prep = dep(FrontierPreparation, f"frontier.prepare.{root}")
            assay = dep(FrontierAssay, f"frontier.assay.{root}")
            return (measure_root(prep, assay),)
        if task in ("frontier.development", "frontier.qualification", "frontier.comparison"):
            if self.phase.phase == "D":
                measured = tuple(
                    dep(FrontierSealedRoot, f"frontier.prospective-evaluation-seal.{r}").measured.chart
                    for r in self.phase.roots
                )
            else:
                measured = tuple(
                    dep(FrontierMeasuredRoot, f"frontier.measure.{r}") for r in self.phase.roots
                )
            if task == "frontier.qualification":
                return (
                    qualification_operands(
                        upstream(context, self.phase, "development", FrontierDevelopment),
                        measured,
                    ),
                )
            seals = tuple(
                dep(FrontierPredictionSeal, f"frontier.predictions.{r}") for r in self.phase.roots
            )
            if task == "frontier.development":
                return (discover(measured, tuple(f for s in seals for f in s.forecasts), seals),)
            return (
                comparison(
                    measured,
                    seals,
                    role="qualification" if self.phase.phase == "C" else "prospective",
                ),
            )
        if task == "frontier.laws":
            qualification = dep(FrontierQualification, "frontier.qualification")
            native_custody = tuple(
                dependency(context, self.custody, FrontierAssay, f"frontier.assay.{r}")
                for r in self.phase.roots
            )
            design = self.phase.design
            design_artifact = ArtifactIdentity(
                design.config_id,
                "frontier-frozen-design",
                design.SCHEMA,
                design.fingerprint(),
                "application/json",
                len(design.canonical_bytes()),
            )
            return (
                qualify_laws(
                    native=self.native,
                    qualification=qualification,
                    manifests=tuple(r[1] for r in native_custody),
                    receipts=tuple(r[2] for r in native_custody),
                    capability=CAPABILITY,
                    publisher=self.publisher,
                    reader=self.reader,
                    design_artifact=design_artifact,
                ),
            )
        if task == "frontier.prospective-evaluation-plan":
            laws = upstream(context, self.phase, "laws", FrontierLaws)
            development = upstream(context, self.phase, "development", FrontierDevelopment)
            return (
                build_prospective_plan(
                    laws=laws,
                    preparations=tuple(
                        dep(FrontierPreparation, f"frontier.prepare.{r}")
                        for r in self.phase.roots
                    ),
                    seals=tuple(
                        dep(FrontierPredictionSeal, f"frontier.predictions.{r}")
                        for r in self.phase.roots
                    ),
                    requests=primary_requests(development, laws.qualified),
                ),
            )
        if task.startswith("frontier.freeze."):
            root = task.removeprefix("frontier.freeze.")
            prep, manifest, receipt = dependency(
                context, self.custody, FrontierPreparation, f"frontier.prepare.{root}"
            )
            return (
                freeze_root(
                    laws=upstream(context, self.phase, "laws", FrontierLaws),
                    preparation=prep,
                    seal=dep(FrontierPredictionSeal, f"frontier.predictions.{root}"),
                    plan=dep(ReactorFiniteControlFrontierProspectivePlan, "frontier.prospective-evaluation-plan"),
                    publisher=self.control.open_control_store(root),
                    reader=self.reader,
                    control=self.control,
                    causal_receipt=receipt,
                    causal_artifact=artifact(manifest),
                ),
            )
        if task.startswith(("frontier.prospective-evaluation-seal.", "frontier.prospective-evaluation-reveal.")):
            kind, root = task.split(".", 2)[1:]
            native, manifest, receipt = dependency(
                context, self.custody, FrontierNativeRoot, f"frontier.action.{root}"
            )
            frozen = dep(FrontierFrozenRoot, f"frontier.freeze.{root}")
            plan = dep(ReactorFiniteControlFrontierProspectivePlan, "frontier.prospective-evaluation-plan")
            if kind == "prospective-evaluation-seal":
                return (
                    seal_native_root(
                        native=native,
                        frozen=frozen,
                        plan=plan,
                        preparation=dep(FrontierPreparation, f"frontier.prepare.{root}"),
                        receipt=receipt,
                        artifact=artifact(manifest),
                        store=self.control.open_prepared_store(),
                        occurred_at_utc=self.control.event_clock(root, "seal"),
                    ),
                )
            if context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
                raise ValueError("Controller-use reveal requires the separate issued reveal boundary")
            return (
                reveal_root(
                    sealed=dep(FrontierSealedRoot, f"frontier.prospective-evaluation-seal.{root}"),
                    native=native,
                    frozen=frozen,
                    laws=upstream(context, self.phase, "laws", FrontierLaws),
                    plan=plan,
                    reveal_authority=self.control.reveal,
                ),
            )
        if task == "frontier.prospective-evaluation-cohort":
            return (
                reduce_prospective(
                    dep(ReactorFiniteControlFrontierProspectivePlan, "frontier.prospective-evaluation-plan"),
                    tuple(
                        dep(FrontierRevealedRoot, f"frontier.prospective-evaluation-reveal.{r}")
                        for r in self.phase.roots
                    ),
                ),
            )
        if task == "frontier.adjudication":
            if self.phase.phase == "B":
                development = dep(FrontierDevelopment, "frontier.development")
                status = ScientificStatus.UNEVALUABLE
                reasons = (
                    "EXPLORATORY_NOMINATION_NOT_EMPIRICAL_QUALIFICATION",
                    f"NOMINATED_COORDINATES_{sum(b.nominated for b in development.bounds)}",
                )
            elif self.phase.phase == "C":
                laws = dep(FrontierLaws, "frontier.laws")
                statuses = tuple(r.qualification.scientific_status for r in laws.rows)
                status = (
                    ScientificStatus.SUPPORTED
                    if laws.qualified
                    else ScientificStatus.NOT_SUPPORTED
                    if all(s is ScientificStatus.NOT_SUPPORTED for s in statuses)
                    else ScientificStatus.UNEVALUABLE
                )
                reasons = (
                    f"SUPPORTED_COORDINATES_{len(laws.qualified)}",
                    "PER_COORDINATE_CENSUS_CONTROLS_LOCAL_USE",
                )
            else:
                cohort = dep(ArchivedReactorFrontierCohort, "frontier.prospective-evaluation-cohort")
                supported = sum(
                    p.scientific_status is ScientificStatus.SUPPORTED for p in cohort.pairs
                )
                status = (
                    ScientificStatus.SUPPORTED
                    if supported
                    else ScientificStatus.NOT_SUPPORTED
                    if any(
                        p.scientific_status is ScientificStatus.NOT_SUPPORTED for p in cohort.pairs
                    )
                    else ScientificStatus.UNEVALUABLE
                )
                reasons = (
                    f"SUPPORTED_CONTROLLER_USE_PAIRS_{supported}",
                    "COMPARATOR_CONTRIBUTION_HAS_SEPARATE_ENDPOINTS",
                )
            a = context.scientific_adjudication_context
            system = frontier_system()
            if (
                a is None
                or a.relation
                != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
                or a.evidence_world_id != system.world.world_id
                or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            ):
                raise ValueError("frontier adjudication lost its relation/reveal boundary")
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
                    tuple(sorted(set(reasons))),
                ),
            )
        raise ValueError("unregistered frontier method task")


class FrontierMethodProvider(RecordCampaignProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        runner: FrontierMethodRunner,
        inputs: tuple[ExternalInputPayload, ...],
    ) -> None:
        super().__init__(
            registry,
            CAPABILITY,
            runner,
            runner.phase,
            runner.phase.config_id,
            inputs,
            OUTPUT_RECORDS,
            adjudication=True,
        )
