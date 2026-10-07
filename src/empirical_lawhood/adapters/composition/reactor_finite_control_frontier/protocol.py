"""Three separate immutable phases on the public candidate graph/compiler route."""

from dataclasses import replace
from empirical_lawhood.adapters.composition.experiment_authoring import study_template_from_protocol
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierPrivate, FrontierAssay
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import FrontierQualification
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.law_terminal import FrontierLaws
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_lock import FrontierFrozenRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_seal import FrontierSealedRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_reveal import FrontierRevealedRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_cohort import ArchivedReactorFrontierCohort
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.comparison import FrontierComparison
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref, protocol_outputs
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import StudyTemplate, CandidateGraphExternalInput, CandidateGraphEdge, ContentIdentityPolicy, ScientificInputRole
from empirical_lawhood.runtime.capabilities import CapabilityRegistry, CapabilityPermission
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)


def protocol(phase: FrontierPhase, native: FrontierNativeConfig) -> ProtocolTemplate:
    if phase.design != native.design:
        raise ValueError("phase/native design mismatch")
    steps = []
    configs = {
        c.capability_key: capability_config_ref(v, ObjectIdentity.from_record(v.config_id, v), c)
        for c, v in ((METHOD, phase), (SOURCE, native))
    }

    def add(
        task: str,
        dependencies: tuple[str, ...],
        records: tuple[tuple[str, str], ...],
        *,
        source: bool = False,
        reveal: bool = False,
        terminal: bool = False,
        cpu: int = 30,
        output_mb: int = 256,
    ) -> None:
        cap = SOURCE if source else METHOD
        prepare = task.startswith("frontier.prepare.")
        stage = (
            ScientificStage.PREPARE
            if prepare
            else ScientificStage.ACQUIRE
            if source
            else ScientificStage.EVALUATE
            if reveal
            else ScientificStage.FREEZE
        )
        steps.append(
            ProtocolStepTemplate(
                task,
                stage,
                cap.capability_key,
                cap.capability_version,
                configs[cap.capability_key],
                tuple(sorted(set(dependencies))),
                protocol_outputs(records),
                cap.permissions
                if source or reveal
                else tuple(
                    p for p in cap.permissions if p is not CapabilityPermission.REVEAL_OUTCOMES
                ),
                OutcomeAccess.EVALUATOR_REVEAL if reveal else OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.OUTCOME_VISIBLE if terminal else VisibilityCeiling.PROSPECTIVE,
                ResourceBudget(
                    1,
                    8 * 1024**3,
                    0,
                    cpu,
                    8 * 1024**3
                    if task
                    in (
                        "frontier.development",
                        "frontier.qualification",
                        "frontier.laws",
                        "frontier.prospective-evaluation-plan",
                        "frontier.prospective-evaluation-cohort",
                        "frontier.comparison",
                    )
                    else 1024**3,
                    output_mb * 1024**2,
                ),
                (),
                BarrierKind.REVEAL
                if reveal
                else BarrierKind.NONE
                if prepare
                else BarrierKind.FREEZE,
                1,
                (f"{phase.config_id}.single-terminal",)
                if terminal
                else (f"{phase.config_id}.{task}",),
            )
        )

    for root in phase.roots:
        prep, predictions = f"frontier.prepare.{root}", f"frontier.predictions.{root}"
        add(
            prep,
            (),
            (("causal", FrontierPreparation.SCHEMA), ("private", FrontierPrivate.SCHEMA)),
            source=True,
            cpu=30 if phase.phase == "B" else 60,
        )
        add(predictions, (prep,), (("science", FrontierPredictionSeal.SCHEMA),), cpu=120)
        if phase.phase in ("B", "C"):
            assay = f"frontier.assay.{root}"
            add(
                assay,
                (prep, predictions),
                (("assay", FrontierAssay.SCHEMA),),
                source=True,
                cpu=1200 if phase.phase == "B" else 900,
            )
            add(
                f"frontier.measure.{root}",
                (prep, assay),
                (("science", FrontierMeasuredRoot.SCHEMA),),
            )
        else:
            frozen, action, seal = (
                f"frontier.freeze.{root}",
                f"frontier.action.{root}",
                f"frontier.prospective-evaluation-seal.{root}",
            )
            add(
                frozen,
                (prep, predictions, "frontier.prospective-evaluation-plan"),
                (("science", FrontierFrozenRoot.SCHEMA),),
                cpu=1000,
            )
            add(
                action,
                (prep, frozen, "frontier.prospective-evaluation-plan"),
                (("assay", FrontierNativeRoot.SCHEMA),),
                source=True,
                cpu=1000,
            )
            add(
                seal,
                (prep, frozen, action, "frontier.prospective-evaluation-plan"),
                (("science", FrontierSealedRoot.SCHEMA),),
                cpu=400,
            )
            add(
                f"frontier.prospective-evaluation-reveal.{root}",
                (frozen, action, seal, "frontier.prospective-evaluation-plan"),
                (("science", FrontierRevealedRoot.SCHEMA),),
                reveal=True,
                cpu=200,
            )
    measurements = tuple(f"frontier.measure.{r}" for r in phase.roots)
    all_predictions = tuple(f"frontier.predictions.{r}" for r in phase.roots)
    terminal: tuple[str, ...]
    if phase.phase == "B":
        add(
            "frontier.development",
            (*measurements, *all_predictions),
            (("science", FrontierDevelopment.SCHEMA),),
            cpu=900,
        )
        terminal = ("frontier.development",)
    elif phase.phase == "C":
        add(
            "frontier.qualification",
            measurements,
            (("science", FrontierQualification.SCHEMA),),
            cpu=900,
        )
        add(
            "frontier.laws",
            ("frontier.qualification", *(f"frontier.assay.{r}" for r in phase.roots)),
            (("science", FrontierLaws.SCHEMA),),
            cpu=7200,
        )
        add(
            "frontier.comparison",
            (*measurements, *all_predictions),
            (("science", FrontierComparison.SCHEMA),),
            reveal=True,
            cpu=900,
        )
        terminal = ("frontier.laws", "frontier.comparison")
    else:
        add(
            "frontier.prospective-evaluation-plan",
            (*all_predictions, *(f"frontier.prepare.{r}" for r in phase.roots)),
            (("science", ReactorFiniteControlFrontierProspectivePlan.SCHEMA),),
            cpu=1800,
        )
        add(
            "frontier.prospective-evaluation-cohort",
            ("frontier.prospective-evaluation-plan", *(f"frontier.prospective-evaluation-reveal.{r}" for r in phase.roots)),
            (("science", ArchivedReactorFrontierCohort.SCHEMA),),
            reveal=True,
            cpu=1800,
        )
        add(
            "frontier.comparison",
            (*all_predictions, *(f"frontier.prospective-evaluation-seal.{r}" for r in phase.roots)),
            (("science", FrontierComparison.SCHEMA),),
            reveal=True,
            cpu=900,
        )
        terminal = ("frontier.prospective-evaluation-cohort", "frontier.comparison")
    add(
        "frontier.adjudication",
        terminal,
        (("science", ScientificAdjudicationRecord.SCHEMA),),
        reveal=True,
        terminal=True,
        cpu=300,
    )
    if sum(s.resource_budget.wall_time_seconds for s in steps) > phase.cpu_seconds:
        raise ValueError("phase task reservations exceed the declared CPU slice")
    return ProtocolTemplate(
        f"{phase.config_id}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda s: s.step_id)),
        False,
        False,
        True,
    )


def study_template(
    phase: FrontierPhase,
    native: FrontierNativeConfig,
    source: EmpiricalStudySource,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    p = protocol(phase, native)
    first = f"frontier.prepare.{phase.roots[0]}"
    template = study_template_from_protocol(
        p,
        source,
        f"{phase.config_id}.source-bundle",
        experiment,
        registry,
        prefix=phase.config_id,
        source_task_id=first,
        terminal_task_id="frontier.adjudication",
        primary_output_ids={
            ScientificStage.PREPARE: "causal",
            ScientificStage.ACQUIRE: "assay",
            ScientificStage.FREEZE: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    original_source = next(e for e in template.graph.edges if e.external_input_id is not None)
    edges = [
        e
        for e in template.graph.edges
        if e.producer_output_id != "private"
        or e.consumer_node_id.startswith(("frontier.assay.", "frontier.action."))
    ]
    edges.extend(
        replace(
            original_source,
            edge_id=f"{phase.config_id}.source.{s.step_id}",
            consumer_node_id=s.step_id,
        )
        for s in p.steps
        if s.step_id != first
        and (
            s.capability_key == SOURCE.capability_key
            or s.step_id.startswith("frontier.predictions.")
        )
    )
    external = list(template.graph.external_inputs)
    for parent in phase.upstream:
        a = parent.artifact
        # This is a previously exposed configuration/evidence input, never a fresh trial.
        external.append(
            CandidateGraphExternalInput(
                a.artifact_id,
                ScientificInputRole.MODEL,
                a.artifact_id,
                ContentIdentityPolicy.EXACT_SHA256,
                a.sha256,
                a.payload_schema,
                a.media_type,
                a.size_bytes,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
        consumers = (
            (f"frontier.prepare.{parent.key}",)
            if phase.phase == "B"
            else tuple(
                s.step_id
                for s in p.steps
                if s.step_id.startswith("frontier.predictions.")
                or (
                    parent.key == "development"
                    and s.step_id in ("frontier.qualification", "frontier.prospective-evaluation-plan")
                )
                or (
                    parent.key == "laws"
                    and (
                        s.step_id == "frontier.prospective-evaluation-plan"
                        or s.step_id.startswith(("frontier.freeze.", "frontier.prospective-evaluation-reveal."))
                    )
                )
            )
        )
        for consumer in consumers:
            edges.append(
                CandidateGraphEdge(
                    f"edge.upstream.{parent.key}.{consumer}",
                    None,
                    None,
                    a.artifact_id,
                    consumer,
                    f"input-upstream-{parent.key}",
                    ScientificInputRole.MODEL,
                    a.artifact_id,
                    a.payload_schema,
                    a.media_type,
                    a.size_bytes,
                    OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.FREEZE,
                )
            )
    graph = replace(
        template.graph,
        edges=tuple(sorted(edges, key=lambda e: e.edge_id)),
        external_inputs=tuple(sorted(external, key=lambda e: e.input_id)),
    )
    coverage = replace(
        template.coverage,
        bindings=tuple(
            replace(
                b,
                contributor_edge_ids=tuple(
                    e.edge_id for e in graph.edges if e.consumer_node_id == b.proof_owner_node_id
                ),
            )
            for b in template.coverage.bindings
        ),
    )
    return replace(template, graph=graph, coverage=coverage)
