"""One predeclared qualification-to-use graph compiled by the public platform."""

from dataclasses import replace

from empirical_lawhood.adapters.composition.experiment_authoring import study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import PREFIX, ROOTS, ClassicalDesign
from empirical_lawhood.adapters.methods.reactor_selected_action_response.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalCausal, ClassicalPrivate, ClassicalDecision, ClassicalAssay, ClassicalQualification
from empirical_lawhood.adapters.methods.reactor_selected_action_response.law_terminal import ClassicalLaw
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_closeout import ClassicalSealResult
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_reveal import ClassicalRevealResult
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_cohort import ReactorSelectedActionResponseProspectiveCohort
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.prospective_records import ClassicalNativeRoot
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import StudyTemplate
from empirical_lawhood.runtime.capabilities import CapabilityRegistry, CapabilityPermission
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)


def protocol(design: ClassicalDesign, native: ClassicalNativeConfig) -> ProtocolTemplate:
    if native.design != design:
        raise ValueError("classical native/scientific specifications differ")
    steps = []
    configs = {
        c.capability_key: _config_ref(v, ObjectIdentity.from_record(v.config_id, v), c)
        for c, v in ((METHOD, design), (SOURCE, native))
    }

    def add(
        task: str,
        dependencies: tuple[str, ...],
        outputs: tuple[tuple[str, str], ...],
        *,
        source: bool = False,
        reveal: bool = False,
        terminal: bool = False,
        cpu: int = 60,
    ) -> None:
        capability = SOURCE if source else METHOD
        prepare = task.startswith("classical.prepare.")
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
                capability.capability_key,
                capability.capability_version,
                configs[capability.capability_key],
                tuple(sorted(set(dependencies))),
                _outputs(outputs),
                capability.permissions
                if source or reveal
                else tuple(
                    p
                    for p in capability.permissions
                    if p is not CapabilityPermission.REVEAL_OUTCOMES
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
                        "classical.qualification",
                        "classical.law",
                        "classical.prospective-evaluation-plan",
                        "classical.prospective-evaluation-cohort",
                    )
                    else 256 * 1024**2,
                    128 * 1024**2,
                ),
                (),
                BarrierKind.REVEAL
                if reveal
                else BarrierKind.NONE
                if prepare
                else BarrierKind.FREEZE,
                1,
                (f"{PREFIX}.single-terminal",) if terminal else (f"{PREFIX}.{task}",),
            )
        )

    for root, role, _, _ in ROOTS:
        prep, decision = f"classical.prepare.{root}", f"classical.decision.{root}"
        release = ("classical.law",) if role == "prospective" else ()
        add(
            prep,
            release,
            (("causal", ClassicalCausal.SCHEMA), ("private", ClassicalPrivate.SCHEMA)),
            source=True,
            cpu=120,
        )
        add(decision, (prep, *release), (("science", ClassicalDecision.SCHEMA),), cpu=30)
        if role == "qualification":
            add(
                f"classical.assay.{root}",
                (prep, decision),
                (("assay", ClassicalAssay.SCHEMA),),
                source=True,
                cpu=240,
            )
        else:
            action, seal = f"classical.action.{root}", f"classical.prospective-evaluation-seal.{root}"
            add(
                action,
                (prep, decision, "classical.law", "classical.qualification", "classical.prospective-evaluation-plan"),
                (("assay", ClassicalNativeRoot.SCHEMA),),
                source=True,
                cpu=360,
            )
            add(
                seal,
                (prep, decision, action, "classical.prospective-evaluation-plan"),
                (("science", ClassicalSealResult.SCHEMA),),
                cpu=45,
            )
            add(
                f"classical.prospective-evaluation-reveal.{root}",
                (prep, decision, action, seal, "classical.prospective-evaluation-plan"),
                (("science", ClassicalRevealResult.SCHEMA),),
                reveal=True,
                cpu=45,
            )
    q = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
    p = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
    add(
        "classical.qualification",
        tuple(
            f"classical.{kind}.{root}" for root in q for kind in ("prepare", "decision", "assay")
        ),
        (("science", ClassicalQualification.SCHEMA),),
        cpu=1800,
    )
    add(
        "classical.law",
        ("classical.qualification", *(f"classical.assay.{root}" for root in q)),
        (("science", ClassicalLaw.SCHEMA),),
        cpu=1800,
    )
    add(
        "classical.prospective-evaluation-plan",
        (
            "classical.law",
            *(f"classical.{kind}.{root}" for root in p for kind in ("prepare", "decision")),
        ),
        (("science", ReactorSelectedActionResponseProspectivePlan.SCHEMA),),
        cpu=1800,
    )
    add(
        "classical.prospective-evaluation-cohort",
        ("classical.prospective-evaluation-plan", *(f"classical.prospective-evaluation-reveal.{root}" for root in p)),
        (("science", ReactorSelectedActionResponseProspectiveCohort.SCHEMA),),
        reveal=True,
        cpu=1800,
    )
    add(
        "classical.adjudication",
        ("classical.law", "classical.prospective-evaluation-cohort"),
        (("science", ScientificAdjudicationRecord.SCHEMA),),
        reveal=True,
        terminal=True,
        cpu=300,
    )
    return ProtocolTemplate(
        f"{PREFIX}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda s: s.step_id)),
        False,
        False,
        True,
    )


def study_template(
    design: ClassicalDesign,
    native: ClassicalNativeConfig,
    source: EmpiricalStudySource,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    p = protocol(design, native)
    first = next(
        s.step_id
        for s in p.steps
        if s.step_id.startswith("classical.prepare.") and "qualification" in s.step_id
    )
    template = _programme_template(
        p,
        source,
        f"{PREFIX}.source-bundle",
        experiment,
        registry,
        prefix=PREFIX,
        source_task_id=first,
        terminal_task_id="classical.adjudication",
        primary_output_ids={
            ScientificStage.PREPARE: "causal",
            ScientificStage.ACQUIRE: "assay",
            ScientificStage.FREEZE: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    source_edge = next(e for e in template.graph.edges if e.external_input_id is not None)
    # Truth never enters a decision, law predictor, prospective controller evaluation plan or qualification summary input.
    private_consumers = (
        "classical.assay.",
        "classical.action.",
        "classical.prospective-evaluation-seal.",
        "classical.prospective-evaluation-reveal.",
    )
    edges = [
        e
        for e in template.graph.edges
        if e.producer_output_id != "private" or e.consumer_node_id.startswith(private_consumers)
    ]
    edges.extend(
        replace(source_edge, edge_id=f"{PREFIX}.source.{s.step_id}", consumer_node_id=s.step_id)
        for s in p.steps
        if s.capability_key == SOURCE.capability_key and s.step_id != first
    )
    graph = replace(template.graph, edges=tuple(sorted(edges, key=lambda e: e.edge_id)))
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
