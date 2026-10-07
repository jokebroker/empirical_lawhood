"""Nine separately issued stages; native futures never become an ad hoc child DAG."""

from dataclasses import replace

from empirical_lawhood.adapters.composition.experiment_authoring import study_template_from_protocol
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref, protocol_outputs
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.calibration import ClassicalCalibration
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.comparison import ClassicalComparison
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalStage, DEVELOPMENT_ROOTS, retained_key
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.discovery import ClassicalNomination
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.law_terminal import ClassicalLaws
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.measurement import ClassicalMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_cohort import ReactorStagedPulseResponseProspectiveCohort
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_lock import ClassicalFrozenRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_plan import ReactorStagedPulseResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_reveal import ClassicalRevealedRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_seal import ClassicalSealedRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.phase_records import ClassicalFirstScreen, ClassicalInducedBundle
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prediction import ClassicalPredictionSeal
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.qualification import ClassicalQualification
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalPreparation, ClassicalPrivate, ClassicalAssay
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, ContentIdentityPolicy, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.capabilities import CapabilityPermission, CapabilityRegistry
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)


def protocol(stage: ClassicalStage, native: ClassicalNativeConfig) -> ProtocolTemplate:
    if stage.design != native.design:
        raise ValueError("protocol changes its exact native scientific design")
    steps = []
    terminal: tuple[str, ...]
    configs = {
        c.capability_key: capability_config_ref(v, ObjectIdentity.from_record(v.config_id, v), c)
        for c, v in ((METHOD, stage), (SOURCE, native))
    }

    def add(
        task: str,
        dependencies: tuple[str, ...],
        records: tuple[tuple[str, type[CanonicalRecord]], ...],
        *,
        source: bool = False,
        reveal: bool = False,
        cpu: int = 10,
        output_mb: int = 32,
    ) -> None:
        task = f"classical.{task}"
        cap = SOURCE if source else METHOD
        prepare = task.startswith("classical.prepare.")
        terminal = task == "classical.adjudication"
        science = (
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
                science,
                cap.capability_key,
                cap.capability_version,
                configs[cap.capability_key],
                tuple(sorted(set(dependencies))),
                protocol_outputs(tuple((key, r.SCHEMA) for key, r in records)),
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
                    8 * 1024**3 if "." not in task.removeprefix("classical.") else 1024**3,
                    output_mb * 1024**2,
                ),
                (),
                BarrierKind.REVEAL
                if reveal
                else BarrierKind.NONE
                if prepare
                else BarrierKind.FREEZE,
                1,
                (
                    f"{stage.config_id}.single-terminal"
                    if terminal
                    else f"{stage.config_id}.{task}",
                ),
            )
        )

    if stage.stage == "base-menu-comparison-NOMINATION":
        add(
            "calibration",
            (),
            (("science", ClassicalNomination), ("calibration", ClassicalCalibration)),
            cpu=1170,
            output_mb=16,
        )
        terminal = ("classical.calibration",)
    else:
        if stage.stage == "staged-sequence-comparison-NOMINATION":
            add("first-screen", (), (("science", ClassicalFirstScreen),), cpu=20)
        prepare_cpu = (
            (8 if stage.block == "staged-sequence-comparison" else 10)
            if stage.role == "NOMINATION"
            else {"base-menu-comparison": 25, "expanded-menu-comparison": 30, "staged-sequence-comparison": 16}[stage.block]
            if stage.role == "QUALIFICATION"
            else {"base-menu-comparison": 30, "expanded-menu-comparison": 30, "staged-sequence-comparison": 20}[stage.block]
        )
        for root in stage.root_ids:
            prepare, predict, assay = (
                f"classical.{op}.{root}" for op in ("prepare", "predictions", "assay")
            )
            add(
                f"prepare.{root}",
                (),
                (("causal", ClassicalPreparation), ("private", ClassicalPrivate)),
                source=True,
                cpu=prepare_cpu,
                output_mb=32,
            )
            add(
                f"predictions.{root}",
                (prepare,),
                (("science", ClassicalPredictionSeal),),
                cpu=8 if stage.block == "staged-sequence-comparison" and stage.role != "PROSPECTIVE" else 10,
                output_mb=8,
            )
            if stage.role != "PROSPECTIVE":
                induced = (("induced", ClassicalInducedBundle),) if stage.block == "staged-sequence-comparison" else ()
                parents = (
                    prepare,
                    predict,
                    *(("classical.first-screen",) if stage.stage == "staged-sequence-comparison-NOMINATION" else ()),
                )
                seconds = (
                    {"expanded-menu-comparison": 240, "staged-sequence-comparison": 70}[stage.block]
                    if stage.role == "NOMINATION"
                    else {"base-menu-comparison": 150, "expanded-menu-comparison": 240, "staged-sequence-comparison": 48}[stage.block]
                )
                add(
                    f"assay.{root}",
                    parents,
                    (("assay", ClassicalAssay), *induced),
                    source=True,
                    cpu=seconds,
                    output_mb=8,
                )
                add(
                    f"measure.{root}",
                    (prepare, assay),
                    (("science", ClassicalMeasuredRoot),),
                    cpu=8 if stage.block == "staged-sequence-comparison" else 10,
                    output_mb=8,
                )
            else:
                freeze, action, seal = (
                    f"classical.{op}.{root}" for op in ("freeze", "action", "prospective-evaluation-seal")
                )
                add(
                    f"freeze.{root}",
                    (prepare, predict, "classical.prospective-evaluation-plan"),
                    (("science", ClassicalFrozenRoot),),
                    cpu={"base-menu-comparison": 280, "expanded-menu-comparison": 260, "staged-sequence-comparison": 95}[stage.block],
                    output_mb=48,
                )
                add(
                    f"action.{root}",
                    (prepare, freeze, "classical.prospective-evaluation-plan"),
                    (("assay", ClassicalNativeRoot),),
                    source=True,
                    cpu={"base-menu-comparison": 170, "expanded-menu-comparison": 150, "staged-sequence-comparison": 170}[stage.block],
                    output_mb=64,
                )
                add(
                    f"prospective-evaluation-seal.{root}",
                    (prepare, freeze, action, "classical.prospective-evaluation-plan"),
                    (("science", ClassicalSealedRoot),),
                    cpu={"base-menu-comparison": 100, "expanded-menu-comparison": 130, "staged-sequence-comparison": 100}[stage.block],
                    output_mb=48,
                )
                add(
                    f"prospective-evaluation-reveal.{root}",
                    (action, seal, "classical.prospective-evaluation-plan"),
                    (("science", ClassicalRevealedRoot),),
                    reveal=True,
                    cpu={"base-menu-comparison": 70, "expanded-menu-comparison": 90, "staged-sequence-comparison": 80}[stage.block],
                    output_mb=48,
                )
        preparations = tuple(f"classical.prepare.{r}" for r in stage.root_ids)
        predictions = tuple(f"classical.predictions.{r}" for r in stage.root_ids)
        measurements = tuple(f"classical.measure.{r}" for r in stage.root_ids)
        assays = tuple(f"classical.assay.{r}" for r in stage.root_ids)
        if stage.role == "NOMINATION":
            add(
                "nomination",
                (
                    *measurements,
                    *predictions,
                    *assays,
                    *(("classical.first-screen",) if stage.block == "staged-sequence-comparison" else ()),
                ),
                (("science", ClassicalNomination),),
                cpu=200,
                output_mb=32,
            )
            terminal = ("classical.nomination",)
        elif stage.role == "QUALIFICATION":
            add(
                "qualification",
                (*measurements, *predictions, *assays),
                (("science", ClassicalQualification),),
                cpu=200 if stage.block == "staged-sequence-comparison" else 300,
                output_mb=32,
            )
            add(
                "laws",
                ("classical.qualification", *assays),
                (("science", ClassicalLaws),),
                cpu={"base-menu-comparison": 600, "expanded-menu-comparison": 1000, "staged-sequence-comparison": 200}[stage.block],
                output_mb=256,
            )
            terminal = ("classical.laws",)
        else:
            reveals = tuple(f"classical.prospective-evaluation-reveal.{r}" for r in stage.root_ids)
            add(
                "prospective-evaluation-plan",
                (*preparations, *predictions),
                (("science", ReactorStagedPulseResponseProspectivePlan),),
                cpu=400,
                output_mb=256,
            )
            add(
                "prospective-evaluation-cohort",
                ("classical.prospective-evaluation-plan", *reveals),
                (("science", ReactorStagedPulseResponseProspectiveCohort),),
                reveal=True,
                cpu=300,
                output_mb=256,
            )
            add(
                "comparison",
                ("classical.prospective-evaluation-cohort", *reveals),
                (("science", ClassicalComparison),),
                reveal=True,
                cpu=300,
                output_mb=8,
            )
            terminal = ("classical.prospective-evaluation-cohort", "classical.comparison")
    add(
        "adjudication",
        terminal,
        (("science", ScientificAdjudicationRecord),),
        reveal=True,
        cpu=20 if stage.block == "staged-sequence-comparison" and stage.role != "PROSPECTIVE" else 30,
        output_mb=1,
    )
    if sum(s.resource_budget.wall_time_seconds for s in steps) > stage.cpu_seconds:
        raise ValueError("task reservations exceed this stage CPU allocation")
    return ProtocolTemplate(
        f"{stage.config_id}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda s: s.step_id)),
        False,
        False,
        True,
    )


def upstream_consumers(
    stage: ClassicalStage, key: str, tasks: tuple[str, ...]
) -> tuple[str, ...]:
    if stage.stage == "base-menu-comparison-NOMINATION":
        return ("classical.calibration",)
    if stage.role == "NOMINATION":
        if key == "historical-b":
            return ("classical.nomination",) if stage.block == "expanded-menu-comparison" else ("classical.first-screen",)
        root, role = next(
            (r, role)
            for r in DEVELOPMENT_ROOTS
            for role in ("assay", "measurement", "preparation", "private")
            if retained_key(r, role) == key
        )
        if stage.block == "staged-sequence-comparison" and role in ("preparation", "measurement"):
            return tuple(
                sorted(
                    (
                        "classical.first-screen",
                        *((f"classical.prepare.{root}",) if role == "preparation" else ()),
                    )
                )
            )
        return (
            tuple(
                sorted(
                    (
                        f"classical.prepare.{root}",
                        *((f"classical.measure.{root}",) if role == "preparation" else ()),
                    )
                )
            )
            if role in ("preparation", "private")
            else (f"classical.assay.{root}", f"classical.measure.{root}")
        )
    if key == "nomination":
        return tuple(
            t
            for t in tasks
            if t.startswith(
                (
                    "classical.prepare.",
                    "classical.assay.",
                    "classical.action.",
                    "classical.predictions.",
                )
            )
            or t == "classical.qualification"
        )
    return tuple(
        t
        for t in tasks
        if t.startswith(("classical.prepare.", "classical.freeze.", "classical.action."))
        or t == "classical.prospective-evaluation-plan"
    )


def study_template(
    stage: ClassicalStage,
    native: ClassicalNativeConfig,
    source: EmpiricalStudySource,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    p = protocol(stage, native)
    first = (
        "classical.calibration"
        if stage.stage == "base-menu-comparison-NOMINATION"
        else f"classical.prepare.{stage.root_ids[0]}"
    )
    template = study_template_from_protocol(
        p,
        source,
        f"{stage.config_id}.source-bundle",
        experiment,
        registry,
        prefix=stage.config_id,
        source_task_id=first,
        terminal_task_id="classical.adjudication",
        primary_output_ids={
            ScientificStage.PREPARE: "causal",
            ScientificStage.ACQUIRE: "assay",
            ScientificStage.FREEZE: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    original = next(e for e in template.graph.edges if e.external_input_id is not None)
    edges = [
        e
        for e in template.graph.edges
        if e.producer_output_id != "private"
        or e.consumer_node_id.startswith(("classical.assay.", "classical.action."))
    ]
    edges.extend(
        replace(
            original, edge_id=f"{stage.config_id}.source.{s.step_id}", consumer_node_id=s.step_id
        )
        for s in p.steps
        if s.step_id != first
        and (
            s.capability_key == SOURCE.capability_key
            or s.step_id.startswith(("classical.predictions.", "classical.freeze."))
            or s.step_id == "classical.prospective-evaluation-plan"
        )
    )
    external = list(template.graph.external_inputs)
    for parent in stage.upstream:
        a = parent.artifact
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
        for consumer in upstream_consumers(stage, parent.key, tuple(s.step_id for s in p.steps)):
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
