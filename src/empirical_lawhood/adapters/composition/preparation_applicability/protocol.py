"""Adapter-owned authoring protocol; separate causal seals precede effects."""

from typing import Any


from dataclasses import replace
from empirical_lawhood.adapters.composition.experiment_authoring import study_template_from_protocol
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref, protocol_outputs
from empirical_lawhood.adapters.methods.preparation_applicability.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.simulators.preparation_applicability.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.adapters.methods.preparation_applicability.records import (
    PreparationApplicabilitySelection,
    PreparationApplicabilityMeasuredRoot,
)
from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityLowerSeal
from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityParents
from empirical_lawhood.adapters.simulators.preparation_applicability.records import (
    PreparationApplicabilityPrefix,
    PreparationApplicabilityPanel,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.runtime.capabilities import CapabilityPermission
from empirical_lawhood.runtime.plans import (
    ProtocolTemplate,
    ProtocolStepTemplate,
    ScientificStage,
    BarrierKind,
    ScientificInputRole,
)
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateGraphExternalInput,
    CandidateGraphEdge,
    ContentIdentityPolicy,
)


def study_template(stage: Any, native: Any, source: Any, experiment: Any, registry: Any) -> Any:
    steps = []
    configs = {
        c.capability_key: capability_config_ref(v, ObjectIdentity.from_record(v.config_id, v), c)
        for c, v in ((METHOD, stage), (SOURCE, native))
    }

    def add(
        task: Any,
        dependencies: Any,
        record: Any,
        science: Any,
        *,
        source: Any = False,
        cpu: Any = 10,
        mb: Any = 4,
        reveal: Any = False,
    ) -> Any:
        cap = SOURCE if source else METHOD
        steps.append(
            ProtocolStepTemplate(
                task,
                science,
                cap.capability_key,
                cap.capability_version,
                configs[cap.capability_key],
                tuple(sorted(set(dependencies))),
                protocol_outputs((("science", record.SCHEMA),)),
                cap.permissions
                if reveal
                else tuple(
                    p for p in cap.permissions if p is not CapabilityPermission.REVEAL_OUTCOMES
                ),
                OutcomeAccess.EVALUATOR_REVEAL if reveal else OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.OUTCOME_VISIBLE
                if task == "ap.adjudication"
                else VisibilityCeiling.PROSPECTIVE,
                ResourceBudget(
                    1,
                    8 * 1024**3,
                    0,
                    cpu,
                    (
                        1024
                        if task in ("ap.report",)
                        else 128
                        if task.startswith("ap.measure.")
                        else 32
                    )
                    * 1024**2,
                    mb * 1024**2,
                ),
                (),
                BarrierKind.REVEAL
                if reveal
                else BarrierKind.NONE
                if science is ScientificStage.PREPARE
                else BarrierKind.FREEZE,
                1,
                (
                    f"{stage.config_id}.single-terminal"
                    if task == "ap.adjudication"
                    else f"{stage.config_id}.{task}",
                ),
            )
        )

    for root in stage.root_ids:
        prefix, select, parents, lower, assay, measure = (
            f"ap.{p}.{root}" for p in ("prefix", "select", "parents", "lower", "assay", "measure")
        )
        add(prefix, (), PreparationApplicabilityPrefix, ScientificStage.PREPARE, source=True, cpu=600, mb=16)
        add(select, (prefix,), PreparationApplicabilitySelection, ScientificStage.FREEZE)
        add(
            parents,
            (prefix, select),
            PreparationApplicabilityParents,
            ScientificStage.ACQUIRE,
            source=True,
            cpu=100,
            mb=8,
        )
        add(lower, (prefix, parents), PreparationApplicabilityLowerSeal, ScientificStage.FREEZE)
        add(
            assay,
            (prefix, select, parents, lower),
            PreparationApplicabilityPanel,
            ScientificStage.ACQUIRE,
            source=True,
            cpu=2400,
            mb=96,
        )
        add(
            measure,
            (prefix, assay, lower),
            PreparationApplicabilityMeasuredRoot,
            ScientificStage.FREEZE,
            cpu=20,
        )
    reduction = "ap.report"
    deps = tuple(f"ap.measure.{r}" for r in stage.root_ids)
    add(reduction, deps, PreparationApplicabilityScreenReport, ScientificStage.FREEZE, cpu=1000)
    add(
        "ap.adjudication",
        (reduction,),
        ScientificAdjudicationRecord,
        ScientificStage.EVALUATE,
        cpu=20,
        reveal=True,
    )
    protocol = ProtocolTemplate(
        f"{stage.config_id}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda s: s.step_id)),
        False,
        False,
        True,
    )
    template = study_template_from_protocol(
        protocol,
        source,
        f"{stage.config_id}.source-bundle",
        experiment,
        registry,
        prefix=stage.config_id,
        source_task_id=f"ap.prefix.{stage.root_ids[0]}",
        terminal_task_id="ap.adjudication",
        primary_output_ids={s: "science" for s in ScientificStage},
    )
    first_edge = next(e for e in template.graph.edges if e.external_input_id is not None)
    # No unrelated configuration bundle is sent to method tasks.
    edges = list(template.graph.edges)
    edges.extend(
        replace(first_edge, edge_id=f"edge.source.{s.step_id}", consumer_node_id=s.step_id)
        for s in protocol.steps
        if s.capability_key == SOURCE.capability_key and s.step_id != first_edge.consumer_node_id
    )
    externals = list(template.graph.external_inputs)
    for parent in stage.upstream:
        a = parent.artifact
        externals.append(
            CandidateGraphExternalInput(
                a.artifact_id,
                ScientificInputRole.MODEL,
                a.artifact_id,
                ContentIdentityPolicy.EXACT_SHA256,
                a.sha256,
                a.payload_schema,
                a.media_type,
                a.size_bytes,
                OutcomeAccess.OUTCOME_BLIND if parent.key == "lower" else OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
        consumers = (
            tuple(
                s.step_id
                for s in protocol.steps
                if s.step_id.startswith(("ap.lower.", "ap.measure.")) or s.step_id == "ap.report"
            )
            if parent.key == "lower"
            else tuple(
                s.step_id
                for s in protocol.steps
                if s.step_id.startswith(("ap.select.", "ap.prefix.", "ap.parents.", "ap.assay."))
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
                    OutcomeAccess.OUTCOME_BLIND if parent.key == "lower" else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.FREEZE,
                )
            )
    graph = replace(
        template.graph,
        external_inputs=tuple(sorted(externals, key=lambda i: i.input_id)),
        edges=tuple(sorted(edges, key=lambda e: e.edge_id)),
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
