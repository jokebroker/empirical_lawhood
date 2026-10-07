"""Mechanical lowering for frozen multi-partition post-hoc portfolios."""

from __future__ import annotations

import hashlib

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.posthoc import POSTHOC_ANALYSIS_IDS, PosthocSourceArtifact, PosthocTrancheFreeze

from .artifacts import ArtifactProfile
from .capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityPermission,
    CapabilityRequirement,
)
from .plans import ArtifactOutputSpec, BarrierKind, CandidateExecutionPlan, ExecutionTask, ExternalInputSpec, PlanLane, ScientificInputRole, ScientificInputSpec, ScientificStage


ANALYSIS_RESULT_SCHEMA = 'empirical-lawhood/runtime/analysis-result'
SKEPTIC_RESULT_SCHEMA = 'empirical-lawhood/runtime/skeptic-report'
INTEGRATED_SKEPTIC_SCHEMA = 'empirical-lawhood/runtime/integrated-skeptic'
METATHEORY_SYNTHESIS_SCHEMA = 'empirical-lawhood/runtime/metatheory-synthesis'
RESULT_INDEX_SCHEMA = 'empirical-lawhood/runtime/result-index'
POSTHOC_CONFIG_SCHEMA = 'empirical-lawhood/runtime/posthoc-portfolio'
POSTHOC_CAPABILITY_KEY = "posthoc-portfolio.analysis"
POSTHOC_CAPABILITY_VERSION = "1.0.0"


def _qualified_payload_schema(source: PosthocSourceArtifact) -> str:
    if source.qualification_status != "QUALIFIED" or source.payload_schema is None:
        raise ValueError(f"unqualified source entered the execution graph: {source.artifact_id}")
    return source.payload_schema


def _task_budget(*, source_scan_bytes: int, output_bytes: int) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=2 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=2 * 60 * 60,
        source_scan_bytes=max(source_scan_bytes, 1),
        output_bytes=output_bytes,
    )


def _config_ref(freeze: PosthocTrancheFreeze) -> CapabilityConfigRef:
    schema_sha256 = hashlib.sha256(POSTHOC_CONFIG_SCHEMA.encode("utf-8")).hexdigest()
    return CapabilityConfigRef(
        config_id="effective-law-posthoc",
        config_schema=POSTHOC_CONFIG_SCHEMA,
        config_schema_sha256=schema_sha256,
        content_sha256=freeze.fingerprint(),
        artifact_id="effective-law-posthoc.freeze",
    )


def _requirement(
    freeze: PosthocTrancheFreeze,
    *,
    input_schemas: tuple[str, ...],
    output_schema: str,
    source_scan_bytes: int,
) -> CapabilityRequirement:
    return CapabilityRequirement(
        capability_key=POSTHOC_CAPABILITY_KEY,
        capability_version=POSTHOC_CAPABILITY_VERSION,
        kind=CapabilityKind.ANALYSIS,
        config=_config_ref(freeze),
        required_input_schema_ids=tuple(sorted(set(input_schemas))),
        required_output_schema_ids=(output_schema,),
        required_permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_OUTCOME_VISIBLE,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        requested_resources=_task_budget(
            source_scan_bytes=source_scan_bytes,
            output_bytes=16 * 1024**2,
        ),
    )


def _output(
    *,
    run_root: str,
    task_id: str,
    output_schema: str,
) -> ArtifactOutputSpec:
    return ArtifactOutputSpec(
        output_id=f"{task_id}-result",
        logical_artifact_id=f"effective-law-posthoc.{task_id}",
        relative_path=f"{run_root}/outputs/{task_id}/result.json",
        payload_schema=output_schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def _external_inputs(
    freeze: PosthocTrancheFreeze,
    parent_ids: tuple[str, ...],
) -> tuple[ExternalInputSpec, ...]:
    parents = {parent.parent_id: parent for parent in freeze.parents}
    return tuple(
        ExternalInputSpec(
            input_id=f"source-{source.artifact_id}",
            logical_artifact_id=source.artifact_id,
            expected_content_sha256=source.content_sha256,
            expected_payload_schema=_qualified_payload_schema(source),
            expected_media_type="application/json",
            expected_size_bytes=source.size_bytes,
            expected_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            expected_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            identity_scope_sha256=None,
        )
        for parent_id in parent_ids
        for source in parents[parent_id].source_artifacts
        if source.qualification_status == "QUALIFIED"
    )


def _external_edges(
    freeze: PosthocTrancheFreeze,
    parent_ids: tuple[str, ...],
) -> tuple[ScientificInputSpec, ...]:
    parents = {parent.parent_id: parent for parent in freeze.parents}
    return tuple(
        ScientificInputSpec(
            edge_id=f"edge-{source.artifact_id}",
            consumer_input_id=f"source-{source.artifact_id}",
            scientific_role=ScientificInputRole.OUTCOME,
            producer_task_id=None,
            producer_output_id=None,
            external_input_id=f"source-{source.artifact_id}",
            logical_artifact_id=source.artifact_id,
            operational_logical_artifact_id=source.artifact_id,
            payload_schema=_qualified_payload_schema(source),
            media_type="application/json",
            maximum_size_bytes=source.size_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
        )
        for parent_id in parent_ids
        for source in parents[parent_id].source_artifacts
        if source.qualification_status == "QUALIFIED"
    )


def _dependency_edges(
    *,
    task_id: str,
    dependencies: tuple[str, ...],
    schemas: dict[str, str],
) -> tuple[ScientificInputSpec, ...]:
    return tuple(
        ScientificInputSpec(
            edge_id=f"edge-{dependency}-to-{task_id}",
            consumer_input_id=f"input-{dependency}",
            scientific_role=ScientificInputRole.OUTCOME,
            producer_task_id=dependency,
            producer_output_id=f"{dependency}-result",
            external_input_id=None,
            logical_artifact_id=f"effective-law-posthoc.{dependency}",
            operational_logical_artifact_id=f"effective-law-posthoc.{dependency}",
            payload_schema=schemas[dependency],
            media_type="application/json",
            maximum_size_bytes=16 * 1024**2,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
        )
        for dependency in dependencies
    )


def compile_posthoc_portfolio_execution_plan(
    freeze: PosthocTrancheFreeze,
    *,
    run_root: str,
    registry_sha256: str,
) -> CandidateExecutionPlan:
    "Lower the fixed scientific-purpose analysis/skeptic/synthesis graph without config DAG input."

    if run_root.startswith("/") or ".." in run_root.split("/"):
        raise ValueError("run_root must be an external-root-relative locator")
    analysis_by_id = {analysis.analysis_id: analysis for analysis in freeze.analyses}
    schemas: dict[str, str] = {
        **{analysis_id: ANALYSIS_RESULT_SCHEMA for analysis_id in analysis_by_id},
        **{f"skeptic.{analysis_id.removeprefix('analysis.')}": SKEPTIC_RESULT_SCHEMA for analysis_id in POSTHOC_ANALYSIS_IDS},
        "skeptic.integrated": INTEGRATED_SKEPTIC_SCHEMA,
        "synthesis.metatheory": METATHEORY_SYNTHESIS_SCHEMA,
        "index.results": RESULT_INDEX_SCHEMA,
    }
    tasks: list[ExecutionTask] = []
    for task_id in POSTHOC_ANALYSIS_IDS:
        analysis = analysis_by_id[task_id]
        external = _external_inputs(freeze, analysis.eligible_parent_ids)
        input_schemas = tuple(
            sorted(
                {
                    value.expected_payload_schema
                    for value in external
                    if value.expected_payload_schema
                }
            )
        )
        tasks.append(
            ExecutionTask(
                task_id=task_id,
                stage=ScientificStage.EXPLORE,
                capability=_requirement(
                    freeze,
                    input_schemas=input_schemas,
                    output_schema=ANALYSIS_RESULT_SCHEMA,
                    source_scan_bytes=sum(value.expected_size_bytes or 0 for value in external),
                ),
                capability_implementation_sha256=freeze.implementation_sha256,
                dependency_task_ids=(),
                external_inputs=external,
                outputs=(
                    _output(run_root=run_root, task_id=task_id, output_schema=schemas[task_id]),
                ),
                resource_lock_ids=(),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=(
                    "complete-family",
                    "counterexample-precedence",
                    "no-cross-partition-pooling",
                ),
                scientific_inputs=_external_edges(freeze, analysis.eligible_parent_ids),
            )
        )
        skeptic_id = f"skeptic.{task_id.removeprefix('analysis.')}"
        deps = (task_id,)
        tasks.append(
            ExecutionTask(
                task_id=skeptic_id,
                stage=ScientificStage.FALSIFY,
                capability=_requirement(
                    freeze,
                    input_schemas=(ANALYSIS_RESULT_SCHEMA,),
                    output_schema=SKEPTIC_RESULT_SCHEMA,
                    source_scan_bytes=16 * 1024**2,
                ),
                capability_implementation_sha256=freeze.implementation_sha256,
                dependency_task_ids=deps,
                external_inputs=(),
                outputs=(
                    _output(
                        run_root=run_root, task_id=skeptic_id, output_schema=schemas[skeptic_id]
                    ),
                ),
                resource_lock_ids=(),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=("counterexample-precedence", "family-completeness"),
                scientific_inputs=_dependency_edges(
                    task_id=skeptic_id,
                    dependencies=deps,
                    schemas=schemas,
                ),
            )
        )
    skeptic_dependencies = tuple(sorted(f"skeptic.{analysis_id.removeprefix('analysis.')}" for analysis_id in POSTHOC_ANALYSIS_IDS))
    tasks.append(
        ExecutionTask(
            task_id="skeptic.integrated",
            stage=ScientificStage.FALSIFY,
            capability=_requirement(
                freeze,
                input_schemas=(SKEPTIC_RESULT_SCHEMA,),
                output_schema=INTEGRATED_SKEPTIC_SCHEMA,
                source_scan_bytes=len(skeptic_dependencies) * 16 * 1024**2,
            ),
            capability_implementation_sha256=freeze.implementation_sha256,
            dependency_task_ids=skeptic_dependencies,
            external_inputs=(),
            outputs=(_output(run_root=run_root, task_id="skeptic.integrated", output_schema=schemas["skeptic.integrated"]),),
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("complete-analysis-family", "no-target-specific-rescue"),
            scientific_inputs=_dependency_edges(
                task_id="skeptic.integrated",
                dependencies=skeptic_dependencies,
                schemas=schemas,
            ),
        )
    )
    synthesis_dependencies = tuple(sorted(("skeptic.integrated", *POSTHOC_ANALYSIS_IDS)))
    tasks.append(
        ExecutionTask(
            task_id="synthesis.metatheory",
            stage=ScientificStage.SYNTHESIZE,
            capability=_requirement(
                freeze,
                input_schemas=(ANALYSIS_RESULT_SCHEMA, INTEGRATED_SKEPTIC_SCHEMA),
                output_schema=METATHEORY_SYNTHESIS_SCHEMA,
                source_scan_bytes=len(synthesis_dependencies) * 16 * 1024**2,
            ),
            capability_implementation_sha256=freeze.implementation_sha256,
            dependency_task_ids=synthesis_dependencies,
            external_inputs=(),
            outputs=(_output(run_root=run_root, task_id="synthesis.metatheory", output_schema=schemas["synthesis.metatheory"]),),
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=(
                "claim-ceiling",
                "counterexample-precedence",
                "separate-evidence-worlds",
            ),
            scientific_inputs=_dependency_edges(
                task_id="synthesis.metatheory",
                dependencies=synthesis_dependencies,
                schemas=schemas,
            ),
        )
    )
    index_dependencies = tuple(sorted((*schemas.keys(),)))
    index_dependencies = tuple(value for value in index_dependencies if value != "index.results")
    tasks.append(
        ExecutionTask(
            task_id="index.results",
            stage=ScientificStage.REPORT,
            capability=_requirement(
                freeze,
                input_schemas=tuple(sorted(set(schemas[value] for value in index_dependencies))),
                output_schema=RESULT_INDEX_SCHEMA,
                source_scan_bytes=len(index_dependencies) * 16 * 1024**2,
            ),
            capability_implementation_sha256=freeze.implementation_sha256,
            dependency_task_ids=index_dependencies,
            external_inputs=(),
            outputs=(_output(run_root=run_root, task_id="index.results", output_schema=schemas["index.results"]),),
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("receipt-completeness", "terminal-status-completeness"),
            scientific_inputs=_dependency_edges(
                task_id="index.results",
                dependencies=index_dependencies,
                schemas=schemas,
            ),
        )
    )
    return CandidateExecutionPlan(
        execution_plan_id="effective-law-posthoc.execution",
        lane=PlanLane.EXPLORATORY,
        source_plan=ObjectIdentity.from_record(freeze.freeze_id, freeze),
        registry_sha256=registry_sha256,
        implementation_commit=freeze.implementation_commit,
        tasks=tuple(sorted(tasks, key=lambda value: value.task_id)),
        nonactuating=True,
        frozen=True,
        candidate=ObjectIdentity.from_record(freeze.freeze_id, freeze),
    )


__all__ = [
    "ANALYSIS_RESULT_SCHEMA",
    "INTEGRATED_SKEPTIC_SCHEMA",
    "METATHEORY_SYNTHESIS_SCHEMA",
    "POSTHOC_CAPABILITY_KEY",
    "POSTHOC_CAPABILITY_VERSION",
    "POSTHOC_CONFIG_SCHEMA",
    "RESULT_INDEX_SCHEMA",
    "SKEPTIC_RESULT_SCHEMA",
    "compile_posthoc_portfolio_execution_plan",
]
