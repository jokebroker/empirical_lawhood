"Candidate/compiler/runtime composition for the bounded Gym--TORAX prepared-base campaign.\n\nThe module is intentionally additive.  It reuses the generic candidate,\nprogramme-issue, scheduler, artifact and receipt boundaries and only supplies\nthe exact prepared-base source/configuration and task implementations.\n"

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from decimal import Decimal
import hashlib
from typing import Any, Callable

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ReadinessStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.worlds import NumericalCoordinateKind, NumericalCoordinateSpec
from empirical_lawhood.planning.experiment_entry import StudyDefinition
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)
from empirical_lawhood.planning.formal_gaps import FormalGapCoverage, FormalGapRegister
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    StreamingOutputEmitter,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import (
    SourceMaterializationConfig,
    SourceReadMode,
)

from .open_campaigns import (
    OpenSimulatorKind,
    OpenSimulatorSourceManifest,
    _entry_package,
    build_open_simulator_authoring_bundle,
    open_simulator_campaign,
    open_simulator_experiment,
    open_simulator_formal_catalog,
    open_simulator_system,
)
from .prepared_base import DevelopmentDecision, EvaluationAdjudication, PreparedBaseAcquisitionBatch, PreparedBaseConfig, PreparedBaseStage, PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID, PreparedBasePrimaryResult, acquire_stage, adjudicate_development, adjudicate_evaluation, development_cells, evaluation_master_cells


_VERSION = "1.0.0"
_SOURCE_STEP_ID = "source"
_FREEZE_STEP_ID = "freeze"
_ACQUIRE_STEP_ID = "acquire"
_REPORT_STEP_ID = "report"
_SCIENTIFIC_OUTPUT_ID = "scientific-adjudication"
_SOURCE_OUTPUT_ID = "source-ready"
_FROZEN_OUTPUT_ID = "frozen-source"
_BATCH_OUTPUT_ID = "prepared-base-batch"
_DEVELOPMENT_OUTPUT_ID = "development-decision"
_EVALUATION_OUTPUT_ID = "evaluation-adjudication"

_READ_WRITE = tuple(
    sorted(
        (
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        key=lambda value: value.value,
    )
)
_DEVELOP = tuple(
    sorted(
        (*_READ_WRITE, CapabilityPermission.READ_DEVELOPMENT),
        key=lambda value: value.value,
    )
)
_EVALUATE = tuple(
    sorted(
        (
            *_READ_WRITE,
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        ),
        key=lambda value: value.value,
    )
)
_SMALL_TASK = ResourceBudget(
    cpu_cores=1,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=1024**2,
    output_bytes=16 * 1024**2,
)
_ACQUISITION_TASK = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=4 * 3600,
    source_scan_bytes=1024**2,
    output_bytes=1024**3,
)
_REPORT_TASK = ResourceBudget(
    cpu_cores=1,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=1024**3,
    output_bytes=16 * 1024**2,
)
_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=8 * 3600,
    source_scan_bytes=1027 * 1024**2,
    output_bytes=2 * 1024**3,
)


@dataclass(frozen=True, slots=True)
class PreparedBaseSourceFile(CanonicalRecord):
    """One exact source file in the bounded dirty-worktree closure."""

    SCHEMA = 'empirical-lawhood/adapters/prepared-base-source-file'

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        if self.size_bytes <= 0:
            raise ValueError("prepared-base source file must be nonempty")
        validate_sha256(self.sha256, field_name="sha256")


@dataclass(frozen=True, slots=True)
class PreparedBaseImplementationManifest(CanonicalRecord):
    """Exact implementation closure used for candidate and issue identity."""

    SCHEMA = 'empirical-lawhood/adapters/prepared-base-implementation-manifest'

    manifest_id: str
    source_files: tuple[PreparedBaseSourceFile, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        require_sorted_unique_strings(
            tuple(value.relative_path for value in self.source_files),
            field_name="source_files",
            allow_empty=False,
        )
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        expected = hashlib.sha256(
            canonical_json_bytes(
                tuple(
                    {
                        "relative_path": value.relative_path,
                        "sha256": value.sha256,
                        "size_bytes": value.size_bytes,
                    }
                    for value in self.source_files
                )
            )
        ).hexdigest()
        if self.implementation_sha256 != expected:
            raise ValueError("prepared-base implementation digest differs from source files")


def _config_suffix(config: PreparedBaseConfig) -> str:
    return config.config_id.removeprefix("config.")


@dataclass(frozen=True, slots=True)
class PreparedBaseAuthoringBundle:
    source_manifest: OpenSimulatorSourceManifest
    source_config: SourceMaterializationConfig
    capability_config: PreparedBaseConfig
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    qualification: MaterializationQualificationReceipt
    inventory: FormalGapSourceCapabilityInventory
    registry: CapabilityRegistry
    template: StudyTemplate
    draft: StudyDraft
    package: StudyDefinition
    formal_methods: FormalMethodCatalog
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


def _config_ref(config: PreparedBaseConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=config.config_id,
        config_schema=config.SCHEMA,
        config_schema_sha256=hashlib.sha256(config.SCHEMA.encode("ascii")).hexdigest(),
        content_sha256=config.fingerprint(),
        artifact_id=f"config-artifact.{config.config_id}",
    )


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        filename_suffix=".json",
    )


def _capability_key(stage: PreparedBaseStage, step_id: str) -> str:
    return f"open-sim.gym-torax-full-iter.prepared-base-{stage.value.lower()}-{step_id}"


def _step(
    *,
    config: PreparedBaseConfig,
    step_id: str,
    stage: ScientificStage,
    kind: CapabilityKind,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    permissions: tuple[CapabilityPermission, ...],
    outcome_access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligations: tuple[str, ...],
    budget: ResourceBudget,
) -> ProtocolStepTemplate:
    del kind
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=_capability_key(config.stage, step_id),
        capability_version=_VERSION,
        config=_config_ref(config),
        dependency_step_ids=dependencies,
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=tuple(sorted(permissions, key=lambda value: value.value)),
        requested_outcome_access=outcome_access,
        visibility_ceiling=visibility,
        resource_budget=budget,
        resource_lock_ids=("lock.open-sim.gym-torax-full-iter",),
        barrier=barrier,
        maximum_attempts=1,
        obligation_ids=tuple(sorted(obligations)),
    )


def _protocol(config: PreparedBaseConfig) -> ProtocolTemplate:
    evaluation = config.stage is PreparedBaseStage.EVALUATION
    final_access = (
        OutcomeAccess.EVALUATOR_REVEAL if evaluation else OutcomeAccess.DEVELOPMENT_VISIBLE
    )
    final_visibility = (
        VisibilityCeiling.OUTCOME_VISIBLE if evaluation else VisibilityCeiling.DEVELOPMENT_ONLY
    )
    acquisition_parent = _FREEZE_STEP_ID if evaluation else _SOURCE_STEP_ID
    steps = [
        _step(
            config=config,
            step_id=_SOURCE_STEP_ID,
            stage=ScientificStage.PREPARE,
            kind=CapabilityKind.SOURCE,
            dependencies=(),
            outputs=(_output(_SOURCE_OUTPUT_ID, OpenSimulatorSourceManifest.SCHEMA),),
            permissions=_READ_WRITE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=("prepared-response-exact-source-runtime",),
            budget=_SMALL_TASK,
        ),
    ]
    if evaluation:
        steps.append(
            _step(
                config=config,
                step_id=_FREEZE_STEP_ID,
                stage=ScientificStage.FREEZE,
                kind=CapabilityKind.TRANSFORM,
                dependencies=(_SOURCE_STEP_ID,),
                outputs=(
                    _output(
                        _FROZEN_OUTPUT_ID,
                        OpenSimulatorSourceManifest.SCHEMA,
                    ),
                ),
                permissions=_READ_WRITE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility=VisibilityCeiling.PROSPECTIVE,
                barrier=BarrierKind.FREEZE,
                obligations=("prepared-response-evaluation-design-frozen",),
                budget=_SMALL_TASK,
            )
        )
    steps.extend(
        (
            _step(
                config=config,
                step_id=_ACQUIRE_STEP_ID,
                stage=ScientificStage.ACQUIRE,
                kind=CapabilityKind.SIMULATOR,
                dependencies=(acquisition_parent,),
                outputs=(_output(_BATCH_OUTPUT_ID, PreparedBaseAcquisitionBatch.SCHEMA),),
                permissions=_DEVELOP,
                outcome_access=config.outcome_access,
                visibility=(
                    VisibilityCeiling.PROSPECTIVE
                    if evaluation
                    else VisibilityCeiling.DEVELOPMENT_ONLY
                ),
                barrier=BarrierKind.NONE,
                obligations=(
                    "prepared-response-action-clock-realization",
                    "prepared-response-paired-view-preparation-acquisition",
                    "prepared-response-partial-termination-disposition",
                ),
                budget=_ACQUISITION_TASK,
            ),
            _step(
                config=config,
                step_id=_REPORT_STEP_ID,
                stage=ScientificStage.EVALUATE if evaluation else ScientificStage.REPORT,
                kind=CapabilityKind.EVALUATOR if evaluation else CapabilityKind.REPORTER,
                dependencies=(_ACQUIRE_STEP_ID,),
                outputs=(
                    _output(
                        _EVALUATION_OUTPUT_ID if evaluation else _DEVELOPMENT_OUTPUT_ID,
                        (
                            EvaluationAdjudication.SCHEMA
                            if evaluation
                            else DevelopmentDecision.SCHEMA
                        ),
                    ),
                    _output(_SCIENTIFIC_OUTPUT_ID, ScientificAdjudicationRecord.SCHEMA),
                ),
                permissions=_EVALUATE if evaluation else _DEVELOP,
                outcome_access=final_access,
                visibility=final_visibility,
                barrier=BarrierKind.REVEAL if evaluation else BarrierKind.NONE,
                obligations=(
                    "prepared-response-frozen-development-selection"
                    if not evaluation
                    else "prepared-response-frozen-evaluation-adjudication",
                ),
                budget=_REPORT_TASK,
            ),
        )
    )
    return ProtocolTemplate(
        template_id=(f"protocol.{config.stage.value.lower()}.{_config_suffix(config)}"),
        template_version=_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _manifests(
    config: PreparedBaseConfig,
    implementation_sha256: str,
) -> tuple[CapabilityManifest, ...]:
    evaluation = config.stage is PreparedBaseStage.EVALUATION
    values: list[
        tuple[
            str,
            CapabilityKind,
            tuple[str, ...],
            tuple[str, ...],
            tuple[CapabilityPermission, ...],
            OutcomeAccess,
            EvidenceCeiling,
            ResourceBudget,
        ]
    ] = [
        (
            _SOURCE_STEP_ID,
            CapabilityKind.SOURCE,
            (OpenSimulatorSourceManifest.SCHEMA,),
            (OpenSimulatorSourceManifest.SCHEMA,),
            _READ_WRITE,
            OutcomeAccess.OUTCOME_BLIND,
            EvidenceCeiling.MEASUREMENT,
            _SMALL_TASK,
        ),
    ]
    if evaluation:
        values.append(
            (
                _FREEZE_STEP_ID,
                CapabilityKind.TRANSFORM,
                (OpenSimulatorSourceManifest.SCHEMA,),
                (OpenSimulatorSourceManifest.SCHEMA,),
                _READ_WRITE,
                OutcomeAccess.OUTCOME_BLIND,
                EvidenceCeiling.MEASUREMENT,
                _SMALL_TASK,
            )
        )
    values.extend(
        (
            (
                _ACQUIRE_STEP_ID,
                CapabilityKind.SIMULATOR,
                (OpenSimulatorSourceManifest.SCHEMA,),
                (PreparedBaseAcquisitionBatch.SCHEMA,),
                _DEVELOP,
                config.outcome_access,
                EvidenceCeiling.ORDER_RELATION,
                _ACQUISITION_TASK,
            ),
            (
                _REPORT_STEP_ID,
                CapabilityKind.EVALUATOR if evaluation else CapabilityKind.REPORTER,
                (PreparedBaseAcquisitionBatch.SCHEMA,),
                tuple(
                    sorted(
                        (
                            EvaluationAdjudication.SCHEMA
                            if evaluation
                            else DevelopmentDecision.SCHEMA,
                            ScientificAdjudicationRecord.SCHEMA,
                        )
                    )
                ),
                _EVALUATE if evaluation else _DEVELOP,
                (
                    OutcomeAccess.EVALUATOR_REVEAL
                    if evaluation
                    else OutcomeAccess.DEVELOPMENT_VISIBLE
                ),
                EvidenceCeiling.ORDER_RELATION,
                _REPORT_TASK,
            ),
        )
    )
    return tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=_capability_key(config.stage, step_id),
                    capability_version=_VERSION,
                    kind=kind,
                    config_schema=PreparedBaseConfig.SCHEMA,
                    config_schema_sha256=hashlib.sha256(
                        PreparedBaseConfig.SCHEMA.encode("ascii")
                    ).hexdigest(),
                    input_schema_ids=tuple(sorted(inputs)),
                    output_schema_ids=tuple(sorted(outputs)),
                    permissions=tuple(sorted(permissions, key=lambda value: value.value)),
                    maximum_evidence_ceiling=ceiling,
                    maximum_outcome_access=access,
                    resource_ceiling=budget,
                    deterministic=False,
                    seed_required=False,
                    language_id="python",
                    runtime_id="runtime.cpu-jax-float64",
                    requires_clean_commit=False,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=(
                        f"prepared-response-{config.stage.value.lower()}-{step_id}-exact-config",
                        f"prepared-response-{config.stage.value.lower()}-{step_id}-path-free",
                    ),
                    implementation_sha256=implementation_sha256,
                )
                for (
                    step_id,
                    kind,
                    inputs,
                    outputs,
                    permissions,
                    access,
                    ceiling,
                    budget,
                ) in values
            ),
            key=lambda value: value.registry_id,
        )
    )


def _graph(
    *,
    config: PreparedBaseConfig,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source: OpenSimulatorSourceManifest,
) -> CandidateScientificGraph:
    source_input = CandidateGraphExternalInput(
        input_id=source.source_id,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=f"artifact.{source.source_id}",
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=source.fingerprint(),
        payload_schema=OpenSimulatorSourceManifest.SCHEMA,
        media_type="application/json",
        maximum_size_bytes=1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    manifests = {value.registry_id: value for value in registry.capabilities}
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=manifests[
                f"{step.capability_key}@{step.capability_version}"
            ].implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    steps = {value.step_id: value for value in protocol.steps}
    edges = [
        CandidateGraphEdge(
            edge_id=f"edge.external.{source.source_id}.{_SOURCE_STEP_ID}",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=source.source_id,
            consumer_node_id=_SOURCE_STEP_ID,
            consumer_input_id="held-source-manifest",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=f"artifact.{source.source_id}",
            payload_schema=OpenSimulatorSourceManifest.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=1024**2,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        )
    ]
    for step in protocol.steps:
        for parent_id in step.dependency_step_ids:
            parent = steps[parent_id]
            output = parent.outputs[0]
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.{parent_id}.{step.step_id}",
                    producer_node_id=parent_id,
                    producer_output_id=output.output_id,
                    external_input_id=None,
                    consumer_node_id=step.step_id,
                    consumer_input_id=f"input-{parent_id}",
                    scientific_role=ScientificInputRole.OUTCOME,
                    logical_artifact_id=f"artifact.{parent_id}.{output.output_id}",
                    payload_schema=output.payload_schema,
                    media_type=output.media_type,
                    maximum_size_bytes=parent.resource_budget.output_bytes,
                    outcome_access=parent.requested_outcome_access,
                    visibility_ceiling=parent.visibility_ceiling,
                    barrier=step.barrier,
                )
            )
    return CandidateScientificGraph(
        graph_id=f"graph.{config.stage.value.lower()}.{_config_suffix(config)}",
        external_inputs=(source_input,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _obligation_coverage(
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
    config: PreparedBaseConfig,
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    fallback = (
        _REPORT_STEP_ID,
        next(value for value in protocol.steps if value.step_id == _REPORT_STEP_ID)
        .outputs[0]
        .output_id,
    )
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation_id,
            proof_owner_node_id=owners.get(obligation_id, fallback)[0],
            required_output_id=owners.get(obligation_id, fallback)[1],
            contributor_edge_ids=incoming[owners.get(obligation_id, fallback)[0]],
        )
        for obligation_id in required_candidate_obligation_ids(experiment, protocol)
    )
    return ObligationCoverage(
        coverage_id=(f"obligation-coverage.{config.stage.value.lower()}.{_config_suffix(config)}"),
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def _prepared_source_system() -> SystemSpec:
    base = open_simulator_system(OpenSimulatorKind.GYM_TORAX)
    view_by_level = {
        0: (
            PREPARED_SOURCE_PRIMARY_VIEW_ID,
            Decimal("1"),
            25,
            10,
        ),
        1: (
            PREPARED_SOURCE_REFINED_VIEW_ID,
            Decimal("0.5"),
            33,
            20,
        ),
    }
    views = []
    for base_view in base.numerical_views:
        level = base_view.coordinates[0].refinement_level
        view_id, timestep, radial_cells, corrector_steps = view_by_level[level]
        coordinates = (
            NumericalCoordinateSpec(
                coordinate_id=f"coordinate.{view_id}.corrector-steps",
                kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                value=Decimal(corrector_steps),
                unit="1",
                refinement_level=level,
            ),
            NumericalCoordinateSpec(
                coordinate_id=f"coordinate.{view_id}.radial-cells",
                kind=NumericalCoordinateKind.SPATIAL_GRID,
                value=Decimal(radial_cells),
                unit="cells",
                refinement_level=level,
            ),
            NumericalCoordinateSpec(
                coordinate_id=f"coordinate.{view_id}.timestep",
                kind=NumericalCoordinateKind.TIMESTEP,
                value=timestep,
                unit="s",
                refinement_level=level,
            ),
        )
        views.append(
            replace(
                base_view,
                view_id=view_id,
                coordinates=tuple(sorted(coordinates, key=lambda value: value.coordinate_id)),
                solver_id="solver.torax.linear-theta-predictor-corrector",
            )
        )
    view_ids = tuple(sorted(value.view_id for value in views))
    components = tuple(
        replace(value, numerical_view_ids=view_ids) if value.numerical_view_ids else value
        for value in base.components
    )
    return replace(
        base,
        label="Gym--TORAX frozen preparation-local paired-view base",
        components=components,
        numerical_views=tuple(sorted(views, key=lambda value: value.view_id)),
        authority_policy=replace(
            base.authority_policy,
            required_gate_ids=(
                "exact-source-closure",
                "qualified-cpu-float64-route",
                "separate-execution-and-reveal-authority",
            ),
        ),
    )


def _prepared_source_experiment(
    *,
    system: SystemSpec,
    config: PreparedBaseConfig,
) -> ExperimentSpec:
    base = open_simulator_experiment(OpenSimulatorKind.GYM_TORAX, system)
    claim = replace(
        base.claims[0],
        claim_id='claim.prepared-base.preparation-local-measurement-to-candidate-family',
        proposition=(
            "The frozen preparation word satisfies the complete q/H98/fGW "
            "preparation-local recurrence contract in both nested numerical views."
        ),
        estimand=(
            "Preparation-unit cross-view all-predicate recurrence with exact "
            "binomial uncertainty and a confirmatory paired-view difference."
        ),
        requested_rung=EvidenceRung.ORDER_RELATION,
        evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
        promotion_rule=(
            "Only a separately issued fresh evaluation may support this order relation "
            "preparation-base claim; controller admission and prospective controller evaluation are not tested."
        ),
        assumption_ids=(
            "cpu-float64-source-runtime-closure-exact",
            "nested-numerical-views-not-independent-units",
            "predeclared-development-learning-rule",
        ),
        numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
    )
    controls = tuple(
        replace(
            value,
            capability_key=_capability_key(config.stage, _ACQUIRE_STEP_ID),
            decisive_rule=(
                "The installed reference is nonselectable and every candidate "
                "uses the exact source-native action and clock ledger."
            ),
        )
        for value in base.controls
    )
    obligations = replace(
        base.obligations,
        obligations_id='obligations.prepared-base.preparation-base',
        support=replace(
            base.obligations.support,
            support_id='support.prepared-base.preparation-local-box',
            physical_unit_count=(
                24
                if config.stage is PreparedBaseStage.DEVELOPMENT
                else len(config.primary_cell_ids)
            ),
            nested_numerical_view_count=2,
            information_cutoff_id=base.information_cutoffs[0].cutoff_id,
            denominator_cell_ids=('cell.prepared-base.local-parameter-box',),
        ),
        validity=replace(
            base.obligations.validity,
            validity_id='validity.prepared-base.complete-temporal-contract',
            validity_domain_ids=('domain.prepared-base.q-h98-fgw-preservation',),
            assumption_ids=(
                "simulator-denominator-is-response-medium",
                "source-runtime-backend-closure-exact",
            ),
        ),
        uncertainty=replace(
            base.obligations.uncertainty,
            uncertainty_id='uncertainty.prepared-base.preparation-unit-exact-binomial',
            method_key="clopper-pearson-preparation-unit",
            confidence_level=Decimal("0.95"),
            limitation_codes=(
                "CLOCKS_NOT_REPLICATION",
                "NESTED_VIEWS_NOT_REPLICATION",
                "SIMULATOR_PARAMETER_LOCAL",
            ),
        ),
        falsifiers=tuple(
            replace(
                value,
                falsifier_id='falsifier.prepared-base.action-clock-view-contract',
                capability_key=_capability_key(config.stage, _ACQUIRE_STEP_ID),
                description=(
                    "Wrong action occurrence, delayed clock, clipped delivery, "
                    "or mismatched paired preparation invalidates the affected route."
                ),
                decisive_rule=(
                    "Preserve a typed terminal result; technical replacement is "
                    "allowed only before evaluator access under the frozen reserve rule."
                ),
            )
            for value in base.obligations.falsifiers
        ),
        closure=replace(
            base.obligations.closure,
            closure_id='closure.prepared-base.preparation-local-history',
            recurrence_cell_ids=('cell.prepared-base.local-parameter-box',),
        ),
        structural_convergence=replace(
            base.obligations.structural_convergence,
            convergence_id='convergence.prepared-base.paired-preparation-base',
            required_structure_ids=("preparation-base-recurrence",),
            numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
        ),
        computability=replace(
            base.obligations.computability,
            computability_id='computability.prepared-base.cpu-x64',
            numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
            evidence_link_ids=('qualification.prepared-base.gym-torax-paired-route',),
        ),
    )
    evaluation_ids = tuple(sorted(config.primary_cell_ids))
    reveal = replace(
        base.reveal_barrier,
        barrier_id=f"reveal.{_config_suffix(config)}",
        development_unit_ids=tuple(sorted(value.cell_id for value in development_cells())),
        evaluation_cohort_id=(f"cohort.{_config_suffix(config)}"),
        evaluation_manifest_sha256=hashlib.sha256(
            "\n".join(evaluation_ids).encode("ascii")
        ).hexdigest(),
        sealed_outcome_artifact_ids=(f"artifact.{_ACQUIRE_STEP_ID}.{_BATCH_OUTPUT_ID}",),
    )
    return replace(
        base,
        experiment_id=f"experiment.{_config_suffix(config)}",
        claims=(claim,),
        controls=tuple(sorted(controls, key=lambda value: value.control_id)),
        precision_goals=tuple(
            replace(
                value,
                goal_id='precision.prepared-base.frozen-finite-roster',
                metric_id="preparation-unit-exact-binomial-and-paired-view",
                target_width=Decimal("0.2"),
                maximum_independent_units=24,
                stopping_rule=(
                    "Use the frozen finite roster and reserves only for blinded "
                    "technical bundle failures; never outcome-retune."
                ),
            )
            for value in base.precision_goals
        ),
        reveal_barrier=reveal,
        obligations=obligations,
        design_visibility_ceiling=(
            VisibilityCeiling.DEVELOPMENT_ONLY
            if config.stage is PreparedBaseStage.EVALUATION
            else VisibilityCeiling.PROSPECTIVE
        ),
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
        authorization_record_id=None,
        predecessor_experiment_ids=(),
    )


def build_prepared_base_authoring_bundle(
    *,
    config: PreparedBaseConfig,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
    inventory: FormalGapSourceCapabilityInventory,
    implementation_sha256: str,
    development_decision: DevelopmentDecision | None = None,
) -> PreparedBaseAuthoringBundle:
    """Build one exact development or fresh-evaluation standard candidate."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if config.formal_coverage != ObjectIdentity.from_record(coverage.coverage_id, coverage):
        raise ValueError("prepared-base config and formal coverage differ")
    if coverage.denominator_id != inventory.denominator_id:
        raise ValueError("prepared-base formal inventory denominator differs")
    if (config.stage is PreparedBaseStage.EVALUATION) != (development_decision is not None):
        raise ValueError("fresh evaluation requires exactly one development decision")

    base_bundle = build_open_simulator_authoring_bundle(
        kind=OpenSimulatorKind.GYM_TORAX,
        register=register,
        implementation_sha256=implementation_sha256,
    )
    source = base_bundle.source_manifest
    if config.source_manifest != ObjectIdentity.from_record(source.source_id, source):
        raise ValueError("prepared-base config binds another Gym--TORAX source")
    system = _prepared_source_system()
    experiment = _prepared_source_experiment(system=system, config=config)
    base_campaign = open_simulator_campaign(OpenSimulatorKind.GYM_TORAX, system, experiment)
    node = replace(
        base_campaign.nodes[0],
        node_id=f"campaign-node.prepared-base.{config.stage.value.lower()}",
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
    )
    campaign = replace(
        base_campaign,
        campaign_id=f"campaign.{_config_suffix(config)}",
        objective=(
            "Execute only the frozen Gym--TORAX preparation-local measurement/order-relation "
            f"{config.stage.value.lower()} act."
        ),
        target_claim_ids=tuple(claim.claim_id for claim in experiment.claims),
        budget=_PROGRAMME_BUDGET,
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
    )
    source_config = SourceMaterializationConfig(
        config_id='source-config.prepared-base.gym-torax-held',
        source_id=source.source_id,
        role=SourceMaterializationRole.PREPARED_MEDIUM,
        content_sha256=source.fingerprint(),
        expected_size_bytes=len(source.canonical_bytes()),
        maximum_bytes=1024**2,
        payload_schema=OpenSimulatorSourceManifest.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    observation_operator = ObjectIdentity(
        object_id='observer.prepared-base.gym-torax-prepared-base',
        object_schema='empirical-lawhood/simulators/prepared-tokamak-response/observation-operator',
        object_version=_VERSION,
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id='qualification.prepared-base.gym-torax-held-source',
        source_id=source.source_id,
        materialization=ObjectIdentity.from_record(source.source_id, source),
        content_sha256=source.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
        native_unit_ids=tuple(sorted({value.native_unit for value in system.quantities})),
        frame_ids=tuple(sorted({value.coordinate_frame for value in system.quantities})),
        clock_ids=tuple(value.clock_id for value in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    protocol = _protocol(config)
    manifests = _manifests(config, implementation_sha256)
    registry = CapabilityRegistry(
        registry_id=(f"registry.prepared-base.{config.stage.value.lower()}.{config.fingerprint()[:24]}"),
        capabilities=manifests,
    )
    graph = _graph(
        config=config,
        protocol=protocol,
        registry=registry,
        source=source,
    )
    obligation_coverage = _obligation_coverage(
        experiment,
        protocol,
        graph,
        config,
    )
    template = StudyTemplate(
        template_key=(
            f"open-sim.gym-torax-full-iter.prepared-base-{config.stage.value.lower()}-prepared-base"
        ),
        template_version=_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=obligation_coverage,
    )
    design_inputs = [
        DesignInputRecord(
            input_id='design-input.prepared-base.frozen-parent',
            object_identity=ObjectIdentity.from_record(config.parent_design_id, config),
            materialization_sha256=config.fingerprint(),
            information_cutoff=experiment.information_cutoffs[0],
            role=DesignInputRole.READINESS_METADATA,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            operator_id="human.project-owner",
        )
    ]
    if development_decision is not None:
        design_inputs.append(
            DesignInputRecord(
                input_id='design-input.prepared-base.development-learning-rule-output',
                object_identity=ObjectIdentity.from_record(
                    development_decision.decision_id,
                    development_decision,
                ),
                materialization_sha256=development_decision.fingerprint(),
                information_cutoff=experiment.information_cutoffs[0],
                role=DesignInputRole.DEVELOPMENT_TUNING,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
                operator_id="human.project-owner",
                physical_unit_ids=tuple(sorted(value.cell_id for value in development_cells())),
            )
        )
    input_values = tuple(sorted(design_inputs, key=lambda value: value.input_id))
    draft_id = f"draft.{_config_suffix(config)}"
    origin = DesignOrigin(
        origin_id=f"origin.{draft_id.removeprefix('draft.')}",
        kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
        declared_input_ids=tuple(value.input_id for value in input_values),
        parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        nomination=None,
        requests_fresh_child=False,
    )
    all_development_units = tuple(sorted(value.cell_id for value in development_cells()))
    if config.stage is PreparedBaseStage.EVALUATION:
        all_evaluation_units = tuple(sorted(value.cell_id for value in config.cells))
    else:
        all_evaluation_units = tuple(sorted(value.cell_id for value in evaluation_master_cells()))
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the selected exact Gym--TORAX preparation word support the "
            "frozen preparation-local recurrence base across paired numerical views?"
        ),
        alternative_ids=tuple(
            sorted(
                (
                    'alternative.prepared-base.supported-cross-view-base',
                    'alternative.prepared-base.view-sensitive-base',
                    'alternative.prepared-base.no-supported-preparation',
                    'alternative.prepared-base.partial-or-terminated',
                    'alternative.prepared-base.technical-observation-failure',
                    'alternative.prepared-base.unevaluable-operand',
                )
            )
        ),
        design_origin=origin,
        design_inputs=input_values,
        development_unit_ids=all_development_units,
        evaluation_unit_ids=all_evaluation_units,
        development_seed_ids=tuple(sorted(f"seed.{value}" for value in all_development_units)),
        evaluation_seed_ids=tuple(sorted(f"seed.{value}" for value in all_evaluation_units)),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in registry.capabilities
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=source.source_id,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                evidence_world_id=system.world.world_id,
                materialization=ObjectIdentity.from_record(source.source_id, source),
                content_sha256=source.fingerprint(),
                source_config_sha256=source_config.fingerprint(),
                observation_operator=observation_operator,
                numerical_view_ids=(PREPARED_SOURCE_PRIMARY_VIEW_ID, PREPARED_SOURCE_REFINED_VIEW_ID),
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id,
                    qualification,
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=_PROGRAMME_BUDGET,
        conditional_successor=None,
    )
    entry = _entry_package(
        definition=base_bundle.definition,
        draft=draft,
        register=register,
        coverage=coverage,
    )
    package = StudyDefinition(
        package_id=(f"programme-authoring-package.{_config_suffix(config)}"),
        draft=draft,
        entry_package=entry,
    )
    return PreparedBaseAuthoringBundle(
        source_manifest=source,
        source_config=source_config,
        capability_config=config,
        system=system,
        experiment=experiment,
        campaign=campaign,
        qualification=qualification,
        inventory=inventory,
        registry=registry,
        template=template,
        draft=draft,
        package=package,
        formal_methods=open_simulator_formal_catalog(register),
        implementation_sha256=implementation_sha256,
    )


def compile_prepared_base_bundle(
    bundle: PreparedBaseAuthoringBundle,
) -> StudyCompilationReport:
    raw = bundle.package.canonical_bytes()
    raw_sha256 = hashlib.sha256(b"application/json\x00" + raw).hexdigest()
    context = CandidateCompilationContext(
        context_id=(
            f"context.prepared-base.{bundle.capability_config.stage.value.lower()}."
            f"{bundle.capability_config.fingerprint()[:24]}"
        ),
        registry=bundle.registry,
        templates=(bundle.template,),
        qualifications=(bundle.qualification,),
        known_design_inputs=bundle.draft.design_inputs,
        implementation_sha256=bundle.implementation_sha256,
    )
    standard = StandardCandidateCompilationContext(
        context_id=f"standard.{context.context_id}",
        base=context,
        formal_methods=bundle.formal_methods,
        source_inventories=(bundle.inventory,),
    )
    return compile_study_candidate(
        authoring_package=bundle.package,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(raw),
            raw_materialization_sha256=raw_sha256,
        ),
        context=standard,
    )


class PreparedBaseConfigDecoder:
    """Static exact-config validator for one issued prepared-base bundle."""

    def __init__(self, manifest: CapabilityManifest, config: PreparedBaseConfig) -> None:
        self.manifest = manifest
        self.config = config

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        if expected_schema != PreparedBaseConfig.SCHEMA:
            raise ValueError("prepared-base config schema differs")
        observed = decode_canonical_bytes(
            payload,
            PreparedBaseConfig,
            maximum_bytes=16 * 1024**2,
        )
        if observed != self.config:
            raise ValueError("prepared-base config differs from static registration")


class PreparedBaseTaskRunner:
    """Path-free prepared-base source, acquisition and adjudication runner."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        bundle: PreparedBaseAuthoringBundle,
        acquirer: Callable[..., PreparedBaseAcquisitionBatch] = acquire_stage,
    ) -> None:
        self.manifest = manifest
        self.bundle = bundle
        self.acquirer = acquirer
        self.step_id = manifest.capability_key.rsplit("-", 1)[-1]

    @staticmethod
    def _decode_config(context: TaskContext) -> PreparedBaseConfig:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        ports = tuple(
            value
            for value in context.input_ports
            if value.payload_schema == PreparedBaseConfig.SCHEMA
        )
        if len(ports) != 1:
            raise ValueError("prepared-base task requires one exact config")
        return decode_canonical_bytes(
            ports[0].read(),
            PreparedBaseConfig,
            maximum_bytes=ports[0].size_bytes,
        )

    @staticmethod
    def _output_id(context: TaskContext, payload_schema: str) -> str:
        matches = tuple(
            value.output_id
            for value in context.output_ports
            if value.payload_schema == payload_schema
        )
        if len(matches) != 1:
            raise ValueError("prepared-base output schema does not resolve one frozen port")
        return matches[0]

    def execute(self, context: TaskContext) -> RunnerResult:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        config = self._decode_config(context)
        if config != self.bundle.capability_config:
            raise ValueError("prepared-base runtime config differs from issued bundle")
        if self.step_id == _SOURCE_STEP_ID:
            sources = tuple(
                value
                for value in context.input_ports
                if value.payload_schema == OpenSimulatorSourceManifest.SCHEMA
            )
            if len(sources) != 1:
                raise ValueError("prepared-base source task requires one source manifest")
            source = decode_canonical_bytes(
                sources[0].read(),
                OpenSimulatorSourceManifest,
                maximum_bytes=sources[0].size_bytes,
            )
            if source != self.bundle.source_manifest:
                raise ValueError("prepared-base source manifest identity changed")
            return RunnerResult(
                outputs=(
                    TaskOutputPayload(
                        output_id=self._output_id(
                            context,
                            OpenSimulatorSourceManifest.SCHEMA,
                        ),
                        payload=source.canonical_bytes(),
                    ),
                ),
                checks=(ReceiptCheck("prepared-response-source-manifest-exact", True, ()),),
            )
        if self.step_id == _FREEZE_STEP_ID:
            sources = tuple(
                value
                for value in context.input_ports
                if value.payload_schema == OpenSimulatorSourceManifest.SCHEMA
            )
            if len(sources) != 1:
                raise ValueError("prepared-base freeze requires one source receipt")
            source = decode_canonical_bytes(
                sources[0].read(),
                OpenSimulatorSourceManifest,
                maximum_bytes=sources[0].size_bytes,
            )
            if source != self.bundle.source_manifest:
                raise ValueError("prepared-base frozen source identity changed")
            return RunnerResult(
                outputs=(
                    TaskOutputPayload(
                        output_id=self._output_id(
                            context,
                            OpenSimulatorSourceManifest.SCHEMA,
                        ),
                        payload=source.canonical_bytes(),
                    ),
                ),
                checks=tuple(
                    sorted(
                        (
                            ReceiptCheck(
                                "prepared-response-evaluation-design-frozen",
                                True,
                                (),
                            ),
                            ReceiptCheck(
                                "prepared-response-development-selection-bound",
                                True,
                                (),
                            ),
                        ),
                        key=lambda value: value.check_id,
                    )
                ),
            )
        if self.step_id == _ACQUIRE_STEP_ID:
            sources = tuple(
                value
                for value in context.input_ports
                if value.payload_schema == OpenSimulatorSourceManifest.SCHEMA
            )
            if len(sources) != 1:
                raise ValueError("prepared-base acquisition lacks source receipt")
            source = decode_canonical_bytes(
                sources[0].read(),
                OpenSimulatorSourceManifest,
                maximum_bytes=sources[0].size_bytes,
            )
            if source != self.bundle.source_manifest:
                raise ValueError("prepared-base acquisition source differs")
            batch = self.acquirer(config=config)
            return RunnerResult(
                outputs=(
                    TaskOutputPayload(
                        output_id=self._output_id(
                            context,
                            PreparedBaseAcquisitionBatch.SCHEMA,
                        ),
                        payload=batch.canonical_bytes(),
                    ),
                ),
                checks=(
                    ReceiptCheck("prepared-response-active-roster-typed", True, ()),
                    ReceiptCheck("prepared-response-cpu-x64-no-fallback", True, ()),
                    ReceiptCheck("prepared-response-reserve-rule-outcome-blind", True, ()),
                ),
            )
        batches = tuple(
            decode_canonical_bytes(
                value.read(),
                PreparedBaseAcquisitionBatch,
                maximum_bytes=value.size_bytes,
            )
            for value in context.input_ports
            if value.payload_schema == PreparedBaseAcquisitionBatch.SCHEMA
        )
        if len(batches) != 1:
            raise ValueError("prepared-base adjudicator requires one acquisition batch")
        batch = batches[0]
        if batch.config != ObjectIdentity.from_record(config.config_id, config):
            raise ValueError("prepared-base adjudicator batch config differs")
        if config.stage is PreparedBaseStage.DEVELOPMENT:
            development_result = adjudicate_development(
                config=config,
                episodes=batch.episodes,
                active_cell_ids=batch.active_cell_ids,
            )
            result: CanonicalRecord = development_result
            evaluability = (
                AdjudicationEvaluability.UNEVALUABLE
                if development_result.disposition == "TECHNICAL_OBSERVATION_FAILURE"
                else AdjudicationEvaluability.EVALUABLE
            )
            scientific_status = (
                ScientificStatus.UNEVALUABLE
                if evaluability is AdjudicationEvaluability.UNEVALUABLE
                else ScientificStatus.PARTIAL
            )
            reasons = tuple(
                sorted(
                    {
                        f"DEVELOPMENT_{development_result.disposition}",
                        *development_result.reason_codes,
                    }
                )
            )
            outcome_access = OutcomeAccess.DEVELOPMENT_VISIBLE
            visibility = VisibilityCeiling.DEVELOPMENT_ONLY
        else:
            evaluation_result = adjudicate_evaluation(
                config=config,
                episodes=batch.episodes,
                active_cell_ids=batch.active_cell_ids,
            )
            result = evaluation_result
            evaluability = (
                AdjudicationEvaluability.UNEVALUABLE
                if evaluation_result.primary_result
                in {
                    PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE,
                    PreparedBasePrimaryResult.UNEVALUABLE_OPERAND,
                }
                else AdjudicationEvaluability.EVALUABLE
            )
            scientific_status = {
                PreparedBasePrimaryResult.SUPPORTED_CROSS_VIEW_BASE: ScientificStatus.SUPPORTED,
                PreparedBasePrimaryResult.VIEW_SENSITIVE_BASE: ScientificStatus.MIXED,
                PreparedBasePrimaryResult.NO_SUPPORTED_PREPARATION: ScientificStatus.NOT_SUPPORTED,
                PreparedBasePrimaryResult.PARTIAL_OR_TERMINATED: ScientificStatus.PARTIAL,
                PreparedBasePrimaryResult.TECHNICAL_OBSERVATION_FAILURE: ScientificStatus.UNEVALUABLE,
                PreparedBasePrimaryResult.UNEVALUABLE_OPERAND: ScientificStatus.UNEVALUABLE,
            }[evaluation_result.primary_result]
            reasons = tuple(
                sorted(
                    {
                        f"PREPARED_RESPONSE_{evaluation_result.primary_result.value}",
                        *evaluation_result.reason_codes,
                    }
                )
            )
            outcome_access = OutcomeAccess.EVALUATION_REVEALED
            visibility = VisibilityCeiling.OUTCOME_VISIBLE
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError("prepared-base scientific adjudication context is absent")
        output_ids = tuple(
            sorted(
                value.logical_artifact_id
                for value in context.output_ports
                if value.logical_artifact_id is not None
            )
        )
        scientific = ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication_context.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication_context.evidence_world_id,
            evidence_world_kind=adjudication_context.evidence_world_kind,
            relation=adjudication_context.relation,
            independent_unit_id=adjudication_context.independent_unit_id,
            information_cutoffs=adjudication_context.information_cutoffs,
            visibility_ceiling=visibility,
            outcome_access=outcome_access,
            evaluability=evaluability,
            scientific_status=scientific_status,
            admission_status=(
                AdmissionStatus.UNEVALUABLE
                if evaluability is AdjudicationEvaluability.UNEVALUABLE
                else AdmissionStatus.NOT_EVALUATED
            ),
            reason_codes=reasons or ("PREPARED_RESPONSE_TYPED_TERMINAL",),
        )
        return RunnerResult(
            outputs=tuple(
                sorted(
                    (
                        TaskOutputPayload(
                            output_id=self._output_id(context, result.SCHEMA),
                            payload=result.canonical_bytes(),
                        ),
                        TaskOutputPayload(
                            output_id=self._output_id(
                                context,
                                ScientificAdjudicationRecord.SCHEMA,
                            ),
                            payload=scientific.canonical_bytes(),
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            checks=(
                ReceiptCheck("prepared-response-adjudication-rule-frozen", True, ()),
                ReceiptCheck("prepared-response-controller-admission-prospective-controller-evaluation-not-tested", True, ()),
            ),
        )

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        "Keep canonical prepared-base batches off the bounded worker-result pipe."

        result = self.execute(context)
        for output in result.outputs:
            if not isinstance(output, TaskOutputPayload):
                raise TypeError("prepared-base runner produced an unexpected output handle")
            for offset in range(0, len(output.payload), 64 * 1024):
                emitter.write(
                    output.output_id,
                    output.payload[offset : offset + 64 * 1024],
                )
        return result.checks


class PreparedBaseCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Exact issued-registry provider for one prepared-base candidate."""

    issued_source_schema_ids = (OpenSimulatorSourceManifest.SCHEMA,)

    def __init__(
        self,
        bundle: PreparedBaseAuthoringBundle,
        *,
        registry: CapabilityRegistry | None = None,
        acquirer: Callable[..., PreparedBaseAcquisitionBatch] = acquire_stage,
    ) -> None:
        self.bundle = bundle
        self.acquirer = acquirer
        self.registry = bundle.registry if registry is None else registry
        if self.registry.capabilities != bundle.registry.capabilities:
            raise ValueError("prepared-base runtime registry capabilities differ")
        self.registry_sha256 = self.registry.fingerprint()
        self.capability_count = len(self.registry.capabilities)

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("prepared-base provider registry identity differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        self._validate_registry(registry)
        return tuple(
            PreparedBaseTaskRunner(manifest, self.bundle, self.acquirer)
            for manifest in registry.capabilities
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        del source_records
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("prepared-base execution plan registry differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        values = []
        config_artifact_ids = {task.capability.config.artifact_id for task in plan.tasks}
        for artifact_id, spec in sorted(specs.items()):
            if artifact_id in config_artifact_ids:
                record: CanonicalRecord = self.bundle.capability_config
                parent_access = OutcomeAccess.OUTCOME_BLIND
                parent_visibility = VisibilityCeiling.PROSPECTIVE
            elif artifact_id == self.bundle.source_manifest.source_id:
                record = self.bundle.source_manifest
                parent_access = OutcomeAccess.OUTCOME_BLIND
                parent_visibility = VisibilityCeiling.PROSPECTIVE
            else:
                raise ValueError("prepared-base plan requests an unknown external input")
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(
                    (
                        self.bundle.capability_config.config_id
                        if isinstance(record, PreparedBaseConfig)
                        else self.bundle.source_manifest.source_id
                    ),
                    record,
                ),
                visibility_ceiling=parent_visibility,
                outcome_access=parent_access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=spec.expected_content_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._validate_registry(registry)
        record_types: dict[str, type[Any]] = {
            OpenSimulatorSourceManifest.SCHEMA: OpenSimulatorSourceManifest,
            PreparedBaseAcquisitionBatch.SCHEMA: PreparedBaseAcquisitionBatch,
            DevelopmentDecision.SCHEMA: DevelopmentDecision,
            EvaluationAdjudication.SCHEMA: EvaluationAdjudication,
            ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
        }
        values = tuple(
            sorted(
                (
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(field.name for field in fields(record_types[schema]))
                        ),
                    )
                    for manifest in registry.capabilities
                    for schema in manifest.output_schema_ids
                ),
                key=lambda value: value.key,
            )
        )
        if execution_plan is None:
            return values
        planned = {
            (
                task.capability.capability_key,
                task.capability.capability_version,
                output.payload_schema,
                output.profile,
            )
            for task in execution_plan.tasks
            for output in task.outputs
        }
        return tuple(value for value in values if value.key in planned)

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        del execution_plan
        self._validate_registry(registry)
        manifest = registry.resolve(
            _capability_key(self.bundle.capability_config.stage, _REPORT_STEP_ID),
            _VERSION,
        )
        return ScientificAdjudicationOutputContract(
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            output_id=f"{_REPORT_STEP_ID}.{_SCIENTIFIC_OUTPUT_ID}",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = [
    "PreparedBaseAuthoringBundle",
    "PreparedBaseCampaignRuntimeProvider",
    "PreparedBaseConfigDecoder",
    "PreparedBaseImplementationManifest",
    "PreparedBaseSourceFile",
    "PreparedBaseTaskRunner",
    "build_prepared_base_authoring_bundle",
    "compile_prepared_base_bundle",
]
