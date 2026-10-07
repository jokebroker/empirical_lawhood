"""Noncontact candidate template for the nine-task retained development analysis."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentCloseoutConfig, ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentDevelopmentResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_continuation import RETAINED_ANALYSIS_PREFIX, RETAINED_ANALYSIS_RUN, ResponseGeometryDevelopmentAnalysisContinuationConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentSupportResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import DEVELOPMENT_VALIDATION_SCHEMA
from empirical_lawhood.adapters.methods.retained_response_geometry_analysis.executable_binding import RETAINED_ANALYSIS_BINDINGS
from empirical_lawhood.adapters.methods.retained_response_geometry_analysis.extension_bundle import RETAINED_ANALYSIS_CAPABILITIES, RETAINED_ANALYSIS_ROLES
from empirical_lawhood.adapters.simulators.response_geometry_prospective.executable_binding import _config_ref, _outputs
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)

from .design import _entry
from .development_authoring import ResponseGeometryDevelopmentAuthoringBundle


def build_response_geometry_development_analysis_template(
    continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    """Author dependency and imported-measurement edges before plan lowering.

    Every estimator, schema, split and scientific owner is reused. Native and
    projection capabilities/tasks are absent; the original panel is an input.
    This function does not read inputs, construct runners or issue anything.
    """
    manifests = dict(zip(RETAINED_ANALYSIS_ROLES, RETAINED_ANALYSIS_CAPABILITIES, strict=True))
    if any(
        registry.resolve(value.capability_key, value.capability_version) != value
        for value in manifests.values()
    ):
        raise ValueError("development analysis template changes its installed method identities")
    qualification = continuation.qualification
    steps: dict[str, ProtocolStepTemplate] = {}
    prefix = RETAINED_ANALYSIS_PREFIX

    def add(
        task: str,
        stage: ScientificStage,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        config_id: str,
        dependencies: tuple[str, ...],
        reports: tuple[tuple[str, str], ...],
        data: tuple[str, str] | None,
        output_bytes: int,
        seconds: int,
    ) -> None:
        scan = len(config.canonical_bytes()) + sum(
            row.artifact.size_bytes for row in continuation.inputs_for_task(task)
        )
        if ".fit." in task or ".assess." in task:
            scan += len(continuation.canonical_bytes())
        scan += sum(
            steps[dependency].resource_budget.output_bytes
            for dependency in dependencies
        )
        if (
            scan > manifest.resource_ceiling.source_scan_bytes
            or output_bytes > manifest.resource_ceiling.output_bytes
        ):
            raise ValueError(
                "development retained analysis exceeds its installed input/output bound"
            )
        steps[task] = ProtocolStepTemplate(
            task,
            stage,
            manifest.capability_key,
            manifest.capability_version,
            _config_ref(
                config, ObjectIdentity.from_record(config_id, config), manifest
            ),
            tuple(sorted(dependencies)),
            _outputs((*reports, ("stage", LinkedCampaignStageEnvelope.SCHEMA)), data),
            manifest.permissions,
            OutcomeAccess.EVALUATION_REVEALED
            if stage is ScientificStage.EVALUATE
            else OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            replace(
                manifest.resource_ceiling,
                source_scan_bytes=scan,
                output_bytes=output_bytes,
                wall_time_seconds=seconds,
            ),
            (),
            BarrierKind.REVEAL
            if stage is ScientificStage.EVALUATE
            else BarrierKind.FREEZE,
            1,
            (f"{RETAINED_ANALYSIS_RUN}.single-terminal",)
            if stage is ScientificStage.EVALUATE
            else (f"custody.{task}",),
        )

    for context in ("assembling", "prepared"):
        fit, calibrate, support, assess = (
            f"{prefix}.{role}.{context}"
            for role in ("fit", "calibrate", "support", "assess")
        )
        for task, dependencies, report, data, size in (
            (fit, (), ResponseGeometryDevelopmentFitResult, ("data", DEVELOPMENT_FIT_SCHEMA), 65 * 1024**2),
            (calibrate, (fit,), ResponseGeometryDevelopmentCalibrationResult, None, 128 * 1024),
            (support, (fit, calibrate), ResponseGeometryDevelopmentSupportResult, None, 1024**2),
        ):
            add(
                task,
                ScientificStage.DEVELOP,
                manifests["development"],
                qualification.method,
                qualification.method.config_id,
                dependencies,
                (("report", report.SCHEMA),),
                data,
                size,
                28800,
            )
        add(
            assess,
            ScientificStage.QUALIFY,
            manifests["assessment"],
            qualification,
            qualification.config_id,
            (fit, calibrate, support),
            (
                ("report", ResponseGeometryDevelopmentContextResult.SCHEMA),
                ("geometry", ResponseGeometryDevelopmentGeometryReport.SCHEMA),
            ),
            ("data", DEVELOPMENT_VALIDATION_SCHEMA),
            297 * 1024**2,
            28800,
        )
    closeout = ResponseGeometryDevelopmentCloseoutConfig(f"{prefix}.closeout-config", qualification)
    terminal = f"{prefix}.evaluate"
    add(
        terminal,
        ScientificStage.EVALUATE,
        manifests["evaluation"],
        closeout,
        closeout.config_id,
        tuple(f"{prefix}.assess.{context}" for context in ("assembling", "prepared")),
        (
            ("report", ResponseGeometryDevelopmentDevelopmentResult.SCHEMA),
            ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
        ),
        None,
        8 * 1024**2,
        120,
    )
    if tuple(sorted(steps)) != continuation.method_task_ids:
        raise ValueError("development analysis template changes its exact unfinished task roster")
    protocol = ProtocolTemplate(
        f"{RETAINED_ANALYSIS_RUN}.protocol",
        "1.0.0",
        tuple(steps[key] for key in sorted(steps)),
        False,
        False,
        True,
    )
    declaration = CandidateGraphExternalInput(
        continuation.config_id,
        ScientificInputRole.MODEL,
        continuation.config_id,
        ContentIdentityPolicy.EXACT_SHA256,
        continuation.fingerprint(),
        continuation.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        len(continuation.canonical_bytes()),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    external_by_id: dict[str, CandidateGraphExternalInput] = {}
    edges: list[CandidateGraphEdge] = []
    for child in protocol.steps:
        if ".fit." in child.step_id or ".assess." in child.step_id:
            declaration_id = continuation.artifact_id(
                RETAINED_ANALYSIS_RUN, continuation.config_id, task_id=child.step_id
            )
            selected_declaration = replace(
                declaration, input_id=declaration_id, logical_artifact_id=declaration_id
            )
            external_by_id[declaration_id] = selected_declaration
            edges.append(
                CandidateGraphEdge(
                    f"edge.{continuation.config_id}.{child.step_id}",
                    None,
                    None,
                    selected_declaration.input_id,
                    child.step_id,
                    "retained-input-declaration",
                    ScientificInputRole.MODEL,
                    selected_declaration.logical_artifact_id,
                    declaration.payload_schema,
                    declaration.media_type,
                    declaration.maximum_size_bytes,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    child.barrier,
                )
            )
        for row in continuation.inputs_for_task(child.step_id):
            imported_id = continuation.artifact_id(
                RETAINED_ANALYSIS_RUN, row.slot_id, task_id=child.step_id
            )
            external_by_id[imported_id] = CandidateGraphExternalInput(
                imported_id,
                ScientificInputRole.OUTCOME,
                imported_id,
                ContentIdentityPolicy.EXACT_SHA256,
                row.artifact.sha256,
                row.artifact.payload_schema,
                row.artifact.media_type,
                row.artifact.size_bytes,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
            )
            edges.append(
                CandidateGraphEdge(
                    f"edge.{row.slot_id}.{child.step_id}",
                    None,
                    None,
                    imported_id,
                    child.step_id,
                    f"input.{row.slot_id}",
                    ScientificInputRole.OUTCOME,
                    imported_id,
                    row.artifact.payload_schema,
                    row.artifact.media_type,
                    row.artifact.size_bytes,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    child.barrier,
                )
            )
        for parent_id in child.dependency_step_ids:
            parent = steps[parent_id]
            for output in parent.outputs:
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{parent_id}.{output.output_id}.{child.step_id}",
                        parent_id,
                        output.output_id,
                        None,
                        child.step_id,
                        f"input.{parent_id}.{output.output_id}",
                        ScientificInputRole.OUTCOME,
                        f"artifact.{parent_id}.{output.output_id}",
                        output.payload_schema,
                        output.media_type,
                        parent.resource_budget.output_bytes,
                        child.requested_outcome_access,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        child.barrier,
                    )
                )
    nodes = tuple(
        CandidateGraphNode(
            step.step_id,
            step.stage,
            step.capability_key,
            step.capability_version,
            registry.resolve(
                step.capability_key, step.capability_version
            ).implementation_sha256,
            step.fingerprint(),
            step.obligation_ids,
            step.requested_outcome_access,
            step.visibility_ceiling,
            step.resource_budget,
        )
        for step in protocol.steps
    )
    graph = CandidateScientificGraph(
        f"{RETAINED_ANALYSIS_RUN}.graph",
        tuple(external_by_id[key] for key in sorted(external_by_id)),
        nodes,
        tuple(sorted(edges, key=lambda row: row.edge_id)),
    )
    owners = {
        obligation: step.step_id
        for step in protocol.steps
        for obligation in step.obligation_ids
    }
    coverage = tuple(
        ObligationCoverageBinding(
            obligation,
            owners.get(obligation, terminal),
            "report",
            tuple(
                sorted(
                    edge.edge_id
                    for edge in graph.edges
                    if edge.consumer_node_id == owners.get(obligation, terminal)
                )
            ),
        )
        for obligation in required_candidate_obligation_ids(experiment, protocol)
    )
    return StudyTemplate(
        f"{RETAINED_ANALYSIS_RUN}.template",
        "1.0.0",
        protocol,
        graph,
        ObligationCoverage(
            f"{RETAINED_ANALYSIS_RUN}.coverage",
            tuple(sorted(coverage, key=lambda row: row.obligation_id)),
        ),
    )


def build_response_geometry_development_analysis_authoring(
    continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig,
    original: ResponseGeometryDevelopmentAuthoringBundle,
    *,
    implementation_sha256: str,
) -> ResponseGeometryDevelopmentAuthoringBundle:
    """Retain development's scientific specification, replacing only its execution route.

    The predecessor bundle is a nonexecuting reconstruction of the original
    declarations. Native strict roots are retained inside the qualification
    provenance, rather than claimed as new prospective acquisition tasks.
    """
    original_draft = original.authoring.base.draft
    experiment = original_draft.experiment
    assert experiment is not None
    if continuation.qualification not in original.payloads:
        raise ValueError(
            "development analysis authoring changes the original frozen qualification science"
        )
    registry = CapabilityRegistry(
        f"{RETAINED_ANALYSIS_RUN}.registry",
        tuple(sorted(RETAINED_ANALYSIS_CAPABILITIES, key=lambda value: value.registry_id)),
    )
    template = build_response_geometry_development_analysis_template(continuation, experiment, registry)
    declaration = ObjectIdentity.from_record(continuation.config_id, continuation)
    original_science = next(
        value
        for value in original_draft.design_inputs
        if value.role is DesignInputRole.MOTIVATION
    )
    added = [
        DesignInputRecord(
            f"{RETAINED_ANALYSIS_RUN}.retained-input-declaration",
            declaration,
            declaration.object_fingerprint,
            original_science.information_cutoff,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            original_science.operator_id,
            parent_input_ids=(original_science.input_id,),
        )
    ]
    if continuation.owner_amendment is not None:
        added.append(
            DesignInputRecord(
                f"{RETAINED_ANALYSIS_RUN}.owner-amendment",
                continuation.owner_amendment,
                continuation.owner_amendment.object_fingerprint,
                original_science.information_cutoff,
                DesignInputRole.READINESS_METADATA,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                original_science.operator_id,
                parent_input_ids=(original_science.input_id,),
            )
        )
    design_inputs = tuple(
        sorted(
            (*original_draft.design_inputs, *added), key=lambda value: value.input_id
        )
    )
    prior_source = original.standard_context.base.qualifications[0]
    qualification_receipt = replace(
        prior_source,
        receipt_id=f"{RETAINED_ANALYSIS_RUN}.declaration-qualification",
        source_id=continuation.config_id,
        materialization=declaration,
        content_sha256=continuation.fingerprint(),
    )
    source_ref = SourceMaterializationRef(
        continuation.config_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        qualification_receipt.evidence_world_id,
        declaration,
        continuation.fingerprint(),
        continuation.fingerprint(),
        qualification_receipt.observation_operator,
        qualification_receipt.numerical_view_ids,
        ObjectIdentity.from_record(
            qualification_receipt.receipt_id, qualification_receipt
        ),
        qualification_receipt.access_disposition,
    )
    assessment_qualification = replace(
        qualification_receipt,
        receipt_id=f"{qualification_receipt.receipt_id}.assessment",
        source_id=f"{continuation.config_id}.assessment",
        materialization=ObjectIdentity.from_record(
            f"{continuation.config_id}.assessment", continuation
        ),
    )
    assessment_source = replace(
        source_ref,
        source_id=assessment_qualification.source_id,
        materialization=assessment_qualification.materialization,
        qualification_receipt=ObjectIdentity.from_record(
            assessment_qualification.receipt_id, assessment_qualification
        ),
    )
    draft = replace(
        original_draft,
        draft_id=f"{RETAINED_ANALYSIS_RUN}.draft",
        design_inputs=design_inputs,
        design_origin=replace(
            original_draft.design_origin,
            origin_id=f"{RETAINED_ANALYSIS_RUN}.origin",
            declared_input_ids=tuple(
                sorted(
                    (
                        *original_draft.design_origin.declared_input_ids,
                        *(value.input_id for value in added),
                    )
                )
            ),
        ),
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                value.capability_key,
                value.capability_version,
                value.implementation_sha256,
            )
            for value in sorted(
                RETAINED_ANALYSIS_CAPABILITIES, key=lambda value: value.capability_key
            )
        ),
        source_materializations=(source_ref, assessment_source),
    )
    context = CandidateCompilationContext(
        f"{RETAINED_ANALYSIS_RUN}.context",
        registry,
        (template,),
        (qualification_receipt, assessment_qualification),
        design_inputs,
        implementation_sha256,
    )
    base, standard = _entry(
        draft,
        context,
        original_science.object_identity,
        prefix=RETAINED_ANALYSIS_RUN,
        minimum_units=64,
        operand_description="Unchanged finite development {domain} operands from the retained original panel; no new acquisition or confirmation.",
        estimator="frozen-development-development-estimator",
        uncertainty="rootwise-model-conditional-calibration",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=RETAINED_ANALYSIS_CAPABILITIES[RETAINED_ANALYSIS_ROLES.index("evaluation")],
        input_schema=ResponseGeometryDevelopmentContextResult.SCHEMA,
        output_schema=ResponseGeometryDevelopmentDevelopmentResult.SCHEMA,
    )
    payloads = tuple(
        sorted(
            (
                continuation,
                continuation.qualification.method,
                continuation.qualification,
                ResponseGeometryDevelopmentCloseoutConfig(
                    f"{RETAINED_ANALYSIS_PREFIX}.closeout-config", continuation.qualification
                ),
            ),
            key=lambda value: value.SCHEMA,
        )
    )
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in RETAINED_ANALYSIS_BINDINGS
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[value.SCHEMA] for value in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{RETAINED_ANALYSIS_RUN}.extension.{i:02d}",
            f"{RETAINED_ANALYSIS_RUN}.namespace.{i:02d}",
            ObjectIdentity.from_record(value.config_id, value),
            len(value.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, (value, decoder) in enumerate(zip(payloads, decoders, strict=True))
    )
    extensions = ProposedStudyExtensionSet(
        f"{RETAINED_ANALYSIS_RUN}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(value.namespace_id for value in proposed),
        proposed,
    )
    selected = {value.capability_key for value in RETAINED_ANALYSIS_CAPABILITIES}
    evidence = replace(
        original.evidence_profile,
        selection_id=f"{RETAINED_ANALYSIS_RUN}.evidence-profile",
        draft_id=draft.draft_id,
    )
    return ResponseGeometryDevelopmentAuthoringBundle(
        ExecutableStudyDefinition(f"{RETAINED_ANALYSIS_RUN}.authoring", base, extensions),
        standard,
        CandidateCapabilityCatalog(
            f"{RETAINED_ANALYSIS_RUN}.catalog",
            tuple(
                value
                for value in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if value.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )


def build_response_geometry_development_analysis_resources(
    continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    if tuple(
        step.step_id for step in protocol.steps
    ) != continuation.method_task_ids or any(
        step.stage
        not in (
            ScientificStage.DEVELOP,
            ScientificStage.QUALIFY,
            ScientificStage.EVALUATE,
        )
        or step.maximum_attempts != 1
        or step.capability_key
        not in {m.capability_key for m in RETAINED_ANALYSIS_CAPABILITIES}
        for step in protocol.steps
    ):
        raise ValueError(
            "development analysis resources require nine methods, zero native tasks and no retries"
        )
    cells = tuple(
        ExecutionResourceTaskCellSpec(
            f"cell.{step.step_id}",
            step.step_id,
            RETAINED_ANALYSIS_RUN,
            None,
            None,
            1,
            0,
            0,
            False,
            NonTimeResourceBudget(
                f"budget.{step.step_id}",
                step.resource_budget.cpu_cores,
                step.resource_budget.memory_bytes,
                step.resource_budget.output_bytes,
                0,
                step.resource_budget.source_scan_bytes,
            ),
            None,
            None,
            None,
        )
        for step in protocol.steps
    )
    return ExecutionResourceEnvelopeSpec(
        f"{RETAINED_ANALYSIS_RUN}.resources",
        issued_extensions,
        cells,
        (ChildResourceTokenLimit(RETAINED_ANALYSIS_RUN, 0, 0),),
        (),
        (),
        tuple(sorted(failure.value for failure in OperationalFailureClass)),
        None,
        None,
        2,
        16 * 1024**3,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "continuation": continuation.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_updates": 0,
                    "native_execution_tokens": 0,
                    "retry_tokens": 0,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
