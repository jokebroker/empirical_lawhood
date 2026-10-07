"""Noncontact preparation authoring through the installed native source/law route.

The complete initial protocol is supplied by the same adapter-owned builder as
its registered expansion. This preserves one authenticated prefix edge per root
without a private execution graph or a new interpretation of physical units.
"""
from empirical_lawhood.kernel.authority import AuthorityAction


from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_profiles import EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.native_source import NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeSourceProfile
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.prospective_config import ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, ContentIdentityPolicy, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.response_experiment import compile_response_experiment
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle, ResponseGeometryAssayCandidateContextProvider
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.protocol_helpers import record_stable_id as _record_id
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.control.backbone_linked_campaign.executable_binding import (
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
    PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
)
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.methods.matrix_preparation.closeout import PreparationCloseoutConfig
from empirical_lawhood.adapters.methods.matrix_preparation.contracts import PreparationMethodConfig, PreparationProjectionConfig, PreparationProjectionReport
from empirical_lawhood.adapters.methods.matrix_preparation.executable_binding import METHOD_BINDINGS
from empirical_lawhood.adapters.methods.matrix_preparation.continuation import PreparationProjectionContinuation
from empirical_lawhood.adapters.methods.matrix_preparation.extension_bundle import METHOD_CAPABILITIES
from empirical_lawhood.adapters.methods.matrix_preparation.qualification import PreparationQualificationProfile
from empirical_lawhood.adapters.methods.matrix_preparation.records import PreparationForecastConfig, PreparationDecisionConfig, PreparationTaskAssessmentConfig, PreparationAssessmentConfig, PreparationNumericalSemanticsConfig
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_CLOCK as NATIVE_REFERENCE_CLOCK, DEVELOPMENT_EPISODE_FRAME as NATIVE_EPISODE_FRAME, DEVELOPMENT_FORCE_DIRECTION as POSITIVE_MODE_DIRECTION, DEVELOPMENT_FORCE_FRAME as FROZEN_PREPARATION_MODE
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, NATIVE_SCHEMA, PreparationNativeResult, PreparationRoot, PreparationSourceConfig
from empirical_lawhood.adapters.simulators.matrix_preparation.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.simulators.matrix_preparation.continuation import PreparationDevelopmentContinuation
from empirical_lawhood.adapters.simulators.matrix_preparation.extension_bundle import SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.matrix_preparation.imports import PreparationSourceImportInventory
from empirical_lawhood.adapters.simulators.matrix_preparation.protocol import preparation_clock_reference, preparation_protocol_steps
from empirical_lawhood.adapters.simulators.matrix_preparation.roster import preparation_native_declarations
from .entry import preparation_entry
from .science import ACTION, DEVELOPMENT_BUDGET, RECEIVER, preparation_experiment, preparation_system


@dataclass(frozen=True, slots=True)
class PreparationAuthoringBundle(ResponseGeometryAssayAuthoringBundle):
    """Exact installed records, closed catalog and source-free authoring context."""


@dataclass(frozen=True, slots=True)
class PreparationCandidateContextProvider(ResponseGeometryAssayCandidateContextProvider):
    bundle: PreparationAuthoringBundle


def _lineage() -> tuple[ProspectivePackageLineage, ProspectiveTerminalMatrix]:
    prefix, imported = DEVELOPMENT, f"{DEVELOPMENT}.import-qualification"
    lineage = ProspectivePackageLineage(
        f"{prefix}.lineage",
        imported,
        f"{prefix}.authenticated-continuation-branch",
        tuple(
            sorted(
                (
                    ProspectivePackageLineageNode(
                        imported,
                        "Current outcome-blind custody authentication of exposed historical prefixes",
                        ProspectivePackageKind.EXCLUDED_QUALIFICATION,
                        (),
                        (f"{prefix}.authenticated-prefix-custody",),
                        EvidenceCeiling.MEASUREMENT,
                        None,
                        False,
                        False,
                        False,
                        False,
                    ),
                    ProspectivePackageLineageNode(
                        prefix,
                        "New conditional outcomes on exposed roots; development-only nomination",
                        ProspectivePackageKind.PROSPECTIVE_CHILD,
                        (imported,),
                        (f"{prefix}.exact-route-gate", f"{prefix}.typed-authority-gate"),
                        EvidenceCeiling.LOCAL_LAW,
                        None,
                        False,
                        False,
                        False,
                        False,
                    ),
                ),
                key=lambda n: n.package_id,
            )
        ),
        (f"{prefix}.unauthenticated-input-stop",),
        True,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    terminal = ProspectiveTerminalMatrix(
        f"{prefix}.terminal-matrix",
        (
            ProspectiveSelectorTransition(
                1,
                "Exact retained custody, public route and typed authority close",
                "Enter all 128 exposed roots and all predeclared analyses",
            ),
            ProspectiveSelectorTransition(
                2,
                "A required input or route is unresolved",
                "Do not issue or contact native source",
            ),
        ),
        (
            ProspectiveTerminalRule(
                1,
                "Any necessary numerical or observation operand is missing",
                "UNEVALUABLE",
                "Retain available outcomes and every assigned root",
                "Record the affected gate failure and leave dependent fresh work unentered",
            ),
            ProspectiveTerminalRule(
                2,
                "Complete development operands",
                "RETAIN_SEPARATE_DEVELOPMENT_GATES_AND_SHARED_LAW_STATUS",
                "Exposed development evidence only",
                "Fresh calibration and efficacy require all context gates plus their own exact freeze, route and issue",
            ),
        ),
        True,
        True,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    return lineage, terminal


def _template(
    source: PreparationSourceConfig,
    projection: PreparationProjectionConfig,
    numerical_semantics: PreparationNumericalSemanticsConfig,
    method: PreparationMethodConfig,
    assessment: PreparationAssessmentConfig,
    closeout: PreparationCloseoutConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    continuation: PreparationDevelopmentContinuation | None = None,
) -> StudyTemplate:
    steps = preparation_protocol_steps(
        source, projection, numerical_semantics, method, assessment, closeout, continuation
    )
    base = _programme_template(
        ProtocolTemplate(f"{DEVELOPMENT}.protocol", "1.0.0", steps, False, False, True),
        source,
        source.config_id,
        experiment,
        registry,
        prefix=DEVELOPMENT,
        source_task_id=next(s.step_id for s in steps if s.stage is ScientificStage.PREPARE),
        terminal_task_id=f"{DEVELOPMENT}.evaluate",
        primary_output_ids={
            s.stage: "native-result" if s.stage is ScientificStage.PREPARE else "report"
            for s in steps
        },
    )
    by_id = {s.step_id: s for s in steps}
    edges = [
        e
        for e in base.graph.edges
        if e.producer_node_id is not None
        and e.payload_schema
        in registry.resolve(
            by_id[e.consumer_node_id].capability_key, by_id[e.consumer_node_id].capability_version
        ).input_schema_ids
    ]
    external = list(base.graph.external_inputs)
    model_edge = next(e for e in base.graph.edges if e.external_input_id == source.config_id)
    active_roots = source.roots if continuation is None else continuation.missing_roots
    for bundle in source.prefix_bundles:
        if not set(bundle.roots).intersection(active_roots):
            continue
        artifact = bundle.artifact
        external.append(
            CandidateGraphExternalInput(
                bundle.imported_artifact_id,
                ScientificInputRole.PREPARED_MEDIUM,
                bundle.imported_artifact_id,
                ContentIdentityPolicy.EXACT_SHA256,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.OUTCOME_VISIBLE,
            )
        )
    for prefix in source.prefixes:
        if prefix.root not in active_roots:
            continue
        task = f"{prefix.root_id}.native"
        edges.append(
            replace(model_edge, edge_id=f"{model_edge.edge_id}.{task}", consumer_node_id=task)
        )
        bundle = next(b for b in source.prefix_bundles if prefix.root in b.roots)
        artifact = bundle.artifact
        edges.append(
            CandidateGraphEdge(
                f"edge.{task}.retained-prefix",
                None,
                None,
                bundle.imported_artifact_id,
                task,
                "retained-prefix-bundle",
                ScientificInputRole.PREPARED_MEDIUM,
                bundle.imported_artifact_id,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.OUTCOME_VISIBLE,
                BarrierKind.NONE,
            )
        )
    if continuation is not None:
        for retained in continuation.native_inputs:
            artifact = retained.artifact
            external.append(
                CandidateGraphExternalInput(
                    retained.imported_artifact_id,
                    ScientificInputRole.OUTCOME,
                    retained.imported_artifact_id,
                    ContentIdentityPolicy.EXACT_SHA256,
                    artifact.sha256,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                )
            )
            for view in (1, 2):
                task_id = f"{retained.root.root_id}.project.r{view}"
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{retained.slot_id}.{task_id}",
                        None,
                        None,
                        retained.imported_artifact_id,
                        task_id,
                        f"input-{retained.task_id}-{retained.output_id}",
                        ScientificInputRole.OUTCOME,
                        retained.imported_artifact_id,
                        artifact.payload_schema,
                        artifact.media_type,
                        artifact.size_bytes,
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        BarrierKind.FREEZE,
                    )
                )
    graph = replace(
        base.graph,
        external_inputs=tuple(sorted(external, key=lambda e: e.input_id)),
        edges=tuple(sorted(edges, key=lambda e: e.edge_id)),
    )
    coverage = replace(
        base.coverage,
        bindings=tuple(
            replace(
                b,
                contributor_edge_ids=tuple(
                    sorted(e.edge_id for e in edges if e.consumer_node_id == b.proof_owner_node_id)
                ),
            )
            for b in base.coverage.bindings
        ),
    )
    return replace(base, graph=graph, coverage=coverage)


def build_preparation_authoring(
    *,
    implementation_plan_sha256: str,
    implementation_sha256: str,
    design_sha256: str,
    imported_inventory: PreparationSourceImportInventory,
    continuation: PreparationDevelopmentContinuation | None = None,
) -> PreparationAuthoringBundle:
    """Build exact full-size records without reading sources or constructing runners."""
    prefix = DEVELOPMENT
    imported_id = ObjectIdentity.from_record(imported_inventory.inventory_id, imported_inventory)
    source = PreparationSourceConfig(
        f"{prefix}.source-config",
        implementation_plan_sha256,
        imported_id,
        imported_inventory.prefix_source_config,
        imported_inventory.prefixes,
        imported_inventory.prefix_bundles,
        design_sha256,
    )
    source_id = ObjectIdentity.from_record(source.config_id, source)
    if continuation is not None:
        continuation.validate_source(source)
    projection = PreparationProjectionConfig(f"{prefix}.projection-config", source_id)
    projection_id = ObjectIdentity.from_record(projection.config_id, projection)
    numerical_semantics = PreparationNumericalSemanticsConfig(f"{prefix}.numerical-semantics-config", projection_id)
    method = PreparationMethodConfig(
        f"{prefix}.method-config", projection_id, implementation_plan_sha256
    )
    system = preparation_system()
    if continuation is not None:
        system = replace(
            system,
            authority_policy=replace(
                system.authority_policy,
                policy_id=f"{continuation.run_id}.authority-policy",
                scope_ids=(continuation.run_id,),
            ),
        )
    assessment = PreparationAssessmentConfig(
        f"{prefix}.assessment-config", system, source_id, projection, numerical_semantics, method
    )
    closeout = PreparationCloseoutConfig(f"{prefix}.closeout-config", assessment)
    manifest = METHOD_CAPABILITIES[3]
    native = preparation_native_declarations(source, preparation_clock_reference(assessment))
    profile = PreparationQualificationProfile(
        QualificationProofOwner(
            f"{prefix}.finite-law-proof-owner",
            manifest.capability_key,
            manifest.capability_version,
            manifest.implementation_sha256,
        )
    ).profile
    experiment = preparation_experiment(system, source, METHOD_CAPABILITIES[-1].capability_key)
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=DEVELOPMENT_BUDGET,
        objective="Complete the exposed-root preparation, observable prediction, forecast and task feasibility panel, preserving separate gates and nonpromotable law assessments.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    evidence_registry = build_observation_evidence_world_registry()
    world = next(
        w
        for w in evidence_registry.world_profiles
        if w.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{prefix}.evidence-profile",
        draft_id=f"{prefix}.draft",
        registry=evidence_registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=tuple(sorted((
            EvidenceRung.MEASUREMENT,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        ), key=lambda rung: rung.value)),
    )
    feasibility = resolve_candidate_profile_feasibility(
        selection=evidence,
        resolver=EvidenceProfileRegistryResolver(
            f"{prefix}.profile-resolver", (evidence_registry,)
        ),
    )
    if feasibility.disposition is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION:
        raise ValueError(
            f"preparation evidence profile is not compilable: {feasibility.disposition}"
        )
    source_profile = NativeSourceProfile(
        f"{prefix}.native-source-profile",
        CapabilitySelection(
            SOURCE_CAPABILITY.capability_key,
            SOURCE_CAPABILITY.capability_version,
            SOURCE_CAPABILITY.implementation_sha256,
        ),
        source_id,
        NATIVE_SCHEMA,
        tuple(u.physical_independent_unit_id for u in native.units),
        128,
        source.maximum_native_updates,
        DEVELOPMENT_BUDGET,
        imported_id,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    evidence_id = ObjectIdentity.from_record(evidence.selection_id, evidence)
    native_id = ObjectIdentity.from_record(source_profile.profile_id, source_profile)
    campaign_id = ObjectIdentity.from_record(campaign.campaign_id, campaign)
    lineage, terminal = _lineage()
    law_qualification = NativeLawQualificationConfig(
        f"{prefix}.native-law-qualification-config",
        evidence_id,
        native_id,
        campaign_id,
        native.chart,
        lineage,
        terminal,
        native.units,
        native.acquisitions,
        native.views,
        source_profile.physical_independent_unit_ids,
        tuple(
            u.physical_independent_unit_id
            for u in native.units
            if u.split_role.value == "DEVELOPMENT"
        ),
        tuple(
            u.physical_independent_unit_id
            for u in native.units
            if u.split_role.value == "EVALUATION"
        ),
        ("r1", "r2"),
        projection_id,
        ObjectIdentity.from_record(method.config_id, method),
        ObjectIdentity.from_record(profile.profile_id, profile),
        ObjectIdentity.from_record(manifest.capability_key, manifest),
        None,
        EvidenceCeiling.LOCAL_LAW,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        native.projections,
    )
    extension = NativeLawQualificationExperiment(
        f"{prefix}.carrier",
        evidence_id,
        native_id,
        campaign_id,
        law_qualification,
        None,
        None,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    compile_response_experiment(
        extension_set=extension,
        evidence_profile_selection=evidence,
        source_pipeline_profile=source_profile,
        linked_campaign_profile=campaign,
    )
    substrate = ResponseSubstrateBinding(
        f"{prefix}.native-binding",
        f"{prefix}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        SOURCE_CAPABILITY.capability_key,
        SOURCE_CAPABILITY.capability_version,
        SOURCE_CAPABILITY.implementation_sha256,
        ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
        NativeSourceProfile.SCHEMA,
        PreparationRoot.SCHEMA,
        None,
        PreparationNativeResult.SCHEMA,
        PreparationProjectionReport.SCHEMA,
        source_id,
        (
            NativeReceiverContract(
                f"{prefix}.receiver",
                RECEIVER,
                "hilbert-schmidt-native",
                FROZEN_PREPARATION_MODE,
                POSITIVE_MODE_DIRECTION,
                (NATIVE_REFERENCE_CLOCK,),
            ),
        ),
        (NativeClockContract(NATIVE_REFERENCE_CLOCK, "reference-tick", NATIVE_EPISODE_FRAME),),
        NativeActionContract(ACTION, True, True, True, True, "HOLD"),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    configs: tuple[CanonicalRecord, ...] = (
        source,
        projection,
        numerical_semantics,
        method,
        assessment,
        closeout,
        PreparationForecastConfig(f"{prefix}.forecast-config", method),
        PreparationDecisionConfig(f"{prefix}.decision-config", method),
        PreparationTaskAssessmentConfig(f"{prefix}.task-assessment-config", method),
    )
    payloads = tuple(
        sorted(
            (
                *configs,
                source_profile,
                law_qualification,
                extension,
                substrate,
                *(
                    (
                        continuation,
                        PreparationProjectionContinuation(
                            f"{continuation.run_id}.projection-continuation",
                            projection_id,
                            ObjectIdentity.from_record(continuation.config_id, continuation),
                        ),
                    )
                    if continuation is not None
                    else ()
                ),
            ),
            key=lambda r: r.SCHEMA,
        )
    )
    decoder_by_schema = {
        d.payload_schema: d
        for b in (
            SOURCE_BINDING,
            *METHOD_BINDINGS,
            NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
            PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
        )
        for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[r.SCHEMA] for r in payloads)
    qualification = MaterializationQualificationReceipt(
        f"{prefix}.config-qualification",
        source.config_id,
        source_id,
        source.fingerprint(),
        system.world.world_id,
        projection_id,
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({q.native_unit for q in system.quantities})),
        tuple(sorted({q.coordinate_frame for q in system.quantities})),
        (NATIVE_REFERENCE_CLOCK,),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        source.config_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        source_id,
        source.fingerprint(),
        source.fingerprint(),
        projection_id,
        tuple(v.view_id for v in system.numerical_views),
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design = ObjectIdentity(
        f"{prefix}.design", 'empirical-lawhood/document/markdown', "1.0.0", source.design_sha256
    )
    qualifications = [qualification]
    sources = [source_ref]
    active_roots = source.roots if continuation is None else continuation.missing_roots
    # Prior native task products retain their OUTCOME role and exact issued
    # custody declaration. Source qualification belongs to the prefix inputs.
    retained_materializations = tuple(
        b for b in source.prefix_bundles if set(b.roots).intersection(active_roots)
    )
    for retained in retained_materializations:
        materialization = ObjectIdentity(
            retained.artifact.artifact_id,
            retained.artifact.payload_schema,
            "1.0.0",
            retained.artifact.sha256,
        )
        receipt = replace(
            qualification,
            receipt_id=f"qualification.{retained.imported_artifact_id}",
            source_id=retained.imported_artifact_id,
            materialization=materialization,
            content_sha256=retained.artifact.sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        qualifications.append(receipt)
        sources.append(
            replace(
                source_ref,
                source_id=retained.imported_artifact_id,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                materialization=materialization,
                content_sha256=retained.artifact.sha256,
                qualification_receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
            )
        )
    plan = ObjectIdentity(
        f"{prefix}.implementation-plan",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        implementation_plan_sha256,
    )
    cutoff = InformationCutoff(
        f"{prefix}.design-freeze", NATIVE_REFERENCE_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    units = source_profile.physical_independent_unit_ids
    seeds = tuple(f"{u}.native-rng" for u in units)
    design_inputs = tuple(
        sorted(
            (
                DesignInputRecord(
                    f"{prefix}.design-input",
                    design,
                    design.object_fingerprint,
                    cutoff,
                    DesignInputRole.MOTIVATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    "owner.empirical-lawhood",
                ),
                DesignInputRecord(
                    f"{prefix}.implementation-input",
                    plan,
                    plan.object_fingerprint,
                    cutoff,
                    DesignInputRole.MOTIVATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    "owner.empirical-lawhood",
                    parent_input_ids=(f"{prefix}.design-input",),
                ),
                DesignInputRecord(
                    f"{prefix}.all-original-roots-exposed",
                    imported_id,
                    imported_id.object_fingerprint,
                    cutoff,
                    DesignInputRole.DEVELOPMENT_TUNING,
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    "owner.empirical-lawhood",
                    units,
                    seeds,
                    (f"{prefix}.implementation-input",),
                ),
            ),
            key=lambda i: i.input_id,
        )
    )
    if continuation is not None:
        plan = continuation.implementation_plan
        design_inputs = tuple(
            sorted(
                (
                    *design_inputs,
                    DesignInputRecord(
                        f"{continuation.run_id}.continuation-input",
                        ObjectIdentity.from_record(continuation.config_id, continuation),
                        continuation.fingerprint(),
                        cutoff,
                        DesignInputRole.DEVELOPMENT_TUNING,
                        OutcomeAccess.EVALUATION_REVEALED,
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        "owner.empirical-lawhood",
                        units,
                        seeds,
                        (f"{prefix}.all-original-roots-exposed",),
                    ),
                    DesignInputRecord(
                        f"{continuation.run_id}.implementation-input",
                        plan,
                        plan.object_fingerprint,
                        cutoff,
                        DesignInputRole.MOTIVATION,
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        "owner.empirical-lawhood",
                        parent_input_ids=(f"{prefix}.implementation-input",),
                    ),
                ),
                key=lambda i: i.input_id,
            )
        )
    capabilities = tuple(
        sorted((SOURCE_CAPABILITY, *METHOD_CAPABILITIES), key=lambda c: c.capability_key)
    )
    registry = CapabilityRegistry(
        f"{prefix}.registry", tuple(sorted(capabilities, key=lambda c: c.registry_id))
    )
    template = _template(
        source, projection, numerical_semantics, method, assessment, closeout, experiment, registry, continuation
    )
    draft = StudyDraft(
        f"{prefix}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (
            f"{prefix}.alternative.development-feasible",
            f"{prefix}.alternative.development-not-qualified",
        ),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (f"{prefix}.all-original-roots-exposed",)
            if continuation is None
            else (
                f"{continuation.run_id}.continuation-input",
                f"{continuation.run_id}.implementation-input",
            ),
            # The new design precedes its continuation outcomes. The complete
            # transitive input ledger and ExperimentSpec retain OUTCOME_VISIBLE
            # design visibility, and the draft has no fresh evaluation units.
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        units,
        (),
        seeds,
        (),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in capabilities
        ),
        tuple(sorted(sources, key=lambda s: s.source_id)),
        DEVELOPMENT_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        tuple(sorted(qualifications, key=lambda q: q.receipt_id)),
        design_inputs,
        implementation_sha256,
    )
    expanded = compose_parameterised_candidate_context_for_template(
        template_key=template.template_key,
        context=context,
        records=payloads,
        factories=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
    )
    assert isinstance(expanded, CandidateCompilationContext)
    base, standard = preparation_entry(draft, expanded, plan)
    proposed = tuple(
        ProposedStudyExtension(
            f"{prefix}.extension.{i:02d}",
            f"{prefix}.namespace.{i:02d}",
            ObjectIdentity.from_record(_record_id(r), r),
            len(r.canonical_bytes()),
            d.decoder_key,
            d.decoder_version,
            d.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, (r, d) in enumerate(zip(payloads, decoders, strict=True))
    )
    extensions = ProposedStudyExtensionSet(
        f"{prefix}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(p.namespace_id for p in proposed),
        proposed,
    )
    selected = {c.capability_key for c in capabilities}
    return PreparationAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.authoring", base, extensions),
        replace(standard, base=context),
        CandidateCapabilityCatalog(
            f"{prefix}.catalog",
            tuple(
                c
                for c in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if c.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )
