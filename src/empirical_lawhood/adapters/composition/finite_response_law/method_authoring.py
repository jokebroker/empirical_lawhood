"Assigned calibration method configuration on the existing ordinary authoring/issue route."
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import replace
from decimal import Decimal as D

from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.finite_response_law.calibration_operands import PREDICTION_SCHEMA
from empirical_lawhood.adapters.methods.finite_response_law.executable_binding import METHOD_BINDING
from empirical_lawhood.adapters.methods.finite_response_law.extension_bundle import METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.finite_response_law.law_binding import output_quantities
from empirical_lawhood.adapters.methods.finite_response_law.method_records import ADJUDICATE, CALIBRATE, FREEZE, QUALIFY, READOUT_SCHEMA, FiniteResponseLawAssignedCalibrationMethodConfig, FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawAssignedQualificationReport
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import ExperimentSpec, PrecisionGoal
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_profiles import bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, MaterializationQualificationReceipt, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateCompilationContext,
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import (
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)

from .authoring import build_native_authoring
from .design import native_experiment, qualification_system
from .exposure import native_seed_ids

PREFIX = f"{PROGRAMME}.calibration.method"
BUDGET = ResourceBudget(1, 1024**3, 0, 480, 128 * 1024**2, 32 * 1024**2)


def method_resource_envelope(
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
    *,
    task_ids: tuple[str, ...] = (FREEZE, CALIBRATE, QUALIFY, ADJUDICATE),
    prefix: str = PREFIX,
) -> ExecutionResourceEnvelopeSpec:
    """Ordinary, nonacquiring tasks on the existing resource envelope."""
    from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceTaskCellSpec, NonTimeResourceBudget

    if {s.step_id for s in protocol.steps} != set(task_ids):
        raise ValueError(
            "Assigned calibration method resource census differs from its three declared tasks"
        )
    cells = tuple(
        ExecutionResourceTaskCellSpec(
            f"cell.{s.step_id}",
            s.step_id,
            prefix,
            None,
            None,
            1,
            0,
            0,
            False,
            NonTimeResourceBudget(
                f"budget.{s.step_id}",
                s.resource_budget.cpu_cores,
                s.resource_budget.memory_bytes,
                s.resource_budget.output_bytes,
                0,
                s.resource_budget.source_scan_bytes,
            ),
            None,
            None,
            None,
        )
        for s in protocol.steps
    )
    return ExecutionResourceEnvelopeSpec(
        f"{prefix}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(prefix, 0, 0),),
        (),
        (),
        ("RESOURCE_COMPUTABILITY_UNAVAILABLE",),
        None,
        None,
        BUDGET.cpu_cores,
        BUDGET.memory_bytes,
        (),
        (),
        protocol.fingerprint(),
        True,
        True,
        True,
        True,
    )


def method_experiment(config: FiniteResponseLawAssignedCalibrationMethodConfig) -> ExperimentSpec:
    system = qualification_system(config.native_source)
    base = native_experiment(config.native_source, system)
    obligations = base.obligations
    obligations = replace(
        obligations,
        uncertainty=replace(
            obligations.uncertainty,
            confidence_level=D(".90"),
            status=ObligationStatus.REQUIRED,
            method_key=f"{PREFIX}.rank-30-complete-root-maximum",
            limitation_codes=("joint-predictive-only-no-latent-decomposition",),
        ),
        validity=replace(
            obligations.validity,
            assumption_ids=(
                "all-roots-accounted-including-failures",
                "frozen-precalibration-predictors",
                "simulator-local-denominator",
            ),
            exclusion_reason_codes=("UNACCOUNTED_OR_CONTRADICTORY_CUSTODY",),
        ),
        falsifiers=tuple(
            replace(
                f,
                capability_key=METHOD_CAPABILITY.capability_key,
                description="Nonfinite rank-30 multiplier refuses that boundary; contradictory custody is UNEVALUABLE. Missing/invalid/unsupported/numerically discrepant roots remain infinite scores in n=32.",
                decisive_rule="No refit, root exclusion, replacement acquisition, or extra global error/width/work threshold.",
            )
            for f in obligations.falsifiers
        ),
    )
    claim = replace(
        base.claims[0],
        claim_id=f"{PREFIX}.claim",
        proposition="The frozen native-action lower/composed laws have a finite joint 90% complete-root calibration bound; comparator qualifications and usability are reported separately.",
        estimand="Rank 30 of all 32 fresh assigned-root maxima over four pairs, eight outputs, two futures and two coupled numerical views, including infinity.",
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        promotion_rule="Sole response-law qualification; primary refusal or absence of a joint usable request stops protected use. No fitting, additional width/error/work gate or calibration success-fraction gate.",
    )
    return replace(
        base,
        experiment_id=PREFIX,
        claims=(claim,),
        obligations=obligations,
        precision_goals=tuple(
            PrecisionGoal(
                f"{PREFIX}.precision.{i}",
                quantity.quantity_id,
                config.native_source.science.delta[i],
                quantity.native_unit,
                32,
                "Per-selected-word half-width only; report the complete fixed request census. No full-menu or population width threshold qualifies/refuses a law.",
            )
            for i, quantity in enumerate(output_quantities())
        ),
        controls=tuple(
            replace(c, capability_key=METHOD_CAPABILITY.capability_key)
            for c in base.controls
        ),
    )


def build_method_authoring(
    *,
    config: FiniteResponseLawAssignedCalibrationMethodConfig,
    implementation_sha256: str,
    exposure: ObjectIdentity,
    exposed_unit_ids: tuple[str, ...],
    exposed_seed_ids: tuple[str, ...],
) -> ResponseGeometryAssayAuthoringBundle:
    if type(config) is not FiniteResponseLawAssignedCalibrationMethodConfig:
        raise ValueError(
            "Assigned calibration authoring requires the assigned finite calibration contract"
        )
    # Reuse the programme's existing design/exposure/entry records. Source
    # declarations are not included in this separate, nonacquiring method act.
    native = build_native_authoring(
        source=config.native_source,
        implementation_sha256=implementation_sha256,
        exposure=exposure,
        exposed_unit_ids=exposed_unit_ids,
        exposed_seed_ids=exposed_seed_ids,
        new_seed_ids=native_seed_ids(config.native_source),
    )
    system = qualification_system(config.native_source)
    experiment = method_experiment(config)
    manifest = METHOD_CAPABILITY
    registry = CapabilityRegistry(f"{PREFIX}.registry", (manifest,))
    ref = ObjectIdentity.from_record(config.config_id, config)
    config_ref = _config_ref(config, ref, manifest)
    steps = []
    for task, products, dependencies, obligation in (
        (
            FREEZE,
            (("report", ObjectIdentity.SCHEMA),),
            (),
            f"{PREFIX}.input-commitment",
        ),
        (
            CALIBRATE,
            (
                ("report", FiniteResponseLawAssignedCalibrationReport.SCHEMA),
                ("predictions", PREDICTION_SCHEMA),
                ("readout", READOUT_SCHEMA),
            ),
            (FREEZE,),
            f"{PREFIX}.calibration-custody",
        ),
        (
            QUALIFY,
            (("report", FiniteResponseLawAssignedQualificationReport.SCHEMA),),
            (CALIBRATE,),
            f"{PREFIX}.qualification-custody",
        ),
        (
            ADJUDICATE,
            (("report", ScientificAdjudicationRecord.SCHEMA),),
            (QUALIFY,),
            f"{PREFIX}.single-terminal",
        ),
    ):
        outputs = []
        for name, schema in (
            products
            if task in (FREEZE, ADJUDICATE)
            else (*products, ("stage", LinkedCampaignStageEnvelope.SCHEMA))
        ):
            profile, media, suffix = (
                (ArtifactProfile.CANONICAL_JSON, "application/json", ".json")
                if schema == PREDICTION_SCHEMA
                else (ArtifactProfile.TEXT_PARAMETERS, "application/json", ".json")
                if schema == READOUT_SCHEMA
                else (
                    ArtifactProfile.CANONICAL_JSON,
                    "application/vnd.empirical-lawhood.canonical+json",
                    ".json",
                )
            )
            outputs.append(OutputTemplate(name, schema, profile, media, suffix))
        steps.append(
            ProtocolStepTemplate(
                task,
                ScientificStage.FREEZE if task == FREEZE else ScientificStage.EVALUATE,
                manifest.capability_key,
                manifest.capability_version,
                config_ref,
                dependencies,
                tuple(sorted(outputs, key=lambda o: o.output_id)),
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
                if task == FREEZE
                else manifest.permissions,
                OutcomeAccess.OUTCOME_BLIND
                if task == FREEZE
                else OutcomeAccess.EVALUATOR_REVEAL,
                VisibilityCeiling.PROSPECTIVE,
                manifest.resource_ceiling,
                (),
                BarrierKind.FREEZE if task == FREEZE else BarrierKind.REVEAL,
                1,
                (obligation,),
            )
        )
    template = _programme_template(
        ProtocolTemplate(
            f"{PREFIX}.protocol",
            "1.0.0",
            tuple(sorted(steps, key=lambda s: s.step_id)),
            False,
            False,
            True,
        ),
        config,
        config.config_id,
        experiment,
        registry,
        prefix=PREFIX,
        source_task_id=FREEZE,
        terminal_task_id=ADJUDICATE,
        primary_output_ids={
            ScientificStage.EVALUATE: "report",
            ScientificStage.FREEZE: "report",
        },
    )
    external, edges, sources, qualifications = (
        [],
        [e for e in template.graph.edges if e.producer_node_id is not None],
        [],
        [],
    )
    for artifact in config.inputs:
        is_native = artifact in (config.native_evaluation, config.native_receipt)
        access = (
            OutcomeAccess.EVALUATOR_REVEAL if is_native else OutcomeAccess.OUTCOME_BLIND
        )
        role = (
            ScientificInputRole.PARENT_RECEIPT
            if artifact == config.native_receipt
            else ScientificInputRole.OUTCOME
            if is_native
            else ScientificInputRole.MODEL
        )
        external.append(
            CandidateGraphExternalInput(
                artifact.artifact_id,
                role,
                artifact.artifact_id,
                ContentIdentityPolicy.EXACT_SHA256,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                access,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
        for task in (CALIBRATE, QUALIFY):
            edges.append(
                CandidateGraphEdge(
                    f"edge.{task}.{artifact.artifact_id}",
                    None,
                    None,
                    artifact.artifact_id,
                    task,
                    f"input.{artifact.artifact_id}",
                    role,
                    artifact.artifact_id,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    access,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.REVEAL if is_native else BarrierKind.NONE,
                )
            )
        if is_native:
            # Protected outcomes/receipts are exact evaluator inputs, not
            # outcome-blind materialization-qualification attestations.
            continue
        identity = ObjectIdentity(
            artifact.artifact_id, artifact.payload_schema, "1.0.0", artifact.sha256
        )
        qualification = MaterializationQualificationReceipt(
            f"qualification.{artifact.artifact_id}",
            artifact.artifact_id,
            identity,
            artifact.sha256,
            system.world.world_id,
            ref,
            tuple(v.view_id for v in system.numerical_views),
            tuple(sorted({q.native_unit for q in system.quantities})),
            (Q_FRAME,),
            (Q_CLOCK,),
            system.relation.relation_id,
            experiment.obligations.validity.validity_id,
            experiment.obligations.uncertainty.uncertainty_id,
            SourceAccessDisposition.VERIFIED_ACCESS,
            access,
            VisibilityCeiling.PROSPECTIVE,
        )
        qualifications.append(qualification)
        sources.append(
            SourceMaterializationRef(
                artifact.artifact_id,
                SourceMaterializationRole.CALIBRATION
                if is_native
                else SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                system.world.world_id,
                identity,
                artifact.sha256,
                config.fingerprint(),
                ref,
                qualification.numerical_view_ids,
                ObjectIdentity.from_record(qualification.receipt_id, qualification),
                SourceAccessDisposition.VERIFIED_ACCESS,
            )
        )
    graph = replace(
        template.graph,
        external_inputs=tuple(sorted(external, key=lambda e: e.input_id)),
        edges=tuple(sorted(edges, key=lambda e: e.edge_id)),
    )
    template = replace(
        template,
        graph=graph,
        coverage=replace(
            template.coverage,
            bindings=tuple(
                replace(
                    b,
                    contributor_edge_ids=tuple(
                        sorted(
                            e.edge_id
                            for e in graph.edges
                            if e.consumer_node_id == b.proof_owner_node_id
                        )
                    ),
                )
                for b in template.coverage.bindings
            ),
        ),
    )
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=BUDGET,
        objective='Measure complete-root uncertainty and finite-menu usability using the frozen fitted laws. Preserve valid negative and unavailable boundaries.',
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    draft = replace(
        native.authoring.base.draft,
        draft_id=f"{PREFIX}.draft",
        question=campaign.objective,
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=(
            CapabilitySelection(
                manifest.capability_key,
                manifest.capability_version,
                manifest.implementation_sha256,
            ),
        ),
        source_materializations=tuple(sorted(sources, key=lambda s: s.source_id)),
        resource_ceiling=BUDGET,
    )
    context = CandidateCompilationContext(
        f"{PREFIX}.context",
        registry,
        (template,),
        tuple(sorted(qualifications, key=lambda q: q.receipt_id)),
        draft.design_inputs,
        implementation_sha256,
    )
    science = next(
        d.object_identity
        for d in draft.design_inputs
        if d.input_id.endswith(".science")
    )
    base, standard = _entry(
        draft,
        context,
        science,
        prefix=PREFIX,
        minimum_units=32,
        operand_description="Frozen joint predictive {domain} assessment on all 32 assigned roots, with explicit infinity and custody refusal.",
        estimator="frozen-rank-30",
        uncertainty="joint-complete-root-90",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=manifest,
        input_schema=FiniteResponseLawAssignedCalibrationReport.SCHEMA,
        output_schema=FiniteResponseLawAssignedQualificationReport.SCHEMA,
        evidence_unit_ids=tuple(r.physical_unit_id for r in config.native_source.roots),
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    (decoder,) = METHOD_BINDING.issued_decoder_registrations
    proposed = ProposedStudyExtension(
        f"{PREFIX}.extension",
        f"{PREFIX}.namespace",
        ref,
        len(config.canonical_bytes()),
        decoder.decoder_key,
        decoder.decoder_version,
        decoder.config_sha256,
        True,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    extensions = ProposedStudyExtensionSet(
        f"{PREFIX}.extensions",
        ObjectIdentity.from_record(base.package_id, base),
        (proposed.namespace_id,),
        (proposed,),
    )
    evidence = bind_profile_selection(
        selection_id=f"{PREFIX}.evidence-profile",
        draft_id=draft.draft_id,
        registry=build_observation_evidence_world_registry(),
        world_profile_id=native.evidence_profile.evidence_world_profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(
            EvidenceRung.LOCAL_LAW,
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
        ),
    )
    catalog = CandidateCapabilityCatalog(
        f"{PREFIX}.catalog",
        tuple(
            r
            for r in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
            if r.manifest.capability_key == manifest.capability_key
        ),
        (template,),
    )
    return ResponseGeometryAssayAuthoringBundle(
        ExecutableStudyDefinition(f"{PREFIX}.executable-study-definition", base, extensions),
        standard,
        catalog,
        (config,),
        (decoder,),
        evidence,
    )
