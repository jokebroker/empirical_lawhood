"""Outcome-aware development authoring for the bounded preparation-policy development preparation-policy screen."""
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

from empirical_lawhood.adapters.composition.finite_response_law.design import native_budget, native_experiment, native_system
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_EVALUATOR_TASK_ID
from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyScreenConfig
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.discovery import EVALUATION_CAPABILITY, PROJECTION_CAPABILITY, SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.executable_binding import EVALUATION_BINDING, PROJECTION_BINDING, SOURCE_BINDING
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.protocol import preparation_policy_protocol_steps
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.roster import Q_RECEIVERS, preparation_policy_declarations, preparation_policy_substrate_binding
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, preparation_policy_native_invocations
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_profiles import EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateCompilationContext,
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolTemplate, ScientificStage


def _record_id(record: CanonicalRecord) -> str:
    for field in ("extension_set_id", "binding_id", "config_id", "spec_id"):
        value = getattr(record, field, None)
        if isinstance(value, str):
            return value
    raise ValueError("preparation-policy development issued record lacks a stable identity")


def _artifact_external(
    artifact: ArtifactIdentity, role: ScientificInputRole
) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        artifact.artifact_id,
        role,
        artifact.artifact_id,
        ContentIdentityPolicy.EXACT_SHA256,
        artifact.sha256,
        artifact.payload_schema,
        artifact.media_type,
        artifact.size_bytes,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.OUTCOME_VISIBLE,
    )


def build_preparation_policy_authoring(
    *,
    source: FiniteResponseLawPreparationPolicyNativeConfig,
    screen: FiniteResponseLawPreparationPolicyScreenConfig,
    implementation_sha256: str,
    prospective_eligibility: ObjectIdentity,
) -> ResponseGeometryAssayAuthoringBundle:
    """Build the exact public-route package without contacting retained sources."""
    raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
    projection = screen.projection
    if projection.native_spec != source:
        raise ValueError("preparation-policy development screen and source configurations differ")
    capabilities = (EVALUATION_CAPABILITY, PROJECTION_CAPABILITY, SOURCE_CAPABILITY)
    bindings = (EVALUATION_BINDING, PROJECTION_BINDING, SOURCE_BINDING)
    selected = {capability.capability_key for capability in capabilities}
    registry = CapabilityRegistry(
        f"{PROGRAMME}.preparation-screening.registry",
        tuple(
            capability
            for capability in GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry.capabilities
            if capability.capability_key in selected
        ),
    )
    prefix = f"{PROGRAMME}.preparation-policy"
    world_registry = build_observation_evidence_world_registry()
    world = next(
        value
        for value in world_registry.world_profiles
        if value.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{prefix}.evidence-profile",
        draft_id=f"{prefix}.draft",
        registry=world_registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
        ),
    )
    declarations = preparation_policy_declarations(source)
    carrier = PredecessorBoundSourceQualificationExperiment(
        f"{prefix}.source-qualification",
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        declarations.units,
        declarations.segments,
        declarations.views,
        Q_RECEIVERS,
        (Q_CLOCK,),
        ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
        ObjectIdentity.from_record(PROJECTION_CAPABILITY.capability_key, PROJECTION_CAPABILITY),
        ObjectIdentity.from_record(EVALUATION_CAPABILITY.capability_key, EVALUATION_CAPABILITY),
        ObjectIdentity.from_record(source.spec_id, source),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(screen.config_id, screen),
        PREPARATION_POLICY_EVALUATOR_TASK_ID,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        retained_predecessors=tuple(row.declaration for row in source.retained_prefixes),
    )
    association = preparation_policy_substrate_binding(
        source,
        carrier,
        ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
    )
    identified = tuple(
        sorted(
            (
                (source.spec_id, source),
                (projection.config_id, projection),
                (screen.config_id, screen),
                (carrier.extension_set_id, carrier),
                (association.binding_id, association),
            ),
            key=lambda value: value[1].SCHEMA,
        )
    )
    payloads = tuple(record for _, record in identified)
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in bindings
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[record.SCHEMA] for record in payloads)

    # Reuse the established native denominator owner, changing only the preparation-policy
    # development claim and the descriptions that differ from Tier 1.
    system = native_system(source)  # type: ignore[arg-type]
    system = replace(
        system,
        system_id=f"{prefix}.system",
        label="Nine-schedule retained-root preparation-policy development screen",
        world=replace(system.world, maximum_evidence=EvidenceCeiling.RESPONSE),
        authority_policy=replace(
            system.authority_policy,
            policy_id=f"{prefix}.authority-policy",
            scope_ids=(prefix,),
            budget_ceiling=native_budget(source),  # type: ignore[arg-type]
        ),
    )
    experiment_source = SimpleNamespace(
        stage=source.stage,
        science=source.science,
        roots=tuple(sorted(source.roots, key=lambda root: root.physical_unit_id)),
    )
    base_experiment = native_experiment(experiment_source, system)  # type: ignore[arg-type]
    experiment = replace(
        base_experiment,
        experiment_id=prefix,
        claims=(
            replace(
                base_experiment.claims[0],
                claim_id=f"{prefix}.development-screen",
                proposition="The frozen preparation-policy development cross-fitted upper-composition screen passes all three declared development gates on the exact 24 retained roots.",
                estimand="Separate full-menu adequacy A, actual joint-consumer success J, and C=A AND J under nine +400 preparation schedules.",
                requested_rung=EvidenceRung.RESPONSE,
                evidence_ceiling=EvidenceCeiling.RESPONSE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                promotion_rule="A positive result freezes the upper package and permits prospective preparation-policy confirmation entry; a valid negative closes prospective preparation-policy confirmation and later dependent stages without retuning.",
            ),
        ),
        controls=tuple(
            replace(control, capability_key=EVALUATION_CAPABILITY.capability_key)
            for control in base_experiment.controls
        ),
        obligations=replace(
            base_experiment.obligations,
            uncertainty=replace(
                base_experiment.obligations.uncertainty,
                method_key=f"{prefix}.four-outer-three-coefficient-rank18-provisional",
                limitation_codes=(
                    "DEVELOPMENT_SCREEN_NOT_PROSPECTIVE_CONFIRMATION",
                    "PROVISIONAL_CROSSFITTED_UNCERTAINTY",
                ),
                status=ObligationStatus.REQUIRED,
            ),
            falsifiers=tuple(
                replace(
                    falsifier,
                    capability_key=EVALUATION_CAPABILITY.capability_key,
                    description="Fewer than six headroom roots, selected-minus-fixed adequacy below four roots, or actual J improvement below 0.05 fails preparation-policy development.",
                    decisive_rule="Retain all roots and missingness; no reschedule, refit after reveal, replacement root, or gate reinterpretation.",
                )
                for falsifier in base_experiment.obligations.falsifiers
            ),
        ),
        reveal_barrier=replace(
            base_experiment.reveal_barrier,
            development_unit_ids=tuple(sorted(root.physical_unit_id for root in source.roots)),
            evaluation_cohort_id=f"{prefix}.exact-24-retained-roots",
            sealed_outcome_artifact_ids=(f"{prefix}.sealed-root-panels",),
        ),
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
    )
    steps = preparation_policy_protocol_steps(source, carrier, projection, screen)
    protocol = ProtocolTemplate(f"{prefix}.protocol", "1.0.0", steps, False, False, True)
    template = _programme_template(
        protocol,
        source,
        source.spec_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id=preparation_policy_native_invocations(source)[0].task_id,
        terminal_task_id=PREPARATION_POLICY_EVALUATOR_TASK_ID,
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.TRANSFORM: "root-panel",
            ScientificStage.EVALUATE: "screen-result",
        },
    )
    graph = template.graph
    config_edge = next(edge for edge in graph.edges if edge.external_input_id == source.spec_id)
    edges = [edge for edge in graph.edges if edge != config_edge]
    external = list(graph.external_inputs)
    artifacts = tuple(
        sorted(
            (
                *(
                    artifact
                    for row in source.retained_prefixes
                    for artifact in (*row.declaration.artifacts, row.declaration.task_receipt)
                ),
                screen.lower_artifact,
                screen.lower_qualification,
                screen.prospective_adjudication,
                screen.prospective_closeout,
            ),
            key=lambda artifact: artifact.artifact_id,
        )
    )
    receipt_ids = {
        row.declaration.task_receipt.artifact_id for row in source.retained_prefixes
    }
    evidence_ids = {
        screen.lower_artifact.artifact_id,
        screen.lower_qualification.artifact_id,
        screen.prospective_adjudication.artifact_id,
        screen.prospective_closeout.artifact_id,
    }
    external.extend(
        _artifact_external(
            artifact,
            ScientificInputRole.QUALIFICATION
            if artifact.artifact_id in evidence_ids
            else ScientificInputRole.PARENT_RECEIPT
            if artifact.artifact_id in receipt_ids
            else ScientificInputRole.PREPARED_MEDIUM,
        )
        for artifact in artifacts
    )
    retained = {
        row.declaration.segment_id: row.declaration for row in source.retained_prefixes
    }
    for task in preparation_policy_native_invocations(source):
        if not task.dependency_task_ids:
            edges.append(
                replace(
                    config_edge,
                    edge_id=f"{config_edge.edge_id}.{task.task_id}",
                    consumer_node_id=task.task_id,
                )
            )
        prior = retained.get(task.predecessor_segment_id or "")
        if prior is None:
            continue
        for index, artifact in enumerate((*prior.artifacts, prior.task_receipt)):
            edges.append(
                CandidateGraphEdge(
                    f"edge.{task.task_id}.retained.{index}",
                    None,
                    None,
                    artifact.artifact_id,
                    task.task_id,
                    f"retained-{index}",
                    ScientificInputRole.PARENT_RECEIPT
                    if artifact == prior.task_receipt
                    else ScientificInputRole.PREPARED_MEDIUM,
                    artifact.artifact_id,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    prior.outcome_access,
                    prior.visibility_ceiling,
                    BarrierKind.NONE,
                )
            )
    for artifact in (
        screen.lower_artifact,
        screen.lower_qualification,
        screen.prospective_adjudication,
        screen.prospective_closeout,
    ):
        edges.append(
            CandidateGraphEdge(
                f"edge.{PREPARATION_POLICY_EVALUATOR_TASK_ID}.{artifact.artifact_id}",
                None,
                None,
                artifact.artifact_id,
                PREPARATION_POLICY_EVALUATOR_TASK_ID,
                f"entry-{artifact.artifact_id}",
                ScientificInputRole.QUALIFICATION,
                artifact.artifact_id,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.OUTCOME_VISIBLE,
                BarrierKind.REVEAL,
            )
        )
    graph = replace(
        graph,
        external_inputs=tuple(sorted(external, key=lambda value: value.input_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )
    coverage = replace(
        template.coverage,
        bindings=tuple(
            replace(
                binding,
                contributor_edge_ids=tuple(
                    sorted(
                        edge.edge_id
                        for edge in graph.edges
                        if edge.consumer_node_id == binding.proof_owner_node_id
                    )
                ),
            )
            for binding in template.coverage.bindings
        ),
    )
    template = replace(template, graph=graph, coverage=coverage)
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=native_budget(source),  # type: ignore[arg-type]
        objective="Run the exact preparation-policy development development screen and freeze an upper package only if all three declared gates pass.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    science = ObjectIdentity(
        f"{PROGRAMME}.frozen-plan",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        source.science.plan_sha256,
    )
    cutoff = InformationCutoff(f"{prefix}.design-cutoff", Q_CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    design_inputs = (
        DesignInputRecord(
            f"{prefix}.prospective-evaluation-eligibility",
            prospective_eligibility,
            prospective_eligibility.object_fingerprint,
            cutoff,
            DesignInputRole.DEVELOPMENT_TUNING,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.OUTCOME_VISIBLE,
            "owner.empirical-lawhood",
            tuple(sorted(root.physical_unit_id for root in source.roots)),
            (),
        ),
        DesignInputRecord(
            f"{prefix}.science",
            science,
            science.object_fingerprint,
            cutoff,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.OUTCOME_VISIBLE,
            "owner.empirical-lawhood",
            tuple(sorted(root.physical_unit_id for root in source.roots)),
            (),
        ),
    )
    config_identity = ObjectIdentity.from_record(source.spec_id, source)
    qualification = MaterializationQualificationReceipt(
        f"{prefix}.config-qualification",
        source.spec_id,
        config_identity,
        source.fingerprint(),
        system.world.world_id,
        ObjectIdentity.from_record(projection.config_id, projection),
        tuple(view.view_id for view in system.numerical_views),
        tuple(sorted({quantity.native_unit for quantity in system.quantities})),
        (Q_FRAME,),
        (Q_CLOCK,),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        source.spec_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        config_identity,
        source.fingerprint(),
        source.fingerprint(),
        qualification.observation_operator,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    qualifications = [qualification]
    sources = [source_ref]
    for artifact in artifacts:
        if artifact.artifact_id in receipt_ids:
            # The receipt is consumed as a lineage-bound parent receipt. It is
            # not a prepared-medium materialization in the compiler's role map.
            continue
        identity = ObjectIdentity(
            artifact.artifact_id, artifact.payload_schema, "1.0.0", artifact.sha256
        )
        receipt = replace(
            qualification,
            receipt_id=f"qualification.{artifact.artifact_id}",
            source_id=artifact.artifact_id,
            materialization=identity,
            content_sha256=artifact.sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        qualifications.append(receipt)
        sources.append(
            replace(
                source_ref,
                source_id=artifact.artifact_id,
                role=SourceMaterializationRole.CALIBRATION
                if artifact in (
                    screen.lower_artifact,
                    screen.lower_qualification,
                    screen.prospective_adjudication,
                    screen.prospective_closeout,
                )
                else SourceMaterializationRole.PREPARED_MEDIUM,
                materialization=identity,
                content_sha256=artifact.sha256,
                qualification_receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
            )
        )
    units = tuple(sorted(root.physical_unit_id for root in source.roots))
    draft = StudyDraft(
        f"{prefix}.draft",
        StudyDraftLifecycle.DRAFT,
        campaign.objective,
        (f"{prefix}.alternative.freeze-upper", f"{prefix}.alternative.valid-negative"),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(value.input_id for value in design_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        units,
        (),
        (),
        (),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                capability.capability_key,
                capability.capability_version,
                capability.implementation_sha256,
            )
            for capability in capabilities
        ),
        tuple(sorted(sources, key=lambda value: value.source_id)),
        native_budget(source),  # type: ignore[arg-type]
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        tuple(sorted(qualifications, key=lambda value: value.receipt_id)),
        design_inputs,
        implementation_sha256,
    )
    expanded = compose_parameterised_candidate_context_for_template(
        template_key=template.template_key,
        context=context,
        records=payloads,
        factories=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
    )
    if not isinstance(expanded, CandidateCompilationContext):
        raise TypeError("preparation-policy development requires its installed parameterised candidate context")
    base, standard = _entry(
        draft,
        expanded,
        science,
        prefix=prefix,
        minimum_units=24,
        operand_description="Nine-schedule preparation-policy development {domain} development screen on retained roots; no prospective preparation-policy confirmation.",
        estimator="four-outer-three-coefficient-u2-provider-screen",
        uncertainty="rank18-provisional-cross-fitted",
        ceiling=EvidenceCeiling.RESPONSE,
        evaluator=EVALUATION_CAPABILITY,
        input_schema='empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-root-panel',
        output_schema=FiniteResponseLawPreparationScreenResult.SCHEMA,
        evidence_unit_ids=units,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    proposed = tuple(
        ProposedStudyExtension(
            f"{prefix}.extension.{index:02d}",
            f"{prefix}.namespace.{index:02d}",
            ObjectIdentity.from_record(record_id, record),
            len(record.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for index, ((record_id, record), decoder) in enumerate(
            zip(identified, decoders, strict=True)
        )
    )
    extensions = ProposedStudyExtensionSet(
        f"{prefix}.extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(value.namespace_id for value in proposed),
        proposed,
    )
    return ResponseGeometryAssayAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.executable-study-definition", base, extensions),
        replace(standard, base=context),
        CandidateCapabilityCatalog(
            f"{prefix}.catalog",
            tuple(
                registration
                for registration in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if registration.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )
