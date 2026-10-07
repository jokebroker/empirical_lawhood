"""Two nonacquiring tasks on the ordinary authoring/issue route."""
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import replace
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_profiles import bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection
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
from empirical_lawhood.runtime.capabilities import CapabilityRegistry, CapabilityPermission
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref
from empirical_lawhood.adapters.methods.finite_response_law.completion.contracts import FiniteResponseLawAssignedRetainedCompletionConfig, FiniteResponseLawRetainedCompletionConfig
from empirical_lawhood.adapters.methods.finite_response_law.completion.extension_bundle import ASSIGNED_CAPABILITY, CAPABILITY
from empirical_lawhood.adapters.methods.finite_response_law.completion.executable_binding import ASSIGNED_BINDING, BINDING
from empirical_lawhood.adapters.methods.finite_response_law.native_records import native_method_types
from .authoring import build_native_authoring
from .exposure import native_seed_ids

BUDGET = ResourceBudget(1, 1024**3, 0, 540, 288 * 1024**2, 48 * 1024**2)


def build_completion_authoring(
    *,
    config: FiniteResponseLawRetainedCompletionConfig,
    implementation_sha256: str,
    exposure: ObjectIdentity,
    exposed_unit_ids: tuple[str, ...],
    exposed_seed_ids: tuple[str, ...],
) -> ResponseGeometryAssayAuthoringBundle:
    if type(config) not in (
        FiniteResponseLawRetainedCompletionConfig,
        FiniteResponseLawAssignedRetainedCompletionConfig,
    ):
        raise ValueError("Retained completion requires a closed versioned configuration")
    assigned = type(config) is FiniteResponseLawAssignedRetainedCompletionConfig
    capability = ASSIGNED_CAPABILITY if assigned else CAPABILITY
    binding = ASSIGNED_BINDING if assigned else BINDING
    prefix = config.study_prefix
    freeze, aggregate, adjudicate = (
        config.freeze_task_id,
        config.aggregate_task_id,
        config.adjudicate_task_id,
    )
    _, _, _, view_type, completion_type = native_method_types(config.native_source)
    native = build_native_authoring(
        source=config.native_source,
        implementation_sha256=implementation_sha256,
        exposure=exposure,
        exposed_unit_ids=exposed_unit_ids,
        exposed_seed_ids=exposed_seed_ids,
        new_seed_ids=() if assigned else native_seed_ids(config.native_source),
        control=config.control if assigned else None,
        retention=config.retention if assigned else None,
    )
    original = native.authoring.base.draft
    assert original.system is not None and original.experiment is not None
    system, experiment = original.system, original.experiment
    experiment = replace(
        experiment,
        experiment_id=prefix,
        claims=tuple(
            replace(
                c,
                proposition=f"Complete the original native measurement census from all {len(config.projections)} authenticated retained projections; no new acquisition, law qualification or use claim.",
            )
            for c in experiment.claims
        ),
        controls=tuple(
            replace(c, capability_key=capability.capability_key) for c in experiment.controls
        ),
        obligations=replace(
            experiment.obligations,
            falsifiers=tuple(
                replace(f, capability_key=capability.capability_key)
                for f in experiment.obligations.falsifiers
            ),
        ),
    )
    registry = CapabilityRegistry(f"{prefix}.registry", (capability,))
    identity = ObjectIdentity.from_record(config.config_id, config)
    config_ref = _config_ref(config, identity, capability)
    steps = []
    for task, stage, products, dependencies in (
        (freeze, ScientificStage.FREEZE, (("report", ObjectIdentity.SCHEMA),), ()),
        (
            aggregate,
            ScientificStage.EVALUATE,
            (
                ("report", completion_type.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ),
            (freeze,),
        ),
        (
            adjudicate,
            ScientificStage.EVALUATE,
            (("report", ScientificAdjudicationRecord.SCHEMA),),
            (aggregate,),
        ),
    ):
        steps.append(
            ProtocolStepTemplate(
                task,
                stage,
                capability.capability_key,
                capability.capability_version,
                config_ref,
                dependencies,
                tuple(
                    OutputTemplate(
                        name,
                        schema,
                        ArtifactProfile.CANONICAL_JSON,
                        "application/vnd.empirical-lawhood.canonical+json",
                        ".json",
                    )
                    for name, schema in products
                ),
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
                if task == freeze
                else capability.permissions,
                OutcomeAccess.OUTCOME_BLIND if task == freeze else OutcomeAccess.EVALUATOR_REVEAL,
                VisibilityCeiling.PROSPECTIVE,
                capability.resource_ceiling,
                (),
                BarrierKind.FREEZE if task == freeze else BarrierKind.REVEAL,
                1,
                (
                    f"{prefix}.single-terminal"
                    if task == adjudicate
                    else f"{prefix}.retained-custody",
                ),
            )
        )
    template = _programme_template(
        ProtocolTemplate(
            f"{prefix}.protocol",
            "1.0.0",
            tuple(sorted(steps, key=lambda s: s.step_id)),
            False,
            False,
            True,
        ),
        config.native_source,
        config.native_source.spec_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id=freeze,
        terminal_task_id=adjudicate,
        primary_output_ids={ScientificStage.EVALUATE: "report", ScientificStage.FREEZE: "report"},
    )
    external = list(template.graph.external_inputs)
    edges = list(template.graph.edges)
    receipt_ids = {row.receipt.artifact_id for row in config.projections}
    for artifact in config.inputs:
        role = (
            ScientificInputRole.PARENT_RECEIPT
            if artifact.artifact_id in receipt_ids
            else ScientificInputRole.OUTCOME
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
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
        edges.append(
            CandidateGraphEdge(
                f"edge.{aggregate}.{artifact.artifact_id}",
                None,
                None,
                artifact.artifact_id,
                aggregate,
                f"input.{artifact.artifact_id}",
                role,
                artifact.artifact_id,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.REVEAL,
            )
        )
    graph = replace(
        template.graph,
        external_inputs=tuple(sorted(external, key=lambda a: a.input_id)),
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
        prefix=prefix,
        budget=BUDGET,
        objective=f"Complete and separately adjudicate the retained {len(config.native_source.roots)}-root native measurement census without reacquisition.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    draft = replace(
        original,
        draft_id=f"{prefix}.draft",
        question=campaign.objective,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=(
            CapabilitySelection(
                capability.capability_key,
                capability.capability_version,
                capability.implementation_sha256,
            ),
        ),
        resource_ceiling=BUDGET,
        source_materializations=original.source_materializations,
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        native.standard_context.base.qualifications,
        draft.design_inputs,
        implementation_sha256,
    )
    science = next(
        d.object_identity for d in draft.design_inputs if d.input_id.endswith(".science")
    )
    base, standard = _entry(
        draft,
        context,
        science,
        prefix=prefix,
        minimum_units=len(config.native_source.roots),
        operand_description="Retained native {domain} measurement census; all original projection receipts required.",
        estimator="unchanged-native-measurement-aggregation",
        uncertainty="original-native-numerical-views",
        ceiling=EvidenceCeiling.MEASUREMENT,
        evaluator=capability,
        input_schema=view_type.SCHEMA,
        output_schema=completion_type.SCHEMA,
        evidence_unit_ids=tuple(r.physical_unit_id for r in config.native_source.roots),
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    (decoder,) = binding.issued_decoder_registrations
    proposed = ProposedStudyExtension(
        f"{prefix}.extension",
        f"{prefix}.namespace",
        identity,
        len(config.canonical_bytes()),
        decoder.decoder_key,
        decoder.decoder_version,
        decoder.config_sha256,
        True,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    extensions = ProposedStudyExtensionSet(
        f"{prefix}.extensions",
        ObjectIdentity.from_record(base.package_id, base),
        (proposed.namespace_id,),
        (proposed,),
    )
    evidence = bind_profile_selection(
        selection_id=f"{prefix}.evidence-profile",
        draft_id=draft.draft_id,
        registry=build_observation_evidence_world_registry(),
        world_profile_id=native.evidence_profile.evidence_world_profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(EvidenceRung.MEASUREMENT,),
    )
    catalog = CandidateCapabilityCatalog(
        f"{prefix}.catalog",
        tuple(
            r
            for r in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
            if r.manifest.capability_key == capability.capability_key
        ),
        (template,),
    )
    return ResponseGeometryAssayAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.executable-study-definition", base, extensions),
        standard,
        catalog,
        (config,),
        (decoder,),
        evidence,
    )
