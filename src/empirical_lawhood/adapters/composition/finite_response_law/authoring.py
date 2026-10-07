"""Finite native authoring on the existing candidate/issue/qualification route.

The source config distinguishes retained development, canary and fresh calibration
units. The source-only terminal establishes measurement availability; law
qualification, the finite consumer and prospective use have separate gates.
"""
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import replace, fields
from decimal import Decimal

from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_profiles import EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.planning.source_qualification import QualifiedSourceUseExperiment, ProspectiveRetainedSourceUse, ProspectiveRetainedSourceQualification, QualifiedSourceUseStage
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, ContentIdentityPolicy, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry, CapabilityManifest
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityBinding
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolTemplate, ScientificStage
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.finite_response_law.native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeEvaluationConfig
from empirical_lawhood.adapters.methods.finite_response_law.native_provider import EVALUATOR_TASK_ID
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig, native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.protocol import native_protocol_steps
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME, Q_RECEIVERS, native_declarations, substrate_binding
from .design import native_budget, native_experiment, native_system
from .exposure import native_seed_ids, native_seed_collisions
from empirical_lawhood.adapters.simulators.finite_response_law.protocol import native_capabilities
from empirical_lawhood.adapters.simulators.finite_response_law.executable_binding import native_bindings
from empirical_lawhood.adapters.methods.finite_response_law.native_records import native_method_types
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig, FiniteResponseLawEvaluationCohort, COHORT
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationCompletionConfig


from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_retention import FiniteResponseLawEvaluationRetention


def _template(
    source: FiniteResponseLawNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    projection: FiniteResponseLawProjectionConfig,
    evaluation: FiniteResponseLawNativeEvaluationConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    control: FiniteResponseLawControlConfig | None = None,
    retention: FiniteResponseLawEvaluationRetention | None = None,
) -> StudyTemplate:
    prefix = f"{PROGRAMME}.{source.stage}"
    if type(source) is FiniteResponseLawAssignedEvaluationConfig:
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.protocol import evaluation_protocol_steps

        if (
            control is None
            or type(projection) is not FiniteResponseLawAssignedEvaluationProjectionConfig
            or type(evaluation) is not FiniteResponseLawAssignedEvaluationCompletionConfig
        ):
            raise ValueError("Independent evaluation cannot author a source route without its complete causal controls")
        steps = evaluation_protocol_steps(
            source,
            carrier,
            projection,
            evaluation,
            control,
            FiniteResponseLawEvaluationRevealConfig(control),
            retention,
        )
    else:
        if control is not None:
            raise ValueError("Independent evaluation controls cannot reinterpret a predecessor native tranche")
        steps = native_protocol_steps(source, carrier, projection, evaluation)
    native = native_invocations(source)
    base = _programme_template(
        ProtocolTemplate(
            f"{prefix}.protocol",
            "1.0.0",
            steps,
            control is not None,
            control is not None,
            control is None,
        ),
        source,
        source.spec_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id=native[0].task_id,
        terminal_task_id=COHORT if control else EVALUATOR_TASK_ID,
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.TRANSFORM: "product" if control else "view-report",
            ScientificStage.EVALUATE: "product" if control else "evaluation",
            **(
                {
                    ScientificStage.CONTROLLER: "product",
                    ScientificStage.FREEZE: "product",
                    ScientificStage.REVEAL: "scientific-adjudication",
                }
                if control
                else {}
            ),
        },
    )
    # The full adapter-owned protocol already names every consumer. Expansion
    # therefore preserves exact root-specific edges, rather than cloning a
    # prototype with every retained artifact into every source task.
    edges = [e for e in base.graph.edges if e.producer_node_id is not None]
    config_edge = next(e for e in base.graph.edges if e.external_input_id == source.spec_id)
    external = [
        e for e in base.graph.external_inputs if retention is None or e.input_id != source.spec_id
    ]
    for prior in carrier.retained_predecessors:
        for artifact in (*prior.artifacts, prior.task_receipt):
            role = (
                ScientificInputRole.PARENT_RECEIPT
                if artifact == prior.task_receipt
                else ScientificInputRole.PREPARED_MEDIUM
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
                    prior.outcome_access,
                    prior.visibility_ceiling,
                )
            )
    if control is not None:
        from empirical_lawhood.adapters.methods.finite_response_law.control_provider import control_tasks

        for artifact in control.inputs:
            role = (
                ScientificInputRole.PARENT_RECEIPT
                if artifact in (control.qualification_receipt, control.calibration_native_receipt)
                else ScientificInputRole.QUALIFICATION
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
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                )
            )
            for task_id, (_, operation) in control_tasks(control).items():
                if operation not in ("forecast", "lock-choices", "freeze-prior"):
                    continue
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{task_id}.{artifact.artifact_id}",
                        None,
                        None,
                        artifact.artifact_id,
                        task_id,
                        f"input.{artifact.artifact_id}",
                        role,
                        artifact.artifact_id,
                        artifact.payload_schema,
                        artifact.media_type,
                        artifact.size_bytes,
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                    )
                )
    priors = {p.segment_id: p for p in carrier.retained_predecessors}
    for task in native:
        if not task.dependency_task_ids and retention is None:
            edges.append(
                replace(
                    config_edge,
                    edge_id=f"{config_edge.edge_id}.{task.task_id}",
                    consumer_node_id=task.task_id,
                )
            )
        predecessor = priors.get(
            task.task_id if retention is not None else task.predecessor_segment_id or ""
        )
        if predecessor is None:
            continue
        for index, artifact in enumerate((*predecessor.artifacts, predecessor.task_receipt)):
            role = (
                ScientificInputRole.PARENT_RECEIPT
                if artifact == predecessor.task_receipt
                else ScientificInputRole.PREPARED_MEDIUM
            )
            edges.append(
                CandidateGraphEdge(
                    f"edge.{task.task_id}.retained.{index}",
                    None,
                    None,
                    artifact.artifact_id,
                    task.task_id,
                    f"retained-{index}",
                    role,
                    artifact.artifact_id,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    predecessor.outcome_access,
                    predecessor.visibility_ceiling,
                    BarrierKind.NONE,
                )
            )
    graph = replace(
        base.graph,
        external_inputs=tuple(sorted(external, key=lambda e: e.input_id)),
        edges=tuple(sorted(edges, key=lambda e: e.edge_id)),
    )
    if control is not None:
        by_id = {s.step_id: s for s in steps}
        graph = replace(
            graph,
            edges=tuple(
                replace(
                    e,
                    outcome_access=by_id[e.producer_node_id].requested_outcome_access,
                    visibility_ceiling=by_id[e.producer_node_id].visibility_ceiling,
                )
                if e.producer_node_id is not None
                else e
                for e in graph.edges
            ),
        )
    coverage = replace(
        base.coverage,
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
            for b in base.coverage.bindings
        ),
    )
    return replace(base, graph=graph, coverage=coverage)


def build_native_authoring(
    *,
    source: FiniteResponseLawNativeConfig,
    implementation_sha256: str,
    exposure: ObjectIdentity,
    exposed_unit_ids: tuple[str, ...],
    exposed_seed_ids: tuple[str, ...],
    new_seed_ids: tuple[str, ...],
    control: FiniteResponseLawControlConfig | None = None,
    retention: FiniteResponseLawEvaluationRetention | None = None,
) -> ResponseGeometryAssayAuthoringBundle:
    """Consume operator-authenticated exposure metadata; never inspect sources."""
    source_capability, projection_capability, evaluation_capability = native_capabilities(source)
    if (
        (type(source) is FiniteResponseLawAssignedEvaluationConfig) != (control is not None)
        or control is not None
        and control.source != source
    ):
        raise ValueError("Fresh independent evaluation authoring requires its exact causal-control configuration")
    source_binding, projection_binding, evaluation_binding = native_bindings(source)
    if retention is not None:
        if retention.source != source or control is None:
            raise ValueError("Independent evaluation retention requires unchanged source and full causal controls")
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation.discovery import SOURCE_CAPABILITY as CONTINUATION_SOURCE, IMPORT_CAPABILITY
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation.executable_binding import SOURCE_BINDING as CONTINUATION_BINDING, IMPORT_BINDING

        source_capability, source_binding = CONTINUATION_SOURCE, CONTINUATION_BINDING
    projection_type, evaluation_config_type, _, view_type, evaluation_type = native_method_types(
        source
    )
    prefix = f"{PROGRAMME}.{source.stage}"
    units = tuple(r.physical_unit_id for r in source.roots)
    collisions = set(units).intersection(exposed_unit_ids)
    expected = set(units) if source.stage == "supplemental-development" or retention is not None else set()
    assigned_seeds = set(native_seed_ids(source))
    if (
        not exposed_unit_ids
        or not exposed_seed_ids
        or collisions != expected
        or (
            retention is not None
            and (new_seed_ids or not assigned_seeds <= set(exposed_seed_ids))
        )
        or (
            retention is None
            and (
                new_seed_ids != native_seed_ids(source)
                or native_seed_collisions(new_seed_ids, exposed_seed_ids)
            )
        )
    ):
        raise ValueError(
            "FLH exposure must retain exactly its known D1 units and reserve fresh future streams"
        )
    projection = projection_type(source)
    evaluation = evaluation_config_type(projection)
    world_registry = build_observation_evidence_world_registry()
    world = next(
        w
        for w in world_registry.world_profiles
        if w.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{prefix}.evidence-profile",
        draft_id=f"{prefix}.draft",
        registry=world_registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=tuple(sorted(EvidenceRung, key=lambda rung: rung.value)) if control else (EvidenceRung.MEASUREMENT,),
    )
    native = native_declarations(source)
    imported = set() if retention is None else {p.segment_id for p in retention.prefixes}
    carrier_class: type[PredecessorBoundSourceQualificationExperiment] = (
        PredecessorBoundSourceQualificationExperiment
        if retention is None
        else ProspectiveRetainedSourceQualification
    )
    carrier = carrier_class(
        f"{prefix}.source-qualification",
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        native.units,
        tuple(s for s in native.segments if s.segment_id not in imported),
        tuple(
            replace(v, segment_ids=tuple(k for k in v.segment_ids if k not in imported))
            for v in native.views
        ),
        Q_RECEIVERS,
        (Q_CLOCK,),
        ObjectIdentity.from_record(source_capability.capability_key, source_capability),
        ObjectIdentity.from_record(projection_capability.capability_key, projection_capability),
        ObjectIdentity.from_record(evaluation_capability.capability_key, evaluation_capability),
        ObjectIdentity.from_record(source.spec_id, source),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(evaluation.config_id, evaluation),
        EVALUATOR_TASK_ID,
        EvidenceCeiling.MEASUREMENT,
        OutcomeAccess.EVALUATION_SEALED,
        retained_predecessors=source.retained_predecessors
        if retention is None
        else retention.prefixes,
    )
    if control is not None:
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.protocol import evaluation_protocol_steps
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.discovery import CONTROL_CAPABILITY, REVEAL_CAPABILITY

        if (
            type(source) is not FiniteResponseLawAssignedEvaluationConfig
            or type(projection) is not FiniteResponseLawAssignedEvaluationProjectionConfig
            or type(evaluation) is not FiniteResponseLawAssignedEvaluationCompletionConfig
        ):
            raise ValueError(
                "Independent evaluation composition requires exact source/projection/completion configurations"
            )
        manifests = {
            m.capability_key: m
            for m in (
                source_capability,
                projection_capability,
                evaluation_capability,
                CONTROL_CAPABILITY,
                REVEAL_CAPABILITY,
            )
        }
        if retention is not None:
            manifests[IMPORT_CAPABILITY.capability_key] = IMPORT_CAPABILITY
        composition_steps = evaluation_protocol_steps(
            source,
            carrier,
            projection,
            evaluation,
            control,
            FiniteResponseLawEvaluationRevealConfig(control),
            retention,
        )
        version_by_schema = {
            record.SCHEMA: record.VERSION
            for record in (
                source,
                projection,
                evaluation,
                control,
                FiniteResponseLawEvaluationRevealConfig(control),
                *((retention,) if retention is not None else ()),
            )
        }
        use_class = (
            QualifiedSourceUseExperiment if retention is None else ProspectiveRetainedSourceUse
        )
        carrier = use_class(
            **{f.name: getattr(carrier, f.name) for f in fields(PredecessorBoundSourceQualificationExperiment)},
            stages=tuple(
                QualifiedSourceUseStage(
                    s.step_id,
                    s.stage.value,
                    s.dependency_step_ids,
                    ObjectIdentity.from_record(s.capability_key, manifests[s.capability_key]),
                    ObjectIdentity(
                        s.config.config_id,
                        s.config.config_schema,
                        version_by_schema[s.config.config_schema],
                        s.config.content_sha256,
                    ),
                )
                for s in composition_steps
            ),
        )
    association = substrate_binding(
        source, carrier, ObjectIdentity.from_record(source_binding.binding_id, source_binding)
    )
    identified = sorted(
        (
            (source.spec_id, source),
            (projection.config_id, projection),
            (evaluation.config_id, evaluation),
            (carrier.extension_set_id, carrier),
            (association.binding_id, association),
            *(((retention.config_id, retention),) if retention is not None else ()),
            *(((control.config_id, control),) if control is not None else ()),
            *(
                (
                    (
                        FiniteResponseLawEvaluationRevealConfig(control).config_id,
                        FiniteResponseLawEvaluationRevealConfig(control),
                    ),
                )
                if control is not None
                else ()
            ),
        ),
        key=lambda v: v[1].SCHEMA,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(v for _, v in identified)
    bindings: tuple[ExecutableCapabilityBinding, ...] = (
        source_binding,
        projection_binding,
        evaluation_binding,
    )
    if control is not None:
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.executable_binding import CONTROL_BINDING, REVEAL_BINDING

        bindings = (*bindings, CONTROL_BINDING, REVEAL_BINDING)
    if retention is not None:
        bindings = (*bindings, IMPORT_BINDING)
    decoder_by_schema = {
        d.payload_schema: d for b in bindings for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[v.SCHEMA] for v in payloads)
    system = native_system(source)
    experiment = native_experiment(source, system)
    if control is not None:
        from .design import qualification_system, evaluation_experiment

        system = qualification_system(source)
        system = replace(
            system, world=replace(system.world, maximum_evidence=EvidenceCeiling.CONTROLLER_USE)
        )
        experiment = evaluation_experiment(source, system)
        experiment = replace(
            experiment,
            reveal_barrier=replace(
                experiment.reveal_barrier, development_unit_ids=exposed_unit_ids
            ),
        )
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=native_budget(source),
        objective="Test frozen pre-parent predictions for finite-native two-consumer use and added interface information on all 64 fresh assigned roots."
        if control
        else "Complete the frozen finite native measurement census; leave two-future opportunity and predictive usefulness to their separate gates.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    qualification = MaterializationQualificationReceipt(
        f"{prefix}.config-qualification",
        source.spec_id,
        ObjectIdentity.from_record(source.spec_id, source),
        source.fingerprint(),
        system.world.world_id,
        ObjectIdentity.from_record(projection.config_id, projection),
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({q.native_unit for q in system.quantities})),
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
        qualification.materialization,
        source.fingerprint(),
        source.fingerprint(),
        qualification.observation_operator,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    qualifications, sources = [qualification], [source_ref]
    if retention is not None:
        # Imported preparations replace acquisition from the numerical-config
        # source. The unchanged config remains an issued executable payload.
        qualifications, sources = [], []
    if control is not None:
        for artifact in (control.qualification, control.calibration_native):
            identity = ObjectIdentity(
                artifact.artifact_id, artifact.payload_schema, "1.0.0", artifact.sha256
            )
            prior_qualification = replace(
                qualification,
                receipt_id=f"qualification.{artifact.artifact_id}",
                source_id=artifact.artifact_id,
                materialization=identity,
                content_sha256=artifact.sha256,
            )
            qualifications.append(prior_qualification)
            sources.append(
                replace(
                    source_ref,
                    source_id=artifact.artifact_id,
                    role=SourceMaterializationRole.CALIBRATION,
                    materialization=identity,
                    content_sha256=artifact.sha256,
                    qualification_receipt=ObjectIdentity.from_record(
                        prior_qualification.receipt_id, prior_qualification
                    ),
                )
            )
    for prior in carrier.retained_predecessors:
        for artifact in prior.artifacts:
            materialization = ObjectIdentity(
                artifact.artifact_id, artifact.payload_schema, "1.0.0", artifact.sha256
            )
            receipt = replace(
                qualification,
                receipt_id=f"qualification.{artifact.artifact_id}",
                source_id=artifact.artifact_id,
                materialization=materialization,
                content_sha256=artifact.sha256,
                outcome_access=prior.outcome_access,
                visibility_ceiling=prior.visibility_ceiling,
            )
            qualifications.append(receipt)
            sources.append(
                replace(
                    source_ref,
                    source_id=artifact.artifact_id,
                    role=SourceMaterializationRole.PREPARED_MEDIUM,
                    materialization=materialization,
                    content_sha256=artifact.sha256,
                    qualification_receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
                )
            )
    science = ObjectIdentity(
        f"{PROGRAMME}.frozen-plan",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        source.science.plan_sha256,
    )
    cutoff = InformationCutoff(
        f"{prefix}.design-cutoff", Q_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    design_inputs = tuple(
        sorted(
            (
                DesignInputRecord(
                    f"{prefix}.science",
                    science,
                    science.object_fingerprint,
                    cutoff,
                    DesignInputRole.MOTIVATION,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    "owner.empirical-lawhood",
                    exposed_unit_ids,
                    exposed_seed_ids,
                ),
                DesignInputRecord(
                    f"{prefix}.exposure",
                    exposure,
                    exposure.object_fingerprint,
                    cutoff,
                    DesignInputRole.READINESS_METADATA,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    "owner.empirical-lawhood",
                    () if control else exposed_unit_ids,
                    () if control else exposed_seed_ids,
                    (f"{prefix}.science",) if control else (),
                ),
            ),
            key=lambda d: d.input_id,
        )
    )
    if control is not None:
        models = control.primary_model_set
        if models.target_world_id != system.world.world_id:
            raise ValueError("Independent evaluation frozen primary model set targets another evidence world")
        design_inputs = tuple(
            sorted(
                (
                    *design_inputs,
                    DesignInputRecord(
                        f"{prefix}.primary-model-set",
                        ObjectIdentity.from_record(models.model_set_id, models),
                        models.fingerprint(),
                        cutoff,
                        DesignInputRole.READINESS_METADATA,
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        "owner.empirical-lawhood",
                        (),
                        (),
                        (f"{prefix}.science",),
                    ),
                ),
                key=lambda d: d.input_id,
            )
        )
    capabilities: tuple[CapabilityManifest, ...] = (
        evaluation_capability,
        projection_capability,
        source_capability,
    )
    if control is not None:
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.discovery import CONTROL_CAPABILITY, REVEAL_CAPABILITY

        capabilities = (*capabilities, CONTROL_CAPABILITY, REVEAL_CAPABILITY)
    if retention is not None:
        capabilities = (*capabilities, IMPORT_CAPABILITY)
    selected = {c.capability_key for c in capabilities}
    registry = CapabilityRegistry(
        f"{prefix}.registry",
        tuple(
            c
            for c in GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry.capabilities
            if c.capability_key in selected
        ),
    )
    template = _template(
        source, carrier, projection, evaluation, experiment, registry, control, retention
    )
    draft = StudyDraft(
        f"{prefix}.draft",
        StudyDraftLifecycle.DRAFT,
        campaign.objective,
        (f"{prefix}.alternative.complete", f"{prefix}.alternative.unavailable"),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(d.input_id for d in design_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        exposed_unit_ids if control else tuple(sorted(set(exposed_unit_ids).union(units))),
        tuple(sorted(units)) if control else (),
        exposed_seed_ids if control else tuple(sorted(set(exposed_seed_ids).union(new_seed_ids))),
        new_seed_ids if control else (),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in sorted(capabilities, key=lambda c: c.capability_key)
        ),
        tuple(sorted(sources, key=lambda s: s.source_id)),
        native_budget(source),
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
    base, standard = _entry(
        draft,
        expanded,
        science,
        prefix=prefix,
        minimum_units=len(units),
        operand_description="Frozen response-law {domain} information assessment; all 64 assigned roots and explicit paired receivers. controller admission/prospective controller evaluation are separate controller obligations."
        if control
        else "Finite native {domain} measurement availability; no fitted response law or population inference.",
        estimator="complete-root-use-and-response-loss" if control else "complete-native-census",
        uncertainty="one-sided-cp-and-paired-root-bootstrap"
        if control
        else "explicit-unavailable-operands",
        ceiling=EvidenceCeiling.LOCAL_LAW if control else EvidenceCeiling.MEASUREMENT,
        evaluator=REVEAL_CAPABILITY if control else evaluation_capability,
        input_schema=FiniteResponseLawEvaluationCohort.SCHEMA if control else view_type.SCHEMA,
        output_schema=FiniteResponseLawEvaluationCohort.SCHEMA if control else evaluation_type.SCHEMA,
        evidence_unit_ids=units,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    proposed = tuple(
        ProposedStudyExtension(
            f"{prefix}.extension.{i:02d}",
            f"{prefix}.namespace.{i:02d}",
            ObjectIdentity.from_record(record_id, record),
            len(record.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, ((record_id, record), decoder) in enumerate(zip(identified, decoders, strict=True))
    )
    extensions = ProposedStudyExtensionSet(
        f"{prefix}.extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(p.namespace_id for p in proposed),
        proposed,
    )
    return ResponseGeometryAssayAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.executable-study-definition", base, extensions),
        replace(standard, base=context),
        CandidateCapabilityCatalog(
            f"{prefix}.catalog",
            tuple(
                r
                for r in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if r.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )
