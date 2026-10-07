"Exact development declarations entering the shared native measurement through law qualification candidate route."

from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_scientific_inputs import ResponseGeometryScientificInputs

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.runtime.maintenance import ContinuationBoundInterruptionAuthority
from empirical_lawhood.planning.evidence_profiles import EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.native_source import NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeSourceProfile
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.prospective_config import ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.response_experiment import compile_response_experiment
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
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
from empirical_lawhood.adapters.methods.qualification_profiles import (
    QualificationProofOwner,
)
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_assessment_provider import response_geometry_development_assessment_references
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentCloseoutConfig, ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentDevelopmentResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_projection import DEVELOPMENT_DATA_SCHEMA, ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentViewReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_qualification import ResponseGeometryDevelopmentAffineQualificationProfile
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentMethodConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import DEVELOPMENT_VALIDATION_SCHEMA, ResponseGeometryDevelopmentQualificationConfig
from empirical_lawhood.adapters.methods.response_geometry_development.extension_bundle import DEVELOPMENT_CAPABILITIES, DEVELOPMENT_ROLES
from empirical_lawhood.adapters.methods.response_geometry_development.executable_binding import DEVELOPMENT_BINDINGS
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import CANONICAL_MEDIA_TYPE
from empirical_lawhood.adapters.simulators.response_geometry_prospective.executable_binding import _config_ref, _outputs
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_CLOCK, DEVELOPMENT_EPISODE_FRAME, DEVELOPMENT_FORCE_DIRECTION, DEVELOPMENT_FORCE_FRAME
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_roster import response_geometry_development_native_declarations
from empirical_lawhood.adapters.simulators.response_geometry_development.extension_bundle import DEVELOPMENT_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.response_geometry_development.executable_binding import DEVELOPMENT_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_HDF5_SCHEMA, DEVELOPMENT_SEED_SHA256, DEVELOPMENT_SOURCE_CONFIG_ID, ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeSegmentResult, ResponseGeometryDevelopmentNativeSegment, development_roots, development_segments

from .design import ResponseGeometryAssayAuthoringBundle, ResponseGeometryAssayCandidateContextProvider, _campaign, _entry, _study_template, _record_id
from .exposure import ResponsePanelExposureInspection
from .development_design import DEVELOPMENT_ACTION, DEVELOPMENT_BUDGET, DEVELOPMENT_PREFIX, DEVELOPMENT_RECEIVER, response_geometry_development_experiment, response_geometry_development_system


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAuthoringBundle(ResponseGeometryAssayAuthoringBundle):
    """The shared nonexecuting authoring bundle, restricted to development's native records."""


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentCandidateContextProvider(ResponseGeometryAssayCandidateContextProvider):
    bundle: ResponseGeometryDevelopmentAuthoringBundle


def _lineage() -> tuple[ProspectivePackageLineage, ProspectiveTerminalMatrix]:
    prefix, q = DEVELOPMENT_PREFIX, "response-geometry-assay"
    lineage = ProspectivePackageLineage(
        f"{prefix}.lineage",
        q,
        f"{prefix}.qualified-assay-branch",
        tuple(
            sorted(
                (
                    ProspectivePackageLineageNode(
                        q,
                        "Completed excluded assay qualification",
                        ProspectivePackageKind.EXCLUDED_QUALIFICATION,
                        (),
                        ("response-geometry-assay.exact-issued-route",),
                        EvidenceCeiling.RESPONSE,
                        None,
                        False,
                        False,
                        False,
                        False,
                    ),
                    ProspectivePackageLineageNode(
                        prefix,
                        "Fresh development development under assay's predeclared selector",
                        ProspectivePackageKind.PROSPECTIVE_CHILD,
                        (q,),
                        (
                            f"{prefix}.authentic-assay-selected",
                            f"{prefix}.exact-route-gate",
                        ),
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
        (f"{prefix}.assay-not-qualified-stop",),
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
                "Authentic assay selected short-pulse-response and both context origins",
                "Enter fixed development only after exact production readiness and typed authority",
            ),
            ProspectiveSelectorTransition(
                2, "assay or exact readiness is absent", "STOP without native development contact"
            ),
        ),
        (
            ProspectiveTerminalRule(
                1,
                "Missing authentic native/method operands",
                "UNEVALUABLE",
                "Authenticated available development evidence",
                "Stop the affected consuming claim; preserve all outcomes",
            ),
            ProspectiveTerminalRule(
                2,
                "Complete operands and sole local-law adjudication",
                "RETAIN_SOLE_OWNER_AND_SEPARATE_CONTRAST_ELIGIBILITY",
                "Local supported, mixed or unsupported development objects",
                "REP and ACQ each need their own finite strata; ACT remains stopped pending separate state/window feasibility and authority",
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


def _prototype(
    configs: tuple[CanonicalRecord, ...],
    experiment: object,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    from empirical_lawhood.kernel.experiments import ExperimentSpec

    assert isinstance(experiment, ExperimentSpec)
    source, projection, method, qualification, closeout = configs
    definitions = (
        (
            "linked-source-materialization",
            ScientificStage.PREPARE,
            DEVELOPMENT_SOURCE_CAPABILITY,
            source,
            (("native-result", ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA),),
            ("native-observations", DEVELOPMENT_HDF5_SCHEMA),
        ),
        (
            "linked-evidence-projection",
            ScientificStage.TRANSFORM,
            DEVELOPMENT_CAPABILITIES[0],
            projection,
            (("report", ResponseGeometryDevelopmentViewReport.SCHEMA),),
            ("data", DEVELOPMENT_DATA_SCHEMA),
        ),
        (
            f"{DEVELOPMENT_PREFIX}.prototype.develop",
            ScientificStage.DEVELOP,
            DEVELOPMENT_CAPABILITIES[1],
            method,
            (("report", ResponseGeometryDevelopmentFitResult.SCHEMA),),
            ("data", DEVELOPMENT_FIT_SCHEMA),
        ),
        (
            f"{DEVELOPMENT_PREFIX}.prototype.assess",
            ScientificStage.QUALIFY,
            DEVELOPMENT_CAPABILITIES[2],
            qualification,
            (
                ("report", ResponseGeometryDevelopmentContextResult.SCHEMA),
                ("geometry", ResponseGeometryDevelopmentGeometryReport.SCHEMA),
            ),
            ("data", DEVELOPMENT_VALIDATION_SCHEMA),
        ),
        (
            f"{DEVELOPMENT_PREFIX}.evaluate",
            ScientificStage.EVALUATE,
            DEVELOPMENT_CAPABILITIES[3],
            closeout,
            (
                ("report", ResponseGeometryDevelopmentDevelopmentResult.SCHEMA),
                ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
            ),
            None,
        ),
    )
    steps: list[ProtocolStepTemplate] = []
    for task, stage, manifest, config, reports, data in definitions:
        config_id = _record_id(config)
        steps.append(
            ProtocolStepTemplate(
                task,
                stage,
                manifest.capability_key,
                manifest.capability_version,
                _config_ref(
                    config, ObjectIdentity.from_record(config_id, config), manifest
                ),
                () if not steps else (steps[-1].step_id,),
                _outputs(
                    (
                        *reports,
                        (
                            "stage-envelope"
                            if stage is ScientificStage.PREPARE
                            else "stage",
                            LinkedCampaignStageEnvelope.SCHEMA,
                        ),
                    ),
                    data,
                ),
                manifest.permissions,
                OutcomeAccess.EVALUATION_REVEALED
                if stage is ScientificStage.EVALUATE
                else OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.PROSPECTIVE,
                manifest.resource_ceiling,
                (),
                BarrierKind.REVEAL
                if stage is ScientificStage.EVALUATE
                else BarrierKind.NONE
                if stage is ScientificStage.PREPARE
                else BarrierKind.FREEZE,
                1,
                (f"{DEVELOPMENT_PREFIX}.single-terminal",)
                if stage is ScientificStage.EVALUATE
                else (f"{DEVELOPMENT_PREFIX}.prototype.{stage.value.lower()}",),
            )
        )
    return _study_template(
        ProtocolTemplate(
            f"{DEVELOPMENT_PREFIX}.protocol",
            "1.0.0",
            tuple(sorted(steps, key=lambda s: s.step_id)),
            False,
            False,
            True,
        ),
        source,
        _record_id(source),
        experiment,
        registry,
        prefix=DEVELOPMENT_PREFIX,
        source_task_id="linked-source-materialization",
        terminal_task_id=f"{DEVELOPMENT_PREFIX}.evaluate",
        primary_output_ids={
            stage: "native-result" if stage is ScientificStage.PREPARE else "report"
            for _, stage, *_ in definitions
        },
    )


def build_response_geometry_development_authoring(
    *,
    assay_evaluation: ObjectIdentity,
    design_packet_sha256: str,
    scientific_inputs: ResponseGeometryScientificInputs,
    implementation_sha256: str,
    prior_assay_exposure: ResponsePanelExposureInspection,
    maintenance_predecessor: ContinuationBoundInterruptionAuthority | None = None,
    storage_continuation_amendment: ObjectIdentity | None = None,
) -> ResponseGeometryDevelopmentAuthoringBundle:
    """No reads, writes or source contact; the operator must authenticate assay separately."""
    prefix = DEVELOPMENT_PREFIX
    if storage_continuation_amendment is not None and (
        maintenance_predecessor is None
        or maintenance_predecessor.continuation_run_id != f"{prefix}-maintenance-continuation"
        or storage_continuation_amendment.object_id != f"{prefix}.storage-continuation-amendment"
        or storage_continuation_amendment.object_schema
        != 'empirical-lawhood/document/markdown'
        or storage_continuation_amendment.object_version != "1.0.0"
    ):
        raise ValueError(
            "development storage continuation requires its exact amendment and predecessors"
        )
    prior_units = tuple(
        sorted(
            set(prior_assay_exposure.excluded_unit_ids)
            | set(prior_assay_exposure.proposed_unit_ids)
        )
    )
    prior_seeds = tuple(
        sorted(
            set(prior_assay_exposure.excluded_seed_ids)
            | set(prior_assay_exposure.proposed_seed_ids)
        )
    )
    source = ResponseGeometryDevelopmentNativeConfig(
        DEVELOPMENT_SOURCE_CONFIG_ID,
        design_packet_sha256,
        DEVELOPMENT_SEED_SHA256,
        development_roots(),
        scientific_inputs,
        assay_evaluation,
    )
    source_id = ObjectIdentity.from_record(source.config_id, source)
    projection = ResponseGeometryDevelopmentProjectionConfig(f"{prefix}.projection-config", source_id)
    projection_id = ObjectIdentity.from_record(projection.config_id, projection)
    method = ResponseGeometryDevelopmentMethodConfig(
        f"{prefix}.method-config", projection_id, design_packet_sha256
    )
    system = response_geometry_development_system()
    qualification = ResponseGeometryDevelopmentQualificationConfig(
        f"{prefix}.qualification-config", system, source, projection, method
    )
    closeout = ResponseGeometryDevelopmentCloseoutConfig(f"{prefix}.closeout-config", qualification)
    assessment_manifest = DEVELOPMENT_CAPABILITIES[DEVELOPMENT_ROLES.index("assessment")]
    clock, _ = response_geometry_development_assessment_references(
        qualification,
        assessment_manifest,
        ArtifactIdentity(
            f"config-artifact.{qualification.config_id}",
            "development-authenticated-input",
            qualification.SCHEMA,
            qualification.fingerprint(),
            CANONICAL_MEDIA_TYPE,
            len(qualification.canonical_bytes()),
        ),
    )
    native = response_geometry_development_native_declarations(source, clock)
    profile = ResponseGeometryDevelopmentAffineQualificationProfile(
        QualificationProofOwner(
            f"{prefix}.finite-law-qualification-owner",
            assessment_manifest.capability_key,
            assessment_manifest.capability_version,
            assessment_manifest.implementation_sha256,
        )
    ).profile
    experiment = response_geometry_development_experiment(system, source, prior_units)
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=DEVELOPMENT_BUDGET,
        objective="Complete the fixed development local-law, representation and offline acquisition development panel; preserve each separate eligibility and stopping result.",
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
    if (
        feasibility.disposition
        is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION
    ):
        raise ValueError(
            f"development evidence profile is not compilable: {feasibility.disposition}"
        )
    source_profile = NativeSourceProfile(
        f"{prefix}.native-source-profile",
        CapabilitySelection(
            DEVELOPMENT_SOURCE_CAPABILITY.capability_key,
            DEVELOPMENT_SOURCE_CAPABILITY.capability_version,
            DEVELOPMENT_SOURCE_CAPABILITY.implementation_sha256,
        ),
        source_id,
        DEVELOPMENT_HDF5_SCHEMA,
        tuple(u.physical_independent_unit_id for u in native.units),
        len(development_segments()),
        sum(s.native_updates for s in development_segments()),
        DEVELOPMENT_BUDGET,
        assay_evaluation,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    evidence_id = ObjectIdentity.from_record(evidence.selection_id, evidence)
    native_id = ObjectIdentity.from_record(source_profile.profile_id, source_profile)
    campaign_id = ObjectIdentity.from_record(campaign.campaign_id, campaign)
    lineage, terminal = _lineage()
    qualification_config = NativeLawQualificationConfig(
        f"{prefix}.law-qualification-config",
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
        ObjectIdentity.from_record(
            assessment_manifest.capability_key, assessment_manifest
        ),
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
        qualification_config,
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
        "response-geometry.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        DEVELOPMENT_SOURCE_CAPABILITY.capability_key,
        DEVELOPMENT_SOURCE_CAPABILITY.capability_version,
        DEVELOPMENT_SOURCE_CAPABILITY.implementation_sha256,
        ObjectIdentity.from_record(DEVELOPMENT_SOURCE_BINDING.binding_id, DEVELOPMENT_SOURCE_BINDING),
        NativeSourceProfile.SCHEMA,
        ResponseGeometryDevelopmentNativeSegment.SCHEMA,
        None,
        ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA,
        ResponseGeometryDevelopmentViewReport.SCHEMA,
        source_id,
        (
            NativeReceiverContract(
                f"{prefix}.receiver",
                DEVELOPMENT_RECEIVER,
                "hilbert-schmidt-native",
                DEVELOPMENT_FORCE_FRAME,
                DEVELOPMENT_FORCE_DIRECTION,
                (DEVELOPMENT_CLOCK,),
            ),
        ),
        (NativeClockContract(DEVELOPMENT_CLOCK, "reference-tick", DEVELOPMENT_EPISODE_FRAME),),
        NativeActionContract(DEVELOPMENT_ACTION, True, True, True, True, "HOLD"),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    configs = (source, projection, method, qualification, closeout)
    payloads = tuple(
        sorted(
            (*configs, source_profile, qualification_config, extension, substrate), key=lambda r: r.SCHEMA
        )
    )
    decoder_by_schema = {
        d.payload_schema: d
        for b in (
            DEVELOPMENT_SOURCE_BINDING,
            *DEVELOPMENT_BINDINGS,
            NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
            PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
        )
        for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[r.SCHEMA] for r in payloads)
    config_qualification = MaterializationQualificationReceipt(
        f"{prefix}.config-qualification",
        source.config_id,
        source_id,
        source.fingerprint(),
        system.world.world_id,
        projection_id,
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({q.native_unit for q in system.quantities})),
        tuple(sorted({q.coordinate_frame for q in system.quantities})),
        (DEVELOPMENT_CLOCK,),
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
        ObjectIdentity.from_record(
            config_qualification.receipt_id, config_qualification
        ),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    science = ObjectIdentity(
        f"{prefix}.design",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        design_packet_sha256,
    )
    cutoff = InformationCutoff(
        f"{prefix}.design-freeze", DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    design_inputs: tuple[DesignInputRecord, ...] = (
        DesignInputRecord(
            f"{prefix}.design-input",
            science,
            design_packet_sha256,
            cutoff,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.response-geometry",
        ),
        DesignInputRecord(
            f"{prefix}.assay-prerequisite",
            assay_evaluation,
            assay_evaluation.object_fingerprint,
            cutoff,
            DesignInputRole.DEVELOPMENT_TUNING,
            OutcomeAccess.EVALUATION_REVEALED,
            VisibilityCeiling.OUTCOME_VISIBLE,
            "owner.response-geometry",
            physical_unit_ids=prior_assay_exposure.proposed_unit_ids,
            seed_ids=prior_assay_exposure.proposed_seed_ids,
            parent_input_ids=(f"{prefix}.design-input",),
        ),
        DesignInputRecord(
            f"{prefix}.assay-prior-exposure",
            ObjectIdentity.from_record(
                prior_assay_exposure.inspection_id, prior_assay_exposure
            ),
            prior_assay_exposure.fingerprint(),
            cutoff,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.response-geometry",
            prior_units,
            prior_seeds,
            (f"{prefix}.design-input",),
        ),
    )
    if maintenance_predecessor is not None:
        if maintenance_predecessor.service_stop.run_id != DEVELOPMENT_PREFIX:
            raise ValueError("development maintenance predecessor belongs to another experiment")
        design_inputs = tuple(
            sorted(
                (
                    *design_inputs,
                    DesignInputRecord(
                        f"{prefix}.maintenance-interruption",
                        ObjectIdentity.from_record(
                            maintenance_predecessor.service_stop.stop_id,
                            maintenance_predecessor,
                        ),
                        maintenance_predecessor.fingerprint(),
                        cutoff,
                        DesignInputRole.READINESS_METADATA,
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        maintenance_predecessor.service_stop.owner_id,
                        parent_input_ids=(f"{prefix}.design-input",),
                    ),
                ),
                key=lambda value: value.input_id,
            )
        )
    if storage_continuation_amendment is not None:
        design_inputs = tuple(
            sorted(
                (
                    *design_inputs,
                    DesignInputRecord(
                        f"{prefix}.storage-continuation",
                        storage_continuation_amendment,
                        storage_continuation_amendment.object_fingerprint,
                        cutoff,
                        DesignInputRole.READINESS_METADATA,
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        "owner.empirical-lawhood",
                        parent_input_ids=(
                            f"{prefix}.design-input",
                            f"{prefix}.maintenance-interruption",
                        ),
                    ),
                ),
                key=lambda value: value.input_id,
            )
        )
    capabilities = tuple(
        sorted((DEVELOPMENT_SOURCE_CAPABILITY, *DEVELOPMENT_CAPABILITIES), key=lambda c: c.capability_key)
    )
    selected = {c.capability_key for c in capabilities}
    registry = CapabilityRegistry(
        f"{prefix}.registry", tuple(sorted(capabilities, key=lambda c: c.registry_id))
    )
    template = _prototype(configs, experiment, registry)
    draft = StudyDraft(
        f"{prefix}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{prefix}.alternative.local-law", f"{prefix}.alternative.unqualified"),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(
                sorted(
                    (
                        f"{prefix}.assay-prerequisite",
                        f"{prefix}.assay-prior-exposure",
                        *(
                            (f"{prefix}.maintenance-interruption",)
                            if maintenance_predecessor
                            else ()
                        ),
                        *(
                            (f"{prefix}.storage-continuation",)
                            if storage_continuation_amendment
                            else ()
                        ),
                    )
                )
            ),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        experiment.reveal_barrier.development_unit_ids,
        qualification_config.evaluation_unit_ids,
        tuple(
            sorted(
                set(prior_seeds) | {f"{u}.native-rng" for u in qualification_config.development_unit_ids}
            )
        ),
        tuple(f"{u}.native-rng" for u in qualification_config.evaluation_unit_ids),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                c.capability_key, c.capability_version, c.implementation_sha256
            )
            for c in capabilities
        ),
        (source_ref,),
        DEVELOPMENT_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        (config_qualification,),
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
        minimum_units=64,
        operand_description="Finite development {domain} response-law operands and declared comparisons on fresh held roots; no general theorem or physical transport claim.",
        estimator="frozen-development-development-estimator",
        uncertainty="rootwise-model-conditional-calibration",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=DEVELOPMENT_CAPABILITIES[DEVELOPMENT_ROLES.index("evaluation")],
        input_schema=ResponseGeometryDevelopmentContextResult.SCHEMA,
        output_schema=ResponseGeometryDevelopmentDevelopmentResult.SCHEMA,
    )
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
    return ResponseGeometryDevelopmentAuthoringBundle(
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
