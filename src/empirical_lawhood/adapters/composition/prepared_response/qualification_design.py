"Outcome-blind prepared source qualification authoring, using the public parameterised route."
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.prepared_response.executable_binding import SOURCE_QUALIFICATION_EVALUATION_BINDING, SOURCE_QUALIFICATION_PROJECTION_BINDING
from empirical_lawhood.adapters.methods.prepared_response.extension_bundle import SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.prepared_response.qualification import PreparedResponseSourceQualificationEvaluationConfig, PreparedResponseSourceQualificationEvaluation
from empirical_lawhood.adapters.methods.prepared_response.qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import SOURCE_QUALIFICATION_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import SOURCE_QUALIFICATION_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.qualification_roster import SOURCE_QUALIFICATION_CLOCK, SOURCE_QUALIFICATION_FRAME, SOURCE_QUALIFICATION_RECEIVERS, prepared_response_source_qualification_native_declarations, prepared_response_source_qualification_substrate
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    AssignmentKind,
    AssignmentSpec,
    ControlKind,
    ControlSpec,
    ExperimentSpec,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    IndependentUnitSpec,
    PortDirection,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    InformationCutoff,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate, StandardCandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

from .exposure import PreparedExposureInspection

PREFIX = "prepared-response.source-qualification"
UNIT = f"{PREFIX}.independent-stochastic-root"
# Provisional authoring maxima. The separate measured full-pipeline resource
# envelope must admit the 19.2-hour central budget before any scientific issue.
BUDGET = ResourceBudget(8, 24 * 1024**3, 0, 86_400, 256 * 1024**3, 64 * 1024**3)
_CAPABILITIES = (SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, SOURCE_QUALIFICATION_SOURCE_CAPABILITY)
_BINDINGS = (SOURCE_QUALIFICATION_EVALUATION_BINDING, SOURCE_QUALIFICATION_PROJECTION_BINDING, SOURCE_QUALIFICATION_SOURCE_BINDING)


def prepared_response_source_qualification_system(
    source: PreparedNativeSpec, *, prefix: str = PREFIX
) -> SystemSpec:
    unit = f"{prefix}.independent-stochastic-root"
    pre = AvailabilitySpec(
        SOURCE_QUALIFICATION_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    observed = replace(
        pre, phase=CausalPhase.RECEIVER, outcome_access=OutcomeAccess.EVALUATION_SEALED
    )
    quantities = tuple(
        QuantitySpec(
            f"{prefix}.quantity.{key}",
            label,
            kind,
            dimension,
            unit,
            SOURCE_QUALIFICATION_FRAME,
            SOURCE_QUALIFICATION_CLOCK,
            available,
            direction,
        )
        for key, label, kind, dimension, unit, available, direction in (
            (
                "two-port-action",
                "Two frozen HS force components and parent coupling schedules",
                QuantityKind.ACTION,
                "native-action",
                "native-couplings-and-force",
                replace(pre, phase=CausalPhase.ACTION_REQUESTED),
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "denominator",
                "Six-matrix q=2 BAOAB medium with paired numerical views",
                QuantityKind.DENOMINATOR,
                "medium-identity",
                "identity",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "history",
                "Fresh causal native history and pre-parent frozen two-port frame",
                QuantityKind.HISTORY,
                "native-history",
                "native-hs-phase-history",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "receiver.m1",
                "Absolute handoff-relative first receiver displacement",
                QuantityKind.RECEIVER,
                "displacement",
                "hilbert-schmidt-native",
                observed,
                ResponseDirection.SIGNED_VECTOR,
            ),
            (
                "receiver.m2",
                "Absolute handoff-relative second receiver displacement",
                QuantityKind.RECEIVER,
                "displacement",
                "hilbert-schmidt-native",
                observed,
                ResponseDirection.SIGNED_VECTOR,
            ),
        )
    )
    relation = RelationalIdentity(
        f"{prefix}.relation",
        (quantities[1].quantity_id,),
        (quantities[2].quantity_id,),
        False,
        (quantities[0].quantity_id,),
        tuple(q.quantity_id for q in quantities[3:]),
        HorizonSpec(f"{prefix}.horizon", SOURCE_QUALIFICATION_CLOCK, Decimal(320), "reference-tick"),
    )
    world = WorldSpec(
        f"{prefix}.world",
        "Prepared six-matrix numerical medium",
        WorldKind.NUMERICAL_SIMULATOR,
        (
            "BAOAB-unit-kinetic-mass",
            "q2-six-Hermitian-matrices",
            "two-frozen-preparent-HS-ports",
        ),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.RESPONSE,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{prefix}.computability",
        ("finite-32-root-source qualification",),
        ("complete-pipeline-canary-pending",),
        ("bounded-native-phase-checkpoint",),
        ("bounded-native-phase-checkpoint",),
        8,
        24 * 1024**3,
        0,
        86_400,
        64 * 1024**3,
        Decimal(69_120),
        None,
    )
    views = tuple(
        NumericalViewSpec(
            view.view_id,
            world.world_id,
            unit,
            f"{CAMPAIGN}.six-matrix-baoab-equations",
            (f"{CAMPAIGN}.native-closure",),
            (f"{CAMPAIGN}.finite-assay-boundary",),
            (
                NumericalCoordinateSpec(
                    f"{prefix}.dt.r{index + 1}",
                    NumericalCoordinateKind.TIMESTEP,
                    Decimal("0.001") / (index + 1),
                    "dimensionless-langevin-time",
                    index,
                ),
            ),
            f"{CAMPAIGN}.solver.baoab",
            "1.0.0",
            "complex128",
            "cpu",
            f"{CAMPAIGN}.native-runtime",
            RandomnessSemantics.GENERATIVE_PREPARATION,
            f"{CAMPAIGN}.native-observer",
            compute.envelope_id,
        )
        for index, view in enumerate(source.numerical_views)
    )
    policy = AuthorityPolicy(
        f"{prefix}.authority-policy",
        "owner.prepared-response" if prefix == PREFIX else "owner.empirical-lawhood",
        "operator.prepared-response" if prefix == PREFIX else "operator.empirical-lawhood",
        (prefix,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset(
            (
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            )
        ),
        frozenset((SourceAccessClass.NONE,)),
        (f"{prefix}.exact-route-gate", f"{prefix}.typed-authority-gate"),
        frozenset(),
        BUDGET,
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{prefix}.system",
        "Excluded finite prepared-response chart qualification",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                SOURCE_QUALIFICATION_CLOCK,
                "One reference tick is 0.001 Langevin time",
                "reference-tick",
                SOURCE_QUALIFICATION_FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        tuple(sorted(quantities, key=lambda value: value.quantity_id)),
        IndependentUnitSpec(
            unit,
            "One fresh root; parents, words and views are nested",
            f"{prefix}.root-id",
        ),
        policy,
        tuple(
            sorted(
                (
                    PortSpec(
                        f"{prefix}.force-port",
                        quantities[0].quantity_id,
                        SOURCE_QUALIFICATION_CLOCK,
                        PortDirection.INPUT,
                        BalanceRole.NONE,
                        AuthorityAction.SIMULATION_EXECUTION,
                    ),
                    *(
                        PortSpec(
                            receiver,
                            quantity.quantity_id,
                            SOURCE_QUALIFICATION_CLOCK,
                            PortDirection.OUTPUT,
                            BalanceRole.OBSERVATION,
                        )
                        for receiver, quantity in zip(
                            SOURCE_QUALIFICATION_RECEIVERS, quantities[3:], strict=True
                        )
                    ),
                ),
                key=lambda value: value.port_id,
            )
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def _experiment(
    system: SystemSpec, exposure: PreparedExposureInspection, *, prefix: str = PREFIX
) -> ExperimentSpec:
    unit = f"{prefix}.independent-stochastic-root"
    receivers = system.relation.receiver_quantity_ids
    cutoff = InformationCutoff(
        f"{prefix}.design-cutoff", SOURCE_QUALIFICATION_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{prefix}.obligations",
        SupportSpec(
            f"{prefix}.support",
            system.relation.relation_id,
            unit,
            32,
            2,
            cutoff.cutoff_id,
            (),
            (f"{prefix}.denominator",),
            (),
            required,
        ),
        ValiditySpec(
            f"{prefix}.validity",
            (f"{prefix}.fixed-roster",),
            ("finite-complete-native-delivery", "simulator-local-denominator"),
            ("NUMERICAL_OR_OBSERVATION_FAILURE",),
            required,
        ),
        UncertaintySpec(
            f"{prefix}.uncertainty",
            f"{prefix}.fixed-root-descriptive-counts",
            unit,
            Decimal("0.95"),
            receivers,
            ("FINITE_EXCLUDED_PANEL_NOT_POPULATION_GUARANTEE",),
            required,
        ),
        (
            FalsifierSpec(
                f"{prefix}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                SOURCE_QUALIFICATION_EVALUATION_CAPABILITY.capability_key,
                "No common charter passes complete source, numerical, preservation, directional contact and fixed-parent comparisons in both contexts.",
                "Keep all assigned roots; no amplitude, threshold, mode or sample-size retuning.",
                required,
            ),
        ),
        ClosureSpec(
            f"{prefix}.closure",
            (f"{prefix}.five-parent-seventeen-word-chart",),
            (f"{prefix}.absolute-handoff-relative-two-receiver-response",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{prefix}.convergence",
            ("all-five-readouts-primary-half-epsilon-over-eight",),
            tuple(v.view_id for v in system.numerical_views),
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{prefix}.compute-evidence",
            system.computability_envelopes[0].envelope_id,
            tuple(v.view_id for v in system.numerical_views),
            ReadinessStatus.READY,
            (),
            (f"{prefix}.provisional-bounded-reservation",),
        ),
    )
    claim = ClaimSpec(
        f"{prefix}.finite-contact-claim",
        system.world.world_id,
        system.relation.relation_id,
        "The first passing frozen charter provides informative finite two-port task contacts in both declared contexts.",
        "Sixteen assigned independent roots per context; fixed-parent and fixed-command contact counts with paired numerical views.",
        unit,
        EvidenceRung.RESPONSE,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "source qualification nominates one chart/task charter; law support and controller efficacy require later independent stages.",
        ("frozen-q2-native-medium", "independent-root-only-inference"),
        numerical_view_ids=tuple(v.view_id for v in system.numerical_views),
    )
    return ExperimentSpec(
        prefix,
        system.system_id,
        system.world.world_id,
        system.relation,
        unit,
        (claim,),
        AssignmentSpec(
            f"{prefix}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            unit,
            system.relation.action_quantity_ids,
            "Five time-matched parents per root; seventeen words share each parent handoff and absolute-time common-response innovations. Primary ports are frozen before any parent.",
            (f"{prefix}.fixed-roster",),
        ),
        receivers,
        (
            ControlSpec(
                f"{prefix}.control",
                ControlKind.BASELINE_COMPARATOR,
                SOURCE_QUALIFICATION_EVALUATION_CAPABILITY.capability_key,
                receivers,
                "Measured HOLD, every fixed parent and every fixed command, all nine tasks, independent endpoint equation checker, primary/half views.",
            ),
        ),
        (
            PrecisionGoal(
                f"{prefix}.precision",
                f"{prefix}.qualified-assigned-root-fraction",
                Decimal(1),
                "proportion",
                32,
                "All sixteen assigned roots per context qualify the chart; fixed 12/8/4 contact counts and 0.20 oracle advantage, no top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{prefix}.reveal",
            exposure.excluded_unit_ids,
            f"{prefix}.excluded-cohort",
            sha256(canonical_json_bytes(exposure.proposed_unit_ids)).hexdigest(),
            (f"{prefix}.sealed-evaluation",),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.OUTCOME_VISIBLE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def _template(
    source: PreparedNativeSpec,
    projection: PreparedResponseSourceQualificationProjectionConfig,
    evaluation: PreparedResponseSourceQualificationEvaluationConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    prefix: str = PREFIX,
) -> StudyTemplate:
    definitions = (
        (
            "linked-source-materialization",
            ScientificStage.PREPARE,
            source,
            source.spec_id,
            SOURCE_QUALIFICATION_SOURCE_CAPABILITY,
            (),
            _outputs(
                (
                    ("native-result", PreparedNativeTaskResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            BarrierKind.NONE,
        ),
        (
            "linked-evidence-projection",
            ScientificStage.TRANSFORM,
            projection,
            projection.config_id,
            SOURCE_QUALIFICATION_PROJECTION_CAPABILITY,
            ("linked-source-materialization",),
            _outputs(
                (
                    ("view-report", PreparedResponseSourceQualificationViewObservation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                )
            ),
            BarrierKind.FREEZE,
        ),
        (
            f"{prefix}.evaluate",
            ScientificStage.EVALUATE,
            evaluation,
            evaluation.config_id,
            SOURCE_QUALIFICATION_EVALUATION_CAPABILITY,
            ("linked-evidence-projection",),
            _outputs(
                (
                    ("evaluation", PreparedResponseSourceQualificationEvaluation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
            BarrierKind.REVEAL,
        ),
    )
    steps = tuple(
        sorted(
            (
                ProtocolStepTemplate(
                    key,
                    stage,
                    manifest.capability_key,
                    manifest.capability_version,
                    _config_ref(
                        config, ObjectIdentity.from_record(config_id, config), manifest
                    ),
                    dependencies,
                    outputs,
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    manifest.resource_ceiling,
                    (),
                    barrier,
                    1,
                    (f"{prefix}.single-terminal",)
                    if stage is ScientificStage.EVALUATE
                    else (f"{prefix}.prototype.{stage.value.lower()}",),
                )
                for key, stage, config, config_id, manifest, dependencies, outputs, barrier in definitions
            ),
            key=lambda s: s.step_id,
        )
    )
    return _programme_template(
        ProtocolTemplate(f"{prefix}.protocol", "1.0.0", steps, False, False, True),
        source,
        source.spec_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id="linked-source-materialization",
        terminal_task_id=f"{prefix}.evaluate",
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.TRANSFORM: "view-report",
            ScientificStage.EVALUATE: "evaluation",
        },
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_prepared_response_source_qualification_authoring(
    *,
    design_packet_sha256: str,
    implementation_sha256: str,
    exposure: PreparedExposureInspection,
    prefix: str = PREFIX,
) -> PreparedResponseSourceQualificationAuthoringBundle:
    validate_stable_id(prefix, field_name="prepared response authoring prefix")
    source = exposure.source_config
    if source.stage != 'qualification' or not exposure.units_and_seeds_unexposed:
        raise ValueError(
            "prepared source qualification authoring requires its exact unexposed source inspection"
        )
    projection = PreparedResponseSourceQualificationProjectionConfig(f"{prefix}.projection-config", source)
    evaluation = PreparedResponseSourceQualificationEvaluationConfig(f"{prefix}.evaluation-config", projection)
    evidence_registry = build_observation_evidence_world_registry()
    world_profile = next(
        v
        for v in evidence_registry.world_profiles
        if v.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{prefix}.evidence-profile",
        draft_id=f"{prefix}.draft",
        registry=evidence_registry,
        world_profile_id=world_profile.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(EvidenceRung.MEASUREMENT, EvidenceRung.RESPONSE),
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
            f"prepared source qualification evidence profile is not compilable: {feasibility.disposition}"
        )
    native = prepared_response_source_qualification_native_declarations(source)
    carrier = FreshSourceQualificationExperiment(
        f"{prefix}.source-qualification",
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        native.units,
        native.segments,
        native.views,
        SOURCE_QUALIFICATION_RECEIVERS,
        (SOURCE_QUALIFICATION_CLOCK,),
        ObjectIdentity.from_record(
            SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_key, SOURCE_QUALIFICATION_SOURCE_CAPABILITY
        ),
        ObjectIdentity.from_record(
            SOURCE_QUALIFICATION_PROJECTION_CAPABILITY.capability_key, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY
        ),
        ObjectIdentity.from_record(
            SOURCE_QUALIFICATION_EVALUATION_CAPABILITY.capability_key, SOURCE_QUALIFICATION_EVALUATION_CAPABILITY
        ),
        ObjectIdentity.from_record(source.spec_id, source),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(evaluation.config_id, evaluation),
        f"{prefix}.evaluate",
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
    )
    association = prepared_response_source_qualification_substrate(
        source,
        carrier,
        ObjectIdentity.from_record(SOURCE_QUALIFICATION_SOURCE_BINDING.binding_id, SOURCE_QUALIFICATION_SOURCE_BINDING),
    )
    identified = sorted(
        (
            (source.spec_id, source),
            (projection.config_id, projection),
            (evaluation.config_id, evaluation),
            (carrier.extension_set_id, carrier),
            (association.binding_id, association),
        ),
        key=lambda value: value[1].SCHEMA,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(v for _, v in identified)
    decoder_by_schema = {
        d.payload_schema: d for b in _BINDINGS for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[v.SCHEMA] for v in payloads)
    system = prepared_response_source_qualification_system(source, prefix=prefix)
    experiment = _experiment(system, exposure, prefix=prefix)
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=BUDGET,
        objective="Qualify the frozen prepared two-port chart without fitted laws.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    source_identity = ObjectIdentity.from_record(source.spec_id, source)
    qualification = MaterializationQualificationReceipt(
        f"{prefix}.config-qualification",
        source.spec_id,
        source_identity,
        source.fingerprint(),
        system.world.world_id,
        ObjectIdentity.from_record(projection.config_id, projection),
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({v.native_unit for v in system.quantities})),
        (SOURCE_QUALIFICATION_FRAME,),
        (SOURCE_QUALIFICATION_CLOCK,),
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
        source_identity,
        source.fingerprint(),
        source.fingerprint(),
        qualification.observation_operator,
        tuple(v.view_id for v in system.numerical_views),
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    science = ObjectIdentity(
        f"{prefix}.final-design",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        design_packet_sha256,
    )
    cutoff = InformationCutoff(
        f"{prefix}.design-freeze", SOURCE_QUALIFICATION_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    design = DesignInputRecord(
        f"{prefix}.design-input",
        science,
        design_packet_sha256,
        cutoff,
        DesignInputRole.MOTIVATION,
        OutcomeAccess.EVALUATION_REVEALED,
        VisibilityCeiling.OUTCOME_VISIBLE,
        "owner.prepared-response" if prefix == PREFIX else "owner.empirical-lawhood",
    )
    implementation_plan = DesignInputRecord(
        f"{prefix}.implementation-plan-input",
        ObjectIdentity(
            f"{prefix}.implementation-plan",
            'empirical-lawhood/document/markdown',
            "1.0.0",
            source.implementation_plan_sha256,
        ),
        source.implementation_plan_sha256,
        cutoff,
        DesignInputRole.MOTIVATION,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.prepared-response" if prefix == PREFIX else "owner.empirical-lawhood",
        parent_input_ids=(design.input_id,),
    )
    inspection = DesignInputRecord(
        f"{prefix}.exposure-input",
        ObjectIdentity.from_record(exposure.inspection_id, exposure),
        exposure.fingerprint(),
        cutoff,
        DesignInputRole.READINESS_METADATA,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.prepared-response" if prefix == PREFIX else "owner.empirical-lawhood",
        exposure.excluded_unit_ids,
        exposure.excluded_seed_ids,
        tuple(sorted((design.input_id, implementation_plan.input_id))),
    )
    design_inputs = tuple(
        sorted((design, implementation_plan, inspection), key=lambda v: v.input_id)
    )
    selected = {v.capability_key for v in _CAPABILITIES}
    registry = CapabilityRegistry(
        f"{prefix}.registry",
        tuple(
            v
            for v in GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry.capabilities
            if v.capability_key in selected
        ),
    )
    template = _template(
        source, projection, evaluation, experiment, registry, prefix=prefix
    )
    draft = StudyDraft(
        f"{prefix}.draft",
        StudyDraftLifecycle.DRAFT,
        "Select the first jointly qualifying finite task charter in the frozen 48-entry order.",
        (f"{prefix}.alternative.finite-contact", f"{prefix}.alternative.unqualified"),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (inspection.input_id,),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        exposure.excluded_unit_ids,
        exposure.proposed_unit_ids,
        exposure.excluded_seed_ids,
        exposure.proposed_seed_ids,
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                v.capability_key, v.capability_version, v.implementation_sha256
            )
            for v in _CAPABILITIES
        ),
        (source_ref,),
        BUDGET,
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        (qualification,),
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
        minimum_units=32,
        operand_description="Prepared finite source qualification {domain} operand qualification in the fixed two-port chart.",
        estimator="fixed-charter-contact-screen",
        uncertainty="fixed-root-descriptive-counts",
        evaluator=SOURCE_QUALIFICATION_EVALUATION_CAPABILITY,
        input_schema=PreparedResponseSourceQualificationViewObservation.SCHEMA,
        output_schema=PreparedResponseSourceQualificationEvaluation.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
        ceiling=EvidenceCeiling.RESPONSE,
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
        f"{prefix}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(v.namespace_id for v in proposed),
        proposed,
    )
    return PreparedResponseSourceQualificationAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.authoring", base, extensions),
        replace(standard, base=context),
        CandidateCapabilityCatalog(
            f"{prefix}.catalog",
            tuple(
                v
                for v in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if v.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationCandidateContextProvider:
    bundle: PreparedResponseSourceQualificationAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("draft differs from its exact prepared source qualification bundle")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError(
                "standard authoring differs from its exact prepared source qualification bundle"
            )
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )
