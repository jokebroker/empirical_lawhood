"Outcome-blind causal response prediction authoring, using the public parameterised route."
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
from empirical_lawhood.adapters.methods.causal_response.executable_binding import EVALUATION_BINDING, PROJECTION_BINDING
from empirical_lawhood.adapters.methods.causal_response.extension_bundle import EVALUATION_CAPABILITY, PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.causal_response.records import CausalResponseCommittedPrediction, CausalResponseEvaluationConfig, CausalResponseEvaluation, CausalResponseProjectionConfig, CausalResponseViewObservation
from empirical_lawhood.adapters.simulators.causal_response.contracts import CausalResponseNativeConfig
from empirical_lawhood.adapters.simulators.causal_response.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.simulators.causal_response.extension_bundle import SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.causal_response.roster import REFERENCE_CLOCK, PREPARENT_FRAME, NATIVE_RECEIVERS, native_declarations, substrate_binding
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
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

from .exposure import CausalResponseExposureInspection

PREFIX = "causal-response-prediction"
UNIT = f"{PREFIX}.independent-stochastic-root"
# Provisional authoring maxima. The separate measured full-pipeline resource
# envelope must admit the 19.2-hour central budget before any scientific issue.
BUDGET = ResourceBudget(8, 24 * 1024**3, 0, 86_400, 256 * 1024**3, 64 * 1024**3)
_CAPABILITIES = (EVALUATION_CAPABILITY, PROJECTION_CAPABILITY, SOURCE_CAPABILITY)
_BINDINGS = (EVALUATION_BINDING, PROJECTION_BINDING, SOURCE_BINDING)


def causal_response_system(source: CausalResponseNativeConfig, *, prefix: str = PREFIX) -> SystemSpec:
    unit = f"{prefix}.independent-stochastic-root"
    pre = AvailabilitySpec(
        REFERENCE_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
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
            PREPARENT_FRAME,
            REFERENCE_CLOCK,
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
        HorizonSpec(f"{prefix}.horizon", REFERENCE_CLOCK, Decimal(320), "reference-tick"),
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
        ("finite-64-root-causal response prediction",),
        ("complete-no-effect-production-proof-required",),
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
        for index, view in enumerate(source.recipe.numerical_views)
    )
    policy = AuthorityPolicy(
        f"{prefix}.authority-policy",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
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
        "Prospective causal prediction of preparation-induced finite response",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                REFERENCE_CLOCK,
                "One reference tick is 0.001 Langevin time",
                "reference-tick",
                PREPARENT_FRAME,
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
                        REFERENCE_CLOCK,
                        PortDirection.INPUT,
                        BalanceRole.NONE,
                        AuthorityAction.SIMULATION_EXECUTION,
                    ),
                    *(
                        PortSpec(
                            receiver,
                            quantity.quantity_id,
                            REFERENCE_CLOCK,
                            PortDirection.OUTPUT,
                            BalanceRole.OBSERVATION,
                        )
                        for receiver, quantity in zip(
                            NATIVE_RECEIVERS, quantities[3:], strict=True
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
    system: SystemSpec, exposure: CausalResponseExposureInspection, *, prefix: str = PREFIX
) -> ExperimentSpec:
    unit = f"{prefix}.independent-stochastic-root"
    receivers = system.relation.receiver_quantity_ids
    cutoff = InformationCutoff(
        f"{prefix}.design-cutoff", REFERENCE_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{prefix}.obligations",
        SupportSpec(
            f"{prefix}.support",
            system.relation.relation_id,
            unit,
            64,
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
            f"{prefix}.context-separated-exact-root-sign-tests",
            unit,
            Decimal("0.95"),
            receivers,
            (
                "32_ROOTS_PER_CONTEXT_23_CONCORDANT_WINS_TWO_SIDED_0025_AND_OBSERVED_MSE_20_PERCENT",
            ),
            required,
        ),
        (
            FalsifierSpec(
                f"{prefix}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                EVALUATION_CAPABILITY.capability_key,
                "The context-parent-horizon mean defeats the frozen causal predictor, or the improvement remains unresolved, under the predeclared per-context rule.",
                "Keep all assigned roots; no amplitude, threshold, mode or sample-size retuning.",
                required,
            ),
        ),
        ClosureSpec(
            f"{prefix}.closure",
            (f"{prefix}.five-parent-four-axial-word-two-innovation-chart",),
            (f"{prefix}.amplitude-scaled-central-gain-preparation-contrast",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{prefix}.convergence",
            ("primary-192-320-contrast-own-view-discrepancy-eightfold-qualification",),
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
        f"{prefix}.causal-prediction-claim",
        system.world.world_id,
        system.relation.relation_id,
        "Causal handoff measurements improve prediction of preparation-induced response contrast beyond a context-parent-horizon mean; each context is adjudicated separately.",
        "Thirty-two assigned independent roots per context; paired future losses and numerical-view-concordant root wins.",
        unit,
        EvidenceRung.RESPONSE,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Finite action response predictive comparison only; no law qualification law, history necessity, pre-parent composition, controller or constitutive closure claim.",
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
            "Five time-matched parents per root; four signed axial words, two independent active future innovations, paired numerical views. Forecasts commit at handoff before future continuation; ports freeze before parents.",
            (f"{prefix}.fixed-roster",),
        ),
        receivers,
        (
            ControlSpec(
                f"{prefix}.control",
                ControlKind.BASELINE_COMPARATOR,
                EVALUATION_CAPABILITY.capability_key,
                receivers,
                "Frozen context-parent-horizon mean and causal predictor on identical fresh futures; parent HOLD defines the contrast; primary/half numerical views.",
            ),
        ),
        (
            PrecisionGoal(
                f"{prefix}.precision",
                f"{prefix}.concordant-root-predictive-improvement",
                Decimal(1),
                "proportion",
                64,
                "At least 23/32 concordant wins, observed MSE reduction >=20 percent in both views, and at least 24/32 resolved roots per context. No top-up.",
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
    source: CausalResponseNativeConfig,
    projection: CausalResponseProjectionConfig,
    evaluation: CausalResponseEvaluationConfig,
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
            SOURCE_CAPABILITY,
            (),
            _outputs(
                (
                    ("native-result", PreparedNativeTaskResult.SCHEMA),
                    ("committed-prediction", CausalResponseCommittedPrediction.SCHEMA),
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
            PROJECTION_CAPABILITY,
            ("linked-source-materialization",),
            _outputs(
                (
                    ("view-report", CausalResponseViewObservation.SCHEMA),
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
            EVALUATION_CAPABILITY,
            ("linked-evidence-projection",),
            _outputs(
                (
                    ("evaluation", CausalResponseEvaluation.SCHEMA),
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
class CausalResponseAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_causal_response_authoring(
    *,
    design_packet_sha256: str,
    implementation_sha256: str,
    exposure: CausalResponseExposureInspection,
    prefix: str = PREFIX,
) -> CausalResponseAuthoringBundle:
    validate_stable_id(prefix, field_name="prepared response authoring prefix")
    source = exposure.source_config
    if not exposure.units_and_seeds_unexposed:
        raise ValueError(
            "causal response prediction authoring requires its exact unexposed source inspection"
        )
    projection = CausalResponseProjectionConfig(f"{prefix}.projection-config", source)
    evaluation = CausalResponseEvaluationConfig(f"{prefix}.evaluation-config", projection)
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
            f"causal response prediction evidence profile is not compilable: {feasibility.disposition}"
        )
    native = native_declarations(source)
    carrier = FreshSourceQualificationExperiment(
        f"{prefix}.source-qualification",
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        native.units,
        native.segments,
        native.views,
        NATIVE_RECEIVERS,
        (REFERENCE_CLOCK,),
        ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
        ObjectIdentity.from_record(
            PROJECTION_CAPABILITY.capability_key, PROJECTION_CAPABILITY
        ),
        ObjectIdentity.from_record(
            EVALUATION_CAPABILITY.capability_key, EVALUATION_CAPABILITY
        ),
        ObjectIdentity.from_record(source.spec_id, source),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(evaluation.config_id, evaluation),
        f"{prefix}.evaluate",
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
    )
    association = substrate_binding(
        source,
        carrier,
        ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
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
    system = causal_response_system(source, prefix=prefix)
    experiment = _experiment(system, exposure, prefix=prefix)
    campaign = _campaign(
        system,
        experiment,
        prefix=prefix,
        budget=BUDGET,
        objective="Compare the frozen causal predictor with a competent finite mean on fresh paired response contrasts.",
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
        (PREPARENT_FRAME,),
        (REFERENCE_CLOCK,),
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
        "causal-response-prediction.exposed-source-qualification-motivation",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        design_packet_sha256,
    )
    cutoff = InformationCutoff(
        f"{prefix}.design-freeze", REFERENCE_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    design = DesignInputRecord(
        f"{prefix}.design-input",
        science,
        design_packet_sha256,
        cutoff,
        DesignInputRole.MOTIVATION,
        OutcomeAccess.EVALUATION_REVEALED,
        VisibilityCeiling.OUTCOME_VISIBLE,
        "owner.empirical-lawhood",
    )
    implementation_plan = DesignInputRecord(
        f"{prefix}.implementation-plan-input",
        ObjectIdentity(
            "causal-response-prediction.implementation-plan",
            'empirical-lawhood/document/markdown',
            "1.0.0",
            source.recipe.implementation_plan_sha256,
        ),
        source.recipe.implementation_plan_sha256,
        cutoff,
        DesignInputRole.MOTIVATION,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.empirical-lawhood",
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
        "owner.empirical-lawhood",
        exposure.excluded_unit_ids,
        exposure.excluded_seed_ids,
        tuple(sorted((design.input_id, implementation_plan.input_id))),
    )
    fitted = DesignInputRecord(
        f"{prefix}.frozen-model-bank-input",
        ObjectIdentity.from_record(source.model_bank.bank_id, source.model_bank),
        source.model_bank.fingerprint(),
        cutoff,
        DesignInputRole.DEVELOPMENT_TUNING,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        "owner.empirical-lawhood",
        tuple(
            sorted(
                r.physical_unit_id
                for c in source.model_bank.contexts
                for r in c.training_roots
            )
        ),
        (),
        (design.input_id,),
    )
    design_inputs = tuple(
        sorted(
            (design, implementation_plan, inspection, fitted), key=lambda v: v.input_id
        )
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
        "Test causal prediction against the context-parent-horizon mean separately in two contexts on fresh replicated futures.",
        (
            f"{prefix}.alternative.baseline-or-unresolved",
            f"{prefix}.alternative.causal-advantage",
        ),
        DesignOrigin(
            f"{prefix}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(sorted((inspection.input_id, fitted.input_id))),
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
        minimum_units=64,
        operand_description="Prospective causal response prediction {domain} operand qualification and preparation-contrast prediction in a fixed two-port chart.",
        estimator="frozen-causal-vs-context-parent-horizon-mean",
        uncertainty="context-separated-exact-root-sign-tests",
        evaluator=EVALUATION_CAPABILITY,
        input_schema=CausalResponseViewObservation.SCHEMA,
        output_schema=CausalResponseEvaluation.SCHEMA,
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
    return CausalResponseAuthoringBundle(
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
class CausalResponseCandidateContextProvider:
    bundle: CausalResponseAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("draft differs from its exact causal response prediction bundle")
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
            raise ValueError("standard authoring differs from its exact causal response prediction bundle")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )
