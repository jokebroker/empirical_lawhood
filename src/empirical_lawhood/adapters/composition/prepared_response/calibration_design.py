"Outcome-blind fresh calibration authoring through the public finite calibration route."
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
from empirical_lawhood.adapters.methods.prepared_response.calibration_provider import PreparedResponseCalibrationCalibrationTask
from empirical_lawhood.adapters.methods.prepared_response.calibration_records import PreparedResponseCalibrationCalibratedLibrary, PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig, PreparedResponseCalibrationViewProjection
from empirical_lawhood.adapters.methods.prepared_response.development_selection import PreparedResponseDevelopmentNominalLibrary
from empirical_lawhood.adapters.methods.prepared_response.executable_binding import CALIBRATION_CALIBRATION_BINDING, CALIBRATION_POLICY_BINDING, CALIBRATION_PROJECTION_BINDING
from empirical_lawhood.adapters.methods.prepared_response.extension_bundle import CALIBRATION_CALIBRATION_CAPABILITY, CALIBRATION_POLICY_CAPABILITY, CALIBRATION_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.prepared_response.policy_decision import PreparedParentDecision, PreparedPolicyDecisionConfig
from empirical_lawhood.adapters.methods.prepared_response.statistics import PreparedStatisticalSpec
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import CALIBRATION_SOURCE_BINDING, PreparedResponseCalibrationSourceFactory
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import CALIBRATION_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.policy_native import PreparedResponseCalibrationNativeTaskResult, prepared_response_calibration_native_invocations
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
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
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
from empirical_lawhood.planning.native_source import NativeSourceProfile
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate, StandardCandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope, rebind_study_template_to_expanded_protocol
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

from .exposure import PreparedExposureInspection


PREFIX = "prepared-response.fresh-response-calibration"
CALIBRATION_CLOCK = f"{PREFIX}.episode-clock"
CALIBRATION_CLOCK_FRAME = f"{PREFIX}.handoff-relative-episode"
CALIBRATION_ACTION_FRAME = f"{PREFIX}.frozen-preparent-two-port-frame"
UNIT = f"{PREFIX}.independent-stochastic-root"
RECEIVERS = (f"{PREFIX}.receiver.m1", f"{PREFIX}.receiver.m2")
CALIBRATION_BUDGET = ResourceBudget(
    8,
    24 * 1024**3,
    0,
    8 * 24 * 60 * 60,
    2 * 1024**4,
    256 * 1024**3,
)
_CAPABILITIES = (
    CALIBRATION_CALIBRATION_CAPABILITY,
    CALIBRATION_POLICY_CAPABILITY,
    CALIBRATION_PROJECTION_CAPABILITY,
    CALIBRATION_SOURCE_CAPABILITY,
)
_BINDINGS = (
    CALIBRATION_CALIBRATION_BINDING,
    CALIBRATION_POLICY_BINDING,
    CALIBRATION_PROJECTION_BINDING,
    CALIBRATION_SOURCE_BINDING,
)


def prepared_response_calibration_system(source: PreparedNativeSpec) -> SystemSpec:
    if source.stage != 'calibration':
        raise ValueError("prepared fresh calibration system requires the fresh calibration native denominator")
    pre = AvailabilitySpec(CALIBRATION_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0))
    observed = replace(pre, phase=CausalPhase.RECEIVER, outcome_access=OutcomeAccess.EVALUATION_SEALED)
    quantities = tuple(
        QuantitySpec(
            f"{PREFIX}.quantity.{key}",
            label,
            kind,
            dimension,
            unit,
            CALIBRATION_ACTION_FRAME,
            CALIBRATION_CLOCK,
            availability,
            direction,
        )
        for key, label, kind, dimension, unit, availability, direction in (
            (
                "two-port-action",
                "dependent refinement-frozen policy parent and complete nine-word audit menu",
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
                "Target-blind pre-parent and handoff instrument histories",
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
        f"{PREFIX}.relation",
        (quantities[1].quantity_id,),
        (quantities[2].quantity_id,),
        False,
        (quantities[0].quantity_id,),
        tuple(value.quantity_id for value in quantities[3:]),
        HorizonSpec(f"{PREFIX}.horizon", CALIBRATION_CLOCK, Decimal(320), "reference-tick"),
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Prepared six-matrix numerical medium",
        WorldKind.NUMERICAL_SIMULATOR,
        (
            "BAOAB-unit-kinetic-mass",
            "q2-six-Hermitian-matrices",
            "two-frozen-preparent-HS-ports",
        ),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.LOCAL_LAW,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{PREFIX}.computability",
        ("finite-192-root-fresh-response-calibration",),
        ("complete-pipeline-canary-pending",),
        ("bounded-native-phase-checkpoint",),
        ("bounded-native-phase-checkpoint",),
        8,
        24 * 1024**3,
        0,
        86_400,
        256 * 1024**3,
        Decimal(69_120),
        None,
    )
    views = tuple(
        NumericalViewSpec(
            view.view_id,
            world.world_id,
            UNIT,
            f"{CAMPAIGN}.six-matrix-baoab-equations",
            (f"{CAMPAIGN}.native-closure",),
            (f"{CAMPAIGN}.finite-assay-boundary",),
            (
                NumericalCoordinateSpec(
                    f"{PREFIX}.dt.r{index + 1}",
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
    authority = AuthorityPolicy(
        f"{PREFIX}.authority-policy",
        "owner.prepared-response",
        "operator.prepared-response",
        (PREFIX,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset(
            (
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            )
        ),
        frozenset((SourceAccessClass.NONE,)),
        (f"{PREFIX}.exact-route-gate", f"{PREFIX}.typed-authority-gate"),
        frozenset(),
        CALIBRATION_BUDGET,
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Fresh simultaneous calibration of dependent refinement-frozen prepared response",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                CALIBRATION_CLOCK,
                "One reference tick is 0.001 Langevin time",
                "reference-tick",
                CALIBRATION_CLOCK_FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        tuple(sorted(quantities, key=lambda value: value.quantity_id)),
        IndependentUnitSpec(
            UNIT,
            "One fresh calibration root; policies, words and views are nested",
            f"{PREFIX}.root-id",
        ),
        authority,
        (
            PortSpec(
                f"{PREFIX}.force-port",
                quantities[0].quantity_id,
                CALIBRATION_CLOCK,
                PortDirection.INPUT,
                BalanceRole.NONE,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
            *(
                PortSpec(
                    receiver,
                    quantity.quantity_id,
                    CALIBRATION_CLOCK,
                    PortDirection.OUTPUT,
                    BalanceRole.OBSERVATION,
                )
                for receiver, quantity in zip(RECEIVERS, quantities[3:], strict=True)
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def _experiment(system: SystemSpec, exposure: PreparedExposureInspection) -> ExperimentSpec:
    cutoff = InformationCutoff(
        f"{PREFIX}.design-cutoff", CALIBRATION_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{PREFIX}.obligations",
        SupportSpec(
            f"{PREFIX}.support",
            system.relation.relation_id,
            UNIT,
            192,
            2,
            cutoff.cutoff_id,
            (),
            (f"{PREFIX}.denominator",),
            (),
            required,
        ),
        ValiditySpec(
            f"{PREFIX}.validity",
            (f"{PREFIX}.fixed-roster", f"{PREFIX}.whole-root-simultaneous-score"),
            ("finite-complete-native-delivery", "simulator-local-denominator"),
            ("NUMERICAL_OR_OBSERVATION_FAILURE",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            f"{PREFIX}.root-maximum-split-conformal",
            UNIT,
            Decimal("0.95"),
            system.relation.receiver_quantity_ids,
            ("ORDER_INDEX_93_OF_96_PER_CONTEXT",),
            required,
        ),
        (
            FalsifierSpec(
                f"{PREFIX}.falsifier",
                FalsifierKind.BASELINE_COMPARATOR,
                CALIBRATION_CALIBRATION_CAPABILITY.capability_key,
                "The protected root-maximum order statistic is nonfinite in either context.",
                "Retain the valid negative and stop conditional response qualification without refitting scales or replacing roots.",
                required,
            ),
        ),
        ClosureSpec(
            f"{PREFIX}.closure",
            (f"{PREFIX}.four-policy-nine-word-chart",),
            (f"{PREFIX}.absolute-two-receiver-response",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{PREFIX}.convergence",
            ("paired-primary-half-complete-root-products",),
            tuple(value.view_id for value in system.numerical_views),
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{PREFIX}.compute-evidence",
            system.computability_envelopes[0].envelope_id,
            tuple(value.view_id for value in system.numerical_views),
            ReadinessStatus.READY,
            (),
            (f"{PREFIX}.provisional-bounded-reservation",),
        ),
    )
    claim = ClaimSpec(
        f"{PREFIX}.calibration-claim",
        system.world.world_id,
        system.relation.relation_id,
        "The frozen dependent refinement model and scales admit, or fail to admit, a finite simultaneous fresh calibration multiplier in each context.",
        "Maximum residual over policy, view, word, readout and output for 96 fresh roots per context.",
        UNIT,
        EvidenceRung.RESPONSE,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "fresh calibration calibrates prediction sets only; independent conditional response qualification qualification remains required.",
        ("frozen-q2-native-medium", "independent-root-only-inference"),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
    units = exposure.proposed_unit_ids
    return ExperimentSpec(
        PREFIX,
        system.system_id,
        system.world.world_id,
        system.relation,
        UNIT,
        (claim,),
        AssignmentSpec(
            f"{PREFIX}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            UNIT,
            system.relation.action_quantity_ids,
            "Four dependent refinement-frozen provider policies and the complete nine-word menu are nested within each root; paired views share each native acquisition.",
            (f"{PREFIX}.fixed-roster",),
        ),
        system.relation.receiver_quantity_ids,
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                CALIBRATION_POLICY_CAPABILITY.capability_key,
                system.relation.receiver_quantity_ids,
                "Primary, HOLD, dependent refinement-best-fixed and conventional provider policies share the same pre-parent cutoff.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                f"{PREFIX}.simultaneous-95-percent",
                Decimal("0.95"),
                "root-maximum-coverage",
                192,
                "Rank ceil(97*0.95)=93 independently in each 96-root context; no top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            units,
            "prepared-response.conditional-risk-calibration.fresh-cohort",
            sha256(
                canonical_json_bytes(
                    tuple(f"prepared-response.conditional-risk-calibration.{index:03d}" for index in range(192))
                )
            ).hexdigest(),
            ("prepared-response.conditional-risk-calibration.sealed-risk-validation",),
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
    policy: PreparedPolicyDecisionConfig,
    projection: PreparedResponseCalibrationProjectionConfig,
    calibration: PreparedResponseCalibrationCalibrationConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    definitions = (
        (
            "linked-source-materialization",
            ScientificStage.PREPARE,
            source,
            source.spec_id,
            CALIBRATION_SOURCE_CAPABILITY,
            (),
            _outputs(
                (
                    ("native-result", PreparedResponseCalibrationNativeTaskResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            BarrierKind.NONE,
        ),
        (
            f"{PREFIX}.prototype.policy",
            ScientificStage.DEVELOP,
            policy,
            policy.config_id,
            CALIBRATION_POLICY_CAPABILITY,
            ("linked-source-materialization",),
            _outputs(
                (
                    ("parent-decision", PreparedParentDecision.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                )
            ),
            BarrierKind.FREEZE,
        ),
        (
            f"{PREFIX}.prototype.parent",
            ScientificStage.ACQUIRE,
            source,
            source.spec_id,
            CALIBRATION_SOURCE_CAPABILITY,
            tuple(sorted(("linked-source-materialization", f"{PREFIX}.prototype.policy"))),
            _outputs(
                (
                    ("native-result", PreparedResponseCalibrationNativeTaskResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            BarrierKind.NONE,
        ),
        (
            f"{PREFIX}.prototype.projection",
            ScientificStage.TRANSFORM,
            projection,
            projection.config_id,
            CALIBRATION_PROJECTION_CAPABILITY,
            tuple(
                sorted(
                    (
                        "linked-source-materialization",
                        f"{PREFIX}.prototype.parent",
                        f"{PREFIX}.prototype.policy",
                    )
                )
            ),
            _outputs(
                (
                    ("view-report", PreparedResponseCalibrationViewProjection.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                )
            ),
            BarrierKind.FREEZE,
        ),
        (
            PreparedResponseCalibrationCalibrationTask.TASK_ID,
            ScientificStage.EVALUATE,
            calibration,
            calibration.config_id,
            CALIBRATION_CALIBRATION_CAPABILITY,
            (f"{PREFIX}.prototype.projection",),
            _outputs(
                (
                    ("calibrated-library", PreparedResponseCalibrationCalibratedLibrary.SCHEMA),
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
                    task_id,
                    stage,
                    capability.capability_key,
                    capability.capability_version,
                    _config_ref(
                        config,
                        ObjectIdentity.from_record(config_id, config),
                        capability,
                    ),
                    dependencies,
                    outputs,
                    capability.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if stage is ScientificStage.EVALUATE
                    else VisibilityCeiling.DEVELOPMENT_ONLY,
                    capability.resource_ceiling,
                    (),
                    barrier,
                    1,
                    (f"{PREFIX}.single-terminal",)
                    if stage is ScientificStage.EVALUATE
                    else (f"{PREFIX}.prototype.{stage.value.lower()}",),
                )
                for (
                    task_id,
                    stage,
                    config,
                    config_id,
                    capability,
                    dependencies,
                    outputs,
                    barrier,
                ) in definitions
            ),
            key=lambda value: value.step_id,
        )
    )
    return _programme_template(
        ProtocolTemplate(f"{PREFIX}.protocol", "1.0.0", steps, False, False, True),
        source,
        source.spec_id,
        experiment,
        registry,
        prefix=PREFIX,
        source_task_id="linked-source-materialization",
        terminal_task_id=PreparedResponseCalibrationCalibrationTask.TASK_ID,
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.ACQUIRE: "native-result",
            ScientificStage.TRANSFORM: "view-report",
            ScientificStage.DEVELOP: "parent-decision",
            ScientificStage.EVALUATE: "calibrated-library",
        },
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_prepared_response_calibration_authoring(
    *,
    design_packet_sha256: str,
    implementation_sha256: str,
    exposure: PreparedExposureInspection,
    development_library: PreparedResponseDevelopmentNominalLibrary,
) -> PreparedResponseCalibrationAuthoringBundle:
    source = exposure.source_config
    if (
        source.stage != 'calibration'
        or not exposure.units_and_seeds_unexposed
        or development_library.disposition != "NOMINATED_FOR_FRESH_CALIBRATION"
        or development_library.policy_library is None
        or source.selected_amplitude
        != development_library.config.fit.projection.native_spec.selected_amplitude
    ):
        raise ValueError("prepared fresh calibration authoring requires fresh roots and an authentic dependent refinement nomination")
    policy = PreparedPolicyDecisionConfig(f"{PREFIX}.policy-config", source, development_library)
    projection = PreparedResponseCalibrationProjectionConfig(
        f"{PREFIX}.projection-config", source, policy
    )
    calibration = PreparedResponseCalibrationCalibrationConfig(
        f"{PREFIX}.calibration-config",
        projection,
        PreparedStatisticalSpec(f"{PREFIX}.statistics"),
    )
    system = prepared_response_calibration_system(source)
    experiment = _experiment(system, exposure)
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=CALIBRATION_BUDGET,
        objective=(
            "Calibrate or reject one simultaneous dependent refinement-frozen prediction-set package "
            "using 96 fresh roots per context."
        ),
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    registry_root = build_observation_evidence_world_registry()
    world = next(
        value
        for value in registry_root.world_profiles
        if value.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{PREFIX}.evidence-profile",
        draft_id=f"{PREFIX}.draft",
        registry=registry_root,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(EvidenceRung.MEASUREMENT, EvidenceRung.RESPONSE),
    )
    feasibility = resolve_candidate_profile_feasibility(
        selection=evidence,
        resolver=EvidenceProfileRegistryResolver(
            f"{PREFIX}.profile-resolver", (registry_root,)
        ),
    )
    if feasibility.disposition is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION:
        raise ValueError(f"prepared fresh calibration evidence profile is not compilable: {feasibility.disposition}")
    source_id = ObjectIdentity.from_record(source.spec_id, source)
    development_identity = ObjectIdentity.from_record("prepared-response.dependent-refinement.nominal-library", development_library)
    native_invocations = prepared_response_calibration_native_invocations(source)
    source_profile = NativeSourceProfile(
        f"{PREFIX}.native-source-profile",
        CapabilitySelection(
            CALIBRATION_SOURCE_CAPABILITY.capability_key,
            CALIBRATION_SOURCE_CAPABILITY.capability_version,
            CALIBRATION_SOURCE_CAPABILITY.implementation_sha256,
        ),
        source_id,
        NATIVE_PAIR_SCHEMA,
        tuple(root.physical_unit_id for root in source.roots),
        len(native_invocations),
        sum(value.maximum_native_updates for value in native_invocations),
        CALIBRATION_BUDGET,
        development_identity,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    evaluation_units = source_profile.physical_independent_unit_ids
    identified = sorted(
        (
            (source.spec_id, source),
            (policy.config_id, policy),
            (projection.config_id, projection),
            (calibration.config_id, calibration),
            (source_profile.profile_id, source_profile),
        ),
        key=lambda value: value[1].SCHEMA,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(value for _, value in identified)
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in _BINDINGS
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[value.SCHEMA] for value in payloads)
    qualification = MaterializationQualificationReceipt(
        f"{PREFIX}.config-qualification",
        source.spec_id,
        source_id,
        source.fingerprint(),
        system.world.world_id,
        ObjectIdentity.from_record(projection.config_id, projection),
        tuple(value.view_id for value in system.numerical_views),
        tuple(sorted({value.native_unit for value in system.quantities})),
        (CALIBRATION_ACTION_FRAME,),
        (CALIBRATION_CLOCK,),
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
        source_id,
        source.fingerprint(),
        source.fingerprint(),
        qualification.observation_operator,
        tuple(value.view_id for value in system.numerical_views),
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    science = ObjectIdentity(
        "prepared-response.final-design",
        'empirical-lawhood/document/markdown',
        "1.0.0",
        design_packet_sha256,
    )
    cutoff = InformationCutoff(
        f"{PREFIX}.design-freeze", CALIBRATION_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    design_inputs = tuple(
        sorted(
            (
                DesignInputRecord(
                    f"{PREFIX}.design-input",
                    science,
                    design_packet_sha256,
                    cutoff,
                    DesignInputRole.MOTIVATION,
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    "owner.prepared-response",
                ),
                DesignInputRecord(
                    f"{PREFIX}.d-prerequisite",
                    development_identity,
                    development_library.fingerprint(),
                    cutoff,
                    DesignInputRole.DEVELOPMENT_TUNING,
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    "owner.prepared-response",
                    parent_input_ids=(f"{PREFIX}.design-input",),
                ),
                DesignInputRecord(
                    f"{PREFIX}.exposure-input",
                    ObjectIdentity.from_record(exposure.inspection_id, exposure),
                    exposure.fingerprint(),
                    cutoff,
                    DesignInputRole.READINESS_METADATA,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    "owner.prepared-response",
                    exposure.excluded_unit_ids,
                    exposure.excluded_seed_ids,
                    (f"{PREFIX}.d-prerequisite", f"{PREFIX}.design-input"),
                ),
            ),
            key=lambda value: value.input_id,
        )
    )
    selected = {value.capability_key for value in _CAPABILITIES}
    registry = CapabilityRegistry(
        f"{PREFIX}.registry",
        tuple(
            value
            for value in GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry.capabilities
            if value.capability_key in selected
        ),
    )
    template = _template(source, policy, projection, calibration, experiment, registry)
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(CALIBRATION_SOURCE_BINDING.binding_id)
    if not isinstance(factory, PreparedResponseCalibrationSourceFactory):
        raise ValueError("prepared fresh calibration authoring requires its registered source expansion")
    template = rebind_study_template_to_expanded_protocol(
        study_template=template,
        expanded_protocol=factory.expand_parameterised_protocol(
            records=payloads, template=template.protocol
        ),
        capability_registry=registry,
    )
    draft = StudyDraft(
        f"{PREFIX}.draft",
        StudyDraftLifecycle.DRAFT,
        "Calibrate the dependent refinement-frozen simultaneous prediction set, or retain a valid noncalibratable result.",
        (f"{PREFIX}.alternative.calibrated", f"{PREFIX}.alternative.not-calibratable"),
        DesignOrigin(
            f"{PREFIX}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (f"{PREFIX}.d-prerequisite", f"{PREFIX}.exposure-input"),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        (),
        evaluation_units,
        (),
        exposure.proposed_seed_ids,
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                value.capability_key,
                value.capability_version,
                value.implementation_sha256,
            )
            for value in _CAPABILITIES
        ),
        (source_ref,),
        CALIBRATION_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{PREFIX}.context",
        registry,
        (template,),
        (qualification,),
        design_inputs,
        implementation_sha256,
    )
    base, standard = _entry(
        draft,
        context,
        science,
        prefix=PREFIX,
        minimum_units=192,
        operand_description="Prepared fresh calibration whole-root {domain} calibration operand.",
        estimator="root-maximum-split-conformal",
        uncertainty="simultaneous-95-percent-rank-93",
        ceiling=EvidenceCeiling.RESPONSE,
        evaluator=CALIBRATION_CALIBRATION_CAPABILITY,
        input_schema=PreparedResponseCalibrationViewProjection.SCHEMA,
        output_schema=PreparedResponseCalibrationCalibratedLibrary.SCHEMA,
        evidence_unit_ids=evaluation_units,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    proposed = tuple(
        ProposedStudyExtension(
            f"{PREFIX}.extension.{index:02d}",
            f"{PREFIX}.namespace.{index:02d}",
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
        f"{PREFIX}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(value.namespace_id for value in proposed),
        proposed,
    )
    return PreparedResponseCalibrationAuthoringBundle(
        ExecutableStudyDefinition(f"{PREFIX}.authoring", base, extensions),
        replace(standard, base=context),
        CandidateCapabilityCatalog(
            f"{PREFIX}.catalog",
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


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationCandidateContextProvider:
    bundle: PreparedResponseCalibrationAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("draft differs from its exact prepared fresh calibration bundle")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("standard authoring differs from its exact prepared fresh calibration bundle")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )


__all__ = [
    "CALIBRATION_BUDGET",
    'PreparedResponseCalibrationAuthoringBundle',
    'PreparedResponseCalibrationCandidateContextProvider',
    'build_prepared_response_calibration_authoring',
    'prepared_response_calibration_system',
]
