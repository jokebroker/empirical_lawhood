"Outcome-blind prepared dependent refinement authoring through the public native-crossfit route."
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

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
from empirical_lawhood.planning.native_source import NativeCrossfitConfig, NativeCrossfitExperiment, NativeSourceProfile
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.prospective_config import ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
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
from empirical_lawhood.runtime.response_experiment import compile_response_experiment
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.development_roster import DEVELOPMENT_ACTION_FRAME, DEVELOPMENT_CLOCK, DEVELOPMENT_CLOCK_FRAME, DEVELOPMENT_NAMESPACE, prepared_response_development_clock_evaluator, prepared_response_development_native_declarations
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import DEVELOPMENT_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import DEVELOPMENT_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation, prepared_static_native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.control.backbone_linked_campaign.executable_binding import (
    NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING,
    PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
)
from empirical_lawhood.adapters.methods.qualification_profiles import ComponentQualificationProfile
from empirical_lawhood.adapters.methods.prepared_response.development_fit import PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit
from empirical_lawhood.adapters.methods.prepared_response.development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection
from empirical_lawhood.adapters.methods.prepared_response.development_provider import PreparedResponseDevelopmentSelectionTask
from empirical_lawhood.adapters.methods.prepared_response.development_selection import PreparedResponseDevelopmentCandidateCost, PreparedResponseDevelopmentNominalLibrary, PreparedResponseDevelopmentSelectionConfig
from empirical_lawhood.adapters.methods.prepared_response.development_benchmarks import PreparedResponseDevelopmentConstantGainReport
from empirical_lawhood.adapters.methods.prepared_response.executable_binding import DEVELOPMENT_FIT_BINDING, DEVELOPMENT_PROJECTION_BINDING, DEVELOPMENT_SELECTION_BINDING
from empirical_lawhood.adapters.methods.prepared_response.extension_bundle import DEVELOPMENT_FIT_CAPABILITY, DEVELOPMENT_PROJECTION_CAPABILITY, DEVELOPMENT_SELECTION_CAPABILITY
from empirical_lawhood.adapters.methods.prepared_response.qualification import PreparedResponseSourceQualificationEvaluation

from .exposure import PreparedExposureInspection
PREFIX = DEVELOPMENT_NAMESPACE
UNIT = f"{PREFIX}.independent-stochastic-root"
RECEIVERS = (f"{PREFIX}.receiver.m1", f"{PREFIX}.receiver.m2")
# ResourceBudget aggregates task ceilings.  Eight workers may therefore carry
# up to eight task-seconds per elapsed second; the separate campaign-elapsed
# contract remains the binding 24-hour/19.2-hour-central scientific stop.
DEVELOPMENT_BUDGET = ResourceBudget(
    8,
    24 * 1024**3,
    0,
    8 * 24 * 60 * 60,
    1024**4,
    256 * 1024**3,
)
_CAPABILITIES = (
    DEVELOPMENT_FIT_CAPABILITY,
    DEVELOPMENT_PROJECTION_CAPABILITY,
    DEVELOPMENT_SELECTION_CAPABILITY,
    DEVELOPMENT_SOURCE_CAPABILITY,
)
_BINDINGS = (DEVELOPMENT_FIT_BINDING, DEVELOPMENT_PROJECTION_BINDING, DEVELOPMENT_SELECTION_BINDING, DEVELOPMENT_SOURCE_BINDING)

# dependent refinement may nominate but may not qualify a law.  These exact identities reserve the
# separately implemented fresh calibration qualification/finalization owners before dependent refinement
# issue; they are neither selected dependent refinement capabilities nor an authority grant.
CALIBRATION_QUALIFICATION_PROFILE = ObjectIdentity(
    "prepared-response.fresh-response-calibration.response-law-qualification-profile",
    ComponentQualificationProfile.SCHEMA,
    "1.0.0",
    sha256(b"prepared-response.fresh-response-calibration.response-law-qualification-profile").hexdigest(),
)
CALIBRATION_LAW_FINALIZER = ObjectIdentity(
    "prepared-response.fresh-response-calibration.response-law-finalizer",
    'empirical-lawhood/runtime/capability-manifest',
    "1.0.0",
    sha256(b"prepared-response.fresh-response-calibration.response-law-finalizer").hexdigest(),
)


def prepared_response_development_system(source: PreparedNativeSpec) -> SystemSpec:
    if source.stage != 'development':
        raise ValueError("prepared dependent refinement system requires the dependent refinement native denominator")
    pre = AvailabilitySpec(DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0))
    observed = replace(
        pre, phase=CausalPhase.RECEIVER, outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE
    )
    quantities = tuple(
        QuantitySpec(
            f"{PREFIX}.quantity.{key}",
            label,
            kind,
            dimension,
            unit,
            DEVELOPMENT_ACTION_FRAME,
            DEVELOPMENT_CLOCK,
            availability,
            direction,
        )
        for key, label, kind, dimension, unit, availability, direction in (
            (
                "two-port-action",
                "Frozen source qualification-nominated two-port force and time-matched parent schedules",
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
                "Causal pre-parent and handoff I0/I1 instrument histories",
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
        HorizonSpec(f"{PREFIX}.horizon", DEVELOPMENT_CLOCK, Decimal(320), "reference-tick"),
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Prepared six-matrix numerical medium",
        WorldKind.NUMERICAL_SIMULATOR,
        ("BAOAB-unit-kinetic-mass", "q2-six-Hermitian-matrices", "two-frozen-preparent-HS-ports"),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.LOCAL_LAW,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{PREFIX}.computability",
        ("finite-128-root-dependent refinement",),
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
    policy = AuthorityPolicy(
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
        DEVELOPMENT_BUDGET,
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Fresh whole-root nested development of absolute prepared response",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                DEVELOPMENT_CLOCK,
                "One reference tick is 0.001 Langevin time",
                "reference-tick",
                DEVELOPMENT_CLOCK_FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        tuple(sorted(quantities, key=lambda value: value.quantity_id)),
        IndependentUnitSpec(
            UNIT, "One fresh root; parents, words and views are nested", f"{PREFIX}.root-id"
        ),
        policy,
        (
            PortSpec(
                f"{PREFIX}.force-port",
                quantities[0].quantity_id,
                DEVELOPMENT_CLOCK,
                PortDirection.INPUT,
                BalanceRole.NONE,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
            *(
                PortSpec(
                    receiver,
                    quantity.quantity_id,
                    DEVELOPMENT_CLOCK,
                    PortDirection.OUTPUT,
                    BalanceRole.OBSERVATION,
                )
                for receiver, quantity in zip(RECEIVERS, quantities[3:], strict=True)
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def _lineage() -> tuple[ProspectivePackageLineage, ProspectiveTerminalMatrix]:
    q = "prepared-response.source-qualification"
    return (
        ProspectivePackageLineage(
            f"{PREFIX}.lineage",
            q,
            f"{PREFIX}.qualified-q-branch",
            tuple(sorted((
                ProspectivePackageLineageNode(
                    q,
                    "Completed excluded source qualification qualification",
                    ProspectivePackageKind.EXCLUDED_QUALIFICATION,
                    (),
                    ("prepared-response.source-qualification.exact-issued-route",),
                    EvidenceCeiling.RESPONSE,
                    None,
                    False,
                    False,
                    False,
                    False,
                ),
                ProspectivePackageLineageNode(
                    PREFIX,
                    "Fresh nominal dependent refinement development under source qualification's selected charter",
                    ProspectivePackageKind.PROSPECTIVE_CHILD,
                    (q,),
                    (f"{PREFIX}.authentic-q-selected", f"{PREFIX}.exact-route-gate"),
                    EvidenceCeiling.LOCAL_LAW,
                    None,
                    False,
                    False,
                    False,
                    False,
                ),
            ), key=lambda value: value.package_id)),
            (f"{PREFIX}.q-not-qualified-stop",),
            True,
            False,
            False,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
        ProspectiveTerminalMatrix(
            f"{PREFIX}.terminal-matrix",
            (
                ProspectiveSelectorTransition(
                    1,
                    "source qualification selected one frozen charter and amplitude",
                    "Enter fixed dependent refinement only after exact route, authority and resource admission",
                ),
                ProspectiveSelectorTransition(
                    2, "source qualification selection or exact readiness is absent", "STOP without native dependent refinement contact"
                ),
            ),
            (
                ProspectiveTerminalRule(
                    1,
                    "Missing authentic native or method operands",
                    "UNEVALUABLE",
                    "Authenticated available dependent refinement evidence",
                    "Stop the affected follow-up and preserve every acquired outcome",
                ),
                ProspectiveTerminalRule(
                    2,
                    "Complete dependent refinement operands and nominal selection",
                    "NOMINATE_OR_NOT_SUPPORTED",
                    "Whole-root outer predictions and all-dependent refinement refits",
                    "Only a nominated library may enter fresh calibration; dependent refinement publishes no law",
                ),
            ),
            True,
            True,
            False,
            False,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
    )


def _experiment(system: SystemSpec, exposure: PreparedExposureInspection) -> ExperimentSpec:
    units = exposure.proposed_unit_ids
    cutoff = InformationCutoff(
        f"{PREFIX}.design-cutoff", DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{PREFIX}.obligations",
        SupportSpec(
            f"{PREFIX}.support",
            system.relation.relation_id,
            UNIT,
            128,
            2,
            cutoff.cutoff_id,
            (),
            (f"{PREFIX}.denominator",),
            (),
            required,
        ),
        ValiditySpec(
            f"{PREFIX}.validity",
            (f"{PREFIX}.fixed-roster", f"{PREFIX}.whole-root-crossfit"),
            ("finite-complete-native-delivery", "simulator-local-denominator"),
            ("NUMERICAL_OR_OBSERVATION_FAILURE",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            f"{PREFIX}.nested-whole-root-development",
            UNIT,
            Decimal("0.95"),
            system.relation.receiver_quantity_ids,
            ("NOMINAL_DEVELOPMENT_NOT_PROTECTED_CALIBRATION",),
            required,
        ),
        (
            FalsifierSpec(
                f"{PREFIX}.falsifier",
                FalsifierKind.BASELINE_COMPARATOR,
                DEVELOPMENT_SELECTION_CAPABILITY.capability_key,
                "No common candidate satisfies finite-root, nominal coverage, sharpness and intercept-improvement screens in both contexts.",
                "Retain the valid negative; do not refit, top up roots or change the family.",
                required,
            ),
        ),
        ClosureSpec(
            f"{PREFIX}.closure",
            (f"{PREFIX}.five-parent-nine-word-chart",),
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
        f"{PREFIX}.nominal-library-claim",
        system.world.world_id,
        system.relation.relation_id,
        "The frozen finite family yields, or fails to yield, one common nominal absolute-response library for fresh calibration.",
        "Whole-root nested crossfit over 64 independent dependent refinement roots per context with paired numerical views.",
        UNIT,
        EvidenceRung.RESPONSE,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "The dependent refinement result is a nomination only; fresh calibration is required before any calibrated law or admission claim.",
        ("frozen-q2-native-medium", "independent-root-only-inference"),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
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
            "Five time-matched parents and the source qualification-selected nine-word chart are nested within each root; paired views share each native acquisition.",
            (f"{PREFIX}.fixed-roster",),
        ),
        system.relation.receiver_quantity_ids,
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                DEVELOPMENT_SELECTION_CAPABILITY.capability_key,
                system.relation.receiver_quantity_ids,
                "Training-only context intercept, four frozen candidate structures and measured excluded-canary cost tie-break.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                f"{PREFIX}.finite-outer-root-count",
                Decimal(2),
                "roots",
                128,
                "At least 62 finite outer-heldout roots per context, no top-up; all-dependent refinement refit only after selection.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            units,
            "prepared-response.fresh-response-calibration.fresh-cohort",
            sha256(canonical_json_bytes(tuple(f"prepared-response.fresh-response-calibration.{i:03d}" for i in range(192)))).hexdigest(),
            ("prepared-response.fresh-response-calibration.sealed-calibration",),
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
    projection: PreparedResponseDevelopmentProjectionConfig,
    fit: PreparedResponseDevelopmentFitConfig,
    selection: PreparedResponseDevelopmentSelectionConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    definitions = (
        (
            "linked-source-materialization",
            ScientificStage.PREPARE,
            source,
            source.spec_id,
            DEVELOPMENT_SOURCE_CAPABILITY,
            (),
            _outputs(
                (("native-result", PreparedNativeTaskResult.SCHEMA),
                 ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA)),
                ("native-observations", NATIVE_PAIR_SCHEMA),
            ),
            BarrierKind.NONE,
        ),
        (
            "linked-evidence-projection",
            ScientificStage.TRANSFORM,
            projection,
            projection.config_id,
            DEVELOPMENT_PROJECTION_CAPABILITY,
            ("linked-source-materialization",),
            _outputs((("view-report", PreparedResponseDevelopmentViewProjection.SCHEMA),
                      ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA))),
            BarrierKind.FREEZE,
        ),
        (
            f"{PREFIX}.prototype.fit",
            ScientificStage.DEVELOP,
            fit,
            fit.config_id,
            DEVELOPMENT_FIT_CAPABILITY,
            ("linked-evidence-projection",),
            _outputs((("model-fit", PreparedResponseDevelopmentModelFit.SCHEMA),
                      ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA))),
            BarrierKind.FREEZE,
        ),
        (
            PreparedResponseDevelopmentSelectionTask.TASK_ID,
            ScientificStage.EVALUATE,
            selection,
            selection.config_id,
            DEVELOPMENT_SELECTION_CAPABILITY,
            (f"{PREFIX}.prototype.fit",),
            _outputs((("nominal-library", PreparedResponseDevelopmentNominalLibrary.SCHEMA),
                      ("constant-gain-benchmark", PreparedResponseDevelopmentConstantGainReport.SCHEMA),
                      ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                      ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA))),
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
                    else OutcomeAccess.DEVELOPMENT_VISIBLE,
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
                for task_id, stage, config, config_id, capability, dependencies, outputs, barrier in definitions
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
        terminal_task_id=PreparedResponseDevelopmentSelectionTask.TASK_ID,
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.TRANSFORM: "view-report",
            ScientificStage.DEVELOP: "model-fit",
            ScientificStage.EVALUATE: "nominal-library",
        },
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_prepared_response_development_authoring(
    *,
    design_packet_sha256: str,
    implementation_sha256: str,
    exposure: PreparedExposureInspection,
    source_qualification: PreparedResponseSourceQualificationEvaluation,
    candidate_costs: tuple[PreparedResponseDevelopmentCandidateCost, ...],
) -> PreparedResponseDevelopmentAuthoringBundle:
    source = exposure.source_config
    if (
        source.stage != 'development'
        or not exposure.units_and_seeds_unexposed
        or source_qualification.selected_charter is None
        or source.selected_amplitude != source_qualification.selected_charter.amplitude
    ):
        raise ValueError("prepared dependent refinement authoring requires fresh dependent refinement roots and an authentic source qualification selection")
    projection = PreparedResponseDevelopmentProjectionConfig(f"{PREFIX}.projection-config", source, source_qualification)
    declarations = prepared_response_development_native_declarations(source, prepared_response_development_clock_evaluator())
    fit = PreparedResponseDevelopmentFitConfig(f"{PREFIX}.fit-config", projection, declarations.crossfit)
    selection = PreparedResponseDevelopmentSelectionConfig(f"{PREFIX}.selection-config", fit, candidate_costs)
    system = prepared_response_development_system(source)
    experiment = _experiment(system, exposure)
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=DEVELOPMENT_BUDGET,
        objective="Nominate or reject one frozen absolute-response library using whole-root dependent refinement cross-fitting; publish no calibrated law.",
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
        raise ValueError(f"prepared dependent refinement evidence profile is not compilable: {feasibility.disposition}")
    source_id = ObjectIdentity.from_record(source.spec_id, source)
    source_qualification_identity = ObjectIdentity.from_record(source_qualification.evaluation_id, source_qualification)
    source_profile = NativeSourceProfile(
        f"{PREFIX}.native-source-profile",
        CapabilitySelection(
            DEVELOPMENT_SOURCE_CAPABILITY.capability_key,
            DEVELOPMENT_SOURCE_CAPABILITY.capability_version,
            DEVELOPMENT_SOURCE_CAPABILITY.implementation_sha256,
        ),
        source_id,
        NATIVE_PAIR_SCHEMA,
        tuple(value.physical_independent_unit_id for value in declarations.units),
        len(prepared_static_native_invocations(source)),
        sum(
            value.maximum_native_updates
            for value in prepared_static_native_invocations(source)
        ),
        DEVELOPMENT_BUDGET,
        source_qualification_identity,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    evidence_id = ObjectIdentity.from_record(evidence.selection_id, evidence)
    profile_id = ObjectIdentity.from_record(source_profile.profile_id, source_profile)
    campaign_id = ObjectIdentity.from_record(campaign.campaign_id, campaign)
    lineage, terminal = _lineage()
    qualification_config = NativeCrossfitConfig(
        f"{PREFIX}.response-law-qualification-config",
        evidence_id,
        profile_id,
        campaign_id,
        declarations.chart,
        lineage,
        terminal,
        declarations.units,
        declarations.acquisitions,
        declarations.views,
        source_profile.physical_independent_unit_ids,
        declarations.crossfit.development_unit_ids,
        (),
        ("r1", "r2"),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(fit.config_id, fit),
        CALIBRATION_QUALIFICATION_PROFILE,
        CALIBRATION_LAW_FINALIZER,
        None,
        EvidenceCeiling.LOCAL_LAW,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        declarations.projections,
        declarations.crossfit,
    )
    extension = NativeCrossfitExperiment(
        f"{PREFIX}.carrier",
        evidence_id,
        profile_id,
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
        f"{PREFIX}.native-binding",
        f"{CAMPAIGN}.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        DEVELOPMENT_SOURCE_CAPABILITY.capability_key,
        DEVELOPMENT_SOURCE_CAPABILITY.capability_version,
        DEVELOPMENT_SOURCE_CAPABILITY.implementation_sha256,
        ObjectIdentity.from_record(DEVELOPMENT_SOURCE_BINDING.binding_id, DEVELOPMENT_SOURCE_BINDING),
        NativeSourceProfile.SCHEMA,
        PreparedNativeInvocation.SCHEMA,
        None,
        PreparedNativeTaskResult.SCHEMA,
        PreparedResponseDevelopmentViewProjection.SCHEMA,
        source_id,
        tuple(
            NativeReceiverContract(
                receiver,
                f"{PREFIX}.quantity.receiver.m{index}",
                "hilbert-schmidt-native",
                DEVELOPMENT_ACTION_FRAME,
                f"{CAMPAIGN}.frozen-primary-port.m{index}",
                (DEVELOPMENT_CLOCK,),
            )
            for index, receiver in enumerate(RECEIVERS, 1)
        ),
        (NativeClockContract(DEVELOPMENT_CLOCK, "reference-tick", DEVELOPMENT_CLOCK_FRAME),),
        NativeActionContract(
            f"{PREFIX}.quantity.two-port-action", True, True, True, True, "HOLD"
        ),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    identified = sorted(
        (
            (source.spec_id, source),
            (projection.config_id, projection),
            (fit.config_id, fit),
            (selection.config_id, selection),
            (source_profile.profile_id, source_profile),
            (qualification_config.config_id, qualification_config),
            (extension.extension_set_id, extension),
            (substrate.binding_id, substrate),
        ),
        key=lambda value: value[1].SCHEMA,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(value for _, value in identified)
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in (
            *_BINDINGS,
            NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING,
            PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
        )
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
        (DEVELOPMENT_ACTION_FRAME,),
        (DEVELOPMENT_CLOCK,),
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
        "prepared-response.final-design", 'empirical-lawhood/document/markdown', "1.0.0", design_packet_sha256
    )
    cutoff = InformationCutoff(
        f"{PREFIX}.design-freeze", DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
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
                    f"{PREFIX}.q-prerequisite",
                    source_qualification_identity,
                    source_qualification.fingerprint(),
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
                    (f"{PREFIX}.design-input", f"{PREFIX}.q-prerequisite"),
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
    template = _template(source, projection, fit, selection, experiment, registry)
    draft = StudyDraft(
        f"{PREFIX}.draft",
        StudyDraftLifecycle.DRAFT,
        "Nominate the lowest measured-cost common eligible absolute-response family, or retain a valid negative.",
        (f"{PREFIX}.alternative.nominal-library", f"{PREFIX}.alternative.not-supported"),
        DesignOrigin(
            f"{PREFIX}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (f"{PREFIX}.exposure-input", f"{PREFIX}.q-prerequisite"),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        declarations.crossfit.development_unit_ids,
        (),
        tuple(f"{value}.native-rng" for value in declarations.crossfit.development_unit_ids),
        (),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                value.capability_key, value.capability_version, value.implementation_sha256
            )
            for value in _CAPABILITIES
        ),
        (source_ref,),
        DEVELOPMENT_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{PREFIX}.context",
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
        prefix=PREFIX,
        minimum_units=128,
        operand_description="Prepared dependent refinement whole-root {domain} nomination operand.",
        estimator="nested-whole-root-development",
        uncertainty="nominal-development-only",
        ceiling=EvidenceCeiling.RESPONSE,
        evaluator=DEVELOPMENT_SELECTION_CAPABILITY,
        input_schema=PreparedResponseDevelopmentModelFit.SCHEMA,
        output_schema=PreparedResponseDevelopmentNominalLibrary.SCHEMA,
        evidence_unit_ids=declarations.crossfit.development_unit_ids,
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
    return PreparedResponseDevelopmentAuthoringBundle(
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
class PreparedResponseDevelopmentCandidateContextProvider:
    bundle: PreparedResponseDevelopmentAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("draft differs from its exact prepared dependent refinement bundle")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("standard authoring differs from its exact prepared dependent refinement bundle")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )


__all__ = [
    "CALIBRATION_LAW_FINALIZER",
    "CALIBRATION_QUALIFICATION_PROFILE",
    "DEVELOPMENT_BUDGET",
    'PreparedResponseDevelopmentAuthoringBundle',
    'PreparedResponseDevelopmentCandidateContextProvider',
    'build_prepared_response_development_authoring',
    'prepared_response_development_system',
]
