"""Outcome-blind assay authoring through the existing candidate/compiler/runtime-owner route.

The source profile is the strict native ResponseGeometryAssayNativeConfig. No file transformation
or pre-existing trajectory is represented as the prospective simulator source.
Construction performs no reads, writes, provider execution or authority grant.
"""

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign, study_template_from_protocol, formal_entry

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from .exposure import ResponsePanelExposureInspection, MappedResponsePanelExposureInspection
from empirical_lawhood.adapters.simulators.six_matrix_response.response_panel import assay_physical_unit_id

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
from empirical_lawhood.adapters.simulators.six_matrix_response.response_scientific_inputs import ResponseGeometryScientificInputs

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
from empirical_lawhood.planning.campaigns import (
    CampaignSpec,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, build_observation_evidence_world_registry, bind_profile_selection
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.formal_gaps import (
    FormalGapEvidenceWorld,
)
from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, SourceQualificationSegment, SourceQualificationView
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate, StandardCandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver, ProfileFeasibilityDisposition, resolve_candidate_profile_feasibility
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import CANONICAL_MEDIA_TYPE
from empirical_lawhood.adapters.simulators.response_geometry_prospective.extension_bundle import ASSAY_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.response_geometry_prospective.executable_binding import ASSAY_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeSegment, ResponseGeometryAssayNativeSegmentResult, assay_roots, assay_segments
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA
from empirical_lawhood.adapters.methods.response_geometry_prospective.qualification import ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig, ResponseGeometryAssayEvaluation, ResponseGeometryAssayViewReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.projection import ASSAY_DIAGNOSTICS_SCHEMA
from empirical_lawhood.adapters.methods.response_geometry_prospective.provider import EVALUATOR_TASK_ID
from empirical_lawhood.adapters.methods.response_geometry_prospective.extension_bundle import ASSAY_PROJECTION_CAPABILITY, ASSAY_EVALUATION_CAPABILITY
from empirical_lawhood.adapters.methods.response_geometry_prospective.executable_binding import ASSAY_PROJECTION_BINDING, ASSAY_EVALUATION_BINDING


PREFIX = "response-geometry-assay"
CLOCK = f"{PREFIX}.reference-clock"
FRAME = "response-geometry.fixed-laboratory-hs-frame"
UNIT = f"{PREFIX}.independent-stochastic-root"
# authoring authoring time coordinates are forecasts only. The compiler/runtime-owner resource
# envelope admits non-time limits; the owner removed elapsed-time stopping.
BUDGET = ResourceBudget(8, 32 * 1024**3, 0, 57_600, 64 * 1024**3, 16 * 1024**3)
_CAPABILITIES = (ASSAY_EVALUATION_CAPABILITY, ASSAY_PROJECTION_CAPABILITY, ASSAY_SOURCE_CAPABILITY)
_BINDINGS = (ASSAY_EVALUATION_BINDING, ASSAY_PROJECTION_BINDING, ASSAY_SOURCE_BINDING)


def _record_id(record: CanonicalRecord) -> str:
    for field in ("extension_set_id", "binding_id", "config_id", "profile_id"):
        value = getattr(record, field, None)
        if isinstance(value, str):
            return value
    raise ValueError("native authoring record lacks its declared stable identity")


def _system() -> SystemSpec:
    pre = AvailabilitySpec(CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0))
    observed = AvailabilitySpec(
        CLOCK, CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED, Decimal(32)
    )
    definitions = (
        (
            "action",
            "Parent coupling word and inner signed X force",
            QuantityKind.ACTION,
            "native-action",
            "native-couplings-and-force",
            replace(pre, phase=CausalPhase.ACTION_REQUESTED),
            ResponseDirection.NOT_APPLICABLE,
        ),
        (
            "denominator",
            "q=2 six-matrix native medium",
            QuantityKind.DENOMINATOR,
            "medium-identity",
            "identity",
            pre,
            ResponseDirection.NOT_APPLICABLE,
        ),
        (
            "history",
            "Causal native state and mature observer history",
            QuantityKind.HISTORY,
            "native-history",
            "native-hs-phase-history",
            pre,
            ResponseDirection.NOT_APPLICABLE,
        ),
        (
            "receiver",
            "Delayed signed X displacement and preservation operands",
            QuantityKind.RECEIVER,
            "displacement",
            "hilbert-schmidt-native",
            observed,
            ResponseDirection.SIGNED_VECTOR,
        ),
    )
    quantities = tuple(
        QuantitySpec(
            f"{PREFIX}.quantity.{key}",
            label,
            kind,
            dimension,
            unit,
            FRAME,
            CLOCK,
            available,
            direction,
        )
        for key, label, kind, dimension, unit, available, direction in definitions
    )
    relation = RelationalIdentity(
        f"{PREFIX}.relation",
        (quantities[1].quantity_id,),
        (quantities[2].quantity_id,),
        False,
        (quantities[0].quantity_id,),
        (quantities[3].quantity_id,),
        HorizonSpec(f"{PREFIX}.horizon", CLOCK, Decimal(640), "reference-tick"),
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Declared six-matrix numerical medium",
        WorldKind.NUMERICAL_SIMULATOR,
        ("BAOAB-unit-kinetic-mass", "q2-six-Hermitian-matrices"),
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
        f"{PREFIX}.computability",
        ("finite-native-assay",),
        ("empirical-full-unit-throughput-unmeasured",),
        ("bounded-native-observer-checkpoint",),
        ("bounded-native-observer-checkpoint",),
        8,
        32 * 1024**3,
        0,
        7200,
        16 * 1024**3,
        Decimal(7200),
        None,
    )
    views = tuple(
        NumericalViewSpec(
            f"response-geometry.view.r{refinement}",
            world.world_id,
            UNIT,
            "response-geometry.six-matrix-baoab-equations",
            ("response-geometry.native-closure",),
            ("response-geometry.finite-assay-boundary",),
            (
                NumericalCoordinateSpec(
                    f"{PREFIX}.dt.r{refinement}",
                    NumericalCoordinateKind.TIMESTEP,
                    Decimal("0.001") / refinement,
                    "dimensionless-langevin-time",
                    index,
                ),
            ),
            "response-geometry.solver.baoab",
            "1.0.0",
            "complex128",
            "cpu",
            "response-geometry.native-runtime",
            RandomnessSemantics.GENERATIVE_PREPARATION,
            f"{PREFIX}.native-observer",
            compute.envelope_id,
        )
        for index, refinement in enumerate((1, 2, 4))
    )
    policy = AuthorityPolicy(
        f"{PREFIX}.authority-policy",
        "owner.response-geometry",
        "operator.response-geometry",
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
        BUDGET,
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Excluded native response qualification",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                CLOCK,
                "Reference clock: one tick is 0.001 Langevin time",
                "reference-tick",
                FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        quantities,
        IndependentUnitSpec(
            UNIT,
            "One fresh independently seeded native root; views and branches are nested",
            f"{PREFIX}.root-id",
        ),
        policy,
        (
            PortSpec(
                f"{PREFIX}.force-port",
                quantities[0].quantity_id,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.NONE,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
            PortSpec(
                f"{PREFIX}.receiver-port",
                quantities[3].quantity_id,
                CLOCK,
                PortDirection.OUTPUT,
                BalanceRole.OBSERVATION,
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def _carrier(
    source: ResponseGeometryAssayNativeConfig,
    projection: ResponseGeometryAssayProjectionConfig,
    evaluation: ResponseGeometryAssayEvaluationConfig,
    evidence: EvidenceProfileSelection,
) -> FreshSourceQualificationExperiment:
    native = assay_segments()
    units = tuple(
        ObservationPhysicalUnit(
            assay_physical_unit_id(source, root),
            f"{PREFIX}.coordinate.{root.context}.t{root.landmark_tick}",
            f"{assay_physical_unit_id(source, root)}.instance",
            f"{PREFIX}.family.{root.context}",
            root.fingerprint(),
            None,
        )
        for root in source.roots
    )
    segments = []
    for item in native:
        action_ids: tuple[str, ...] = ()
        if item.phase == "parent":
            action_ids = (f"action.{item.task_id}",)
        elif item.phase in {"inner", "resume"}:
            occurrence = item if item.phase == "inner" else item.predecessor
            assert occurrence is not None
            action_ids = (f"action.{occurrence.task_id}",)
        segments.append(
            SourceQualificationSegment(
                item.task_id,
                f"group.{item.task_id}",
                assay_physical_unit_id(source, item.root),
                None if item.predecessor is None else item.predecessor.task_id,
                CLOCK,
                Decimal(item.start_tick),
                Decimal(item.end_tick),
                tuple(
                    f"{item.root.root_id}.project.r{refinement}"
                    for refinement in item.root.refinements
                ),
                action_ids,
            )
        )
    views = tuple(
        SourceQualificationView(
            f"{root.root_id}.project.r{refinement}",
            assay_physical_unit_id(source, root),
            f"response-geometry.view.r{refinement}",
            tuple(item.task_id for item in native if item.root == root),
        )
        for root in source.roots
        for refinement in root.refinements
    )
    return FreshSourceQualificationExperiment(
        f"{PREFIX}.carrier",
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        units,
        tuple(segments),
        views,
        (f"{PREFIX}.receiver",),
        (CLOCK,),
        ObjectIdentity.from_record(ASSAY_SOURCE_CAPABILITY.capability_key, ASSAY_SOURCE_CAPABILITY),
        ObjectIdentity.from_record(ASSAY_PROJECTION_CAPABILITY.capability_key, ASSAY_PROJECTION_CAPABILITY),
        ObjectIdentity.from_record(ASSAY_EVALUATION_CAPABILITY.capability_key, ASSAY_EVALUATION_CAPABILITY),
        ObjectIdentity.from_record(source.config_id, source),
        ObjectIdentity.from_record(projection.config_id, projection),
        ObjectIdentity.from_record(evaluation.config_id, evaluation),
        EVALUATOR_TASK_ID,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
    )


def _substrate(
    source: ResponseGeometryAssayNativeConfig, carrier: FreshSourceQualificationExperiment
) -> FreshSourceQualificationSubstrateBinding:
    source_id = ObjectIdentity.from_record(source.config_id, source)
    native = ResponseSubstrateBinding(
        f"{PREFIX}.native-binding",
        "response-geometry.six-matrix-q2-medium",
        NativeInteractionKind.INTERACTIVE_EXECUTION,
        ASSAY_SOURCE_CAPABILITY.capability_key,
        ASSAY_SOURCE_CAPABILITY.capability_version,
        ASSAY_SOURCE_CAPABILITY.implementation_sha256,
        ObjectIdentity.from_record(ASSAY_SOURCE_BINDING.binding_id, ASSAY_SOURCE_BINDING),
        ResponseGeometryAssayNativeConfig.SCHEMA,
        ResponseGeometryAssayNativeSegment.SCHEMA,
        None,
        ResponseGeometryAssayNativeSegmentResult.SCHEMA,
        ResponseGeometryAssayViewReport.SCHEMA,
        source_id,
        (
            NativeReceiverContract(
                f"{PREFIX}.receiver",
                f"{PREFIX}.quantity.receiver",
                "hilbert-schmidt-native",
                FRAME,
                "response-geometry.frozen-signed-mode",
                (CLOCK,),
            ),
        ),
        (NativeClockContract(CLOCK, "reference-tick", FRAME),),
        NativeActionContract(f"{PREFIX}.quantity.action", True, True, True, True, "HOLD"),
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return FreshSourceQualificationSubstrateBinding(
        f"{PREFIX}.substrate",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        native,
        source_id,
    )


def _experiment(
    system: SystemSpec,
    carrier: FreshSourceQualificationExperiment,
    exposure: ResponsePanelExposureInspection | MappedResponsePanelExposureInspection,
) -> ExperimentSpec:
    receiver = f"{PREFIX}.quantity.receiver"
    cutoff = InformationCutoff(f"{PREFIX}.design-cutoff", CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    units = tuple(unit.physical_independent_unit_id for unit in carrier.physical_units)
    obligations = ScientificObligations(
        f"{PREFIX}.obligations",
        SupportSpec(
            f"{PREFIX}.support",
            system.relation.relation_id,
            UNIT,
            32,
            2,
            cutoff.cutoff_id,
            (),
            (f"{PREFIX}.denominator",),
            (),
            ObligationStatus.REQUIRED,
        ),
        ValiditySpec(
            f"{PREFIX}.validity",
            (f"{PREFIX}.fixed-roster",),
            ("finite-complete-native-delivery", "simulator-local-denominator"),
            ("NUMERICAL_OR_OBSERVATION_FAILURE",),
            ObligationStatus.REQUIRED,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            f"{PREFIX}.exact-binomial-bonferroni",
            UNIT,
            Decimal("0.95"),
            (receiver,),
            ("FINITE_PANEL_NOT_POPULATION_GUARANTEE",),
            ObligationStatus.REQUIRED,
        ),
        (
            FalsifierSpec(
                f"{PREFIX}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                ASSAY_EVALUATION_CAPABILITY.capability_key,
                "Insufficient positive paired response, preservation or numerical agreement refuses the finite assay.",
                "No extra root, amplitude, refinement or post-response mode choice.",
                ObligationStatus.REQUIRED,
            ),
        ),
        ClosureSpec(
            f"{PREFIX}.closure",
            (f"{PREFIX}.five-parent-three-sign-cells",),
            (f"{PREFIX}.fixed-origin-response",),
            system.relation.history_quantity_ids,
            ObligationStatus.REQUIRED,
        ),
        StructuralConvergenceSpec(
            f"{PREFIX}.convergence",
            ("primary-half-four-of-four-decision-agreement",),
            tuple(view.view_id for view in system.numerical_views[:2]),
            (),
            ObligationStatus.REQUIRED,
        ),
        ComputabilityEvidence(
            f"{PREFIX}.compute-evidence",
            system.computability_envelopes[0].envelope_id,
            tuple(view.view_id for view in system.numerical_views),
            ReadinessStatus.READY,
            (),
            (f"{PREFIX}.bounded-reservation",),
        ),
    )
    claim = ClaimSpec(
        f"{PREFIX}.finite-contact-claim",
        system.world.world_id,
        system.relation.relation_id,
        "The frozen finite X-force assay resolves delayed native response and preservation under paired numerical views.",
        "Per-context and per-landmark direct contact and numerical decision vectors in 32 excluded roots.",
        UNIT,
        EvidenceRung.RESPONSE,
        EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Only the frozen common-assay selector may qualify a finite task; no law qualification/prospective evaluation conclusion.",
        ("frozen-q2-native-medium", "independent-root-only-inference"),
        numerical_view_ids=tuple(view.view_id for view in system.numerical_views[:2]),
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
            "All five parent schedules fork each root at its single assigned landmark; all three signed inner actions share its parent state and absolute-time noise.",
            (f"{PREFIX}.fixed-roster",),
        ),
        (receiver,),
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                ASSAY_EVALUATION_CAPABILITY.capability_key,
                (receiver,),
                "Complete NEG/HOLD/POS, free damped impulse, primary/half views, quarter diagnostic, and fixed restart/covariance subsets.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                f"{PREFIX}.observed-decision-agreement",
                Decimal("0.90"),
                "proportion",
                32,
                "Four of four paired vectors per cell/assay; descriptive simultaneous exact intervals, no top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            exposure.excluded_unit_ids,
            f"{PREFIX}.excluded-cohort",
            sha256(canonical_json_bytes(units)).hexdigest(),
            (f"{PREFIX}.sealed-evaluation",),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.OUTCOME_VISIBLE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def _campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
    *,
    prefix: str = PREFIX,
    budget: ResourceBudget = BUDGET,
    objective: str = "Obtain the excluded finite native-response qualification without constructing a fitted law.",
) -> CampaignSpec:
    return single_experiment_campaign(
        system,
        experiment,
        prefix=prefix,
        budget=budget,
        objective=objective,
        actions=(
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
            ("simulation", AuthorityAction.SIMULATION_EXECUTION),
        ),
    )


def _template(
    source: ResponseGeometryAssayNativeConfig,
    projection: ResponseGeometryAssayProjectionConfig,
    evaluation: ResponseGeometryAssayEvaluationConfig,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    def outputs(
        records: tuple[tuple[str, str], ...], hdf5: tuple[str, str] | None
    ) -> tuple[OutputTemplate, ...]:
        values = [
            OutputTemplate(
                key, schema, ArtifactProfile.CANONICAL_JSON, CANONICAL_MEDIA_TYPE, ".canonical.json"
            )
            for key, schema in records
        ]
        if hdf5 is not None:
            values.append(
                OutputTemplate(*hdf5, ArtifactProfile.AUDITED_HDF5, "application/x-hdf5", ".h5")
            )
        return tuple(sorted(values, key=lambda value: value.output_id))

    definitions = (
        (
            "linked-source-materialization",
            ScientificStage.PREPARE,
            source,
            ASSAY_SOURCE_CAPABILITY,
            (),
            outputs(
                (
                    ("native-result", ResponseGeometryAssayNativeSegmentResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", ASSAY_HDF5_SCHEMA),
            ),
            BarrierKind.NONE,
        ),
        (
            "linked-evidence-projection",
            ScientificStage.TRANSFORM,
            projection,
            ASSAY_PROJECTION_CAPABILITY,
            ("linked-source-materialization",),
            outputs(
                (
                    ("view-report", ResponseGeometryAssayViewReport.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("diagnostics", ASSAY_DIAGNOSTICS_SCHEMA),
            ),
            BarrierKind.FREEZE,
        ),
        (
            EVALUATOR_TASK_ID,
            ScientificStage.EVALUATE,
            evaluation,
            ASSAY_EVALUATION_CAPABILITY,
            ("linked-evidence-projection",),
            outputs(
                (
                    ("evaluation", ResponseGeometryAssayEvaluation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                ),
                None,
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
                    CapabilityConfigRef(
                        config.config_id,
                        config.SCHEMA,
                        manifest.config_schema_sha256,
                        config.fingerprint(),
                        f"config-artifact.{config.config_id}",
                    ),
                    dependencies,
                    emitted,
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    manifest.resource_ceiling,
                    (),
                    barrier,
                    1,
                    (f"{PREFIX}.single-terminal",)
                    if stage is ScientificStage.EVALUATE
                    else (f"{PREFIX}.prototype.{stage.value.lower()}",),
                )
                for key, stage, config, manifest, dependencies, emitted, barrier in definitions
            ),
            key=lambda value: value.step_id,
        )
    )
    protocol = ProtocolTemplate(f"{PREFIX}.protocol", "1.0.0", steps, False, False, True)
    return _study_template(
        protocol,
        source,
        source.config_id,
        experiment,
        registry,
        prefix=PREFIX,
        source_task_id="linked-source-materialization",
        terminal_task_id=EVALUATOR_TASK_ID,
        primary_output_ids={
            ScientificStage.PREPARE: "native-result",
            ScientificStage.TRANSFORM: "view-report",
            ScientificStage.EVALUATE: "evaluation",
        },
    )


def _study_template(
    protocol: ProtocolTemplate,
    source: CanonicalRecord,
    source_id: str,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    *,
    prefix: str,
    source_task_id: str,
    terminal_task_id: str,
    primary_output_ids: dict[ScientificStage, str],
) -> StudyTemplate:
    return study_template_from_protocol(
        protocol,
        source,
        source_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id=source_task_id,
        terminal_task_id=terminal_task_id,
        primary_output_ids=primary_output_ids,
    )


def _entry(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    science: ObjectIdentity,
    *,
    prefix: str = PREFIX,
    minimum_units: int = 32,
    operand_description: str = "Finite assay {domain} operand qualification under the frozen common assay assay.",
    estimator: str = "frozen-assay-estimator",
    uncertainty: str = "exact-binomial-bonferroni",
    ceiling: EvidenceCeiling = EvidenceCeiling.RESPONSE,
    evaluator: CapabilityManifest = ASSAY_EVALUATION_CAPABILITY,
    input_schema: str = ResponseGeometryAssayViewReport.SCHEMA,
    output_schema: str = ResponseGeometryAssayEvaluation.SCHEMA,
    evidence_unit_ids: tuple[str, ...] | None = None,
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    return formal_entry(
        draft,
        context,
        science,
        prefix=prefix,
        minimum_units=minimum_units,
        operand_description=operand_description,
        estimator=estimator,
        uncertainty=uncertainty,
        ceiling=ceiling,
        evaluator=evaluator,
        input_schema=input_schema,
        output_schema=output_schema,
        evidence_unit_ids=evidence_unit_ids,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_response_geometry_assay_authoring(
    *,
    design_packet_sha256: str,
    seed_root_sha256: str,
    scientific_inputs: ResponseGeometryScientificInputs,
    implementation_sha256: str,
    exposure: ResponsePanelExposureInspection | MappedResponsePanelExposureInspection,
) -> ResponseGeometryAssayAuthoringBundle:
    source = ResponseGeometryAssayNativeConfig(
        exposure.source_config.config_id
        if isinstance(exposure, MappedResponsePanelExposureInspection)
        else f"{PREFIX}.source-config",
        design_packet_sha256,
        seed_root_sha256,
        assay_roots(),
        scientific_inputs,
    )
    if exposure.source_config != source or not exposure.units_and_seeds_unexposed:
        raise ValueError("assay authoring requires its exact disjoint exposure inspection")
    projection = ResponseGeometryAssayProjectionConfig(
        f"{PREFIX}.projection-config", ObjectIdentity.from_record(source.config_id, source)
    )
    evaluation = ResponseGeometryAssayEvaluationConfig(
        f"{PREFIX}.evaluation-config", ObjectIdentity.from_record(projection.config_id, projection)
    )
    evidence_registry = build_observation_evidence_world_registry()
    world_profile = next(
        value
        for value in evidence_registry.world_profiles
        if value.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{PREFIX}.evidence-profile",
        draft_id=f"{PREFIX}.draft",
        registry=evidence_registry,
        world_profile_id=world_profile.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(EvidenceRung.MEASUREMENT, EvidenceRung.RESPONSE),
    )
    feasibility = resolve_candidate_profile_feasibility(
        selection=evidence,
        resolver=EvidenceProfileRegistryResolver(
            f"{PREFIX}.profile-resolver", (evidence_registry,)
        ),
    )
    if feasibility.disposition is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION:
        raise ValueError(f'assay measurement/response evidence profile is not compilable: {feasibility.disposition}')
    carrier = _carrier(source, projection, evaluation, evidence)
    association = _substrate(source, carrier)
    payloads: tuple[CanonicalRecord, ...] = tuple(
        sorted(
            (source, projection, evaluation, carrier, association), key=lambda value: value.SCHEMA
        )
    )
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in _BINDINGS
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[value.SCHEMA] for value in payloads)
    system = _system()
    experiment = _experiment(system, carrier, exposure)
    campaign = _campaign(system, experiment)
    source_identity = ObjectIdentity.from_record(source.config_id, source)
    # This receipt qualifies only the exactly decoded numerical configuration;
    # future native trajectories and assay support remain unmeasured.
    qualification = MaterializationQualificationReceipt(
        f"{PREFIX}.config-qualification",
        source.config_id,
        source_identity,
        source.fingerprint(),
        system.world.world_id,
        ObjectIdentity.from_record(projection.config_id, projection),
        tuple(value.view_id for value in system.numerical_views),
        tuple(sorted({value.native_unit for value in system.quantities})),
        (FRAME,),
        (CLOCK,),
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
        source_identity,
        source.fingerprint(),
        source.fingerprint(),
        qualification.observation_operator,
        tuple(value.view_id for value in system.numerical_views),
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    science = ObjectIdentity(
        "response-geometry.frozen-common-assay-design", 'empirical-lawhood/document/markdown', "1.0.0", design_packet_sha256
    )
    design = DesignInputRecord(
        f"{PREFIX}.design-input",
        science,
        design_packet_sha256,
        InformationCutoff(f"{PREFIX}.design-freeze", CLOCK, CausalPhase.PRE_ACTION, Decimal(0)),
        DesignInputRole.MOTIVATION,
        OutcomeAccess.EVALUATION_REVEALED,
        VisibilityCeiling.OUTCOME_VISIBLE,
        "owner.response-geometry",
    )
    amendment_input = (
        DesignInputRecord(
            f"{PREFIX}.amendment-input",
            ObjectIdentity(
                exposure.panel_binding.binding_id.removesuffix(".slot-instance-binding")
                + ".scope-amendment",
                'empirical-lawhood/document/markdown',
                "1.0.0",
                exposure.scope_amendment.content_sha256,
            ),
            exposure.scope_amendment.content_sha256,
            design.information_cutoff,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.response-geometry",
            parent_input_ids=(design.input_id,),
        )
        if isinstance(exposure, MappedResponsePanelExposureInspection)
        else None
    )
    inspection_input = DesignInputRecord(
        f"{PREFIX}.exposure-input",
        ObjectIdentity.from_record(exposure.inspection_id, exposure),
        exposure.fingerprint(),
        design.information_cutoff,
        DesignInputRole.READINESS_METADATA,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.response-geometry",
        exposure.excluded_unit_ids,
        exposure.excluded_seed_ids,
        tuple(sorted((design.input_id, *((amendment_input.input_id,) if amendment_input else ())))),
    )
    design_inputs = tuple(
        sorted(
            (design, inspection_input, *((amendment_input,) if amendment_input else ())),
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
    template = _template(source, projection, evaluation, experiment, registry)
    draft = StudyDraft(
        f"{PREFIX}.draft",
        StudyDraftLifecycle.DRAFT,
        "Qualify the frozen finite native X-response assay in both declared preparation contexts.",
        (f"{PREFIX}.alternative.finite-contact", f"{PREFIX}.alternative.unqualified"),
        DesignOrigin(
            f"{PREFIX}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (inspection_input.input_id,),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        exposure.excluded_unit_ids,
        tuple(value.physical_independent_unit_id for value in carrier.physical_units),
        exposure.excluded_seed_ids,
        exposure.proposed_seed_ids,
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
        BUDGET,
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
    base, standard = _entry(draft, expanded, science)
    proposed = tuple(
        ProposedStudyExtension(
            f"{PREFIX}.extension.{index:02d}",
            f"{PREFIX}.namespace.{index:02d}",
            ObjectIdentity.from_record(_record_id(record), record),
            len(record.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for index, (record, decoder) in enumerate(zip(payloads, decoders, strict=True))
    )
    extensions = ProposedStudyExtensionSet(
        f"{PREFIX}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(value.namespace_id for value in proposed),
        proposed,
    )
    package = ExecutableStudyDefinition(f"{PREFIX}.authoring", base, extensions)
    registrations = tuple(
        value
        for value in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
        if value.manifest.capability_key in selected
    )
    return ResponseGeometryAssayAuthoringBundle(
        package,
        replace(standard, base=context),
        CandidateCapabilityCatalog(f"{PREFIX}.catalog", registrations, (template,)),
        payloads,
        decoders,
        evidence,
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayCandidateContextProvider:
    bundle: ResponseGeometryAssayAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("draft differs from its exact native authoring bundle")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("base native authoring package differs")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )
