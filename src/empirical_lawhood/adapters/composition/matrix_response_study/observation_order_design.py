"""Declarative, outcome-blind authoring root for Six-matrix response GRR observation order."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.matrix_response_study.executable_binding import OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING, OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING
from empirical_lawhood.adapters.methods.matrix_response_study.extension_bundle import OBSERVATION_ORDER_EVALUATION_CAPABILITY, OBSERVATION_ORDER_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.matrix_response_study.history_conditioned_geometry import HistoryViewTrace
from empirical_lawhood.adapters.methods.matrix_response_study.observation_order import MatrixObservationOrderEvaluationConfig, MatrixObservationOrderEvaluation, MatrixObservationOrderProjectionConfig, MatrixObservationOrderViewReport
from empirical_lawhood.adapters.simulators.six_matrix_response.executable_binding import OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING
from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import OBSERVATION_ORDER_HDF5_VALIDATOR, OBSERVATION_ORDER_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order_scientific_inputs import observation_order_scientific_stream_inputs
from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order import MatrixObservationOrderPairedHistoryRequest, MatrixObservationOrderPairedHistoryResult, MatrixObservationOrderPairedHistorySourceConfig, OBSERVATION_ORDER_HDF5_MEDIA_TYPE, OBSERVATION_ORDER_HDF5_SCHEMA, OBSERVATION_ORDER_PRODUCTION_SEED_ROOT, build_observation_order_history_slots
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
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
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
    EvidenceUnitScope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, build_observation_evidence_world_registry, bind_profile_selection
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodBinding,
    FormalMethodCatalog,
    FormalMethodRole,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalDomain,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
    FormalGapSpec,
)
from empirical_lawhood.planning.observation_order import ObservationAcquisitionGroup, ObservationNestedView, ObservationOrderOwnerBinding, ObservationOrderOwnerRole, ObservationPhysicalUnit, ObservationOrderExperimentExtension
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.prospective_config import ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
from empirical_lawhood.planning.source_pipelines import SourcePipelineArtifactProfile, SourcePipelineChangeSemantics, SourcePipelineCoordinate, SourcePipelineMode, SourcePipelineProfile, SourcePipelineTransformEdge
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef, CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.observation_order import ObservationOrderSubstrateBinding
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context_for_template
from empirical_lawhood.runtime.response_experiment_ports import NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

from empirical_lawhood.adapters.simulators.six_matrix_response.history_preparation import FOUR_FAMILY_PREPARATION_IDS

from .contracts import MatrixResponseDesignBinding


_TOKEN = "matrix-observation-order"
_FAMILIES = FOUR_FAMILY_PREPARATION_IDS
_BUDGET = ResourceBudget(8, 8 * 1024**3, 0, 43_200, 16 * 1024**3, 5 * 1024**3)


def _identity(object_id: str, schema: str, purpose: str) -> ObjectIdentity:
    return ObjectIdentity(object_id, schema, "1.0.0", sha256(purpose.encode()).hexdigest())


def _record_id(record: CanonicalRecord) -> str:
    for name in ("extension_set_id", "binding_id", "config_id"):
        value = getattr(record, name, None)
        if isinstance(value, str):
            return value
    raise TypeError("observation order extension record has no stable root ID")


def _seed(purpose: str) -> str:
    # Fixed original numerical inputs; public names do not allocate streams.
    seeds = {'geometry-probe-roster': 'dd996dd7b74e2ce62b2ec8a1c174a7a73b0bc18ac56048968f345f6fcdb16cc7', 'inference-bootstrap': 'fd17161d59ece91d5245d10e84ebf2288a177cb22226620eeebceae0f776b355'}
    if purpose not in seeds:
        raise ValueError("observation-order scientific seed purpose is unsupported")
    return seeds[purpose]


def _configs(binding: MatrixResponseDesignBinding, science: ObjectIdentity) -> tuple[MatrixObservationOrderPairedHistorySourceConfig, MatrixObservationOrderProjectionConfig, MatrixObservationOrderEvaluationConfig]:
    design = binding.source_config
    member = next(
        x for x in design.anisotropic_model.family_members if x.member_id == "six-matrix-response.member.mass-0p5.cross-coupling-1"
    )
    source = MatrixObservationOrderPairedHistorySourceConfig(
        config_id="config.matrix-observation-order.source",
        science_specification=science,
        source_design=ObjectIdentity.from_record(design.config_id, design),
        member=member,
        primary_view=design.primary_view,
        fine_view=design.secondary_view,
        seed_root_hex=OBSERVATION_ORDER_PRODUCTION_SEED_ROOT,
        seed_domain="empirical-lawhood/matrix-observation-order/scientific-stream-input",
        scientific_stream_inputs=observation_order_scientific_stream_inputs(qualification=False),
        slots=build_observation_order_history_slots(),
        target_alpha_tilde_x=Decimal("0.6666666666666666666666666667"),
        target_alpha_tilde_y=Decimal("7.333333333333333333333333334"),
        decreasing_coupling_start=Decimal(8),
        ramp_steps=256,
        primary_total_steps=1024,
        fine_integrator_multiplier=2,
        decoded_hdf5_bytes=9_517_104,
        maximum_hdf5_bytes=16 * 1024**2,
        qualification_fixture_id=None,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    source_id = ObjectIdentity.from_record(source.config_id, source)
    projection = MatrixObservationOrderProjectionConfig(
        "config.matrix-observation-order.projection",
        science,
        source_id,
        member,
        design.primary_view,
        design.secondary_view,
        _FAMILIES,
        tuple(range(0, 1025, 16)),
        "matrix-observation-order.branch.complete-autonomous-history",
        16,
        16,
        12,
        32,
        tuple(range(12)),
        (Decimal("0.25"), Decimal("0.5"), Decimal("1.0")),
        _seed("geometry-probe-roster"),
        '6eb889bd104fb772b64ade4d755d7c76e02094faee4053b3f9b29cde279b99da',
        Decimal("0.35"),
        Decimal("0.95"),
        Decimal("0.30"),
        Decimal("0.25"),
        Decimal("1e-12"),
        Decimal("1e-10"),
        Decimal("1e-10"),
        Decimal("1e-10"),
        Decimal("1e-10"),
        Decimal("0.25"),
        Decimal("0.75"),
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    evaluation = MatrixObservationOrderEvaluationConfig(
        "config.matrix-observation-order.evaluation",
        science,
        ObjectIdentity.from_record(projection.config_id, projection),
        _FAMILIES,
        (384, 512, 640, 768),
        (64, 128, 256),
        (16, 32, 64, 128),
        _seed("inference-bootstrap"),
        4096,
        Decimal("0.95"),
        3687,
        16,
        4,
        4,
        16,
        64,
        256,
        Decimal("0.50"),
        Decimal("0.10"),
        256,
        512,
        None,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    return source, projection, evaluation


def _profiles(
    source: MatrixObservationOrderPairedHistorySourceConfig,
) -> tuple[EvidenceProfileSelection, SourcePipelineProfile]:
    registry = build_observation_evidence_world_registry()
    world = next(
        x
        for x in registry.world_profiles
        if x.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id="selection.matrix-observation-order.observation-order",
        draft_id="draft.matrix-observation-order",
        registry=registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.observation-order",
        requested_rungs=(EvidenceRung.ORDER_RELATION,),
    )
    source_id = ObjectIdentity.from_record(source.config_id, source)
    coordinates = tuple(
        sorted(
            (
                SourcePipelineCoordinate(
                    "coordinate.matrix-observation-order.disposition",
                    "matrix-observation-order.native-disposition",
                    "categorical",
                    "six-matrix-response.frame.simultaneous-unitary-quotient",
                    "clock.matrix-observation-order.primary",
                ),
                SourcePipelineCoordinate(
                    "coordinate.matrix-observation-order.unit",
                    "matrix-observation-order.physical-independent-unit-id",
                    "identity",
                    "six-matrix-response.frame.simultaneous-unitary-quotient",
                    "clock.matrix-observation-order.primary",
                ),
            ),
            key=lambda x: x.coordinate_id,
        )
    )
    validator = ObjectIdentity.from_record(OBSERVATION_ORDER_HDF5_VALIDATOR.registration_id, OBSERVATION_ORDER_HDF5_VALIDATOR)
    selection = CapabilitySelection(
        OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_key, "1.0.0", OBSERVATION_ORDER_SOURCE_CAPABILITY.implementation_sha256
    )
    edge = SourcePipelineTransformEdge(
        "edge.matrix-observation-order.source-binding",
        0,
        "registry.matrix-observation-order.source-binding",
        selection,
        source_id,
        source.SCHEMA,
        source.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        "application/vnd.empirical-lawhood.canonical+json",
        "canonical-json",
        "canonical-json",
        coordinates,
        coordinates,
        SourcePipelineChangeSemantics.PRESERVED,
        None,
        SourcePipelineChangeSemantics.PRESERVED,
        None,
        SourcePipelineChangeSemantics.PRESERVED,
        None,
        SourcePipelineChangeSemantics.PRESERVED,
        None,
        True,
        True,
        validator,
        ResourceBudget(
            1, 64 * 1024**2, 0, 60, len(source.canonical_bytes()), len(source.canonical_bytes())
        ),
    )
    profile = SourcePipelineProfile(
        "source-pipeline.matrix-observation-order",
        "1.0.0",
        SourcePipelineMode.SIMULATED,
        selection,
        source_id,
        "source.matrix-observation-order.config",
        source_id,
        source.fingerprint(),
        len(source.canonical_bytes()),
        "repository-internal",
        "external-root.operator-external-root",
        "source/matrix-observation-order-source-config.json",
        source.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        "canonical-json",
        coordinates,
        (edge,),
        source.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        ".canonical.json",
        SourcePipelineArtifactProfile.CANONICAL_JSON,
        coordinates,
        (validator,),
        "matrix-observation-order.physical-independent-unit-id",
        "matrix-observation-order.native-disposition",
        _identity(
            "cutoff.matrix-observation-order.source-freeze", InformationCutoff.SCHEMA, source.science_specification.object_fingerprint
        ),
        _identity(
            "manifest.matrix-observation-order.source-binding",
            'empirical-lawhood/planning/dataset-transformation-manifest',
            source.fingerprint(),
        ),
        ResourceBudget(
            1, 64 * 1024**2, 0, 60, len(source.canonical_bytes()), len(source.canonical_bytes())
        ),
        256,
        256,
        "scratch/matrix-observation-order",
        "derived/matrix-observation-order/source-config.json",
        "world.matrix-observation-order.six-matrix",
        _identity(
            "observation.matrix-observation-order.native-history",
            'empirical-lawhood/planning/observation-operator',
            "paired-native-history",
        ),
        tuple(sorted((source.primary_view.view_id, source.fine_view.view_id))),
        "relation.matrix-observation-order.history-geometry",
        "validity.matrix-observation-order",
        "uncertainty.matrix-observation-order",
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    return evidence, profile


def _terminal_matrix() -> ProspectiveTerminalMatrix:
    names = (
        "PLATFORM_OBSTRUCTION",
        "STOPPED",
        "OPERATIONALLY_UNEVALUABLE",
        "NUMERICAL_UNEVALUABLE",
        "STRATUM_UNEVALUABLE",
        "CONFIRMED",
        "MIXED_VIEWS",
        "NOT_CONFIRMED",
    )
    return ProspectiveTerminalMatrix(
        "terminal-matrix.matrix-observation-order",
        (ProspectiveSelectorTransition(1, "COMPLETE_PAIRED_ROSTER", "SEALED_EVALUATION"),),
        tuple(
            ProspectiveTerminalRule(i, name, name, "FINAL_OBSERVATION_ORDER_RECORD", "NO_FOLLOW_UP")
            for i, name in enumerate(names, 1)
        ),
        True,
        True,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )


def _carrier(
    source: MatrixObservationOrderPairedHistorySourceConfig,
    projection: MatrixObservationOrderProjectionConfig,
    evaluation: MatrixObservationOrderEvaluationConfig,
    evidence: EvidenceProfileSelection,
    profile: SourcePipelineProfile,
) -> ObservationOrderExperimentExtension:
    units = tuple(
        ObservationPhysicalUnit(
            x.physical_independent_unit_id,
            f"coordinate.{x.family_id}.{x.history_index:03d}",
            x.preparation_instance_id,
            x.family_id,
            sha256(
                f"{source.fingerprint()}\0{x.family_id}\0{x.history_index}".encode()
            ).hexdigest(),
            f"seed.matrix-observation-order.h{x.history_index:03d}",
        )
        for x in source.slots
    )
    groups = tuple(
        ObservationAcquisitionGroup(
            x.acquisition_group_id,
            x.physical_independent_unit_id,
            x.preparation_instance_id,
            tuple(sorted((x.primary_view_id, x.fine_view_id))),
        )
        for x in source.slots
    )
    views = tuple(
        sorted(
            (
                ObservationNestedView(
                    view_id,
                    x.acquisition_group_id,
                    x.physical_independent_unit_id,
                    member_id,
                    ("receiver.matrix-observation-order.geometry",),
                    (clock_id,),
                    ("support.matrix-observation-order.frozen-risk-sets",),
                )
                for x in source.slots
                for view_id, member_id, clock_id in (
                    (x.primary_view_id, source.primary_view.view_id, "clock.matrix-observation-order.primary"),
                    (x.fine_view_id, source.fine_view.view_id, "clock.matrix-observation-order.fine"),
                )
            ),
            key=lambda x: x.view_id,
        )
    )
    identities = {
        ObservationOrderOwnerRole.SOURCE: (OBSERVATION_ORDER_SOURCE_CAPABILITY, source),
        ObservationOrderOwnerRole.EVIDENCE_PROJECTION: (OBSERVATION_ORDER_PROJECTION_CAPABILITY, projection),
        ObservationOrderOwnerRole.METHOD: (OBSERVATION_ORDER_EVALUATION_CAPABILITY, projection),
        ObservationOrderOwnerRole.SEALED_EVALUATOR: (OBSERVATION_ORDER_EVALUATION_CAPABILITY, evaluation),
    }
    owners = tuple(
        ObservationOrderOwnerBinding(
            role,
            ObjectIdentity.from_record(cap.capability_key, cap),
            ObjectIdentity.from_record(_record_id(config), config),
        )
        for role, (cap, config) in sorted(identities.items(), key=lambda x: x[0].value)
    )
    return ObservationOrderExperimentExtension(
        "extension-set.matrix-observation-order",
        _TOKEN,
        source.science_specification,
        ObjectIdentity.from_record(evidence.selection_id, evidence),
        ObjectIdentity.from_record(profile.profile_id, profile),
        units,
        groups,
        views,
        _FAMILIES,
        tuple(sorted((source.primary_view.view_id, source.fine_view.view_id))),
        ("receiver.matrix-observation-order.geometry",),
        ("clock.matrix-observation-order.fine", "clock.matrix-observation-order.primary"),
        ("support.matrix-observation-order.frozen-risk-sets",),
        tuple(f"cutoff.matrix-observation-order.step-{x}" for x in (384, 512, 640, 768)),
        "matrix-observation-order.acquire",
        "matrix-observation-order.project",
        "matrix-observation-order.evaluate",
        owners,
        _terminal_matrix(),
        EvidenceCeiling.ORDER_RELATION,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )


def _substrate(
    source: MatrixObservationOrderPairedHistorySourceConfig, carrier: ObservationOrderExperimentExtension
) -> ObservationOrderSubstrateBinding:
    binding_identity = ObjectIdentity.from_record(
        OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING.binding_id, OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING
    )
    source_id = ObjectIdentity.from_record(source.config_id, source)
    clocks = (
        NativeClockContract(
            "clock.matrix-observation-order.fine",
            "dimensionless-langevin-time",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
        ),
        NativeClockContract(
            "clock.matrix-observation-order.primary",
            "dimensionless-langevin-time",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
        ),
    )
    native = ResponseSubstrateBinding(
        "substrate.matrix-observation-order",
        "medium.six-matrix-response.six-matrix-constitutive",
        NativeInteractionKind.READ_ONLY_ACQUISITION,
        OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_key,
        OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_version,
        OBSERVATION_ORDER_SOURCE_CAPABILITY.implementation_sha256,
        binding_identity,
        SourcePipelineProfile.SCHEMA,
        MatrixObservationOrderPairedHistoryRequest.SCHEMA,
        OBSERVATION_ORDER_HDF5_SCHEMA,
        None,
        HistoryViewTrace.SCHEMA,
        source_id,
        (
            NativeReceiverContract(
                "receiver.matrix-observation-order.geometry",
                "quantity.matrix-observation-order.geometry-state",
                "categorical-four-state",
                "six-matrix-response.frame.simultaneous-unitary-quotient",
                "direction.matrix-observation-order.forward-history",
                tuple(x.clock_id for x in clocks),
            ),
        ),
        clocks,
        None,
        False,
        False,
        False,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return ObservationOrderSubstrateBinding(
        "observation-substrate.matrix-observation-order",
        ObjectIdentity.from_record(carrier.extension_set_id, carrier),
        native,
        source_id,
    )


def _system() -> SystemSpec:
    clock_ids = ("clock.matrix-observation-order.fine", "clock.matrix-observation-order.primary")
    clocks = tuple(
        ClockSpec(
            x,
            x.rsplit(".", 1)[-1].title(),
            "dimensionless-langevin-time",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
            SamplingSemantics.REGULAR,
            HoldSemantics.NONE,
            ClockLabelSemantics.INSTANT,
            Decimal("0.0005") if x.endswith("fine") else Decimal("0.001"),
            Decimal(0),
        )
        for x in clock_ids
    )
    pre = AvailabilitySpec(
        clock_ids[1], CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    recv = AvailabilitySpec(
        clock_ids[1], CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED, Decimal("1.024")
    )
    quantities = (
        QuantitySpec(
            "quantity.matrix-observation-order.denominator",
            "Six-matrix q=2 denominator",
            QuantityKind.DENOMINATOR,
            "six-matrix-denominator",
            "identity",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
            clock_ids[1],
            pre,
        ),
        QuantitySpec(
            "quantity.matrix-observation-order.geometry-state",
            "Observed geometry state",
            QuantityKind.RECEIVER,
            "geometry-state",
            "categorical-four-state",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
            clock_ids[1],
            recv,
            ResponseDirection.TARGET_BAND,
        ),
        QuantitySpec(
            "quantity.matrix-observation-order.preparation-history",
            "Complete preparation history",
            QuantityKind.HISTORY,
            "matrix-history",
            "dimensionless-native-history",
            "six-matrix-response.frame.simultaneous-unitary-quotient",
            clock_ids[1],
            pre,
        ),
    )
    relation = RelationalIdentity(
        "relation.matrix-observation-order.history-geometry",
        (quantities[0].quantity_id,),
        (quantities[2].quantity_id,),
        False,
        (),
        (quantities[1].quantity_id,),
        HorizonSpec(
            "horizon.matrix-observation-order.complete",
            clock_ids[1],
            Decimal("1.024"),
            "dimensionless-langevin-time",
        ),
    )
    world = WorldSpec(
        "world.matrix-observation-order.six-matrix",
        "Six-matrix numerical simulator",
        WorldKind.NUMERICAL_SIMULATOR,
        ("BAOAB-under-damped-Langevin", "six-Hermitian-matrix-constitutive-dynamics"),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.ORDER_RELATION,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    envelope = ComputabilityEnvelope(
        "computability.matrix-observation-order",
        ("paired-complex128-BAOAB",),
        (),
        ("dense-receiver-geometry", "paired-same-driver-views"),
        ("dense-receiver-geometry", "paired-same-driver-views"),
        8,
        8 * 1024**3,
        0,
        43_200,
        5 * 1024**3,
        Decimal(43_200),
        Decimal(43_200),
    )
    views = tuple(
        NumericalViewSpec(
            view_id,
            world.world_id,
            "unit.matrix-observation-order.preparation-history",
            "equations.six-matrix-response.six-matrix-baoab",
            ("closure.six-matrix-response.six-matrix-native",),
            ("boundary.six-matrix-response.complete-history",),
            (
                NumericalCoordinateSpec(
                    f"coordinate.{view_id}.timestep",
                    NumericalCoordinateKind.TIMESTEP,
                    dt,
                    "dimensionless-langevin-time",
                    level,
                ),
            ),
            "solver.six-matrix-response.baoab",
            "1.0.0",
            "complex128",
            "cpu",
            "runtime.matrix-observation-order",
            RandomnessSemantics.GENERATIVE_PREPARATION,
            "observation.matrix-observation-order.native-geometry",
            envelope.envelope_id,
        )
        for view_id, dt, level in (
            ("six-matrix-response.view.baoab-dt-0p0005", Decimal("0.0005"), 1),
            ("six-matrix-response.view.baoab-dt-0p001", Decimal("0.001"), 0),
        )
    )
    policy = AuthorityPolicy(
        "authority-policy.matrix-observation-order",
        "owner.matrix-observation-order",
        "operator.matrix-observation-order",
        (_TOKEN,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset(
            (
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            )
        ),
        frozenset((SourceAccessClass.NONE,)),
        ("gate.matrix-observation-order.readiness", "gate.matrix-observation-order.typed-authorities"),
        frozenset(),
        _BUDGET,
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        "system.matrix-observation-order",
        "Six-matrix response history-conditioned observation system",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        clocks,
        quantities,
        IndependentUnitSpec(
            "unit.matrix-observation-order.preparation-history",
            "One fresh complete preparation history",
            "matrix-observation-order-history-id",
        ),
        policy,
        (
            PortSpec(
                "port.matrix-observation-order.geometry",
                quantities[1].quantity_id,
                clock_ids[1],
                PortDirection.OUTPUT,
                BalanceRole.OBSERVATION,
            ),
        ),
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


def _experiment(system: SystemSpec, carrier: ObservationOrderExperimentExtension) -> ExperimentSpec:
    relation = system.relation
    units = tuple(x.physical_independent_unit_id for x in carrier.physical_units)
    cutoffs = tuple(
        InformationCutoff(
            x,
            "clock.matrix-observation-order.primary",
            CausalPhase.RECEIVER,
            Decimal(x.rsplit("-", 1)[-1]) * Decimal("0.001"),
        )
        for x in carrier.causal_cutoff_ids
    )
    validity = ValiditySpec(
        "validity.matrix-observation-order",
        ("domain.matrix-observation-order.frozen-roster",),
        ("complete-fresh-history", "native-simulator-only"),
        ("NUMERICAL_OR_OBSERVATION_INVALID",),
        ObligationStatus.REQUIRED,
    )
    uncertainty = UncertaintySpec(
        "uncertainty.matrix-observation-order",
        "method.matrix-observation-order.family-stratified-history-bootstrap",
        system.independent_unit.unit_id,
        Decimal("0.95"),
        ("quantity.matrix-observation-order.geometry-state",),
        ("SIMULATOR_LOCAL_ONLY",),
        ObligationStatus.REQUIRED,
    )
    obligations = ScientificObligations(
        "obligations.matrix-observation-order",
        SupportSpec(
            "support.matrix-observation-order",
            relation.relation_id,
            system.independent_unit.unit_id,
            256,
            2,
            cutoffs[0].cutoff_id,
            (),
            ("denominator.matrix-observation-order.dimension-two",),
            (),
            ObligationStatus.REQUIRED,
        ),
        validity,
        uncertainty,
        (
            FalsifierSpec(
                "falsifier.matrix-observation-order.view-or-support-failure",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_key,
                "A failed view or unsupported risk stratum prevents confirmation.",
                "No cross-view compensation and no support relaxation.",
                ObligationStatus.REQUIRED,
            ),
        ),
        ClosureSpec(
            "closure.matrix-observation-order.history",
            ("cell.matrix-observation-order.four-geometry-states",),
            ("factor.matrix-observation-order.full-intersection", "factor.matrix-observation-order.robust-00"),
            relation.history_quantity_ids,
            ObligationStatus.REQUIRED,
        ),
        StructuralConvergenceSpec(
            "convergence.matrix-observation-order.primary-fine",
            ("paired-view-qualitative-agreement",),
            tuple(x.view_id for x in system.numerical_views),
            (),
            ObligationStatus.REQUIRED,
        ),
        ComputabilityEvidence(
            "computability-evidence.matrix-observation-order",
            system.computability_envelopes[0].envelope_id,
            tuple(x.view_id for x in system.numerical_views),
            ReadinessStatus.READY,
            (),
            ("science-freeze.matrix-observation-order",),
        ),
    )
    claim = ClaimSpec(
        "claim.matrix-observation-order.retention-and-reentry",
        system.world.world_id,
        relation.relation_id,
        "Fresh complete histories show the predeclared retention and history-qualified re-entry pattern independently in both numerical views.",
        "Separate primary and fine retention and re-entry probabilities under frozen family-stratified support.",
        system.independent_unit.unit_id,
        EvidenceRung.ORDER_RELATION,
        EvidenceCeiling.ORDER_RELATION,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "CONFIRMED only under both frozen predicates in both views; every other terminal is final.",
        ("fresh-complete-preparation-history", "frozen-native-simulator-denominator"),
        numerical_view_ids=tuple(x.view_id for x in system.numerical_views),
    )
    return ExperimentSpec(
        _TOKEN,
        system.system_id,
        system.world.world_id,
        relation,
        system.independent_unit.unit_id,
        (claim,),
        AssignmentSpec(
            "assignment.matrix-observation-order.observational",
            AssignmentKind.OBSERVATIONAL,
            system.independent_unit.unit_id,
            (),
            "Observe a frozen balanced roster of independently seeded simulator preparations; no action is requested or applied.",
            ("support.matrix-observation-order.frozen-roster",),
        ),
        ("quantity.matrix-observation-order.geometry-state",),
        (
            ControlSpec(
                "control.matrix-observation-order.paired-view",
                ControlKind.BASELINE_COMPARATOR,
                OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_key,
                ("quantity.matrix-observation-order.geometry-state",),
                "Primary and fine views are reported separately and cannot compensate.",
            ),
        ),
        (
            PrecisionGoal(
                "precision.matrix-observation-order.fixed-roster",
                "metric.matrix-observation-order.clustered-interval-width",
                Decimal("0.05"),
                "probability",
                256,
                "Run the fixed 256-history roster once; report achieved width without top-up.",
            ),
        ),
        cutoffs,
        RevealBarrierSpec(
            "reveal.matrix-observation-order",
            ("unit.matrix-observation-order.synthetic-qualification",),
            "cohort.matrix-observation-order.confirmation",
            sha256(canonical_json_bytes(units)).hexdigest(),
            ("artifact.matrix-observation-order.sealed-evaluation",),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.PROSPECTIVE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def _campaign(system: SystemSpec, experiment: ExperimentSpec) -> CampaignSpec:
    node = CampaignNode(
        "node.matrix-observation-order.experiment",
        CampaignNodeKind.EXPERIMENT_SPEC,
        CampaignLane.PROSPECTIVE,
        ObjectIdentity.from_record(experiment.experiment_id, experiment),
        (),
        LifecycleStatus.ACTIVE,
    )
    rights = tuple(
        DecisionRight(
            f"decision-right.matrix-observation-order.{suffix}",
            action,
            system.authority_policy.delegate_id,
            system.authority_policy.policy_id,
            True,
        )
        for suffix, action in (
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
            ("simulation", AuthorityAction.SIMULATION_EXECUTION),
        )
    )
    return CampaignSpec(
        "campaign.matrix-observation-order",
        "Confirm history-conditioned geometric retention and re-entry under fresh complete preparation histories.",
        (system.system_id,),
        (system.world.world_id,),
        tuple(x.claim_id for x in experiment.claims),
        _BUDGET,
        ObjectIdentity.from_record(system.authority_policy.policy_id, system.authority_policy),
        rights,
        (node,),
        (node.node_id,),
        (node.node_id,),
        (),
        LifecycleStatus.ACTIVE,
    )


def _output(output_id: str, schema: str, *, hdf5: bool = False) -> OutputTemplate:
    return OutputTemplate(
        output_id,
        schema,
        ArtifactProfile.AUDITED_HDF5 if hdf5 else ArtifactProfile.CANONICAL_JSON,
        OBSERVATION_ORDER_HDF5_MEDIA_TYPE if hdf5 else "application/vnd.empirical-lawhood.canonical+json",
        ".h5" if hdf5 else ".canonical.json",
    )


def _config_ref(
    record: CanonicalRecord, binding: ExecutableCapabilityBinding
) -> CapabilityConfigRef:
    record_id = _record_id(record)
    return CapabilityConfigRef(
        record_id,
        record.SCHEMA,
        sha256(record.SCHEMA.encode()).hexdigest(),
        record.fingerprint(),
        f"config-artifact.{record_id}",
    )


def _template(
    source: MatrixObservationOrderPairedHistorySourceConfig,
    projection: MatrixObservationOrderProjectionConfig,
    evaluation: MatrixObservationOrderEvaluationConfig,
    experiment: ExperimentSpec,
) -> StudyTemplate:
    source_step = ProtocolStepTemplate(
        "linked-source-materialization",
        ScientificStage.PREPARE,
        OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_key,
        "1.0.0",
        _config_ref(source, OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING),
        (),
        tuple(
            sorted(
                (
                    _output("paired-history-result", MatrixObservationOrderPairedHistoryResult.SCHEMA),
                    _output("paired-native-history", OBSERVATION_ORDER_HDF5_SCHEMA, hdf5=True),
                    _output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                key=lambda x: x.output_id,
            )
        ),
        OBSERVATION_ORDER_SOURCE_CAPABILITY.permissions,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        ResourceBudget(1, 2 * 1024**3, 0, 80, 0, 16 * 1024**2),
        (),
        BarrierKind.NONE,
        1,
        ("matrix-observation-order-paired-history-custody",),
    )
    projection_step = ProtocolStepTemplate(
        "linked-evidence-projection",
        ScientificStage.TRANSFORM,
        OBSERVATION_ORDER_PROJECTION_CAPABILITY.capability_key,
        "1.0.0",
        _config_ref(projection, OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING),
        (source_step.step_id,),
        (
            _output("geometry-trace", HistoryViewTrace.SCHEMA),
            _output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
        ),
        OBSERVATION_ORDER_PROJECTION_CAPABILITY.permissions,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        ResourceBudget(1, 2 * 1024**3, 0, 10, 16 * 1024**2, 512 * 1024),
        (),
        BarrierKind.FREEZE,
        1,
        ("matrix-observation-order-dense-geometry-projection",),
    )
    evaluation_step = ProtocolStepTemplate(
        "matrix-observation-order.evaluate",
        ScientificStage.EVALUATE,
        OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_key,
        "1.0.0",
        _config_ref(evaluation, OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING),
        (projection_step.step_id,),
        tuple(
            sorted(
                (
                    _output("evaluation", MatrixObservationOrderEvaluation.SCHEMA),
                    _output("fine-report", MatrixObservationOrderViewReport.SCHEMA),
                    _output("primary-report", MatrixObservationOrderViewReport.SCHEMA),
                    _output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                    _output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                key=lambda x: x.output_id,
            )
        ),
        OBSERVATION_ORDER_EVALUATION_CAPABILITY.permissions,
        OutcomeAccess.EVALUATION_REVEALED,
        VisibilityCeiling.PROSPECTIVE,
        ResourceBudget(1, 8 * 1024**3, 0, 17_600, 5 * 1024**3, 32 * 1024**2),
        (),
        BarrierKind.REVEAL,
        1,
        ("matrix-observation-order-single-terminal",),
    )
    steps = tuple(sorted((source_step, projection_step, evaluation_step), key=lambda x: x.step_id))
    protocol = ProtocolTemplate("protocol.matrix-observation-order", "1.0.0", steps, False, False, True)
    registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    nodes = tuple(
        CandidateGraphNode(
            x.step_id,
            x.stage,
            x.capability_key,
            x.capability_version,
            registry.resolve(x.capability_key, x.capability_version).implementation_sha256,
            x.fingerprint(),
            x.obligation_ids,
            x.requested_outcome_access,
            x.visibility_ceiling,
            x.resource_budget,
        )
        for x in steps
    )
    external = CandidateGraphExternalInput(
        "source.matrix-observation-order.config",
        ScientificInputRole.MODEL,
        "materialization.matrix-observation-order.source-config",
        ContentIdentityPolicy.EXACT_SHA256,
        source.fingerprint(),
        source.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        len(source.canonical_bytes()),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    edges: list[CandidateGraphEdge] = [
        CandidateGraphEdge(
            "edge.external.matrix-observation-order.source-config",
            None,
            None,
            external.input_id,
            source_step.step_id,
            "source-config",
            ScientificInputRole.MODEL,
            external.logical_artifact_id,
            source.SCHEMA,
            external.media_type,
            external.maximum_size_bytes,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            BarrierKind.NONE,
        )
    ]
    for parent, child in ((source_step, projection_step), (projection_step, evaluation_step)):
        manifest = registry.resolve(child.capability_key, child.capability_version)
        for output in parent.outputs:
            if output.payload_schema in manifest.input_schema_ids:
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{parent.step_id}.{output.output_id}.{child.step_id}",
                        parent.step_id,
                        output.output_id,
                        None,
                        child.step_id,
                        f"input-{parent.step_id}-{output.output_id}",
                        ScientificInputRole.OUTCOME,
                        f"artifact.{parent.step_id}.{output.output_id}",
                        output.payload_schema,
                        output.media_type,
                        parent.resource_budget.output_bytes,
                        child.requested_outcome_access,
                        child.visibility_ceiling,
                        child.barrier,
                    )
                )
    graph = CandidateScientificGraph(
        "graph.matrix-observation-order",
        (external,),
        tuple(sorted(nodes, key=lambda x: x.node_id)),
        tuple(sorted(edges, key=lambda x: x.edge_id)),
    )
    owners = {o: (s.step_id, s.outputs[0].output_id) for s in steps for o in s.obligation_ids}
    bindings = tuple(
        ObligationCoverageBinding(
            o,
            *owners.get(o, (evaluation_step.step_id, "evaluation")),
            tuple(
                sorted(
                    x.edge_id
                    for x in graph.edges
                    if x.consumer_node_id
                    == owners.get(o, (evaluation_step.step_id, "evaluation"))[0]
                )
            ),
        )
        for o in required_candidate_obligation_ids(experiment, protocol)
    )
    return StudyTemplate(
        "template.matrix-observation-order",
        "1.0.0",
        protocol,
        graph,
        ObligationCoverage(
            "coverage.matrix-observation-order", tuple(sorted(bindings, key=lambda x: x.obligation_id))
        ),
    )


def _entry(
    draft: StudyDraft, context: CandidateCompilationContext, science: ObjectIdentity
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    assert draft.system is not None and draft.experiment is not None
    world = FormalGapEvidenceWorld.NUMERICAL_SIMULATOR
    active = {FormalDomain.DYNAMICS, FormalDomain.GEOMETRY}
    gaps = tuple(
        FormalGapSpec(
            f"gap.matrix-observation-order.{x.value.lower()}",
            x,
            f"order relation observation-order {x.value.lower()} applicability.",
            ("operand.matrix-observation-order.geometry-traces",),
            (world,),
            16,
            2,
            ("estimator.matrix-observation-order.interval-and-transition",),
            (draft.experiment.controls[0].control_id,),
            ("falsifier.matrix-observation-order.view-or-support-failure",),
            ("support.matrix-observation-order.frozen-risk-sets",),
            "multiplicity.matrix-observation-order.family-stratified",
            EvidenceCeiling.ORDER_RELATION,
            (science.object_id,),
        )
        for x in FormalDomain
    )
    register = FormalGapRegister(
        "formal-gap-register.matrix-observation-order",
        "1.0.0",
        (science,),
        tuple(sorted(gaps, key=lambda x: x.gap_id)),
    )
    inactive_ids = tuple(sorted(x.gap_id for x in gaps if x.domain not in active))
    inventory = FormalGapSourceCapabilityInventory(
        "formal-source-inventory.matrix-observation-order",
        draft.system.system_id,
        world,
        tuple(x.materialization for x in draft.source_materializations),
        ("operand.matrix-observation-order.geometry-traces",),
        ("support.matrix-observation-order.frozen-risk-sets",),
        draft.evaluation_unit_ids,
        EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        tuple(x.view_id for x in draft.system.numerical_views),
        ("estimator.matrix-observation-order.interval-and-transition",),
        (draft.experiment.controls[0].control_id,),
        ("multiplicity.matrix-observation-order.family-stratified",),
        inactive_ids,
        (),
        EvidenceCeiling.ORDER_RELATION,
        OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    template = context.template(draft.dag_template_key)
    assert template is not None
    terminal = next(
        x for x in template.coverage.bindings if x.obligation_id == "matrix-observation-order-single-terminal"
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            x.gap_id,
            FormalGapCoverageDisposition.TEST_IN_THIS_ACT
            if x.domain in active
            else FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
            None,
            () if x.domain in active else ("ORDER_RELATION_OBSERVATION_DENOMINATOR_ONLY",),
            "estimator.matrix-observation-order.interval-and-transition" if x.domain in active else None,
            (draft.experiment.controls[0].control_id,) if x.domain in active else (),
            "multiplicity.matrix-observation-order.family-stratified" if x.domain in active else None,
            (terminal.obligation_id,) if x.domain in active else (),
            (terminal.required_output_id,) if x.domain in active else (),
            (terminal.proof_owner_node_id,) if x.domain in active else (),
        )
        for x in gaps
    )
    coverage = FormalGapCoverage(
        "formal-gap-coverage.matrix-observation-order",
        ObjectIdentity.from_record(register.register_id, register),
        draft.system.system_id,
        draft.draft_id,
        applicability,
        tuple(sorted(assignments, key=lambda x: x.gap_id)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    requirement_bindings = tuple(
        ExperimentEntryRequirementBinding(
            x,
            (f"binding.matrix-observation-order.{x.value.lower()}",),
            ReadinessStatus.AUTHORITY_REQUIRED
            if x is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else ReadinessStatus.READY,
            ("EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
            if x is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else (),
        )
        for x in sorted(ExperimentEntryRequirement, key=lambda x: x.value)
    )
    checklist = ExperimentEntryChecklist(
        "entry-checklist.matrix-observation-order",
        ObjectIdentity.from_record(draft.draft_id, draft),
        ObjectIdentity.from_record(register.register_id, register),
        ObjectIdentity.from_record(coverage.coverage_id, coverage),
        world,
        "route.matrix-observation-order.issued-compilation",
        "durability.matrix-observation-order.receipt-first",
        requirement_bindings,
        tuple(sorted(ExperimentEntryTransition, key=lambda x: x.value)),
        tuple(sorted(ExperimentTerminalClass, key=lambda x: x.value)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    entry = ExperimentEntryPackage("entry-package.matrix-observation-order", register, coverage, checklist)
    authoring = StudyDefinition("programme-authoring-package.matrix-observation-order", draft, entry)
    gap_ids = tuple(sorted(x.gap_id for x in gaps if x.domain in active))
    method = context.registry.resolve(OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_key, "1.0.0")
    methods = tuple(
        FormalMethodBinding(
            name,
            role,
            method.capability_key,
            method.capability_version,
            method.implementation_sha256,
            gap_ids,
            HistoryViewTrace.SCHEMA,
            MatrixObservationOrderEvaluation.SCHEMA,
            EvidenceCeiling.ORDER_RELATION,
            OutcomeAccess.EVALUATOR_REVEAL,
        )
        for role, name in (
            (FormalMethodRole.ESTIMATOR, "estimator.matrix-observation-order.interval-and-transition"),
            (FormalMethodRole.MULTIPLICITY, "multiplicity.matrix-observation-order.family-stratified"),
        )
    )
    standard = StandardCandidateCompilationContext(
        "standard-context.matrix-observation-order",
        context,
        FormalMethodCatalog(
            "formal-method-catalog.matrix-observation-order",
            tuple(sorted(methods, key=lambda x: x.binding_id)),
        ),
        (inventory,),
    )
    return authoring, standard


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderObservationAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection
    source_profile: SourcePipelineProfile


def build_observation_order_observation_authoring(
    binding: MatrixResponseDesignBinding, science: ObjectIdentity
) -> MatrixObservationOrderObservationAuthoringBundle:
    source, projection, evaluation = _configs(binding, science)
    evidence, source_profile = _profiles(source)
    carrier = _carrier(source, projection, evaluation, evidence, source_profile)
    association = _substrate(source, carrier)
    payloads: tuple[CanonicalRecord, ...] = tuple(
        sorted((carrier, association, source, projection, evaluation), key=lambda x: x.SCHEMA)
    )
    decoder_map = {
        x.payload_schema: x
        for binding in (
            OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING,
            OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING,
            OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING,
        )
        for x in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_map[x.SCHEMA] for x in payloads)
    system = _system()
    experiment = _experiment(system, carrier)
    campaign = _campaign(system, experiment)
    source_identity = ObjectIdentity.from_record(source.config_id, source)
    qualification = MaterializationQualificationReceipt(
        "qualification.matrix-observation-order.source-config",
        "source.matrix-observation-order.config",
        source_identity,
        source.fingerprint(),
        system.world.world_id,
        source_profile.observation_operator,
        tuple(x.view_id for x in system.numerical_views),
        tuple(sorted({x.native_unit for x in system.quantities})),
        tuple(sorted({x.coordinate_frame for x in system.quantities})),
        tuple(x.clock_id for x in system.clocks),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        "source.matrix-observation-order.config",
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        source_identity,
        source.fingerprint(),
        source.fingerprint(),
        source_profile.observation_operator,
        tuple(x.view_id for x in system.numerical_views),
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design_input = DesignInputRecord(
        "input.matrix-observation-order.science-freeze",
        source.science_specification,
        source.science_specification.object_fingerprint,
        InformationCutoff(
            "cutoff.matrix-observation-order.design-freeze",
            "clock.matrix-observation-order.primary",
            CausalPhase.PRE_ACTION,
            Decimal(0),
        ),
        DesignInputRole.READINESS_METADATA,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.matrix-observation-order-science-review",
    )
    template = _template(source, projection, evaluation, experiment)
    selections = tuple(
        CapabilitySelection(x.capability_key, x.capability_version, x.implementation_sha256)
        for x in (OBSERVATION_ORDER_EVALUATION_CAPABILITY, OBSERVATION_ORDER_PROJECTION_CAPABILITY, OBSERVATION_ORDER_SOURCE_CAPABILITY)
    )
    draft = StudyDraft(
        "draft.matrix-observation-order",
        StudyDraftLifecycle.DRAFT,
        "Confirm history-conditioned geometric retention and re-entry under fresh complete preparation histories.",
        ("alternative.matrix-observation-order.confirmed", "alternative.matrix-observation-order.not-confirmed-or-unevaluable"),
        DesignOrigin(
            "origin.matrix-observation-order.ordinary-predeclared",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (design_input.input_id,),
            VisibilityCeiling.PROSPECTIVE,
        ),
        (design_input,),
        ("unit.matrix-observation-order.synthetic-qualification",),
        tuple(x.physical_independent_unit_id for x in carrier.physical_units),
        ("seed.matrix-observation-order.synthetic-qualification",),
        tuple(f"seed.matrix-observation-order.h{x.history_index:03d}" for x in source.slots),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(sorted(selections, key=lambda x: x.selection_id)),
        (source_ref,),
        _BUDGET,
    )
    installed_registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    selected_keys = {
        x.capability_key
        for x in (OBSERVATION_ORDER_SOURCE_CAPABILITY, OBSERVATION_ORDER_PROJECTION_CAPABILITY, OBSERVATION_ORDER_EVALUATION_CAPABILITY)
    }
    registry = CapabilityRegistry(
        "registry.matrix-observation-order",
        tuple(x for x in installed_registry.capabilities if x.capability_key in selected_keys),
    )
    context = CandidateCompilationContext(
        "context.matrix-observation-order",
        registry,
        (template,),
        (qualification,),
        (design_input,),
        sha256(b"matrix-observation-order-declarative-authoring-context").hexdigest(),
    )
    expanded = compose_parameterised_candidate_context_for_template(
        template_key=template.template_key,
        context=context,
        records=payloads,
        factories=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
    )
    assert isinstance(expanded, CandidateCompilationContext)
    base, standard = _entry(draft, expanded, source.science_specification)
    proposed = tuple(
        ProposedStudyExtension(
            f"extension.matrix-observation-order.{i:02d}",
            f"matrix-observation-order-{i:02d}",
            ObjectIdentity.from_record(_record_id(record), record),
            len(record.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, (record, decoder) in enumerate(zip(payloads, decoders, strict=True))
    )
    extension_set = ProposedStudyExtensionSet(
        "proposed-extensions.matrix-observation-order",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(x.namespace_id for x in proposed),
        proposed,
    )
    package = ExecutableStudyDefinition("authoring-package.matrix-observation-order", base, extension_set)
    installed = GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog
    registrations = tuple(
        x for x in installed.registrations if x.manifest.capability_key in selected_keys
    )
    catalog = CandidateCapabilityCatalog("catalog.matrix-observation-order", registrations, (template,))
    return MatrixObservationOrderObservationAuthoringBundle(
        package,
        replace(standard, base=context),
        catalog,
        payloads,
        decoders,
        evidence,
        source_profile,
    )


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderObservationCandidateContextProvider:
    bundle: MatrixObservationOrderObservationAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("observation order candidate draft differs")
        source = CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ())
        return CandidateContextResolution(self.bundle.standard_context.base, (), source)

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("observation order standard authoring package differs")
        source = CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ())
        return StandardCandidateContextResolution(self.bundle.standard_context, (), source)


__all__ = [
    'MatrixObservationOrderObservationAuthoringBundle',
    'MatrixObservationOrderObservationCandidateContextProvider',
    'build_observation_order_observation_authoring',
]
