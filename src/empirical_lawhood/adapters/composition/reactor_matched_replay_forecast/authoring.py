"""Noncontact authoring for 42 five-episode native units, prospective qualification and separate reveal."""
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.transport import ReactorTraceEnvelope

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.science import CHART, CLOCK, CUTOFF, PREFIX, SUPPORT, UNIT, ALL_UNITS, BUDGET, ReactorForecastDesign, forecast_design, forecast_system
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.calibration import RECEIVERS
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.records import ReactorForecastResult, QUALIFY, REVEAL
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.design import ReactorBatchConfig, ReactorBatchSource, BRANCHES, native_task_id
from empirical_lawhood.kernel.authority import ResourceBudget
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
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateCompilationContext,
    StandardCandidateCompilationContext,
)
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

SOURCE_TASKS = tuple(native_task_id(*b) for b in BRANCHES)
SCIENCE_TASK = QUALIFY
TERMINAL_TASK = REVEAL


def _experiment(system: SystemSpec, design: ReactorForecastDesign) -> ExperimentSpec:
    required = ObligationStatus.REQUIRED
    views = tuple(v.view_id for v in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    obligations = ScientificObligations(
        f"{PREFIX}.obligations",
        SupportSpec(
            f"{PREFIX}.support",
            system.relation.relation_id,
            UNIT,
            42,
            2,
            CUTOFF,
            (CHART,),
            (SUPPORT,),
            (),
            required,
        ),
        ValiditySpec(
            f"{PREFIX}.validity",
            (SUPPORT,),
            ("fixed-reference-policy-only", "native-delivery-observed"),
            ("INCOMPLETE_NATIVE_DELIVERY",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            "whole-episode-bonferroni-rank32-of32",
            UNIT,
            design.confidence_level,
            tuple(sorted(RECEIVERS)),
            ("NO_PRIVATE_PANEL_COVERAGE_GUARANTEE",),
            required,
        ),
        (
            FalsifierSpec(
                f"{PREFIX}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                METHOD_CAPABILITY.capability_key,
                "Nonfinite calibrated bounds or insufficient joint heldout coverage falsifies the on-policy forecast claim.",
                "All 32 calibration scores must be finite; at least nine of ten heldout episodes covered; numerical allowances fixed in design.",
                required,
            ),
        ),
        ClosureSpec(
            f"{PREFIX}.closure",
            (SUPPORT,),
            ("native-integrator-timestep",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{PREFIX}.convergence", ("absolute-temperature-dose-conversion",), views, (), required
        ),
        ComputabilityEvidence(
            f"{PREFIX}.computability-evidence",
            system.computability_envelopes[0].envelope_id,
            views,
            ReadinessStatus.READY,
            (),
            (f"{PREFIX}.bounded-full-batch",),
        ),
    )
    claim = ClaimSpec(
        f"{PREFIX}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "Causal absolute forecasts under the fixed reference policy qualify for rolling full-batch admission.",
        "Next-interval temperature peak, endpoint dose and conversion across complete assigned episodes; singleton on-policy support only.",
        UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "The registered joint predictive profile and existing sole law owner qualify the forecast; admission and prospective validation remain separate.",
        ("native-delay-retained", "on-policy-only"),
        numerical_view_ids=views,
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
            "Each of 42 fresh roots receives two feedback views, matched native-command replay and two prescribed pulse views; 32 calibration and ten heldout roots.",
            (SUPPORT,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Persistence holds the same causal observer state over the next interval; diagnostic comparison only, with no extra point-error admission gate.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                "heldout-joint-coverage-fraction",
                Decimal(".9"),
                "1",
                42,
                "Exactly 32 calibration and ten heldout units, no top-up; callbacks and two numerical views are nested.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            tuple(s.unit_id for s in design.native.scenarios if s.split == "calibration"),
            f"{PREFIX}.fixed-heldout-assigned-units",
            sha256(
                canonical_json_bytes(
                    tuple(s.unit_id for s in design.native.scenarios if s.split == "heldout")
                )
            ).hexdigest(),
            tuple(f"artifact.{PREFIX}.{task}.native-panel" for task in SOURCE_TASKS),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.PROSPECTIVE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


@dataclass(frozen=True, slots=True)
class ReactorAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_reactor_authoring(
    *,
    source: ReactorBatchSource,
    implementation_sha256: str,
) -> ReactorAuthoringBundle:
    design = forecast_design()
    system = forecast_system(design)
    experiment = _experiment(system, design)
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=BUDGET,
        objective="Qualify absolute forecasts with matched-input numerical fidelity and report feedback robustness and prescribed recovery separately.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    capabilities = tuple(
        sorted((METHOD_CAPABILITY, SOURCE_CAPABILITY), key=lambda c: c.registry_id)
    )
    registry = CapabilityRegistry(f"{PREFIX}.registry", capabilities)
    steps = (
        ProtocolStepTemplate(
            SCIENCE_TASK,
            ScientificStage.EVALUATE,
            METHOD_CAPABILITY.capability_key,
            METHOD_CAPABILITY.capability_version,
            _config_ref(
                design, ObjectIdentity.from_record(design.config_id, design), METHOD_CAPABILITY
            ),
            SOURCE_TASKS,
            _outputs((("science", ReactorForecastResult.SCHEMA),)),
            METHOD_CAPABILITY.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.PROSPECTIVE,
            METHOD_CAPABILITY.resource_ceiling,
            (),
            BarrierKind.REVEAL,
            1,
            (f"{PREFIX}.forecast-qualification",),
        ),
        ProtocolStepTemplate(
            TERMINAL_TASK,
            ScientificStage.EVALUATE,
            METHOD_CAPABILITY.capability_key,
            METHOD_CAPABILITY.capability_version,
            _config_ref(
                design, ObjectIdentity.from_record(design.config_id, design), METHOD_CAPABILITY
            ),
            (SCIENCE_TASK,),
            _outputs((("science", ScientificAdjudicationRecord.SCHEMA),)),
            METHOD_CAPABILITY.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.OUTCOME_VISIBLE,
            ResourceBudget(1, 4 * 1024**3, 0, 300, 16 * 1024**2, 1024**2),
            (),
            BarrierKind.REVEAL,
            1,
            (f"{PREFIX}.single-terminal",),
        ),
        *(
            ProtocolStepTemplate(
                task_id,
                ScientificStage.PREPARE,
                SOURCE_CAPABILITY.capability_key,
                SOURCE_CAPABILITY.capability_version,
                _config_ref(
                    design.native,
                    ObjectIdentity.from_record(design.native.config_id, design.native),
                    SOURCE_CAPABILITY,
                ),
                (),
                _outputs((("native-panel", ReactorTraceEnvelope.SCHEMA),)),
                SOURCE_CAPABILITY.permissions,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                SOURCE_CAPABILITY.resource_ceiling,
                (),
                BarrierKind.FREEZE,
                1,
                (f"{PREFIX}.native-operands.{task_id}",),
            )
            for task_id in SOURCE_TASKS
        ),
    )
    template = _programme_template(
        ProtocolTemplate(
            f"{PREFIX}.protocol",
            "1.0.0",
            tuple(sorted(steps, key=lambda s: s.step_id)),
            False,
            False,
            True,
        ),
        source,
        f"{PREFIX}.source-bundle",
        experiment,
        registry,
        prefix=PREFIX,
        source_task_id=SOURCE_TASKS[0],
        terminal_task_id=TERMINAL_TASK,
        primary_output_ids={
            ScientificStage.PREPARE: "native-panel",
            ScientificStage.QUALIFY: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    first_edge = next(e for e in template.graph.edges if e.external_input_id is not None)
    edges = tuple(
        sorted(
            (
                *template.graph.edges,
                *(
                    replace(
                        first_edge,
                        edge_id=f"{PREFIX}.source-edge.{task_id}",
                        consumer_node_id=task_id,
                    )
                    for task_id in SOURCE_TASKS[1:]
                ),
            ),
            key=lambda e: e.edge_id,
        )
    )
    graph = replace(template.graph, edges=edges)
    coverage = replace(
        template.coverage,
        bindings=tuple(
            replace(
                b,
                contributor_edge_ids=tuple(
                    e.edge_id for e in edges if e.consumer_node_id == b.proof_owner_node_id
                ),
            )
            for b in template.coverage.bindings
        ),
    )
    template = replace(template, graph=graph, coverage=coverage)
    native_identity = ObjectIdentity.from_record(f"{PREFIX}.source-bundle", source)
    observer = ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY)
    qualification = MaterializationQualificationReceipt(
        f"{PREFIX}.configuration-qualification",
        native_identity.object_id,
        native_identity,
        source.fingerprint(),
        system.world.world_id,
        observer,
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({q.native_unit for q in system.quantities})),
        tuple(sorted({q.coordinate_frame for q in system.quantities})),
        (CLOCK,),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        native_identity.object_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        native_identity,
        source.fingerprint(),
        design.native.fingerprint(),
        observer,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design_identity = ObjectIdentity.from_record(design.config_id, design)
    design_input = DesignInputRecord(
        f"{PREFIX}.design-input",
        design_identity,
        design.fingerprint(),
        experiment.information_cutoffs[0],
        DesignInputRole.MOTIVATION,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.empirical-lawhood",
    )
    design_inputs = (design_input,)
    draft = StudyDraft(
        f"{PREFIX}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{PREFIX}.forecast-qualified", f"{PREFIX}.forecast-unqualified"),
        DesignOrigin(
            f"{PREFIX}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(i.input_id for i in design_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        (),
        ALL_UNITS,
        (),
        tuple(sorted(f"seed.{s.seed}" for s in design.native.scenarios)),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in capabilities
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
    base, standard = _entry(
        draft,
        context,
        design_identity,
        prefix=PREFIX,
        minimum_units=32,
        operand_description="On-policy absolute {domain} forecast operands across complete preassigned episodes, causal observer histories, three native receivers and two coupled views; no off-policy or private-panel coverage claim.",
        estimator="fixed-causal-native-forecast",
        uncertainty="whole-episode-joint-calibration",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=METHOD_CAPABILITY,
        input_schema=ReactorTraceEnvelope.SCHEMA,
        output_schema=ReactorForecastResult.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[ReactorForecastDesign | ReactorBatchConfig, ...] = tuple(
        sorted((design, design.native), key=lambda r: r.SCHEMA)
    )
    by_schema = {
        d.payload_schema: d
        for b in (METHOD_BINDING, SOURCE_BINDING)
        for d in b.issued_decoder_registrations
    }
    decoders = tuple(by_schema[r.SCHEMA] for r in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{PREFIX}.extension.{i}",
            f"{PREFIX}.namespace.{i}",
            ObjectIdentity.from_record(r.config_id, r),
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
    package = ExecutableStudyDefinition(
        f"{PREFIX}.executable-study-definition",
        base,
        ProposedStudyExtensionSet(
            f"{PREFIX}.extensions",
            ObjectIdentity.from_record(base.package_id, base),
            tuple(p.namespace_id for p in proposed),
            proposed,
        ),
    )
    evidence_registry = build_observation_evidence_world_registry()
    world_profile = next(
        p
        for p in evidence_registry.world_profiles
        if p.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{PREFIX}.evidence-profile",
        draft_id=draft.draft_id,
        registry=evidence_registry,
        world_profile_id=world_profile.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=tuple(sorted((
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        ), key=lambda rung: rung.value)),
    )
    selected = {c.capability_key for c in capabilities}
    return ReactorAuthoringBundle(
        package,
        standard,
        CandidateCapabilityCatalog(
            f"{PREFIX}.catalog",
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


@dataclass(frozen=True, slots=True)
class ReactorMatchedReplayForecastCandidateContextProvider:
    bundle: ReactorAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("reactor draft differs from its frozen authoring context")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )

    def resolve_standard(
        self, package: StudyDefinition
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("reactor package differs from its frozen authoring context")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )
