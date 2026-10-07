"Outcome-blind measurement through local law plus conditional admission/controller use and comparative study authoring.\n\nThe final law adjudication remains local law; root-level controller use, native scores and paired\ncontributions have separate typed outputs and no invented aggregate verdict.\n"
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.science import CHART, CLOCK, CUTOFF, PREFIX, SUPPORT, UNIT, RECEIVERS, empirical_system
from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe
from empirical_lawhood.adapters.methods.reactor_causal_response.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_causal_response.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_causal_response.transport import CausalReactorRootEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope, EmpiricalQualificationTerminal
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_causal_response.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_causal_response.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.reactor_causal_response.config import EmpiricalNativeConfig, NATIVE_TASKS
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource, TimedReactorConfirmationEnvelope, EmpiricalRootAnalysis, EmpiricalContributionResult, EmpiricalNativeBenchmarkResult
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_runner import ROOT_TASKS, NATIVE_BENCHMARK_TASKS

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
    ScientificInputRole,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

from empirical_lawhood.adapters.methods.reactor_causal_response.science import PROTOCOL_BUDGET

CAMPAIGN_BUDGET = PROTOCOL_BUDGET

SOURCE_TASKS = tuple(sorted(t for t, _, _ in NATIVE_TASKS))
SCIENCE_TASK = "empirical.qualification"
TERMINAL_TASK = "empirical.adjudication"
ALL_UNITS = tuple(t.removeprefix("empirical.") for t in SOURCE_TASKS)


def _experiment(system: SystemSpec, design: EmpiricalRecipe) -> ExperimentSpec:
    required = ObligationStatus.REQUIRED
    views = tuple(v.view_id for v in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    obligations = ScientificObligations(
        f"{PREFIX}.obligations",
        SupportSpec(
            f"{PREFIX}.support",
            system.relation.relation_id,
            UNIT,
            32,
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
            ("frozen-empirical-policy-only", "native-delivery-observed"),
            ("INCOMPLETE_NATIVE_DELIVERY",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            "whole-root-joint-rank32-of32",
            UNIT,
            Decimal(".95"),
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
                "All 32 calibration scores must be finite; all 32 fresh qualification roots adequate with one-sided CP lower bound at least .90; numerical allowances fixed in design.",
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
            f"{PREFIX}.convergence",
            ("absolute-window-temperature-and-end-conversion",),
            views,
            (),
            required,
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
        "Observation-learned causal response laws qualify a frozen rolling policy over complete batches.",
        "Closed-interval temperature maximum and endpoint conversion; 32 independent calibration roots and 32 fresh qualification roots; frozen chosen policy only.",
        UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "The registered joint predictive profile and existing sole law owner qualify the forecast; admission and prospective validation remain separate.",
        ("frozen-policy-path-only", "native-delay-retained"),
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
            "Sixteen fit and eight nomination roots each receive exploration-unshifted/exploration-shifted-one-position/exploration-shifted-two-positions/feed-intervention/jacket-intervention; 32 calibration and 32 qualification roots follow the frozen chosen policy. Every nominal tape is replayed at half the integration step.",
            (SUPPORT,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Fixed F0 and F1 rivals and matched empirical branches are retained; diagnostics do not alter nomination, calibration or qualification.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                "heldout-joint-coverage-fraction",
                Decimal(".9"),
                "1",
                32,
                "Exactly 32 calibration and 32 fresh qualification roots; all 2880 callbacks and both numerical views are nested, without top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            tuple(f"reactor-empirical-calibration-{i:03d}" for i in range(32)),
            f"{PREFIX}.fixed-heldout-assigned-units",
            sha256(
                canonical_json_bytes(
                    tuple(f"reactor-empirical-qualification-{i:03d}" for i in range(32))
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
class EmpiricalAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_empirical_authoring(
    *,
    source: EmpiricalStudySource,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
) -> EmpiricalAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import require_specifications

    require_specifications(specifications, count=5)
    design = EmpiricalRecipe()
    native = EmpiricalNativeConfig()
    system = empirical_system(design)
    experiment = _experiment(system, design)
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=CAMPAIGN_BUDGET,
        objective="Learn empirical response laws, freeze nomination and calibration, and qualify the exact chosen policy on fresh independent roots.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    capabilities = tuple(
        sorted((METHOD_CAPABILITY, SOURCE_CAPABILITY), key=lambda c: c.registry_id)
    )
    registry = CapabilityRegistry(f"{PREFIX}.registry", capabilities)

    def method_step(
        task: str,
        dependencies: tuple[str, ...],
        schema: str,
        seconds: int,
        *,
        terminal: bool = False,
    ) -> ProtocolStepTemplate:
        return ProtocolStepTemplate(
            task,
            ScientificStage.EVALUATE,
            METHOD_CAPABILITY.capability_key,
            METHOD_CAPABILITY.capability_version,
            _config_ref(
                design, ObjectIdentity.from_record(design.config_id, design), METHOD_CAPABILITY
            ),
            tuple(sorted(dependencies)),
            _outputs((("science", schema),)),
            METHOD_CAPABILITY.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.OUTCOME_VISIBLE if terminal else VisibilityCeiling.PROSPECTIVE,
            ResourceBudget(
                1,
                8 * 1024**3,
                0,
                seconds,
                (
                    3 * 1024**3
                    if task in ROOT_TASKS
                    else 6 * 1024**3
                    if task == "empirical.discovery"
                    else 1280 * 1024**2
                    if task == "empirical.calibration"
                    else 2176 * 1024**2
                    if task == SCIENCE_TASK
                    else 768 * 1024**2
                    if terminal
                    else 256 * 1024**2
                ),
                (
                    256
                    if task == "empirical.discovery"
                    else 128
                    if task == "empirical.contribution"
                    else 64
                    if task in NATIVE_BENCHMARK_TASKS
                    else 16
                    if task == SCIENCE_TASK
                    else 4
                    if task in ROOT_TASKS or task == "empirical.calibration"
                    else 1
                )
                * 1024**2,
            ),
            (),
            BarrierKind.REVEAL,
            1,
            (f"{PREFIX}.single-terminal" if terminal else f"{PREFIX}.{task}",),
        )

    steps = (
        method_step(
            "empirical.discovery",
            tuple(t for t, r, _ in NATIVE_TASKS if r in ("fit", "nomination")),
            EmpiricalDiscoveryEnvelope.SCHEMA,
            3600,
        ),
        method_step(
            "empirical.calibration",
            ("empirical.discovery",) + tuple(t for t, r, _ in NATIVE_TASKS if r == "calibration"),
            EmpiricalCalibrationEnvelope.SCHEMA,
            1200,
        ),
        method_step(
            SCIENCE_TASK,
            ("empirical.calibration",)
            + tuple(t for t, r, _ in NATIVE_TASKS if r in ("calibration", "qualification")),
            EmpiricalQualificationTerminal.SCHEMA,
            2400,
        ),
        method_step(
            TERMINAL_TASK,
            (SCIENCE_TASK, "empirical.contribution", *NATIVE_BENCHMARK_TASKS),
            ScientificAdjudicationRecord.SCHEMA,
            300,
            terminal=True,
        ),
        *(
            method_step(
                task,
                (SCIENCE_TASK, "empirical.discovery", task.replace("empirical.prospective-evaluation.", "empirical.")),
                EmpiricalRootAnalysis.SCHEMA,
                7200,
            )
            for task in ROOT_TASKS
        ),
        method_step(
            "empirical.contribution",
            (SCIENCE_TASK, *ROOT_TASKS),
            EmpiricalContributionResult.SCHEMA,
            1200,
        ),
        *(
            method_step(
                task,
                (SCIENCE_TASK, "empirical.discovery"),
                EmpiricalNativeBenchmarkResult.SCHEMA,
                1800,
            )
            for task in NATIVE_BENCHMARK_TASKS
        ),
        *(
            ProtocolStepTemplate(
                task,
                ScientificStage.PREPARE,
                SOURCE_CAPABILITY.capability_key,
                SOURCE_CAPABILITY.capability_version,
                _config_ref(
                    native, ObjectIdentity.from_record(native.config_id, native), SOURCE_CAPABILITY
                ),
                ()
                if role in ("fit", "nomination")
                else ("empirical.discovery",)
                if role == "calibration"
                else ("empirical.discovery", SCIENCE_TASK)
                if role == "confirmation"
                else ("empirical.calibration",),
                _outputs(
                    (
                        (
                            "native-panel",
                            EmpiricalAcquisitionEnvelope.SCHEMA
                            if role in ("fit", "nomination")
                            else TimedReactorConfirmationEnvelope.SCHEMA
                            if role == "confirmation"
                            else CausalReactorRootEnvelope.SCHEMA,
                        ),
                    )
                ),
                SOURCE_CAPABILITY.permissions,
                OutcomeAccess.EVALUATION_SEALED
                if role in ("fit", "nomination")
                else OutcomeAccess.EVALUATOR_REVEAL,
                VisibilityCeiling.PROSPECTIVE,
                ResourceBudget(
                    1,
                    8 * 1024**3,
                    0,
                    1200
                    if role in ("fit", "nomination")
                    else 14400
                    if role == "confirmation"
                    else 300,
                    (256 if role in ("confirmation", "calibration") else 64) * 1024**2,
                    256 * 1024**2
                    if role in ("fit", "nomination", "confirmation")
                    else 32 * 1024**2,
                ),
                (),
                BarrierKind.FREEZE,
                1,
                (f"{PREFIX}.native-operands.{task}",),
            )
            for task, role, _ in NATIVE_TASKS
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
    method_source_id = f"{PREFIX}.method-source-bundle"
    method_source = replace(
        template.graph.external_inputs[0],
        input_id=method_source_id,
        logical_artifact_id=method_source_id,
    )
    edges = tuple(
        sorted(
            (
                *(
                    replace(edge, scientific_role=ScientificInputRole.MODEL)
                    if edge.consumer_node_id in SOURCE_TASKS and edge.producer_node_id is not None
                    else edge
                    for edge in template.graph.edges
                ),
                *(
                    replace(
                        first_edge,
                        edge_id=f"{PREFIX}.source-edge.{task_id}",
                        consumer_node_id=task_id,
                        external_input_id=method_source_id
                        if task_id in (*ROOT_TASKS, *NATIVE_BENCHMARK_TASKS)
                        else first_edge.external_input_id,
                        logical_artifact_id=method_source_id
                        if task_id in (*ROOT_TASKS, *NATIVE_BENCHMARK_TASKS)
                        else first_edge.logical_artifact_id,
                    )
                    for task_id in (*SOURCE_TASKS[1:], *ROOT_TASKS, *NATIVE_BENCHMARK_TASKS)
                ),
            ),
            key=lambda e: e.edge_id,
        )
    )
    graph = replace(
        template.graph,
        edges=edges,
        external_inputs=tuple(
            sorted((*template.graph.external_inputs, method_source), key=lambda v: v.input_id)
        ),
    )
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
        native.fingerprint(),
        observer,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design_identity = ObjectIdentity.from_record(design.config_id, design)
    method_materialization = ObjectIdentity.from_record(method_source_id, source)
    method_qualification = replace(
        qualification,
        receipt_id=f"{PREFIX}.method-configuration-qualification",
        source_id=method_source_id,
        materialization=method_materialization,
    )
    method_source_ref = replace(
        source_ref,
        source_id=method_source_id,
        materialization=method_materialization,
        qualification_receipt=ObjectIdentity.from_record(
            method_qualification.receipt_id, method_qualification
        ),
    )
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
    design_inputs = (
        design_input,
        *(
            DesignInputRecord(
                f"{PREFIX}.specification-{i}",
                ObjectIdentity(
                    f"{PREFIX}.specification-{i}",
                    'empirical-lawhood/document/scientific-specification',
                    "1.0.0",
                    digest,
                ),
                digest,
                experiment.information_cutoffs[0],
                DesignInputRole.MOTIVATION,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                "owner.empirical-lawhood",
            )
            for i, (_, digest) in enumerate(specifications)
        ),
    )
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
        tuple(sorted(f"seed.{seed + i}" for role, n, seed in design.roles for i in range(n))),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in capabilities
        ),
        tuple(sorted((source_ref, method_source_ref), key=lambda s: s.source_id)),
        CAMPAIGN_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{PREFIX}.context",
        registry,
        (template,),
        tuple(sorted((qualification, method_qualification), key=lambda q: q.receipt_id)),
        design_inputs,
        implementation_sha256,
    )
    base, standard = _entry(
        draft,
        context,
        design_identity,
        prefix=PREFIX,
        minimum_units=32,
        operand_description="Frozen-policy absolute {domain} forecast operands across complete preassigned episodes, causal observer histories, two native receivers and two coupled views; no off-policy or private-panel coverage claim.",
        estimator="observation-learned-nine-ridge-candidates",
        uncertainty="whole-episode-joint-calibration",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=METHOD_CAPABILITY,
        input_schema=CausalReactorRootEnvelope.SCHEMA,
        output_schema=EmpiricalQualificationTerminal.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[EmpiricalRecipe | EmpiricalNativeConfig, ...] = tuple(
        sorted((design, native), key=lambda r: r.SCHEMA)
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
            EvidenceRung.ADMISSION,
            EvidenceRung.CONTROLLER_USE,
        ), key=lambda rung: rung.value)),
    )
    selected = {c.capability_key for c in capabilities}
    return EmpiricalAuthoringBundle(
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
class EmpiricalCandidateContextProvider:
    bundle: EmpiricalAuthoringBundle

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
