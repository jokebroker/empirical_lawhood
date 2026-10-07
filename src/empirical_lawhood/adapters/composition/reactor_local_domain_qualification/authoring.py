"Fresh local measurement through local law authoring with explicit exposed-development ancestry.\n\nThe frozen initial domain is uncovered; admission/controller use overlays are inapplicable.\n"
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import ROOTS, SOURCE_TASKS, LocalQualificationDesign
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalRootEvidence
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.science import CHART, CLOCK, CUTOFF, PREFIX, PROTOCOL_BUDGET, RECEIVERS, SUPPORT, UNIT, local_system
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.scoring import LocalCalibration
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.terminal import LocalQualificationResult
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.config import LocalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
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
    ScientificInputRole,
    ScientificStage,
)
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

CAMPAIGN_BUDGET = PROTOCOL_BUDGET

SCIENCE_TASK = "local.qualification"
TERMINAL_TASK = "local.adjudication"
ALL_UNITS = tuple(t.removeprefix("local.") for t in SOURCE_TASKS)


def _experiment(
    system: SystemSpec, design: LocalQualificationDesign
) -> ExperimentSpec:
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
            ("frozen-causal-local-domains", "native-delivery-observed"),
            ("INCOMPLETE_NATIVE_DELIVERY",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            "conditional-local-root-max-calibration",
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
                "Nonfinite local bounds, failed action contrasts or insufficient fresh conditional root adequacy defeats the receiver/domain claim.",
                "At least 29 causally contacted roots per panel; root maximum errors within frozen bounds and action-contrast allowance; one-sided CP lower bound at least .90.",
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
        "Separately qualify 16 receiver/domain claims from 15 empirically discovered local operating regimes on fresh native interventions.",
        "Ten-second temperature maximum or endpoint conversion, assessed separately per nominated domain and receiver; fixed exploratory donors and causal local assays; conditional encounter coverage only.",
        UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "The registered joint predictive profile and existing sole law owner qualify the forecast; admission and prospective validation remain separate.",
        ("causal-local-contact-only", "native-delay-retained"),
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
            "Thirty-two calibration and 32 qualification roots each receive exploration-unshifted/exploration-shifted-one-position/exploration-shifted-two-positions and earliest-causal-encounter local interventions. Replay each exact nominal tape at half the integration step.",
            (SUPPORT,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Matched nominal donor versus one-word intervention retains absolute and contrast errors and the zero-response rival; diagnostics cannot change frozen domains or thresholds.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                "heldout-conditional-domain-root-adequacy-fraction",
                Decimal(".9"),
                "1",
                32,
                "Exactly 32 calibration and 32 fresh qualification roots; causally contacted independent roots counted once, callbacks and views nested, no top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            tuple(f"reactor-local-domain-calibration-{i:03d}" for i in range(32)),
            f"{PREFIX}.fixed-heldout-assigned-units",
            sha256(
                canonical_json_bytes(
                    tuple(f"reactor-local-domain-qualification-{i:03d}" for i in range(32))
                )
            ).hexdigest(),
            tuple(f"artifact.{PREFIX}.{task}.native-panel" for task in SOURCE_TASKS),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


@dataclass(frozen=True, slots=True)
class LocalAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_local_authoring(
    *,
    source: EmpiricalStudySource,
    design: LocalQualificationDesign,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
) -> LocalAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import (
        require_specifications,
    )

    require_specifications(specifications, count=2)
    native = LocalNativeConfig(design)
    system = local_system()
    experiment = _experiment(system, design)
    campaign = _campaign(
        system,
        experiment,
        prefix=PREFIX,
        budget=CAMPAIGN_BUDGET,
        objective="Qualify frozen distinct local response laws on fresh independent roots, retaining uncovered domains and separate receiver verdicts.",
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
                design,
                ObjectIdentity.from_record(design.config_id, design),
                METHOD_CAPABILITY,
            ),
            tuple(sorted(dependencies)),
            _outputs((("science", schema),)),
            METHOD_CAPABILITY.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.OUTCOME_VISIBLE
            if terminal
            else VisibilityCeiling.PROSPECTIVE,
            ResourceBudget(
                1,
                8 * 1024**3,
                0,
                seconds,
                8 * 1024**3,
                128 * 1024**2,
            ),
            (),
            BarrierKind.REVEAL,
            1,
            (f"{PREFIX}.single-terminal" if terminal else f"{PREFIX}.{task}",),
        )

    steps = (
        method_step(
            "local.calibration",
            tuple(f"local.{r}" for r, role, _, _ in ROOTS if role == "calibration"),
            LocalCalibration.SCHEMA,
            1200,
        ),
        method_step(
            SCIENCE_TASK,
            ("local.calibration", *SOURCE_TASKS),
            LocalQualificationResult.SCHEMA,
            7200,
        ),
        method_step(
            TERMINAL_TASK,
            (SCIENCE_TASK,),
            ScientificAdjudicationRecord.SCHEMA,
            300,
            terminal=True,
        ),
        *(
            ProtocolStepTemplate(
                f"local.{root}",
                ScientificStage.PREPARE,
                SOURCE_CAPABILITY.capability_key,
                SOURCE_CAPABILITY.capability_version,
                _config_ref(
                    native,
                    ObjectIdentity.from_record(native.config_id, native),
                    SOURCE_CAPABILITY,
                ),
                () if role == "calibration" else ("local.calibration",),
                _outputs((("native-panel", LocalRootEvidence.SCHEMA),)),
                SOURCE_CAPABILITY.permissions,
                OutcomeAccess.EVALUATION_SEALED
                if role == "calibration"
                else OutcomeAccess.EVALUATOR_REVEAL,
                VisibilityCeiling.PROSPECTIVE,
                ResourceBudget(1, 8 * 1024**3, 0, 1800, 256 * 1024**2, 128 * 1024**2),
                (),
                BarrierKind.FREEZE,
                1,
                (f"{PREFIX}.native-operands.{root}",),
            )
            for root, role, _, _ in ROOTS
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
    first_edge = next(
        e for e in template.graph.edges if e.external_input_id is not None
    )
    edges = tuple(
        sorted(
            (
                *(
                    replace(edge, scientific_role=ScientificInputRole.MODEL)
                    if edge.consumer_node_id in SOURCE_TASKS
                    and edge.producer_node_id is not None
                    else edge
                    for edge in template.graph.edges
                ),
                *(
                    replace(
                        first_edge,
                        edge_id=f"{PREFIX}.source-edge.{task}",
                        consumer_node_id=task,
                    )
                    for task in SOURCE_TASKS[1:]
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
                    e.edge_id
                    for e in edges
                    if e.consumer_node_id == b.proof_owner_node_id
                ),
            )
            for b in template.coverage.bindings
        ),
    )
    template = replace(template, graph=graph, coverage=coverage)
    native_identity = ObjectIdentity.from_record(f"{PREFIX}.source-bundle", source)
    observer = ObjectIdentity.from_record(
        SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY
    )
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
    old_units = tuple(
        sorted(
            f"reactor-empirical-{role}-{i:03d}"
            for role, n in (("fit", 16), ("nomination", 8))
            for i in range(n)
        )
    )
    old_seeds = tuple(
        sorted(
            f"seed.{base + i}"
            for base, n in ((81000, 16), (82000, 8))
            for i in range(n)
        )
    )
    development = DesignInputRecord(
        f"{PREFIX}.development-input",
        ObjectIdentity(
            "reactor-local-development-publication",
            "empirical-lawhood/analysis/reactor-local-discovery/development-publication",
            "1.0.0",
            design.atlas.development_sha256,
        ),
        design.atlas.development_sha256,
        experiment.information_cutoffs[0],
        DesignInputRole.DEVELOPMENT_TUNING,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        "owner.empirical-lawhood",
        old_units,
        old_seeds,
        (),
    )
    design_inputs = tuple(
        sorted((*design_inputs, development), key=lambda d: d.input_id)
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
        old_units,
        ALL_UNITS,
        old_seeds,
        tuple(sorted(f"seed.{seed}" for _, _, _, seed in ROOTS)),
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
        CAMPAIGN_BUDGET,
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
        operand_description="Frozen local {domain} operands, independent root maxima, causal support and applied-word contrasts, receiver-specific qualification; no global, transition, controller or private-panel claim.",
        estimator="observation-learned-distinct-local-response-laws",
        uncertainty="receiver-domain-conditional-root-max-calibration",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=METHOD_CAPABILITY,
        input_schema=LocalRootEvidence.SCHEMA,
        output_schema=LocalQualificationResult.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[LocalQualificationDesign | LocalNativeConfig, ...] = tuple(
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
        ), key=lambda rung: rung.value)),
    )
    selected = {c.capability_key for c in capabilities}
    return LocalAuthoringBundle(
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
class LocalCandidateContextProvider:
    bundle: LocalAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("reactor draft differs from its frozen authoring context")
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
                "reactor package differs from its frozen authoring context"
            )
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )
