"""Noncontact authoring for twenty native prefix tasks and one existing-owner chain."""
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry, study_template_from_protocol as _programme_template
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.reactor_prefix_response.design import CHART, CLOCK, CUTOFF, PREFIX, RECEIVER, SUPPORT, UNIT, ReactorScienceDesign, reactor_science_design, reactor_system, unit_ids
from empirical_lawhood.adapters.methods.reactor_prefix_response.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_prefix_response.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_prefix_response.result import ReactorScienceResult
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_prefix_response.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_prefix_response.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import BRANCHES, ReactorPrefixConfig, ReactorPrefixPanel, ReactorSourceBundle, native_task_id
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
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
    CandidateDiagnostic,
    CandidateDiagnosticClass,
    CandidateDiagnosticCode,
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

from .prior_run import ReactorPriorRun

SOURCE_TASKS = tuple(native_task_id(*b) for b in BRANCHES)
SCIENCE_TASK = "reactor-finite-chain"
BUDGET = ResourceBudget(1, 1024**3, 0, 6300, 28 * 1024**2, 48 * 1024**2)


def _experiment(
    system: SystemSpec,
    design: ReactorScienceDesign,
    *,
    method_capability=METHOD_CAPABILITY,
    source_tasks=SOURCE_TASKS,
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
            5,
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
            ("fixed-public-prefix-only", "native-delivery-observed"),
            ("INCOMPLETE_NATIVE_DELIVERY",),
            required,
        ),
        UncertaintySpec(
            f"{PREFIX}.uncertainty",
            "complete-unit-bootstrap",
            UNIT,
            design.simultaneous_confidence_level,
            (RECEIVER,),
            ("FIXED_PUBLIC_PANEL_NOT_POPULATION_SAFETY",),
            required,
        ),
        (
            FalsifierSpec(
                f"{PREFIX}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                method_capability.capability_key,
                "Heldout contrast error or timestep disagreement falsifies the bounded contrast claim.",
                "Fixed 0.01 K heldout and 0.001 K refinement limits; no retuning or branch replacement.",
                required,
            ),
        ),
        ClosureSpec(
            f"{PREFIX}.closure",
            (SUPPORT,),
            ("first-jacket-command",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{PREFIX}.convergence",
            ("paired-temperature-contrast",),
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
            (f"{PREFIX}.bounded-prefix",),
        ),
    )
    claim = ClaimSpec(
        f"{PREFIX}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "The fixed public panel supports a finite local jacket-temperature contrast under its native delayed observer.",
        "Paired callback-temperature contrast at 20 s for the declared first jacket command; no full-batch or absolute forecast claim.",
        UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Only the existing finite-action proof profile and sole law owner may qualify the contrast.",
        ("fixed-public-panel-only", "native-delay-retained"),
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
            "Every declared scenario/seed receives both arms and both nested numerical views; splits are frozen in design.",
            (SUPPORT,),
        ),
        (RECEIVER,),
        (
            ControlSpec(
                f"{PREFIX}.control",
                ControlKind.BASELINE_COMPARATOR,
                method_capability.capability_key,
                (RECEIVER,),
                "The comparator holds the first jacket command at 316 K; the second command and zero feed are shared.",
            ),
        ),
        (
            PrecisionGoal(
                f"{PREFIX}.precision",
                "heldout-absolute-contrast-error",
                design.maximum_heldout_absolute_error_k,
                "K",
                5,
                "Exactly five fixed units, no top-up; two numerical views are nested.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{PREFIX}.reveal",
            tuple(
                f"unit.{s.replace('_', '-')}"
                for s in design.native.calibration_scenarios
            ),
            f"{PREFIX}.fixed-heldout-public-units",
            sha256(canonical_json_bytes(design.native.heldout_scenarios)).hexdigest(),
            tuple(f"artifact.{PREFIX}.{task}.native-panel" for task in source_tasks),
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
    source: ReactorSourceBundle,
    implementation_sha256: str,
    prior_runs: tuple[ReactorPriorRun, ...] = (),
    fresh_experiment_id: str | None = None,
    assignment=None,
    prior_census=None,
) -> ReactorAuthoringBundle:
    if (assignment is None) != (prior_census is None):
        raise ValueError("REACTOR_ASSIGNMENT_COMPLETE_CENSUS_REQUIRED")
    if assignment is not None:
        assignment.check_prior_census(prior_census)
    if fresh_experiment_id is not None and (
        fresh_experiment_id in (PREFIX, "tbs-reactor-prefix-v1-r4") or prior_runs
    ):
        raise ValueError(
            "fresh reactor authoring requires a new identity and no historical ledgers"
        )
    stem = PREFIX if fresh_experiment_id is None else fresh_experiment_id
    method_capability, source_capability = METHOD_CAPABILITY, SOURCE_CAPABILITY
    method_binding, source_binding = METHOD_BINDING, SOURCE_BINDING
    panel_type, result_type = ReactorPrefixPanel, ReactorScienceResult
    design = reactor_science_design()
    if assignment is not None:
        if fresh_experiment_id is None:
            raise ValueError(
                "assigned reactor requires a fresh target experiment identity"
            )
        from empirical_lawhood.adapters.methods.reactor_prefix_response.assigned.executable_binding import BINDING as assigned_method_binding
        from empirical_lawhood.adapters.methods.reactor_prefix_response.assigned.extension_bundle import CAPABILITY as assigned_method_capability
        from empirical_lawhood.adapters.methods.reactor_prefix_response.assigned.records import ReactorAssignedScienceResult, assigned_science_design
        from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixPanel
        from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned_binding.executable_binding import BINDING as assigned_source_binding
        from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned_binding.extension_bundle import CAPABILITY as assigned_source_capability

        design = assigned_science_design(assignment)
        method_capability, source_capability = (
            assigned_method_capability,
            assigned_source_capability,
        )
        method_binding, source_binding = (
            assigned_method_binding,
            assigned_source_binding,
        )
        panel_type, result_type = (
            ReactorAssignedPrefixPanel,
            ReactorAssignedScienceResult,
        )
    source_tasks = tuple(
        native_task_id(*branch, config=design.native)
        for branch in design.native.branches
    )
    system = reactor_system(design)
    experiment = _experiment(
        system, design, method_capability=method_capability, source_tasks=source_tasks
    )
    if fresh_experiment_id is not None:
        policy = replace(
            system.authority_policy,
            policy_id=f"{stem}.authority-policy",
            scope_ids=(stem,),
            allowed_actions=frozenset(
                (
                    AuthorityAction.SIMULATION_EXECUTION,
                    AuthorityAction.EVALUATOR_REVEAL,
                    AuthorityAction.READ_ONLY_EXPLORATION,
                )
            ),
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            required_gate_ids=(
                f"{stem}.exact-production-proof",
                f"{stem}.typed-execution-authority",
            ),
        )
        system = replace(system, system_id=f"{stem}.system", authority_policy=policy)
        experiment = replace(
            experiment,
            experiment_id=stem,
            system_id=system.system_id,
            claims=(
                replace(
                    experiment.claims[0],
                    claim_id=f"{stem}.claim",
                    proposition=(
                        "The declared public five-scenario reactor panel yields a finite "
                        "paired jacket-temperature contrast under the delayed native observer."
                    ),
                    promotion_rule=(
                        "The existing finite-action owner may qualify only the fixed-panel "
                        "conditional contrast; this public roster supplies no independent "
                        "population confirmation."
                    ),
                ),
            ),
            reveal_barrier=replace(
                experiment.reveal_barrier,
                barrier_id=f"{stem}.reveal",
                evaluation_cohort_id=f"{stem}.heldout-assigned-units",
            ),
            authority_policy_id=policy.policy_id,
        )
    if assignment is not None:
        experiment = replace(
            experiment,
            claims=(
                replace(
                    experiment.claims[0],
                    proposition="Five preassigned native scenario/seed units test the fixed finite jacket-temperature contrast under the delayed native observer.",
                    promotion_rule="Qualify only the declared finite-panel contrast; release qualification does not replicate manuscript claims or establish population transport.",
                ),
            ),
        )
    campaign = _campaign(
        system,
        experiment,
        prefix=stem,
        budget=BUDGET,
        objective='Execute one declared reactor measurement-through-law-qualification contrast chain and retain every two-decision prefix.',
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    capabilities = tuple(
        sorted((method_capability, source_capability), key=lambda c: c.registry_id)
    )
    registry = CapabilityRegistry(f"{stem}.registry", capabilities)
    steps = (
        ProtocolStepTemplate(
            SCIENCE_TASK,
            ScientificStage.EVALUATE,
            method_capability.capability_key,
            method_capability.capability_version,
            _config_ref(
                design,
                ObjectIdentity.from_record(design.config_id, design),
                method_capability,
            ),
            source_tasks,
            _outputs(
                (
                    ("science", result_type.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
            method_capability.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            # This is the terminal evaluator: reveal its complete output batch.
            # Mixed private science/public adjudication cannot publish atomically.
            VisibilityCeiling.OUTCOME_VISIBLE,
            method_capability.resource_ceiling,
            (),
            BarrierKind.REVEAL,
            1,
            (f"{stem}.single-terminal",),
        ),
        *(
            ProtocolStepTemplate(
                task_id,
                ScientificStage.PREPARE,
                source_capability.capability_key,
                source_capability.capability_version,
                _config_ref(
                    design.native,
                    ObjectIdentity.from_record(design.native.config_id, design.native),
                    source_capability,
                ),
                (),
                _outputs((("native-panel", panel_type.SCHEMA),)),
                source_capability.permissions,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                source_capability.resource_ceiling,
                (),
                BarrierKind.FREEZE,
                1,
                (f"{stem}.native-operands.{task_id}",),
            )
            for task_id in source_tasks
        ),
    )
    template = _programme_template(
        ProtocolTemplate(f"{stem}.protocol", "1.0.0", steps, False, False, True),
        source,
        f"{stem}.source-bundle",
        experiment,
        registry,
        prefix=stem,
        source_task_id=source_tasks[0],
        terminal_task_id=SCIENCE_TASK,
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
                *template.graph.edges,
                *(
                    replace(
                        first_edge,
                        edge_id=f"{stem}.source-edge.{task_id}",
                        consumer_node_id=task_id,
                    )
                    for task_id in source_tasks[1:]
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
    native_identity = ObjectIdentity.from_record(f"{stem}.source-bundle", source)
    observer = ObjectIdentity.from_record(
        source_capability.capability_key, source_capability
    )
    qualification = MaterializationQualificationReceipt(
        f"{stem}.configuration-qualification",
        native_identity.object_id,
        native_identity,
        source.fingerprint(),
        system.world.world_id,
        observer,
        tuple(v.view_id for v in system.numerical_views),
        ("K", "kg/s"),
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
        f"{stem}.design-input",
        design_identity,
        design.fingerprint(),
        experiment.information_cutoffs[0],
        DesignInputRole.MOTIVATION,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.empirical-lawhood",
    )
    if fresh_experiment_id is None and tuple(p.run_id for p in prior_runs) != (
        "reactor-prefix-response.output-record-declaration-failure",
        "reactor-prefix-response.evaluator-custody-serialization-failure",
        "reactor-prefix-response.terminal-evidence-class-mixing-failure",
    ):
        raise ValueError("publication-only authoring requires the three declared operational ledgers")
    prior_inputs = tuple(
        DesignInputRecord(
            f"{stem}.prior-operational-run.{i}",
            ObjectIdentity.from_record(prior.run_id, prior),
            prior.fingerprint(),
            experiment.information_cutoffs[0],
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        )
        for i, prior in enumerate(prior_runs)
    )
    if assignment is not None:
        prior_inputs = (
            *prior_inputs,
            DesignInputRecord(
                f"{stem}.prior-exposure",
                assignment.prior_census,
                assignment.prior_census.object_fingerprint,
                experiment.information_cutoffs[0],
                DesignInputRole.READINESS_METADATA,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                "owner.empirical-lawhood",
                prior_census.prior_unit_ids,
                prior_census.authoring_seed_ids,
                (),
            ),
        )
    design_inputs = tuple(
        sorted((design_input, *prior_inputs), key=lambda r: r.input_id)
    )
    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.contrast-qualified", f"{stem}.contrast-unqualified"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(i.input_id for i in design_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs,
        () if prior_census is None else prior_census.prior_unit_ids,
        unit_ids(design),
        () if prior_census is None else prior_census.authoring_seed_ids,
        assignment.seed_ids
        if assignment is not None
        else tuple(
            sorted(
                f"seed.{s.replace('_', '-')}"
                for s in (
                    *design.native.calibration_scenarios,
                    *design.native.heldout_scenarios,
                )
            )
        ),
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
        BUDGET,
    )
    context = CandidateCompilationContext(
        f"{stem}.context",
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
        prefix=stem,
        minimum_units=2,
        operand_description="Finite {domain} operands restricted to the paired jacket contrast, native prefix history and declared two-view support; no smooth atlas or full-batch control claim.",
        estimator="finite-action-compatibility-set",
        uncertainty="whole-unit-bootstrap",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=method_capability,
        input_schema=panel_type.SCHEMA,
        output_schema=result_type.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[ReactorScienceDesign | ReactorPrefixConfig, ...] = tuple(
        sorted((design, design.native), key=lambda r: r.SCHEMA)
    )
    by_schema = {
        d.payload_schema: d
        for b in (method_binding, source_binding)
        for d in b.issued_decoder_registrations
    }
    decoders = tuple(by_schema[r.SCHEMA] for r in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{stem}.extension.{i}",
            f"{stem}.namespace.{i}",
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
        f"{stem}.executable-study-definition",
        base,
        ProposedStudyExtensionSet(
            f"{stem}.extensions",
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
        selection_id=f"{stem}.evidence-profile",
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
            f"{stem}.catalog",
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
class ReactorPrefixResponseCandidateContextProvider:
    bundle: ReactorAuthoringBundle

    @property
    def issue_diagnostics(self) -> tuple[CandidateDiagnostic, ...]:
        native = next(
            value for value in self.bundle.payloads if isinstance(value, ReactorPrefixConfig)
        )
        assignment = getattr(native, "assignment", None)
        if (
            assignment is not None
            and assignment.evidence_role == "PROSPECTIVE_RELEASE_QUALIFICATION"
        ):
            return ()
        return (
            CandidateDiagnostic(
                "reactor.issue.exposed-evaluation-roster",
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED,
                "assignment.evidence_role",
                "PUBLIC_EVALUATION_UNITS_AND_SEEDS_EXPOSED: development authoring "
                "cannot be issued as prospective evidence, regardless of proposer assertions",
            ),
        )

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
            issue_diagnostics=self.issue_diagnostics,
        )
