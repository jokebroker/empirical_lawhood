"""Strict prospective graph for one RC model and an attached numerical check."""

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import formal_entry, study_template_from_protocol, single_experiment_campaign
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.protocol_helpers import (
    capability_config_ref,
    protocol_outputs,
)
from empirical_lawhood.adapters.methods.rc_ladder_numerical_comparison.contracts import ResistorCapacitorLadderEvaluationConfig
from empirical_lawhood.adapters.methods.rc_ladder_numerical_comparison.executable_binding import BINDING as EVALUATOR_BINDING
from empirical_lawhood.adapters.methods.rc_ladder_numerical_comparison.extension_bundle import CAPABILITY as EVALUATOR_CAPABILITY
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderNumericalCheck, ResistorCapacitorLadderStudyConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.executable_binding import BINDING as NATIVE_BINDING
from empirical_lawhood.adapters.simulators.rc_ladder_response.extension_bundle import CAPABILITY as NATIVE_CAPABILITY
from empirical_lawhood.adapters.simulators.rc_ladder_response.runtime_provider import native_task_id
from empirical_lawhood.kernel.authority import AuthorityAction
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
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld
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

from .design import BUDGET, CHART, CLOCK, CUTOFF, DENOMINATOR, HISTORY, RECEIVER_CURRENT, RECEIVER_VOLTAGE, SUPPORT, UNIT, rc_system


def _experiment(
    system: SystemSpec, study: ResistorCapacitorLadderStudyConfig, *, experiment_id: str
) -> ExperimentSpec:
    stem = experiment_id
    required = ObligationStatus.REQUIRED
    views = tuple(view.view_id for view in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            1,
            2,
            CUTOFF,
            (CHART,),
            (DENOMINATOR,),
            (),
            required,
        ),
        ValiditySpec(
            f"{stem}.validity",
            (SUPPORT,),
            ("finite-source-impedance", "frozen-component-roster", "numerical-only"),
            ("PHYSICAL_BOARD_UNOBSERVED",),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "deterministic-grid-error-bound",
            UNIT,
            Decimal("0.95"),
            (RECEIVER_VOLTAGE,),
            ("NO_POPULATION_CONFIDENCE_FROM_ONE_MODEL",),
            ObligationStatus.NOT_APPLICABLE,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                EVALUATOR_CAPABILITY.capability_key,
                "Two native solvers must agree and the output-grid charge balance must hold.",
                "Fail above the frozen voltage or charge-rate defect; do not retune or count views as units.",
                required,
            ),
        ),
        ClosureSpec(
            f"{stem}.closure",
            (SUPPORT,),
            ("left-source-voltage",),
            (HISTORY,),
            required,
        ),
        StructuralConvergenceSpec(
            f"{stem}.convergence",
            ("four-cell-kirchhoff-equations",),
            views,
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{stem}.computability-evidence",
            system.computability_envelopes[0].envelope_id,
            views,
            ReadinessStatus.READY,
            (),
            ("bounded-local-cpu",),
        ),
    )
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "The declared four-cell model meets its frozen two-view numerical and charge-balance limits.",
        "Maximum node-voltage solver difference in V and output-grid charge-rate residual in A for one model.",
        UNIT,
        EvidenceRung.MEASUREMENT,
        EvidenceCeiling.MEASUREMENT,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Only numerical-method conformance for this declared model; no board, population or law claim.",
        ("declared-model-only", "synthetic-numerical-development"),
        numerical_view_ids=views,
    )
    native_tasks = tuple(native_task_id(study, view) for view in views)
    return ExperimentSpec(
        stem,
        system.system_id,
        system.world.world_id,
        system.relation,
        UNIT,
        (claim,),
        AssignmentSpec(
            f"{stem}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            UNIT,
            system.relation.action_quantity_ids,
            "One frozen model and boundary-voltage word is evaluated under both nested solvers.",
            (SUPPORT,),
        ),
        (RECEIVER_CURRENT, RECEIVER_VOLTAGE),
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                EVALUATOR_CAPABILITY.capability_key,
                (RECEIVER_VOLTAGE,),
                "The matrix-exponential solution is the predeclared comparator for backward Euler.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "maximum-numerical-view-defect",
                study.maximum_view_defect_volts,
                "V",
                1,
                "Exactly one prepared model; stop at the frozen voltage and charge-rate limits.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            (f"{stem}.analytic-one-cell-reference",),
            f"{stem}.fixed-model-evaluation",
            sha256(canonical_json_bytes((study.unit_id,))).hexdigest(),
            tuple(f"artifact.{stem}.{task}.native-panel" for task in native_tasks),
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
class RCLadderResponseAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]


def build_rc_authoring(
    *, study: ResistorCapacitorLadderStudyConfig, experiment_id: str, implementation_sha256: str
) -> RCLadderResponseAuthoringBundle:
    if experiment_id == study.study_id:
        raise ValueError("RC experiment identity must differ from the study identity")
    stem = experiment_id
    evaluation = ResistorCapacitorLadderEvaluationConfig(f"{stem}.evaluation-config", study, True)
    system = rc_system(study, experiment_id=stem)
    experiment = _experiment(system, study, experiment_id=stem)
    campaign = single_experiment_campaign(
        system,
        experiment,
        prefix=stem,
        budget=BUDGET,
        objective="Check one frozen RC model numerically without a physical-board claim.",
        actions=(
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
            ("simulation", AuthorityAction.SIMULATION_EXECUTION),
        ),
    )
    capabilities = tuple(
        sorted((NATIVE_CAPABILITY, EVALUATOR_CAPABILITY), key=lambda c: c.registry_id)
    )
    registry = CapabilityRegistry(f"{stem}.registry", capabilities)
    native_tasks = tuple(
        native_task_id(study, view) for view in ("backward-euler", "matrix-exponential")
    )
    evaluate_task = f"{stem}.numerical-evaluate"
    steps = (
        ProtocolStepTemplate(
            evaluate_task,
            ScientificStage.EVALUATE,
            EVALUATOR_CAPABILITY.capability_key,
            EVALUATOR_CAPABILITY.capability_version,
            capability_config_ref(
                evaluation,
                ObjectIdentity.from_record(evaluation.config_id, evaluation),
                EVALUATOR_CAPABILITY,
            ),
            native_tasks,
            protocol_outputs(
                (
                    ("numerical-check", ResistorCapacitorLadderNumericalCheck.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
            EVALUATOR_CAPABILITY.permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.OUTCOME_VISIBLE,
            EVALUATOR_CAPABILITY.resource_ceiling,
            (),
            BarrierKind.REVEAL,
            1,
            (f"{stem}.single-terminal",),
        ),
        *(
            ProtocolStepTemplate(
                task,
                ScientificStage.PREPARE,
                NATIVE_CAPABILITY.capability_key,
                NATIVE_CAPABILITY.capability_version,
                capability_config_ref(
                    study,
                    ObjectIdentity.from_record(study.study_id, study),
                    NATIVE_CAPABILITY,
                ),
                (),
                protocol_outputs((("native-panel", ResistorCapacitorLadderNativePanel.SCHEMA),)),
                NATIVE_CAPABILITY.permissions,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                NATIVE_CAPABILITY.resource_ceiling,
                (),
                BarrierKind.FREEZE,
                1,
                (f"{stem}.native-operands.{task}",),
            )
            for task in native_tasks
        ),
    )
    protocol = ProtocolTemplate(
        f"{stem}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda step: step.step_id)),
        False,
        False,
        True,
    )
    template = study_template_from_protocol(
        protocol,
        study,
        f"{stem}.source-study",
        experiment,
        registry,
        prefix=stem,
        source_task_id=native_tasks[0],
        terminal_task_id=evaluate_task,
        primary_output_ids={
            ScientificStage.PREPARE: "native-panel",
            ScientificStage.QUALIFY: "numerical-check",
            ScientificStage.EVALUATE: "numerical-check",
        },
    )
    first_edge = next(
        edge for edge in template.graph.edges if edge.external_input_id is not None
    )
    edges = tuple(
        sorted(
            (
                *template.graph.edges,
                replace(
                    first_edge,
                    edge_id=f"{stem}.source-edge.{native_tasks[1]}",
                    consumer_node_id=native_tasks[1],
                ),
            ),
            key=lambda edge: edge.edge_id,
        )
    )
    graph = replace(template.graph, edges=edges)
    coverage = replace(
        template.coverage,
        bindings=tuple(
            replace(
                binding,
                contributor_edge_ids=tuple(
                    edge.edge_id
                    for edge in edges
                    if edge.consumer_node_id == binding.proof_owner_node_id
                ),
            )
            for binding in template.coverage.bindings
        ),
    )
    template = replace(template, graph=graph, coverage=coverage)
    source_identity = ObjectIdentity.from_record(f"{stem}.source-study", study)
    observer = ObjectIdentity.from_record(
        NATIVE_CAPABILITY.capability_key, NATIVE_CAPABILITY
    )
    qualification = MaterializationQualificationReceipt(
        f"{stem}.source-qualification",
        source_identity.object_id,
        source_identity,
        study.fingerprint(),
        system.world.world_id,
        observer,
        tuple(view.view_id for view in system.numerical_views),
        ("A", "F-and-ohm", "V", "ohm"),
        tuple(sorted({quantity.coordinate_frame for quantity in system.quantities})),
        (CLOCK,),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        source_identity.object_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        source_identity,
        study.fingerprint(),
        study.fingerprint(),
        observer,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design_identity = ObjectIdentity.from_record(evaluation.config_id, evaluation)
    design_input = DesignInputRecord(
        f"{stem}.design-input",
        design_identity,
        evaluation.fingerprint(),
        experiment.information_cutoffs[0],
        DesignInputRole.MOTIVATION,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        "owner.empirical-lawhood",
    )
    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.numerical-conformant", f"{stem}.numerical-falsified"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (design_input.input_id,),
            VisibilityCeiling.PROSPECTIVE,
        ),
        (design_input,),
        (f"{stem}.analytic-one-cell-reference",),
        (study.unit_id,),
        (),
        (),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                cap.capability_key, cap.capability_version, cap.implementation_sha256
            )
            for cap in capabilities
        ),
        (source_ref,),
        BUDGET,
    )
    context = CandidateCompilationContext(
        f"{stem}.context",
        registry,
        (template,),
        (qualification,),
        (design_input,),
        implementation_sha256,
    )
    base, standard = formal_entry(
        draft,
        context,
        design_identity,
        prefix=stem,
        minimum_units=1,
        operand_description="Numerical {domain} operands for one frozen RC model and two nested solvers; no physical board or population claim.",
        estimator="frozen-numerical-comparison",
        uncertainty="deterministic-grid-bound",
        ceiling=EvidenceCeiling.MEASUREMENT,
        evaluator=EVALUATOR_CAPABILITY,
        input_schema=ResistorCapacitorLadderNativePanel.SCHEMA,
        output_schema=ResistorCapacitorLadderNumericalCheck.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(
        sorted((evaluation, study), key=lambda record: record.SCHEMA)
    )
    by_schema = {
        decoder.payload_schema: decoder
        for binding in (EVALUATOR_BINDING, NATIVE_BINDING)
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(by_schema[record.SCHEMA] for record in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{stem}.extension.{index}",
            f"{stem}.namespace.{index}",
            ObjectIdentity.from_record(
                record.config_id
                if isinstance(record, ResistorCapacitorLadderEvaluationConfig)
                else record.study_id,
                record,
            ),
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
    package = ExecutableStudyDefinition(
        f"{stem}.authoring",
        base,
        ProposedStudyExtensionSet(
            f"{stem}.extensions",
            ObjectIdentity.from_record(base.package_id, base),
            tuple(item.namespace_id for item in proposed),
            proposed,
        ),
    )
    selected = {capability.capability_key for capability in capabilities}
    return RCLadderResponseAuthoringBundle(
        package,
        standard,
        CandidateCapabilityCatalog(
            f"{stem}.catalog",
            tuple(
                registration
                for registration in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if registration.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
    )


@dataclass(frozen=True, slots=True)
class ResistorCapacitorCandidateContextProvider:
    bundle: RCLadderResponseAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("RC draft differs from its frozen authoring context")
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
            raise ValueError("RC package differs from its frozen authoring context")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )


__all__ = ['RCLadderResponseAuthoringBundle', 'ResistorCapacitorCandidateContextProvider', "build_rc_authoring"]
