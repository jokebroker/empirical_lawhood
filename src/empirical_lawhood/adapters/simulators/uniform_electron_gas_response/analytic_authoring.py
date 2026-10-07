"""Prospective graph for a single disclosed uniform electron gas analytic-development unit."""

from dataclasses import dataclass
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
from empirical_lawhood.adapters.methods.uniform_electron_gas_analytic_reference.executable_binding import BINDING as EVALUATOR_BINDING
from empirical_lawhood.adapters.methods.uniform_electron_gas_analytic_reference.extension_bundle import CAPABILITY as EVALUATOR_CAPABILITY
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

from .analytic_contracts import UniformElectronGasAnalyticCheck, UniformElectronGasAnalyticEvaluationConfig, UniformElectronGasAnalyticPanel
from .analytic_design import ACTION, BUDGET, CHART, CLOCK, CUTOFF, DENOMINATOR, RECEIVER, SUPPORT, UNIT, analytic_system
from .executable_binding import BINDING as NATIVE_BINDING
from .extension_bundle import CAPABILITY as NATIVE_CAPABILITY
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig
from .runtime_provider import native_task_id


def _experiment(
    system: SystemSpec, config: UniformElectronGasAnalyticReferenceConfig, *, experiment_id: str
) -> ExperimentSpec:
    stem = experiment_id
    required = ObligationStatus.REQUIRED
    view_ids = tuple(view.view_id for view in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, Decimal(0))
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            1,
            1,
            CUTOFF,
            (CHART,),
            (DENOMINATOR,),
            (),
            required,
        ),
        ValiditySpec(
            f"{stem}.validity",
            (SUPPORT,),
            (
                "analytic-positive-reference",
                "frozen-ueg-denominator",
                "static-transverse-chart",
            ),
            ("EXTERNAL_GAUGE_CLOSED_SOURCE_UNOBSERVED",),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "deterministic-reference-bound",
            UNIT,
            Decimal("0.95"),
            (RECEIVER,),
            ("NO_EMPIRICAL_CONFIDENCE_FROM_SYNTHETIC_UNIT",),
            ObligationStatus.NOT_APPLICABLE,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.RECEIVER_GATE,
                EVALUATOR_CAPABILITY.capability_key,
                "Signed realized-SI current, q intercept and slab inverse must agree with the frozen chart.",
                "Fail wrong sign, action drift, unstable q limit or slab mismatch without retuning.",
                required,
            ),
        ),
        ClosureSpec(
            f"{stem}.closure", (SUPPORT,), ("signed-transverse-u",), (), required
        ),
        StructuralConvergenceSpec(
            f"{stem}.convergence",
            ("finite-q-even-fit",),
            view_ids,
            (),
            ObligationStatus.NOT_APPLICABLE,
        ),
        ComputabilityEvidence(
            f"{stem}.computability-evidence",
            system.computability_envelopes[0].envelope_id,
            view_ids,
            ReadinessStatus.READY,
            (),
            ("bounded-local-cpu",),
        ),
    )
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "The disclosed analytic UEG acquisition satisfies its frozen signed finite-q and slab-reference checks.",
        "Realized-SI transverse kernel, four-q intercept stability and slab-centre inverse for one synthetic unit.",
        UNIT,
        EvidenceRung.MEASUREMENT,
        EvidenceCeiling.MEASUREMENT,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Analytic-development conformance only; no material order, physical measurement or transverse response-receiver admission claim.",
        ("analytic-reference-only", "one-synthetic-acquisition"),
        numerical_view_ids=view_ids,
    )
    task_id = native_task_id(config)
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
            (ACTION,),
            "One frozen analytic source applies the signed q/u chart; conditions are nested.",
            (SUPPORT,),
        ),
        (RECEIVER,),
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                EVALUATOR_CAPABILITY.capability_key,
                (RECEIVER,),
                "The signed zero-action current and even remainder must stay below the frozen bounds.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "q-intercept-stability-relative",
                config.threshold("q_intercept_stability_relative"),
                "dimensionless",
                1,
                "Stop at the frozen q and slab limits for the one synthetic acquisition.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            (f"{stem}.free-electron-derivation",),
            f"{stem}.synthetic-evaluation",
            sha256(canonical_json_bytes((config.independent_unit_id,))).hexdigest(),
            (f"artifact.{stem}.{task_id}.native-panel",),
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
class UniformElectronGasAnalyticAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]


def build_uniform_electron_gas_analytic_authoring(
    *, config: UniformElectronGasAnalyticReferenceConfig, experiment_id: str, implementation_sha256: str
) -> UniformElectronGasAnalyticAuthoringBundle:
    if experiment_id == config.config_id:
        raise ValueError("uniform electron gas experiment identity must differ from source config")
    stem = experiment_id
    evaluation = UniformElectronGasAnalyticEvaluationConfig(
        f"{stem}.evaluation-config", config, True
    )
    system = analytic_system(config, experiment_id=stem)
    experiment = _experiment(system, config, experiment_id=stem)
    campaign = single_experiment_campaign(
        system,
        experiment,
        prefix=stem,
        budget=BUDGET,
        objective="Check one synthetic UEG analytic source without a physical-material claim.",
        actions=(
            ("reference", AuthorityAction.REFERENCE_WORLD_EXECUTION),
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
        ),
    )
    capabilities = tuple(
        sorted(
            (NATIVE_CAPABILITY, EVALUATOR_CAPABILITY),
            key=lambda value: value.registry_id,
        )
    )
    registry = CapabilityRegistry(f"{stem}.registry", capabilities)
    native_task = native_task_id(config)
    evaluate_task = f"{stem}.finite-q-evaluate"
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
            (native_task,),
            protocol_outputs(
                (
                    ("analytic-check", UniformElectronGasAnalyticCheck.SCHEMA),
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
        ProtocolStepTemplate(
            native_task,
            ScientificStage.PREPARE,
            NATIVE_CAPABILITY.capability_key,
            NATIVE_CAPABILITY.capability_version,
            capability_config_ref(
                config,
                ObjectIdentity.from_record(config.config_id, config),
                NATIVE_CAPABILITY,
            ),
            (),
            protocol_outputs((("native-panel", UniformElectronGasAnalyticPanel.SCHEMA),)),
            NATIVE_CAPABILITY.permissions,
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            NATIVE_CAPABILITY.resource_ceiling,
            (),
            BarrierKind.FREEZE,
            1,
            (f"{stem}.native-operands",),
        ),
    )
    protocol = ProtocolTemplate(
        f"{stem}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda value: value.step_id)),
        False,
        False,
        True,
    )
    template = study_template_from_protocol(
        protocol,
        config,
        f"{stem}.source-config",
        experiment,
        registry,
        prefix=stem,
        source_task_id=native_task,
        terminal_task_id=evaluate_task,
        primary_output_ids={
            ScientificStage.PREPARE: "native-panel",
            ScientificStage.QUALIFY: "analytic-check",
            ScientificStage.EVALUATE: "analytic-check",
        },
    )
    source_identity = ObjectIdentity.from_record(f"{stem}.source-config", config)
    observer = ObjectIdentity.from_record(
        NATIVE_CAPABILITY.capability_key, NATIVE_CAPABILITY
    )
    qualification = MaterializationQualificationReceipt(
        f"{stem}.source-qualification",
        source_identity.object_id,
        source_identity,
        config.fingerprint(),
        system.world.world_id,
        observer,
        tuple(view.view_id for view in system.numerical_views),
        tuple(sorted({quantity.native_unit for quantity in system.quantities})),
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
        config.fingerprint(),
        config.fingerprint(),
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
        (f"{stem}.analytic-conformant", f"{stem}.analytic-falsified"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            (design_input.input_id,),
            VisibilityCeiling.PROSPECTIVE,
        ),
        (design_input,),
        (f"{stem}.free-electron-derivation",),
        (config.independent_unit_id,),
        (),
        (),
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
            for value in capabilities
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
        operand_description="Analytic {domain} operands for one disclosed UEG source and 20 nested q/u conditions; no material claim.",
        estimator="signed-finite-q-intercept-and-slab-inverse",
        uncertainty="deterministic-reference-bound",
        ceiling=EvidenceCeiling.MEASUREMENT,
        evaluator=EVALUATOR_CAPABILITY,
        input_schema=UniformElectronGasAnalyticPanel.SCHEMA,
        output_schema=UniformElectronGasAnalyticCheck.SCHEMA,
        world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        minimum_views=1,
    )
    payloads: tuple[CanonicalRecord, ...] = tuple(
        sorted((evaluation, config), key=lambda value: value.SCHEMA)
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
            ObjectIdentity.from_record(record.config_id, record),
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
        f"{stem}.executable-study-definition",
        base,
        ProposedStudyExtensionSet(
            f"{stem}.extensions",
            ObjectIdentity.from_record(base.package_id, base),
            tuple(item.namespace_id for item in proposed),
            proposed,
        ),
    )
    selected = {value.capability_key for value in capabilities}
    return UniformElectronGasAnalyticAuthoringBundle(
        package,
        standard,
        CandidateCapabilityCatalog(
            f"{stem}.catalog",
            tuple(
                value
                for value in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if value.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
    )


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticCandidateContextProvider:
    bundle: UniformElectronGasAnalyticAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("uniform electron gas analytic draft differs from frozen authoring")
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
            raise ValueError("uniform electron gas analytic package differs from frozen authoring")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )


__all__ = [
    'UniformElectronGasAnalyticAuthoringBundle',
    'UniformElectronGasAnalyticCandidateContextProvider',
    "build_uniform_electron_gas_analytic_authoring",
]
