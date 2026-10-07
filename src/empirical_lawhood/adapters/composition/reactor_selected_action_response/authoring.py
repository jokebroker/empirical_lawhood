"""Outcome-blind public authoring for separate reactor B and C issues."""

from __future__ import annotations
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld


from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign, formal_entry as _entry
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import DEVELOPMENT_ROOTS, PREFIX, ROOTS, ClassicalDesign
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_cohort import ReactorSelectedActionResponseProspectiveCohort
from empirical_lawhood.adapters.methods.reactor_selected_action_response.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_selected_action_response.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalAssay
from empirical_lawhood.adapters.methods.reactor_selected_action_response.science import CHART, CLOCK, CUTOFF, RECEIVERS, UNIT, classical_system
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
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
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
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
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution

from .protocol import study_template

BUDGET = ResourceBudget(4, 32 * 1024**3, 0, 129600, 256 * 1024**3, 64 * 1024**3)


def experiment_spec() -> ExperimentSpec:
    system, stem = classical_system(), PREFIX
    views = tuple(v.view_id for v in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, D(0))
    required, support = ObligationStatus.REQUIRED, f"{stem}.causal-chart-support"
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            32,
            2,
            CUTOFF,
            (CHART,),
            (support,),
            (),
            required,
        ),
        ValiditySpec(
            f"{stem}.validity",
            (support,),
            (
                "actual-feed-projection",
                "causal-observation-delay",
                "whole-root-numerical-validity",
            ),
            ("INCOMPLETE_NATIVE_DELIVERY", "INVALID_NUMERICAL_VIEW"),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "fixed-selected-response-interval",
            UNIT,
            D(".95"),
            tuple(sorted(RECEIVERS)),
            ("NO_OUTSIDE_CHART_PROMOTION",),
            required,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                METHOD_CAPABILITY.capability_key,
                "A violated selected-action bound, unsafe trajectory or invalid delivery defeats the local use claim.",
                "64 assigned law qualification roots: CP95 lower at least .90, both views and safety; prospective controller evaluation: 32/32 valid uses and Student lower above .0012 K.",
                required,
            ),
        ),
        ClosureSpec(
            f"{stem}.closure",
            (support,),
            ("native-integrator-timestep",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{stem}.convergence",
            ("absolute-temperature-and-paired-zero-feed-cooling",),
            views,
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{stem}.computability",
            system.computability_envelopes[0].envelope_id,
            views,
            ReadinessStatus.READY,
            (),
            (f"{stem}.bounded-native-panel",),
        ),
    )
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "An empirically nominated finite feed-response bound qualifies one pulse and survives fresh prospective use.",
        "exploration-unshifted prepared-interior callback; fixed jacket, .16 kg feed over ten seconds; two matched numerical views.",
        UNIT,
        EvidenceRung.CONTROLLER_USE,
        EvidenceCeiling.CONTROLLER_USE,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "No full-batch, exact-coefficient or competing-controller superiority claim.",
        ("fixed-root-roster", "native-delay-retained", "no-outcome-driven-features"),
        numerical_view_ids=views,
    )
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
            "64 fresh qualification scenarios, then 32 prospective scenarios conditional on supported law qualification; no replacements.",
            (support,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Matched zero-feed counterfactual at identical jacket and common causal history; audit repeats the nominated positive word.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "all-assigned-selected-action-use",
                D(".90"),
                "1",
                32,
                "law qualification CP95 on 64; prospective controller evaluation all 32 exact safe uses plus Student lower above the frozen request; no top-up.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            DEVELOPMENT_ROOTS,
            f"{stem}.fixed-evaluation-roots",
            sha256(
                canonical_json_bytes(tuple(root for root, _, _, _ in ROOTS))
            ).hexdigest(),
            tuple(
                sorted(
                    f"artifact.classical.{('assay' if role == 'qualification' else 'action')}.{root}.assay"
                    for root, role, _, _ in ROOTS
                )
            ),
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
class ClassicalAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_authoring(
    *,
    source: EmpiricalStudySource,
    native: ClassicalNativeConfig,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
    development: ObjectIdentity,
) -> ClassicalAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import (
        require_specifications,
    )

    require_specifications(specifications, count=1)
    if source.batch.fingerprint() != native.source_sha256:
        raise ValueError("classical authoring changed its exact native source")
    design, stem, system = native.design, PREFIX, classical_system()
    experiment = experiment_spec()
    campaign = _campaign(
        system,
        experiment,
        prefix=stem,
        budget=BUDGET,
        objective="Qualify one empirical response bound and prospectively deliver its selected pulse.",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    capabilities = tuple(
        sorted((METHOD_CAPABILITY, SOURCE_CAPABILITY), key=lambda v: v.registry_id)
    )
    registry = CapabilityRegistry(f"{stem}.registry", capabilities)
    template = study_template(design, native, source, experiment, registry)
    source_id = f"{stem}.source-bundle"
    source_identity = ObjectIdentity.from_record(source_id, source)
    observer = ObjectIdentity.from_record(
        SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY
    )
    qualification = MaterializationQualificationReceipt(
        f"{stem}.configuration-qualification",
        source_id,
        source_identity,
        source.fingerprint(),
        system.world.world_id,
        observer,
        tuple(v.view_id for v in system.numerical_views),
        tuple(sorted({v.native_unit for v in system.quantities})),
        tuple(sorted({v.coordinate_frame for v in system.quantities})),
        (CLOCK,),
        system.relation.relation_id,
        experiment.obligations.validity.validity_id,
        experiment.obligations.uncertainty.uncertainty_id,
        SourceAccessDisposition.VERIFIED_ACCESS,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    source_ref = SourceMaterializationRef(
        source_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        system.world.world_id,
        source_identity,
        source.fingerprint(),
        native.fingerprint(),
        observer,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        SourceAccessDisposition.VERIFIED_ACCESS,
    )
    design_identity = ObjectIdentity.from_record(design.config_id, design)
    design_inputs = [
        DesignInputRecord(
            f"{stem}.design",
            design_identity,
            design.fingerprint(),
            experiment.information_cutoffs[0],
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        )
    ]
    design_inputs.extend(
        DesignInputRecord(
            f"{stem}.specification-{index}",
            ObjectIdentity(
                f"{stem}.specification-{index}",
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
        for index, (_, digest) in enumerate(specifications)
    )
    prior_seeds = tuple(
        sorted(
            f"seed.{start + i}"
            for start, count in ((98000, 32), (98100, 16), (98200, 32), (98300, 64))
            for i in range(count)
        )
    )
    design_inputs.append(
        DesignInputRecord(
            f"{stem}.exposed-discovery",
            development,
            development.object_fingerprint,
            experiment.information_cutoffs[0],
            DesignInputRole.DEVELOPMENT_TUNING,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            "owner.empirical-lawhood",
            DEVELOPMENT_ROOTS,
            prior_seeds,
        )
    )
    inputs = tuple(sorted(design_inputs, key=lambda v: v.input_id))
    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.adequate", f"{stem}.inadequate"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(v.input_id for v in inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        inputs,
        DEVELOPMENT_ROOTS,
        tuple(sorted(root for root, _, _, _ in ROOTS)),
        prior_seeds,
        tuple(sorted(f"seed.{seed}" for _, _, _, seed in ROOTS)),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(
                v.capability_key, v.capability_version, v.implementation_sha256
            )
            for v in capabilities
        ),
        (source_ref,),
        BUDGET,
    )
    context = CandidateCompilationContext(
        f"{stem}.context",
        registry,
        (template,),
        (qualification,),
        inputs,
        implementation_sha256,
    )
    base, standard = _entry(
        draft,
        context,
        design_identity,
        prefix=stem,
        minimum_units=32,
        operand_description="Frozen selected-feed {domain} operands under the bounded ten-second response claim.",
        estimator="selected-finite-response-bound",
        uncertainty="root-cp-and-prospective-controller-evaluation-student",
        ceiling=EvidenceCeiling.LOCAL_LAW,
        evaluator=METHOD_CAPABILITY,
        input_schema=ClassicalAssay.SCHEMA,
        output_schema=ReactorSelectedActionResponseProspectiveCohort.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[ClassicalDesign | ClassicalNativeConfig, ...] = tuple(
        sorted((design, native), key=lambda v: v.SCHEMA)
    )
    by_schema = {
        d.payload_schema: d
        for b in (METHOD_BINDING, SOURCE_BINDING)
        for d in b.issued_decoder_registrations
    }
    decoders = tuple(by_schema[v.SCHEMA] for v in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{stem}.extension.{i}",
            f"{stem}.namespace.{i}",
            ObjectIdentity.from_record(v.config_id, v),
            len(v.canonical_bytes()),
            d.decoder_key,
            d.decoder_version,
            d.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, (v, d) in enumerate(zip(payloads, decoders, strict=True))
    )
    package = ExecutableStudyDefinition(
        f"{stem}.executable-study-definition",
        base,
        ProposedStudyExtensionSet(
            f"{stem}.extensions",
            ObjectIdentity.from_record(base.package_id, base),
            tuple(v.namespace_id for v in proposed),
            proposed,
        ),
    )
    evidence_registry = build_observation_evidence_world_registry()
    world = next(
        v
        for v in evidence_registry.world_profiles
        if v.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{stem}.evidence-profile",
        draft_id=draft.draft_id,
        registry=evidence_registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=tuple(sorted(tuple(EvidenceRung), key=lambda rung: rung.value)),
    )
    return ClassicalAuthoringBundle(
        package,
        standard,
        CandidateCapabilityCatalog(
            f"{stem}.catalog",
            tuple(
                v
                for v in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if v.manifest.capability_key in {c.capability_key for c in capabilities}
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )


@dataclass(frozen=True, slots=True)
class ClassicalContextProvider:
    bundle: ClassicalAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("reactor phase draft differs from its frozen context")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )

    def resolve_standard(self, package: object) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("reactor phase authoring package differs")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(
                self.bundle.standard_context.base.qualifications, (), ()
            ),
        )
