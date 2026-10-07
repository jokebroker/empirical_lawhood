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
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.science import CLOCK, CUTOFF
from empirical_lawhood.adapters.methods.reactor_regime_response.config import PREFIX, ROOTS, ReactorRegimeResponseDesign
from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import RegimeCContinuation
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_cohort import ReactorRegimeResponseCohortProspective
from empirical_lawhood.adapters.methods.reactor_regime_response.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_regime_response.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_regime_response.law_terminal import RegimeJointLawResult
from empirical_lawhood.adapters.methods.reactor_regime_response.nomination_records import RegimeNominationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel
from empirical_lawhood.adapters.methods.reactor_regime_response.science import CHART, RECEIVERS, UNIT, regime_system
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import ReactorRegimeNativeConfig
from empirical_lawhood.adapters.simulators.reactor_regime_response.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_regime_response.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
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
from empirical_lawhood.kernel.references import ArtifactIdentity
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

from .continuation import bind_c_continuation
from .protocol import phase_template

PHASE_BUDGET = ResourceBudget(4, 32 * 1024**3, 0, 129600, 256 * 1024**3, 256 * 1024**3)


def _experiment(phase: str) -> ExperimentSpec:
    system = regime_system()
    stem = f"{PREFIX}.phase-{phase.lower()}"
    phase_roles = (
        ("fit", "nomination")
        if phase == "B"
        else ("calibration", "qualification")
        if phase == "C"
        else ("prospective",)
    )
    roots = tuple(root for root, role, _, _ in ROOTS if role in phase_roles)
    fit_roots = tuple(root for root, role, _, _ in ROOTS if role == "fit")
    evaluated = tuple(
        root
        for root, role, _, _ in ROOTS
        if role
        == (
            "nomination"
            if phase == "B"
            else "qualification"
            if phase == "C"
            else "prospective"
        )
    )
    views = tuple(view.view_id for view in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, D(0))
    required = ObligationStatus.REQUIRED
    support = f"{stem}.causal-chart-support"
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            len(evaluated),
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
            "frozen-action-specific-root-max-bounds",
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
                "Invalid native views, insufficient causal contact or failed frozen joint adequacy defeats the bounded claim.",
                "Retain every assigned root and both views; C needs one-sided CP lower bound at least .90 for joint endpoint and selected preparation.",
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
    rung = (
        EvidenceRung.RESPONSE
        if phase == "B"
        else EvidenceRung.LOCAL_LAW
        if phase == "C"
        else EvidenceRung.CONTROLLER_USE
    )
    ceiling = (
        EvidenceCeiling.RESPONSE
        if phase == "B"
        else EvidenceCeiling.LOCAL_LAW
        if phase == "C"
        else EvidenceCeiling.CONTROLLER_USE
    )
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        "Freeze fit and nomination before independent calibration"
        if phase == "B"
        else "Test the one nominated joint local response law and fixed preparation on 64 fresh roots"
        if phase == "C"
        else "Prospectively deliver four separate paired cooling requests from each qualified prepared root",
        "Matched native absolute peak and paired zero-feed cooling under the fixed exploration-unshifted preparation, separate causal histories and both numerical views.",
        UNIT,
        rung,
        ceiling,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "B nominates only; C requires its predeclared all-assigned joint law and selected-preparation CP criterion.",
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
            "Exactly the allocated roots and frozen native preparation; D has four separate common-parent paired cooling requests and no replacement roots.",
            (support,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Frozen K/S/L, absolute, smooth, history and probe contrasts on root-paired native words; no comparator can change the primary law after nomination.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "all-assigned-joint-adequacy-and-preparation",
                D(".90"),
                "1",
                len(evaluated),
                "At C exactly 64 assignments, no top-up or retuning; B's 16 nominations only freeze the finite selected route.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            fit_roots
            if phase == "B"
            else tuple(
                root
                for root, role, _, _ in ROOTS
                if role == ("calibration" if phase == "C" else "prospective")
            ),
            f"{stem}.fixed-evaluation-roots",
            sha256(canonical_json_bytes(evaluated)).hexdigest(),
            tuple(
                sorted(
                    f"artifact.{stem}.regime.{('action' if phase == 'D' else 'assay')}.{root}.{('native' if phase == 'D' else 'assay')}"
                    for root in roots
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
class RegimePhaseAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def build_phase_authoring(
    *,
    phase: str,
    source: EmpiricalStudySource,
    native: ReactorRegimeNativeConfig,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
    prior_fit: ArtifactIdentity | None = None,
    prior_nomination: ArtifactIdentity | None = None,
    prior_discovery: ArtifactIdentity | None = None,
    prior_calibration: ArtifactIdentity | None = None,
    prior_qualification: ArtifactIdentity | None = None,
    prior_law: ArtifactIdentity | None = None,
    phase_a: ObjectIdentity,
    corrected_phase_a_sha256: str | None = None,
    continuation: RegimeCContinuation | None = None,
) -> RegimePhaseAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import (
        require_specifications,
    )

    require_specifications(specifications, count=1)
    if phase not in ("B", "C", "D"):
        raise ValueError("reactor authoring requires phase B, C or D")
    if source.batch.fingerprint() != native.source_sha256:
        raise ValueError("reactor source bytes differ from the native pin")
    design = native.design
    stem = f"{PREFIX}.phase-{phase.lower()}"
    system = regime_system()
    experiment = _experiment(phase)
    campaign = _campaign(
        system,
        experiment,
        prefix=stem,
        budget=PHASE_BUDGET,
        objective="Freeze B fit/nomination"
        if phase == "B"
        else "Qualify the fixed joint law and selected preparation on independent C roots"
        if phase == "C"
        else "Deliver and evaluate four paired cooling requests from independent D roots",
        actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
    )
    capabilities = tuple(
        sorted(
            (METHOD_CAPABILITY, SOURCE_CAPABILITY), key=lambda value: value.registry_id
        )
    )
    registry = CapabilityRegistry(f"{stem}.registry", capabilities)
    template = phase_template(
        phase,
        design,
        native,
        source,
        experiment,
        registry,
        prior_fit=prior_fit,
        prior_nomination=prior_nomination,
        prior_discovery=prior_discovery,
        prior_calibration=prior_calibration,
        prior_qualification=prior_qualification,
        prior_law=prior_law,
    )
    if continuation is not None:
        if phase != "C":
            raise ValueError("only C has an authorized retained-input continuation")
        template = bind_c_continuation(template, experiment, continuation)
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
        tuple(value.view_id for value in system.numerical_views),
        tuple(sorted({value.native_unit for value in system.quantities})),
        tuple(sorted({value.coordinate_frame for value in system.quantities})),
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
    design_inputs.append(
        DesignInputRecord(
            f"{stem}.exposed-phase-a",
            phase_a,
            phase_a.object_fingerprint,
            experiment.information_cutoffs[0],
            DesignInputRole.DEVELOPMENT_TUNING,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            "owner.empirical-lawhood",
        )
    )
    if corrected_phase_a_sha256 is not None:
        design_inputs.append(
            DesignInputRecord(
                f"{stem}.exposed-phase-a-corrected",
                ObjectIdentity(
                    f"{stem}.exposed-phase-a-corrected",
                    "empirical-lawhood/analysis/reactor-regime-response-corrected-phase-a",
                    "1.0.0",
                    corrected_phase_a_sha256,
                ),
                corrected_phase_a_sha256,
                experiment.information_cutoffs[0],
                DesignInputRole.DEVELOPMENT_TUNING,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                "owner.empirical-lawhood",
            )
        )
    qualifications = [qualification]
    sources = [source_ref]
    if continuation is not None:
        continuation_identity = ObjectIdentity.from_record(
            continuation.continuation_id, continuation
        )
        design_inputs.append(
            DesignInputRecord(
                f"{stem}.retained-input-continuation",
                continuation_identity,
                continuation.fingerprint(),
                experiment.information_cutoffs[0],
                DesignInputRole.READINESS_METADATA,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                "owner.empirical-lawhood",
            )
        )
        # These are the original graph's sealed OUTCOME operands, authenticated
        # by receipt-bound provider inputs. They are not new source acquisitions
        # or model configurations and must not receive fabricated source receipts.
    for label, artifact in (
        ("fit", prior_fit),
        ("nomination", prior_nomination),
        ("discovery", prior_discovery),
        ("calibration", prior_calibration),
        ("qualification", prior_qualification),
        ("law", prior_law),
    ):
        if artifact is not None:
            contributing_roles = (
                ("fit", "nomination")
                if label in ("fit", "nomination", "discovery")
                else ("calibration", "qualification")
            )
            prior_identity = ObjectIdentity(
                artifact.artifact_id, artifact.payload_schema, "1.0.0", artifact.sha256
            )
            design_inputs.append(
                DesignInputRecord(
                    f"{stem}.prior-{label}",
                    prior_identity,
                    artifact.sha256,
                    experiment.information_cutoffs[0],
                    DesignInputRole.DEVELOPMENT_TUNING,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    "owner.empirical-lawhood",
                    tuple(
                        sorted(
                            root
                            for root, role, _, _ in ROOTS
                            if role in contributing_roles
                        )
                    ),
                    tuple(
                        sorted(
                            f"seed.{seed}"
                            for _, role, _, seed in ROOTS
                            if role in contributing_roles
                        )
                    ),
                )
            )
            prior_receipt = MaterializationQualificationReceipt(
                f"{stem}.prior-{label}-qualification",
                artifact.artifact_id,
                prior_identity,
                artifact.sha256,
                system.world.world_id,
                ObjectIdentity.from_record(
                    METHOD_CAPABILITY.capability_key, METHOD_CAPABILITY
                ),
                tuple(value.view_id for value in system.numerical_views),
                tuple(sorted({value.native_unit for value in system.quantities})),
                tuple(sorted({value.coordinate_frame for value in system.quantities})),
                (CLOCK,),
                system.relation.relation_id,
                experiment.obligations.validity.validity_id,
                experiment.obligations.uncertainty.uncertainty_id,
                SourceAccessDisposition.VERIFIED_ACCESS,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            )
            qualifications.append(prior_receipt)
            sources.append(
                SourceMaterializationRef(
                    artifact.artifact_id,
                    SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                    system.world.world_id,
                    prior_identity,
                    artifact.sha256,
                    design.fingerprint(),
                    ObjectIdentity.from_record(
                        METHOD_CAPABILITY.capability_key, METHOD_CAPABILITY
                    ),
                    prior_receipt.numerical_view_ids,
                    ObjectIdentity.from_record(prior_receipt.receipt_id, prior_receipt),
                    SourceAccessDisposition.VERIFIED_ACCESS,
                )
            )
    design_inputs.sort(key=lambda value: value.input_id)
    prior_roles = (
        ()
        if phase == "B"
        else ("fit", "nomination")
        if phase == "C"
        else ("fit", "nomination", "calibration", "qualification")
    )
    current_roles = (
        ("fit", "nomination")
        if phase == "B"
        else ("calibration", "qualification")
        if phase == "C"
        else ("prospective",)
    )
    prior_units = tuple(
        sorted(root for root, role, _, _ in ROOTS if role in prior_roles)
    )
    prior_seeds = tuple(
        sorted(f"seed.{seed}" for _, role, _, seed in ROOTS if role in prior_roles)
    )
    units = tuple(sorted(root for root, role, _, _ in ROOTS if role in current_roles))
    seeds = tuple(
        sorted(f"seed.{seed}" for _, role, _, seed in ROOTS if role in current_roles)
    )
    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.adequate", f"{stem}.inadequate"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(value.input_id for value in design_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        tuple(design_inputs),
        prior_units,
        units,
        prior_seeds,
        seeds,
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
        tuple(sorted(sources, key=lambda value: value.source_id)),
        PHASE_BUDGET,
    )
    context = CandidateCompilationContext(
        f"{stem}.context",
        registry,
        (template,),
        tuple(sorted(qualifications, key=lambda value: value.receipt_id)),
        tuple(design_inputs),
        implementation_sha256,
    )
    base, standard = _entry(
        draft,
        context,
        design_identity,
        prefix=stem,
        minimum_units=16 if phase == "B" else 64,
        operand_description="Frozen {domain} joint temperature/cooling operands on assigned roots, with causal chart support and both numerical views.",
        estimator="frozen-causal-k-s-l-and-information-law",
        uncertainty="action-specific-root-max-calibration",
        ceiling=EvidenceCeiling.RESPONSE
        if phase == "B"
        else EvidenceCeiling.LOCAL_LAW,
        evaluator=METHOD_CAPABILITY,
        input_schema=RegimeDNativeRoot.SCHEMA
        if phase == "D"
        else RegimeAssayPanel.SCHEMA,
        output_schema=RegimeNominationPackage.SCHEMA
        if phase == "B"
        else RegimeJointLawResult.SCHEMA
        if phase == "C"
        else ReactorRegimeResponseCohortProspective.SCHEMA,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    payloads: tuple[
        ReactorRegimeResponseDesign | ReactorRegimeNativeConfig, ...
    ] = tuple(sorted((design, native), key=lambda value: value.SCHEMA))
    decoder_by_schema = {
        decoder.payload_schema: decoder
        for binding in (METHOD_BINDING, SOURCE_BINDING)
        for decoder in binding.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[value.SCHEMA] for value in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{stem}.extension.{index}",
            f"{stem}.namespace.{index}",
            ObjectIdentity.from_record(value.config_id, value),
            len(value.canonical_bytes()),
            decoder.decoder_key,
            decoder.decoder_version,
            decoder.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for index, (value, decoder) in enumerate(zip(payloads, decoders, strict=True))
    )
    package = ExecutableStudyDefinition(
        f"{stem}.executable-study-definition",
        base,
        ProposedStudyExtensionSet(
            f"{stem}.extensions",
            ObjectIdentity.from_record(base.package_id, base),
            tuple(value.namespace_id for value in proposed),
            proposed,
        ),
    )
    evidence_registry = build_observation_evidence_world_registry()
    world_profile = next(
        value
        for value in evidence_registry.world_profiles
        if value.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{stem}.evidence-profile",
        draft_id=draft.draft_id,
        registry=evidence_registry,
        world_profile_id=world_profile.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=(
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
        )
        if phase == "B"
        else (
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        )
        if phase == "C"
        else (
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
            EvidenceRung.ADMISSION,
            EvidenceRung.CONTROLLER_USE,
        ),
    )
    selected = {value.capability_key for value in capabilities}
    return RegimePhaseAuthoringBundle(
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
        evidence,
    )


@dataclass(frozen=True, slots=True)
class RegimePhaseContextProvider:
    bundle: RegimePhaseAuthoringBundle

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
