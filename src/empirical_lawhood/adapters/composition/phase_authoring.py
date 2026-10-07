"""Shared strict simulator authoring completion and exact context resolution."""

from dataclasses import dataclass
from empirical_lawhood.adapters.composition.experiment_authoring import formal_entry
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.formal_gaps import FormalGapEvidenceWorld
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection, EvidenceWorldKind, bind_profile_selection, build_observation_evidence_world_registry
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import StudyDraft, DesignInputRecord, MaterializationQualificationReceipt
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StandardCandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution


@dataclass(frozen=True, slots=True)
class PhaseAuthoringBundle:
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    payloads: tuple[CanonicalRecord, ...]
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]
    evidence_profile: EvidenceProfileSelection


def finish_simulator_authoring(
    *,
    draft: StudyDraft,
    registry: CapabilityRegistry,
    template: StudyTemplate,
    qualifications: tuple[MaterializationQualificationReceipt, ...],
    inputs: tuple[DesignInputRecord, ...],
    implementation_sha256: str,
    design_identity: ObjectIdentity,
    evaluator: CapabilityManifest,
    input_schema: str,
    evidence_units: tuple[str, ...],
    payloads: tuple[CanonicalRecord, ...],
    config_ids: tuple[str, ...],
    executable_bindings: tuple[ExecutableCapabilityBinding, ...],
    operand_description: str,
    estimator: str,
    uncertainty: str,
) -> PhaseAuthoringBundle:
    stem = draft.draft_id.removesuffix(".draft")
    experiment = draft.experiment
    if experiment is None:
        raise ValueError(
            "strict simulator authoring requires its complete experiment specification"
        )
    context = CandidateCompilationContext(
        f"{stem}.context",
        registry,
        (template,),
        tuple(sorted(qualifications, key=lambda q: q.receipt_id)),
        tuple(inputs),
        implementation_sha256,
    )
    base, standard = formal_entry(
        draft,
        context,
        design_identity,
        prefix=stem,
        minimum_units=len(evidence_units),
        operand_description=operand_description,
        estimator=estimator,
        uncertainty=uncertainty,
        # Formal estimators qualify operands through local-law evidence.
        # The experiment/profile retain controller use; their control and nested
        # evaluation owners supply the later evidence rungs.
        ceiling=(
            EvidenceCeiling.LOCAL_LAW
            if experiment.claims[0].evidence_ceiling is EvidenceCeiling.CONTROLLER_USE
            else experiment.claims[0].evidence_ceiling
        ),
        evaluator=evaluator,
        input_schema=input_schema,
        output_schema=ScientificAdjudicationRecord.SCHEMA,
        evidence_unit_ids=evidence_units,
        world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        minimum_views=2,
    )
    decoders_by_schema = {
        d.payload_schema: d for b in executable_bindings for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoders_by_schema[v.SCHEMA] for v in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{stem}.extension.{i}",
            f"{stem}.namespace.{i}",
            ObjectIdentity.from_record(config_ids[i], v),
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
            tuple(p.namespace_id for p in proposed),
            proposed,
        ),
    )
    evidence_registry = build_observation_evidence_world_registry()
    world = next(
        w
        for w in evidence_registry.world_profiles
        if w.world_kind is EvidenceWorldKind.DETERMINISTIC_SIMULATOR
    )
    evidence = bind_profile_selection(
        selection_id=f"{stem}.evidence-profile",
        draft_id=draft.draft_id,
        registry=evidence_registry,
        world_profile_id=world.profile_id,
        objective_profile_id="objective.law-control",
        requested_rungs=tuple(
            r for r in EvidenceRung if experiment.claims[0].evidence_ceiling.allows(r)
        ),
    )
    return PhaseAuthoringBundle(
        package,
        standard,
        CandidateCapabilityCatalog(
            f"{stem}.catalog",
            tuple(
                r
                for r in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if r.manifest.capability_key in {c.capability_key for c in registry.capabilities}
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )


@dataclass(frozen=True, slots=True)
class PhaseContextProvider:
    bundle: PhaseAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("phase draft differs from frozen context")
        return CandidateContextResolution(
            self.bundle.standard_context.base,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )

    def resolve_standard(self, package: object) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("phase authoring package differs")
        return StandardCandidateContextResolution(
            self.bundle.standard_context,
            (),
            CandidateSourceResolution(self.bundle.standard_context.base.qualifications, (), ()),
        )
