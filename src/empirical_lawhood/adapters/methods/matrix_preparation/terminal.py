"""The existing family/assessment/finalizer route for finite development models."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.contracts import CandidateEvaluatorImplementation, ComponentUncertaintyFamilyAssessment
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoderRegistry,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationRequest, LawEvaluationResult
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .contracts import CANDIDATES
from .data import ProjectedContext
from .family import preparation_candidate_evidence, preparation_family
from .fitting import read_fit
from .law import LAW_KEY, LAW_MAXIMUM_BYTES, PreparationLawDecoder, build_law_payload
from .models import root_roles
from .qualification import PreparationQualificationProfile
from .records import PreparationAssessmentConfig, PreparationDescriptionReport, PreparationFitReport, PreparationNumericalSemanticsReport


@dataclass(frozen=True, slots=True)
class PreparationLawReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-law-report'
    report_id: str
    context: str
    config: ObjectIdentity
    description: ObjectIdentity
    scope: QualificationScopeSpec
    claim_units: ClaimUnitBinding
    family: ComponentUncertaintyFamilyAssessment
    qualification: LawQualificationResult

    def __post_init__(self) -> None:
        if (
            self.report_id != f"{DEVELOPMENT}.law.{self.context}"
            or self.context not in ("assembling", "prepared")
            or self.config.object_schema != PreparationAssessmentConfig.SCHEMA
            or self.description.object_schema != PreparationDescriptionReport.SCHEMA
            or self.qualification.dataset_or_projection != self.description
            or self.family.ledger.dataset_or_projection != self.description
            or self.qualification.candidate_family_assessment
            != ObjectIdentity.from_record(self.family.assessment_id, self.family)
            or self.claim_units.scope != ObjectIdentity.from_record(self.scope.scope_id, self.scope)
            or self.qualification.claim_unit_binding
            != ObjectIdentity.from_record(self.claim_units.binding_id, self.claim_units)
            or self.family.ledger.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("finite law report changes the sole finalizer or exposed-root lineage")


def qualify_preparation_description(
    *,
    config: PreparationAssessmentConfig,
    fit: PreparationFitReport,
    fit_payload: bytes,
    fit_artifact: ArtifactIdentity,
    description: PreparationDescriptionReport,
    description_artifact_id: str,
    numerical_semantics: PreparationNumericalSemanticsReport,
    observed: ProjectedContext,
    payload_plane: CandidatePayloadPlane,
    profile_owner: QualificationProofOwner,
    evaluator_implementation: CandidateEvaluatorImplementation,
    selector_capability: ObjectIdentity,
    selector_implementation: ObjectIdentity,
) -> PreparationLawReport:
    if (
        description.fit != ObjectIdentity.from_record(fit.report_id, fit)
        or description.input_reports != observed.identities
        or observed.role != "observable"
        or description.numerical_semantics != ObjectIdentity.from_record(numerical_semantics.report_id, numerical_semantics)
        or fit_artifact.sha256 != fit.data_sha256
        or fit_artifact.size_bytes != len(fit_payload)
        or evaluator_implementation.evaluator_key != LAW_KEY
    ):
        raise ValueError("finite law qualification changes authenticated observable/model inputs")
    groups = read_fit(fit, fit_payload)
    report_ref = ObjectIdentity.from_record(description.report_id, description)
    scope, binding, ledger = preparation_family(
        system=config.system,
        config=config.method,
        context=fit.context,
        projection=report_ref,
        selector_capability=selector_capability,
        selector_implementation=selector_implementation,
    )
    evidence = EvidenceLink(
        f"evidence-link.{description.report_id}",
        EvidenceRelation.DERIVED_FROM,
        report_ref,
        ObjectIdentity.from_record(config.system.relation.relation_id, config.system.relation),
        tuple(sorted((fit_artifact.artifact_id, description_artifact_id))),
        config.system.world.world_id,
        ledger.obligation_template.information_cutoff_id,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.OUTCOME_VISIBLE,
        (VisibilityCeiling.OUTCOME_VISIBLE,),
        "Frozen causal fits, exposed held-root description errors and numerical checks; no protected confirmation or operational controller admission.",
    )
    implementation = ObjectIdentity.from_record(
        evaluator_implementation.implementation_id, evaluator_implementation
    )
    decoder, profile = PreparationLawDecoder(), PreparationQualificationProfile(profile_owner)
    decoders = CandidatePayloadDecoderRegistry((decoder,))
    assembler = LawAssessmentAssembler(
        payload_plane, decoders, QualificationProfileEvaluatorRegistry((profile,))
    )
    roles = root_roles(observed.roots)
    training, screen = roles == 0, roles == 2
    known = np.isfinite(observed.arrays["response"][screen]).all(axis=(2, 3, 4, 5))
    common = (
        *numerical_semantics.metrics,
        NamedDecimal(
            "native-numerical-semantics-qualified",
            Decimal(int(numerical_semantics.numerical_semantics_qualified)),
            "1",
        ),
        NamedDecimal("minimum-audit-bundles", Decimal(4 if known.all() else 0), "1"),
        NamedDecimal("minimum-parent-screen-roots", Decimal(int(known.sum(axis=0).min())), "1"),
    )
    assessments = []
    for candidate in CANDIDATES:
        payload = build_law_payload(
            candidate,
            fit,
            fit_artifact,
            groups.get(candidate),
            observed.arrays["handoff_features"][training],
        )
        artifact = ArtifactIdentity(
            f"payload.{payload.payload_id}",
            "finite-observable-law",
            payload.SCHEMA,
            payload.fingerprint(),
            "application/json",
            len(payload.canonical_bytes()),
        )
        evaluator = ExecutableReference(
            f"evaluator.{payload.payload_id}",
            evaluator_implementation.capability_key,
            evaluator_implementation.capability_version,
            LAW_KEY,
            artifact,
            SafePayloadFormat.CANONICAL_JSON,
            LawEvaluationRequest.SCHEMA,
            LawEvaluationResult.SCHEMA,
            True,
        )
        publication = payload_plane.publish_candidate_payload(
            payload=payload.canonical_bytes(),
            evaluator=evaluator,
            implementation=implementation,
            decoder_schema=decoder.decoder_schema,
            decoder_version=decoder.decoder_version,
            maximum_decode_bytes=LAW_MAXIMUM_BYTES,
        )
        metrics = tuple(
            sorted(
                (
                    *common,
                    *next(s.metrics for s in description.screens if s.candidate == candidate),
                ),
                key=lambda v: v.value_id,
            )
        )
        supplied = preparation_candidate_evidence(
            ledger, binding, payload, publication, metrics, (evidence,)
        )
        assessments.append(
            assembler.assemble(
                system=config.system,
                candidate=supplied,
                qualification_profile=ObjectIdentity.from_record(
                    profile.profile.profile_id, profile.profile
                ),
                metrics=tuple(
                    NamedDecimal(f"{supplied.candidate_id}.{v.value_id}", v.value, v.unit)
                    for v in metrics
                ),
            )
        )
    family = CandidateFamilyAssembler().assemble(ledger, tuple(assessments))
    qualification = ResponseLawQualificationService(payload_plane, decoders).qualify(
        config.system, report_ref, family
    )
    return PreparationLawReport(
        f"{DEVELOPMENT}.law.{fit.context}",
        fit.context,
        ObjectIdentity.from_record(config.config_id, config),
        report_ref,
        scope,
        binding,
        family,
        qualification,
    )
