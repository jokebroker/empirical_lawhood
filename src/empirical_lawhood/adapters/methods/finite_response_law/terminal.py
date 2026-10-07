"Bind committed finite response-law operands to the existing assessment and qualification owners."

from typing import cast

from empirical_lawhood.adapters.methods.contracts import CandidateEvaluatorImplementation, JointUncertaintyFamilyAssessment, CandidateMethodEvidenceReceipt
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoderRegistry,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationRequest,
    LawEvaluationResult,
)
from empirical_lawhood.adapters.methods.qualification_profiles import (
    QualificationProofOwner,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.execution import DependencyReceiptBinding

from .calibration_operands import CalibrationOperands
from .frozen_package import frozen_development
from .law_family import EVIDENCE_KIND, candidate_evidence, law_family
from .law_payloads import MAXIMUM_LAW_BYTES, FiniteResponseLawConditionalDecoder, FiniteResponseLawLowerDecoder, FiniteResponseLawLowerPayload, conditional_payload, lower_payload
from .method_records import QUALIFY, FiniteResponseLawAssignedCalibrationMethodConfig, FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawAssignedQualificationReport, FiniteResponseLawCalibrationMethodConfig, FiniteResponseLawCalibrationReport, FiniteResponseLawQualificationReport
from .qualification import FiniteResponseLawQualificationProfile, calibration_metrics
from .science import PROGRAMME


def qualify_calibration(
    *,
    config: FiniteResponseLawCalibrationMethodConfig | FiniteResponseLawAssignedCalibrationMethodConfig,
    report: FiniteResponseLawCalibrationReport | FiniteResponseLawAssignedCalibrationReport,
    operands: CalibrationOperands,
    development_report_bytes: bytes,
    coefficient_bytes: bytes,
    system: SystemSpec,
    manifest: CapabilityManifest,
    payload_plane: CandidatePayloadPlane,
    dependency: DependencyReceiptBinding,
    calibration_materialization_id: str,
    prediction_materialization_id: str,
) -> FiniteResponseLawQualificationReport | FiniteResponseLawAssignedQualificationReport:
    config_id = ObjectIdentity.from_record(config.config_id, config)
    if report.config != config_id or report.boundaries != operands.boundaries:
        raise ValueError(
            "Qualification changes the committed calibration or configuration"
        )
    points, widths = frozen_development(
        report_bytes=development_report_bytes,
        coefficient_bytes=coefficient_bytes,
        report_identity=config.development_report,
        coefficient_identity=config.coefficients,
    )
    owner = QualificationProofOwner(
        f"{PROGRAMME}.fresh-calibration-owner",
        manifest.capability_key,
        manifest.capability_version,
        manifest.implementation_sha256,
    )
    capability = ObjectIdentity.from_record(manifest.capability_key, manifest)
    implementation = ObjectIdentity.from_record(owner.owner_id, owner)
    decoders = CandidatePayloadDecoderRegistry(
        (FiniteResponseLawConditionalDecoder(), FiniteResponseLawLowerDecoder())
    )
    families, qualifications, publications = [], [], []
    lower: FiniteResponseLawLowerPayload | None = None
    for calibration in report.boundaries:
        boundary = calibration.boundary
        _, binding, ledger = law_family(
            system=system,
            config=config_id,
            calibration=calibration.identity,
            boundary=boundary,
            root_ids=calibration.root_ids,
            selector_capability=capability,
            selector_implementation=implementation,
        )
        payload = (
            lower_payload(
                points,
                widths[boundary],
                payload_id=f"{PROGRAMME}.{boundary}.frozen",
                development_manifest=config.development_manifest,
                frozen_coefficients=config.coefficients,
                calibration=calibration.identity,
                q=calibration.q,
            )
            if boundary == "lower"
            else conditional_payload(
                points,
                widths[boundary],
                lower=lower if boundary == "composed" else None,
                payload_id=f"{PROGRAMME}.{boundary}.frozen",
                development_manifest=config.development_manifest,
                frozen_coefficients=config.coefficients,
                calibration=calibration.identity,
                q=calibration.q,
            )
        )
        if isinstance(payload, FiniteResponseLawLowerPayload):
            lower = payload
        decoder = (
            FiniteResponseLawLowerDecoder() if boundary == "lower" else FiniteResponseLawConditionalDecoder()
        )
        evaluator_impl = CandidateEvaluatorImplementation(
            f"{PROGRAMME}.{boundary}.implementation",
            manifest.capability_key,
            manifest.capability_version,
            ledger.method_key,
            manifest.implementation_sha256,
        )
        raw = payload.canonical_bytes()
        artifact = ArtifactIdentity(
            payload.payload_id,
            "law-payload",
            payload.SCHEMA,
            payload.fingerprint(),
            "application/json",
            len(raw),
        )
        evaluator = ExecutableReference(
            f"{PROGRAMME}.{boundary}.evaluator",
            manifest.capability_key,
            manifest.capability_version,
            ledger.method_key,
            artifact,
            SafePayloadFormat.CANONICAL_JSON,
            LawEvaluationRequest.SCHEMA,
            LawEvaluationResult.SCHEMA,
            True,
        )
        publication = payload_plane.publish_candidate_payload(
            payload=raw,
            evaluator=evaluator,
            implementation=ObjectIdentity.from_record(
                evaluator_impl.implementation_id, evaluator_impl
            ),
            decoder_schema=decoder.decoder_schema,
            decoder_version=decoder.decoder_version,
            maximum_decode_bytes=MAXIMUM_LAW_BYTES,
        )
        artifact_ids = tuple(
            sorted(
                (
                    calibration.prediction_artifact.artifact_id,
                    config.development_manifest.artifact_id,
                    config.coefficients.artifact_id,
                )
            )
        )
        links = tuple(
            EvidenceLink(
                f"{PROGRAMME}.{boundary}.evidence-{i}",
                EvidenceRelation.DERIVED_FROM,
                source,
                ObjectIdentity.from_record(
                    system.relation.relation_id, system.relation
                ),
                artifact_ids,
                system.world.world_id,
                ledger.obligation_template.information_cutoff_id,
                OutcomeAccess.EVALUATOR_REVEAL,
                VisibilityCeiling.PROSPECTIVE,
                (VisibilityCeiling.PROSPECTIVE,),
                "Fresh complete-root calibration with authenticated native receipt and frozen development operands.",
            )
            for i, source in enumerate(
                (
                    calibration.identity,
                    calibration.native_evaluation,
                    calibration.native_task_receipt,
                )
            )
        )
        evidence = CandidateMethodEvidenceReceipt(
            dependency.receipt_id,
            EVIDENCE_KIND,
            calibration_metrics(calibration),
            artifact_ids,
            tuple(e.link_id for e in links),
            (),
        )
        candidate = candidate_evidence(
            ledger, binding, publication, evidence, links, boundary=boundary
        )
        profile = FiniteResponseLawQualificationProfile(
            owner,
            boundary,
            calibration,
            payload,
            operands,
            dependency,
            calibration_materialization_id,
            prediction_materialization_id,
        )
        assessment = LawAssessmentAssembler(
            payload_plane, decoders, QualificationProfileEvaluatorRegistry((profile,))
        ).assemble(
            system=system,
            candidate=candidate,
            qualification_profile=ObjectIdentity.from_record(
                profile.profile.profile_id, profile.profile
            ),
        )
        family = cast(
            JointUncertaintyFamilyAssessment,
            CandidateFamilyAssembler().assemble(ledger, (assessment,)),
        )
        result = ResponseLawQualificationService(payload_plane, decoders).qualify(
            system, calibration.identity, family
        )
        families.append(family)
        qualifications.append(result)
        publications.append(publication)
    result_type = (
        FiniteResponseLawAssignedQualificationReport
        if type(report) is FiniteResponseLawAssignedCalibrationReport
        else FiniteResponseLawQualificationReport
    )
    return result_type(
        f"{QUALIFY}.result",
        config_id,
        report,
        tuple(families),
        tuple(qualifications),
        tuple(publications),
    )
