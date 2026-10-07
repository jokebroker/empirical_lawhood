"""Authenticated operands enter the existing candidate and sole law finalizers."""

from empirical_lawhood.adapters.methods.contracts import (
    CandidateEvaluatorImplementation,
    CandidateMethodEvidenceReceipt,
)
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoderRegistry,
    CandidatePayloadPublisher,
    CandidatePayloadReader,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationRequest, LawEvaluationResult
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.methods.reactor_prefix_response.batch_calibration import ReactorForecastCalibration
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from .law import EVIDENCE_KIND, MAXIMUM_LAW_BYTES, ReactorForecastDecoder, ReactorForecastPayload, candidate_evidence, forecast_family
from .qualification import ReactorForecastQualificationProfile, calibration_metrics
from .records import ReactorBatchCustody, ReactorForecastResult
from .science import CUTOFF, METHOD, PREFIX, ReactorForecastDesign, forecast_system


def qualify_forecasts(
    design: ReactorForecastDesign,
    calibration: ReactorForecastCalibration,
    custody: ReactorBatchCustody,
    source_sha256: str,
    manifest: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> ReactorForecastResult:
    if source_sha256 != design.native.source_bundle_sha256:
        raise ValueError("qualification substitutes its pinned native source bundle")
    design_id = ObjectIdentity.from_record(design.config_id, design)
    calibration_id = ObjectIdentity.from_record(f"{PREFIX}.calibration", calibration)
    expected = tuple(t.object_fingerprint for s in calibration.unit_scores for t in s.traces)
    if (
        expected != tuple(m.logical.content_sha256 for m in custody.manifests)
        or design_artifact.sha256 != design.fingerprint()
    ):
        raise ValueError("qualification changes committed trace or design operands")
    system = forecast_system(design)
    owner = QualificationProofOwner(
        f"{PREFIX}.proof-owner",
        manifest.capability_key,
        manifest.capability_version,
        manifest.implementation_sha256,
    )
    binding, ledger = forecast_family(
        system,
        design_id,
        calibration_id,
        ObjectIdentity.from_record(manifest.capability_key, manifest),
        ObjectIdentity.from_record(owner.owner_id, owner),
    )
    payload = ReactorForecastPayload(design_id, calibration_id, source_sha256, calibration.bounds)
    decoder = ReactorForecastDecoder()
    impl = CandidateEvaluatorImplementation(
        f"{PREFIX}.implementation",
        manifest.capability_key,
        manifest.capability_version,
        METHOD,
        manifest.implementation_sha256,
    )
    raw = payload.canonical_bytes()
    artifact = ArtifactIdentity(
        f"{PREFIX}.forecast-payload",
        "law-payload",
        payload.SCHEMA,
        payload.fingerprint(),
        "application/json",
        len(raw),
    )
    evaluator = ExecutableReference(
        f"{PREFIX}.evaluator",
        manifest.capability_key,
        manifest.capability_version,
        METHOD,
        artifact,
        SafePayloadFormat.CANONICAL_JSON,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        True,
    )
    publication = publisher.publish_candidate_payload(
        payload=raw,
        evaluator=evaluator,
        implementation=ObjectIdentity.from_record(impl.implementation_id, impl),
        decoder_schema=decoder.decoder_schema,
        decoder_version=decoder.decoder_version,
        maximum_decode_bytes=MAXIMUM_LAW_BYTES,
    )
    artifact_ids = tuple(sorted(m.logical.logical_artifact_id for m in custody.manifests))
    links = tuple(
        EvidenceLink(
            f"{PREFIX}.evidence-{index:03d}",
            EvidenceRelation.DERIVED_FROM,
            source,
            ObjectIdentity.from_record(system.relation.relation_id, system.relation),
            artifact_ids,
            system.world.world_id,
            CUTOFF,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "Fresh complete-episode forecast operands from authenticated assigned native tasks.",
        )
        for index, source in enumerate(
            (
                calibration_id,
                *(ObjectIdentity.from_record(r.receipt_id, r) for r in custody.receipts),
            )
        )
    )
    receipts = tuple(
        sorted(
            (
                CandidateMethodEvidenceReceipt(
                    r.receipt_id,
                    EVIDENCE_KIND,
                    calibration_metrics(calibration),
                    (m.logical.logical_artifact_id,),
                    (links[index + 1].link_id,),
                    (),
                )
                for index, (m, r) in enumerate(
                    zip(custody.manifests, custody.receipts, strict=True)
                )
            ),
            key=lambda r: r.receipt_id,
        )
    )
    candidate = candidate_evidence(ledger, binding, publication, receipts, links)
    profile = ReactorForecastQualificationProfile(
        owner, design, calibration, payload, candidate, design_artifact
    )
    decoders = CandidatePayloadDecoderRegistry((decoder,))
    assessment = LawAssessmentAssembler(
        reader, decoders, QualificationProfileEvaluatorRegistry((profile,))
    ).assemble(
        system=system,
        candidate=candidate,
        qualification_profile=ObjectIdentity.from_record(
            profile.profile.profile_id, profile.profile
        ),
    )
    family = CandidateFamilyAssembler().assemble(ledger, (assessment,))
    result = ResponseLawQualificationService(reader, decoders).qualify(
        system, calibration_id, family
    )
    return ReactorForecastResult(
        design_id, calibration, custody, payload, ledger, candidate, result
    )
