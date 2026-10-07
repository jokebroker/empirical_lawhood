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
from .records import DevelopmentBoundReactorQualificationOperands
from .payload import DevelopmentBoundReactorResponsePayload, DevelopmentBoundReactorResponseDecoder
from .numerical import FrozenFit
from .serialization import fit_data, json_bytes
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, CandidateFamilyLedger
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from .law import EVIDENCE_KIND, MAXIMUM_LAW_BYTES, candidate_evidence, empirical_family
from .qualification import EmpiricalQualificationProfile, calibration_metrics
from .config import EmpiricalRecipe
from .science import CUTOFF, METHOD, PREFIX, empirical_system


@dataclass(frozen=True, slots=True)
class EmpiricalQualificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-qualification-result'
    design: ObjectIdentity
    operands: DevelopmentBoundReactorQualificationOperands
    payload: DevelopmentBoundReactorResponsePayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult

    def __post_init__(self) -> None:
        if (
            self.payload.calibration_sha256 != self.operands.calibration_digest
            or self.payload.policy_sha256 != self.design.object_fingerprint
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
            or self.family.dataset_or_projection.object_fingerprint != self.operands.fingerprint()
        ):
            raise ValueError("empirical qualification identity chain differs")


def qualify_empirical(
    design: EmpiricalRecipe,
    calibration: DevelopmentBoundReactorQualificationOperands,
    manifests: tuple[ArtifactManifest, ...],
    task_receipts: tuple[CanonicalTaskReceipt, ...],
    model: FrozenFit,
    manifest: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> EmpiricalQualificationResult:
    from empirical_lawhood.kernel.status import OperationalStatus

    expected_roots = tuple(t.object_id for t in calibration.trace_identities)
    if len(manifests) != 64 or tuple(r.task_id for r in task_receipts) != tuple(
        f"empirical.{root}" for root in expected_roots
    ):
        raise ValueError("qualification needs exact 64 assigned raw operand receipts")
    if (
        len({r.run_id for r in task_receipts}) != 1
        or len({r.implementation_commit for r in task_receipts}) != 1
    ):
        raise ValueError("empirical custody mixes runs or implementation commits")
    for artifact_manifest, receipt in zip(manifests, task_receipts, strict=True):
        if (
            receipt.operational_status is not OperationalStatus.SUCCEEDED
            or receipt.output_materializations != (artifact_manifest.materialization,)
            or receipt.output_logical_artifacts != (artifact_manifest.logical,)
            or artifact_manifest.publication is None
            or artifact_manifest.logical.outcome_access
            not in {OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL}
            or artifact_manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("qualification raw operand custody differs")
    design_id = ObjectIdentity.from_record(design.config_id, design)
    calibration_id = ObjectIdentity.from_record(f"{PREFIX}.calibration", calibration)
    expected = tuple(t.object_fingerprint for t in calibration.trace_identities)
    if (
        expected != tuple(m.logical.content_sha256 for m in manifests)
        or design_artifact.sha256 != design.fingerprint()
    ):
        raise ValueError("qualification changes committed trace or design operands")
    system = empirical_system(design)
    owner = QualificationProofOwner(
        f"{PREFIX}.proof-owner",
        manifest.capability_key,
        manifest.capability_version,
        manifest.implementation_sha256,
    )
    binding, ledger = empirical_family(
        system,
        design_id,
        calibration_id,
        ObjectIdentity.from_record(manifest.capability_key, manifest),
        ObjectIdentity.from_record(owner.owner_id, owner),
    )
    payload = DevelopmentBoundReactorResponsePayload(
        json_bytes(fit_data(model)).decode(),
        design.fingerprint(),
        calibration.calibration_digest,
        calibration.q if calibration.q is not None and calibration.q <= 1 else None,
        development_q=calibration.development_q,
    )
    decoder = DevelopmentBoundReactorResponseDecoder()
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
    artifact_ids = tuple(sorted(m.logical.logical_artifact_id for m in manifests))
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
                *(ObjectIdentity.from_record(r.receipt_id, r) for r in task_receipts),
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
                for index, (m, r) in enumerate(zip(manifests, task_receipts, strict=True))
            ),
            key=lambda r: r.receipt_id,
        )
    )
    candidate = candidate_evidence(ledger, binding, publication, receipts, links)
    profile = EmpiricalQualificationProfile(
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
    return EmpiricalQualificationResult(
        design_id, calibration, payload, ledger, candidate, result
    )
