"""Authenticated local operands enter the existing sole law finalizer."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import (
    CandidateEvaluatorImplementation,
    CandidateMethodEvidenceReceipt,
    CandidateFamilyLedger,
    LawCandidateEvidence,
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
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
    NamedDecimal,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from .config import FeedQualificationDesign, PREFIX, ROOTS
from .payload import FeedPayload, FeedDecoder
from .scoring import FeedCalibration, FeedQualificationOperands
from .law import feed_family, candidate_evidence
from .science import METHOD, CUTOFF, feed_system
from .qualification import FeedQualificationProfile


@dataclass(frozen=True, slots=True)
class FeedLawResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-law-result'
    domain: str
    payload: FeedPayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult

    def __post_init__(self) -> None:
        if (
            self.domain != self.payload.domain.domain_id
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
        ):
            raise ValueError("local law publication identity chain differs")


@dataclass(frozen=True, slots=True)
class FeedQualificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-qualification-result'
    operands: FeedQualificationOperands
    laws: tuple[FeedLawResult, ...]

    def __post_init__(self) -> None:
        if len(self.laws) != 1 or self.laws[0].domain != "d11010":
            raise ValueError("local qualification census or frozen admission/controller use applicability differs")

    @property
    def joint_lower_bound(self) -> D:
        from scipy.stats import beta

        contacted = set(self.operands.cells[0].contacted_roots)
        if any(set(c.contacted_roots) != contacted for c in self.operands.cells):
            raise ValueError("feed receivers changed their common contact population")
        adequate = set.intersection(*(set(c.adequate_roots) for c in self.operands.cells))
        return (
            D(0)
            if not adequate
            else D(repr(float(beta.ppf(0.05, len(adequate), len(contacted) - len(adequate) + 1))))
        )

    @property
    def admission_and_prospective_prerequisite(self) -> str:
        from empirical_lawhood.kernel.status import ScientificStatus

        return (
            "QUALIFIED_PREPARED_FEED_RECEIVERS_CONTROL_ISSUE_REQUIRED"
            if self.joint_lower_bound >= D(".90")
            and all(
                law.qualification.scientific_status is ScientificStatus.SUPPORTED
                for law in self.laws
            )
            else "ADMISSION_CONTROLLER_USE_PREREQUISITE_NONENTRY_FEED_RECEIVERS_NOT_JOINTLY_SUPPORTED"
        )


def qualify_feed_cell(
    design: FeedQualificationDesign,
    calibrated: FeedCalibration,
    assessed: FeedQualificationOperands,
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    capability: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> FeedLawResult:
    expected = tuple(f"feed.{r}" for r, _, _, _ in ROOTS)
    operands = ObjectIdentity.from_record(f"{PREFIX}.qualification-operands", assessed)
    if (
        len(manifests) != 64
        or tuple(r.task_id for r in receipts) != expected
        or len({r.run_id for r in receipts}) != 1
        or len({r.implementation_commit for r in receipts}) != 1
        or design_artifact.sha256 != design.fingerprint()
    ):
        raise ValueError("local qualification requires exact 64-root custody and design")
    if (
        assessed.recipe != ObjectIdentity.from_record(design.config_id, design)
        or assessed.calibration != ObjectIdentity.from_record(f"{PREFIX}.calibration", calibrated)
        or tuple(e.object_fingerprint for e in assessed.evidence)
        != tuple(m.logical.content_sha256 for m in manifests)
    ):
        raise ValueError("local finalization substituted assessed scientific operands")
    for manifest, receipt in zip(manifests, receipts, strict=True):
        if (
            receipt.operational_status is not OperationalStatus.SUCCEEDED
            or receipt.output_materializations != (manifest.materialization,)
            or receipt.output_logical_artifacts != (manifest.logical,)
            or manifest.publication is None
            or manifest.logical.outcome_access
            not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL)
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("local qualification raw operand custody differs")
    if tuple(m.logical.content_sha256 for m in manifests[:32]) != tuple(
        e.object_fingerprint for e in calibrated.evidence
    ):
        raise ValueError("local calibration evidence differs from authenticated raw roots")
    stem = f"{PREFIX}.d11010.joint"
    system = feed_system()
    owner = QualificationProofOwner(
        f"{stem}.proof-owner",
        capability.capability_key,
        capability.capability_version,
        capability.implementation_sha256,
    )
    binding, ledger = feed_family(
        design,
        "d11010",
        operands,
        ObjectIdentity.from_record(capability.capability_key, capability),
        ObjectIdentity.from_record(owner.owner_id, owner),
    )
    model = design.atlas.domains[0]
    available = tuple(
        c.q is not None
        and c.q <= 1
        and (width := assessed.cells[r].halfwidth) is not None
        and width <= (D(".26"), D(".005051"))[r]
        for r, c in enumerate(calibrated.cells)
    )
    payload = FeedPayload(
        model,
        design.fingerprint(),
        calibrated.fingerprint(),
        (
            assessed.cells[0].halfwidth if available[0] else None,
            assessed.cells[1].halfwidth if available[1] else None,
        ),
        (
            calibrated.cells[0].q if available[0] else None,
            calibrated.cells[1].q if available[1] else None,
        ),
    )
    implementation = CandidateEvaluatorImplementation(
        f"{stem}.implementation",
        capability.capability_key,
        capability.capability_version,
        METHOD,
        capability.implementation_sha256,
    )
    artifact = ArtifactIdentity(
        f"{stem}.payload",
        "law-payload",
        payload.SCHEMA,
        payload.fingerprint(),
        "application/json",
        len(payload.canonical_bytes()),
    )
    evaluator = ExecutableReference(
        f"{stem}.evaluator",
        capability.capability_key,
        capability.capability_version,
        METHOD,
        artifact,
        SafePayloadFormat.CANONICAL_JSON,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        True,
    )
    decoder = FeedDecoder()
    publication = publisher.publish_candidate_payload(
        payload=payload.canonical_bytes(),
        evaluator=evaluator,
        implementation=ObjectIdentity.from_record(implementation.implementation_id, implementation),
        decoder_schema=decoder.decoder_schema,
        decoder_version=decoder.decoder_version,
        maximum_decode_bytes=65536,
    )
    artifact_ids = tuple(sorted(m.logical.logical_artifact_id for m in manifests))
    links = tuple(
        EvidenceLink(
            f"{stem}.evidence-{i:03d}",
            EvidenceRelation.DERIVED_FROM,
            source,
            ObjectIdentity.from_record(system.relation.relation_id, system.relation),
            artifact_ids,
            system.world.world_id,
            CUTOFF,
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "Authenticated fresh local measurements; all assigned roots retained and causal-contact denominators explicit.",
        )
        for i, source in enumerate(
            (operands, *(ObjectIdentity.from_record(r.receipt_id, r) for r in receipts))
        )
    )
    scores = {s.root: s for s in (*calibrated.scores, *assessed.heldout_scores) if s.receiver == 0}
    if set(scores) != {r for r, _, _, _ in ROOTS}:
        raise ValueError("local method receipt lost an assigned root score")
    evidence = tuple(
        sorted(
            (
                CandidateMethodEvidenceReceipt(
                    r.receipt_id,
                    f"{PREFIX}.fresh-local-operands",
                    (
                        NamedDecimal(
                            "causal-action-assay",
                            D(int(scores[r.task_id.removeprefix("feed.")].action_assay)),
                            "1",
                        ),
                        NamedDecimal(
                            "covered-callbacks",
                            D(scores[r.task_id.removeprefix("feed.")].covered_rows),
                            "1",
                        ),
                    ),
                    (m.logical.logical_artifact_id,),
                    (links[i + 1].link_id,),
                    (),
                )
                for i, (m, r) in enumerate(zip(manifests, receipts, strict=True))
            ),
            key=lambda r: r.receipt_id,
        )
    )
    candidate = candidate_evidence(ledger, binding, publication, evidence, links)
    profile = FeedQualificationProfile(
        owner, design, calibrated, assessed.cells, payload, candidate, design_artifact
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
    result = ResponseLawQualificationService(reader, decoders).qualify(system, operands, family)
    return FeedLawResult("d11010", payload, ledger, candidate, result)
