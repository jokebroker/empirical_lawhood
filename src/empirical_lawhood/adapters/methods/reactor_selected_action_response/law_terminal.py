"Authenticate C operands and delegate the selected finite-bound law to the installed local law owner."

from __future__ import annotations

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
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest

from .config import ROOTS, ClassicalDesign
from .law_family import CUTOFF, classical_candidate_evidence, classical_law_family
from .law_payload import ClassicalLawDecoder, ClassicalLawPayload
from .law_qualification import STEM, ClassicalQualificationProfile
from .records import ClassicalQualification
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig
from .science import METHOD, classical_system

RESULT_ID = "reactor-selected-action-response-law-result"


@dataclass(frozen=True, slots=True)
class ClassicalLaw(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-law'

    payload: ClassicalLawPayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult
    operands: ClassicalQualification

    def __post_init__(self) -> None:
        if (
            self.family.dataset_or_projection != self.payload.qualification
            or self.payload.qualification
            != ObjectIdentity.from_record(self.operands.package_id, self.operands)
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
            or self.qualification.dataset_or_projection != self.payload.qualification
            or self.candidate.axis_map != self.family.axis_map
        ):
            raise ValueError(
                "selected finite-bound law result lost its exact selected local law owner chain"
            )


def qualify_classical_law(
    design: ClassicalDesign,
    native: ClassicalNativeConfig,
    qualification: ClassicalQualification,
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    capability: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> ClassicalLaw:
    "The method supplies facts; the generic service alone creates the local law."
    expected = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
    if (
        len(manifests) != 64
        or len(receipts) != 64
        or tuple(receipt.task_id for receipt in receipts)
        != tuple(f"classical.assay.{root}" for root in expected)
        or len({receipt.run_id for receipt in receipts}) != 1
        or len({receipt.implementation_commit for receipt in receipts}) != 1
        or design_artifact.sha256 != design.fingerprint()
        or native.design != design
        or tuple(manifest.logical.content_sha256 for manifest in manifests)
        != tuple(value.object_fingerprint for value in qualification.assays)
    ):
        raise ValueError("selected-action local law owner lacks complete 64-root immutable custody")
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
            raise ValueError("selected-action local law owner received substituted native root evidence")
    payload = ClassicalLawPayload(
        design,
        native.prepared_domain,
        ObjectIdentity.from_record(qualification.package_id, qualification),
    )
    q_identity = ObjectIdentity.from_record(qualification.package_id, qualification)
    system = classical_system()
    owner = QualificationProofOwner(
        f"{STEM}.proof-owner",
        capability.capability_key,
        capability.capability_version,
        capability.implementation_sha256,
    )
    binding, ledger = classical_law_family(
        design,
        payload,
        q_identity,
        ObjectIdentity.from_record(capability.capability_key, capability),
        ObjectIdentity.from_record(owner.owner_id, owner),
    )
    implementation = CandidateEvaluatorImplementation(
        f"{STEM}.implementation",
        capability.capability_key,
        capability.capability_version,
        METHOD,
        capability.implementation_sha256,
    )
    artifact = ArtifactIdentity(
        f"{STEM}.payload",
        "law-payload",
        payload.SCHEMA,
        payload.fingerprint(),
        "application/json",
        len(payload.canonical_bytes()),
    )
    evaluator = ExecutableReference(
        f"{STEM}.evaluator",
        capability.capability_key,
        capability.capability_version,
        METHOD,
        artifact,
        SafePayloadFormat.CANONICAL_JSON,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        True,
    )
    decoder = ClassicalLawDecoder()
    publication = publisher.publish_candidate_payload(
        payload=payload.canonical_bytes(),
        evaluator=evaluator,
        implementation=ObjectIdentity.from_record(implementation.implementation_id, implementation),
        decoder_schema=decoder.decoder_schema,
        decoder_version=decoder.decoder_version,
        maximum_decode_bytes=4 * 1024**2,
    )
    artifact_ids = tuple(sorted(manifest.logical.logical_artifact_id for manifest in manifests))
    links = tuple(
        EvidenceLink(
            f"{STEM}.evidence-{index:03d}",
            EvidenceRelation.DERIVED_FROM,
            source,
            ObjectIdentity.from_record(system.relation.relation_id, system.relation),
            artifact_ids,
            system.world.world_id,
            CUTOFF,
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            "Authenticated selected positive native word; all 64 qualification roots retained.",
        )
        for index, source in enumerate(
            (
                q_identity,
                *(ObjectIdentity.from_record(receipt.receipt_id, receipt) for receipt in receipts),
                ObjectIdentity.from_record(design.config_id, design),
            )
        )
    )
    by_root = {value.root: value for value in qualification.roots}
    evidence = tuple(
        sorted(
            (
                CandidateMethodEvidenceReceipt(
                    receipt.receipt_id,
                    f"{STEM}.fresh-joint-operands",
                    (
                        NamedDecimal("law-adequate", D(int(by_root[root].adequate)), "1"),
                        NamedDecimal("preparation-safe", D(int(by_root[root].safe)), "1"),
                    ),
                    (manifest.logical.logical_artifact_id,),
                    (links[index + 1].link_id,),
                    (),
                )
                for index, (root, manifest, receipt) in enumerate(
                    zip(expected, manifests, receipts, strict=True)
                )
            ),
            key=lambda value: value.receipt_id,
        )
    )
    candidate = classical_candidate_evidence(ledger, binding, publication, evidence, links)
    profile = ClassicalQualificationProfile(
        owner,
        design,
        qualification,
        payload,
        candidate,
        design_artifact,
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
    result = ResponseLawQualificationService(reader, decoders).qualify(system, q_identity, family)
    if result.dataset_or_projection != q_identity:
        raise ValueError("sole-owner local law result lost its exact C qualification source")
    return ClassicalLaw(payload, ledger, candidate, result, qualification)
