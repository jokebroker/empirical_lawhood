"""Publish and submit one finite candidate to existing qualification owners."""

from typing import Callable

from empirical_lawhood.adapters.methods.contracts import (
    CandidateEvaluatorImplementation,
    CandidateFamilyLedger,
    CandidateMethodEvidenceReceipt,
    LawCandidateEvidence,
)
from empirical_lawhood.adapters.methods.family_evidence import singleton_candidate_evidence
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoder,
    CandidatePayloadDecoderRegistry,
    CandidatePayloadPublisher,
    CandidatePayloadReader,
    CandidateQualificationProfileEvaluator,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationRequest, LawEvaluationResult
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord, ExtensionBinding
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest


def require_native_custody(
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    task_ids: tuple[str, ...],
    source_hashes: tuple[str, ...],
) -> None:
    if (
        not task_ids
        or len(manifests) != len(task_ids)
        or tuple(r.task_id for r in receipts) != task_ids
        or len({r.run_id for r in receipts}) != 1
        or len({r.implementation_commit for r in receipts}) != 1
        or tuple(m.logical.content_sha256 for m in manifests) != source_hashes
    ):
        raise ValueError("qualification needs the complete immutable native receipt roster")
    for manifest, receipt in zip(manifests, receipts, strict=True):
        if (
            receipt.operational_status is not OperationalStatus.SUCCEEDED
            or manifest.materialization not in receipt.output_materializations
            or manifest.logical not in receipt.output_logical_artifacts
            or manifest.publication is None
            or manifest.logical.outcome_access
            not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL)
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("qualification received substituted native evidence")


def assemble_finite_law(
    *,
    system: SystemSpec,
    stem: str,
    method: str,
    cutoff: str,
    payload: CanonicalRecord,
    decoder: CandidatePayloadDecoder,
    family: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    capability: CapabilityManifest,
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    metrics: tuple[tuple[NamedDecimal, ...], ...],
    evidence_description: str,
    profile: Callable[[LawCandidateEvidence], CandidateQualificationProfileEvaluator],
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
) -> tuple[LawCandidateEvidence, LawQualificationResult]:
    if (
        decoder.extension_namespace is None
        or len(receipts) != len(binding.independent_unit_instance_ids)
        or len(metrics) != len(receipts)
    ):
        raise ValueError("finite candidate lost its extension or physical-unit evidence census")
    implementation = CandidateEvaluatorImplementation(
        f"{stem}.implementation",
        capability.capability_key,
        capability.capability_version,
        method,
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
        method,
        artifact,
        SafePayloadFormat.CANONICAL_JSON,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        True,
    )
    publication = publisher.publish_candidate_payload(
        payload=payload.canonical_bytes(),
        evaluator=evaluator,
        implementation=ObjectIdentity.from_record(implementation.implementation_id, implementation),
        decoder_schema=decoder.decoder_schema,
        decoder_version=decoder.decoder_version,
        maximum_decode_bytes=4 * 1024**2,
    )
    links = tuple(
        EvidenceLink(
            f"{stem}.evidence-{i:03d}",
            EvidenceRelation.DERIVED_FROM,
            ObjectIdentity.from_record(receipt.receipt_id, receipt),
            ObjectIdentity.from_record(system.relation.relation_id, system.relation),
            (manifest.logical.logical_artifact_id,),
            system.world.world_id,
            cutoff,
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            (VisibilityCeiling.PROSPECTIVE,),
            evidence_description,
        )
        for i, (manifest, receipt) in enumerate(zip(manifests, receipts, strict=True))
    )
    evidence = tuple(
        sorted(
            (
                CandidateMethodEvidenceReceipt(
                    receipt.receipt_id,
                    f"{stem}.joint-operands",
                    values,
                    (manifest.logical.logical_artifact_id,),
                    (link.link_id,),
                    (),
                )
                for receipt, manifest, link, values in zip(
                    receipts, manifests, links, metrics, strict=True
                )
            ),
            key=lambda e: e.receipt_id,
        )
    )
    candidate = singleton_candidate_evidence(
        family,
        binding,
        publication,
        evidence,
        links,
        ExtensionBinding(decoder.extension_namespace, payload.SCHEMA, publication.content_sha256),
    )
    selected_profile = profile(candidate)
    decoders = CandidatePayloadDecoderRegistry((decoder,))
    assessment = LawAssessmentAssembler(
        reader, decoders, QualificationProfileEvaluatorRegistry((selected_profile,))
    ).assemble(
        system=system,
        candidate=candidate,
        qualification_profile=ObjectIdentity.from_record(
            selected_profile.profile.profile_id, selected_profile.profile
        ),
    )
    assembled = CandidateFamilyAssembler().assemble(family, (assessment,))
    result = ResponseLawQualificationService(reader, decoders).qualify(
        system, family.dataset_or_projection, assembled
    )
    return candidate, result
