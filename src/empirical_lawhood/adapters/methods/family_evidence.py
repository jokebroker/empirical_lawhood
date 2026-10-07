"""Evidence assembly for one explicitly declared member of a candidate family."""

from empirical_lawhood.adapters.methods.contracts import (
    CandidateFamilyLedger,
    CandidateMethodEvidenceReceipt,
    LawCandidateEvidence,
)
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt


def singleton_candidate_evidence(
    family: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipts: tuple[CandidateMethodEvidenceReceipt, ...],
    links: tuple[EvidenceLink, ...],
    extension: ExtensionBinding,
) -> LawCandidateEvidence:
    if len(family.members) != 1:
        raise ValueError("singleton evidence requires one explicitly declared candidate")
    if family.claim_unit_binding != ObjectIdentity.from_record(binding.binding_id, binding):
        raise ValueError("candidate evidence substituted its independent-unit binding")
    member = family.members[0]
    return LawCandidateEvidence(
        f"{member.candidate_id}.evidence",
        member.candidate_id,
        family.system,
        family.dataset_or_projection,
        member.config,
        family.method_key,
        family.method_version,
        family.method_kind,
        family.representation_kind,
        publication.candidate_evaluator,
        family.axis_map,
        family.claim_unit_binding,
        binding.independent_unit_instance_ids,
        family.claim_template,
        family.obligation_template,
        receipts,
        publication,
        links,
        family.outcome_access,
        family.parent_visibility_ceilings,
        family.visibility_ceiling,
        extension,
    )
