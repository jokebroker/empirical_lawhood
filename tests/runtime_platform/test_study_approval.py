# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.approval import ApprovalCheckerRegistration, ApprovalCheckerRegistry, ApprovalGateAttestation, ApprovalGateKind, ApprovalSignatureAlgorithm, AttestationResult, CompleteApprovalService, DurableAuthorizationRecord, FrozenIssuedStudyApprovalProposal, issue_approval_gate_attestation
from empirical_lawhood.planning.authority import AuthorizationDecision
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.study_issue import StudyPublicationReceipt, build_study_approval_proposal, prepare_draft_candidate_issue
from tests.approval_support import FixedDecisionClock, deterministic_approval_signer
from tests.runtime_platform.test_study_issue import _issue_inputs, _publisher


class _StudyProposalStore:
    def __init__(self) -> None:
        self.records: dict[str, FrozenIssuedStudyApprovalProposal] = {}

    def persist(self, record: FrozenIssuedStudyApprovalProposal) -> ObjectIdentity:
        observed = self.records.get(record.frozen_proposal_id)
        if observed is not None and observed != record:
            raise ValueError("immutable programme proposal conflict")
        self.records[record.frozen_proposal_id] = record
        return ObjectIdentity.from_record(record.frozen_proposal_id, record)

    def load(self, frozen_proposal_id: str) -> FrozenIssuedStudyApprovalProposal:
        return self.records[frozen_proposal_id]


class _AuthorizationStore:
    def __init__(self) -> None:
        self.records: dict[str, DurableAuthorizationRecord] = {}

    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity:
        observed = self.records.get(record.authorization_id)
        if observed is not None and observed != record:
            raise ValueError("immutable authorization conflict")
        self.records[record.authorization_id] = record
        return ObjectIdentity.from_record(record.authorization_id, record)

    def load(self, authorization_id: str) -> DurableAuthorizationRecord:
        return self.records[authorization_id]


class _AttestationStore:
    def __init__(self) -> None:
        self.records: dict[str, ApprovalGateAttestation] = {}

    def load(self, attestation_id: str) -> ApprovalGateAttestation:
        return self.records[attestation_id]


def _published_issue(
    tmp_path: Path,
) -> tuple[dict[str, object], object]:
    root = tmp_path / "external"
    root.mkdir()
    inputs = _issue_inputs()
    authority = inputs["custody_authority"]
    assert isinstance(authority, StudyOperationAuthority)
    preparation = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    publisher: ExternalIssuedStudyPublisher = _publisher(root, authority)
    receipt = publisher.publish(preparation, at_utc="2026-07-24T20:32:00Z")
    return inputs, receipt


def _checker_registrations(
    gate_ids: tuple[str, ...],
) -> tuple[ApprovalCheckerRegistration, ...]:
    registrations = []
    for gate_id in gate_ids:
        signer = deterministic_approval_signer(gate_id)
        registrations.append(
            ApprovalCheckerRegistration(
                checker_registration_id=f"checker-registration.{gate_id}",
                gate_id=gate_id,
                gate_kind=(
                    ApprovalGateKind.IMPLEMENTATION
                    if gate_id == "clean-implementation"
                    else ApprovalGateKind.RESOURCE
                ),
                checker_id=f"checker.{gate_id}",
                implementation_sha256="b" * 64,
                implementation_version="1.0.0",
                signature_algorithm=ApprovalSignatureAlgorithm.ED25519,
                signature_version="1.0.0",
                verification_key_hex=signer.verification_key_hex,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    return tuple(sorted(registrations, key=lambda value: value.gate_id))


def test_ordinary_issued_study_uses_complete_approval_without_nomination(
    tmp_path: Path,
) -> None:
    inputs, publication_receipt = _published_issue(tmp_path)
    candidate = inputs["candidate"]
    draft = inputs["draft"]
    preparation = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    assert hasattr(candidate, "candidate_id")
    assert hasattr(draft, "design_inputs")
    assert hasattr(publication_receipt, "receipt_id")
    proposal = build_study_approval_proposal(
        issued_study=preparation.manifest,
        publication_receipt=publication_receipt,
    )
    proposal_store = _StudyProposalStore()
    authorization_store = _AuthorizationStore()
    attestation_store = _AttestationStore()
    registrations = _checker_registrations(candidate.system.authority_policy.required_gate_ids)
    checker_registry = ApprovalCheckerRegistry(
        registry_id="programme-approval-checkers",
        registrations=registrations,
    )
    service = CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id="programme-approval-clock",
            value="2026-07-24T20:40:00Z",
        ),
        proposal_store=proposal_store,
        checker_registry=checker_registry,
        attestation_store=attestation_store,
        store=authorization_store,
    )
    frozen_identity = service.freeze_study_proposal(
        proposal=proposal,
        system=candidate.system,
        requested_scope_id="reference-world",
        implementation_commit=preparation.manifest.implementation_commit,
        authority_action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
        source_access=SourceAccessClass.NONE,
    )
    frozen = proposal_store.load(frozen_identity.object_id)
    assert frozen.proposal.design_origin == candidate.design_origin
    assert frozen.proposal.issued_study == proposal.issued_study

    authorization_id = "authorization.reference-issued-programme"
    for registration in registrations:
        signer = deterministic_approval_signer(registration.gate_id)
        attestation = issue_approval_gate_attestation(
            registration=registration,
            signer=signer,
            attestation_id=f"attestation.{registration.gate_id}.reference-issued",
            authorization_id=authorization_id,
            subject=frozen_identity,
            result=AttestationResult.PASSED,
            checked_at_utc="2026-07-24T20:35:00Z",
            issued_at_utc="2026-07-24T20:36:00Z",
            information_cutoff=proposal.decision_cutoff,
        )
        attestation_store.records[attestation.attestation_id] = attestation
    identities = tuple(
        ObjectIdentity.from_record(value.attestation_id, value)
        for value in sorted(
            attestation_store.records.values(),
            key=lambda value: value.gate_id,
        )
    )
    approved = service.authorize(
        policy=candidate.system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=identities,
        authorization_id=authorization_id,
        approver_id=candidate.system.authority_policy.delegate_id,
    )

    assert approved.decision is AuthorizationDecision.APPROVED_NONACTUATING
    assert approved.outcome_access is OutcomeAccess.OUTCOME_BLIND
    assert not approved.plan_mutated
    assert not approved.grants_claim_promotion
    replayed = service.replay(
        system=candidate.system,
        frozen_proposal=frozen_identity,
        authorization=ObjectIdentity.from_record(approved.authorization_id, approved),
    )
    assert replayed.authorization_record_id == authorization_id

    with pytest.raises(ValueError, match="distinct"):
        service.preview(
            policy=candidate.system.authority_policy,
            frozen_proposal=frozen_identity,
            attestations=identities,
            authorization_id=authorization_id,
            approver_id=proposal.proposed_by,
        )


def test_study_approval_proposal_builder_refuses_unpublished_identity(
    tmp_path: Path,
) -> None:
    inputs, publication_receipt = _published_issue(tmp_path)
    preparation = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    assert isinstance(publication_receipt, StudyPublicationReceipt)
    substituted = replace(
        publication_receipt,
        issue_manifest=replace(
            publication_receipt.issue_manifest,
            object_fingerprint="f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="another issued programme"):
        build_study_approval_proposal(
            issued_study=preparation.manifest,
            publication_receipt=substituted,
        )
