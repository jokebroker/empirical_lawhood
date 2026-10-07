# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from pathlib import Path

from empirical_lawhood.api.codecs import load_authoring
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.models import IssuedStudyPackage, assemble_issued_study_package
from empirical_lawhood.api.results import CompileCampaignRequest, OperationStatus
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistry,
    AttestationResult,
    CompleteApprovalService,
    issue_approval_gate_attestation,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.study_issue import build_study_approval_proposal
from tests.approval_support import FixedDecisionClock, deterministic_approval_signer
from tests.runtime_platform.test_study_approval import _AttestationStore, _AuthorizationStore, _StudyProposalStore, _checker_registrations
from tests.runtime_platform.test_study_issue import _issue_api_fixture, _issue_request


class _AuthorityStore:
    def __init__(self, authority: StudyOperationAuthority) -> None:
        self.authority = authority

    def load(self, authority_id: str) -> StudyOperationAuthority:
        if authority_id != self.authority.authority_id:
            raise KeyError(authority_id)
        return self.authority


def test_standard_issue_approves_packages_and_compiles_without_identity_loss(
    tmp_path: Path,
) -> None:
    issue_api, paths, _external, _custody = _issue_api_fixture(tmp_path)
    issued_result = issue_api.issue_study(_issue_request(paths, confirmed=True))
    assert issued_result.status is OperationStatus.SUCCEEDED
    assert issued_result.payload is not None
    assert issued_result.payload.publication_receipt is not None
    manifest = issued_result.payload.manifest
    publication_receipt = issued_result.payload.publication_receipt
    candidate = manifest.candidate
    base_candidate = candidate.base_candidate
    proposal = build_study_approval_proposal(
        issued_study=manifest,
        publication_receipt=publication_receipt,
    )
    assert proposal.candidate == ObjectIdentity.from_record(
        candidate.candidate_id,
        candidate,
    )

    proposal_store = _StudyProposalStore()
    authorization_store = _AuthorizationStore()
    attestation_store = _AttestationStore()
    registrations = _checker_registrations(base_candidate.system.authority_policy.required_gate_ids)
    checker_registry = ApprovalCheckerRegistry(
        registry_id="standard-package-approval-checkers",
        registrations=registrations,
    )
    service = CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id="standard-package-approval-clock",
            value="2026-07-24T20:40:00Z",
        ),
        proposal_store=proposal_store,
        checker_registry=checker_registry,
        attestation_store=attestation_store,
        store=authorization_store,
    )
    frozen_identity = service.freeze_study_proposal(
        proposal=proposal,
        system=base_candidate.system,
        requested_scope_id="reference-world",
        implementation_commit=manifest.implementation_commit,
        authority_action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
        source_access=SourceAccessClass.NONE,
    )
    authorization_id = "authorization.standard-issued-package"
    for registration in registrations:
        attestation = issue_approval_gate_attestation(
            registration=registration,
            signer=deterministic_approval_signer(registration.gate_id),
            attestation_id=f"attestation.{registration.gate_id}.standard-package",
            authorization_id=authorization_id,
            subject=frozen_identity,
            result=AttestationResult.PASSED,
            checked_at_utc="2026-07-24T20:35:00Z",
            issued_at_utc="2026-07-24T20:36:00Z",
            information_cutoff=proposal.decision_cutoff,
        )
        attestation_store.records[attestation.attestation_id] = attestation
    attestation_identities = tuple(
        ObjectIdentity.from_record(value.attestation_id, value)
        for value in sorted(
            attestation_store.records.values(),
            key=lambda value: value.gate_id,
        )
    )
    approval = service.authorize(
        policy=base_candidate.system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=attestation_identities,
        authorization_id=authorization_id,
        approver_id=base_candidate.system.authority_policy.delegate_id,
    )
    execution_authority = StudyOperationAuthority(
        authority_id="authority.execution.standard-issued-package",
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=ObjectIdentity.from_record(manifest.issue_id, manifest),
        prerequisite_authority=ObjectIdentity.from_record(
            approval.authorization_id,
            approval,
        ),
        issuer=manifest.proposer_attestation.proposer,
        grantee_id="operator.execution-service",
        scope_id="scope.standard-issued-package-execution",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=False,
        allows_external_publication=False,
        allows_execution=True,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-07-24T20:41:00Z",
        expires_at_utc="2026-07-25T20:41:00Z",
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    package = assemble_issued_study_package(
        issued_study=manifest,
        publication_receipt=publication_receipt,
        frozen_proposal=proposal_store.load(frozen_identity.object_id),
        scientific_approval=approval,
        execution_authority=execution_authority,
        run_plan_id="run.standard-issued-package",
        grantee_id="operator.execution-service",
        at_utc="2026-07-24T20:42:00Z",
    )
    package_path = tmp_path / "standard-issued-package.json"
    package_path.write_bytes(package.canonical_bytes())
    assert load_authoring(package_path) == package
    assert isinstance(load_authoring(package_path), IssuedStudyPackage)

    compile_api = EmpiricalLawhoodApi(
        repo_root=tmp_path,
        approval_service=service,
        study_authority_store=_AuthorityStore(execution_authority),
        external_root=tmp_path / "external",
    )
    compiled = compile_api.compile_campaign(CompileCampaignRequest(package_path))

    assert compiled.status is OperationStatus.SUCCEEDED
    assert compiled.payload is not None
    assert compiled.payload.package_id == package.package_id
