# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from empirical_lawhood.api.codecs import load_authoring
from empirical_lawhood.api.execution import (
    _CatalogSemanticReplayBudget,
    CampaignCapabilityError,
    CampaignExecutionService,
    CampaignRevealAuthorityError,
)
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.models import IssuedCampaignPackage, assemble_issued_campaign_package
from empirical_lawhood.api.results import (
    CompileCampaignRequest,
    OperationStatus,
    RunCampaignRequest,
)
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.execution import LocalScheduler
from empirical_lawhood.infrastructure.recovery import decode_run_recovery_index
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalizationError
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistry,
    AttestationResult,
    CompleteApprovalService,
    issue_approval_gate_attestation,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.compiler import compile_run_plan, lower_run_plan
from empirical_lawhood.runtime.artifacts import (
    ArtifactGenericValidation,
    ArtifactMaterialization,
    CanonicalTaskReceipt,
    ExternalRootContract,
    LogicalArtifactIdentity,
    ReceiptCheck,
)
from empirical_lawhood.runtime.execution import VerifiedArtifactInput
from empirical_lawhood.runtime.plans import ArtifactOutputSpec, CandidateExecutionPlan, ExecutionTask, CandidateRunPlan
from empirical_lawhood.runtime.study_issue import build_study_approval_proposal, prepare_draft_candidate_issue
from empirical_lawhood.runtime.scientific_graph_preservation import prove_scientific_graph_parity
from empirical_lawhood.runtime.recovery import CandidateRunRecoveryIndex, build_run_recovery_index
from empirical_lawhood.runtime.providers import CampaignRuntimeProviderRegistry
from tests.approval_support import FixedDecisionClock, deterministic_approval_signer
from tests.runtime_platform.test_study_approval import _AttestationStore, _AuthorizationStore, _StudyProposalStore, _checker_registrations, _published_issue
from tests.runtime_platform.test_study_issue import _Inspector


def _authorized_inputs(tmp_path: Path) -> dict[str, object]:
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
        registry_id="programme-parity-approval-checkers",
        registrations=registrations,
    )
    service = CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id="programme-parity-approval-clock",
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
    authorization_id = "authorization.reference-issued-parity"
    for registration in registrations:
        attestation = issue_approval_gate_attestation(
            registration=registration,
            signer=deterministic_approval_signer(registration.gate_id),
            attestation_id=f"attestation.{registration.gate_id}.reference-parity",
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
        policy=candidate.system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=attestation_identities,
        authorization_id=authorization_id,
        approver_id=candidate.system.authority_policy.delegate_id,
    )
    execution_authority = StudyOperationAuthority(
        authority_id="authority.execution.reference-issued-parity",
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=ObjectIdentity.from_record(
            preparation.manifest.issue_id,
            preparation.manifest,
        ),
        prerequisite_authority=ObjectIdentity.from_record(
            approval.authorization_id,
            approval,
        ),
        issuer=preparation.manifest.proposer_attestation.proposer,
        grantee_id="operator.execution-service",
        scope_id="scope.reference-issued-execution",
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
    return {
        "candidate": candidate,
        "preparation": preparation,
        "publication_receipt": publication_receipt,
        "frozen": proposal_store.load(frozen_identity.object_id),
        "approval": approval,
        "execution_authority": execution_authority,
        "approval_service": service,
    }


def _compile_authorized(
    values: dict[str, object],
    *,
    run_plan_id: str,
) -> tuple[object, object, object, object]:
    candidate = values["candidate"]
    preparation = values["preparation"]
    receipt = values["publication_receipt"]
    frozen = values["frozen"]
    approval = values["approval"]
    execution_authority = values["execution_authority"]
    approval_service = values["approval_service"]
    assert hasattr(candidate, "scientific_graph")
    assert hasattr(preparation, "manifest")
    assert hasattr(receipt, "receipt_id")
    assert hasattr(frozen, "frozen_proposal_id")
    assert hasattr(approval, "authorization_id")
    assert isinstance(execution_authority, StudyOperationAuthority)
    assert isinstance(approval_service, CompleteApprovalService)
    package = assemble_issued_campaign_package(
        issued_study=preparation.manifest,
        publication_receipt=receipt,
        frozen_proposal=frozen,
        scientific_approval=approval,
        execution_authority=execution_authority,
        run_plan_id=run_plan_id,
        grantee_id="operator.execution-service",
        at_utc="2026-07-24T20:42:00Z",
    )
    run_plan = compile_run_plan(
        run_plan_id=package.run_plan_id,
        campaign=package.campaign,
        system=package.system,
        experiment=package.experiment,
        frozen_proposal=package.frozen_proposal,
        authorization=package.scientific_approval,
        template=package.protocol,
        registry=package.registry,
        implementation_commit=package.implementation_commit,
        approval_service=approval_service,
        scientific_graph=package.issued_study.candidate.scientific_graph,
        candidate=ObjectIdentity.from_record(
            package.issued_study.candidate.candidate_id,
            package.issued_study.candidate,
        ),
    )
    execution_plan = lower_run_plan(run_plan, package.registry)
    parity = prove_scientific_graph_parity(
        candidate=candidate,
        issued_candidate=package.issued_study.candidate,
        authorized_experiment=package.experiment,
        scientific_approval=ObjectIdentity.from_record(
            package.scientific_approval.authorization_id,
            package.scientific_approval,
        ),
        issue_manifest=ObjectIdentity.from_record(
            package.issued_study.issue_id,
            package.issued_study,
        ),
        package=ObjectIdentity.from_record(package.package_id, package),
        execution_authority=ObjectIdentity.from_record(
            package.execution_authority.authority_id,
            package.execution_authority,
        ),
        run_plan=run_plan,
        execution_plan=execution_plan,
        registry=package.registry,
    )
    return package, run_plan, execution_plan, parity


def test_generated_package_and_final_plans_are_deterministic_and_parity_bound(
    tmp_path: Path,
) -> None:
    values = _authorized_inputs(tmp_path)

    first = _compile_authorized(values, run_plan_id="run.reference-issued-parity")
    second = _compile_authorized(values, run_plan_id="run.reference-issued-parity")

    assert first == second
    package, run_plan, execution_plan, parity = first
    assert package.canonical_bytes() == second[0].canonical_bytes()
    assert run_plan.canonical_bytes() == second[1].canonical_bytes()
    assert execution_plan.canonical_bytes() == second[2].canonical_bytes()
    assert (
        parity.candidate_projection_sha256
        == parity.run_plan_projection_sha256
        == parity.execution_plan_projection_sha256
    )
    assert any(
        value.logical_artifact_id == "materialization.reference-medium"
        for step in run_plan.steps
        for value in step.external_inputs
    )

    alternate = _compile_authorized(values, run_plan_id="run.reference-issued-alternate")
    assert alternate[3].candidate_projection_sha256 == parity.candidate_projection_sha256
    assert alternate[1].fingerprint() != run_plan.fingerprint()


def _verified_output(
    output: ArtifactOutputSpec,
    *,
    token: str,
) -> VerifiedArtifactInput:
    payload = token.encode("utf-8")
    physical_sha256 = hashlib.sha256(payload).hexdigest()
    logical = LogicalArtifactIdentity(
        logical_artifact_id=output.logical_artifact_id,
        content_sha256=physical_sha256,
        payload_schema=output.payload_schema,
        profile=output.profile,
        media_type=output.media_type,
        visibility_ceiling=output.visibility_ceiling,
        parent_visibility_ceilings=(),
        outcome_access=output.outcome_access,
        generic_validation=ArtifactGenericValidation(
            validator_key="validator.synthetic-exact-edge",
            validator_version="1.0.0",
            validator_implementation_sha256="a" * 64,
            payload_schema=output.payload_schema,
            profile=output.profile,
        ),
    )
    materialization = ArtifactMaterialization(
        materialization_id=(f"materialization.{output.logical_artifact_id}"),
        logical_artifact_id=output.logical_artifact_id,
        storage_root_id="storage.synthetic-exact-edge",
        relative_path=f"synthetic/{token}.bin",
        physical_sha256=physical_sha256,
        size_bytes=len(payload),
        compression="none",
    )
    return VerifiedArtifactInput(logical, materialization)


def test_exact_edge_plan_roundtrips_and_scheduler_excludes_unlisted_siblings(
    tmp_path: Path,
) -> None:
    values = _authorized_inputs(tmp_path)
    _package, run_plan, execution_plan, _parity = _compile_authorized(
        values,
        run_plan_id="run.reference-issued-exact-inputs",
    )
    assert isinstance(run_plan, CandidateRunPlan)
    assert isinstance(execution_plan, CandidateExecutionPlan)
    assert (
        decode_canonical_bytes(
            run_plan.canonical_bytes(),
            CandidateRunPlan,
            maximum_bytes=10_000_000,
        )
        == run_plan
    )
    assert (
        decode_canonical_bytes(
            execution_plan.canonical_bytes(),
            CandidateExecutionPlan,
            maximum_bytes=10_000_000,
        )
        == execution_plan
    )

    task = next(value for value in execution_plan.tasks if value.dependency_task_ids)
    assert isinstance(task, ExecutionTask)
    tasks = {value.task_id: value for value in execution_plan.tasks}
    outputs: dict[str, tuple[VerifiedArtifactInput, ...]] = {}
    extra: VerifiedArtifactInput | None = None
    for dependency_id in task.dependency_task_ids:
        dependency = tasks[dependency_id]
        expected_ids = {
            value.producer_output_id
            for value in task.scientific_inputs
            if value.producer_task_id == dependency_id
        }
        selected = tuple(
            _verified_output(
                output,
                token=f"{dependency_id}-{index:03d}",
            )
            for index, output in enumerate(dependency.outputs)
            if output.output_id in expected_ids
        )
        if extra is None:
            source = dependency.outputs[0]
            extra_output = replace(
                source,
                output_id=f"{dependency_id}.unlisted",
                logical_artifact_id=f"artifact.{dependency_id}.unlisted",
                relative_path=f"runs/unlisted/{dependency_id}.json",
            )
            extra = _verified_output(
                extra_output,
                token=f"{dependency_id}-unlisted",
            )
            selected = (*selected, extra)
        outputs[dependency_id] = tuple(
            sorted(
                selected,
                key=lambda value: value.logical.logical_artifact_id,
            )
        )
    assert extra is not None

    selected = LocalScheduler._dependency_outputs(task, outputs)
    assert extra not in selected
    assert {value.logical.logical_artifact_id for value in selected} == {
        value.operational_logical_artifact_id
        for value in task.scientific_inputs
        if value.producer_task_id is not None
    }
    receipts = {
        dependency_id: CanonicalTaskReceipt(
            receipt_id=f"receipt.{dependency_id}.synthetic",
            run_id=run_plan.run_plan_id,
            task_id=dependency_id,
            attempt_id=f"{run_plan.run_plan_id}.{dependency_id}.attempt-001",
            implementation_commit=run_plan.implementation_commit,
            input_materialization_ids=(),
            output_materializations=tuple(value.materialization for value in available),
            output_logical_artifacts=tuple(value.logical for value in available),
            checks=(ReceiptCheck("synthetic-check", True, ()),),
            operational_status=OperationalStatus.SUCCEEDED,
            reason_codes=(),
        )
        for dependency_id, available in outputs.items()
    }
    bindings = LocalScheduler._dependency_receipt_bindings(
        task,
        dependency_outputs=selected,
        receipts=receipts,
    )
    assert {
        materialization_id
        for binding in bindings
        for materialization_id in binding.output_materialization_ids
    } == {value.materialization.materialization_id for value in selected}

    recovery = build_run_recovery_index(
        execution_plan,
        run_id=run_plan.run_plan_id,
        wave_id="wave.exact-edge-roundtrip",
        authority_identities=(execution_plan.candidate,),
    )
    assert isinstance(recovery, CandidateRunRecoveryIndex)
    assert decode_run_recovery_index(recovery.canonical_bytes()) == recovery
    for mutation in ("unknown-field", "wrong-version"):
        document = json.loads(recovery.canonical_bytes())
        if mutation == "unknown-field":
            document["value"]["unknown"] = True
        else:
            document["version"] = "9.0.0"
        with pytest.raises(CanonicalizationError):
            decode_run_recovery_index(
                (
                    json.dumps(
                        document,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )


def test_parity_refuses_included_run_and_execution_mutations(
    tmp_path: Path,
) -> None:
    values = _authorized_inputs(tmp_path)
    package, run_plan, execution_plan, _parity = _compile_authorized(
        values,
        run_plan_id="run.reference-issued-mutation",
    )
    changed_step = replace(
        run_plan.steps[0],
        obligation_ids=tuple(sorted((*run_plan.steps[0].obligation_ids, "obligation.foreign"))),
    )
    changed_run = replace(
        run_plan,
        steps=(changed_step, *run_plan.steps[1:]),
    )
    with pytest.raises(ValueError, match="scientific node"):
        prove_scientific_graph_parity(
            candidate=package.issued_study.candidate,
            issued_candidate=package.issued_study.candidate,
            authorized_experiment=package.experiment,
            scientific_approval=ObjectIdentity.from_record(
                package.scientific_approval.authorization_id,
                package.scientific_approval,
            ),
            issue_manifest=ObjectIdentity.from_record(
                package.issued_study.issue_id,
                package.issued_study,
            ),
            package=ObjectIdentity.from_record(package.package_id, package),
            execution_authority=ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ),
            run_plan=changed_run,
            execution_plan=execution_plan,
            registry=package.registry,
        )

    changed_task = replace(
        execution_plan.tasks[0],
        capability_implementation_sha256="f" * 64,
    )
    changed_execution = replace(
        execution_plan,
        tasks=(changed_task, *execution_plan.tasks[1:]),
    )
    with pytest.raises(ValueError, match="execution scientific task"):
        prove_scientific_graph_parity(
            candidate=package.issued_study.candidate,
            issued_candidate=package.issued_study.candidate,
            authorized_experiment=package.experiment,
            scientific_approval=ObjectIdentity.from_record(
                package.scientific_approval.authorization_id,
                package.scientific_approval,
            ),
            issue_manifest=ObjectIdentity.from_record(
                package.issued_study.issue_id,
                package.issued_study,
            ),
            package=ObjectIdentity.from_record(package.package_id, package),
            execution_authority=ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ),
            run_plan=run_plan,
            execution_plan=changed_execution,
            registry=package.registry,
        )


def test_package_assembly_refuses_authority_substitution(tmp_path: Path) -> None:
    values = _authorized_inputs(tmp_path)
    execution_authority = values["execution_authority"]
    assert isinstance(execution_authority, StudyOperationAuthority)
    source_authority = StudyOperationAuthority(
        authority_id="authority.source.reference-issued-parity",
        kind=StudyAuthorityKind.SOURCE_ACQUISITION,
        subject=execution_authority.subject,
        prerequisite_authority=None,
        issuer=execution_authority.issuer,
        grantee_id=execution_authority.grantee_id,
        scope_id="scope.reference-source",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=True,
        allows_external_publication=False,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc=execution_authority.issued_at_utc,
        expires_at_utc=execution_authority.expires_at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    values["execution_authority"] = source_authority

    with pytest.raises(PermissionError, match="kind cannot substitute"):
        _compile_authorized(values, run_plan_id="run.reference-authority-substitution")


class _ExecutionAuthorityStore:
    def __init__(self, authority: StudyOperationAuthority) -> None:
        self.authority = authority

    def load(self, authority_id: str) -> StudyOperationAuthority:
        if authority_id != self.authority.authority_id:
            raise KeyError(authority_id)
        return self.authority


class _StudyAuthorityStore:
    def __init__(self, *authorities: StudyOperationAuthority) -> None:
        self.authorities = {authority.authority_id: authority for authority in authorities}

    def load(self, authority_id: str) -> StudyOperationAuthority:
        return self.authorities[authority_id]


def test_public_compile_accepts_only_the_trusted_issued_package(
    tmp_path: Path,
) -> None:
    values = _authorized_inputs(tmp_path)
    package, _run_plan, _execution_plan, _parity = _compile_authorized(
        values,
        run_plan_id="run.reference-public-issued",
    )
    assert isinstance(package, IssuedCampaignPackage)
    package_path = tmp_path / "issued-campaign.json"
    package_path.write_bytes(package.canonical_bytes())
    assert load_authoring(package_path) == package
    authority = values["execution_authority"]
    service = values["approval_service"]
    assert isinstance(authority, StudyOperationAuthority)
    assert isinstance(service, CompleteApprovalService)
    api = EmpiricalLawhoodApi(
        repo_root=tmp_path,
        approval_service=service,
        study_authority_store=_ExecutionAuthorityStore(authority),
        external_root=tmp_path / "external",
    )

    compiled = api.compile_campaign(CompileCampaignRequest(package_path))

    assert compiled.succeeded
    assert compiled.payload is not None
    assert compiled.payload.package_id == package.package_id
    assert compiled.payload.run_plan_id == package.run_plan_id
    run_attempt = api.run_campaign(RunCampaignRequest(package_path, confirmed=True))
    assert run_attempt.status is OperationStatus.BLOCKED
    assert run_attempt.reason_codes == ("EXECUTION_SERVICE_UNAVAILABLE",)

    untrusted_api = EmpiricalLawhoodApi(
        repo_root=tmp_path,
        approval_service=service,
        external_root=tmp_path / "external",
    )
    refused = untrusted_api.compile_campaign(CompileCampaignRequest(package_path))
    assert refused.status is OperationStatus.INVALID


def test_issued_execution_requires_a_separate_exact_reveal_authority(
    tmp_path: Path,
) -> None:
    values = _authorized_inputs(tmp_path)
    package, run_plan, execution_plan, _parity = _compile_authorized(
        values,
        run_plan_id="run.reference-issued-reveal-gate",
    )
    assert isinstance(package, IssuedCampaignPackage)
    execution_authority = values["execution_authority"]
    approval_service = values["approval_service"]
    assert isinstance(execution_authority, StudyOperationAuthority)
    assert isinstance(approval_service, CompleteApprovalService)
    reveal_authority = StudyOperationAuthority(
        authority_id="authority.reveal.reference-issued-parity",
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=ObjectIdentity.from_record(
            package.issued_study.issue_id,
            package.issued_study,
        ),
        prerequisite_authority=ObjectIdentity.from_record(
            execution_authority.authority_id,
            execution_authority,
        ),
        issuer=execution_authority.issuer,
        grantee_id="operator.evaluator-service",
        scope_id="scope.reference-issued-reveal",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=False,
        allows_external_publication=False,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=True,
        issued_at_utc="2026-07-24T20:43:00Z",
        expires_at_utc="2026-07-25T20:43:00Z",
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    root = tmp_path / "external"
    artifact_plane = ExternalArtifactPlane(
        GuardedExternalRoot(
            ExternalRootContract(
                storage_root_id="storage.synthetic-science",
                logical_name="Synthetic programme execution root",
                canonical_path=str(root.resolve()),
                required_mount_path=str(root.resolve()),
                mount_contract_schema='empirical-lawhood/testing/fixtures/study-execution-root',
                minimum_free_bytes=1,
            ),
            _Inspector(root),
        )
    )
    service = CampaignExecutionService(
        repo_root=Path(__file__).resolve().parents[2],
        artifact_plane=artifact_plane,
        engine_factory=lambda: pytest.fail("authority refusal must precede SQLite"),
        catalog_path=tmp_path / "catalog.sqlite3",
        providers=CampaignRuntimeProviderRegistry(()),
        approval_service=approval_service,
        study_authority_store=_StudyAuthorityStore(
            execution_authority,
            reveal_authority,
        ),
        enforce_git_identity=False,
    )

    with pytest.raises(
        CampaignRevealAuthorityError,
        match="requires separate reveal authority",
    ):
        service.execute(
            package=package,
            run_plan=run_plan,
            execution_plan=execution_plan,
            at_utc="2026-07-24T20:44:00Z",
        )
    with pytest.raises(
        CampaignRevealAuthorityError,
        match="reveal authority replay failed",
    ):
        service.execute(
            package=package,
            run_plan=run_plan,
            execution_plan=execution_plan,
            at_utc="2026-07-24T20:44:00Z",
            reveal_authority_id=execution_authority.authority_id,
        )
    with pytest.raises(CampaignCapabilityError, match="no runtime provider"):
        service.execute(
            package=package,
            run_plan=run_plan,
            execution_plan=execution_plan,
            at_utc="2026-07-24T20:44:00Z",
            reveal_authority_id=reveal_authority.authority_id,
        )
    assert not root.joinpath("runs").exists()

    service._persist_plans(
        package,
        run_plan,
        execution_plan,
        package_name="campaign-package",
        source_plan_name="run-plan",
        minimum_free_bytes=1,
    )
    assert service._load_package(package.run_plan_id) == package
    with pytest.raises(
        CampaignRevealAuthorityError,
        match="requires separate reveal authority",
    ):
        service.resume(
            package.run_plan_id,
            lambda _package: (run_plan, execution_plan),
            at_utc="2026-07-24T20:44:00Z",
        )
    with pytest.raises(
        CampaignCapabilityError,
        match="no runtime provider",
    ):
        service._authenticated_rebuild_plan_semantics(
            package.run_plan_id,
            budget=_CatalogSemanticReplayBudget(),
        )
