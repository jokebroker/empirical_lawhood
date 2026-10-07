# SPDX-License-Identifier: MPL-2.0
"""Closed software test world, never an operator approval or qualification tool.

Ephemeral signers and fabricated human declarations below are ONLY test fixtures.
No production authority, owner key, historical cohort or outcome is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Match the installed CLI bootstrap before importing any numerical module.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile
from empirical_lawhood.api.reactor_preissue import prove_fresh_reactor
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorPrefixAssignedUnit, ReactorPrefixAssignment, ReactorPrefixPriorCensus
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import SCENARIOS
from empirical_lawhood.api.reactor_authoring import author_fresh_reactor
from empirical_lawhood.api.composition import create_cli_api
from empirical_lawhood.api.models import assemble_issued_study_package
from empirical_lawhood.api.results import AssembleExperimentPackageRequest, CompileCampaignRequest, IssueStudyRequest, IssueExtensionsRequest, ResumeCampaignRequest, RunCampaignRequest
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.infrastructure.authority import Ed25519ApprovalAttestationSigner
from empirical_lawhood.infrastructure.study_issue import PROGRAMME_EXECUTION_GRANTEE_ID, PROGRAMME_ISSUE_GRANTEE_ID, PROGRAMME_REVEAL_GRANTEE_ID
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistration, ApprovalCheckerRegistry, ApprovalGateKind,
    ApprovalSignatureAlgorithm, AttestationResult, issue_approval_gate_attestation,
)
from empirical_lawhood.planning.study_issue import AccountableHumanIdentity, HumanProposerAttestation, ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.candidate_compiler import ExecutableStudyCompilationReport
from empirical_lawhood.runtime.operator_profile import OperatorStorageAccessMode, OperatorStorageProfile
from empirical_lawhood.runtime.plans import CandidateExecutionPlan
from scripts.operator_records import (
    authorize_reviewed, export_record, freeze_for_review, identity, open_stores,
    publish_reviewed_authority, read_record,
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def checked(result):
    assert result.succeeded, (result.operation, result.status, result.reason_codes, result.errors)
    assert result.payload is not None
    return result.payload


def snapshot(root):
    return {str(p.relative_to(root)): (sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def scenario(repo, trust_dir, storage, *, scale=False, exposed=False):
    run_id = "synthetic-public-reactor-lifecycle"
    (storage / "artifacts").mkdir()
    (storage / "scratch").mkdir()
    profile = OperatorStorageProfile(
        "synthetic-lifecycle-storage", "external-filesystem", "1.0.0", str(storage),
        "/dev/shm", "artifacts", "scratch", OperatorStorageAccessMode.READ_WRITE,
        None, None, ("tmpfs",), "strict-mount-contained-no-symlink", 1, (), 1, False,
    )
    prior_units = tuple(f"unit.synthetic.prior-programme.context-a.independent-root-{i:06d}" for i in range(20_000)) if scale else ("synthetic.previously-exposed",)
    prior_seeds = tuple(sorted((
        *(f"seed.pcg64dxsm.{i:032x}" for i in range(100_000)),
        *(f"seed.native-reactor.{2**127+i}" for i in range(100_000)),
        *(f"seed.native-reactor.{i+100}" for i in range(30_000)),
    ))) if scale else ("seed.native-reactor.100",)
    census = ReactorPrefixPriorCensus(
        "synthetic.prior", prior_units, prior_seeds,
        ObjectIdentity("synthetic.inventory", 'empirical-lawhood/synthetic/inventory', "1.0.0", "0" * 64),
    )
    assignment = ReactorPrefixAssignment(
        f"{run_id}.cohort", tuple(
            ReactorPrefixAssignedUnit(f"{run_id}.cohort.{s.replace('_', '-')}", s, 930_000_000 + i)
            for i, s in enumerate(SCENARIOS)
        ), identity(census, census.census_id),
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE" if exposed else "PROSPECTIVE_RELEASE_QUALIFICATION",
    )
    authoring = AssignedReactorAuthoringProfile(
        "synthetic.profile", run_id, load_packaged_reactor_source().fingerprint(), assignment, census,
    )
    directory = storage / "artifacts" / "operator-inputs"
    authored = author_fresh_reactor(root=repo, profile=authoring, output_dir=directory)
    assert authored["task_count"] == 21 and authored["native_tasks_executed"] == 0
    # The positive fixture models an eligible assignment only inside this test
    # world. Its disclosed values/roles never become a real qualification roster.
    assert authored["prospective_issue_eligible"] is (not exposed)
    report = read_record(directory / "candidate-report.json", ExecutableStudyCompilationReport)
    candidate = report.candidate
    assert candidate is not None
    base = candidate.base_candidate.base_candidate
    system = base.system
    closure = read_record(directory / "source-closure.json", ImplementationSourceClosure)
    trust_dir.mkdir(mode=0o700)
    registrations, signers = [], {}
    for gate_id in system.authority_policy.required_gate_ids:
        signer = Ed25519ApprovalAttestationSigner.create_private_key_file(trust_dir / f"{gate_id}.key")
        signers[gate_id] = signer
        registrations.append(ApprovalCheckerRegistration(
            f"synthetic.registration.{gate_id}", gate_id,
            ApprovalGateKind.IMPLEMENTATION if gate_id.endswith(".exact-production-proof") else ApprovalGateKind.AUTHORITY,
            f"synthetic.checker.{gate_id}", sha256(Path(__file__).read_bytes()).hexdigest(), "1.0.0",
            ApprovalSignatureAlgorithm.ED25519, "1.0.0", signer.verification_key_hex, OutcomeAccess.OUTCOME_BLIND,
        ))
    trust = export_record(trust_dir / "trust.json", ApprovalCheckerRegistry("synthetic.checkers", tuple(registrations)))
    stores = open_stores(repo_root=repo, profile=profile, trust_path=trust)
    proof = prove_fresh_reactor(repo_root=repo, storage_profile=profile, directory=directory,
                               approval_checker_trust_path=trust)
    assert all(passed for _, passed, _ in proof["closure_sections"])
    assert proof["native_tasks_executed"] == 0 and len(proof["control_persistence"]) == 4
    api = create_cli_api(repo_root=repo, operator_storage_profile=profile,
                         approval_checker_trust_path=trust, reactor_authoring_dir=directory)
    owner = AccountableHumanIdentity("synthetic.human", "synthetic.test-role", "synthetic.provider", "1" * 64)

    def grant(label, kind, subject, prerequisite=None):
        custody = kind is StudyAuthorityKind.CUSTODY_PUBLICATION
        execute = kind is StudyAuthorityKind.EXPERIMENT_EXECUTION
        reveal = kind is StudyAuthorityKind.OUTCOME_REVEAL
        record = StudyOperationAuthority(
            f"synthetic.{label}", kind, subject, prerequisite, identity(owner, owner.human_id),
            PROGRAMME_ISSUE_GRANTEE_ID if custody else PROGRAMME_EXECUTION_GRANTEE_ID if execute else PROGRAMME_REVEAL_GRANTEE_ID,
            run_id, stores.plane.root.contract.storage_root_id if custody else None,
            "issued-programmes" if custody else None, False, custody, execute, False, reveal,
            now(), None, OutcomeAccess.EVALUATOR_REVEAL if reveal else OutcomeAccess.OUTCOME_BLIND,
        )
        publish_reviewed_authority(stores, record)
        export_record(directory / f"{label}.json", record)
        return record

    def proposer(label, selected, materialization):
        # Fabricated test-only declarations. In the negative case they deliberately
        # contradict the exposed role, which the issue boundary must enforce.
        attestation = HumanProposerAttestation(
            f"synthetic.{label}", identity(selected, selected.candidate_id), identity(owner, owner.human_id),
            owner.human_id, base.design_origin.origin_id, base.design_input_ids, base.source_qualification_receipts,
            materialization.raw_materialization_sha256, base.semantic_config_sha256,
            True, True, False, False, now(), OutcomeAccess.OUTCOME_BLIND,
        )
        return export_record(directory / f"{label}.json", attestation)

    base_custody = grant("base-custody", StudyAuthorityKind.CUSTODY_PUBLICATION,
                         identity(candidate.base_candidate, candidate.base_candidate.candidate_id))
    proposer("base-proposer", candidate.base_candidate, base.authoring_materialization)
    request = IssueStudyRequest(
        directory / "base-authoring.json", directory / "base-candidate.json", directory / "base-proposer.json",
        directory / "source-closure.json", directory / "base-custody.json",
    )
    before = snapshot(storage)
    preview_result = api.issue_study(request)
    if exposed:
        for confirmed in (False, True):
            refused = api.issue_study(replace(request, confirmed=confirmed))
            assert refused.reason_codes == ("OUTCOME_SEPARATION_REQUIRED",), refused
            assert refused.payload is None and snapshot(storage) == before
            assert any("PUBLIC_EVALUATION_UNITS_AND_SEEDS_EXPOSED" in e.message for e in refused.errors)
        grant("extensions-custody", StudyAuthorityKind.CUSTODY_PUBLICATION, identity(candidate, candidate.candidate_id))
        proposer("extensions-proposer", candidate, report.authoring_materialization)
        extension_request = IssueExtensionsRequest(
            directory / "authoring.json", tuple(sorted(directory.glob("payload-*.json"))),
            tuple(sorted(directory.glob("decoder-*.json"))), directory / "candidate.json",
            "synthetic.base-never-issued", directory / "extensions-proposer.json", directory / "extensions-custody.json",
        )
        before = snapshot(storage)
        for confirmed in (False, True):
            refused = api.issue_extensions(replace(extension_request, confirmed=confirmed))
            assert refused.reason_codes == ("OUTCOME_SEPARATION_REQUIRED",), refused
            assert refused.payload is None and snapshot(storage) == before
        print('exposed assignment refused base and extension study issue despite contradictory attestations', flush=True)
        return
    assert preview_result.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    preview = preview_result.payload
    assert not preview.confirmed and preview.external_bytes_written == 0 and snapshot(storage) == before
    if scale:
        from empirical_lawhood.runtime.study_issue import MAX_ISSUE_PAYLOAD_BYTES
        assert len(census.prior_unit_ids) == 20_000 and len(census.prior_seed_ids) == 230_000
        assert len(census.authoring_seed_ids) == 130_000
        roundtrip = read_record(directory / "profile.json", AssignedReactorAuthoringProfile)
        assert roundtrip.prior_census == census
        assert all(0 < member.size_bytes <= MAX_ISSUE_PAYLOAD_BYTES for member in preview.manifest.members)
        assert proof["issue_authoring_member_byte_ceiling"] == MAX_ISSUE_PAYLOAD_BYTES
        # The next byte above the real production bound must be refused by proof
        # before it decodes, creates a provider, or writes another control record.
        member = directory / "base-authoring.json"
        original = member.read_bytes()
        with member.open("wb") as stream:
            stream.truncate(MAX_ISSUE_PAYLOAD_BYTES + 1)
        try:
            prove_fresh_reactor(repo_root=repo, storage_profile=profile, directory=directory,
                               approval_checker_trust_path=trust)
        except ValueError as error:
            assert str(error) == "REACTOR_AUTHORING_EXCEEDS_PRODUCTION_ISSUE_BYTE_BOUND"
        else:
            raise AssertionError("production authoring member byte limit was not enforced")
        member.write_bytes(original)
        print("realistic census and production member bounds verified", proof["issue_authoring_member_bytes"], flush=True)
        return
    issued_base = checked(api.issue_study(replace(request, confirmed=True)))
    base_publication = stores.publisher.load_study(issued_base.manifest.issue_id)
    assert base_publication.publication_receipt == issued_base.publication_receipt
    assert base_custody.subject == identity(candidate.base_candidate, candidate.base_candidate.candidate_id)
    grant("extensions-custody", StudyAuthorityKind.CUSTODY_PUBLICATION, identity(candidate, candidate.candidate_id))
    proposer("extensions-proposer", candidate, report.authoring_materialization)
    extension_issue_request = IssueExtensionsRequest(
        directory / "authoring.json", tuple(sorted(directory.glob("payload-*.json"))),
        tuple(sorted(directory.glob("decoder-*.json"))), directory / "candidate.json", issued_base.manifest.issue_id,
        directory / "extensions-proposer.json", directory / "extensions-custody.json",
    )
    before = snapshot(storage)
    extension_issue_preview = api.issue_extensions(extension_issue_request)
    assert extension_issue_preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert extension_issue_preview.payload.external_bytes_written == 0 and snapshot(storage) == before
    issued_extension_study = checked(api.issue_extensions(replace(extension_issue_request, confirmed=True)))
    publication = stores.publisher.load_executable_study(issued_extension_study.manifest.issue_id)
    assert publication.manifest.base == base_publication.manifest
    assert publication.publication_receipt == issued_extension_study.publication_receipt

    def approve(label, published, action=AuthorityAction.SIMULATION_EXECUTION):
        frozen = freeze_for_review(stores, publication=published, system=system,
                                   scope_id=system.authority_policy.scope_ids[0],
                                   implementation_commit=closure.implementation_commit, action=action)
        auth_id = f"synthetic.approval.{label}"
        attestations = tuple(issue_approval_gate_attestation(
            registration=registration, signer=signers[registration.gate_id],
            attestation_id=f"synthetic.attestation.{label}.{registration.gate_id}", authorization_id=auth_id,
            subject=identity(frozen, frozen.frozen_proposal_id), result=AttestationResult.PASSED,
            checked_at_utc=now(), issued_at_utc=now(), information_cutoff=frozen.proposal.decision_cutoff,
        ) for registration in registrations)
        approval = authorize_reviewed(stores, system=system, frozen=frozen, authorization_id=auth_id,
                                      attestations=attestations, approver_id=system.authority_policy.delegate_id)
        assert authorize_reviewed(stores, system=system, frozen=frozen, authorization_id=auth_id,
                                  attestations=attestations, approver_id=system.authority_policy.delegate_id) == approval
        export_record(directory / f"{label}-frozen.json", frozen)
        export_record(directory / f"{label}-approval.json", approval)
        return frozen, approval

    base_frozen, base_approval = approve("base", base_publication)
    base_execution = grant("base-execution", StudyAuthorityKind.EXPERIMENT_EXECUTION,
                            identity(issued_base.manifest, issued_base.manifest.issue_id),
                            identity(base_approval, base_approval.authorization_id))
    base_package = assemble_issued_study_package(
        issued_study=base_publication.manifest, publication_receipt=base_publication.publication_receipt,
        frozen_proposal=base_frozen, scientific_approval=base_approval, execution_authority=base_execution,
        run_plan_id=run_id, grantee_id=PROGRAMME_EXECUTION_GRANTEE_ID, at_utc=now(),
    )
    local_controls = trust_dir.parent / "local-controls"
    local_controls.mkdir(mode=0o700)
    export_record(local_controls / "base-package.json", base_package)
    _, extension_execution_approval = approve("extensions", publication)
    extension_execution_authority = grant("extensions-execution", StudyAuthorityKind.EXPERIMENT_EXECUTION,
                          identity(publication.manifest, publication.manifest.issue_id),
                          identity(extension_execution_approval, extension_execution_approval.authorization_id))
    export_record(directory / "extensions-manifest.json", publication.manifest)
    export_record(directory / "extensions-receipt.json", publication.publication_receipt)
    assembly = checked(api.assemble_package(AssembleExperimentPackageRequest(
        local_controls / "base-package.json", directory / "extensions-manifest.json", directory / "extensions-receipt.json",
        directory / "extensions-frozen.json", directory / "extensions-approval.json", directory / "extensions-execution.json",
        directory / "resources.json", run_id, PROGRAMME_EXECUTION_GRANTEE_ID, now(), emit_path=Path("experiment-package.json"),
    )))
    assert assembly.authority_replayed and not assembly.executed
    package_path = local_controls / "experiment-package.json"
    compiled = checked(api.compile_campaign(CompileCampaignRequest(package_path)))
    assert sum(map(len, compiled.parallel_task_groups)) == 21
    before = snapshot(storage)
    preview = api.run_campaign(RunCampaignRequest(package_path))
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",), preview.to_mapping()
    assert snapshot(storage) == before
    run = api.run_campaign(RunCampaignRequest(package_path, confirmed=True))
    print("sealed run", run.status, run.reason_codes, run.errors, flush=True)
    assert run.status.value == "BLOCKED"
    assert "OUTCOME_REVEAL_AUTHORITY_REQUIRED" in run.reason_codes
    # Keep all receipted acquisition bytes and mtimes across reveal and recovery.
    preissue_plan = read_record(directory / "preissue-execution-plan.json", CandidateExecutionPlan)
    native_locators = tuple(str(stores.plane.root.resolve(output.relative_path, for_write=False).relative_to(storage))
                            for task in preissue_plan.tasks if task.stage.value == "PREPARE" for output in task.outputs)
    assert len(native_locators) == 20
    sealed_files = snapshot(storage)
    native_files = {locator: sealed_files[locator] for locator in native_locators}
    wrong = api.resume_campaign(ResumeCampaignRequest(run_id, True, extension_execution_authority.authority_id))
    assert not wrong.succeeded
    assert {k: snapshot(storage)[k] for k in native_files} == native_files
    approve("reveal", publication, AuthorityAction.EVALUATOR_REVEAL)
    reveal = grant("reveal", StudyAuthorityKind.OUTCOME_REVEAL,
                   identity(publication.manifest, publication.manifest.issue_id),
                   identity(extension_execution_authority, extension_execution_authority.authority_id))
    finished = checked(api.resume_campaign(ResumeCampaignRequest(run_id, True, reveal.authority_id)))
    assert len(finished.completed_task_ids) == len(finished.receipt_ids) == 21
    assert finished.adjudication_fingerprint is not None
    assert finished.admission_status.value != "ADMITTED"
    assert {k: snapshot(storage)[k] for k in native_files} == native_files
    completed_snapshot = snapshot(storage)
    output_locators = tuple(str(stores.plane.root.resolve(output.relative_path, for_write=False).relative_to(storage))
                            for task in preissue_plan.tasks for output in task.outputs)
    completed_outputs = {locator: completed_snapshot[locator] for locator in output_locators}
    # Discard only this test's rebuildable SQLite projection; custody stays intact.
    catalog = repo / ".empirical-lawhood" / "experiment_catalog.sqlite3"
    assert catalog.is_file()
    catalog.rename(catalog.with_suffix(".sqlite3.test-backup"))
    recovered_api = create_cli_api(repo_root=repo, operator_storage_profile=profile,
                                   approval_checker_trust_path=trust, reactor_authoring_dir=directory)
    recovered = checked(recovered_api.resume_campaign(ResumeCampaignRequest(run_id, True, reveal.authority_id)))
    assert recovered.receipt_ids == finished.receipt_ids
    assert recovered.adjudication_fingerprint == finished.adjudication_fingerprint
    assert recovered.scientific_status == finished.scientific_status
    assert {k: snapshot(storage)[k] for k in completed_outputs} == completed_outputs
    print("21 receipted tasks; same-identity recovery verified", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="empirical-lawhood-lifecycle-", dir="/dev/shm") as temporary:
        scenario(Path(sys.argv[1]), Path(sys.argv[2]), Path(temporary),
                 scale=sys.argv[3] == "scale", exposed=sys.argv[3] == "exposed")
