# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import IssueStudyRequest, OperationStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    ExternalArtifactPlane,
    ExternalRootUnavailable,
    FilesystemState,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher, ExternalStudyOperationAuthorityStore, GitStudySourceClosureInspector
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.experiment_entry import StudyDefinition
from empirical_lawhood.planning.study_issue import AccountableHumanIdentity, HumanProposerAttestation, ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority, SourceClosureKind
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, StudyCompilationReport, StandardCandidateCompilationContext, compile_draft_candidate, compile_study_candidate
from empirical_lawhood.runtime.study_issue import ISSUE_PUBLICATION_RELATIVE_ROOT, IssuedStudyMember, MAX_ISSUE_PAYLOAD_BYTES, MAX_STANDARD_PROGRAMME_CONTROL_BYTES, prepare_draft_candidate_issue
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ExternalRootContract
from tests.approval_support import FixedDecisionClock
from tests.experiment_entry_support import entry_package_for_draft
from tests.runtime_platform.test_candidate_compiler import _fixture
from tests.runtime_platform.test_standard_formal_candidate import _standard_fixture
from tests.standard_candidate_support import FixedStandardCandidateContextProvider


def test_generated_compilation_member_capacity_keeps_scientific_payloads_bounded() -> None:
    member = IssuedStudyMember(
        "issue.fixture.compilation",
        "issued-programmes/fixture/report.json",
        StudyCompilationReport.SCHEMA,
        "application/json",
        ArtifactProfile.CANONICAL_JSON,
        MAX_ISSUE_PAYLOAD_BYTES + 1,
        "a" * 64,
        "a" * 64,
    )
    with pytest.raises(ValueError, match="byte bound"):
        replace(member, payload_schema='empirical-lawhood/testing/fixtures/scientific-payload')
    with pytest.raises(ValueError, match="byte bound"):
        replace(member, profile=ArtifactProfile.TEXT_PARAMETERS)
    with pytest.raises(ValueError, match="byte bound"):
        replace(member, size_bytes=MAX_STANDARD_PROGRAMME_CONTROL_BYTES + 1)


def _issue_inputs(*, raw_padding_bytes: int = 0) -> dict[str, object]:
    draft, context = _fixture(conditional=True)
    raw = draft.canonical_bytes() + b" " * raw_padding_bytes
    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(raw),
            raw_materialization_sha256=hashlib.sha256(b"application/json\x00" + raw).hexdigest(),
        ),
        context=context,
    )
    assert report.candidate is not None
    candidate = report.candidate
    human = AccountableHumanIdentity(
        human_id="human.owner",
        role_id="role.accountable-proposer",
        identity_provider_id="identity.local-owner",
        identity_subject_sha256="7" * 64,
    )
    human_identity = ObjectIdentity.from_record(human.human_id, human)
    proposer_attestation = HumanProposerAttestation(
        attestation_id="proposer-attestation.reference-candidate",
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        proposer=human_identity,
        proposer_id=human.human_id,
        design_origin_id=candidate.design_origin.origin_id,
        design_input_ids=candidate.design_input_ids,
        source_qualification_receipts=candidate.source_qualification_receipts,
        raw_materialization_sha256=(candidate.authoring_materialization.raw_materialization_sha256),
        semantic_config_sha256=candidate.semantic_config_sha256,
        known_exposure_lineage_complete=True,
        evaluation_units_and_seeds_unexposed=True,
        fresh_child_outcomes_observed=False,
        codex_or_chat_is_proposer_attestor_approver_or_issuer=False,
        attested_at_utc="2026-07-24T20:30:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    source_closure = ImplementationSourceClosure(
        source_closure_id="source-closure.reference-candidate",
        kind=SourceClosureKind.CLEAN_GIT_COMMIT,
        implementation_commit="a" * 40,
        implementation_sha256=candidate.implementation_sha256,
        source_tree_sha256="8" * 64,
        clean_worktree=True,
    )
    custody_authority = StudyOperationAuthority(
        authority_id="authority.custody.reference-candidate",
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        prerequisite_authority=None,
        issuer=human_identity,
        grantee_id="operator.issue-service",
        scope_id="scope.reference-candidate-issue",
        storage_root_id="storage.synthetic-science",
        relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        allows_source_acquisition=False,
        allows_external_publication=True,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-07-24T20:31:00Z",
        expires_at_utc="2026-07-25T20:31:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return {
        "draft": draft,
        "entry_package": entry_package_for_draft(draft),
        "raw_draft_bytes": raw,
        "current_compilation": report,
        "candidate": candidate,
        "registry": context.registry,
        "materialization_qualifications": context.qualifications,
        "source_closure": source_closure,
        "observed_source_closure": source_closure,
        "proposer_attestation": proposer_attestation,
        "custody_authority": custody_authority,
        "storage_root_id": "storage.synthetic-science",
        "grantee_id": "operator.issue-service",
        "at_utc": "2026-07-24T20:32:00Z",
    }


def test_issue_preparation_is_deterministic_complete_and_write_free() -> None:
    inputs = _issue_inputs()

    first = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    second = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    later_inputs = dict(inputs)
    later_inputs["at_utc"] = "2026-07-24T20:33:00Z"
    later = prepare_draft_candidate_issue(**later_inputs)  # type: ignore[arg-type]

    assert first == second
    assert first.manifest.canonical_bytes() == second.manifest.canonical_bytes()
    assert first.manifest == later.manifest
    assert first.manifest.issue_id.startswith("issue.")
    assert first.manifest.candidate.conditional_successor is not None
    assert first.manifest.implementation_commit == "a" * 40
    assert tuple(value.member for value in first.payloads) == first.manifest.members
    schemas = {value.member.payload_schema for value in first.payloads}
    assert first.manifest.candidate.SCHEMA in schemas
    assert first.manifest.candidate.conditional_successor.SCHEMA in schemas
    assert all(
        value.member.relative_path.startswith(
            f"{ISSUE_PUBLICATION_RELATIVE_ROOT}/{first.manifest.issue_id}/"
        )
        for value in first.payloads
    )
    raw_member = next(
        value for value in first.manifest.members if value.member_id.endswith(".draft-source")
    )
    assert raw_member.logical_content_sha256 == raw_member.physical_sha256
    assert first.manifest.raw_draft.object_fingerprint != raw_member.physical_sha256


def test_issue_member_capacity_accepts_256_and_refuses_257() -> None:
    preparation = prepare_draft_candidate_issue(**_issue_inputs())  # type: ignore[arg-type]
    manifest = preparation.manifest
    first = manifest.members[0]
    members = tuple(
        replace(
            first,
            member_id=f"{manifest.issue_id}.capacity-{index:03d}",
            relative_path=f"issued-programmes/{manifest.issue_id}/capacity-{index:03d}.json",
        )
        for index in range(257)
    )
    accepted = replace(manifest, members=members[:256])
    assert len(accepted.members) == 256
    with pytest.raises(ValueError, match="member count"):
        replace(manifest, members=members)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("raw_draft_bytes", b"changed", "authoring materialization"),
        ("materialization_qualifications", (), "qualification"),
        (
            "storage_root_id",
            "storage.other",
            "storage scope",
        ),
    ),
)
def test_issue_preparation_refuses_stale_or_substituted_inputs(
    field: str,
    replacement: object,
    message: str,
) -> None:
    inputs = _issue_inputs()
    inputs[field] = replacement

    with pytest.raises((PermissionError, ValueError), match=message):
        prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]


def test_issue_preparation_refuses_authority_and_attestation_substitution() -> None:
    inputs = _issue_inputs()
    candidate = inputs["candidate"]
    custody = inputs["custody_authority"]
    attestation = inputs["proposer_attestation"]
    assert not isinstance(candidate, dict)
    assert isinstance(custody, StudyOperationAuthority)
    assert isinstance(attestation, HumanProposerAttestation)
    source_authority = StudyOperationAuthority(
        authority_id="authority.source.reference-candidate",
        kind=StudyAuthorityKind.SOURCE_ACQUISITION,
        subject=custody.subject,
        prerequisite_authority=None,
        issuer=custody.issuer,
        grantee_id=custody.grantee_id,
        scope_id="scope.reference-source",
        storage_root_id=None,
        relative_root=None,
        allows_source_acquisition=True,
        allows_external_publication=False,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc=custody.issued_at_utc,
        expires_at_utc=custody.expires_at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    inputs["custody_authority"] = source_authority
    with pytest.raises(PermissionError, match="kind cannot substitute"):
        prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]

    inputs = _issue_inputs()
    attestation = inputs["proposer_attestation"]
    assert isinstance(attestation, HumanProposerAttestation)
    inputs["proposer_attestation"] = replace(
        attestation,
        candidate=ObjectIdentity(
            object_id="candidate.foreign",
            object_schema=attestation.candidate.object_schema,
            object_version=attestation.candidate.object_version,
            object_fingerprint="9" * 64,
        ),
    )
    with pytest.raises(ValueError, match="another candidate"):
        prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]


def test_issue_preparation_refuses_observed_source_closure_drift() -> None:
    inputs = _issue_inputs()
    closure = inputs["source_closure"]
    assert isinstance(closure, ImplementationSourceClosure)
    inputs["observed_source_closure"] = replace(
        closure,
        source_tree_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="source closure is stale"):
        prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]


class _Inspector:
    def __init__(self, root: object, *, free_bytes: int = 100_000_000) -> None:
        from pathlib import Path

        assert isinstance(root, Path)
        self.root = root
        self.free_bytes = free_bytes

    def inspect(self, _contract: ExternalRootContract) -> FilesystemState:
        return FilesystemState(
            canonical_root=str(self.root.resolve()),
            active_mount=True,
            writable=True,
            free_bytes=self.free_bytes,
            path_is_symlink=False,
        )


class _AuthorityStore:
    def __init__(self, record: StudyOperationAuthority) -> None:
        self.record = record

    def load(self, authority_id: str) -> StudyOperationAuthority:
        if authority_id != self.record.authority_id:
            raise KeyError(authority_id)
        return self.record


def _publisher(
    root: object,
    authority: StudyOperationAuthority,
    *,
    free_bytes: int = 100_000_000,
) -> ExternalIssuedStudyPublisher:
    from pathlib import Path

    assert isinstance(root, Path)
    contract = ExternalRootContract(
        storage_root_id="storage.synthetic-science",
        logical_name="Synthetic programme issue root",
        canonical_path=str(root.resolve()),
        required_mount_path=str(root.resolve()),
        mount_contract_schema='empirical-lawhood/testing/fixtures/study-issue-root',
        minimum_free_bytes=1,
    )
    return ExternalIssuedStudyPublisher(
        artifact_plane=ExternalArtifactPlane(
            GuardedExternalRoot(contract, _Inspector(root, free_bytes=free_bytes))
        ),
        authority_store=_AuthorityStore(authority),
        grantee_id="operator.issue-service",
    )


@pytest.mark.parametrize("in_memory_limit", (32 * 1024**2, 1024))
def test_guarded_issue_publisher_is_atomic_no_replace_and_replay_safe(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    in_memory_limit: int,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    monkeypatch.setattr(
        "empirical_lawhood.infrastructure.artifacts.MAX_IN_MEMORY_ARTIFACT_BYTES", in_memory_limit
    )
    root = tmp_path / "external"
    root.mkdir()
    inputs = _issue_inputs(raw_padding_bytes=1024**2)
    authority = inputs["custody_authority"]
    assert isinstance(authority, StudyOperationAuthority)
    preparation = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    publisher = _publisher(root, authority)

    first = publisher.publish(preparation, at_utc="2026-07-24T20:32:00Z")
    second = publisher.publish(preparation, at_utc="2026-07-24T20:33:00Z")

    assert first == second
    assert publisher.load(preparation.manifest.issue_id).manifest == preparation.manifest
    assert publisher.load(preparation.manifest.issue_id).publication_receipt == first
    assert first.issue_manifest == ObjectIdentity.from_record(
        preparation.manifest.issue_id,
        preparation.manifest,
    )
    assert first.publication.commit_marker_relative_path.endswith(".commit.json")
    assert first.total_payload_bytes > 0
    for payload in preparation.payloads:
        assert (root / payload.member.relative_path).read_bytes() == payload.payload
        assert (root / f"{payload.member.relative_path}.manifest.json").is_file()
    manifest_path = (
        root / ISSUE_PUBLICATION_RELATIVE_ROOT / preparation.manifest.issue_id / "manifest.json"
    )
    assert manifest_path.read_bytes() == preparation.manifest.canonical_bytes()
    receipt_path = manifest_path.with_name("publication-receipt.json")
    assert receipt_path.read_bytes() == first.canonical_bytes()
    assert Path(f"{receipt_path}.manifest.json").is_file()
    assert not (root / ".empirical-lawhood").exists()

    target = root / preparation.payloads[0].member.relative_path
    target.write_bytes(b"tampered")
    with pytest.raises(ArtifactIdentityConflict):
        publisher.load(preparation.manifest.issue_id)
    with pytest.raises(ArtifactIdentityConflict):
        publisher.publish(preparation, at_utc="2026-07-24T20:32:00Z")


@pytest.mark.parametrize("capacity", ("space", "record"))
def test_guarded_issue_publisher_refuses_capacity_before_any_publication(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    capacity: str,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    root = tmp_path / "external"
    root.mkdir()
    inputs = _issue_inputs()
    authority = inputs["custody_authority"]
    assert isinstance(authority, StudyOperationAuthority)
    preparation = prepare_draft_candidate_issue(**inputs)  # type: ignore[arg-type]
    publisher = _publisher(root, authority, free_bytes=1 if capacity == "space" else 100_000_000)
    if capacity == "record":
        monkeypatch.setattr(
            'empirical_lawhood.infrastructure.study_issue.MAX_ISSUED_PROGRAMME_MANIFEST_BYTES', 1
        )

    with pytest.raises(
        ExternalRootUnavailable if capacity == "space" else ValueError,
        match="free-space" if capacity == "space" else "bounded reader capacity",
    ):
        publisher.publish(preparation, at_utc="2026-07-24T20:32:00Z")

    assert list(root.iterdir()) == []


def _git(*arguments: str, cwd: Path) -> bytes:
    return subprocess.check_output(("git", *arguments), cwd=cwd)


def test_git_source_closure_inspector_reobserves_commit_tree_and_cleanliness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This unit test isolates Git observations. The joint executing-source and
    # clean-commit boundary is exercised in test_source_origin.py subprocesses.
    monkeypatch.setattr(
        "empirical_lawhood.infrastructure.source_origin.require_executing_target_source",
        lambda root: root / "src" / "empirical_lawhood",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    _git("config", "user.name", "Programme Issue Test", cwd=repo)
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "source.py", cwd=repo)
    _git("commit", "-q", "-m", "freeze source", cwd=repo)
    commit = _git("rev-parse", "HEAD", cwd=repo).decode("ascii").strip()
    tree = _git("ls-tree", "-r", "-z", "--full-tree", "HEAD", cwd=repo)
    closure = ImplementationSourceClosure(
        source_closure_id="source-closure.git-inspector",
        kind=SourceClosureKind.CLEAN_GIT_COMMIT,
        implementation_commit=commit,
        implementation_sha256="5" * 64,
        source_tree_sha256=hashlib.sha256(tree).hexdigest(),
        clean_worktree=True,
    )
    inspector = GitStudySourceClosureInspector(repo)

    assert inspector.observe(closure) == closure

    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="dirty"):
        inspector.observe(closure)
    (repo / "untracked.txt").unlink()

    with pytest.raises(PermissionError, match="commit changed"):
        inspector.observe(
            replace(
                closure,
                implementation_commit="b" * 40,
            )
        )
    with pytest.raises(PermissionError, match="source tree changed"):
        inspector.observe(
            replace(
                closure,
                source_tree_sha256="f" * 64,
            )
        )


def test_external_study_authority_store_replays_exact_canonical_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "external"
    root.mkdir()
    inputs = _issue_inputs()
    authority = inputs["custody_authority"]
    assert isinstance(authority, StudyOperationAuthority)
    contract = ExternalRootContract(
        storage_root_id="storage.synthetic-science",
        logical_name="Synthetic programme authority root",
        canonical_path=str(root.resolve()),
        required_mount_path=str(root.resolve()),
        mount_contract_schema='empirical-lawhood/testing/fixtures/study-authority-root',
        minimum_free_bytes=1,
    )
    plane = ExternalArtifactPlane(GuardedExternalRoot(contract, _Inspector(root)))
    store = ExternalStudyOperationAuthorityStore(plane)
    expected = ObjectIdentity.from_record(authority.authority_id, authority)

    assert store.persist(authority) == expected
    assert store.persist(authority) == expected

    assert store.load(authority.authority_id) == authority

    with pytest.raises(FileExistsError, match="another identity"):
        store.persist(replace(authority, expires_at_utc="2026-07-26T20:31:00Z"))

    relative_path = f"authority/programme-operation-authorities/{authority.authority_id}.json"
    (root / relative_path).write_bytes(b"tampered")
    with pytest.raises(ArtifactIdentityConflict):
        store.load(authority.authority_id)


class _PassthroughSourceInspector:
    def observe(
        self,
        expected: ImplementationSourceClosure,
    ) -> ImplementationSourceClosure:
        return expected


def _issue_api_fixture(
    tmp_path: Path,
    *,
    standard_fixture: Callable[
        [], tuple[StudyDefinition, StandardCandidateCompilationContext]
    ] = _standard_fixture,
) -> tuple[EmpiricalLawhoodApi, dict[str, Path], Path, StudyOperationAuthority]:
    authoring, context = standard_fixture()
    raw = authoring.canonical_bytes()
    report = compile_study_candidate(
        authoring_package=authoring,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(raw),
            raw_materialization_sha256=hashlib.sha256(b"application/json\x00" + raw).hexdigest(),
        ),
        context=context,
    )
    assert report.candidate is not None
    candidate = report.candidate
    human = AccountableHumanIdentity(
        human_id="human.owner",
        role_id="role.accountable-proposer",
        identity_provider_id="identity.local-owner",
        identity_subject_sha256="7" * 64,
    )
    human_identity = ObjectIdentity.from_record(human.human_id, human)
    attestation = HumanProposerAttestation(
        attestation_id="proposer-attestation.standard-reference-candidate",
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        proposer=human_identity,
        proposer_id=human.human_id,
        design_origin_id=candidate.base_candidate.design_origin.origin_id,
        design_input_ids=candidate.base_candidate.design_input_ids,
        source_qualification_receipts=(candidate.base_candidate.source_qualification_receipts),
        raw_materialization_sha256=(
            candidate.base_candidate.authoring_materialization.raw_materialization_sha256
        ),
        semantic_config_sha256=candidate.base_candidate.semantic_config_sha256,
        known_exposure_lineage_complete=True,
        evaluation_units_and_seeds_unexposed=True,
        fresh_child_outcomes_observed=False,
        codex_or_chat_is_proposer_attestor_approver_or_issuer=False,
        attested_at_utc="2026-07-24T20:30:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    source_closure = ImplementationSourceClosure(
        source_closure_id="source-closure.standard-reference-candidate",
        kind=SourceClosureKind.CLEAN_GIT_COMMIT,
        implementation_commit="a" * 40,
        implementation_sha256=candidate.base_candidate.implementation_sha256,
        source_tree_sha256="8" * 64,
        clean_worktree=True,
    )
    authority = StudyOperationAuthority(
        authority_id="authority.custody.standard-reference-candidate",
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        prerequisite_authority=None,
        issuer=human_identity,
        grantee_id="operator.issue-service",
        scope_id="scope.standard-reference-candidate-issue",
        storage_root_id="storage.synthetic-science",
        relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        allows_source_acquisition=False,
        allows_external_publication=True,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-07-24T20:31:00Z",
        expires_at_utc="2026-07-25T20:31:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    external = tmp_path / "external"
    external.mkdir()
    paths = {
        "authoring": tmp_path / "programme-authoring-package.json",
        "expected_candidate": tmp_path / "standard-candidate.json",
        "attestation": tmp_path / "proposer-attestation.json",
        "source_closure": tmp_path / "source-closure.json",
        "authority": tmp_path / "custody-authority.json",
    }
    records = {
        "authoring": authoring,
        "expected_candidate": candidate,
        "attestation": attestation,
        "source_closure": source_closure,
        "authority": authority,
    }
    for name, record in records.items():
        assert hasattr(record, "canonical_bytes")
        paths[name].write_bytes(record.canonical_bytes())
    api = EmpiricalLawhoodApi(
        repo_root=tmp_path,
        candidate_context_provider=FixedStandardCandidateContextProvider(context),
        study_issue_publisher=_publisher(external, authority),
        study_source_closure_inspector=_PassthroughSourceInspector(),
        study_issue_clock=FixedDecisionClock(
            clock_id="programme-issue-test-clock",
            value="2026-07-24T20:32:00Z",
        ),
        external_root=external,
    )
    return api, paths, external, authority


def _issue_request(paths: dict[str, Path], *, confirmed: bool) -> IssueStudyRequest:
    return IssueStudyRequest(
        authoring_package_path=paths["authoring"],
        expected_candidate_path=paths["expected_candidate"],
        proposer_attestation_path=paths["attestation"],
        source_closure_path=paths["source_closure"],
        custody_authority_path=paths["authority"],
        confirmed=confirmed,
    )


def test_public_issue_preview_is_write_free_and_confirmed_issue_is_replay_safe(
    tmp_path: Path,
) -> None:
    api, paths, external, _authority = _issue_api_fixture(tmp_path)

    preview = api.issue_study(_issue_request(paths, confirmed=False))

    assert preview.status is OperationStatus.BLOCKED
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert preview.payload is not None
    assert preview.payload.confirmed is False
    assert preview.payload.external_bytes_written == 0
    assert preview.payload.workers_created is False
    assert preview.payload.catalog_mutated is False
    assert preview.payload.authority_issued is False
    assert preview.payload.source_acquired is False
    assert preview.payload.protected_outcomes_read is False
    assert list(external.iterdir()) == []
    assert not (tmp_path / ".empirical-lawhood").exists()

    issued = api.issue_study(_issue_request(paths, confirmed=True))
    repeated = api.issue_study(_issue_request(paths, confirmed=True))

    assert issued.status is OperationStatus.SUCCEEDED
    assert repeated.status is OperationStatus.SUCCEEDED
    assert issued.payload == repeated.payload
    assert issued.payload is not None
    assert issued.payload.confirmed is True
    assert issued.payload.publication_receipt is not None
    assert (
        issued.payload.publication_receipt.total_payload_bytes
        == issued.payload.planned_payload_bytes
    )
    assert issued.payload.publication_receipt_bytes == len(
        issued.payload.publication_receipt.canonical_bytes()
    )
    assert issued.payload.external_bytes_written == (
        issued.payload.planned_payload_bytes + issued.payload.publication_receipt_bytes
    )
    assert not (tmp_path / ".empirical-lawhood").exists()


def test_public_issue_refuses_authority_store_substitution(
    tmp_path: Path,
) -> None:
    api, paths, external, authority = _issue_api_fixture(tmp_path)
    paths["authority"].write_bytes(
        replace(
            authority,
            expires_at_utc="2026-07-26T20:31:00Z",
        ).canonical_bytes()
    )

    result = api.issue_study(_issue_request(paths, confirmed=False))

    assert result.status is OperationStatus.BLOCKED
    assert result.reason_codes == ("PROGRAMME_ISSUE_AUTHORITY_OR_SOURCE_REPLAY_FAILED",)
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("mutation", ("unknown-field", "wrong-version"))
def test_public_issue_rejects_noncanonical_closed_input_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    api, paths, external, _authority = _issue_api_fixture(tmp_path)
    document = json.loads(paths["source_closure"].read_text(encoding="utf-8"))
    if mutation == "unknown-field":
        document["value"]["undeclared"] = True
    else:
        document["version"] = "9.0.0"
    paths["source_closure"].write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    result = api.issue_study(_issue_request(paths, confirmed=False))

    assert result.status is OperationStatus.INVALID
    assert result.reason_codes == ("PROGRAMME_ISSUE_INVALID",)
    assert list(external.iterdir()) == []
