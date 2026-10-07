# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from empirical_lawhood.api.codecs import AuthoringCodecError, MAX_AUTHORING_BYTES, load_authoring, load_standard_issued_study, load_standard_study_candidate
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest, IssueExtensionsRequest, OperationStatus
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedExtensionPayloadSource
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.candidate_compiler import bind_standard_candidate_extensions
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.static_codecs import build_canonical_record_codec_registry
from tests.approval_support import FixedDecisionClock
from tests.issued_study_support import _extension_plan
from tests.runtime_platform.test_study_issue import _issue_api_fixture, _issue_request, _publisher
from tests.runtime_platform.test_standard_formal_candidate import _standard_fixture
from tests.standard_candidate_support import FixedStandardCandidateContextProvider


def _write(path: Path, record: CanonicalRecord) -> None:
    path.write_bytes(record.canonical_bytes())


def test_public_compile_preview_publish_restart_and_hostile_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_api, base_paths, external, _ = _issue_api_fixture(tmp_path)
    base_result = base_api.issue_study(_issue_request(base_paths, confirmed=True))
    assert base_result.status is OperationStatus.SUCCEEDED
    assert base_result.payload is not None
    assert base_result.payload.publication_receipt is not None
    base_manifest = base_result.payload.manifest

    extension_plan = _extension_plan()
    extension_payload = extension_plan.canonical_bytes()
    decoder_digest = hashlib.sha256(b"executable-study-decoder-config").hexdigest()
    proposal = ProposedStudyExtension(
        extension_id="extension.public-executable-study",
        namespace_id="fixed-round-acquisition",
        payload=ObjectIdentity.from_record("acquisition.public-executable-study", extension_plan),
        payload_size_bytes=len(extension_payload),
        decoder_key="decoder.public-executable-study",
        decoder_version="1.0.0",
        decoder_config_sha256=decoder_digest,
        required_for_activation=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    extension_set = ProposedStudyExtensionSet(
        extension_set_id="extension-set.public-executable-study",
        authoring_package=ObjectIdentity.from_record(
            base_manifest.authoring_package.package_id,
            base_manifest.authoring_package,
        ),
        namespace_roster=("fixed-round-acquisition",),
        extensions=(proposal,),
    )
    executable_authoring = ExecutableStudyDefinition(
        package_id="authoring-package.public-executable-study",
        base=base_manifest.authoring_package,
        extension_set=extension_set,
    )
    executable_candidate = bind_standard_candidate_extensions(
        base_candidate=base_manifest.candidate,
        authoring_package=executable_authoring,
    )
    registration = StudyExtensionDecoderRegistration(
        registration_id="decoder-registration.public-executable-study",
        decoder_key=proposal.decoder_key,
        decoder_version=proposal.decoder_version,
        payload_schema=extension_plan.SCHEMA,
        payload_version=extension_plan.VERSION,
        config_sha256=decoder_digest,
        implementation_sha256=hashlib.sha256(b"public-executable-study-decoder").hexdigest(),
        maximum_payload_bytes=1_000_000,
    )
    wrapper_payload = executable_authoring.canonical_bytes()
    attestation = replace(
        base_manifest.proposer_attestation,
        attestation_id="proposer-attestation.public-executable-study",
        candidate=ObjectIdentity.from_record(executable_candidate.candidate_id, executable_candidate),
        raw_materialization_sha256=hashlib.sha256(
            b"application/json\x00" + wrapper_payload
        ).hexdigest(),
    )
    authority = StudyOperationAuthority(
        authority_id="authority.custody.public-executable-study",
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=ObjectIdentity.from_record(executable_candidate.candidate_id, executable_candidate),
        prerequisite_authority=None,
        issuer=attestation.proposer,
        grantee_id="operator.issue-service",
        scope_id="scope.public-executable-study",
        storage_root_id="storage.synthetic-science",
        relative_root="issued-programmes",
        allows_source_acquisition=False,
        allows_external_publication=True,
        allows_execution=False,
        allows_actuation=False,
        allows_reveal=False,
        issued_at_utc="2026-07-24T20:31:00Z",
        expires_at_utc="2026-07-25T20:31:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    paths = {
        "authoring": tmp_path / "executable-study-authoring-package.json",
        "payload": tmp_path / "extension-public-executable-study.json",
        "registration": tmp_path / "decoder-registration-public-executable-study.json",
        "candidate": tmp_path / "candidate-public-executable-study.json",
        "attestation": tmp_path / "attestation-public-executable-study.json",
        "authority": tmp_path / "authority-public-executable-study.json",
    }
    _write(paths["authoring"], executable_authoring)
    paths["payload"].write_bytes(extension_payload)
    _write(paths["registration"], registration)
    _write(paths["candidate"], executable_candidate)
    _write(paths["attestation"], attestation)
    _write(paths["authority"], authority)

    base_authoring, context = _standard_fixture()
    assert base_authoring == base_manifest.authoring_package
    codecs = build_canonical_record_codec_registry(
        registry_id="public-executable-study-test-codecs",
        record_types=(type(extension_plan),),
        maximum_bytes_by_schema={extension_plan.SCHEMA: 1_000_000},
        decoder_implementation_sha256=registration.implementation_sha256,
    )

    def api() -> EmpiricalLawhoodApi:
        return EmpiricalLawhoodApi(
            repo_root=tmp_path,
            candidate_context_provider=FixedStandardCandidateContextProvider(context),
            study_issue_publisher=_publisher(external, authority),
            study_issue_clock=FixedDecisionClock(
                clock_id="executable-study-issue-clock",
                value="2026-07-24T20:32:00Z",
            ),
            study_extension_codec_registry=codecs,
            external_root=external,
        )

    compile_request = CompileCandidateRequest(
        path=paths["authoring"],
        extension_payload_paths=(paths["payload"],),
        decoder_registration_paths=(paths["registration"],),
    )
    compiled = api().compile_candidate(compile_request)
    assert compiled.status is OperationStatus.SUCCEEDED
    assert compiled.payload is not None
    assert compiled.payload.report.candidate == executable_candidate
    assert compiled.payload.report.extension_materializations[0].payload == proposal.payload

    candidate_bytes = paths["candidate"].read_bytes()
    paths["candidate"].write_bytes(
        candidate_bytes + b" " * (MAX_AUTHORING_BYTES + 1 - len(candidate_bytes))
    )
    with pytest.raises(AuthoringCodecError, match="byte limit"):
        load_authoring(paths["candidate"])
    assert load_standard_study_candidate(paths["candidate"]) == executable_candidate
    with pytest.raises(AuthoringCodecError, match="unsupported authoring root"):
        load_standard_study_candidate(paths["authoring"])
    with monkeypatch.context() as limits:
        limits.setattr("empirical_lawhood.api.codecs.MAX_COMPOSED_PROGRAMME_CONTROL_BYTES", 1)
        with pytest.raises(AuthoringCodecError, match="byte limit"):
            load_standard_study_candidate(paths["candidate"])
    with monkeypatch.context() as limits:
        limits.setattr("empirical_lawhood.api.codecs.MAX_AUTHORING_NODES", 1)
        assert load_standard_study_candidate(paths["candidate"]) == executable_candidate
        limits.setattr("empirical_lawhood.api.codecs.MAX_COMPOSED_CONTROL_NODES", 1)
        with pytest.raises(AuthoringCodecError, match="node limit"):
            load_standard_study_candidate(paths["candidate"])

    issue_request = IssueExtensionsRequest(
        authoring_package_path=paths["authoring"],
        extension_payload_paths=(paths["payload"],),
        decoder_registration_paths=(paths["registration"],),
        expected_candidate_path=paths["candidate"],
        base_issue_id=base_manifest.issue_id,
        extension_proposer_attestation_path=paths["attestation"],
        extension_custody_authority_path=paths["authority"],
        confirmed=False,
    )
    preview = api().issue_extensions(issue_request)
    assert preview.status is OperationStatus.BLOCKED
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert preview.payload is not None
    assert preview.payload.external_bytes_written == 0

    confirmed_request = replace(issue_request, confirmed=True)
    issued = api().issue_extensions(confirmed_request)
    replayed_after_restart = api().issue_extensions(confirmed_request)
    assert issued.status is OperationStatus.SUCCEEDED
    assert replayed_after_restart.status is OperationStatus.SUCCEEDED
    assert issued.payload == replayed_after_restart.payload
    assert issued.payload is not None
    assert issued.payload.publication_receipt is not None
    composed_path = tmp_path / "composed-manifest.json"
    manifest_bytes = issued.payload.manifest.canonical_bytes()
    composed_path.write_bytes(
        manifest_bytes + b" " * (MAX_AUTHORING_BYTES + 1 - len(manifest_bytes))
    )
    with pytest.raises(AuthoringCodecError, match="byte limit"):
        load_authoring(composed_path)
    assert load_standard_issued_study(composed_path) == issued.payload.manifest
    with pytest.raises(AuthoringCodecError, match="unsupported authoring root"):
        load_standard_issued_study(paths["authoring"])
    with monkeypatch.context() as limits:
        limits.setattr("empirical_lawhood.api.codecs.MAX_AUTHORING_NODES", 1)
        assert load_standard_issued_study(composed_path) == issued.payload.manifest
        limits.setattr("empirical_lawhood.api.codecs.MAX_COMPOSED_CONTROL_NODES", 1)
        with pytest.raises(AuthoringCodecError, match="node limit"):
            load_standard_issued_study(composed_path)
    assert issued.payload.publication_receipt.declared_member_ids == tuple(
        value.member_id for value in issued.payload.manifest.members
    )
    assert issued.payload.publication_receipt.base_publication_receipt == (
        ObjectIdentity.from_record(
            base_result.payload.publication_receipt.receipt_id,
            base_result.payload.publication_receipt,
        )
    )
    published_member = issued.payload.manifest.issued_extensions.members[0]
    authenticated_source = ExternalIssuedExtensionPayloadSource(_publisher(external, authority))
    assert (
        authenticated_source.read_member(
            issue_id=issued.payload.manifest.issue_id,
            member=published_member,
            maximum_bytes=registration.maximum_payload_bytes,
        )
        == extension_payload
    )

    publisher = authenticated_source._publisher
    original_load = publisher.load_executable_study
    load_calls = []

    def recorded_load(issue_id):  # type: ignore[no-untyped-def]
        load_calls.append(issue_id)
        return original_load(issue_id)

    monkeypatch.setattr(publisher, 'load_executable_study', recorded_load)
    requests = ((published_member, registration.maximum_payload_bytes),)
    for _ in range(2):
        assert authenticated_source.read_members(
            issue_id=issued.payload.manifest.issue_id, requests=requests
        ) == (extension_payload,)
    assert len(load_calls) == 2  # No authentication survives a read operation.
    with pytest.raises(ValueError, match="duplicated"):
        authenticated_source.read_members(
            issue_id=issued.payload.manifest.issue_id, requests=requests * 2
        )
    with pytest.raises(ValueError, match="read bound"):
        authenticated_source.read_members(
            issue_id=issued.payload.manifest.issue_id, requests=((published_member, 1),)
        )
    assert len(load_calls) == 2
    published_path = publisher.artifact_plane.root.resolve(
        published_member.relative_path, for_write=False
    )
    published_path.write_bytes(extension_payload + b" ")
    with pytest.raises((RuntimeError, ValueError)):
        authenticated_source.read_members(
            issue_id=issued.payload.manifest.issue_id, requests=requests
        )
    published_path.write_bytes(extension_payload)

    paths["payload"].write_bytes(extension_payload + b" ")
    hostile = api().compile_candidate(compile_request)
    assert hostile.status is OperationStatus.INVALID
    assert hostile.reason_codes == ("EXTENSION_CANDIDATE_INVALID",)
