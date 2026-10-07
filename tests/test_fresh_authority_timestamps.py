# SPDX-License-Identifier: MPL-2.0
"""Strict conformance authority and unchanged historical approval replay."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.api.codecs import load_authoring, load_registered_authoring
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalizationError
from empirical_lawhood.planning import approval, response_algebra
from tests.approval_support import (
    FixedDecisionClock, approval_service_for_package, checker_registry_for_attestations,
    deterministic_approval_signer, empty_approval_service,
)


FIXTURES = Path(__file__).parent / "fixtures"
LEGACY_SHA256 = "22d4b361b6a643ee8691c08335a627a395508242f09a65d078b4c19683b94b45"


def legacy_attestation():
    return load_registered_authoring(
        FIXTURES / "legacy-approval-gate.json",
        root_schemas={approval.ApprovalGateAttestation.SCHEMA: approval.ApprovalGateAttestation},
    )


def attestation_inputs():
    legacy = legacy_attestation()
    return dict(
        registration=checker_registry_for_attestations((legacy,)).registrations[0],
        signer=deterministic_approval_signer(f"synthetic-reference:{legacy.gate_id}"),
        attestation_id=legacy.attestation_id, authorization_id=legacy.authorization_id,
        subject=legacy.subject, result=legacy.result, information_cutoff=legacy.information_cutoff,
        checked_at_utc="2026-07-15T00:00:00Z", issued_at_utc="2026-07-15T00:00:00Z",
    )


@pytest.mark.parametrize("field", ("checked_at_utc", "issued_at_utc"))
@pytest.mark.parametrize("value", (
    "2026-07-15T00:00Z", "20260715T000000Z", "2026-07-15T00:00:00,1Z",
    "2026-02-30T00:00:00Z", "2026-07-15T00:00:00+00:00", None,
))
def test_fresh_attestation_refuses_before_signer(field, value):
    args = attestation_inputs()
    real = args["signer"]
    calls = []

    class Signer:
        signature_algorithm = real.signature_algorithm
        signature_version = real.signature_version
        verification_key_hex = real.verification_key_hex

        def sign(self, raw):
            calls.append(raw)
            return real.sign(raw)

    args["signer"] = Signer()
    args[field] = value
    with pytest.raises(ValueError, match=field):
        approval.issue_approval_gate_attestation(**args)
    assert not calls


def test_fresh_fractional_seconds_and_existing_chronology_are_preserved():
    args = attestation_inputs()
    args.update(checked_at_utc="2026-07-15T00:00:00.123456Z", issued_at_utc="2026-07-15T00:00:00.654321Z")
    record = approval.issue_approval_gate_attestation(**args)
    assert record.checked_at_utc == args["checked_at_utc"]
    assert record.issued_at_utc == args["issued_at_utc"]
    checker_registry_for_attestations((record,)).validate(record)
    args["issued_at_utc"] = "2026-07-15T00:00:00.1Z"
    with pytest.raises(ValueError, match="issued before"):
        approval.issue_approval_gate_attestation(**args)


def test_prechange_broader_valid_signed_bytes_replay_through_real_stores():
    raw = (FIXTURES / "legacy-approval-gate.json").read_bytes()
    assert sha256(raw).hexdigest() == LEGACY_SHA256
    legacy = legacy_attestation()
    assert legacy.canonical_bytes() == raw
    assert legacy.checked_at_utc == legacy.issued_at_utc == "2026-07-15T00:00Z"
    package = load_authoring(FIXTURES / "reference-campaign.json")
    attestations = tuple(
        legacy if value.gate_id == legacy.gate_id else value
        for value in package.authorization.envelope.attestations
    )
    authorization = replace(package.authorization, envelope=replace(
        package.authorization.envelope, attestations=attestations,
    ))
    package = replace(package, authorization=authorization)
    service = approval_service_for_package(package)
    frozen = ObjectIdentity.from_record(package.frozen_proposal.frozen_proposal_id, package.frozen_proposal)
    # The decision envelope is strict and all registry/store identity checks run.
    preview = service.preview(
        policy=package.system.authority_policy, frozen_proposal=frozen,
        attestations=tuple(ObjectIdentity.from_record(a.attestation_id, a) for a in attestations),
        authorization_id=authorization.authorization_id, approver_id=authorization.envelope.approver_id,
    )
    assert preview.decision is approval.AuthorizationDecision.APPROVED_NONACTUATING
    replayed = service.replay(
        system=package.system, frozen_proposal=frozen,
        authorization=ObjectIdentity.from_record(authorization.authorization_id, authorization),
    )
    assert replayed.authorization_record_id == authorization.authorization_id
    assert service._attestation_store.load(legacy.attestation_id).canonical_bytes() == raw


@pytest.mark.parametrize("entry", ("preview", "authorize", "source"))
@pytest.mark.parametrize("value", ("2026-07-15T00:00Z", "2026-02-30T00:00:00Z"))
def test_invalid_fresh_decision_clock_refuses_before_store_or_policy(entry, value):
    calls = []

    class Store:
        def __getattr__(self, name):
            def forbidden(*args, **kwargs):
                calls.append(name)
                pytest.fail("invalid fresh timestamp reached a store")
            return forbidden

    service = empty_approval_service()
    service._clock = FixedDecisionClock(clock_id="synthetic.invalid-clock", value=value)
    service._proposal_store = service._study_proposal_store = service._store = Store()
    service._source_acquisition_store = Store()
    identity = ObjectIdentity("synthetic.frozen", approval.FrozenApprovalProposal.SCHEMA, "1.0.0", "a" * 64)
    args = dict(frozen_proposal=identity, attestations=(), authorization_id="synthetic.authorization", approver_id="synthetic.approver")
    with pytest.raises(ValueError, match="trusted_decision_time"):
        if entry == "source":
            service.authorize_source_acquisition(**args)
        else:
            getattr(service, entry)(policy=object(), **args)
    assert not calls


def conformance_inputs():
    package = load_authoring(FIXTURES / "reference-campaign.json")
    policy = replace(package.system.authority_policy, maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH)
    selected = SimpleNamespace(
        authority_action=next(iter(policy.allowed_actions)), world_kind=next(iter(policy.allowed_world_kinds)),
        source_access=next(iter(policy.allowed_source_classes)), resource_budget=policy.budget_ceiling,
        requested_scope_id=policy.scope_ids[0], package_id="synthetic.conformance", fingerprint=lambda: "a" * 64,
        implementation_commit="a" * 40,
    )
    return dict(package=selected, policy=policy, authorization_id="synthetic.conformance-approval",
                passed_gate_ids=policy.required_gate_ids, proposer_id="synthetic.proposer",
                approver_id=policy.delegate_id, decided_at_utc="2026-07-15T00:00:00Z")


@pytest.mark.parametrize("value", ("2026-07-15T00:00Z", "2026-02-30T00:00:00Z"))
def test_conformance_entry_refuses_invalid_time_before_policy(value):
    args = conformance_inputs()
    args.update(decided_at_utc=value, policy=object(), package=object())
    with pytest.raises(ValueError, match="decided_at_utc"):
        response_algebra.authorize_response_algebra_conformance(**args)


def test_conformance_existing_policy_and_expiry_outcomes_are_preserved():
    args = conformance_inputs()
    result = response_algebra.authorize_response_algebra_conformance(**args)
    assert result.decision is response_algebra.ConformanceAuthorizationDecision.APPROVED_NONACTUATING
    args["policy"] = replace(args["policy"], expires_at_utc="2026-07-14T00:00:00Z")
    result = response_algebra.authorize_response_algebra_conformance(**args)
    assert result.decision is response_algebra.ConformanceAuthorizationDecision.AUTHORITY_REQUIRED
    assert "AUTHORITY_POLICY_EXPIRED" in result.reason_codes


@pytest.mark.parametrize("value", (
    "2026-02-30T00:00:00Z", "2026-99-99T99:99:99Z", "garbageTgarbageZ",
    "2026-07-15T00:00Z", "20260715T000000Z", "2026-07-15T00:00:00,1Z",
    "2026-07-15T00:00:00+00:00", "2026-07-15T00:00:00.1234567Z", None,
))
def test_persisted_conformance_refuses_invalid_or_legacy_only_timestamps(value):
    record = response_algebra.authorize_response_algebra_conformance(**conformance_inputs())
    with pytest.raises(ValueError, match="decided_at_utc"):
        replace(record, decided_at_utc=value)
    document = json.loads(record.canonical_bytes())
    document["value"]["decided_at_utc"] = value
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(CanonicalizationError, match="decided_at_utc"):
        decode_canonical_bytes(raw, type(record), maximum_bytes=10000)


@pytest.mark.parametrize("value", (
    "2026-07-15T00:00:00Z", "2026-07-15T00:00:00.1Z",
    "2026-07-15T00:00:00.123456Z", "2024-02-29T23:59:59Z",
))
def test_persisted_conformance_keeps_valid_timestamp_bytes(value):
    record = response_algebra.authorize_response_algebra_conformance(**conformance_inputs())
    # Frozen before the constructor repair, independently of the new decoder.
    assert record.fingerprint() == "c119b549f0f44dc8ee15d61a97d0a07bd7855dfdbe4cbe75041ba4cb0e6bb515"
    record = replace(record, decided_at_utc=value)
    raw = record.canonical_bytes()
    decoded = decode_canonical_bytes(raw, type(record), maximum_bytes=10000)
    assert decoded.decided_at_utc == value
    assert decoded.canonical_bytes() == raw


@pytest.mark.parametrize("value", (
    "2026-02-30T00:00:00Z", "2026-99-99T99:99:99Z", "garbageTgarbageZ",
    "2026-07-15T00:00:00+00:00", None,
))
def test_persisted_approval_owners_refuse_invalid_calendar_or_format(value):
    package = load_authoring(FIXTURES / "reference-campaign.json")
    gate = legacy_attestation()
    # Exercise both attestation material owners before signature validation,
    # and both durable decision owners, rather than only their shared parser.
    for record, field in (
        (gate, "checked_at_utc"), (gate, "issued_at_utc"),
        (gate.unsigned_payload(), "checked_at_utc"),
        (gate.unsigned_payload(), "issued_at_utc"),
        (package.authorization.envelope, "decided_at_utc"),
        (package.authorization, "decided_at_utc"),
    ):
        with pytest.raises(ValueError, match=field):
            replace(record, **{field: value})


@pytest.mark.parametrize("value", (
    "2026-07-15T00:00Z", "20260715T000000Z", "2026-07-15T00:00:00,1Z",
))
def test_persisted_approval_keeps_declared_legacy_timestamp_spellings(value):
    package = load_authoring(FIXTURES / "reference-campaign.json")
    envelope = replace(package.authorization.envelope, decided_at_utc=value)
    authorization = replace(package.authorization, envelope=envelope, decided_at_utc=value)
    raw = authorization.canonical_bytes()
    decoded = decode_canonical_bytes(raw, type(authorization), maximum_bytes=len(raw))
    assert decoded.decided_at_utc == decoded.envelope.decided_at_utc == value
    assert decoded.canonical_bytes() == raw
