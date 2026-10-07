# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Disk-independent immutable approval-store composition for follow-up tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from empirical_lawhood.api.codecs import load_authoring
from empirical_lawhood.api.models import CampaignPackage
from empirical_lawhood.infrastructure.authority import Ed25519ApprovalAttestationSigner
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistration,
    ApprovalCheckerRegistry,
    ApprovalGateAttestation,
    CompleteApprovalService,
    DurableAuthorizationRecord,
    FrozenApprovalProposal,
)


def deterministic_approval_signer(label: str) -> Ed25519ApprovalAttestationSigner:
    """Return a deterministic private signer confined to non-production tests."""

    private_bytes = hashlib.sha256(f"test-approval-signer:{label}".encode()).digest()
    return Ed25519ApprovalAttestationSigner.from_private_bytes(private_bytes)


class FixedDecisionClock:
    def __init__(self, *, clock_id: str, value: str) -> None:
        self.clock_id = clock_id
        self.value = value

    def now_utc(self) -> str:
        return self.value


class ImmutableProposalStore:
    def __init__(self, records: tuple[FrozenApprovalProposal, ...] = ()) -> None:
        self._records = {record.frozen_proposal_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("test proposal store identities must be unique")

    def persist(self, record: FrozenApprovalProposal) -> ObjectIdentity:
        observed = self._records.get(record.frozen_proposal_id)
        if observed != record:
            raise ValueError("immutable test proposal store conflict")
        return ObjectIdentity.from_record(record.frozen_proposal_id, record)

    def load(self, frozen_proposal_id: str) -> FrozenApprovalProposal:
        return self._records[frozen_proposal_id]


class ImmutableAuthorizationStore:
    def __init__(self, records: tuple[DurableAuthorizationRecord, ...] = ()) -> None:
        self._records = {record.authorization_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("test authorization store identities must be unique")

    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity:
        observed = self._records.get(record.authorization_id)
        if observed != record:
            raise ValueError("immutable test authorization store conflict")
        return ObjectIdentity.from_record(record.authorization_id, record)

    def load(self, authorization_id: str) -> DurableAuthorizationRecord:
        return self._records[authorization_id]


class ImmutableAttestationStore:
    def __init__(self, records: tuple[ApprovalGateAttestation, ...] = ()) -> None:
        self._records = {record.attestation_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("test attestation store identities must be unique")

    def persist(self, record: ApprovalGateAttestation) -> ObjectIdentity:
        observed = self._records.get(record.attestation_id)
        if observed is not None and observed != record:
            raise ValueError("immutable test attestation store conflict")
        self._records[record.attestation_id] = record
        return ObjectIdentity.from_record(record.attestation_id, record)

    def load(self, attestation_id: str) -> ApprovalGateAttestation:
        return self._records[attestation_id]


def checker_registry_for_attestations(
    attestations: tuple[ApprovalGateAttestation, ...],
    *,
    registry_id: str = "test-approval-checkers",
) -> ApprovalCheckerRegistry:
    registrations = tuple(
        sorted(
            (
                ApprovalCheckerRegistration(
                    checker_registration_id=attestation.authenticated_checker.object_id,
                    gate_id=attestation.gate_id,
                    gate_kind=attestation.gate_kind,
                    checker_id=attestation.checker_id,
                    implementation_sha256=attestation.checker_implementation_sha256,
                    implementation_version=attestation.checker_implementation_version,
                    signature_algorithm=attestation.signature_algorithm,
                    signature_version=attestation.signature_version,
                    verification_key_hex=attestation.verification_key_hex,
                    outcome_access=attestation.outcome_access,
                )
                for attestation in attestations
            ),
            key=lambda value: value.gate_id,
        )
    )
    registry = ApprovalCheckerRegistry(
        registry_id=registry_id,
        registrations=registrations,
    )
    for attestation in attestations:
        registry.validate(attestation)
    return registry


def attestation_identities(
    attestations: tuple[ApprovalGateAttestation, ...],
) -> tuple[ObjectIdentity, ...]:
    return tuple(
        ObjectIdentity.from_record(attestation.attestation_id, attestation)
        for attestation in attestations
    )


def approval_service_for_package(package: CampaignPackage) -> CompleteApprovalService:
    attestations = package.authorization.envelope.attestations
    return CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id=package.authorization.decision_clock_id,
            value=package.authorization.decided_at_utc,
        ),
        proposal_store=ImmutableProposalStore((package.frozen_proposal,)),
        checker_registry=checker_registry_for_attestations(attestations),
        attestation_store=ImmutableAttestationStore(attestations),
        store=ImmutableAuthorizationStore((package.authorization,)),
    )


def approval_service_for_path(path: Path) -> CompleteApprovalService:
    package = load_authoring(path)
    if not isinstance(package, CampaignPackage):
        raise TypeError("test approval composition requires a follow-up CampaignPackage")
    return approval_service_for_package(package)


def empty_approval_service() -> CompleteApprovalService:
    return CompleteApprovalService(
        clock=FixedDecisionClock(
            clock_id="empty-test-decision-clock",
            value="2026-07-15T00:00:00Z",
        ),
        proposal_store=ImmutableProposalStore(),
        checker_registry=ApprovalCheckerRegistry(
            registry_id="empty-test-approval-checkers",
            registrations=(),
        ),
        attestation_store=ImmutableAttestationStore(),
        store=ImmutableAuthorizationStore(),
    )


__all__ = [
    "ImmutableAttestationStore",
    "attestation_identities",
    "approval_service_for_package",
    "approval_service_for_path",
    "checker_registry_for_attestations",
    "deterministic_approval_signer",
    "empty_approval_service",
]
