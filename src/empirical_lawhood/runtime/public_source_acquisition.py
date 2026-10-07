"""Receipted one-attempt public source custody behind existing approval/artifact ports."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
from typing import ClassVar, Protocol, TypeVar
from urllib.parse import urlsplit

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.approval import CompleteApprovalService, DecisionClock
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.planning.public_source_acquisition import ChecksummedFrozenSourceAcquisitionProposal, CaptureAssuredFrozenSourceAcquisitionProposal, ChecksummedPublicSourceAcquisitionPlan, CaptureAssuredPublicSourceAcquisitionPlan, ChecksummedPublicSourceMember, CaptureAssuredPublicSourceMember, CaptureAssuredSourceAcquisitionApproval, SourceCaptureAssurance, SourceAcquisitionApprovalStore, validate_public_https_url
from empirical_lawhood.runtime.study_issue import StudySourceClosureInspector


SOURCE_ACQUISITION_GRANTEE = "public-source-acquisition-service"
RAW_SOURCE_SCHEMA = 'empirical-lawhood/source/checksummed-uninterpreted-public-bytes'
CAPTURE_ASSURED_RAW_SOURCE_SCHEMA = 'empirical-lawhood/source/capture-assured-uninterpreted-public-bytes'
RecordT = TypeVar("RecordT", bound=CanonicalRecord)


@dataclass(frozen=True, slots=True)
class ChecksummedSourceAcquisitionInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/checksummed-source-acquisition-invocation'
    PROPOSAL_SCHEMA: ClassVar[str] = ChecksummedFrozenSourceAcquisitionProposal.SCHEMA
    APPROVAL_SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-source-acquisition-approval'

    frozen_proposal: ObjectIdentity
    approval: ObjectIdentity
    acquisition_authority: ObjectIdentity
    publication_authority: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.frozen_proposal.object_schema != self.PROPOSAL_SCHEMA
            or self.approval.object_schema != self.APPROVAL_SCHEMA
            or self.acquisition_authority.object_schema != StudyOperationAuthority.SCHEMA
            or self.publication_authority.object_schema != StudyOperationAuthority.SCHEMA
            or self.acquisition_authority == self.publication_authority
        ):
            raise ValueError("source invocation requires separate exact authority identities")


@dataclass(frozen=True, slots=True)
class ChecksummedSourceAcquisitionAttempt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/checksummed-source-acquisition-attempt'
    PLAN_SCHEMA: ClassVar[str] = ChecksummedPublicSourceAcquisitionPlan.SCHEMA
    MEMBER_SCHEMA: ClassVar[str] = ChecksummedPublicSourceMember.SCHEMA
    attempt_id: str
    plan: ObjectIdentity
    member: ObjectIdentity
    invocation: ChecksummedSourceAcquisitionInvocation
    started_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.plan.object_schema != self.PLAN_SCHEMA or (
            self.member.object_schema != self.MEMBER_SCHEMA
        ):
            raise ValueError("source attempt changes its plan/member kind")
        parse_utc_timestamp(self.started_at_utc, field_name="started_at_utc")


@dataclass(frozen=True, slots=True)
class ChecksummedSourceTransferObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/checksummed-source-transfer-observation'
    ATTEMPT_SCHEMA: ClassVar[str] = ChecksummedSourceAcquisitionAttempt.SCHEMA
    observation_id: str
    attempt: ObjectIdentity
    source_sha256: str
    source_size_bytes: int
    final_url_without_query: str
    received_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        if self.attempt.object_schema != self.ATTEMPT_SCHEMA:
            raise ValueError("transfer observation requires its durable attempt")
        validate_sha256(self.source_sha256)
        if (
            type(self.source_size_bytes) is not int
            or not 0 < self.source_size_bytes <= 64 * 1024**2
        ):
            raise ValueError("transfer observation has an invalid byte count")
        validate_public_https_url(self.final_url_without_query)
        if urlsplit(self.final_url_without_query).query:
            raise ValueError("custody metadata cannot expose signed redirect query parameters")
        parse_utc_timestamp(self.received_at_utc, field_name="received_at_utc")


@dataclass(frozen=True, slots=True)
class ChecksummedSourceCustodyReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/checksummed-source-custody-receipt'
    PLAN_SCHEMA: ClassVar[str] = ChecksummedPublicSourceAcquisitionPlan.SCHEMA
    MEMBER_SCHEMA: ClassVar[str] = ChecksummedPublicSourceMember.SCHEMA
    ATTEMPT_SCHEMA: ClassVar[str] = ChecksummedSourceAcquisitionAttempt.SCHEMA
    TRANSFER_SCHEMA: ClassVar[str] = ChecksummedSourceTransferObservation.SCHEMA
    RAW_SCHEMA: ClassVar[str] = RAW_SOURCE_SCHEMA
    receipt_id: str
    plan: ObjectIdentity
    member: ObjectIdentity
    attempt: ObjectIdentity
    transfer: ObjectIdentity
    raw_artifact: ArtifactIdentity
    raw_relative_locator: str
    publication_manifest_sha256: str
    custodied_at_utc: str

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_relative_locator

        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.plan.object_schema != self.PLAN_SCHEMA
            or self.member.object_schema != self.MEMBER_SCHEMA
            or self.attempt.object_schema != self.ATTEMPT_SCHEMA
            or self.transfer.object_schema != self.TRANSFER_SCHEMA
            or self.raw_artifact.payload_schema != self.RAW_SCHEMA
        ):
            raise ValueError("custody receipt changes an acquisition operand schema")
        validate_relative_locator(self.raw_relative_locator)
        validate_sha256(self.publication_manifest_sha256)
        parse_utc_timestamp(self.custodied_at_utc, field_name="custodied_at_utc")


@dataclass(frozen=True, slots=True)
class SourceAcquisitionFailure(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-acquisition-failure'
    failure_id: str
    attempt: ObjectIdentity
    failed_at_utc: str
    reason_code: str

    def __post_init__(self) -> None:
        validate_stable_id(self.failure_id, field_name="failure_id")
        parse_utc_timestamp(self.failed_at_utc, field_name="failed_at_utc")
        if self.reason_code != "SOURCE_ACQUISITION_ATTEMPT_FAILED":
            raise ValueError("source failure must retain the closed operational disposition")


@dataclass(frozen=True, slots=True)
class SourceAcquisitionPreflight(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-acquisition-preflight'
    plan: ObjectIdentity
    member_count: int
    declared_transfer_bytes: int
    maximum_attempts: int
    storage_verified: bool
    implementation_verified: bool
    authority_verified: bool
    source_contacted: bool = False

    def __post_init__(self) -> None:
        if self.source_contacted is not False or not (
            self.storage_verified and self.implementation_verified and self.authority_verified
        ):
            raise ValueError("successful source preflight must be noncontact and fully verified")


@dataclass(frozen=True, slots=True)
class FetchedPublicSource:
    payload: bytes
    final_url_without_query: str
    observed_etag: str | None = None
    observed_media_type: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceAcquisitionInvocation(ChecksummedSourceAcquisitionInvocation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capture-assured-source-acquisition-invocation'
    PROPOSAL_SCHEMA: ClassVar[str] = CaptureAssuredFrozenSourceAcquisitionProposal.SCHEMA
    APPROVAL_SCHEMA: ClassVar[str] = CaptureAssuredSourceAcquisitionApproval.SCHEMA


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceAcquisitionAttempt(ChecksummedSourceAcquisitionAttempt):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capture-assured-source-acquisition-attempt'
    PLAN_SCHEMA: ClassVar[str] = CaptureAssuredPublicSourceAcquisitionPlan.SCHEMA
    MEMBER_SCHEMA: ClassVar[str] = CaptureAssuredPublicSourceMember.SCHEMA
    invocation: CaptureAssuredSourceAcquisitionInvocation


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceTransferObservation(ChecksummedSourceTransferObservation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capture-assured-source-transfer-observation'
    ATTEMPT_SCHEMA: ClassVar[str] = CaptureAssuredSourceAcquisitionAttempt.SCHEMA
    observed_etag: str | None
    observed_media_type: str | None

    def __post_init__(self) -> None:
        super(CaptureAssuredSourceTransferObservation, self).__post_init__()
        for value in (self.observed_etag, self.observed_media_type):
            if value is not None and (len(value) > 256 or any(ord(c) < 32 for c in value)):
                raise ValueError("invalid bounded response metadata")


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceCustodyReceipt(ChecksummedSourceCustodyReceipt):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capture-assured-source-custody-receipt'
    PLAN_SCHEMA: ClassVar[str] = CaptureAssuredPublicSourceAcquisitionPlan.SCHEMA
    MEMBER_SCHEMA: ClassVar[str] = CaptureAssuredPublicSourceMember.SCHEMA
    ATTEMPT_SCHEMA: ClassVar[str] = CaptureAssuredSourceAcquisitionAttempt.SCHEMA
    TRANSFER_SCHEMA: ClassVar[str] = CaptureAssuredSourceTransferObservation.SCHEMA
    RAW_SCHEMA: ClassVar[str] = CAPTURE_ASSURED_RAW_SOURCE_SCHEMA


def acquisition_record_types(
    plan: ChecksummedPublicSourceAcquisitionPlan,
) -> tuple[
    type[ChecksummedSourceAcquisitionAttempt],
    type[ChecksummedSourceTransferObservation],
    type[ChecksummedSourceCustodyReceipt],
]:
    if type(plan) is ChecksummedPublicSourceAcquisitionPlan:
        return ChecksummedSourceAcquisitionAttempt, ChecksummedSourceTransferObservation, ChecksummedSourceCustodyReceipt
    if type(plan) is CaptureAssuredPublicSourceAcquisitionPlan:
        return CaptureAssuredSourceAcquisitionAttempt, CaptureAssuredSourceTransferObservation, CaptureAssuredSourceCustodyReceipt
    raise ValueError("unregistered acquisition plan version")


class PublicSourceFetcher(Protocol):
    transport_key: str
    transport_version: str

    @property
    def implementation_sha256(self) -> str: ...

    def fetch(
        self, member: ChecksummedPublicSourceMember, *, timeout_seconds: int
    ) -> FetchedPublicSource: ...


class PublicSourceOperationAuthorityStore(Protocol):
    def load(self, authority_id: str) -> StudyOperationAuthority: ...


class PublicSourceCustodyPort(SourceAcquisitionApprovalStore, Protocol):
    def preflight(self, plan: ChecksummedPublicSourceAcquisitionPlan) -> None: ...

    def load_record(
        self, plan: ChecksummedPublicSourceAcquisitionPlan, record_id: str, record_type: type[RecordT]
    ) -> RecordT | None: ...

    def publish_record(
        self, plan: ChecksummedPublicSourceAcquisitionPlan, record_id: str, record: CanonicalRecord
    ) -> None: ...

    def reserve_attempt(
        self, plan: ChecksummedPublicSourceAcquisitionPlan, attempt: ChecksummedSourceAcquisitionAttempt
    ) -> None: ...

    def publish_raw(
        self, plan: ChecksummedPublicSourceAcquisitionPlan, member: ChecksummedPublicSourceMember, payload: bytes
    ) -> tuple[ArtifactIdentity, str]: ...

    def recover_raw(
        self,
        plan: ChecksummedPublicSourceAcquisitionPlan,
        member: ChecksummedPublicSourceMember,
        transfer: ChecksummedSourceTransferObservation,
    ) -> tuple[ArtifactIdentity, str]: ...

    def verify_raw(
        self,
        plan: ChecksummedPublicSourceAcquisitionPlan,
        member: ChecksummedPublicSourceMember,
        receipt: ChecksummedSourceCustodyReceipt,
    ) -> None: ...


def verify_public_source_bytes(member: ChecksummedPublicSourceMember, payload: bytes) -> str:
    if len(payload) != member.expected_size_bytes:
        raise ValueError("source size differs from its exact member selection")
    if (
        type(member) is CaptureAssuredPublicSourceMember
        and member.assurance is SourceCaptureAssurance.HTTPS_ORIGIN_CAPTURE
    ):
        # Exact origin/selection was checked by transport. This is a new local
        # custody hash, explicitly not a match to an unavailable upstream digest.
        return sha256(payload).hexdigest()
    observed = (
        md5(payload, usedforsecurity=False).hexdigest()
        if member.upstream_checksum_algorithm == "md5"
        else sha256(payload).hexdigest()
    )
    if observed != member.upstream_checksum:
        raise ValueError("source bytes differ from their pinned upstream checksum")
    return sha256(payload).hexdigest()


class PublicSourceAcquisitionService:
    def __init__(
        self,
        *,
        approval: CompleteApprovalService,
        authority_store: PublicSourceOperationAuthorityStore,
        custody: PublicSourceCustodyPort,
        fetcher: PublicSourceFetcher,
        source_closure_inspector: StudySourceClosureInspector,
        clock: DecisionClock,
    ) -> None:
        self.approval, self.authority_store = approval, authority_store
        self.custody, self.fetcher = custody, fetcher
        self.source_closure_inspector, self.clock = source_closure_inspector, clock

    def _prepare(self, invocation: ChecksummedSourceAcquisitionInvocation) -> ChecksummedPublicSourceAcquisitionPlan:
        frozen = self.approval.replay_source_acquisition(
            authorization=invocation.approval, frozen_proposal=invocation.frozen_proposal
        )
        plan = frozen.plan
        if (
            (
                type(invocation) is ChecksummedSourceAcquisitionInvocation
                and type(plan) is not ChecksummedPublicSourceAcquisitionPlan
            )
            or (
                type(invocation) is CaptureAssuredSourceAcquisitionInvocation
                and type(plan) is not CaptureAssuredPublicSourceAcquisitionPlan
            )
            or type(invocation)
            not in (ChecksummedSourceAcquisitionInvocation, CaptureAssuredSourceAcquisitionInvocation)
        ):
            raise ValueError("invocation changes the frozen acquisition version")
        owner = ObjectIdentity.from_record(
            frozen.proposer_attestation.proposer.human_id, frozen.proposer_attestation.proposer
        )
        for identity, kind in (
            (invocation.acquisition_authority, StudyAuthorityKind.SOURCE_ACQUISITION),
            (invocation.publication_authority, StudyAuthorityKind.CUSTODY_PUBLICATION),
        ):
            authority = self.authority_store.load(identity.object_id)
            if ObjectIdentity.from_record(authority.authority_id, authority) != identity or (
                authority.issuer != owner or authority.scope_id != plan.scope_id
            ):
                raise PermissionError("source operation authority identity/owner/scope differs")
            publication = kind is StudyAuthorityKind.CUSTODY_PUBLICATION
            require_study_authority(
                authority,
                kind=kind,
                subject=ObjectIdentity.from_record(plan.plan_id, plan),
                prerequisite_authority=None,
                grantee_id=SOURCE_ACQUISITION_GRANTEE,
                storage_root_id=plan.storage_root.object_id if publication else None,
                relative_root=plan.relative_root if publication else None,
                at_utc=self.clock.now_utc(),
            )
            if not publication and authority.outcome_access is not plan.outcome_access:
                raise PermissionError("acquisition authority changes the permitted source access")
        if (
            self.fetcher.transport_key != plan.transport_key
            or self.fetcher.transport_version != plan.transport_version
            or self.fetcher.implementation_sha256 != plan.transport_implementation_sha256
        ):
            raise ValueError("registered transport differs from the frozen selection")
        commit = plan.source_closure.implementation_commit
        if commit is None:
            raise ValueError("source acquisition requires a clean implementation commit")
        if self.source_closure_inspector.observe(plan.source_closure) != plan.source_closure:
            raise PermissionError("source acquisition requires its exact clean implementation")
        self.custody.preflight(plan)
        return plan

    def preview(self, invocation: ChecksummedSourceAcquisitionInvocation) -> SourceAcquisitionPreflight:
        plan = self._prepare(invocation)
        return SourceAcquisitionPreflight(
            ObjectIdentity.from_record(plan.plan_id, plan),
            len(plan.members),
            sum(m.expected_size_bytes for m in plan.members),
            len(plan.members),
            True,
            True,
            True,
        )

    def acquire(
        self, invocation: ChecksummedSourceAcquisitionInvocation
    ) -> tuple[ChecksummedSourceCustodyReceipt, ...]:
        plan = self._prepare(invocation)
        return tuple(self._acquire_one(plan, member, invocation) for member in plan.members)

    def recover(
        self, invocation: ChecksummedSourceAcquisitionInvocation
    ) -> tuple[ChecksummedSourceCustodyReceipt, ...]:
        plan = self._prepare(invocation)
        return tuple(self._recover_one(plan, member, invocation) for member in plan.members)

    def _acquire_one(
        self,
        plan: ChecksummedPublicSourceAcquisitionPlan,
        member: ChecksummedPublicSourceMember,
        invocation: ChecksummedSourceAcquisitionInvocation,
    ) -> ChecksummedSourceCustodyReceipt:
        attempt_type, _, receipt_type = acquisition_record_types(plan)
        receipt = self.custody.load_record(plan, f"custody.{member.source_id}", receipt_type)
        if receipt is not None:
            return self._recover_one(plan, member, invocation)
        attempt = attempt_type(
            f"attempt.{member.source_id}",
            ObjectIdentity.from_record(plan.plan_id, plan),
            ObjectIdentity.from_record(member.source_id, member),
            invocation,
            self.clock.now_utc(),
        )
        self.custody.reserve_attempt(plan, attempt)
        try:
            fetched = self.fetcher.fetch(
                member,
                timeout_seconds=max(1, plan.resource_budget.wall_time_seconds // len(plan.members)),
            )
            digest = verify_public_source_bytes(member, fetched.payload)
            if fetched.final_url_without_query not in (
                member.url,
                *member.allowed_redirect_urls_without_query,
            ):
                raise ValueError("transport returned an undeclared final source URL")
            transfer_values = (
                f"transfer.{member.source_id}",
                ObjectIdentity.from_record(attempt.attempt_id, attempt),
                digest,
                len(fetched.payload),
                fetched.final_url_without_query,
                self.clock.now_utc(),
            )
            transfer: ChecksummedSourceTransferObservation
            if type(plan) is CaptureAssuredPublicSourceAcquisitionPlan:
                transfer = CaptureAssuredSourceTransferObservation(
                    *transfer_values, fetched.observed_etag, fetched.observed_media_type
                )
            else:
                transfer = ChecksummedSourceTransferObservation(*transfer_values)
            self.custody.publish_record(plan, transfer.observation_id, transfer)
            artifact, manifest = self.custody.publish_raw(plan, member, fetched.payload)
            return self._finish(plan, member, attempt, transfer, artifact, manifest)
        except Exception:
            failure = SourceAcquisitionFailure(
                f"failure.{member.source_id}",
                ObjectIdentity.from_record(attempt.attempt_id, attempt),
                self.clock.now_utc(),
                "SOURCE_ACQUISITION_ATTEMPT_FAILED",
            )
            self.custody.publish_record(plan, failure.failure_id, failure)
            raise

    def _recover_one(
        self,
        plan: ChecksummedPublicSourceAcquisitionPlan,
        member: ChecksummedPublicSourceMember,
        invocation: ChecksummedSourceAcquisitionInvocation,
    ) -> ChecksummedSourceCustodyReceipt:
        attempt_type, transfer_type, receipt_type = acquisition_record_types(plan)
        attempt = self.custody.load_record(plan, f"attempt.{member.source_id}", attempt_type)
        transfer = self.custody.load_record(plan, f"transfer.{member.source_id}", transfer_type)
        if attempt is None or transfer is None:
            raise RuntimeError("SOURCE_RECOVERY_INCOMPLETE_NO_REACQUISITION")
        if (
            attempt.plan != ObjectIdentity.from_record(plan.plan_id, plan)
            or attempt.invocation != invocation
            or attempt.member != ObjectIdentity.from_record(member.source_id, member)
            or transfer.attempt != ObjectIdentity.from_record(attempt.attempt_id, attempt)
            or transfer.source_size_bytes != member.expected_size_bytes
        ):
            raise ValueError("source recovery changes its exact recorded operands")
        receipt = self.custody.load_record(plan, f"custody.{member.source_id}", receipt_type)
        if receipt is not None:
            if (
                receipt.plan != attempt.plan
                or receipt.member != attempt.member
                or receipt.attempt != transfer.attempt
                or receipt.transfer != ObjectIdentity.from_record(transfer.observation_id, transfer)
                or receipt.raw_artifact.sha256 != transfer.source_sha256
                or receipt.raw_relative_locator != member.relative_locator
            ):
                raise ValueError("custody receipt changes the received byte identity")
            self.custody.verify_raw(plan, member, receipt)
            return receipt
        artifact, manifest = self.custody.recover_raw(plan, member, transfer)
        return self._finish(plan, member, attempt, transfer, artifact, manifest)

    def _finish(
        self,
        plan: ChecksummedPublicSourceAcquisitionPlan,
        member: ChecksummedPublicSourceMember,
        attempt: ChecksummedSourceAcquisitionAttempt,
        transfer: ChecksummedSourceTransferObservation,
        artifact: ArtifactIdentity,
        manifest: str,
    ) -> ChecksummedSourceCustodyReceipt:
        if (
            artifact.sha256 != transfer.source_sha256
            or artifact.size_bytes != member.expected_size_bytes
        ):
            raise ValueError("raw publication changes the received bytes")
        receipt_type = acquisition_record_types(plan)[2]
        receipt = receipt_type(
            f"custody.{member.source_id}",
            ObjectIdentity.from_record(plan.plan_id, plan),
            ObjectIdentity.from_record(member.source_id, member),
            ObjectIdentity.from_record(attempt.attempt_id, attempt),
            ObjectIdentity.from_record(transfer.observation_id, transfer),
            artifact,
            member.relative_locator,
            manifest,
            self.clock.now_utc(),
        )
        self.custody.publish_record(plan, receipt.receipt_id, receipt)
        self.custody.verify_raw(plan, member, receipt)
        return receipt
