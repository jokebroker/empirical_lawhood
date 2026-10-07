"""Finite public-source intake, separate from operations over held datasets.

These additive records preserve DATASET_REGISTRATION/TRANSFORMATION/BINDING's
no-download contracts. No scientific stage or physical unit is invented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar, Protocol
from urllib.parse import urlsplit

from empirical_lawhood.kernel.authority import AuthorityPolicy, ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import InformationCutoff, parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.study_issue import AccountableHumanIdentity, ImplementationSourceClosure


MAX_ACQUISITION_MEMBERS = 16
MAX_PUBLIC_MEMBER_BYTES = 64 * 1024**2
MAX_ACQUISITION_BYTES = 512 * 1024**2
MAX_ACQUISITION_CONTROL_BYTES = 1024**2


def validate_public_https_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        len(value) > 4096
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port not in (None, 443)
        or any(ord(c) <= 32 for c in value)
    ):
        raise ValueError("public source requires a bounded credential-free HTTPS URL")


@dataclass(frozen=True, slots=True)
class ChecksummedPublicSourceMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-source-member'
    CHECKSUM_LENGTHS: ClassVar[dict[str, int]] = {"md5": 32, "sha256": 64}
    MEDIA_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
            "text/plain",
            "application/pdf",
            "application/zip",
            "chemical/x-mmcif",
        }
    )

    source_id: str
    release_id: str
    url: str
    allowed_redirect_urls_without_query: tuple[str, ...]
    expected_size_bytes: int
    upstream_checksum_algorithm: str
    upstream_checksum: str
    licence_id: str
    media_type: str
    relative_locator: str

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        validate_nonempty(self.release_id, field_name="release_id")
        validate_public_https_url(self.url)
        if urlsplit(self.url).query:
            raise ValueError("initial public source selection cannot contain a signed query")
        require_sorted_unique_strings(
            self.allowed_redirect_urls_without_query,
            field_name="allowed_redirect_urls_without_query",
        )
        if len(self.allowed_redirect_urls_without_query) > 4:
            raise ValueError("redirect destinations exceed the finite roster")
        for url in self.allowed_redirect_urls_without_query:
            validate_public_https_url(url)
            if urlsplit(url).query:
                raise ValueError(
                    "redirect declaration stores the exact public path without secrets"
                )
        if type(self.expected_size_bytes) is not int or not (
            0 < self.expected_size_bytes <= MAX_PUBLIC_MEMBER_BYTES
        ):
            raise ValueError("public source byte envelope is invalid")
        lengths = self.CHECKSUM_LENGTHS
        if (
            self.upstream_checksum_algorithm not in lengths
            or re.fullmatch(
                rf"[0-9a-f]{{{lengths[self.upstream_checksum_algorithm]}}}",
                self.upstream_checksum,
            )
            is None
        ):
            raise ValueError("public source needs an exact upstream checksum")
        validate_nonempty(self.licence_id, field_name="licence_id")
        if self.media_type not in self.MEDIA_TYPES:
            raise ValueError("source media is outside the finite uninterpreted custody roster")
        validate_relative_locator(self.relative_locator)


@dataclass(frozen=True, slots=True)
class ChecksummedPublicSourceAcquisitionPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-source-acquisition-plan'
    WORLD_KINDS: ClassVar[frozenset[WorldKind]] = frozenset({WorldKind.PHYSICAL_EXPERIMENT})
    MEMBER_TYPE: ClassVar[type[ChecksummedPublicSourceMember]] = ChecksummedPublicSourceMember

    plan_id: str
    scope_id: str
    members: tuple[ChecksummedPublicSourceMember, ...]
    storage_root: ObjectIdentity
    relative_root: str
    transport_key: str
    transport_version: str
    transport_implementation_sha256: str
    source_closure: ImplementationSourceClosure
    resource_budget: ResourceBudget
    minimum_free_bytes: int
    outcome_access: OutcomeAccess
    world_kind: WorldKind
    exposure_ledger: ObjectIdentity
    information_cutoff: InformationCutoff
    maximum_attempts_per_member: int = 1

    def __post_init__(self) -> None:
        if any(type(member) is not self.MEMBER_TYPE for member in self.members):
            raise ValueError("source plan requires its exact versioned member records")
        for name in ("plan_id", "scope_id", "transport_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.transport_version != "1.0.0":
            raise ValueError("unsupported public-source transport version")
        validate_sha256(self.transport_implementation_sha256)
        require_sorted_unique_ids(self.members, attribute="source_id", field_name="members")
        if not 1 <= len(self.members) <= MAX_ACQUISITION_MEMBERS:
            raise ValueError("public-source member roster is outside its finite bound")
        total = sum(m.expected_size_bytes for m in self.members)
        if total > MAX_ACQUISITION_BYTES:
            raise ValueError("public-source transfer exceeds its aggregate envelope")
        validate_relative_locator(self.relative_root)
        paths = tuple(m.relative_locator for m in self.members)
        if len(set(paths)) != len(paths) or any(
            not p.startswith(f"{self.relative_root}/raw/") for p in paths
        ):
            raise ValueError("raw source targets must be unique and within the exact scope")
        if self.storage_root.object_schema != 'empirical-lawhood/runtime/external-root-contract':
            raise ValueError("source acquisition requires the guarded external-root contract")
        if (
            self.maximum_attempts_per_member != 1
            or type(self.maximum_attempts_per_member) is not int
        ):
            raise ValueError("Acquisition allows one durable attempt per member")
        if type(self.minimum_free_bytes) is not int or self.minimum_free_bytes <= 0:
            raise ValueError("source acquisition requires an explicit positive space floor")
        budget = self.resource_budget
        if (
            budget.cpu_cores > 4
            or budget.gpu_devices != 0
            or not 1 <= budget.wall_time_seconds <= 3600
            or not (4 * max(m.expected_size_bytes for m in self.members) + 16 * 1024**2)
            <= budget.memory_bytes
            <= 16 * 1024**3
            or budget.source_scan_bytes < 8 * total
            or budget.output_bytes
            < total + (8 + 8 * len(self.members)) * MAX_ACQUISITION_CONTROL_BYTES
        ):
            raise ValueError("acquisition budget omits bounded transfer/verification/control work")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError(
                "this source-intake version supports disclosed historical curation only"
            )
        if self.world_kind not in self.WORLD_KINDS:
            raise ValueError(
                "this intake carries historical physical evidence without new experiments"
            )


@dataclass(frozen=True, slots=True)
class ChecksummedSourceIntakeProposerAttestation(CanonicalRecord):
    """Owner adoption for archive intake, with no claim of unexposed biological labels."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-source-intake-proposer-attestation'
    PLAN_SCHEMA: ClassVar[str] = ChecksummedPublicSourceAcquisitionPlan.SCHEMA

    attestation_id: str
    plan: ObjectIdentity
    proposer: AccountableHumanIdentity
    standing_grant_id: str
    standing_grant_instruction_sha256: str
    exposure_ledger: ObjectIdentity
    known_exposure_lineage_complete: bool
    biological_labels_previously_exposed: bool
    new_physical_experiment_requested: bool
    attested_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attestation_id, field_name="attestation_id")
        validate_stable_id(self.standing_grant_id, field_name="standing_grant_id")
        validate_sha256(self.standing_grant_instruction_sha256)
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        if self.plan.object_schema != self.PLAN_SCHEMA:
            raise ValueError("source proposer attestation binds another plan kind")
        if (
            self.known_exposure_lineage_complete is not True
            or self.biological_labels_previously_exposed is not True
            or self.new_physical_experiment_requested is not False
        ):
            raise ValueError(
                "archive intake must disclose exposure and cannot grant a physical assay"
            )


@dataclass(frozen=True, slots=True)
class ChecksummedFrozenSourceAcquisitionProposal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-frozen-source-acquisition-proposal'
    PLAN_TYPE: ClassVar[type[ChecksummedPublicSourceAcquisitionPlan]] = ChecksummedPublicSourceAcquisitionPlan
    ATTESTATION_TYPE: ClassVar[type[ChecksummedSourceIntakeProposerAttestation]] = (
        ChecksummedSourceIntakeProposerAttestation
    )

    frozen_proposal_id: str
    plan: ChecksummedPublicSourceAcquisitionPlan
    policy: AuthorityPolicy
    proposer_attestation: ChecksummedSourceIntakeProposerAttestation

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not self.PLAN_TYPE
            or type(self.proposer_attestation) is not self.ATTESTATION_TYPE
        ):
            raise ValueError("frozen acquisition requires a matching plan and attestation version")
        validate_stable_id(self.frozen_proposal_id, field_name="frozen_proposal_id")
        if self.proposer_attestation.plan != ObjectIdentity.from_record(
            self.plan.plan_id, self.plan
        ):
            raise ValueError("frozen intake and proposer attestation disagree")
        if self.proposer_attestation.exposure_ledger != self.plan.exposure_ledger:
            raise ValueError("frozen intake changes the inspected exposure ledger")
        if self.proposer_attestation.proposer.human_id != self.policy.delegator_id:
            raise ValueError("source proposer must be the accountable policy owner")
        if not self.policy.budget_ceiling.contains(self.plan.resource_budget):
            raise ValueError("source plan exceeds the adopted policy budget")
        if self.policy.expires_at_utc is not None:
            raise ValueError("perpetual standing grant cannot be silently given an expiry")


@dataclass(frozen=True, slots=True)
class ChecksummedSourceAcquisitionApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/checksummed-source-acquisition-approval'
    PROPOSAL_SCHEMA: ClassVar[str] = ChecksummedFrozenSourceAcquisitionProposal.SCHEMA

    authorization_id: str
    frozen_proposal: ObjectIdentity
    attestations: tuple[ObjectIdentity, ...]
    checker_registry_sha256: str
    approver_id: str
    decided_at_utc: str
    decision_clock_id: str
    approved: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("authorization_id", "approver_id", "decision_clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.frozen_proposal.object_schema != self.PROPOSAL_SCHEMA:
            raise ValueError("acquisition approval requires its frozen source proposal")
        require_sorted_unique_ids(
            self.attestations, attribute="object_id", field_name="attestations"
        )
        if not self.attestations:
            raise ValueError("acquisition approval requires authenticated checker attestations")
        validate_sha256(self.checker_registry_sha256)
        parse_utc_timestamp(self.decided_at_utc, field_name="decided_at_utc")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if type(self.approved) is not bool or self.approved == bool(self.reason_codes):
            raise ValueError("approval decision and refusal reasons disagree")


class SourceCaptureAssurance(StrEnum):
    PROVIDER_CHECKSUM = "PROVIDER_CHECKSUM"
    HTTPS_ORIGIN_CAPTURE = "HTTPS_ORIGIN_CAPTURE"


@dataclass(frozen=True, slots=True)
class CaptureAssuredPublicSourceMember(ChecksummedPublicSourceMember):
    """Capture assurance is explicit; an HTTP ETag never substitutes for a digest."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capture-assured-source-member'
    CHECKSUM_LENGTHS: ClassVar[dict[str, int]] = {
        **ChecksummedPublicSourceMember.CHECKSUM_LENGTHS,
        "none": 0,
    }
    MEDIA_TYPES: ClassVar[frozenset[str]] = ChecksummedPublicSourceMember.MEDIA_TYPES | {"application/gzip"}

    assurance: SourceCaptureAssurance
    evidence_world_kind: WorldKind
    metadata_url: str
    metadata_checked_at_utc: str
    upstream_etag: str | None

    def __post_init__(self) -> None:
        super(CaptureAssuredPublicSourceMember, self).__post_init__()
        validate_public_https_url(self.metadata_url)
        if urlsplit(self.metadata_url).query:
            raise ValueError("metadata URL cannot contain a signed query")
        parse_utc_timestamp(self.metadata_checked_at_utc, field_name="metadata_checked_at_utc")
        if not isinstance(self.assurance, SourceCaptureAssurance) or (
            (self.assurance is SourceCaptureAssurance.HTTPS_ORIGIN_CAPTURE)
            != (self.upstream_checksum_algorithm == "none")
        ):
            raise ValueError("capture assurance and upstream checksum disagree")
        if self.evidence_world_kind not in (
            WorldKind.PHYSICAL_EXPERIMENT,
            WorldKind.NUMERICAL_SIMULATOR,
        ):
            raise ValueError("unsupported source evidence world")
        if self.upstream_etag is not None and (
            not self.upstream_etag
            or len(self.upstream_etag) > 256
            or any(ord(c) < 32 for c in self.upstream_etag)
        ):
            raise ValueError("invalid bounded ETag metadata")


@dataclass(frozen=True, slots=True)
class CaptureAssuredPublicSourceAcquisitionPlan(ChecksummedPublicSourceAcquisitionPlan):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capture-assured-source-acquisition-plan'
    WORLD_KINDS: ClassVar[frozenset[WorldKind]] = frozenset(
        {WorldKind.PHYSICAL_EXPERIMENT, WorldKind.NUMERICAL_SIMULATOR}
    )
    MEMBER_TYPE: ClassVar[type[ChecksummedPublicSourceMember]] = CaptureAssuredPublicSourceMember
    members: tuple[CaptureAssuredPublicSourceMember, ...]

    def __post_init__(self) -> None:
        super(CaptureAssuredPublicSourceAcquisitionPlan, self).__post_init__()
        if any(m.evidence_world_kind is not self.world_kind for m in self.members):
            raise ValueError("different evidence worlds require separate source plans")


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceIntakeProposerAttestation(ChecksummedSourceIntakeProposerAttestation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capture-assured-source-intake-proposer-attestation'
    PLAN_SCHEMA: ClassVar[str] = CaptureAssuredPublicSourceAcquisitionPlan.SCHEMA


@dataclass(frozen=True, slots=True)
class CaptureAssuredFrozenSourceAcquisitionProposal(ChecksummedFrozenSourceAcquisitionProposal):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capture-assured-frozen-source-acquisition-proposal'
    PLAN_TYPE: ClassVar[type[ChecksummedPublicSourceAcquisitionPlan]] = CaptureAssuredPublicSourceAcquisitionPlan
    ATTESTATION_TYPE: ClassVar[type[ChecksummedSourceIntakeProposerAttestation]] = (
        CaptureAssuredSourceIntakeProposerAttestation
    )
    plan: CaptureAssuredPublicSourceAcquisitionPlan
    proposer_attestation: CaptureAssuredSourceIntakeProposerAttestation


@dataclass(frozen=True, slots=True)
class CaptureAssuredSourceAcquisitionApproval(ChecksummedSourceAcquisitionApproval):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/capture-assured-source-acquisition-approval'
    PROPOSAL_SCHEMA: ClassVar[str] = CaptureAssuredFrozenSourceAcquisitionProposal.SCHEMA


class SourceAcquisitionApprovalStore(Protocol):
    def load_source_proposal(self, proposal_id: str) -> ChecksummedFrozenSourceAcquisitionProposal: ...

    def persist_source_proposal(
        self, proposal: ChecksummedFrozenSourceAcquisitionProposal
    ) -> ObjectIdentity: ...

    def load_source_approval(self, authorization_id: str) -> ChecksummedSourceAcquisitionApproval: ...

    def persist_source_approval(self, approval: ChecksummedSourceAcquisitionApproval) -> ObjectIdentity: ...
