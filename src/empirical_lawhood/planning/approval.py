"""Complete outcome-blind approval envelopes and durable authorization acts."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec
from empirical_lawhood.planning.study_authoring import RetrospectiveDesignOrigin

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import ClassVar, Protocol, TypeAlias, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec, validate_experiment_against_system
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import InformationCutoff, parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind

from .authority import AuthorizationDecision
from .design import ExperimentProposal
from .exploration import ProspectiveNomination
from .prospective import NominationObligations, nomination_obligations_extension
from .study_authoring import DesignOrigin, DesignOriginKind
from .public_source_acquisition import ChecksummedFrozenSourceAcquisitionProposal, CaptureAssuredFrozenSourceAcquisitionProposal, SourceAcquisitionApprovalStore, ChecksummedSourceAcquisitionApproval, CaptureAssuredSourceAcquisitionApproval


class ApprovalGateKind(StrEnum):
    IMPLEMENTATION = "IMPLEMENTATION"
    RESOURCE = "RESOURCE"
    CUSTODY = "CUSTODY"
    SUPPORT = "SUPPORT"
    OUTCOME_SEPARATION = "OUTCOME_SEPARATION"
    AUTHORITY = "AUTHORITY"


class AttestationResult(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class ApprovalSignatureAlgorithm(StrEnum):
    """Closed signature algorithms admitted for approval attestations."""

    ED25519 = "ED25519"


APPROVAL_SIGNATURE_VERSION = "1.0.0"


def _validate_lowercase_hex(
    value: str,
    *,
    byte_length: int,
    field_name: str,
) -> bytes:
    if re.fullmatch(rf"[0-9a-f]{{{byte_length * 2}}}", value) is None:
        raise ValueError(
            f"{field_name} must be exactly {byte_length} bytes of lowercase hexadecimal"
        )
    return bytes.fromhex(value)


def _parse_utc(value: str, *, field_name: str) -> datetime:
    """Validate calendar time under the historical signed-record contract.

    Retain ``datetime.fromisoformat`` spellings ending in ``Z``, including
    omitted seconds, compact dates/times and comma fractions. These strings
    remain identity-bearing and are never normalized. Fresh authority inputs
    use the stricter kernel parser before reaching these persisted owners.
    """

    validate_nonempty(value, field_name=field_name)
    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{field_name} is not a valid UTC timestamp") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _validate_utc(value: str, *, field_name: str) -> None:
    _parse_utc(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class ApprovalCheckerRegistration(CanonicalRecord):
    """Immutable trust anchor for one outcome-blind approval checker."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-checker-registration'

    checker_registration_id: str
    gate_id: str
    gate_kind: ApprovalGateKind
    checker_id: str
    implementation_sha256: str
    implementation_version: str
    signature_algorithm: ApprovalSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("checker_registration_id", self.checker_registration_id),
            ("gate_id", self.gate_id),
            ("checker_id", self.checker_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        validate_semantic_version(self.implementation_version)
        if self.signature_algorithm is not ApprovalSignatureAlgorithm.ED25519:
            raise ValueError("unsupported approval attestation signature algorithm")
        if self.signature_version != APPROVAL_SIGNATURE_VERSION:
            raise ValueError("unsupported approval attestation signature version")
        _validate_lowercase_hex(
            self.verification_key_hex,
            byte_length=32,
            field_name="verification_key_hex",
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("approval checkers must be outcome-blind")


@dataclass(frozen=True, slots=True)
class ApprovalCheckerRegistry(CanonicalRecord):
    """Static registry defining the checker trusted for each approval gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-checker-registry'

    registry_id: str
    registrations: tuple[ApprovalCheckerRegistration, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(
            self.registrations,
            attribute="gate_id",
            field_name="registrations",
        )
        checker_ids = tuple(value.checker_id for value in self.registrations)
        registration_ids = tuple(value.checker_registration_id for value in self.registrations)
        if len(set(checker_ids)) != len(checker_ids):
            raise ValueError("approval checker registry contains duplicate checker IDs")
        if len(set(registration_ids)) != len(registration_ids):
            raise ValueError("approval checker registry contains duplicate registration IDs")

    def resolve(self, gate_id: str) -> ApprovalCheckerRegistration:
        for registration in self.registrations:
            if registration.gate_id == gate_id:
                return registration
        raise KeyError(gate_id)

    def validate(self, attestation: ApprovalGateAttestation) -> None:
        """Require exact binding to the independently registered checker."""

        try:
            registration = self.resolve(attestation.gate_id)
        except KeyError as error:
            raise ValueError("attestation uses an unregistered approval gate") from error
        expected_identity = ObjectIdentity.from_record(
            registration.checker_registration_id,
            registration,
        )
        if attestation.authenticated_checker != expected_identity:
            raise ValueError("attestation checker registration identity mismatch")
        if attestation.checker_id != registration.checker_id:
            raise ValueError("attestation checker identity mismatch")
        if attestation.gate_kind is not registration.gate_kind:
            raise ValueError("attestation gate kind mismatch")
        if attestation.checker_implementation_sha256 != registration.implementation_sha256:
            raise ValueError("attestation checker implementation digest mismatch")
        if attestation.checker_implementation_version != registration.implementation_version:
            raise ValueError("attestation checker implementation version mismatch")
        if attestation.signature_algorithm is not registration.signature_algorithm:
            raise ValueError("attestation signature algorithm mismatch")
        if attestation.signature_version != registration.signature_version:
            raise ValueError("attestation signature version mismatch")
        if attestation.verification_key_hex != registration.verification_key_hex:
            raise ValueError("attestation verification key mismatch")
        _verify_approval_attestation_signature(attestation)


@dataclass(frozen=True, slots=True)
class ApprovalGateAttestationPayload(CanonicalRecord):
    """Canonical unsigned bytes covered by an approval-checker signature."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-gate-attestation-payload'

    attestation_id: str
    authorization_id: str
    gate_id: str
    gate_kind: ApprovalGateKind
    subject: ObjectIdentity
    result: AttestationResult
    checker_id: str
    authenticated_checker: ObjectIdentity
    checked_at_utc: str
    issued_at_utc: str
    information_cutoff: InformationCutoff
    checker_implementation_sha256: str
    checker_implementation_version: str
    signature_algorithm: ApprovalSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        _validate_attestation_material(self)


@dataclass(frozen=True, slots=True)
class ApprovalGateAttestation(CanonicalRecord):
    """Typed outcome-blind gate result with a detached Ed25519 signature."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-gate-attestation'

    attestation_id: str
    authorization_id: str
    gate_id: str
    gate_kind: ApprovalGateKind
    subject: ObjectIdentity
    result: AttestationResult
    checker_id: str
    authenticated_checker: ObjectIdentity
    checked_at_utc: str
    issued_at_utc: str
    information_cutoff: InformationCutoff
    checker_implementation_sha256: str
    checker_implementation_version: str
    signature_algorithm: ApprovalSignatureAlgorithm
    signature_version: str
    verification_key_hex: str
    signature_hex: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        _validate_attestation_material(self)
        _validate_lowercase_hex(
            self.signature_hex,
            byte_length=64,
            field_name="signature_hex",
        )
        _verify_approval_attestation_signature(self)

    def unsigned_payload(self) -> ApprovalGateAttestationPayload:
        """Return the exact canonical payload covered by ``signature_hex``."""

        return ApprovalGateAttestationPayload(
            attestation_id=self.attestation_id,
            authorization_id=self.authorization_id,
            gate_id=self.gate_id,
            gate_kind=self.gate_kind,
            subject=self.subject,
            result=self.result,
            checker_id=self.checker_id,
            authenticated_checker=self.authenticated_checker,
            checked_at_utc=self.checked_at_utc,
            issued_at_utc=self.issued_at_utc,
            information_cutoff=self.information_cutoff,
            checker_implementation_sha256=self.checker_implementation_sha256,
            checker_implementation_version=self.checker_implementation_version,
            signature_algorithm=self.signature_algorithm,
            signature_version=self.signature_version,
            verification_key_hex=self.verification_key_hex,
            outcome_access=self.outcome_access,
        )


def _validate_attestation_material(
    value: ApprovalGateAttestation | ApprovalGateAttestationPayload,
) -> None:
    """Validate material fields shared by signed and unsigned records."""

    for name, field_value in (
        ("attestation_id", value.attestation_id),
        ("authorization_id", value.authorization_id),
        ("gate_id", value.gate_id),
        ("checker_id", value.checker_id),
    ):
        validate_stable_id(field_value, field_name=name)
    _validate_utc(value.checked_at_utc, field_name="checked_at_utc")
    _validate_utc(value.issued_at_utc, field_name="issued_at_utc")
    if _parse_utc(
        value.checked_at_utc,
        field_name="checked_at_utc",
    ) > _parse_utc(value.issued_at_utc, field_name="issued_at_utc"):
        raise ValueError("approval attestation cannot be issued before it was checked")
    validate_sha256(
        value.checker_implementation_sha256,
        field_name="checker_implementation_sha256",
    )
    validate_semantic_version(value.checker_implementation_version)
    if value.signature_algorithm is not ApprovalSignatureAlgorithm.ED25519:
        raise ValueError("unsupported approval attestation signature algorithm")
    if value.signature_version != APPROVAL_SIGNATURE_VERSION:
        raise ValueError("unsupported approval attestation signature version")
    _validate_lowercase_hex(
        value.verification_key_hex,
        byte_length=32,
        field_name="verification_key_hex",
    )
    if value.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
        raise ValueError("approval attestations must be outcome-blind")


def _verify_approval_attestation_signature(attestation: ApprovalGateAttestation) -> None:
    public_key_bytes = _validate_lowercase_hex(
        attestation.verification_key_hex,
        byte_length=32,
        field_name="verification_key_hex",
    )
    signature = _validate_lowercase_hex(
        attestation.signature_hex,
        byte_length=64,
        field_name="signature_hex",
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            attestation.unsigned_payload().canonical_bytes(),
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("approval attestation signature verification failed") from error


class ApprovalAttestationSigner(Protocol):
    """Non-canonical private signing boundary used only by a trusted issuer."""

    signature_algorithm: ApprovalSignatureAlgorithm
    signature_version: str
    verification_key_hex: str

    def sign(self, payload: bytes) -> bytes: ...


def issue_approval_gate_attestation(
    *,
    registration: ApprovalCheckerRegistration,
    signer: ApprovalAttestationSigner,
    attestation_id: str,
    authorization_id: str,
    subject: ObjectIdentity,
    result: AttestationResult,
    checked_at_utc: str,
    issued_at_utc: str,
    information_cutoff: InformationCutoff,
) -> ApprovalGateAttestation:
    """Issue one attestation through an independently registered private signer."""

    # Fresh authority uses strict timestamps; persisted signed records retain
    # their original broader parser and exact strings.
    parse_utc_timestamp(checked_at_utc, field_name="checked_at_utc")
    parse_utc_timestamp(issued_at_utc, field_name="issued_at_utc")
    if signer.signature_algorithm is not registration.signature_algorithm:
        raise ValueError("approval signer algorithm differs from checker registration")
    if signer.signature_version != registration.signature_version:
        raise ValueError("approval signer version differs from checker registration")
    if signer.verification_key_hex != registration.verification_key_hex:
        raise ValueError("approval signer key differs from checker registration")
    authenticated_checker = ObjectIdentity.from_record(
        registration.checker_registration_id,
        registration,
    )
    unsigned = ApprovalGateAttestationPayload(
        attestation_id=attestation_id,
        authorization_id=authorization_id,
        gate_id=registration.gate_id,
        gate_kind=registration.gate_kind,
        subject=subject,
        result=result,
        checker_id=registration.checker_id,
        authenticated_checker=authenticated_checker,
        checked_at_utc=checked_at_utc,
        issued_at_utc=issued_at_utc,
        information_cutoff=information_cutoff,
        checker_implementation_sha256=registration.implementation_sha256,
        checker_implementation_version=registration.implementation_version,
        signature_algorithm=registration.signature_algorithm,
        signature_version=registration.signature_version,
        verification_key_hex=registration.verification_key_hex,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    signature = signer.sign(unsigned.canonical_bytes())
    if len(signature) != 64:
        raise ValueError("approval signer returned a non-Ed25519 signature")
    return ApprovalGateAttestation(
        attestation_id=unsigned.attestation_id,
        authorization_id=unsigned.authorization_id,
        gate_id=unsigned.gate_id,
        gate_kind=unsigned.gate_kind,
        subject=unsigned.subject,
        result=unsigned.result,
        checker_id=unsigned.checker_id,
        authenticated_checker=unsigned.authenticated_checker,
        checked_at_utc=unsigned.checked_at_utc,
        issued_at_utc=unsigned.issued_at_utc,
        information_cutoff=unsigned.information_cutoff,
        checker_implementation_sha256=unsigned.checker_implementation_sha256,
        checker_implementation_version=unsigned.checker_implementation_version,
        signature_algorithm=unsigned.signature_algorithm,
        signature_version=unsigned.signature_version,
        verification_key_hex=unsigned.verification_key_hex,
        signature_hex=signature.hex(),
        outcome_access=unsigned.outcome_access,
    )


@dataclass(frozen=True, slots=True)
class ApprovalObligations(CanonicalRecord):
    """Complete authority/custody obligations bound before approval."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-obligations'

    obligations_id: str
    system: ObjectIdentity
    world: ObjectIdentity
    policy: ObjectIdentity
    action: AuthorityAction
    world_kind: WorldKind
    source_access: SourceAccessClass
    requested_scope_id: str
    requested_budget: ResourceBudget
    information_cutoff: InformationCutoff
    implementation_commit: str
    required_gate_ids: tuple[str, ...]
    proposer_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("obligations_id", self.obligations_id),
            ("requested_scope_id", self.requested_scope_id),
            ("proposer_id", self.proposer_id),
        ):
            validate_stable_id(value, field_name=name)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        require_sorted_unique_strings(
            self.required_gate_ids,
            field_name="required_gate_ids",
            allow_empty=False,
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("approval obligations must be outcome-blind")


def approval_obligations_extension(obligations: ApprovalObligations) -> ExtensionBinding:
    return ExtensionBinding(
        namespace="approval-obligations",
        schema=obligations.SCHEMA,
        payload_sha256=obligations.fingerprint(),
    )


@dataclass(frozen=True, slots=True)
class FrozenApprovalProposal(CanonicalRecord):
    """Separately immutable proposal plus its resolved approval obligations."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/frozen-approval-proposal'

    frozen_proposal_id: str
    proposal: ExperimentProposal
    obligations: ApprovalObligations

    def __post_init__(self) -> None:
        validate_stable_id(self.frozen_proposal_id, field_name="frozen_proposal_id")
        expected = approval_obligations_extension(self.obligations)
        if expected not in self.proposal.extensions:
            raise ValueError("proposal does not bind its complete approval obligations")
        if self.proposal.proposed_by != self.obligations.proposer_id:
            raise ValueError("proposal and obligations bind different proposers")
        if self.proposal.budget != self.obligations.requested_budget:
            raise ValueError("proposal and obligations bind different budgets")
        if self.proposal.decision_cutoff != self.obligations.information_cutoff:
            raise ValueError("proposal and obligations bind different information cutoffs")


@dataclass(frozen=True, slots=True)
class IssuedStudyApprovalProposal(CanonicalRecord):
    """Additive ordinary/nomination-compatible proposal over an issued programme."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/issued-study-approval-proposal'

    proposal_id: str
    issued_study: ObjectIdentity
    publication_receipt: ObjectIdentity
    candidate: ObjectIdentity
    candidate_experiment: ExperimentSpec
    design_origin: DesignOrigin
    proposer_attestation: ObjectIdentity
    source_qualification_receipts: tuple[ObjectIdentity, ...]
    alternative_design_ids: tuple[str, ...]
    budget: ResourceBudget
    decision_cutoff: InformationCutoff
    proposed_by: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveApprovalProposal):
            if type(self.candidate_experiment) is not ExperimentSpec:
                raise ValueError(
                    "ProgrammeApprovalProposal requires its original candidate_experiment schema"
                )
            if type(self.design_origin) is not DesignOrigin:
                raise ValueError(
                    "ProgrammeApprovalProposal requires its original design_origin schema"
                )
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        validate_stable_id(self.proposed_by, field_name="proposed_by")
        require_sorted_unique_ids(
            self.source_qualification_receipts,
            attribute="object_id",
            field_name="source_qualification_receipts",
        )
        require_sorted_unique_strings(
            self.alternative_design_ids,
            field_name="alternative_design_ids",
            allow_empty=False,
        )
        if self.candidate_experiment.authorization_record_id is not None:
            raise ValueError("programme proposal cannot contain an authorization")
        if self.candidate_experiment.readiness is not ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("programme proposal experiment must await separate authority")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("programme approval must be blind to fresh-child outcomes")
        if self.parent_visibility_ceiling is not (self.design_origin.parent_visibility_ceiling):
            raise ValueError("programme proposal changes inherited design visibility")
        if self.candidate_experiment.design_visibility_ceiling is not self.visibility_ceiling:
            raise ValueError("programme proposal and experiment visibility differ")
        if (
            self.design_origin.kind is DesignOriginKind.PROSPECTIVE_NOMINATION
            and self.visibility_ceiling.is_promotable
        ):
            raise ValueError("nomination-derived programme proposal remains nonpromotable")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class FrozenIssuedStudyApprovalProposal(CanonicalRecord):
    """Versioned issued-programme proposal plus resolved approval obligations."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/frozen-issued-study-approval-proposal'

    frozen_proposal_id: str
    proposal: IssuedStudyApprovalProposal
    obligations: ApprovalObligations

    def __post_init__(self) -> None:
        if not isinstance(self, FrozenRetrospectiveApproval):
            if type(self.proposal) is not IssuedStudyApprovalProposal:
                raise ValueError(
                    "FrozenProgrammeApprovalProposal requires its original proposal schema"
                )
        validate_stable_id(self.frozen_proposal_id, field_name="frozen_proposal_id")
        expected = approval_obligations_extension(self.obligations)
        if expected not in self.proposal.extensions:
            raise ValueError("programme proposal does not bind its approval obligations")
        if self.proposal.proposed_by != self.obligations.proposer_id:
            raise ValueError("programme proposal and obligations bind different proposers")
        if self.proposal.budget != self.obligations.requested_budget:
            raise ValueError("programme proposal and obligations bind different budgets")
        if self.proposal.decision_cutoff != self.obligations.information_cutoff:
            raise ValueError("programme proposal and obligations bind different cutoffs")


FrozenApprovalRoot: TypeAlias = FrozenApprovalProposal | FrozenIssuedStudyApprovalProposal


def freeze_study_approval_proposal(
    *,
    proposal: IssuedStudyApprovalProposal,
    system: SystemSpec,
    requested_scope_id: str,
    implementation_commit: str,
    authority_action: AuthorityAction,
    source_access: SourceAccessClass,
) -> FrozenIssuedStudyApprovalProposal:
    """Map either accepted design-origin variant into existing obligations."""

    record_type_frozenprogrammeapprovalproposal: type[FrozenIssuedStudyApprovalProposal] = (
        FrozenRetrospectiveApproval
        if isinstance(proposal, RetrospectiveApprovalProposal)
        else FrozenIssuedStudyApprovalProposal
    )
    validate_experiment_against_system(proposal.candidate_experiment, system)
    policy = system.authority_policy
    if requested_scope_id not in policy.scope_ids:
        raise ValueError("approval scope is absent from the bound authority policy")
    obligations = ApprovalObligations(
        obligations_id=f"approval-obligations.{proposal.proposal_id}",
        system=ObjectIdentity.from_record(system.system_id, system),
        world=ObjectIdentity.from_record(system.world.world_id, system.world),
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        action=authority_action,
        world_kind=system.world.kind,
        source_access=source_access,
        requested_scope_id=requested_scope_id,
        requested_budget=proposal.budget,
        information_cutoff=proposal.decision_cutoff,
        implementation_commit=implementation_commit,
        required_gate_ids=policy.required_gate_ids,
        proposer_id=proposal.proposed_by,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    binding = approval_obligations_extension(obligations)
    extensions = tuple(
        sorted(
            (*proposal.extensions, binding),
            key=lambda value: value.namespace,
        )
    )
    require_extensions(extensions)
    bound_proposal = replace(proposal, extensions=extensions)
    return record_type_frozenprogrammeapprovalproposal(
        frozen_proposal_id=f"frozen-programme-approval.{proposal.proposal_id}",
        proposal=bound_proposal,
        obligations=obligations,
    )


def freeze_approval_proposal(
    *,
    proposal: ExperimentProposal,
    nomination: ProspectiveNomination,
    design_obligations: NominationObligations,
    system: SystemSpec,
    requested_scope_id: str,
    implementation_commit: str,
) -> FrozenApprovalProposal:
    """Resolve trusted system/design facts into a separately frozen root."""

    validate_experiment_against_system(proposal.candidate_experiment, system)
    if proposal.nomination != ObjectIdentity.from_record(nomination.nomination_id, nomination):
        raise ValueError("proposal does not bind the supplied prospective nomination")
    if design_obligations.nomination_id != nomination.nomination_id:
        raise ValueError("prospective obligations bind another nomination")
    if nomination_obligations_extension(design_obligations) not in nomination.extensions:
        raise ValueError("nomination does not bind the supplied prospective obligations")
    if proposal.budget != design_obligations.budget:
        raise ValueError("proposal lowers or changes the nomination budget")
    if proposal.decision_cutoff != design_obligations.information_cutoff:
        raise ValueError("proposal changes the nomination information cutoff")
    policy = system.authority_policy
    if requested_scope_id not in policy.scope_ids:
        raise ValueError("approval scope is absent from the bound authority policy")
    obligations = ApprovalObligations(
        obligations_id=f"approval-obligations.{proposal.proposal_id}",
        system=ObjectIdentity.from_record(system.system_id, system),
        world=ObjectIdentity.from_record(system.world.world_id, system.world),
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        action=design_obligations.authority_action,
        world_kind=system.world.kind,
        source_access=design_obligations.source_access,
        requested_scope_id=requested_scope_id,
        requested_budget=proposal.budget,
        information_cutoff=proposal.decision_cutoff,
        implementation_commit=implementation_commit,
        required_gate_ids=policy.required_gate_ids,
        proposer_id=proposal.proposed_by,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    binding = approval_obligations_extension(obligations)
    extensions = tuple(
        sorted(
            (*proposal.extensions, binding),
            key=lambda value: value.namespace,
        )
    )
    require_extensions(extensions)
    bound_proposal = replace(proposal, extensions=extensions)
    return FrozenApprovalProposal(
        frozen_proposal_id=f"frozen-approval.{proposal.proposal_id}",
        proposal=bound_proposal,
        obligations=obligations,
    )


@dataclass(frozen=True, slots=True)
class ApprovalEnvelope(CanonicalRecord):
    """Exact trusted-time envelope presented to the complete approval gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-envelope'

    envelope_id: str
    authorization_id: str
    frozen_proposal: ObjectIdentity
    proposal: ObjectIdentity
    experiment: ObjectIdentity
    obligations: ObjectIdentity
    system: ObjectIdentity
    world: ObjectIdentity
    policy: ObjectIdentity
    action: AuthorityAction
    world_kind: WorldKind
    source_access: SourceAccessClass
    requested_scope_id: str
    requested_budget: ResourceBudget
    information_cutoff: InformationCutoff
    implementation_commit: str
    attestations: tuple[ApprovalGateAttestation, ...]
    proposer_id: str
    approver_id: str
    decided_at_utc: str
    decision_clock_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("envelope_id", self.envelope_id),
            ("authorization_id", self.authorization_id),
            ("requested_scope_id", self.requested_scope_id),
            ("proposer_id", self.proposer_id),
            ("approver_id", self.approver_id),
            ("decision_clock_id", self.decision_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        require_sorted_unique_ids(
            self.attestations,
            attribute="gate_id",
            field_name="attestations",
        )
        if not self.attestations:
            raise ValueError("approval envelope requires typed gate attestations")
        if self.proposer_id == self.approver_id:
            raise ValueError("proposer and approver must be distinct")
        _validate_utc(self.decided_at_utc, field_name="decided_at_utc")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("approval envelope must be outcome-blind")


@dataclass(frozen=True, slots=True)
class DurableAuthorizationRecord(CanonicalRecord):
    """Persisted authorization act bound to a complete approval envelope."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/durable-authorization-record'

    authorization_id: str
    envelope: ApprovalEnvelope
    policy: ObjectIdentity
    proposal: ObjectIdentity
    experiment: ObjectIdentity
    decision: AuthorizationDecision
    reason_codes: tuple[str, ...]
    failed_gate_ids: tuple[str, ...]
    decided_at_utc: str
    decision_clock_id: str
    outcome_access: OutcomeAccess
    plan_mutated: bool
    grants_claim_promotion: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.authorization_id, field_name="authorization_id")
        validate_stable_id(self.decision_clock_id, field_name="decision_clock_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.failed_gate_ids, field_name="failed_gate_ids")
        _validate_utc(self.decided_at_utc, field_name="decided_at_utc")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("authorization act must be outcome-blind")
        if self.plan_mutated:
            raise ValueError("authorization act cannot mutate the proposal")
        if self.grants_claim_promotion:
            raise ValueError("authorization act cannot grant claim promotion")
        if self.authorization_id != self.envelope.authorization_id:
            raise ValueError("authorization act and envelope IDs differ")
        if self.policy != self.envelope.policy:
            raise ValueError("authorization act and envelope policies differ")
        if self.proposal != self.envelope.proposal:
            raise ValueError("authorization act and envelope proposals differ")
        if self.experiment != self.envelope.experiment:
            raise ValueError("authorization act and envelope experiments differ")
        if self.decided_at_utc != self.envelope.decided_at_utc:
            raise ValueError("authorization act and envelope decision times differ")
        if self.decision_clock_id != self.envelope.decision_clock_id:
            raise ValueError("authorization act and envelope clocks differ")

    @property
    def action(self) -> AuthorityAction:
        return self.envelope.action

    @property
    def world_kind(self) -> WorldKind:
        return self.envelope.world_kind

    @property
    def source_access(self) -> SourceAccessClass:
        return self.envelope.source_access

    @property
    def requested_budget(self) -> ResourceBudget:
        return self.envelope.requested_budget

    @property
    def implementation_commit(self) -> str:
        return self.envelope.implementation_commit


@dataclass(frozen=True, slots=True)
class ApprovalDecisionPreview:
    envelope: ApprovalEnvelope
    decision: AuthorizationDecision
    reason_codes: tuple[str, ...]
    failed_gate_ids: tuple[str, ...]


class DecisionClock(Protocol):
    clock_id: str

    def now_utc(self) -> str: ...


class SystemDecisionClock:
    clock_id = "system-utc-clock"

    def now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class AuthorizationRecordStore(Protocol):
    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity: ...

    def load(self, authorization_id: str) -> DurableAuthorizationRecord: ...


class FrozenApprovalProposalStore(Protocol):
    def persist(self, frozen: FrozenApprovalProposal) -> ObjectIdentity: ...

    def load(self, frozen_proposal_id: str) -> FrozenApprovalProposal: ...


class FrozenStudyApprovalProposalStore(Protocol):
    def persist(self, frozen: FrozenIssuedStudyApprovalProposal) -> ObjectIdentity: ...

    def load(self, frozen_proposal_id: str) -> FrozenIssuedStudyApprovalProposal: ...


class ApprovalGateAttestationStore(Protocol):
    """Read boundary for attestations issued into an immutable trusted store."""

    def load(self, attestation_id: str) -> ApprovalGateAttestation: ...


_NONDELEGABLE_SOURCE_ACCESS = {
    SourceAccessClass.CLICK_THROUGH_TERMS,
    SourceAccessClass.PAID,
    SourceAccessClass.PRIVATE_CREDENTIAL,
}


class CompleteApprovalService:
    """Trusted application service; callers cannot restate envelope facts."""

    def __init__(
        self,
        *,
        clock: DecisionClock,
        proposal_store: (FrozenApprovalProposalStore | FrozenStudyApprovalProposalStore),
        study_proposal_store: FrozenStudyApprovalProposalStore | None = None,
        checker_registry: ApprovalCheckerRegistry,
        attestation_store: ApprovalGateAttestationStore,
        store: AuthorizationRecordStore | None = None,
        source_acquisition_store: SourceAcquisitionApprovalStore | None = None,
    ) -> None:
        self._clock = clock
        self._proposal_store = proposal_store
        self._study_proposal_store = (
            cast(FrozenStudyApprovalProposalStore, proposal_store)
            if study_proposal_store is None
            else study_proposal_store
        )
        self._checker_registry = checker_registry
        self._attestation_store = attestation_store
        self._store = store
        self._source_acquisition_store = source_acquisition_store

    def authorize_source_acquisition(
        self,
        *,
        frozen_proposal: ObjectIdentity,
        attestations: tuple[ObjectIdentity, ...],
        authorization_id: str,
        approver_id: str,
    ) -> ChecksummedSourceAcquisitionApproval:
        """Approve a finite archive intake without fabricating an experiment proposal."""

        store = self._source_acquisition_store
        if store is None:
            raise RuntimeError("source acquisition approval store is not composed")
        decided_at = self._clock.now_utc()
        parse_utc_timestamp(decided_at, field_name="trusted_decision_time")
        reasons = self._source_acquisition_reasons(
            frozen_proposal=frozen_proposal,
            attestations=attestations,
            authorization_id=authorization_id,
            approver_id=approver_id,
            decided_at_utc=decided_at,
        )
        approval_type = {
            ChecksummedFrozenSourceAcquisitionProposal.SCHEMA: ChecksummedSourceAcquisitionApproval,
            CaptureAssuredFrozenSourceAcquisitionProposal.SCHEMA: CaptureAssuredSourceAcquisitionApproval,
        }.get(frozen_proposal.object_schema)
        if approval_type is None:
            raise ValueError("unregistered frozen source proposal version")
        record = approval_type(
            authorization_id,
            frozen_proposal,
            attestations,
            self._checker_registry.fingerprint(),
            approver_id,
            decided_at,
            self._clock.clock_id,
            not reasons,
            reasons,
        )
        stored = store.persist_source_approval(record)
        if stored != ObjectIdentity.from_record(authorization_id, record) or (
            store.load_source_approval(authorization_id) != record
        ):
            raise RuntimeError("source approval store failed immutable identity verification")
        if not record.approved:
            raise PermissionError("source acquisition refused: " + ",".join(record.reason_codes))
        return record

    def replay_source_acquisition(
        self, *, authorization: ObjectIdentity, frozen_proposal: ObjectIdentity
    ) -> ChecksummedFrozenSourceAcquisitionProposal:
        """Authenticate recorded checker acts and all immutable source/policy operands."""

        store = self._source_acquisition_store
        if store is None:
            raise RuntimeError("source acquisition approval store is not composed")
        record = store.load_source_approval(authorization.object_id)
        if (
            ObjectIdentity.from_record(record.authorization_id, record) != authorization
            or record.frozen_proposal != frozen_proposal
            or record.checker_registry_sha256 != self._checker_registry.fingerprint()
            or record.decision_clock_id != self._clock.clock_id
            or not record.approved
            or record.reason_codes
            or _parse_utc(record.decided_at_utc, field_name="decided_at_utc")
            > _parse_utc(self._clock.now_utc(), field_name="replay_time")
        ):
            raise PermissionError("source acquisition approval identity/time differs")
        reasons = self._source_acquisition_reasons(
            frozen_proposal=frozen_proposal,
            attestations=record.attestations,
            authorization_id=record.authorization_id,
            approver_id=record.approver_id,
            decided_at_utc=record.decided_at_utc,
        )
        if reasons:
            raise PermissionError(
                "source acquisition approval replay refused: " + ",".join(reasons)
            )
        return store.load_source_proposal(frozen_proposal.object_id)

    def _source_acquisition_reasons(
        self,
        *,
        frozen_proposal: ObjectIdentity,
        attestations: tuple[ObjectIdentity, ...],
        authorization_id: str,
        approver_id: str,
        decided_at_utc: str,
    ) -> tuple[str, ...]:
        store = self._source_acquisition_store
        if store is None:
            raise RuntimeError("source acquisition approval store is not composed")
        frozen = store.load_source_proposal(frozen_proposal.object_id)
        if ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen) != frozen_proposal:
            raise ValueError("source proposal store returned another immutable identity")
        require_sorted_unique_ids(attestations, attribute="object_id", field_name="attestations")
        time = _parse_utc(decided_at_utc, field_name="decided_at_utc")
        proposed = _parse_utc(
            frozen.proposer_attestation.attested_at_utc, field_name="attested_at_utc"
        )
        passed: set[str] = set()
        seen: set[str] = set()
        reasons: set[str] = set()
        for identity in attestations:
            gate = self._attestation_store.load(identity.object_id)
            if (
                ObjectIdentity.from_record(gate.attestation_id, gate) != identity
                or gate.subject != frozen_proposal
                or gate.authorization_id != authorization_id
                or gate.information_cutoff != frozen.plan.information_cutoff
                or gate.gate_id in seen
                or gate.gate_id not in frozen.policy.required_gate_ids
                or not proposed
                <= _parse_utc(gate.checked_at_utc, field_name="checked_at_utc")
                <= _parse_utc(gate.issued_at_utc, field_name="issued_at_utc")
                <= time
            ):
                raise ValueError("source checker attestation identity/cutoff/chronology differs")
            self._checker_registry.validate(gate)
            seen.add(gate.gate_id)
            if gate.result is AttestationResult.PASSED:
                passed.add(gate.gate_id)
            else:
                reasons.add("SOURCE_ACQUISITION_GATE_FAILED")
        reasons.update(
            frozen.policy.refusal_reasons(
                action=AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
                world_kind=frozen.plan.world_kind,
                source_access=SourceAccessClass.OFFICIAL_OPEN_PUBLIC,
                passed_gate_ids=frozenset(passed),
                requested_budget=frozen.plan.resource_budget,
                requested_outcome_access=frozen.plan.outcome_access,
                requested_scope_id=frozen.plan.scope_id,
                proposer_id=frozen.proposer_attestation.proposer.human_id,
                approver_id=approver_id,
                at_utc=decided_at_utc,
            )
        )
        return tuple(sorted(reasons))

    @property
    def decision_clock_id(self) -> str:
        """Identity of the trusted clock composed for authorization and replay."""

        return self._clock.clock_id

    def freeze_proposal(
        self,
        *,
        proposal: ExperimentProposal,
        nomination: ProspectiveNomination,
        design_obligations: NominationObligations,
        system: SystemSpec,
        requested_scope_id: str,
        implementation_commit: str,
    ) -> ObjectIdentity:
        frozen = freeze_approval_proposal(
            proposal=proposal,
            nomination=nomination,
            design_obligations=design_obligations,
            system=system,
            requested_scope_id=requested_scope_id,
            implementation_commit=implementation_commit,
        )
        expected = ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen)
        store = cast(FrozenApprovalProposalStore, self._proposal_store)
        stored = store.persist(frozen)
        if stored != expected or store.load(frozen.frozen_proposal_id) != frozen:
            raise RuntimeError("proposal store failed immutable identity verification")
        return expected

    def freeze_study_proposal(
        self,
        *,
        proposal: IssuedStudyApprovalProposal,
        system: SystemSpec,
        requested_scope_id: str,
        implementation_commit: str,
        authority_action: AuthorityAction,
        source_access: SourceAccessClass,
    ) -> ObjectIdentity:
        """Freeze the additive issued-programme route without a fake nomination."""

        frozen = freeze_study_approval_proposal(
            proposal=proposal,
            system=system,
            requested_scope_id=requested_scope_id,
            implementation_commit=implementation_commit,
            authority_action=authority_action,
            source_access=source_access,
        )
        expected = ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen)
        store = self._study_proposal_store
        stored = store.persist(frozen)
        if stored != expected or store.load(frozen.frozen_proposal_id) != frozen:
            raise RuntimeError("programme proposal store failed immutable identity verification")
        return expected

    def preview(
        self,
        *,
        policy: AuthorityPolicy,
        frozen_proposal: ObjectIdentity,
        attestations: tuple[ObjectIdentity, ...],
        authorization_id: str,
        approver_id: str,
    ) -> ApprovalDecisionPreview:
        decided_at = self._clock.now_utc()
        parse_utc_timestamp(decided_at, field_name="trusted_decision_time")
        frozen = self._load_frozen(frozen_proposal)
        envelope = self._envelope(
            policy=policy,
            frozen=frozen,
            attestations=attestations,
            authorization_id=authorization_id,
            approver_id=approver_id,
            decided_at_utc=decided_at,
        )
        return self._decide(policy=policy, frozen=frozen, envelope=envelope)

    def authorize(
        self,
        *,
        policy: AuthorityPolicy,
        frozen_proposal: ObjectIdentity,
        attestations: tuple[ObjectIdentity, ...],
        authorization_id: str,
        approver_id: str,
    ) -> DurableAuthorizationRecord:
        if self._store is None:
            raise RuntimeError("durable authorization requires an authorization record store")
        preview = self.preview(
            policy=policy,
            frozen_proposal=frozen_proposal,
            attestations=attestations,
            authorization_id=authorization_id,
            approver_id=approver_id,
        )
        if preview.decision is not AuthorizationDecision.APPROVED_NONACTUATING:
            raise PermissionError(
                "authorization was not approved: " + ",".join(preview.reason_codes)
            )
        envelope = preview.envelope
        record = DurableAuthorizationRecord(
            authorization_id=authorization_id,
            envelope=envelope,
            policy=envelope.policy,
            proposal=envelope.proposal,
            experiment=envelope.experiment,
            decision=preview.decision,
            reason_codes=preview.reason_codes,
            failed_gate_ids=preview.failed_gate_ids,
            decided_at_utc=envelope.decided_at_utc,
            decision_clock_id=envelope.decision_clock_id,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            plan_mutated=False,
            grants_claim_promotion=False,
        )
        stored = self._store.persist(record)
        expected = ObjectIdentity.from_record(record.authorization_id, record)
        if stored != expected or self._store.load(authorization_id) != record:
            raise RuntimeError("authorization store failed immutable identity verification")
        return record

    def replay(
        self,
        *,
        system: SystemSpec,
        frozen_proposal: ObjectIdentity,
        authorization: ObjectIdentity,
    ) -> ExperimentSpec:
        """Load and completely replay one previously persisted authorization act."""

        if self._store is None:
            raise RuntimeError("authorization replay requires an authorization record store")
        frozen = self._load_frozen(frozen_proposal)
        record = self._store.load(authorization.object_id)
        if record.authorization_id != authorization.object_id:
            raise ValueError("authorization store returned a substituted record")
        if ObjectIdentity.from_record(record.authorization_id, record) != authorization:
            raise ValueError("authorization store returned a fingerprint-mismatched record")
        trusted_attestations = self._load_trusted_attestations(
            expected=tuple(
                ObjectIdentity.from_record(value.attestation_id, value)
                for value in record.envelope.attestations
            ),
            frozen=frozen,
            authorization_id=record.authorization_id,
            decided_at_utc=record.decided_at_utc,
        )
        if trusted_attestations != record.envelope.attestations:
            raise ValueError("authorization embeds substituted approval attestations")
        return _replay_durable_authorization_semantics(
            system=system,
            frozen=frozen,
            record=record,
            trusted_decision_clock_id=self._clock.clock_id,
        )

    def _load_frozen(self, expected: ObjectIdentity) -> FrozenApprovalRoot:
        if expected.object_schema in {
            FrozenIssuedStudyApprovalProposal.SCHEMA,
            FrozenRetrospectiveApproval.SCHEMA,
        }:
            frozen: FrozenApprovalRoot = self._study_proposal_store.load(expected.object_id)
        elif expected.object_schema == FrozenApprovalProposal.SCHEMA:
            store = cast(FrozenApprovalProposalStore, self._proposal_store)
            frozen = store.load(expected.object_id)
        else:
            raise ValueError("authorization names an unsupported frozen proposal schema")
        if frozen.frozen_proposal_id != expected.object_id:
            raise ValueError("proposal store returned a substituted frozen proposal")
        if ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen) != expected:
            raise ValueError("proposal store returned a fingerprint-mismatched frozen proposal")
        return frozen

    def _envelope(
        self,
        *,
        policy: AuthorityPolicy,
        frozen: FrozenApprovalRoot,
        attestations: tuple[ObjectIdentity, ...],
        authorization_id: str,
        approver_id: str,
        decided_at_utc: str,
    ) -> ApprovalEnvelope:
        obligations = frozen.obligations
        expected_policy = ObjectIdentity.from_record(policy.policy_id, policy)
        if obligations.policy != expected_policy:
            raise ValueError("frozen approval obligations bind another policy")
        decided_at = decided_at_utc
        _validate_utc(decided_at, field_name="trusted_decision_time")
        trusted_attestations = self._load_trusted_attestations(
            expected=attestations,
            frozen=frozen,
            authorization_id=authorization_id,
            decided_at_utc=decided_at,
        )
        return ApprovalEnvelope(
            envelope_id=f"approval-envelope.{authorization_id}",
            authorization_id=authorization_id,
            frozen_proposal=ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen),
            proposal=ObjectIdentity.from_record(frozen.proposal.proposal_id, frozen.proposal),
            experiment=ObjectIdentity.from_record(
                frozen.proposal.candidate_experiment.experiment_id,
                frozen.proposal.candidate_experiment,
            ),
            obligations=ObjectIdentity.from_record(
                obligations.obligations_id,
                obligations,
            ),
            system=obligations.system,
            world=obligations.world,
            policy=obligations.policy,
            action=obligations.action,
            world_kind=obligations.world_kind,
            source_access=obligations.source_access,
            requested_scope_id=obligations.requested_scope_id,
            requested_budget=obligations.requested_budget,
            information_cutoff=obligations.information_cutoff,
            implementation_commit=obligations.implementation_commit,
            attestations=trusted_attestations,
            proposer_id=obligations.proposer_id,
            approver_id=approver_id,
            decided_at_utc=decided_at,
            decision_clock_id=self._clock.clock_id,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )

    def _load_trusted_attestations(
        self,
        *,
        expected: tuple[ObjectIdentity, ...],
        frozen: FrozenApprovalRoot,
        authorization_id: str,
        decided_at_utc: str,
    ) -> tuple[ApprovalGateAttestation, ...]:
        """Load exact store records and revalidate every trust binding."""

        frozen_identity = ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen)
        decision_time = _parse_utc(decided_at_utc, field_name="decided_at_utc")
        loaded: list[ApprovalGateAttestation] = []
        for identity in expected:
            attestation = self._attestation_store.load(identity.object_id)
            if attestation.attestation_id != identity.object_id:
                raise ValueError("attestation store returned a substituted record")
            if ObjectIdentity.from_record(attestation.attestation_id, attestation) != identity:
                raise ValueError("attestation store returned a fingerprint-mismatched record")
            self._checker_registry.validate(attestation)
            if attestation.authorization_id != authorization_id:
                raise ValueError("approval attestation was issued for another authorization")
            if attestation.subject != frozen_identity:
                raise ValueError("approval attestation binds another frozen proposal")
            if attestation.information_cutoff != frozen.obligations.information_cutoff:
                raise ValueError("approval attestation binds another information cutoff")
            if (
                _parse_utc(
                    attestation.issued_at_utc,
                    field_name="issued_at_utc",
                )
                > decision_time
            ):
                raise ValueError("approval attestation was issued after the approval decision")
            loaded.append(attestation)
        return tuple(sorted(loaded, key=lambda value: value.gate_id))

    @staticmethod
    def _decide(
        *,
        policy: AuthorityPolicy,
        frozen: FrozenApprovalRoot,
        envelope: ApprovalEnvelope,
    ) -> ApprovalDecisionPreview:
        obligations = frozen.obligations
        reasons: set[str] = set()
        expected = {
            "frozen_proposal": ObjectIdentity.from_record(
                frozen.frozen_proposal_id,
                frozen,
            ),
            "proposal": ObjectIdentity.from_record(
                frozen.proposal.proposal_id,
                frozen.proposal,
            ),
            "experiment": ObjectIdentity.from_record(
                frozen.proposal.candidate_experiment.experiment_id,
                frozen.proposal.candidate_experiment,
            ),
            "obligations": ObjectIdentity.from_record(
                obligations.obligations_id,
                obligations,
            ),
            "system": obligations.system,
            "world": obligations.world,
            "policy": obligations.policy,
        }
        for field_name, identity in expected.items():
            if getattr(envelope, field_name) != identity:
                reasons.add(f"{field_name.upper()}_IDENTITY_MISMATCH")
        for field_name in (
            "action",
            "world_kind",
            "source_access",
            "requested_scope_id",
            "requested_budget",
            "information_cutoff",
            "implementation_commit",
            "proposer_id",
        ):
            if getattr(envelope, field_name) != getattr(obligations, field_name):
                reasons.add(f"{field_name.upper()}_MISMATCH")
        frozen_identity = expected["frozen_proposal"]
        attested_gate_ids: set[str] = set()
        for attestation in envelope.attestations:
            if attestation.authorization_id != envelope.authorization_id:
                reasons.add("ATTESTATION_AUTHORIZATION_MISMATCH")
            if attestation.subject != frozen_identity:
                reasons.add("ATTESTATION_SUBJECT_MISMATCH")
            if attestation.information_cutoff != obligations.information_cutoff:
                reasons.add("ATTESTATION_CUTOFF_MISMATCH")
            if _parse_utc(
                attestation.issued_at_utc,
                field_name="issued_at_utc",
            ) > _parse_utc(envelope.decided_at_utc, field_name="decided_at_utc"):
                reasons.add("ATTESTATION_ISSUED_AFTER_DECISION")
            if attestation.result is AttestationResult.PASSED:
                attested_gate_ids.add(attestation.gate_id)
            else:
                reasons.add("GATE_ATTESTATION_FAILED")
        if attested_gate_ids - set(policy.required_gate_ids):
            reasons.add("UNEXPECTED_GATE_ATTESTATION")
        failed_gate_ids = tuple(sorted(set(policy.required_gate_ids) - attested_gate_ids))
        if failed_gate_ids:
            reasons.add("REQUIRED_GATES_MISSING")
        reasons.update(
            policy.refusal_reasons(
                action=envelope.action,
                world_kind=envelope.world_kind,
                source_access=envelope.source_access,
                passed_gate_ids=frozenset(attested_gate_ids),
                requested_budget=envelope.requested_budget,
                requested_outcome_access=envelope.outcome_access,
                requested_scope_id=envelope.requested_scope_id,
                proposer_id=envelope.proposer_id,
                approver_id=envelope.approver_id,
                at_utc=envelope.decided_at_utc,
            )
        )
        if envelope.source_access in _NONDELEGABLE_SOURCE_ACCESS:
            reasons.add("NONDELEGABLE_SOURCE_AUTHORITY_REQUIRED")
        if envelope.action in policy.nondelegable_actions:
            reasons.add("NONDELEGABLE_ACTION_AUTHORITY_REQUIRED")
        if any(reason.endswith("AUTHORITY_REQUIRED") for reason in reasons):
            decision = AuthorizationDecision.AUTHORITY_REQUIRED
        elif reasons:
            decision = AuthorizationDecision.REFUSED
        else:
            decision = AuthorizationDecision.APPROVED_NONACTUATING
            reasons.add("COMPLETE_ENVELOPE_APPROVED")
        return ApprovalDecisionPreview(
            envelope=envelope,
            decision=decision,
            reason_codes=tuple(sorted(reasons)),
            failed_gate_ids=failed_gate_ids,
        )


def _replay_durable_authorization_semantics(
    *,
    system: SystemSpec,
    frozen: FrozenApprovalRoot,
    record: DurableAuthorizationRecord,
    trusted_decision_clock_id: str | None = None,
) -> ExperimentSpec:
    """Recompute embedded approval invariants after trusted-store replay.

    This function is deliberately private: embedded records alone do not
    authenticate attestations. ``CompleteApprovalService.replay`` first reloads
    their exact immutable store identities and checker registrations.
    """

    policy = system.authority_policy
    expected_policy = ObjectIdentity.from_record(policy.policy_id, policy)
    envelope = record.envelope
    obligations = frozen.obligations
    expected_proposal = ObjectIdentity.from_record(
        frozen.proposal.proposal_id,
        frozen.proposal,
    )
    expected_experiment = ObjectIdentity.from_record(
        frozen.proposal.candidate_experiment.experiment_id,
        frozen.proposal.candidate_experiment,
    )
    if record.decision is not AuthorizationDecision.APPROVED_NONACTUATING:
        raise ValueError("only a complete approved authorization can make an experiment ready")
    if record.policy != expected_policy or envelope.policy != expected_policy:
        raise ValueError("authorization replay binds another policy")
    if record.proposal != expected_proposal or envelope.proposal != expected_proposal:
        raise ValueError("authorization replay binds another proposal")
    if record.experiment != expected_experiment or envelope.experiment != expected_experiment:
        raise ValueError("authorization replay binds another experiment")
    if envelope.frozen_proposal != ObjectIdentity.from_record(
        frozen.frozen_proposal_id,
        frozen,
    ):
        raise ValueError("authorization replay binds another frozen proposal")
    if obligations.system != ObjectIdentity.from_record(system.system_id, system):
        raise ValueError("authorization replay binds another system")
    if obligations.world != ObjectIdentity.from_record(system.world.world_id, system.world):
        raise ValueError("authorization replay binds another evidence world")
    if obligations.world_kind is not system.world.kind:
        raise ValueError("authorization replay binds another evidence-world kind")
    if obligations.required_gate_ids != policy.required_gate_ids:
        raise ValueError("authorization replay changes the policy-required gates")
    if envelope.envelope_id != f"approval-envelope.{record.authorization_id}":
        raise ValueError("authorization replay has a noncanonical envelope identity")
    if (
        trusted_decision_clock_id is not None
        and record.decision_clock_id != trusted_decision_clock_id
    ):
        raise ValueError("authorization replay binds an untrusted decision clock")

    replayed = CompleteApprovalService._decide(
        policy=policy,
        frozen=frozen,
        envelope=envelope,
    )
    if replayed.decision is not record.decision:
        raise ValueError("authorization replay decision differs from the durable record")
    if replayed.reason_codes != record.reason_codes:
        raise ValueError("authorization replay reasons differ from the durable record")
    if replayed.failed_gate_ids != record.failed_gate_ids:
        raise ValueError("authorization replay failed gates differ from the durable record")
    if replayed.decision is not AuthorizationDecision.APPROVED_NONACTUATING:
        raise ValueError("only a completely replayed approval can make an experiment ready")

    validate_experiment_against_system(frozen.proposal.candidate_experiment, system)
    return replace(
        frozen.proposal.candidate_experiment,
        authorization_record_id=record.authorization_id,
        readiness=ReadinessStatus.READY,
    )


def validate_durable_authorization_structure(
    *,
    system: SystemSpec,
    frozen: FrozenApprovalRoot,
    record: DurableAuthorizationRecord,
) -> None:
    """Validate embedded package consistency without authenticating or granting readiness."""

    _replay_durable_authorization_semantics(
        system=system,
        frozen=frozen,
        record=record,
    )


__all__ = [
    "APPROVAL_SIGNATURE_VERSION",
    "ApprovalAttestationSigner",
    "ApprovalCheckerRegistration",
    "ApprovalCheckerRegistry",
    "ApprovalDecisionPreview",
    "ApprovalEnvelope",
    "ApprovalGateAttestation",
    "ApprovalGateAttestationPayload",
    "ApprovalGateAttestationStore",
    "ApprovalGateKind",
    "ApprovalObligations",
    "AttestationResult",
    "ApprovalSignatureAlgorithm",
    "AuthorizationRecordStore",
    "CompleteApprovalService",
    "DecisionClock",
    "DurableAuthorizationRecord",
    "FrozenApprovalProposal",
    "FrozenApprovalRoot",
    "FrozenApprovalProposalStore",
    'FrozenIssuedStudyApprovalProposal',
    'FrozenStudyApprovalProposalStore',
    'IssuedStudyApprovalProposal',
    "SystemDecisionClock",
    "approval_obligations_extension",
    "freeze_approval_proposal",
    'freeze_study_approval_proposal',
    "issue_approval_gate_attestation",
    "validate_durable_authorization_structure",
]


@dataclass(frozen=True, slots=True)
class RetrospectiveApprovalProposal(IssuedStudyApprovalProposal):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-approval-proposal'

    candidate_experiment: RetrospectiveExperimentSpec
    design_origin: RetrospectiveDesignOrigin

    def __post_init__(self) -> None:
        if type(self.candidate_experiment) is not RetrospectiveExperimentSpec:
            raise ValueError(
                "RetrospectiveApprovalProposal requires its exact candidate_experiment schema"
            )
        if type(self.design_origin) is not RetrospectiveDesignOrigin:
            raise ValueError("RetrospectiveApprovalProposal requires its exact design_origin schema")
        IssuedStudyApprovalProposal.__post_init__(self)


@dataclass(frozen=True, slots=True)
class FrozenRetrospectiveApproval(FrozenIssuedStudyApprovalProposal):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/frozen-retrospective-approval'

    proposal: RetrospectiveApprovalProposal

    def __post_init__(self) -> None:
        if type(self.proposal) is not RetrospectiveApprovalProposal:
            raise ValueError("FrozenRetrospectiveApproval requires its exact proposal schema")
        FrozenIssuedStudyApprovalProposal.__post_init__(self)
