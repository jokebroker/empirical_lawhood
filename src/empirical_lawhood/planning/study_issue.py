"""Typed human-attestation, source-closure and operation-authority records.

These records describe prerequisites for issuing and executing a compiled
programme.  Constructing one grants no authority: guarded services must load
the exact record through an independently composed trusted store before the
corresponding operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp


class SourceClosureKind(StrEnum):
    CLEAN_GIT_COMMIT = "CLEAN_GIT_COMMIT"
    EXACT_SOURCE_CLOSURE = "EXACT_SOURCE_CLOSURE"


class StudyAuthorityKind(StrEnum):
    SOURCE_ACQUISITION = "SOURCE_ACQUISITION"
    CUSTODY_PUBLICATION = "CUSTODY_PUBLICATION"
    EXPERIMENT_EXECUTION = "EXPERIMENT_EXECUTION"
    OUTCOME_REVEAL = "OUTCOME_REVEAL"


@dataclass(frozen=True, slots=True)
class AccountableHumanIdentity(CanonicalRecord):
    """Minimal identity record for an accountable human scientific role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/accountable-human-identity'

    human_id: str
    role_id: str
    identity_provider_id: str
    identity_subject_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("human_id", self.human_id),
            ("role_id", self.role_id),
            ("identity_provider_id", self.identity_provider_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.identity_subject_sha256,
            field_name="identity_subject_sha256",
        )


@dataclass(frozen=True, slots=True)
class HumanProposerAttestation(CanonicalRecord):
    """Accountable declaration over the candidate and complete input ledger."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/human-proposer-attestation'

    attestation_id: str
    candidate: ObjectIdentity
    proposer: ObjectIdentity
    proposer_id: str
    design_origin_id: str
    design_input_ids: tuple[str, ...]
    source_qualification_receipts: tuple[ObjectIdentity, ...]
    raw_materialization_sha256: str
    semantic_config_sha256: str
    known_exposure_lineage_complete: bool
    evaluation_units_and_seeds_unexposed: bool
    fresh_child_outcomes_observed: bool
    codex_or_chat_is_proposer_attestor_approver_or_issuer: bool
    attested_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("attestation_id", self.attestation_id),
            ("proposer_id", self.proposer_id),
            ("design_origin_id", self.design_origin_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.design_input_ids,
            field_name="design_input_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.source_qualification_receipts,
            attribute="object_id",
            field_name="source_qualification_receipts",
        )
        validate_sha256(
            self.raw_materialization_sha256,
            field_name="raw_materialization_sha256",
        )
        validate_sha256(
            self.semantic_config_sha256,
            field_name="semantic_config_sha256",
        )
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        if self.proposer.object_id != self.proposer_id:
            raise ValueError("proposer identity and proposer ID differ")
        if not self.known_exposure_lineage_complete:
            raise ValueError("proposer must attest complete known exposure lineage")
        self._validate_evaluation_exposure()
        if self.fresh_child_outcomes_observed:
            raise ValueError("proposer attestation cannot follow fresh-child outcome access")
        if self.codex_or_chat_is_proposer_attestor_approver_or_issuer:
            raise ValueError("Codex/chat cannot hold a scientific or authority role")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("proposer attestation must be blind to fresh-child outcomes")

    def _validate_evaluation_exposure(self) -> None:
        if not self.evaluation_units_and_seeds_unexposed:
            raise ValueError("proposer must attest evaluation-unit and seed exclusion")


@dataclass(frozen=True, slots=True)
class RetrospectiveProposerAttestation(HumanProposerAttestation):
    """Owner adoption of exposed historical evidence; no fresh evaluation attested."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-proposer-attestation'

    def _validate_evaluation_exposure(self) -> None:
        if self.evaluation_units_and_seeds_unexposed:
            raise ValueError("historical proposer cannot attest unexposed evaluation")
        if self.candidate.object_schema not in {
            'empirical-lawhood/runtime/retrospective-standard-candidate',
            'empirical-lawhood/runtime/retrospective-extension-candidate',
        }:
            raise ValueError("historical proposer requires an exact historical candidate")


@dataclass(frozen=True, slots=True)
class ImplementationSourceClosure(CanonicalRecord):
    """Exact implementation closure checked again immediately before issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/implementation-source-closure'

    source_closure_id: str
    kind: SourceClosureKind
    implementation_commit: str
    implementation_sha256: str
    source_tree_sha256: str
    clean_worktree: bool
    exact_source_manifest: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.source_closure_id, field_name="source_closure_id")
        if len(self.implementation_commit) != 40 or any(
            value not in "0123456789abcdef" for value in self.implementation_commit
        ):
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        validate_sha256(self.source_tree_sha256, field_name="source_tree_sha256")
        if self.kind is SourceClosureKind.CLEAN_GIT_COMMIT:
            if not self.clean_worktree:
                raise ValueError("clean Git closure requires a clean worktree")
            if self.exact_source_manifest is not None:
                raise ValueError("clean Git closure cannot substitute an exact-source manifest")
        elif self.exact_source_manifest is None:
            raise ValueError("exact source closure requires its immutable source manifest")


@dataclass(frozen=True, slots=True)
class StudyOperationAuthority(CanonicalRecord):
    """One nontransitive authority grant for exactly one operation class."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-operation-authority'

    authority_id: str
    kind: StudyAuthorityKind
    subject: ObjectIdentity
    prerequisite_authority: ObjectIdentity | None
    issuer: ObjectIdentity
    grantee_id: str
    scope_id: str
    storage_root_id: str | None
    relative_root: str | None
    allows_source_acquisition: bool
    allows_external_publication: bool
    allows_execution: bool
    allows_actuation: bool
    allows_reveal: bool
    issued_at_utc: str
    expires_at_utc: str | None
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("authority_id", self.authority_id),
            ("grantee_id", self.grantee_id),
            ("scope_id", self.scope_id),
        ):
            validate_stable_id(value, field_name=name)
        issued_at = parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if self.expires_at_utc is not None:
            expires_at = parse_utc_timestamp(
                self.expires_at_utc,
                field_name="expires_at_utc",
            )
            if expires_at <= issued_at:
                raise ValueError("operation authority must expire after issue")
        if self.relative_root is not None:
            validate_relative_locator(self.relative_root)
        observed = (
            self.allows_source_acquisition,
            self.allows_external_publication,
            self.allows_execution,
            self.allows_reveal,
        )
        expected = {
            StudyAuthorityKind.SOURCE_ACQUISITION: (True, False, False, False),
            StudyAuthorityKind.CUSTODY_PUBLICATION: (False, True, False, False),
            StudyAuthorityKind.EXPERIMENT_EXECUTION: (False, False, True, False),
            StudyAuthorityKind.OUTCOME_REVEAL: (False, False, False, True),
        }[self.kind]
        if observed != expected:
            raise ValueError("operation authority permission profile differs from its kind")
        if self.kind is StudyAuthorityKind.CUSTODY_PUBLICATION:
            if self.storage_root_id is None or self.relative_root is None:
                raise ValueError("custody publication requires an exact storage scope")
            if self.prerequisite_authority is not None:
                raise ValueError("custody publication cannot inherit another authority")
            if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
                raise ValueError("custody publication must remain outcome-blind")
        else:
            if self.storage_root_id is not None or self.relative_root is not None:
                raise ValueError("nonpublication authority cannot name a storage target")
        if self.kind is StudyAuthorityKind.SOURCE_ACQUISITION:
            if self.prerequisite_authority is not None:
                raise ValueError("source acquisition cannot inherit another authority")
            if self.allows_actuation:
                raise ValueError("source acquisition cannot permit actuation")
        elif self.kind is StudyAuthorityKind.EXPERIMENT_EXECUTION:
            if self.prerequisite_authority is None:
                raise ValueError("execution authority requires scientific approval")
            if self.outcome_access not in {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
            }:
                raise ValueError("execution authority cannot reveal outcomes")
        elif self.kind is StudyAuthorityKind.OUTCOME_REVEAL:
            if self.prerequisite_authority is None:
                raise ValueError("reveal authority requires an exact execution authority")
            if self.allows_actuation:
                raise ValueError("reveal authority cannot permit actuation")
            if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
                raise ValueError("reveal authority requires evaluator-only outcome access")
        elif self.allows_actuation:
            raise ValueError("only experiment execution may permit actuation")


def require_study_authority(
    authority: StudyOperationAuthority,
    *,
    kind: StudyAuthorityKind,
    subject: ObjectIdentity,
    prerequisite_authority: ObjectIdentity | None,
    grantee_id: str,
    storage_root_id: str | None = None,
    relative_root: str | None = None,
    at_utc: str | None = None,
) -> None:
    """Validate an exact loaded authority without granting a later operation."""

    if authority.kind is not kind:
        raise PermissionError("operation authority kind cannot substitute for this act")
    if authority.subject != subject:
        raise PermissionError("operation authority binds another subject")
    if authority.prerequisite_authority != prerequisite_authority:
        raise PermissionError("operation authority binds another prerequisite act")
    if authority.grantee_id != grantee_id:
        raise PermissionError("operation authority binds another grantee")
    if authority.storage_root_id != storage_root_id or authority.relative_root != relative_root:
        raise PermissionError("operation authority binds another storage scope")
    if at_utc is not None:
        checked_at = parse_utc_timestamp(at_utc, field_name="at_utc")
        issued_at = parse_utc_timestamp(
            authority.issued_at_utc,
            field_name="issued_at_utc",
        )
        if checked_at < issued_at:
            raise PermissionError("operation authority is not yet effective")
        if authority.expires_at_utc is not None and checked_at > parse_utc_timestamp(
            authority.expires_at_utc,
            field_name="expires_at_utc",
        ):
            raise PermissionError("operation authority has expired")


__all__ = [
    "AccountableHumanIdentity",
    "HumanProposerAttestation",
    "ImplementationSourceClosure",
    'StudyAuthorityKind',
    'StudyOperationAuthority',
    "SourceClosureKind",
    'require_study_authority',
]
