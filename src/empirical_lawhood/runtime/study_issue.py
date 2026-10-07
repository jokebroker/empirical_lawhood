"""Deterministic validation and materialization plan for programme issue/freeze."""

from __future__ import annotations

from empirical_lawhood.runtime.candidate_compiler import RetrospectiveStandardCandidate, RetrospectiveExtensionCandidate
from empirical_lawhood.planning.experiment_entry import RetrospectiveAuthoringBase
from empirical_lawhood.planning.approval import RetrospectiveApprovalProposal

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Protocol, TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.study_authoring import MaterializationQualificationReceipt, StudyDraft, SourceAccessDisposition
from empirical_lawhood.planning.approval import IssuedStudyApprovalProposal
from empirical_lawhood.planning.experiment_entry import ExperimentEntryPackage, StudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet, validate_experiment_entry
from empirical_lawhood.planning.study_issue import HumanProposerAttestation, RetrospectiveProposerAttestation, ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority, require_study_authority

from .artifacts import (
    MAX_ARTIFACT_PUBLICATION_MEMBERS,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    LogicalArtifactIdentity,
)
from .candidate_compiler import CandidateAxisState, CandidateCompilationDisposition, CandidateCompilationReport, DraftStudyCandidate, StudyCompilationReport, ExecutableStudyCompilationReport, StudyCandidate, ExecutableStudyCandidate
from .capabilities import CapabilityRegistry
from .study_extensions import StudyExtensionMaterializationReceipt


MAX_ISSUE_MEMBERS = MAX_ARTIFACT_PUBLICATION_MEMBERS
MAX_ISSUE_PAYLOAD_BYTES = 64 * 1024 * 1024
# Generated standard reports duplicate full candidates and their provenance.
# These finite capacities do not enlarge raw authoring or extension payloads.
MAX_STANDARD_PROGRAMME_CONTROL_BYTES = 256 * 1024**2
MAX_COMPOSED_PROGRAMME_CONTROL_BYTES = 256 * 1024**2
ISSUE_PUBLICATION_RELATIVE_ROOT = "issued-programmes"


_ExtensionRecordT = TypeVar("_ExtensionRecordT", bound=CanonicalRecord)


class StudySourceClosureInspector(Protocol):
    """Re-observe the exact implementation closure immediately before issue."""

    def observe(
        self,
        expected: ImplementationSourceClosure,
    ) -> ImplementationSourceClosure: ...


@dataclass(frozen=True, slots=True)
class IssuedStudyMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-study-member'

    member_id: str
    relative_path: str
    payload_schema: str
    media_type: str
    profile: ArtifactProfile
    size_bytes: int
    physical_sha256: str
    logical_content_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_relative_locator(self.relative_path)
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        maximum = (
            MAX_STANDARD_PROGRAMME_CONTROL_BYTES
            if self.payload_schema == StudyCompilationReport.SCHEMA
            and self.profile is ArtifactProfile.CANONICAL_JSON
            else MAX_ISSUE_PAYLOAD_BYTES
        )
        if self.size_bytes <= 0 or self.size_bytes > maximum:
            raise ValueError("issued member violates its byte bound")
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        validate_sha256(
            self.logical_content_sha256,
            field_name="logical_content_sha256",
        )


@dataclass(frozen=True, slots=True)
class IssuedDraftManifest(CanonicalRecord):
    """Complete frozen scientific issue root before approval or execution."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-draft-manifest'

    issue_id: str
    candidate: DraftStudyCandidate
    draft: StudyDraft
    entry_package: ExperimentEntryPackage
    raw_draft: ObjectIdentity
    canonical_draft: ObjectIdentity
    source_closure: ImplementationSourceClosure
    proposer_attestation: HumanProposerAttestation
    custody_authority: ObjectIdentity
    registry: CapabilityRegistry
    materialization_qualifications: tuple[MaterializationQualificationReceipt, ...]
    members: tuple[IssuedStudyMember, ...]
    storage_root_id: str
    publication_relative_root: str
    issued_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.publication_relative_root)
        require_sorted_unique_ids(
            self.materialization_qualifications,
            attribute="receipt_id",
            field_name="materialization_qualifications",
        )
        require_sorted_unique_ids(
            self.members,
            attribute="member_id",
            field_name="members",
        )
        if not self.members or len(self.members) > MAX_ISSUE_MEMBERS:
            raise ValueError("issued programme member count violates its bound")
        if self.canonical_draft != ObjectIdentity.from_record(
            self.draft.draft_id,
            self.draft,
        ):
            raise ValueError("issued canonical draft identity differs")
        validate_experiment_entry(self.draft, self.entry_package, for_issue=True)
        if (
            self.raw_draft.object_fingerprint
            != self.candidate.authoring_materialization.raw_materialization_sha256
        ):
            raise ValueError("issued raw draft identity differs from the candidate")
        if self.draft.fingerprint() != self.candidate.semantic_config_sha256:
            raise ValueError("issued semantic draft identity differs from the candidate")
        if self.source_closure.implementation_sha256 != self.candidate.implementation_sha256:
            raise ValueError("issued source closure differs from the candidate implementation")
        candidate_identity = ObjectIdentity.from_record(
            self.candidate.candidate_id,
            self.candidate,
        )
        if self.proposer_attestation.candidate != candidate_identity:
            raise ValueError("issued proposer attestation binds another candidate")
        if self.proposer_attestation.design_origin_id != self.candidate.design_origin.origin_id:
            raise ValueError("issued proposer attestation binds another design origin")
        if self.proposer_attestation.design_input_ids != self.candidate.design_input_ids:
            raise ValueError("issued proposer attestation binds another design-input ledger")
        if (
            self.proposer_attestation.source_qualification_receipts
            != self.candidate.source_qualification_receipts
        ):
            raise ValueError("issued proposer attestation binds other source qualifications")
        if (
            self.proposer_attestation.raw_materialization_sha256
            != self.candidate.authoring_materialization.raw_materialization_sha256
            or self.proposer_attestation.semantic_config_sha256
            != self.candidate.semantic_config_sha256
        ):
            raise ValueError("issued proposer attestation binds another config")
        qualification_identities = tuple(
            ObjectIdentity.from_record(value.receipt_id, value)
            for value in self.materialization_qualifications
        )
        if qualification_identities != self.candidate.source_qualification_receipts:
            raise ValueError("issued qualification records differ from candidate locks")
        _validate_registry_locks(self.candidate, self.registry)
        if self.publication_relative_root != ISSUE_PUBLICATION_RELATIVE_ROOT:
            raise ValueError("issued programme uses another publication namespace")
        expected_prefix = f"{self.publication_relative_root}/{self.issue_id}/"
        if any(not value.relative_path.startswith(expected_prefix) for value in self.members):
            raise ValueError("issued programme member escapes its content-addressed issue root")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("issue/freeze cannot read protected outcomes")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("issued programme must remain prospective")

    @property
    def implementation_commit(self) -> str:
        return self.source_closure.implementation_commit


@dataclass(frozen=True, slots=True)
class StudyIssuePayload:
    member: IssuedStudyMember
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("issued payload must be immutable bytes")
        if len(self.payload) != self.member.size_bytes:
            raise ValueError("issued payload size differs from its member declaration")
        if hashlib.sha256(self.payload).hexdigest() != self.member.physical_sha256:
            raise ValueError("issued payload digest differs from its member declaration")


@dataclass(frozen=True, slots=True)
class DraftIssuePreparation:
    manifest: IssuedDraftManifest
    payloads: tuple[StudyIssuePayload, ...]

    def __post_init__(self) -> None:
        if tuple(value.member for value in self.payloads) != self.manifest.members:
            raise ValueError("issue preparation payload order differs from its manifest")


@dataclass(frozen=True, slots=True)
class StudyPublicationReceipt(CanonicalRecord):
    """Verified atomic publication result for one issued programme."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-publication-receipt'

    receipt_id: str
    issue_manifest: ObjectIdentity
    manifest_logical: LogicalArtifactIdentity
    manifest_materialization: ArtifactMaterialization
    publication: ArtifactPublicationBinding
    declared_member_ids: tuple[str, ...]
    total_payload_bytes: int
    published_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.declared_member_ids,
            field_name="declared_member_ids",
            allow_empty=False,
        )
        if self.total_payload_bytes <= 0:
            raise ValueError("issued publication receipt requires positive payload bytes")
        parse_utc_timestamp(self.published_at_utc, field_name="published_at_utc")
        if (
            self.manifest_logical.logical_artifact_id
            != self.manifest_materialization.logical_artifact_id
        ):
            raise ValueError("issued manifest logical and physical identities differ")
        if (
            self.manifest_materialization.storage_root_id
            != self.publication.publication_scope.storage_root_id
        ):
            raise ValueError("issued manifest lies on another storage root")
        publication_member_ids = tuple(
            sorted(value.logical_artifact_id for value in self.publication.members)
        )
        expected_ids = tuple(
            sorted(
                (
                    *self.declared_member_ids,
                    self.manifest_logical.logical_artifact_id,
                )
            )
        )
        if publication_member_ids != expected_ids:
            raise ValueError("issued publication batch differs from declared members")


@dataclass(frozen=True, slots=True)
class ExtensionPublicationReceipt(CanonicalRecord):
    """Composite custody for an immutable base issue plus one extension batch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/extension-publication-receipt'

    receipt_id: str
    issue_manifest: ObjectIdentity
    base_publication_receipt: ObjectIdentity
    manifest_logical: LogicalArtifactIdentity
    manifest_materialization: ArtifactMaterialization
    extension_publication: ArtifactPublicationBinding
    declared_base_member_ids: tuple[str, ...]
    declared_extension_member_ids: tuple[str, ...]
    extension_payload_bytes: int
    total_payload_bytes: int
    published_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.issue_manifest.object_schema != (
            RetrospectiveIssuedStudy.SCHEMA
            if isinstance(self, RetrospectivePublicationReceipt)
            else IssuedExecutableStudyManifest.SCHEMA
        ):
            raise ValueError("extension publication receipt binds another manifest schema")
        if self.base_publication_receipt.object_schema != (
            StudyPublicationReceipt.SCHEMA
        ):
            raise ValueError("extension publication receipt binds another base receipt schema")
        require_sorted_unique_strings(
            self.declared_base_member_ids,
            field_name="declared_base_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.declared_extension_member_ids,
            field_name="declared_extension_member_ids",
            allow_empty=False,
        )
        if set(self.declared_base_member_ids) & set(self.declared_extension_member_ids):
            raise ValueError("base and extension publication members overlap")
        if self.extension_payload_bytes <= 0 or self.total_payload_bytes < (
            self.extension_payload_bytes
        ):
            raise ValueError("extension publication receipt requires positive payload bytes")
        parse_utc_timestamp(self.published_at_utc, field_name="published_at_utc")
        if (
            self.manifest_logical.logical_artifact_id
            != self.manifest_materialization.logical_artifact_id
            or self.manifest_materialization.storage_root_id
            != self.extension_publication.publication_scope.storage_root_id
        ):
            raise ValueError("issued executable study manifest logical/physical publication differs")
        publication_member_ids = tuple(
            sorted(value.logical_artifact_id for value in self.extension_publication.members)
        )
        expected_ids = tuple(
            sorted(
                (
                    *self.declared_extension_member_ids,
                    self.manifest_logical.logical_artifact_id,
                )
            )
        )
        if publication_member_ids != expected_ids:
            raise ValueError("extension publication batch differs from declared members")

    @property
    def declared_member_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self.declared_base_member_ids, *self.declared_extension_member_ids)))


@dataclass(frozen=True, slots=True)
class PublishedDraftStudy:
    """Verified external manifest/receipt pair returned by trusted infrastructure."""

    manifest: IssuedDraftManifest
    publication_receipt: StudyPublicationReceipt

    def __post_init__(self) -> None:
        if self.publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.issue_id,
            self.manifest,
        ):
            raise ValueError("published issue receipt binds another manifest")


@dataclass(frozen=True, slots=True)
class IssuedStudyManifest(CanonicalRecord):
    """Additive issue root for the mandatory draft-plus-entry authoring package."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-study-manifest'

    issue_id: str
    candidate: StudyCandidate
    authoring_package: StudyDefinition
    raw_authoring_package: ObjectIdentity
    canonical_authoring_package: ObjectIdentity
    source_closure: ImplementationSourceClosure
    proposer_attestation: HumanProposerAttestation
    custody_authority: ObjectIdentity
    registry: CapabilityRegistry
    materialization_qualifications: tuple[MaterializationQualificationReceipt, ...]
    members: tuple[IssuedStudyMember, ...]
    storage_root_id: str
    publication_relative_root: str
    issued_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveIssuedBase):
            if type(self.proposer_attestation) is not HumanProposerAttestation:
                raise ValueError("prospective issue requires its original attestation")
            if type(self.candidate) is not StudyCandidate:
                raise ValueError(
                    "StandardIssuedProgrammeManifest requires its original candidate schema"
                )
            if type(self.authoring_package) is not StudyDefinition:
                raise ValueError(
                    "StandardIssuedProgrammeManifest requires its original authoring_package schema"
                )
        validate_stable_id(self.issue_id, field_name="issue_id")
        require_sorted_unique_ids(
            self.members,
            attribute="member_id",
            field_name="members",
        )
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.publication_relative_root)
        require_sorted_unique_ids(
            self.materialization_qualifications,
            attribute="receipt_id",
            field_name="materialization_qualifications",
        )
        require_sorted_unique_ids(
            self.members,
            attribute="member_id",
            field_name="members",
        )
        if not self.members or len(self.members) > MAX_ISSUE_MEMBERS:
            raise ValueError("standard issued programme member count violates its bound")
        if self.canonical_authoring_package != ObjectIdentity.from_record(
            self.authoring_package.package_id,
            self.authoring_package,
        ):
            raise ValueError("issued canonical authoring-package identity differs")
        if self.candidate.authoring_package != self.canonical_authoring_package:
            raise ValueError("issued authoring package differs from the standard candidate")
        validate_experiment_entry(
            self.authoring_package.draft,
            self.authoring_package.entry_package,
            for_issue=True,
        )
        base = self.candidate.base_candidate
        if (
            self.raw_authoring_package.object_fingerprint
            != base.authoring_materialization.raw_materialization_sha256
        ):
            raise ValueError("issued raw authoring identity differs from the candidate")
        if self.authoring_package.draft.fingerprint() != base.semantic_config_sha256:
            raise ValueError("issued draft identity differs from the base candidate")
        if self.source_closure.implementation_sha256 != base.implementation_sha256:
            raise ValueError("issued source closure differs from the candidate implementation")
        candidate_identity = ObjectIdentity.from_record(
            self.candidate.candidate_id,
            self.candidate,
        )
        if self.proposer_attestation.candidate != candidate_identity:
            raise ValueError("issued proposer attestation binds another standard candidate")
        if self.proposer_attestation.design_origin_id != base.design_origin.origin_id:
            raise ValueError("issued proposer attestation binds another design origin")
        if self.proposer_attestation.design_input_ids != base.design_input_ids:
            raise ValueError("issued proposer attestation binds another design-input ledger")
        if (
            self.proposer_attestation.source_qualification_receipts
            != base.source_qualification_receipts
        ):
            raise ValueError("issued proposer attestation binds other source qualifications")
        if (
            self.proposer_attestation.raw_materialization_sha256
            != base.authoring_materialization.raw_materialization_sha256
            or self.proposer_attestation.semantic_config_sha256 != base.semantic_config_sha256
        ):
            raise ValueError("issued proposer attestation binds another authoring package")
        qualification_identities = tuple(
            ObjectIdentity.from_record(value.receipt_id, value)
            for value in self.materialization_qualifications
        )
        if qualification_identities != base.source_qualification_receipts:
            raise ValueError("issued qualification records differ from candidate locks")
        _validate_registry_locks(base, self.registry)
        if self.publication_relative_root != ISSUE_PUBLICATION_RELATIVE_ROOT:
            raise ValueError("standard issued programme uses another publication namespace")
        expected_prefix = f"{self.publication_relative_root}/{self.issue_id}/"
        if any(not value.relative_path.startswith(expected_prefix) for value in self.members):
            raise ValueError("standard issue member escapes its content-addressed root")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("standard issue cannot read protected outcomes")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("standard issue must remain prospective")

    @property
    def implementation_commit(self) -> str:
        return self.source_closure.implementation_commit


@dataclass(frozen=True, slots=True)
class StudyExtensionDecoderRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-extension-decoder-registration'

    registration_id: str
    decoder_key: str
    decoder_version: str
    payload_schema: str
    payload_version: str
    config_sha256: str
    implementation_sha256: str
    maximum_payload_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("registration_id", self.registration_id),
            ("decoder_key", self.decoder_key),
        ):
            validate_stable_id(value, field_name=name)
        from empirical_lawhood.kernel.serialization import validate_semantic_version

        validate_semantic_version(self.decoder_version)
        validate_semantic_version(self.payload_version)
        validate_schema(self.payload_schema)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        if not 0 < self.maximum_payload_bytes <= MAX_ISSUE_PAYLOAD_BYTES:
            raise ValueError("extension decoder byte bound is invalid")


@dataclass(frozen=True, slots=True)
class IssuedExtensionBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-extension-binding'

    extension_id: str
    proposed_extension: ObjectIdentity
    issued_member: ObjectIdentity
    decoder_registration: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_id, field_name="extension_id")
        if self.proposed_extension.object_schema != ProposedStudyExtension.SCHEMA:
            raise ValueError("issued extension binds another proposal schema")
        if self.issued_member.object_schema != IssuedStudyMember.SCHEMA:
            raise ValueError("issued extension binds another member schema")
        if self.decoder_registration.object_schema != StudyExtensionDecoderRegistration.SCHEMA:
            raise ValueError("issued extension binds another decoder schema")


@dataclass(frozen=True, slots=True)
class IssuedExtensionSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-extension-set'

    issued_extension_set_id: str
    proposed_extension_set: ObjectIdentity
    members: tuple[IssuedStudyMember, ...]
    bindings: tuple[IssuedExtensionBinding, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(
            self.issued_extension_set_id,
            field_name="issued_extension_set_id",
        )
        if self.proposed_extension_set.object_schema != ProposedStudyExtensionSet.SCHEMA:
            raise ValueError("issued extension set binds another proposed-set schema")
        require_sorted_unique_ids(
            self.members,
            attribute="member_id",
            field_name="members",
        )
        require_sorted_unique_ids(
            self.bindings,
            attribute="extension_id",
            field_name="bindings",
        )
        if not self.bindings:
            raise ValueError("issued extension set cannot be empty")
        member_identities = {
            ObjectIdentity.from_record(value.member_id, value) for value in self.members
        }
        if {value.issued_member for value in self.bindings} != member_identities:
            raise ValueError("issued extension bindings differ from their member roster")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("issued extensions must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("issued extensions must remain prospective")


@dataclass(frozen=True, slots=True)
class StudyExtensionValidationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-extension-validation-receipt'

    receipt_id: str
    issued_extension_set: ObjectIdentity
    validated_extension_ids: tuple[str, ...]
    complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.validated_extension_ids,
            field_name="validated_extension_ids",
            allow_empty=False,
        )
        if not self.complete:
            raise ValueError("issued extension validation receipt must be complete")


@dataclass(frozen=True, slots=True)
class IssuedExecutableStudyManifest(CanonicalRecord):
    "Issued extension wrapper retaining the complete standard manifest."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-executable-study-manifest'

    issue_id: str
    base: IssuedStudyManifest
    candidate: ExecutableStudyCandidate
    issued_extensions: IssuedExtensionSet
    extension_validation: StudyExtensionValidationReceipt
    extension_proposer_attestation: HumanProposerAttestation
    extension_custody_authority: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveIssuedStudy):
            if type(self.extension_proposer_attestation) is not HumanProposerAttestation:
                raise ValueError("prospective issue requires its original attestation")
            if type(self.base) is not IssuedStudyManifest:
                raise ValueError(
                    "IssuedExecutableStudyManifest requires its original base schema"
                )
            if type(self.candidate) is not ExecutableStudyCandidate:
                raise ValueError(
                    "IssuedExecutableStudyManifest requires its original candidate schema"
                )
        validate_stable_id(self.issue_id, field_name="issue_id")
        require_sorted_unique_ids(
            self.members,
            attribute="member_id",
            field_name="members",
        )
        if len(self.members) > MAX_ISSUE_MEMBERS:
            raise ValueError("issued executable study manifest exceeds its member bound")
        if self.candidate.base_candidate != self.base.candidate:
            raise ValueError("issued executable study manifest changes its base candidate")
        if self.issued_extensions.proposed_extension_set != (self.candidate.proposed_extension_set):
            raise ValueError("issued executable study manifest binds another proposed extension set")
        if self.extension_validation.issued_extension_set != ObjectIdentity.from_record(
            self.issued_extensions.issued_extension_set_id,
            self.issued_extensions,
        ):
            raise ValueError("issued executable study manifest lacks exact extension validation")
        if self.extension_custody_authority.object_schema != StudyOperationAuthority.SCHEMA:
            raise ValueError("issued executable study manifest lacks typed extension custody authority")
        candidate_identity = ObjectIdentity.from_record(
            self.candidate.candidate_id,
            self.candidate,
        )
        base_candidate = self.candidate.base_candidate.base_candidate
        if (
            self.extension_proposer_attestation.candidate != candidate_identity
            or self.extension_proposer_attestation.design_origin_id
            != base_candidate.design_origin.origin_id
            or self.extension_proposer_attestation.design_input_ids
            != base_candidate.design_input_ids
            or self.extension_proposer_attestation.source_qualification_receipts
            != base_candidate.source_qualification_receipts
            or self.extension_proposer_attestation.semantic_config_sha256
            != base_candidate.semantic_config_sha256
        ):
            raise ValueError("issued executable study manifest proposer attestation binds another candidate")

    @property
    def members(self) -> tuple[IssuedStudyMember, ...]:
        return tuple(
            sorted(
                (*self.base.members, *self.issued_extensions.members),
                key=lambda value: value.member_id,
            )
        )

    @property
    def proposer_attestation(self) -> HumanProposerAttestation:
        return self.extension_proposer_attestation

    @property
    def materialization_qualifications(
        self,
    ) -> tuple[MaterializationQualificationReceipt, ...]:
        return self.base.materialization_qualifications

    @property
    def custody_authority(self) -> ObjectIdentity:
        return self.base.custody_authority

    @property
    def registry(self) -> CapabilityRegistry:
        return self.base.registry

    @property
    def source_closure(self) -> ImplementationSourceClosure:
        return self.base.source_closure

    @property
    def implementation_commit(self) -> str:
        return self.base.implementation_commit


def bind_standard_issued_extensions(
    *,
    base: IssuedStudyManifest,
    candidate: ExecutableStudyCandidate,
    issued_extensions: IssuedExtensionSet,
    extension_validation: StudyExtensionValidationReceipt,
    extension_proposer_attestation: HumanProposerAttestation,
    extension_custody_authority: StudyOperationAuthority,
) -> IssuedExecutableStudyManifest:
    record_type_issued_executable_study_manifest: type[IssuedExecutableStudyManifest] = (
        RetrospectiveIssuedStudy
        if isinstance(candidate, RetrospectiveExtensionCandidate)
        else IssuedExecutableStudyManifest
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "base": ObjectIdentity.from_record(base.issue_id, base),
                "candidate": ObjectIdentity.from_record(
                    candidate.candidate_id,
                    candidate,
                ),
                "issued_extensions": ObjectIdentity.from_record(
                    issued_extensions.issued_extension_set_id,
                    issued_extensions,
                ),
                "extension_validation": ObjectIdentity.from_record(
                    extension_validation.receipt_id,
                    extension_validation,
                ),
                "extension_custody_authority": ObjectIdentity.from_record(
                    extension_custody_authority.authority_id,
                    extension_custody_authority,
                ),
                "extension_proposer_attestation": ObjectIdentity.from_record(
                    extension_proposer_attestation.attestation_id,
                    extension_proposer_attestation,
                ),
            }
        )
    ).hexdigest()
    return record_type_issued_executable_study_manifest(
        issue_id=f"issued-executable-study.{digest[:32]}",
        base=base,
        candidate=candidate,
        issued_extensions=issued_extensions,
        extension_validation=extension_validation,
        extension_proposer_attestation=extension_proposer_attestation,
        extension_custody_authority=ObjectIdentity.from_record(
            extension_custody_authority.authority_id,
            extension_custody_authority,
        ),
    )


def validate_issued_study_extensions(
    *,
    proposed: ProposedStudyExtensionSet,
    issued_members: tuple[IssuedStudyMember, ...],
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...],
) -> tuple[IssuedExtensionSet, StudyExtensionValidationReceipt]:
    """Reject incomplete, duplicate, unexpected or decoder-conflicting extensions."""

    require_sorted_unique_ids(
        issued_members,
        attribute="member_id",
        field_name="issued_members",
    )
    require_sorted_unique_ids(
        decoder_registrations,
        attribute="registration_id",
        field_name="decoder_registrations",
    )
    if len(issued_members) != len(proposed.extensions):
        raise ValueError("issued extension member roster differs from proposal")
    registrations = {
        (value.decoder_key, value.decoder_version): value for value in decoder_registrations
    }
    if len(registrations) != len(decoder_registrations):
        raise ValueError("extension decoder key/version is duplicated")
    bindings: list[IssuedExtensionBinding] = []
    used_member_ids: set[str] = set()
    used_registration_ids: set[str] = set()
    for extension in proposed.extensions:
        matching_members = tuple(
            member
            for member in issued_members
            if member.payload_schema == extension.payload.object_schema
            and member.logical_content_sha256 == extension.payload.object_fingerprint
            and member.size_bytes == extension.payload_size_bytes
        )
        if len(matching_members) != 1:
            raise ValueError("issued extension member differs from proposal")
        member = matching_members[0]
        if member.member_id in used_member_ids:
            raise ValueError("one issued member satisfies several extension proposals")
        used_member_ids.add(member.member_id)
        registration = registrations.get((extension.decoder_key, extension.decoder_version))
        if (
            registration is None
            or member.payload_schema != extension.payload.object_schema
            or member.logical_content_sha256 != extension.payload.object_fingerprint
            or member.size_bytes != extension.payload_size_bytes
            or registration.payload_schema != extension.payload.object_schema
            or registration.payload_version != extension.payload.object_version
            or registration.config_sha256 != extension.decoder_config_sha256
            or member.size_bytes > registration.maximum_payload_bytes
        ):
            raise ValueError("issued extension member/decoder differs from proposal")
        used_registration_ids.add(registration.registration_id)
        bindings.append(
            IssuedExtensionBinding(
                extension_id=extension.extension_id,
                proposed_extension=ObjectIdentity.from_record(
                    extension.extension_id,
                    extension,
                ),
                issued_member=ObjectIdentity.from_record(member.member_id, member),
                decoder_registration=ObjectIdentity.from_record(
                    registration.registration_id,
                    registration,
                ),
            )
        )
    if used_member_ids != {value.member_id for value in issued_members} or (
        used_registration_ids != {value.registration_id for value in decoder_registrations}
    ):
        raise ValueError("issued extension member/decoder roster has unexpected entries")
    issued = IssuedExtensionSet(
        issued_extension_set_id=f"issued-extensions.{proposed.extension_set_id}",
        proposed_extension_set=ObjectIdentity.from_record(
            proposed.extension_set_id,
            proposed,
        ),
        members=issued_members,
        bindings=tuple(sorted(bindings, key=lambda value: value.extension_id)),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    receipt = StudyExtensionValidationReceipt(
        receipt_id=f"extension-validation.{issued.issued_extension_set_id}",
        issued_extension_set=ObjectIdentity.from_record(
            issued.issued_extension_set_id,
            issued,
        ),
        validated_extension_ids=tuple(value.extension_id for value in issued.bindings),
        complete=True,
    )
    return issued, receipt


def decode_issued_extension_payload(
    *,
    payload: bytes,
    proposed: ProposedStudyExtension,
    registration: StudyExtensionDecoderRegistration,
    record_type: type[_ExtensionRecordT],
) -> _ExtensionRecordT:
    if (
        record_type.SCHEMA != proposed.payload.object_schema
        or registration.payload_schema != record_type.SCHEMA
        or registration.payload_version != proposed.payload.object_version
        or registration.decoder_key != proposed.decoder_key
        or registration.decoder_version != proposed.decoder_version
        or registration.config_sha256 != proposed.decoder_config_sha256
        or len(payload) != proposed.payload_size_bytes
        or len(payload) > registration.maximum_payload_bytes
        or hashlib.sha256(payload).hexdigest() != proposed.payload.object_fingerprint
    ):
        raise ValueError("extension payload differs from its issued decoder binding")
    return decode_canonical_bytes(
        payload,
        record_type,
        maximum_bytes=registration.maximum_payload_bytes,
    )


@dataclass(frozen=True, slots=True)
class StudyIssuePreparation:
    """Write-free standard issue bundle with exact member payloads."""

    manifest: IssuedStudyManifest
    payloads: tuple[StudyIssuePayload, ...]

    def __post_init__(self) -> None:
        if tuple(value.member for value in self.payloads) != self.manifest.members:
            raise ValueError("standard issue payload order differs from its manifest")


@dataclass(frozen=True, slots=True)
class ExecutableStudyIssuePreparation:
    "Write-free extension issue over an already published immutable base."

    manifest: IssuedExecutableStudyManifest
    base_publication_receipt: StudyPublicationReceipt
    extension_custody_authority: StudyOperationAuthority
    extension_payloads: tuple[StudyIssuePayload, ...]

    def __post_init__(self) -> None:
        if self.base_publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.base.issue_id,
            self.manifest.base,
        ):
            raise ValueError("extension preparation binds another base publication")
        if (
            ObjectIdentity.from_record(
                self.extension_custody_authority.authority_id,
                self.extension_custody_authority,
            )
            != self.manifest.extension_custody_authority
        ):
            raise ValueError("extension preparation changes its custody authority")
        if tuple(value.member for value in self.extension_payloads) != (
            self.manifest.issued_extensions.members
        ):
            raise ValueError("extension payload order differs from its issued extension set")


@dataclass(frozen=True, slots=True)
class PublishedExecutableStudy:
    """Verified base plus additive extension publication."""

    manifest: IssuedExecutableStudyManifest
    publication_receipt: ExtensionPublicationReceipt

    def __post_init__(self) -> None:
        if self.publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.issue_id,
            self.manifest,
        ):
            raise ValueError("published executable study receipt binds another manifest")


def project_standard_study_extensions(
    *,
    candidate: ExecutableStudyCandidate,
    extension_payload_bytes: tuple[bytes, ...],
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...],
    extension_materializations: tuple[StudyExtensionMaterializationReceipt, ...],
) -> tuple[
    IssuedExtensionSet,
    StudyExtensionValidationReceipt,
    tuple[StudyIssuePayload, ...],
]:
    """Derive future extension members without publication, authority, or evidence status."""

    proposals = candidate.authoring_package.extension_set.extensions
    if not (
        len(proposals)
        == len(extension_payload_bytes)
        == len(decoder_registrations)
        == len(extension_materializations)
    ):
        raise ValueError("extension issue inputs differ from the proposed roster")
    payloads: list[StudyIssuePayload] = []
    for proposal, payload, registration, materialization in zip(
        proposals,
        extension_payload_bytes,
        decoder_registrations,
        extension_materializations,
        strict=True,
    ):
        if (
            materialization.extension_id != proposal.extension_id
            or materialization.proposed_extension
            != ObjectIdentity.from_record(proposal.extension_id, proposal)
            or materialization.payload != proposal.payload
            or materialization.byte_count != len(payload)
            or materialization.physical_sha256 != hashlib.sha256(payload).hexdigest()
            or materialization.decoder_registration
            != ObjectIdentity.from_record(registration.registration_id, registration)
        ):
            raise ValueError("extension issue bytes/decoder differ from compilation custody")
        member = _member(
            issue_id=candidate.candidate_id,
            member_label=f"extension.{proposal.extension_id}",
            relative_path=(
                f"{ISSUE_PUBLICATION_RELATIVE_ROOT}/extensions/{candidate.candidate_id}/"
                f"{proposal.extension_id}.json"
            ),
            payload_schema=proposal.payload.object_schema,
            media_type="application/json",
            profile=ArtifactProfile.CANONICAL_JSON,
            payload=payload,
            logical_content_sha256=proposal.payload.object_fingerprint,
        )
        payloads.append(StudyIssuePayload(member=member, payload=payload))
    issued, validation = validate_issued_study_extensions(
        proposed=candidate.authoring_package.extension_set,
        issued_members=tuple(
            sorted((value.member for value in payloads), key=lambda x: x.member_id)
        ),
        decoder_registrations=tuple(
            sorted(decoder_registrations, key=lambda value: value.registration_id)
        ),
    )
    payload_by_id = {value.member.member_id: value for value in payloads}
    return issued, validation, tuple(payload_by_id[value.member_id] for value in issued.members)


def prepare_executable_study_issue(
    *,
    base_manifest: IssuedStudyManifest,
    base_publication_receipt: StudyPublicationReceipt,
    current_compilation: ExecutableStudyCompilationReport,
    candidate: ExecutableStudyCandidate,
    extension_payload_bytes: tuple[bytes, ...],
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...],
    extension_materializations: tuple[StudyExtensionMaterializationReceipt, ...],
    extension_proposer_attestation: HumanProposerAttestation,
    extension_custody_authority: StudyOperationAuthority,
    storage_root_id: str,
    grantee_id: str,
    at_utc: str,
) -> ExecutableStudyIssuePreparation:
    """Prepare exact extension members without republishing immutable base bytes."""

    if (
        current_compilation.candidate != candidate
        or current_compilation.base_report.candidate != base_manifest.candidate
        or candidate.base_candidate != base_manifest.candidate
        or current_compilation.extension_materializations != extension_materializations
        or extension_proposer_attestation.raw_materialization_sha256
        != current_compilation.authoring_materialization.raw_materialization_sha256
    ):
        raise ValueError("extension issue differs from its current compilation")
    require_study_authority(
        extension_custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        prerequisite_authority=None,
        grantee_id=grantee_id,
        storage_root_id=storage_root_id,
        relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        at_utc=at_utc,
    )
    issued, validation, payloads = project_standard_study_extensions(
        candidate=candidate,
        extension_payload_bytes=extension_payload_bytes,
        decoder_registrations=decoder_registrations,
        extension_materializations=extension_materializations,
    )
    manifest = bind_standard_issued_extensions(
        base=base_manifest,
        candidate=candidate,
        issued_extensions=issued,
        extension_validation=validation,
        extension_proposer_attestation=extension_proposer_attestation,
        extension_custody_authority=extension_custody_authority,
    )
    return ExecutableStudyIssuePreparation(
        manifest=manifest,
        base_publication_receipt=base_publication_receipt,
        extension_custody_authority=extension_custody_authority,
        extension_payloads=payloads,
    )


@dataclass(frozen=True, slots=True)
class PublishedStudy:
    """Verified external standard manifest/receipt pair."""

    manifest: IssuedStudyManifest
    publication_receipt: StudyPublicationReceipt

    def __post_init__(self) -> None:
        if self.publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.issue_id,
            self.manifest,
        ):
            raise ValueError("published standard issue receipt binds another manifest")


def prepare_draft_candidate_issue(
    *,
    draft: StudyDraft,
    raw_draft_bytes: bytes,
    current_compilation: CandidateCompilationReport,
    candidate: DraftStudyCandidate,
    registry: CapabilityRegistry,
    materialization_qualifications: tuple[MaterializationQualificationReceipt, ...],
    source_closure: ImplementationSourceClosure,
    observed_source_closure: ImplementationSourceClosure,
    proposer_attestation: HumanProposerAttestation,
    entry_package: ExperimentEntryPackage,
    custody_authority: StudyOperationAuthority,
    storage_root_id: str,
    grantee_id: str,
    at_utc: str,
) -> DraftIssuePreparation:
    """Revalidate exact candidate inputs and prepare a write-free issue bundle."""

    if not isinstance(raw_draft_bytes, bytes) or not raw_draft_bytes:
        raise ValueError("issue requires the exact nonempty draft bytes")
    if len(raw_draft_bytes) > MAX_ISSUE_PAYLOAD_BYTES:
        raise ValueError("raw programme draft exceeds the issue byte bound")
    if source_closure != observed_source_closure:
        raise ValueError("implementation source closure is stale")
    validate_experiment_entry(draft, entry_package, for_issue=True)
    if (
        current_compilation.disposition
        is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        or current_compilation.candidate != candidate
    ):
        raise ValueError("issue requires an exact current READY_TO_ISSUE compilation")
    if (
        current_compilation.design_validity_state,
        current_compilation.source_readiness_state,
        current_compilation.freeze_readiness_state,
        current_compilation.authority_state,
    ) != (
        CandidateAxisState.READY,
        CandidateAxisState.READY,
        CandidateAxisState.READY,
        CandidateAxisState.EXPECTED_AUTHORITY_GATE,
    ):
        raise ValueError("candidate readiness axes changed before issue")
    raw_materialization_sha256 = hashlib.sha256(
        candidate.authoring_materialization.media_type.encode("ascii") + b"\x00" + raw_draft_bytes
    ).hexdigest()
    if (
        len(raw_draft_bytes) != candidate.authoring_materialization.byte_count
        or raw_materialization_sha256
        != candidate.authoring_materialization.raw_materialization_sha256
    ):
        raise ValueError("exact authoring materialization changed before issue")
    if draft.fingerprint() != candidate.semantic_config_sha256:
        raise ValueError("semantic config changed before issue")
    if (
        draft.system != candidate.system
        or draft.experiment != candidate.experiment
        or draft.campaign != candidate.campaign
        or draft.design_origin != candidate.design_origin
        or tuple(value.input_id for value in draft.design_inputs) != candidate.design_input_ids
        or draft.capability_selections != candidate.capability_locks
        or draft.source_materializations != candidate.source_locks
        or draft.resource_ceiling != candidate.resource_ceiling
    ):
        raise ValueError("draft scientific components changed before issue")
    if any(
        value.access_disposition is not SourceAccessDisposition.VERIFIED_ACCESS
        for value in candidate.source_locks
    ):
        raise ValueError("issued source lock is no longer verified")
    _validate_registry_locks(candidate, registry)
    candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=candidate_identity,
        prerequisite_authority=None,
        grantee_id=grantee_id,
        storage_root_id=storage_root_id,
        relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        at_utc=at_utc,
    )
    if proposer_attestation.candidate != candidate_identity:
        raise ValueError("proposer attestation binds another candidate")
    if source_closure.implementation_sha256 != candidate.implementation_sha256:
        raise ValueError("source closure implementation differs from the candidate")

    qualification_identities = tuple(
        ObjectIdentity.from_record(value.receipt_id, value)
        for value in materialization_qualifications
    )
    if qualification_identities != candidate.source_qualification_receipts:
        raise ValueError("materialization qualification records are missing or stale")

    issue_seed = {
        "candidate": candidate_identity,
        "raw_draft_sha256": raw_materialization_sha256,
        "semantic_config_sha256": candidate.semantic_config_sha256,
        "source_closure": ObjectIdentity.from_record(
            source_closure.source_closure_id,
            source_closure,
        ),
        "proposer_attestation": ObjectIdentity.from_record(
            proposer_attestation.attestation_id,
            proposer_attestation,
        ),
        "entry_package": ObjectIdentity.from_record(
            entry_package.package_id,
            entry_package,
        ),
        "custody_authority": ObjectIdentity.from_record(
            custody_authority.authority_id,
            custody_authority,
        ),
        "registry_sha256": registry.fingerprint(),
        "qualifications": qualification_identities,
        "storage_root_id": storage_root_id,
    }
    issue_digest = hashlib.sha256(canonical_json_bytes(issue_seed)).hexdigest()
    issue_id = f"issue.{issue_digest[:32]}"
    prefix = f"{ISSUE_PUBLICATION_RELATIVE_ROOT}/{issue_id}"

    raw_identity = ObjectIdentity(
        object_id=f"raw-draft.{issue_id}",
        object_schema='empirical-lawhood/planning/study-draft-raw-input',
        object_version="1.0.0",
        object_fingerprint=raw_materialization_sha256,
    )
    canonical_identity = ObjectIdentity.from_record(draft.draft_id, draft)
    payload_specs: list[tuple[str, str, str, ArtifactProfile, bytes, str]] = [
        (
            "draft-source",
            f"{prefix}/draft-source",
            StudyDraft.SCHEMA,
            ArtifactProfile.TEXT_PARAMETERS,
            raw_draft_bytes,
            hashlib.sha256(raw_draft_bytes).hexdigest(),
        ),
        (
            "draft-canonical",
            f"{prefix}/draft-canonical.json",
            StudyDraft.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            draft.canonical_bytes(),
            draft.fingerprint(),
        ),
        (
            "candidate",
            f"{prefix}/candidate.json",
            DraftStudyCandidate.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            candidate.canonical_bytes(),
            candidate.fingerprint(),
        ),
        (
            "experiment-entry-package",
            f"{prefix}/experiment-entry-package.json",
            ExperimentEntryPackage.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            entry_package.canonical_bytes(),
            entry_package.fingerprint(),
        ),
        (
            "registry",
            f"{prefix}/capability-registry.json",
            CapabilityRegistry.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            registry.canonical_bytes(),
            registry.fingerprint(),
        ),
        (
            "source-closure",
            f"{prefix}/source-closure.json",
            ImplementationSourceClosure.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            source_closure.canonical_bytes(),
            source_closure.fingerprint(),
        ),
        (
            "proposer-attestation",
            f"{prefix}/proposer-attestation.json",
            HumanProposerAttestation.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            proposer_attestation.canonical_bytes(),
            proposer_attestation.fingerprint(),
        ),
        (
            "custody-authority",
            f"{prefix}/custody-authority.json",
            StudyOperationAuthority.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            custody_authority.canonical_bytes(),
            custody_authority.fingerprint(),
        ),
    ]
    component_records: tuple[CanonicalRecord, ...] = (
        candidate.system,
        candidate.experiment,
        candidate.campaign,
        candidate.protocol,
        candidate.scientific_graph,
        candidate.obligation_coverage,
        *materialization_qualifications,
        *((candidate.conditional_successor,) if candidate.conditional_successor else ()),
    )
    for index, record in enumerate(component_records):
        payload_specs.append(
            (
                f"component-{index:03d}",
                f"{prefix}/components/{index:03d}-{record.SCHEMA.rsplit('/', 2)[-1]}.json",
                record.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                record.canonical_bytes(),
                record.fingerprint(),
            )
        )
    members = tuple(
        sorted(
            (
                _member(
                    issue_id=issue_id,
                    member_label=label,
                    relative_path=path,
                    payload_schema=schema,
                    profile=profile,
                    payload=payload,
                    logical_content_sha256=logical_sha256,
                    media_type=(
                        candidate.authoring_materialization.media_type
                        if label == "draft-source"
                        else "application/json"
                    ),
                )
                for label, path, schema, profile, payload, logical_sha256 in payload_specs
            ),
            key=lambda value: value.member_id,
        )
    )
    payload_by_member_id = {
        f"{issue_id}.{label}": payload
        for label, _path, _schema, _profile, payload, _logical_sha256 in payload_specs
    }
    manifest = IssuedDraftManifest(
        issue_id=issue_id,
        candidate=candidate,
        draft=draft,
        entry_package=entry_package,
        raw_draft=raw_identity,
        canonical_draft=canonical_identity,
        source_closure=source_closure,
        proposer_attestation=proposer_attestation,
        custody_authority=ObjectIdentity.from_record(
            custody_authority.authority_id,
            custody_authority,
        ),
        registry=registry,
        materialization_qualifications=materialization_qualifications,
        members=members,
        storage_root_id=storage_root_id,
        publication_relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        issued_at_utc=custody_authority.issued_at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return DraftIssuePreparation(
        manifest=manifest,
        payloads=tuple(
            StudyIssuePayload(
                member=member,
                payload=payload_by_member_id[member.member_id],
            )
            for member in members
        ),
    )


def prepare_study_issue(
    *,
    authoring_package: StudyDefinition,
    raw_authoring_package_bytes: bytes,
    current_compilation: StudyCompilationReport,
    candidate: StudyCandidate,
    registry: CapabilityRegistry,
    materialization_qualifications: tuple[MaterializationQualificationReceipt, ...],
    source_closure: ImplementationSourceClosure,
    observed_source_closure: ImplementationSourceClosure,
    proposer_attestation: HumanProposerAttestation,
    custody_authority: StudyOperationAuthority,
    storage_root_id: str,
    grantee_id: str,
    at_utc: str,
) -> StudyIssuePreparation:
    """Recompile-bound issue preparation for one mandatory standard authoring root."""

    record_type_standardissuedprogrammemanifest: type[IssuedStudyManifest] = (
        RetrospectiveIssuedBase
        if isinstance(authoring_package, RetrospectiveAuthoringBase)
        else IssuedStudyManifest
    )
    if not isinstance(raw_authoring_package_bytes, bytes) or not raw_authoring_package_bytes:
        raise ValueError("standard issue requires exact nonempty authoring-package bytes")
    if len(raw_authoring_package_bytes) > MAX_ISSUE_PAYLOAD_BYTES:
        raise ValueError("raw authoring package exceeds the issue byte bound")
    if source_closure != observed_source_closure:
        raise ValueError("implementation source closure is stale")
    validate_experiment_entry(
        authoring_package.draft,
        authoring_package.entry_package,
        for_issue=True,
    )
    if (
        current_compilation.disposition
        is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        or current_compilation.candidate != candidate
    ):
        raise ValueError("standard issue requires the exact current READY_TO_ISSUE candidate")
    base_report = current_compilation.base_report
    base = candidate.base_candidate
    if (
        base_report.design_validity_state,
        base_report.source_readiness_state,
        base_report.freeze_readiness_state,
        base_report.authority_state,
    ) != (
        CandidateAxisState.READY,
        CandidateAxisState.READY,
        CandidateAxisState.READY,
        CandidateAxisState.EXPECTED_AUTHORITY_GATE,
    ):
        raise ValueError("standard candidate readiness axes changed before issue")
    raw_materialization_sha256 = hashlib.sha256(
        base.authoring_materialization.media_type.encode("ascii")
        + b"\x00"
        + raw_authoring_package_bytes
    ).hexdigest()
    if (
        len(raw_authoring_package_bytes) != base.authoring_materialization.byte_count
        or raw_materialization_sha256 != base.authoring_materialization.raw_materialization_sha256
    ):
        raise ValueError("exact standard authoring materialization changed before issue")
    if (
        ObjectIdentity.from_record(
            authoring_package.package_id,
            authoring_package,
        )
        != candidate.authoring_package
    ):
        raise ValueError("standard authoring package changed before issue")
    draft = authoring_package.draft
    if (
        draft.fingerprint() != base.semantic_config_sha256
        or draft.system != base.system
        or draft.experiment != base.experiment
        or draft.campaign != base.campaign
        or draft.design_origin != base.design_origin
        or tuple(value.input_id for value in draft.design_inputs) != base.design_input_ids
        or draft.capability_selections != base.capability_locks
        or draft.source_materializations != base.source_locks
        or draft.resource_ceiling != base.resource_ceiling
    ):
        raise ValueError("standard package scientific components changed before issue")
    if any(
        value.access_disposition is not SourceAccessDisposition.VERIFIED_ACCESS
        for value in base.source_locks
    ):
        raise ValueError("issued source lock is no longer verified")
    _validate_registry_locks(base, registry)
    candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=candidate_identity,
        prerequisite_authority=None,
        grantee_id=grantee_id,
        storage_root_id=storage_root_id,
        relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        at_utc=at_utc,
    )
    if proposer_attestation.candidate != candidate_identity:
        raise ValueError("proposer attestation binds another standard candidate")
    if source_closure.implementation_sha256 != base.implementation_sha256:
        raise ValueError("source closure implementation differs from the candidate")
    qualification_identities = tuple(
        ObjectIdentity.from_record(value.receipt_id, value)
        for value in materialization_qualifications
    )
    if qualification_identities != base.source_qualification_receipts:
        raise ValueError("materialization qualification records are missing or stale")

    issue_seed = {
        "candidate": candidate_identity,
        "raw_authoring_package_sha256": raw_materialization_sha256,
        "authoring_package": candidate.authoring_package,
        "source_closure": ObjectIdentity.from_record(
            source_closure.source_closure_id,
            source_closure,
        ),
        "proposer_attestation": ObjectIdentity.from_record(
            proposer_attestation.attestation_id,
            proposer_attestation,
        ),
        "custody_authority": ObjectIdentity.from_record(
            custody_authority.authority_id,
            custody_authority,
        ),
        "registry_sha256": registry.fingerprint(),
        "qualifications": qualification_identities,
        "storage_root_id": storage_root_id,
    }
    issue_digest = hashlib.sha256(canonical_json_bytes(issue_seed)).hexdigest()
    issue_id = f"standard-issue.{issue_digest[:32]}"
    prefix = f"{ISSUE_PUBLICATION_RELATIVE_ROOT}/{issue_id}"
    raw_identity = ObjectIdentity(
        object_id=f"raw-authoring-package.{issue_id}",
        object_schema='empirical-lawhood/planning/study-authoring-package-raw-input',
        object_version="1.0.0",
        object_fingerprint=raw_materialization_sha256,
    )
    canonical_identity = ObjectIdentity.from_record(
        authoring_package.package_id,
        authoring_package,
    )
    payload_specs: list[tuple[str, str, str, ArtifactProfile, bytes, str]] = [
        (
            "authoring-source",
            f"{prefix}/authoring-source",
            authoring_package.SCHEMA,
            ArtifactProfile.TEXT_PARAMETERS,
            raw_authoring_package_bytes,
            hashlib.sha256(raw_authoring_package_bytes).hexdigest(),
        ),
        (
            "authoring-canonical",
            f"{prefix}/authoring-canonical.json",
            authoring_package.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            authoring_package.canonical_bytes(),
            authoring_package.fingerprint(),
        ),
        (
            "standard-candidate",
            f"{prefix}/standard-candidate.json",
            candidate.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            candidate.canonical_bytes(),
            candidate.fingerprint(),
        ),
        (
            "base-candidate",
            f"{prefix}/base-candidate.json",
            base.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            base.canonical_bytes(),
            base.fingerprint(),
        ),
        (
            "standard-compilation-report",
            f"{prefix}/standard-compilation-report.json",
            current_compilation.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            current_compilation.canonical_bytes(),
            current_compilation.fingerprint(),
        ),
        (
            "experiment-entry-package",
            f"{prefix}/experiment-entry-package.json",
            authoring_package.entry_package.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            authoring_package.entry_package.canonical_bytes(),
            authoring_package.entry_package.fingerprint(),
        ),
        (
            "registry",
            f"{prefix}/capability-registry.json",
            CapabilityRegistry.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            registry.canonical_bytes(),
            registry.fingerprint(),
        ),
        (
            "source-closure",
            f"{prefix}/source-closure.json",
            ImplementationSourceClosure.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            source_closure.canonical_bytes(),
            source_closure.fingerprint(),
        ),
        (
            "proposer-attestation",
            f"{prefix}/proposer-attestation.json",
            proposer_attestation.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            proposer_attestation.canonical_bytes(),
            proposer_attestation.fingerprint(),
        ),
        (
            "custody-authority",
            f"{prefix}/custody-authority.json",
            StudyOperationAuthority.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            custody_authority.canonical_bytes(),
            custody_authority.fingerprint(),
        ),
    ]
    component_records: tuple[CanonicalRecord, ...] = (
        base.system,
        base.experiment,
        base.campaign,
        base.protocol,
        base.scientific_graph,
        base.obligation_coverage,
        authoring_package.entry_package.register,
        authoring_package.entry_package.coverage,
        *materialization_qualifications,
        *((base.conditional_successor,) if base.conditional_successor else ()),
    )
    for index, record in enumerate(component_records):
        payload_specs.append(
            (
                f"component-{index:03d}",
                f"{prefix}/components/{index:03d}-{record.SCHEMA.rsplit('/', 2)[-1]}.json",
                record.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                record.canonical_bytes(),
                record.fingerprint(),
            )
        )
    members = tuple(
        sorted(
            (
                _member(
                    issue_id=issue_id,
                    member_label=label,
                    relative_path=path,
                    payload_schema=schema,
                    profile=profile,
                    payload=payload,
                    logical_content_sha256=logical_sha256,
                    media_type=(
                        base.authoring_materialization.media_type
                        if label == "authoring-source"
                        else "application/json"
                    ),
                )
                for label, path, schema, profile, payload, logical_sha256 in payload_specs
            ),
            key=lambda value: value.member_id,
        )
    )
    payload_by_member_id = {
        f"{issue_id}.{label}": payload
        for label, _path, _schema, _profile, payload, _logical_sha256 in payload_specs
    }
    manifest = record_type_standardissuedprogrammemanifest(
        issue_id=issue_id,
        candidate=candidate,
        authoring_package=authoring_package,
        raw_authoring_package=raw_identity,
        canonical_authoring_package=canonical_identity,
        source_closure=source_closure,
        proposer_attestation=proposer_attestation,
        custody_authority=ObjectIdentity.from_record(
            custody_authority.authority_id,
            custody_authority,
        ),
        registry=registry,
        materialization_qualifications=materialization_qualifications,
        members=members,
        storage_root_id=storage_root_id,
        publication_relative_root=ISSUE_PUBLICATION_RELATIVE_ROOT,
        issued_at_utc=custody_authority.issued_at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return StudyIssuePreparation(
        manifest=manifest,
        payloads=tuple(
            StudyIssuePayload(
                member=member,
                payload=payload_by_member_id[member.member_id],
            )
            for member in members
        ),
    )


def _member(
    *,
    issue_id: str,
    member_label: str,
    relative_path: str,
    payload_schema: str,
    media_type: str,
    profile: ArtifactProfile,
    payload: bytes,
    logical_content_sha256: str,
) -> IssuedStudyMember:
    physical_sha256 = hashlib.sha256(payload).hexdigest()
    return IssuedStudyMember(
        member_id=f"{issue_id}.{member_label}",
        relative_path=relative_path,
        payload_schema=payload_schema,
        media_type=media_type,
        profile=profile,
        size_bytes=len(payload),
        physical_sha256=physical_sha256,
        logical_content_sha256=logical_content_sha256,
    )


def _validate_registry_locks(
    candidate: DraftStudyCandidate,
    registry: CapabilityRegistry,
) -> None:
    for lock in candidate.capability_locks:
        manifest = registry.resolve(lock.capability_key, lock.capability_version)
        if manifest.implementation_sha256 != lock.implementation_sha256:
            raise ValueError("capability registry differs from candidate locks")


def build_study_approval_proposal(
    *,
    issued_study: (
        IssuedDraftManifest
        | IssuedStudyManifest
        | IssuedExecutableStudyManifest
    ),
    publication_receipt: (StudyPublicationReceipt | ExtensionPublicationReceipt),
) -> IssuedStudyApprovalProposal:
    """Generate the exact outcome-blind approval proposal from an issued bundle."""

    record_type_programmeapprovalproposal: type[IssuedStudyApprovalProposal] = (
        RetrospectiveApprovalProposal
        if isinstance(issued_study, (RetrospectiveIssuedBase, RetrospectiveIssuedStudy))
        else IssuedStudyApprovalProposal
    )
    manifest_identity = ObjectIdentity.from_record(
        issued_study.issue_id,
        issued_study,
    )
    if publication_receipt.issue_manifest != manifest_identity:
        raise ValueError("publication receipt binds another issued programme")
    expected_member_ids = tuple(sorted(value.member_id for value in issued_study.members))
    if publication_receipt.declared_member_ids != expected_member_ids:
        raise ValueError("publication receipt omits issued programme members")
    standard_candidate = issued_study.candidate
    candidate = (
        standard_candidate.base_candidate.base_candidate
        if isinstance(standard_candidate, ExecutableStudyCandidate)
        else standard_candidate.base_candidate
        if isinstance(standard_candidate, StudyCandidate)
        else standard_candidate
    )
    draft = (
        issued_study.candidate.authoring_package.base.draft
        if isinstance(issued_study, IssuedExecutableStudyManifest)
        else issued_study.authoring_package.draft
        if isinstance(issued_study, IssuedStudyManifest)
        else issued_study.draft
    )
    cutoffs = {
        value.information_cutoff.fingerprint(): value.information_cutoff
        for value in draft.design_inputs
    }
    if len(cutoffs) != 1:
        raise ValueError("issued programme lacks one exact shared approval information cutoff")
    return record_type_programmeapprovalproposal(
        proposal_id=f"programme-proposal.{issued_study.issue_id}",
        issued_study=manifest_identity,
        publication_receipt=ObjectIdentity.from_record(
            publication_receipt.receipt_id,
            publication_receipt,
        ),
        candidate=ObjectIdentity.from_record(
            standard_candidate.candidate_id,
            standard_candidate,
        ),
        candidate_experiment=candidate.experiment,
        design_origin=candidate.design_origin,
        proposer_attestation=ObjectIdentity.from_record(
            issued_study.proposer_attestation.attestation_id,
            issued_study.proposer_attestation,
        ),
        source_qualification_receipts=candidate.source_qualification_receipts,
        alternative_design_ids=draft.alternative_ids,
        budget=candidate.resource_ceiling,
        decision_cutoff=next(iter(cutoffs.values())),
        proposed_by=issued_study.proposer_attestation.proposer_id,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceiling=candidate.design_origin.parent_visibility_ceiling,
        visibility_ceiling=candidate.experiment.design_visibility_ceiling,
    )


__all__ = [
    "ISSUE_PUBLICATION_RELATIVE_ROOT",
    'IssuedDraftManifest',
    'IssuedExtensionBinding',
    'IssuedExtensionSet',
    'IssuedStudyMember',
    'StudyPublicationReceipt',
    'ExtensionPublicationReceipt',
    'StudyIssuePayload',
    'DraftIssuePreparation',
    'StudySourceClosureInspector',
    'StudyExtensionDecoderRegistration',
    'StudyExtensionValidationReceipt',
    'PublishedDraftStudy',
    'PublishedStudy',
    'PublishedExecutableStudy',
    'IssuedStudyManifest',
    'IssuedExecutableStudyManifest',
    'StudyIssuePreparation',
    'ExecutableStudyIssuePreparation',
    'build_study_approval_proposal',
    "bind_standard_issued_extensions",
    "decode_issued_extension_payload",
    'prepare_draft_candidate_issue',
    'prepare_study_issue',
    'prepare_executable_study_issue',
    'project_standard_study_extensions',
    'validate_issued_study_extensions',
]


@dataclass(frozen=True, slots=True)
class RetrospectiveIssuedBase(IssuedStudyManifest):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-issued-base'

    candidate: RetrospectiveStandardCandidate
    authoring_package: RetrospectiveAuthoringBase

    proposer_attestation: RetrospectiveProposerAttestation

    def __post_init__(self) -> None:
        if type(self.proposer_attestation) is not RetrospectiveProposerAttestation:
            raise ValueError("historical issue requires its exact exposed attestation")
        if type(self.candidate) is not RetrospectiveStandardCandidate:
            raise ValueError("RetrospectiveIssuedBase requires its exact candidate schema")
        if type(self.authoring_package) is not RetrospectiveAuthoringBase:
            raise ValueError("RetrospectiveIssuedBase requires its exact authoring_package schema")
        IssuedStudyManifest.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveIssuedStudy(IssuedExecutableStudyManifest):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-issued-study'

    base: RetrospectiveIssuedBase
    candidate: RetrospectiveExtensionCandidate

    extension_proposer_attestation: RetrospectiveProposerAttestation

    def __post_init__(self) -> None:
        if type(self.extension_proposer_attestation) is not RetrospectiveProposerAttestation:
            raise ValueError("historical issue requires its exact exposed attestation")
        if type(self.base) is not RetrospectiveIssuedBase:
            raise ValueError("RetrospectiveIssuedStudy requires its exact base schema")
        if type(self.candidate) is not RetrospectiveExtensionCandidate:
            raise ValueError("RetrospectiveIssuedStudy requires its exact candidate schema")
        IssuedExecutableStudyManifest.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectivePublicationReceipt(ExtensionPublicationReceipt):
    """Composite custody explicitly bound to a historical extension issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-publication-receipt'
