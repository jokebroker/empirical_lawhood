"""Guarded atomic publisher for deterministic programme issue/freeze bundles."""

from __future__ import annotations

from empirical_lawhood.runtime.study_issue import RetrospectiveIssuedBase, RetrospectiveIssuedStudy, RetrospectivePublicationReceipt

import hashlib
from pathlib import Path
from typing import Protocol

from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.bounded_process import run_bounded_command
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority, SourceClosureKind, require_study_authority
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    ArtifactWriteRequest,
    LogicalArtifactIdentity,
    MAX_ARTIFACT_PUBLICATION_MEMBERS,
)
from empirical_lawhood.runtime.issued_extension_payloads import MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES
from empirical_lawhood.runtime.study_issue import IssuedStudyMember, ExtensionPublicationReceipt, StudyPublicationReceipt, IssuedDraftManifest, MAX_ISSUE_PAYLOAD_BYTES, MAX_STANDARD_PROGRAMME_CONTROL_BYTES, MAX_COMPOSED_PROGRAMME_CONTROL_BYTES, DraftIssuePreparation, PublishedDraftStudy, PublishedStudy, PublishedExecutableStudy, IssuedStudyManifest, IssuedExecutableStudyManifest, StudyIssuePreparation, ExecutableStudyIssuePreparation
from empirical_lawhood.runtime.multi_world_study_issue import IssuedMultiWorldStudy, MultiWorldStudyIssueManifest, MultiWorldStudyIssuePreparation, MultiWorldStudyPublicationReceipt

from .artifacts import ExternalArtifactPlane
from .task_receipts import decode_artifact_manifest


MAX_PROGRAMME_AUTHORITY_BYTES = 10_000_000
MAX_ISSUED_PROGRAMME_MANIFEST_BYTES = MAX_ISSUE_PAYLOAD_BYTES
# Composed controls retain full provenance; scientific payload bounds stay fixed.
MAX_EXECUTABLE_STUDY_MANIFEST_BYTES = MAX_COMPOSED_PROGRAMME_CONTROL_BYTES
PROGRAMME_EXECUTION_GRANTEE_ID = "operator.execution-service"
PROGRAMME_ISSUE_GRANTEE_ID = "operator.issue-service"
PROGRAMME_REVEAL_GRANTEE_ID = "operator.evaluator-service"


class StudyOperationAuthorityStore(Protocol):
    def load(self, authority_id: str) -> StudyOperationAuthority: ...


class ExternalStudyOperationAuthorityStore:
    """Read exact owner-issued operation authorities from the evidence plane."""

    def __init__(self, artifact_plane: ExternalArtifactPlane) -> None:
        self._artifact_plane = artifact_plane

    @staticmethod
    def _relative_path(authority_id: str) -> str:
        validate_stable_id(authority_id, field_name="authority_id")
        return f"authority/programme-operation-authorities/{authority_id}.json"

    def load(self, authority_id: str) -> StudyOperationAuthority:
        relative_path = self._relative_path(authority_id)
        payload_path = self._artifact_plane.root.resolve(
            relative_path,
            for_write=False,
        )
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise KeyError(authority_id)
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        if (
            sidecar.logical.logical_artifact_id != authority_id
            or sidecar.logical.payload_schema != StudyOperationAuthority.SCHEMA
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
            or sidecar.materialization.size_bytes > MAX_PROGRAMME_AUTHORITY_BYTES
        ):
            raise ValueError(
                "programme operation authority manifest differs from its closed contract"
            )
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=sidecar.logical,
                materialization=sidecar.materialization,
            )
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        authority = decode_canonical_bytes(
            payload,
            StudyOperationAuthority,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        if (
            authority.authority_id != authority_id
            or authority.canonical_bytes() != payload
            or authority.fingerprint() != sidecar.logical.content_sha256
        ):
            raise ValueError("programme operation authority identity differs from its manifest")
        return authority

    def persist(self, authority: StudyOperationAuthority) -> ObjectIdentity:
        """Persist or exactly replay one owner-issued operation authority.

        Authority construction remains outside this store.  The method only
        provides the guarded, no-substitution persistence boundary needed by
        owner tooling before issue, execution or reveal.
        """

        if not isinstance(authority, StudyOperationAuthority):
            raise TypeError("programme authority store received another record type")
        expected = ObjectIdentity.from_record(authority.authority_id, authority)
        relative_path = self._relative_path(authority.authority_id)
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if payload_path.is_file() or sidecar_path.is_file():
            replayed = self.load(authority.authority_id)
            if (
                replayed != authority
                or ObjectIdentity.from_record(
                    replayed.authority_id,
                    replayed,
                )
                != expected
            ):
                raise FileExistsError("existing programme operation authority has another identity")
            return expected
        parent = ArtifactLineageParent(
            identity=authority.issuer,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        result = self._artifact_plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=authority.authority_id,
                relative_path=relative_path,
                payload_schema=authority.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"authority-scope.{authority.authority_id}",
                publication_scope_relative_root=("authority/programme-operation-authorities"),
                payload=authority.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=authority.fingerprint(),
                lineage_parents=(parent,),
            )
        )
        if (
            result.logical.logical_artifact_id != expected.object_id
            or result.logical.content_sha256 != expected.object_fingerprint
            or self.load(authority.authority_id) != authority
        ):
            raise RuntimeError("programme authority failed immutable replay")
        return expected


class GitStudySourceClosureInspector:
    """Bounded clean-commit inspector for the current repository worktree."""

    _MAX_STATUS_BYTES = 16 * 1024 * 1024
    _MAX_TREE_BYTES = 64 * 1024 * 1024

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve(strict=True)

    def observe(
        self,
        expected: ImplementationSourceClosure,
    ) -> ImplementationSourceClosure:
        from .source_origin import require_executing_target_source

        require_executing_target_source(self._repo_root)
        if expected.kind is not SourceClosureKind.CLEAN_GIT_COMMIT:
            raise PermissionError(
                "exact-source closure requires an independently composed manifest inspector"
            )
        head = (
            self._run_git(
                ("rev-parse", "--verify", "HEAD"),
                maximum_stdout_bytes=128,
            )
            .decode("ascii")
            .strip()
        )
        if head != expected.implementation_commit:
            raise PermissionError("implementation commit changed before issue")
        status = self._run_git(
            ("status", "--porcelain=v1", "--untracked-files=all"),
            maximum_stdout_bytes=self._MAX_STATUS_BYTES,
        )
        if status:
            raise PermissionError("implementation worktree is dirty before issue")
        tree = self._run_git(
            ("ls-tree", "-r", "-z", "--full-tree", "HEAD"),
            maximum_stdout_bytes=self._MAX_TREE_BYTES,
        )
        if hashlib.sha256(tree).hexdigest() != expected.source_tree_sha256:
            raise PermissionError("implementation source tree changed before issue")
        return expected

    def _run_git(
        self,
        arguments: tuple[str, ...],
        *,
        maximum_stdout_bytes: int,
    ) -> bytes:
        result = run_bounded_command(
            ("git", *arguments),
            cwd=self._repo_root,
            timeout_seconds=10.0,
            maximum_stdout_bytes=maximum_stdout_bytes,
            maximum_stderr_bytes=64 * 1024,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise PermissionError(
                f"implementation Git closure inspection failed: {message or result.returncode}"
            )
        return result.stdout


class ExternalIssuedStudyPublisher:
    """Publish one already validated issue bundle under exact custody authority."""

    def __init__(
        self,
        *,
        artifact_plane: ExternalArtifactPlane,
        authority_store: StudyOperationAuthorityStore,
        grantee_id: str,
    ) -> None:
        self._artifact_plane = artifact_plane
        self._authority_store = authority_store
        self._grantee_id = grantee_id

    @property
    def storage_root_id(self) -> str:
        return self._artifact_plane.root.contract.storage_root_id

    @property
    def grantee_id(self) -> str:
        return self._grantee_id

    @property
    def authority_store(self) -> StudyOperationAuthorityStore:
        return self._authority_store

    @property
    def artifact_plane(self) -> ExternalArtifactPlane:
        """Expose the already-guarded plane for authenticated issued-byte replay."""

        return self._artifact_plane

    def load(self, issue_id: str) -> PublishedDraftStudy:
        """Load and verify the exact externally published issue and receipt."""

        validate_stable_id(issue_id, field_name="issue_id")
        relative_path = f"issued-programmes/{issue_id}/manifest.json"
        payload_path = self._artifact_plane.root.resolve(
            relative_path,
            for_write=False,
        )
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise KeyError(issue_id)
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        if (
            sidecar.logical.logical_artifact_id != f"issued-manifest.{issue_id}"
            or sidecar.logical.payload_schema != IssuedDraftManifest.SCHEMA
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
            or sidecar.materialization.size_bytes > MAX_ISSUED_PROGRAMME_MANIFEST_BYTES
            or sidecar.publication is None
        ):
            raise RuntimeError("issued programme manifest sidecar differs from its closed contract")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=sidecar.logical,
                materialization=sidecar.materialization,
            )
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_ISSUED_PROGRAMME_MANIFEST_BYTES,
        )
        manifest = decode_canonical_bytes(
            payload,
            IssuedDraftManifest,
            maximum_bytes=MAX_ISSUED_PROGRAMME_MANIFEST_BYTES,
        )
        if (
            manifest.issue_id != issue_id
            or manifest.canonical_bytes() != payload
            or manifest.fingerprint() != sidecar.logical.content_sha256
            or manifest.storage_root_id != self.storage_root_id
        ):
            raise RuntimeError("issued programme manifest identity differs")
        receipt_id = f"issue-publication-receipt.{issue_id}"
        receipt = self._load_study_publication_receipt(receipt_id)
        if receipt is None:
            raise RuntimeError("issued programme publication receipt is absent")
        declared_member_ids = tuple(sorted(value.member_id for value in manifest.members))
        self._validate_manifest_members(
            manifest,
            publication=sidecar.publication,
        )
        total_payload_bytes = sum(value.size_bytes for value in manifest.members)
        total_payload_bytes += len(payload)
        self._validate_publication_receipt(
            receipt,
            manifest=manifest,
            manifest_logical=sidecar.logical,
            manifest_materialization=sidecar.materialization,
            publication=sidecar.publication,
            declared_member_ids=declared_member_ids,
            total_payload_bytes=total_payload_bytes,
        )
        return PublishedDraftStudy(
            manifest=manifest,
            publication_receipt=receipt,
        )

    def load_study(self, issue_id: str) -> PublishedStudy:
        """Load and verify one exact additive standard issue."""

        validate_stable_id(issue_id, field_name="issue_id")
        relative_path = f"issued-programmes/{issue_id}/manifest.json"
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise KeyError(issue_id)
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        if (
            sidecar.logical.logical_artifact_id != f"issued-manifest.{issue_id}"
            or sidecar.logical.payload_schema
            not in {IssuedStudyManifest.SCHEMA, RetrospectiveIssuedBase.SCHEMA}
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
            or sidecar.materialization.size_bytes > MAX_STANDARD_PROGRAMME_CONTROL_BYTES
            or sidecar.publication is None
        ):
            raise RuntimeError("standard issued manifest sidecar differs from its contract")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=sidecar.logical,
                materialization=sidecar.materialization,
            )
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_STANDARD_PROGRAMME_CONTROL_BYTES,
        )
        manifest_type: type[IssuedStudyManifest] = (
            RetrospectiveIssuedBase
            if sidecar.logical.payload_schema == RetrospectiveIssuedBase.SCHEMA
            else IssuedStudyManifest
        )
        manifest = decode_canonical_bytes(
            payload,
            manifest_type,
            maximum_bytes=MAX_STANDARD_PROGRAMME_CONTROL_BYTES,
        )
        if (
            manifest.issue_id != issue_id
            or manifest.canonical_bytes() != payload
            or manifest.fingerprint() != sidecar.logical.content_sha256
            or manifest.storage_root_id != self.storage_root_id
        ):
            raise RuntimeError("standard issued manifest identity differs")
        receipt_id = f"issue-publication-receipt.{issue_id}"
        receipt = self._load_study_publication_receipt(receipt_id)
        if receipt is None:
            raise RuntimeError("standard issue publication receipt is absent")
        declared_member_ids = tuple(sorted(value.member_id for value in manifest.members))
        self._validate_manifest_members(manifest, publication=sidecar.publication)
        total_payload_bytes = sum(value.size_bytes for value in manifest.members) + len(payload)
        self._validate_publication_receipt(
            receipt,
            manifest=manifest,
            manifest_logical=sidecar.logical,
            manifest_materialization=sidecar.materialization,
            publication=sidecar.publication,
            declared_member_ids=declared_member_ids,
            total_payload_bytes=total_payload_bytes,
        )
        return PublishedStudy(
            manifest=manifest,
            publication_receipt=receipt,
        )

    def validate_standard(
        self,
        preparation: ExecutableStudyIssuePreparation,
        *,
        at_utc: str,
    ) -> int:
        """Replay base custody plus exact fresh extension authority without writing."""

        manifest = preparation.manifest
        published_base = self.load_study(manifest.base.issue_id)
        if (
            published_base.manifest != manifest.base
            or published_base.publication_receipt != preparation.base_publication_receipt
        ):
            raise PermissionError("extension issue base publication replay differs")
        authority = self._authority_store.load(manifest.extension_custody_authority.object_id)
        if (
            authority != preparation.extension_custody_authority
            or ObjectIdentity.from_record(authority.authority_id, authority)
            != manifest.extension_custody_authority
        ):
            raise PermissionError("extension custody authority replay differs")
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
            subject=ObjectIdentity.from_record(
                manifest.candidate.candidate_id,
                manifest.candidate,
            ),
            prerequisite_authority=None,
            grantee_id=self._grantee_id,
            storage_root_id=self.storage_root_id,
            relative_root="issued-programmes",
            at_utc=at_utc,
        )
        manifest_payload = manifest.canonical_bytes()
        if len(manifest_payload) > MAX_EXECUTABLE_STUDY_MANIFEST_BYTES:
            raise ValueError("composed executable study manifest exceeds its bounded reader capacity")
        planned_bytes = sum(
            value.member.size_bytes for value in preparation.extension_payloads
        ) + len(manifest_payload)
        self._artifact_plane.root.verify(
            for_write=True,
            operation_minimum_free_bytes=max(1, planned_bytes * 4),
        )
        return planned_bytes

    def publish_standard(
        self,
        preparation: ExecutableStudyIssuePreparation,
        *,
        at_utc: str,
    ) -> ExtensionPublicationReceipt:
        "Atomically publish only extension bytes and their manifest."

        extension_payload_bytes = self.validate_standard(preparation, at_utc=at_utc)
        manifest = preparation.manifest
        scope_root = "issued-programmes"
        scope_id = f"programme-extension-issue.{manifest.issue_id}"
        manifest_logical_id = f"issued-executable-study-manifest.{manifest.issue_id}"
        manifest_relative_path = f"issued-programmes/{manifest.issue_id}/executable-study-manifest.json"
        manifest_payload = manifest.canonical_bytes()
        minimum_free_bytes = max(1, extension_payload_bytes * 4)
        candidate_parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(
                manifest.candidate.candidate_id,
                manifest.candidate,
            ),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        requests = tuple(
            ArtifactWriteRequest(
                logical_artifact_id=value.member.member_id,
                relative_path=value.member.relative_path,
                payload_schema=value.member.payload_schema,
                profile=value.member.profile,
                media_type=value.member.media_type,
                publication_scope_id=scope_id,
                publication_scope_relative_root=scope_root,
                payload=value.payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=value.member.logical_content_sha256,
                lineage_parents=(candidate_parent,),
                minimum_free_bytes=minimum_free_bytes,
            )
            for value in preparation.extension_payloads
        ) + (
            ArtifactWriteRequest(
                logical_artifact_id=manifest_logical_id,
                relative_path=manifest_relative_path,
                payload_schema=manifest.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=scope_id,
                publication_scope_relative_root=scope_root,
                payload=manifest_payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=manifest.fingerprint(),
                lineage_parents=(candidate_parent,),
                minimum_free_bytes=minimum_free_bytes,
            ),
        )
        results = self._artifact_plane.write_batch(
            tuple(request.as_stream() for request in requests)
        )
        manifest_result = next(
            value for value in results if value.logical.logical_artifact_id == manifest_logical_id
        )
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                self._artifact_plane.root.resolve(
                    f"{manifest_relative_path}.manifest.json",
                    for_write=False,
                ),
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        self._artifact_plane.verify_manifest(sidecar)
        if (
            sidecar.publication is None
            or sidecar.logical != manifest_result.logical
            or sidecar.materialization != manifest_result.materialization
        ):
            raise RuntimeError("issued executable study manifest lacks exact extension publication binding")
        receipt_type: type[ExtensionPublicationReceipt] = (
            RetrospectivePublicationReceipt
            if isinstance(manifest, RetrospectiveIssuedStudy)
            else ExtensionPublicationReceipt
        )
        receipt = receipt_type(
            receipt_id=f"issue-publication-receipt.{manifest.issue_id}",
            issue_manifest=ObjectIdentity.from_record(manifest.issue_id, manifest),
            base_publication_receipt=ObjectIdentity.from_record(
                preparation.base_publication_receipt.receipt_id,
                preparation.base_publication_receipt,
            ),
            manifest_logical=manifest_result.logical,
            manifest_materialization=manifest_result.materialization,
            extension_publication=sidecar.publication,
            declared_base_member_ids=tuple(
                sorted(value.member_id for value in manifest.base.members)
            ),
            declared_extension_member_ids=tuple(
                sorted(value.member_id for value in manifest.issued_extensions.members)
            ),
            extension_payload_bytes=extension_payload_bytes,
            total_payload_bytes=(
                preparation.base_publication_receipt.total_payload_bytes + extension_payload_bytes
            ),
            published_at_utc=at_utc,
        )
        existing = self._load_extension_publication_receipt(receipt.receipt_id)
        if existing is not None:
            if existing != receipt:
                raise RuntimeError("existing extension publication receipt binds another extension publication")
            return existing
        receipt_relative_path = f"issued-programmes/{manifest.issue_id}/extension-publication-receipt.json"
        self._artifact_plane.write_stream(
            ArtifactWriteRequest(
                logical_artifact_id=receipt.receipt_id,
                relative_path=receipt_relative_path,
                payload_schema=receipt.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"programme-extension-receipt.{manifest.issue_id}",
                publication_scope_relative_root=f"issued-programmes/{manifest.issue_id}",
                payload=receipt.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=receipt.fingerprint(),
                lineage_parents=(
                    ArtifactLineageParent(
                        identity=receipt.issue_manifest,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
                minimum_free_bytes=minimum_free_bytes,
            ).as_stream()
        )
        loaded = self._load_extension_publication_receipt(receipt.receipt_id)
        if loaded != receipt:
            raise RuntimeError("extension publication receipt failed immutable replay")
        return receipt

    def load_executable_study(self, issue_id: str) -> PublishedExecutableStudy:
        """Load and verify one composite base-plus-extension issue after restart."""

        validate_stable_id(issue_id, field_name="issue_id")
        relative_path = f"issued-programmes/{issue_id}/executable-study-manifest.json"
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise KeyError(issue_id)
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(sidecar_path, maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES)
        )
        if (
            sidecar.logical.logical_artifact_id != f"issued-executable-study-manifest.{issue_id}"
            or sidecar.logical.payload_schema
            not in {IssuedExecutableStudyManifest.SCHEMA, RetrospectiveIssuedStudy.SCHEMA}
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
            or sidecar.materialization.size_bytes > MAX_EXECUTABLE_STUDY_MANIFEST_BYTES
            or sidecar.publication is None
        ):
            raise RuntimeError("issued executable study manifest sidecar differs from its contract")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(logical=sidecar.logical, materialization=sidecar.materialization)
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_EXECUTABLE_STUDY_MANIFEST_BYTES,
        )
        manifest_type: type[IssuedExecutableStudyManifest] = (
            RetrospectiveIssuedStudy
            if sidecar.logical.payload_schema == RetrospectiveIssuedStudy.SCHEMA
            else IssuedExecutableStudyManifest
        )
        manifest = decode_canonical_bytes(
            payload,
            manifest_type,
            maximum_bytes=MAX_EXECUTABLE_STUDY_MANIFEST_BYTES,
        )
        if (
            manifest.issue_id != issue_id
            or manifest.canonical_bytes() != payload
            or manifest.fingerprint() != sidecar.logical.content_sha256
        ):
            raise RuntimeError("issued executable study manifest identity differs")
        published_base = self.load_study(manifest.base.issue_id)
        if published_base.manifest != manifest.base:
            raise RuntimeError("issued study base manifest replay differs")
        receipt = self._load_extension_publication_receipt(f"issue-publication-receipt.{issue_id}")
        if receipt is None:
            raise RuntimeError("extension publication receipt is absent")
        if receipt.base_publication_receipt != ObjectIdentity.from_record(
            published_base.publication_receipt.receipt_id,
            published_base.publication_receipt,
        ):
            raise RuntimeError("extension publication receipt changes its base custody")
        published_by_id = {
            value.logical_artifact_id: value for value in sidecar.publication.members
        }
        for member in manifest.issued_extensions.members:
            published = published_by_id.get(member.member_id)
            if (
                published is None
                or published.storage_root_id != self.storage_root_id
                or published.relative_path != member.relative_path
                or published.physical_sha256 != member.physical_sha256
                or published.size_bytes != member.size_bytes
            ):
                raise RuntimeError("executable study extension member differs from publication")
        if (
            receipt.issue_manifest != ObjectIdentity.from_record(issue_id, manifest)
            or receipt.manifest_logical != sidecar.logical
            or receipt.manifest_materialization != sidecar.materialization
            or receipt.extension_publication != sidecar.publication
            or receipt.declared_member_ids
            != tuple(sorted(value.member_id for value in manifest.members))
        ):
            raise RuntimeError("extension publication receipt differs from manifest/publication")
        return PublishedExecutableStudy(
            manifest=manifest,
            publication_receipt=receipt,
        )

    def validate_bundle(
        self,
        preparation: MultiWorldStudyIssuePreparation,
        *,
        at_utc: str,
    ) -> int:
        """Replay bundle custody and all three immutable child publications."""

        manifest = preparation.manifest
        if self._artifact_plane.root.contract.storage_root_id != manifest.storage_root_id:
            raise PermissionError("bundle publisher is composed for another storage root")
        authority = self._authority_store.load(manifest.custody_authority.object_id)
        if authority != preparation.custody_authority:
            raise PermissionError("bundle custody authority store returned another record")
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
            subject=ObjectIdentity.from_record(
                manifest.candidate.candidate_id,
                manifest.candidate,
            ),
            prerequisite_authority=None,
            grantee_id=self._grantee_id,
            storage_root_id=manifest.storage_root_id,
            relative_root=manifest.publication_relative_root,
            at_utc=at_utc,
        )
        for child in manifest.children:
            published = self.load_executable_study(child.manifest.issue_id)
            if (
                published.manifest != child.manifest
                or published.publication_receipt != child.publication_receipt
            ):
                raise RuntimeError("bundle child publication replay differs")
        payload_bytes = len(manifest.canonical_bytes())
        self._artifact_plane.root.verify(
            for_write=True,
            operation_minimum_free_bytes=max(1, payload_bytes * 4),
        )
        return payload_bytes

    def publish_bundle(
        self,
        preparation: MultiWorldStudyIssuePreparation,
        *,
        at_utc: str,
    ) -> MultiWorldStudyPublicationReceipt:
        """Publish only the parent manifest; child publication bytes remain immutable."""

        payload_bytes = self.validate_bundle(preparation, at_utc=at_utc)
        manifest = preparation.manifest
        scope_root = f"{manifest.publication_relative_root}/{manifest.issue_id}"
        scope_id = f"programme-bundle-issue.{manifest.issue_id}"
        logical_id = f"issued-multi-world-study-manifest.{manifest.issue_id}"
        relative_path = f"{scope_root}/multi-world-study-manifest.json"
        candidate_parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(
                manifest.candidate.candidate_id,
                manifest.candidate,
            ),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        child_parents = tuple(
            ArtifactLineageParent(
                identity=ObjectIdentity.from_record(
                    value.publication_receipt.receipt_id,
                    value.publication_receipt,
                ),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            for value in manifest.children
        )
        results = self._artifact_plane.write_batch(
            (
                ArtifactWriteRequest(
                    logical_artifact_id=logical_id,
                    relative_path=relative_path,
                    payload_schema=manifest.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    publication_scope_id=scope_id,
                    publication_scope_relative_root=scope_root,
                    payload=manifest.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    parent_visibility_ceilings=tuple(
                        VisibilityCeiling.PROSPECTIVE for _ in (candidate_parent, *child_parents)
                    ),
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    logical_content_sha256=manifest.fingerprint(),
                    lineage_parents=(candidate_parent, *child_parents),
                    minimum_free_bytes=max(1, payload_bytes * 4),
                ),
            )
        )
        result = results[0]
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                self._artifact_plane.root.resolve(
                    f"{relative_path}.manifest.json",
                    for_write=False,
                ),
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        self._artifact_plane.verify_manifest(sidecar)
        if (
            sidecar.publication is None
            or sidecar.logical != result.logical
            or sidecar.materialization != result.materialization
        ):
            raise RuntimeError("bundle manifest lacks its atomic publication binding")
        receipt_id = f"programme-bundle-publication-receipt.{manifest.issue_id}"
        existing = self._load_bundle_publication_receipt(receipt_id)
        expected_child_receipts = tuple(
            sorted(
                (
                    ObjectIdentity.from_record(
                        value.publication_receipt.receipt_id,
                        value.publication_receipt,
                    )
                    for value in manifest.children
                ),
                key=lambda value: value.object_id,
            )
        )
        if existing is not None:
            if (
                existing.issue_manifest
                != ObjectIdentity.from_record(
                    manifest.issue_id,
                    manifest,
                )
                or existing.child_publication_receipts != expected_child_receipts
                or existing.manifest_logical != result.logical
                or existing.manifest_materialization != result.materialization
                or existing.publication != sidecar.publication
                or existing.manifest_payload_bytes != payload_bytes
            ):
                raise RuntimeError("existing bundle receipt binds another publication")
            return existing
        receipt = MultiWorldStudyPublicationReceipt(
            receipt_id=receipt_id,
            issue_manifest=ObjectIdentity.from_record(manifest.issue_id, manifest),
            child_publication_receipts=expected_child_receipts,
            manifest_logical=result.logical,
            manifest_materialization=result.materialization,
            publication=sidecar.publication,
            manifest_payload_bytes=payload_bytes,
            published_at_utc=at_utc,
        )
        receipt_relative_path = f"{scope_root}/multi-world-publication-receipt.json"
        self._artifact_plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=receipt.receipt_id,
                relative_path=receipt_relative_path,
                payload_schema=receipt.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"programme-bundle-receipt.{manifest.issue_id}",
                publication_scope_relative_root=scope_root,
                payload=receipt.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=receipt.fingerprint(),
                lineage_parents=(
                    ArtifactLineageParent(
                        identity=receipt.issue_manifest,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
                minimum_free_bytes=max(1, payload_bytes * 4),
            )
        )
        loaded = self._load_bundle_publication_receipt(receipt.receipt_id)
        if loaded != receipt:
            raise RuntimeError("bundle publication receipt failed immutable replay")
        return receipt

    def load_bundle(self, issue_id: str) -> IssuedMultiWorldStudy:
        """Load and verify the exact parent manifest, receipt and child lineage."""

        validate_stable_id(issue_id, field_name="issue_id")
        relative_path = f"issued-programme-bundles/{issue_id}/multi-world-study-manifest.json"
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise KeyError(issue_id)
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(sidecar_path, maximum_bytes=MAX_ISSUED_PROGRAMME_MANIFEST_BYTES)
        )
        if (
            sidecar.logical.logical_artifact_id != f"issued-multi-world-study-manifest.{issue_id}"
            or sidecar.logical.payload_schema != MultiWorldStudyIssueManifest.SCHEMA
            or sidecar.materialization.relative_path != relative_path
            or sidecar.publication is None
        ):
            raise RuntimeError("issued bundle manifest sidecar differs")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(logical=sidecar.logical, materialization=sidecar.materialization)
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_ISSUED_PROGRAMME_MANIFEST_BYTES,
        )
        manifest = decode_canonical_bytes(
            payload,
            MultiWorldStudyIssueManifest,
            maximum_bytes=MAX_ISSUED_PROGRAMME_MANIFEST_BYTES,
        )
        if (
            manifest.issue_id != issue_id
            or manifest.canonical_bytes() != payload
            or manifest.fingerprint() != sidecar.logical.content_sha256
        ):
            raise RuntimeError("issued bundle manifest identity differs")
        for child in manifest.children:
            published = self.load_executable_study(child.manifest.issue_id)
            if (
                published.manifest != child.manifest
                or published.publication_receipt != child.publication_receipt
            ):
                raise RuntimeError("issued bundle child replay differs")
        receipt = self._load_bundle_publication_receipt(
            f"programme-bundle-publication-receipt.{issue_id}"
        )
        if receipt is None or (
            receipt.issue_manifest != ObjectIdentity.from_record(issue_id, manifest)
            or receipt.manifest_logical != sidecar.logical
            or receipt.manifest_materialization != sidecar.materialization
            or receipt.publication != sidecar.publication
        ):
            raise RuntimeError("issued bundle publication receipt differs")
        return IssuedMultiWorldStudy(
            bundle_id=f"issued-bundle.{issue_id}",
            manifest=manifest,
            publication_receipt=receipt,
        )

    def _load_bundle_publication_receipt(
        self,
        receipt_id: str,
    ) -> MultiWorldStudyPublicationReceipt | None:
        validate_stable_id(receipt_id, field_name="receipt_id")
        issue_id = receipt_id.removeprefix("programme-bundle-publication-receipt.")
        relative_path = f"issued-programme-bundles/{issue_id}/multi-world-publication-receipt.json"
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.exists() and not sidecar_path.exists():
            return None
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError("bundle publication receipt is partially materialized")
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(sidecar_path, maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES)
        )
        if (
            sidecar.logical.logical_artifact_id != receipt_id
            or sidecar.logical.payload_schema != MultiWorldStudyPublicationReceipt.SCHEMA
            or sidecar.materialization.relative_path != relative_path
        ):
            raise RuntimeError("bundle publication receipt sidecar differs")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(logical=sidecar.logical, materialization=sidecar.materialization)
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        receipt = decode_canonical_bytes(
            payload,
            MultiWorldStudyPublicationReceipt,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        if (
            receipt.receipt_id != receipt_id
            or receipt.canonical_bytes() != payload
            or receipt.fingerprint() != sidecar.logical.content_sha256
        ):
            raise RuntimeError("bundle publication receipt identity differs")
        return receipt

    def _load_extension_publication_receipt(
        self,
        receipt_id: str,
    ) -> ExtensionPublicationReceipt | None:
        validate_stable_id(receipt_id, field_name="receipt_id")
        issue_id = receipt_id.removeprefix("issue-publication-receipt.")
        relative_path = f"issued-programmes/{issue_id}/extension-publication-receipt.json"
        payload_path = self._artifact_plane.root.resolve(relative_path, for_write=False)
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.exists() and not sidecar_path.exists():
            return None
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError("extension publication receipt is partially materialized")
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(sidecar_path, maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES)
        )
        if (
            sidecar.logical.logical_artifact_id != receipt_id
            or sidecar.logical.payload_schema
            not in {
                ExtensionPublicationReceipt.SCHEMA,
                RetrospectivePublicationReceipt.SCHEMA,
            }
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
        ):
            raise RuntimeError("extension publication receipt sidecar differs")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(logical=sidecar.logical, materialization=sidecar.materialization)
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        receipt_type: type[ExtensionPublicationReceipt] = (
            RetrospectivePublicationReceipt
            if sidecar.logical.payload_schema == RetrospectivePublicationReceipt.SCHEMA
            else ExtensionPublicationReceipt
        )
        receipt = decode_canonical_bytes(
            payload,
            receipt_type,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        if (
            receipt.receipt_id != receipt_id
            or receipt.canonical_bytes() != payload
            or receipt.fingerprint() != sidecar.logical.content_sha256
        ):
            raise RuntimeError("extension publication receipt identity differs")
        return receipt

    def _validate_manifest_members(
        self,
        manifest: IssuedDraftManifest | IssuedStudyManifest,
        *,
        publication: ArtifactPublicationBinding,
    ) -> None:
        published_by_id = {value.logical_artifact_id: value for value in publication.members}
        for member in manifest.members:
            published = published_by_id.get(member.member_id)
            if (
                published is None
                or published.storage_root_id != manifest.storage_root_id
                or published.relative_path != member.relative_path
                or published.physical_sha256 != member.physical_sha256
                or published.size_bytes != member.size_bytes
                or published.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
                or published.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            ):
                raise RuntimeError("issued programme member differs from its committed publication")
            member_sidecar_path = self._artifact_plane.root.resolve(
                f"{member.relative_path}.manifest.json",
                for_write=False,
            )
            member_sidecar = decode_artifact_manifest(
                read_bounded_bytes(
                    member_sidecar_path,
                    maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
                )
            )
            if (
                member_sidecar.publication != publication
                or member_sidecar.logical.logical_artifact_id != member.member_id
                or member_sidecar.logical.payload_schema != member.payload_schema
                or member_sidecar.logical.profile is not member.profile
                or member_sidecar.logical.media_type != member.media_type
                or member_sidecar.logical.content_sha256 != member.logical_content_sha256
                or member_sidecar.materialization.relative_path != member.relative_path
                or member_sidecar.materialization.physical_sha256 != member.physical_sha256
                or member_sidecar.materialization.size_bytes != member.size_bytes
            ):
                raise RuntimeError("issued programme member sidecar differs from its manifest")

    def validate(
        self,
        preparation: DraftIssuePreparation | StudyIssuePreparation,
        *,
        at_utc: str,
    ) -> int:
        """Replay exact custody authority and storage readiness without writing."""

        manifest = preparation.manifest
        if self._artifact_plane.root.contract.storage_root_id != manifest.storage_root_id:
            raise PermissionError("issue publisher is composed for another storage root")
        authority = self._authority_store.load(manifest.custody_authority.object_id)
        if (
            ObjectIdentity.from_record(authority.authority_id, authority)
            != manifest.custody_authority
        ):
            raise PermissionError("custody authority store returned a substituted record")
        candidate_identity = ObjectIdentity.from_record(
            manifest.candidate.candidate_id,
            manifest.candidate,
        )
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
            subject=candidate_identity,
            prerequisite_authority=None,
            grantee_id=self._grantee_id,
            storage_root_id=manifest.storage_root_id,
            relative_root=manifest.publication_relative_root,
            at_utc=at_utc,
        )
        manifest_payload = manifest.canonical_bytes()
        manifest_limit = (
            MAX_STANDARD_PROGRAMME_CONTROL_BYTES
            if isinstance(manifest, IssuedStudyManifest)
            else MAX_ISSUED_PROGRAMME_MANIFEST_BYTES
        )
        if len(manifest_payload) > manifest_limit:
            raise ValueError("issued manifest exceeds its bounded reader capacity")
        total_payload_bytes = sum(value.member.size_bytes for value in preparation.payloads)
        total_payload_bytes += len(manifest_payload)
        minimum_free_bytes = max(1, total_payload_bytes * 4)
        self._artifact_plane.root.verify(
            for_write=True,
            operation_minimum_free_bytes=minimum_free_bytes,
        )
        return total_payload_bytes

    def publish(
        self,
        preparation: DraftIssuePreparation | StudyIssuePreparation,
        *,
        at_utc: str,
    ) -> StudyPublicationReceipt:
        total_payload_bytes = self.validate(preparation, at_utc=at_utc)
        manifest = preparation.manifest
        scope_root = f"{manifest.publication_relative_root}/{manifest.issue_id}"
        scope_id = f"programme-issue.{manifest.issue_id}"
        manifest_logical_id = f"issued-manifest.{manifest.issue_id}"
        manifest_relative_path = f"{scope_root}/manifest.json"
        manifest_payload = manifest.canonical_bytes()
        minimum_free_bytes = max(1, total_payload_bytes * 4)
        candidate_identity = ObjectIdentity.from_record(
            manifest.candidate.candidate_id,
            manifest.candidate,
        )
        candidate_parent = ArtifactLineageParent(
            identity=candidate_identity,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        requests = tuple(
            ArtifactWriteRequest(
                logical_artifact_id=value.member.member_id,
                relative_path=value.member.relative_path,
                payload_schema=value.member.payload_schema,
                profile=value.member.profile,
                media_type=value.member.media_type,
                publication_scope_id=scope_id,
                publication_scope_relative_root=scope_root,
                payload=value.payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=value.member.logical_content_sha256,
                lineage_parents=(candidate_parent,),
                minimum_free_bytes=minimum_free_bytes,
            )
            for value in preparation.payloads
        ) + (
            ArtifactWriteRequest(
                logical_artifact_id=manifest_logical_id,
                relative_path=manifest_relative_path,
                payload_schema=manifest.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=scope_id,
                publication_scope_relative_root=scope_root,
                payload=manifest_payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=manifest.fingerprint(),
                lineage_parents=(candidate_parent,),
                minimum_free_bytes=minimum_free_bytes,
            ),
        )
        results = self._artifact_plane.write_batch(
            tuple(request.as_stream() for request in requests)
        )
        manifest_result = next(
            value for value in results if value.logical.logical_artifact_id == manifest_logical_id
        )
        sidecar_path = self._artifact_plane.root.resolve(
            f"{manifest_relative_path}.manifest.json",
            for_write=False,
        )
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(sidecar_path, maximum_bytes=10_000_000)
        )
        self._artifact_plane.verify_manifest(sidecar)
        if (
            sidecar.publication is None
            or sidecar.logical != manifest_result.logical
            or sidecar.materialization != manifest_result.materialization
        ):
            raise RuntimeError("issued manifest lacks its exact atomic publication binding")
        declared_member_ids = tuple(
            sorted(value.member.member_id for value in preparation.payloads)
        )
        receipt_id = f"issue-publication-receipt.{manifest.issue_id}"
        existing = self._load_study_publication_receipt(receipt_id)
        if existing is not None:
            self._validate_publication_receipt(
                existing,
                manifest=preparation.manifest,
                manifest_logical=manifest_result.logical,
                manifest_materialization=manifest_result.materialization,
                publication=sidecar.publication,
                declared_member_ids=declared_member_ids,
                total_payload_bytes=total_payload_bytes,
            )
            return existing
        receipt = StudyPublicationReceipt(
            receipt_id=receipt_id,
            issue_manifest=ObjectIdentity.from_record(manifest.issue_id, manifest),
            manifest_logical=manifest_result.logical,
            manifest_materialization=manifest_result.materialization,
            publication=sidecar.publication,
            declared_member_ids=declared_member_ids,
            total_payload_bytes=total_payload_bytes,
            published_at_utc=at_utc,
        )
        receipt_relative_path = f"{scope_root}/publication-receipt.json"
        self._artifact_plane.write_stream(
            ArtifactWriteRequest(
                logical_artifact_id=receipt.receipt_id,
                relative_path=receipt_relative_path,
                payload_schema=receipt.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"programme-issue-receipt.{manifest.issue_id}",
                publication_scope_relative_root=scope_root,
                payload=receipt.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=receipt.fingerprint(),
                lineage_parents=(
                    ArtifactLineageParent(
                        identity=receipt.issue_manifest,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
                minimum_free_bytes=minimum_free_bytes,
            ).as_stream()
        )
        loaded = self._load_study_publication_receipt(receipt.receipt_id)
        if loaded != receipt:
            raise RuntimeError("programme issue publication receipt failed immutable replay")
        return receipt

    def _load_study_publication_receipt(
        self,
        receipt_id: str,
    ) -> StudyPublicationReceipt | None:
        validate_stable_id(receipt_id, field_name="receipt_id")
        issue_id = receipt_id.removeprefix("issue-publication-receipt.")
        relative_path = f"issued-programmes/{issue_id}/publication-receipt.json"
        payload_path = self._artifact_plane.root.resolve(
            relative_path,
            for_write=False,
        )
        sidecar_path = self._artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not payload_path.exists() and not sidecar_path.exists():
            return None
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError("programme issue publication receipt is partially materialized")
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        if (
            sidecar.logical.logical_artifact_id != receipt_id
            or sidecar.logical.payload_schema != StudyPublicationReceipt.SCHEMA
            or sidecar.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or sidecar.logical.media_type != "application/json"
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.relative_path != relative_path
            or sidecar.materialization.size_bytes > MAX_PROGRAMME_AUTHORITY_BYTES
        ):
            raise RuntimeError("programme issue publication receipt sidecar differs")
        self._artifact_plane.verify_manifest(
            ArtifactManifest(
                logical=sidecar.logical,
                materialization=sidecar.materialization,
            )
        )
        payload = read_bounded_bytes(
            payload_path,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        receipt = decode_canonical_bytes(
            payload,
            StudyPublicationReceipt,
            maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
        )
        if (
            receipt.receipt_id != receipt_id
            or receipt.canonical_bytes() != payload
            or receipt.fingerprint() != sidecar.logical.content_sha256
        ):
            raise RuntimeError("programme issue publication receipt identity differs")
        return receipt

    @staticmethod
    def _validate_publication_receipt(
        receipt: StudyPublicationReceipt,
        *,
        manifest: IssuedDraftManifest | IssuedStudyManifest,
        manifest_logical: LogicalArtifactIdentity,
        manifest_materialization: ArtifactMaterialization,
        publication: ArtifactPublicationBinding,
        declared_member_ids: tuple[str, ...],
        total_payload_bytes: int,
    ) -> None:
        if (
            receipt.issue_manifest
            != ObjectIdentity.from_record(
                manifest.issue_id,
                manifest,
            )
            or receipt.manifest_logical != manifest_logical
            or receipt.manifest_materialization != manifest_materialization
            or receipt.publication != publication
            or receipt.declared_member_ids != declared_member_ids
            or receipt.total_payload_bytes != total_payload_bytes
        ):
            raise RuntimeError("existing programme issue receipt binds another publication")


class ExternalIssuedExtensionPayloadSource:
    "Bounded reader for members of an authenticated published extension issue."

    def __init__(self, publisher: ExternalIssuedStudyPublisher) -> None:
        self._publisher = publisher

    def read_member(
        self,
        *,
        issue_id: str,
        member: IssuedStudyMember,
        maximum_bytes: int,
    ) -> bytes:
        return self.read_members(issue_id=issue_id, requests=((member, maximum_bytes),))[0]

    def read_members(
        self,
        *,
        issue_id: str,
        requests: tuple[tuple[IssuedStudyMember, int], ...],
    ) -> tuple[bytes, ...]:
        validate_stable_id(issue_id, field_name="issue_id")
        if not 0 < len(requests) <= MAX_ARTIFACT_PUBLICATION_MEMBERS:
            raise ValueError("issued extension member count exceeds its read bound")
        if len({member.member_id for member, _ in requests}) != len(requests):
            raise ValueError("issued extension member requests are duplicated")
        if any(bound < 1 or member.size_bytes > bound for member, bound in requests):
            raise ValueError("issued extension member exceeds its read bound")
        if sum(member.size_bytes for member, _ in requests) > MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES:
            raise ValueError("issued extension members exceed the aggregate read bound")
        # All trust is local to this call. Later resolution/recovery reauthenticates.
        published = self._publisher.load_executable_study(issue_id)
        sidecars = tuple(self._member_sidecar(published, member) for member, _ in requests)
        plane = self._publisher.artifact_plane
        # The existing batch verifier retains sibling/commit-marker and snapshot guards.
        plane.verify_manifests(sidecars)
        payloads: list[bytes] = []
        for member, maximum_bytes in requests:
            payload = read_bounded_bytes(
                plane.root.resolve(member.relative_path, for_write=False),
                maximum_bytes=maximum_bytes,
            )
            if (
                len(payload) != member.size_bytes
                or hashlib.sha256(payload).hexdigest() != member.physical_sha256
            ):
                raise RuntimeError("issued extension member bytes differ after authenticated open")
            payloads.append(payload)
        return tuple(payloads)

    def _member_sidecar(
        self,
        published: PublishedExecutableStudy,
        member: IssuedStudyMember,
    ) -> ArtifactManifest:
        exact = next(
            (
                value
                for value in published.manifest.issued_extensions.members
                if value.member_id == member.member_id
            ),
            None,
        )
        if exact != member:
            raise RuntimeError("requested extension member differs from published issue")
        publication = published.publication_receipt.extension_publication
        publication_member = next(
            (
                value
                for value in publication.members
                if value.logical_artifact_id == member.member_id
            ),
            None,
        )
        if (
            publication_member is None
            or publication_member.storage_root_id
            != self._publisher.artifact_plane.root.contract.storage_root_id
            or publication_member.relative_path != member.relative_path
            or publication_member.physical_sha256 != member.physical_sha256
            or publication_member.size_bytes != member.size_bytes
            or publication_member.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or publication_member.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise RuntimeError("published extension member differs from its custody binding")
        plane = self._publisher.artifact_plane
        sidecar_path = plane.root.resolve(
            f"{member.relative_path}.manifest.json",
            for_write=False,
        )
        sidecar = decode_artifact_manifest(
            read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_PROGRAMME_AUTHORITY_BYTES,
            )
        )
        if (
            sidecar.publication != publication
            or sidecar.logical.logical_artifact_id != member.member_id
            or sidecar.logical.payload_schema != member.payload_schema
            or sidecar.logical.profile is not member.profile
            or sidecar.logical.media_type != member.media_type
            or sidecar.logical.content_sha256 != member.logical_content_sha256
            or sidecar.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or sidecar.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or sidecar.materialization.storage_root_id != plane.root.contract.storage_root_id
            or sidecar.materialization.relative_path != member.relative_path
            or sidecar.materialization.physical_sha256 != member.physical_sha256
            or sidecar.materialization.size_bytes != member.size_bytes
        ):
            raise RuntimeError("issued extension member sidecar differs from its contract")
        return sidecar


__all__ = [
    "ExternalIssuedExtensionPayloadSource",
    'ExternalStudyOperationAuthorityStore',
    'ExternalIssuedStudyPublisher',
    'GitStudySourceClosureInspector',
    "MAX_ISSUED_PROGRAMME_MANIFEST_BYTES",
    "MAX_PROGRAMME_AUTHORITY_BYTES",
    "PROGRAMME_EXECUTION_GRANTEE_ID",
    "PROGRAMME_ISSUE_GRANTEE_ID",
    "PROGRAMME_REVEAL_GRANTEE_ID",
    'StudyOperationAuthorityStore',
]
