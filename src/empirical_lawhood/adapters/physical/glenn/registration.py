"""Exact process-local inspection of the already-held Glenn archive.

The adapter accepts one precomposed registration manifest and one registered
source opener.  It can neither select another path nor write evidence.  Its
result is the canonical, proof-free input to the separately authorized
registration publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityKind,
    DatasetRegistrationManifest,
    DatasetSelectorMemberContract,
)
from empirical_lawhood.planning.datasets import (
    CurrentMaterializationVerification,
    CustodyState,
    DatasetMaterialization,
    DatasetMaterializationVerificationObservations,
    DatasetMaterializationVerificationSubject,
    EvidenceReferenceKind,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.dataset_io import (
    DatasetSourceGuardReceipt,
    DatasetSourceOpener,
    GuardedDatasetSource,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationSupportingRecord,
    PreparedDatasetRegistration,
)
from empirical_lawhood.runtime.datasets import (
    DatasetFormatInspectorRegistration,
    DatasetInspectorImplementationBinding,
)

from .contracts import (
    GlennAdapterError,
    GlennArchiveInput,
    GlennTransformProfile,
)
from .manifests import GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT
from .registry import GLENN_INSPECTOR_REGISTRY_KEY
from .source import GlennArchiveAudit, GlennMemberReceipt, inspect_glenn_archive


GLENN_ARCHIVE_VERIFICATION_POLICY_ID: Final = "policy.glenn-archive-verification"
GLENN_ARCHIVE_REGISTRATION_INSPECTION_ID: Final = "inspection.glenn-archive-registration"
_CURRENT_VERIFICATION_VALIDITY_SECONDS: Final = 300


def _positive_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _expected_members(
    profile: GlennTransformProfile,
) -> tuple[DatasetSelectorMemberContract, ...]:
    return tuple(
        sorted(
            (
                DatasetSelectorMemberContract(
                    member_id=value.member_id,
                    relative_locator=value.relative_locator,
                    expected_physical_sha256=value.expected_physical_sha256,
                    expected_compressed_size_bytes=value.expected_compressed_size_bytes,
                    expected_uncompressed_size_bytes=value.expected_uncompressed_size_bytes,
                    expected_crc32=value.expected_crc32,
                    media_type=value.media_type,
                    format_profile_id=value.format_profile_id,
                    inspection_policy_id=value.inspection_policy_id,
                )
                for value in profile.members
            ),
            key=lambda value: value.member_id,
        )
    )


@dataclass(frozen=True, slots=True)
class GlennArchiveVerificationPolicy(CanonicalRecord):
    """Exact structural policy applied before the source can be registered."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-archive-verification-policy'

    policy_id: str
    manifest: ObjectIdentity
    inspector_registration: ObjectIdentity
    archive_inspection_profile_id: str
    materialization_format_profile_id: str
    expected_archive_sha256: str
    expected_archive_size_bytes: int
    expected_members: tuple[DatasetSelectorMemberContract, ...]
    maximum_archive_bytes: int
    maximum_members: int
    maximum_central_directory_bytes: int
    maximum_member_uncompressed_bytes: int
    maximum_total_uncompressed_bytes: int
    maximum_selected_uncompressed_bytes: int
    maximum_member_compression_ratio_hex: str
    maximum_total_compression_ratio_hex: str
    stream_chunk_bytes: int
    archive_full_hash_passes: int
    current_verification_validity_seconds: int
    network_required: bool
    source_mutation_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        if self.policy_id != GLENN_ARCHIVE_VERIFICATION_POLICY_ID:
            raise ValueError("Glenn archive verification policy has another identity")
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetRegistrationManifest.SCHEMA
        ):
            raise ValueError("manifest must identify a DatasetRegistrationManifest")
        if (
            not isinstance(self.inspector_registration, ObjectIdentity)
            or self.inspector_registration.object_schema
            != DatasetFormatInspectorRegistration.SCHEMA
        ):
            raise ValueError(
                "inspector_registration must identify a DatasetFormatInspectorRegistration"
            )
        for field_name, text_value in (
            ("archive_inspection_profile_id", self.archive_inspection_profile_id),
            ("materialization_format_profile_id", self.materialization_format_profile_id),
        ):
            validate_stable_id(text_value, field_name=field_name)
        validate_sha256(self.expected_archive_sha256, field_name="expected_archive_sha256")
        for field_name, numeric_value in (
            ("expected_archive_size_bytes", self.expected_archive_size_bytes),
            ("maximum_archive_bytes", self.maximum_archive_bytes),
            ("maximum_members", self.maximum_members),
            ("maximum_central_directory_bytes", self.maximum_central_directory_bytes),
            ("maximum_member_uncompressed_bytes", self.maximum_member_uncompressed_bytes),
            ("maximum_total_uncompressed_bytes", self.maximum_total_uncompressed_bytes),
            ("maximum_selected_uncompressed_bytes", self.maximum_selected_uncompressed_bytes),
            ("stream_chunk_bytes", self.stream_chunk_bytes),
            ("archive_full_hash_passes", self.archive_full_hash_passes),
            (
                "current_verification_validity_seconds",
                self.current_verification_validity_seconds,
            ),
        ):
            _positive_integer(numeric_value, field_name=field_name)
        if self.expected_archive_size_bytes > self.maximum_archive_bytes:
            raise ValueError("expected archive exceeds its verification byte ceiling")
        if (
            not isinstance(self.expected_members, tuple)
            or not self.expected_members
            or any(
                not isinstance(value, DatasetSelectorMemberContract)
                for value in self.expected_members
            )
        ):
            raise ValueError("expected_members must be exact selector member contracts")
        require_sorted_unique_ids(
            self.expected_members,
            attribute="member_id",
            field_name="expected_members",
        )
        if len(self.expected_members) > self.maximum_members:
            raise ValueError("expected safe members exceed the archive member ceiling")
        for field_name, hex_value in (
            (
                "maximum_member_compression_ratio_hex",
                self.maximum_member_compression_ratio_hex,
            ),
            (
                "maximum_total_compression_ratio_hex",
                self.maximum_total_compression_ratio_hex,
            ),
        ):
            try:
                decoded = float.fromhex(hex_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field_name} must be a canonical positive float hex") from error
            if decoded <= 0.0 or decoded.hex() != hex_value:
                raise ValueError(f"{field_name} must be a canonical positive float hex")
        if self.archive_full_hash_passes != GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT:
            raise ValueError("Glenn verification must retain its exact full-hash pass count")
        if self.current_verification_validity_seconds != (_CURRENT_VERIFICATION_VALIDITY_SECONDS):
            raise ValueError("Glenn current verification has another validity interval")
        if self.network_required is not False or self.source_mutation_allowed is not False:
            raise ValueError("held Glenn verification must be zero-network and read-only")


def build_glenn_archive_verification_policy(
    *,
    manifest: DatasetRegistrationManifest,
    inspector_registration: DatasetFormatInspectorRegistration,
    profile: GlennTransformProfile,
) -> GlennArchiveVerificationPolicy:
    """Bind the manifest, registered implementation, exact members and limits."""

    if len(manifest.expected_format_profile_ids) != 1:
        raise ValueError("Glenn registration requires one materialization format profile")
    limits = profile.limits
    return GlennArchiveVerificationPolicy(
        policy_id=GLENN_ARCHIVE_VERIFICATION_POLICY_ID,
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        inspector_registration=ObjectIdentity.from_record(
            inspector_registration.capability_key,
            inspector_registration,
        ),
        archive_inspection_profile_id=profile.profile_id,
        materialization_format_profile_id=manifest.expected_format_profile_ids[0],
        expected_archive_sha256=profile.expected_archive_sha256,
        expected_archive_size_bytes=profile.expected_archive_size_bytes,
        expected_members=_expected_members(profile),
        maximum_archive_bytes=limits.maximum_archive_bytes,
        maximum_members=limits.maximum_members,
        maximum_central_directory_bytes=limits.maximum_central_directory_bytes,
        maximum_member_uncompressed_bytes=limits.maximum_member_uncompressed_bytes,
        maximum_total_uncompressed_bytes=limits.maximum_total_uncompressed_bytes,
        maximum_selected_uncompressed_bytes=limits.maximum_selected_uncompressed_bytes,
        maximum_member_compression_ratio_hex=limits.maximum_member_compression_ratio.hex(),
        maximum_total_compression_ratio_hex=limits.maximum_total_compression_ratio.hex(),
        stream_chunk_bytes=limits.stream_chunk_bytes,
        archive_full_hash_passes=GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT,
        current_verification_validity_seconds=_CURRENT_VERIFICATION_VALIDITY_SECONDS,
        network_required=False,
        source_mutation_allowed=False,
    )


def _member_receipt_matches_contract(
    receipt: GlennMemberReceipt,
    contract: DatasetSelectorMemberContract,
) -> bool:
    return (
        receipt.member_id == contract.member_id
        and receipt.relative_locator == contract.relative_locator
        and receipt.physical_sha256 == contract.expected_physical_sha256
        and receipt.compressed_size_bytes == contract.expected_compressed_size_bytes
        and receipt.uncompressed_size_bytes == contract.expected_uncompressed_size_bytes
        and receipt.crc32 == contract.expected_crc32
        and receipt.media_type == contract.media_type
        and receipt.format_profile_id == contract.format_profile_id
        and receipt.inspection_policy_id == contract.inspection_policy_id
    )


@dataclass(frozen=True, slots=True)
class GlennArchiveRegistrationInspection(CanonicalRecord):
    """Canonical non-writing result of one exact in-place archive inspection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-archive-registration-inspection'

    inspection_id: str
    manifest: ObjectIdentity
    inspector: RegisteredImplementationIdentity
    policy: GlennArchiveVerificationPolicy
    source_guard_receipt: DatasetSourceGuardReceipt
    archive_audit: GlennArchiveAudit
    subject: DatasetMaterializationVerificationSubject
    observations: DatasetMaterializationVerificationObservations
    distinct_source_bytes: int
    archive_full_hash_passes: int
    archive_full_hash_read_ceiling_bytes: int
    network_bytes: int
    source_mutated: bool
    evidence_persisted: bool
    catalog_written: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        if self.inspection_id != GLENN_ARCHIVE_REGISTRATION_INSPECTION_ID:
            raise ValueError("Glenn registration inspection has another identity")
        if not isinstance(self.manifest, ObjectIdentity) or self.manifest != self.policy.manifest:
            raise ValueError("inspection manifest differs from its verification policy")
        if not isinstance(self.inspector, RegisteredImplementationIdentity):
            raise ValueError("inspector must be a RegisteredImplementationIdentity")
        if not isinstance(self.source_guard_receipt, DatasetSourceGuardReceipt):
            raise ValueError("source_guard_receipt must be completed guard evidence")
        if not isinstance(self.archive_audit, GlennArchiveAudit):
            raise ValueError("archive_audit must be a GlennArchiveAudit")
        if not isinstance(self.subject, DatasetMaterializationVerificationSubject):
            raise ValueError("subject must be a verification subject")
        if not isinstance(self.observations, DatasetMaterializationVerificationObservations):
            raise ValueError("observations must be materialization verification observations")
        if not self.observations.validates(self.subject):
            raise ValueError("inspection observations differ from the proof-free subject")
        guard = self.source_guard_receipt
        if (
            guard.guard_receipt_id != self.archive_audit.guard_evidence_id
            or guard.relative_locator != self.archive_audit.locator
            or guard.observed_size_bytes != self.archive_audit.archive_size_bytes
            or guard.observed_sha256 != self.archive_audit.archive_sha256
            or guard.observed_size_bytes != self.policy.expected_archive_size_bytes
            or guard.observed_sha256 != self.policy.expected_archive_sha256
        ):
            raise ValueError("source guard, archive audit and policy identities differ")
        if self.archive_audit.profile_id != self.policy.archive_inspection_profile_id:
            raise ValueError("archive audit used another inspection profile")
        if len(self.archive_audit.selected_members) != len(self.policy.expected_members) or any(
            not _member_receipt_matches_contract(receipt, contract)
            for receipt, contract in zip(
                self.archive_audit.selected_members,
                self.policy.expected_members,
                strict=True,
            )
        ):
            raise ValueError("archive audit differs from the exact safe-member policy")
        if (
            self.subject.storage_root_id != guard.storage_root.object_id
            or self.subject.relative_locator != guard.relative_locator
            or self.subject.physical_sha256 != guard.observed_sha256
            or self.subject.byte_size != guard.observed_size_bytes
            or self.subject.file_count != 1
            or self.subject.format_profile_id != self.policy.materialization_format_profile_id
        ):
            raise ValueError("verification subject differs from the guarded archive")
        if self.distinct_source_bytes != guard.observed_size_bytes:
            raise ValueError("distinct source-byte accounting differs from the archive")
        if self.archive_full_hash_passes != self.policy.archive_full_hash_passes:
            raise ValueError("inspection full-hash pass count differs from policy")
        if self.archive_full_hash_read_ceiling_bytes != (
            self.archive_full_hash_passes * self.distinct_source_bytes
        ):
            raise ValueError("inspection full-hash read ceiling is inconsistent")
        if self.network_bytes != 0:
            raise ValueError("held Glenn inspection must use zero network bytes")
        for field_name in ("source_mutated", "evidence_persisted", "catalog_written"):
            if getattr(self, field_name) is not False:
                raise ValueError(f"non-writing Glenn inspection requires {field_name}=false")

    def unverified_materialization(self) -> DatasetMaterialization:
        """Return the observed subject without manufacturing publication proof."""

        subject = self.subject
        return DatasetMaterialization(
            materialization_id=subject.materialization_id,
            release_id=subject.release_id,
            materialization_class=subject.materialization_class,
            evidence_class=subject.evidence_class,
            outcome_access=subject.outcome_access,
            storage_root_id=subject.storage_root_id,
            relative_locator=subject.relative_locator,
            selector=subject.selector,
            physical_sha256=subject.physical_sha256,
            byte_size=subject.byte_size,
            file_count=subject.file_count,
            media_type=subject.media_type,
            format_profile_id=subject.format_profile_id,
            logical_identity=subject.logical_identity,
            custody_state=CustodyState.UNVERIFIED,
            verifier=None,
            verification_policy_id=None,
            verification_policy_sha256=None,
            verified_at_utc=None,
            verification_evidence_refs=(),
            manifest_relative_locator=None,
            receipt_relative_locator=None,
            reason_codes=(),
        )


class GlennArchiveRegistrationInspector:
    """Process-local inspector/current-verifier bound to one exact manifest."""

    capability_key: str
    capability_version: str
    implementation_sha256: str
    capability_manifest_fingerprint: str
    registration_fingerprint: str

    def __init__(
        self,
        *,
        registration: DatasetFormatInspectorRegistration,
        manifest: DatasetRegistrationManifest,
        profile: GlennTransformProfile,
        source_opener: DatasetSourceOpener,
        trusted_at_utc: str,
    ) -> None:
        parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")
        if registration.capability_key != GLENN_INSPECTOR_REGISTRY_KEY:
            raise ValueError("registration is not the exact Glenn inspector")
        if registration.accepted_media_types != (manifest.expected_media_type,):
            raise ValueError("Glenn inspector media type differs from the manifest")
        if not set(manifest.expected_format_profile_ids).issubset(
            registration.supported_format_profile_ids
        ):
            raise ValueError("Glenn inspector does not support the manifest format profile")
        if (
            profile.expected_archive_size_bytes != manifest.expected_byte_size
            or profile.expected_archive_sha256 != manifest.expected_physical_sha256
            or registration.limits.max_input_bytes < manifest.expected_byte_size
            or registration.limits.max_files < manifest.expected_file_count
            or registration.limits.max_archive_members < profile.limits.maximum_members
        ):
            raise ValueError("Glenn inspector limits or archive identity differ from the manifest")
        inspector_bindings = tuple(
            value
            for value in manifest.capabilities
            if value.kind is DatasetCapabilityKind.INSPECTOR
        )
        expected_capability_identity = ObjectIdentity.from_record(
            registration.capability_key,
            registration.capability,
        )
        if (
            len(inspector_bindings) != 1
            or inspector_bindings[0].implementation != expected_capability_identity
        ):
            raise ValueError("manifest does not bind this exact Glenn inspector registration")
        self._registration = registration
        self._manifest = manifest
        self._manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        self._profile = profile
        self._source_opener = source_opener
        self._trusted_at_utc = trusted_at_utc
        self._policy = build_glenn_archive_verification_policy(
            manifest=manifest,
            inspector_registration=registration,
            profile=profile,
        )
        self._implementation_identity = RegisteredImplementationIdentity(
            implementation_key=registration.capability_key,
            implementation_version=registration.capability_version,
            implementation_sha256=registration.implementation_sha256,
        )
        self.capability_key = registration.capability_key
        self.capability_version = registration.capability_version
        self.implementation_sha256 = registration.implementation_sha256
        self.capability_manifest_fingerprint = registration.capability.fingerprint()
        self.registration_fingerprint = registration.fingerprint()

    @property
    def policy(self) -> GlennArchiveVerificationPolicy:
        return self._policy

    @property
    def implementation_identity(self) -> RegisteredImplementationIdentity:
        return self._implementation_identity

    def inspect_registration(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str | None = None,
    ) -> GlennArchiveRegistrationInspection:
        """Inspect only the manifest-bound source and return no publication claim."""

        if request != self._manifest_identity:
            raise ValueError("Glenn inspector request differs from its exact manifest")
        verified_at_utc = self._trusted_at_utc if trusted_at_utc is None else trusted_at_utc
        parse_utc_timestamp(verified_at_utc, field_name="trusted_at_utc")
        manifest = self._manifest
        with self._source_opener.open_exact(
            source_scope=manifest.source_scope,
            expected_size_bytes=manifest.expected_byte_size,
            expected_sha256=manifest.expected_physical_sha256,
            maximum_bytes=manifest.work_envelope.resources.source_scan_bytes,
            trusted_at_utc=verified_at_utc,
        ) as guarded:
            if not isinstance(guarded, GuardedDatasetSource):
                raise GlennAdapterError("source opener returned another guarded-source type")
            decoded = inspect_glenn_archive(
                GlennArchiveInput(
                    stream=guarded.stream,
                    locator=manifest.source_scope.relative_prefix,
                    guard_evidence_id=guarded.guard_receipt_id,
                ),
                profile=self._profile,
            )
            archive_audit = decoded.audit

        source_guard_receipt = guarded.guard_receipt
        subject = DatasetMaterializationVerificationSubject(
            materialization_id=manifest.proposed_materialization_id,
            release_id=manifest.release.release_id,
            materialization_class=manifest.materialization_class,
            evidence_class=manifest.evidence_class,
            outcome_access=manifest.outcome_access,
            storage_root_id=manifest.source_scope.storage_root.object_id,
            relative_locator=manifest.source_scope.relative_prefix,
            selector=manifest.selector,
            physical_sha256=source_guard_receipt.observed_sha256,
            byte_size=source_guard_receipt.observed_size_bytes,
            file_count=manifest.expected_file_count,
            media_type=manifest.expected_media_type,
            format_profile_id=manifest.expected_format_profile_ids[0],
            logical_identity=None,
        )
        observations = DatasetMaterializationVerificationObservations(
            observed_storage_root_id=subject.storage_root_id,
            observed_relative_locator=subject.relative_locator,
            observed_selector=subject.selector,
            observed_physical_sha256=source_guard_receipt.observed_sha256,
            observed_byte_size=source_guard_receipt.observed_size_bytes,
            observed_file_count=manifest.expected_file_count,
            observed_media_type=subject.media_type,
            observed_format_profile_id=subject.format_profile_id,
            observed_logical_identity=None,
            complete_eof=source_guard_receipt.complete_eof,
            trusted_root=True,
            locator_contained=True,
            read_only=source_guard_receipt.opened_read_only,
        )
        return GlennArchiveRegistrationInspection(
            inspection_id=GLENN_ARCHIVE_REGISTRATION_INSPECTION_ID,
            manifest=self._manifest_identity,
            inspector=self._implementation_identity,
            policy=self._policy,
            source_guard_receipt=source_guard_receipt,
            archive_audit=archive_audit,
            subject=subject,
            observations=observations,
            distinct_source_bytes=manifest.expected_byte_size,
            archive_full_hash_passes=GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT,
            archive_full_hash_read_ceiling_bytes=(
                GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT * manifest.expected_byte_size
            ),
            network_bytes=0,
            source_mutated=False,
            evidence_persisted=False,
            catalog_written=False,
        )

    def inspect(self, request: ObjectIdentity) -> DatasetMaterialization:
        """Satisfy the generic inspector port without manufacturing verification proof."""

        return self.inspect_registration(request).unverified_materialization()

    def prepare_registration(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str,
    ) -> PreparedDatasetRegistration:
        """Adapt the Glenn audit to the generic authorized publisher boundary."""

        inspection = self.inspect_registration(
            request,
            trusted_at_utc=trusted_at_utc,
        )
        audit_id = f"audit.glenn-archive.{inspection.archive_audit.fingerprint()[:24]}"
        return PreparedDatasetRegistration(
            inspection_id=inspection.inspection_id,
            inspection_record=inspection,
            manifest=inspection.manifest,
            inspector=inspection.inspector,
            verification_policy_id=inspection.policy.policy_id,
            verification_policy_record=inspection.policy,
            source_guard_receipts=(inspection.source_guard_receipt,),
            supporting_records=(
                DatasetRegistrationSupportingRecord(
                    record_id=audit_id,
                    record=inspection.archive_audit,
                ),
            ),
            subject=inspection.subject,
            observations=inspection.observations,
            distinct_source_bytes=inspection.distinct_source_bytes,
            source_full_hash_passes=inspection.archive_full_hash_passes,
            source_full_hash_read_ceiling_bytes=(inspection.archive_full_hash_read_ceiling_bytes),
            network_bytes=inspection.network_bytes,
            source_mutated=inspection.source_mutated,
        )

    def revalidate(
        self,
        materialization: DatasetMaterialization,
        *,
        trusted_at_utc: str,
    ) -> CurrentMaterializationVerification:
        """Re-read the exact archive before a no-download acquisition decision."""

        if (
            not isinstance(materialization, DatasetMaterialization)
            or materialization.custody_state is not CustodyState.VERIFIED
            or materialization.verifier != self._implementation_identity
            or materialization.verification_policy_id != self._policy.policy_id
            or materialization.verification_policy_sha256 != self._policy.fingerprint()
        ):
            raise ValueError("materialization is not verified by this exact Glenn inspector")
        inspection = self.inspect_registration(
            self._manifest_identity,
            trusted_at_utc=trusted_at_utc,
        )
        if inspection.subject != materialization.verification_subject():
            raise ValueError("current Glenn inspection differs from the registered subject")
        evidence = tuple(
            value
            for value in materialization.verification_evidence_refs
            if value.kind is EvidenceReferenceKind.VERIFICATION
        )
        if len(evidence) != 1:
            raise ValueError("verified Glenn materialization lacks one exact receipt reference")
        verified_at = parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")
        valid_until = verified_at + timedelta(
            seconds=self._policy.current_verification_validity_seconds
        )
        valid_until_utc = valid_until.isoformat(timespec="seconds").replace("+00:00", "Z")
        verification_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "materialization_fingerprint": materialization.fingerprint(),
                    "policy_fingerprint": self._policy.fingerprint(),
                    "trusted_at_utc": trusted_at_utc,
                }
            )
        ).hexdigest()
        return CurrentMaterializationVerification(
            verification_id=f"verification.glenn-current.{verification_digest[:24]}",
            materialization_id=materialization.materialization_id,
            materialization_fingerprint=materialization.fingerprint(),
            observed_physical_sha256=inspection.archive_audit.archive_sha256,
            verifier=self._implementation_identity,
            verification_policy_id=self._policy.policy_id,
            verification_policy_sha256=self._policy.fingerprint(),
            verified_at_utc=trusted_at_utc,
            valid_until_utc=valid_until_utc,
            evidence=evidence[0],
            accessible=True,
        )


def bind_glenn_archive_registration_inspector(
    *,
    registration: DatasetFormatInspectorRegistration,
    manifest: DatasetRegistrationManifest,
    profile: GlennTransformProfile,
    source_opener: DatasetSourceOpener,
    trusted_at_utc: str,
) -> DatasetInspectorImplementationBinding:
    """Construct the process-local exact implementation binding."""

    inspector = GlennArchiveRegistrationInspector(
        registration=registration,
        manifest=manifest,
        profile=profile,
        source_opener=source_opener,
        trusted_at_utc=trusted_at_utc,
    )
    return DatasetInspectorImplementationBinding(
        registry_id=registration.registry_id,
        registration_fingerprint=registration.fingerprint(),
        inspector=inspector,
        current_verifier=inspector,
    )


__all__ = [
    "GLENN_ARCHIVE_REGISTRATION_INSPECTION_ID",
    "GLENN_ARCHIVE_VERIFICATION_POLICY_ID",
    "GlennArchiveRegistrationInspection",
    "GlennArchiveRegistrationInspector",
    "GlennArchiveVerificationPolicy",
    "bind_glenn_archive_registration_inspector",
    "build_glenn_archive_verification_policy",
]
