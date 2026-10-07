"""Registered held-directory inspection for simulator software and generations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.kernel.serialization import validate_sha256, validate_stable_id
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityKind,
    DatasetDirectorySelectorManifest,
    DatasetRegistrationManifest,
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
    DatasetDirectoryGuardReceipt,
    DatasetDirectorySourceInspector,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationSupportingRecord,
    PreparedDatasetRegistration,
)
from empirical_lawhood.runtime.datasets import (
    DatasetFormatInspectorRegistration,
    DatasetInspectorImplementationBinding,
)

from .registry import (
    SIMULATOR_DATASET_INSPECTOR_KEY,
    SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,
    SIMULATOR_DIRECTORY_MEDIA_TYPE,
    SIMULATOR_MAX_SINGLE_FILE_BYTES,
)
from .lineage import SimulatorGenerationLineageManifest
from .manifests import build_simulator_directory_output_contract


@dataclass(frozen=True, slots=True)
class SimulatorDirectoryVerificationPolicy(CanonicalRecord):
    """Exact selector-bound verification policy for one held directory."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-directory-verification-policy'

    policy_id: str
    manifest: ObjectIdentity
    selector: ObjectIdentity
    generation_lineage: ObjectIdentity | None
    expected_content_sha256: str
    expected_total_size_bytes: int
    expected_file_count: int
    current_verification_validity_seconds: int
    source_full_hash_passes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != DatasetRegistrationManifest.SCHEMA
        ):
            raise ValueError("manifest must identify a registration manifest")
        if (
            not isinstance(self.selector, ObjectIdentity)
            or self.selector.object_schema != DatasetDirectorySelectorManifest.SCHEMA
        ):
            raise ValueError("selector must identify a directory selector")
        if self.generation_lineage is not None and (
            not isinstance(self.generation_lineage, ObjectIdentity)
            or self.generation_lineage.object_schema != SimulatorGenerationLineageManifest.SCHEMA
        ):
            raise ValueError("generation_lineage must identify simulator lineage")
        validate_sha256(
            self.expected_content_sha256,
            field_name="expected_content_sha256",
        )
        for field_name, value in (
            ("expected_total_size_bytes", self.expected_total_size_bytes),
            ("expected_file_count", self.expected_file_count),
            (
                "current_verification_validity_seconds",
                self.current_verification_validity_seconds,
            ),
            ("source_full_hash_passes", self.source_full_hash_passes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.source_full_hash_passes != 1:
            raise ValueError("directory registration performs exactly one full hash pass")


@dataclass(frozen=True, slots=True)
class SimulatorDirectoryRegistrationInspection(CanonicalRecord):
    """Non-writing result of exact allowlisted directory inspection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-directory-registration-inspection'

    inspection_id: str
    manifest: ObjectIdentity
    inspector: RegisteredImplementationIdentity
    policy: SimulatorDirectoryVerificationPolicy
    selector: DatasetDirectorySelectorManifest
    directory_guard: DatasetDirectoryGuardReceipt
    subject: DatasetMaterializationVerificationSubject
    observations: DatasetMaterializationVerificationObservations
    network_bytes: int
    source_mutated: bool
    evidence_persisted: bool
    catalog_written: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        if not isinstance(self.policy, SimulatorDirectoryVerificationPolicy):
            raise TypeError("policy must be a SimulatorDirectoryVerificationPolicy")
        if not isinstance(self.directory_guard, DatasetDirectoryGuardReceipt):
            raise TypeError("directory_guard must be a DatasetDirectoryGuardReceipt")
        if not isinstance(self.subject, DatasetMaterializationVerificationSubject):
            raise TypeError("subject must be a verification subject")
        if not isinstance(
            self.observations,
            DatasetMaterializationVerificationObservations,
        ):
            raise TypeError("observations must be verification observations")
        if self.manifest != self.policy.manifest:
            raise ValueError("inspection manifest differs from its policy")
        if not isinstance(self.inspector, RegisteredImplementationIdentity):
            raise TypeError("inspector must be a RegisteredImplementationIdentity")
        if not isinstance(self.selector, DatasetDirectorySelectorManifest):
            raise TypeError("selector must be a DatasetDirectorySelectorManifest")
        if self.directory_guard.selector != ObjectIdentity.from_record(
            self.selector.selector_id,
            self.selector,
        ):
            raise ValueError("directory guard differs from the exact selector")
        if self.directory_guard.observed_content_sha256 != self.policy.expected_content_sha256:
            raise ValueError("directory guard content differs from policy")
        if self.directory_guard.observed_size_bytes != self.policy.expected_total_size_bytes:
            raise ValueError("directory guard bytes differ from policy")
        if self.directory_guard.observed_file_count != self.policy.expected_file_count:
            raise ValueError("directory guard file count differs from policy")
        if not self.observations.validates(self.subject):
            raise ValueError("inspection observations differ from their subject")
        if (
            self.subject.physical_sha256 != self.directory_guard.observed_content_sha256
            or self.subject.byte_size != self.directory_guard.observed_size_bytes
            or self.subject.file_count != self.directory_guard.observed_file_count
            or self.subject.storage_root_id != self.directory_guard.storage_root.object_id
            or self.subject.relative_locator != self.directory_guard.source_scope.relative_prefix
        ):
            raise ValueError("inspection subject differs from directory guard")
        if self.network_bytes != 0:
            raise ValueError("held directory inspection must use zero network bytes")
        for field_name in ("source_mutated", "evidence_persisted", "catalog_written"):
            if getattr(self, field_name) is not False:
                raise ValueError(f"non-writing inspection requires {field_name}=false")

    def unverified_materialization(self) -> DatasetMaterialization:
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


class SimulatorDirectoryRegistrationInspector:
    """Exact manifest/selector-bound inspector and current verifier."""

    def __init__(
        self,
        *,
        registration: DatasetFormatInspectorRegistration,
        manifest: DatasetRegistrationManifest,
        selector: DatasetDirectorySelectorManifest,
        generation_lineage: SimulatorGenerationLineageManifest | None = None,
        directory_inspector: DatasetDirectorySourceInspector,
        trusted_at_utc: str,
    ) -> None:
        parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")
        if registration.capability_key != SIMULATOR_DATASET_INSPECTOR_KEY:
            raise ValueError("registration is not the simulator directory inspector")
        if registration.accepted_media_types != (SIMULATOR_DIRECTORY_MEDIA_TYPE,):
            raise ValueError("directory inspector media type differs")
        if registration.supported_format_profile_ids != (SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,):
            raise ValueError("directory inspector format profile differs")
        selector_identity = ObjectIdentity.from_record(selector.selector_id, selector)
        lineage_identity = (
            ObjectIdentity.from_record(generation_lineage.lineage_id, generation_lineage)
            if generation_lineage is not None
            else None
        )
        expected_release_manifest_sha256 = (
            selector.fingerprint()
            if generation_lineage is None
            else generation_lineage.fingerprint()
        )
        if (
            manifest.selector.selector_id != selector.selector_id
            or manifest.selector.selector_schema != selector.SCHEMA
            or manifest.selector.selector_sha256 != selector.fingerprint()
            or manifest.selector.complete_release != selector.complete_subject
            or manifest.expected_physical_sha256 != selector.expected_content_sha256
            or manifest.expected_byte_size != selector.expected_total_size_bytes
            or manifest.expected_file_count != len(selector.members)
            or manifest.expected_media_type != SIMULATOR_DIRECTORY_MEDIA_TYPE
            or manifest.expected_format_profile_ids != (SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,)
            or manifest.release.expected_manifest_sha256 != expected_release_manifest_sha256
        ):
            raise ValueError("manifest differs from its exact directory selector")
        if generation_lineage is not None:
            if (
                generation_lineage.proposed_materialization_id
                != manifest.proposed_materialization_id
                or generation_lineage.outcome_access is not manifest.outcome_access
                or generation_lineage.visibility_ceiling is not manifest.visibility_ceiling
            ):
                raise ValueError("generation lineage differs from the registration manifest")
            generation_lineage.validate_selected_evidence(selector)
        if (
            registration.limits.max_input_bytes < manifest.expected_byte_size
            or registration.limits.max_files < manifest.expected_file_count
        ):
            raise ValueError("directory inspector limits are below the manifest")
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
            raise ValueError("manifest does not bind this exact directory inspector")
        self._registration = registration
        self._manifest = manifest
        self._manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        self._selector = selector
        self._selector_identity = selector_identity
        self._generation_lineage = generation_lineage
        self._generation_lineage_identity = lineage_identity
        self._directory_inspector = directory_inspector
        self._trusted_at_utc = trusted_at_utc
        policy_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "manifest": self._manifest_identity.object_fingerprint,
                    "selector": selector_identity.object_fingerprint,
                    "generation_lineage": (
                        None if lineage_identity is None else lineage_identity.object_fingerprint
                    ),
                }
            )
        ).hexdigest()
        self._policy = SimulatorDirectoryVerificationPolicy(
            policy_id=f"policy.simulator-directory.{policy_digest[:24]}",
            manifest=self._manifest_identity,
            selector=selector_identity,
            generation_lineage=lineage_identity,
            expected_content_sha256=selector.expected_content_sha256,
            expected_total_size_bytes=selector.expected_total_size_bytes,
            expected_file_count=len(selector.members),
            current_verification_validity_seconds=3600,
            source_full_hash_passes=1,
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
    def policy(self) -> SimulatorDirectoryVerificationPolicy:
        return self._policy

    def inspect_registration(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str | None = None,
    ) -> SimulatorDirectoryRegistrationInspection:
        if request != self._manifest_identity:
            raise ValueError("directory inspector request differs from its manifest")
        verified_at_utc = self._trusted_at_utc if trusted_at_utc is None else trusted_at_utc
        parse_utc_timestamp(verified_at_utc, field_name="trusted_at_utc")
        directory_guard = self._directory_inspector.inspect_exact(
            source_scope=self._manifest.source_scope,
            selector=self._selector,
            maximum_bytes=self._manifest.work_envelope.resources.source_scan_bytes,
            maximum_files=self._manifest.work_envelope.files.max_source_files,
            maximum_single_file_bytes=min(
                self._manifest.work_envelope.files.max_single_file_bytes,
                SIMULATOR_MAX_SINGLE_FILE_BYTES,
            ),
            trusted_at_utc=verified_at_utc,
        )
        subject = DatasetMaterializationVerificationSubject(
            materialization_id=self._manifest.proposed_materialization_id,
            release_id=self._manifest.release.release_id,
            materialization_class=self._manifest.materialization_class,
            evidence_class=self._manifest.evidence_class,
            outcome_access=self._manifest.outcome_access,
            storage_root_id=self._manifest.source_scope.storage_root.object_id,
            relative_locator=self._manifest.source_scope.relative_prefix,
            selector=self._manifest.selector,
            physical_sha256=directory_guard.observed_content_sha256,
            byte_size=directory_guard.observed_size_bytes,
            file_count=directory_guard.observed_file_count,
            media_type=self._manifest.expected_media_type,
            format_profile_id=self._manifest.expected_format_profile_ids[0],
            logical_identity=None,
        )
        observations = DatasetMaterializationVerificationObservations(
            observed_storage_root_id=subject.storage_root_id,
            observed_relative_locator=subject.relative_locator,
            observed_selector=subject.selector,
            observed_physical_sha256=subject.physical_sha256,
            observed_byte_size=subject.byte_size,
            observed_file_count=subject.file_count,
            observed_media_type=subject.media_type,
            observed_format_profile_id=subject.format_profile_id,
            observed_logical_identity=None,
            complete_eof=directory_guard.complete_eof,
            trusted_root=True,
            locator_contained=True,
            read_only=directory_guard.opened_read_only,
        )
        inspection_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "guard": directory_guard.fingerprint(),
                    "manifest": self._manifest_identity.object_fingerprint,
                }
            )
        ).hexdigest()
        return SimulatorDirectoryRegistrationInspection(
            inspection_id=f"inspection.simulator-directory.{inspection_digest[:24]}",
            manifest=self._manifest_identity,
            inspector=self._implementation_identity,
            policy=self._policy,
            selector=self._selector,
            directory_guard=directory_guard,
            subject=subject,
            observations=observations,
            network_bytes=0,
            source_mutated=False,
            evidence_persisted=False,
            catalog_written=False,
        )

    def inspect(self, request: ObjectIdentity) -> DatasetMaterialization:
        return self.inspect_registration(request).unverified_materialization()

    def prepare_registration(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str,
    ) -> PreparedDatasetRegistration:
        inspection = self.inspect_registration(request, trusted_at_utc=trusted_at_utc)
        output_contract = build_simulator_directory_output_contract(
            self._manifest,
            generation_lineage=self._generation_lineage,
        )
        return PreparedDatasetRegistration(
            inspection_id=inspection.inspection_id,
            inspection_record=inspection,
            manifest=inspection.manifest,
            inspector=inspection.inspector,
            verification_policy_id=inspection.policy.policy_id,
            verification_policy_record=inspection.policy,
            source_guard_receipts=(inspection.directory_guard,),
            supporting_records=tuple(
                sorted(
                    (
                        DatasetRegistrationSupportingRecord(
                            record_id=output_contract.output_id,
                            record=output_contract,
                        ),
                        DatasetRegistrationSupportingRecord(
                            record_id=inspection.selector.selector_id,
                            record=inspection.selector,
                        ),
                        *(
                            ()
                            if self._generation_lineage is None
                            else (
                                DatasetRegistrationSupportingRecord(
                                    record_id=self._generation_lineage.lineage_id,
                                    record=self._generation_lineage,
                                ),
                            )
                        ),
                    ),
                    key=lambda value: value.record_id,
                )
            ),
            subject=inspection.subject,
            observations=inspection.observations,
            distinct_source_bytes=inspection.directory_guard.observed_size_bytes,
            source_full_hash_passes=inspection.policy.source_full_hash_passes,
            source_full_hash_read_ceiling_bytes=(
                inspection.directory_guard.observed_size_bytes
                * inspection.policy.source_full_hash_passes
            ),
            network_bytes=0,
            source_mutated=False,
        )

    def revalidate(
        self,
        materialization: DatasetMaterialization,
        *,
        trusted_at_utc: str,
    ) -> CurrentMaterializationVerification:
        if (
            not isinstance(materialization, DatasetMaterialization)
            or materialization.custody_state is not CustodyState.VERIFIED
            or materialization.verifier != self._implementation_identity
            or materialization.verification_policy_id != self._policy.policy_id
            or materialization.verification_policy_sha256 != self._policy.fingerprint()
        ):
            raise ValueError("materialization is not verified by this directory inspector")
        inspection = self.inspect_registration(
            self._manifest_identity,
            trusted_at_utc=trusted_at_utc,
        )
        if inspection.subject != materialization.verification_subject():
            raise ValueError("current directory inspection differs from materialization")
        evidence = tuple(
            value
            for value in materialization.verification_evidence_refs
            if value.kind is EvidenceReferenceKind.VERIFICATION
        )
        if len(evidence) != 1:
            raise ValueError("verified directory materialization lacks exact evidence")
        verified_at = parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")
        valid_until = verified_at + timedelta(
            seconds=self._policy.current_verification_validity_seconds
        )
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "materialization": materialization.fingerprint(),
                    "policy": self._policy.fingerprint(),
                    "trusted_at_utc": trusted_at_utc,
                }
            )
        ).hexdigest()
        return CurrentMaterializationVerification(
            verification_id=f"verification.simulator-directory.{digest[:24]}",
            materialization_id=materialization.materialization_id,
            materialization_fingerprint=materialization.fingerprint(),
            observed_physical_sha256=inspection.subject.physical_sha256,
            verifier=self._implementation_identity,
            verification_policy_id=self._policy.policy_id,
            verification_policy_sha256=self._policy.fingerprint(),
            verified_at_utc=trusted_at_utc,
            valid_until_utc=valid_until.isoformat(timespec="seconds").replace(
                "+00:00",
                "Z",
            ),
            evidence=evidence[0],
            accessible=True,
        )


def bind_simulator_directory_registration_inspector(
    *,
    registration: DatasetFormatInspectorRegistration,
    manifest: DatasetRegistrationManifest,
    selector: DatasetDirectorySelectorManifest,
    generation_lineage: SimulatorGenerationLineageManifest | None = None,
    directory_inspector: DatasetDirectorySourceInspector,
    trusted_at_utc: str,
) -> DatasetInspectorImplementationBinding:
    inspector = SimulatorDirectoryRegistrationInspector(
        registration=registration,
        manifest=manifest,
        selector=selector,
        generation_lineage=generation_lineage,
        directory_inspector=directory_inspector,
        trusted_at_utc=trusted_at_utc,
    )
    return DatasetInspectorImplementationBinding(
        registry_id=registration.registry_id,
        registration_fingerprint=registration.fingerprint(),
        inspector=inspector,
        current_verifier=inspector,
    )


__all__ = [
    "SimulatorDirectoryRegistrationInspection",
    "SimulatorDirectoryRegistrationInspector",
    "SimulatorDirectoryVerificationPolicy",
    "bind_simulator_directory_registration_inspector",
]
