"""Registered, fail-closed dataset no-redownload decision service.

Pure planning can identify exact local custody that is eligible for fresh
verification, but it cannot authenticate a caller-supplied verification
record.  This module is the composition boundary that resolves the exact
registered implementation, obtains trusted time, replays subject-bound
external evidence, and only then emits ``ALREADY_PRESENT``.
"""

from __future__ import annotations

from typing import Final

from empirical_lawhood.kernel.serialization import validate_nonempty, validate_stable_id
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import DatasetDecisionClock
from empirical_lawhood.planning.dataset_manifests import (
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
)
from empirical_lawhood.planning.datasets import (
    AcquisitionDecision,
    AcquisitionDisposition,
    CurrentMaterializationVerification,
    DatasetMaterialization,
    DatasetMaterializationVerificationReceipt,
    DatasetRelease,
    DatasetSelectorRef,
    EvidenceReference,
    EvidenceReferenceKind,
    LogicalContentIdentity,
    RegisteredImplementationIdentity,
    acquisition_revalidation_candidates,
    decide_acquisition,
)

from .datasets import (
    DatasetEvidenceOwnerKind,
    DatasetEvidenceVerification,
    DatasetEvidenceVerificationRequest,
    DatasetEvidenceVerifier,
    DatasetEvidenceVerifierRegistration,
    DatasetFormatInspectorRegistration,
    DatasetImplementationRegistry,
)


MAX_CURRENT_VERIFICATION_EVIDENCE_BYTES: Final[int] = 64 * 1024**2


class RegisteredAcquisitionDecisionError(ValueError):
    """A registered verifier or its evidence broke the trusted contract."""


def _registered_verifier_identity(
    registration: DatasetFormatInspectorRegistration,
) -> RegisteredImplementationIdentity:
    return RegisteredImplementationIdentity(
        implementation_key=registration.capability_key,
        implementation_version=registration.capability_version,
        implementation_sha256=registration.implementation_sha256,
    )


def _exact_manifest_evidence(
    materialization: DatasetMaterialization,
) -> EvidenceReference:
    matches = tuple(
        reference
        for reference in materialization.verification_evidence_refs
        if reference.kind is EvidenceReferenceKind.MANIFEST
    )
    if len(matches) != 1:
        raise RegisteredAcquisitionDecisionError(
            "verified materialization lacks one exact manifest evidence reference"
        )
    return matches[0]


def _require_registration_accepts(
    registration: DatasetFormatInspectorRegistration,
    materialization: DatasetMaterialization,
) -> None:
    if (
        materialization.media_type is None
        or materialization.media_type not in registration.accepted_media_types
    ):
        raise RegisteredAcquisitionDecisionError(
            "registered current verifier does not accept the materialization media type"
        )
    if (
        materialization.format_profile_id is None
        or materialization.format_profile_id not in registration.supported_format_profile_ids
    ):
        raise RegisteredAcquisitionDecisionError(
            "registered current verifier does not support the materialization format profile"
        )
    if (
        materialization.byte_size is None
        or materialization.byte_size > registration.limits.max_input_bytes
    ):
        raise RegisteredAcquisitionDecisionError(
            "materialization exceeds the registered current-verifier byte limit"
        )
    if (
        materialization.file_count is None
        or materialization.file_count > registration.limits.max_files
    ):
        raise RegisteredAcquisitionDecisionError(
            "materialization exceeds the registered current-verifier file limit"
        )


def _verify_current_evidence(
    *,
    materialization: DatasetMaterialization,
    verification_evidence: EvidenceReference,
    expected_verifier: RegisteredImplementationIdentity,
    expected_verified_at_utc: str,
    expected_verification_policy_id: str,
    expected_verification_policy_sha256: str,
    inspector_registration: DatasetFormatInspectorRegistration,
    evidence_registration: DatasetEvidenceVerifierRegistration,
    evidence_verifier: DatasetEvidenceVerifier,
) -> None:
    if (
        verification_evidence.kind is not EvidenceReferenceKind.VERIFICATION
        or verification_evidence.evidence_schema != DatasetMaterializationVerificationReceipt.SCHEMA
    ):
        raise RegisteredAcquisitionDecisionError(
            "current verification requires one typed materialization receipt"
        )
    if verification_evidence.evidence_schema not in (
        evidence_registration.supported_evidence_schema_ids
    ):
        raise RegisteredAcquisitionDecisionError(
            "registered evidence verifier does not support the receipt schema"
        )
    manifest_evidence = _exact_manifest_evidence(materialization)
    if manifest_evidence.evidence_schema not in {
        DatasetRegistrationManifest.SCHEMA,
        DatasetTransformationManifest.SCHEMA,
    }:
        raise RegisteredAcquisitionDecisionError(
            "current verification requires a registration or transformation manifest"
        )
    if manifest_evidence.evidence_schema not in (
        evidence_registration.supported_evidence_schema_ids
    ):
        raise RegisteredAcquisitionDecisionError(
            "registered evidence verifier does not support the materialization manifest"
        )
    if evidence_registration.limits.max_files < 2:
        raise RegisteredAcquisitionDecisionError(
            "registered evidence verifier admits fewer than two evidence files"
        )
    if evidence_registration.limits.max_records < 2:
        raise RegisteredAcquisitionDecisionError(
            "registered evidence verifier admits fewer than two evidence records"
        )
    maximum_bytes = min(
        MAX_CURRENT_VERIFICATION_EVIDENCE_BYTES,
        inspector_registration.limits.max_output_bytes,
        evidence_registration.limits.max_input_bytes,
    )
    if maximum_bytes <= 0:
        raise RegisteredAcquisitionDecisionError(
            "registered current verifier admits no bounded evidence output"
        )
    remaining = maximum_bytes
    remaining_output_bytes = evidence_registration.limits.max_output_bytes
    for reference in (manifest_evidence, verification_evidence):
        if remaining <= 0:
            raise RegisteredAcquisitionDecisionError(
                "current-verification evidence exceeds its aggregate byte envelope"
            )
        request = DatasetEvidenceVerificationRequest(
            owner_kind=DatasetEvidenceOwnerKind.MATERIALIZATION,
            owner_id=materialization.materialization_id,
            reference=reference,
            materialization_subject=materialization.verification_subject(),
            expected_verifier=expected_verifier,
            expected_verification_policy_id=expected_verification_policy_id,
            expected_verification_policy_sha256=expected_verification_policy_sha256,
            expected_verified_at_utc=expected_verified_at_utc,
            expected_manifest_evidence=manifest_evidence,
        )
        try:
            result = evidence_verifier.verify(request, maximum_bytes=remaining)
        except Exception as error:
            raise RegisteredAcquisitionDecisionError(
                "current-verification evidence replay failed"
            ) from error
        if (
            not isinstance(result, DatasetEvidenceVerification)
            or result.observed_size_bytes > remaining
            or not result.validates(request)
        ):
            raise RegisteredAcquisitionDecisionError(
                "current-verification evidence does not bind the exact subject"
            )
        result_size_bytes = len(result.canonical_bytes())
        if result_size_bytes > remaining_output_bytes:
            raise RegisteredAcquisitionDecisionError(
                "current-verification evidence exceeds its aggregate output byte envelope"
            )
        remaining -= result.observed_size_bytes
        remaining_output_bytes -= result_size_bytes


def decide_registered_acquisition(
    requested_release: DatasetRelease,
    materializations: tuple[DatasetMaterialization, ...],
    *,
    inspector_registry_id: str,
    evidence_verifier_registry_id: str,
    implementation_registry: DatasetImplementationRegistry,
    clock: DatasetDecisionClock,
    required_selector: DatasetSelectorRef | None = None,
    expected_physical_sha256: str | None = None,
    expected_logical_identity: LogicalContentIdentity | None = None,
    required_format_profile_id: str | None = None,
) -> AcquisitionDecision:
    """Return ``ALREADY_PRESENT`` only after registered current verification.

    The clock and executable registries are application-composed dependencies,
    never values decoded from a caller request.  Invalid registered output is a
    contract error; stale or inaccessible output remains the pure
    ``VERIFY_EXISTING`` decision and cannot authorize a no-download result.
    """

    if not isinstance(implementation_registry, DatasetImplementationRegistry):
        raise TypeError("implementation_registry must be a DatasetImplementationRegistry")
    validate_nonempty(inspector_registry_id, field_name="inspector_registry_id")
    validate_nonempty(
        evidence_verifier_registry_id,
        field_name="evidence_verifier_registry_id",
    )
    clock_id = clock.clock_id
    validate_stable_id(clock_id, field_name="decision_clock_id")
    trusted_at_utc = clock.now_utc()
    parse_utc_timestamp(trusted_at_utc, field_name="trusted_at_utc")

    base_decision = decide_acquisition(
        requested_release,
        materializations,
        required_selector=required_selector,
        expected_physical_sha256=expected_physical_sha256,
        expected_logical_identity=expected_logical_identity,
        required_format_profile_id=required_format_profile_id,
    )
    if base_decision.disposition is not AcquisitionDisposition.VERIFY_EXISTING:
        return base_decision

    candidates = acquisition_revalidation_candidates(
        requested_release,
        materializations,
        required_selector=required_selector,
        expected_physical_sha256=expected_physical_sha256,
        expected_logical_identity=expected_logical_identity,
        required_format_profile_id=required_format_profile_id,
    )
    if not candidates:
        return base_decision

    registration = implementation_registry.static_registry.inspector(inspector_registry_id)
    _, current_verifier = implementation_registry.inspector(inspector_registry_id)
    evidence_registration = implementation_registry.static_registry.evidence_verifier(
        evidence_verifier_registry_id
    )
    evidence_verifier = implementation_registry.evidence_verifier(evidence_verifier_registry_id)
    expected_verifier = _registered_verifier_identity(registration)

    for materialization in candidates:
        _require_registration_accepts(registration, materialization)
        expected_policy_id = materialization.verification_policy_id
        expected_policy_sha256 = materialization.verification_policy_sha256
        if expected_policy_id is None or expected_policy_sha256 is None:
            raise RegisteredAcquisitionDecisionError(
                "verified materialization lacks its exact verification policy"
            )
        try:
            verification = current_verifier.revalidate(
                materialization,
                trusted_at_utc=trusted_at_utc,
            )
        except Exception as error:
            raise RegisteredAcquisitionDecisionError(
                "registered current verifier failed"
            ) from error
        if not isinstance(verification, CurrentMaterializationVerification):
            raise RegisteredAcquisitionDecisionError(
                "registered current verifier returned an invalid record type"
            )
        if not verification.validates(materialization):
            raise RegisteredAcquisitionDecisionError(
                "registered current verification names a different materialization"
            )
        if verification.verifier != expected_verifier:
            raise RegisteredAcquisitionDecisionError(
                "registered current-verifier identity differs from its registration"
            )
        if (
            verification.verification_policy_id != expected_policy_id
            or verification.verification_policy_sha256 != expected_policy_sha256
        ):
            raise RegisteredAcquisitionDecisionError(
                "current-verification policy differs from the exact registration"
            )
        if not verification.is_current_at(trusted_at_utc):
            continue
        _verify_current_evidence(
            materialization=materialization,
            verification_evidence=verification.evidence,
            expected_verifier=expected_verifier,
            expected_verified_at_utc=verification.verified_at_utc,
            expected_verification_policy_id=expected_policy_id,
            expected_verification_policy_sha256=expected_policy_sha256,
            inspector_registration=registration,
            evidence_registration=evidence_registration,
            evidence_verifier=evidence_verifier,
        )
        return AcquisitionDecision(
            requested_release_id=requested_release.release_id,
            disposition=AcquisitionDisposition.ALREADY_PRESENT,
            selected_materialization_id=materialization.materialization_id,
            considered_materialization_ids=base_decision.considered_materialization_ids,
            reason_codes=("EXACT_REGISTERED_VERIFIED_MATERIALIZATION_PRESENT",),
        )
    return base_decision


__all__ = [
    "MAX_CURRENT_VERIFICATION_EVIDENCE_BYTES",
    "RegisteredAcquisitionDecisionError",
    "decide_registered_acquisition",
]
