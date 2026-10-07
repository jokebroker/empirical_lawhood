"""Canonical architecture-only release boundary for backbone extensions F0--F4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.artifacts import (
    ArtifactMaterialization,
    ArtifactPublicationBinding,
    LogicalArtifactIdentity,
)
from empirical_lawhood.runtime.extension_bundles import GeneratedExtensionBundleAggregate
from empirical_lawhood.runtime.route_conformance import RunEnvelopeRouteConformanceReceipt, RunEnvelopeOperationalConformanceReceipt
from empirical_lawhood.runtime.response_law_release import ConsolidationClaimCeiling, ReleaseConformanceAttestation, ReleaseRepositoryStateReceipt, ReleaseVerificationReceipt, ResponseLawConsolidationRelease, MultiWorldReadinessConsumerPin


class BackboneConformanceCaseKind(StrEnum):
    SUPPORTED_PUBLIC_ROUTE = "SUPPORTED_PUBLIC_ROUTE"
    OBSTRUCTED_ZERO_LAW_ATLAS = "OBSTRUCTED_ZERO_LAW_ATLAS"
    STRUCTURAL_NONPROMOTABLE = "STRUCTURAL_NONPROMOTABLE"


class BackboneExtensionGateStatus(StrEnum):
    F4_BACKBONE_RELEASE_READY = "F4_BACKBONE_RELEASE_READY"


@dataclass(frozen=True, slots=True)
class BackboneConformanceCaseReceipt(CanonicalRecord):
    """One actual terminal conformance output, never a new empirical claim."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/backbone-conformance-case-receipt'

    case_id: str
    kind: BackboneConformanceCaseKind
    evidence_records: tuple[ObjectIdentity, ...]
    public_route_receipt: ObjectIdentity | None
    operational_conformance: ObjectIdentity | None
    provider_registry_sha256: str | None
    terminal_disposition: str
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        require_sorted_unique_ids(
            self.evidence_records,
            attribute="object_id",
            field_name="evidence_records",
        )
        validate_nonempty(self.terminal_disposition, field_name="terminal_disposition")
        if not self.evidence_records or not self.terminal or self.created_empirical_evidence:
            raise ValueError("backbone conformance case is not terminal architecture evidence")
        route_values = (
            self.public_route_receipt,
            self.operational_conformance,
            self.provider_registry_sha256,
        )
        if self.kind is BackboneConformanceCaseKind.SUPPORTED_PUBLIC_ROUTE:
            if any(value is None for value in route_values):
                raise ValueError("supported backbone case lacks its actual public route")
            assert self.public_route_receipt is not None
            assert self.operational_conformance is not None
            assert self.provider_registry_sha256 is not None
            if (
                self.public_route_receipt.object_schema != RunEnvelopeRouteConformanceReceipt.SCHEMA
                or self.operational_conformance.object_schema
                != RunEnvelopeOperationalConformanceReceipt.SCHEMA
            ):
                raise ValueError("supported backbone case binds another route topology")
            validate_sha256(
                self.provider_registry_sha256,
                field_name="provider_registry_sha256",
            )
        elif any(value is not None for value in route_values):
            raise ValueError("non-route backbone case claims public-route execution")


@dataclass(frozen=True, slots=True)
class BackboneExtensionConformancePayload(CanonicalRecord):
    """Compact external payload over exact records; contains no scientific arrays/rows."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/backbone-extension-conformance-payload'

    payload_id: str
    source_closure: ImplementationSourceClosure
    extension_aggregate: ObjectIdentity
    cases: tuple[BackboneConformanceCaseReceipt, ...]
    terminal: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.payload_id, field_name="payload_id")
        if self.extension_aggregate.object_schema != GeneratedExtensionBundleAggregate.SCHEMA:
            raise ValueError("backbone conformance payload binds another extension aggregate")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if {value.kind for value in self.cases} != set(BackboneConformanceCaseKind):
            raise ValueError("backbone conformance payload case roster is incomplete")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or not self.terminal
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("backbone conformance payload overclaims its architecture boundary")


@dataclass(frozen=True, slots=True)
class BackboneExternalPublicationRecoveryReceipt(CanonicalRecord):
    """Actual immutable publication plus bounded path-free recovery identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/backbone-external-publication-recovery-receipt'

    receipt_id: str
    storage_root_id: str
    authoritative_relative_locator: str
    logical_artifact: ObjectIdentity
    materialization: ObjectIdentity
    publication: ObjectIdentity
    source_payload_sha256: str
    recovered_payload_sha256: str
    recovered_payload_size_bytes: int
    decoder_schema: str
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.authoritative_relative_locator)
        expected_schemas = (
            (self.logical_artifact, LogicalArtifactIdentity.SCHEMA),
            (self.materialization, ArtifactMaterialization.SCHEMA),
            (self.publication, ArtifactPublicationBinding.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("backbone publication receipt binds another artifact topology")
        validate_sha256(self.source_payload_sha256, field_name="source_payload_sha256")
        validate_sha256(self.recovered_payload_sha256, field_name="recovered_payload_sha256")
        if (
            self.source_payload_sha256 != self.recovered_payload_sha256
            or self.recovered_payload_size_bytes <= 0
            or self.decoder_schema != BackboneExtensionConformancePayload.SCHEMA
            or not self.terminal
        ):
            raise ValueError("backbone external publication did not recover exact payload bytes")


@dataclass(frozen=True, slots=True)
class ConsolidatedBackboneExtensionRelease(CanonicalRecord):
    """F0--F4 release gate, limited to architecture/conformance claims."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/consolidated-backbone-extension-release'

    release_id: str
    gate_status: BackboneExtensionGateStatus
    source_closure: ImplementationSourceClosure
    repository_state: ReleaseRepositoryStateReceipt
    extension_aggregate: ObjectIdentity
    corrected_base_release: ObjectIdentity
    corrected_base_attestation: ObjectIdentity
    current_consumer_pin: ObjectIdentity
    conformance_payload: ObjectIdentity
    conformance_cases: tuple[BackboneConformanceCaseReceipt, ...]
    external_publication_recovery: BackboneExternalPublicationRecoveryReceipt
    verification_receipts: tuple[ReleaseVerificationReceipt, ...]
    maximum_claim: ConsolidationClaimCeiling
    limitations: tuple[str, ...]
    repository_contains_scientific_payloads: bool
    created_empirical_evidence: bool
    released_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.release_id, field_name="release_id")
        parse_utc_timestamp(self.released_at_utc, field_name="released_at_utc")
        expected_schemas = (
            (self.extension_aggregate, GeneratedExtensionBundleAggregate.SCHEMA),
            (self.corrected_base_release, ResponseLawConsolidationRelease.SCHEMA),
            (self.corrected_base_attestation, ReleaseConformanceAttestation.SCHEMA),
            (self.current_consumer_pin, MultiWorldReadinessConsumerPin.SCHEMA),
            (self.conformance_payload, BackboneExtensionConformancePayload.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("backbone release dependency schema differs")
        require_sorted_unique_ids(
            self.conformance_cases,
            attribute="case_id",
            field_name="conformance_cases",
        )
        if {value.kind for value in self.conformance_cases} != set(BackboneConformanceCaseKind):
            raise ValueError("backbone release conformance roster is incomplete")
        require_sorted_unique_ids(
            self.verification_receipts,
            attribute="receipt_id",
            field_name="verification_receipts",
        )
        if not self.verification_receipts or any(
            value.implementation_commit != self.source_closure.implementation_commit
            for value in self.verification_receipts
        ):
            raise ValueError("backbone verification differs from its source closure")
        require_sorted_unique_strings(self.limitations, field_name="limitations", allow_empty=False)
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.repository_state.implementation_commit
            != self.source_closure.implementation_commit
            or self.repository_state.source_tree_sha256 != self.source_closure.source_tree_sha256
            or self.conformance_payload.object_fingerprint
            != self.external_publication_recovery.source_payload_sha256
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.repository_contains_scientific_payloads
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("backbone release overclaims its F4 boundary")


@dataclass(frozen=True, slots=True)
class ConsolidatedBackboneExtensionAttestation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/consolidated-backbone-extension-attestation'

    attestation_id: str
    release: ObjectIdentity
    external_publication_recovery: ObjectIdentity
    source_closure: ImplementationSourceClosure
    terminal: bool
    created_empirical_evidence: bool
    attested_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attestation_id, field_name="attestation_id")
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        if (
            self.release.object_schema != ConsolidatedBackboneExtensionRelease.SCHEMA
            or self.external_publication_recovery.object_schema
            != BackboneExternalPublicationRecoveryReceipt.SCHEMA
            or self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("backbone release attestation is not terminal and source-exact")


@dataclass(frozen=True, slots=True)
class ConsolidatedBackboneExtensionCorrection(CanonicalRecord):
    """Additive correction retaining the initial F4 release as provenance."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/consolidated-backbone-extension-correction'

    correction_id: str
    superseded_release: ObjectIdentity
    superseded_attestation: ObjectIdentity
    corrected_release: ObjectIdentity
    corrected_attestation: ObjectIdentity
    source_closure: ImplementationSourceClosure
    correction_reason_codes: tuple[str, ...]
    terminal: bool
    created_empirical_evidence: bool
    corrected_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.correction_id, field_name="correction_id")
        parse_utc_timestamp(self.corrected_at_utc, field_name="corrected_at_utc")
        for value in (self.superseded_release, self.corrected_release):
            if value.object_schema != ConsolidatedBackboneExtensionRelease.SCHEMA:
                raise ValueError("backbone correction binds another release schema")
        for value in (self.superseded_attestation, self.corrected_attestation):
            if value.object_schema != ConsolidatedBackboneExtensionAttestation.SCHEMA:
                raise ValueError("backbone correction binds another attestation schema")
        require_sorted_unique_strings(
            self.correction_reason_codes,
            field_name="correction_reason_codes",
            allow_empty=False,
        )
        if (
            self.superseded_release == self.corrected_release
            or self.superseded_attestation == self.corrected_attestation
            or self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("backbone release correction is not an additive terminal record")


def decode_backbone_extension_release(payload: bytes) -> ConsolidatedBackboneExtensionRelease:
    return decode_canonical_bytes(
        payload,
        ConsolidatedBackboneExtensionRelease,
        maximum_bytes=8 * 1024 * 1024,
    )


def decode_backbone_extension_attestation(
    payload: bytes,
) -> ConsolidatedBackboneExtensionAttestation:
    return decode_canonical_bytes(
        payload,
        ConsolidatedBackboneExtensionAttestation,
        maximum_bytes=2 * 1024 * 1024,
    )


def decode_backbone_extension_correction(
    payload: bytes,
) -> ConsolidatedBackboneExtensionCorrection:
    return decode_canonical_bytes(
        payload,
        ConsolidatedBackboneExtensionCorrection,
        maximum_bytes=2 * 1024 * 1024,
    )


__all__ = [
    'BackboneConformanceCaseKind',
    'BackboneConformanceCaseReceipt',
    'BackboneExtensionConformancePayload',
    'BackboneExtensionGateStatus',
    'BackboneExternalPublicationRecoveryReceipt',
    'ConsolidatedBackboneExtensionAttestation',
    'ConsolidatedBackboneExtensionCorrection',
    'ConsolidatedBackboneExtensionRelease',
    'decode_backbone_extension_attestation',
    'decode_backbone_extension_correction',
    'decode_backbone_extension_release',
]
