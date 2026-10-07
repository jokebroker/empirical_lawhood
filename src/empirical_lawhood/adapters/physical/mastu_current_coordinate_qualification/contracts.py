"""Claim-limited source contracts for the public MAST-U current follow-up."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.laws import CausalStrength
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

from empirical_lawhood.adapters.physical.mastu_release_qualification import MASTUPublicArchiveInventory, MASTUPublicReleaseObjectPolicy, MASTUPublicTrustedUseAcceptance


CURRENT_SOURCE_LIMITATIONS: tuple[str, ...] = (
    "ARCHIVE_INCLUDED_AND_FINITE_VALIDITY_ONLY",
    "EXACT_CALIBRATION_INSTANCE_UNKNOWN",
    "NO_ACTION_OR_CONTROLLER_SEMANTICS",
    "QUALITY_FLAGS_ABSENT",
    "SENSOR_LATENCY_AND_SYNCHRONISATION_UNKNOWN",
    "TOTAL_CURRENT_UNCERTAINTY_UNKNOWN",
)


class MASTUCurrentCoordinateRole(StrEnum):
    PROCESSED_DIAGNOSTIC_CONSTRAINT_TARGET = "PROCESSED_DIAGNOSTIC_CONSTRAINT_TARGET"


class MASTUCurrentSourceDisposition(StrEnum):
    ADMITTED_WITH_LIMITATIONS = "ADMITTED_WITH_LIMITATIONS"


@dataclass(frozen=True, slots=True)
class MASTUPublicEvidenceDocument(CanonicalRecord):
    """Exact public document bytes and the narrow facts used from them."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-current-coordinate-qualification/mastu-public-evidence-document'

    document_id: str
    title: str
    doi: str
    public_url: str
    held_relative_locator: str
    expected_size_bytes: int
    expected_sha256: str
    admitted_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.document_id, field_name="document_id")
        validate_nonempty(self.title, field_name="title")
        validate_nonempty(self.doi, field_name="doi")
        if not self.public_url.startswith("https://scientific-publications.ukaea.uk/"):
            raise ValueError("supporting evidence must use the official UKAEA HTTPS origin")
        validate_relative_locator(self.held_relative_locator)
        if not self.held_relative_locator.endswith(".pdf"):
            raise ValueError("supporting evidence locator must name one PDF")
        if isinstance(self.expected_size_bytes, bool) or self.expected_size_bytes <= 0:
            raise ValueError("expected_size_bytes must be positive")
        validate_sha256(self.expected_sha256, field_name="expected_sha256")
        require_sorted_unique_strings(
            self.admitted_fact_ids,
            field_name="admitted_fact_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class MASTUPublicCurrentSourceQualification(CanonicalRecord):
    """Outcome-blind admission of one release-native observational coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-current-coordinate-qualification/mastu-public-current-source-qualification'

    qualification_id: str
    experiment_id: str
    predecessor_terminal_id: str
    trusted_use_acceptance: ObjectIdentity
    trace_release_policy: ObjectIdentity
    trace_inventory: ObjectIdentity
    machine_release_policy: ObjectIdentity
    machine_inventory: ObjectIdentity
    evidence_documents: tuple[ObjectIdentity, ...]
    coordinate_id: str
    coordinate_role: MASTUCurrentCoordinateRole
    source_label_path: str
    source_value_path: str
    serialized_label_field: str
    serialized_value_field: str
    included_active_channel_count: int
    excluded_channel_labels: tuple[str, ...]
    native_unit: str
    positive_direction: str
    calibration_semantics: str
    validity_semantics: str
    clock_semantics: str
    action_role: str
    causal_strength_ceiling: CausalStrength
    limitations: tuple[str, ...]
    development_conversion_eligible: bool
    protected_conversion_requires_frozen_issue: bool
    realized_current_observation_admitted: bool
    voltage_or_current_action_admitted: bool
    passive_current_measurement_admitted: bool
    disposition: MASTUCurrentSourceDisposition
    maximum_claim_ceiling: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        validate_stable_id(self.predecessor_terminal_id, field_name="predecessor_terminal_id")
        expected_schemas = (
            (self.trusted_use_acceptance, MASTUPublicTrustedUseAcceptance.SCHEMA),
            (self.trace_release_policy, MASTUPublicReleaseObjectPolicy.SCHEMA),
            (self.trace_inventory, MASTUPublicArchiveInventory.SCHEMA),
            (self.machine_release_policy, MASTUPublicReleaseObjectPolicy.SCHEMA),
            (self.machine_inventory, MASTUPublicArchiveInventory.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("current qualification binds another release/inventory schema")
        require_sorted_unique_ids(
            self.evidence_documents,
            attribute="object_id",
            field_name="evidence_documents",
        )
        if len(self.evidence_documents) != 2 or any(
            value.object_schema != MASTUPublicEvidenceDocument.SCHEMA
            for value in self.evidence_documents
        ):
            raise ValueError("current qualification requires the two exact public documents")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.coordinate_role is not (
            MASTUCurrentCoordinateRole.PROCESSED_DIAGNOSTIC_CONSTRAINT_TARGET
        ):
            raise ValueError("current source role differs from its narrow observational role")
        if self.source_label_path != "/epm/input/constraints/pfcircuits/shortname":
            raise ValueError("current label source path differs from the released script")
        if self.source_value_path != "/epm/input/constraints/pfcircuits/target":
            raise ValueError("current value source path differs from the released script")
        if (self.serialized_label_field, self.serialized_value_field) != (
            "coil_name",
            "currents_input",
        ):
            raise ValueError("current serialized fields differ from the release")
        if self.included_active_channel_count != 23 or self.excluded_channel_labels != ("pc",):
            raise ValueError("current coordinate must be the 23 connected non-PC active channels")
        if self.native_unit != "A":
            raise ValueError("current coordinate must retain amperes")
        if self.positive_direction != "TOROIDAL_ANTICLOCKWISE_VIEWED_FROM_ABOVE_POSITIVE":
            raise ValueError("current positive-direction convention differs")
        if self.calibration_semantics != (
            "PUBLIC_PROCEDURE_BOUND_EXACT_INSTANCE_AND_TOTAL_UNCERTAINTY_UNAVAILABLE"
        ):
            raise ValueError("current calibration ceiling differs")
        if self.validity_semantics != "PUBLISHED_ANALYSIS_WINDOW_AND_NUMERIC_FINITE_ONLY":
            raise ValueError("current validity ceiling differs")
        if self.clock_semantics != (
            "EPM_RECONSTRUCTION_TIME_SECONDS_SENSOR_LATENCY_AND_SYNCHRONY_UNQUALIFIED"
        ):
            raise ValueError("current clock ceiling differs")
        if self.action_role != "NONE" or self.causal_strength_ceiling is not (
            CausalStrength.OBSERVATIONAL_ASSOCIATION
        ):
            raise ValueError("current coordinate cannot acquire an action or causal role")
        require_sorted_unique_strings(self.limitations, field_name="limitations")
        if self.limitations != CURRENT_SOURCE_LIMITATIONS:
            raise ValueError("current qualification limitation roster differs")
        if not self.development_conversion_eligible or not (
            self.protected_conversion_requires_frozen_issue
        ):
            raise ValueError("current conversion ordering differs")
        if any(
            (
                self.realized_current_observation_admitted,
                self.voltage_or_current_action_admitted,
                self.passive_current_measurement_admitted,
            )
        ):
            raise ValueError("current qualification overstates public source semantics")
        if self.disposition is not MASTUCurrentSourceDisposition.ADMITTED_WITH_LIMITATIONS:
            raise ValueError("current qualification disposition differs")
        if self.maximum_claim_ceiling != (
            "RELEASE_NATIVE_CURRENT_COORDINATE_CONDITIONED_OBSERVATIONAL_ASSOCIATION"
        ):
            raise ValueError("current qualification claim ceiling differs")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification must remain outcome-blind")


__all__ = [
    "CURRENT_SOURCE_LIMITATIONS",
    'MASTUCurrentCoordinateRole',
    'MASTUCurrentSourceDisposition',
    'MASTUPublicCurrentSourceQualification',
    'MASTUPublicEvidenceDocument',
]
