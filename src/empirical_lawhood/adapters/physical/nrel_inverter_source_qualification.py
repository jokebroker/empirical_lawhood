"""held-source qualification held-source inventory and honest physical-unit stop."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateIndependenceClass, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff
from empirical_lawhood.adapters.methods.structural_recurrence import TargetLevel
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


class NRELSourceQualificationDisposition(StrEnum):
    PASS = "PASS"
    PHYSICAL_UNIT_UNRESOLVED = "PHYSICAL_UNIT_UNRESOLVED"
    ACTION_SEMANTICS_UNRESOLVED = "ACTION_SEMANTICS_UNRESOLVED"
    UNSAFE_SOURCE_FORMAT = "UNSAFE_SOURCE_FORMAT"
    SOURCE_INTEGRITY_FAILURE = "SOURCE_INTEGRITY_FAILURE"


@dataclass(frozen=True, slots=True)
class NRELPublicCSVInventory(CanonicalRecord):
    """Structural facts for one exact public byte stream, without role guesses."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-public-csv-inventory'

    member_id: str
    relative_locator: str
    content_sha256: str
    size_bytes: int
    row_count: int
    minimum_column_count: int
    maximum_column_count: int
    header_present: bool
    strict_ascii: bool
    nul_byte_present: bool
    first_column_minimum: Decimal
    first_column_maximum: Decimal
    second_column_minimum: Decimal
    second_column_maximum: Decimal
    second_column_backstep_count: int
    first_column_one_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_nonempty(self.relative_locator, field_name="relative_locator")
        path = PurePosixPath(self.relative_locator)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_locator:
            raise ValueError("NREL public CSV locator is unsafe")
        if path.suffix.lower() != ".csv":
            raise ValueError("NREL public source member is not CSV")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if (
            min(
                self.size_bytes,
                self.row_count,
                self.minimum_column_count,
                self.maximum_column_count,
            )
            <= 0
        ):
            raise ValueError("NREL public CSV structural counts must be positive")
        if self.minimum_column_count > self.maximum_column_count:
            raise ValueError("NREL public CSV column bounds are inverted")
        for name in (
            "first_column_minimum",
            "first_column_maximum",
            "second_column_minimum",
            "second_column_maximum",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.first_column_minimum > self.first_column_maximum
            or self.second_column_minimum > self.second_column_maximum
            or min(self.second_column_backstep_count, self.first_column_one_count) < 0
        ):
            raise ValueError("NREL public CSV numeric structural summary is invalid")
        if not self.strict_ascii or self.nul_byte_present:
            raise ValueError("NREL public CSV is not a safe strict text stream")


@dataclass(frozen=True, slots=True)
class NRELSourceQualification(CanonicalRecord):
    """Exact held-source qualification result; missing source roles remain missing."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-source-qualification'

    qualification_id: str
    catalogue_record_id: str
    catalogue_snapshot_sha256: str
    catalogue_terms: str
    members: tuple[NRELPublicCSVInventory, ...]
    header_or_sidecar_schema_available: bool
    apparatus_identity_available: bool
    run_identity_available: bool
    preparation_reset_identity_available: bool
    requested_action_available: bool
    accepted_action_available: bool
    applied_action_available: bool
    realized_action_available: bool
    receiver_names_and_units_available: bool
    calibration_uncertainty_available: bool
    trip_saturation_validity_available: bool
    prior_receiver_outcome_accessed: bool
    cryptographic_split_committed_before_receiver_access: bool
    rows_count_as_units: bool
    scientific_unit_count: int
    prospective_prediction_eligible: bool
    disposition: NRELSourceQualificationDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.catalogue_record_id, field_name="catalogue_record_id")
        validate_sha256(
            self.catalogue_snapshot_sha256,
            field_name="catalogue_snapshot_sha256",
        )
        validate_nonempty(self.catalogue_terms, field_name="catalogue_terms")
        require_sorted_unique_ids(self.members, attribute="member_id", field_name="members")
        if not self.members:
            raise ValueError("NREL qualification has no acquired source members")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.rows_count_as_units:
            raise ValueError("NREL waveform rows cannot count as physical preparations")
        required_roles = (
            self.header_or_sidecar_schema_available,
            self.apparatus_identity_available,
            self.run_identity_available,
            self.preparation_reset_identity_available,
            self.requested_action_available,
            self.accepted_action_available,
            self.applied_action_available,
            self.realized_action_available,
            self.receiver_names_and_units_available,
            self.calibration_uncertainty_available,
            self.trip_saturation_validity_available,
        )
        controlled = (
            not self.prior_receiver_outcome_accessed
            and self.cryptographic_split_committed_before_receiver_access
        )
        passed = all(required_roles) and controlled and self.scientific_unit_count > 1
        if self.prospective_prediction_eligible != passed:
            raise ValueError("NREL prospective eligibility is not source-role-derived")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("NREL public-source qualification access differs")
        if passed:
            if self.disposition is not NRELSourceQualificationDisposition.PASS or (
                self.reason_codes
            ):
                raise ValueError("passing NREL source qualification retains a stop")
        else:
            expected = (
                NRELSourceQualificationDisposition.PHYSICAL_UNIT_UNRESOLVED
                if not (
                    self.run_identity_available
                    and self.preparation_reset_identity_available
                    and self.scientific_unit_count > 0
                )
                else NRELSourceQualificationDisposition.ACTION_SEMANTICS_UNRESOLVED
            )
            if self.disposition is not expected:
                raise ValueError("NREL stopped disposition differs from missing source roles")
            if not self.reason_codes or expected.value not in self.reason_codes:
                raise ValueError("NREL stopped qualification lacks its controlling reason")


def nrel_source_qualification_stop_handoff(
    qualification: NRELSourceQualification,
) -> IndependentSubstrateTargetTerminalHandoff:
    """Project physical-unit grouping without converting waveform rows into physical evidence."""

    if qualification.disposition is NRELSourceQualificationDisposition.PASS:
        raise ValueError("passing NREL source qualification requires predictive unit-mapping design, not a stop")
    reasons = tuple(
        sorted(
            {
                *qualification.reason_codes,
                "NREL_PREDICTIVE_UNIT_MAPPING_PREREQUISITE_NONATTEMPT",
                "RETROSPECTIVE_PHYSICAL_CONSISTENCY_UNEVALUABLE",
            }
        )
    )
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id="handoff.independent-substrate-grounding.source-continuation.nrel-inverter-physical-unit-grouping-stop",
        slot=IndependentSubstrateTargetKind.NREL_INVERTER,
        evidence_world=EvidenceWorld.RETROSPECTIVE_PHYSICAL_ARCHIVE,
        attained_level=TargetLevel.MEASUREMENT_READINESS,
        independence_class=IndependentSubstrateIndependenceClass.UNEVALUABLE,
        evaluation_eligible=False,
        categorical_support=False,
        decisive_opposition=False,
        unsafe_false_admission_count=0,
        action_ontology_clock_error_count=0,
        panel_envelope_limited=False,
        structural_recurrence_restrictiveness_supported=False,
        comparator_tied_or_won=False,
        structural_recurrence_less_safe_or_exact_than_comparator=False,
        metric_topology_exchanges=(),
        physical_consistency="UNEVALUABLE",
        prediction_receipt_sha256=None,
        match_receipt_sha256=qualification.fingerprint(),
        maximum_claim_ceiling=(
            "ACQUIRED_PUBLIC_WAVEFORM_SOURCE_PHYSICAL_UNIT_AND_ACTION_ROLES_UNRESOLVED"
        ),
        reason_codes=reasons,
    )


__all__ = [
    'NRELPublicCSVInventory',
    'NRELSourceQualificationDisposition',
    'NRELSourceQualification',
    "nrel_source_qualification_stop_handoff",
]
