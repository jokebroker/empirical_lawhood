"""Pre-conversion semantic gate for the public MAST-U voltage source.

The current release fails this gate, so trusted deserialization and numeric
conversion are deliberately absent from the reachable adapter surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .contracts import MASTUPublicArchiveInventory, MASTUPublicDataCiteMetadataReceipt, MASTUPublicMemberRole, MASTUPublicReleaseObjectPolicy, MASTUPublicTrustedUseAcceptance


VOLTAGE_SIGNAL_PATHS: tuple[str, ...] = (
    "/XCM/D1/VOUT",
    "/XCM/D2/VOUT",
    "/XCM/D3/VOUT",
    "/XCM/D5/VOUT",
    "/XCM/D6/VOUT",
    "/XCM/D7/VOUT",
    "/XCM/DP/VOUT",
    "/XCM/MFPS/VOLTS",
    "/XCM/PX/VOUT",
    "/XCM/RFPS/VOUT",
    "/XCM/SFPS/VOLTS",
    "/xcm/p1ps/volts",
)
SEMANTIC_REFUSAL_CODES: tuple[str, ...] = (
    "FLAGSHIP_UNIT_QUALIFIED_REALIZED_VOLTAGE_PREMISE_NOT_ADMITTED",
    "VOLTAGE_CALIBRATION_UNQUALIFIED",
    "VOLTAGE_MEASUREMENT_POINT_UNQUALIFIED",
    "VOLTAGE_POLARITY_UNQUALIFIED",
    "VOLTAGE_UNIT_UNQUALIFIED",
    "VOLTAGE_VALIDITY_UNQUALIFIED",
)


class MASTUPublicSourceQualificationDisposition(StrEnum):
    PASS = "PASS"
    SOURCE_SEMANTICS_UNEVALUABLE = "SOURCE_SEMANTICS_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MASTUPublicVoltageSourceAudit(CanonicalRecord):
    """Outcome-blind audit of exact DataCite and released provenance source."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-voltage-source-audit'

    audit_id: str
    release_policy: ObjectIdentity
    trace_inventory: ObjectIdentity
    datacite_metadata: ObjectIdentity
    datacite_metadata_sha256: str
    datacite_metadata_size_bytes: int
    provenance_member_id: str
    provenance_source_sha256: str
    provenance_source_size_bytes: int
    voltage_signal_paths: tuple[str, ...]
    serialized_voltage_fields: tuple[str, ...]
    voltage_unit_serialized: bool
    voltage_polarity_or_sign_serialized: bool
    voltage_measurement_point_definition_serialized: bool
    voltage_calibration_serialized: bool
    voltage_validity_serialized: bool
    controller_history_serialized: bool
    requested_accepted_applied_action_chain_serialized: bool
    shot_45425_deserialized: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if self.release_policy.object_schema != MASTUPublicReleaseObjectPolicy.SCHEMA:
            raise ValueError("voltage audit binds another release-policy schema")
        if self.trace_inventory.object_schema != MASTUPublicArchiveInventory.SCHEMA:
            raise ValueError("voltage audit binds another inventory schema")
        if self.datacite_metadata.object_schema != MASTUPublicDataCiteMetadataReceipt.SCHEMA:
            raise ValueError("voltage audit binds another DataCite receipt schema")
        validate_sha256(self.datacite_metadata_sha256, field_name="datacite_metadata_sha256")
        validate_sha256(self.provenance_source_sha256, field_name="provenance_source_sha256")
        if min(self.datacite_metadata_size_bytes, self.provenance_source_size_bytes) <= 0:
            raise ValueError("voltage audit source sizes must be positive")
        validate_stable_id(self.provenance_member_id, field_name="provenance_member_id")
        if self.voltage_signal_paths != VOLTAGE_SIGNAL_PATHS:
            raise ValueError("voltage audit signal-path roster differs from released source")
        if self.serialized_voltage_fields != ("times", "voltages"):
            raise ValueError("voltage audit serialized-field roster differs")
        semantic_assertions = (
            self.voltage_unit_serialized,
            self.voltage_polarity_or_sign_serialized,
            self.voltage_measurement_point_definition_serialized,
            self.voltage_calibration_serialized,
            self.voltage_validity_serialized,
            self.controller_history_serialized,
            self.requested_accepted_applied_action_chain_serialized,
            self.shot_45425_deserialized,
        )
        if any(semantic_assertions):
            raise ValueError("exact released source audit cannot assert absent semantics/access")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source-semantic audit must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class MASTUPublicSourceSemanticQualification(CanonicalRecord):
    """Typed F3 refusal for the unit-qualified realized-voltage premise."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-source-semantic-qualification'

    qualification_id: str
    trusted_use_acceptance: ObjectIdentity
    trace_inventory: ObjectIdentity
    jmpp_inventory: ObjectIdentity
    jmpp_datacite_metadata: ObjectIdentity
    voltage_source_audit: ObjectIdentity
    unit_established: bool
    polarity_or_sign_established: bool
    measurement_point_established: bool
    calibration_established: bool
    validity_established: bool
    unit_qualified_realized_voltage_premise_admitted: bool
    campaign_a_issue_eligible: bool
    trusted_deserialization_started: bool
    shot_45425_deserialized: bool
    disposition: MASTUPublicSourceQualificationDisposition
    maximum_claim_ceiling: str
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        expected_schemas = (
            (self.trusted_use_acceptance, MASTUPublicTrustedUseAcceptance.SCHEMA),
            (self.trace_inventory, MASTUPublicArchiveInventory.SCHEMA),
            (self.jmpp_inventory, MASTUPublicArchiveInventory.SCHEMA),
            (
                self.jmpp_datacite_metadata,
                MASTUPublicDataCiteMetadataReceipt.SCHEMA,
            ),
            (self.voltage_source_audit, MASTUPublicVoltageSourceAudit.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("source qualification binds another record schema")
        established = (
            self.unit_established,
            self.polarity_or_sign_established,
            self.measurement_point_established,
            self.calibration_established,
            self.validity_established,
        )
        if any(established) or any(
            (
                self.unit_qualified_realized_voltage_premise_admitted,
                self.campaign_a_issue_eligible,
                self.trusted_deserialization_started,
                self.shot_45425_deserialized,
            )
        ):
            raise ValueError("failed source semantics cannot admit or deserialize the campaign")
        if self.disposition is not (
            MASTUPublicSourceQualificationDisposition.SOURCE_SEMANTICS_UNEVALUABLE
        ):
            raise ValueError("exact current release must retain its semantic refusal")
        if self.maximum_claim_ceiling != "PUBLIC_HELD_BYTES_AND_SIGNAL_PATH_PROVENANCE_ONLY":
            raise ValueError("source qualification claim ceiling differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.reason_codes != SEMANTIC_REFUSAL_CODES:
            raise ValueError("source qualification reason roster differs")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification must remain outcome-blind")


def audit_voltage_source(
    policy: MASTUPublicReleaseObjectPolicy,
    inventory: MASTUPublicArchiveInventory,
    datacite_metadata: MASTUPublicDataCiteMetadataReceipt,
) -> MASTUPublicVoltageSourceAudit:
    """Bind the code-owned text audit to exact held metadata/source identities."""

    policy_identity = ObjectIdentity.from_record(policy.release_object_id, policy)
    if inventory.release_policy != policy_identity:
        raise ValueError("trace inventory differs from the exact release policy")
    if (
        datacite_metadata.release_policy != policy_identity
        or datacite_metadata.physical_sha256 != policy.datacite_metadata_sha256
        or datacite_metadata.size_bytes != policy.datacite_metadata_size_bytes
    ):
        raise ValueError("DataCite metadata receipt differs from the exact release policy")
    matches = tuple(
        value
        for value in inventory.members
        if value.selected_role is MASTUPublicMemberRole.PROVENANCE_SOURCE
    )
    if len(matches) != 1:
        raise ValueError("trace inventory lacks the exact provenance-source member")
    source = matches[0]
    source_sha256 = source.selected_content_sha256
    if source_sha256 is None:
        raise ValueError("trace inventory lacks the provenance-source content hash")
    return MASTUPublicVoltageSourceAudit(
        audit_id="audit.mastu-public.26m5-voltage-source",
        release_policy=policy_identity,
        trace_inventory=ObjectIdentity.from_record(inventory.inventory_id, inventory),
        datacite_metadata=ObjectIdentity.from_record(
            datacite_metadata.receipt_id, datacite_metadata
        ),
        datacite_metadata_sha256=policy.datacite_metadata_sha256,
        datacite_metadata_size_bytes=policy.datacite_metadata_size_bytes,
        provenance_member_id=source.member_id,
        provenance_source_sha256=source_sha256,
        provenance_source_size_bytes=source.uncompressed_size_bytes,
        voltage_signal_paths=VOLTAGE_SIGNAL_PATHS,
        serialized_voltage_fields=("times", "voltages"),
        voltage_unit_serialized=False,
        voltage_polarity_or_sign_serialized=False,
        voltage_measurement_point_definition_serialized=False,
        voltage_calibration_serialized=False,
        voltage_validity_serialized=False,
        controller_history_serialized=False,
        requested_accepted_applied_action_chain_serialized=False,
        shot_45425_deserialized=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def qualify_primary_source_semantics(
    *,
    acceptance: MASTUPublicTrustedUseAcceptance,
    trace_inventory: MASTUPublicArchiveInventory,
    jmpp_inventory: MASTUPublicArchiveInventory,
    jmpp_datacite_metadata: MASTUPublicDataCiteMetadataReceipt,
    audit: MASTUPublicVoltageSourceAudit,
) -> MASTUPublicSourceSemanticQualification:
    """Emit the first honest terminal without deserializing either shot."""

    inventory_policies = (trace_inventory.release_policy, jmpp_inventory.release_policy)
    if inventory_policies != acceptance.accepted_release_policies:
        raise ValueError("inventories differ from exact trusted-use acceptance")
    if jmpp_datacite_metadata.release_policy != jmpp_inventory.release_policy:
        raise ValueError("JMPP DataCite metadata differs from the exact JMPP inventory")
    exact_audit = ObjectIdentity.from_record(audit.audit_id, audit)
    if audit.trace_inventory != ObjectIdentity.from_record(
        trace_inventory.inventory_id, trace_inventory
    ):
        raise ValueError("voltage audit differs from the exact trace inventory")
    return MASTUPublicSourceSemanticQualification(
        qualification_id="qualification.mastu-public.primary-source-semantics",
        trusted_use_acceptance=ObjectIdentity.from_record(acceptance.acceptance_id, acceptance),
        trace_inventory=ObjectIdentity.from_record(trace_inventory.inventory_id, trace_inventory),
        jmpp_inventory=ObjectIdentity.from_record(jmpp_inventory.inventory_id, jmpp_inventory),
        jmpp_datacite_metadata=ObjectIdentity.from_record(
            jmpp_datacite_metadata.receipt_id, jmpp_datacite_metadata
        ),
        voltage_source_audit=exact_audit,
        unit_established=False,
        polarity_or_sign_established=False,
        measurement_point_established=False,
        calibration_established=False,
        validity_established=False,
        unit_qualified_realized_voltage_premise_admitted=False,
        campaign_a_issue_eligible=False,
        trusted_deserialization_started=False,
        shot_45425_deserialized=False,
        disposition=(MASTUPublicSourceQualificationDisposition.SOURCE_SEMANTICS_UNEVALUABLE),
        maximum_claim_ceiling="PUBLIC_HELD_BYTES_AND_SIGNAL_PATH_PROVENANCE_ONLY",
        reason_codes=SEMANTIC_REFUSAL_CODES,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    'MASTUPublicSourceQualificationDisposition',
    'MASTUPublicSourceSemanticQualification',
    'MASTUPublicVoltageSourceAudit',
    "SEMANTIC_REFUSAL_CODES",
    "VOLTAGE_SIGNAL_PATHS",
    "audit_voltage_source",
    "qualify_primary_source_semantics",
]
