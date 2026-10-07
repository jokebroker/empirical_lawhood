"Exact prepared-denominator/local-support compatibility for admission handoff.\n\nThis module owns a family-neutral, pure evidence join.  Adapter-owned method\nrecords project only their exact identities, current action words and finite\nentry coordinates into the records below.  The evaluator neither decodes a\npayload nor interprets a local-support fingerprint.  In particular, it cannot\ninterpolate, enlarge support, finalize a law, create admission truth or grant\nauthority.\n"

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus

from .evidence_geometry import ReceiptAdmissionSupportCell


_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_CHECK_IDS = (
    "action-word-roster",
    "admission-action-bounds",
    "admission-cell-roster",
    "admission-chart",
    "admission-denominator",
    "exact-qualified-coordinates",
    "local-support-fingerprints",
    "local-support-roster",
    "payload-publication-identity",
    "prepared-denominator",
    "region-compatibility-identity",
    "supported-law-result",
)


def _validate_reason_codes(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> None:
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    if any(_REASON_CODE.fullmatch(value) is None for value in values):
        raise ValueError(f"{field_name} must contain canonical UPPER_SNAKE_CASE codes")


def _validate_identity_roster(
    values: tuple[ObjectIdentity, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> None:
    require_sorted_unique_ids(values, attribute="object_id", field_name=field_name)
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")


class LocalSupportRegionDisposition(StrEnum):
    """Upstream region/panel qualification presented to the shared join."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class LocalSupportCompatibilityCheckStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNEVALUABLE = "UNEVALUABLE"


class PreparedDenominatorLocalSupportCompatibilityDisposition(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class LocalSupportFingerprint(CanonicalRecord):
    "One opaque typed local coordinate and its exact admission action-bound roster."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/local-support-fingerprint'

    local_support_id: str
    coordinate_fingerprint: ObjectIdentity
    admission_action_bound_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.local_support_id, field_name="local_support_id")
        require_sorted_unique_strings(
            self.admission_action_bound_ids,
            field_name="admission_action_bound_ids",
            allow_empty=False,
        )
        for bound_id in self.admission_action_bound_ids:
            validate_stable_id(bound_id, field_name="admission_action_bound_ids")


@dataclass(frozen=True, slots=True)
class PreparedDenominatorLocalSupportCompatibilitySpec(CanonicalRecord):
    """Outcome-blind issued meaning of one prepared-D/local-support join."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/prepared-denominator-local-support-compatibility-spec'
    )

    spec_id: str
    prepared_denominator_id: str
    chart_id: str
    region_support_spec: ObjectIdentity
    payload_schema: str
    payload_publication_schema: str
    region_compatibility_schema: str
    local_supports: tuple[LocalSupportFingerprint, ...]
    action_words: tuple[ObjectIdentity, ...]
    evaluator_implementation: ObjectIdentity
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND
    maximum_outcome_access: OutcomeAccess = OutcomeAccess.EVALUATOR_REVEAL
    frozen_before_issue: bool = True
    interpolation_allowed: bool = False
    geometry_interpretation_allowed: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        validate_stable_id(
            self.prepared_denominator_id,
            field_name="prepared_denominator_id",
        )
        validate_stable_id(self.chart_id, field_name="chart_id")
        for schema in (
            self.payload_schema,
            self.payload_publication_schema,
            self.region_compatibility_schema,
        ):
            validate_schema(schema)
        require_sorted_unique_ids(
            self.local_supports,
            attribute="local_support_id",
            field_name="local_supports",
        )
        if not self.local_supports:
            raise ValueError("local-support compatibility requires at least one local")
        if self.prepared_denominator_id in {
            value.local_support_id for value in self.local_supports
        }:
            raise ValueError("prepared denominator cannot alias a local support")
        coordinate_ids = tuple(
            value.coordinate_fingerprint.object_id for value in self.local_supports
        )
        if len(set(coordinate_ids)) != len(coordinate_ids):
            raise ValueError("local supports require distinct typed coordinate identities")
        _validate_identity_roster(
            self.action_words,
            field_name="action_words",
            allow_empty=False,
        )
        if any(value.object_schema != OccurrenceActionWord.SCHEMA for value in self.action_words):
            raise ValueError("local-support spec requires current ActionWord identities")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("local-support compatibility spec must be outcome-blind")
        if self.maximum_outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("local-support compatibility is evaluator-reveal bounded")
        if not self.frozen_before_issue:
            raise ValueError("local-support compatibility spec must freeze before issue")
        if self.interpolation_allowed:
            raise ValueError("local-support compatibility cannot authorize interpolation")
        if self.geometry_interpretation_allowed:
            raise ValueError("the shared compatibility seam cannot interpret geometry")


@dataclass(frozen=True, slots=True)
class LocalSupportPayloadEntry(CanonicalRecord):
    """Exact identity projection of one adapter-owned finite payload entry."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/local-support-payload-entry'

    entry: ObjectIdentity
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    local_support_id: str
    local_support_fingerprint: ObjectIdentity
    action_word: ObjectIdentity
    coordinate_qualified: bool
    interpolation_used: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("local_support_id", self.local_support_id),
        ):
            validate_stable_id(value, field_name=field_name)
        require_sorted_unique_strings(
            self.qualification_view_ids,
            field_name="qualification_view_ids",
            allow_empty=False,
        )
        for value in self.qualification_view_ids:
            validate_stable_id(value, field_name="qualification_view_ids")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("local payload entry requires a current ActionWord identity")

    @property
    def entry_id(self) -> str:
        return self.entry.object_id

    @property
    def coordinate_key(self) -> tuple[str, str, tuple[str, ...], str, str]:
        return (
            self.denominator_member_id,
            self.candidate_version_id,
            self.qualification_view_ids,
            self.local_support_id,
            self.action_word.object_id,
        )


@dataclass(frozen=True, slots=True)
class PreparedDenominatorLocalSupportPayloadEvidence(CanonicalRecord):
    """Complete, identity-bound projection of an adapter-owned local payload."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/prepared-denominator-local-support-payload-evidence'
    )

    evidence_id: str
    payload: ObjectIdentity
    publication: ObjectIdentity
    published_content_sha256: str
    region_compatibility: ObjectIdentity
    prepared_denominator_id: str
    action_words: tuple[OccurrenceActionWord, ...]
    entries: tuple[LocalSupportPayloadEntry, ...]
    complete_entry_roster: bool
    outcome_access: OutcomeAccess = OutcomeAccess.EVALUATOR_REVEAL

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(
            self.prepared_denominator_id,
            field_name="prepared_denominator_id",
        )
        validate_sha256(
            self.published_content_sha256,
            field_name="published_content_sha256",
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.entries,
            attribute="entry_id",
            field_name="entries",
        )
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("local payload evidence is evaluator-reveal only")


@dataclass(frozen=True, slots=True)
class PreparedDenominatorLocalSupportRegionEvidence(CanonicalRecord):
    """Authenticated generic view of an adapter-owned region compatibility."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/prepared-denominator-local-support-region-evidence'
    )

    evidence_id: str
    compatibility: ObjectIdentity
    region_support_spec: ObjectIdentity
    payload: ObjectIdentity
    local_supports: tuple[LocalSupportFingerprint, ...]
    disposition: LocalSupportRegionDisposition
    reason_codes: tuple[str, ...]
    interpolation_used: bool
    support_expanded: bool
    unqualified_coordinate_ids: tuple[str, ...]
    outcome_access: OutcomeAccess = OutcomeAccess.EVALUATOR_REVEAL

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        require_sorted_unique_ids(
            self.local_supports,
            attribute="local_support_id",
            field_name="local_supports",
        )
        require_sorted_unique_strings(
            self.unqualified_coordinate_ids,
            field_name="unqualified_coordinate_ids",
        )
        supported = self.disposition is LocalSupportRegionDisposition.SUPPORTED
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=supported,
        )
        if supported and self.reason_codes:
            raise ValueError("supported region compatibility cannot carry reasons")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("region compatibility evidence is evaluator-reveal only")


@dataclass(frozen=True, slots=True)
class LocalSupportCompatibilityCheck(CanonicalRecord):
    """One noncompensating compatibility predicate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/local-support-compatibility-check'

    check_id: str
    status: LocalSupportCompatibilityCheckStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        passed = self.status is LocalSupportCompatibilityCheckStatus.PASSED
        _validate_reason_codes(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=passed,
        )
        if passed and self.reason_codes:
            raise ValueError("passed compatibility check cannot carry reasons")


@dataclass(frozen=True, slots=True)
class PreparedDenominatorLocalSupportCompatibility(CanonicalRecord):
    "Terminal compatibility receipt; never a law, admission verdict or authority."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prepared-denominator-local-support-compatibility'

    receipt_id: str
    compatibility_spec: ObjectIdentity
    evaluator_implementation: ObjectIdentity
    payload_evidence: ObjectIdentity | None
    payload: ObjectIdentity | None
    payload_publication: ObjectIdentity | None
    region_evidence: ObjectIdentity | None
    region_compatibility: ObjectIdentity | None
    qualification_result: ObjectIdentity | None
    response_law: ObjectIdentity | None
    prepared_denominator_id: str
    chart_id: str
    local_supports: tuple[LocalSupportFingerprint, ...]
    action_words: tuple[ObjectIdentity, ...]
    law_support_ids: tuple[str, ...]
    law_action_bound_ids: tuple[str, ...]
    payload_local_support_ids: tuple[str, ...]
    admission_support_cells: tuple[ReceiptAdmissionSupportCell, ...]
    checks: tuple[LocalSupportCompatibilityCheck, ...]
    disposition: PreparedDenominatorLocalSupportCompatibilityDisposition
    reason_codes: tuple[str, ...]
    input_interpolation_detected: bool
    input_unqualified_coordinate_detected: bool
    outcome_access: OutcomeAccess = OutcomeAccess.EVALUATOR_REVEAL
    geometry_interpreted: bool = False
    interpolation_performed: bool = False
    scientific_finalization_performed: bool = False
    grants_admission_truth: bool = False
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.prepared_denominator_id,
            field_name="prepared_denominator_id",
        )
        validate_stable_id(self.chart_id, field_name="chart_id")
        require_sorted_unique_ids(
            self.local_supports,
            attribute="local_support_id",
            field_name="local_supports",
        )
        _validate_identity_roster(
            self.action_words,
            field_name="action_words",
            allow_empty=False,
        )
        for field_name, values in (
            ("law_support_ids", self.law_support_ids),
            ("law_action_bound_ids", self.law_action_bound_ids),
            ("payload_local_support_ids", self.payload_local_support_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        expected_cells = tuple(
            sorted(
                self.admission_support_cells,
                key=lambda value: (value.cell_id, value.fingerprint()),
            )
        )
        if self.admission_support_cells != expected_cells:
            raise ValueError("admission support cells must use canonical cell/fingerprint order")
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        if tuple(value.check_id for value in self.checks) != _CHECK_IDS:
            raise ValueError("local-support receipt lacks the closed compatibility checks")
        expected_reasons = tuple(
            sorted({reason for value in self.checks for reason in value.reason_codes})
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("local-support receipt reasons differ from its exact checks")
        expected_disposition = (
            PreparedDenominatorLocalSupportCompatibilityDisposition.UNEVALUABLE
            if any(
                value.status is LocalSupportCompatibilityCheckStatus.UNEVALUABLE
                for value in self.checks
            )
            else PreparedDenominatorLocalSupportCompatibilityDisposition.INCOMPATIBLE
            if any(
                value.status is LocalSupportCompatibilityCheckStatus.FAILED
                for value in self.checks
            )
            else PreparedDenominatorLocalSupportCompatibilityDisposition.COMPATIBLE
        )
        if self.disposition is not expected_disposition:
            raise ValueError("local-support receipt disposition is not its check intersection")
        if self.disposition is (
            PreparedDenominatorLocalSupportCompatibilityDisposition.COMPATIBLE
        ):
            if self.reason_codes:
                raise ValueError("compatible local-support receipt cannot carry reasons")
            if self.input_interpolation_detected or self.input_unqualified_coordinate_detected:
                raise ValueError("compatible local-support receipt contains an inexact coordinate")
        elif not self.reason_codes:
            raise ValueError("non-compatible local-support receipt requires decisive reasons")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("local-support receipt is evaluator-reveal only")
        if self.geometry_interpreted:
            raise ValueError("local-support compatibility cannot interpret geometry")
        if self.interpolation_performed:
            raise ValueError("local-support compatibility cannot interpolate")
        if self.scientific_finalization_performed:
            raise ValueError("local-support compatibility cannot finalize science")
        if self.grants_admission_truth:
            raise ValueError("local-support compatibility cannot create admission truth")
        if self.grants_authority:
            raise ValueError("local-support compatibility cannot grant authority")


def _check(
    check_id: str,
    *,
    failed: tuple[str, ...] = (),
    unevaluable: tuple[str, ...] = (),
) -> LocalSupportCompatibilityCheck:
    failed = tuple(sorted(set(failed)))
    unevaluable = tuple(sorted(set(unevaluable)))
    if unevaluable:
        status = LocalSupportCompatibilityCheckStatus.UNEVALUABLE
        reasons = tuple(sorted({*failed, *unevaluable}))
    elif failed:
        status = LocalSupportCompatibilityCheckStatus.FAILED
        reasons = failed
    else:
        status = LocalSupportCompatibilityCheckStatus.PASSED
        reasons = ()
    return LocalSupportCompatibilityCheck(
        check_id=check_id,
        status=status,
        reason_codes=reasons,
    )


@dataclass(frozen=True, slots=True)
class PreparedDenominatorLocalSupportCompatibilityEvaluator:
    """Evaluate the exact join without I/O, geometry logic or finalization."""

    def evaluate(
        self,
        *,
        receipt_id: str,
        issued_spec: PreparedDenominatorLocalSupportCompatibilitySpec,
        payload_evidence: PreparedDenominatorLocalSupportPayloadEvidence | None,
        region_evidence: PreparedDenominatorLocalSupportRegionEvidence | None,
        qualification_result: LawQualificationResult | None,
        response_law: ResponseLaw | None,
        admission_support_cells: tuple[ReceiptAdmissionSupportCell, ...],
    ) -> PreparedDenominatorLocalSupportCompatibility:
        validate_stable_id(receipt_id, field_name="receipt_id")
        local_ids = tuple(value.local_support_id for value in issued_spec.local_supports)
        local_id_set = set(local_ids)
        spec_fingerprints = {
            value.local_support_id: value.coordinate_fingerprint
            for value in issued_spec.local_supports
        }
        spec_bounds = {
            value.local_support_id: value.admission_action_bound_ids
            for value in issued_spec.local_supports
        }
        spec_action_ids = {value.object_id for value in issued_spec.action_words}

        law_support_ids = (
            response_law.obligations.support.denominator_cell_ids
            if response_law is not None
            else ()
        )
        law_action_bound_ids = (
            tuple(value.bound_id for value in response_law.obligations.support.action_bounds)
            if response_law is not None
            else ()
        )
        payload_entries = payload_evidence.entries if payload_evidence is not None else ()
        payload_local_support_ids = tuple(
            sorted({value.local_support_id for value in payload_entries})
        )
        cells = tuple(
            sorted(
                admission_support_cells,
                key=lambda value: (value.cell_id, value.fingerprint()),
            )
        )

        supported_law_failed: list[str] = []
        supported_law_unevaluable: list[str] = []
        if qualification_result is None:
            supported_law_unevaluable.append("QUALIFICATION_RESULT_ABSENT")
        elif qualification_result.scientific_status is not ScientificStatus.SUPPORTED:
            supported_law_unevaluable.append("QUALIFICATION_RESULT_NOT_SUPPORTED")
        if response_law is None:
            supported_law_unevaluable.append("RESPONSE_LAW_ABSENT")
        if qualification_result is not None and response_law is not None:
            if qualification_result.response_law != response_law:
                supported_law_failed.append("QUALIFICATION_RESULT_LAW_MISMATCH")
            if qualification_result.chart_id != response_law.chart_id:
                supported_law_failed.append("QUALIFICATION_RESULT_CHART_MISMATCH")

        publication_failed: list[str] = []
        publication_unevaluable: list[str] = []
        if payload_evidence is None:
            publication_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        else:
            if payload_evidence.payload.object_schema != issued_spec.payload_schema:
                publication_failed.append("PAYLOAD_SCHEMA_MISMATCH")
            if payload_evidence.publication.object_schema != issued_spec.payload_publication_schema:
                publication_failed.append("PAYLOAD_PUBLICATION_SCHEMA_MISMATCH")
            if (
                payload_evidence.published_content_sha256
                != payload_evidence.payload.object_fingerprint
            ):
                publication_failed.append("PAYLOAD_PUBLICATION_CONTENT_MISMATCH")
            if qualification_result is not None:
                if qualification_result.payload_publication is None:
                    publication_unevaluable.append("PAYLOAD_PUBLICATION_ABSENT")
                elif qualification_result.payload_publication != payload_evidence.publication:
                    publication_failed.append("QUALIFICATION_PAYLOAD_PUBLICATION_MISMATCH")

        region_failed: list[str] = []
        region_unevaluable: list[str] = []
        if region_evidence is None:
            region_unevaluable.append("REGION_COMPATIBILITY_EVIDENCE_ABSENT")
        else:
            if (
                region_evidence.compatibility.object_schema
                != issued_spec.region_compatibility_schema
            ):
                region_failed.append("REGION_COMPATIBILITY_SCHEMA_MISMATCH")
            if region_evidence.region_support_spec != issued_spec.region_support_spec:
                region_failed.append("REGION_SUPPORT_SPEC_MISMATCH")
            if payload_evidence is not None:
                if region_evidence.payload != payload_evidence.payload:
                    region_failed.append("REGION_PAYLOAD_MISMATCH")
                if payload_evidence.region_compatibility != region_evidence.compatibility:
                    region_failed.append("REGION_COMPATIBILITY_IDENTITY_MISMATCH")
            if region_evidence.disposition is LocalSupportRegionDisposition.UNEVALUABLE:
                region_unevaluable.append("REGION_COMPATIBILITY_UNEVALUABLE")
            elif region_evidence.disposition is LocalSupportRegionDisposition.NOT_SUPPORTED:
                region_failed.append("REGION_COMPATIBILITY_NOT_SUPPORTED")

        denominator_failed: list[str] = []
        denominator_unevaluable: list[str] = []
        if payload_evidence is None:
            denominator_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        else:
            if payload_evidence.prepared_denominator_id != issued_spec.prepared_denominator_id:
                denominator_failed.append("PAYLOAD_PREPARED_DENOMINATOR_MISMATCH")
            if any(
                value.denominator_id != issued_spec.prepared_denominator_id
                for value in payload_evidence.action_words
            ):
                denominator_failed.append("ACTION_WORD_DENOMINATOR_MISMATCH")

        action_failed: list[str] = []
        action_unevaluable: list[str] = []
        if payload_evidence is None:
            action_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        else:
            payload_action_identities = tuple(
                ObjectIdentity.from_record(value.word_id, value)
                for value in payload_evidence.action_words
            )
            expected_actions = {value.object_id: value for value in issued_spec.action_words}
            if payload_action_identities != issued_spec.action_words:
                action_failed.append("ACTION_WORD_ROSTER_MISMATCH")
            if any(
                value.support_status is not ActionWordSupportStatus.SUPPORTED
                for value in payload_evidence.action_words
            ):
                action_failed.append("ACTION_WORD_NOT_SUPPORTED")
            if any(
                value.action_word.object_id not in expected_actions
                or value.action_word != expected_actions[value.action_word.object_id]
                for value in payload_entries
            ):
                action_failed.append("PAYLOAD_ENTRY_ACTION_WORD_FOREIGN")
            coordinate_keys = tuple(value.coordinate_key for value in payload_entries)
            if len(set(coordinate_keys)) != len(coordinate_keys):
                action_failed.append("PAYLOAD_ENTRY_COORDINATE_DUPLICATED")
            action_ids_by_local = {
                local_id: {
                    value.action_word.object_id
                    for value in payload_entries
                    if value.local_support_id == local_id
                }
                for local_id in local_ids
            }
            if any(value != spec_action_ids for value in action_ids_by_local.values()):
                action_failed.append("PAYLOAD_LOCAL_ACTION_ROSTER_MISMATCH")

        local_failed: list[str] = []
        local_unevaluable: list[str] = []
        expected_law_support = tuple(sorted((issued_spec.prepared_denominator_id, *local_ids)))
        if response_law is None:
            local_unevaluable.append("RESPONSE_LAW_ABSENT")
        elif law_support_ids != expected_law_support:
            local_failed.append("LAW_LOCAL_SUPPORT_ROSTER_MISMATCH")
        if payload_evidence is None:
            local_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        elif payload_local_support_ids != local_ids:
            local_failed.append("PAYLOAD_LOCAL_SUPPORT_ROSTER_MISMATCH")
        if issued_spec.prepared_denominator_id in payload_local_support_ids:
            local_failed.append("PAYLOAD_PREPARED_DENOMINATOR_ROW_PRESENT")

        fingerprint_failed: list[str] = []
        fingerprint_unevaluable: list[str] = []
        if payload_evidence is None:
            fingerprint_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        elif any(
            spec_fingerprints.get(value.local_support_id) != value.local_support_fingerprint
            for value in payload_entries
        ):
            fingerprint_failed.append("PAYLOAD_LOCAL_FINGERPRINT_MISMATCH")
        if region_evidence is None:
            fingerprint_unevaluable.append("REGION_COMPATIBILITY_EVIDENCE_ABSENT")
        elif region_evidence.local_supports != issued_spec.local_supports:
            fingerprint_failed.append("REGION_LOCAL_FINGERPRINT_ROSTER_MISMATCH")

        cell_ids = tuple(value.cell_id for value in cells)
        admission_roster_failed: list[str] = []
        if len(set(cell_ids)) != len(cell_ids):
            admission_roster_failed.append("ADMISSION_CELL_DUPLICATED")
        if cell_ids != local_ids:
            admission_roster_failed.append("ADMISSION_LOCAL_CELL_ROSTER_MISMATCH")

        admission_chart_failed: list[str] = []
        admission_chart_unevaluable: list[str] = []
        if response_law is None:
            admission_chart_unevaluable.append("RESPONSE_LAW_ABSENT")
        else:
            if response_law.chart_id != issued_spec.chart_id:
                admission_chart_failed.append("LAW_CHART_MISMATCH")
            if response_law.chart_id not in response_law.obligations.support.chart_ids:
                admission_chart_failed.append("LAW_CHART_OUTSIDE_SUPPORT")
            if any(value.chart_id != response_law.chart_id for value in cells):
                admission_chart_failed.append("ADMISSION_CHART_MISMATCH")

        admission_denominator_failed: list[str] = []
        if any(value.denominator_cell_id != issued_spec.prepared_denominator_id for value in cells):
            admission_denominator_failed.append("ADMISSION_PREPARED_DENOMINATOR_MISMATCH")
        if any(value.denominator_cell_id in local_id_set for value in cells):
            admission_denominator_failed.append("ADMISSION_LOCAL_USED_AS_DENOMINATOR")

        admission_bounds_failed: list[str] = []
        admission_bounds_unevaluable: list[str] = []
        if response_law is None:
            admission_bounds_unevaluable.append("RESPONSE_LAW_ABSENT")
        else:
            law_bound_set = set(law_action_bound_ids)
            if any(
                value.cell_id not in spec_bounds
                or value.action_bound_ids != spec_bounds[value.cell_id]
                for value in cells
            ):
                admission_bounds_failed.append("ADMISSION_ACTION_BOUND_ROSTER_MISMATCH")
            if any(not set(value.action_bound_ids).issubset(law_bound_set) for value in cells):
                admission_bounds_failed.append("ADMISSION_ACTION_BOUND_OUTSIDE_LAW_SUPPORT")

        coordinate_failed: list[str] = []
        coordinate_unevaluable: list[str] = []
        if payload_evidence is None:
            coordinate_unevaluable.append("PAYLOAD_EVIDENCE_ABSENT")
        else:
            if not payload_evidence.complete_entry_roster:
                coordinate_unevaluable.append("PAYLOAD_ENTRY_ROSTER_INCOMPLETE")
            if any(not value.coordinate_qualified for value in payload_entries):
                coordinate_failed.append("PAYLOAD_COORDINATE_UNQUALIFIED")
            if any(value.interpolation_used for value in payload_entries):
                coordinate_failed.append("PAYLOAD_INTERPOLATION_DETECTED")
        if region_evidence is None:
            coordinate_unevaluable.append("REGION_COMPATIBILITY_EVIDENCE_ABSENT")
        else:
            if region_evidence.interpolation_used:
                coordinate_failed.append("REGION_INTERPOLATION_DETECTED")
            if region_evidence.support_expanded:
                coordinate_failed.append("REGION_SUPPORT_EXPANSION_DETECTED")
            if region_evidence.unqualified_coordinate_ids:
                coordinate_failed.append("REGION_UNQUALIFIED_COORDINATE_DETECTED")

        checks = tuple(
            sorted(
                (
                    _check(
                        "action-word-roster",
                        failed=tuple(action_failed),
                        unevaluable=tuple(action_unevaluable),
                    ),
                    _check(
                        "exact-qualified-coordinates",
                        failed=tuple(coordinate_failed),
                        unevaluable=tuple(coordinate_unevaluable),
                    ),
                    _check(
                        "local-support-fingerprints",
                        failed=tuple(fingerprint_failed),
                        unevaluable=tuple(fingerprint_unevaluable),
                    ),
                    _check(
                        "local-support-roster",
                        failed=tuple(local_failed),
                        unevaluable=tuple(local_unevaluable),
                    ),
                    _check(
                        "admission-action-bounds",
                        failed=tuple(admission_bounds_failed),
                        unevaluable=tuple(admission_bounds_unevaluable),
                    ),
                    _check("admission-cell-roster", failed=tuple(admission_roster_failed)),
                    _check(
                        "admission-chart",
                        failed=tuple(admission_chart_failed),
                        unevaluable=tuple(admission_chart_unevaluable),
                    ),
                    _check("admission-denominator", failed=tuple(admission_denominator_failed)),
                    _check(
                        "payload-publication-identity",
                        failed=tuple(publication_failed),
                        unevaluable=tuple(publication_unevaluable),
                    ),
                    _check(
                        "prepared-denominator",
                        failed=tuple(denominator_failed),
                        unevaluable=tuple(denominator_unevaluable),
                    ),
                    _check(
                        "region-compatibility-identity",
                        failed=tuple(region_failed),
                        unevaluable=tuple(region_unevaluable),
                    ),
                    _check(
                        "supported-law-result",
                        failed=tuple(supported_law_failed),
                        unevaluable=tuple(supported_law_unevaluable),
                    ),
                ),
                key=lambda value: value.check_id,
            )
        )
        disposition = (
            PreparedDenominatorLocalSupportCompatibilityDisposition.UNEVALUABLE
            if any(
                value.status is LocalSupportCompatibilityCheckStatus.UNEVALUABLE
                for value in checks
            )
            else PreparedDenominatorLocalSupportCompatibilityDisposition.INCOMPATIBLE
            if any(
                value.status is LocalSupportCompatibilityCheckStatus.FAILED for value in checks
            )
            else PreparedDenominatorLocalSupportCompatibilityDisposition.COMPATIBLE
        )
        reason_codes = tuple(sorted({reason for value in checks for reason in value.reason_codes}))
        input_interpolation_detected = bool(
            payload_evidence is not None
            and any(value.interpolation_used for value in payload_entries)
        ) or bool(region_evidence is not None and region_evidence.interpolation_used)
        input_unqualified_coordinate_detected = bool(
            payload_evidence is not None
            and any(not value.coordinate_qualified for value in payload_entries)
        ) or bool(region_evidence is not None and region_evidence.unqualified_coordinate_ids)

        return PreparedDenominatorLocalSupportCompatibility(
            receipt_id=receipt_id,
            compatibility_spec=ObjectIdentity.from_record(issued_spec.spec_id, issued_spec),
            evaluator_implementation=issued_spec.evaluator_implementation,
            payload_evidence=(
                ObjectIdentity.from_record(payload_evidence.evidence_id, payload_evidence)
                if payload_evidence is not None
                else None
            ),
            payload=payload_evidence.payload if payload_evidence is not None else None,
            payload_publication=(
                payload_evidence.publication if payload_evidence is not None else None
            ),
            region_evidence=(
                ObjectIdentity.from_record(region_evidence.evidence_id, region_evidence)
                if region_evidence is not None
                else None
            ),
            region_compatibility=(
                region_evidence.compatibility if region_evidence is not None else None
            ),
            qualification_result=(
                ObjectIdentity.from_record(
                    qualification_result.result_id,
                    qualification_result,
                )
                if qualification_result is not None
                else None
            ),
            response_law=(
                ObjectIdentity.from_record(response_law.law_id, response_law)
                if response_law is not None
                else None
            ),
            prepared_denominator_id=issued_spec.prepared_denominator_id,
            chart_id=issued_spec.chart_id,
            local_supports=issued_spec.local_supports,
            action_words=issued_spec.action_words,
            law_support_ids=law_support_ids,
            law_action_bound_ids=law_action_bound_ids,
            payload_local_support_ids=payload_local_support_ids,
            admission_support_cells=cells,
            checks=checks,
            disposition=disposition,
            reason_codes=reason_codes,
            input_interpolation_detected=input_interpolation_detected,
            input_unqualified_coordinate_detected=input_unqualified_coordinate_detected,
        )


def evaluate_prepared_denominator_local_support_compatibility(
    *,
    receipt_id: str,
    issued_spec: PreparedDenominatorLocalSupportCompatibilitySpec,
    payload_evidence: PreparedDenominatorLocalSupportPayloadEvidence | None,
    region_evidence: PreparedDenominatorLocalSupportRegionEvidence | None,
    qualification_result: LawQualificationResult | None,
    response_law: ResponseLaw | None,
    admission_support_cells: tuple[ReceiptAdmissionSupportCell, ...],
) -> PreparedDenominatorLocalSupportCompatibility:
    """Functional entry point for the stateless compatibility evaluator."""

    return PreparedDenominatorLocalSupportCompatibilityEvaluator().evaluate(
        receipt_id=receipt_id,
        issued_spec=issued_spec,
        payload_evidence=payload_evidence,
        region_evidence=region_evidence,
        qualification_result=qualification_result,
        response_law=response_law,
        admission_support_cells=admission_support_cells,
    )


__all__ = [
    'LocalSupportCompatibilityCheckStatus',
    'LocalSupportCompatibilityCheck',
    'LocalSupportFingerprint',
    'LocalSupportPayloadEntry',
    'LocalSupportRegionDisposition',
    'PreparedDenominatorLocalSupportCompatibilityDisposition',
    'PreparedDenominatorLocalSupportCompatibilityEvaluator',
    'PreparedDenominatorLocalSupportCompatibilitySpec',
    'PreparedDenominatorLocalSupportCompatibility',
    'PreparedDenominatorLocalSupportPayloadEvidence',
    'PreparedDenominatorLocalSupportRegionEvidence',
    'evaluate_prepared_denominator_local_support_compatibility',
]
