"""Strict source, decode and physical-unit contracts for independent substrate grounding NREL held-source and predictive-unit contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
import re
from typing import ClassVar

from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateReceiverBackactionClass, IndependentSubstrateTargetPhase
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


class NRELArchiveMemberFormat(StrEnum):
    CSV_UTF8 = "CSV_UTF8"
    TSV_UTF8 = "TSV_UTF8"


class NRELOutcomeVisibilityRoute(StrEnum):
    CUSTODIAN_HELD_PARTITION = "CUSTODIAN_HELD_PARTITION"
    CRYPTOGRAPHIC_COMMITTED_DECODE_FILTER = "CRYPTOGRAPHIC_COMMITTED_DECODE_FILTER"
    RETROSPECTIVE_OUTCOME_VISIBILITY_UNCONTROLLED = "RETROSPECTIVE_OUTCOME_VISIBILITY_UNCONTROLLED"


class NRELActionKind(StrEnum):
    ACTION = "ACTION"
    HOLD = "HOLD"


class NRELColumnRole(StrEnum):
    SAMPLE_ID = "SAMPLE_ID"
    PREPARATION_ID = "PREPARATION_ID"
    RUN_ID = "RUN_ID"
    SEQUENCE_INDEX = "SEQUENCE_INDEX"
    TIMESTAMP_UTC = "TIMESTAMP_UTC"
    ACTION_KIND = "ACTION_KIND"
    ACTION_ID = "ACTION_ID"
    REQUESTED_ACTION = "REQUESTED_ACTION"
    ACTION_ACCEPTED_FLAG = "ACTION_ACCEPTED_FLAG"
    ACCEPTED_ACTION = "ACCEPTED_ACTION"
    APPLIED_ACTION = "APPLIED_ACTION"
    REALIZED_ACTION = "REALIZED_ACTION"
    REQUEST_TICK = "REQUEST_TICK"
    ACCEPTANCE_TICK = "ACCEPTANCE_TICK"
    APPLICATION_TICK = "APPLICATION_TICK"
    REALIZATION_TICK = "REALIZATION_TICK"
    AC_POWER_W = "AC_POWER_W"
    DC_POWER_W = "DC_POWER_W"
    AC_UNCERTAINTY_W = "AC_UNCERTAINTY_W"
    DC_UNCERTAINTY_W = "DC_UNCERTAINTY_W"
    TRIPPED = "TRIPPED"
    SATURATED = "SATURATED"
    OBSERVATION_VALID = "OBSERVATION_VALID"


@dataclass(frozen=True, slots=True)
class NRELArchiveMemberBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-archive-member-binding'

    member_id: str
    relative_locator: str
    content_sha256: str
    size_bytes: int
    format: NRELArchiveMemberFormat
    role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_nonempty(self.relative_locator, field_name="relative_locator")
        path = PurePosixPath(self.relative_locator)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_locator:
            raise ValueError("NREL archive member locator is unsafe")
        suffix = path.suffix.lower()
        expected = {
            NRELArchiveMemberFormat.CSV_UTF8: ".csv",
            NRELArchiveMemberFormat.TSV_UTF8: ".tsv",
        }[self.format]
        if suffix != expected:
            raise ValueError("NREL archive member suffix differs from safe format")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if self.size_bytes <= 0:
            raise ValueError("NREL archive member size must be positive")
        validate_nonempty(self.role, field_name="role")
        if self.role != "LOGGED_INTERVENTION_TABLE":
            raise ValueError("NREL selected member role is unsupported")


@dataclass(frozen=True, slots=True)
class NRELArchiveSourceBinding(CanonicalRecord):
    """Exact held physical archive bytes and public-outcome protection route."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-archive-source-binding'

    binding_id: str
    catalogue_record_id: str
    archive_sha256: str
    archive_size_bytes: int
    terms_receipt: ObjectIdentity
    acquisition_receipt: ObjectIdentity
    apparatus_id: str
    site_id: str
    members: tuple[NRELArchiveMemberBinding, ...]
    outcome_visibility_route: NRELOutcomeVisibilityRoute
    prior_receiver_access_investigator_ids: tuple[str, ...]
    prospective_prediction_eligible: bool
    source_network_required: bool
    apparatus_site_count: int
    ac_receiver_backaction: IndependentSubstrateReceiverBackactionClass
    dc_receiver_backaction: IndependentSubstrateReceiverBackactionClass

    def __post_init__(self) -> None:
        for name in ("binding_id", "catalogue_record_id", "apparatus_id", "site_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.archive_sha256, field_name="archive_sha256")
        if self.archive_size_bytes <= 0:
            raise ValueError("NREL archive size must be positive")
        require_sorted_unique_ids(self.members, attribute="member_id", field_name="members")
        if not self.members:
            raise ValueError("NREL source binding requires a safe selected member")
        require_sorted_unique_strings(
            self.prior_receiver_access_investigator_ids,
            field_name="prior_receiver_access_investigator_ids",
        )
        controlled = self.outcome_visibility_route is not (
            NRELOutcomeVisibilityRoute.RETROSPECTIVE_OUTCOME_VISIBILITY_UNCONTROLLED
        )
        if self.prospective_prediction_eligible != controlled:
            raise ValueError("NREL prediction eligibility differs from visibility route")
        if self.source_network_required:
            raise ValueError("NREL analysis must use held offline source bytes")
        if self.apparatus_site_count != 1:
            raise ValueError("independent substrate grounding NREL archive has an exact one-apparatus/site ceiling")


@dataclass(frozen=True, slots=True)
class NRELColumnBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-column-binding'

    binding_id: str
    role: NRELColumnRole
    header: str
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_nonempty(self.header, field_name="header")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class NRELSafeDecodeProfile(CanonicalRecord):
    """Exact bounded table decoder and outcome partition, frozen before decode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-safe-decode-profile'

    profile_id: str
    source_binding: ObjectIdentity
    member: ObjectIdentity
    phase: IndependentSubstrateTargetPhase
    expected_headers: tuple[str, ...]
    column_bindings: tuple[NRELColumnBinding, ...]
    included_preparation_ids: tuple[str, ...]
    maximum_payload_bytes: int
    maximum_rows: int
    maximum_columns: int
    maximum_cell_bytes: int
    delimiter: str
    encoding: str
    executable_or_object_deserialization_allowed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        if self.source_binding.object_schema != NRELArchiveSourceBinding.SCHEMA:
            raise ValueError("NREL decode profile source identity differs")
        if self.member.object_schema != NRELArchiveMemberBinding.SCHEMA:
            raise ValueError("NREL decode profile member identity differs")
        require_sorted_unique_ids(
            self.column_bindings,
            attribute="binding_id",
            field_name="column_bindings",
        )
        if {value.role for value in self.column_bindings} != set(NRELColumnRole):
            raise ValueError("NREL decode profile column-role roster differs")
        headers = tuple(value.header for value in self.column_bindings)
        if len(set(headers)) != len(headers) or set(headers) != set(self.expected_headers):
            raise ValueError("NREL decode profile headers differ from column bindings")
        if not self.expected_headers or len(set(self.expected_headers)) != len(
            self.expected_headers
        ):
            raise ValueError("NREL expected headers must be unique")
        require_sorted_unique_strings(
            self.included_preparation_ids,
            field_name="included_preparation_ids",
            allow_empty=False,
        )
        if (
            min(
                self.maximum_payload_bytes,
                self.maximum_rows,
                self.maximum_columns,
                self.maximum_cell_bytes,
            )
            <= 0
        ):
            raise ValueError("NREL decode limits must be positive")
        if self.maximum_columns != len(self.expected_headers):
            raise ValueError("NREL maximum column count must equal the exact header roster")
        if self.delimiter not in {",", "\t"} or self.encoding != "utf-8":
            raise ValueError("NREL decoder supports only exact UTF-8 CSV/TSV")
        if self.executable_or_object_deserialization_allowed:
            raise ValueError("NREL decoder cannot deserialize executable/object payloads")
        expected_access = {
            IndependentSubstrateTargetPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            IndependentSubstrateTargetPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }.get(self.phase)
        if expected_access is None:
            raise ValueError("NREL retrospective archive cannot define controller use")
        if self.outcome_access is not expected_access:
            raise ValueError("NREL decode outcome access differs from phase")


@dataclass(frozen=True, slots=True)
class NRELNativeSample(CanonicalRecord):
    """One nested logged intervention/measurement row in native physical units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-native-sample'

    sample_id: str
    preparation_id: str
    run_id: str
    sequence_index: int
    timestamp_utc: str
    action_kind: NRELActionKind
    action_id: str
    requested_action_code: str
    action_accepted: bool
    accepted_action_code: str | None
    applied_action_code: str | None
    realized_action_code: str | None
    request_tick: int
    acceptance_tick: int | None
    application_tick: int | None
    realization_tick: int | None
    ac_power_w: Decimal | None
    dc_power_w: Decimal | None
    ac_uncertainty_w: Decimal | None
    dc_uncertainty_w: Decimal | None
    tripped: bool
    saturated: bool
    observation_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("sample_id", "preparation_id", "run_id", "action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.sequence_index < 0:
            raise ValueError("NREL sequence index must be nonnegative")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.timestamp_utc) is None:
            raise ValueError("NREL timestamp must be canonical UTC seconds")
        validate_nonempty(self.requested_action_code, field_name="requested_action_code")
        for name in (
            "accepted_action_code",
            "applied_action_code",
            "realized_action_code",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, field_name=name)
        if self.action_kind is NRELActionKind.HOLD and self.requested_action_code != "HOLD":
            raise ValueError("NREL hold sample must request native HOLD")
        if not self.action_accepted and any(
            value is not None
            for value in (
                self.accepted_action_code,
                self.applied_action_code,
                self.realized_action_code,
                self.acceptance_tick,
                self.application_tick,
                self.realization_tick,
            )
        ):
            raise ValueError("unaccepted NREL action cannot claim later semantics")
        ticks = (
            self.request_tick,
            self.acceptance_tick,
            self.application_tick,
            self.realization_tick,
        )
        if self.request_tick < 0 or any(value is not None and value < 0 for value in ticks):
            raise ValueError("NREL action ticks must be nonnegative")
        present = tuple(value for value in ticks if value is not None)
        if tuple(sorted(present)) != present:
            raise ValueError("NREL action clocks violate causal order")
        for name in (
            "ac_power_w",
            "dc_power_w",
            "ac_uncertainty_w",
            "dc_uncertainty_w",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(
                    value,
                    field_name=name,
                    minimum=(Decimal(0) if "uncertainty" in name else None),
                )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.observation_valid:
            if (
                self.ac_power_w is None
                or self.dc_power_w is None
                or self.ac_uncertainty_w is None
                or self.dc_uncertainty_w is None
                or self.saturated
                or self.reason_codes
            ):
                raise ValueError("valid NREL observation lacks bounded receiver values")
        elif not self.reason_codes:
            raise ValueError("invalid NREL observation requires a typed reason")


@dataclass(frozen=True, slots=True)
class NRELPreparationTrace(CanonicalRecord):
    """One physical preparation/run unit; samples and channels remain nested views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-preparation-trace'

    trace_id: str
    preparation_id: str
    run_id: str
    apparatus_id: str
    site_id: str
    samples: tuple[NRELNativeSample, ...]
    scientific_unit_count: int

    def __post_init__(self) -> None:
        for name in (
            "trace_id",
            "preparation_id",
            "run_id",
            "apparatus_id",
            "site_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        sample_ids = tuple(value.sample_id for value in self.samples)
        if not self.samples or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("NREL preparation trace requires unique samples")
        sequence_indexes = tuple(value.sequence_index for value in self.samples)
        if sequence_indexes != tuple(sorted(set(sequence_indexes))):
            raise ValueError("NREL preparation samples must be sequence ordered and unique")
        if any(
            value.preparation_id != self.preparation_id or value.run_id != self.run_id
            for value in self.samples
        ):
            raise ValueError("NREL preparation trace mixes physical units")
        if self.scientific_unit_count != 1:
            raise ValueError("one NREL preparation trace must count as one unit")


@dataclass(frozen=True, slots=True)
class NRELArchiveDecodeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-archive-decode-result'

    result_id: str
    source_binding: ObjectIdentity
    decode_profile: ObjectIdentity
    member_predecode_sha256: str
    member_postdecode_sha256: str
    scanned_row_count: int
    selected_row_count: int
    traces: tuple[NRELPreparationTrace, ...]
    scientific_unit_count: int
    outcome_access: OutcomeAccess
    sealed_evaluation_decode_authorized: bool
    evaluator_reveal_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.source_binding.object_schema != NRELArchiveSourceBinding.SCHEMA:
            raise ValueError("NREL decode result source differs")
        if self.decode_profile.object_schema != NRELSafeDecodeProfile.SCHEMA:
            raise ValueError("NREL decode result profile differs")
        for name in ("member_predecode_sha256", "member_postdecode_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.member_predecode_sha256 != self.member_postdecode_sha256:
            raise ValueError("NREL member changed during decode")
        if not 0 <= self.selected_row_count <= self.scanned_row_count:
            raise ValueError("NREL decode row accounting is invalid")
        require_sorted_unique_ids(self.traces, attribute="trace_id", field_name="traces")
        if self.scientific_unit_count != len(self.traces):
            raise ValueError("NREL scientific units must equal preparation traces")
        if self.outcome_access is OutcomeAccess.EVALUATION_SEALED:
            if not (self.sealed_evaluation_decode_authorized or self.evaluator_reveal_authorized):
                raise ValueError("sealed NREL result requires sealed decode authority")
        elif self.sealed_evaluation_decode_authorized or self.evaluator_reveal_authorized:
            raise ValueError("development NREL result cannot claim sealed/reveal authority")


__all__ = [
    'NRELActionKind',
    'NRELArchiveDecodeResult',
    'NRELArchiveMemberBinding',
    'NRELArchiveMemberFormat',
    'NRELArchiveSourceBinding',
    'NRELColumnBinding',
    'NRELColumnRole',
    'NRELNativeSample',
    'NRELOutcomeVisibilityRoute',
    'NRELPreparationTrace',
    'NRELSafeDecodeProfile',
]
