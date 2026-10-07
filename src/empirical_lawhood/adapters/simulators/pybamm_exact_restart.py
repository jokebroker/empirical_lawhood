"""Bounded PyBaMM battery exact restart exact-restart reconstruction qualification.

battery exact restart is a numerical-source qualification selected by the terminal battery electrothermal result.
It does not identify a new response law, construct an admission chart, or
enter admission or controller use.  The adapter owns only the frozen preparation/history chart,
lossless numeric checkpoint capsule, finite reconstruction routes, receiver
comparison, and pure adjudication.  Storage, issue, execution authority, and
reveal remain outer concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
from hashlib import sha256
import math
import re
import time
from typing import Any, ClassVar, Mapping, Sequence, TypedDict

import numpy as np

from empirical_lawhood.adapters.simulators import pybamm_electrothermal_hierarchy as battery_electrothermal
from empirical_lawhood.adapters.simulators.staged_hybrid_response import StagedHybridResponseSourceFile
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp


BATTERY_EXACT_RESTART_PLAN_ID = "battery-exact-restart-reconstruction-qualification"
BATTERY_EXACT_RESTART_DESIGN_ID = "design.battery-exact-restart.qualification"
BATTERY_EXACT_RESTART_PROVIDER_KEY = "open-sim.battery-exact-restart"
BATTERY_EXACT_RESTART_SOURCE_VERSION = battery_electrothermal.BATTERY_ELECTROTHERMAL_SOURCE_VERSION
BATTERY_EXACT_RESTART_EXTERNAL_ROOT = "runs/battery-exact-restart-qualification"
BATTERY_EXACT_RESTART_LOCALIZATION_SEED = 2_026_072_9091
BATTERY_EXACT_RESTART_QUALIFICATION_SEED = 2_026_072_9092
BATTERY_EXACT_RESTART_RESERVE_SEED = 2_026_072_9093
BATTERY_EXACT_RESTART_MINIMUM_FREE_BYTES = 100 * 1024**3
BATTERY_EXACT_RESTART_CAPACITY_FLOOR_AH = Decimal("1e-10")
BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K = Decimal("1e-8")
BATTERY_EXACT_RESTART_VOLTAGE_FLOOR_V = Decimal("1e-8")
BATTERY_EXACT_RESTART_MAXIMUM_RECORD_BYTES = 64 * 1024**2
_Q = Decimal("0.0000001")


class BatteryExactRestartStage(StrEnum):
    CANARY = "CANARY"
    LOCALIZATION = "LOCALIZATION"
    QUALIFICATION = "QUALIFICATION"


class BatteryExactRestartUnitRole(StrEnum):
    CANARY = "CANARY"
    LOCALIZATION = "LOCALIZATION"
    QUALIFICATION_PRIMARY = "QUALIFICATION_PRIMARY"
    QUALIFICATION_RESERVE = "QUALIFICATION_RESERVE"


class BatteryExactRestartRoute(StrEnum):
    NATIVE_SAME_OBJECT = "NATIVE-SAME-OBJECT"
    DECIMAL15_LIVE_PREFIX_MAP = "DECIMAL15-LIVE-PREFIX-MAP"
    LOSSLESS_LIVE_PREFIX_MAP = "LOSSLESS-LIVE-PREFIX-MAP"
    LOSSLESS_TARGET_DIRECT = "LOSSLESS-TARGET-DIRECT"
    LOSSLESS_PUBLIC_MAP = "LOSSLESS-PUBLIC-MAP"
    LOSSLESS_PRECONSISTENT = "LOSSLESS-PRECONSISTENT"


BATTERY_EXACT_RESTART_ROUTE_ROSTER = tuple(BatteryExactRestartRoute)
BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE = (
    BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP,
    BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT,
    BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT,
)


class BatteryExactRestartRouteFeasibility(StrEnum):
    FEASIBLE = "FEASIBLE"
    TYPED_EXCLUSION = "TYPED_EXCLUSION"
    UNTRUSTWORTHY_CAPABILITY = "UNTRUSTWORTHY_CAPABILITY"


class _RouteSemantics(TypedDict):
    api_status: str
    supported_denominator_ids: tuple[str, ...]
    consumes_state_y: bool
    consumes_derivative_yp: bool
    consumes_inputs: bool
    consumes_model_identity: bool
    consumes_time: bool
    consistency_initialization_behavior: str
    model_mapping_behavior: str
    process_assumptions: tuple[str, ...]


class _RouteResultCommon(TypedDict):
    result_id: str
    config: ObjectIdentity
    capture: ObjectIdentity
    unit_id: str
    denominator_id: str
    history_id: str
    route: BatteryExactRestartRoute
    outcome_access: OutcomeAccess


class BatteryExactRestartCaptureDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    SOURCE_DENOMINATOR_FAILURE = "SOURCE_DENOMINATOR_FAILURE"
    CAPSULE_INVALID = "CAPSULE_INVALID"
    UNEVALUABLE_IDENTITY = "UNEVALUABLE_IDENTITY"


class BatteryExactRestartRouteDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    RECEIVER_OPPOSED = "RECEIVER_OPPOSED"
    SOURCE_ROUTE_FAILURE = "SOURCE_ROUTE_FAILURE"
    CAPSULE_INVALID = "CAPSULE_INVALID"
    TYPED_EXCLUSION = "TYPED_EXCLUSION"
    UNEVALUABLE_IDENTITY = "UNEVALUABLE_IDENTITY"


class BatteryExactRestartMechanismDisposition(StrEnum):
    NUMERIC_ENCODING_SENSITIVE = "NUMERIC_ENCODING_SENSITIVE"
    MODEL_MAPPING_SENSITIVE = "MODEL_MAPPING_SENSITIVE"
    CONSISTENCY_INITIALIZATION_SENSITIVE = "CONSISTENCY_INITIALIZATION_SENSITIVE"
    MULTIFACTOR_OR_UNRESOLVED = "MULTIFACTOR_OR_UNRESOLVED"
    OBSTRUCTION_NOT_REPRODUCED = "OBSTRUCTION_NOT_REPRODUCED"
    SOURCE_API_INSUFFICIENT = "SOURCE_API_INSUFFICIENT"


class BatteryExactRestartLocalizationVerdict(StrEnum):
    ROUTE_SELECTED = "BATTERY_EXACT_RESTART_LOCALIZATION_ROUTE_SELECTED"
    OBSTRUCTION_NOT_REPRODUCED = "BATTERY_EXACT_RESTART_OBSTRUCTION_NOT_REPRODUCED"
    SOURCE_ROUTE_INSUFFICIENT = "BATTERY_EXACT_RESTART_SOURCE_ROUTE_INSUFFICIENT"
    TECHNICAL_PARTIAL = "BATTERY_EXACT_RESTART_TECHNICAL_PARTIAL"
    UNEVALUABLE = "BATTERY_EXACT_RESTART_UNEVALUABLE"


class BatteryExactRestartQualificationVerdict(StrEnum):
    QUALIFIED_ALL = "BATTERY_EXACT_RESTART_EXACT_RECONSTRUCTION_QUALIFIED_ALL_ENTERED_DENOMINATORS"
    DENOMINATOR_CONDITIONAL = "BATTERY_EXACT_RESTART_EXACT_RECONSTRUCTION_DENOMINATOR_CONDITIONAL"
    NOT_QUALIFIED = "BATTERY_EXACT_RESTART_EXACT_RECONSTRUCTION_NOT_QUALIFIED"
    TECHNICAL_PARTIAL = "BATTERY_EXACT_RESTART_TECHNICAL_PARTIAL"
    UNEVALUABLE = "BATTERY_EXACT_RESTART_UNEVALUABLE"


class BatteryExactRestartClosureClass(StrEnum):
    CLOSED = "EXACT_RECONSTRUCTION_CLOSED"
    OPPOSED = "EXACT_RECONSTRUCTION_OPPOSED"
    UNEVALUABLE = "EXACT_RECONSTRUCTION_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class BatteryExactRestartPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-preparation'

    unit_id: str
    stage: BatteryExactRestartStage
    role: BatteryExactRestartUnitRole
    stratum_id: str
    initial_soc: Decimal
    initial_temperature_k: Decimal
    reserve: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        validate_decimal(
            self.initial_soc,
            field_name="initial_soc",
            minimum=Decimal("0.20"),
        )
        validate_decimal(
            self.initial_temperature_k,
            field_name="initial_temperature_k",
            minimum=Decimal("285"),
        )
        if self.initial_soc > Decimal("0.85") or self.initial_temperature_k > Decimal("305"):
            raise ValueError("battery exact restart preparation leaves the frozen support")
        if self.reserve is not (self.role is BatteryExactRestartUnitRole.QUALIFICATION_RESERVE):
            raise ValueError("battery exact restart reserve flag differs from the unit role")
        expected_stage = (
            BatteryExactRestartStage.CANARY
            if self.role is BatteryExactRestartUnitRole.CANARY
            else (
                BatteryExactRestartStage.LOCALIZATION
                if self.role is BatteryExactRestartUnitRole.LOCALIZATION
                else BatteryExactRestartStage.QUALIFICATION
            )
        )
        if self.stage is not expected_stage:
            raise ValueError("battery exact restart stage differs from the unit role")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartHistory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-history'

    history_id: str
    word: battery_electrothermal.BatteryElectrothermalActionWord
    checkpoint_s: int
    continuation_end_s: int

    def __post_init__(self) -> None:
        validate_stable_id(self.history_id, field_name="history_id")
        expected_id = (
            "history.battery-exact-restart."
            + self.word.word_id.removeprefix("word.battery-electrothermal.")
            + f".{self.checkpoint_s}.{self.continuation_end_s}"
        )
        if (
            self.history_id != expected_id
            or self.checkpoint_s not in {300, 450}
            or self.continuation_end_s - self.checkpoint_s != 300
        ):
            raise ValueError("battery exact restart history differs from the frozen chart")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartExactTolerance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-exact-tolerance'

    capacity_ah: Decimal
    mean_temperature_k: Decimal
    maximum_temperature_k: Decimal
    terminal_voltage_v: Decimal

    def __post_init__(self) -> None:
        expected = (
            BATTERY_EXACT_RESTART_CAPACITY_FLOOR_AH,
            BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K,
            BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K,
            BATTERY_EXACT_RESTART_VOLTAGE_FLOOR_V,
        )
        if (
            self.capacity_ah,
            self.mean_temperature_k,
            self.maximum_temperature_k,
            self.terminal_voltage_v,
        ) != expected:
            raise ValueError("battery exact restart exact floors differ from battery electrothermal")


def exact_tolerance() -> BatteryExactRestartExactTolerance:
    return BatteryExactRestartExactTolerance(
        capacity_ah=BATTERY_EXACT_RESTART_CAPACITY_FLOOR_AH,
        mean_temperature_k=BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K,
        maximum_temperature_k=BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K,
        terminal_voltage_v=BATTERY_EXACT_RESTART_VOLTAGE_FLOOR_V,
    )


@dataclass(frozen=True, slots=True)
class BatteryExactRestartDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-design'

    design_id: str
    localization_units: tuple[BatteryExactRestartPreparation, ...]
    qualification_units: tuple[BatteryExactRestartPreparation, ...]
    reserve_units: tuple[BatteryExactRestartPreparation, ...]
    all_denominators: tuple[battery_electrothermal.BatteryElectrothermalDenominator, ...]
    localization_denominator_ids: tuple[str, ...]
    histories: tuple[BatteryExactRestartHistory, ...]
    route_roster: tuple[BatteryExactRestartRoute, ...]
    tolerance: BatteryExactRestartExactTolerance
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        require_sorted_unique_strings(
            self.localization_denominator_ids,
            field_name="localization_denominator_ids",
        )
        all_units = (*self.localization_units, *self.qualification_units, *self.reserve_units)
        ids = tuple(item.unit_id for item in all_units)
        coordinates = tuple((item.initial_soc, item.initial_temperature_k) for item in all_units)
        denominator_ids = tuple(item.denominator_id for item in self.all_denominators)
        history_ids = tuple(item.history_id for item in self.histories)
        if (
            self.design_id != BATTERY_EXACT_RESTART_DESIGN_ID
            or len(self.localization_units) != 4
            or len(self.qualification_units) != 6
            or len(self.reserve_units) != 6
            or len(set(ids)) != 16
            or len(set(coordinates)) != 16
            or denominator_ids != tuple(sorted(denominator_ids))
            or len(denominator_ids) != 8
            or not set(self.localization_denominator_ids).issubset(denominator_ids)
            or len(self.localization_denominator_ids) != 4
            or history_ids != tuple(sorted(history_ids))
            or len(history_ids) != 5
            or self.route_roster != BATTERY_EXACT_RESTART_ROUTE_ROSTER
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart design differs from the frozen finite chart")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartRouteAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-route-audit'

    route: BatteryExactRestartRoute
    feasibility: BatteryExactRestartRouteFeasibility
    source_symbols: tuple[str, ...]
    consumed_operands: tuple[str, ...]
    capsule_only_eligible: bool
    reason: str
    api_status: str
    supported_denominator_ids: tuple[str, ...]
    consumes_state_y: bool
    consumes_derivative_yp: bool
    consumes_inputs: bool
    consumes_model_identity: bool
    consumes_time: bool
    consistency_initialization_behavior: str
    model_mapping_behavior: str
    process_assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.source_symbols, field_name="source_symbols")
        require_sorted_unique_strings(
            self.consumed_operands,
            field_name="consumed_operands",
        )
        require_sorted_unique_strings(
            self.supported_denominator_ids,
            field_name="supported_denominator_ids",
        )
        require_sorted_unique_strings(
            self.process_assumptions,
            field_name="process_assumptions",
        )
        if (
            not self.reason
            or self.api_status
            not in {
                "PUBLIC_API",
                "PUBLIC_API_WITH_PINNED_INTERNAL_BRANCH_SEMANTICS",
            }
            or len(self.supported_denominator_ids) != 8
            or not self.consistency_initialization_behavior
            or not self.model_mapping_behavior
            or not self.process_assumptions
        ):
            raise ValueError("battery exact restart route audit semantics are incomplete")
        if (
            self.route
            in {
                BatteryExactRestartRoute.NATIVE_SAME_OBJECT,
                BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP,
                BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP,
            }
            and self.capsule_only_eligible
        ):
            raise ValueError("battery exact restart historical/native route cannot enter qualification")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartSourceAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-source-audit'

    audit_id: str
    source_version: str
    distribution_record_sha256: str
    python_version: str
    platform_identity: str
    numpy_version: str
    casadi_version: str
    source_files: tuple[StagedHybridResponseSourceFile, ...]
    semantic_facts: tuple[str, ...]
    route_audits: tuple[BatteryExactRestartRouteAudit, ...]
    audited_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        validate_sha256(
            self.distribution_record_sha256,
            field_name="distribution_record_sha256",
        )
        parse_utc_timestamp(self.audited_at_utc, field_name="audited_at_utc")
        require_sorted_unique_strings(self.semantic_facts, field_name="semantic_facts")
        paths = tuple(item.relative_path for item in self.source_files)
        routes = tuple(item.route for item in self.route_audits)
        if (
            self.source_version != BATTERY_EXACT_RESTART_SOURCE_VERSION
            or not self.python_version
            or not self.platform_identity
            or not self.numpy_version
            or not self.casadi_version
            or paths != tuple(sorted(paths))
            or routes != tuple(BatteryExactRestartRoute)
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart source audit is incomplete or unordered")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartStageConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-stage-config'

    config_id: str
    design: ObjectIdentity
    stage: BatteryExactRestartStage
    units: tuple[BatteryExactRestartPreparation, ...]
    primary_unit_ids: tuple[str, ...]
    reserve_unit_ids: tuple[str, ...]
    denominators: tuple[battery_electrothermal.BatteryElectrothermalDenominator, ...]
    histories: tuple[BatteryExactRestartHistory, ...]
    routes: tuple[BatteryExactRestartRoute, ...]
    selected_route: BatteryExactRestartRoute | None
    tolerance: BatteryExactRestartExactTolerance
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.primary_unit_ids,
            field_name="primary_unit_ids",
        )
        require_sorted_unique_strings(
            self.reserve_unit_ids,
            field_name="reserve_unit_ids",
        )
        unit_ids = {item.unit_id for item in self.units}
        denominator_ids = tuple(item.denominator_id for item in self.denominators)
        history_ids = tuple(item.history_id for item in self.histories)
        if (
            not set(self.primary_unit_ids).issubset(unit_ids)
            or not set(self.reserve_unit_ids).issubset(unit_ids)
            or set(self.primary_unit_ids) & set(self.reserve_unit_ids)
            or denominator_ids != tuple(sorted(denominator_ids))
            or history_ids != tuple(sorted(history_ids))
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart stage config has incompatible identities")
        if self.stage is BatteryExactRestartStage.CANARY:
            if (
                len(self.primary_unit_ids) != 4
                or self.reserve_unit_ids
                or len(self.denominators) != 8
                or self.routes != BATTERY_EXACT_RESTART_ROUTE_ROSTER
                or self.selected_route is not None
                or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            ):
                raise ValueError("battery exact restart canary config differs from the source panel")
        elif self.stage is BatteryExactRestartStage.LOCALIZATION:
            if (
                len(self.primary_unit_ids) != 4
                or self.reserve_unit_ids
                or len(self.denominators) != 4
                or self.routes != BATTERY_EXACT_RESTART_ROUTE_ROSTER
                or self.selected_route is not None
                or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            ):
                raise ValueError("battery exact restart localization config differs from the design")
        elif (
            len(self.primary_unit_ids) != 6
            or len(self.reserve_unit_ids) != 6
            or len(self.denominators) != 8
            or self.selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or self.routes != (BatteryExactRestartRoute.NATIVE_SAME_OBJECT, self.selected_route)
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("battery exact restart qualification config differs from the design")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartResourceEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-resource-envelope'

    envelope_id: str
    stage: BatteryExactRestartStage
    worker_process_count: int
    observed_peak_worker_rss_bytes: int
    coordinator_reserve_bytes: int
    available_memory_bytes_at_freeze: int
    maximum_record_bytes: int
    minimum_free_external_bytes: int
    expected_capture_count: int
    expected_route_result_count: int
    expected_maximum_wall_seconds: Decimal
    expected_maximum_output_bytes: int
    canary_identity: ObjectIdentity
    frozen_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")
        validate_decimal(
            self.expected_maximum_wall_seconds,
            field_name="expected_maximum_wall_seconds",
            minimum=Decimal(0),
        )
        if (
            self.worker_process_count not in {1, 2, 4}
            or self.observed_peak_worker_rss_bytes <= 0
            or self.coordinator_reserve_bytes <= 0
            or self.available_memory_bytes_at_freeze <= 0
            or self.maximum_record_bytes < 1024**2
            or self.minimum_free_external_bytes < BATTERY_EXACT_RESTART_MINIMUM_FREE_BYTES
            or self.expected_capture_count <= 0
            or self.expected_route_result_count <= 0
            or self.expected_maximum_output_bytes <= 0
        ):
            raise ValueError("battery exact restart resource envelope is invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartRunIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-run-identity'

    run_id: str
    stage: BatteryExactRestartStage
    git_commit: str
    config: ObjectIdentity
    source_audit: ObjectIdentity
    implementation: ObjectIdentity
    resource_envelope: ObjectIdentity | None
    issue: ObjectIdentity
    prerequisite: ObjectIdentity | None
    created_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.run_id, field_name="run_id")
        parse_utc_timestamp(self.created_at_utc, field_name="created_at_utc")
        expected_access = (
            OutcomeAccess.EVALUATION_SEALED
            if self.stage is BatteryExactRestartStage.QUALIFICATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        )
        if (
            re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None
            or (self.resource_envelope is None and self.stage is not BatteryExactRestartStage.CANARY)
            or self.outcome_access is not expected_access
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart run identity is invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartExecutionBinding(CanonicalRecord):
    """Finite issue/run/source binding embedded into every checkpoint capsule."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-execution-binding'

    binding_id: str
    plan_id: str
    external_root: str
    config: ObjectIdentity
    issue: ObjectIdentity
    run: ObjectIdentity
    source_audit: ObjectIdentity
    implementation: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.plan_id, field_name="plan_id")
        if (
            self.plan_id != BATTERY_EXACT_RESTART_PLAN_ID
            or self.external_root != BATTERY_EXACT_RESTART_EXTERNAL_ROOT
            or self.outcome_access
            not in {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
            }
        ):
            raise ValueError("battery exact restart execution binding is invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartCanaryReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-canary-report'

    report_id: str
    source_audit: ObjectIdentity
    implementation: ObjectIdentity
    canary_config: ObjectIdentity
    built_denominator_ids: tuple[str, ...]
    route_panels: tuple[BatteryExactRestartRoutePanel, ...]
    repeated_route_ids: tuple[BatteryExactRestartRoute, ...]
    repeat_count_per_route: int
    deterministic_repeat_passed: bool
    observed_peak_worker_rss_bytes: int
    maximum_capture_size_bytes: int
    maximum_route_size_bytes: int
    maximum_capture_runtime_seconds: Decimal
    maximum_route_runtime_seconds: Decimal
    safe_worker_counts: tuple[int, ...]
    available_memory_bytes: int
    filesystem_type: str
    completed_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_strings(
            self.built_denominator_ids,
            field_name="built_denominator_ids",
        )
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")
        routes = tuple(item.route for item in self.route_panels)
        if (
            len(self.built_denominator_ids) != 8
            or routes != tuple(BatteryExactRestartRoute)
            or self.repeated_route_ids != BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or self.repeat_count_per_route != 3
            or not self.deterministic_repeat_passed
            or self.observed_peak_worker_rss_bytes <= 0
            or self.maximum_capture_size_bytes <= 0
            or self.maximum_route_size_bytes <= 0
            or self.maximum_capture_runtime_seconds <= 0
            or self.maximum_route_runtime_seconds <= 0
            or self.safe_worker_counts not in {(1,), (1, 2), (1, 2, 4)}
            or self.available_memory_bytes <= 0
            or not self.filesystem_type
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart canary report is incomplete")


def _validate_hex_tuple(values: tuple[str, ...], *, field_name: str) -> None:
    for value in values:
        try:
            decoded = float.fromhex(value)
        except ValueError as error:
            raise ValueError(f"{field_name} is not canonical float hex") from error
        if not math.isfinite(decoded) or decoded.hex() != value:
            raise ValueError(f"{field_name} is not finite canonical float hex")


def _numeric_payload_sha256(
    state_hex: tuple[str, ...],
    derivative_hex: tuple[str, ...],
    input_hex: tuple[tuple[str, str], ...],
) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "derivative_hex": derivative_hex,
                "input_hex": input_hex,
                "state_hex": state_hex,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BatteryExactRestartInputLedger(CanonicalRecord):
    """Requested-to-realized simulator input identity at one causal boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-input-ledger'

    input_name: str
    requested_hex: str
    accepted_hex: str
    applied_hex: str
    realized_hex: str

    def __post_init__(self) -> None:
        if not self.input_name:
            raise ValueError("battery exact restart input ledger name is absent")
        values = (
            self.requested_hex,
            self.accepted_hex,
            self.applied_hex,
            self.realized_hex,
        )
        _validate_hex_tuple(values, field_name="input_ledger")
        if len(set(values)) != 1:
            raise ValueError("battery exact restart simulator input ledger changed an exact input")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartCheckpointCapsule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-checkpoint-capsule'

    capsule_id: str
    execution_binding: BatteryExactRestartExecutionBinding
    source_audit: ObjectIdentity
    unit: ObjectIdentity
    denominator: ObjectIdentity
    history: ObjectIdentity
    checkpoint_s: int
    source_version: str
    distribution_record_sha256: str
    python_version: str
    platform_identity: str
    numpy_version: str
    casadi_version: str
    parameter_set: str
    evaluated_parameter_fingerprint_sha256: str
    model_fingerprint_sha256: str
    model_options_fingerprint_sha256: str
    geometry_mesh_discretization_fingerprint_sha256: str
    solver_fingerprint_sha256: str
    state_layout_sha256: str
    observation_clock_fingerprint_sha256: str
    receiver_schema_fingerprint_sha256: str
    state_vector_length: int
    differential_state_length: int
    algebraic_state_length: int
    state_hex: tuple[str, ...]
    derivative_hex: tuple[str, ...]
    source_input_hex: tuple[tuple[str, str], ...]
    checkpoint_input_ledger: tuple[BatteryExactRestartInputLedger, ...]
    continuation_input_ledger: tuple[BatteryExactRestartInputLedger, ...]
    continuation_time_origin_s: int
    continuation_end_s: int
    prefix_termination: str
    prefix_valid: bool
    route_compatible_ids: tuple[BatteryExactRestartRoute, ...]
    numeric_payload_sha256: str
    bit_exact_roundtrip: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.capsule_id, field_name="capsule_id")
        for name, value in (
            ("distribution_record_sha256", self.distribution_record_sha256),
            (
                "evaluated_parameter_fingerprint_sha256",
                self.evaluated_parameter_fingerprint_sha256,
            ),
            ("model_fingerprint_sha256", self.model_fingerprint_sha256),
            (
                "model_options_fingerprint_sha256",
                self.model_options_fingerprint_sha256,
            ),
            (
                "geometry_mesh_discretization_fingerprint_sha256",
                self.geometry_mesh_discretization_fingerprint_sha256,
            ),
            ("solver_fingerprint_sha256", self.solver_fingerprint_sha256),
            ("state_layout_sha256", self.state_layout_sha256),
            (
                "observation_clock_fingerprint_sha256",
                self.observation_clock_fingerprint_sha256,
            ),
            (
                "receiver_schema_fingerprint_sha256",
                self.receiver_schema_fingerprint_sha256,
            ),
            ("numeric_payload_sha256", self.numeric_payload_sha256),
        ):
            validate_sha256(value, field_name=name)
        _validate_hex_tuple(self.state_hex, field_name="state_hex")
        _validate_hex_tuple(self.derivative_hex, field_name="derivative_hex")
        input_names = tuple(item[0] for item in self.source_input_hex)
        checkpoint_names = tuple(item.input_name for item in self.checkpoint_input_ledger)
        continuation_names = tuple(item.input_name for item in self.continuation_input_ledger)
        if input_names != tuple(sorted(input_names)) or len(input_names) != len(set(input_names)):
            raise ValueError("battery exact restart capsule inputs are not sorted and unique")
        if (
            checkpoint_names != tuple(sorted(checkpoint_names))
            or checkpoint_names != input_names
            or continuation_names != tuple(sorted(continuation_names))
            or continuation_names != input_names
        ):
            raise ValueError("battery exact restart capsule input ledgers are incomplete")
        _validate_hex_tuple(
            tuple(item[1] for item in self.source_input_hex),
            field_name="source_input_hex",
        )
        if (
            self.source_version != BATTERY_EXACT_RESTART_SOURCE_VERSION
            or self.parameter_set != battery_electrothermal.BATTERY_ELECTROTHERMAL_PARAMETER_SET
            or self.checkpoint_s not in {300, 450}
            or self.continuation_time_origin_s != self.checkpoint_s
            or self.continuation_end_s - self.checkpoint_s != 300
            or self.prefix_termination != "final time"
            or not self.prefix_valid
            or self.route_compatible_ids != BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or self.state_vector_length != len(self.state_hex)
            or self.differential_state_length + self.algebraic_state_length
            != self.state_vector_length
            or (self.derivative_hex and len(self.derivative_hex) != self.state_vector_length)
            or not self.bit_exact_roundtrip
            or self.numeric_payload_sha256
            != _numeric_payload_sha256(
                self.state_hex,
                self.derivative_hex,
                self.source_input_hex,
            )
        ):
            raise ValueError("battery exact restart checkpoint capsule is incomplete")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartReceiverTrace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-receiver-trace'

    time_hex: tuple[str, ...]
    capacity_hex: tuple[str, ...]
    mean_temperature_hex: tuple[str, ...]
    maximum_temperature_hex: tuple[str, ...]
    terminal_voltage_hex: tuple[str, ...]
    termination: str
    complete: bool

    def __post_init__(self) -> None:
        values = (
            self.time_hex,
            self.capacity_hex,
            self.mean_temperature_hex,
            self.maximum_temperature_hex,
            self.terminal_voltage_hex,
        )
        for index, item in enumerate(values):
            _validate_hex_tuple(item, field_name=f"trace_{index}")
        lengths = {len(item) for item in values}
        if len(lengths) != 1 or (
            self.complete and (not self.time_hex or self.termination != "final time")
        ):
            raise ValueError("battery exact restart receiver trace is incomplete")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartCapture(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-capture'

    capture_id: str
    config: ObjectIdentity
    unit_id: str
    denominator_id: str
    history_id: str
    capsule: BatteryExactRestartCheckpointCapsule | None
    native_trace: BatteryExactRestartReceiverTrace | None
    disposition: BatteryExactRestartCaptureDisposition
    reason_codes: tuple[str, ...]
    source_build_seconds: Decimal
    runtime_seconds: Decimal
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.capture_id, field_name="capture_id")
        for name, value in (
            ("unit_id", self.unit_id),
            ("denominator_id", self.denominator_id),
            ("history_id", self.history_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name, scalar_value in (
            ("source_build_seconds", self.source_build_seconds),
            ("runtime_seconds", self.runtime_seconds),
        ):
            validate_decimal(scalar_value, field_name=name, minimum=Decimal(0))
        if self.disposition is BatteryExactRestartCaptureDisposition.COMPLETE:
            if (
                self.capsule is None
                or self.native_trace is None
                or not self.native_trace.complete
                or self.reason_codes
            ):
                raise ValueError("complete battery exact restart capture is incomplete")
        elif self.capsule is not None or self.native_trace is not None or not self.reason_codes:
            raise ValueError("failed battery exact restart capture carries scientific payload")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartRouteResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-route-result'

    result_id: str
    config: ObjectIdentity
    capture: ObjectIdentity
    unit_id: str
    denominator_id: str
    history_id: str
    route: BatteryExactRestartRoute
    trace: BatteryExactRestartReceiverTrace | None
    capacity_error_ah: Decimal | None
    mean_temperature_error_k: Decimal | None
    maximum_temperature_error_k: Decimal | None
    terminal_voltage_error_v: Decimal | None
    first_sample_max_normalized_error: Decimal | None
    preconsistent_correction_max_abs: Decimal | None
    pre_algebraic_residual_max_abs: Decimal | None
    post_algebraic_residual_max_abs: Decimal | None
    capsule_bit_exact: bool
    fresh_process_isolation: bool
    receiver_closed: bool
    disposition: BatteryExactRestartRouteDisposition
    reason_codes: tuple[str, ...]
    runtime_seconds: Decimal
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        for name, value in (
            ("unit_id", self.unit_id),
            ("denominator_id", self.denominator_id),
            ("history_id", self.history_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        validate_decimal(
            self.runtime_seconds,
            field_name="runtime_seconds",
            minimum=Decimal(0),
        )
        errors = (
            self.capacity_error_ah,
            self.mean_temperature_error_k,
            self.maximum_temperature_error_k,
            self.terminal_voltage_error_v,
        )
        if self.disposition in {
            BatteryExactRestartRouteDisposition.COMPLETE,
            BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED,
        }:
            if (
                self.trace is None
                or not self.trace.complete
                or any(item is None for item in errors)
                or self.reason_codes
                or self.receiver_closed is not (self.disposition is BatteryExactRestartRouteDisposition.COMPLETE)
            ):
                raise ValueError("evaluable battery exact restart route result is incomplete")
        elif self.trace is not None or any(item is not None for item in errors):
            raise ValueError("failed battery exact restart route result carries evaluable receivers")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartRoutePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-route-panel'

    route: BatteryExactRestartRoute
    intended_rows: int
    evaluable_rows: int
    closed_rows: int
    maximum_capacity_error_ah: Decimal | None
    maximum_mean_temperature_error_k: Decimal | None
    maximum_maximum_temperature_error_k: Decimal | None
    maximum_terminal_voltage_error_v: Decimal | None
    passes_all_rows: bool

    def __post_init__(self) -> None:
        if (
            self.intended_rows <= 0
            or not 0 <= self.evaluable_rows <= self.intended_rows
            or not 0 <= self.closed_rows <= self.evaluable_rows
            or self.passes_all_rows
            is not (
                self.evaluable_rows == self.intended_rows and self.closed_rows == self.intended_rows
            )
        ):
            raise ValueError("battery exact restart route panel counts are invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartLocalizationDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-localization-decision'

    decision_id: str
    config: ObjectIdentity
    source_audit: ObjectIdentity
    route_panels: tuple[BatteryExactRestartRoutePanel, ...]
    obstruction_reproduced: bool
    mechanism_disposition: BatteryExactRestartMechanismDisposition
    selected_route: BatteryExactRestartRoute | None
    verdict: BatteryExactRestartLocalizationVerdict
    fresh_process_isolation_verified: bool
    decided_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        parse_utc_timestamp(self.decided_at_utc, field_name="decided_at_utc")
        routes = tuple(item.route for item in self.route_panels)
        if (
            routes != tuple(BatteryExactRestartRoute)
            or self.selected_route not in {*BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE, None}
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart localization decision is incomplete")
        if self.verdict is BatteryExactRestartLocalizationVerdict.ROUTE_SELECTED:
            if (
                self.selected_route is None
                or not self.obstruction_reproduced
                or not self.fresh_process_isolation_verified
            ):
                raise ValueError("battery exact restart route selection lacks required evidence")
        elif self.selected_route is not None:
            raise ValueError("condition-false battery exact restart localization selected a route")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartQualificationFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-qualification-freeze'

    freeze_id: str
    localization_decision: ObjectIdentity
    source_audit: ObjectIdentity
    implementation: ObjectIdentity
    selected_route_canary: ObjectIdentity
    qualification_config: ObjectIdentity
    resource_envelope: ObjectIdentity
    selected_route: BatteryExactRestartRoute
    primary_roster_sha256: str
    reserve_roster_sha256: str
    tolerance: BatteryExactRestartExactTolerance
    expected_capture_count: int
    expected_route_result_count: int
    planned_issue_id: str
    planned_run_id: str
    planned_execution_authority_id: str
    planned_reveal_authority_id: str
    external_root: str
    capsule_schema: str
    route_result_schema: str
    receipt_schema: str
    terminal_index_schema: str
    entrypoint_help_sha256: str
    adjudicator_source_sha256: str
    frozen_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_sha256(self.primary_roster_sha256, field_name="primary_roster_sha256")
        validate_sha256(self.reserve_roster_sha256, field_name="reserve_roster_sha256")
        validate_sha256(self.entrypoint_help_sha256, field_name="entrypoint_help_sha256")
        validate_sha256(
            self.adjudicator_source_sha256,
            field_name="adjudicator_source_sha256",
        )
        for name, value in (
            ("planned_issue_id", self.planned_issue_id),
            ("planned_run_id", self.planned_run_id),
            ("planned_execution_authority_id", self.planned_execution_authority_id),
            ("planned_reveal_authority_id", self.planned_reveal_authority_id),
        ):
            validate_stable_id(value, field_name=name)
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")
        if (
            self.selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or self.expected_capture_count != 240
            or self.expected_route_result_count != 240
            or self.external_root != BATTERY_EXACT_RESTART_EXTERNAL_ROOT
            or self.capsule_schema != BatteryExactRestartCheckpointCapsule.SCHEMA
            or self.route_result_schema != BatteryExactRestartRouteResult.SCHEMA
            or self.receipt_schema != 'empirical-lawhood/adapters/simulators/staged-hybrid-response-episode-artifact-receipt'
            or self.terminal_index_schema != 'empirical-lawhood/adapters/simulators/staged-hybrid-response-acquisition-index'
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart qualification freeze differs from the plan")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartSelectedRouteCanary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-selected-route-canary'

    canary_id: str
    localization_decision: ObjectIdentity
    capture: ObjectIdentity
    selected_route: BatteryExactRestartRoute
    repeat_results: tuple[ObjectIdentity, ...]
    maximum_capacity_error_ah: Decimal
    maximum_mean_temperature_error_k: Decimal
    maximum_maximum_temperature_error_k: Decimal
    maximum_terminal_voltage_error_v: Decimal
    all_closed: bool
    completed_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.canary_id, field_name="canary_id")
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")
        for name, value in (
            ("maximum_capacity_error_ah", self.maximum_capacity_error_ah),
            ("maximum_mean_temperature_error_k", self.maximum_mean_temperature_error_k),
            (
                "maximum_maximum_temperature_error_k",
                self.maximum_maximum_temperature_error_k,
            ),
            ("maximum_terminal_voltage_error_v", self.maximum_terminal_voltage_error_v),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if (
            self.selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or len(self.repeat_results) != 3
            or len({item.object_id for item in self.repeat_results}) != 3
            or not self.all_closed
            or self.maximum_capacity_error_ah > BATTERY_EXACT_RESTART_CAPACITY_FLOOR_AH
            or self.maximum_mean_temperature_error_k > BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K
            or self.maximum_maximum_temperature_error_k > BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K
            or self.maximum_terminal_voltage_error_v > BATTERY_EXACT_RESTART_VOLTAGE_FLOOR_V
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart selected-route canary did not qualify")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartClosureRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-closure-row'

    denominator_id: str
    history_id: str
    intended_unit_count: int
    evaluable_unit_count: int
    closed_unit_count: int
    maximum_capacity_error_ah: Decimal | None
    maximum_mean_temperature_error_k: Decimal | None
    maximum_maximum_temperature_error_k: Decimal | None
    maximum_terminal_voltage_error_v: Decimal | None
    closure_class: BatteryExactRestartClosureClass

    def __post_init__(self) -> None:
        validate_stable_id(self.denominator_id, field_name="denominator_id")
        validate_stable_id(self.history_id, field_name="history_id")
        if (
            self.intended_unit_count != 6
            or not 0 <= self.evaluable_unit_count <= 6
            or not 0 <= self.closed_unit_count <= self.evaluable_unit_count
        ):
            raise ValueError("battery exact restart closure row counts are invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-adjudication'

    adjudication_id: str
    freeze: ObjectIdentity
    config: ObjectIdentity
    selected_route: BatteryExactRestartRoute
    closure_rows: tuple[BatteryExactRestartClosureRow, ...]
    complete_capture_count: int
    intended_capture_count: int
    complete_route_result_count: int
    intended_route_result_count: int
    qualified_denominator_ids: tuple[str, ...]
    verdict: BatteryExactRestartQualificationVerdict
    revealed_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        parse_utc_timestamp(self.revealed_at_utc, field_name="revealed_at_utc")
        require_sorted_unique_strings(
            self.qualified_denominator_ids,
            field_name="qualified_denominator_ids",
        )
        keys = tuple((item.denominator_id, item.history_id) for item in self.closure_rows)
        if (
            self.selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or keys != tuple(sorted(keys))
            or len(keys) != 40
            or len(set(keys)) != 40
            or self.intended_capture_count != 240
            or self.intended_route_result_count != 240
            or not 0 <= self.complete_capture_count <= 240
            or not 0 <= self.complete_route_result_count <= 240
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart adjudication is incomplete")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartErrorSummaryRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-error-summary-row'

    panel: str
    route: BatteryExactRestartRoute
    denominator_id: str
    history_id: str
    intended_count: int
    evaluable_count: int
    closed_count: int
    median_capacity_error_ah: Decimal | None
    p95_capacity_error_ah: Decimal | None
    maximum_capacity_error_ah: Decimal | None
    median_mean_temperature_error_k: Decimal | None
    p95_mean_temperature_error_k: Decimal | None
    maximum_mean_temperature_error_k: Decimal | None
    median_maximum_temperature_error_k: Decimal | None
    p95_maximum_temperature_error_k: Decimal | None
    maximum_maximum_temperature_error_k: Decimal | None
    median_terminal_voltage_error_v: Decimal | None
    p95_terminal_voltage_error_v: Decimal | None
    maximum_terminal_voltage_error_v: Decimal | None
    maximum_first_sample_normalized_error: Decimal | None
    maximum_preconsistent_correction_abs: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.panel, field_name="panel")
        validate_stable_id(self.denominator_id, field_name="denominator_id")
        validate_stable_id(self.history_id, field_name="history_id")
        if (
            self.panel not in {"localization", "qualification"}
            or self.intended_count <= 0
            or not 0 <= self.evaluable_count <= self.intended_count
            or not 0 <= self.closed_count <= self.evaluable_count
        ):
            raise ValueError("battery exact restart descriptive row counts are invalid")


@dataclass(frozen=True, slots=True)
class BatteryExactRestartDescriptiveAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-exact-restart-descriptive-analysis'

    analysis_id: str
    localization_decision: ObjectIdentity
    adjudication: ObjectIdentity
    selected_route: BatteryExactRestartRoute
    rows: tuple[BatteryExactRestartErrorSummaryRow, ...]
    analyzed_at_utc: str
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        parse_utc_timestamp(self.analyzed_at_utc, field_name="analyzed_at_utc")
        keys = tuple(
            (item.panel, item.route.value, item.denominator_id, item.history_id)
            for item in self.rows
        )
        if (
            self.selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            or keys != tuple(sorted(keys))
            or len(keys) != len(set(keys))
            or not self.rows
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery exact restart descriptive analysis is incomplete")


_STRATA = (
    ('low-state-of-charge-cool', Decimal("0.25"), Decimal("0.40"), Decimal("291.15"), Decimal("293.15")),
    ('low-state-of-charge-warm', Decimal("0.25"), Decimal("0.40"), Decimal("295.15"), Decimal("297.15")),
    ('middle-state-of-charge-cool', Decimal("0.45"), Decimal("0.60"), Decimal("291.15"), Decimal("293.15")),
    ('middle-state-of-charge-warm', Decimal("0.45"), Decimal("0.60"), Decimal("295.15"), Decimal("297.15")),
    ('high-state-of-charge-cool', Decimal("0.65"), Decimal("0.80"), Decimal("291.15"), Decimal("293.15")),
    ('high-state-of-charge-warm', Decimal("0.65"), Decimal("0.80"), Decimal("295.15"), Decimal("297.15")),
)

_FROZEN_STRATUM_HASH_OPERANDS = (
    (115, 49, 45, 108, 111, 119, 45, 99, 111, 111, 108),
    (115, 50, 45, 108, 111, 119, 45, 119, 97, 114, 109),
    (115, 51, 45, 109, 105, 100, 100, 108, 101, 45, 99, 111, 111, 108),
    (115, 52, 45, 109, 105, 100, 100, 108, 101, 45, 119, 97, 114, 109),
    (115, 53, 45, 104, 105, 103, 104, 45, 99, 111, 111, 108),
    (115, 54, 45, 104, 105, 103, 104, 45, 119, 97, 114, 109),
)


def _fraction(seed: int, *tokens: object) -> Decimal:
    digest = sha256(b":".join((str(seed).encode(), *(item if isinstance(item, bytes) else str(item).encode() for item in tokens)))).digest()
    return Decimal(int.from_bytes(digest[:8], "big")) / Decimal(2**64)


def _preparation(
    *,
    stage: BatteryExactRestartStage,
    role: BatteryExactRestartUnitRole,
    seed: int,
    token: str,
    stratum_index: int,
) -> BatteryExactRestartPreparation:
    stratum_id, soc_lo, soc_hi, temp_lo, temp_hi = _STRATA[stratum_index - 1]
    soc = (soc_lo + (soc_hi - soc_lo) * _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "soc")).quantize(
        _Q, rounding=ROUND_HALF_EVEN
    )
    temperature = (
        temp_lo + (temp_hi - temp_lo) * _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "temp")
    ).quantize(_Q, rounding=ROUND_HALF_EVEN)
    return BatteryExactRestartPreparation(
        unit_id=f"unit.battery-exact-restart.{token}.s{stratum_index:02d}.01",
        stage=stage,
        role=role,
        stratum_id=stratum_id,
        initial_soc=soc,
        initial_temperature_k=temperature,
        reserve=role is BatteryExactRestartUnitRole.QUALIFICATION_RESERVE,
    )


def histories() -> tuple[BatteryExactRestartHistory, ...]:
    by_id = {item.word_id: item for item in battery_electrothermal.action_words()}
    return tuple(
        sorted(
            (
                BatteryExactRestartHistory(
                    history_id=(
                        "history.battery-exact-restart."
                        + word_id.removeprefix("word.battery-electrothermal.")
                        + f".{checkpoint_s}.{end_s}"
                    ),
                    word=by_id[word_id],
                    checkpoint_s=checkpoint_s,
                    continuation_end_s=end_s,
                )
                for word_id, checkpoint_s, end_s in battery_electrothermal.BATTERY_ELECTROTHERMAL_RESTART_HISTORIES
            ),
            key=lambda item: item.history_id,
        )
    )


def build_design() -> BatteryExactRestartDesign:
    localization_units = tuple(
        sorted(
            (
                _preparation(
                    stage=BatteryExactRestartStage.LOCALIZATION,
                    role=BatteryExactRestartUnitRole.LOCALIZATION,
                    seed=BATTERY_EXACT_RESTART_LOCALIZATION_SEED,
                    token="localization",
                    stratum_index=index,
                )
                for index in (1, 2, 5, 6)
            ),
            key=lambda item: item.unit_id,
        )
    )
    qualification_units = tuple(
        _preparation(
            stage=BatteryExactRestartStage.QUALIFICATION,
            role=BatteryExactRestartUnitRole.QUALIFICATION_PRIMARY,
            seed=BATTERY_EXACT_RESTART_QUALIFICATION_SEED,
            token="qualification",
            stratum_index=index,
        )
        for index in range(1, 7)
    )
    reserve_units = tuple(
        _preparation(
            stage=BatteryExactRestartStage.QUALIFICATION,
            role=BatteryExactRestartUnitRole.QUALIFICATION_RESERVE,
            seed=BATTERY_EXACT_RESTART_RESERVE_SEED,
            token="qualification-reserve",
            stratum_index=index,
        )
        for index in range(1, 7)
    )
    all_denominators = battery_electrothermal.denominators(include_x_full=False)
    localization = tuple(
        sorted(
            (
                "denominator.pybamm.dfn.isothermal.idaklu-refined",
                "denominator.pybamm.dfn.lumped.idaklu-refined",
                "denominator.pybamm.spme.lumped.casadi-coarse",
                "denominator.pybamm.spme.lumped.idaklu-refined",
            )
        )
    )
    return BatteryExactRestartDesign(
        design_id=BATTERY_EXACT_RESTART_DESIGN_ID,
        localization_units=localization_units,
        qualification_units=qualification_units,
        reserve_units=reserve_units,
        all_denominators=all_denominators,
        localization_denominator_ids=localization,
        histories=histories(),
        route_roster=BATTERY_EXACT_RESTART_ROUTE_ROSTER,
        tolerance=exact_tolerance(),
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def build_localization_config(design: BatteryExactRestartDesign) -> BatteryExactRestartStageConfig:
    wanted = set(design.localization_denominator_ids)
    denominators = tuple(item for item in design.all_denominators if item.denominator_id in wanted)
    return BatteryExactRestartStageConfig(
        config_id="config.battery-exact-restart.localization",
        design=ObjectIdentity.from_record(design.design_id, design),
        stage=BatteryExactRestartStage.LOCALIZATION,
        units=design.localization_units,
        primary_unit_ids=tuple(item.unit_id for item in design.localization_units),
        reserve_unit_ids=(),
        denominators=denominators,
        histories=design.histories,
        routes=BATTERY_EXACT_RESTART_ROUTE_ROSTER,
        selected_route=None,
        tolerance=design.tolerance,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def build_qualification_config(
    design: BatteryExactRestartDesign,
    *,
    selected_route: BatteryExactRestartRoute,
) -> BatteryExactRestartStageConfig:
    if selected_route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE:
        raise ValueError("battery exact restart selected route is not qualification eligible")
    units = (*design.qualification_units, *design.reserve_units)
    return BatteryExactRestartStageConfig(
        config_id=f"config.battery-exact-restart.qualification.{selected_route.name.lower()}",
        design=ObjectIdentity.from_record(design.design_id, design),
        stage=BatteryExactRestartStage.QUALIFICATION,
        units=tuple(sorted(units, key=lambda item: item.unit_id)),
        primary_unit_ids=tuple(item.unit_id for item in design.qualification_units),
        reserve_unit_ids=tuple(item.unit_id for item in design.reserve_units),
        denominators=design.all_denominators,
        histories=design.histories,
        routes=(BatteryExactRestartRoute.NATIVE_SAME_OBJECT, selected_route),
        selected_route=selected_route,
        tolerance=design.tolerance,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def _route_semantics(route: BatteryExactRestartRoute) -> _RouteSemantics:
    direct = route in {
        BatteryExactRestartRoute.NATIVE_SAME_OBJECT,
        BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT,
        BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT,
    }
    live = route in {
        BatteryExactRestartRoute.NATIVE_SAME_OBJECT,
        BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP,
        BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP,
    }
    return {
        "api_status": (
            "PUBLIC_API"
            if route is BatteryExactRestartRoute.NATIVE_SAME_OBJECT
            else "PUBLIC_API_WITH_PINNED_INTERNAL_BRANCH_SEMANTICS"
        ),
        "supported_denominator_ids": tuple(
            item.denominator_id for item in build_design().all_denominators
        ),
        "consumes_state_y": True,
        "consumes_derivative_yp": False,
        "consumes_inputs": True,
        "consumes_model_identity": True,
        "consumes_time": True,
        "consistency_initialization_behavior": {
            BatteryExactRestartRoute.NATIVE_SAME_OBJECT: "source-native step initialization",
            BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP: "target solver source-default consistency",
            BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP: ("target solver source-default consistency"),
            BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT: ("target solver source-default consistency"),
            BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP: (
                "public initial-condition map then source-default consistency"
            ),
            BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT: (
                "explicit public consistent-state preparation for DAE; identity for ODE"
            ),
        }[route],
        "model_mapping_behavior": (
            "same built-model identity; no public variable map"
            if direct
            else "rebuilt source model mapped into distinct rebuilt target model"
        ),
        "process_assumptions": tuple(
            sorted(
                (
                    (
                        "live-prefix-object-retained",
                        "localization-only",
                        "same-worker-source-control",
                    )
                    if live
                    else (
                        "capsule-only-input",
                        "fresh-reconstruction-worker",
                        "pinned-source-build",
                    )
                )
            )
        ),
    }


def route_audits() -> tuple[BatteryExactRestartRouteAudit, ...]:
    """Return the finite source-audited battery exact restart route manifest."""

    values = {
        BatteryExactRestartRoute.NATIVE_SAME_OBJECT: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.NATIVE_SAME_OBJECT,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.Simulation.step",
                "pybamm.Solution.last_state",
            ),
            consumed_operands=(
                "live-model",
                "live-simulation",
                "native-last-state",
            ),
            capsule_only_eligible=False,
            reason="truth-known native same-object continuation reference",
            **_route_semantics(BatteryExactRestartRoute.NATIVE_SAME_OBJECT),
        ),
        BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.BaseModel.set_initial_conditions_from",
                "pybamm.BaseSolver.step",
                "pybamm.Solution",
            ),
            consumed_operands=(
                "decimal15-state",
                "live-prefix-model",
                "rebuilt-target-model",
            ),
            capsule_only_eligible=False,
            reason="exact historical battery electrothermal reconstruction calibration route",
            **_route_semantics(BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP),
        ),
        BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.BaseModel.set_initial_conditions_from",
                "pybamm.BaseSolver.step",
                "pybamm.Solution",
            ),
            consumed_operands=(
                "live-prefix-model",
                "lossless-state",
                "rebuilt-target-model",
            ),
            capsule_only_eligible=False,
            reason="one-factor numeric-encoding contrast through the battery electrothermal seam",
            **_route_semantics(BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP),
        ),
        BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.BaseSolver.step",
                "pybamm.Solution",
            ),
            consumed_operands=(
                "lossless-state",
                "rebuilt-target-model",
            ),
            capsule_only_eligible=True,
            reason="source-supported direct y0 placement in the identical target model",
            **_route_semantics(BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT),
        ),
        BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.BaseModel.set_initial_conditions_from",
                "pybamm.BaseSolver.step",
                "pybamm.Solution",
            ),
            consumed_operands=(
                "lossless-state",
                "rebuilt-source-model",
                "rebuilt-target-model",
            ),
            capsule_only_eligible=True,
            reason="public variable-ID and scale/reference mapping between rebuilt models",
            **_route_semantics(BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP),
        ),
        BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT: BatteryExactRestartRouteAudit(
            route=BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT,
            feasibility=BatteryExactRestartRouteFeasibility.FEASIBLE,
            source_symbols=(
                "pybamm.BaseSolver.calculate_consistent_state",
                "pybamm.BaseSolver.step",
                "pybamm.Solution",
            ),
            consumed_operands=(
                "lossless-state",
                "rebuilt-target-model",
                "source-consistent-state",
            ),
            capsule_only_eligible=True,
            reason="public consistency preparation followed by direct target placement",
            **_route_semantics(BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT),
        ),
    }
    return tuple(values[route] for route in BatteryExactRestartRoute)


def source_semantic_facts() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                "all-yps-is-retained-by-last-state-but-not-consumed-by-base-solver-step",
                "base-model-equality-is-object-identity",
                "base-solver-direct-branch-consumes-last-state-all-ys-for-identical-model",
                "base-solver-map-branch-calls-set-initial-conditions-from-for-distinct-model",
                "base-solver-recalculates-consistent-initialization-on-every-step",
                "idaklu-recalculates-y0-and-ydot0-on-every-step",
                "electrothermal-fabricated-solution-bound-live-prefix-model-distinct-from-target-model",
            )
        )
    )


def validate_repository_config(document: Mapping[str, object]) -> None:
    expected: dict[str, object] = {
        "campaign_token": "BATTERY-EXACT-RESTART-PYBAMM-EXACT-RESTART-QUALIFICATION",
        "claim_ceiling": "NON_PROMOTABLE_SOURCE_QUALIFICATION",
        "denominator_ids": [item.denominator_id for item in build_design().all_denominators],
        "design_id": BATTERY_EXACT_RESTART_DESIGN_ID,
        "external_root": BATTERY_EXACT_RESTART_EXTERNAL_ROOT,
        "histories": [
            [item.word.word_id, item.checkpoint_s, item.continuation_end_s] for item in histories()
        ],
        "localization": {
            "denominator_ids": list(build_design().localization_denominator_ids),
            "preparation_count": 4,
            "route_ids": [item.value for item in BATTERY_EXACT_RESTART_ROUTE_ROSTER],
            "seed": BATTERY_EXACT_RESTART_LOCALIZATION_SEED,
        },
        "native_threads_per_worker": 1,
        "parameter_set": battery_electrothermal.BATTERY_ELECTROTHERMAL_PARAMETER_SET,
        "provider_key": BATTERY_EXACT_RESTART_PROVIDER_KEY,
        "qualification": {
            "primary_count": 6,
            "primary_seed": BATTERY_EXACT_RESTART_QUALIFICATION_SEED,
            "reserve_count": 6,
            "reserve_seed": BATTERY_EXACT_RESTART_RESERVE_SEED,
            "route_preference": [item.value for item in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE],
        },
        "receiver_floors": {
            "capacity_ah": "1e-10",
            "maximum_temperature_k": "1e-8",
            "mean_temperature_k": "1e-8",
            "terminal_voltage_v": "1e-8",
        },
        "schema": 'empirical-lawhood/simulators/battery-exact-state-restart/config',
        "source_version": BATTERY_EXACT_RESTART_SOURCE_VERSION,
        "version": "1.0.0",
    }
    if dict(document) != expected:
        raise ValueError("battery exact restart repository config differs from the bounded chart")


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("battery exact restart produced a nonfinite scalar")
    return Decimal(repr(float(value)))


def _hex_values(values: np.ndarray | Sequence[float]) -> tuple[str, ...]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flattened)):
        raise ValueError("battery exact restart produced a nonfinite numeric payload")
    return tuple(float(item).hex() for item in flattened)


def decode_hex(values: Sequence[str]) -> np.ndarray:
    decoded = np.asarray([float.fromhex(item) for item in values], dtype=np.float64)
    if tuple(float(item).hex() for item in decoded) != tuple(values):
        raise ValueError("battery exact restart float-hex round trip changed numeric state")
    return decoded


def _source_options(denominator: battery_electrothermal.BatteryElectrothermalDenominator) -> dict[str, str]:
    values = {
        "thermal": {
            battery_electrothermal.BatteryElectrothermalThermal.ISOTHERMAL: "isothermal",
            battery_electrothermal.BatteryElectrothermalThermal.LUMPED: "lumped",
            battery_electrothermal.BatteryElectrothermalThermal.X_FULL: "x-full",
        }[denominator.thermal],
        "cell geometry": "pouch",
    }
    if denominator.thermal is battery_electrothermal.BatteryElectrothermalThermal.ISOTHERMAL:
        values["calculate heat source for isothermal models"] = "true"
    return values


def _inputs(
    unit: BatteryExactRestartPreparation,
    current_delta_a: Decimal,
    ambient_delta_k: Decimal,
) -> dict[str, float]:
    return {
        "Current function [A]": float(current_delta_a),
        "Ambient temperature [K]": float(unit.initial_temperature_k + ambient_delta_k),
    }


def _runtime_objects(
    unit: BatteryExactRestartPreparation,
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
) -> tuple[Any, Any]:
    import pybamm  # type: ignore[import-untyped]

    if pybamm.__version__ != BATTERY_EXACT_RESTART_SOURCE_VERSION:
        raise RuntimeError("installed PyBaMM differs from the battery exact restart source")
    model_class = (
        pybamm.lithium_ion.SPMe
        if denominator.model is battery_electrothermal.BatteryElectrothermalModel.SPME
        else pybamm.lithium_ion.DFN
    )
    model = model_class(options=_source_options(denominator))
    parameters = pybamm.ParameterValues(battery_electrothermal.BATTERY_ELECTROTHERMAL_PARAMETER_SET)
    parameters["Current function [A]"] = "[input]"
    parameters["Ambient temperature [K]"] = "[input]"
    parameters["Initial temperature [K]"] = float(unit.initial_temperature_k)
    if denominator.view.solver is battery_electrothermal.BatteryElectrothermalSolver.CASADI:
        solver = pybamm.CasadiSolver(
            mode="safe",
            rtol=float(denominator.view.rtol),
            atol=float(denominator.view.atol),
        )
    else:
        solver = pybamm.IDAKLUSolver(
            rtol=float(denominator.view.rtol),
            atol=float(denominator.view.atol),
        )
    points = denominator.view.spatial_points
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        solver=solver,
        var_pts={
            "x_n": points,
            "x_s": points,
            "x_p": points,
            "r_n": points,
            "r_p": points,
        },
    )
    return pybamm, simulation


def _state_layout_sha256(model: Any) -> str:
    layout = tuple(
        sorted(
            (
                str(variable),
                tuple(
                    (
                        int(item.start or 0),
                        int(item.stop or 0),
                        int(item.step or 1),
                    )
                    for item in (slices if isinstance(slices, list) else [slices])
                ),
            )
            for variable, slices in model.y_slices.items()
        )
    )
    return sha256(canonical_json_bytes(layout)).hexdigest()


def _model_fingerprint_sha256(
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
    model: Any,
) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "denominator": denominator.fingerprint(),
                "len_alg": int(model.len_alg),
                "len_rhs": int(model.len_rhs),
                "name": str(model.name),
                "state_layout_sha256": _state_layout_sha256(model),
            }
        )
    ).hexdigest()


def _configuration_fingerprints(
    *,
    unit: BatteryExactRestartPreparation,
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
    history: BatteryExactRestartHistory,
) -> tuple[str, str, str, str, str, str]:
    """Fingerprint the finite build/clock/receiver specifications.

    The installed source digest lives in the source audit.  These hashes bind
    the deterministic parameter overrides and numerical build chart that,
    together with that source digest, produce the evaluated model.
    """

    parameter = sha256(
        canonical_json_bytes(
            {
                "ambient_parameter": "[input]",
                "current_parameter": "[input]",
                "initial_temperature_k_hex": float(unit.initial_temperature_k).hex(),
                "parameter_set": denominator.parameter_set,
                "source_version": denominator.source_version,
            }
        )
    ).hexdigest()
    options = sha256(canonical_json_bytes(_source_options(denominator))).hexdigest()
    geometry_mesh = sha256(
        canonical_json_bytes(
            {
                "cell_geometry": denominator.cell_geometry,
                "spatial_points": denominator.view.spatial_points,
                "var_pts": ("r_n", "r_p", "x_n", "x_p", "x_s"),
            }
        )
    ).hexdigest()
    solver = sha256(
        canonical_json_bytes(
            {
                "atol": denominator.view.atol,
                "mode": (
                    "safe" if denominator.view.solver is battery_electrothermal.BatteryElectrothermalSolver.CASADI else "source-default"
                ),
                "rtol": denominator.view.rtol,
                "solver": denominator.view.solver.value,
            }
        )
    ).hexdigest()
    clock = sha256(
        canonical_json_bytes(_hex_values(_sample_clock(denominator, history)))
    ).hexdigest()
    receiver = sha256(
        canonical_json_bytes(
            (
                ("Discharge capacity [A.h]", "scalar"),
                ("Volume-averaged cell temperature [K]", "scalar"),
                ("Cell temperature [K]", "spatial-maximum"),
                ("Terminal voltage [V]", "scalar"),
            )
        )
    ).hexdigest()
    return parameter, options, geometry_mesh, solver, clock, receiver


def _input_ledger(values: Mapping[str, float]) -> tuple[BatteryExactRestartInputLedger, ...]:
    return tuple(
        BatteryExactRestartInputLedger(
            input_name=name,
            requested_hex=float(value).hex(),
            accepted_hex=float(value).hex(),
            applied_hex=float(value).hex(),
            realized_hex=float(value).hex(),
        )
        for name, value in sorted(values.items())
    )


def _step_prefix(
    *,
    unit: BatteryExactRestartPreparation,
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
    history: BatteryExactRestartHistory,
) -> tuple[Any, Any, Any, Decimal]:
    started = time.monotonic()
    pybamm, simulation = _runtime_objects(unit, denominator)
    current, ambient = history.word.values_at(0)
    simulation.build(
        initial_soc=float(unit.initial_soc),
        inputs=_inputs(unit, current, ambient),
    )
    build_seconds = _decimal(time.monotonic() - started)
    interval = float(denominator.view.nominal_output_interval_s)
    prefix = None
    for start, stop in zip(battery_electrothermal.BATTERY_ELECTROTHERMAL_BOUNDARIES_S[:-1], battery_electrothermal.BATTERY_ELECTROTHERMAL_BOUNDARIES_S[1:], strict=True):
        if start >= history.checkpoint_s:
            break
        segment_stop = min(stop, history.checkpoint_s)
        current, ambient = history.word.values_at(start)
        duration = segment_stop - start
        relative = np.arange(0.0, duration + interval, interval, dtype=np.float64)
        relative[-1] = float(duration)
        simulation.step(
            duration,
            t_eval=np.unique(relative),
            inputs=_inputs(unit, current, ambient),
        )
        if segment_stop == history.checkpoint_s:
            prefix = simulation.solution.last_state
            break
    if prefix is None:
        raise RuntimeError("battery exact restart prefix did not reach its checkpoint")
    return pybamm, simulation, prefix, build_seconds


def _sample_clock(
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
    history: BatteryExactRestartHistory,
) -> np.ndarray:
    interval = float(denominator.view.nominal_output_interval_s)
    values = np.arange(
        history.checkpoint_s,
        history.continuation_end_s + interval,
        interval,
        dtype=np.float64,
    )
    values[-1] = float(history.continuation_end_s)
    return np.unique(values)


def _variable(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = np.asarray(solution[name](times), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"battery exact restart source variable {name!r} is nonfinite")
    return values


def _scalar_series(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = _variable(solution, name, times).reshape(-1)
    if values.size != times.size:
        raise RuntimeError(f"battery exact restart source variable {name!r} is not scalar by clock")
    return values


def _field_by_time(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = _variable(solution, name, times)
    if values.ndim == 1:
        if values.size != times.size:
            raise RuntimeError(f"battery exact restart source field {name!r} has invalid shape")
        return values.reshape(1, -1)
    if values.shape[-1] == times.size:
        return values.reshape(-1, times.size)
    if values.shape[0] == times.size:
        return np.moveaxis(values, 0, -1).reshape(-1, times.size)
    raise RuntimeError(f"battery exact restart source field {name!r} lacks a clock axis")


def _trace(solution: Any, clock: np.ndarray) -> BatteryExactRestartReceiverTrace:
    return BatteryExactRestartReceiverTrace(
        time_hex=_hex_values(clock),
        capacity_hex=_hex_values(_scalar_series(solution, "Discharge capacity [A.h]", clock)),
        mean_temperature_hex=_hex_values(
            _scalar_series(solution, "Volume-averaged cell temperature [K]", clock)
        ),
        maximum_temperature_hex=_hex_values(
            np.max(_field_by_time(solution, "Cell temperature [K]", clock), axis=0)
        ),
        terminal_voltage_hex=_hex_values(_scalar_series(solution, "Terminal voltage [V]", clock)),
        termination=str(solution.termination),
        complete=str(solution.termination) == "final time",
    )


def _lookup(
    config: BatteryExactRestartStageConfig,
    *,
    unit_id: str,
    denominator_id: str,
    history_id: str,
) -> tuple[BatteryExactRestartPreparation, battery_electrothermal.BatteryElectrothermalDenominator, BatteryExactRestartHistory]:
    try:
        unit = next(item for item in config.units if item.unit_id == unit_id)
        denominator = next(
            item for item in config.denominators if item.denominator_id == denominator_id
        )
        history = next(item for item in config.histories if item.history_id == history_id)
    except StopIteration as error:
        raise ValueError("battery exact restart work item leaves the issued config") from error
    return unit, denominator, history


def verify_denominator_builds(
    *,
    unit: BatteryExactRestartPreparation,
    denominators: Sequence[battery_electrothermal.BatteryElectrothermalDenominator],
) -> tuple[str, ...]:
    """Build every supplied denominator and verify required source operands."""

    built = []
    required = {
        "Discharge capacity [A.h]",
        "Volume-averaged cell temperature [K]",
        "Cell temperature [K]",
        "Terminal voltage [V]",
    }
    for denominator in denominators:
        _, simulation = _runtime_objects(unit, denominator)
        simulation.build(
            initial_soc=float(unit.initial_soc),
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        model = simulation.built_model
        if (
            not required.issubset(model.variables)
            or int(model.len_rhs) + int(model.len_alg) <= 0
            or not _state_layout_sha256(model)
        ):
            raise RuntimeError(f"battery exact restart source build is incomplete: {denominator.denominator_id}")
        built.append(denominator.denominator_id)
    return tuple(sorted(built))


def capture_reference(
    *,
    config: BatteryExactRestartStageConfig,
    unit_id: str,
    denominator_id: str,
    history_id: str,
    source_audit: BatteryExactRestartSourceAudit,
    execution_binding: BatteryExactRestartExecutionBinding,
) -> BatteryExactRestartCapture:
    """Capture a lossless checkpoint and native same-object hold continuation."""

    started = time.monotonic()
    build_seconds = Decimal(0)
    unit, denominator, history = _lookup(
        config,
        unit_id=unit_id,
        denominator_id=denominator_id,
        history_id=history_id,
    )
    token = (
        f"{unit.unit_id.removeprefix('unit.battery-exact-restart.')}."
        f"{denominator.denominator_id.removeprefix('denominator.pybamm.')}."
        f"{history.history_id.removeprefix('history.battery-exact-restart.')}"
    )
    capture_id = f"capture.battery-exact-restart.{token}"
    config_identity = ObjectIdentity.from_record(config.config_id, config)
    if (
        execution_binding.config != config_identity
        or execution_binding.source_audit
        != ObjectIdentity.from_record(source_audit.audit_id, source_audit)
        or execution_binding.outcome_access is not config.outcome_access
    ):
        raise ValueError("battery exact restart capture execution binding differs from the config")
    try:
        _, simulation, prefix, build_seconds = _step_prefix(
            unit=unit,
            denominator=denominator,
            history=history,
        )
        model = prefix.all_models[-1]
        state = np.asarray(prefix.all_ys[0], dtype=np.float64).reshape(-1)
        derivative = (
            np.asarray(prefix.all_yps[0], dtype=np.float64).reshape(-1)
            if prefix.all_yps is not None
            else np.asarray([], dtype=np.float64)
        )
        inputs = tuple(
            sorted(
                (
                    str(name),
                    float(np.asarray(value, dtype=np.float64).reshape(-1)[0]).hex(),
                )
                for name, value in prefix.all_inputs[-1].items()
            )
        )
        state_hex = _hex_values(state)
        derivative_hex = _hex_values(derivative)
        layout = _state_layout_sha256(model)
        (
            parameter_fingerprint,
            options_fingerprint,
            geometry_mesh_fingerprint,
            solver_fingerprint,
            clock_fingerprint,
            receiver_fingerprint,
        ) = _configuration_fingerprints(
            unit=unit,
            denominator=denominator,
            history=history,
        )
        checkpoint_values = {name: float.fromhex(value) for name, value in inputs}
        continuation_values = _inputs(unit, Decimal(0), Decimal(0))
        capsule = BatteryExactRestartCheckpointCapsule(
            capsule_id=f"capsule.battery-exact-restart.{token}",
            execution_binding=execution_binding,
            source_audit=ObjectIdentity.from_record(
                source_audit.audit_id,
                source_audit,
            ),
            unit=ObjectIdentity.from_record(unit.unit_id, unit),
            denominator=ObjectIdentity.from_record(
                denominator.denominator_id,
                denominator,
            ),
            history=ObjectIdentity.from_record(history.history_id, history),
            checkpoint_s=history.checkpoint_s,
            source_version=BATTERY_EXACT_RESTART_SOURCE_VERSION,
            distribution_record_sha256=source_audit.distribution_record_sha256,
            python_version=source_audit.python_version,
            platform_identity=source_audit.platform_identity,
            numpy_version=source_audit.numpy_version,
            casadi_version=source_audit.casadi_version,
            parameter_set=denominator.parameter_set,
            evaluated_parameter_fingerprint_sha256=parameter_fingerprint,
            model_fingerprint_sha256=_model_fingerprint_sha256(denominator, model),
            model_options_fingerprint_sha256=options_fingerprint,
            geometry_mesh_discretization_fingerprint_sha256=(geometry_mesh_fingerprint),
            solver_fingerprint_sha256=solver_fingerprint,
            state_layout_sha256=layout,
            observation_clock_fingerprint_sha256=clock_fingerprint,
            receiver_schema_fingerprint_sha256=receiver_fingerprint,
            state_vector_length=len(state_hex),
            differential_state_length=int(model.len_rhs),
            algebraic_state_length=int(model.len_alg),
            state_hex=state_hex,
            derivative_hex=derivative_hex,
            source_input_hex=inputs,
            checkpoint_input_ledger=_input_ledger(checkpoint_values),
            continuation_input_ledger=_input_ledger(continuation_values),
            continuation_time_origin_s=history.checkpoint_s,
            continuation_end_s=history.continuation_end_s,
            prefix_termination=str(prefix.termination),
            prefix_valid=bool(
                str(prefix.termination) == "final time"
                and np.all(np.isfinite(state))
                and (not derivative.size or np.all(np.isfinite(derivative)))
            ),
            route_compatible_ids=BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE,
            numeric_payload_sha256=_numeric_payload_sha256(
                state_hex,
                derivative_hex,
                inputs,
            ),
            bit_exact_roundtrip=tuple(_hex_values(decode_hex(state_hex))) == state_hex,
            outcome_access=config.outcome_access,
        )
        clock = _sample_clock(denominator, history)
        relative = clock - history.checkpoint_s
        native_solution = simulation.step(
            history.continuation_end_s - history.checkpoint_s,
            t_eval=relative,
            starting_solution=prefix,
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        native_trace = _trace(native_solution, clock)
        if not native_trace.complete:
            raise RuntimeError(f"native termination: {native_trace.termination}")
        return BatteryExactRestartCapture(
            capture_id=capture_id,
            config=config_identity,
            unit_id=unit_id,
            denominator_id=denominator_id,
            history_id=history_id,
            capsule=capsule,
            native_trace=native_trace,
            disposition=BatteryExactRestartCaptureDisposition.COMPLETE,
            reason_codes=(),
            source_build_seconds=build_seconds,
            runtime_seconds=_decimal(time.monotonic() - started),
            outcome_access=config.outcome_access,
        )
    except Exception as error:
        return BatteryExactRestartCapture(
            capture_id=capture_id,
            config=config_identity,
            unit_id=unit_id,
            denominator_id=denominator_id,
            history_id=history_id,
            capsule=None,
            native_trace=None,
            disposition=BatteryExactRestartCaptureDisposition.SOURCE_DENOMINATOR_FAILURE,
            reason_codes=(f"CAPTURE_{type(error).__name__.upper()}",),
            source_build_seconds=build_seconds,
            runtime_seconds=_decimal(time.monotonic() - started),
            outcome_access=config.outcome_access,
        )


def _capsule_inputs(capsule: BatteryExactRestartCheckpointCapsule) -> dict[str, float]:
    return {name: float.fromhex(value) for name, value in capsule.source_input_hex}


def _starting_solution(
    pybamm: Any,
    *,
    checkpoint_s: int,
    state: np.ndarray,
    model: Any,
    inputs: Mapping[str, float],
) -> Any:
    return pybamm.Solution(
        [np.asarray([float(checkpoint_s)], dtype=np.float64)],
        [state.reshape(-1, 1)],
        [model],
        [{name: np.asarray([float(value)], dtype=np.float64) for name, value in inputs.items()}],
        termination="final time",
        check_solution=False,
    )


def _preconsistent_state(
    *,
    unit: BatteryExactRestartPreparation,
    denominator: battery_electrothermal.BatteryElectrothermalDenominator,
    checkpoint_s: int,
    raw_state: np.ndarray,
) -> tuple[np.ndarray, Decimal, Decimal | None, Decimal | None]:
    _, audit_simulation = _runtime_objects(unit, denominator)
    hold_inputs = _inputs(unit, Decimal(0), Decimal(0))
    audit_simulation.build(
        initial_soc=float(unit.initial_soc),
        inputs=hold_inputs,
    )
    model = audit_simulation.built_model
    solver = audit_simulation.solver
    if int(model.len_alg) == 0:
        return raw_state.copy(), Decimal(0), Decimal(0), Decimal(0)
    solver.set_up(model, hold_inputs)
    model.y0_list = [raw_state.reshape(-1, 1)]
    time_value = float(np.nextafter(float(checkpoint_s), np.inf))

    def residual(values: np.ndarray) -> Decimal | None:
        try:
            evaluated = np.asarray(
                model.algebraic_eval(time_value, values, hold_inputs),
                dtype=np.float64,
            )
            if evaluated.size == 0:
                return Decimal(0)
            return _decimal(float(np.max(np.abs(evaluated))))
        except Exception:
            return None

    pre_residual = residual(raw_state)
    consistent = np.asarray(
        solver.calculate_consistent_state(model, time_value, [hold_inputs])[0],
        dtype=np.float64,
    ).reshape(-1)
    post_residual = residual(consistent)
    correction = _decimal(float(np.max(np.abs(consistent - raw_state))))
    return consistent, correction, pre_residual, post_residual


def _trace_errors(
    trace: BatteryExactRestartReceiverTrace,
    native: BatteryExactRestartReceiverTrace,
    tolerance: BatteryExactRestartExactTolerance,
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], Decimal, bool]:
    if trace.time_hex != native.time_hex:
        raise ValueError("battery exact restart reconstruction and native clocks differ")
    pairs = (
        (trace.capacity_hex, native.capacity_hex),
        (trace.mean_temperature_hex, native.mean_temperature_hex),
        (trace.maximum_temperature_hex, native.maximum_temperature_hex),
        (trace.terminal_voltage_hex, native.terminal_voltage_hex),
    )
    errors = tuple(
        _decimal(float(np.max(np.abs(decode_hex(left) - decode_hex(right)))))
        for left, right in pairs
    )
    floors = (
        tolerance.capacity_ah,
        tolerance.mean_temperature_k,
        tolerance.maximum_temperature_k,
        tolerance.terminal_voltage_v,
    )
    normalized_first = max(
        Decimal(repr(abs(float.fromhex(left[0]) - float.fromhex(right[0])))) / floor
        for (left, right), floor in zip(pairs, floors, strict=True)
    )
    return (
        (errors[0], errors[1], errors[2], errors[3]),
        normalized_first,
        all(error <= floor for error, floor in zip(errors, floors, strict=True)),
    )


def acquire_route_result(
    *,
    config: BatteryExactRestartStageConfig,
    capture: BatteryExactRestartCapture,
    route: BatteryExactRestartRoute,
    source_audit: BatteryExactRestartSourceAudit,
    execution_binding: BatteryExactRestartExecutionBinding,
    fresh_process_isolation: bool,
    result_suffix: str = "primary",
) -> BatteryExactRestartRouteResult:
    """Reconstruct one persisted checkpoint through one predeclared route."""

    if route is BatteryExactRestartRoute.NATIVE_SAME_OBJECT:
        raise ValueError("Native continuation is acquired only by capture_reference")
    if route not in config.routes:
        raise ValueError("battery exact restart route leaves the issued config")
    validate_stable_id(result_suffix, field_name="result_suffix")
    started = time.monotonic()
    common: _RouteResultCommon = {
        "result_id": (
            f"route-result.battery-exact-restart."
            f"{capture.capture_id.removeprefix('capture.battery-exact-restart.')}."
            f"{route.name.lower()}.{result_suffix}"
        ),
        "config": ObjectIdentity.from_record(config.config_id, config),
        "capture": ObjectIdentity.from_record(capture.capture_id, capture),
        "unit_id": capture.unit_id,
        "denominator_id": capture.denominator_id,
        "history_id": capture.history_id,
        "route": route,
        "outcome_access": config.outcome_access,
    }
    if (
        capture.disposition is not BatteryExactRestartCaptureDisposition.COMPLETE
        or capture.capsule is None
        or capture.native_trace is None
    ):
        return BatteryExactRestartRouteResult(
            **common,
            trace=None,
            capacity_error_ah=None,
            mean_temperature_error_k=None,
            maximum_temperature_error_k=None,
            terminal_voltage_error_v=None,
            first_sample_max_normalized_error=None,
            preconsistent_correction_max_abs=None,
            pre_algebraic_residual_max_abs=None,
            post_algebraic_residual_max_abs=None,
            capsule_bit_exact=False,
            fresh_process_isolation=fresh_process_isolation,
            receiver_closed=False,
            disposition=BatteryExactRestartRouteDisposition.SOURCE_ROUTE_FAILURE,
            reason_codes=("CAPTURE_NOT_COMPLETE",),
            runtime_seconds=_decimal(time.monotonic() - started),
        )
    capsule = capture.capsule
    unit, denominator, history = _lookup(
        config,
        unit_id=capture.unit_id,
        denominator_id=capture.denominator_id,
        history_id=capture.history_id,
    )
    if (
        capture.config != ObjectIdentity.from_record(config.config_id, config)
        or capsule.execution_binding != execution_binding
        or execution_binding.config != ObjectIdentity.from_record(config.config_id, config)
        or capsule.source_audit != ObjectIdentity.from_record(source_audit.audit_id, source_audit)
        or capsule.distribution_record_sha256 != source_audit.distribution_record_sha256
        or capsule.python_version != source_audit.python_version
        or capsule.platform_identity != source_audit.platform_identity
        or capsule.numpy_version != source_audit.numpy_version
        or capsule.casadi_version != source_audit.casadi_version
        or capsule.denominator.object_id != denominator.denominator_id
        or capsule.unit.object_id != unit.unit_id
        or capsule.history.object_id != history.history_id
    ):
        return BatteryExactRestartRouteResult(
            **common,
            trace=None,
            capacity_error_ah=None,
            mean_temperature_error_k=None,
            maximum_temperature_error_k=None,
            terminal_voltage_error_v=None,
            first_sample_max_normalized_error=None,
            preconsistent_correction_max_abs=None,
            pre_algebraic_residual_max_abs=None,
            post_algebraic_residual_max_abs=None,
            capsule_bit_exact=False,
            fresh_process_isolation=fresh_process_isolation,
            receiver_closed=False,
            disposition=BatteryExactRestartRouteDisposition.UNEVALUABLE_IDENTITY,
            reason_codes=("CAPSULE_IDENTITY_MISMATCH",),
            runtime_seconds=_decimal(time.monotonic() - started),
        )
    correction = pre_residual = post_residual = None
    try:
        raw_state = decode_hex(capsule.state_hex)
        bit_exact = _hex_values(raw_state) == capsule.state_hex
        if (
            not bit_exact
            or _numeric_payload_sha256(
                capsule.state_hex,
                capsule.derivative_hex,
                capsule.source_input_hex,
            )
            != capsule.numeric_payload_sha256
        ):
            raise ValueError("capsule round trip changed")
        pybamm, target_simulation = _runtime_objects(unit, denominator)
        hold_inputs = _inputs(unit, Decimal(0), Decimal(0))
        target_simulation.build(
            initial_soc=float(unit.initial_soc),
            inputs=hold_inputs,
        )
        target_model = target_simulation.built_model
        (
            parameter_fingerprint,
            options_fingerprint,
            geometry_mesh_fingerprint,
            solver_fingerprint,
            clock_fingerprint,
            receiver_fingerprint,
        ) = _configuration_fingerprints(
            unit=unit,
            denominator=denominator,
            history=history,
        )
        if (
            capsule.model_fingerprint_sha256 != _model_fingerprint_sha256(denominator, target_model)
            or capsule.state_layout_sha256 != _state_layout_sha256(target_model)
            or capsule.evaluated_parameter_fingerprint_sha256 != parameter_fingerprint
            or capsule.model_options_fingerprint_sha256 != options_fingerprint
            or capsule.geometry_mesh_discretization_fingerprint_sha256 != geometry_mesh_fingerprint
            or capsule.solver_fingerprint_sha256 != solver_fingerprint
            or capsule.observation_clock_fingerprint_sha256 != clock_fingerprint
            or capsule.receiver_schema_fingerprint_sha256 != receiver_fingerprint
            or capsule.checkpoint_input_ledger != _input_ledger(_capsule_inputs(capsule))
            or capsule.continuation_input_ledger != _input_ledger(hold_inputs)
        ):
            raise ValueError("capsule rebuild identity changed")
        state = raw_state
        start_model = target_model
        start_inputs: Mapping[str, float] = _capsule_inputs(capsule)
        if route in {
            BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP,
            BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP,
        }:
            _, _, prefix, _ = _step_prefix(
                unit=unit,
                denominator=denominator,
                history=history,
            )
            start_model = prefix.all_models[-1]
            start_inputs = {
                name: float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
                for name, value in prefix.all_inputs[-1].items()
            }
            if route is BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP:
                state = np.asarray(
                    [float(f"{item:.15g}") for item in raw_state],
                    dtype=np.float64,
                )
        elif route is BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP:
            _, source_simulation = _runtime_objects(unit, denominator)
            source_simulation.build(
                initial_soc=float(unit.initial_soc),
                inputs=_capsule_inputs(capsule),
            )
            start_model = source_simulation.built_model
        elif route is BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT:
            state, correction, pre_residual, post_residual = _preconsistent_state(
                unit=unit,
                denominator=denominator,
                checkpoint_s=history.checkpoint_s,
                raw_state=raw_state,
            )
        elif route is not BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT:
            raise ValueError("unsupported battery exact restart route")
        start = _starting_solution(
            pybamm,
            checkpoint_s=history.checkpoint_s,
            state=state,
            model=start_model,
            inputs=start_inputs,
        )
        clock = _sample_clock(denominator, history)
        solution = target_simulation.step(
            history.continuation_end_s - history.checkpoint_s,
            t_eval=clock - history.checkpoint_s,
            starting_solution=start,
            inputs=hold_inputs,
        )
        trace = _trace(solution, clock)
        if not trace.complete:
            raise RuntimeError(f"route termination: {trace.termination}")
        errors, first_error, closed = _trace_errors(
            trace,
            capture.native_trace,
            config.tolerance,
        )
        return BatteryExactRestartRouteResult(
            **common,
            trace=trace,
            capacity_error_ah=errors[0],
            mean_temperature_error_k=errors[1],
            maximum_temperature_error_k=errors[2],
            terminal_voltage_error_v=errors[3],
            first_sample_max_normalized_error=first_error,
            preconsistent_correction_max_abs=correction,
            pre_algebraic_residual_max_abs=pre_residual,
            post_algebraic_residual_max_abs=post_residual,
            capsule_bit_exact=bit_exact,
            fresh_process_isolation=fresh_process_isolation,
            receiver_closed=closed,
            disposition=(
                BatteryExactRestartRouteDisposition.COMPLETE if closed else BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED
            ),
            reason_codes=(),
            runtime_seconds=_decimal(time.monotonic() - started),
        )
    except Exception as error:
        return BatteryExactRestartRouteResult(
            **common,
            trace=None,
            capacity_error_ah=None,
            mean_temperature_error_k=None,
            maximum_temperature_error_k=None,
            terminal_voltage_error_v=None,
            first_sample_max_normalized_error=None,
            preconsistent_correction_max_abs=correction,
            pre_algebraic_residual_max_abs=pre_residual,
            post_algebraic_residual_max_abs=post_residual,
            capsule_bit_exact=False,
            fresh_process_isolation=fresh_process_isolation,
            receiver_closed=False,
            disposition=BatteryExactRestartRouteDisposition.SOURCE_ROUTE_FAILURE,
            reason_codes=(f"ROUTE_{type(error).__name__.upper()}",),
            runtime_seconds=_decimal(time.monotonic() - started),
        )


def _maximum(
    results: Sequence[BatteryExactRestartRouteResult],
    field: str,
) -> Decimal | None:
    values = [getattr(item, field) for item in results if getattr(item, field) is not None]
    return max(values) if values else None


def _route_panels(
    results: Sequence[BatteryExactRestartRouteResult],
    *,
    intended_rows: int,
    native_evaluable_rows: int,
    native_closed_rows: int,
) -> tuple[BatteryExactRestartRoutePanel, ...]:
    panels = []
    for route in BatteryExactRestartRoute:
        if route is BatteryExactRestartRoute.NATIVE_SAME_OBJECT:
            local: list[BatteryExactRestartRouteResult] = []
            evaluable = native_evaluable_rows
            closed = native_closed_rows
            maxima: tuple[
                Decimal | None,
                Decimal | None,
                Decimal | None,
                Decimal | None,
            ] = (Decimal(0),) * 4 if native_evaluable_rows == intended_rows else (None,) * 4
        else:
            local = [item for item in results if item.route is route]
            evaluable = sum(
                item.disposition
                in {BatteryExactRestartRouteDisposition.COMPLETE, BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED}
                for item in local
            )
            closed = sum(item.receiver_closed for item in local)
            maxima = (
                _maximum(local, "capacity_error_ah"),
                _maximum(local, "mean_temperature_error_k"),
                _maximum(local, "maximum_temperature_error_k"),
                _maximum(local, "terminal_voltage_error_v"),
            )
        panels.append(
            BatteryExactRestartRoutePanel(
                route=route,
                intended_rows=intended_rows,
                evaluable_rows=evaluable,
                closed_rows=closed,
                maximum_capacity_error_ah=maxima[0],
                maximum_mean_temperature_error_k=maxima[1],
                maximum_maximum_temperature_error_k=maxima[2],
                maximum_terminal_voltage_error_v=maxima[3],
                passes_all_rows=evaluable == intended_rows and closed == intended_rows,
            )
        )
    return tuple(panels)


def summarize_route_panels(
    *,
    captures: Sequence[BatteryExactRestartCapture],
    results: Sequence[BatteryExactRestartRouteResult],
    intended_rows: int,
) -> tuple[BatteryExactRestartRoutePanel, ...]:
    complete = sum(item.disposition is BatteryExactRestartCaptureDisposition.COMPLETE for item in captures)
    return _route_panels(
        results,
        intended_rows=intended_rows,
        native_evaluable_rows=complete,
        native_closed_rows=complete,
    )


def decide_localization(
    *,
    config: BatteryExactRestartStageConfig,
    source_audit: BatteryExactRestartSourceAudit,
    captures: Sequence[BatteryExactRestartCapture],
    results: Sequence[BatteryExactRestartRouteResult],
    decided_at_utc: str,
) -> BatteryExactRestartLocalizationDecision:
    if config.stage is not BatteryExactRestartStage.LOCALIZATION:
        raise ValueError("battery exact restart localization decision requires localization config")
    intended = len(config.primary_unit_ids) * len(config.denominators) * len(config.histories)
    expected_keys = {
        (unit_id, denominator.denominator_id, history.history_id)
        for unit_id in config.primary_unit_ids
        for denominator in config.denominators
        for history in config.histories
    }
    capture_keys = {(item.unit_id, item.denominator_id, item.history_id) for item in captures}
    config_identity = ObjectIdentity.from_record(config.config_id, config)
    capture_by_key = {
        (item.unit_id, item.denominator_id, item.history_id): item for item in captures
    }
    capture_identity_valid = all(
        item.config == config_identity
        and item.outcome_access is OutcomeAccess.DEVELOPMENT_VISIBLE
        and (
            item.capsule is None
            or (
                item.capsule.execution_binding.config == config_identity
                and item.capsule.source_audit
                == ObjectIdentity.from_record(source_audit.audit_id, source_audit)
            )
        )
        for item in captures
    )
    result_identity_valid = all(
        item.config == config_identity
        and item.outcome_access is OutcomeAccess.DEVELOPMENT_VISIBLE
        and (key := (item.unit_id, item.denominator_id, item.history_id)) in capture_by_key
        and item.capture
        == ObjectIdentity.from_record(
            capture_by_key[key].capture_id,
            capture_by_key[key],
        )
        and (item.route not in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE or item.fresh_process_isolation)
        for item in results
    )
    complete_captures = sum(item.disposition is BatteryExactRestartCaptureDisposition.COMPLETE for item in captures)
    panels = _route_panels(
        results,
        intended_rows=intended,
        native_evaluable_rows=complete_captures,
        native_closed_rows=complete_captures,
    )
    result_roster_complete = all(
        len([item for item in results if item.route is route]) == intended
        and {
            (item.unit_id, item.denominator_id, item.history_id)
            for item in results
            if item.route is route
        }
        == expected_keys
        for route in BatteryExactRestartRoute
        if route is not BatteryExactRestartRoute.NATIVE_SAME_OBJECT
    )
    if (
        len(captures) != intended
        or capture_keys != expected_keys
        or not result_roster_complete
        or not capture_identity_valid
        or not result_identity_valid
    ):
        return BatteryExactRestartLocalizationDecision(
            decision_id="decision.battery-exact-restart.localization",
            config=ObjectIdentity.from_record(config.config_id, config),
            source_audit=ObjectIdentity.from_record(source_audit.audit_id, source_audit),
            route_panels=panels,
            obstruction_reproduced=False,
            mechanism_disposition=BatteryExactRestartMechanismDisposition.MULTIFACTOR_OR_UNRESOLVED,
            selected_route=None,
            verdict=BatteryExactRestartLocalizationVerdict.TECHNICAL_PARTIAL,
            fresh_process_isolation_verified=False,
            decided_at_utc=decided_at_utc,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )
    if complete_captures != intended:
        return BatteryExactRestartLocalizationDecision(
            decision_id="decision.battery-exact-restart.localization",
            config=ObjectIdentity.from_record(config.config_id, config),
            source_audit=ObjectIdentity.from_record(source_audit.audit_id, source_audit),
            route_panels=panels,
            obstruction_reproduced=False,
            mechanism_disposition=BatteryExactRestartMechanismDisposition.MULTIFACTOR_OR_UNRESOLVED,
            selected_route=None,
            verdict=BatteryExactRestartLocalizationVerdict.UNEVALUABLE,
            fresh_process_isolation_verified=False,
            decided_at_utc=decided_at_utc,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )
    obstructed_denominators = {
        "denominator.pybamm.dfn.isothermal.idaklu-refined",
        "denominator.pybamm.dfn.lumped.idaklu-refined",
    }
    r1_required = [
        item
        for item in results
        if item.route is BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP
        and (
            item.denominator_id in obstructed_denominators
            or (
                item.denominator_id == "denominator.pybamm.spme.lumped.casadi-coarse"
                and ".i-then-t." in item.history_id
            )
        )
    ]
    obstruction_reproduced = bool(r1_required) and any(
        not item.receiver_closed for item in r1_required
    )
    if not obstruction_reproduced:
        return BatteryExactRestartLocalizationDecision(
            decision_id="decision.battery-exact-restart.localization",
            config=ObjectIdentity.from_record(config.config_id, config),
            source_audit=ObjectIdentity.from_record(source_audit.audit_id, source_audit),
            route_panels=panels,
            obstruction_reproduced=False,
            mechanism_disposition=BatteryExactRestartMechanismDisposition.OBSTRUCTION_NOT_REPRODUCED,
            selected_route=None,
            verdict=BatteryExactRestartLocalizationVerdict.OBSTRUCTION_NOT_REPRODUCED,
            fresh_process_isolation_verified=False,
            decided_at_utc=decided_at_utc,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )
    by_route = {item.route: item for item in panels}
    route_audits = {item.route: item for item in source_audit.route_audits}
    selected = next(
        (
            route
            for route in BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE
            if by_route[route].passes_all_rows
            and route_audits[route].feasibility is BatteryExactRestartRouteFeasibility.FEASIBLE
            and route_audits[route].capsule_only_eligible
            and all(item.fresh_process_isolation for item in results if item.route is route)
        ),
        None,
    )
    r1 = {
        (item.unit_id, item.denominator_id, item.history_id): item
        for item in results
        if item.route is BatteryExactRestartRoute.DECIMAL15_LIVE_PREFIX_MAP
    }
    r2 = {
        (item.unit_id, item.denominator_id, item.history_id): item
        for item in results
        if item.route is BatteryExactRestartRoute.LOSSLESS_LIVE_PREFIX_MAP
    }
    encoding_sensitive = any(
        not left.receiver_closed and key in r2 and r2[key].receiver_closed
        for key, left in r1.items()
    )
    mapping_sensitive = (
        by_route[BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT].passes_all_rows
        != by_route[BatteryExactRestartRoute.LOSSLESS_PUBLIC_MAP].passes_all_rows
    )
    consistency_sensitive = (
        by_route[BatteryExactRestartRoute.LOSSLESS_PRECONSISTENT].passes_all_rows
        and not by_route[BatteryExactRestartRoute.LOSSLESS_TARGET_DIRECT].passes_all_rows
    )
    mechanisms = sum((encoding_sensitive, mapping_sensitive, consistency_sensitive))
    if mechanisms != 1:
        mechanism = BatteryExactRestartMechanismDisposition.MULTIFACTOR_OR_UNRESOLVED
    elif encoding_sensitive:
        mechanism = BatteryExactRestartMechanismDisposition.NUMERIC_ENCODING_SENSITIVE
    elif mapping_sensitive:
        mechanism = BatteryExactRestartMechanismDisposition.MODEL_MAPPING_SENSITIVE
    else:
        mechanism = BatteryExactRestartMechanismDisposition.CONSISTENCY_INITIALIZATION_SENSITIVE
    return BatteryExactRestartLocalizationDecision(
        decision_id="decision.battery-exact-restart.localization",
        config=ObjectIdentity.from_record(config.config_id, config),
        source_audit=ObjectIdentity.from_record(source_audit.audit_id, source_audit),
        route_panels=panels,
        obstruction_reproduced=True,
        mechanism_disposition=mechanism,
        selected_route=selected,
        verdict=(
            BatteryExactRestartLocalizationVerdict.ROUTE_SELECTED
            if selected is not None
            else BatteryExactRestartLocalizationVerdict.SOURCE_ROUTE_INSUFFICIENT
        ),
        fresh_process_isolation_verified=selected is not None,
        decided_at_utc=decided_at_utc,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def adjudicate_qualification(
    *,
    freeze: BatteryExactRestartQualificationFreeze,
    config: BatteryExactRestartStageConfig,
    captures: Sequence[BatteryExactRestartCapture],
    results: Sequence[BatteryExactRestartRouteResult],
    revealed_at_utc: str,
) -> BatteryExactRestartAdjudication:
    if (
        config.stage is not BatteryExactRestartStage.QUALIFICATION
        or config.selected_route is None
        or freeze.selected_route is not config.selected_route
        or freeze.qualification_config != ObjectIdentity.from_record(config.config_id, config)
        or freeze.tolerance != config.tolerance
    ):
        raise ValueError("battery exact restart qualification adjudication has incompatible inputs")
    wanted_units = set(config.primary_unit_ids)
    capture_rows = [item for item in captures if item.unit_id in wanted_units]
    route_rows = [
        item
        for item in results
        if item.unit_id in wanted_units and item.route is config.selected_route
    ]
    config_identity = ObjectIdentity.from_record(config.config_id, config)
    capture_by_key = {
        (item.unit_id, item.denominator_id, item.history_id): item for item in capture_rows
    }
    structural_identity_valid = all(
        item.config == config_identity
        and item.outcome_access is OutcomeAccess.EVALUATION_SEALED
        and (
            item.capsule is None
            or (
                item.capsule.execution_binding.config == config_identity
                and item.capsule.execution_binding.source_audit == freeze.source_audit
                and item.capsule.execution_binding.implementation == freeze.implementation
            )
        )
        for item in capture_rows
    ) and all(
        item.config == config_identity
        and item.outcome_access is OutcomeAccess.EVALUATION_SEALED
        and item.fresh_process_isolation
        and (key := (item.unit_id, item.denominator_id, item.history_id)) in capture_by_key
        and item.capture
        == ObjectIdentity.from_record(
            capture_by_key[key].capture_id,
            capture_by_key[key],
        )
        for item in route_rows
    )
    expected_keys = {
        (unit_id, denominator.denominator_id, history.history_id)
        for unit_id in config.primary_unit_ids
        for denominator in config.denominators
        for history in config.histories
    }
    capture_keys = {(item.unit_id, item.denominator_id, item.history_id) for item in capture_rows}
    route_keys = {(item.unit_id, item.denominator_id, item.history_id) for item in route_rows}
    rows: list[BatteryExactRestartClosureRow] = []
    for denominator in config.denominators:
        for history in config.histories:
            local = [
                item
                for item in route_rows
                if item.denominator_id == denominator.denominator_id
                and item.history_id == history.history_id
            ]
            evaluable = [
                item
                for item in local
                if item.disposition
                in {BatteryExactRestartRouteDisposition.COMPLETE, BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED}
            ]
            closed = [item for item in evaluable if item.receiver_closed]
            closure = (
                BatteryExactRestartClosureClass.CLOSED
                if len(closed) == 6
                else (
                    BatteryExactRestartClosureClass.OPPOSED if len(evaluable) == 6 else BatteryExactRestartClosureClass.UNEVALUABLE
                )
            )
            rows.append(
                BatteryExactRestartClosureRow(
                    denominator_id=denominator.denominator_id,
                    history_id=history.history_id,
                    intended_unit_count=6,
                    evaluable_unit_count=len(evaluable),
                    closed_unit_count=len(closed),
                    maximum_capacity_error_ah=_maximum(local, "capacity_error_ah"),
                    maximum_mean_temperature_error_k=_maximum(
                        local,
                        "mean_temperature_error_k",
                    ),
                    maximum_maximum_temperature_error_k=_maximum(
                        local,
                        "maximum_temperature_error_k",
                    ),
                    maximum_terminal_voltage_error_v=_maximum(
                        local,
                        "terminal_voltage_error_v",
                    ),
                    closure_class=closure,
                )
            )
    ordered = tuple(sorted(rows, key=lambda item: (item.denominator_id, item.history_id)))
    qualified = tuple(
        sorted(
            denominator.denominator_id
            for denominator in config.denominators
            if all(
                item.closure_class is BatteryExactRestartClosureClass.CLOSED
                for item in ordered
                if item.denominator_id == denominator.denominator_id
            )
        )
    )
    if (
        len(capture_rows) != 240
        or len(route_rows) != 240
        or capture_keys != expected_keys
        or route_keys != expected_keys
        or not structural_identity_valid
    ):
        verdict = BatteryExactRestartQualificationVerdict.TECHNICAL_PARTIAL
    elif any(item.closure_class is BatteryExactRestartClosureClass.UNEVALUABLE for item in ordered):
        verdict = BatteryExactRestartQualificationVerdict.UNEVALUABLE
    elif len(qualified) == len(config.denominators):
        verdict = BatteryExactRestartQualificationVerdict.QUALIFIED_ALL
    elif qualified:
        verdict = BatteryExactRestartQualificationVerdict.DENOMINATOR_CONDITIONAL
    else:
        verdict = BatteryExactRestartQualificationVerdict.NOT_QUALIFIED
    return BatteryExactRestartAdjudication(
        adjudication_id="adjudication.battery-exact-restart.qualification",
        freeze=ObjectIdentity.from_record(freeze.freeze_id, freeze),
        config=ObjectIdentity.from_record(config.config_id, config),
        selected_route=config.selected_route,
        closure_rows=ordered,
        complete_capture_count=sum(
            item.disposition is BatteryExactRestartCaptureDisposition.COMPLETE for item in capture_rows
        ),
        intended_capture_count=240,
        complete_route_result_count=sum(
            item.disposition in {BatteryExactRestartRouteDisposition.COMPLETE, BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED}
            for item in route_rows
        ),
        intended_route_result_count=240,
        qualified_denominator_ids=(
            qualified
            if verdict
            not in {
                BatteryExactRestartQualificationVerdict.TECHNICAL_PARTIAL,
                BatteryExactRestartQualificationVerdict.UNEVALUABLE,
            }
            else ()
        ),
        verdict=verdict,
        revealed_at_utc=revealed_at_utc,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def _three_number_summary(
    results: Sequence[BatteryExactRestartRouteResult],
    field: str,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    values = np.asarray(
        [float(value) for item in results if (value := getattr(item, field)) is not None],
        dtype=np.float64,
    )
    if not values.size:
        return None, None, None
    return (
        _decimal(float(np.median(values))),
        _decimal(float(np.quantile(values, 0.95, method="linear"))),
        _decimal(float(np.max(values))),
    )


def build_descriptive_analysis(
    *,
    localization_decision: BatteryExactRestartLocalizationDecision,
    localization_config: BatteryExactRestartStageConfig,
    localization_results: Sequence[BatteryExactRestartRouteResult],
    adjudication: BatteryExactRestartAdjudication,
    qualification_config: BatteryExactRestartStageConfig,
    qualification_results: Sequence[BatteryExactRestartRouteResult],
    analyzed_at_utc: str,
) -> BatteryExactRestartDescriptiveAnalysis:
    if localization_decision.selected_route is None:
        raise ValueError("battery exact restart descriptive analysis requires a selected route")
    rows: list[BatteryExactRestartErrorSummaryRow] = []
    panels = (
        (
            "localization",
            localization_config,
            tuple(BatteryExactRestartRoute)[1:],
            localization_results,
        ),
        (
            "qualification",
            qualification_config,
            (qualification_config.selected_route,),
            qualification_results,
        ),
    )
    fields = (
        "capacity_error_ah",
        "mean_temperature_error_k",
        "maximum_temperature_error_k",
        "terminal_voltage_error_v",
    )
    for panel, config, routes, results in panels:
        intended = len(config.primary_unit_ids)
        for route in routes:
            assert route is not None
            for denominator in config.denominators:
                for history in config.histories:
                    local = [
                        item
                        for item in results
                        if item.route is route
                        and item.denominator_id == denominator.denominator_id
                        and item.history_id == history.history_id
                        and item.unit_id in config.primary_unit_ids
                    ]
                    evaluable = [
                        item
                        for item in local
                        if item.disposition
                        in {
                            BatteryExactRestartRouteDisposition.COMPLETE,
                            BatteryExactRestartRouteDisposition.RECEIVER_OPPOSED,
                        }
                    ]
                    summaries = tuple(_three_number_summary(local, field) for field in fields)
                    rows.append(
                        BatteryExactRestartErrorSummaryRow(
                            panel=panel,
                            route=route,
                            denominator_id=denominator.denominator_id,
                            history_id=history.history_id,
                            intended_count=intended,
                            evaluable_count=len(evaluable),
                            closed_count=sum(item.receiver_closed for item in evaluable),
                            median_capacity_error_ah=summaries[0][0],
                            p95_capacity_error_ah=summaries[0][1],
                            maximum_capacity_error_ah=summaries[0][2],
                            median_mean_temperature_error_k=summaries[1][0],
                            p95_mean_temperature_error_k=summaries[1][1],
                            maximum_mean_temperature_error_k=summaries[1][2],
                            median_maximum_temperature_error_k=summaries[2][0],
                            p95_maximum_temperature_error_k=summaries[2][1],
                            maximum_maximum_temperature_error_k=summaries[2][2],
                            median_terminal_voltage_error_v=summaries[3][0],
                            p95_terminal_voltage_error_v=summaries[3][1],
                            maximum_terminal_voltage_error_v=summaries[3][2],
                            maximum_first_sample_normalized_error=_maximum(
                                local,
                                "first_sample_max_normalized_error",
                            ),
                            maximum_preconsistent_correction_abs=_maximum(
                                local,
                                "preconsistent_correction_max_abs",
                            ),
                        )
                    )
    return BatteryExactRestartDescriptiveAnalysis(
        analysis_id="analysis.battery-exact-restart.descriptive",
        localization_decision=ObjectIdentity.from_record(
            localization_decision.decision_id,
            localization_decision,
        ),
        adjudication=ObjectIdentity.from_record(
            adjudication.adjudication_id,
            adjudication,
        ),
        selected_route=localization_decision.selected_route,
        rows=tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.panel,
                    item.route.value,
                    item.denominator_id,
                    item.history_id,
                ),
            )
        ),
        analyzed_at_utc=analyzed_at_utc,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'BatteryExactRestartAdjudication',
    'BatteryExactRestartCapture',
    'BatteryExactRestartCaptureDisposition',
    'BatteryExactRestartCanaryReport',
    'BatteryExactRestartCheckpointCapsule',
    'BatteryExactRestartClosureClass',
    'BatteryExactRestartClosureRow',
    'BatteryExactRestartDesign',
    'BatteryExactRestartDescriptiveAnalysis',
    'BatteryExactRestartErrorSummaryRow',
    'BatteryExactRestartExecutionBinding',
    'BatteryExactRestartExactTolerance',
    'BatteryExactRestartHistory',
    'BatteryExactRestartInputLedger',
    'BatteryExactRestartLocalizationDecision',
    'BatteryExactRestartLocalizationVerdict',
    'BatteryExactRestartMechanismDisposition',
    'BatteryExactRestartPreparation',
    'BatteryExactRestartQualificationFreeze',
    'BatteryExactRestartQualificationVerdict',
    'BatteryExactRestartResourceEnvelope',
    'BatteryExactRestartReceiverTrace',
    'BatteryExactRestartRunIdentity',
    'BatteryExactRestartSelectedRouteCanary',
    'BatteryExactRestartRoute',
    'BatteryExactRestartRouteAudit',
    'BatteryExactRestartRouteDisposition',
    'BatteryExactRestartRouteFeasibility',
    'BatteryExactRestartRoutePanel',
    'BatteryExactRestartSourceAudit',
    'BatteryExactRestartStage',
    'BatteryExactRestartStageConfig',
    'BatteryExactRestartUnitRole',
    "BATTERY_EXACT_RESTART_CAPACITY_FLOOR_AH",
    "BATTERY_EXACT_RESTART_DESIGN_ID",
    "BATTERY_EXACT_RESTART_EXTERNAL_ROOT",
    "BATTERY_EXACT_RESTART_MAXIMUM_RECORD_BYTES",
    "BATTERY_EXACT_RESTART_MINIMUM_FREE_BYTES",
    "BATTERY_EXACT_RESTART_PLAN_ID",
    "BATTERY_EXACT_RESTART_PROVIDER_KEY",
    "BATTERY_EXACT_RESTART_QUALIFICATION_ROUTE_PREFERENCE",
    "BATTERY_EXACT_RESTART_ROUTE_ROSTER",
    "BATTERY_EXACT_RESTART_SOURCE_VERSION",
    "BATTERY_EXACT_RESTART_TEMPERATURE_FLOOR_K",
    "BATTERY_EXACT_RESTART_VOLTAGE_FLOOR_V",
    "acquire_route_result",
    "adjudicate_qualification",
    "build_design",
    "build_descriptive_analysis",
    "build_localization_config",
    "build_qualification_config",
    "capture_reference",
    "decode_hex",
    "decide_localization",
    "exact_tolerance",
    "histories",
    "route_audits",
    "source_semantic_facts",
    "summarize_route_panels",
    "validate_repository_config",
    "verify_denominator_builds",
]
