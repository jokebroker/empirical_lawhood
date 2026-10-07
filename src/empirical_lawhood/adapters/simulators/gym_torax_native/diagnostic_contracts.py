"""Current Gym--TORAX native source and delivery contracts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import hashlib
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .extraction_manifest import GymToraxBoundedExtractionManifest
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


GYM_TORAX_HORIZON_REQUESTS = 120
GYM_TORAX_PRIMARY_MEMBER_ID = 'member.tokamak-control.primary'
GYM_TORAX_REFINED_MEMBER_ID = 'member.tokamak-control.refined'


class GymToraxSourceDisposition(StrEnum):
    AVAILABLE = "AVAILABLE"
    OPERATOR_API_UNAVAILABLE = "OPERATOR_API_UNAVAILABLE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


class GymToraxDeliveryDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    CLIPPED = "CLIPPED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


class GymToraxNumericalDisposition(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    SOLVER_FAILURE = "SOLVER_FAILURE"
    UNEVALUABLE = "UNEVALUABLE"


class GymToraxObservationDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class GymToraxPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-preparation'

    preparation_id: str
    physical_independent_unit_id: str
    environment_seed: int
    initial_temperature_scale: Decimal
    initial_density_nbar: Decimal
    bootstrap_multiplier: Decimal
    inner_transport_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        validate_stable_id(
            self.physical_independent_unit_id,
            field_name="physical_independent_unit_id",
        )
        if self.environment_seed < 0:
            raise ValueError("Gym--TORAX environment seed must be nonnegative")
        for field_name, value, lower, upper in (
            (
                "initial_temperature_scale",
                self.initial_temperature_scale,
                Decimal("0.990"),
                Decimal("1.020"),
            ),
            (
                "initial_density_nbar",
                self.initial_density_nbar,
                Decimal("0.848"),
                Decimal("0.852"),
            ),
            (
                "bootstrap_multiplier",
                self.bootstrap_multiplier,
                Decimal("0.990"),
                Decimal("1.005"),
            ),
            (
                "inner_transport_scale",
                self.inner_transport_scale,
                Decimal("0.990"),
                Decimal("1.010"),
            ),
        ):
            validate_decimal(value, field_name=field_name, minimum=lower)
            if value > upper:
                raise ValueError(f"{field_name} exceeds the frozen preparation chart")


@dataclass(frozen=True, slots=True)
class GymToraxNumericalMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-numerical-member'

    member_id: str
    internal_timestep_s: Decimal
    radial_cells: int
    corrector_steps: int
    solver_id: str
    predictor_corrector: bool
    pereverzev_enabled: bool
    pereverzev_chi: Decimal
    pereverzev_d: Decimal
    backend: str
    precision: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_stable_id(self.solver_id, field_name="solver_id")
        expected = {
            GYM_TORAX_PRIMARY_MEMBER_ID: (Decimal("1"), 25, 10),
            GYM_TORAX_REFINED_MEMBER_ID: (Decimal("0.5"), 33, 20),
        }
        if expected.get(self.member_id) != (
            self.internal_timestep_s,
            self.radial_cells,
            self.corrector_steps,
        ):
            raise ValueError("Gym--TORAX numerical member identity/mesh differs")
        if (
            self.solver_id != "solver.torax.linear-theta"
            or not self.predictor_corrector
            or not self.pereverzev_enabled
            or self.pereverzev_chi != Decimal("30")
            or self.pereverzev_d != Decimal("15")
            or self.backend != "cpu"
            or self.precision != "float64"
        ):
            raise ValueError("Gym--TORAX numerical denominator changed")


def gym_torax_numerical_members() -> tuple[GymToraxNumericalMember, ...]:
    """Return the exact nonpooled primary/refined denominator members."""

    return tuple(
        sorted(
            (
                GymToraxNumericalMember(
                    member_id=GYM_TORAX_PRIMARY_MEMBER_ID,
                    internal_timestep_s=Decimal("1"),
                    radial_cells=25,
                    corrector_steps=10,
                    solver_id="solver.torax.linear-theta",
                    predictor_corrector=True,
                    pereverzev_enabled=True,
                    pereverzev_chi=Decimal("30"),
                    pereverzev_d=Decimal("15"),
                    backend="cpu",
                    precision="float64",
                ),
                GymToraxNumericalMember(
                    member_id=GYM_TORAX_REFINED_MEMBER_ID,
                    internal_timestep_s=Decimal("0.5"),
                    radial_cells=33,
                    corrector_steps=20,
                    solver_id="solver.torax.linear-theta",
                    predictor_corrector=True,
                    pereverzev_enabled=True,
                    pereverzev_chi=Decimal("30"),
                    pereverzev_d=Decimal("15"),
                    backend="cpu",
                    precision="float64",
                ),
            ),
            key=lambda value: value.member_id,
        )
    )


@dataclass(frozen=True, slots=True)
class GymToraxNativeAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-action'

    ip_a: Decimal
    nbi_power_w: Decimal
    nbi_location: Decimal
    nbi_width: Decimal
    ecrh_power_w: Decimal
    ecrh_location: Decimal
    ecrh_width: Decimal

    def __post_init__(self) -> None:
        for field_name, value, lower, upper in (
            ("ip_a", self.ip_a, Decimal("100000"), Decimal("15000000")),
            ("nbi_power_w", self.nbi_power_w, Decimal(0), Decimal("33000000.5")),
            ("nbi_location", self.nbi_location, Decimal(0), Decimal(1)),
            ("nbi_width", self.nbi_width, Decimal("0.01"), Decimal(1)),
            ("ecrh_power_w", self.ecrh_power_w, Decimal(0), Decimal("20000000.5")),
            ("ecrh_location", self.ecrh_location, Decimal(0), Decimal(1)),
            ("ecrh_width", self.ecrh_width, Decimal("0.01"), Decimal(1)),
        ):
            validate_decimal(value, field_name=field_name, minimum=lower)
            if value > upper:
                raise ValueError(f"{field_name} exceeds the native action bound")
        if (
            self.nbi_location != Decimal("0.25")
            or self.nbi_width != Decimal("0.25")
            or self.ecrh_location != Decimal("0.35")
            or self.ecrh_width != Decimal("0.05")
        ):
            raise ValueError("Gym--TORAX native deposition coordinates changed")


@dataclass(frozen=True, slots=True)
class GymToraxNativeActionRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-action-row'

    row_id: str
    request_clock: int
    action: GymToraxNativeAction
    controlled_occurrence_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        if not 0 <= self.request_clock < GYM_TORAX_HORIZON_REQUESTS:
            raise ValueError("Gym--TORAX action row is outside requests 0--119")
        if self.controlled_occurrence_id is not None:
            validate_stable_id(
                self.controlled_occurrence_id,
                field_name="controlled_occurrence_id",
            )


@dataclass(frozen=True, slots=True)
class GymToraxNativeSchedule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-schedule'

    schedule_id: str
    action_word: OccurrenceActionWord
    rows: tuple[GymToraxNativeActionRow, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.schedule_id, field_name="schedule_id")
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        if tuple(value.request_clock for value in self.rows) != tuple(
            range(GYM_TORAX_HORIZON_REQUESTS)
        ):
            raise ValueError("Gym--TORAX native schedule must cover requests 0--119")
        if max(
            abs(right.action.ip_a - left.action.ip_a)
            for left, right in zip(self.rows, self.rows[1:], strict=False)
        ) > Decimal("200000"):
            raise ValueError("Gym--TORAX native schedule exceeds the Ip slew bound")
        occurrence_by_clock = {
            value.request_clock: value.controlled_occurrence_id
            for value in self.rows
            if value.controlled_occurrence_id is not None
        }
        expected = {
            int(occurrence.requested.coordinate.coordinate): occurrence.occurrence_id
            for occurrence in self.action_word.occurrences
        }
        if occurrence_by_clock != expected:
            raise ValueError("native schedule changes the exact ActionWord occurrence map")


@dataclass(frozen=True, slots=True)
class GymToraxActionDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-action-delivery'

    delivery_id: str
    request_clock: int
    receiver_clock: int
    occurrence_id: str | None
    requested: GymToraxNativeAction
    accepted: GymToraxNativeAction | None
    applied: GymToraxNativeAction | None
    realized: GymToraxNativeAction | None
    disposition: GymToraxDeliveryDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.delivery_id, field_name="delivery_id")
        if not 0 <= self.request_clock < GYM_TORAX_HORIZON_REQUESTS:
            raise ValueError("Gym--TORAX delivery request clock is outside the horizon")
        if self.receiver_clock != self.request_clock + 1:
            raise ValueError("Gym--TORAX delivery changed request/receiver clock order")
        if self.occurrence_id is not None:
            validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        stages_complete = all(
            value is not None for value in (self.accepted, self.applied, self.realized)
        )
        if self.disposition is GymToraxDeliveryDisposition.COMPLETE:
            if not stages_complete or self.reason_codes:
                raise ValueError("complete Gym--TORAX delivery requires all exact stages")
        elif not self.reason_codes:
            raise ValueError("noncomplete Gym--TORAX delivery requires a reason")


@dataclass(frozen=True, slots=True)
class GymToraxDiagnosticFloat64Block(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-diagnostic-float64-block'

    block_id: str
    category: str
    native_field_id: str
    native_unit: str
    dimension_ids: tuple[str, ...]
    shape: tuple[int, ...]
    clock_values: tuple[int, ...]
    data_base64: str
    data_sha256: str
    finite_value_count: int
    nonfinite_value_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.block_id, field_name="block_id")
        validate_stable_id(self.category, field_name="category")
        validate_nonempty(self.native_field_id, field_name="native_field_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if not self.dimension_ids or any(not value for value in self.dimension_ids):
            raise ValueError("dimension_ids must contain nonempty ordered axis labels")
        if len(set(self.dimension_ids)) != len(self.dimension_ids):
            raise ValueError("dimension_ids must be unique")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("Gym--TORAX block shape must be positive")
        if self.finite_value_count < 0 or self.nonfinite_value_count < 0:
            raise ValueError("Gym--TORAX block counts must be nonnegative")
        if self.finite_value_count + self.nonfinite_value_count != int(np.prod(self.shape)):
            raise ValueError("Gym--TORAX block counts differ from shape")
        payload = base64.b64decode(self.data_base64, validate=True)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if len(payload) != int(np.prod(self.shape)) * 8:
            raise ValueError("Gym--TORAX block byte count differs from little-endian float64")
        if hashlib.sha256(payload).hexdigest() != self.data_sha256:
            raise ValueError("Gym--TORAX block digest differs")
        if self.clock_values != tuple(sorted(set(self.clock_values))):
            raise ValueError("Gym--TORAX block clocks must be strictly ordered and unique")
        if self.clock_values and self.shape[0] != len(self.clock_values):
            raise ValueError("Gym--TORAX block clock axis differs from shape")

    def array(self) -> npt.NDArray[np.float64]:
        return np.frombuffer(base64.b64decode(self.data_base64), dtype="<f8").reshape(self.shape)


def encode_gym_torax_float64_block(
    *,
    block_id: str,
    category: str,
    native_field_id: str,
    native_unit: str,
    dimension_ids: tuple[str, ...],
    values: npt.ArrayLike,
    clock_values: tuple[int, ...] = (),
) -> GymToraxDiagnosticFloat64Block:
    """Encode one bounded little-endian native array without interpretation."""

    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim == 0:
        array = array.reshape(1)
    payload = array.tobytes(order="C")
    finite_count = int(np.isfinite(array).sum())
    return GymToraxDiagnosticFloat64Block(
        block_id=block_id,
        category=category,
        native_field_id=native_field_id,
        native_unit=native_unit,
        dimension_ids=dimension_ids,
        shape=tuple(int(value) for value in array.shape),
        clock_values=clock_values,
        data_base64=base64.b64encode(payload).decode("ascii"),
        data_sha256=hashlib.sha256(payload).hexdigest(),
        finite_value_count=finite_count,
        nonfinite_value_count=array.size - finite_count,
    )


@dataclass(frozen=True, slots=True)
class GymToraxOperatorSourceSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-operator-source-summary'

    summary_id: str
    disposition: GymToraxSourceDisposition
    source_method_id: str
    state_coordinate_ids: tuple[str, ...]
    input_coordinate_ids: tuple[str, ...]
    receiver_coordinate_ids: tuple[str, ...]
    operator_block_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.source_method_id, field_name="source_method_id")
        coordinate_fields = (
            ("state_coordinate_ids", self.state_coordinate_ids),
            ("input_coordinate_ids", self.input_coordinate_ids),
            ("receiver_coordinate_ids", self.receiver_coordinate_ids),
        )
        for field_name, values in coordinate_fields:
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique ordered labels")
            if self.disposition is GymToraxSourceDisposition.AVAILABLE and not values:
                raise ValueError(f"{field_name} must not be empty")
        require_sorted_unique_strings(
            self.operator_block_ids,
            field_name="operator_block_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is GymToraxSourceDisposition.AVAILABLE:
            if (
                any(not values for _, values in coordinate_fields)
                or not self.operator_block_ids
                or self.reason_codes
            ):
                raise ValueError(
                    "available Gym--TORAX operators require coordinates, blocks, and no reasons"
                )
        elif not self.reason_codes or self.operator_block_ids:
            raise ValueError("unavailable Gym--TORAX operators require reasons and no blocks")


@dataclass(frozen=True, slots=True)
class GymToraxDiagnosticEpisodeRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-diagnostic-episode-request'

    request_id: str
    source_qualification: ObjectIdentity
    extraction_manifest: ObjectIdentity
    preparation: GymToraxPreparation
    numerical_member: GymToraxNumericalMember
    schedule: GymToraxNativeSchedule
    maximum_output_bytes: int
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        if self.extraction_manifest.object_schema != GymToraxBoundedExtractionManifest.SCHEMA:
            raise ValueError("Gym--TORAX request requires the extraction manifest")
        if self.maximum_output_bytes <= 0 or self.maximum_output_bytes > 512 * 1024**2:
            raise ValueError("Gym--TORAX episode output bound is outside 512 MiB")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("Gym--TORAX request has an unsupported outcome-access lane")
        if self.evidence_ceiling not in {
            EvidenceCeiling.NON_PROMOTABLE,
            EvidenceCeiling.MEASUREMENT,
        }:
            raise ValueError("Gym--TORAX source request cannot claim response-law truth")


@dataclass(frozen=True, slots=True)
class GymToraxDiagnosticNativeEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-diagnostic-native-episode'

    episode_id: str
    request: ObjectIdentity
    preparation: ObjectIdentity
    numerical_member: ObjectIdentity
    action_word: ObjectIdentity
    state_clocks: tuple[int, ...]
    missing_required_state_clocks: tuple[int, ...]
    last_valid_state_clock: int | None
    deliveries: tuple[GymToraxActionDelivery, ...]
    blocks: tuple[GymToraxDiagnosticFloat64Block, ...]
    operator_source: GymToraxOperatorSourceSummary
    source_disposition: GymToraxSourceDisposition
    delivery_disposition: GymToraxDeliveryDisposition
    numerical_disposition: GymToraxNumericalDisposition
    observation_disposition: GymToraxObservationDisposition
    termination: bool
    truncation: bool
    backend: str
    precision: str
    gymtorax_version: str
    torax_version: str
    jax_version: str
    jaxlib_version: str
    runtime_seconds: Decimal
    peak_rss_bytes: int
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        if self.request.object_schema != GymToraxDiagnosticEpisodeRequest.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another request schema")
        if self.preparation.object_schema != GymToraxPreparation.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another preparation schema")
        if self.numerical_member.object_schema != GymToraxNumericalMember.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another member schema")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("Gym--TORAX episode requires an exact current ActionWord")
        if self.state_clocks != tuple(sorted(set(self.state_clocks))):
            raise ValueError("Gym--TORAX state clocks must be strictly ordered and unique")
        if self.missing_required_state_clocks != tuple(
            sorted(set(self.missing_required_state_clocks))
        ):
            raise ValueError("Gym--TORAX missing clocks must be sorted and unique")
        require_sorted_unique_ids(
            self.deliveries,
            attribute="delivery_id",
            field_name="deliveries",
        )
        require_sorted_unique_ids(self.blocks, attribute="block_id", field_name="blocks")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.last_valid_state_clock != (max(self.state_clocks) if self.state_clocks else None):
            raise ValueError("Gym--TORAX last-valid clock differs from its state roster")
        validate_decimal(self.runtime_seconds, field_name="runtime_seconds", minimum=Decimal(0))
        if self.peak_rss_bytes < 0:
            raise ValueError("Gym--TORAX peak RSS must be nonnegative")
        if (
            self.backend != "cpu"
            or self.precision != "float64"
            or self.gymtorax_version != "1.1.1"
            or self.torax_version != "1.4.2"
            or self.jax_version != "0.10.2"
            or self.jaxlib_version != "0.10.2"
        ):
            raise ValueError("Gym--TORAX episode runtime denominator differs")
        if self.evidence_ceiling not in {
            EvidenceCeiling.NON_PROMOTABLE,
            EvidenceCeiling.MEASUREMENT,
        }:
            raise ValueError("Gym--TORAX episode cannot claim response-law truth")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("Gym--TORAX source episode cannot reveal outcomes")
        if self.source_disposition is GymToraxSourceDisposition.OPERATOR_API_UNAVAILABLE:
            raise ValueError("episode source disposition cannot conflate operator API feasibility")
        if not self.reason_codes:
            complete = (
                self.source_disposition is GymToraxSourceDisposition.AVAILABLE
                and self.delivery_disposition is GymToraxDeliveryDisposition.COMPLETE
                and self.numerical_disposition is GymToraxNumericalDisposition.VALID
                and self.observation_disposition is GymToraxObservationDisposition.COMPLETE
                and not self.termination
                and not self.truncation
                and not self.missing_required_state_clocks
            )
            if not complete:
                raise ValueError("noncomplete Gym--TORAX episode requires typed reasons")


__all__ = [
    "GYM_TORAX_HORIZON_REQUESTS",
    "GYM_TORAX_PRIMARY_MEMBER_ID",
    "GYM_TORAX_REFINED_MEMBER_ID",
    'GymToraxActionDelivery',
    "GymToraxDeliveryDisposition",
    'GymToraxDiagnosticEpisodeRequest',
    'GymToraxDiagnosticFloat64Block',
    'GymToraxNativeActionRow',
    'GymToraxNativeAction',
    'GymToraxDiagnosticNativeEpisode',
    'GymToraxNativeSchedule',
    "GymToraxNumericalDisposition",
    'GymToraxNumericalMember',
    "GymToraxObservationDisposition",
    'GymToraxOperatorSourceSummary',
    'GymToraxPreparation',
    "GymToraxSourceDisposition",
    'encode_gym_torax_float64_block',
    'gym_torax_numerical_members',
]
