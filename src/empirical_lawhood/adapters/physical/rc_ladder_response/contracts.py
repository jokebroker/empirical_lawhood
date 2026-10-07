"""Draft-profile physical bundle contracts for a read-only physical scale morphism source."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


class RcLadderResponseActionDeliveryStatus(StrEnum):
    REALIZED = "REALIZED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"


class RcLadderResponseEpisodeRole(StrEnum):
    CANARY = "CANARY"
    COMMISSIONING = "COMMISSIONING"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION_SEALED = "EVALUATION_SEALED"


@dataclass(frozen=True, slots=True)
class RcLadderResponseClockFitRecord(CanonicalRecord):
    """Bounded native-to-reference clock fit for one acquisition clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-clock-fit-record'

    clock_fit_id: str
    native_clock_id: str
    reference_clock_id: str
    offset_seconds: Decimal
    drift_seconds_per_second: Decimal
    maximum_residual_seconds: Decimal
    uncertainty_seconds: Decimal
    calibration_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("clock_fit_id", "native_clock_id", "reference_clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "offset_seconds",
            "drift_seconds_per_second",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        for name in ("maximum_residual_seconds", "uncertainty_seconds"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(
            self.calibration_evidence_ids,
            field_name="calibration_evidence_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class RcLadderResponseChannelCalibration(CanonicalRecord):
    """Exact native channel order, ranges, resolution and uncertainty."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-channel-calibration'

    calibration_id: str
    channel_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    range_lower_native: tuple[Decimal, ...]
    range_upper_native: tuple[Decimal, ...]
    resolution_native: tuple[Decimal, ...]
    uncertainty_native: tuple[Decimal, ...]
    calibration_evidence_ids: tuple[str, ...]
    measured_before_response: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.calibration_id, field_name="calibration_id")
        require_sorted_unique_strings(self.channel_ids, field_name="channel_ids", allow_empty=False)
        if len(self.native_units) != len(self.channel_ids):
            raise ValueError("channel calibration unit roster differs from channel order")
        for index, unit in enumerate(self.native_units):
            validate_nonempty(unit, field_name=f"native_units[{index}]")
        values = (
            self.range_lower_native,
            self.range_upper_native,
            self.resolution_native,
            self.uncertainty_native,
        )
        if any(len(value) != len(self.channel_ids) for value in values):
            raise ValueError("channel calibration value rosters differ from channel order")
        for index, (lower, upper, resolution, uncertainty) in enumerate(zip(*values, strict=True)):
            validate_decimal(lower, field_name=f"range_lower_native[{index}]")
            validate_decimal(upper, field_name=f"range_upper_native[{index}]")
            if upper <= lower:
                raise ValueError("channel calibration range must have positive width")
            validate_decimal(
                resolution, field_name=f"resolution_native[{index}]", minimum=Decimal(0)
            )
            validate_decimal(
                uncertainty,
                field_name=f"uncertainty_native[{index}]",
                minimum=Decimal(0),
            )
            if resolution == 0:
                raise ValueError("channel calibration resolution must be positive")
        require_sorted_unique_strings(
            self.calibration_evidence_ids,
            field_name="calibration_evidence_ids",
            allow_empty=False,
        )
        if not self.measured_before_response:
            raise ValueError("evaluation channel calibration must precede response exposure")


@dataclass(frozen=True, slots=True)
class RcLadderResponseBoardIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-board-identity'

    board_id: str
    implementation_id: str
    serial_id: str
    scale_cells: int
    batch_id: str
    panel_id: str
    physical_length_millimetres: Decimal
    resistance_component_ids: tuple[str, ...]
    capacitance_component_ids: tuple[str, ...]
    source_resistance_component_id: str
    termination_resistance_component_id: str

    def __post_init__(self) -> None:
        for name in (
            "board_id",
            "implementation_id",
            "serial_id",
            "batch_id",
            "panel_id",
            "source_resistance_component_id",
            "termination_resistance_component_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scale_cells not in {16, 32, 64}:
            raise ValueError("physical physical scale morphism board must have 16, 32 or 64 cells")
        for name, values, expected_count in (
            ("resistance_component_ids", self.resistance_component_ids, self.scale_cells - 1),
            ("capacitance_component_ids", self.capacitance_component_ids, self.scale_cells),
        ):
            if len(values) != expected_count:
                raise ValueError(f"{name} differs from the board topology")
            for index, value in enumerate(values):
                validate_stable_id(value, field_name=f"{name}[{index}]")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} repeats a physical component")
        component_ids = (
            *self.resistance_component_ids,
            *self.capacitance_component_ids,
            self.source_resistance_component_id,
            self.termination_resistance_component_id,
        )
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("board component identities must be globally unique")
        validate_decimal(
            self.physical_length_millimetres,
            field_name="physical_length_millimetres",
            minimum=Decimal(0),
        )
        if self.physical_length_millimetres == 0:
            raise ValueError("physical board length must be positive")


@dataclass(frozen=True, slots=True)
class RcLadderResponseComponentMetrology(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-component-metrology'

    metrology_id: str
    board_id: str
    resistance_component_ids: tuple[str, ...]
    capacitance_component_ids: tuple[str, ...]
    source_resistance_component_id: str
    termination_resistance_component_id: str
    resistance_ohms: tuple[Decimal, ...]
    resistance_uncertainty_ohms: tuple[Decimal, ...]
    capacitance_farads: tuple[Decimal, ...]
    capacitance_uncertainty_farads: tuple[Decimal, ...]
    source_resistance_ohms: Decimal
    source_resistance_uncertainty_ohms: Decimal
    termination_resistance_ohms: Decimal
    termination_resistance_uncertainty_ohms: Decimal
    voltage_uncertainty_volts: Decimal
    current_uncertainty_amperes: Decimal
    measured_before_response: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.metrology_id, field_name="metrology_id")
        validate_stable_id(self.board_id, field_name="board_id")
        for name in (
            "source_resistance_component_id",
            "termination_resistance_component_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name, values in (
            ("resistance_component_ids", self.resistance_component_ids),
            ("capacitance_component_ids", self.capacitance_component_ids),
        ):
            if not values:
                raise ValueError(f"{name} cannot be empty")
            for index, value in enumerate(values):
                validate_stable_id(value, field_name=f"{name}[{index}]")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} repeats a component")
        component_ids = (
            *self.resistance_component_ids,
            *self.capacitance_component_ids,
            self.source_resistance_component_id,
            self.termination_resistance_component_id,
        )
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("metrology component identities must be globally unique")
        for value_name, uncertainty_name, component_name in (
            (
                "resistance_ohms",
                "resistance_uncertainty_ohms",
                "resistance_component_ids",
            ),
            (
                "capacitance_farads",
                "capacitance_uncertainty_farads",
                "capacitance_component_ids",
            ),
        ):
            values = getattr(self, value_name)
            uncertainties = getattr(self, uncertainty_name)
            measured_component_ids = getattr(self, component_name)
            if len(values) != len(measured_component_ids) or len(uncertainties) != len(values):
                raise ValueError(f"{value_name} value/uncertainty/component rosters differ")
            for index, value in enumerate(values):
                validate_decimal(value, field_name=f"{value_name}[{index}]", minimum=Decimal(0))
                if value == 0:
                    raise ValueError(f"{value_name} must be positive")
                validate_decimal(
                    uncertainties[index],
                    field_name=f"{uncertainty_name}[{index}]",
                    minimum=Decimal(0),
                )
        for name in (
            "source_resistance_ohms",
            "source_resistance_uncertainty_ohms",
            "termination_resistance_ohms",
            "termination_resistance_uncertainty_ohms",
            "voltage_uncertainty_volts",
            "current_uncertainty_amperes",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.source_resistance_ohms == 0 or self.termination_resistance_ohms == 0:
            raise ValueError("source/termination resistances must be positive")
        if not self.measured_before_response:
            raise ValueError("evaluation numerical twin may use only pre-response metrology")


@dataclass(frozen=True, slots=True)
class RcLadderResponsePreparationRecord(CanonicalRecord):
    """Requested and measured initial state for one physical board/context."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-preparation-record'

    preparation_id: str
    board_id: str
    requested_equivalence_class_id: str
    requested_node_voltages_volts: tuple[Decimal, ...]
    realized_node_voltages_volts: tuple[Decimal, ...]
    voltage_uncertainty_volts: Decimal
    reset_evidence_ids: tuple[str, ...]
    equilibration_evidence_ids: tuple[str, ...]
    realization_complete_before_action_request: bool

    def __post_init__(self) -> None:
        for name in ("preparation_id", "board_id", "requested_equivalence_class_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not self.requested_node_voltages_volts or not self.realized_node_voltages_volts:
            raise ValueError("preparation profiles cannot be empty")
        if len(self.requested_node_voltages_volts) != len(self.realized_node_voltages_volts):
            raise ValueError("requested and realized preparation profiles differ in dimension")
        for name in ("requested_node_voltages_volts", "realized_node_voltages_volts"):
            for index, value in enumerate(getattr(self, name)):
                validate_decimal(value, field_name=f"{name}[{index}]")
        validate_decimal(
            self.voltage_uncertainty_volts,
            field_name="voltage_uncertainty_volts",
            minimum=Decimal(0),
        )
        require_sorted_unique_strings(
            self.reset_evidence_ids, field_name="reset_evidence_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.equilibration_evidence_ids,
            field_name="equilibration_evidence_ids",
            allow_empty=False,
        )
        if not self.realization_complete_before_action_request:
            raise ValueError("preparation must be measured before its action request")


@dataclass(frozen=True, slots=True)
class RcLadderResponseActionLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-action-ledger'

    ledger_id: str
    episode_id: str
    requested_action_id: str
    accepted_action_id: str | None
    applied_action_id: str | None
    realized_action_id: str | None
    delivery_status: RcLadderResponseActionDeliveryStatus
    request_time_seconds: Decimal
    acceptance_time_seconds: Decimal | None
    application_time_seconds: Decimal | None
    realization_time_seconds: Decimal | None
    clipping_observed: bool
    interlock_observed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        validate_stable_id(self.episode_id, field_name="episode_id")
        validate_stable_id(self.requested_action_id, field_name="requested_action_id")
        validate_decimal(
            self.request_time_seconds, field_name="request_time_seconds", minimum=Decimal(0)
        )
        stages = (
            (self.accepted_action_id, self.acceptance_time_seconds),
            (self.applied_action_id, self.application_time_seconds),
            (self.realized_action_id, self.realization_time_seconds),
        )
        previous = self.request_time_seconds
        stage_present = tuple(action_id is not None for action_id, _ in stages)
        if stage_present not in {
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        }:
            raise ValueError("action stages must form a requested-to-realized prefix")
        for index, (action_id, clock) in enumerate(stages):
            if (action_id is None) != (clock is None):
                raise ValueError("action stage identity and clock must be jointly present")
            if action_id is not None:
                validate_stable_id(action_id, field_name=f"stage_action_id[{index}]")
                assert clock is not None
                validate_decimal(clock, field_name=f"stage_clock[{index}]", minimum=previous)
                previous = clock
        if self.delivery_status is RcLadderResponseActionDeliveryStatus.REALIZED:
            if any(action_id is None for action_id, _ in stages):
                raise ValueError("realized delivery requires all action stages")
        elif self.realized_action_id is not None:
            raise ValueError("non-realized delivery cannot claim a realized action")


@dataclass(frozen=True, slots=True)
class RcLadderResponseEpisodeManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-episode-manifest'

    episode_id: str
    role: RcLadderResponseEpisodeRole
    board: ObjectIdentity
    metrology: ObjectIdentity
    preparation: ObjectIdentity
    complete_context_id: str
    matched_hold_episode_id: str
    is_hold: bool
    action_ledger: RcLadderResponseActionLedger
    native_clock_id: str
    clock_fit: ObjectIdentity
    channel_calibration: ObjectIdentity
    previous_episode_id: str | None
    causal_cutoff_seconds: Decimal
    receiver_window_seconds: tuple[Decimal, Decimal]
    sample_count: int
    channel_ids: tuple[str, ...]
    environment_channel_ids: tuple[str, ...]
    voltage_unit: str
    current_unit: str
    clock_unit: str
    gauge_id: str
    terminal_current_sign_id: str
    member_relative_path: str
    member_sha256: str
    member_size_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "complete_context_id",
            "matched_hold_episode_id",
            "native_clock_id",
            "gauge_id",
            "terminal_current_sign_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.previous_episode_id is not None:
            validate_stable_id(self.previous_episode_id, field_name="previous_episode_id")
        if self.board.object_schema != RcLadderResponseBoardIdentity.SCHEMA:
            raise ValueError("episode board identity has the wrong schema")
        if self.metrology.object_schema != RcLadderResponseComponentMetrology.SCHEMA:
            raise ValueError("episode metrology identity has the wrong schema")
        if self.preparation.object_schema != RcLadderResponsePreparationRecord.SCHEMA:
            raise ValueError("episode preparation identity has the wrong schema")
        if self.action_ledger.episode_id != self.episode_id:
            raise ValueError("episode and action-ledger identities differ")
        if self.clock_fit.object_schema != RcLadderResponseClockFitRecord.SCHEMA:
            raise ValueError("episode clock fit has the wrong schema")
        if self.channel_calibration.object_schema != RcLadderResponseChannelCalibration.SCHEMA:
            raise ValueError("episode channel calibration has the wrong schema")
        if self.is_hold != (self.episode_id == self.matched_hold_episode_id):
            raise ValueError("hold episode must be its context's matched-hold identity")
        validate_decimal(
            self.causal_cutoff_seconds,
            field_name="causal_cutoff_seconds",
            minimum=Decimal(0),
        )
        window_start, window_end = self.receiver_window_seconds
        validate_decimal(window_start, field_name="receiver_window_seconds[0]", minimum=Decimal(0))
        validate_decimal(window_end, field_name="receiver_window_seconds[1]", minimum=window_start)
        if window_end <= window_start or self.causal_cutoff_seconds > window_start:
            raise ValueError("causal cutoff must precede a positive receiver window")
        if self.action_ledger.realization_time_seconds is not None:
            if self.action_ledger.realization_time_seconds > self.causal_cutoff_seconds:
                raise ValueError("realized action must precede the causal cutoff")
        if self.sample_count < 2:
            raise ValueError("episode must contain at least two samples")
        require_sorted_unique_strings(self.channel_ids, field_name="channel_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.environment_channel_ids,
            field_name="environment_channel_ids",
        )
        if self.voltage_unit != "V" or self.current_unit != "A" or self.clock_unit != "s":
            raise ValueError("draft physical scale morphism profile requires exact native SI units")
        validate_relative_locator(self.member_relative_path)
        if not self.member_relative_path.endswith(".npz"):
            raise ValueError("draft physical scale morphism episode member must be safe NumPy NPZ")
        validate_sha256(self.member_sha256, field_name="member_sha256")
        if not 0 < self.member_size_bytes <= 128 * 1024 * 1024:
            raise ValueError("episode member size lies outside the bounded profile")


@dataclass(frozen=True, slots=True)
class RcLadderResponseBundleManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-bundle-manifest'

    manifest_id: str
    dossier: ObjectIdentity
    protocol: ObjectIdentity
    implementation_id: str
    board: RcLadderResponseBoardIdentity
    metrology: RcLadderResponseComponentMetrology
    clock_fits: tuple[RcLadderResponseClockFitRecord, ...]
    channel_calibrations: tuple[RcLadderResponseChannelCalibration, ...]
    preparations: tuple[RcLadderResponsePreparationRecord, ...]
    episodes: tuple[RcLadderResponseEpisodeManifest, ...]
    semantic_profile: ObjectIdentity
    laboratory_authority: ObjectIdentity
    custodian_id: str
    custody_seal: ObjectIdentity
    manifest_signature: ObjectIdentity
    publication_receipt: ObjectIdentity
    sealed: bool

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "implementation_id",
            "custodian_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.implementation_id != self.board.implementation_id:
            raise ValueError("bundle implementation and board differ")
        if self.metrology.board_id != self.board.board_id:
            raise ValueError("bundle metrology and board differ")
        if (
            self.metrology.resistance_component_ids != self.board.resistance_component_ids
            or self.metrology.capacitance_component_ids != self.board.capacitance_component_ids
            or self.metrology.source_resistance_component_id
            != self.board.source_resistance_component_id
            or self.metrology.termination_resistance_component_id
            != self.board.termination_resistance_component_id
        ):
            raise ValueError("bundle metrology component roster/order differs from the board")
        if len(self.metrology.capacitance_farads) != self.board.scale_cells:
            raise ValueError("bundle capacitance roster differs from board scale")
        if len(self.metrology.resistance_ohms) != self.board.scale_cells - 1:
            raise ValueError("bundle resistance roster differs from board topology")
        require_sorted_unique_ids(
            self.clock_fits, attribute="clock_fit_id", field_name="clock_fits"
        )
        require_sorted_unique_ids(
            self.channel_calibrations,
            attribute="calibration_id",
            field_name="channel_calibrations",
        )
        if not self.clock_fits or not self.channel_calibrations:
            raise ValueError("bundle must bind clock-fit and channel-calibration records")
        require_sorted_unique_ids(
            self.preparations,
            attribute="preparation_id",
            field_name="preparations",
        )
        if not self.preparations:
            raise ValueError("bundle must contain realized preparation records")
        if any(value.board_id != self.board.board_id for value in self.preparations):
            raise ValueError("bundle preparation belongs to another board")
        if any(
            len(value.realized_node_voltages_volts) != self.board.scale_cells
            for value in self.preparations
        ):
            raise ValueError("bundle preparation dimension differs from board scale")
        require_sorted_unique_ids(self.episodes, attribute="episode_id", field_name="episodes")
        if not self.episodes:
            raise ValueError("bundle must contain episodes")
        ledger_ids = tuple(value.action_ledger.ledger_id for value in self.episodes)
        if len(set(ledger_ids)) != len(ledger_ids):
            raise ValueError("bundle repeats an action-ledger identity")
        board_identity = ObjectIdentity.from_record(self.board.board_id, self.board)
        metrology_identity = ObjectIdentity.from_record(self.metrology.metrology_id, self.metrology)
        preparation_identities = {
            ObjectIdentity.from_record(value.preparation_id, value) for value in self.preparations
        }
        clock_fit_identities = {
            ObjectIdentity.from_record(value.clock_fit_id, value) for value in self.clock_fits
        }
        calibration_identities = {
            ObjectIdentity.from_record(value.calibration_id, value)
            for value in self.channel_calibrations
        }
        if any(
            value.board != board_identity
            or value.metrology != metrology_identity
            or value.preparation not in preparation_identities
            or value.clock_fit not in clock_fit_identities
            or value.channel_calibration not in calibration_identities
            for value in self.episodes
        ):
            raise ValueError(
                "episode linkage differs from bundle board/metrology/preparation/calibration"
            )
        clock_fit_by_identity = {
            ObjectIdentity.from_record(value.clock_fit_id, value): value
            for value in self.clock_fits
        }
        calibration_by_identity = {
            ObjectIdentity.from_record(value.calibration_id, value): value
            for value in self.channel_calibrations
        }
        for episode in self.episodes:
            if clock_fit_by_identity[episode.clock_fit].native_clock_id != episode.native_clock_id:
                raise ValueError("episode native clock differs from its exact clock fit")
            expected_calibration_channels = tuple(
                sorted((*episode.channel_ids, *episode.environment_channel_ids))
            )
            if calibration_by_identity[episode.channel_calibration].channel_ids != (
                expected_calibration_channels
            ):
                raise ValueError("episode observed channels differ from calibration order")
        paths = tuple(value.member_relative_path for value in self.episodes)
        if len(set(paths)) != len(paths):
            raise ValueError("bundle repeats an episode member path")
        by_context: dict[str, list[RcLadderResponseEpisodeManifest]] = {}
        for episode in self.episodes:
            by_context.setdefault(episode.complete_context_id, []).append(episode)
        for context_id, context_episodes in by_context.items():
            holds = [value for value in context_episodes if value.is_hold]
            if len(holds) != 1:
                raise ValueError(f"context {context_id} lacks exactly one matched hold")
            if any(
                value.matched_hold_episode_id != holds[0].episode_id for value in context_episodes
            ):
                raise ValueError("context episodes do not link the same realized hold")
            if holds[0].action_ledger.requested_action_id != "hold":
                raise ValueError("matched hold episode does not request the hold action")
        episode_ids = {value.episode_id for value in self.episodes}
        if any(
            value.previous_episode_id is not None and value.previous_episode_id not in episode_ids
            for value in self.episodes
        ):
            raise ValueError("episode previous linkage lies outside the sealed bundle")
        if any(value.previous_episode_id == value.episode_id for value in self.episodes):
            raise ValueError("episode cannot identify itself as its predecessor")
        predecessor_by_episode = {
            value.episode_id: value.previous_episode_id for value in self.episodes
        }
        for episode_id in predecessor_by_episode:
            visited: set[str] = set()
            current: str | None = episode_id
            while current is not None:
                if current in visited:
                    raise ValueError("episode predecessor linkage contains a cycle")
                visited.add(current)
                current = predecessor_by_episode[current]
        if not self.sealed:
            raise ValueError("read-only physical bundle must be immutable/sealed")


__all__ = [
    'RcLadderResponseActionDeliveryStatus',
    'RcLadderResponseActionLedger',
    'RcLadderResponseBoardIdentity',
    'RcLadderResponseBundleManifest',
    'RcLadderResponseChannelCalibration',
    'RcLadderResponseClockFitRecord',
    'RcLadderResponseComponentMetrology',
    'RcLadderResponseEpisodeManifest',
    'RcLadderResponseEpisodeRole',
    'RcLadderResponsePreparationRecord',
]
