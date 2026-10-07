"""Strict scientific records for the FreeGSNKE magnetic-response adapter.

The records in this module are deliberately path-free.  Exact external source
and runtime locators are installed by the trusted composition root; scientific
configuration binds only immutable content identities and static capability
keys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
import re
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal, QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


FREEGSNKE_ACTIVE_CIRCUIT_IDS: Final = (
    "d1",
    "d2",
    "d3",
    "d5",
    "d6",
    "d7",
    "dp",
    "p4",
    "p5",
    "p6",
    "px",
    "solenoid",
)
FREEGSNKE_PORT_IDS: Final = ("p4", "p5")
FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS: Final = (
    "passive-current-l1",
    "passive-current-max-abs",
    "passive-current-signed-sum",
)
FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA: Final = 'empirical-lawhood/simulators/freegsnke/free-gsnke-saved-preparation'
FREEGSNKE_PRIMARY_STATE_UNITS: Final = {
    "beta-p": "1",
    "ip-realized": "A",
    "kappa": "1",
    "li": "1",
}
FREEGSNKE_RECEIVER_UNITS: Final = {
    "axis-r": "m",
    "axis-z": "m",
    "elongation": "1",
    "gap-inner": "m",
    "gap-lower": "m",
    "gap-outer": "m",
    "gap-top": "m",
    "pickup-1": "T",
    "pickup-2": "T",
    "psi-loop-1": "Wb",
    "psi-loop-2": "Wb",
    "triangularity-lower": "1",
    "xpoint-r": "m",
    "xpoint-z": "m",
}
FREEGSNKE_RECEIVER_FAMILIES: Final = {
    "boundary": ("gap-inner", "gap-lower", "gap-outer", "gap-top"),
    "magnetic": ("pickup-1", "pickup-2", "psi-loop-1", "psi-loop-2"),
    "shape": (
        "axis-r",
        "axis-z",
        "elongation",
        "triangularity-lower",
        "xpoint-r",
        "xpoint-z",
    ),
}


class FreeGsnkePhase(StrEnum):
    MICROFIXTURE = "MICROFIXTURE"
    SCOUT = "SCOUT"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"
    ADMISSION_EVALUATION = "admission-evaluation"
    PROSPECTIVE_VALIDATION = "prospective-validation"


class FreeGsnkeBranchKind(StrEnum):
    COMPARATOR = "COMPARATOR"
    SINGLE_PORT = "SINGLE_PORT"
    HALF_DOSE = "HALF_DOSE"
    JOINT = "JOINT"


class FreeGsnkeTargetEpisodeStatus(StrEnum):
    OBSERVED_COMPLETE = "OBSERVED_COMPLETE"
    OBSERVED_INVALID = "OBSERVED_INVALID"
    WORKER_FAILED = "WORKER_FAILED"
    WORKER_TIMED_OUT = "WORKER_TIMED_OUT"


@dataclass(frozen=True, slots=True)
class FreeGsnkeSourceBinding(CanonicalRecord):
    """Exact source/runtime/machine identity, never an executable locator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-source-binding'

    binding_id: str
    freegsnke_version: str
    freegsnke_commit: str
    freegsnke_archive_sha256: str
    freegs4e_version: str
    freegs4e_commit: str
    freegs4e_archive_sha256: str
    runtime_tree_sha256: str
    runtime_python_version: str
    machine_id: str
    machine_members: tuple[ArtifactIdentity, ...]
    active_circuit_ids: tuple[str, ...]
    passive_structure_count: int
    source_network_required: bool
    private_uda_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.machine_id, field_name="machine_id")
        for name, value in (
            ("freegsnke_version", self.freegsnke_version),
            ("freegs4e_version", self.freegs4e_version),
            ("runtime_python_version", self.runtime_python_version),
        ):
            validate_nonempty(value, field_name=name)
        for name, value in (
            ("freegsnke_commit", self.freegsnke_commit),
            ("freegs4e_commit", self.freegs4e_commit),
        ):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{name} must be a lowercase Git object ID")
        for name, value in (
            ("freegsnke_archive_sha256", self.freegsnke_archive_sha256),
            ("freegs4e_archive_sha256", self.freegs4e_archive_sha256),
            ("runtime_tree_sha256", self.runtime_tree_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(
            self.machine_members,
            attribute="artifact_id",
            field_name="machine_members",
        )
        if len(self.machine_members) < 5:
            raise ValueError("FreeGSNKE binding requires active/passive/wall/limiter/probe bytes")
        require_sorted_unique_strings(
            self.active_circuit_ids,
            field_name="active_circuit_ids",
            allow_empty=False,
        )
        if self.active_circuit_ids != FREEGSNKE_ACTIVE_CIRCUIT_IDS:
            raise ValueError("FreeGSNKE active-circuit role set differs")
        if self.passive_structure_count != 138:
            raise ValueError("FreeGSNKE MAST-U-like passive-structure count differs")
        if self.source_network_required or self.private_uda_required:
            raise ValueError("selected FreeGSNKE denominator must be offline and public")


@dataclass(frozen=True, slots=True)
class FreeGsnkeNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-numerical-view'

    view_id: str
    r_min_m: Decimal
    r_max_m: Decimal
    z_min_m: Decimal
    z_max_m: Decimal
    radial_points: int
    vertical_points: int
    timestep_s: Decimal
    static_relative_tolerance: Decimal
    plasma_resistivity_ohm_m: Decimal
    maximum_mode_frequency_hz: Decimal
    linear_only: bool
    precision: str

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        for name, value in (
            ("r_min_m", self.r_min_m),
            ("r_max_m", self.r_max_m),
            ("z_min_m", self.z_min_m),
            ("z_max_m", self.z_max_m),
        ):
            validate_decimal(value, field_name=name)
        if self.r_min_m >= self.r_max_m or self.z_min_m >= self.z_max_m:
            raise ValueError("FreeGSNKE grid bounds must increase")
        if self.radial_points < 3 or self.vertical_points < 3:
            raise ValueError("FreeGSNKE grid is too small")
        if (self.radial_points - 1) & (self.radial_points - 2) or (self.vertical_points - 1) & (
            self.vertical_points - 2
        ):
            raise ValueError("FreeGSNKE grid dimensions must be 2**n + 1")
        for name, value in (
            ("timestep_s", self.timestep_s),
            ("static_relative_tolerance", self.static_relative_tolerance),
            ("plasma_resistivity_ohm_m", self.plasma_resistivity_ohm_m),
            ("maximum_mode_frequency_hz", self.maximum_mode_frequency_hz),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if not self.linear_only:
            raise ValueError("Measurement through admission requires the predeclared linear evolutive solver")
        if self.precision != "float64":
            raise ValueError("FreeGSNKE numerical view must retain float64")


@dataclass(frozen=True, slots=True)
class FreeGsnkePreparation(CanonicalRecord):
    """One independent numerical/plasma preparation; branches remain nested."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation'

    preparation_id: str
    phase: FreeGsnkePhase
    plasma_current_target_a: Decimal
    pressure_axis_pa: Decimal
    current_profile_alpha_m: Decimal
    current_profile_alpha_n: Decimal
    elongation_target: Decimal
    fvac_t_m: Decimal
    seed: int
    saved_state: ArtifactIdentity | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        if not isinstance(self.phase, FreeGsnkePhase):
            raise ValueError("unknown FreeGSNKE phase")
        for name, value, minimum in (
            ("plasma_current_target_a", self.plasma_current_target_a, Decimal("1")),
            ("pressure_axis_pa", self.pressure_axis_pa, Decimal("0")),
            ("current_profile_alpha_m", self.current_profile_alpha_m, Decimal("0.01")),
            ("current_profile_alpha_n", self.current_profile_alpha_n, Decimal("0.01")),
            ("elongation_target", self.elongation_target, Decimal("1")),
            ("fvac_t_m", self.fvac_t_m, Decimal("0")),
        ):
            validate_decimal(value, field_name=name, minimum=minimum)
        if self.seed < 0:
            raise ValueError("FreeGSNKE preparation seed must be nonnegative")
        if self.saved_state is not None:
            if self.saved_state.role != "freegsnke-saved-preparation":
                raise ValueError("FreeGSNKE saved-state role differs")
            if self.saved_state.payload_schema != FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA:
                raise ValueError("FreeGSNKE saved state requires its current canonical preparation identity; original archives require a separately verified export")
            if self.saved_state.media_type != "application/vnd.empirical-lawhood.canonical+json":
                raise ValueError("FreeGSNKE saved state must use non-executable JSON")


@dataclass(frozen=True, slots=True)
class FreeGsnkeActionPort(CanonicalRecord):
    """One fixed native port and its exact individual-coil voltage contraction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-port'

    port_id: str
    semantic_label: str
    native_unit: str
    coil_voltage_weights: tuple[NamedDecimal, ...]
    source_current_bound: QuantityBound
    baseline_current_a: Decimal
    full_increment_v: Decimal
    experiment_voltage_bound: QuantityBound | None = None

    def __post_init__(self) -> None:
        if self.port_id not in FREEGSNKE_PORT_IDS:
            raise ValueError("FreeGSNKE port is outside the frozen two-port chart")
        validate_nonempty(self.semantic_label, field_name="semantic_label")
        if self.native_unit != "V":
            raise ValueError("FreeGSNKE native action must remain circuit voltage in V")
        require_sorted_unique_ids(
            self.coil_voltage_weights,
            attribute="value_id",
            field_name="coil_voltage_weights",
        )
        if tuple(value.value_id for value in self.coil_voltage_weights) != FREEGSNKE_PORT_IDS:
            raise ValueError("FreeGSNKE port contraction must bind both selected coils")
        if any(value.unit != "1" for value in self.coil_voltage_weights):
            raise ValueError("coil-voltage weights must be dimensionless")
        expected = {value.value_id: value.value for value in self.coil_voltage_weights}
        if expected != {value: Decimal(value == self.port_id) for value in FREEGSNKE_PORT_IDS}:
            raise ValueError("direct fallback port must be an exact one-hot contraction")
        if self.source_current_bound.quantity_id != f"{self.port_id}-current":
            raise ValueError("source current bound is attached to the wrong port")
        if self.source_current_bound.native_unit != "A":
            raise ValueError("source current bound must remain in A")
        if self.source_current_bound.lower is None or self.source_current_bound.upper is None:
            raise ValueError("selected direct PF group requires a two-sided current bound")
        validate_decimal(self.baseline_current_a, field_name="baseline_current_a")
        if not (
            self.source_current_bound.lower
            < self.baseline_current_a
            < self.source_current_bound.upper
        ):
            raise ValueError("baseline current lacks symmetric source-limit headroom")
        validate_decimal(self.full_increment_v, field_name="full_increment_v", minimum=Decimal(0))
        if self.full_increment_v == 0:
            raise ValueError("FreeGSNKE action increment must be positive")
        # Current execution requires the experiment voltage bound explicitly.
        if self.experiment_voltage_bound is not None:
            bound = self.experiment_voltage_bound
            if bound.quantity_id != f"{self.port_id}-voltage-increment":
                raise ValueError("experiment voltage bound is attached to the wrong port")
            if bound.native_unit != "V":
                raise ValueError("experiment voltage bound must remain in V")
            if bound.lower != -self.full_increment_v or bound.upper != self.full_increment_v:
                raise ValueError("experiment voltage bound must equal the exact symmetric dose")


@dataclass(frozen=True, slots=True)
class FreeGsnkeActionChart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-chart'

    chart_id: str
    selection_rule_id: str
    geometry_coordinate_ids: tuple[str, ...]
    geometry_condition_number: Decimal
    ports: tuple[FreeGsnkeActionPort, ...]
    circuit_timescale_s: Decimal
    pulse_duration_s: Decimal
    primary_horizons_s: tuple[Decimal, ...]
    diagnostic_horizons_s: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        validate_stable_id(self.selection_rule_id, field_name="selection_rule_id")
        require_sorted_unique_strings(
            self.geometry_coordinate_ids,
            field_name="geometry_coordinate_ids",
            allow_empty=False,
        )
        if self.geometry_coordinate_ids != ("bz-at-axis", "dbz-dr-at-axis"):
            raise ValueError("FreeGSNKE fallback geometry coordinates differ")
        validate_decimal(
            self.geometry_condition_number,
            field_name="geometry_condition_number",
            minimum=Decimal(1),
        )
        if tuple(value.port_id for value in self.ports) != FREEGSNKE_PORT_IDS:
            raise ValueError("FreeGSNKE action chart must contain sorted admission and controller-use ports")
        for name, value in (
            ("circuit_timescale_s", self.circuit_timescale_s),
            ("pulse_duration_s", self.pulse_duration_s),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.pulse_duration_s != self.circuit_timescale_s / Decimal(4):
            raise ValueError("pulse duration must be exactly 0.25 tau_c")
        if tuple(sorted(set(self.primary_horizons_s))) != self.primary_horizons_s:
            raise ValueError("primary horizons must be sorted and unique")
        if tuple(sorted(set(self.diagnostic_horizons_s))) != self.diagnostic_horizons_s:
            raise ValueError("diagnostic horizons must be sorted and unique")
        if len(self.primary_horizons_s) != 2 or len(self.diagnostic_horizons_s) != 2:
            raise ValueError("FreeGSNKE chart requires two primary and two diagnostic horizons")


@dataclass(frozen=True, slots=True)
class FreeGsnkeVoltageSample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-voltage-sample'

    sample_id: str
    clock_s: Decimal
    coil_voltage_increments: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        validate_decimal(self.clock_s, field_name="clock_s", minimum=Decimal(0))
        require_sorted_unique_ids(
            self.coil_voltage_increments,
            attribute="value_id",
            field_name="coil_voltage_increments",
        )
        if tuple(value.value_id for value in self.coil_voltage_increments) != FREEGSNKE_PORT_IDS:
            raise ValueError("applied sample must contain exact admission and controller-use coil increments")
        if any(value.unit != "V" for value in self.coil_voltage_increments):
            raise ValueError("applied voltage sample units differ")


@dataclass(frozen=True, slots=True)
class FreeGsnkeCurrentSnapshot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-current-snapshot'

    snapshot_id: str
    receiver_clock_s: Decimal
    active_currents: tuple[NamedDecimal, ...]
    passive_currents: tuple[NamedDecimal, ...]
    plasma_current: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_decimal(self.receiver_clock_s, field_name="receiver_clock_s", minimum=Decimal(0))
        require_sorted_unique_ids(
            self.active_currents,
            attribute="value_id",
            field_name="active_currents",
        )
        if tuple(value.value_id for value in self.active_currents) != FREEGSNKE_ACTIVE_CIRCUIT_IDS:
            raise ValueError("realized active-current role set differs")
        require_sorted_unique_ids(
            self.passive_currents,
            attribute="value_id",
            field_name="passive_currents",
        )
        if len(self.passive_currents) != 138:
            raise ValueError("realized passive-current role set differs")
        if any(value.unit != "A" for value in (*self.active_currents, *self.passive_currents)):
            raise ValueError("realized metal currents must remain in A")
        if self.plasma_current.value_id != "ip-realized" or self.plasma_current.unit != "A":
            raise ValueError("realized plasma current identity/unit differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeActionLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-ledger'

    ledger_id: str
    branch_kind: FreeGsnkeBranchKind
    request_clock_s: Decimal
    acceptance_clock_s: Decimal
    requested_port_increments: tuple[NamedDecimal, ...]
    accepted_coil_increments: tuple[NamedDecimal, ...]
    applied_samples: tuple[FreeGsnkeVoltageSample, ...]
    clipping_flags: tuple[str, ...]
    realized_currents: tuple[FreeGsnkeCurrentSnapshot, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if not isinstance(self.branch_kind, FreeGsnkeBranchKind):
            raise ValueError("unknown FreeGSNKE branch kind")
        for name, value in (
            ("request_clock_s", self.request_clock_s),
            ("acceptance_clock_s", self.acceptance_clock_s),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.acceptance_clock_s < self.request_clock_s:
            raise ValueError("FreeGSNKE acceptance precedes request")
        for name, values in (
            ("requested_port_increments", self.requested_port_increments),
            ("accepted_coil_increments", self.accepted_coil_increments),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if tuple(value.value_id for value in values) != FREEGSNKE_PORT_IDS:
                raise ValueError(f'{name} must bind exact admission and controller-use roles')
            if any(value.unit != "V" for value in values):
                raise ValueError(f"{name} must remain in V")
        if self.requested_port_increments != self.accepted_coil_increments:
            raise ValueError("direct fallback acceptance must equal its one-hot request")
        if not self.applied_samples:
            raise ValueError("FreeGSNKE action ledger requires applied samples")
        clocks = tuple(value.clock_s for value in self.applied_samples)
        if tuple(sorted(set(clocks))) != clocks or clocks[0] < self.acceptance_clock_s:
            raise ValueError("FreeGSNKE applied-sample clocks differ")
        requested = {value.value_id: value.value for value in self.requested_port_increments}
        for sample in self.applied_samples:
            if {
                value.value_id: value.value for value in sample.coil_voltage_increments
            } != requested:
                raise ValueError("applied voltage differs from accepted increment")
        require_sorted_unique_strings(self.clipping_flags, field_name="clipping_flags")
        if self.clipping_flags:
            raise ValueError("claim-bearing FreeGSNKE action may not be materially clipped")
        if not self.realized_currents:
            raise ValueError("FreeGSNKE ledger requires realized current snapshots")
        realized_clocks = tuple(value.receiver_clock_s for value in self.realized_currents)
        if tuple(sorted(set(realized_clocks))) != realized_clocks:
            raise ValueError("realized-current clocks must be sorted and unique")
        if realized_clocks[0] <= self.request_clock_s:
            raise ValueError("realized-current receiver clock must follow request")
        is_zero = all(value == 0 for value in requested.values())
        if self.branch_kind is FreeGsnkeBranchKind.COMPARATOR and not is_zero:
            raise ValueError("comparator must request exact zero increments")
        if self.branch_kind is not FreeGsnkeBranchKind.COMPARATOR and is_zero:
            raise ValueError("non-comparator branch must request a nonzero increment")


@dataclass(frozen=True, slots=True)
class FreeGsnkeCausalState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-causal-state'

    state_id: str
    cutoff_clock_s: Decimal
    primary_values: tuple[NamedDecimal, ...]
    profile_prefix_values: tuple[NamedDecimal, ...]
    boundary_sha256: str
    active_current_sha256: str
    passive_current_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        validate_decimal(self.cutoff_clock_s, field_name="cutoff_clock_s", minimum=Decimal(0))
        require_sorted_unique_ids(
            self.primary_values,
            attribute="value_id",
            field_name="primary_values",
        )
        if {
            value.value_id: value.unit for value in self.primary_values
        } != FREEGSNKE_PRIMARY_STATE_UNITS:
            raise ValueError("FreeGSNKE primary causal state identities/units differ")
        require_sorted_unique_ids(
            self.profile_prefix_values,
            attribute="value_id",
            field_name="profile_prefix_values",
        )
        if not self.profile_prefix_values:
            raise ValueError("FreeGSNKE causal profile prefix must be retained")
        for name, value in (
            ("boundary_sha256", self.boundary_sha256),
            ("active_current_sha256", self.active_current_sha256),
            ("passive_current_sha256", self.passive_current_sha256),
        ):
            validate_sha256(value, field_name=name)


@dataclass(frozen=True, slots=True)
class FreeGsnkeReceiverSnapshot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-receiver-snapshot'

    snapshot_id: str
    receiver_clock_s: Decimal
    values: tuple[NamedDecimal, ...]
    topology_id: str
    valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_stable_id(self.topology_id, field_name="topology_id")
        validate_decimal(self.receiver_clock_s, field_name="receiver_clock_s", minimum=Decimal(0))
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")
        if {value.value_id: value.unit for value in self.values} != FREEGSNKE_RECEIVER_UNITS:
            raise ValueError("FreeGSNKE receiver identities/units differ")
        if self.topology_id != "single-null-diverted":
            raise ValueError("FreeGSNKE claim denominator lost its single-null topology")


@dataclass(frozen=True, slots=True)
class FreeGsnkeEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-episode'

    episode_id: str
    source_binding: ObjectIdentity
    preparation: ObjectIdentity
    numerical_view: ObjectIdentity
    action_chart: ObjectIdentity
    physical_preparation_id: str
    phase: FreeGsnkePhase
    causal_state: FreeGsnkeCausalState
    action_ledger: FreeGsnkeActionLedger
    receiver_snapshots: tuple[FreeGsnkeReceiverSnapshot, ...]
    valid: bool
    error_code: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        validate_stable_id(self.physical_preparation_id, field_name="physical_preparation_id")
        for value, schema, label in (
            (self.source_binding, FreeGsnkeSourceBinding.SCHEMA, "source binding"),
            (self.preparation, FreeGsnkePreparation.SCHEMA, "preparation"),
            (self.numerical_view, FreeGsnkeNumericalView.SCHEMA, "numerical view"),
            (self.action_chart, FreeGsnkeActionChart.SCHEMA, "action chart"),
        ):
            if not isinstance(value, ObjectIdentity) or value.object_schema != schema:
                raise ValueError(f"FreeGSNKE episode {label} identity differs")
        if not isinstance(self.phase, FreeGsnkePhase):
            raise ValueError("unknown FreeGSNKE episode phase")
        if self.causal_state.cutoff_clock_s > self.action_ledger.request_clock_s:
            raise ValueError("FreeGSNKE causal state crosses the request cutoff")
        if not self.receiver_snapshots:
            raise ValueError("FreeGSNKE episode requires receiver snapshots")
        clocks = tuple(value.receiver_clock_s for value in self.receiver_snapshots)
        if tuple(sorted(set(clocks))) != clocks or clocks[0] <= self.action_ledger.request_clock_s:
            raise ValueError("FreeGSNKE receiver clocks differ or cross the request")
        ledger_clocks = tuple(
            value.receiver_clock_s for value in self.action_ledger.realized_currents
        )
        if clocks != ledger_clocks:
            raise ValueError("receiver and realized-current clocks must be identical")
        if self.valid:
            if self.error_code is not None or not all(
                value.valid for value in self.receiver_snapshots
            ):
                raise ValueError("valid FreeGSNKE episode contains an error/invalid receiver")
        else:
            if self.error_code is None:
                raise ValueError("invalid FreeGSNKE episode requires a typed error code")
            validate_stable_id(self.error_code, field_name="error_code")


@dataclass(frozen=True, slots=True)
class FreeGsnkeProcessRequest(CanonicalRecord):
    """Bounded, path-free request passed to the fixed isolated worker."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-process-request'

    request_id: str
    source_binding: FreeGsnkeSourceBinding
    preparation: FreeGsnkePreparation
    numerical_view: FreeGsnkeNumericalView
    action_chart: FreeGsnkeActionChart
    phase: FreeGsnkePhase
    branch_id: str
    branch_kind: FreeGsnkeBranchKind
    requested_port_increments: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        if self.phase is not self.preparation.phase:
            raise ValueError("FreeGSNKE request phase differs from preparation")
        require_sorted_unique_ids(
            self.requested_port_increments,
            attribute="value_id",
            field_name="requested_port_increments",
        )
        if tuple(value.value_id for value in self.requested_port_increments) != FREEGSNKE_PORT_IDS:
            raise ValueError("FreeGSNKE process request must bind exact admission and controller-use ports")
        if any(value.unit != "V" for value in self.requested_port_increments):
            raise ValueError("FreeGSNKE process request unit differs")
        values = {value.value_id: value.value for value in self.requested_port_increments}
        if self.branch_kind is FreeGsnkeBranchKind.COMPARATOR:
            if any(values.values()):
                raise ValueError("FreeGSNKE comparator request must be exact zero")
        elif sum(value != 0 for value in values.values()) < 1:
            raise ValueError("FreeGSNKE non-comparator request must be nonzero")
        for port in self.action_chart.ports:
            bound = port.experiment_voltage_bound
            if bound is not None and (
                bound.lower is None
                or bound.upper is None
                or values[port.port_id] < bound.lower
                or values[port.port_id] > bound.upper
            ):
                raise ValueError("FreeGSNKE request exceeds the experiment voltage bound")
            if abs(values[port.port_id]) > port.full_increment_v:
                raise ValueError("FreeGSNKE request exceeds the frozen full dose")


@dataclass(frozen=True, slots=True)
class FreeGsnkeProcessResponse(CanonicalRecord):
    """Closed response envelope: one request identity and one episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-process-response'

    response_id: str
    request: ObjectIdentity
    episode: FreeGsnkeEpisode

    def __post_init__(self) -> None:
        validate_stable_id(self.response_id, field_name="response_id")
        if self.request.object_schema != FreeGsnkeProcessRequest.SCHEMA:
            raise ValueError("FreeGSNKE process response request identity differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetCurrentSnapshot(CanonicalRecord):
    """Outcome-capable current view; missing native roles remain observable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-current-snapshot'

    snapshot_id: str
    receiver_clock_s: Decimal
    active_currents: tuple[NamedDecimal, ...]
    passive_currents: tuple[NamedDecimal, ...]
    plasma_current: NamedDecimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_decimal(
            self.receiver_clock_s,
            field_name="receiver_clock_s",
            minimum=Decimal(0),
        )
        require_sorted_unique_ids(
            self.active_currents,
            attribute="value_id",
            field_name="active_currents",
        )
        if any(
            value.value_id not in FREEGSNKE_ACTIVE_CIRCUIT_IDS for value in self.active_currents
        ):
            raise ValueError("target active-current role is unknown")
        require_sorted_unique_ids(
            self.passive_currents,
            attribute="value_id",
            field_name="passive_currents",
        )
        if any(value.unit != "A" for value in (*self.active_currents, *self.passive_currents)):
            raise ValueError("target metal currents must remain in A")
        if self.plasma_current is not None and (
            self.plasma_current.value_id != "ip-realized" or self.plasma_current.unit != "A"
        ):
            raise ValueError("target plasma-current identity/unit differs")

    @property
    def complete(self) -> bool:
        return (
            tuple(value.value_id for value in self.active_currents) == FREEGSNKE_ACTIVE_CIRCUIT_IDS
            and len(self.passive_currents) == 138
            and self.plasma_current is not None
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetReceiverSnapshot(CanonicalRecord):
    """Receiver view that preserves invalid topology and missing-role outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-receiver-snapshot'

    snapshot_id: str
    receiver_clock_s: Decimal
    values: tuple[NamedDecimal, ...]
    topology_id: str
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_decimal(
            self.receiver_clock_s,
            field_name="receiver_clock_s",
            minimum=Decimal(0),
        )
        require_sorted_unique_ids(
            self.values,
            attribute="value_id",
            field_name="values",
        )
        if any(FREEGSNKE_RECEIVER_UNITS.get(value.value_id) != value.unit for value in self.values):
            raise ValueError("target receiver identity/unit is unknown")
        validate_stable_id(self.topology_id, field_name="topology_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid and self.reason_codes:
            raise ValueError("valid target receiver cannot carry failure reasons")

    @property
    def complete(self) -> bool:
        return {value.value_id: value.unit for value in self.values} == FREEGSNKE_RECEIVER_UNITS


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetActionObservation(CanonicalRecord):
    """Outcome-capable action fibre; mismatches remain evidence, not decode errors."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-action-observation'

    observation_id: str
    request_clock_s: Decimal
    acceptance_clock_s: Decimal | None
    requested_port_increments: tuple[NamedDecimal, ...]
    accepted_port_increments: tuple[NamedDecimal, ...]
    applied_samples: tuple[FreeGsnkeVoltageSample, ...]
    clipping_flags: tuple[str, ...]
    realized_currents: tuple[FreeGsnkeTargetCurrentSnapshot, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_decimal(self.request_clock_s, field_name="request_clock_s", minimum=Decimal(0))
        if self.acceptance_clock_s is not None:
            validate_decimal(
                self.acceptance_clock_s,
                field_name="acceptance_clock_s",
                minimum=Decimal(0),
            )
        for name, values in (
            ("requested_port_increments", self.requested_port_increments),
            ("accepted_port_increments", self.accepted_port_increments),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if values and tuple(value.value_id for value in values) != FREEGSNKE_PORT_IDS:
                raise ValueError(f'{name} must bind exact admission and controller-use roles when observed')
            if any(value.unit != "V" for value in values):
                raise ValueError(f"{name} must remain in V")
        if tuple(value.value_id for value in self.requested_port_increments) != (
            FREEGSNKE_PORT_IDS
        ):
            raise ValueError("target action observation requires the exact request roles")
        applied_clocks = tuple(value.clock_s for value in self.applied_samples)
        if tuple(sorted(set(applied_clocks))) != applied_clocks:
            raise ValueError("target action applied clocks must be sorted and unique")
        require_sorted_unique_strings(self.clipping_flags, field_name="clipping_flags")
        realized_clocks = tuple(value.receiver_clock_s for value in self.realized_currents)
        if tuple(sorted(set(realized_clocks))) != realized_clocks:
            raise ValueError("target realized-current clocks must be sorted and unique")


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetEpisode(CanonicalRecord):
    """One issued branch outcome, including unfavorable or incomplete response."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-episode'

    episode_id: str
    source_binding: ObjectIdentity
    preparation: ObjectIdentity
    numerical_view: ObjectIdentity
    action_chart: ObjectIdentity
    physical_preparation_id: str
    phase: FreeGsnkePhase
    causal_state: FreeGsnkeCausalState | None
    action_observation: FreeGsnkeTargetActionObservation
    receiver_snapshots: tuple[FreeGsnkeTargetReceiverSnapshot, ...]
    status: FreeGsnkeTargetEpisodeStatus
    error_code: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        validate_stable_id(
            self.physical_preparation_id,
            field_name="physical_preparation_id",
        )
        for value, schema, label in (
            (self.source_binding, FreeGsnkeSourceBinding.SCHEMA, "source binding"),
            (self.preparation, FreeGsnkePreparation.SCHEMA, "preparation"),
            (self.numerical_view, FreeGsnkeNumericalView.SCHEMA, "numerical view"),
            (self.action_chart, FreeGsnkeActionChart.SCHEMA, "action chart"),
        ):
            if value.object_schema != schema:
                raise ValueError(f"FreeGSNKE target episode {label} identity differs")
        receiver_clocks = tuple(value.receiver_clock_s for value in self.receiver_snapshots)
        if tuple(sorted(set(receiver_clocks))) != receiver_clocks:
            raise ValueError("FreeGSNKE target receiver clocks must be sorted and unique")
        if self.error_code is not None:
            validate_stable_id(self.error_code, field_name="error_code")
        if not isinstance(self.status, FreeGsnkeTargetEpisodeStatus):
            raise ValueError("unknown FreeGSNKE target episode status")
        if self.status is FreeGsnkeTargetEpisodeStatus.OBSERVED_COMPLETE:
            if (
                self.error_code is not None
                or self.causal_state is None
                or not self.receiver_snapshots
                or not self.action_observation.accepted_port_increments
                or self.action_observation.acceptance_clock_s is None
                or not self.action_observation.applied_samples
                or not self.action_observation.realized_currents
                or not all(
                    value.valid and value.complete and value.topology_id == "single-null-diverted"
                    for value in self.receiver_snapshots
                )
                or not all(value.complete for value in self.action_observation.realized_currents)
            ):
                raise ValueError("complete target episode lacks its observed fibre")
        elif self.error_code is None:
            raise ValueError("unfavorable target episode requires a typed error code")


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetProcessResponse(CanonicalRecord):
    """Request-bound target response that can carry a scientific adverse outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-process-response'

    response_id: str
    request: ObjectIdentity
    episode: FreeGsnkeTargetEpisode

    def __post_init__(self) -> None:
        validate_stable_id(self.response_id, field_name="response_id")
        if self.request.object_schema != FreeGsnkeProcessRequest.SCHEMA:
            raise ValueError("FreeGSNKE target response request identity differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeSavedPreparation(CanonicalRecord):
    """One excluded-qualification saved state used by every nested branch.

    The record deliberately binds the preparation recipe *without* its saved
    state reference.  This avoids a content-hash cycle while still proving
    that the issued request restores the exact qualified preparation bytes.
    """

    SCHEMA: ClassVar[str] = FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA

    saved_state_id: str
    source_binding: ObjectIdentity
    preparation_id: str
    preparation_recipe_sha256: str
    phase: FreeGsnkePhase
    numerical_view: ObjectIdentity
    action_chart: ObjectIdentity
    initial_causal_state: FreeGsnkeCausalState
    initial_currents: FreeGsnkeTargetCurrentSnapshot
    initial_receivers: FreeGsnkeTargetReceiverSnapshot
    static_solver_converged: bool
    source_current_limits_passed: bool
    wall_and_limiter_clear: bool
    lower_single_null_valid: bool
    elongation_target_passed: bool
    native_action_chart_passed: bool
    eligible: bool
    reason_codes: tuple[str, ...]
    excluded_qualification_unit_count: int
    target_response_count: int
    target_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.saved_state_id, field_name="saved_state_id")
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        validate_sha256(
            self.preparation_recipe_sha256,
            field_name="preparation_recipe_sha256",
        )
        if self.source_binding.object_schema != FreeGsnkeSourceBinding.SCHEMA:
            raise ValueError("saved preparation source binding differs")
        if self.numerical_view.object_schema != FreeGsnkeNumericalView.SCHEMA:
            raise ValueError("saved preparation numerical view differs")
        if self.action_chart.object_schema != FreeGsnkeActionChart.SCHEMA:
            raise ValueError("saved preparation action chart differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        qualified = all(
            (
                self.static_solver_converged,
                self.source_current_limits_passed,
                self.wall_and_limiter_clear,
                self.lower_single_null_valid,
                self.elongation_target_passed,
                self.native_action_chart_passed,
                self.initial_currents.complete,
                self.initial_receivers.complete,
                self.initial_receivers.valid,
                self.initial_receivers.topology_id == "single-null-diverted",
            )
        )
        if self.eligible != qualified or self.eligible == bool(self.reason_codes):
            raise ValueError("saved preparation eligibility is not fact-derived")
        if self.initial_causal_state.cutoff_clock_s != 0:
            raise ValueError("saved preparation causal state must precede action")
        if (
            self.initial_currents.receiver_clock_s != 0
            or self.initial_receivers.receiver_clock_s != 0
        ):
            raise ValueError("saved preparation observations must use the preparation clock")
        if self.excluded_qualification_unit_count != 1:
            raise ValueError("saved preparation must count one excluded qualification unit")
        if self.target_response_count or self.target_outcome_access_count:
            raise ValueError("saved preparation crossed target outcome access")


def freegsnke_saved_preparation_artifact(
    saved: FreeGsnkeSavedPreparation,
) -> ArtifactIdentity:
    """Return the exact non-executable artifact identity embedded in a request."""

    return ArtifactIdentity(
        artifact_id=saved.saved_state_id,
        role="freegsnke-saved-preparation",
        payload_schema=saved.SCHEMA,
        sha256=saved.fingerprint(),
        media_type="application/vnd.empirical-lawhood.canonical+json",
        size_bytes=len(saved.canonical_bytes()),
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetWorkerInput(CanonicalRecord):
    """Exact process input joining one issued branch to its saved preparation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-worker-input'

    input_id: str
    request: FreeGsnkeProcessRequest
    saved_preparation: FreeGsnkeSavedPreparation

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        request = self.request
        saved = self.saved_preparation
        recipe = replace(request.preparation, saved_state=None)
        if (
            request.preparation.saved_state != freegsnke_saved_preparation_artifact(saved)
            or saved.preparation_id != request.preparation.preparation_id
            or saved.preparation_recipe_sha256 != recipe.fingerprint()
            or saved.phase is not request.phase
            or saved.source_binding
            != ObjectIdentity.from_record(
                request.source_binding.binding_id,
                request.source_binding,
            )
            or saved.numerical_view
            != ObjectIdentity.from_record(
                request.numerical_view.view_id,
                request.numerical_view,
            )
            or saved.action_chart
            != ObjectIdentity.from_record(
                request.action_chart.chart_id,
                request.action_chart,
            )
            or not saved.eligible
        ):
            raise ValueError("FreeGSNKE worker input changed its saved preparation")


__all__ = [
    "FREEGSNKE_ACTIVE_CIRCUIT_IDS",
    "FREEGSNKE_PORT_IDS",
    "FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS",
    "FREEGSNKE_PRIMARY_STATE_UNITS",
    "FREEGSNKE_RECEIVER_FAMILIES",
    "FREEGSNKE_RECEIVER_UNITS",
    "FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA",
    "FreeGsnkeActionChart",
    "FreeGsnkeActionLedger",
    "FreeGsnkeActionPort",
    "FreeGsnkeBranchKind",
    "FreeGsnkeCausalState",
    "FreeGsnkeCurrentSnapshot",
    "FreeGsnkeEpisode",
    "FreeGsnkeNumericalView",
    "FreeGsnkePhase",
    "FreeGsnkePreparation",
    "FreeGsnkeProcessRequest",
    "FreeGsnkeProcessResponse",
    "FreeGsnkeReceiverSnapshot",
    'FreeGsnkeSavedPreparation',
    "FreeGsnkeSourceBinding",
    'FreeGsnkeTargetActionObservation',
    'FreeGsnkeTargetCurrentSnapshot',
    "FreeGsnkeTargetEpisodeStatus",
    'FreeGsnkeTargetEpisode',
    'FreeGsnkeTargetProcessResponse',
    'FreeGsnkeTargetReceiverSnapshot',
    'FreeGsnkeTargetWorkerInput',
    "FreeGsnkeVoltageSample",
    "freegsnke_saved_preparation_artifact",
]
