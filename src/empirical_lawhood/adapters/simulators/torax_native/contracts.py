"""Substrate-general native direct-TORAX preparation and output records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .._native_admission import (
    NativeTimeGridAdmission,
    TORAX_ESTIMATED_BYTES_PER_RADIAL_CELL,
    TORAX_ESTIMATED_BYTES_PER_RETAINED_SAMPLE,
    TORAX_MAX_ESTIMATED_HISTORY_BYTES,
    TORAX_MAX_HISTORY_CELLS,
    admit_native_time_grid,
)


class NativeToraxEpisodeDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    SOLVER_FAILURE = "SOLVER_FAILURE"


@dataclass(frozen=True, slots=True)
class NativeToraxView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-view'

    view_id: str
    radial_cells: int
    timestep_s: Decimal
    horizon_s: Decimal
    solver_id: str
    linear_solver_id: str
    precision: str
    closure_id: str

    def __post_init__(self) -> None:
        for name in ("view_id", "solver_id", "linear_solver_id", "closure_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("timestep_s", "horizon_s"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if type(self.radial_cells) is not int or self.radial_cells < 4:
            raise ValueError("native TORAX view requires at least four radial cells")
        if self.solver_id != "linear-theta-fully-implicit":
            raise ValueError("native TORAX requires the fully implicit linear solver")
        if self.linear_solver_id != "thomas" or self.precision != "float64":
            raise ValueError("native TORAX requires Thomas float64 execution")
        self.operational_admission()

    def operational_admission(self) -> NativeTimeGridAdmission:
        admission = admit_native_time_grid(
            timestep=self.timestep_s,
            intervals=(self.horizon_s,),
            label="TORAX",
            require_integral_intervals=False,
            sample_offset=1,
            estimated_bytes_per_sample=(
                self.radial_cells * TORAX_ESTIMATED_BYTES_PER_RADIAL_CELL
                + TORAX_ESTIMATED_BYTES_PER_RETAINED_SAMPLE
            ),
            maximum_estimated_bytes=TORAX_MAX_ESTIMATED_HISTORY_BYTES,
        )
        if admission.sample_count * self.radial_cells > TORAX_MAX_HISTORY_CELLS:
            raise ValueError(f"TORAX native history exceeds {TORAX_MAX_HISTORY_CELLS} radial cells")
        return admission


@dataclass(frozen=True, slots=True)
class NativeToraxPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-preparation'

    preparation_id: str
    radial_coordinates: tuple[Decimal, ...]
    electron_temperature_ev: tuple[Decimal, ...]
    ion_temperature_ev: tuple[Decimal, ...]
    electron_density_m3: tuple[Decimal, ...]
    plasma_current_a: Decimal
    main_ion: str
    impurity: str
    zeff: Decimal
    major_radius_m: Decimal
    minor_radius_m: Decimal
    toroidal_field_t: Decimal
    elongation_lcfs: Decimal
    source_radial_location: Decimal
    source_width: Decimal
    electron_heat_fraction: Decimal
    absorbed_power_fraction: Decimal
    chi_i_m2_s: Decimal
    chi_e_m2_s: Decimal
    particle_diffusivity_m2_s: Decimal
    particle_convection_m_s: Decimal
    maximum_core_temperature_ev: Decimal
    assumption_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        size = len(self.radial_coordinates)
        if size < 3 or any(
            len(values) != size
            for values in (
                self.electron_temperature_ev,
                self.ion_temperature_ev,
                self.electron_density_m3,
            )
        ):
            raise ValueError("native TORAX profiles require one common radial grid")
        if tuple(sorted(self.radial_coordinates)) != self.radial_coordinates:
            raise ValueError("native TORAX radial coordinates must be ordered")
        if self.radial_coordinates[0] != 0 or self.radial_coordinates[-1] != 1:
            raise ValueError("native TORAX radial coordinates must cover normalized [0,1]")
        for value in (
            *self.radial_coordinates,
            *self.electron_temperature_ev,
            *self.ion_temperature_ev,
            *self.electron_density_m3,
        ):
            validate_decimal(value, field_name="profile_value", minimum=Decimal(0))
        if any(value <= 0 for value in self.electron_temperature_ev):
            raise ValueError("electron-temperature profile must be positive")
        if any(value <= 0 for value in self.ion_temperature_ev):
            raise ValueError("ion-temperature profile must be positive")
        if any(value <= 0 for value in self.electron_density_m3):
            raise ValueError("electron-density profile must be positive")
        for name in (
            "plasma_current_a",
            "zeff",
            "major_radius_m",
            "minor_radius_m",
            "toroidal_field_t",
            "elongation_lcfs",
            "source_width",
            "chi_i_m2_s",
            "chi_e_m2_s",
            "particle_diffusivity_m2_s",
            "maximum_core_temperature_ev",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "source_radial_location",
            "electron_heat_fraction",
            "absorbed_power_fraction",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError(f"{name} must not exceed one")
        validate_decimal(
            self.particle_convection_m_s,
            field_name="particle_convection_m_s",
        )
        if self.minor_radius_m > self.major_radius_m:
            raise ValueError("minor radius exceeds major radius")
        if self.main_ion not in {"deuterium", "tritium"}:
            raise ValueError("native TORAX main-ion symbol is unsupported")
        if self.impurity not in {"argon", "carbon", "neon"}:
            raise ValueError("native TORAX impurity symbol is unsupported")
        require_sorted_unique_strings(
            self.assumption_ids,
            field_name="assumption_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class NativeToraxActionStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-action-stage'

    stage: ActionDeliveryStage
    power_w: Decimal
    coordinate_s: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.power_w, field_name="power_w", minimum=Decimal(0))
        validate_decimal(self.coordinate_s, field_name="coordinate_s", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class NativeToraxAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-action'

    action_id: str
    action_label: str
    stages: tuple[NativeToraxActionStage, ...]
    duration_s: Decimal
    source_model_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        validate_stable_id(self.action_label, field_name="action_label")
        validate_stable_id(self.source_model_id, field_name="source_model_id")
        if tuple(value.stage for value in self.stages) != tuple(ActionDeliveryStage):
            raise ValueError("native TORAX action requires every delivery stage")
        validate_decimal(self.duration_s, field_name="duration_s", minimum=Decimal(0))
        if self.duration_s == 0:
            raise ValueError("native TORAX action duration must be positive")
        if len({value.power_w for value in self.stages}) != 1:
            raise ValueError("native TORAX direct delivery requires stage-exact power")
        if self.source_model_id != "generic-ion-electron-gaussian-proxy":
            raise ValueError("native TORAX source-model identity differs")

    @property
    def realized_power_w(self) -> Decimal:
        return self.stages[-1].power_w


@dataclass(frozen=True, slots=True)
class NativeToraxTrajectory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-trajectory'

    trajectory_id: str
    preparation_id: str
    action_id: str
    view_id: str
    time_s: tuple[Decimal, ...]
    core_electron_temperature_ev: tuple[Decimal, ...]
    radial_mean_electron_temperature_ev: tuple[Decimal, ...]
    core_edge_contrast_ev: tuple[Decimal, ...]
    minimum_electron_temperature_ev: tuple[Decimal, ...]
    minimum_ion_temperature_ev: tuple[Decimal, ...]
    torax_sim_error: str

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "preparation_id", "action_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        size = len(self.time_s)
        if size < 2 or any(
            len(values) != size
            for values in (
                self.core_electron_temperature_ev,
                self.radial_mean_electron_temperature_ev,
                self.core_edge_contrast_ev,
                self.minimum_electron_temperature_ev,
                self.minimum_ion_temperature_ev,
            )
        ):
            raise ValueError("native TORAX trajectory arrays have inconsistent lengths")
        if tuple(sorted(self.time_s)) != self.time_s or len(set(self.time_s)) != size:
            raise ValueError("native TORAX trajectory clock must be strictly increasing")
        for value in (
            *self.time_s,
            *self.core_electron_temperature_ev,
            *self.radial_mean_electron_temperature_ev,
            *self.core_edge_contrast_ev,
            *self.minimum_electron_temperature_ev,
            *self.minimum_ion_temperature_ev,
        ):
            validate_decimal(value, field_name="trajectory_value")
        if not self.torax_sim_error.strip():
            raise ValueError("native TORAX trajectory must retain simulator status")


@dataclass(frozen=True, slots=True)
class NativeToraxEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-episode'

    episode_id: str
    preparation_id: str
    action_id: str
    view_id: str
    disposition: NativeToraxEpisodeDisposition
    endpoint_delta_core_temperature_ev: Decimal | None
    endpoint_core_edge_contrast_ev: Decimal | None
    minimum_electron_temperature_ev: Decimal | None
    minimum_ion_temperature_ev: Decimal | None
    safety_margin_ev: Decimal | None
    effort_j: Decimal | None
    trajectory_sha256: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("episode_id", "preparation_id", "action_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.trajectory_sha256 is not None:
            validate_sha256(self.trajectory_sha256, field_name="trajectory_sha256")
        complete = self.disposition is NativeToraxEpisodeDisposition.COMPLETE
        outputs = (
            self.endpoint_delta_core_temperature_ev,
            self.endpoint_core_edge_contrast_ev,
            self.minimum_electron_temperature_ev,
            self.minimum_ion_temperature_ev,
            self.safety_margin_ev,
            self.effort_j,
        )
        for name, value in zip(
            (
                "endpoint_delta_core_temperature_ev",
                "endpoint_core_edge_contrast_ev",
                "minimum_electron_temperature_ev",
                "minimum_ion_temperature_ev",
                "safety_margin_ev",
                "effort_j",
            ),
            outputs,
            strict=True,
        ):
            if value is not None:
                validate_decimal(value, field_name=name)
        if complete:
            if any(value is None for value in outputs) or self.trajectory_sha256 is None:
                raise ValueError("complete native TORAX episode lacks outputs")
            if self.reason_codes:
                raise ValueError("complete native TORAX episode cannot carry reasons")
        elif any(value is not None for value in outputs) or self.trajectory_sha256 is not None:
            raise ValueError("invalid native TORAX episode cannot expose scoreable outputs")
        elif not self.reason_codes:
            raise ValueError("invalid native TORAX episode requires reasons")


__all__ = [
    'NativeToraxActionStage',
    'NativeToraxAction',
    "NativeToraxEpisodeDisposition",
    'NativeToraxEpisode',
    'NativeToraxPreparation',
    'NativeToraxTrajectory',
    'NativeToraxView',
]
