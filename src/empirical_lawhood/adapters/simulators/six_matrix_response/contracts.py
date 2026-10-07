"""Strict scientific and numerical contracts for the Six-matrix response six-matrix world."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import prod
from pathlib import PurePosixPath
from typing import ClassVar

from .history_preparation import FOUR_FAMILY_PREPARATION_IDS

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_stable_id,
)


class BetaCouplingRule(StrEnum):
    """Closed mass/coupling convention; runtime cannot switch semantics."""

    PUBLISHED_DETERMINISTIC = "PUBLISHED_DETERMINISTIC"


class MatrixIntegratorKind(StrEnum):
    BAOAB_UNDERDAMPED_LANGEVIN = "BAOAB_UNDERDAMPED_LANGEVIN"


class MatrixPrecision(StrEnum):
    COMPLEX128 = "COMPLEX128"


@dataclass(frozen=True, slots=True)
class CanonicalSixMatrixIsotropicModel(CanonicalRecord):
    """Exact published isotropic action convention used only by numerical qualification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/canonical-six-matrix-isotropic-model'
    VERSION: ClassVar[str] = "1.0.0"

    model_id: str
    matrix_dimension_rule: str
    representation_dimension_q: int
    mass_m: Decimal
    cross_coupling_gamma: Decimal
    beta_rule: BetaCouplingRule
    scaled_coupling_id: str
    scaled_coupling_alpha_tilde: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        validate_stable_id(self.scaled_coupling_id, field_name="scaled_coupling_id")
        if self.matrix_dimension_rule != "n=q^2":
            raise ValueError("canonical model requires n=q^2")
        if self.representation_dimension_q not in {2, 3, 4}:
            raise ValueError("Six-matrix response canonical q must be one of 2, 3 or 4")
        validate_decimal(self.mass_m, field_name="mass_m", minimum=Decimal("0"))
        validate_decimal(
            self.cross_coupling_gamma,
            field_name="cross_coupling_gamma",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.scaled_coupling_alpha_tilde,
            field_name="scaled_coupling_alpha_tilde",
            minimum=Decimal("0"),
        )
        if self.mass_m == 0 or self.cross_coupling_gamma != Decimal("1"):
            raise ValueError("canonical numerical qualification identity requires M>0 and published gamma=1")
        if self.beta_rule is not BetaCouplingRule.PUBLISHED_DETERMINISTIC:
            raise ValueError("canonical numerical qualification identity requires the published beta map")

    @property
    def matrix_dimension_n(self) -> int:
        return self.representation_dimension_q**2

    @property
    def quadratic_casimir(self) -> Decimal:
        q = Decimal(self.representation_dimension_q)
        return (q * q - Decimal(1)) / Decimal(4)

    @property
    def alpha(self) -> Decimal:
        return self.scaled_coupling_alpha_tilde / Decimal(self.representation_dimension_q)

    @property
    def mu(self) -> Decimal:
        return (
            Decimal(2)
            * (Decimal(4) * self.quadratic_casimir * self.mass_m - Decimal(1))
            / Decimal(9)
        )

    @property
    def beta(self) -> Decimal:
        return -(self.alpha**2) * self.mu


@dataclass(frozen=True, slots=True)
class SixMatrixResponseModelFamilyMember(CanonicalRecord):
    """One symmetric anisotropic feasibility member; alpha remains the action."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-model-family-member'
    VERSION: ClassVar[str] = "1.0.0"

    member_id: str
    mass_x: Decimal
    mass_y: Decimal
    cross_coupling_gamma: Decimal
    beta_rule: BetaCouplingRule

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        for name in ("mass_x", "mass_y", "cross_coupling_gamma"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
            if value == 0:
                raise ValueError("Six-matrix response feasibility masses and gamma must be positive")
        if self.mass_x != self.mass_y:
            raise ValueError("Six-matrix response baseline predeclares only symmetric mass members")
        if self.beta_rule is not BetaCouplingRule.PUBLISHED_DETERMINISTIC:
            raise ValueError("Six-matrix response baseline has exactly one deterministic beta rule")


@dataclass(frozen=True, slots=True)
class AnisotropicSixMatrixConstitutiveModel(CanonicalRecord):
    """Frozen anisotropic family identity, distinct from the canonical numerical qualification model."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/anisotropic-six-matrix-constitutive-model'
    VERSION: ClassVar[str] = "1.0.0"

    model_family_id: str
    matrix_dimension_rule: str
    scaled_coupling_id_x: str
    scaled_coupling_id_y: str
    beta_rule: BetaCouplingRule
    family_members: tuple[SixMatrixResponseModelFamilyMember, ...]
    protected_action_quantity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.model_family_id, field_name="model_family_id")
        validate_stable_id(self.scaled_coupling_id_x, field_name="scaled_coupling_id_x")
        validate_stable_id(self.scaled_coupling_id_y, field_name="scaled_coupling_id_y")
        if self.matrix_dimension_rule != "n=q^2":
            raise ValueError("anisotropic model requires n=q^2")
        require_sorted_unique_ids(
            self.family_members,
            attribute="member_id",
            field_name="family_members",
        )
        if not self.family_members:
            raise ValueError("anisotropic model requires a nonempty frozen family")
        require_sorted_unique_strings(
            self.protected_action_quantity_ids,
            field_name="protected_action_quantity_ids",
            allow_empty=False,
        )
        if self.protected_action_quantity_ids != (
            "six-matrix-response.quantity.alpha-tilde-x",
            "six-matrix-response.quantity.alpha-tilde-y",
        ):
            raise ValueError("Six-matrix response action may change only the two scaled couplings")
        if self.beta_rule is not BetaCouplingRule.PUBLISHED_DETERMINISTIC or any(
            value.beta_rule is not self.beta_rule for value in self.family_members
        ):
            raise ValueError("anisotropic family cannot switch beta semantics")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-numerical-view'
    VERSION: ClassVar[str] = "1.0.0"

    view_id: str
    integrator: MatrixIntegratorKind
    timestep: Decimal
    time_unit: str
    friction_gamma: Decimal
    bath_temperature: Decimal
    precision: MatrixPrecision
    rng_algorithm: str
    rng_version: str
    stream_derivation_rule_id: str
    checkpoint_interval_steps: int

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        validate_stable_id(
            self.stream_derivation_rule_id,
            field_name="stream_derivation_rule_id",
        )
        validate_semantic_version(self.rng_version)
        if self.time_unit != "dimensionless-langevin-time":
            raise ValueError("Six-matrix response uses one dimensionless Langevin clock")
        for name in ("timestep", "friction_gamma", "bath_temperature"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
            if value == 0:
                raise ValueError("Six-matrix response numerical view values must be positive")
        if self.checkpoint_interval_steps < 1:
            raise ValueError("checkpoint interval must be positive")
        if self.precision is not MatrixPrecision.COMPLEX128:
            raise ValueError("Six-matrix response baseline persists and computes in complex128")


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationSchedule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-schedule'
    VERSION: ClassVar[str] = "1.0.0"

    schedule_id: str
    ramp_steps: int
    dwell_steps: int
    qualification_steps: int
    fast_receiver_cadence_steps: int
    spectral_receiver_cadence_steps: int

    def __post_init__(self) -> None:
        validate_stable_id(self.schedule_id, field_name="schedule_id")
        for name in (
            "ramp_steps",
            "dwell_steps",
            "qualification_steps",
            "fast_receiver_cadence_steps",
            "spectral_receiver_cadence_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.spectral_receiver_cadence_steps < self.fast_receiver_cadence_steps:
            raise ValueError("spectral cadence cannot be faster than the fast receiver")

    @property
    def total_steps(self) -> int:
        return self.ramp_steps + self.dwell_steps + self.qualification_steps


@dataclass(frozen=True, slots=True)
class SixMatrixResponseFeasibilityEnvelope(CanonicalRecord):
    """No-rescue anisotropic feasibility atlas and deterministic selection contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-feasibility-envelope'
    VERSION: ClassVar[str] = "1.0.0"

    envelope_id: str
    feasibility_alpha_tilde_min: Decimal
    feasibility_alpha_tilde_max: Decimal
    feasibility_axis_count: int
    feasibility_history_ids: tuple[str, ...]
    feasibility_seed_count: int
    feasibility_schedule: SixMatrixResponsePreparationSchedule
    confirmation_alpha_tilde_min: Decimal
    confirmation_alpha_tilde_max: Decimal
    confirmation_axis_count: int
    confirmation_history_ids: tuple[str, ...]
    confirmation_seed_count: int
    confirmation_schedule: SixMatrixResponsePreparationSchedule
    secondary_view_stratification_rule_id: str
    family_selection_metric_ids: tuple[str, ...]
    family_selection_tie_break: str
    maximum_anisotropic_feasibility_integration_steps: int

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        validate_stable_id(
            self.secondary_view_stratification_rule_id,
            field_name="secondary_view_stratification_rule_id",
        )
        for prefix in ("feasibility", "confirmation"):
            lower = getattr(self, f"{prefix}_alpha_tilde_min")
            upper = getattr(self, f"{prefix}_alpha_tilde_max")
            validate_decimal(lower, field_name=f"{prefix}_alpha_tilde_min", minimum=Decimal(0))
            validate_decimal(upper, field_name=f"{prefix}_alpha_tilde_max", minimum=lower)
            if lower == upper:
                raise ValueError("anisotropic feasibility alpha grids must have nonzero width")
            if getattr(self, f"{prefix}_axis_count") < 3:
                raise ValueError("anisotropic feasibility alpha grids require at least three points per axis")
            if getattr(self, f"{prefix}_seed_count") < 1:
                raise ValueError("anisotropic feasibility seed counts must be positive")
            expected_history_order = (
                tuple(FOUR_FAMILY_PREPARATION_IDS[index] for index in (0, 2, 3))
                if prefix == "feasibility" else FOUR_FAMILY_PREPARATION_IDS
            )
            if getattr(self, f"{prefix}_history_ids") != expected_history_order:
                raise ValueError("feasibility histories differ from the complete scientific acquisition order")
        if self.feasibility_axis_count != 13 or self.feasibility_seed_count != 3:
            raise ValueError("Six-matrix response baseline q=2 atlas is the frozen 13x13x3-seed design")
        if self.confirmation_axis_count != 9 or self.confirmation_seed_count != 4:
            raise ValueError("Six-matrix response baseline q=3 atlas is the frozen 9x9x4-seed design")
        require_sorted_unique_strings(
            self.family_selection_metric_ids,
            field_name="family_selection_metric_ids",
            allow_empty=False,
        )
        if self.family_selection_tie_break != "ascending-member-id":
            raise ValueError("anisotropic feasibility family tie-break must be bytewise member identity")
        if self.maximum_anisotropic_feasibility_integration_steps < 1:
            raise ValueError("anisotropic feasibility integration ceiling must be positive")

    @property
    def feasibility_rollouts_per_member(self) -> int:
        return self.feasibility_axis_count**2 * len(self.feasibility_history_ids) * self.feasibility_seed_count

    @property
    def confirmation_rollouts(self) -> int:
        return self.confirmation_axis_count**2 * len(self.confirmation_history_ids) * self.confirmation_seed_count


@dataclass(frozen=True, slots=True)
class SixMatrixResponseNumericalThresholds(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-numerical-thresholds'
    VERSION: ClassVar[str] = "1.0.0"

    thresholds_id: str
    gradient_relative_error_max: Decimal
    hermiticity_residual_max: Decimal
    invariance_relative_error_max: Decimal
    ideal_spectrum_absolute_error_max: Decimal
    radius_phi_min: Decimal
    radius_phi_max: Decimal
    su2_closure_ratio_max: Decimal
    approximate_kernel_band_ratio_max: Decimal
    persistence_fraction_min: Decimal
    confirmation_seed_recurrence_minimum: int
    confirmation_history_recurrence_minimum: int
    confirmation_minimum_cells_per_axis: int
    numerical_view_phase_concordance_min: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.thresholds_id, field_name="thresholds_id")
        for name in (
            "gradient_relative_error_max",
            "hermiticity_residual_max",
            "invariance_relative_error_max",
            "ideal_spectrum_absolute_error_max",
            "radius_phi_min",
            "radius_phi_max",
            "su2_closure_ratio_max",
            "approximate_kernel_band_ratio_max",
            "persistence_fraction_min",
            "numerical_view_phase_concordance_min",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.radius_phi_min < self.radius_phi_max:
            raise ValueError("radius interval must be strictly ordered")
        for name in (
            "persistence_fraction_min",
            "numerical_view_phase_concordance_min",
        ):
            if getattr(self, name) > Decimal(1):
                raise ValueError(f"{name} cannot exceed one")
        if (
            self.confirmation_seed_recurrence_minimum != 3
            or self.confirmation_history_recurrence_minimum != 2
            or self.confirmation_minimum_cells_per_axis != 2
        ):
            raise ValueError("Six-matrix response baseline freezes the anisotropic feasibility hard feasibility minima")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseSizeCompatibility(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-size-compatibility'
    VERSION: ClassVar[str] = "1.0.0"

    compatibility_id: str
    assay_roster: tuple[int, ...]
    raw_coupling_formula: str
    scaled_coupling_formula: str
    action_normalization_formula: str
    time_normalization_id: str
    friction_normalization_id: str
    bath_noise_normalization_id: str
    receiver_normalization_ids: tuple[str, ...]
    effort_normalization_id: str
    spectral_transport_kind: str
    projector_transport_kind: str
    decisive_falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.compatibility_id, field_name="compatibility_id")
        for name in (
            "time_normalization_id",
            "friction_normalization_id",
            "bath_noise_normalization_id",
            "effort_normalization_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.assay_roster != (2, 3, 4):
            raise ValueError("Six-matrix response finite-size route is exactly q=2,3,4")
        if self.raw_coupling_formula != "alpha=alpha_tilde/q":
            raise ValueError("raw coupling must use the published q scaling")
        if self.scaled_coupling_formula != "alpha_tilde=q*alpha":
            raise ValueError("scaled coupling must use the published convention")
        if self.action_normalization_formula != "S_density=S/n":
            raise ValueError("Six-matrix response action receiver must be normalized by n")
        require_sorted_unique_strings(
            self.receiver_normalization_ids,
            field_name="receiver_normalization_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )
        if self.spectral_transport_kind != "quotient-invariant-normalized-bands":
            raise ValueError("cross-q spectra require quotient-valid summaries")
        if self.projector_transport_kind != "not-applicable-across-q":
            raise ValueError("Six-matrix response forbids direct cross-q projector comparison")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseArtifactDataset(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-artifact-dataset'
    VERSION: ClassVar[str] = "1.0.0"

    dataset_id: str
    path: str
    dtype: str
    minimum_shape: tuple[int, ...]
    maximum_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    fill_value_id: str
    native_unit: str
    frame_id: str
    clock_id: str
    key_role_id: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "fill_value_id",
            "frame_id",
            "clock_id",
            "key_role_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        path = PurePosixPath(self.path)
        if not self.path.startswith("/") or str(path) != self.path or self.path == "/":
            raise ValueError("Six-matrix response HDF5 dataset path must be normalized and absolute")
        if self.dtype not in {"<c16", "<f8", "<i8", "<u1"}:
            raise ValueError("Six-matrix response HDF5 dtype is outside the closed inventory")
        if (
            not self.minimum_shape
            or len(self.minimum_shape) != len(self.maximum_shape)
            or len(self.chunk_shape) != len(self.minimum_shape)
            or any(value < 0 for value in (*self.minimum_shape, *self.maximum_shape))
            or any(value < 1 for value in self.chunk_shape)
            or any(
                chunk > minimum
                for chunk, minimum in zip(
                    self.chunk_shape,
                    self.minimum_shape,
                    strict=True,
                )
            )
            or any(
                lower > upper
                for lower, upper in zip(
                    self.minimum_shape,
                    self.maximum_shape,
                    strict=True,
                )
            )
        ):
            raise ValueError("Six-matrix response HDF5 dataset shape bounds are invalid")
        if not self.native_unit:
            raise ValueError("Six-matrix response HDF5 datasets require native units")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseArtifactProfileConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-artifact-profile-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    payload_schema: str
    media_type: str
    datasets: tuple[SixMatrixResponseArtifactDataset, ...]
    maximum_objects: int
    maximum_dataset_elements: int
    maximum_total_elements: int
    maximum_decoded_bytes: int
    maximum_shard_bytes: int
    deterministic_writer_id: str
    compression_filter_id: str
    shuffle_filter_enabled: bool
    checksum_filter_id: str
    object_creation_order_id: str
    root_track_object_times: bool
    dataset_track_object_times: bool
    group_object_time_policy_id: str
    hdf5_library_version_policy_id: str
    libver_bounds_id: str
    one_writer_per_shard: bool
    extra_objects_forbidden: bool
    external_links_forbidden: bool
    variable_length_values_forbidden: bool

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_schema

        validate_stable_id(self.config_id, field_name="config_id")
        validate_schema(self.payload_schema)
        for name in (
            "deterministic_writer_id",
            "compression_filter_id",
            "checksum_filter_id",
            "object_creation_order_id",
            "group_object_time_policy_id",
            "hdf5_library_version_policy_id",
            "libver_bounds_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.media_type != "application/x-hdf5":
            raise ValueError("Six-matrix response binary profile is audited HDF5")
        require_sorted_unique_ids(self.datasets, attribute="dataset_id", field_name="datasets")
        paths = tuple(value.path for value in self.datasets)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("Six-matrix response HDF5 dataset paths must be sorted and unique")
        groups = {"/"}
        for dataset_path in paths:
            parent = PurePosixPath(dataset_path).parent
            while str(parent) != ".":
                groups.add(str(parent))
                if str(parent) == "/":
                    break
                parent = parent.parent
        dataset_elements = tuple(prod(value.maximum_shape) for value in self.datasets)
        if (
            min(
                self.maximum_objects,
                self.maximum_dataset_elements,
                self.maximum_total_elements,
                self.maximum_decoded_bytes,
                self.maximum_shard_bytes,
            )
            < 1
        ):
            raise ValueError("Six-matrix response HDF5 bounds must be positive")
        if self.maximum_objects < len(groups) + len(self.datasets):
            raise ValueError("Six-matrix response HDF5 object ceiling is below its exact inventory")
        if (
            max(dataset_elements) > self.maximum_dataset_elements
            or sum(dataset_elements) > self.maximum_total_elements
        ):
            raise ValueError("Six-matrix response HDF5 element ceilings are below dataset geometry")
        dtype_bytes = {"<c16": 16, "<f8": 8, "<i8": 8, "<u1": 1}
        if (
            sum(
                elements * dtype_bytes[dataset.dtype]
                for dataset, elements in zip(
                    self.datasets,
                    dataset_elements,
                    strict=True,
                )
            )
            > self.maximum_decoded_bytes
        ):
            raise ValueError("Six-matrix response HDF5 decoded-byte ceiling is below dataset geometry")
        if self.maximum_shard_bytes > 536_870_912:
            raise ValueError("Six-matrix response shards retain a 512 MiB vfat safety ceiling")
        if self.maximum_decoded_bytes < self.maximum_shard_bytes:
            raise ValueError("decoded-byte ceiling cannot be below encoded shard ceiling")
        if not (
            self.extra_objects_forbidden
            and self.external_links_forbidden
            and self.variable_length_values_forbidden
        ):
            raise ValueError("Six-matrix response audited HDF5 forbids undeclared or unbounded content")
        if (
            self.compression_filter_id != "six-matrix-response.hdf5.compression.none"
            or self.shuffle_filter_enabled
            or self.checksum_filter_id != "six-matrix-response.hdf5.checksum.none"
            or self.object_creation_order_id != "six-matrix-response.hdf5.creation-order.lexicographic-path"
            or self.root_track_object_times
            or self.dataset_track_object_times
            or self.group_object_time_policy_id
            != "six-matrix-response.hdf5.group-time-disable-request-byte-parity"
            or self.hdf5_library_version_policy_id != "six-matrix-response.hdf5.library-lock-receipt-exact"
            or self.libver_bounds_id != "six-matrix-response.hdf5.libver-earliest-to-hdf5-1-14"
            or not self.one_writer_per_shard
        ):
            raise ValueError("Six-matrix response HDF5 writer semantics are not fully deterministic")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseSixMatrixSourceConfig(CanonicalRecord):
    """Complete source-free Six-matrix response simulator selection and work bound."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-six-matrix-source-config'
    VERSION: ClassVar[str] = "1.0.0"

    config_id: str
    config_version: str
    canonical_reference_models: tuple[CanonicalSixMatrixIsotropicModel, ...]
    anisotropic_model: AnisotropicSixMatrixConstitutiveModel
    primary_view: SixMatrixResponseNumericalView
    secondary_view: SixMatrixResponseNumericalView
    feasibility_envelope: SixMatrixResponseFeasibilityEnvelope
    numerical_thresholds: SixMatrixResponseNumericalThresholds
    finite_size_compatibility: SixMatrixResponseSizeCompatibility
    independent_unit_kind: str
    acquisition_group_rule_id: str
    maximum_total_integration_steps: int
    maximum_worker_processes: int
    blas_threads_per_worker: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_semantic_version(self.config_version)
        validate_stable_id(
            self.acquisition_group_rule_id,
            field_name="acquisition_group_rule_id",
        )
        require_sorted_unique_ids(
            self.canonical_reference_models,
            attribute="model_id",
            field_name="canonical_reference_models",
        )
        if not self.canonical_reference_models:
            raise ValueError("Six-matrix numerical qualification requires a frozen canonical reference roster")
        if self.primary_view.timestep != self.secondary_view.timestep * Decimal(2):
            raise ValueError("secondary Six-matrix response numerical view must be exactly dt/2")
        if (
            self.primary_view.friction_gamma != self.secondary_view.friction_gamma
            or self.primary_view.bath_temperature != self.secondary_view.bath_temperature
            or self.primary_view.integrator is not self.secondary_view.integrator
        ):
            raise ValueError("Six-matrix response numerical views may differ only in timestep identity")
        if self.independent_unit_kind != "rng-seeded-matrix-trajectory":
            raise ValueError("Six-matrix response independent units are RNG-seeded matrix trajectories")
        if (
            self.maximum_total_integration_steps
            < self.feasibility_envelope.maximum_anisotropic_feasibility_integration_steps
        ):
            raise ValueError("total integration ceiling cannot be below anisotropic feasibility")
        if self.maximum_worker_processes != 8 or self.blas_threads_per_worker != 1:
            raise ValueError("Six-matrix response baseline is admitted only for the measured 8-core thread-capped host")
        if self.grants_authority:
            raise ValueError("source configuration cannot grant execution authority")
