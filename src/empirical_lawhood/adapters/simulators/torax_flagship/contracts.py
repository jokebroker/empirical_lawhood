"""Closed preparation, action, numerical-view and matched-panel records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


TORAX_HORIZON_S: Final = Decimal("0.040")
TORAX_RUNTIME_VERSIONS: Final = {
    "python": "3.11.14",
    "torax": "1.4.2",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
}


class ToraxFieldOrigin(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    MARGINALIZED = "MARGINALIZED"


class ToraxMappingDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


class ToraxMappingRejectionCode(StrEnum):
    INVALID_ACTION_UNIT = "INVALID_ACTION_UNIT"
    INSUFFICIENT_PROFILE = "INSUFFICIENT_PROFILE"
    INSUFFICIENT_GEOMETRY = "INSUFFICIENT_GEOMETRY"
    UNQUALIFIED_BOUNDARY = "UNQUALIFIED_BOUNDARY"
    UNQUALIFIED_COMPOSITION = "UNQUALIFIED_COMPOSITION"
    UNQUALIFIED_SOURCE = "UNQUALIFIED_SOURCE"
    UNQUALIFIED_TRANSPORT = "UNQUALIFIED_TRANSPORT"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    OUTCOME_ACCESS_FORBIDDEN = "OUTCOME_ACCESS_FORBIDDEN"


class ToraxAction(StrEnum):
    DOWN = "DOWN"
    HOLD = "HOLD"
    UP = "UP"


class ToraxViewRole(StrEnum):
    PRIMARY = "PRIMARY"
    SENSITIVITY = "SENSITIVITY"


class ToraxRolloutState(StrEnum):
    COMPLETE = "COMPLETE"
    PHYSICAL_SINK = "PHYSICAL_SINK"
    NUMERICAL_SINK = "NUMERICAL_SINK"
    SOLVER_FAILURE = "SOLVER_FAILURE"
    INVALID = "INVALID"
    INTERRUPTED = "INTERRUPTED"


class ToraxPanelDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ToraxRuntimeIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-runtime-identity'

    runtime_id: str
    python_version: str
    torax_version: str
    jax_version: str
    jaxlib_version: str
    numpy_version: str
    scipy_version: str
    platform_id: str
    backend: str
    device_id: str
    precision: str
    x64_enabled: bool
    runtime_artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.runtime_id, field_name="runtime_id")
        for name, text_value in (
            ("python_version", self.python_version),
            ("torax_version", self.torax_version),
            ("jax_version", self.jax_version),
            ("jaxlib_version", self.jaxlib_version),
            ("numpy_version", self.numpy_version),
            ("scipy_version", self.scipy_version),
            ("platform_id", self.platform_id),
            ("backend", self.backend),
            ("device_id", self.device_id),
            ("precision", self.precision),
        ):
            validate_nonempty(text_value, field_name=name)
        actual = {
            "python": self.python_version,
            "torax": self.torax_version,
            "jax": self.jax_version,
            "jaxlib": self.jaxlib_version,
            "numpy": self.numpy_version,
            "scipy": self.scipy_version,
        }
        if actual != TORAX_RUNTIME_VERSIONS:
            raise ValueError("TORAX runtime differs from the Phase 1 freeze")
        if self.backend != "cpu" or self.precision != "float64" or not self.x64_enabled:
            raise ValueError("flagship TORAX runtime must be CPU-first x64")
        require_sorted_unique_ids(
            self.runtime_artifacts, attribute="artifact_id", field_name="runtime_artifacts"
        )


@dataclass(frozen=True, slots=True)
class ToraxNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-numerical-view'

    view_id: str
    role: ToraxViewRole
    radial_cells: int
    timestep_s: Decimal
    horizon_s: Decimal
    solver_id: str
    linear_solver_id: str
    precision: str
    closure_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        validate_stable_id(self.solver_id, field_name="solver_id")
        validate_stable_id(self.linear_solver_id, field_name="linear_solver_id")
        validate_stable_id(self.closure_id, field_name="closure_id")
        validate_nonempty(self.precision, field_name="precision")
        validate_decimal(self.timestep_s, field_name="timestep_s", minimum=Decimal(0))
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        expected = {
            ToraxViewRole.PRIMARY: (24, Decimal("0.001")),
            ToraxViewRole.SENSITIVITY: (12, Decimal("0.002")),
        }
        if (self.radial_cells, self.timestep_s) != expected[self.role]:
            raise ValueError(
                "TORAX numerical view differs from the declared primary/sensitivity view"
            )
        if self.horizon_s != TORAX_HORIZON_S:
            raise ValueError("TORAX view horizon differs from 40 ms")
        if self.solver_id != "linear-theta-fully-implicit":
            raise ValueError("TORAX solver differs from the frozen fully implicit method")
        if self.linear_solver_id != "thomas" or self.precision != "float64":
            raise ValueError("TORAX linear solver or precision differs")


@dataclass(frozen=True, slots=True)
class ToraxThetaMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-theta-member'

    theta_id: str
    ti_over_te: Decimal
    main_ion: str
    impurity: str
    zeff: Decimal
    geometry_id: str
    major_radius_m: Decimal
    minor_radius_m: Decimal
    toroidal_field_t: Decimal
    elongation_lcfs: Decimal
    boundary_id: str
    current_convention_id: str
    source_radial_location: Decimal
    source_width: Decimal
    electron_heat_fraction: Decimal
    absorbed_power_fraction: Decimal
    chi_i_m2_s: Decimal
    chi_e_m2_s: Decimal
    particle_diffusivity_m2_s: Decimal
    particle_convection_m_s: Decimal
    radial_mapping_id: str

    def __post_init__(self) -> None:
        for name, text_value in (
            ("theta_id", self.theta_id),
            ("main_ion", self.main_ion),
            ("impurity", self.impurity),
            ("geometry_id", self.geometry_id),
            ("boundary_id", self.boundary_id),
            ("current_convention_id", self.current_convention_id),
            ("radial_mapping_id", self.radial_mapping_id),
        ):
            validate_stable_id(text_value, field_name=name)
        for name, positive_value in (
            ("ti_over_te", self.ti_over_te),
            ("zeff", self.zeff),
            ("source_width", self.source_width),
            ("chi_i_m2_s", self.chi_i_m2_s),
            ("chi_e_m2_s", self.chi_e_m2_s),
            ("particle_diffusivity_m2_s", self.particle_diffusivity_m2_s),
            ("major_radius_m", self.major_radius_m),
            ("minor_radius_m", self.minor_radius_m),
            ("toroidal_field_t", self.toroidal_field_t),
            ("elongation_lcfs", self.elongation_lcfs),
        ):
            validate_decimal(positive_value, field_name=name, minimum=Decimal(0))
            if positive_value == 0:
                raise ValueError(f"{name} must be positive")
        for name, fraction_value in (
            ("source_radial_location", self.source_radial_location),
            ("electron_heat_fraction", self.electron_heat_fraction),
            ("absorbed_power_fraction", self.absorbed_power_fraction),
        ):
            validate_decimal(fraction_value, field_name=name, minimum=Decimal(0))
            if fraction_value > 1:
                raise ValueError(f"{name} must be at most one")
        validate_decimal(self.particle_convection_m_s, field_name="particle_convection_m_s")
        if self.geometry_id != "circular-geometry-testing-assumption":
            raise ValueError("TORAX geometry must remain explicitly testing-only")
        if self.minor_radius_m > self.major_radius_m:
            raise ValueError("TORAX circular minor radius exceeds major radius")


@dataclass(frozen=True, slots=True)
class ToraxMappedField(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-mapped-field'

    field_id: str
    origin: ToraxFieldOrigin
    values: tuple[Decimal, ...]
    radial_coordinates: tuple[Decimal, ...]
    native_unit: str
    source_feature_id: str | None
    assumption_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if not self.values:
            raise ValueError("mapped TORAX field cannot be empty")
        if self.radial_coordinates and len(self.radial_coordinates) != len(self.values):
            raise ValueError("mapped field radial coordinates differ")
        for value in (*self.values, *self.radial_coordinates):
            validate_decimal(value, field_name="mapped_value")
        if self.source_feature_id is not None:
            validate_stable_id(self.source_feature_id, field_name="source_feature_id")
        if self.assumption_id is not None:
            validate_stable_id(self.assumption_id, field_name="assumption_id")
        if self.origin in {ToraxFieldOrigin.OBSERVED, ToraxFieldOrigin.DERIVED}:
            if self.source_feature_id is None:
                raise ValueError("observed/derived field requires a source feature")
        elif self.assumption_id is None:
            raise ValueError("assumed/marginalized field requires an assumption identity")


@dataclass(frozen=True, slots=True)
class ToraxFieldClassification(CanonicalRecord):
    """Explicit provenance class for every mapped profile or closure field."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-field-classification'

    field_id: str
    origin: ToraxFieldOrigin
    source_feature_id: str | None
    assumption_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        if self.source_feature_id is not None:
            validate_stable_id(self.source_feature_id, field_name="source_feature_id")
        if self.assumption_id is not None:
            validate_stable_id(self.assumption_id, field_name="assumption_id")
        if self.origin in {ToraxFieldOrigin.OBSERVED, ToraxFieldOrigin.DERIVED}:
            if self.source_feature_id is None or self.assumption_id is not None:
                raise ValueError("observed/derived classification requires only source provenance")
        elif self.assumption_id is None or self.source_feature_id is not None:
            raise ValueError("assumed/marginalized classification requires only an assumption")


@dataclass(frozen=True, slots=True)
class ToraxPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-preparation'

    preparation_id: str
    mast_state: ObjectIdentity
    theta: ToraxThetaMember
    fields: tuple[ToraxMappedField, ...]
    classifications: tuple[ToraxFieldClassification, ...]
    runtime: ObjectIdentity
    stock_iter_hybrid: bool
    outcome_blind_mapping: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        require_sorted_unique_ids(self.fields, attribute="field_id", field_name="fields")
        required = {"electron-density", "electron-temperature", "ion-temperature", "plasma-current"}
        if {value.field_id for value in self.fields} != required:
            raise ValueError("TORAX preparation field set differs")
        require_sorted_unique_ids(
            self.classifications, attribute="field_id", field_name="classifications"
        )
        classified = required | {
            "boundary",
            "composition",
            "current-convention",
            "geometry",
            "radial-mapping",
            "source",
            "transport",
        }
        if {value.field_id for value in self.classifications} != classified:
            raise ValueError("TORAX preparation classification ledger is incomplete")
        if self.stock_iter_hybrid:
            raise ValueError("stock ITER-hybrid preparation is forbidden")
        if not self.outcome_blind_mapping:
            raise ValueError("MAST-to-TORAX preparation mapping must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class ToraxPreparationEnsemble(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-preparation-ensemble'

    ensemble_id: str
    mast_state: ObjectIdentity
    preparations: tuple[ToraxPreparation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ensemble_id, field_name="ensemble_id")
        require_sorted_unique_ids(
            self.preparations, attribute="preparation_id", field_name="preparations"
        )
        if not self.preparations:
            raise ValueError("TORAX preparation ensemble cannot be empty")
        if any(value.mast_state != self.mast_state for value in self.preparations):
            raise ValueError("TORAX ensemble crosses MAST state identities")


@dataclass(frozen=True, slots=True)
class ToraxMappingResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-mapping-result'

    result_id: str
    mast_state: ObjectIdentity
    disposition: ToraxMappingDisposition
    ensemble: ToraxPreparationEnsemble | None
    rejection_codes: tuple[ToraxMappingRejectionCode, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if tuple(sorted(set(self.rejection_codes))) != self.rejection_codes:
            raise ValueError("mapping rejection codes must be sorted and unique")
        if self.disposition is ToraxMappingDisposition.ACCEPTED:
            if self.ensemble is None or self.rejection_codes:
                raise ValueError("accepted mapping requires only a preparation ensemble")
        elif self.ensemble is not None or not self.rejection_codes:
            raise ValueError("non-accepted mapping requires only typed rejection reasons")


@dataclass(frozen=True, slots=True)
class ToraxActionStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-action-stage'

    stage: ActionDeliveryStage
    power_w: Decimal
    native_unit: str
    coordinate_s: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.power_w, field_name="power_w", minimum=Decimal(0))
        if self.native_unit != "W":
            raise ValueError("TORAX proxy-source action must remain in watts")
        validate_decimal(self.coordinate_s, field_name="coordinate_s", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class ToraxActionSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-action-spec'

    action_id: str
    action: ToraxAction
    stages: tuple[ToraxActionStage, ...]
    duration_s: Decimal
    horizon_s: Decimal
    source_model_id: str
    source_is_proxy: bool
    current_drive_model_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        validate_stable_id(self.source_model_id, field_name="source_model_id")
        validate_stable_id(self.current_drive_model_id, field_name="current_drive_model_id")
        if tuple(value.stage for value in self.stages) != tuple(ActionDeliveryStage):
            raise ValueError("TORAX action requires requested/accepted/applied/realized stages")
        validate_decimal(self.duration_s, field_name="duration_s", minimum=Decimal(0))
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        if self.duration_s != TORAX_HORIZON_S or self.horizon_s != TORAX_HORIZON_S:
            raise ValueError("TORAX action duration/horizon differ from 40 ms")
        if (
            self.source_model_id != "generic-ion-electron-gaussian-proxy"
            or not self.source_is_proxy
        ):
            raise ValueError("TORAX source must be labelled as the generic Gaussian proxy")
        if self.current_drive_model_id != "fixed-current-zero-generic-source":
            raise ValueError("TORAX current-source assumption differs")
        if self.stages[-1].power_w == 0:
            raise ValueError("TORAX candidates use nonzero absolute proxy-source power")


@dataclass(frozen=True, slots=True)
class ToraxValidityChecks(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-validity-checks'

    converged: bool
    conserved: bool
    boundary_valid: bool
    solver_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        valid = self.converged and self.conserved and self.boundary_valid and self.solver_valid
        if valid == bool(self.reason_codes):
            raise ValueError("TORAX validity flags differ from their reasons")

    @property
    def valid(self) -> bool:
        return self.converged and self.conserved and self.boundary_valid and self.solver_valid


@dataclass(frozen=True, slots=True)
class ToraxRolloutResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-rollout-result'

    rollout_id: str
    preparation: ObjectIdentity
    theta_id: str
    view: ToraxNumericalView
    action: ToraxActionSpec
    state: ToraxRolloutState
    endpoint_delta_te_core_ev: Decimal | None
    preservation_margin: Decimal | None
    effort_j: Decimal | None
    validity: ToraxValidityChecks
    state_artifact: ArtifactIdentity | None
    action_artifact: ArtifactIdentity | None
    runtime_artifacts: tuple[ArtifactIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.rollout_id, field_name="rollout_id")
        validate_stable_id(self.theta_id, field_name="theta_id")
        for name, value in (
            ("endpoint_delta_te_core_ev", self.endpoint_delta_te_core_ev),
            ("preservation_margin", self.preservation_margin),
            ("effort_j", self.effort_j),
        ):
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_ids(
            self.runtime_artifacts, attribute="artifact_id", field_name="runtime_artifacts"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.state is ToraxRolloutState.COMPLETE
        if complete:
            if not self.validity.valid or any(
                value is None
                for value in (
                    self.endpoint_delta_te_core_ev,
                    self.preservation_margin,
                    self.effort_j,
                    self.state_artifact,
                    self.action_artifact,
                )
            ):
                raise ValueError("complete TORAX rollout lacks valid scientific outputs")
            if not self.runtime_artifacts:
                raise ValueError("complete TORAX rollout requires external runtime/JIT artifacts")
            if self.reason_codes:
                raise ValueError("complete TORAX rollout cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("non-complete TORAX rollout requires reasons")
        elif any(
            value is not None
            for value in (
                self.endpoint_delta_te_core_ev,
                self.preservation_margin,
                self.effort_j,
                self.state_artifact,
                self.action_artifact,
            )
        ):
            raise ValueError("invalid/incomplete rollout cannot expose scoreable outputs")


@dataclass(frozen=True, slots=True)
class ToraxMatchedPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-matched-panel'

    panel_id: str
    preparation: ObjectIdentity
    theta_id: str
    view: ToraxNumericalView
    rollouts: tuple[ToraxRolloutResult, ...]
    disposition: ToraxPanelDisposition
    independent_preparation_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.theta_id, field_name="theta_id")
        require_sorted_unique_ids(self.rollouts, attribute="rollout_id", field_name="rollouts")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.independent_preparation_count != 0:
            raise ValueError("Theta-member panels are nested, not independent preparations")
        if any(
            value.preparation != self.preparation
            or value.theta_id != self.theta_id
            or value.view != self.view
            for value in self.rollouts
        ):
            raise ValueError("matched TORAX panel crosses preparation/theta/view identities")
        actions = {value.action.action for value in self.rollouts}
        complete = (
            len(self.rollouts) == 3
            and actions == set(ToraxAction)
            and all(value.state is ToraxRolloutState.COMPLETE for value in self.rollouts)
        )
        if (self.disposition is ToraxPanelDisposition.COMPLETE) != complete:
            raise ValueError("TORAX panel disposition differs from action completeness")
        if complete and self.reason_codes:
            raise ValueError("complete TORAX panel cannot carry reasons")
        if not complete and not self.reason_codes:
            raise ValueError("incomplete/unevaluable TORAX panel requires reasons")


__all__ = [
    "TORAX_HORIZON_S",
    "TORAX_RUNTIME_VERSIONS",
    "ToraxAction",
    "ToraxActionSpec",
    "ToraxActionStage",
    "ToraxFieldOrigin",
    "ToraxFieldClassification",
    "ToraxMappedField",
    "ToraxMappingDisposition",
    "ToraxMappingRejectionCode",
    "ToraxMappingResult",
    "ToraxMatchedPanel",
    "ToraxNumericalView",
    "ToraxPanelDisposition",
    "ToraxPreparation",
    "ToraxPreparationEnsemble",
    "ToraxRolloutResult",
    "ToraxRolloutState",
    "ToraxRuntimeIdentity",
    "ToraxThetaMember",
    "ToraxValidityChecks",
    "ToraxViewRole",
]
