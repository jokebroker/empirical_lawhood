"""Finite prediction/task/window operands frozen before native development."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, READOUTS, PreparationRoot, PreparationSourceConfig


FEATURES = (
    "receiver_position",
    "receiver_velocity",
    "position_X_spectral_norm_squared",
    "position_Y_spectral_norm_squared",
    "momentum_X_spectral_norm_squared",
    "momentum_Y_spectral_norm_squared",
    "position_X_norm_history_slope",
    "receiver_history_acceleration",
    "invocation_offset_time",
)
CANDIDATES = ("constant_gain", "direct_linear", "direct_quadratic")
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
PROJECTION_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/projected-observable-root-arrays-hdf5'
PRIVILEGED_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/privileged-root-diagnostics-hdf5'
PROSPECTIVE_TASK_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/sealed-task-root-outcomes-hdf5'
FIT_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/development-fit-arrays-hdf5'
ASSESSMENT_SCHEMA = 'empirical-lawhood/methods/matrix-preparation/development-assessment-arrays-hdf5'


@dataclass(frozen=True, slots=True)
class PreparationProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-projection-config'
    config_id: str
    source_config: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.config_id != f"{DEVELOPMENT}.projection-config"
            or self.source_config.object_schema != PreparationSourceConfig.SCHEMA
        ):
            raise ValueError("preparation projection has another source/owner identity")


@dataclass(frozen=True, slots=True)
class PreparationMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-method-config'
    config_id: str
    projection_config: ObjectIdentity
    implementation_plan_sha256: str
    absolute_tolerance: Decimal = Decimal("0.125")
    odd_tolerance: Decimal = Decimal("0.015625")
    maximum_halfwidth: Decimal = Decimal("0.125")
    residual_scale_floor: Decimal = Decimal("0.0078125")
    numerical_absolute_tolerance: Decimal = Decimal("0.0078125")
    numerical_odd_tolerance: Decimal = Decimal("0.00390625")
    nominal_interval_level: Decimal = Decimal("0.95")
    admission_threshold: Decimal = Decimal("0.90")
    target_centers: tuple[Decimal, ...] = (Decimal("-0.375"), Decimal(0), Decimal("0.375"))
    target_radius: Decimal = Decimal("0.125")
    readouts: tuple[int, ...] = READOUTS
    features: tuple[str, ...] = FEATURES
    candidates: tuple[str, ...] = CANDIDATES
    ridge_penalties: tuple[Decimal, ...] = tuple(Decimal(str(v)) for v in PENALTIES)
    independent_unit: str = "ORIGINAL_STOCHASTIC_ROOT_ALL_BRANCHES_NESTED"
    inference_scope: str = "EXPOSED_DEVELOPMENT_ONLY"
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_sha256(self.implementation_plan_sha256, field_name="implementation_plan_sha256")
        expected = {
            "absolute_tolerance": "0.125",
            "odd_tolerance": "0.015625",
            "maximum_halfwidth": "0.125",
            "residual_scale_floor": "0.0078125",
            "numerical_absolute_tolerance": "0.0078125",
            "numerical_odd_tolerance": "0.00390625",
            "nominal_interval_level": "0.95",
            "admission_threshold": "0.90",
            "target_radius": "0.125",
        }
        if (
            self.config_id != f"{DEVELOPMENT}.method-config"
            or self.projection_config.object_schema != PreparationProjectionConfig.SCHEMA
            or any(
                type(getattr(self, k)) is not Decimal or getattr(self, k) != Decimal(v)
                for k, v in expected.items()
            )
            or self.target_centers != (Decimal("-0.375"), Decimal(0), Decimal("0.375"))
            or any(type(v) is not Decimal for v in self.target_centers)
            or self.readouts != READOUTS
            or self.features != FEATURES
            or self.candidates != CANDIDATES
            or self.ridge_penalties != tuple(Decimal(str(v)) for v in PENALTIES)
            or self.independent_unit != "ORIGINAL_STOCHASTIC_ROOT_ALL_BRANCHES_NESTED"
            or self.inference_scope != "EXPOSED_DEVELOPMENT_ONLY"
            or self.grants_authority is not False
        ):
            raise ValueError(
                "preparation method changes a frozen task, model, numerical or exposure rule"
            )


@dataclass(frozen=True, slots=True)
class _ProjectionReport(CanonicalRecord):
    PART: ClassVar[str]
    report_id: str
    root: PreparationRoot
    refinement: int
    projection_config: ObjectIdentity
    native_result: ObjectIdentity
    data_sha256: str
    complete_native_occurrences: int
    incomplete_native_occurrences: int

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.data_sha256, field_name="data_sha256")
        suffix = "" if self.PART == "observable" else f".{self.PART}"
        count = {"observable": 65, "prospective-task": 15, "privileged": 45 if self.root.numerical_semantics else 0}[
            self.PART
        ]
        if (
            self.report_id != f"report.{self.root.root_id}{suffix}.r{self.refinement}"
            or self.refinement not in (1, 2)
            or type(self.refinement) is not int
            or self.projection_config.object_schema != PreparationProjectionConfig.SCHEMA
            or self.native_result.object_schema
            != 'empirical-lawhood/simulators/matrix-preparation/preparation-native-result'
            or any(
                type(v) is not int or v < 0
                for v in (self.complete_native_occurrences, self.incomplete_native_occurrences)
            )
            or self.complete_native_occurrences + self.incomplete_native_occurrences != count
        ):
            raise ValueError("preparation projection changes its root/member/native lineage")


@dataclass(frozen=True, slots=True)
class PreparationProjectionReport(_ProjectionReport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-projection-report'
    PART: ClassVar[str] = "observable"


@dataclass(frozen=True, slots=True)
class PreparationPrivilegedReport(_ProjectionReport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-privileged-report'
    PART: ClassVar[str] = "privileged"
    failed_mechanism_parents: tuple[str, ...]
    failed_benchmark_parents: tuple[str, ...]

    def __post_init__(self) -> None:
        _ProjectionReport.__post_init__(self)
        for field in ("failed_mechanism_parents", "failed_benchmark_parents"):
            values = getattr(self, field)
            require_sorted_unique_strings(values, field_name=field)
            if any(
                v not in ("hold", "x-negative", "x-positive", "y-negative", "y-positive")
                for v in values
            ):
                raise ValueError("privileged report changes its parent roster")


@dataclass(frozen=True, slots=True)
class PreparationProspectiveTaskReport(_ProjectionReport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-task-report'
    PART: ClassVar[str] = "prospective-task"


PreparationProjectionRecord = (
    PreparationProjectionReport | PreparationPrivilegedReport | PreparationProspectiveTaskReport
)
