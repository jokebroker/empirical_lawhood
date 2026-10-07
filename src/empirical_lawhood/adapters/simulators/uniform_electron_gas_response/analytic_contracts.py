"""Typed, explicitly synthetic uniform electron gas analytic-development source and check."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .contracts import ActionCurrentRow
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/analytic-panel'

    panel_id: str
    config_sha256: str
    independent_unit_id: str
    rows: tuple[ActionCurrentRow, ...]
    slab_profile: tuple[tuple[Decimal, Decimal], ...]
    synthetic_reference: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if len(self.config_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.config_sha256
        ):
            raise ValueError("analytic panel requires a canonical config digest")
        if not self.synthetic_reference or len(self.rows) != 20:
            raise ValueError(
                "uniform electron gas analytic panel has one synthetic 20-condition acquisition"
            )
        if len({row.row_id for row in self.rows}) != 20:
            raise ValueError("uniform electron gas analytic panel has duplicate row IDs")
        if len(self.slab_profile) < 3 or len(self.slab_profile) % 2 == 0:
            raise ValueError("uniform electron gas slab profile needs an odd number of points")
        for x, ratio in self.slab_profile:
            validate_decimal(x, field_name="slab_x_m")
            validate_decimal(ratio, field_name="slab_field_ratio", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/analytic-evaluation-config'

    config_id: str
    source: UniformElectronGasAnalyticReferenceConfig
    synthetic_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if not self.synthetic_only:
            raise ValueError(
                "analytic uniform electron gas check cannot promote a physical measurement"
            )


@dataclass(frozen=True, slots=True)
class UniformElectronGasAnalyticCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/analytic-check'

    check_id: str
    independent_unit_id: str
    kernel_intercept_A_T_m3: Decimal | None
    q_intercept_stability_relative: Decimal | None
    slab_depth_m: Decimal | None
    passed: bool
    reason_codes: tuple[str, ...]
    synthetic_reference: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        for name in (
            "kernel_intercept_A_T_m3",
            "q_intercept_stability_relative",
            "slab_depth_m",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        if not self.synthetic_reference:
            raise ValueError(
                "analytic uniform electron gas check cannot claim native material evidence"
            )
        if self.passed and self.reason_codes:
            raise ValueError("passing analytic check cannot carry failure codes")
        if not self.passed and not self.reason_codes:
            raise ValueError("failed analytic check needs reason codes")


__all__ = [
    'UniformElectronGasAnalyticCheck',
    'UniformElectronGasAnalyticEvaluationConfig',
    'UniformElectronGasAnalyticPanel',
]
