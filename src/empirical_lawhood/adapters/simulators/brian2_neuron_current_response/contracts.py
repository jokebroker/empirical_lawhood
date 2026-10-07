"""Typed, bounded leaky integrate-and-fire canary input."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .._native_admission import (
    BRIAN2_ESTIMATED_BYTES_PER_UPDATE,
    BRIAN2_MAX_ESTIMATED_HISTORY_BYTES,
    NativeTimeGridAdmission,
    admit_native_time_grid,
)


@dataclass(frozen=True, slots=True)
class Brian2LIFNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/brian2-neuron-current-response/brian2-lif-native-config'

    config_id: str
    independent_unit_id: str
    timestep_ms: Decimal
    baseline_ms: Decimal
    horizon_ms: Decimal
    membrane_tau_ms: Decimal
    membrane_resistance_mohm: Decimal
    rest_mv: Decimal
    reset_mv: Decimal
    threshold_mv: Decimal
    requested_step_pa: Decimal
    maximum_accepted_pa: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_stable_id(self.independent_unit_id)
        for name in (
            "timestep_ms",
            "baseline_ms",
            "horizon_ms",
            "membrane_tau_ms",
            "membrane_resistance_mohm",
            "maximum_accepted_pa",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("rest_mv", "reset_mv", "threshold_mv", "requested_step_pa"):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.reset_mv != self.rest_mv
            or self.threshold_mv <= self.rest_mv
            or self.requested_step_pa <= 0
            or self.baseline_ms + self.horizon_ms > Decimal(1000)
        ):
            raise ValueError("Brian2 LIF input changes reset, threshold, grid or bounded action")
        self.operational_admission()

    def operational_admission(self) -> NativeTimeGridAdmission:
        return admit_native_time_grid(
            timestep=self.timestep_ms,
            intervals=(self.baseline_ms, self.horizon_ms),
            label="Brian2",
            require_integral_intervals=True,
            sample_offset=0,
            estimated_bytes_per_sample=BRIAN2_ESTIMATED_BYTES_PER_UPDATE,
            maximum_estimated_bytes=BRIAN2_MAX_ESTIMATED_HISTORY_BYTES,
        )
