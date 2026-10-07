"""Native reactor observations and delivery evidence, without a safety verdict."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)


UPSTREAM_COMMIT = "749bc764b667502d47d2d99de988ee6e089b6aa4"
PLANT_SHA256 = "562a486d20597ebb49c74c3066933fa9762edbd2c4c5f20d06a78f300360c5b7"
PARAMS_SHA256 = "ef0d043f4d1d64f0b57f146cd9fc2da4d03f7f9aca56ee085a7df3a9e40e911c"
PUBLIC_SCENARIOS_SHA256 = "435c79e1ab1a5aed446997fa95f1a09dd636e1c940cee83cd2b18a77ca655016"


@dataclass(frozen=True, slots=True)
class ReactorMeasurement(CanonicalRecord):
    """Exactly the upstream callback's visible information; dose is not delayed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-measurement'
    time_s: Decimal
    sample_dt_s: Decimal
    t_reactor_k: Decimal
    t_jacket_k: Decimal
    dosed_kg: Decimal

    def __post_init__(self) -> None:
        for name in ("time_s", "sample_dt_s", "t_reactor_k", "t_jacket_k", "dosed_kg"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.sample_dt_s <= 0:
            raise ValueError("reactor sample duration must be positive")


@dataclass(frozen=True, slots=True)
class ReactorCommand(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-command'
    decision_id: str
    time_s: Decimal
    feed_kg_s: Decimal
    jacket_k: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_decimal(self.time_s, field_name="time_s", minimum=Decimal(0))
        # Bounds are applied by the native actuator, not silently by this adapter.
        validate_decimal(self.feed_kg_s, field_name="feed_kg_s")
        validate_decimal(self.jacket_k, field_name="jacket_k")


@dataclass(frozen=True, slots=True)
class ReactorExposure(CanonicalRecord):
    """Inputs actually consumed by one completed native RK4 update."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-exposure'
    time_s: Decimal
    duration_s: Decimal
    feed_kg_s: Decimal
    jacket_k: Decimal

    def __post_init__(self) -> None:
        for name in ("time_s", "duration_s", "feed_kg_s", "jacket_k"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.duration_s <= 0:
            raise ValueError("reactor exposure duration must be positive")


@dataclass(frozen=True, slots=True)
class ReactorDelivery(CanonicalRecord):
    """One completed sample, with native targets, actuator settings and exposures."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-delivery'
    command: ReactorCommand
    accepted_feed_kg_s: Decimal
    accepted_jacket_k: Decimal
    applied_feed_kg_s: Decimal
    applied_jacket_k: Decimal
    exposures: tuple[ReactorExposure, ...]
    next_measurement: ReactorMeasurement

    def __post_init__(self) -> None:
        for name in (
            "accepted_feed_kg_s",
            "accepted_jacket_k",
            "applied_feed_kg_s",
            "applied_jacket_k",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not self.exposures or len(self.exposures) > 20:
            raise ValueError("bounded reactor delivery requires 1..20 observed native updates")
        clock = self.command.time_s
        for exposure in self.exposures:
            if exposure.time_s != clock:
                raise ValueError("native exposure clocks have a gap or overlap")
            clock += exposure.duration_s
        if (
            clock != self.next_measurement.time_s
            or clock - self.command.time_s != self.next_measurement.sample_dt_s
        ):
            raise ValueError("reactor delivery did not complete exactly one native sample")


@dataclass(frozen=True, slots=True)
class ReactorFinalDelivery(CanonicalRecord):
    """The last native interval, which has no following measurement callback."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-final-delivery'
    command: ReactorCommand
    accepted_feed_kg_s: Decimal
    accepted_jacket_k: Decimal
    applied_feed_kg_s: Decimal
    applied_jacket_k: Decimal
    exposures: tuple[ReactorExposure, ...]
    terminal_time_s: Decimal

    def __post_init__(self) -> None:
        for name in (
            "accepted_feed_kg_s",
            "accepted_jacket_k",
            "applied_feed_kg_s",
            "applied_jacket_k",
            "terminal_time_s",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.command.time_s != Decimal(28790) or self.terminal_time_s != Decimal(28800):
            raise ValueError("final delivery requires the exact last full-batch interval")
        if not self.exposures or len(self.exposures) > 20:
            raise ValueError("final delivery requires observed native updates")
        clock = self.command.time_s
        for exposure in self.exposures:
            if exposure.time_s != clock:
                raise ValueError("final native exposure clocks have a gap or overlap")
            clock += exposure.duration_s
        if clock != self.terminal_time_s:
            raise ValueError("final delivery has not reached the native batch horizon")
