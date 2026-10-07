"""Exact affine map from a root's native callback to the local ten-second receiver."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

LOCAL_CLOCK = "reactor-local-ten-second-clock"
LOCAL_FRAME = "reactor-local-ten-second-window"
NATIVE_CLOCK = "reactor-clock"


@dataclass(frozen=True, slots=True)
class ReactorLocalClockMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-local-clock-map'

    map_id: str
    root: str
    callback: int
    native_start_s: D
    native_clock_id: str = NATIVE_CLOCK
    local_clock_id: str = LOCAL_CLOCK
    local_frame_id: str = LOCAL_FRAME
    duration_s: D = D(10)

    def __post_init__(self) -> None:
        if (
            self.map_id != f"{self.root}.callback-{self.callback:04d}.local-clock"
            or not 0 < self.callback < 2880
            or self.native_start_s != D(10 * self.callback)
            or self.native_clock_id != NATIVE_CLOCK
            or self.local_clock_id != LOCAL_CLOCK
            or self.local_frame_id != LOCAL_FRAME
            or self.duration_s != D(10)
        ):
            raise ValueError("reactor local clock changes its exact native callback affine map")

    def native_time(self, local_s: D) -> D:
        if not D(0) <= local_s <= self.duration_s:
            raise ValueError("reactor local readout lies outside its ten-second window")
        return self.native_start_s + local_s
