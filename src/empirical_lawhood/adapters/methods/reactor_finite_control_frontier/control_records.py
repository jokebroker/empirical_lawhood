"""Explicit response-window/guard compatibility and the frozen local consumer."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseCoordinate
from .config import CONTEXTS, HORIZONS, FrontierDesign, assignment
from .law_terminal import FrontierLaws
from .science import LOCAL_CLOCK, LOCAL_FRAME, RECEIVERS


@dataclass(frozen=True, slots=True)
class FrontierReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-readout'
    context: str
    response_window_s: int
    delivery_guard_s: int = 120

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.response_window_s not in HORIZONS
            or self.delivery_guard_s != 120
        ):
            raise ValueError("receiver changed its inner response window or full guard")

    @property
    def record_id(self) -> str:
        return f"reactor-finite-control-frontier.{self.context}.t{self.response_window_s:03d}.readout"


@dataclass(frozen=True, slots=True)
class FrontierClockMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-clock-map'
    root: str
    callback: int
    readout: FrontierReadout

    def __post_init__(self) -> None:
        assignment(self.root)
        if type(self.callback) is not int or not 60 <= self.callback <= 2508:
            raise ValueError("local clock is outside the native complete-word window")

    @property
    def map_id(self) -> str:
        return f"{self.root}.{self.readout.context}.t{self.readout.response_window_s:03d}.clock"

    def native_time(self, local_s: D) -> D:
        if not D(0) <= local_s <= 120:
            raise ValueError("local readout is outside the complete guard")
        return D(self.callback * 10) + local_s


def consumer(laws: FrontierLaws, readout: FrontierReadout) -> FrozenFeedbackConsumer:
    artifacts = tuple(
        sorted(
            (
                r.candidate.payload_publication.artifact
                for r in laws.rows
                if r.payload.bound.coordinate.context == readout.context
                and r.payload.bound.coordinate.horizon_s == readout.response_window_s
                and r.payload.bound.coordinate.coordinate_id in laws.qualified
            ),
            key=lambda a: a.artifact_id,
        )
    )
    design = FrontierDesign()
    return FrozenFeedbackConsumer(
        f"{readout.record_id}.consumer",
        ObjectIdentity.from_record(design.config_id, design),
        artifacts,
        laws.source,
        ObjectIdentity.from_record(laws.record_id, laws),
        ObjectIdentity.from_record(readout.record_id, readout),
    )


def response_coordinates(readout: FrontierReadout) -> tuple[FiniteResponseCoordinate, ...]:
    """At guard completion, read S_120 and the explicitly defined inner C_tau.

    C_tau is a scalar post-window statistic, not a temperature observation at
    tau or 120. Its observation-operator identity binds the nested window.
    """
    operator = ObjectIdentity.from_record(readout.record_id, readout)
    return tuple(
        sorted(
            (
                FiniteResponseCoordinate(
                    key,
                    key,
                    key,
                    operator,
                    FiniteReadoutKind.WINDOW_MAXIMUM if i == 0 else FiniteReadoutKind.ENDPOINT,
                    "K",
                    LOCAL_FRAME,
                    LOCAL_CLOCK,
                    D(0) if i == 0 else D(120),
                    D(120),
                )
                for i, key in enumerate(RECEIVERS)
            ),
            key=lambda c: c.coordinate_id,
        )
    )
