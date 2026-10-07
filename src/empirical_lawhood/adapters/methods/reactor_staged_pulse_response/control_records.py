"""Frozen readout/consumer contracts, including the committed saturation parameter."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.finite_response_geometry import FiniteReadoutKind, FiniteResponseCoordinate
from .config import ARMS, MENUS, SEQUENCES, ClassicalDesign, Request, assignment
from .law_terminal import ClassicalLaws
from .science import LOCAL_CLOCK, LOCAL_FRAME


@dataclass(frozen=True, slots=True)
class ClassicalReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-readout'
    policy: str
    request: Request
    stage_kind: str
    laws: ObjectIdentity
    mapping: str = "raw-or-min-raw-committed-q-after-raw-numerical-tests"

    def __post_init__(self) -> None:
        if (
            self.policy not in (*ARMS, *MENUS, *SEQUENCES)
            or self.stage_kind not in ("local", "first", "baseline", "joint")
            or self.laws.object_schema != ClassicalLaws.SCHEMA
            or self.mapping != "raw-or-min-raw-committed-q-after-raw-numerical-tests"
        ):
            raise ValueError("readout changes its fixed receiver map or owner-law source")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.policy.lower()}.{self.request.request_id.lower()}.{self.stage_kind}.readout"

    @property
    def guard_s(self) -> int:
        return 240 if self.stage_kind == "baseline" else 120

    @property
    def receivers(self) -> tuple[str, ...]:
        return (
            ("c1", "c2", "g", "s", "s2")
            if self.stage_kind == "joint"
            else ("c1", "s")
            if self.stage_kind in ("first", "baseline")
            else ("c", "s")
        )


@dataclass(frozen=True, slots=True)
class ClassicalClockMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-clock-map'
    root: str
    callback: int
    first_callback: int
    readout: ClassicalReadout

    def __post_init__(self) -> None:
        assignment(self.root)
        if not 60 <= self.first_callback <= 600 or self.callback != self.first_callback + (
            12 if self.readout.stage_kind == "joint" else 0
        ):
            raise ValueError("readout changes the native first/second cutoff map")

    @property
    def map_id(self) -> str:
        return f"{self.root}.{self.readout.record_id}.clock"

    def native_time(self, local_s: D) -> D:
        if not 0 <= local_s <= self.readout.guard_s:
            raise ValueError("readout is outside its native guard")
        return self.callback * 10 + local_s


def response_coordinates(readout: ClassicalReadout) -> tuple[FiniteResponseCoordinate, ...]:
    operator = ObjectIdentity.from_record(readout.record_id, readout)
    return tuple(
        FiniteResponseCoordinate(
            key,
            key,
            key,
            operator,
            FiniteReadoutKind.WINDOW_MAXIMUM if key.startswith("s") else FiniteReadoutKind.ENDPOINT,
            "K",
            LOCAL_FRAME,
            LOCAL_CLOCK,
            D(0) if key.startswith("s") else D(readout.guard_s),
            D(readout.guard_s),
        )
        for key in readout.receivers
    )


def consumer(laws: ClassicalLaws, readout: ClassicalReadout) -> FrozenFeedbackConsumer:
    if readout.laws != ObjectIdentity.from_record(laws.record_id, laws):
        raise ValueError("consumer substitutes its qualified law census")
    design = ClassicalDesign()
    arm = (
        readout.policy
        if readout.policy in ARMS
        else "EL_SERVICE_MENU"
        if readout.policy in MENUS
        else "EL_SEQUENCE_RELATION"
    )
    artifacts = tuple(
        sorted(
            (
                r.candidate.payload_publication.artifact
                for r in laws.rows
                if r.payload.recipe.arm == arm
                and r.payload.recipe.bound.coordinate.kind == readout.stage_kind
                and r.payload.recipe.recipe_id in laws.qualified
                and (
                    readout.stage_kind != "local"
                    or (
                        r.payload.recipe.bound.coordinate.context,
                        r.payload.recipe.bound.coordinate.horizon_s,
                    )
                    == (readout.request.context, readout.request.horizon_s)
                )
            ),
            key=lambda a: a.artifact_id,
        )
    )
    return FrozenFeedbackConsumer(
        f"{readout.record_id}.consumer",
        ObjectIdentity.from_record(design.config_id, design),
        artifacts,
        ObjectIdentity.from_record(laws.operands.record_id, laws.operands),
        ObjectIdentity.from_record(laws.record_id, laws),
        ObjectIdentity.from_record(readout.record_id, readout),
    )
