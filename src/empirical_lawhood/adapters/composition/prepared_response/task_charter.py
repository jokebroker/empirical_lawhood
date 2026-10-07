"""Outcome-blind finite task charter and measured frame translation.

The task owner uses no fitted predictor or native future. Selection from this
charter belongs to source qualification's separate registered evaluator.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN


@dataclass(frozen=True, slots=True)
class PreparedTaskCharterEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/prepared-response/prepared-task-charter-entry'
    amplitude: Decimal
    horizon_ticks: int
    epsilon: Decimal
    distance_multiple: int
    reference_kind: str = "FREE_DAMPED_PREPARENT_VELOCITY_GAMMA_ONE"
    parent_and_delay_ticks: int = 272
    loss_kind: str = "TWO_PORT_ABSOLUTE_ENDPOINT_INFINITY_NORM"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.amplitude, Decimal)
            or not self.amplitude.is_finite()
            or self.amplitude not in (8, 16)
            or type(self.horizon_ticks) is not int
            or self.horizon_ticks not in (320, 256, 192)
            or not isinstance(self.epsilon, Decimal)
            or not self.epsilon.is_finite()
            or self.epsilon
            not in (Decimal(1) / 32, Decimal(1) / 16, Decimal(3) / 32, Decimal(1) / 8)
            or type(self.distance_multiple) is not int
            or self.distance_multiple not in (4, 6)
            or self.reference_kind != "FREE_DAMPED_PREPARENT_VELOCITY_GAMMA_ONE"
            or type(self.parent_and_delay_ticks) is not int
            or self.parent_and_delay_ticks != 272
            or self.loss_kind != "TWO_PORT_ABSOLUTE_ENDPOINT_INFINITY_NORM"
        ):
            raise ValueError("prepared task changes the finite pre-source qualification charter")

    @property
    def entry_id(self) -> str:
        return f"{CAMPAIGN}.charter.a{int(self.amplitude)}.t{self.horizon_ticks}.e{int(self.epsilon * 32)}-32.d{self.distance_multiple}"

    @property
    def distance(self) -> Decimal:
        return self.epsilon * self.distance_multiple


def prepared_task_charter() -> tuple[PreparedTaskCharterEntry, ...]:
    return tuple(
        PreparedTaskCharterEntry(amplitude, horizon, epsilon, multiple)
        for amplitude in (Decimal(8), Decimal(16))
        for horizon in (320, 256, 192)
        for epsilon in (Decimal(1) / 32, Decimal(1) / 16, Decimal(3) / 32, Decimal(1) / 8)
        for multiple in (4, 6)
    )


def _vector(value: tuple[Decimal, ...]) -> None:
    if len(value) != 2 or any(not isinstance(x, Decimal) or not x.is_finite() for x in value):
        raise ValueError("prepared task requires two finite native Decimal coordinates")


def task_direction(index: int) -> tuple[Decimal, Decimal]:
    if type(index) is not int or index not in range(9):
        raise ValueError("prepared task draw is outside the nine equiprobable requests")
    with localcontext() as context:
        context.prec = 28
        diagonal = Decimal(1) / Decimal(2).sqrt()
        return (
            (Decimal(0), Decimal(0)),
            (Decimal(-1), Decimal(0)),
            (Decimal(1), Decimal(0)),
            (Decimal(0), Decimal(-1)),
            (Decimal(0), Decimal(1)),
            (-diagonal, -diagonal),
            (diagonal, diagonal),
            (-diagonal, diagonal),
            (diagonal, -diagonal),
        )[index]


@dataclass(frozen=True, slots=True)
class PreparedTaskTarget(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/prepared-response/prepared-task-target'
    target_id: str
    charter: PreparedTaskCharterEntry
    root_id: str
    preparent_frame: ObjectIdentity
    preparent_velocity: tuple[Decimal, ...]
    target_draw: int
    absolute_center: tuple[Decimal, ...]
    native_unit: str = "hilbert-schmidt-native"
    reference_frame: str = "FROZEN_PREPARENT_X_PORTS_DISPLACEMENT_FROM_PREPARENT"

    def __post_init__(self) -> None:
        validate_stable_id(self.target_id, field_name="target_id")
        validate_stable_id(self.root_id, field_name="root_id")
        _vector(self.preparent_velocity)
        _vector(self.absolute_center)
        expected = _center(self.charter, self.preparent_velocity, self.target_draw)
        if (
            self.absolute_center != expected
            or self.native_unit != "hilbert-schmidt-native"
            or self.reference_frame != "FROZEN_PREPARENT_X_PORTS_DISPLACEMENT_FROM_PREPARENT"
        ):
            raise ValueError("prepared target was recentered or changes its task convention")


def _center(
    charter: PreparedTaskCharterEntry, velocity: tuple[Decimal, ...], draw: int
) -> tuple[Decimal, ...]:
    _vector(velocity)
    direction = task_direction(draw)
    with localcontext() as context:
        context.prec = 28
        delta = Decimal("0.001") * (charter.parent_and_delay_ticks + charter.horizon_ticks)
        drift_coefficient = 1 - (-delta).exp()
        return tuple(
            v * drift_coefficient + charter.distance * q
            for v, q in zip(velocity, direction, strict=True)
        )


def prepared_task_target(
    *,
    charter: PreparedTaskCharterEntry,
    root_id: str,
    frame: ObjectIdentity,
    velocity: tuple[Decimal, ...],
    draw: int,
) -> PreparedTaskTarget:
    return PreparedTaskTarget(
        f"{root_id}.task-target",
        charter,
        root_id,
        frame,
        velocity,
        draw,
        _center(charter, velocity, draw),
    )


@dataclass(frozen=True, slots=True)
class PreparedTargetTranslation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/prepared-response/prepared-target-translation'
    target: PreparedTaskTarget
    handoff_observation: ObjectIdentity
    observed_handoff_displacement: tuple[Decimal, ...]
    remaining_center: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        _vector(self.observed_handoff_displacement)
        _vector(self.remaining_center)
        if self.remaining_center != _translate(self.target, self.observed_handoff_displacement):
            raise ValueError(
                "target translation must retain the original absolute pre-parent target"
            )


def _translate(target: PreparedTaskTarget, handoff: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec = 28
        return tuple(a - b for a, b in zip(target.absolute_center, handoff, strict=True))


def translate_prepared_target(
    target: PreparedTaskTarget,
    *,
    handoff_observation: ObjectIdentity,
    observed_handoff_displacement: tuple[Decimal, ...],
) -> PreparedTargetTranslation:
    _vector(observed_handoff_displacement)
    return PreparedTargetTranslation(
        target,
        handoff_observation,
        observed_handoff_displacement,
        _translate(target, observed_handoff_displacement),
    )
