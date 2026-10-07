"""Lossless command-to-substep map into the ordinary action-word/delivery owner.

Twenty occurrences represent two simultaneous channels over ten native steps,
not twenty controller requests. Requested/accepted events retain the original
callback clock; exact affine transports locate each held applied subinterval.
The native command and full native delivery remain separate retained records.
"""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, Protocol, cast
from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus, ObservedActionOccurrence
from empirical_lawhood.kernel.control import ScientificCommitmentKind, OperationalDeliveryState
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import (
    ClockCoordinate,
    ClockTransport,
    ClockTransportKind,
    ClockTransportAvailability,
    ClockTransportMonotonicity,
    CoordinateOrigin,
)
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.runtime.controller_runtime import DeliveryPortResult
from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Projection
from empirical_lawhood.adapters.methods.reactor_causal_response.science import CLOCK, SUPPORT
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand, ReactorDelivery, ReactorFinalDelivery

FRAME = "reactor-episode-time"
ORIGIN = CoordinateOrigin.EPISODE_RELATIVE
_CLOCK_RECIPE = EmpiricalRecipe()


def coordinate(t: D) -> ClockCoordinate:
    return ClockCoordinate(CLOCK, t, "s", FRAME, ORIGIN)


def clock_transport(key: str, offset: D = D(0)) -> ClockTransport:
    # Keep the immutable recipe so its canonical encoding is reused across the
    # many substep transports. Each transport still binds its own key/offset.
    recipe = _CLOCK_RECIPE
    parameters = ArtifactIdentity(
        recipe.config_id,
        "clock-transport-parameters",
        recipe.SCHEMA,
        recipe.fingerprint(),
        "application/json",
        len(recipe.canonical_bytes()),
    )
    reference = ExecutableReference(
        "reactor-exact-substep-clock",
        "kernel.clock-transport",
        "1.0.0",
        "exact-affine-clock-transport",
        parameters,
        SafePayloadFormat.CANONICAL_JSON,
        ClockCoordinate.SCHEMA,
        ClockCoordinate.SCHEMA,
        True,
    )
    return ClockTransport(
        key,
        CLOCK,
        CLOCK,
        "s",
        "s",
        FRAME,
        FRAME,
        ORIGIN,
        ORIGIN,
        ClockTransportKind.IDENTITY if offset == 0 else ClockTransportKind.AFFINE_EXACT,
        ClockTransportAvailability.AVAILABLE,
        D(1),
        offset,
        D(0),
        D(0),
        D(28800),
        ClockTransportMonotonicity.STRICTLY_INCREASING,
        reference,
        (),
    )


def exposure_word(
    command: ReactorCommand,
    accepted: tuple[D, D],
    applied: tuple[D, D],
    exposure: tuple[tuple[D, D, D, D], ...],
    *,
    history_id: str,
    horizon_id: str,
    receiver_id: str = "reactor-peak-temperature",
) -> OccurrenceActionWord:
    if (
        len(exposure) != 10
        or tuple(e[0] for e in exposure) != tuple(command.time_s + i for i in range(10))
        or any(e[1] != 1 for e in exposure)
    ):
        raise ValueError("owner word requires all ten nominal substeps")
    # Lossless run-length encoding of equal held inputs. The compatibility
    # record retains all ten native substeps; a run never averages values.
    # Expanding each run onto those substep clocks reproduces the full profile.
    runs: list[tuple[D, D, D, D]] = []
    for time, dt, feed, jacket in exposure:
        if runs and runs[-1][0] + runs[-1][1] == time and runs[-1][2:] == (feed, jacket):
            start, duration, _, _ = runs[-1]
            runs[-1] = (start, duration + dt, feed, jacket)
        else:
            runs.append((time, dt, feed, jacket))
    occurrences, groups = [], []
    for step, (time, dt, feed, jacket) in enumerate(runs):
        orders = []
        for channel, unit, index in (
            ("reactor-feed", "kg/s", 0),
            ("reactor-jacket-command", "K", 1),
        ):
            key = f"{command.decision_id}.{step:02d}.{channel}"
            binding = ActionChannelBinding(
                f"channel.{channel}",
                f"port.{channel}",
                channel,
                tuple(
                    ActionStageQuantityBinding(
                        s, f"{channel}-{s.value.lower()}", CLOCK, "s", FRAME, ORIGIN
                    )
                    for s in ActionDeliveryStage
                ),
                unit,
                "reactor-native-actuator",
                "increasing-native-command",
                SUPPORT,
                "native-reactor-delivery",
            )
            values = (
                (command.feed_kg_s, command.jacket_k)[index],
                accepted[index],
                applied[index],
                (feed, jacket)[index],
            )
            events = tuple(
                ActionStageEvent(
                    s,
                    f"{channel}-{s.value.lower()}",
                    value,
                    unit,
                    "reactor-native-actuator",
                    "increasing-native-command",
                    coordinate(
                        command.time_s
                        if s in (ActionDeliveryStage.REQUESTED, ActionDeliveryStage.ACCEPTED)
                        else time
                    ),
                )
                for s, value in zip(ActionDeliveryStage, values, strict=True)
            )
            occurrence = ActionOccurrence(
                key,
                binding,
                events[0],
                events[1],
                events[2],
                events[3],
                clock_transport(f"{key}.request"),
                clock_transport(f"{key}.held", time - command.time_s),
                clock_transport(f"{key}.realized"),
                dt,
                "s",
            )
            occurrences.append(occurrence)
            orders.append(ActionOccurrenceOrder(key, clock_transport(f"{key}.order")))
        groups.append(
            ActionOccurrenceGroup(f"{command.decision_id}.group.{step:02d}", tuple(orders))
        )
    return OccurrenceActionWord(
        f"{command.decision_id}.word",
        ActionWordMode.SIMULTANEOUS if len(groups) == 1 else ActionWordMode.MIXED,
        tuple(occurrences),
        tuple(groups),
        CLOCK,
        "s",
        FRAME,
        ORIGIN,
        SUPPORT,
        history_id,
        receiver_id,
        horizon_id,
        tuple(f"{command.decision_id}.prefix.{i:02d}" for i in range(len(occurrences) + 1)),
        ActionWordSupportStatus.SUPPORTED,
        (),
    )


@dataclass(frozen=True, slots=True)
class ReactorExposureWordMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-causal-response/reactor-exposure-word-map'
    command: ReactorCommand
    accepted: tuple[D, D]
    applied: tuple[D, D]
    exposure: tuple[tuple[D, D, D, D], ...]
    word: OccurrenceActionWord

    def __post_init__(self) -> None:
        if self.word != exposure_word(
            self.command,
            self.accepted,
            self.applied,
            self.exposure,
            history_id=self.word.retained_history_id,
            horizon_id=self.word.horizon_id,
            receiver_id=self.word.receiver_id,
        ):
            raise ValueError(
                "substep compatibility map changes command, stages, clocks or exposure"
            )

    @classmethod
    def from_projection(
        cls, projection: Projection, *, decision_id: str, history_id: str, horizon_id: str
    ) -> 'ReactorExposureWordMap':
        def decimal(x: float) -> D:
            return D(repr(float(x)))

        command = ReactorCommand(
            decision_id,
            decimal(projection.exposure[0][0]),
            decimal(projection.requested[0]),
            decimal(projection.requested[1]),
        )
        accepted = (decimal(projection.accepted[0]), decimal(projection.accepted[1]))
        applied = (decimal(projection.applied[0]), decimal(projection.applied[1]))
        exposure = tuple(
            cast(tuple[D, D, D, D], tuple(decimal(x) for x in e)) for e in projection.exposure
        )
        return cls(
            command,
            accepted,
            applied,
            exposure,
            exposure_word(
                command, accepted, applied, exposure, history_id=history_id, horizon_id=horizon_id
            ),
        )


class BatchDeliverySession(Protocol):
    def advance(self, command: ReactorCommand) -> ReactorDelivery | ReactorFinalDelivery: ...


@dataclass
class ReactorProfileDeliveryPort:
    implementation_binding: ImplementationBinding
    session: BatchDeliverySession
    mapping: ReactorExposureWordMap
    last_delivery: ReactorDelivery | ReactorFinalDelivery | None = None

    def deliver(
        self, *, action_word: OccurrenceActionWord, commitment_kind: ScientificCommitmentKind
    ) -> DeliveryPortResult:
        if (
            self.implementation_binding.role is not ImplementationRole.DELIVERY
            or commitment_kind is not ScientificCommitmentKind.ACTION
            or action_word != self.mapping.word
        ):
            raise ValueError("native delivery requires the exact committed profile and ACTION")
        try:
            raw = self.session.advance(self.mapping.command)
        except RuntimeError as error:
            if str(error) not in {
                "reactor delivery timed out; no completion evidence",
                "native reactor failed; no completion evidence",
            }:
                raise
            return DeliveryPortResult(
                f"delivery.{self.mapping.command.decision_id}",
                (),
                OperationalDeliveryState.TERMINATED,
            )
        self.last_delivery = raw
        if raw.command != self.mapping.command:
            raise ValueError("native session substituted command")
        if isinstance(raw, ReactorFinalDelivery) != (raw.command.time_s == D(28790)):
            raise ValueError("last native interval requires terminal delivery without a callback")
        actual = exposure_word(
            raw.command,
            (raw.accepted_feed_kg_s, raw.accepted_jacket_k),
            (raw.applied_feed_kg_s, raw.applied_jacket_k),
            tuple((e.time_s, e.duration_s, e.feed_kg_s, e.jacket_k) for e in raw.exposures),
            history_id=action_word.retained_history_id,
            horizon_id=action_word.horizon_id,
        )
        observed = tuple(
            ObservedActionOccurrence(
                f"observed.{o.occurrence_id}",
                o.occurrence_id,
                o.requested,
                o.accepted,
                o.applied,
                o.realized,
                (),
            )
            for o in actual.occurrences
        )
        return DeliveryPortResult(
            f"delivery.{raw.command.decision_id}",
            observed,
            None if actual == action_word else OperationalDeliveryState.TERMINATED,
        )
