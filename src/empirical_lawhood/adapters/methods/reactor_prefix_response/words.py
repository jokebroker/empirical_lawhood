"""Finite native action chart, with shared occurrence identities across arms."""

from decimal import Decimal

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.time import (
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
)
from .design import CLOCK, FRAME, HISTORY, PREFIX, RECEIVER, SUPPORT, ReactorScienceDesign


def action_words(design: ReactorScienceDesign) -> tuple[OccurrenceActionWord, ...]:
    """Requested chart descriptors; actual stages must be checked before use."""
    origin = CoordinateOrigin.EPISODE_RELATIVE
    parameters = ArtifactIdentity(
        design.config_id,
        "clock-transport-parameters",
        design.SCHEMA,
        design.fingerprint(),
        "application/vnd.empirical-lawhood.canonical+json",
        len(design.canonical_bytes()),
    )
    evaluator = ExecutableReference(
        f"{PREFIX}.identity-clock",
        "kernel.clock-transport",
        "1.0.0",
        "identity-clock-transport",
        parameters,
        SafePayloadFormat.CANONICAL_JSON,
        ClockCoordinate.SCHEMA,
        ClockCoordinate.SCHEMA,
        True,
    )

    def transport(key: str) -> ClockTransport:
        return ClockTransport(
            key,
            CLOCK,
            CLOCK,
            "s",
            "s",
            FRAME,
            FRAME,
            origin,
            origin,
            ClockTransportKind.IDENTITY,
            ClockTransportAvailability.AVAILABLE,
            Decimal(1),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(20),
            ClockTransportMonotonicity.STRICTLY_INCREASING,
            evaluator,
            (),
        )

    words = []
    for arm, first_jacket in (("comparator", Decimal(316)), ("pulse", Decimal(315))):
        occurrences, groups = [], []
        for step in range(2):
            orders = []
            for channel, unit, value in (
                ("reactor-feed", "kg/s", Decimal(0)),
                ("reactor-jacket-command", "K", first_jacket if step == 0 else Decimal(316)),
            ):
                key = f"{channel}.{step}"
                binding = ActionChannelBinding(
                    f"channel.{channel}",
                    f"port.{channel}",
                    channel,
                    tuple(
                        ActionStageQuantityBinding(
                            stage, f"{channel}-{stage.value.lower()}", CLOCK, "s", FRAME, origin
                        )
                        for stage in ActionDeliveryStage
                    ),
                    unit,
                    "reactor-native-actuator",
                    "increasing-native-command",
                    SUPPORT,
                    "native-reactor-delivery",
                )
                events = tuple(
                    ActionStageEvent(
                        stage,
                        f"{channel}-{stage.value.lower()}",
                        value,
                        unit,
                        "reactor-native-actuator",
                        "increasing-native-command",
                        ClockCoordinate(CLOCK, Decimal(10 * step), "s", FRAME, origin),
                    )
                    for stage in ActionDeliveryStage
                )
                occurrence = ActionOccurrence(
                    f"occurrence.{key}",
                    binding,
                    events[0],
                    events[1],
                    events[2],
                    events[3],
                    transport(f"transport.{key}.requested-accepted"),
                    transport(f"transport.{key}.accepted-applied"),
                    transport(f"transport.{key}.applied-realized"),
                    Decimal(10),
                    "s",
                )
                occurrences.append(occurrence)
                orders.append(
                    ActionOccurrenceOrder(
                        occurrence.occurrence_id, transport(f"transport.{key}.ordering")
                    )
                )
            groups.append(ActionOccurrenceGroup(f"group.reactor.{step}", tuple(orders)))
        words.append(
            OccurrenceActionWord(
                f"word.reactor.{arm}",
                ActionWordMode.MIXED,
                tuple(occurrences),
                tuple(groups),
                CLOCK,
                "s",
                FRAME,
                origin,
                SUPPORT,
                HISTORY,
                RECEIVER,
                f"{PREFIX}.horizon",
                tuple(f"prefix.reactor.{i}" for i in range(5)),
                ActionWordSupportStatus.SUPPORTED,
                (),
            )
        )
    return tuple(words)
