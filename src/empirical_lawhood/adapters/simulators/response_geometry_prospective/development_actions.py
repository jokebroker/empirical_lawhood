"""development's expected native pulse chart in the existing action/clock contracts.

These are declared delivery expectations, never observed delivery or admission.
Native segment receipts separately retain acceptance, completed intervals, both
force evaluations, signed impulse and the realized state/checkpoint lineage.
"""

from decimal import Decimal

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.references import ExecutableReference
from empirical_lawhood.kernel.time import (
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import ResponseGeometryNativeForcePulse


DEVELOPMENT_ACTION_KNOTS = (0, 64)
DEVELOPMENT_CLOCK = "response-geometry.native-reference-tick"
DEVELOPMENT_EPISODE_FRAME = "response-geometry.native-episode"
DEVELOPMENT_FORCE_UNIT = "native-hs-force"
DEVELOPMENT_FORCE_FRAME = "response-geometry.frozen-preparent-mode"
DEVELOPMENT_FORCE_DIRECTION = "response-geometry.positive-oriented-mode"


def response_geometry_development_action_word(
    pulse: ResponseGeometryNativeForcePulse,
    *,
    denominator_id: str,
    history_id: str,
    receiver_id: str,
    horizon_id: str,
    clock_evaluator: ExecutableReference,
) -> OccurrenceActionWord:
    """Bind one native short-pulse-response pulse; the receiver horizon includes the 256-tick coast."""
    if pulse.assay != "short-pulse-response":
        raise ValueError("development action chart is qualified only for the short-pulse-response pulse")
    stem = pulse.occurrence_id
    clock = DEVELOPMENT_CLOCK
    frame = DEVELOPMENT_EPISODE_FRAME
    origin = CoordinateOrigin.EPISODE_RELATIVE
    transport = ClockTransport(
        transport_id=f"{stem}.clock",
        source_clock_id=clock,
        target_clock_id=clock,
        source_time_unit="reference-tick",
        target_time_unit="reference-tick",
        source_coordinate_frame=frame,
        target_coordinate_frame=frame,
        source_origin=origin,
        target_origin=origin,
        kind=ClockTransportKind.IDENTITY,
        availability=ClockTransportAvailability.AVAILABLE,
        scale=Decimal(1),
        offset=Decimal(0),
        tolerance=Decimal(0),
        valid_source_lower=Decimal(pulse.invocation_tick),
        valid_source_upper=Decimal(pulse.invocation_tick + 320),
        monotonicity=ClockTransportMonotonicity.STRICTLY_INCREASING,
        evaluator=clock_evaluator,
        reason_codes=(),
    )
    channel = ActionChannelBinding(
        f"{stem}.channel",
        "response-geometry.x-force",
        "response-geometry.x-force-command",
        tuple(
            ActionStageQuantityBinding(
                stage,
                f"response-geometry.x-force.{stage.value.lower()}",
                clock,
                "reference-tick",
                frame,
                origin,
            )
            for stage in ActionDeliveryStage
        ),
        DEVELOPMENT_FORCE_UNIT,
        DEVELOPMENT_FORCE_FRAME,
        DEVELOPMENT_FORCE_DIRECTION,
        "response-geometry.short-pulse-response-native-force-support",
        "response-geometry.both-kicks-completed-interval-delivery",
    )
    occurrences, groups = [], []
    for index, (start, end) in enumerate(zip(DEVELOPMENT_ACTION_KNOTS, DEVELOPMENT_ACTION_KNOTS[1:], strict=False)):
        occurrence_id = f"{stem}.interval.{index}"
        tick = pulse.invocation_tick + start
        coordinate = ClockCoordinate(clock, Decimal(tick), "reference-tick", frame, origin)
        force = Decimal(str(pulse.interval_force(native_step=tick, refinement=1)))
        events = tuple(
            ActionStageEvent(
                stage.stage,
                stage.quantity_id,
                force,
                DEVELOPMENT_FORCE_UNIT,
                DEVELOPMENT_FORCE_FRAME,
                DEVELOPMENT_FORCE_DIRECTION,
                coordinate,
            )
            for stage in channel.stages
        )
        occurrences.append(
            ActionOccurrence(
                occurrence_id,
                channel,
                events[0],
                events[1],
                events[2],
                events[3],
                transport,
                transport,
                transport,
                Decimal(end - start),
                "reference-tick",
            )
        )
        groups.append(
            ActionOccurrenceGroup(
                f"{stem}.group.{index}",
                (ActionOccurrenceOrder(occurrence_id, transport),),
            )
        )
    return OccurrenceActionWord(
        f"{stem}.word",
        ActionWordMode.SEQUENTIAL,
        tuple(occurrences),
        tuple(groups),
        clock,
        "reference-tick",
        frame,
        origin,
        denominator_id,
        history_id,
        receiver_id,
        horizon_id,
        tuple(f"{stem}.support.prefix.{index}" for index in range(2)),
        ActionWordSupportStatus.UNEVALUABLE,
        ("SUPPORT_EXPECTATION_REQUIRES_EVIDENCE",),
    )
