"Exact current Gym-TORAX action words and Gym--TORAX native schedules.\n\nThe constructors in this module are pure source translations.  They preserve\nthe retained preparation schedule while giving the four Gym-TORAX words fresh\ncurrent-schema identities and occurrence-safe four-stage delivery records.\nThey do not qualify observed delivery or grant source/execution authority.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus, decode_action_word
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.time import (
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
)

from .diagnostic_contracts import GYM_TORAX_HORIZON_REQUESTS, GymToraxNativeActionRow, GymToraxNativeAction, GymToraxNativeSchedule


GYM_TORAX_FUTURE_IP_ACTION_WORD_ID = 'action-word.tokamak-control.future-ip'
GYM_TORAX_LOWER_IP_ACTION_WORD_ID = 'action-word.tokamak-control.lower-ip'
GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID = 'action-word.tokamak-control.native-hold'
GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID = 'action-word.tokamak-control.wrong-sign-ip'

GYM_TORAX_ACTION_WORD_IDS = (
    GYM_TORAX_FUTURE_IP_ACTION_WORD_ID,
    GYM_TORAX_LOWER_IP_ACTION_WORD_ID,
    GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID,
    GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID,
)

_REQUEST_CLOCK_ID = 'clock.tokamak-control.request-index'
_REQUEST_TIME_UNIT = "request"
_REQUEST_FRAME = 'frame.tokamak-control.request-index'
_IP_QUANTITY_ID = 'quantity.tokamak-control.ip-target'
_IP_NATIVE_UNIT = "A"
_IP_ACTION_FRAME = 'frame.tokamak-control.absolute-ip-target'
_IP_NATIVE_DIRECTION = 'direction.tokamak-control.absolute-ip-target'
_CLOCK_PAYLOAD = b'{"kind":"identity","version":"1.0.0"}'
_DELIVERY_STAGES = (
    ActionDeliveryStage.REQUESTED,
    ActionDeliveryStage.ACCEPTED,
    ActionDeliveryStage.APPLIED,
    ActionDeliveryStage.REALIZED,
)


@dataclass(frozen=True, slots=True)
class _GymToraxActionWordSpec:
    word_id: str
    slug: str
    event_value_a: Decimal
    request_coordinates: tuple[int, ...]


_WORD_SPECS = (
    _GymToraxActionWordSpec(
        word_id=GYM_TORAX_FUTURE_IP_ACTION_WORD_ID,
        slug="future-ip",
        event_value_a=Decimal("12600000"),
        request_coordinates=(111, 112, 113, 114, 115, 116),
    ),
    _GymToraxActionWordSpec(
        word_id=GYM_TORAX_LOWER_IP_ACTION_WORD_ID,
        slug="lower-ip",
        event_value_a=Decimal("12400000"),
        request_coordinates=(104, 105, 106, 107, 108, 109),
    ),
    _GymToraxActionWordSpec(
        word_id=GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID,
        slug="native-hold",
        event_value_a=Decimal("12500000"),
        request_coordinates=(104, 105, 106, 107, 108, 109),
    ),
    _GymToraxActionWordSpec(
        word_id=GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID,
        slug="wrong-sign-ip",
        event_value_a=Decimal("12600000"),
        request_coordinates=(104, 105, 106, 107, 108, 109),
    ),
)
_WORD_SPEC_BY_ID = {value.word_id: value for value in _WORD_SPECS}

if tuple(value.word_id for value in _WORD_SPECS) != GYM_TORAX_ACTION_WORD_IDS:
    raise AssertionError("Gym-TORAX action-word specs are not in unsigned-UTF-8 ID order")


def gym_torax_identity_request_clock_evaluator() -> ExecutableReference:
    """Return the content-bound evaluator for exact request-clock identity maps."""

    return ExecutableReference(
        reference_id='evaluator.tokamak-control.identity-request-clock',
        capability_key='gym-torax-native.identity-request-clock',
        capability_version="1.0.0",
        evaluator_key="gym-torax-native-identity-request-clock",
        payload=ArtifactIdentity(
            artifact_id='artifact.tokamak-control.identity-request-clock',
            role="evaluator",
            payload_schema=(
                'empirical-lawhood/simulators/gym-torax-native/request-clock-evaluator'
            ),
            sha256=sha256(_CLOCK_PAYLOAD).hexdigest(),
            media_type="application/json",
            size_bytes=len(_CLOCK_PAYLOAD),
        ),
        payload_format=SafePayloadFormat.CANONICAL_JSON,
        input_schema=ClockCoordinate.SCHEMA,
        output_schema=ClockCoordinate.SCHEMA,
        deterministic=True,
    )


def gym_torax_ip_action_channel() -> ActionChannelBinding:
    """Return the one exact shared absolute-Ip channel used by all four words."""

    return ActionChannelBinding(
        binding_id='action-channel.tokamak-control.ip-target',
        port_id="port.gym-torax.ip-target",
        controller_quantity_id=_IP_QUANTITY_ID,
        stages=tuple(
            ActionStageQuantityBinding(
                stage=stage,
                quantity_id=f"{_IP_QUANTITY_ID}.{stage.value.lower()}",
                clock_id=_REQUEST_CLOCK_ID,
                time_unit=_REQUEST_TIME_UNIT,
                clock_coordinate_frame=_REQUEST_FRAME,
                clock_origin=CoordinateOrigin.EPISODE_RELATIVE,
            )
            for stage in _DELIVERY_STAGES
        ),
        native_unit=_IP_NATIVE_UNIT,
        native_action_frame=_IP_ACTION_FRAME,
        native_direction=_IP_NATIVE_DIRECTION,
        support_contract_id='support-contract.tokamak-control.ip-target',
        delivery_contract_id='delivery-contract.tokamak-control.ip-target-exact',
    )


def _word_spec(word_id: str) -> _GymToraxActionWordSpec:
    try:
        return _WORD_SPEC_BY_ID[word_id]
    except (KeyError, TypeError) as error:
        raise ValueError("word_id is outside the exact Gym-TORAX current chart") from error


def _coordinate(request_coordinate: int) -> ClockCoordinate:
    return ClockCoordinate(
        clock_id=_REQUEST_CLOCK_ID,
        coordinate=Decimal(request_coordinate),
        time_unit=_REQUEST_TIME_UNIT,
        coordinate_frame=_REQUEST_FRAME,
        origin=CoordinateOrigin.EPISODE_RELATIVE,
    )


def _identity_transport(
    *,
    transport_id: str,
    request_coordinate: int,
) -> ClockTransport:
    coordinate = Decimal(request_coordinate)
    return ClockTransport(
        transport_id=transport_id,
        source_clock_id=_REQUEST_CLOCK_ID,
        target_clock_id=_REQUEST_CLOCK_ID,
        source_time_unit=_REQUEST_TIME_UNIT,
        target_time_unit=_REQUEST_TIME_UNIT,
        source_coordinate_frame=_REQUEST_FRAME,
        target_coordinate_frame=_REQUEST_FRAME,
        source_origin=CoordinateOrigin.EPISODE_RELATIVE,
        target_origin=CoordinateOrigin.EPISODE_RELATIVE,
        kind=ClockTransportKind.IDENTITY,
        availability=ClockTransportAvailability.AVAILABLE,
        scale=Decimal(1),
        offset=Decimal(0),
        tolerance=Decimal(0),
        valid_source_lower=coordinate,
        valid_source_upper=coordinate,
        monotonicity=ClockTransportMonotonicity.STRICTLY_INCREASING,
        evaluator=gym_torax_identity_request_clock_evaluator(),
        reason_codes=(),
    )


def _occurrence(
    spec: _GymToraxActionWordSpec,
    *,
    request_coordinate: int,
    channel: ActionChannelBinding,
) -> ActionOccurrence:
    request_label = f"{request_coordinate:04d}"
    occurrence_stem = f'occurrence.tokamak-control.{spec.slug}.request-{request_label}'
    transport_stem = f'transport.tokamak-control.{spec.slug}.request-{request_label}'

    def event(stage: ActionDeliveryStage) -> ActionStageEvent:
        return ActionStageEvent(
            stage=stage,
            quantity_id=f"{_IP_QUANTITY_ID}.{stage.value.lower()}",
            value=spec.event_value_a,
            native_unit=_IP_NATIVE_UNIT,
            native_action_frame=_IP_ACTION_FRAME,
            native_direction=_IP_NATIVE_DIRECTION,
            coordinate=_coordinate(request_coordinate),
        )

    return ActionOccurrence(
        occurrence_id=occurrence_stem,
        channel=channel,
        requested=event(ActionDeliveryStage.REQUESTED),
        accepted=event(ActionDeliveryStage.ACCEPTED),
        applied=event(ActionDeliveryStage.APPLIED),
        realized=event(ActionDeliveryStage.REALIZED),
        requested_to_accepted=_identity_transport(
            transport_id=f"{transport_stem}.requested-to-accepted",
            request_coordinate=request_coordinate,
        ),
        accepted_to_applied=_identity_transport(
            transport_id=f"{transport_stem}.accepted-to-applied",
            request_coordinate=request_coordinate,
        ),
        applied_to_realized=_identity_transport(
            transport_id=f"{transport_stem}.applied-to-realized",
            request_coordinate=request_coordinate,
        ),
        duration=Decimal(1),
        duration_unit="s",
    )


def build_gym_torax_action_word(word_id: str) -> OccurrenceActionWord:
    """Build one exact supported six-occurrence current Gym-TORAX action word."""

    spec = _word_spec(word_id)
    channel = gym_torax_ip_action_channel()
    occurrences = tuple(
        _occurrence(spec, request_coordinate=request, channel=channel)
        for request in spec.request_coordinates
    )
    groups = tuple(
        ActionOccurrenceGroup(
            group_id=f'group.tokamak-control.{spec.slug}.request-{request:04d}',
            members=(
                ActionOccurrenceOrder(
                    occurrence_id=occurrence.occurrence_id,
                    transport=_identity_transport(
                        transport_id=(
                            f'transport.tokamak-control.{spec.slug}.request-{request:04d}.'
                            "applied-to-ordering"
                        ),
                        request_coordinate=request,
                    ),
                ),
            ),
        )
        for request, occurrence in zip(
            spec.request_coordinates,
            occurrences,
            strict=True,
        )
    )
    prefix_support_ids = (
        f'support-prefix.tokamak-control.{spec.slug}.identity',
        *(
            f'support-prefix.tokamak-control.{spec.slug}.through-request-{request:04d}'
            for request in spec.request_coordinates
        ),
    )
    return OccurrenceActionWord(
        word_id=spec.word_id,
        mode=ActionWordMode.SEQUENTIAL,
        occurrences=occurrences,
        groups=groups,
        ordering_clock_id=_REQUEST_CLOCK_ID,
        ordering_time_unit=_REQUEST_TIME_UNIT,
        ordering_coordinate_frame=_REQUEST_FRAME,
        ordering_origin=CoordinateOrigin.EPISODE_RELATIVE,
        denominator_id='denominator.tokamak-control.gym-torax-prepared',
        retained_history_id='history.tokamak-control.preaction-through-state-0104',
        receiver_id='receiver.tokamak-control.q-fusion-phase-mean-0105-0110',
        horizon_id='horizon.tokamak-control.states-0001-0120',
        prefix_support_ids=prefix_support_ids,
        support_status=ActionWordSupportStatus.SUPPORTED,
        reason_codes=(),
    )


def build_gym_torax_action_word_chart() -> tuple[OccurrenceActionWord, ...]:
    """Build the exact four-word chart in unsigned-UTF-8 word-ID order."""

    return tuple(build_gym_torax_action_word(word_id) for word_id in GYM_TORAX_ACTION_WORD_IDS)


def decode_gym_torax_action_word(payload: bytes) -> OccurrenceActionWord:
    """Decode a current ActionWord and require exact Gym-TORAX chart membership."""

    action_word = decode_action_word(payload)
    expected = build_gym_torax_action_word(action_word.word_id)
    if action_word != expected:
        raise ValueError("decoded action word differs from the exact Gym-TORAX current chart")
    return action_word


def build_gym_torax_retained_baseline_action(
    request_clock: int,
) -> GymToraxNativeAction:
    "Reconstruct the exact 120-request Ip/NBI/ECRH preparation schedule."

    if (
        not isinstance(request_clock, int)
        or isinstance(request_clock, bool)
        or not 0 <= request_clock < GYM_TORAX_HORIZON_REQUESTS
    ):
        raise ValueError("Gym--TORAX baseline request clock is outside 0--119")
    ip_a = (
        Decimal("3000000")
        + Decimal(request_clock + 1) * (Decimal("12500000") - Decimal("3000000")) / Decimal(100)
        if request_clock < 99
        else Decimal("12500000")
    )
    return GymToraxNativeAction(
        ip_a=ip_a,
        nbi_power_w=Decimal("33000000") if request_clock >= 99 else Decimal(0),
        nbi_location=Decimal("0.25"),
        nbi_width=Decimal("0.25"),
        ecrh_power_w=Decimal("20000000") if request_clock >= 30 else Decimal(0),
        ecrh_location=Decimal("0.35"),
        ecrh_width=Decimal("0.05"),
    )


def build_gym_torax_native_schedule(action_word: OccurrenceActionWord) -> GymToraxNativeSchedule:
    """Build one exact 120-row native schedule for an authenticated Gym-TORAX word."""

    expected_word = build_gym_torax_action_word(action_word.word_id)
    if action_word != expected_word:
        raise ValueError("native schedule requires an exact supported Gym-TORAX action word")
    spec = _word_spec(action_word.word_id)
    occurrence_by_request = {
        int(occurrence.requested.coordinate.coordinate): occurrence.occurrence_id
        for occurrence in action_word.occurrences
    }
    rows = []
    for request_clock in range(GYM_TORAX_HORIZON_REQUESTS):
        baseline = build_gym_torax_retained_baseline_action(request_clock)
        controlled_occurrence_id = occurrence_by_request.get(request_clock)
        action = baseline
        if controlled_occurrence_id is not None:
            action = GymToraxNativeAction(
                ip_a=spec.event_value_a,
                nbi_power_w=baseline.nbi_power_w,
                nbi_location=baseline.nbi_location,
                nbi_width=baseline.nbi_width,
                ecrh_power_w=baseline.ecrh_power_w,
                ecrh_location=baseline.ecrh_location,
                ecrh_width=baseline.ecrh_width,
            )
        rows.append(
            GymToraxNativeActionRow(
                row_id=(f'native-action-row.tokamak-control.{spec.slug}.request-{request_clock:04d}'),
                request_clock=request_clock,
                action=action,
                controlled_occurrence_id=controlled_occurrence_id,
            )
        )
    return GymToraxNativeSchedule(
        schedule_id=f'native-schedule.tokamak-control.{spec.slug}.requests-0000-0119',
        action_word=action_word,
        rows=tuple(rows),
    )


def build_gym_torax_native_schedule_chart() -> tuple[GymToraxNativeSchedule, ...]:
    """Build the four exact schedules in current action-word ID order."""

    return tuple(
        build_gym_torax_native_schedule(action_word)
        for action_word in build_gym_torax_action_word_chart()
    )


__all__ = [
    'GYM_TORAX_ACTION_WORD_IDS',
    'GYM_TORAX_FUTURE_IP_ACTION_WORD_ID',
    'GYM_TORAX_LOWER_IP_ACTION_WORD_ID',
    'GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID',
    'GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID',
    'build_gym_torax_action_word_chart',
    'build_gym_torax_action_word',
    'build_gym_torax_retained_baseline_action',
    'build_gym_torax_native_schedule_chart',
    'build_gym_torax_native_schedule',
    'decode_gym_torax_action_word',
    'gym_torax_identity_request_clock_evaluator',
    'gym_torax_ip_action_channel',
]
