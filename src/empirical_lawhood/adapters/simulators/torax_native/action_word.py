"""Substrate-general native TORAX action-word translation."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
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

from .contracts import NativeToraxAction


_CLOCK_PAYLOAD = b'{"kind":"identity","version":"1.0.0"}'


def native_torax_clock_evaluator() -> ExecutableReference:
    return ExecutableReference(
        reference_id="evaluator.torax-native-clock",
        capability_key="torax-native.identity-clock",
        capability_version="1.0.0",
        evaluator_key="torax-native-identity-clock",
        payload=ArtifactIdentity(
            artifact_id="artifact.torax-native-clock",
            role="evaluator",
            payload_schema='empirical-lawhood/simulators/torax-native/clock-evaluator',
            sha256=sha256(_CLOCK_PAYLOAD).hexdigest(),
            media_type="application/json",
            size_bytes=len(_CLOCK_PAYLOAD),
        ),
        payload_format=SafePayloadFormat.CANONICAL_JSON,
        input_schema=ClockCoordinate.SCHEMA,
        output_schema=ClockCoordinate.SCHEMA,
        deterministic=True,
    )


def _coordinate(value: Decimal) -> ClockCoordinate:
    return ClockCoordinate(
        clock_id="torax-episode-clock",
        coordinate=value,
        time_unit="s",
        coordinate_frame="torax-episode-relative-time",
        origin=CoordinateOrigin.EPISODE_RELATIVE,
    )


def _transport(transport_id: str) -> ClockTransport:
    return ClockTransport(
        transport_id=transport_id,
        source_clock_id="torax-episode-clock",
        target_clock_id="torax-episode-clock",
        source_time_unit="s",
        target_time_unit="s",
        source_coordinate_frame="torax-episode-relative-time",
        target_coordinate_frame="torax-episode-relative-time",
        source_origin=CoordinateOrigin.EPISODE_RELATIVE,
        target_origin=CoordinateOrigin.EPISODE_RELATIVE,
        kind=ClockTransportKind.IDENTITY,
        availability=ClockTransportAvailability.AVAILABLE,
        scale=Decimal(1),
        offset=Decimal(0),
        tolerance=Decimal(0),
        valid_source_lower=Decimal(0),
        valid_source_upper=Decimal("0.04"),
        monotonicity=ClockTransportMonotonicity.STRICTLY_INCREASING,
        evaluator=native_torax_clock_evaluator(),
        reason_codes=(),
    )


def build_native_torax_action_word(
    action: NativeToraxAction,
    *,
    preparation_id: str,
) -> OccurrenceActionWord:
    """Translate one exact four-stage native action without losing occurrence identity."""

    word_id = f"word.{preparation_id}.{action.action_label}"
    occurrence_id = f"occurrence.{preparation_id}.{action.action_label}"
    channel = ActionChannelBinding(
        binding_id=f"channel.{preparation_id}.{action.action_label}.proxy-power",
        port_id="generic-heat-total-power",
        controller_quantity_id="torax-proxy-heating-power",
        stages=tuple(
            ActionStageQuantityBinding(
                stage=stage,
                quantity_id=f"torax-proxy-power-{stage.value.lower()}",
                clock_id="torax-episode-clock",
                time_unit="s",
                clock_coordinate_frame="torax-episode-relative-time",
                clock_origin=CoordinateOrigin.EPISODE_RELATIVE,
            )
            for stage in ActionDeliveryStage
        ),
        native_unit="W",
        native_action_frame="torax-generic-heat-source",
        native_direction=f"absolute-{action.action_label}-power",
        support_contract_id="support.torax-native-proxy-chart",
        delivery_contract_id="delivery.torax-direct-stage-exact",
    )

    def event(stage: ActionDeliveryStage) -> ActionStageEvent:
        source = next(value for value in action.stages if value.stage is stage)
        return ActionStageEvent(
            stage=stage,
            quantity_id=f"torax-proxy-power-{stage.value.lower()}",
            value=source.power_w,
            native_unit="W",
            native_action_frame="torax-generic-heat-source",
            native_direction=f"absolute-{action.action_label}-power",
            coordinate=_coordinate(source.coordinate_s),
        )

    occurrence = ActionOccurrence(
        occurrence_id=occurrence_id,
        channel=channel,
        requested=event(ActionDeliveryStage.REQUESTED),
        accepted=event(ActionDeliveryStage.ACCEPTED),
        applied=event(ActionDeliveryStage.APPLIED),
        realized=event(ActionDeliveryStage.REALIZED),
        requested_to_accepted=_transport(
            f"transport.{preparation_id}.{action.action_label}.requested-accepted"
        ),
        accepted_to_applied=_transport(
            f"transport.{preparation_id}.{action.action_label}.accepted-applied"
        ),
        applied_to_realized=_transport(
            f"transport.{preparation_id}.{action.action_label}.applied-realized"
        ),
        duration=action.duration_s,
        duration_unit="s",
    )
    return OccurrenceActionWord(
        word_id=word_id,
        mode=ActionWordMode.SEQUENTIAL,
        occurrences=(occurrence,),
        groups=(
            ActionOccurrenceGroup(
                group_id=f"group.{preparation_id}.{action.action_label}",
                members=(
                    ActionOccurrenceOrder(
                        occurrence_id=occurrence_id,
                        transport=_transport(
                            f"transport.{preparation_id}.{action.action_label}.applied-order"
                        ),
                    ),
                ),
            ),
        ),
        ordering_clock_id="torax-episode-clock",
        ordering_time_unit="s",
        ordering_coordinate_frame="torax-episode-relative-time",
        ordering_origin=CoordinateOrigin.EPISODE_RELATIVE,
        denominator_id=preparation_id,
        retained_history_id="history.torax-preaction-profiles",
        receiver_id="receiver.torax-core-te-and-contrast",
        horizon_id="horizon.torax-40ms",
        prefix_support_ids=(
            f"support-prefix.{preparation_id}.identity",
            f"support-prefix.{preparation_id}.{action.action_label}",
        ),
        support_status=ActionWordSupportStatus.SUPPORTED,
        reason_codes=(),
    )


__all__ = ["build_native_torax_action_word", "native_torax_clock_evaluator"]
