"""Translate observed reactor deliveries into the existing native delivery port."""

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Protocol

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionStageEvent, OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.kernel.time import CoordinateOrigin
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.runtime.controller_runtime import DeliveryPortResult

from .contracts import ReactorCommand, ReactorDelivery


class ReactorDeliverySession(Protocol):
    def deliver(self, command: ReactorCommand) -> ReactorDelivery: ...


@dataclass(frozen=True, slots=True)
class ReactorNativeDeliveryPort:
    """One sample per word; no claimed HOLD fibre or inferred future delivery."""

    implementation_binding: ImplementationBinding
    session: ReactorDeliverySession

    def __post_init__(self) -> None:
        if self.implementation_binding.role is not ImplementationRole.DELIVERY:
            raise ValueError("reactor delivery requires the native-delivery role")

    def deliver(
        self, *, action_word: OccurrenceActionWord, commitment_kind: ScientificCommitmentKind
    ) -> DeliveryPortResult:
        if commitment_kind is not ScientificCommitmentKind.ACTION:
            raise ValueError("reactor bridge has no measured native HOLD declaration")
        occurrences = action_word.occurrences
        by_channel = {o.channel.controller_quantity_id: o for o in occurrences}
        if len(occurrences) != 2 or set(by_channel) != {"reactor-feed", "reactor-jacket-command"}:
            raise ValueError("reactor sample requires exactly its feed and jacket channels")
        feed = by_channel["reactor-feed"]
        jacket = by_channel["reactor-jacket-command"]
        coordinate = feed.requested.coordinate
        if coordinate != jacket.requested.coordinate:
            raise ValueError("reactor sample requests must be simultaneous")
        for occurrence, unit in ((feed, "kg/s"), (jacket, "K")):
            if (
                occurrence.channel.native_unit != unit
                or occurrence.channel.native_action_frame != "reactor-native-actuator"
                or occurrence.channel.native_direction != "increasing-native-command"
                or occurrence.requested.coordinate.clock_id != "reactor-clock"
                or occurrence.requested.coordinate.coordinate_frame != "reactor-episode-time"
                or occurrence.requested.coordinate.time_unit != "s"
                or occurrence.requested.coordinate.origin is not CoordinateOrigin.EPISODE_RELATIVE
                or occurrence.duration != Decimal("10")
                or occurrence.duration_unit != "s"
            ):
                raise ValueError("reactor word changes its native channel, duration or clock")
        command = ReactorCommand(
            action_word.word_id, coordinate.coordinate, feed.requested.value, jacket.requested.value
        )
        delivery = self.session.deliver(command)
        if delivery.command != command:
            raise ValueError("reactor session returned another command")
        observed = []
        for occurrence in occurrences:
            is_feed = occurrence is feed
            realized_values = {e.feed_kg_s if is_feed else e.jacket_k for e in delivery.exposures}
            accepted = delivery.accepted_feed_kg_s if is_feed else delivery.accepted_jacket_k
            applied = delivery.applied_feed_kg_s if is_feed else delivery.applied_jacket_k

            def event(stage: ActionDeliveryStage, value: Decimal) -> ActionStageEvent:
                # Channel metadata is validated above; the value comes from the
                # observed native stage, never from the word's expected stage.
                return replace(
                    occurrence.requested,
                    stage=stage,
                    quantity_id=occurrence.channel.stage_binding(stage).quantity_id,
                    value=value,
                )

            reasons = []
            if accepted != occurrence.requested.value:
                reasons.append("REACTOR_NATIVE_TARGET_CLIPPED")
            if applied != accepted:
                reasons.append("REACTOR_NATIVE_RATE_LIMITED")
            realized = None
            if len(realized_values) == 1:
                realized = event(ActionDeliveryStage.REALIZED, next(iter(realized_values)))
                if realized.value != applied:
                    reasons.append("REACTOR_NATIVE_DOSE_LIMITED")
            else:
                # A scalar occurrence cannot erase a varying exposure. Retain
                # the complete native ledger and return an incomplete stage.
                reasons.append("REACTOR_VARYING_EXPOSURE_REQUIRES_FINER_WORD")
            observed.append(
                ObservedActionOccurrence(
                    observation_id=f"observed.{action_word.word_id}.{occurrence.occurrence_id}",
                    expected_occurrence_id=occurrence.occurrence_id,
                    requested=event(ActionDeliveryStage.REQUESTED, occurrence.requested.value),
                    accepted=event(ActionDeliveryStage.ACCEPTED, accepted),
                    applied=event(ActionDeliveryStage.APPLIED, applied),
                    realized=realized,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        return DeliveryPortResult(f"delivery.{action_word.word_id}", tuple(observed), None)
