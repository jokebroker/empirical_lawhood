"""Actual native delivery of one bounded, already committed feed word."""

from dataclasses import replace

from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession, ReactorExposureWordMap, exposure_word
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.control import OperationalDeliveryState
from empirical_lawhood.runtime.controller_runtime import DeliveryPortResult
from .contracts import ReactorDelivery
from .finite_words import feed_action_word


def deliver_feed_word(
    session: BatchDeliverySession,
    action_word: OccurrenceActionWord,
    mappings: tuple[ReactorExposureWordMap, ...],
    last_deliveries: list[ReactorDelivery],
) -> DeliveryPortResult:
    if last_deliveries or not 1 <= len(mappings) <= 24:
        raise ValueError("finite native delivery requires one unconsumed bounded word")
    chart = action_word.denominator_id
    observed, actual = [], []
    for expected in mappings:
        try:
            raw = session.advance(expected.command)
        except RuntimeError as error:
            if str(error) not in {
                "reactor delivery timed out; no completion evidence",
                "native reactor failed; no completion evidence",
            }:
                raise
            break
        if not isinstance(raw, ReactorDelivery) or raw.command != expected.command:
            raise ValueError("native pulse substituted its command or ended before its guard")
        last_deliveries.append(raw)
        exposure = tuple((e.time_s, e.duration_s, e.feed_kg_s, e.jacket_k) for e in raw.exposures)
        accepted = (raw.accepted_feed_kg_s, raw.accepted_jacket_k)
        applied = (raw.applied_feed_kg_s, raw.applied_jacket_k)
        word = exposure_word(
            raw.command,
            accepted,
            applied,
            exposure,
            history_id=action_word.retained_history_id,
            horizon_id=action_word.horizon_id,
        )
        actual.append(ReactorExposureWordMap(raw.command, accepted, applied, exposure, word))
        for occurrence in word.occurrences:
            if occurrence.channel.controller_quantity_id != "reactor-feed":
                continue
            occurrence = replace(
                occurrence, channel=replace(occurrence.channel, support_contract_id=chart)
            )
            observed.append(
                ObservedActionOccurrence(
                    f"observed.{occurrence.occurrence_id}",
                    occurrence.occurrence_id,
                    occurrence.requested,
                    occurrence.accepted,
                    occurrence.applied,
                    occurrence.realized,
                    (),
                )
            )
        if actual[-1] != expected:
            break
    exact = (
        len(actual) == len(mappings)
        and feed_action_word(
            tuple(actual),
            word_id=action_word.word_id,
            history_id=action_word.retained_history_id,
            horizon_id=action_word.horizon_id,
            chart_id=chart,
            receiver_id=action_word.receiver_id,
        )
        == action_word
    )
    return DeliveryPortResult(
        f"delivery.{action_word.word_id}",
        tuple(observed),
        None if exact else OperationalDeliveryState.TERMINATED,
    )
