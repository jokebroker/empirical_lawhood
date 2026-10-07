"Finite response-law values in the existing quantity, clock and native action contracts.\n\nThese declarations confer no support or delivery evidence. The finite word\ncontains the expected four delivery stages; native receipts establish what\nactually happened. The receiver is explicitly a paired assay.\n"

from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
    HorizonSpec,
)
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PreparedForceWord
from .law_payloads import FiniteResponseLawResponseChart
from .science import PROGRAMME

HORIZON = HorizonSpec(f"{PROGRAMME}.horizon", Q_CLOCK, D(192), "reference-tick")
PAIRED_RECEIVER = f"{PROGRAMME}.paired-receiver"
PARENT_QUANTITY = f"{PROGRAMME}.assigned-parent"
NATIVE_WORDS = tuple(
    PreparedForceWord(D(magnitude), direction, sign)
    for magnitude, direction, sign in FiniteResponseLawResponseChart().native_words
)


def feature_quantities(boundary: str) -> tuple[QuantitySpec, ...]:
    if boundary not in ("lower", "composed", "cached", "direct"):
        raise ValueError("Unknown finite response-law boundary")
    if boundary == "cached":
        return ()
    stage, cutoff = ("handoff", 4368) if boundary == "lower" else ("prefix", 4096)
    return tuple(
        QuantitySpec(
            f"{PROGRAMME}.{stage}.{name}",
            name,
            QuantityKind.OBSERVATION,
            unit,
            unit,
            Q_FRAME,
            Q_CLOCK,
            AvailabilitySpec(
                Q_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, D(cutoff)
            ),
        )
        for name, unit in zip(
            REFERENCE_INSTRUMENT.feature_ids, REFERENCE_INSTRUMENT.feature_units, strict=True
        )
    )


def parent_quantity() -> QuantitySpec:
    return QuantitySpec(
        PARENT_QUANTITY,
        "Preassigned native parent index",
        QuantityKind.HISTORY,
        "native-parent-index",
        "1",
        Q_FRAME,
        Q_CLOCK,
        AvailabilitySpec(Q_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, D(4096)),
    )


def output_quantities() -> tuple[QuantitySpec, ...]:
    chart = FiniteResponseLawResponseChart()
    return tuple(
        QuantitySpec(
            f"{PROGRAMME}.paired.{name}",
            f"Paired assay: {name}",
            QuantityKind.RECEIVER,
            unit,
            unit,
            Q_FRAME,
            Q_CLOCK,
            AvailabilitySpec(
                Q_CLOCK, CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED, D(4560)
            ),
            ResponseDirection.SIGNED_VECTOR if i < 2 else ResponseDirection.LOWER_IS_BETTER,
        )
        for i, (name, unit) in enumerate(zip(chart.output_ids, chart.output_units, strict=True))
    )


def clock_evaluator() -> ExecutableReference:
    payload = canonical_json_bytes(
        {
            "clock_id": Q_CLOCK,
            "frame_id": Q_FRAME,
            "native_unit": "reference-tick",
            "origin": CoordinateOrigin.ABSOLUTE.value,
            "map": "IDENTITY",
        }
    )
    return ExecutableReference(
        f"{PROGRAMME}.clock-identity",
        f"{PROGRAMME}.clock-identity",
        "1.0.0",
        f"{PROGRAMME}.clock-identity-evaluator",
        ArtifactIdentity(
            f"{PROGRAMME}.clock-identity-payload",
            "clock-evaluator",
            'empirical-lawhood/methods/finite-response-law/clock-identity',
            sha256(payload).hexdigest(),
            "application/json",
            len(payload),
        ),
        SafePayloadFormat.CANONICAL_JSON,
        ClockCoordinate.SCHEMA,
        ClockCoordinate.SCHEMA,
        True,
    )


def native_action_word(
    force: PreparedForceWord,
    *,
    denominator_id: str,
    history_id: str,
) -> OccurrenceActionWord:
    """Declare one existing 64-tick axial pulse at the +272 handoff.

    HOLD has no differential response fibre in this law. Actual matched HOLD
    remains in the paired measurement/preservation records, not a fake zero
    response prediction or an admitted fallback.
    """
    if force not in NATIVE_WORDS:
        raise ValueError("Finite response law requires one of eight native signed axial pulse words")
    stem = f"{PROGRAMME}.native.{force.word_id}"
    origin = CoordinateOrigin.ABSOLUTE
    coordinate = ClockCoordinate(Q_CLOCK, D(4368), "reference-tick", Q_FRAME, origin)
    transport = ClockTransport(
        f"{stem}.clock",
        Q_CLOCK,
        Q_CLOCK,
        "reference-tick",
        "reference-tick",
        Q_FRAME,
        Q_FRAME,
        origin,
        origin,
        ClockTransportKind.IDENTITY,
        ClockTransportAvailability.AVAILABLE,
        D(1),
        D(0),
        D(0),
        D(4096),
        D(4560),
        ClockTransportMonotonicity.STRICTLY_INCREASING,
        clock_evaluator(),
        (),
    )
    direction = f"{CAMPAIGN}.two-port-direction.d{force.direction_index}"
    channel = ActionChannelBinding(
        f"{stem}.channel",
        f"{PROGRAMME}.two-port-force",
        f"{PROGRAMME}.two-port-force-command",
        tuple(
            ActionStageQuantityBinding(
                stage,
                f"{PROGRAMME}.two-port-force.{stage.value.lower()}",
                Q_CLOCK,
                "reference-tick",
                Q_FRAME,
                origin,
            )
            for stage in ActionDeliveryStage
        ),
        "native-hs-force",
        Q_FRAME,
        direction,
        f"{PROGRAMME}.finite-force-source-support",
        f"{PROGRAMME}.both-kicks-completed-interval-delivery",
    )
    events = tuple(
        ActionStageEvent(
            binding.stage,
            binding.quantity_id,
            force.magnitude * force.sign,
            channel.native_unit,
            Q_FRAME,
            direction,
            coordinate,
        )
        for binding in channel.stages
    )
    occurrence = ActionOccurrence(
        f"{stem}.interval.0",
        channel,
        events[0],
        events[1],
        events[2],
        events[3],
        transport,
        transport,
        transport,
        D(64),
        "reference-tick",
    )
    return OccurrenceActionWord(
        stem,
        ActionWordMode.SEQUENTIAL,
        (occurrence,),
        (
            ActionOccurrenceGroup(
                f"{stem}.group.0", (ActionOccurrenceOrder(occurrence.occurrence_id, transport),)
            ),
        ),
        Q_CLOCK,
        "reference-tick",
        Q_FRAME,
        origin,
        denominator_id,
        history_id,
        PAIRED_RECEIVER,
        HORIZON.horizon_id,
        (f"{stem}.support.prefix.0", f"{stem}.support.prefix.1"),
        ActionWordSupportStatus.UNEVALUABLE,
        ("SUPPORT_EXPECTATION_REQUIRES_EVIDENCE",),
    )
