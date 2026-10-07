"""Explicit identifier compatibility map into the existing finite controller.

The qualified chart is an eight-output paired assay. ActionWord has one receiver
identifier, so its controller form anchors that block to its first declared
quantity. The complete paired block remains mandatory in the law request, task
and readouts. No pulse, occurrence, clock, frame or response arithmetic changes.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .law_binding import NATIVE_WORDS, native_action_word, output_quantities
from .science import PROGRAMME


def _controller_word(word: OccurrenceActionWord) -> OccurrenceActionWord:
    return replace(
        word,
        word_id=f"{word.word_id}.controller",
        receiver_id=output_quantities()[0].quantity_id,
        occurrences=tuple(
            replace(
                o,
                channel=replace(
                    o.channel, controller_quantity_id=f"{PROGRAMME}.quantity.two-port-action"
                ),
            )
            for o in word.occurrences
        ),
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawControllerWordMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-controller-word-map'

    qualified_word: OccurrenceActionWord
    controller_word: OccurrenceActionWord
    paired_receiver_quantity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        word = self.qualified_word
        originals = tuple(
            replace(
                native_action_word(
                    force, denominator_id=word.denominator_id, history_id=word.retained_history_id
                ),
                support_status=word.support_status,
                reason_codes=word.reason_codes,
            )
            for force in NATIVE_WORDS
        )
        if (
            word not in originals
            or self.controller_word != _controller_word(word)
            or self.paired_receiver_quantity_ids
            != tuple(sorted(q.quantity_id for q in output_quantities()))
        ):
            raise ValueError("Finite response-law controller compatibility changes native pulse or paired receiver")


def controller_word_map(word: OccurrenceActionWord) -> FiniteResponseLawControllerWordMap:
    return FiniteResponseLawControllerWordMap(
        word, _controller_word(word), tuple(sorted(q.quantity_id for q in output_quantities()))
    )
