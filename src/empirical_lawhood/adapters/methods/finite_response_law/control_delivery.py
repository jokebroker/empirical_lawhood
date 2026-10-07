"""Receipt-backed native delivery binding for one paired-purpose numerical view.

The source provider owns acquisition. This port authenticates its actual bytes
and reports the frozen command's delivery; it never invokes a simulator or
reselects an action. Both independent futures must satisfy every delivery stage.
"""

from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.control import (
    OperationalDeliveryState,
    ScientificCommitmentKind,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.runtime.controller_runtime import DeliveryPortResult
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawEvaluationTaskResult, decode_task_native
from .control_word import controller_word_map
from .control_evaluation import measured_hold_word
from .law_binding import native_action_word, NATIVE_WORDS


def observation_words(root_id: str) -> tuple[OccurrenceActionWord, ...]:
    """Native observation labels only; no support or controller admission claim.

    Denominator/history labels do not enter action-stage observations. The
    eventual delivery trace must still bind its exact precommitted full word.
    """
    return tuple(
        controller_word_map(
            native_action_word(
                force, denominator_id=root_id, history_id=f"{root_id}.native-history"
            )
        ).controller_word
        for force in NATIVE_WORDS
    ) + (measured_hold_word(root_id, f"{root_id}.native-history"),)


def _force(word: OccurrenceActionWord) -> PreparedForceWord:
    for force in NATIVE_WORDS:
        original = replace(
            native_action_word(
                force,
                denominator_id=word.denominator_id,
                history_id=word.retained_history_id,
            ),
            support_status=word.support_status,
            reason_codes=word.reason_codes,
        )
        if word == controller_word_map(original).controller_word:
            return force
    if word == measured_hold_word(word.denominator_id, word.retained_history_id):
        return PreparedForceWord(D(0), 0, 0)
    raise ValueError(
        "Finite response-law delivery substitutes the exact native word, frame, waveform or timing"
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawObservedNativeWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-observed-native-word'
    )
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawEvaluationTaskResult]] = FiniteResponseLawEvaluationTaskResult
    word: OccurrenceActionWord
    refinement: int
    sources: tuple[FiniteResponseLawEvaluationTaskResult, ...]
    observed: ObservedActionOccurrence

    def __post_init__(self) -> None:
        force = _force(self.word)
        if (
            type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or tuple(r.invocation.purpose for r in self.sources)
            != ("future-1", "future-2")
            or len({r.invocation.root for r in self.sources}) != 1
            or len({r.invocation.source for r in self.sources}) != 1
            or len({r.predecessors for r in self.sources}) != 1
            or len({r.common_start for r in self.sources}) != 1
            or len({r.frozen_ports_base64 for r in self.sources}) != 1
            or any(type(r) is not self.SOURCE_TYPE for r in self.sources)
            or any(r.invocation.word != force for r in self.sources)
            or self.observed.expected_occurrence_id
            != self.word.occurrences[0].occurrence_id
        ):
            raise ValueError(
                "Finite response-law observed delivery loses its exact root, two futures or common start"
            )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedObservedNativeWord(FiniteResponseLawObservedNativeWord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-observed-native-word'
    )
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawEvaluationTaskResult]] = (
        FiniteResponseLawAssignedEvaluationTaskResult
    )
    sources: tuple[FiniteResponseLawAssignedEvaluationTaskResult, ...]


def observe_native_word(
    *,
    word: OccurrenceActionWord,
    refinement: int,
    inputs: tuple[tuple[FiniteResponseLawEvaluationTaskResult, bytes], ...],
) -> FiniteResponseLawObservedNativeWord:
    """Authenticate full interval traces; an absent/partial stage stays absent."""
    force = _force(word)
    if type(refinement) is not int or refinement not in (1, 2) or len(inputs) != 2:
        raise ValueError(
            "Finite response-law delivery needs both purposes of one declared numerical view"
        )
    accepted = True
    applied = True
    realized = True
    reasons: set[str] = set()
    sources = tuple(record for record, _ in inputs)
    result_type = type(sources[0])
    if result_type not in (
        FiniteResponseLawEvaluationTaskResult,
        FiniteResponseLawAssignedEvaluationTaskResult,
    ) or any(type(record) is not result_type for record in sources):
        raise ValueError("Finite response-law delivery mixes versioned native sources")
    for record, raw in inputs:
        if record.invocation.word != force:
            raise ValueError("Finite response-law delivery binds another native invocation")
        pair = decode_task_native(record, raw)
        if pair is None:
            accepted = applied = realized = False
            reasons.add(record.unentered_reason or "NATIVE_SOURCE_UNAVAILABLE")
            continue
        phase = pair[refinement - 1]
        delivery = phase.delivery
        accepted = accepted and delivery.accepted
        trace = phase.realized_trace
        start = 4368 * refinement
        steps = np.arange(start, 4560 * refinement)
        shape_valid = trace.shape == (len(steps), 9)
        if not shape_valid or not np.array_equal(trace[:, 0], steps):
            applied = realized = False
            reasons.add("NATIVE_INTERVAL_TRACE_INCOMPLETE")
            continue
        components = prepared_force_components(
            force, native_step=start, invocation_tick=4368, refinement=refinement
        )
        expected = (steps < start + 64 * refinement)[:, None] * components
        applied_here = np.array_equal(trace[:, 5:7], expected)
        realized_here = np.array_equal(trace[:, 7:9], expected)
        applied = applied and applied_here
        realized = realized and realized_here and delivery.disposition == "COMPLETE"
        if not applied_here or not realized_here:
            reasons.add("NATIVE_FORCE_TRACE_MISMATCH")
        if delivery.disposition != "COMPLETE":
            reasons.add(delivery.reason or "NATIVE_SOURCE_INCOMPLETE")
    occurrence = word.occurrences[0]
    # These event values are admitted only after authentication of BOTH full
    # native traces in the declared frame. No interval/future errors are averaged.
    observed = ObservedActionOccurrence(
        f"{sources[0].invocation.root.stage_unit}.{force.word_id}.r{refinement}.observed",
        occurrence.occurrence_id,
        occurrence.requested,
        occurrence.accepted if accepted else None,
        occurrence.applied if applied else None,
        occurrence.realized if realized else None,
        tuple(sorted(reasons)),
    )
    output_type = (
        FiniteResponseLawAssignedObservedNativeWord
        if result_type is FiniteResponseLawAssignedEvaluationTaskResult
        else FiniteResponseLawObservedNativeWord
    )
    return output_type(word, refinement, sources, observed)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeReceiptDelivery:
    implementation_binding: ImplementationBinding
    evidence: FiniteResponseLawObservedNativeWord | None = None

    def __post_init__(self) -> None:
        if self.implementation_binding.role is not ImplementationRole.DELIVERY:
            raise ValueError(
                "Finite response-law native receipt port requires its exact delivery implementation"
            )

    def deliver(
        self, *, action_word: OccurrenceActionWord, commitment_kind: ScientificCommitmentKind
    ) -> DeliveryPortResult:
        if self.evidence is None:
            raise ValueError(
                "No actual native receipts: pre-parent preparation cannot deliver"
            )
        force = _force(action_word)
        kind = (
            ScientificCommitmentKind.MEASURED_HOLD
            if force.sign == 0
            else ScientificCommitmentKind.ACTION
        )
        if action_word != self.evidence.word or commitment_kind is not kind:
            raise ValueError(
                "Finite response-law receipt delivery substitutes the precommitted native action"
            )
        observed = self.evidence.observed
        return DeliveryPortResult(
            f"{observed.observation_id}.delivery",
            (observed,),
            None
            if observed.complete and not observed.reason_codes
            else OperationalDeliveryState.TERMINATED,
        )
