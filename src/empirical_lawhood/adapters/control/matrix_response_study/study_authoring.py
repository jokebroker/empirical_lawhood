"""Six-matrix response action-grammar validation around the sole generic programme compiler author.

This module does not fit a law, derive programme admission truth, compile a controller or run a
delivery.  It proves that an exact Six-matrix response primitive word is represented without
loss by the current occurrence/group-aware ``ActionWord`` and then delegates
the scientific gate to :class:`AdmissionControllerAuthor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

from empirical_lawhood.adapters.control.controller_authoring import AdmissionControllerAuthor, AdmissionControllerAuthoringRequest
from empirical_lawhood.kernel.action_contracts import ActionOccurrence, OccurrenceActionWord, ActionWordMode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy

from .contracts import MatrixResponseActionDomain, MatrixResponseActionGrammar, MatrixResponseOuterStudyAuthoringConfig, MatrixResponsePrimitiveAction


@dataclass(frozen=True, slots=True)
class MatrixResponsePrimitiveWordBinding(CanonicalRecord):
    """Exact interpretation of one generic action fibre in the Six-matrix response grammar."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-primitive-word-binding'
    VERSION: ClassVar[str] = "1.0.0"

    binding_id: str
    action_binding_id: str
    action_word: ObjectIdentity
    primitive_ids: tuple[str, ...]
    x_controller_quantity_id: str
    y_controller_quantity_id: str

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "action_binding_id",
            "x_controller_quantity_id",
            "y_controller_quantity_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("Six-matrix response primitive binding requires an exact current ActionWord")
        if not self.primitive_ids:
            raise ValueError("Six-matrix response primitive binding requires a nonempty primitive word")
        for primitive_id in self.primitive_ids:
            validate_stable_id(primitive_id, field_name="primitive_ids")
        if self.x_controller_quantity_id == self.y_controller_quantity_id:
            raise ValueError("Six-matrix response primitive binding must retain two distinct native channels")


@dataclass(frozen=True, slots=True)
class MatrixResponseStudyAuthoringCompatibilityReceipt(CanonicalRecord):
    """Outcome-blind proof that the Six-matrix response chart reached the unchanged programme compiler owner."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-study-authoring-compatibility-receipt'
    VERSION: ClassVar[str] = "1.0.0"

    receipt_id: str
    authoring_config: ObjectIdentity
    study: ObjectIdentity
    primitive_word_bindings: tuple[ObjectIdentity, ...]
    action_fibre_count: int
    occurrence_count: int
    simultaneity_group_count: int
    synthetic_compatibility_only: bool
    scientific_authoring_complete: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.authoring_config.object_schema != MatrixResponseOuterStudyAuthoringConfig.SCHEMA
            or self.study.object_schema != AdmissionControllerStudy.SCHEMA
        ):
            raise ValueError("Six-matrix response authoring receipt binds another config/programme schema")
        require_sorted_unique_ids(
            self.primitive_word_bindings,
            attribute="object_id",
            field_name="primitive_word_bindings",
        )
        if (
            min(
                self.action_fibre_count,
                self.occurrence_count,
                self.simultaneity_group_count,
            )
            < 1
        ):
            raise ValueError("Six-matrix response authoring receipt requires a nonempty chart proof")
        if self.synthetic_compatibility_only == self.scientific_authoring_complete:
            raise ValueError("synthetic Six-matrix response authoring can never claim a complete chart")
        if self.grants_authority:
            raise ValueError("Six-matrix response authoring compatibility cannot grant authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseAuthoredStudy:
    study: AdmissionControllerStudy
    receipt: MatrixResponseStudyAuthoringCompatibilityReceipt


def _expected_complete_words(grammar: MatrixResponseActionGrammar) -> frozenset[tuple[str, ...]]:
    active = tuple(value.primitive_id for value in grammar.primitives if not value.measured_hold)
    hold = next(value.primitive_id for value in grammar.primitives if value.measured_hold)
    return frozenset(
        (
            (hold,),
            *(
                word
                for depth in range(1, grammar.maximum_active_depth + 1)
                for word in product(active, repeat=depth)
            ),
        )
    )


def _stage_values(occurrence: ActionOccurrence) -> tuple[object, ...]:
    return (
        occurrence.requested.value,
        occurrence.accepted.value,
        occurrence.applied.value,
        occurrence.realized.value,
    )


def _validate_group(
    *,
    word: OccurrenceActionWord,
    group_index: int,
    primitive: MatrixResponsePrimitiveAction,
    binding: MatrixResponsePrimitiveWordBinding,
) -> bool:
    group = word.groups[group_index]
    occurrences = {value.occurrence_id: value for value in word.occurrences}
    members = tuple(occurrences[value.occurrence_id] for value in group.members)
    by_channel = {value.channel.controller_quantity_id: value for value in members}
    expected: dict[str, object] = {}
    if primitive.measured_hold:
        if len(members) != 1:
            raise ValueError("Six-matrix response measured HOLD must remain one standalone zero-ramp group")
        expected[members[0].channel.controller_quantity_id] = primitive.delta_x
    else:
        if primitive.delta_x != 0:
            expected[binding.x_controller_quantity_id] = primitive.delta_x
        if primitive.delta_y != 0:
            expected[binding.y_controller_quantity_id] = primitive.delta_y
    if set(by_channel) != set(expected):
        raise ValueError("Six-matrix response primitive occurrence channels differ from its native chart")
    if primitive.simultaneous_channels != (len(members) == 2):
        raise ValueError("Six-matrix response simultaneity-group multiplicity differs from its primitive")
    for channel_id, expected_value in expected.items():
        occurrence = by_channel[channel_id]
        if any(value != expected_value for value in _stage_values(occurrence)):
            raise ValueError("Six-matrix response requested/accepted/applied/realized values differ")
        if (
            occurrence.channel.native_unit != primitive.native_unit
            or occurrence.channel.native_action_frame != primitive.native_frame
        ):
            raise ValueError("Six-matrix response primitive uses another native unit or frame")
    return len(members) == 2


def _validate_word(
    *,
    grammar: MatrixResponseActionGrammar,
    action_word: OccurrenceActionWord,
    binding: MatrixResponsePrimitiveWordBinding,
) -> int:
    if binding.action_word != ObjectIdentity.from_record(action_word.word_id, action_word):
        raise ValueError("Six-matrix response primitive binding substitutes its exact ActionWord")
    primitive_by_id = {value.primitive_id: value for value in grammar.primitives}
    try:
        primitives = tuple(primitive_by_id[value] for value in binding.primitive_ids)
    except KeyError as error:
        raise ValueError("Six-matrix response primitive binding names an undeclared primitive") from error
    holds = tuple(value for value in primitives if value.measured_hold)
    if holds:
        if len(primitives) != 1 or len(holds) != 1:
            raise ValueError("Six-matrix response HOLD cannot be composed with an active primitive")
    elif not 1 <= len(primitives) <= grammar.maximum_active_depth:
        raise ValueError("Six-matrix response active word exceeds its frozen composition depth")
    if len(action_word.groups) != len(primitives):
        raise ValueError("Six-matrix response primitive depth differs from ActionWord group depth")
    simultaneous_groups = sum(
        _validate_group(
            word=action_word,
            group_index=index,
            primitive=primitive,
            binding=binding,
        )
        for index, primitive in enumerate(primitives)
    )
    expected_occurrences = sum(2 if value.simultaneous_channels else 1 for value in primitives)
    if (
        len(action_word.occurrences) != expected_occurrences
        or len(action_word.occurrences) > grammar.maximum_occurrences_per_word
    ):
        raise ValueError("Six-matrix response ActionWord occurrence count differs from its primitive word")
    if any(
        max(abs(value.delta_x), abs(value.delta_y)) / grammar.ramp_steps
        > grammar.maximum_rate_per_step
        for value in primitives
    ):
        raise ValueError("Six-matrix response primitive word exceeds its frozen per-step rate")
    return simultaneous_groups


class MatrixResponseArbitraryChartStudyAuthor:
    """Validate arbitrary Six-matrix response words, then invoke the unchanged programme compiler final author."""

    def __init__(self, config: MatrixResponseOuterStudyAuthoringConfig) -> None:
        self.config = config
        self._delegate = AdmissionControllerAuthor()

    def _author(
        self,
        *,
        request: AdmissionControllerAuthoringRequest,
        grammar: MatrixResponseActionGrammar,
        bindings: tuple[MatrixResponsePrimitiveWordBinding, ...],
        synthetic_compatibility_only: bool,
    ) -> MatrixResponseAuthoredStudy:
        require_sorted_unique_ids(
            bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        by_action = {value.action_binding_id: value for value in bindings}
        actions = {value.action_binding_id: value for value in request.action_bindings}
        if set(by_action) != set(actions) or len(bindings) != len(actions):
            raise ValueError("Six-matrix response primitive bindings differ from the complete programme admission action roster")
        simultaneous_group_count = 0
        for action_id, action in actions.items():
            simultaneous_group_count += _validate_word(
                grammar=grammar,
                action_word=action.action_word,
                binding=by_action[action_id],
            )
        words = frozenset(value.primitive_ids for value in bindings)
        if synthetic_compatibility_only:
            has_mixed_depth_three = any(
                len(value.primitive_ids) == 3
                and actions[value.action_binding_id].action_word.mode is ActionWordMode.MIXED
                for value in bindings
            )
            has_hold = any(
                len(value.primitive_ids) == 1
                and next(
                    item
                    for item in grammar.primitives
                    if item.primitive_id == value.primitive_ids[0]
                ).measured_hold
                for value in bindings
            )
            if not (has_mixed_depth_three and has_hold and simultaneous_group_count):
                raise ValueError(
                    "Six-matrix response synthetic proof requires depth-three mixed action and measured HOLD"
                )
        elif words != _expected_complete_words(grammar):
            raise ValueError("Six-matrix response scientific authoring requires the complete frozen action chart")
        programme = self._delegate.author(request)
        receipt = MatrixResponseStudyAuthoringCompatibilityReceipt(
            receipt_id=f"six-matrix-response-authoring-compatibility.{programme.study_id}",
            authoring_config=ObjectIdentity.from_record(self.config.config_id, self.config),
            study=ObjectIdentity.from_record(programme.study_id, programme),
            primitive_word_bindings=tuple(
                ObjectIdentity.from_record(value.binding_id, value) for value in bindings
            ),
            action_fibre_count=len(bindings),
            occurrence_count=sum(
                len(value.action_word.occurrences) for value in request.action_bindings
            ),
            simultaneity_group_count=simultaneous_group_count,
            synthetic_compatibility_only=synthetic_compatibility_only,
            scientific_authoring_complete=not synthetic_compatibility_only,
            grants_authority=False,
        )
        return MatrixResponseAuthoredStudy(study=programme, receipt=receipt)

    def prove_synthetic_compatibility(
        self,
        *,
        request: AdmissionControllerAuthoringRequest,
        bindings: tuple[MatrixResponsePrimitiveWordBinding, ...],
    ) -> MatrixResponseAuthoredStudy:
        """Exercise mixed occurrence semantics without producing runnable Six-matrix response science."""

        return self._author(
            request=request,
            grammar=self.config.outer_action_grammar,
            bindings=bindings,
            synthetic_compatibility_only=True,
        )

    def author_scientific(
        self,
        *,
        request: AdmissionControllerAuthoringRequest,
        domain: MatrixResponseActionDomain,
        bindings: tuple[MatrixResponsePrimitiveWordBinding, ...],
    ) -> MatrixResponseAuthoredStudy:
        """Author only a complete outer or lower chart; authority remains separate."""

        grammar = (
            self.config.outer_action_grammar
            if domain is MatrixResponseActionDomain.PARENT_COUPLING
            else self.config.inner_action_grammar
        )
        return self._author(
            request=request,
            grammar=grammar,
            bindings=bindings,
            synthetic_compatibility_only=False,
        )


__all__ = [
    'MatrixResponseArbitraryChartStudyAuthor',
    'MatrixResponseAuthoredStudy',
    'MatrixResponsePrimitiveWordBinding',
    'MatrixResponseStudyAuthoringCompatibilityReceipt',
]
