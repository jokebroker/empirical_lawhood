"Conjunctive finite-closure assessment for witnessed action-word descent.\n\nThis additive adapter consumes eligible transformation assessments through\nan explicit compatibility record.  A mixed square can never imply fully witnessed conjunctive square closure on\nits own: identities, supported composition, tested associativity, both typed\ndirections, eligible squares and every declared square prerequisite are all\nrequired.\n"

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
import json
from typing import ClassVar, Mapping, Sequence, cast

from empirical_lawhood.adapters.methods.action_word_descent import MapDisposition, TransformationAssessment
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_document_shape,
    validate_stable_id,
)


class ClosureRung(StrEnum):
    BELOW_TYPED_PARTIAL_GRAPH = "BELOW_TYPED_PARTIAL_GRAPH"
    TYPED_PARTIAL_GRAPH = "TYPED_PARTIAL_GRAPH"
    FINITE_CATEGORY = "FINITE_CATEGORY"
    ADDITIONAL_FINITE_STRUCTURE = "ADDITIONAL_FINITE_STRUCTURE"
    CONJUNCTIVE_SQUARE_CLOSURE = "CONJUNCTIVE_SQUARE_CLOSURE"


@dataclass(frozen=True, slots=True)
class ClosurePrerequisiteEvidence(CanonicalRecord):
    """Explicit witnesses required by the bounded finite closure ladder."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/closure-prerequisite-evidence'

    evidence_id: str
    typed_arrows_tested: bool
    explicit_undefined_set_complete: bool
    identity_witness_ids: tuple[str, ...]
    composition_witness_ids: tuple[str, ...]
    associativity_witness_ids: tuple[str, ...]
    additional_structure_witness_ids: tuple[str, ...]
    horizontal_direction_witness_ids: tuple[str, ...]
    vertical_direction_witness_ids: tuple[str, ...]
    eligible_mixed_square_ids: tuple[str, ...]
    passed_mixed_square_ids: tuple[str, ...]
    declared_square_prerequisite_ids: tuple[str, ...]
    satisfied_square_prerequisite_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        for name, values in (
            ("identity_witness_ids", self.identity_witness_ids),
            ("composition_witness_ids", self.composition_witness_ids),
            ("associativity_witness_ids", self.associativity_witness_ids),
            ("additional_structure_witness_ids", self.additional_structure_witness_ids),
            ("horizontal_direction_witness_ids", self.horizontal_direction_witness_ids),
            ("vertical_direction_witness_ids", self.vertical_direction_witness_ids),
            ("eligible_mixed_square_ids", self.eligible_mixed_square_ids),
            ("passed_mixed_square_ids", self.passed_mixed_square_ids),
            (
                "declared_square_prerequisite_ids",
                self.declared_square_prerequisite_ids,
            ),
            (
                "satisfied_square_prerequisite_ids",
                self.satisfied_square_prerequisite_ids,
            ),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=True)
        if not set(self.passed_mixed_square_ids).issubset(self.eligible_mixed_square_ids):
            raise ValueError("passed mixed squares must be eligible")
        if not set(self.satisfied_square_prerequisite_ids).issubset(
            self.declared_square_prerequisite_ids
        ):
            raise ValueError("satisfied square prerequisites must be declared")


@dataclass(frozen=True, slots=True)
class ClosureCompatibilityMap(CanonicalRecord):
    "Identity-preserving map from eligible assessment records."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/closure-compatibility-map'

    map_id: str
    source_schema: str
    source_assessment_ids: tuple[str, ...]
    target_capability_key: str
    scientific_meaning_preserved: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.map_id, field_name="map_id")
        if self.source_schema != TransformationAssessment.SCHEMA:
            raise ValueError("closure compatibility source schema differs")
        require_sorted_unique_strings(
            self.source_assessment_ids,
            field_name="source_assessment_ids",
        )
        if self.target_capability_key != 'witnessed-action-word-closure':
            raise ValueError("closure compatibility target differs")
        if not self.scientific_meaning_preserved:
            raise ValueError("incompatible transformation assessments cannot enter witnessed closure")


@dataclass(frozen=True, slots=True)
class WitnessedActionWordClosureAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/witnessed-action-word-closure-assessment'

    assessment_id: str
    compatibility_map_id: str
    admitted_word_ids: tuple[str, ...]
    minimal_counterexample_word_ids: tuple[str, ...]
    undefined_word_ids: tuple[str, ...]
    supported_properties: tuple[str, ...]
    prerequisite_witness_ids: tuple[str, ...]
    maximum_formal_rung: ClosureRung
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.compatibility_map_id, field_name="compatibility_map_id")
        for name, values in (
            ("admitted_word_ids", self.admitted_word_ids),
            ("minimal_counterexample_word_ids", self.minimal_counterexample_word_ids),
            ("undefined_word_ids", self.undefined_word_ids),
            ("supported_properties", self.supported_properties),
            ("prerequisite_witness_ids", self.prerequisite_witness_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=True)
        if not self.reason_codes:
            raise ValueError("closure assessment requires rung reasons")


def _all_admitted(
    assessments: Sequence[TransformationAssessment],
) -> tuple[TransformationAssessment, ...]:
    return tuple(
        item
        for item in assessments
        if item.disposition in {MapDisposition.EXACT, MapDisposition.EQUIVALENT}
    )


def assess_closure(
    assessments: Sequence[TransformationAssessment],
    prerequisites: ClosurePrerequisiteEvidence,
    compatibility: ClosureCompatibilityMap,
    *,
    assessment_id: str,
) -> WitnessedActionWordClosureAssessment:
    """Return the greatest fully witnessed rung without vacuous promotion."""

    if not assessments:
        raise ValueError("closure assessment requires transformation evidence")
    assessment_ids = tuple(sorted(item.assessment_id for item in assessments))
    if assessment_ids != compatibility.source_assessment_ids:
        raise ValueError("closure compatibility assessment set differs")
    word_ids = tuple(item.word_id for item in assessments)
    if len(set(word_ids)) != len(word_ids):
        raise ValueError("closure assessment contains duplicate word identities")

    admitted_assessments = _all_admitted(assessments)
    admitted = tuple(sorted(item.word_id for item in admitted_assessments))
    counterexamples = tuple(
        sorted(item.word_id for item in assessments if item.disposition is MapDisposition.MATERIAL)
    )
    undefined = tuple(
        sorted(
            item.word_id
            for item in assessments
            if item.disposition
            in {MapDisposition.UNDEFINED, MapDisposition.INVALID, MapDisposition.UNEVALUABLE}
        )
    )

    properties: set[str] = set()
    reasons: set[str] = set()
    typed_partial_graph_complete = (
        bool(admitted)
        and prerequisites.typed_arrows_tested
        and prerequisites.explicit_undefined_set_complete
    )
    if typed_partial_graph_complete:
        properties.update(("EXPLICIT_UNDEFINED_SET", "TYPED_ARROWS"))
        reasons.add("TYPED_PARTIAL_GRAPH_COMPLETE")
    else:
        reasons.add("TYPED_PARTIAL_GRAPH_PREREQUISITE_INCOMPLETE")

    identities = bool(prerequisites.identity_witness_ids) and all(
        item.identity_valid for item in admitted_assessments
    )
    composition = bool(prerequisites.composition_witness_ids) and all(
        item.composition_valid for item in admitted_assessments
    )
    tested_associativity = bool(prerequisites.associativity_witness_ids)
    associativity = tested_associativity and all(
        item.associativity_valid is True
        for item in admitted_assessments
        if item.word_id in prerequisites.associativity_witness_ids
    )
    associativity = associativity and all(
        witness in admitted for witness in prerequisites.associativity_witness_ids
    )
    finite_category_complete = typed_partial_graph_complete and identities and composition and associativity
    if identities:
        properties.add("IDENTITIES")
    if composition:
        properties.add("SUPPORTED_COMPOSITION")
    if associativity:
        properties.add("TESTED_ASSOCIATIVITY")
    if finite_category_complete:
        reasons.add("FINITE_CATEGORY_PREREQUISITES_COMPLETE")
    else:
        reasons.add("FINITE_CATEGORY_PREREQUISITE_INCOMPLETE")

    additional_structure_witnessed = finite_category_complete and bool(prerequisites.additional_structure_witness_ids)
    if additional_structure_witnessed:
        properties.add("DECLARED_ADDITIONAL_FINITE_STRUCTURE")
        reasons.add("ADDITIONAL_FINITE_STRUCTURE_WITNESSED")

    mixed_squares_complete = (
        bool(prerequisites.eligible_mixed_square_ids)
        and prerequisites.passed_mixed_square_ids == prerequisites.eligible_mixed_square_ids
    )
    square_prerequisites_complete = (
        bool(prerequisites.declared_square_prerequisite_ids)
        and prerequisites.satisfied_square_prerequisite_ids
        == prerequisites.declared_square_prerequisite_ids
    )
    both_directions = bool(prerequisites.horizontal_direction_witness_ids) and bool(
        prerequisites.vertical_direction_witness_ids
    )
    conjunctive_square_closure_complete = finite_category_complete and both_directions and mixed_squares_complete and square_prerequisites_complete
    if both_directions:
        properties.add("BOTH_TYPED_DIRECTIONS")
    if mixed_squares_complete:
        properties.add("ALL_ELIGIBLE_MIXED_SQUARES")
    if square_prerequisites_complete:
        properties.add("ALL_DECLARED_SQUARE_PREREQUISITES")
    if conjunctive_square_closure_complete:
        reasons.add("CONJUNCTIVE_SQUARE_CLOSURE_PREREQUISITES_COMPLETE")
    elif prerequisites.passed_mixed_square_ids:
        reasons.add("MIXED_SQUARE_ALONE_CANNOT_PROMOTE_CONJUNCTIVE_CLOSURE")

    rung = ClosureRung.BELOW_TYPED_PARTIAL_GRAPH
    if typed_partial_graph_complete:
        rung = ClosureRung.TYPED_PARTIAL_GRAPH
    if finite_category_complete:
        rung = ClosureRung.FINITE_CATEGORY
    if additional_structure_witnessed:
        rung = ClosureRung.ADDITIONAL_FINITE_STRUCTURE
    if conjunctive_square_closure_complete:
        rung = ClosureRung.CONJUNCTIVE_SQUARE_CLOSURE

    witness_ids = tuple(
        sorted(
            {
                *prerequisites.identity_witness_ids,
                *prerequisites.composition_witness_ids,
                *prerequisites.associativity_witness_ids,
                *prerequisites.additional_structure_witness_ids,
                *prerequisites.horizontal_direction_witness_ids,
                *prerequisites.vertical_direction_witness_ids,
                *prerequisites.passed_mixed_square_ids,
                *prerequisites.satisfied_square_prerequisite_ids,
            }
        )
    )
    return WitnessedActionWordClosureAssessment(
        assessment_id=assessment_id,
        compatibility_map_id=compatibility.map_id,
        admitted_word_ids=admitted,
        minimal_counterexample_word_ids=counterexamples,
        undefined_word_ids=undefined,
        supported_properties=tuple(sorted(properties)),
        prerequisite_witness_ids=witness_ids,
        maximum_formal_rung=rung,
        reason_codes=tuple(sorted(reasons)),
    )


def decode_closure_prerequisites(payload: bytes) -> ClosurePrerequisiteEvidence:
    """Strictly decode the externally persisted prerequisite contract."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("closure-prerequisite payload is not JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("closure-prerequisite document is not a mapping")
    raw = validate_document_shape(
        cast(Mapping[str, object], document),
        expected_schema=ClosurePrerequisiteEvidence.SCHEMA,
        expected_version=ClosurePrerequisiteEvidence.VERSION,
        field_names=frozenset(field.name for field in fields(ClosurePrerequisiteEvidence)),
    )

    def string_tuple(name: str) -> tuple[str, ...]:
        value = raw[name]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{name} differs")
        return tuple(value)

    return ClosurePrerequisiteEvidence(
        evidence_id=str(raw["evidence_id"]),
        typed_arrows_tested=bool(raw["typed_arrows_tested"]),
        explicit_undefined_set_complete=bool(raw["explicit_undefined_set_complete"]),
        identity_witness_ids=string_tuple("identity_witness_ids"),
        composition_witness_ids=string_tuple("composition_witness_ids"),
        associativity_witness_ids=string_tuple("associativity_witness_ids"),
        additional_structure_witness_ids=string_tuple("additional_structure_witness_ids"),
        horizontal_direction_witness_ids=string_tuple("horizontal_direction_witness_ids"),
        vertical_direction_witness_ids=string_tuple("vertical_direction_witness_ids"),
        eligible_mixed_square_ids=string_tuple("eligible_mixed_square_ids"),
        passed_mixed_square_ids=string_tuple("passed_mixed_square_ids"),
        declared_square_prerequisite_ids=string_tuple("declared_square_prerequisite_ids"),
        satisfied_square_prerequisite_ids=string_tuple("satisfied_square_prerequisite_ids"),
    )


__all__ = [
    'WitnessedActionWordClosureAssessment',
    'ClosureCompatibilityMap',
    'ClosurePrerequisiteEvidence',
    'ClosureRung',
    'assess_closure',
    'decode_closure_prerequisites',
]
