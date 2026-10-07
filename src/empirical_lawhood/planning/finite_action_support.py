"Explicit opt-in compatibility from qualified finite evaluation to admission support.\n\nA finite operator need not invent continuous scalar action bounds. This map\nretains its exact law, member, support cell and complete word roster. It grants\nno support itself: every view must carry the registered law evaluator's exact\nterminal binding. The existing scalar-bound route is unchanged without the map.\n"

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
)
from .evidence_geometry import LawEvaluationBindingDisposition, LawMemberEvaluationBinding, ReceiptAdmissionReceiptProductionPlan, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionSupportCell, validate_law_evaluation_binding

NAMESPACE = "empirical-lawhood.admission.qualified-finite-action-support"


@dataclass(frozen=True, slots=True)
class QualifiedFiniteActionSupportMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/qualified-finite-action-support-map'

    law: ObjectIdentity
    qualification: ObjectIdentity
    member: ObjectIdentity
    support: ReceiptAdmissionSupportCell
    action_words: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.action_words, attribute="object_id", field_name="action_words"
        )
        if not 1 <= len(self.action_words) <= 32 or self.support.action_bound_ids != tuple(
            w.object_id for w in self.action_words
        ):
            raise ValueError("Finite support map must bind the exact complete native-word chart")
        if any(w.object_schema != OccurrenceActionWord.SCHEMA for w in self.action_words):
            raise ValueError("Finite support map requires exact ActionWord identities")

    @property
    def extension(self) -> ExtensionBinding:
        return ExtensionBinding(NAMESPACE, self.SCHEMA, self.fingerprint())


def support_map_for_plan(plan: ReceiptAdmissionReceiptProductionPlan) -> QualifiedFiniteActionSupportMap:
    if (
        len(plan.model_set.members) != 1
        or len(plan.atlas.laws) != 1
        or len(plan.support_cells) != 1
    ):
        raise ValueError("Finite support compatibility requires one exact qualified boundary")
    member, law, cell = plan.model_set.members[0], plan.atlas.laws[0], plan.support_cells[0]
    if (
        law.representation_kind is not LawRepresentationKind.FINITE_ACTION_OPERATOR
        or law.obligations.support.action_bounds
    ):
        raise ValueError("Finite support compatibility cannot replace scalar action bounds")
    return QualifiedFiniteActionSupportMap(
        ObjectIdentity.from_record(law.law_id, law),
        member.qualification_result,
        ObjectIdentity.from_record(member.binding_id, member),
        cell,
        tuple(
            sorted((f.action_word_identity for f in plan.action_fibres), key=lambda w: w.object_id)
        ),
    )


def mapped_coordinate_outside_support(
    plan: ReceiptAdmissionReceiptProductionPlan,
    coordinate: ReceiptAdmissionPlannedCoordinate,
    bindings: tuple[LawMemberEvaluationBinding, ...],
) -> bool:
    mapping = support_map_for_plan(plan)
    extensions = tuple(e for e in plan.model_set.extensions if e.namespace == NAMESPACE)
    if extensions != (mapping.extension,):
        raise ValueError("Finite support map is absent or differs from its complete frozen plan")
    selected = tuple(
        b
        for b in bindings
        if b.planned_coordinate == ObjectIdentity.from_record(coordinate.coordinate_id, coordinate)
    )
    if (
        len(selected) != len(coordinate.qualification_view_ids)
        or tuple(sorted(b.qualification_view_id for b in selected))
        != coordinate.qualification_view_ids
    ):
        raise ValueError("Finite support requires exact law evaluation in every planned view")
    word = plan.action_fibre(coordinate.action_fibre).action_word_identity
    for binding in selected:
        validate_law_evaluation_binding(coordinate, binding)
        if (
            binding.action_word != word
            or binding.law_evaluation_result is None
            or binding.evaluator_implementation is None
        ):
            raise ValueError("Finite support loses its exact word or terminal evaluation binding")
    law = plan.atlas.laws[0]
    action = plan.action_fibre(coordinate.action_fibre).action_word
    return (
        action.denominator_id != mapping.support.denominator_cell_id
        or mapping.support.denominator_cell_id not in law.obligations.support.denominator_cell_ids
        or mapping.support.chart_id != law.chart_id
        or any(
            b.evaluation_disposition is LawEvaluationBindingDisposition.OUTSIDE_SUPPORT
            for b in selected
        )
    )
