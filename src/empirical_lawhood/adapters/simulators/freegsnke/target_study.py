"""Exact parent/conditional-child composition for independent substrate grounding FreeGSNKE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.study_authoring import ConditionalChildRequest
from empirical_lawhood.runtime.candidate_compiler import StudyTemplate

from .target_design import FreeGsnkePowerFreeze
from .target_evaluation import FreeGsnkeAdmissionEvaluationEvaluation
from .target_lifecycle_protocol import FREEGSNKE_ADMISSION_EVALUATION_STEP_ID
from .target_validation_protocol import FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID, freegsnke_prospective_conditional_request


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetStudyComposition(CanonicalRecord):
    """One development-through-admission evaluation parent with one fully frozen conditional prospective validation child."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-study-composition'

    composition_id: str
    parent_template: StudyTemplate
    conditional_controller_evaluation_template: StudyTemplate
    conditional_successor: ConditionalChildRequest
    power_freeze: FreeGsnkePowerFreeze
    terminalization_requires_actual_receipts: bool
    outcome_time_scientific_edits_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.composition_id, field_name="composition_id")
        parent_steps = {value.step_id: value for value in self.parent_template.protocol.steps}
        admission_evaluation = parent_steps.get(FREEGSNKE_ADMISSION_EVALUATION_STEP_ID)
        child_slots = tuple(
            value
            for value in self.conditional_controller_evaluation_template.graph.external_inputs
            if value.input_id == FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID
        )
        if (
            admission_evaluation is None
            or {value.payload_schema for value in admission_evaluation.outputs} != {FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA}
            or self.conditional_successor.parent_node_id != FREEGSNKE_ADMISSION_EVALUATION_STEP_ID
            or self.conditional_successor.parent_receipt_input_id != FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID
            or self.conditional_successor.template_key != self.conditional_controller_evaluation_template.template_key
            or self.conditional_successor.evaluation_unit_ids != self.power_freeze.prospective_validation_unit_ids
            or len(child_slots) != 1
            or not self.terminalization_requires_actual_receipts
            or self.outcome_time_scientific_edits_allowed
        ):
            raise ValueError("FreeGSNKE parent/conditional-child composition differs")


def compose_freegsnke_target_study(
    *,
    composition_id: str,
    conditional_request_id: str,
    parent_template: StudyTemplate,
    conditional_controller_evaluation_template: StudyTemplate,
    power_freeze: FreeGsnkePowerFreeze,
) -> FreeGsnkeTargetStudyComposition:
    """Attach the exact prospective validation template to the sole admission evaluation parent before issue."""

    request = freegsnke_prospective_conditional_request(
        request_id=conditional_request_id,
        parent_node_id=FREEGSNKE_ADMISSION_EVALUATION_STEP_ID,
        power_freeze=power_freeze,
        child_template=conditional_controller_evaluation_template,
    )
    return FreeGsnkeTargetStudyComposition(
        composition_id=composition_id,
        parent_template=parent_template,
        conditional_controller_evaluation_template=conditional_controller_evaluation_template,
        conditional_successor=request,
        power_freeze=power_freeze,
        terminalization_requires_actual_receipts=True,
        outcome_time_scientific_edits_allowed=False,
    )


__all__ = [
    'FreeGsnkeTargetStudyComposition',
    'compose_freegsnke_target_study',
]
