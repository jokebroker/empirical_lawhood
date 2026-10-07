"""Five non-substitutable morphism defects and measured-hold grading."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismDefectPanel, PhysicalScaleMorphismGateSign, PhysicalScaleMorphismHoldDisposition, PhysicalScaleMorphismHoldFibreRecord


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismDefectThresholds(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-defect-thresholds'

    thresholds_id: str
    calibration: Decimal
    observational: Decimal
    held_future_semantic: Decimal
    interventional: Decimal
    decision: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.thresholds_id, field_name="thresholds_id")
        for name in (
            "calibration",
            "observational",
            "held_future_semantic",
            "interventional",
            "decision",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


def make_defect_panel(
    *,
    panel_id: str,
    morphism_id: str,
    thresholds: PhysicalScaleMorphismDefectThresholds,
    calibration_defect: Decimal | None,
    observational_defect: Decimal | None,
    held_future_semantic_defect: Decimal | None,
    interventional_defect: Decimal | None,
    decision_defect: Decimal | None,
    unmatched_operand_ids: tuple[str, ...] = (),
) -> PhysicalScaleMorphismDefectPanel:
    values = (
        calibration_defect,
        observational_defect,
        held_future_semantic_defect,
        interventional_defect,
        decision_defect,
    )
    for index, value in enumerate(values):
        if value is not None:
            validate_decimal(value, field_name=f"defect[{index}]", minimum=Decimal(0))
    calibration_pass = (
        calibration_defect is not None and calibration_defect <= thresholds.calibration
    )
    observation_pass = (
        observational_defect is not None and observational_defect <= thresholds.observational
    )
    future_pass = (
        held_future_semantic_defect is not None
        and held_future_semantic_defect <= thresholds.held_future_semantic
    )
    intervention_pass = (
        interventional_defect is not None and interventional_defect <= thresholds.interventional
    )
    decision_pass = decision_defect is not None and decision_defect <= thresholds.decision
    calibration_only = calibration_pass and not observation_pass
    observational_only = (
        calibration_pass
        and observation_pass
        and not (future_pass and intervention_pass and decision_pass)
    )
    return PhysicalScaleMorphismDefectPanel(
        panel_id=panel_id,
        morphism_id=morphism_id,
        calibration_defect=calibration_defect,
        observational_defect=observational_defect,
        held_future_semantic_defect=held_future_semantic_defect,
        interventional_defect=interventional_defect,
        decision_defect=decision_defect,
        unmatched_operand_ids=unmatched_operand_ids,
        calibration_only=calibration_only,
        observational_only=observational_only,
    )


def evaluate_hold_fibre(
    *,
    hold_id: str,
    context_id: str,
    board_id: str,
    realized_episode_id: str | None,
    gate_signs: tuple[PhysicalScaleMorphismGateSign, ...],
    mapped_disposition: PhysicalScaleMorphismHoldDisposition | None,
) -> PhysicalScaleMorphismHoldFibreRecord:
    if not gate_signs:
        raise ValueError("hold fibre needs assessed gates")
    if realized_episode_id is None:
        disposition = PhysicalScaleMorphismHoldDisposition.HOLD_INVALID_OR_NOT_REALIZED
    elif any(value in {PhysicalScaleMorphismGateSign.UNEVALUABLE, PhysicalScaleMorphismGateSign.AMBIGUOUS} for value in gate_signs):
        disposition = PhysicalScaleMorphismHoldDisposition.HOLD_UNQUALIFIED
    elif any(value in {PhysicalScaleMorphismGateSign.FAIL, PhysicalScaleMorphismGateSign.OUT_OF_SUPPORT} for value in gate_signs):
        disposition = PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_UNSAFE
    else:
        disposition = PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE
    return PhysicalScaleMorphismHoldFibreRecord(
        hold_id=hold_id,
        context_id=context_id,
        board_id=board_id,
        realized_episode_id=realized_episode_id,
        gate_signs=gate_signs,
        disposition=disposition,
        mapped_disposition=mapped_disposition,
        false_safe_hold=mapped_disposition is PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE
        and disposition is not PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE,
    )


def decision_defect(
    *,
    fine_signs: tuple[PhysicalScaleMorphismGateSign, ...],
    mapped_signs: tuple[PhysicalScaleMorphismGateSign, ...],
) -> Decimal:
    """Finite decision-mismatch rate; false-safe cells remain a separate veto."""

    if len(fine_signs) != len(mapped_signs) or not fine_signs:
        raise ValueError("decision sign panels must have the same nonzero shape")
    mismatches = 0
    for fine, mapped in zip(fine_signs, mapped_signs, strict=True):
        mismatches += fine is not mapped
    return Decimal(mismatches) / Decimal(len(fine_signs))


__all__ = [
    'PhysicalScaleMorphismDefectThresholds',
    "decision_defect",
    "evaluate_hold_fibre",
    "make_defect_panel",
]
