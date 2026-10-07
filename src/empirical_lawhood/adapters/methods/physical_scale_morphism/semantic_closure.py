"""Semantic-role and present-close/held-future fibre closure."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismSemanticRolePanel, PHYSICAL_SCALE_MORPHISM_SEMANTIC_ROLE_IDS


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSemanticFibrePair(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-semantic-fibre-pair'

    pair_id: str
    left_unit_id: str
    right_unit_id: str
    present_defect: Decimal
    future_defect: Decimal
    role_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("pair_id", "left_unit_id", "right_unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.left_unit_id == self.right_unit_id:
            raise ValueError("semantic fibre pair requires distinct units")
        validate_decimal(self.present_defect, field_name="present_defect", minimum=Decimal(0))
        validate_decimal(self.future_defect, field_name="future_defect", minimum=Decimal(0))
        require_sorted_unique_strings(self.role_ids, field_name="role_ids", allow_empty=False)
        if not set(self.role_ids).issubset(PHYSICAL_SCALE_MORPHISM_SEMANTIC_ROLE_IDS):
            raise ValueError("semantic fibre pair contains an unknown role")


def evaluate_semantic_closure(
    *,
    panel_id: str,
    morphism_id: str,
    preserved_role_ids: tuple[str, ...],
    omitted_role_ids: tuple[str, ...],
    merged_role_ids: tuple[str, ...],
    fibre_pairs: tuple[PhysicalScaleMorphismSemanticFibrePair, ...],
    present_tolerance: Decimal,
    future_tolerance: Decimal,
) -> PhysicalScaleMorphismSemanticRolePanel:
    require_sorted_unique_ids(fibre_pairs, attribute="pair_id", field_name="fibre_pairs")
    validate_decimal(present_tolerance, field_name="present_tolerance", minimum=Decimal(0))
    validate_decimal(future_tolerance, field_name="future_tolerance", minimum=Decimal(0))
    present_close = tuple(
        value.pair_id for value in fibre_pairs if value.present_defect <= present_tolerance
    )
    future_diverged = tuple(
        value.pair_id
        for value in fibre_pairs
        if value.present_defect <= present_tolerance and value.future_defect > future_tolerance
    )
    return PhysicalScaleMorphismSemanticRolePanel(
        panel_id=panel_id,
        morphism_id=morphism_id,
        preserved_role_ids=preserved_role_ids,
        omitted_role_ids=omitted_role_ids,
        merged_role_ids=merged_role_ids,
        present_close_pair_ids=present_close,
        future_diverged_pair_ids=future_diverged,
        closure_supported=not omitted_role_ids and not merged_role_ids and not future_diverged,
    )


__all__ = ['PhysicalScaleMorphismSemanticFibrePair', "evaluate_semantic_closure"]
