"""Explicit complete-unit numerical preparations bound to current native context."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePreparationInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/selective-dependence-response/preparation-input"
    complete_unit_id: str
    target_id: str
    target_design: ObjectIdentity
    phase: str
    roster_index: int
    full_seed: int
    draw_variable_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("complete_unit_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            type(self.target_design) is not ObjectIdentity
            or self.phase not in ("CANARY", "DEVELOPMENT", "EVALUATION", "RESERVE")
            or type(self.roster_index) is not int
            or self.roster_index < 0
            or type(self.full_seed) is not int
            or not 0 <= self.full_seed < 2**128
            or type(self.draw_variable_ids) is not tuple
            or not self.draw_variable_ids
            or len(set(self.draw_variable_ids)) != len(self.draw_variable_ids)
        ):
            raise ValueError("native preparation requires its full numerical seed and exact ordered current context")
        for variable_id in self.draw_variable_ids:
            validate_stable_id(variable_id, field_name="draw_variable_id")

    def validate_context(
        self,
        *,
        complete_unit_id: str,
        target_id: str,
        target_design: ObjectIdentity,
        phase: str,
        draw_variable_ids: tuple[str, ...],
    ) -> None:
        if (
            self.complete_unit_id != complete_unit_id
            or self.target_id != target_id
            or self.target_design != target_design
            or self.phase != phase
            or self.draw_variable_ids != draw_variable_ids
        ):
            raise ValueError("native preparation input differs from its exact current unit/design/phase/draw order")
