"""Explicit original numerical inputs and current custody for reactive entrance.

Original mappings require external verification before source issue. This record
keeps original identities separate and transfers no historical qualification.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256


@dataclass(frozen=True, slots=True)
class MatrixReactiveEntranceRootScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-reactive-entrance-root"

    slot_index: int
    fine_selection_rank_sha256: str
    source_seed_sha256: str
    bridge_seed_sha256: str
    has_fine_view: bool

    def __post_init__(self) -> None:
        if type(self.slot_index) is not int or not 0 <= self.slot_index < 512:
            raise ValueError("reactive-entrance scientific root index differs")
        if type(self.has_fine_view) is not bool:
            raise ValueError("reactive-entrance fine selection must be explicit")
        for name in ("fine_selection_rank_sha256", "source_seed_sha256", "bridge_seed_sha256"):
            validate_sha256(getattr(self, name), field_name=name)


def reactive_entrance_numeric_census_sha256(roots: tuple[MatrixReactiveEntranceRootScientificInput, ...]) -> str:
    return sha256(canonical_json_bytes(tuple(
        (value.slot_index, value.fine_selection_rank_sha256, value.source_seed_sha256,
         value.bridge_seed_sha256, value.has_fine_view) for value in roots
    ))).hexdigest()


@dataclass(frozen=True, slots=True)
class MatrixReactiveEntranceScientificInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-reactive-entrance"

    original_plan_sha256: str
    target_plan_sha256: str
    original_source_cohort_sha256: str
    target_source_cohort_sha256: str
    original_checkpoint_sha256: str
    target_checkpoint_sha256: str
    replay_scientific_seed_sha256: str
    original_numeric_census_sha256: str
    roots: tuple[MatrixReactiveEntranceRootScientificInput, ...]

    def __post_init__(self) -> None:
        for name in (
            "original_plan_sha256", "target_plan_sha256", "original_source_cohort_sha256",
            "target_source_cohort_sha256", "original_checkpoint_sha256", "target_checkpoint_sha256",
            "replay_scientific_seed_sha256", "original_numeric_census_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            type(self.roots) is not tuple
            or any(not isinstance(value, MatrixReactiveEntranceRootScientificInput) for value in self.roots)
            or tuple(value.slot_index for value in self.roots) != tuple(range(512))
        ):
            raise ValueError("reactive-entrance requires the complete ordered 512-root scientific input census")
        ranked = sorted(self.roots, key=lambda value: bytes.fromhex(value.fine_selection_rank_sha256))
        fine = {value.slot_index for value in ranked[:128]}
        if any(value.has_fine_view != (value.slot_index in fine) for value in self.roots):
            raise ValueError("reactive-entrance original fine selection differs from its full numeric ranking")
        if reactive_entrance_numeric_census_sha256(self.roots) != self.original_numeric_census_sha256:
            raise ValueError("reactive-entrance original numeric census custody differs")

    def require_target_roots(self, *, plan_sha256: str, source_cohort_sha256: str) -> None:
        if self.target_plan_sha256 != plan_sha256 or self.target_source_cohort_sha256 != source_cohort_sha256:
            raise ValueError("reactive-entrance scientific input target plan/cohort custody differs")

    def require_target_checkpoint(self, checkpoint_sha256: str) -> None:
        if self.target_checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("reactive-entrance scientific input target checkpoint custody differs")
