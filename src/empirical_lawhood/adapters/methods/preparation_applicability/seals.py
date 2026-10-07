"""Two separate causal seals: preparation choice and post-handoff lower use."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.simulators.preparation_applicability.records import PreparationApplicabilityNativePhase
from .records import finite


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityParents(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/parents"
    root_id: str
    prefix_sha256: str
    selection_sha256: str
    phases: tuple[PreparationApplicabilityNativePhase, ...]

    def __post_init__(self) -> None:
        for v in (self.prefix_sha256, self.selection_sha256):
            validate_sha256(v, field_name="parents_sha256")
        keys = tuple((p.schedule_index, p.refinement) for p in self.phases)
        if len(keys) != len(set(keys)) or any(
            p.root_id != self.root_id or p.phase != "parent" for p in self.phases
        ):
            raise ValueError("parents duplicate or substitute native acquisitions")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityLowerSeal(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/lower-seal"
    root_id: str
    parents_sha256: str
    lower_sha256: str
    complete: bool
    mean: tuple[Decimal, ...]
    width: tuple[Decimal, ...]
    support: tuple[bool, ...]
    choices: tuple[int, ...]
    directions: tuple[int, ...]
    requirements: tuple[Decimal, ...]
    cutoff_tick: int = 4496

    def __post_init__(self) -> None:
        for v in (self.parents_sha256, self.lower_sha256):
            validate_sha256(v, field_name="lower_seal_sha256")
        finite(self.mean, 96 if self.complete else 0)
        finite(self.width, 96 if self.complete else 0)
        if self.cutoff_tick != 4496 or len(self.support) != (3 if self.complete else 0):
            raise ValueError("lower seal changes its actual handoff or support")
        if len(self.choices) not in (0, 1536) or any(
            type(v) is not int or v not in range(-1, 8) for v in self.choices
        ):
            raise ValueError("lower seal changes the finite word/NONATTEMPT census")
        if self.choices and (len(self.directions) != 512 or len(self.requirements) != 512):
            raise ValueError("lower seal omits its exact new consumer assignment")
