"""Explicit numerical allocations for orthogonal operator comparisons.

The operator supplies a separately authenticated original export and its current
configuration binding. This input cannot authenticate that export or transfer
qualification or permission to access a native trajectory.
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id


@dataclass(frozen=True, slots=True)
class MatrixResponseOperatorShuffleScientificInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-response-operator-shuffle"

    original_source: ObjectIdentity
    export_receipt: ObjectIdentity
    source_original_context_sha256: str
    current_config_sha256: str
    current_trajectory_id: str
    origin_steps: tuple[int, ...]
    full_seed_sha256s: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.original_source, ObjectIdentity) or not isinstance(self.export_receipt, ObjectIdentity):
            raise ValueError("operator comparisons require original source and external export custody")
        validate_sha256(self.source_original_context_sha256, field_name="source_original_context_sha256")
        validate_sha256(self.current_config_sha256, field_name="current_config_sha256")
        validate_stable_id(self.current_trajectory_id, field_name="current_trajectory_id")
        if self.origin_steps != (816, 848, 880, 896, 928, 960):
            raise ValueError("operator comparison inputs require the complete six-origin acquisition order")
        if not isinstance(self.full_seed_sha256s, tuple) or len(self.full_seed_sha256s) != 6:
            raise ValueError("operator comparisons require all six numerical allocation rows")
        for row in self.full_seed_sha256s:
            if not isinstance(row, tuple) or len(row) != 32:
                raise ValueError("operator comparisons require all thirty-two ordered shuffles per origin")
            for seed in row:
                validate_sha256(seed, field_name="full_seed_sha256")
        prefixes = tuple(seed[:32] for row in self.full_seed_sha256s for seed in row)
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("operator comparison inputs repeat a PCG allocation")

    def require_current_binding(
        self, *, current_config_sha256: str, current_trajectory_id: str,
        origin_steps: tuple[int, ...] | None = None,
    ) -> None:
        if (
            self.current_config_sha256 != current_config_sha256
            or self.current_trajectory_id != current_trajectory_id
            or (origin_steps is not None and self.origin_steps != origin_steps)
        ):
            raise ValueError("operator comparison inputs bind another current configuration, trajectory or origin order")

    def seeds_for_origin(self, origin_step: int) -> tuple[str, ...]:
        if type(origin_step) is not int or origin_step not in self.origin_steps:
            raise ValueError("operator comparison origin is outside the complete numerical input")
        return self.full_seed_sha256s[self.origin_steps.index(origin_step)]


def require_operator_shuffle_scientific_inputs(
    value: MatrixResponseOperatorShuffleScientificInputs, *,
    current_config_sha256: str, current_trajectory_id: str,
    origin_steps: tuple[int, ...] | None = None,
) -> MatrixResponseOperatorShuffleScientificInputs:
    if not isinstance(value, MatrixResponseOperatorShuffleScientificInputs):
        raise ValueError("operator comparisons require complete typed original numerical inputs and external export custody")
    value.require_current_binding(
        current_config_sha256=current_config_sha256,
        current_trajectory_id=current_trajectory_id,
        origin_steps=origin_steps,
    )
    return value
