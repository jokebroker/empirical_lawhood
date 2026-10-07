"""Caller-supplied full numerical stream inputs with separate source custody.

A typed specimen binds current operands and explicit original export references.
Callers must authenticate the export before issue; this record cannot transfer
source qualification or permission to execute a native source.
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id


@dataclass(frozen=True, slots=True)
class SixMatrixResponseScientificSeedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/six-matrix-native-stream"

    scientific_role: str
    current_root_id: str
    current_context_sha256: str
    stream_index: int
    full_seed_sha256: str
    source_original_context_sha256: str
    original_source: ObjectIdentity
    export_receipt: ObjectIdentity

    def __post_init__(self) -> None:
        if self.scientific_role not in (
            "shooting-branch", "shooting-bridge", "controlled-parent-replay",
            "controlled-noise-block", "preparation-root-identity",
            "preparation-tape", "observation-tape",
        ):
            raise ValueError("six-matrix scientific seed has another physical role")
        validate_stable_id(self.current_root_id, field_name="current_root_id")
        if type(self.stream_index) is not int or self.stream_index < 0:
            raise ValueError("six-matrix scientific stream index must be a nonnegative integer")
        for name in ("current_context_sha256", "full_seed_sha256", "source_original_context_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if not isinstance(self.original_source, ObjectIdentity) or not isinstance(self.export_receipt, ObjectIdentity):
            raise ValueError("six-matrix numerical inputs require explicit original source and export custody")

    def require_current_binding(
        self, *, scientific_role: str, current_root_id: str,
        current_context_sha256: str, stream_index: int,
    ) -> None:
        if (
            self.scientific_role != scientific_role
            or self.current_root_id != current_root_id
            or self.current_context_sha256 != current_context_sha256
            or self.stream_index != stream_index
        ):
            raise ValueError("six-matrix scientific seed input binds another current root, role, context or index")


def require_six_matrix_scientific_seed_input(
    value: SixMatrixResponseScientificSeedInput | None, *, scientific_role: str,
    current_root_id: str, current_context_sha256: str, stream_index: int,
) -> SixMatrixResponseScientificSeedInput:
    if not isinstance(value, SixMatrixResponseScientificSeedInput):
        raise ValueError("six-matrix source requires its complete typed full numerical seed input and explicit external export custody")
    value.require_current_binding(
        scientific_role=scientific_role, current_root_id=current_root_id,
        current_context_sha256=current_context_sha256, stream_index=stream_index,
    )
    return value
