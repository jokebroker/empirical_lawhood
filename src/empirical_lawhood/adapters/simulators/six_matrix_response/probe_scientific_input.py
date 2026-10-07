"""Explicit probe seed input with separate original and current custody.

The caller supplies an externally verified original-to-current mapping. This
record does not authenticate that mapping or transfer historical qualification.
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256


@dataclass(frozen=True, slots=True)
class MatrixResponseProbeScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/matrix-response-probe"

    original_config_sha256: str
    target_config_sha256: str
    scientific_seed_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "original_config_sha256", "target_config_sha256", "scientific_seed_sha256"
        ):
            validate_sha256(getattr(self, name), field_name=name)

    def require_config_custody(
        self, *, original_config_sha256: str, target_config_sha256: str
    ) -> None:
        if (
            self.original_config_sha256 != original_config_sha256
            or self.target_config_sha256 != target_config_sha256
        ):
            raise ValueError("matrix response probe scientific input configuration custody differs")
