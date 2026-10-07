"Compact canonical retained preparation screen phase identity; arrays remain external transports."

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256


@dataclass(frozen=True, slots=True)
class RetainedPreparationScreenManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/retained-entry-readiness/retained-preparation-screen-manifest'
    phase: str
    plan_sha256: str
    operands_sha256: str
    source_sha256: str
    authority_grant: str
    native_calls: int = 0
    issue_calls: int = 0
    scheduler_calls: int = 0

    def __post_init__(self) -> None:
        for value in (self.plan_sha256, self.operands_sha256, self.source_sha256):
            validate_sha256(value, field_name="digest")
        if self.phase != "retained-preparation-screen" or (self.native_calls, self.issue_calls, self.scheduler_calls) != (
            0,
            0,
            0,
        ):
            raise ValueError("Retained preparation screen permits retained analysis only")
        if self.authority_grant != "empirical-lawhood.owner-standing-authority.2026-09-06":
            raise ValueError("Unknown authority grant")
