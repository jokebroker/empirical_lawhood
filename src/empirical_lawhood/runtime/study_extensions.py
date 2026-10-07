"""Shared identities for extension-aware candidate and issue custody."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.experiment_entry import ProposedStudyExtension


@dataclass(frozen=True, slots=True)
class StudyExtensionMaterializationReceipt(CanonicalRecord):
    """Exact outcome-blind bytes and decoder observed by the public compiler."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/study-extension-materialization-receipt'

    receipt_id: str
    extension_id: str
    proposed_extension: ObjectIdentity
    payload: ObjectIdentity
    byte_count: int
    physical_sha256: str
    decoder_registration: ObjectIdentity
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.extension_id, field_name="extension_id")
        if self.proposed_extension.object_schema != ProposedStudyExtension.SCHEMA:
            raise ValueError("extension materialization binds another proposal schema")
        if self.byte_count < 1:
            raise ValueError("extension materialization must be nonempty")
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.payload.object_fingerprint != self.physical_sha256:
            raise ValueError("extension materialization physical/logical digest differs")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("extension materialization must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("extension materialization must remain prospective")


__all__ = ['StudyExtensionMaterializationReceipt']
