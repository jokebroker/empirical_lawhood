"Outcome-visible review and prospective compilation for the fine-steering mirror."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.planning.exploration import AnomalySignal


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorFreshEvidenceDesign(CanonicalRecord):
    """Advisory fresh-evidence design; it delegates no physical authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-fresh-evidence-design'

    design_id: str
    parent_result_sha256: str
    readiness_status: ReadinessStatus
    requested_evidence_rung: str
    fresh_independent_blocks_per_action: int
    requirement_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    admission_role_ids: tuple[str, ...]
    nonclaim_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        validate_sha256(self.parent_result_sha256, field_name="parent_result_sha256")
        if self.readiness_status is not ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("fresh FSM acquisition requires facility/instrument authority")
        if self.requested_evidence_rung != "LOCAL_LAW":
            raise ValueError("Fine-steering-mirror confirmation design must request a local-law result")
        if self.fresh_independent_blocks_per_action < 2:
            raise ValueError("follow-up must support held-out within-cell recurrence")
        for field_name, values in (
            ("requirement_ids", self.requirement_ids),
            ("control_ids", self.control_ids),
            ("admission_role_ids", self.admission_role_ids),
            ("nonclaim_codes", self.nonclaim_codes),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorOutcomeVisibleReview(CanonicalRecord):
    """Immutable R7A detector output and separately authorizable R7B design."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-outcome-visible-review'

    review_id: str
    parent_result_sha256: str
    detector_registry_fingerprint: str
    diagnostic_projection_fingerprint: str
    signals: tuple[AnomalySignal, ...]
    complete_detector_count: int
    interpretation_codes: tuple[str, ...]
    fresh_evidence_design: FineSteeringMirrorFreshEvidenceDesign
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.review_id, field_name="review_id")
        for name, value in (
            ("parent_result_sha256", self.parent_result_sha256),
            ("detector_registry_fingerprint", self.detector_registry_fingerprint),
            ("diagnostic_projection_fingerprint", self.diagnostic_projection_fingerprint),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(self.signals, attribute="signal_id", field_name="signals")
        if self.complete_detector_count != 11:
            raise ValueError("R7A review must retain the complete detector family")
        require_sorted_unique_strings(
            self.interpretation_codes,
            field_name="interpretation_codes",
            allow_empty=False,
        )
        if self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("post-hoc review cannot promote revealed evidence")
