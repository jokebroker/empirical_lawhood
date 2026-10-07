"""Outcome-blind independent substrate grounding Grid2Op follow-up/source readiness boundary.

This module does not emulate Grid2Op.  It records whether the exact fresh,
independent substrate grounding-only package/backend/chronic route exists before any target outcome can
be generated.  A failed route is a typed pre-issue stop and cannot be converted
into a scientific counterexample or silently replaced by the SCRAC child.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateIndependenceClass, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff
from empirical_lawhood.adapters.methods.structural_recurrence import TargetLevel
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)


class Grid2OpReadinessDisposition(StrEnum):
    READY_FOR_SOURCE_QUALIFICATION = "READY_FOR_SOURCE_QUALIFICATION"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    BACKEND_UNAVAILABLE = "GRID2OP_BACKEND_UNAVAILABLE"
    CHRONICS_UNAVAILABLE = "GRID2OP_CHRONICS_UNAVAILABLE"
    IDENTITY_COLLISION_STOP = "INDEPENDENT_RECURRENCE_IDENTITY_COLLISION_STOP"


@dataclass(frozen=True, slots=True)
class Grid2OpSourceReadiness(CanonicalRecord):
    """source readiness facts before source qualification or target issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2-op-source-readiness'

    readiness_id: str
    fresh_target_identity_disjoint: bool
    scrac_outcomes_accessed: bool
    exact_package_present: bool
    exact_backend_present: bool
    exact_chronics_present: bool
    acquisition_attempted: bool
    scientific_unit_count: int
    disposition: Grid2OpReadinessDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.readiness_id, field_name="readiness_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.acquisition_attempted:
            raise ValueError("independent substrate grounding source readiness cannot acquire Grid2Op source")
        if self.scientific_unit_count != 0:
            raise ValueError("Grid2Op source readiness cannot count as a scientific unit")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Grid2Op source readiness must remain outcome blind")
        if self.scrac_outcomes_accessed:
            expected = Grid2OpReadinessDisposition.IDENTITY_COLLISION_STOP
        elif not self.fresh_target_identity_disjoint:
            expected = Grid2OpReadinessDisposition.IDENTITY_COLLISION_STOP
        elif not self.exact_package_present:
            expected = Grid2OpReadinessDisposition.SOURCE_UNAVAILABLE
        elif not self.exact_backend_present:
            expected = Grid2OpReadinessDisposition.BACKEND_UNAVAILABLE
        elif not self.exact_chronics_present:
            expected = Grid2OpReadinessDisposition.CHRONICS_UNAVAILABLE
        else:
            expected = Grid2OpReadinessDisposition.READY_FOR_SOURCE_QUALIFICATION
        if self.disposition is not expected:
            raise ValueError("Grid2Op source readiness disposition is not fact-derived")
        if expected is Grid2OpReadinessDisposition.READY_FOR_SOURCE_QUALIFICATION:
            if self.reason_codes:
                raise ValueError("ready Grid2Op source readiness cannot retain stop reasons")
        elif not self.reason_codes or expected.value not in self.reason_codes:
            raise ValueError("stopped Grid2Op source readiness lacks its controlling reason")


def grid2op_source_unavailable_readiness() -> Grid2OpSourceReadiness:
    """Return the exact current no-acquisition source readiness state."""

    disposition = Grid2OpReadinessDisposition.SOURCE_UNAVAILABLE
    return Grid2OpSourceReadiness(
        readiness_id="readiness.independent-substrate-grounding.grid2op-source-readiness",
        fresh_target_identity_disjoint=True,
        scrac_outcomes_accessed=False,
        exact_package_present=False,
        exact_backend_present=False,
        exact_chronics_present=False,
        acquisition_attempted=False,
        scientific_unit_count=0,
        disposition=disposition,
        reason_codes=(disposition.value,),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def grid2op_readiness_stop_handoff(
    readiness: Grid2OpSourceReadiness,
) -> IndependentSubstrateTargetTerminalHandoff:
    """Project a stopped source readiness boundary without fabricating target evidence."""

    if readiness.disposition is Grid2OpReadinessDisposition.READY_FOR_SOURCE_QUALIFICATION:
        raise ValueError("ready Grid2Op source readiness requires the target lifecycle, not a stop handoff")
    reasons = tuple(
        sorted(
            {
                *readiness.reason_codes,
                "GRID2OP_TARGET_NOT_ISSUED",
                "ADMISSION_NOT_ATTEMPTED_PREREQUISITE_NOT_MET",
                "CONTROLLER_USE_NOT_ATTEMPTED_PREREQUISITE_NOT_MET",
            }
        )
    )
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id="handoff.independent-substrate-grounding.grid2op-source-readiness-stop",
        slot=IndependentSubstrateTargetKind.GRID2OP,
        evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
        attained_level=TargetLevel.MEASUREMENT_READINESS,
        independence_class=IndependentSubstrateIndependenceClass.UNEVALUABLE,
        evaluation_eligible=False,
        categorical_support=False,
        decisive_opposition=False,
        unsafe_false_admission_count=0,
        action_ontology_clock_error_count=0,
        panel_envelope_limited=False,
        structural_recurrence_restrictiveness_supported=False,
        comparator_tied_or_won=False,
        structural_recurrence_less_safe_or_exact_than_comparator=False,
        metric_topology_exchanges=(),
        physical_consistency="NOT_APPLICABLE",
        prediction_receipt_sha256=None,
        match_receipt_sha256=readiness.fingerprint(),
        maximum_claim_ceiling=(
            "OUTCOME_BLIND_source readiness_SOURCE_STOP_NO_GRID2OP_RESPONSE_OR_RECURRENCE_CLAIM"
        ),
        reason_codes=reasons,
    )


__all__ = [
    "Grid2OpReadinessDisposition",
    'Grid2OpSourceReadiness',
    "grid2op_readiness_stop_handoff",
    "grid2op_source_unavailable_readiness",
]
