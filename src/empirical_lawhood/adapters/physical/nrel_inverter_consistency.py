"""Outcome-blind independent substrate grounding NREL inverter archive source/authority boundary."""

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


class NRELArchiveReadinessDisposition(StrEnum):
    READY_FOR_AUTHORIZED_ACQUISITION = "SOURCE_ROUTE_ACCEPTED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    TERMS_UNRESOLVED = "SOURCE_TERMS_UNRESOLVED"
    SECURITY_STOP = "UNSAFE_ARCHIVE_FORMAT_STOP"


@dataclass(frozen=True, slots=True)
class NRELArchiveReadiness(CanonicalRecord):
    """source-authority readiness facts without source acquisition or protected receiver decode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-archive-readiness'

    readiness_id: str
    catalogue_route_identified: bool
    source_terms_accepted: bool
    safe_decode_route_predeclared: bool
    source_authority_present: bool
    source_bytes_present: bool
    acquisition_attempted: bool
    protected_receiver_decoded: bool
    scientific_unit_count: int
    disposition: NRELArchiveReadinessDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.readiness_id, field_name="readiness_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.acquisition_attempted or self.protected_receiver_decoded:
            raise ValueError("NREL source-authority readiness cannot acquire or decode protected receiver outcomes")
        if self.scientific_unit_count != 0:
            raise ValueError("NREL source-authority readiness cannot count as a physical unit")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("NREL source-authority readiness must remain outcome blind")
        if not self.catalogue_route_identified:
            expected = NRELArchiveReadinessDisposition.SOURCE_UNAVAILABLE
        elif not self.source_authority_present:
            expected = NRELArchiveReadinessDisposition.AUTHORITY_REQUIRED
        elif not self.source_terms_accepted:
            expected = NRELArchiveReadinessDisposition.TERMS_UNRESOLVED
        elif not self.safe_decode_route_predeclared:
            expected = NRELArchiveReadinessDisposition.SECURITY_STOP
        elif not self.source_bytes_present:
            expected = NRELArchiveReadinessDisposition.SOURCE_UNAVAILABLE
        else:
            expected = NRELArchiveReadinessDisposition.READY_FOR_AUTHORIZED_ACQUISITION
        if self.disposition is not expected:
            raise ValueError("NREL source-authority readiness disposition is not fact-derived")
        if expected is NRELArchiveReadinessDisposition.READY_FOR_AUTHORIZED_ACQUISITION:
            if self.reason_codes:
                raise ValueError("ready NREL source-authority readiness cannot retain stop reasons")
        elif not self.reason_codes or expected.value not in self.reason_codes:
            raise ValueError("stopped NREL source-authority readiness lacks its controlling reason")


def nrel_authority_required_readiness() -> NRELArchiveReadiness:
    """Return the current source-authority readiness authority stop without accessing the archive."""

    disposition = NRELArchiveReadinessDisposition.AUTHORITY_REQUIRED
    return NRELArchiveReadiness(
        readiness_id="readiness.independent-substrate-grounding.nrel-inverter-source-authority-readiness",
        catalogue_route_identified=True,
        source_terms_accepted=False,
        safe_decode_route_predeclared=False,
        source_authority_present=False,
        source_bytes_present=False,
        acquisition_attempted=False,
        protected_receiver_decoded=False,
        scientific_unit_count=0,
        disposition=disposition,
        reason_codes=(disposition.value,),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def nrel_readiness_stop_handoff(
    readiness: NRELArchiveReadiness,
) -> IndependentSubstrateTargetTerminalHandoff:
    """Project an source-authority readiness stop without treating absent bytes as physical evidence."""

    if readiness.disposition is NRELArchiveReadinessDisposition.READY_FOR_AUTHORIZED_ACQUISITION:
        raise ValueError("ready NREL source-authority readiness requires authorized acquisition, not a stop handoff")
    reasons = tuple(
        sorted(
            {
                *readiness.reason_codes,
                "NREL_ARCHIVE_NOT_ACQUIRED",
                "PHYSICAL_UNIT_UNRESOLVED",
                "RETROSPECTIVE_PHYSICAL_ANALYSIS_NOT_ATTEMPTED",
            }
        )
    )
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id="handoff.independent-substrate-grounding.nrel-inverter-source-authority-readiness-stop",
        slot=IndependentSubstrateTargetKind.NREL_INVERTER,
        evidence_world=EvidenceWorld.RETROSPECTIVE_PHYSICAL_ARCHIVE,
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
        physical_consistency="UNEVALUABLE",
        prediction_receipt_sha256=None,
        match_receipt_sha256=readiness.fingerprint(),
        maximum_claim_ceiling=("OUTCOME_BLIND_source-authority readiness_AUTHORITY_STOP_NO_PHYSICAL_CONSISTENCY_CLAIM"),
        reason_codes=reasons,
    )


__all__ = [
    "NRELArchiveReadinessDisposition",
    'NRELArchiveReadiness',
    "nrel_authority_required_readiness",
    "nrel_readiness_stop_handoff",
]
