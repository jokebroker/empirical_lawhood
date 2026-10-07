"""Compact terminal projection for the independent substrate grounding FreeGSNKE target.

The terminal record retains exact scientific receipt hashes and the complete
target validation record, then emits the small cross-target handoff consumed by
independent substrate cross-target.  It does not reopen simulator payloads or average target-native
metrics across substrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentImplementationDossier, IndependentSubstrateIndependenceClass, IndependentSubstrateRestrictivenessAxis, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff
from empirical_lawhood.adapters.methods.structural_recurrence import TargetLevel
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)

from .target_validation import FreeGsnkeTargetValidation


def _attained_level(validation: FreeGsnkeTargetValidation) -> TargetLevel:
    if not validation.parent_admission_evaluation.evaluation_eligible:
        return TargetLevel.LAW_QUALIFICATION
    result = validation.target_result
    if (
        validation.prospective_validation_child_issued
        and result is not None
        and result.policy_action_executed
        and validation.structural_recurrence_validation_evidence is not None
    ):
        return TargetLevel.PROSPECTIVE_USE
    return TargetLevel.ADMISSION


def _terminal_facts(
    validation: FreeGsnkeTargetValidation,
) -> tuple[bool, bool, int, int, bool]:
    admission_evaluation = validation.parent_admission_evaluation
    match = validation.target_match
    panel_limited = admission_evaluation.panel_envelope_limited or (
        validation.prospective_validation_reduction is not None and validation.prospective_validation_reduction.panel_envelope_limited
    )
    clock_errors = admission_evaluation.action_ontology_clock_error_count + (
        validation.prospective_validation_reduction.action_ontology_error_count
        if validation.prospective_validation_reduction is not None
        else 0
    )
    categorical_support = bool(
        admission_evaluation.categorical_support
        and match is not None
        and match.passed
        and not panel_limited
        and not clock_errors
    )
    validation_mismatch = bool(
        match is not None
        and not match.validation_exact
        and admission_evaluation.evaluation_eligible
        and not panel_limited
    )
    decisive_opposition = bool(
        admission_evaluation.decisive_opposition
        or validation.validation_opposed
        or validation_mismatch
        or clock_errors
    )
    return (
        categorical_support,
        decisive_opposition,
        admission_evaluation.unsafe_false_admission_count,
        clock_errors,
        panel_limited,
    )


def _claim_ceiling(
    *,
    validation: FreeGsnkeTargetValidation,
    dossier: IndependentImplementationDossier,
) -> str:
    level = _attained_level(validation)
    support, opposition, unsafe, clocks, limited = _terminal_facts(validation)
    if dossier.classification is not IndependentSubstrateIndependenceClass.NOVEL_EXTERNAL_IMPLEMENTATION:
        return "FREEGSNKE_INDEPENDENCE_UNEVALUABLE"
    if unsafe or clocks:
        return "FREEGSNKE_DECISIVE_SAFETY_OR_ACTION_ONTOLOGY_COUNTEREVIDENCE"
    if limited:
        return f"FREEGSNKE_PANEL_LIMITED_{level.value}_EVIDENCE"
    if opposition:
        return "FREEGSNKE_WELL_RESOLVED_PREDICTIVE_COUNTEREXAMPLE"
    if support and level is TargetLevel.PROSPECTIVE_USE:
        return "FREEGSNKE_INDEPENDENT_SIMULATOR_CONTROLLER_ADMISSION_RECURRENCE_WITH_PROSPECTIVE_CONTROLLER_EVALUATION_VALIDATION"
    if support:
        return "FREEGSNKE_INDEPENDENT_SIMULATOR_CONTROLLER_ADMISSION_RECURRENCE"
    return f"FREEGSNKE_{level.value}_MIXED_OR_NONSUPPORTING"


def _project_handoff(
    *,
    handoff_id: str,
    validation: FreeGsnkeTargetValidation,
    dossier: IndependentImplementationDossier,
    prediction_receipt_sha256: str,
    target_match_receipt_sha256: str | None,
) -> IndependentSubstrateTargetTerminalHandoff:
    support, opposition, unsafe, clocks, limited = _terminal_facts(validation)
    comparator = validation.parent_admission_evaluation.comparator_adjudication
    reasons = {
        *validation.reason_codes,
        *dossier.reason_codes,
        *(("TARGET_MATCH_RECEIPT_ABSENT",) if validation.target_match is None else ()),
        _claim_ceiling(validation=validation, dossier=dossier),
    }
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id=handoff_id,
        slot=IndependentSubstrateTargetKind.FREEGSNKE,
        evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
        attained_level=_attained_level(validation),
        independence_class=dossier.classification,
        evaluation_eligible=validation.parent_admission_evaluation.evaluation_eligible,
        categorical_support=support,
        decisive_opposition=opposition,
        unsafe_false_admission_count=unsafe,
        action_ontology_clock_error_count=clocks,
        panel_envelope_limited=limited,
        structural_recurrence_restrictiveness_supported=(
            comparator.disposition is IndependentSubstrateRestrictivenessAxis.SUPPORTED
        ),
        comparator_tied_or_won=comparator.comparator_tied_or_won,
        structural_recurrence_less_safe_or_exact_than_comparator=(
            comparator.structural_recurrence_less_safe_or_exact_than_comparator
        ),
        metric_topology_exchanges=(validation.parent_admission_evaluation.metric_topology_analysis.exchanges),
        physical_consistency="NOT_APPLICABLE",
        prediction_receipt_sha256=prediction_receipt_sha256,
        match_receipt_sha256=target_match_receipt_sha256,
        maximum_claim_ceiling=_claim_ceiling(
            validation=validation,
            dossier=dossier,
        ),
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetTerminal(CanonicalRecord):
    """Receipt-bound target terminal result and compact independent substrate cross-target handoff."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-terminal'

    terminal_id: str
    validation: FreeGsnkeTargetValidation
    independence_dossier: IndependentImplementationDossier
    prediction_receipt_sha256: str
    admission_evaluation_evaluation_receipt_sha256: str
    prospective_validation_issue_receipt_sha256: str | None
    prospective_validation_evaluation_receipt_sha256: str | None
    target_match_receipt_sha256: str | None
    handoff: IndependentSubstrateTargetTerminalHandoff

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        for name in (
            "prediction_receipt_sha256",
            "admission_evaluation_evaluation_receipt_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "prospective_validation_issue_receipt_sha256",
            "prospective_validation_evaluation_receipt_sha256",
            "target_match_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_sha256(value, field_name=name)
        if self.independence_dossier.slot is not IndependentSubstrateTargetKind.FREEGSNKE:
            raise ValueError("FreeGSNKE terminal dossier names another target")
        if self.validation.prospective_validation_child_issued != (
            self.prospective_validation_issue_receipt_sha256 is not None
            and self.prospective_validation_evaluation_receipt_sha256 is not None
        ):
            raise ValueError("FreeGSNKE terminal prospective validation receipt presence differs from execution")
        if (self.validation.target_match is not None) != (
            self.target_match_receipt_sha256 is not None
        ):
            raise ValueError("FreeGSNKE terminal match receipt presence differs")
        expected = _project_handoff(
            handoff_id=self.handoff.handoff_id,
            validation=self.validation,
            dossier=self.independence_dossier,
            prediction_receipt_sha256=self.prediction_receipt_sha256,
            target_match_receipt_sha256=self.target_match_receipt_sha256,
        )
        if self.handoff != expected:
            raise ValueError("FreeGSNKE compact handoff is not terminal-record-derived")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.terminal_id, self)


def build_freegsnke_target_terminal(
    *,
    terminal_id: str,
    handoff_id: str,
    validation: FreeGsnkeTargetValidation,
    independence_dossier: IndependentImplementationDossier,
    prediction_receipt_sha256: str,
    admission_evaluation_evaluation_receipt_sha256: str,
    prospective_validation_issue_receipt_sha256: str | None,
    prospective_validation_evaluation_receipt_sha256: str | None,
    target_match_receipt_sha256: str | None,
) -> FreeGsnkeTargetTerminal:
    """Build a target terminal only after its required receipts exist."""

    for name, value in (
        ("prediction_receipt_sha256", prediction_receipt_sha256),
        ("admission_evaluation_evaluation_receipt_sha256", admission_evaluation_evaluation_receipt_sha256),
    ):
        validate_sha256(value, field_name=name)
    if independence_dossier.slot is not IndependentSubstrateTargetKind.FREEGSNKE:
        raise ValueError("FreeGSNKE terminal requires the FreeGSNKE dossier")
    if not independence_dossier.prediction_precedes_outcome_access:
        raise ValueError("FreeGSNKE terminal cannot close a contaminated prediction order")
    handoff = _project_handoff(
        handoff_id=handoff_id,
        validation=validation,
        dossier=independence_dossier,
        prediction_receipt_sha256=prediction_receipt_sha256,
        target_match_receipt_sha256=target_match_receipt_sha256,
    )
    return FreeGsnkeTargetTerminal(
        terminal_id=terminal_id,
        validation=validation,
        independence_dossier=independence_dossier,
        prediction_receipt_sha256=prediction_receipt_sha256,
        admission_evaluation_evaluation_receipt_sha256=admission_evaluation_evaluation_receipt_sha256,
        prospective_validation_issue_receipt_sha256=prospective_validation_issue_receipt_sha256,
        prospective_validation_evaluation_receipt_sha256=prospective_validation_evaluation_receipt_sha256,
        target_match_receipt_sha256=target_match_receipt_sha256,
        handoff=handoff,
    )


__all__ = [
    'FreeGsnkeTargetTerminal',
    "build_freegsnke_target_terminal",
]
