"""Outcome-blind construct nomination-authoring boundary closeout that unlocks metadata nomination only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .conformance import TargetConstructValidationConformanceReport


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNominationAuthoringManifest(CanonicalRecord):
    """construct nomination-authoring boundary result: method closed, metadata nomination unlocked, issue still locked."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-nomination-authoring-manifest'

    manifest_id: str
    donor_binding: ObjectIdentity
    relation_codebook: ObjectIdentity
    method_question_freeze: ObjectIdentity
    conformance_report: ObjectIdentity
    conformance_case_count: int
    conformance_all_passed: bool
    maximum_candidate_count: int
    allowed_nomination_output_ids: tuple[str, ...]
    metadata_nomination_unlocked: bool
    source_acquisition_unlocked: bool
    target_issue_unlocked: bool
    excluded_scope_ids: tuple[str, ...]
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        require_sorted_unique_strings(
            self.allowed_nomination_output_ids,
            field_name="allowed_nomination_output_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.excluded_scope_ids,
            field_name="excluded_scope_ids",
            allow_empty=False,
        )
        if self.conformance_case_count != 11 or not self.conformance_all_passed:
            raise ValueError("nomination authoring requires the exact passed truth-known construct conformance roster")
        if self.maximum_candidate_count != 5:
            raise ValueError("nomination authoring must retain the five-candidate bound")
        if not self.metadata_nomination_unlocked:
            raise ValueError("construct nomination-authoring boundary must unlock metadata-only nomination")
        if self.source_acquisition_unlocked or self.target_issue_unlocked:
            raise ValueError("construct nomination-authoring boundary cannot authorize source acquisition or target issue")
        required_exclusions = {
            "cpaim",
            "nature-publication",
            "physical-claim",
            "prospective-controller-validation",
        }
        if not required_exclusions.issubset(self.excluded_scope_ids):
            raise ValueError("construct nomination-authoring boundary reactivates deferred scope")
        if self.protected_outcome_access_count:
            raise ValueError("construct nomination-authoring boundary cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct nomination-authoring boundary must remain outcome-blind")


def freeze_nomination_authoring(
    *,
    donor_binding: ObjectIdentity,
    relation_codebook: ObjectIdentity,
    method_question_freeze: ObjectIdentity,
    conformance: TargetConstructValidationConformanceReport,
) -> TargetConstructValidationNominationAuthoringManifest:
    if conformance.method_question_freeze != method_question_freeze:
        raise ValueError("truth-known construct conformance report and method question freeze differ")
    if not conformance.all_passed:
        raise ValueError("METHOD_CONFORMANCE_FAILED")
    return TargetConstructValidationNominationAuthoringManifest(
        manifest_id="target-construct-validation.nomination-authoring-manifest",
        donor_binding=donor_binding,
        relation_codebook=relation_codebook,
        method_question_freeze=method_question_freeze,
        conformance_report=ObjectIdentity.from_record(conformance.report_id, conformance),
        conformance_case_count=len(conformance.case_results),
        conformance_all_passed=conformance.all_passed,
        maximum_candidate_count=5,
        allowed_nomination_output_ids=(
            "candidate-registry",
            "contamination-ledger",
            "nomination-result",
            "source-resource-authority-envelope",
            "target-native-dossier",
        ),
        metadata_nomination_unlocked=True,
        source_acquisition_unlocked=False,
        target_issue_unlocked=False,
        excluded_scope_ids=(
            "cpaim",
            "nature-publication",
            "physical-claim",
            "prospective-controller-validation",
            "scale-covariance",
        ),
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
