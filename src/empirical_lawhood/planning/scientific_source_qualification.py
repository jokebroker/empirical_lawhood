"""Scientific qualification after source-pipeline custody and before claims."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryApplicability, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile, SourcePipelineQualificationReceipt


class ScientificSourceGateKind(StrEnum):
    SOURCE_IDENTITY = "SOURCE_IDENTITY"
    NUMERICAL_OR_INSTRUMENT_VALIDITY = "NUMERICAL_OR_INSTRUMENT_VALIDITY"
    PREPARATION = "PREPARATION"
    STATIONARITY_OR_RECURRENCE = "STATIONARITY_OR_RECURRENCE"
    OBSERVATION_OPERATOR = "OBSERVATION_OPERATOR"
    PHYSICAL_UNIT_COMPLETENESS = "PHYSICAL_UNIT_COMPLETENESS"
    CAUSAL_CUTOFF = "CAUSAL_CUTOFF"


class ScientificSourceInvalidUnitPolicy(StrEnum):
    OPPOSES = "OPPOSES"
    UNEVALUABLE = "UNEVALUABLE"


class ScientificSourceQualificationStopKind(StrEnum):
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    SOURCE_NOT_READY = "SOURCE_NOT_READY"
    RESOURCE_NOT_QUALIFIED = "RESOURCE_NOT_QUALIFIED"
    INPUT_IDENTITY_MISMATCH = "INPUT_IDENTITY_MISMATCH"
    EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class ScientificSourceGateSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-gate-spec'

    gate_id: str
    gate_kind: ScientificSourceGateKind
    applicability: MetatheoryApplicability
    evaluator: MetatheoryMethodSelection | None
    required_operands: tuple[ObjectIdentity, ...]
    independent_unit_definition_id: str
    uncertainty_operation_id: str
    falsifier_ids: tuple[str, ...]
    applicability_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        validate_stable_id(
            self.independent_unit_definition_id,
            field_name="independent_unit_definition_id",
        )
        validate_stable_id(self.uncertainty_operation_id, field_name="uncertainty_operation_id")
        require_sorted_unique_ids(
            self.required_operands,
            attribute="object_id",
            field_name="required_operands",
        )
        require_sorted_unique_strings(self.falsifier_ids, field_name="falsifier_ids")
        require_sorted_unique_strings(
            self.applicability_reason_codes,
            field_name="applicability_reason_codes",
        )
        if self.applicability is MetatheoryApplicability.REQUIRED:
            if self.evaluator is None or not self.required_operands:
                raise ValueError("required scientific source gate lacks evaluator or operands")
            if self.applicability_reason_codes:
                raise ValueError("required scientific source gate cannot carry N/A reasons")
        elif (
            self.evaluator is not None
            or self.required_operands
            or not self.applicability_reason_codes
        ):
            raise ValueError("inapplicable scientific source gate must remain nonexecuting")


@dataclass(frozen=True, slots=True)
class ScientificSourceQualificationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-qualification-spec'

    spec_id: str
    source_pipeline_profile: ObjectIdentity
    source_pipeline_qualification: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    source_identity: ObjectIdentity
    source_materialization: ObjectIdentity
    gates: tuple[ScientificSourceGateSpec, ...]
    physical_unit_ids: tuple[str, ...]
    expected_physical_unit_count: int
    invalid_unit_policy: ScientificSourceInvalidUnitPolicy
    preparation: ObjectIdentity
    observation_operator: ObjectIdentity
    numerical_view: ObjectIdentity
    causal_cutoff: ObjectIdentity
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if self.source_pipeline_profile.object_schema != SourcePipelineProfile.SCHEMA:
            raise ValueError("scientific source spec requires a source pipeline profile")
        if (
            self.source_pipeline_qualification.object_schema
            != SourcePipelineQualificationReceipt.SCHEMA
        ):
            raise ValueError("scientific source spec requires pipeline qualification custody")
        if self.evidence_profile_selection.object_schema != EvidenceProfileSelection.SCHEMA:
            raise ValueError("scientific source spec requires an evidence profile selection")
        require_sorted_unique_ids(self.gates, attribute="gate_id", field_name="gates")
        if {value.gate_kind for value in self.gates} != set(ScientificSourceGateKind):
            raise ValueError("scientific source spec must declare every gate kind exactly once")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        if type(
            self.expected_physical_unit_count
        ) is not int or self.expected_physical_unit_count != len(self.physical_unit_ids):
            raise ValueError("scientific source expected unit count differs from its roster")
        required_visibility = inherited_visibility((), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("scientific source spec visibility understates outcome access")


@dataclass(frozen=True, slots=True)
class ScientificSourceGateEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-gate-evidence'

    evidence_id: str
    qualification_spec: ObjectIdentity
    gate_spec: ObjectIdentity
    input_operands: tuple[ObjectIdentity, ...]
    method: MetatheoryMethodSelection
    observed_physical_unit_ids: tuple[str, ...]
    eligible_physical_unit_ids: tuple[str, ...]
    invalid_physical_unit_ids: tuple[str, ...]
    metrics: tuple[NamedDecimal, ...]
    publication: ObjectIdentity
    recovery: ObjectIdentity
    evidence_links: tuple[ObjectIdentity, ...]
    disposition: MetatheoryCellDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if self.qualification_spec.object_schema != ScientificSourceQualificationSpec.SCHEMA:
            raise ValueError("gate evidence names another qualification schema")
        if self.gate_spec.object_schema != ScientificSourceGateSpec.SCHEMA:
            raise ValueError("gate evidence names another gate schema")
        require_sorted_unique_ids(
            self.input_operands,
            attribute="object_id",
            field_name="input_operands",
        )
        for name, values in (
            ("observed_physical_unit_ids", self.observed_physical_unit_ids),
            ("eligible_physical_unit_ids", self.eligible_physical_unit_ids),
            ("invalid_physical_unit_ids", self.invalid_physical_unit_ids),
        ):
            require_sorted_unique_strings(
                values, field_name=name, allow_empty=name != "observed_physical_unit_ids"
            )
        if set(self.eligible_physical_unit_ids) & set(self.invalid_physical_unit_ids):
            raise ValueError("eligible and invalid scientific source units overlap")
        if set(self.eligible_physical_unit_ids) | set(self.invalid_physical_unit_ids) != set(
            self.observed_physical_unit_ids
        ):
            raise ValueError("scientific source gate evidence does not account for every unit")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MetatheoryCellDisposition.NOT_APPLICABLE:
            raise ValueError("nonexecuting gate applicability cannot be authored as evidence")
        if self.disposition is MetatheoryCellDisposition.SUPPORTED:
            if self.invalid_physical_unit_ids or self.reason_codes:
                raise ValueError("supported source gate cannot retain invalid units or reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported source gate evidence requires reasons")


@dataclass(frozen=True, slots=True)
class ScientificSourceGateResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-gate-result'

    result_id: str
    qualification_spec: ObjectIdentity
    gate_spec: ObjectIdentity
    evidence: ObjectIdentity | None
    disposition: MetatheoryCellDisposition
    decisive_witness_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.qualification_spec.object_schema != ScientificSourceQualificationSpec.SCHEMA:
            raise ValueError("source gate result names another qualification schema")
        if self.gate_spec.object_schema != ScientificSourceGateSpec.SCHEMA:
            raise ValueError("source gate result names another gate schema")
        require_sorted_unique_strings(
            self.decisive_witness_ids,
            field_name="decisive_witness_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MetatheoryCellDisposition.NOT_APPLICABLE:
            if self.evidence is not None or not self.reason_codes:
                raise ValueError("N/A source gate result must be nonexecuting and reasoned")
        elif self.evidence is None:
            raise ValueError("applicable source gate result requires exact evidence")
        elif self.disposition is MetatheoryCellDisposition.SUPPORTED:
            if self.reason_codes:
                raise ValueError("supported source gate result cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported source gate result requires reasons")


@dataclass(frozen=True, slots=True)
class ScientificSourceQualificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-qualification-result'

    result_id: str
    qualification_spec: ObjectIdentity
    source_pipeline_qualification: ObjectIdentity
    gate_evidence: tuple[ScientificSourceGateEvidence, ...]
    gate_results: tuple[ScientificSourceGateResult, ...]
    complete_physical_unit_count: int
    invalid_physical_unit_count: int
    eligible_physical_unit_count: int
    disposition: MetatheoryAggregateDisposition
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    evidence_links: tuple[ObjectIdentity, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.qualification_spec.object_schema != ScientificSourceQualificationSpec.SCHEMA:
            raise ValueError("source qualification result names another spec schema")
        if (
            self.source_pipeline_qualification.object_schema
            != SourcePipelineQualificationReceipt.SCHEMA
        ):
            raise ValueError("source qualification result names another pipeline receipt")
        require_sorted_unique_ids(
            self.gate_evidence,
            attribute="evidence_id",
            field_name="gate_evidence",
        )
        require_sorted_unique_ids(
            self.gate_results,
            attribute="result_id",
            field_name="gate_results",
        )
        for name in (
            "complete_physical_unit_count",
            "invalid_physical_unit_count",
            "eligible_physical_unit_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.invalid_physical_unit_count + self.eligible_physical_unit_count
            > self.complete_physical_unit_count
        ):
            raise ValueError("source qualification unit counts exceed complete units")
        if self.disposition not in {
            MetatheoryAggregateDisposition.SUPPORTED,
            MetatheoryAggregateDisposition.OPPOSED,
            MetatheoryAggregateDisposition.UNEVALUABLE,
        }:
            raise ValueError("source qualification aggregate uses an invalid disposition")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        required_visibility = inherited_visibility((), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("source qualification visibility understates outcome access")


@dataclass(frozen=True, slots=True)
class ScientificSourceQualificationStop(CanonicalRecord):
    """Operational stop that constructs no scientific gate result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/scientific-source-qualification-stop'

    stop_id: str
    qualification_spec: ObjectIdentity
    stop_kind: ScientificSourceQualificationStopKind
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.stop_id, field_name="stop_id")
        if self.qualification_spec.object_schema != ScientificSourceQualificationSpec.SCHEMA:
            raise ValueError("source qualification stop names another spec schema")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification operational stop must remain outcome-blind")
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(
            inherited_visibility((), self.outcome_access)
        ):
            raise ValueError("source qualification stop visibility is too permissive")


__all__ = [
    'ScientificSourceGateEvidence',
    'ScientificSourceGateKind',
    'ScientificSourceGateResult',
    'ScientificSourceGateSpec',
    'ScientificSourceInvalidUnitPolicy',
    'ScientificSourceQualificationResult',
    'ScientificSourceQualificationSpec',
    'ScientificSourceQualificationStopKind',
    'ScientificSourceQualificationStop',
]
