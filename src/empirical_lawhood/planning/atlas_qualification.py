"""Uncertainty-qualified finite chart and atlas companion records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeResult
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection, MetatheoryPredictiveLevel


class ChartSupportLevel(StrEnum):
    RESPONSE_SPAN = "RESPONSE_SPAN"
    TRANSITION_NEIGHBOURHOOD = "TRANSITION_NEIGHBOURHOOD"
    RECURRENT_LOCAL_CHART = "RECURRENT_LOCAL_CHART"


class AtlasLawInputKind(StrEnum):
    LAW_ATLAS = "LAW_ATLAS"
    LAW_OBSTRUCTION = "LAW_OBSTRUCTION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class AtlasQualificationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-qualification-spec'

    spec_id: str
    system_id: str
    evidence_world_id: str
    relation_id: str
    chart_id: str
    law_input_kind: AtlasLawInputKind
    law_or_obstruction: ObjectIdentity | None
    coordinate_challenge_result: ObjectIdentity
    coordinate_cell_ids: tuple[str, ...]
    gate_owner_ids: tuple[str, ...]
    physical_unit_ids: tuple[str, ...]
    n: int
    alpha: NamedDecimal
    delta: NamedDecimal
    response_span_method: MetatheoryMethodSelection
    transition_method: MetatheoryMethodSelection
    uncertainty_method: MetatheoryMethodSelection
    required_predictive_levels: tuple[MetatheoryPredictiveLevel, ...]
    boundary_containment_rule_id: str
    ambiguity_width_rule_id: str
    recurrence_rule_id: str
    time_ordering_rule_id: str | None
    censoring_rule_id: str | None
    falsifier_ids: tuple[str, ...]
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("spec_id", "system_id", "evidence_world_id", "relation_id", "chart_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.law_input_kind is AtlasLawInputKind.NOT_APPLICABLE:
            if self.law_or_obstruction is not None:
                raise ValueError("N/A atlas qualification cannot bind law output")
        elif self.law_or_obstruction is None:
            raise ValueError("atlas qualification requires its declared law/obstruction identity")
        if self.coordinate_challenge_result.object_schema != CoordinateChallengeResult.SCHEMA:
            raise ValueError("atlas qualification requires a coordinate challenge result")
        for name, values in (
            ("coordinate_cell_ids", self.coordinate_cell_ids),
            ("gate_owner_ids", self.gate_owner_ids),
            ("physical_unit_ids", self.physical_unit_ids),
            ("falsifier_ids", self.falsifier_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if type(self.n) is not int or self.n != len(self.physical_unit_ids):
            raise ValueError("atlas qualification n differs from complete physical units")
        if (
            tuple(sorted(set(self.required_predictive_levels), key=lambda value: value.value))
            != self.required_predictive_levels
        ):
            raise ValueError("atlas predictive levels must be sorted and unique")
        if not self.required_predictive_levels:
            raise ValueError("atlas qualification requires predictive levels")
        for name in (
            "boundary_containment_rule_id",
            "ambiguity_width_rule_id",
            "recurrence_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        dynamical = MetatheoryPredictiveLevel.DYNAMICAL in self.required_predictive_levels
        if dynamical != (
            self.time_ordering_rule_id is not None and self.censoring_rule_id is not None
        ):
            raise ValueError("dynamical atlas qualification requires ordering and censoring rules")
        for value, name in (
            (self.time_ordering_rule_id, "time_ordering_rule_id"),
            (self.censoring_rule_id, "censoring_rule_id"),
        ):
            if value is not None:
                validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class SetValuedChartRegion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/set-valued-chart-region'

    region_id: str
    coordinate_system_id: str
    support_cell_ids: tuple[str, ...]
    boundary_labels: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    uncertainty_representation: ObjectIdentity
    resolution: NamedDecimal
    artifact: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.region_id, field_name="region_id")
        validate_stable_id(self.coordinate_system_id, field_name="coordinate_system_id")
        for name, values in (
            ("support_cell_ids", self.support_cell_ids),
            ("boundary_labels", self.boundary_labels),
            ("stratum_labels", self.stratum_labels),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)


@dataclass(frozen=True, slots=True)
class ChartTransitionEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/chart-transition-evidence'

    evidence_id: str
    qualification_spec: ObjectIdentity
    predictive_level: MetatheoryPredictiveLevel
    source_cell_id: str
    target_cell_id: str
    physical_unit_ids: tuple[str, ...]
    direction_id: str
    gate_margin_vector: ObjectIdentity
    chart_region: ObjectIdentity
    response_span_supported: bool
    transition_recurrent: bool
    uncertainty_localized: bool
    ambiguity_width_within_rule: bool
    gate_labels_preserved: bool
    observations_time_ordered: bool
    first_passage_censored: bool
    ambiguity_width: NamedDecimal
    method: MetatheoryMethodSelection
    publication: ObjectIdentity
    recovery: ObjectIdentity
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if self.qualification_spec.object_schema != AtlasQualificationSpec.SCHEMA:
            raise ValueError("chart transition evidence names another qualification schema")
        for name in ("source_cell_id", "target_cell_id", "direction_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        if self.chart_region.object_schema != SetValuedChartRegion.SCHEMA:
            raise ValueError("chart transition evidence requires a set-valued region")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


@dataclass(frozen=True, slots=True)
class AtlasQualificationCellResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-qualification-cell-result'

    cell_id: str
    qualification_spec: ObjectIdentity
    predictive_level: MetatheoryPredictiveLevel
    evidence: ObjectIdentity
    response_span: MetatheoryCellDisposition
    transition_recurrence: MetatheoryCellDisposition
    uncertainty_localization: MetatheoryCellDisposition
    labelled_boundary_preservation: MetatheoryCellDisposition
    ambiguity_width: NamedDecimal
    disposition: MetatheoryCellDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class AtlasQualificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-qualification-result'

    result_id: str
    qualification_spec: ObjectIdentity
    chart_region: ObjectIdentity
    transition_evidence: tuple[ChartTransitionEvidence, ...]
    cells: tuple[AtlasQualificationCellResult, ...]
    achieved_predictive_levels: tuple[MetatheoryPredictiveLevel, ...]
    disposition: MetatheoryAggregateDisposition
    gap_or_obstruction_refs: tuple[ObjectIdentity, ...]
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    response_law_construction_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.transition_evidence,
            attribute="evidence_id",
            field_name="transition_evidence",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if len(self.transition_evidence) != len(self.cells):
            raise ValueError("atlas qualification must retain every transition cell")
        if (
            tuple(sorted(set(self.achieved_predictive_levels), key=lambda value: value.value))
            != self.achieved_predictive_levels
        ):
            raise ValueError("achieved atlas predictive levels must be sorted and unique")
        require_sorted_unique_ids(
            self.gap_or_obstruction_refs,
            attribute="object_id",
            field_name="gap_or_obstruction_refs",
        )
        if self.response_law_construction_authorized:
            raise ValueError("atlas qualification cannot construct a ResponseLaw")


__all__ = [
    'AtlasLawInputKind',
    'AtlasQualificationCellResult',
    'AtlasQualificationResult',
    'AtlasQualificationSpec',
    'ChartSupportLevel',
    'ChartTransitionEvidence',
    'SetValuedChartRegion',
]
