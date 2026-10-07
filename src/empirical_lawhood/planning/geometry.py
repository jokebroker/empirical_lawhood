"""Planning contracts for atlas, admission, and reachable geometry services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.admission import (
    AdmissionGateKind,
    AdmissionSet,
    GateStatus,
    ReachabilityResult,
)
from empirical_lawhood.kernel.atlases import ChartTransition, ResponseAtlas
from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.models import ViewModelSetSpec
from empirical_lawhood.kernel.obligations import ComputabilityEvidence, StructuralConvergenceSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import HorizonSpec


@dataclass(frozen=True, slots=True)
class AtlasDomainCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-domain-cell'

    domain_cell_id: str
    chart_id: str
    denominator_cell_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("domain_cell_id", self.domain_cell_id),
            ("chart_id", self.chart_id),
            ("denominator_cell_id", self.denominator_cell_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class AtlasAssemblySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-assembly-spec'

    assembly_id: str
    system: ObjectIdentity
    laws: tuple[ObjectIdentity, ...]
    domain_cells: tuple[AtlasDomainCell, ...]
    transitions: tuple[ChartTransition, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.assembly_id, field_name="assembly_id")
        require_sorted_unique_ids(self.laws, attribute="object_id", field_name="laws")
        require_sorted_unique_ids(
            self.domain_cells,
            attribute="domain_cell_id",
            field_name="domain_cells",
        )
        require_sorted_unique_ids(
            self.transitions,
            attribute="transition_id",
            field_name="transitions",
        )
        if not self.laws or not self.domain_cells:
            raise ValueError("atlas assembly requires laws and a declared domain")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("atlas assembly ceiling must be exactly local law")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot assemble a claim-bearing atlas")


@dataclass(frozen=True, slots=True)
class AtlasAssemblyResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-assembly-result'

    result_id: str
    assembly: ObjectIdentity
    atlas: ResponseAtlas
    overlap_domain_cell_ids: tuple[str, ...]
    uncovered_domain_cell_ids: tuple[str, ...]
    supported_transition_ids: tuple[str, ...]
    rejected_transition_ids: tuple[str, ...]
    source_law_fingerprints_preserved: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        for field_name, values in (
            ("overlap_domain_cell_ids", self.overlap_domain_cell_ids),
            ("uncovered_domain_cell_ids", self.uncovered_domain_cell_ids),
            ("supported_transition_ids", self.supported_transition_ids),
            ("rejected_transition_ids", self.rejected_transition_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        if not self.source_law_fingerprints_preserved:
            raise ValueError("atlas assembly cannot rewrite source laws")


@dataclass(frozen=True, slots=True)
class QualificationBatchAtlasAssemblySpec(CanonicalRecord):
    "Batch-aware atlas request; direct law assembly remains an exact replay surface."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/qualification-batch-atlas-assembly-spec'

    assembly_id: str
    system: ObjectIdentity
    qualification_batch: ObjectIdentity
    domain_cells: tuple[AtlasDomainCell, ...]
    transitions: tuple[ChartTransition, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.assembly_id, field_name="assembly_id")
        if self.qualification_batch.object_schema != (
            'empirical-lawhood/methods/law-qualification-batch'
        ):
            raise ValueError("batch atlas assembly requires a LawQualificationBatch identity")
        require_sorted_unique_ids(
            self.domain_cells,
            attribute="domain_cell_id",
            field_name="domain_cells",
        )
        require_sorted_unique_ids(
            self.transitions,
            attribute="transition_id",
            field_name="transitions",
        )
        if not self.domain_cells:
            raise ValueError("batch atlas assembly requires a declared domain")
        coordinates = tuple(
            (value.chart_id, value.denominator_cell_id) for value in self.domain_cells
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("batch atlas domain repeats a chart/cell coordinate")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("batch atlas assembly ceiling must be exactly local law")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot assemble a claim-bearing atlas")


class AtlasAssemblyObstructionKind(StrEnum):
    ZERO_SUPPORTED_LAWS = "ZERO_SUPPORTED_LAWS"


@dataclass(frozen=True, slots=True)
class AtlasAssemblyObstruction(CanonicalRecord):
    """Typed zero-law terminal; it deliberately contains no ResponseAtlas."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-assembly-obstruction'

    obstruction_id: str
    kind: AtlasAssemblyObstructionKind
    assembly: ObjectIdentity
    qualification_batch: ObjectIdentity
    coordinate_ids: tuple[str, ...]
    qualification_result_ids: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.obstruction_id, field_name="obstruction_id")
        for name, values in (
            ("coordinate_ids", self.coordinate_ids),
            ("qualification_result_ids", self.qualification_result_ids),
            ("evidence_link_ids", self.evidence_link_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.kind is not AtlasAssemblyObstructionKind.ZERO_SUPPORTED_LAWS:
            raise ValueError("unknown atlas assembly obstruction")


@dataclass(frozen=True, slots=True)
class BatchAtlasAssemblyResult(CanonicalRecord):
    """Successful batch-aware projection with exact negative-source accounting."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/batch-atlas-assembly-result'

    result_id: str
    assembly: ObjectIdentity
    qualification_batch: ObjectIdentity
    atlas: ResponseAtlas
    negative_coordinate_ids: tuple[str, ...]
    overlap_disposition_ids: tuple[str, ...]
    supported_transition_ids: tuple[str, ...]
    rejected_transition_ids: tuple[str, ...]
    source_law_fingerprints_preserved: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        for name, values in (
            ("negative_coordinate_ids", self.negative_coordinate_ids),
            ("overlap_disposition_ids", self.overlap_disposition_ids),
            ("supported_transition_ids", self.supported_transition_ids),
            ("rejected_transition_ids", self.rejected_transition_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if not self.source_law_fingerprints_preserved:
            raise ValueError("batch atlas assembly cannot rewrite source laws")


@dataclass(frozen=True, slots=True)
class AdmissionCandidateCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-candidate-cell'

    cell_id: str
    denominator_cell_id: str
    chart_id: str
    action_bound_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("chart_id", self.chart_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.action_bound_ids,
            field_name="action_bound_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ModelGateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/model-gate-assessment'

    assessment_id: str
    cell_id: str
    model_member_id: str
    kind: AdmissionGateKind
    status: GateStatus
    constraint_ids: tuple[str, ...]
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("cell_id", self.cell_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.constraint_ids, field_name="constraint_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        if self.status is GateStatus.PASS and self.reason_codes:
            raise ValueError("passing model gate cannot retain exclusion reasons")
        if self.status is not GateStatus.PASS and not self.reason_codes:
            raise ValueError("non-passing model gate requires exclusion reasons")

    @property
    def coordinate_id(self) -> str:
        return f"{self.cell_id}.{self.model_member_id}.{self.kind.value.lower()}"


@dataclass(frozen=True, slots=True)
class AdmissionEvaluationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-evaluation-spec'

    evaluation_id: str
    atlas: ObjectIdentity
    model_set: ObjectIdentity
    nominal_model_member_id: str
    model_member_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    candidate_cells: tuple[AdmissionCandidateCell, ...]
    assessments: tuple[ModelGateAssessment, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.nominal_model_member_id, field_name="nominal_model_member_id")
        for field_name, values in (
            ("model_member_ids", self.model_member_ids),
            ("receiver_quantity_ids", self.receiver_quantity_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if self.nominal_model_member_id not in self.model_member_ids:
            raise ValueError("nominal model is absent from the frozen plausible set")
        require_sorted_unique_ids(
            self.candidate_cells, attribute="cell_id", field_name="candidate_cells"
        )
        require_sorted_unique_ids(
            self.assessments, attribute="assessment_id", field_name="assessments"
        )
        if not self.candidate_cells:
            raise ValueError("admission evaluation requires candidate cells")
        expected = {
            f"{cell.cell_id}.{member}.{kind.value.lower()}"
            for cell in self.candidate_cells
            for member in self.model_member_ids
            for kind in AdmissionGateKind
        }
        observed = {assessment.coordinate_id for assessment in self.assessments}
        if observed != expected:
            raise ValueError("admission assessment grid is incomplete or contains extras")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("admission evaluation ceiling must be exactly admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot establish receiver admission")


@dataclass(frozen=True, slots=True)
class AdmissionComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-comparison'

    comparison_id: str
    evaluation: ObjectIdentity
    nominal: AdmissionSet
    robust: AdmissionSet
    structurally_stable: bool
    disagreement_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        require_sorted_unique_strings(
            self.disagreement_cell_ids, field_name="disagreement_cell_ids"
        )
        observed_stability = self.nominal.admitted_cell_ids == self.robust.admitted_cell_ids
        if self.structurally_stable != observed_stability:
            raise ValueError("admission structural-stability flag differs from the intersections")
        observed_disagreement = tuple(
            sorted(
                set(self.nominal.admitted_cell_ids).symmetric_difference(
                    self.robust.admitted_cell_ids
                )
            )
        )
        if self.disagreement_cell_ids != observed_disagreement:
            raise ValueError("admission disagreement cells differ from nominal/robust results")


class ReachabilityMethodKind(StrEnum):
    FINITE_GRID = "FINITE_GRID"


class ReachabilityCellStatus(StrEnum):
    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ModelReachabilityAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/model-reachability-assessment'

    assessment_id: str
    admission_cell_id: str
    model_member_id: str
    status: ReachabilityCellStatus
    viable_direction_rank: int
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("admission_cell_id", self.admission_cell_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.viable_direction_rank < 0:
            raise ValueError("viable direction rank must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        if self.status is ReachabilityCellStatus.REACHABLE:
            if self.viable_direction_rank == 0 or self.reason_codes:
                raise ValueError("reachable cell needs viable directions and no failure")
        elif self.viable_direction_rank != 0 or not self.reason_codes:
            raise ValueError("unreachable/unevaluable cell needs zero rank and reasons")

    @property
    def coordinate_id(self) -> str:
        return f"{self.admission_cell_id}.{self.model_member_id}"


@dataclass(frozen=True, slots=True)
class ReachabilityEvaluationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/reachability-evaluation-spec'

    evaluation_id: str
    admission_comparison: ObjectIdentity
    model_set: ObjectIdentity
    nominal_model_member_id: str
    model_member_ids: tuple[str, ...]
    method_kind: ReachabilityMethodKind
    method_key: str
    method_version: str
    initial_set_id: str
    action_chart_ids: tuple[str, ...]
    dynamics_law_ids: tuple[str, ...]
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    structural_convergence: StructuralConvergenceSpec
    computability: ComputabilityEvidence
    assessments: tuple[ModelReachabilityAssessment, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("nominal_model_member_id", self.nominal_model_member_id),
            ("method_key", self.method_key),
            ("initial_set_id", self.initial_set_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        for field_name, values in (
            ("model_member_ids", self.model_member_ids),
            ("action_chart_ids", self.action_chart_ids),
            ("dynamics_law_ids", self.dynamics_law_ids),
            ("constraint_ids", self.constraint_ids),
            ("numerical_view_ids", self.numerical_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if self.nominal_model_member_id not in self.model_member_ids:
            raise ValueError("nominal reachability model is absent from the plausible set")
        require_sorted_unique_ids(
            self.assessments, attribute="assessment_id", field_name="assessments"
        )
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("reachability evaluation ceiling must be exactly admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot establish reachability")


@dataclass(frozen=True, slots=True)
class ReachabilityComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/reachability-comparison'

    comparison_id: str
    evaluation: ObjectIdentity
    nominal: ReachabilityResult
    robust: ReachabilityResult
    structurally_stable: bool
    disagreement_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        require_sorted_unique_strings(
            self.disagreement_cell_ids, field_name="disagreement_cell_ids"
        )
        observed_stability = (
            self.nominal.reachable_cell_ids == self.robust.reachable_cell_ids
            and self.nominal.viable_direction_rank == self.robust.viable_direction_rank
        )
        if self.structurally_stable != observed_stability:
            raise ValueError("reachability structural-stability flag differs")
        observed_disagreement = tuple(
            sorted(
                set(self.nominal.reachable_cell_ids).symmetric_difference(
                    self.robust.reachable_cell_ids
                )
            )
        )
        if self.disagreement_cell_ids != observed_disagreement:
            raise ValueError("reachability disagreement cells differ from its results")


def validate_model_members(spec: ViewModelSetSpec, declared_ids: tuple[str, ...]) -> None:
    if spec.member_view_ids != declared_ids:
        raise ValueError("evaluation model members differ from the frozen ModelSetSpec")
