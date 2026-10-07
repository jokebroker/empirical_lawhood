"""Pluggable finite-grid reachable/viable geometry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

from empirical_lawhood.kernel.admission import (
    AdmissionSet,
    ReachabilityResult,
    ReachabilityStatus,
    validate_reachability_against_admission,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.models import ViewModelSetSpec
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.status import AdmissionStatus, ReadinessStatus
from empirical_lawhood.planning.geometry import (
    AdmissionComparison,
    ReachabilityCellStatus,
    ReachabilityComparison,
    ReachabilityEvaluationSpec,
    ReachabilityMethodKind,
    validate_model_members,
)

from .services import _require_known_evidence


class ReachabilityMethod(Protocol):
    capability_key: ClassVar[str]
    capability_version: ClassVar[str]
    method_kind: ClassVar[ReachabilityMethodKind]

    def evaluate(
        self,
        *,
        admission: AdmissionComparison,
        model_set: ViewModelSetSpec,
        spec: ReachabilityEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> ReachabilityComparison: ...


class FiniteGridReachability:
    capability_key = "geometry.reachability.finite-grid"
    capability_version = "1.0.0"
    method_kind = ReachabilityMethodKind.FINITE_GRID

    def evaluate(
        self,
        *,
        admission: AdmissionComparison,
        model_set: ViewModelSetSpec,
        spec: ReachabilityEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> ReachabilityComparison:
        self._validate(admission, model_set, spec, evidence_links)
        nominal = self._result(
            admission=admission.nominal,
            model_set=model_set,
            spec=spec,
            evidence_links=evidence_links,
            member_ids=(spec.nominal_model_member_id,),
            reachability_id=f"reachability.{spec.evaluation_id}.nominal",
        )
        robust = self._result(
            admission=admission.robust,
            model_set=model_set,
            spec=spec,
            evidence_links=evidence_links,
            member_ids=spec.model_member_ids,
            reachability_id=f"reachability.{spec.evaluation_id}.robust",
        )
        disagreement = tuple(
            sorted(set(nominal.reachable_cell_ids).symmetric_difference(robust.reachable_cell_ids))
        )
        return ReachabilityComparison(
            comparison_id=f"reachability-comparison.{spec.evaluation_id}",
            evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
            nominal=nominal,
            robust=robust,
            structurally_stable=(
                not disagreement and nominal.viable_direction_rank == robust.viable_direction_rank
            ),
            disagreement_cell_ids=disagreement,
        )

    def _validate(
        self,
        admission: AdmissionComparison,
        model_set: ViewModelSetSpec,
        spec: ReachabilityEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> None:
        if spec.admission_comparison != ObjectIdentity.from_record(
            admission.comparison_id, admission
        ):
            raise ValueError("reachability binds another admission comparison")
        if spec.model_set != ObjectIdentity.from_record(model_set.model_set_id, model_set):
            raise ValueError("reachability binds another plausible model set")
        validate_model_members(model_set, spec.model_member_ids)
        if spec.numerical_view_ids != model_set.member_view_ids:
            raise ValueError("reachability numerical views differ from its frozen model set")
        if spec.method_kind is not self.method_kind or spec.method_key != self.capability_key:
            raise ValueError("reachability configuration selects another registered method")
        cell_ids = tuple(cell.cell_id for cell in admission.robust.cells)
        chart_ids = {cell.chart_id for cell in admission.robust.cells}
        if not set(spec.action_chart_ids).issubset(chart_ids):
            raise ValueError("reachability action chart lies outside receiver admission")
        expected = {
            f"{cell_id}.{member}" for cell_id in cell_ids for member in spec.model_member_ids
        }
        observed = {assessment.coordinate_id for assessment in spec.assessments}
        if observed != expected:
            raise ValueError("reachability assessment grid is incomplete or contains extras")
        _require_known_evidence(
            (
                evidence_id
                for assessment in spec.assessments
                for evidence_id in assessment.evidence_link_ids
            ),
            evidence_links,
        )

    @staticmethod
    def _result(
        *,
        admission: AdmissionSet,
        model_set: ViewModelSetSpec,
        spec: ReachabilityEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
        member_ids: tuple[str, ...],
        reachability_id: str,
    ) -> ReachabilityResult:
        reachable: list[str] = []
        ranks: list[int] = []
        unevaluable = False
        for cell in admission.cells:
            if not cell.admitted:
                continue
            assessments = tuple(
                assessment
                for assessment in spec.assessments
                if assessment.admission_cell_id == cell.cell_id
                and assessment.model_member_id in member_ids
            )
            statuses = {assessment.status for assessment in assessments}
            if statuses == {ReachabilityCellStatus.REACHABLE}:
                reachable.append(cell.cell_id)
                ranks.append(min(item.viable_direction_rank for item in assessments))
            elif ReachabilityCellStatus.UNEVALUABLE in statuses:
                unevaluable = True
        status = FiniteGridReachability._status(admission, spec, tuple(reachable), unevaluable)
        if status in {ReachabilityStatus.REACHABLE, ReachabilityStatus.PARTIAL}:
            reachable_ids = tuple(sorted(reachable))
            viable_rank = min(ranks)
        else:
            reachable_ids = ()
            viable_rank = 0
        result = ReachabilityResult(
            reachability_id=reachability_id,
            admission=ObjectIdentity.from_record(admission.admission_id, admission),
            initial_set_id=spec.initial_set_id,
            action_chart_ids=spec.action_chart_ids,
            dynamics_law_ids=spec.dynamics_law_ids,
            horizon=spec.horizon,
            constraint_ids=spec.constraint_ids,
            numerical_view_ids=spec.numerical_view_ids,
            structural_convergence=spec.structural_convergence,
            computability=spec.computability,
            reachable_cell_ids=reachable_ids,
            viable_direction_rank=viable_rank,
            status=status,
            evidence_links=evidence_links,
            evidence_ceiling=EvidenceCeiling.lowest(
                spec.evidence_ceiling, admission.evidence_ceiling, model_set.evidence_ceiling
            ),
            visibility_ceiling=VisibilityCeiling.most_restrictive(
                spec.visibility_ceiling,
                admission.visibility_ceiling,
                model_set.visibility_ceiling,
            ),
        )
        validate_reachability_against_admission(admission, result)
        return result

    @staticmethod
    def _status(
        admission: AdmissionSet,
        spec: ReachabilityEvaluationSpec,
        reachable: tuple[str, ...],
        unevaluable: bool,
    ) -> ReachabilityStatus:
        if spec.computability.readiness is not ReadinessStatus.READY:
            return ReachabilityStatus.COMPUTABILITY_BOUNDARY
        if spec.structural_convergence.status is not ObligationStatus.SATISFIED:
            return ReachabilityStatus.UNEVALUABLE
        if admission.status is AdmissionStatus.UNEVALUABLE:
            return ReachabilityStatus.UNEVALUABLE
        if admission.status is AdmissionStatus.EMPTY:
            return ReachabilityStatus.EMPTY
        if len(reachable) == len(admission.admitted_cell_ids):
            return ReachabilityStatus.REACHABLE
        if reachable:
            return ReachabilityStatus.PARTIAL
        if unevaluable:
            return ReachabilityStatus.UNEVALUABLE
        return ReachabilityStatus.EMPTY


@dataclass(frozen=True, slots=True)
class ReachabilityRegistry:
    methods: tuple[ReachabilityMethod, ...]

    def __post_init__(self) -> None:
        kinds = tuple(method.method_kind for method in self.methods)
        keys = tuple(method.capability_key for method in self.methods)
        if len(set(kinds)) != len(kinds) or len(set(keys)) != len(keys):
            raise ValueError("reachability method registry contains duplicates")
        if set(kinds) != set(ReachabilityMethodKind):
            raise ValueError("reachability registry does not cover every method kind")

    def resolve(self, kind: ReachabilityMethodKind) -> ReachabilityMethod:
        for method in self.methods:
            if method.method_kind is kind:
                return method
        raise KeyError(kind)


def default_reachability_registry() -> ReachabilityRegistry:
    return ReachabilityRegistry(methods=(cast(ReachabilityMethod, FiniteGridReachability()),))
