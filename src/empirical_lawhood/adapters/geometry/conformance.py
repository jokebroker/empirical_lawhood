"Atlas, admission, and reachability reference conformance."

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.methods.baselines import NonlinearLocalIdentifier
from empirical_lawhood.adapters.methods.conformance import (
    build_identification_config,
    identification_cases,
)
from empirical_lawhood.adapters.methods.conformance_payloads import EphemeralCandidatePayloadPlane
from empirical_lawhood.adapters.methods.identification import build_response_method_identification_service
from empirical_lawhood.kernel.admission import (
    AdmissionGateKind,
    GateStatus,
    ReachabilityStatus,
)
from empirical_lawhood.kernel.atlases import ChartTransition, ChartTransitionStatus
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ViewModelSetSpec
from empirical_lawhood.kernel.obligations import ObligationStatus, ValiditySpec
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.geometry import (
    AdmissionCandidateCell,
    AdmissionComparison,
    AdmissionEvaluationSpec,
    AtlasAssemblyResult,
    AtlasAssemblySpec,
    AtlasDomainCell,
    ModelGateAssessment,
    ModelReachabilityAssessment,
    ReachabilityCellStatus,
    ReachabilityComparison,
    ReachabilityEvaluationSpec,
    ReachabilityMethodKind,
)

from .reachability import FiniteGridReachability, default_reachability_registry
from .services import AtlasAssembler, ReceiverAdmissionEvaluator


@dataclass(frozen=True, slots=True)
class AtlasReferenceConformanceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/geometry/atlas-reference-conformance-report'

    report_id: str
    atlas_fingerprint: str
    overlap_domain_cell_ids: tuple[str, ...]
    uncovered_domain_cell_ids: tuple[str, ...]
    supported_transition_ids: tuple[str, ...]
    rejected_transition_ids: tuple[str, ...]
    strong_response_admission_status: AdmissionStatus
    nominal_admission_status: AdmissionStatus
    robust_admission_status: AdmissionStatus
    stable_reachability_status: ReachabilityStatus
    nominal_reachability_status: ReachabilityStatus
    robust_reachability_status: ReachabilityStatus
    unreachable_status: ReachabilityStatus
    reachability_method_keys: tuple[str, ...]
    source_law_fingerprints_preserved: bool
    physical_observation_to_controller_use_execution: str
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.atlas_fingerprint, field_name="atlas_fingerprint")
        for field_name, values in (
            ("overlap_domain_cell_ids", self.overlap_domain_cell_ids),
            ("uncovered_domain_cell_ids", self.uncovered_domain_cell_ids),
            ("supported_transition_ids", self.supported_transition_ids),
            ("rejected_transition_ids", self.rejected_transition_ids),
            ("reachability_method_keys", self.reachability_method_keys),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        if not self.source_law_fingerprints_preserved:
            raise ValueError("Atlas reference conformance cannot mutate parametric response-method source laws")
        if self.physical_observation_to_controller_use_execution != "NONE" or self.status != "PASS":
            raise ValueError("Atlas reference report has an invalid evidence/status claim")


def _supported_laws() -> tuple[SystemSpec, tuple[ResponseLaw, ...]]:
    stable = next(case for case in identification_cases() if case.case_id == "response-method-stable-linear")
    payload_plane = EphemeralCandidatePayloadPlane()
    service = build_response_method_identification_service(
        payload_publisher=payload_plane,
        payload_reader=payload_plane,
    )
    primary_result = service.identify(
        stable.system, stable.dataset, stable.config, stable.identifier
    )
    if primary_result.response_law is None:
        raise ValueError("Atlas reference requires the supported parametric response-method stable reference law")
    nonlinear_config = build_identification_config(
        stable.system,
        config_id="config.atlas-overlap-nonlinear",
        method_key=NonlinearLocalIdentifier.method_key,
        method_kind=LawMethodKind.NONLINEAR_LOCAL,
        polynomial_degree=2,
        include_history=False,
    )
    overlap_result = service.identify(
        stable.system,
        stable.dataset,
        nonlinear_config,
        NonlinearLocalIdentifier(),
    )
    secondary_dataset = replace(
        stable.dataset,
        dataset_id="atlas-secondary-chart-dataset",
        observations=tuple(
            replace(
                observation,
                observation_id=f"{observation.observation_id}.secondary",
                chart_id="atlas-secondary-chart",
            )
            for observation in stable.dataset.observations
        ),
    )
    secondary_config = replace(
        stable.config,
        config_id="config.atlas-secondary-chart",
        chart_id="atlas-secondary-chart",
    )
    secondary_result = service.identify(
        stable.system,
        secondary_dataset,
        secondary_config,
        stable.identifier,
    )
    laws = tuple(
        sorted(
            (
                law
                for law in (
                    primary_result.response_law,
                    overlap_result.response_law,
                    secondary_result.response_law,
                )
                if law is not None
            ),
            key=lambda law: law.law_id,
        )
    )
    if len(laws) != 3 or any(
        law.scientific_status is not ScientificStatus.SUPPORTED for law in laws
    ):
        raise ValueError("Atlas reference conformance requires three supported local laws")
    return stable.system, laws


def _evidence_links(laws: tuple[ResponseLaw, ...]) -> tuple[EvidenceLink, ...]:
    values = {link.link_id: link for law in laws for link in law.evidence_links}
    return tuple(sorted(values.values(), key=lambda link: link.link_id))


def _model_set(laws: tuple[ResponseLaw, ...]) -> ViewModelSetSpec:
    link_id = next(law for law in laws if law.chart_id == "response-method-local-chart" and law.evaluator.capability_key == "baseline.local-linear").evidence_links[0].link_id
    return ViewModelSetSpec(
        model_set_id="atlas-reference-model-set",
        target_world_id=next(law for law in laws if law.chart_id == "response-method-local-chart" and law.evaluator.capability_key == "baseline.local-linear").world_id,
        member_view_ids=("coarse-view", "fine-view"),
        model_relation_ids=("reference-coarse-fine-relation",),
        plausibility_rule="Retain both frozen numerical views until robust gates agree.",
        support_rule="Every member must remain inside the local law support.",
        uncertainty_set_ids=("reference-view-disagreement",),
        validity=ValiditySpec(
            validity_id="atlas-model-set-validity",
            validity_domain_ids=("response-method-cell-a", "response-method-cell-b"),
            assumption_ids=("frozen-reference-views",),
            exclusion_reason_codes=(),
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=(link_id,),
        ),
        intersection_semantics=ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _atlas() -> tuple[
    SystemSpec,
    tuple[ResponseLaw, ...],
    tuple[EvidenceLink, ...],
    AtlasAssemblyResult,
]:
    system, laws = _supported_laws()
    evidence_links = _evidence_links(laws)
    primary_chart = "response-method-local-chart"
    secondary_chart = "atlas-secondary-chart"
    primary = next(law for law in laws if law.chart_id == primary_chart)
    transitions = (
        ChartTransition(
            transition_id="transition.primary-to-secondary",
            source_chart_id=primary_chart,
            target_chart_id=secondary_chart,
            status=ChartTransitionStatus.SUPPORTED,
            transport_evaluator=primary.evaluator,
            reason_codes=(),
            evidence_link_ids=(primary.evidence_links[0].link_id,),
        ),
        ChartTransition(
            transition_id="transition.secondary-to-primary",
            source_chart_id=secondary_chart,
            target_chart_id=primary_chart,
            status=ChartTransitionStatus.REJECTED,
            transport_evaluator=None,
            reason_codes=("held-out-directional-transport-failed",),
            evidence_link_ids=(),
        ),
    )
    domain_cells = tuple(
        sorted(
            (
                AtlasDomainCell(
                    domain_cell_id=f"domain.{chart}.{cell}",
                    chart_id=chart,
                    denominator_cell_id=cell,
                )
                for chart in (primary_chart, secondary_chart)
                for cell in ("response-method-cell-a", "response-method-cell-b", "atlas-unsupported-cell")
            ),
            key=lambda item: item.domain_cell_id,
        )
    )
    spec = AtlasAssemblySpec(
        assembly_id="atlas-reference-atlas",
        system=ObjectIdentity.from_record(system.system_id, system),
        laws=tuple(
            sorted(
                (ObjectIdentity.from_record(law.law_id, law) for law in laws),
                key=lambda item: item.object_id,
            )
        ),
        domain_cells=domain_cells,
        transitions=transitions,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    result = AtlasAssembler().assemble(
        system=system,
        laws=laws,
        spec=spec,
        evidence_links=evidence_links,
    )
    return system, laws, evidence_links, result


def _gate_assessments(
    *,
    evaluation_id: str,
    cell_id: str,
    link_id: str,
    nominal_failure: AdmissionGateKind | None = None,
    robust_failure: AdmissionGateKind | None = None,
) -> tuple[ModelGateAssessment, ...]:
    values = []
    for member in ("coarse-view", "fine-view"):
        for kind in AdmissionGateKind:
            failed = (member == "coarse-view" and kind is nominal_failure) or (
                member == "fine-view" and kind is robust_failure
            )
            status = GateStatus.FAIL if failed else GateStatus.PASS
            slug = kind.value.lower().replace("_", "-")
            values.append(
                ModelGateAssessment(
                    assessment_id=f"assessment.{evaluation_id}.{member}.{slug}",
                    cell_id=cell_id,
                    model_member_id=member,
                    kind=kind,
                    status=status,
                    constraint_ids=(f"constraint.{slug}",),
                    margin=NamedDecimal(
                        value_id=f"margin.{evaluation_id}.{member}.{slug}",
                        value=Decimal("-0.1") if failed else Decimal("0.5"),
                        unit="1",
                    ),
                    reason_codes=(f"{slug}-failed",) if failed else (),
                    evidence_link_ids=(link_id,),
                )
            )
    return tuple(sorted(values, key=lambda item: item.assessment_id))


def _admission(
    *,
    evaluation_id: str,
    atlas_result: AtlasAssemblyResult,
    model_set: ViewModelSetSpec,
    evidence_links: tuple[EvidenceLink, ...],
    nominal_failure: AdmissionGateKind | None = None,
    robust_failure: AdmissionGateKind | None = None,
) -> AdmissionComparison:
    spec = _admission_spec(
        evaluation_id=evaluation_id,
        atlas_result=atlas_result,
        model_set=model_set,
        link_id=evidence_links[0].link_id,
        nominal_failure=nominal_failure,
        robust_failure=robust_failure,
    )
    return ReceiverAdmissionEvaluator().evaluate(
        atlas=atlas_result.atlas,
        model_set=model_set,
        spec=spec,
        evidence_links=evidence_links,
    )


def _admission_spec(
    *,
    evaluation_id: str,
    atlas_result: AtlasAssemblyResult,
    model_set: ViewModelSetSpec,
    link_id: str,
    nominal_failure: AdmissionGateKind | None = None,
    robust_failure: AdmissionGateKind | None = None,
) -> AdmissionEvaluationSpec:
    cell_id = f"admission-cell.{evaluation_id}"
    return AdmissionEvaluationSpec(
        evaluation_id=evaluation_id,
        atlas=ObjectIdentity.from_record(atlas_result.atlas.atlas_id, atlas_result.atlas),
        model_set=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        nominal_model_member_id="coarse-view",
        model_member_ids=model_set.member_view_ids,
        receiver_quantity_ids=("receiver",),
        candidate_cells=(
            AdmissionCandidateCell(
                cell_id=cell_id,
                denominator_cell_id="response-method-cell-a",
                chart_id="response-method-local-chart",
                action_bound_ids=("bound.action",),
            ),
        ),
        assessments=_gate_assessments(
            evaluation_id=evaluation_id,
            cell_id=cell_id,
            link_id=link_id,
            nominal_failure=nominal_failure,
            robust_failure=robust_failure,
        ),
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _reachability_assessments(
    evaluation_id: str,
    cell_id: str,
    link_id: str,
    *,
    nominal: ReachabilityCellStatus,
    robust: ReachabilityCellStatus,
) -> tuple[ModelReachabilityAssessment, ...]:
    values = []
    for member, status in (("coarse-view", nominal), ("fine-view", robust)):
        values.append(
            ModelReachabilityAssessment(
                assessment_id=f"reach-assessment.{evaluation_id}.{member}",
                admission_cell_id=cell_id,
                model_member_id=member,
                status=status,
                viable_direction_rank=1 if status is ReachabilityCellStatus.REACHABLE else 0,
                reason_codes=(
                    () if status is ReachabilityCellStatus.REACHABLE else ("target-unreachable",)
                ),
                evidence_link_ids=(link_id,),
            )
        )
    return tuple(sorted(values, key=lambda item: item.assessment_id))


def _reachability(
    *,
    evaluation_id: str,
    admission: AdmissionComparison,
    model_set: ViewModelSetSpec,
    law: ResponseLaw,
    evidence_links: tuple[EvidenceLink, ...],
    nominal: ReachabilityCellStatus,
    robust: ReachabilityCellStatus,
) -> ReachabilityComparison:
    spec = _reachability_spec(
        evaluation_id=evaluation_id,
        admission=admission,
        model_set=model_set,
        law=law,
        link_id=evidence_links[0].link_id,
        nominal=nominal,
        robust=robust,
    )
    return FiniteGridReachability().evaluate(
        admission=admission,
        model_set=model_set,
        spec=spec,
        evidence_links=evidence_links,
    )


def _reachability_spec(
    *,
    evaluation_id: str,
    admission: AdmissionComparison,
    model_set: ViewModelSetSpec,
    law: ResponseLaw,
    link_id: str,
    nominal: ReachabilityCellStatus,
    robust: ReachabilityCellStatus,
) -> ReachabilityEvaluationSpec:
    cell_id = admission.robust.cells[0].cell_id
    return ReachabilityEvaluationSpec(
        evaluation_id=evaluation_id,
        admission_comparison=ObjectIdentity.from_record(admission.comparison_id, admission),
        model_set=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        nominal_model_member_id="coarse-view",
        model_member_ids=model_set.member_view_ids,
        method_kind=ReachabilityMethodKind.FINITE_GRID,
        method_key=FiniteGridReachability.capability_key,
        method_version=FiniteGridReachability.capability_version,
        initial_set_id=f"initial-set.{evaluation_id}",
        action_chart_ids=("response-method-local-chart",),
        dynamics_law_ids=(law.law_id,),
        horizon=law.relation.horizon,
        constraint_ids=("receiver-basin", "sink-boundary"),
        numerical_view_ids=model_set.member_view_ids,
        structural_convergence=law.obligations.structural_convergence,
        computability=law.obligations.computability,
        assessments=_reachability_assessments(
            evaluation_id,
            cell_id,
            link_id,
            nominal=nominal,
            robust=robust,
        ),
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def run_atlas_reference_conformance() -> AtlasReferenceConformanceReport:
    _, laws, evidence_links, atlas_result = _atlas()
    model_set = _model_set(laws)
    strong_empty = _admission(
        evaluation_id="strong-response-empty-admission",
        atlas_result=atlas_result,
        model_set=model_set,
        evidence_links=evidence_links,
        nominal_failure=AdmissionGateKind.PHYSICAL_SINK,
        robust_failure=AdmissionGateKind.PHYSICAL_SINK,
    )
    divergent = _admission(
        evaluation_id="nominal-pass-robust-hold",
        atlas_result=atlas_result,
        model_set=model_set,
        evidence_links=evidence_links,
        robust_failure=AdmissionGateKind.UNCERTAINTY,
    )
    stable = _admission(
        evaluation_id="stable-admission",
        atlas_result=atlas_result,
        model_set=model_set,
        evidence_links=evidence_links,
    )
    law = next(law for law in laws if law.chart_id == "response-method-local-chart")
    stable_reachability = _reachability(
        evaluation_id="stable-reachable",
        admission=stable,
        model_set=model_set,
        law=law,
        evidence_links=evidence_links,
        nominal=ReachabilityCellStatus.REACHABLE,
        robust=ReachabilityCellStatus.REACHABLE,
    )
    divergent_reachability = _reachability(
        evaluation_id="nominal-reachable-robust-empty",
        admission=stable,
        model_set=model_set,
        law=law,
        evidence_links=evidence_links,
        nominal=ReachabilityCellStatus.REACHABLE,
        robust=ReachabilityCellStatus.UNREACHABLE,
    )
    unreachable = _reachability(
        evaluation_id="unreachable-control",
        admission=stable,
        model_set=model_set,
        law=law,
        evidence_links=evidence_links,
        nominal=ReachabilityCellStatus.UNREACHABLE,
        robust=ReachabilityCellStatus.UNREACHABLE,
    )
    registry = default_reachability_registry()
    return AtlasReferenceConformanceReport(
        report_id="atlas-reference-conformance",
        atlas_fingerprint=atlas_result.atlas.fingerprint(),
        overlap_domain_cell_ids=atlas_result.overlap_domain_cell_ids,
        uncovered_domain_cell_ids=atlas_result.uncovered_domain_cell_ids,
        supported_transition_ids=atlas_result.supported_transition_ids,
        rejected_transition_ids=atlas_result.rejected_transition_ids,
        strong_response_admission_status=strong_empty.robust.status,
        nominal_admission_status=divergent.nominal.status,
        robust_admission_status=divergent.robust.status,
        stable_reachability_status=stable_reachability.robust.status,
        nominal_reachability_status=divergent_reachability.nominal.status,
        robust_reachability_status=divergent_reachability.robust.status,
        unreachable_status=unreachable.robust.status,
        reachability_method_keys=tuple(method.capability_key for method in registry.methods),
        source_law_fingerprints_preserved=atlas_result.source_law_fingerprints_preserved,
        physical_observation_to_controller_use_execution="NONE",
        status="PASS",
    )
