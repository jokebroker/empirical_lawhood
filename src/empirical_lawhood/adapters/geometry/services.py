"""Method-neutral atlas assembly and robust receiver-admission intersection."""

from __future__ import annotations

from collections.abc import Iterable

from empirical_lawhood.kernel.admission import (
    AdmissionCellResult,
    AdmissionGateKind,
    AdmissionGateResult,
    AdmissionSet,
    GateStatus,
    validate_admission_against_atlas,
)
from empirical_lawhood.kernel.atlases import (
    AtlasGap,
    AtlasGapKind,
    ChartTransitionStatus,
    ResponseAtlas,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw, validate_law_against_system
from empirical_lawhood.kernel.models import ViewModelSetSpec
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.status import AdmissionStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.adapters.methods.contracts import (
    LawQualificationBatch,
    LawQualificationCoordinateDisposition,
    LawQualificationOverlapKind,
)
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison, derive_finite_certificate_admission_comparison, derive_finite_certificate_reachability_comparison
from empirical_lawhood.planning.geometry import (
    AdmissionCandidateCell,
    AdmissionComparison,
    AdmissionEvaluationSpec,
    AtlasAssemblyObstruction,
    AtlasAssemblyObstructionKind,
    AtlasAssemblyResult,
    AtlasAssemblySpec,
    BatchAtlasAssemblyResult,
    ModelGateAssessment,
    QualificationBatchAtlasAssemblySpec,
    validate_model_members,
)
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionSpec, ControlledMapReachabilitySpec, ReceiptAdmissionReceiptProductionPlan, VerifiedAdmissionReachabilityCategoricalProjection, ControlledMapReachabilityComparison, derive_receipt_admission_comparison, derive_controlled_map_categorical_projection, derive_controlled_map_reachability_comparison


def _links_by_id(evidence_links: tuple[EvidenceLink, ...]) -> dict[str, EvidenceLink]:
    return {link.link_id: link for link in evidence_links}


def _require_known_evidence(
    evidence_link_ids: Iterable[str], evidence_links: tuple[EvidenceLink, ...]
) -> None:
    unknown = set(evidence_link_ids) - set(_links_by_id(evidence_links))
    if unknown:
        raise ValueError(
            f"geometry assessment references unknown evidence links: {sorted(unknown)}"
        )


class AtlasAssembler:
    capability_key = "geometry.atlas-assembler"
    capability_version = "1.0.0"

    def assemble_batch(
        self,
        *,
        system: SystemSpec,
        batch: LawQualificationBatch,
        spec: QualificationBatchAtlasAssemblySpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> BatchAtlasAssemblyResult | AtlasAssemblyObstruction:
        """Project a complete terminal batch without inventing support."""

        self._validate_batch(system, batch, spec, evidence_links)
        laws = batch.supported_laws
        batch_identity = ObjectIdentity.from_record(batch.batch_id, batch)
        assembly_identity = ObjectIdentity.from_record(spec.assembly_id, spec)
        if not laws:
            return AtlasAssemblyObstruction(
                obstruction_id=f"atlas-obstruction.{spec.assembly_id}",
                kind=AtlasAssemblyObstructionKind.ZERO_SUPPORTED_LAWS,
                assembly=assembly_identity,
                qualification_batch=batch_identity,
                coordinate_ids=tuple(value.coordinate_id for value in batch.coordinates),
                qualification_result_ids=tuple(
                    sorted(value.qualification_result.result_id for value in batch.coordinates)
                ),
                evidence_link_ids=tuple(value.link_id for value in evidence_links),
                reason_codes=tuple(
                    sorted(
                        {
                            "zero-supported-laws",
                            *(
                                reason
                                for value in batch.coordinates
                                for reason in value.reason_codes
                            ),
                        }
                    )
                ),
            )
        before = tuple(value.fingerprint() for value in laws)
        gaps = tuple(
            sorted(
                (
                    *(
                        AtlasGap(
                            gap_id=f"gap.{batch.batch_id}.{coordinate.coordinate_id}",
                            kind=self._batch_gap_kind(coordinate.disposition),
                            chart_ids=(coordinate.chart_id,),
                            denominator_cell_ids=(coordinate.denominator_cell_id,),
                            reason_codes=coordinate.reason_codes,
                            evidence_link_ids=coordinate.evidence_link_ids,
                        )
                        for coordinate in batch.coordinates
                        if coordinate.response_law is None
                    ),
                    *(
                        AtlasGap(
                            gap_id=f"gap.{batch.batch_id}.{overlap.overlap_id}",
                            kind=AtlasGapKind.REJECTED_TRANSITION,
                            chart_ids=(overlap.chart_id,),
                            denominator_cell_ids=(overlap.denominator_cell_id,),
                            reason_codes=overlap.reason_codes,
                            evidence_link_ids=overlap.evidence_link_ids,
                        )
                        for overlap in batch.overlaps
                        if overlap.kind is LawQualificationOverlapKind.REJECTED_COMPOSITION
                    ),
                ),
                key=lambda value: value.gap_id,
            )
        )
        atlas = ResponseAtlas(
            atlas_id=f"atlas.{spec.assembly_id}",
            system_id=system.system_id,
            world_id=system.world.world_id,
            laws=laws,
            transitions=spec.transitions,
            gaps=gaps,
            evidence_links=evidence_links,
            evidence_ceiling=EvidenceCeiling.lowest(
                batch.evidence_ceiling,
                spec.evidence_ceiling,
                *(value.evidence_ceiling for value in laws),
            ),
            visibility_ceiling=VisibilityCeiling.most_restrictive(
                batch.visibility_ceiling,
                spec.visibility_ceiling,
                *(value.visibility_ceiling for value in laws),
            ),
        )
        after = tuple(value.fingerprint() for value in laws)
        return BatchAtlasAssemblyResult(
            result_id=f"batch-atlas-assembly-result.{spec.assembly_id}",
            assembly=assembly_identity,
            qualification_batch=batch_identity,
            atlas=atlas,
            negative_coordinate_ids=tuple(
                value.coordinate_id for value in batch.coordinates if value.response_law is None
            ),
            overlap_disposition_ids=tuple(value.overlap_id for value in batch.overlaps),
            supported_transition_ids=tuple(
                value.transition_id
                for value in spec.transitions
                if value.status is ChartTransitionStatus.SUPPORTED
            ),
            rejected_transition_ids=tuple(
                value.transition_id
                for value in spec.transitions
                if value.status is not ChartTransitionStatus.SUPPORTED
            ),
            source_law_fingerprints_preserved=before == after,
        )

    def assemble(
        self,
        *,
        system: SystemSpec,
        laws: tuple[ResponseLaw, ...],
        spec: AtlasAssemblySpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> AtlasAssemblyResult:
        self._validate(system, laws, spec, evidence_links)
        before = tuple(law.fingerprint() for law in laws)
        coverage = {
            cell.domain_cell_id: tuple(
                law
                for law in laws
                if law.chart_id == cell.chart_id
                and cell.denominator_cell_id in law.obligations.support.denominator_cell_ids
            )
            for cell in spec.domain_cells
        }
        evidence_ids = tuple(link.link_id for link in evidence_links)
        gaps = tuple(
            AtlasGap(
                gap_id=f"gap.{spec.assembly_id}.{cell.domain_cell_id}",
                kind=AtlasGapKind.UNSUPPORTED,
                chart_ids=(cell.chart_id,),
                denominator_cell_ids=(cell.denominator_cell_id,),
                reason_codes=("no-supported-law-for-declared-domain-cell",),
                evidence_link_ids=evidence_ids,
            )
            for cell in spec.domain_cells
            if not coverage[cell.domain_cell_id]
        )
        atlas = ResponseAtlas(
            atlas_id=f"atlas.{spec.assembly_id}",
            system_id=system.system_id,
            world_id=system.world.world_id,
            laws=laws,
            transitions=spec.transitions,
            gaps=gaps,
            evidence_links=evidence_links,
            evidence_ceiling=EvidenceCeiling.lowest(
                spec.evidence_ceiling, *(law.evidence_ceiling for law in laws)
            ),
            visibility_ceiling=VisibilityCeiling.most_restrictive(
                spec.visibility_ceiling, *(law.visibility_ceiling for law in laws)
            ),
        )
        after = tuple(law.fingerprint() for law in laws)
        return AtlasAssemblyResult(
            result_id=f"atlas-assembly-result.{spec.assembly_id}",
            assembly=ObjectIdentity.from_record(spec.assembly_id, spec),
            atlas=atlas,
            overlap_domain_cell_ids=tuple(
                cell_id for cell_id, cell_laws in coverage.items() if len(cell_laws) > 1
            ),
            uncovered_domain_cell_ids=tuple(
                cell_id for cell_id, cell_laws in coverage.items() if not cell_laws
            ),
            supported_transition_ids=tuple(
                transition.transition_id
                for transition in spec.transitions
                if transition.status is ChartTransitionStatus.SUPPORTED
            ),
            rejected_transition_ids=tuple(
                transition.transition_id
                for transition in spec.transitions
                if transition.status is not ChartTransitionStatus.SUPPORTED
            ),
            source_law_fingerprints_preserved=before == after,
        )

    @staticmethod
    def _validate(
        system: SystemSpec,
        laws: tuple[ResponseLaw, ...],
        spec: AtlasAssemblySpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> None:
        if spec.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("atlas assembly binds another immutable system")
        identities = tuple(
            sorted(
                (ObjectIdentity.from_record(law.law_id, law) for law in laws),
                key=lambda item: item.object_id,
            )
        )
        if spec.laws != identities:
            raise ValueError("atlas assembly law identities differ from its inputs")
        if not evidence_links:
            raise ValueError("atlas assembly requires evidence links")
        coordinates = tuple((cell.chart_id, cell.denominator_cell_id) for cell in spec.domain_cells)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("atlas declared domain repeats a chart/denominator coordinate")
        for law in laws:
            validate_law_against_system(law, system)
        known_link_ids = {link.link_id for link in evidence_links}
        required_link_ids = {link.link_id for law in laws for link in law.evidence_links} | {
            evidence_id
            for transition in spec.transitions
            for evidence_id in transition.evidence_link_ids
        }
        if required_link_ids != known_link_ids:
            raise ValueError("atlas evidence corpus differs from its laws and transitions")
        declared_coordinates = set(coordinates)
        supported_coordinates = {
            (law.chart_id, denominator_cell_id)
            for law in laws
            for denominator_cell_id in law.obligations.support.denominator_cell_ids
        }
        if not supported_coordinates.issubset(declared_coordinates):
            raise ValueError("atlas declared domain omits supported law coordinates")
        if any(
            evidence_id not in known_link_ids
            for transition in spec.transitions
            for evidence_id in transition.evidence_link_ids
        ):
            raise ValueError("chart transition references unknown evidence")

    @staticmethod
    def _batch_gap_kind(
        disposition: LawQualificationCoordinateDisposition,
    ) -> AtlasGapKind:
        if disposition is LawQualificationCoordinateDisposition.NO_DATA:
            return AtlasGapKind.NO_DATA
        if disposition is LawQualificationCoordinateDisposition.COMPUTABILITY_BOUNDARY:
            return AtlasGapKind.COMPUTABILITY_BOUNDARY
        if disposition in {
            LawQualificationCoordinateDisposition.PREREQUISITE_OBSTRUCTION,
            LawQualificationCoordinateDisposition.CONTRACT_OBSTRUCTION,
        }:
            return AtlasGapKind.INVALID_COORDINATE
        if disposition in {
            LawQualificationCoordinateDisposition.SUPPORTED_LAW,
            LawQualificationCoordinateDisposition.SUPPORTED_CONSTITUENT,
        }:
            raise ValueError("supported qualification coordinate is not an atlas gap")
        return AtlasGapKind.UNSUPPORTED

    @staticmethod
    def _validate_batch(
        system: SystemSpec,
        batch: LawQualificationBatch,
        spec: QualificationBatchAtlasAssemblySpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> None:
        if spec.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("batch atlas assembly binds another immutable system")
        if spec.qualification_batch != ObjectIdentity.from_record(batch.batch_id, batch):
            raise ValueError("batch atlas assembly binds another qualification batch")
        if batch.system_id != system.system_id or batch.world_id != system.world.world_id:
            raise ValueError("qualification batch uses another system or evidence world")
        declared = {
            (value.domain_cell_id, value.chart_id, value.denominator_cell_id)
            for value in spec.domain_cells
        }
        observed = {
            (value.domain_cell_id, value.chart_id, value.denominator_cell_id)
            for value in batch.coordinates
        }
        if declared != observed:
            raise ValueError("batch atlas declared domain differs from terminal coordinates")
        known = {value.link_id: value for value in evidence_links}
        if len(known) != len(evidence_links):
            raise ValueError("batch atlas evidence link IDs are duplicated")
        required_ids = {value.link_id for value in batch.evidence_links} | {
            evidence_id
            for transition in spec.transitions
            for evidence_id in transition.evidence_link_ids
        }
        if set(known) != required_ids:
            raise ValueError("batch atlas evidence differs from results and transitions")
        for link in batch.evidence_links:
            if known[link.link_id] != link:
                raise ValueError("batch atlas evidence ID resolves to different exact evidence")
        for law in batch.supported_laws:
            validate_law_against_system(law, system)
        law_charts = {value.chart_id for value in batch.supported_laws}
        for transition in spec.transitions:
            if not {transition.source_chart_id, transition.target_chart_id} <= law_charts:
                raise ValueError("batch atlas transition references a chart without a law")


class AdmissionReceiptPlanValidator:
    "Cross-layer lineage check from terminal local law batch into the frozen admission roster."

    capability_key = "geometry.admission-receipt-plan-validator"
    capability_version = "1.0.0"

    @staticmethod
    def validate(
        *,
        batch: LawQualificationBatch,
        plan: ReceiptAdmissionReceiptProductionPlan,
    ) -> None:
        if plan.qualification_batch != ObjectIdentity.from_record(batch.batch_id, batch):
            raise ValueError("Admission receipt plan binds another qualification batch")
        supported = tuple(value for value in batch.coordinates if value.response_law is not None)
        for member in plan.model_set.members:
            member_sources = tuple(
                value
                for value in supported
                if value.denominator_member_id == member.denominator_member_id
            )
            if any(value.response_law != member.response_law for value in member_sources):
                raise ValueError("Admission model member would collapse several supported local laws")
            matches = tuple(
                value
                for value in supported
                if value.denominator_member_id == member.denominator_member_id
                and value.response_law == member.response_law
                and ObjectIdentity.from_record(
                    value.qualification_result.result_id,
                    value.qualification_result,
                )
                == member.qualification_result
            )
            if not matches:
                raise ValueError("Admission model member lacks an exact supported batch constituent")
            if any(
                not set(member.candidate_version_ids)
                <= set(value.qualification_result.qualification_trace.candidate_version_member_ids)
                or member.qualification_view_ids
                != value.qualification_result.qualification_trace.qualification_view_ids
                for value in matches
            ):
                raise ValueError("Admission model member rewrites batch candidate/refinement axes")
        member_ids = {value.denominator_member_id for value in plan.model_set.members}
        supported_member_ids = {value.denominator_member_id for value in supported}
        if member_ids != supported_member_ids:
            raise ValueError("Admission model set omits or adds a supported batch denominator member")


class RawAdmissionProjector:
    "Sole pure projection of one raw corpus into admission."

    capability_key = "geometry.raw-admission-projector"
    capability_version = "1.0.0"

    @staticmethod
    def admission(spec: ReceiptAdmissionSpec) -> AdmissionComparison:
        return derive_receipt_admission_comparison(spec)


class RawReachabilityAdmissionProjector:
    """Separate full-map reachability projection over the same corpus."""

    capability_key = "geometry.raw-reachability-admission-projector"
    capability_version = "1.0.0"

    @staticmethod
    def reachability(
        spec: ControlledMapReachabilitySpec,
    ) -> ControlledMapReachabilityComparison:
        return derive_controlled_map_reachability_comparison(spec)


class FiniteAdmissionProjector:
    """Finite-set corpus entry to the shared noncompensating admission owner."""

    capability_key = "geometry.finite-admission-projector"
    capability_version = "1.0.0"

    @staticmethod
    def admission(spec: FiniteCertificateAdmissionSpec) -> AdmissionComparison:
        return derive_finite_certificate_admission_comparison(spec)


class FiniteReachabilityAdmissionProjector:
    """Separate finite certificate intersection over the same complete corpus."""

    capability_key = "geometry.finite-reachability-admission-projector"
    capability_version = "1.0.0"

    @staticmethod
    def reachability(spec: FiniteCertificateReachabilitySpec) -> FiniteCertificateReachabilityComparison:
        return derive_finite_certificate_reachability_comparison(spec)


class RawAdmissionReachabilityCategoricalProjectionEvaluator:
    """Equality-check the categorical R8 view without reconstructing raw truth."""

    capability_key = "geometry.raw-admission-reachability-categorical-projection-evaluator"
    capability_version = "1.0.0"

    @staticmethod
    def atlas_projection(
        admission_spec: ReceiptAdmissionSpec,
        reachability_spec: ControlledMapReachabilitySpec,
    ) -> VerifiedAdmissionReachabilityCategoricalProjection:
        return derive_controlled_map_categorical_projection(admission_spec, reachability_spec)


def _aggregate_status(assessments: tuple[ModelGateAssessment, ...]) -> GateStatus:
    statuses = {assessment.status for assessment in assessments}
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.UNEVALUABLE in statuses:
        return GateStatus.UNEVALUABLE
    return GateStatus.PASS


def _aggregate_margin(
    gate_id: str, assessments: tuple[ModelGateAssessment, ...]
) -> NamedDecimal | None:
    margins = tuple(item.margin for item in assessments if item.margin is not None)
    if not margins:
        return None
    units = {margin.unit for margin in margins}
    if len(units) != 1:
        raise ValueError("model-set gate margins use unlike native units")
    return NamedDecimal(
        value_id=f"margin.{gate_id}",
        value=min(margin.value for margin in margins),
        unit=margins[0].unit,
    )


class ReceiverAdmissionEvaluator:
    capability_key = "geometry.receiver-admission"
    capability_version = "1.0.0"

    def evaluate(
        self,
        *,
        atlas: ResponseAtlas,
        model_set: ViewModelSetSpec,
        spec: AdmissionEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> AdmissionComparison:
        self._validate(atlas, model_set, spec, evidence_links)
        nominal = self._intersection(
            atlas=atlas,
            model_set=model_set,
            spec=spec,
            evidence_links=evidence_links,
            member_ids=(spec.nominal_model_member_id,),
            admission_id=f"admission.{spec.evaluation_id}.nominal",
            model_set_id=f"{model_set.model_set_id}.nominal",
        )
        robust = self._intersection(
            atlas=atlas,
            model_set=model_set,
            spec=spec,
            evidence_links=evidence_links,
            member_ids=spec.model_member_ids,
            admission_id=f"admission.{spec.evaluation_id}.robust",
            model_set_id=model_set.model_set_id,
        )
        disagreement = tuple(
            sorted(set(nominal.admitted_cell_ids).symmetric_difference(robust.admitted_cell_ids))
        )
        return AdmissionComparison(
            comparison_id=f"admission-comparison.{spec.evaluation_id}",
            evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
            nominal=nominal,
            robust=robust,
            structurally_stable=not disagreement,
            disagreement_cell_ids=disagreement,
        )

    @staticmethod
    def _validate(
        atlas: ResponseAtlas,
        model_set: ViewModelSetSpec,
        spec: AdmissionEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> None:
        if spec.atlas != ObjectIdentity.from_record(atlas.atlas_id, atlas):
            raise ValueError("admission evaluation binds another response atlas")
        if spec.model_set != ObjectIdentity.from_record(model_set.model_set_id, model_set):
            raise ValueError("admission evaluation binds another plausible model set")
        validate_model_members(model_set, spec.model_member_ids)
        if model_set.target_world_id != atlas.world_id:
            raise ValueError("admission model set targets another evidence world")
        _require_known_evidence(
            (
                evidence_id
                for assessment in spec.assessments
                for evidence_id in assessment.evidence_link_ids
            ),
            evidence_links,
        )
        gap_coordinates = {
            (chart_id, denominator_cell_id)
            for gap in atlas.gaps
            for chart_id in gap.chart_ids
            for denominator_cell_id in gap.denominator_cell_ids
        }
        if any(
            (cell.chart_id, cell.denominator_cell_id) in gap_coordinates
            for cell in spec.candidate_cells
        ):
            raise ValueError("admission candidate occupies an explicit atlas gap")
        if any(
            not atlas.laws_for_coordinate(cell.chart_id, cell.denominator_cell_id)
            for cell in spec.candidate_cells
        ):
            raise ValueError("admission candidate lies outside explicit atlas law support")

    @staticmethod
    def _intersection(
        *,
        atlas: ResponseAtlas,
        model_set: ViewModelSetSpec,
        spec: AdmissionEvaluationSpec,
        evidence_links: tuple[EvidenceLink, ...],
        member_ids: tuple[str, ...],
        admission_id: str,
        model_set_id: str,
    ) -> AdmissionSet:
        cells = tuple(
            ReceiverAdmissionEvaluator._cell(cell, spec.assessments, member_ids)
            for cell in spec.candidate_cells
        )
        if all(cell.admitted for cell in cells):
            status = AdmissionStatus.ADMITTED
        elif any(cell.admitted for cell in cells):
            status = AdmissionStatus.PARTIAL
        elif any(gate.status is GateStatus.UNEVALUABLE for cell in cells for gate in cell.gates):
            status = AdmissionStatus.UNEVALUABLE
        else:
            status = AdmissionStatus.EMPTY
        admission = AdmissionSet(
            admission_id=admission_id,
            atlas=ObjectIdentity.from_record(atlas.atlas_id, atlas),
            model_set_id=model_set_id,
            receiver_quantity_ids=spec.receiver_quantity_ids,
            cells=cells,
            status=status,
            evidence_links=evidence_links,
            evidence_ceiling=EvidenceCeiling.lowest(
                spec.evidence_ceiling, model_set.evidence_ceiling
            ),
            visibility_ceiling=VisibilityCeiling.most_restrictive(
                spec.visibility_ceiling,
                model_set.visibility_ceiling,
                atlas.visibility_ceiling,
            ),
        )
        validate_admission_against_atlas(atlas, admission)
        return admission

    @staticmethod
    def _cell(
        cell: AdmissionCandidateCell,
        assessments: tuple[ModelGateAssessment, ...],
        member_ids: tuple[str, ...],
    ) -> AdmissionCellResult:
        gates = []
        for kind in AdmissionGateKind:
            selected = tuple(
                assessment
                for assessment in assessments
                if assessment.cell_id == cell.cell_id
                and assessment.model_member_id in member_ids
                and assessment.kind is kind
            )
            status = _aggregate_status(selected)
            gate_id = f"gate.{cell.cell_id}.{kind.value.lower()}"
            gates.append(
                AdmissionGateResult(
                    gate_id=gate_id,
                    kind=kind,
                    status=status,
                    constraint_ids=tuple(
                        sorted(
                            {
                                value
                                for assessment in selected
                                for value in assessment.constraint_ids
                            }
                        )
                    ),
                    margin=_aggregate_margin(gate_id, selected),
                    reason_codes=(
                        ()
                        if status is GateStatus.PASS
                        else tuple(
                            sorted(
                                {
                                    value
                                    for assessment in selected
                                    if assessment.status is not GateStatus.PASS
                                    for value in assessment.reason_codes
                                }
                            )
                        )
                    ),
                    evidence_link_ids=tuple(
                        sorted(
                            {
                                value
                                for assessment in selected
                                for value in assessment.evidence_link_ids
                            }
                        )
                    ),
                )
            )
        gate_tuple = tuple(sorted(gates, key=lambda item: item.gate_id))
        return AdmissionCellResult(
            cell_id=cell.cell_id,
            denominator_cell_id=cell.denominator_cell_id,
            chart_id=cell.chart_id,
            action_bound_ids=cell.action_bound_ids,
            gates=gate_tuple,
            admitted=all(gate.status is GateStatus.PASS for gate in gate_tuple),
        )
