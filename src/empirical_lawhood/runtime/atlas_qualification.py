"""Finalization of already-typed finite chart transition evidence."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.atlas_qualification import AtlasLawInputKind, AtlasQualificationCellResult, AtlasQualificationResult, AtlasQualificationSpec, ChartTransitionEvidence, SetValuedChartRegion
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryMethodSelection, MetatheoryPredictiveLevel
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class AtlasQualificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AtlasQualificationService:
    capability_registry: CapabilityRegistry

    def finalize(
        self,
        *,
        spec: AtlasQualificationSpec,
        region: SetValuedChartRegion,
        transition_evidence: tuple[ChartTransitionEvidence, ...],
    ) -> AtlasQualificationResult:
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        region_identity = ObjectIdentity.from_record(region.region_id, region)
        if (
            tuple(value.predictive_level for value in transition_evidence)
            != spec.required_predictive_levels
        ):
            raise AtlasQualificationError("ATLAS_PREDICTIVE_LEVEL_ROSTER_MISMATCH")
        for method in (spec.response_span_method, spec.transition_method, spec.uncertainty_method):
            self._require_method(method)
        cells = []
        gaps: dict[str, ObjectIdentity] = {}
        for observed in transition_evidence:
            if (
                observed.qualification_spec != spec_identity
                or observed.chart_region != region_identity
                or observed.physical_unit_ids != spec.physical_unit_ids
                or observed.method != spec.transition_method
            ):
                raise AtlasQualificationError("ATLAS_TRANSITION_IDENTITY_MISMATCH")
            response = self._bool_disposition(observed.response_span_supported)
            transition = self._bool_disposition(observed.transition_recurrent)
            uncertainty = (
                MetatheoryCellDisposition.SUPPORTED
                if observed.uncertainty_localized and observed.ambiguity_width_within_rule
                else MetatheoryCellDisposition.UNEVALUABLE
            )
            boundary = self._bool_disposition(observed.gate_labels_preserved)
            reasons: list[str] = []
            if not observed.response_span_supported:
                reasons.append("RESPONSE_SPAN_NOT_SUPPORTED")
            if not observed.transition_recurrent:
                reasons.append("TRANSITION_NOT_RECURRENT")
            if uncertainty is MetatheoryCellDisposition.UNEVALUABLE:
                reasons.append("UNCERTAINTY_REGION_TOO_WIDE_OR_UNLOCALIZED")
            if not observed.gate_labels_preserved:
                reasons.append("LABELLED_BOUNDARY_NOT_PRESERVED")
            if observed.predictive_level is MetatheoryPredictiveLevel.DYNAMICAL:
                if not observed.observations_time_ordered or observed.first_passage_censored:
                    reasons.append("DYNAMICAL_ORDER_OR_CENSORING_UNRESOLVED")
            axes = (response, transition, uncertainty, boundary)
            if MetatheoryCellDisposition.OPPOSED in axes:
                disposition = MetatheoryCellDisposition.OPPOSED
            elif reasons:
                disposition = MetatheoryCellDisposition.UNEVALUABLE
            else:
                disposition = MetatheoryCellDisposition.SUPPORTED
            cells.append(
                AtlasQualificationCellResult(
                    cell_id=f"cell.{observed.evidence_id}",
                    qualification_spec=spec_identity,
                    predictive_level=observed.predictive_level,
                    evidence=ObjectIdentity.from_record(observed.evidence_id, observed),
                    response_span=response,
                    transition_recurrence=transition,
                    uncertainty_localization=uncertainty,
                    labelled_boundary_preservation=boundary,
                    ambiguity_width=observed.ambiguity_width,
                    disposition=disposition,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
            for ref in observed.evidence_links:
                gaps[ref.object_id] = ref
        law_obstructed = spec.law_input_kind is AtlasLawInputKind.LAW_OBSTRUCTION
        if law_obstructed:
            assert spec.law_or_obstruction is not None
            gaps[spec.law_or_obstruction.object_id] = spec.law_or_obstruction
        dispositions = {cell.disposition for cell in cells}
        if law_obstructed:
            aggregate = MetatheoryAggregateDisposition.UNEVALUABLE
        elif dispositions == {MetatheoryCellDisposition.SUPPORTED}:
            aggregate = MetatheoryAggregateDisposition.SUPPORTED
        elif MetatheoryCellDisposition.OPPOSED in dispositions:
            aggregate = (
                MetatheoryAggregateDisposition.OPPOSED
                if dispositions == {MetatheoryCellDisposition.OPPOSED}
                else MetatheoryAggregateDisposition.MIXED
            )
        else:
            aggregate = MetatheoryAggregateDisposition.UNEVALUABLE
        achieved = (
            ()
            if law_obstructed
            else tuple(
                sorted(
                    (
                        cell.predictive_level
                        for cell in cells
                        if cell.disposition is MetatheoryCellDisposition.SUPPORTED
                    ),
                    key=lambda value: value.value,
                )
            )
        )
        return AtlasQualificationResult(
            result_id=f"result.{spec.spec_id}",
            qualification_spec=spec_identity,
            chart_region=region_identity,
            transition_evidence=transition_evidence,
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            achieved_predictive_levels=achieved,
            disposition=aggregate,
            gap_or_obstruction_refs=tuple(gaps[key] for key in sorted(gaps)),
            maximum_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            response_law_construction_authorized=False,
        )

    def _require_method(self, method: MetatheoryMethodSelection) -> None:
        try:
            manifest = self.capability_registry.resolve(
                method.capability_key,
                method.capability_version,
            )
        except KeyError as error:
            raise AtlasQualificationError("ATLAS_METHOD_UNREGISTERED") from error
        if manifest.implementation_sha256 != method.implementation_sha256:
            raise AtlasQualificationError("ATLAS_METHOD_BINDING_DRIFT")

    @staticmethod
    def _bool_disposition(value: bool) -> MetatheoryCellDisposition:
        return (
            MetatheoryCellDisposition.SUPPORTED if value else MetatheoryCellDisposition.OPPOSED
        )


__all__ = ["AtlasQualificationError", "AtlasQualificationService"]
