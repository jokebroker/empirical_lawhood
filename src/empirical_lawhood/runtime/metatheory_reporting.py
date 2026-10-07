"""Generic report finalization over exact metatheory owner products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition
from empirical_lawhood.planning.metatheory_prediction import CustodyBoundMetatheoryAdjudicationResult, MetatheoryAdjudicationStop
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationResult
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeResult
from empirical_lawhood.planning.obstruction_atlas import ObstructionAtlasBuildStop, ObstructionAtlas, ObstructionClaimEffect
from empirical_lawhood.planning.property_survival import FacePairedPropertySurvivalAssessment
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceQualificationResult, ScientificSourceQualificationStop
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationContext,
    ScientificAdjudicationRecord,
)


@dataclass(frozen=True, slots=True)
class MetatheoryReportInput(CanonicalRecord):
    """Exact terminal and full owner-product roster consumed by the reporter."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metatheory-report-input'

    report_input_id: str
    adjudication: ObjectIdentity
    obstruction: ObjectIdentity
    owner_products: tuple[ObjectIdentity, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_input_id, field_name="report_input_id")
        if self.adjudication.object_schema not in {
            AtlasQualificationResult.SCHEMA,
            CoordinateChallengeResult.SCHEMA,
            CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
            MetatheoryAdjudicationStop.SCHEMA,
            FacePairedPropertySurvivalAssessment.SCHEMA,
            ScientificSourceQualificationResult.SCHEMA,
            ScientificSourceQualificationStop.SCHEMA,
        }:
            raise ValueError("metatheory report input names another decisive terminal schema")
        if self.obstruction.object_schema not in {
            ObstructionAtlas.SCHEMA,
            ObstructionAtlasBuildStop.SCHEMA,
        }:
            raise ValueError("metatheory report input names another obstruction schema")
        require_sorted_unique_ids(
            self.owner_products,
            attribute="object_id",
            field_name="owner_products",
        )
        if not self.owner_products:
            raise ValueError("metatheory report input requires owner products")
        if not {self.adjudication, self.obstruction}.issubset(set(self.owner_products)):
            raise ValueError("metatheory report input omits its terminal owner products")
        if self.grants_authority:
            raise ValueError("metatheory report input cannot grant authority")


@dataclass(frozen=True, slots=True)
class MetatheoryReportService:
    """Project an owner-derived terminal into the generic scientific report schema."""

    def finalize(
        self,
        *,
        report_input: MetatheoryReportInput,
        adjudication: CustodyBoundMetatheoryAdjudicationResult | MetatheoryAdjudicationStop | None,
        obstruction: ObstructionAtlas | ObstructionAtlasBuildStop,
        context: ScientificAdjudicationContext,
        run_id: str,
        task_id: str,
        input_materialization_ids: tuple[str, ...],
        output_logical_artifact_ids: tuple[str, ...],
        required_receipt_ids: tuple[str, ...],
    ) -> ScientificAdjudicationRecord:
        obstruction_id = ObjectIdentity.from_record(
            obstruction.atlas_id
            if isinstance(obstruction, ObstructionAtlas)
            else obstruction.stop_id,
            obstruction,
        )
        if report_input.obstruction != obstruction_id:
            raise ValueError("METATHEORY_REPORT_TERMINAL_IDENTITY_MISMATCH")
        if adjudication is not None:
            adjudication_id = ObjectIdentity.from_record(
                adjudication.result_id
                if isinstance(adjudication, CustodyBoundMetatheoryAdjudicationResult)
                else adjudication.stop_id,
                adjudication,
            )
            if report_input.adjudication != adjudication_id:
                raise ValueError("METATHEORY_REPORT_TERMINAL_IDENTITY_MISMATCH")
        elif isinstance(obstruction, ObstructionAtlasBuildStop):
            disposition = MetatheoryAggregateDisposition.UNEVALUABLE
            reason_codes = obstruction.reason_codes
        else:
            terminal_ids = {value.terminal for value in obstruction.source_bindings}
            if report_input.adjudication not in terminal_ids:
                raise ValueError("METATHEORY_REPORT_ADJUDICATION_NOT_IN_OBSTRUCTION")
            dispositions = set(obstruction.scientific_dispositions)
            reason_codes = tuple(
                sorted(
                    {
                        reason
                        for source in obstruction.source_bindings
                        for reason in source.source_reason_codes
                    }
                )
            )
            if obstruction.maximum_claim_effect is ObstructionClaimEffect.PREREQUISITE_NONATTEMPT:
                disposition = MetatheoryAggregateDisposition.PREREQUISITE_NONATTEMPT
            elif dispositions == {MetatheoryCellDisposition.SUPPORTED}:
                disposition = MetatheoryAggregateDisposition.SUPPORTED
            elif dispositions == {MetatheoryCellDisposition.OPPOSED}:
                disposition = MetatheoryAggregateDisposition.OPPOSED
            elif MetatheoryCellDisposition.OPPOSED in dispositions and (
                MetatheoryCellDisposition.SUPPORTED in dispositions
            ):
                disposition = MetatheoryAggregateDisposition.MIXED
            else:
                disposition = MetatheoryAggregateDisposition.UNEVALUABLE
            if not reason_codes:
                reason_codes = (f"METATHEORY_{disposition.value}",)
        if isinstance(adjudication, CustodyBoundMetatheoryAdjudicationResult):
            disposition = adjudication.adjudication_result.disposition
            reason_codes = tuple(
                sorted(
                    {
                        reason
                        for cell in adjudication.adjudication_result.cells
                        for reason in cell.reason_codes
                    }
                )
            )
            if not reason_codes:
                reason_codes = (f"METATHEORY_{disposition.value}",)
        elif isinstance(adjudication, MetatheoryAdjudicationStop):
            disposition = MetatheoryAggregateDisposition.UNEVALUABLE
            reason_codes = adjudication.reason_codes
        evaluable = disposition in {
            MetatheoryAggregateDisposition.SUPPORTED,
            MetatheoryAggregateDisposition.OPPOSED,
            MetatheoryAggregateDisposition.MIXED,
        }
        status = {
            MetatheoryAggregateDisposition.SUPPORTED: ScientificStatus.SUPPORTED,
            MetatheoryAggregateDisposition.OPPOSED: ScientificStatus.NOT_SUPPORTED,
            MetatheoryAggregateDisposition.MIXED: ScientificStatus.MIXED,
        }.get(disposition, ScientificStatus.UNEVALUABLE)
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{run_id}.{task_id}",
            run_id=run_id,
            adjudication_task_id=task_id,
            execution_plan=context.execution_plan,
            input_materialization_ids=input_materialization_ids,
            output_logical_artifact_ids=output_logical_artifact_ids,
            required_receipt_ids=required_receipt_ids,
            evidence_world_id=context.evidence_world_id,
            evidence_world_kind=context.evidence_world_kind,
            relation=context.relation,
            independent_unit_id=context.independent_unit_id,
            information_cutoffs=context.information_cutoffs,
            visibility_ceiling=context.visibility_ceiling,
            outcome_access=context.outcome_access,
            evaluability=(
                AdjudicationEvaluability.EVALUABLE
                if evaluable
                else AdjudicationEvaluability.UNEVALUABLE
            ),
            scientific_status=status,
            admission_status=(
                AdmissionStatus.NOT_EVALUATED if evaluable else AdmissionStatus.UNEVALUABLE
            ),
            reason_codes=reason_codes,
            fixture_scope_id=context.fixture_scope_id,
            plumbing_only=context.plumbing_only,
        )


__all__ = ['MetatheoryReportInput', "MetatheoryReportService"]
