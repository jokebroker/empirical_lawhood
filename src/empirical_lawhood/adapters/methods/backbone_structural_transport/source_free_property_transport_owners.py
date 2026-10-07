"""Corrected source-free metatheory stage runners invoking canonical owners."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationResult, AtlasQualificationSpec, ChartTransitionEvidence, SetValuedChartRegion
from empirical_lawhood.planning.dataset_authority import DatasetDecisionClock, DatasetOperationRequest
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeEvidence, CoordinateChallengeNomination, CoordinateChallengeResult, CoordinateChallengeSpec, MetatheoryCoordinateConstructionProduct
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceResult, DecisionAssuranceSpec, DecisionComparisonEvidence
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.planning.metatheory_prediction import CustodyBoundMetatheoryAdjudicationResult, MetatheoryAdjudicationSpec, MetatheoryAdjudicationStop, MetatheoryPredictionIssueReceipt, MetatheoryPredictionPackage, MetatheoryRevealAuthorizationBinding, MetatheorySealedTarget, MetatheoryTargetAcquisitionIndex, MetatheoryTargetOutcome, RevealedTargetOutcomeBinding
from empirical_lawhood.planning.obstruction_atlas import ObstructionAtlasBuildStop, ObstructionAtlasSpec, ObstructionAtlas, ObstructionSourceBinding
from empirical_lawhood.planning.property_survival import PropertyPathCorrespondence, FacePairedPropertySurvivalAssessment, PropertySurvivalCellEvidence, PropertySurvivalSpec
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateEvidence, ScientificSourceQualificationResult, ScientificSourceQualificationSpec, ScientificSourceQualificationStopKind, ScientificSourceQualificationStop
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile, SourcePipelineQualificationDisposition, SourcePipelineQualificationReceipt
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationContext,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.atlas_qualification import AtlasQualificationService
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.coordinate_challenges import CoordinateChallengeService
from empirical_lawhood.runtime.coordinate_construction import MetatheoryCoordinateConstructionService
from empirical_lawhood.runtime.datasets import DatasetCapabilityRegistry
from empirical_lawhood.runtime.decision_assurance import DecisionAssuranceService
from empirical_lawhood.runtime.evidence_lineage import EvidenceDependenceService
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryStageOwnerContract
from empirical_lawhood.runtime.metatheory_prediction import (
    MetatheoryAdjudicationService,
    MetatheoryPredictionIssueService,
    MetatheoryTargetAcquisitionService,
)
from empirical_lawhood.runtime.metatheory_reporting import MetatheoryReportInput, MetatheoryReportService
from empirical_lawhood.runtime.obstruction_atlas import ObstructionAtlasBuilder
from empirical_lawhood.runtime.property_survival import PropertySurvivalService
from empirical_lawhood.runtime.scientific_source_qualification import (
    ScientificSourceQualificationService,
)
from empirical_lawhood.runtime.source_pipelines import SourcePipelineExecutionEvidence, authority_required_source_pipeline_receipt, compile_source_pipeline, qualify_source_pipeline
from empirical_lawhood.runtime.sources import SourceCapabilityManifest


def _require_role(
    contract: MetatheoryStageOwnerContract,
    role: MetatheoryCampaignStageRole,
) -> None:
    if contract.role is not role:
        raise ValueError("METATHEORY_STAGE_RUNNER_ROLE_MISMATCH")


@dataclass(frozen=True, slots=True)
class SourcePipelineOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract

    def __post_init__(self) -> None:
        _require_role(self.owner_contract, MetatheoryCampaignStageRole.SOURCE_PIPELINE)

    def run(
        self,
        *,
        profile: SourcePipelineProfile,
        source_manifest: SourceCapabilityManifest,
        capability_registry: CapabilityRegistry,
        dataset_registry: DatasetCapabilityRegistry,
        dataset_operation_request: DatasetOperationRequest | None,
        execution_evidence: SourcePipelineExecutionEvidence | None,
        decision_clock: DatasetDecisionClock | None,
    ) -> SourcePipelineQualificationReceipt:
        compilation = compile_source_pipeline(
            profile,
            source_manifest=source_manifest,
            capability_registry=capability_registry,
            dataset_registry=dataset_registry,
            dataset_operation_request=dataset_operation_request,
        )
        if dataset_operation_request is None:
            if execution_evidence is not None or decision_clock is not None:
                raise ValueError("AUTHORITY_STOP_SOURCE_PIPELINE_RECEIVED_EXECUTION_EVIDENCE")
            return authority_required_source_pipeline_receipt(profile)
        if execution_evidence is None or decision_clock is None:
            raise ValueError("QUALIFIED_SOURCE_PIPELINE_LACKS_EXECUTION_EVIDENCE")
        return qualify_source_pipeline(
            profile,
            compilation,
            execution_evidence,
            source_manifest=source_manifest,
            decision_clock=decision_clock,
        )


@dataclass(frozen=True, slots=True)
class ScientificSourceQualificationOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: ScientificSourceQualificationService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
        )

    def run(
        self,
        *,
        spec: ScientificSourceQualificationSpec,
        pipeline_receipt: SourcePipelineQualificationReceipt,
        gate_evidence: tuple[ScientificSourceGateEvidence, ...],
    ) -> ScientificSourceQualificationResult | ScientificSourceQualificationStop:
        if pipeline_receipt.disposition is not SourcePipelineQualificationDisposition.QUALIFIED:
            return self.service.stop(
                spec=spec,
                stop_kind=ScientificSourceQualificationStopKind.AUTHORITY_REQUIRED,
                reason_codes=(
                    pipeline_receipt.reason_codes
                    or ("SCIENTIFIC_SOURCE_QUALIFICATION_PREREQUISITE_NONATTEMPT",)
                ),
            )
        return self.service.finalize(
            spec=spec,
            pipeline_receipt=pipeline_receipt,
            gate_evidence=gate_evidence,
        )


@dataclass(frozen=True, slots=True)
class CoordinateConstructionOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: MetatheoryCoordinateConstructionService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
        )

    def run(
        self,
        *,
        spec: CoordinateChallengeSpec,
        source_qualification: ScientificSourceQualificationResult | None,
        nominations: tuple[CoordinateChallengeNomination, ...],
        evidence: tuple[CoordinateChallengeEvidence, ...],
    ) -> MetatheoryCoordinateConstructionProduct:
        return self.service.construct(
            spec=spec,
            source_qualification=source_qualification,
            nominations=nominations,
            evidence=evidence,
        )


@dataclass(frozen=True, slots=True)
class CoordinateEvaluationOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: CoordinateChallengeService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        )

    def run(
        self,
        *,
        spec: CoordinateChallengeSpec,
        construction: MetatheoryCoordinateConstructionProduct,
    ) -> CoordinateChallengeResult:
        if construction.challenge_spec != ObjectIdentity.from_record(spec.spec_id, spec):
            raise ValueError("COORDINATE_EVALUATION_CONSTRUCTION_IDENTITY_MISMATCH")
        return self.service.finalize(
            spec=spec,
            nominations=construction.nominations,
            evidence=construction.evidence,
        )


@dataclass(frozen=True, slots=True)
class AtlasQualificationOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: AtlasQualificationService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
        )

    def run(
        self,
        *,
        coordinate_result: CoordinateChallengeResult,
        spec: AtlasQualificationSpec,
        region: SetValuedChartRegion,
        transition_evidence: tuple[ChartTransitionEvidence, ...],
    ) -> AtlasQualificationResult:
        if spec.coordinate_challenge_result != ObjectIdentity.from_record(
            coordinate_result.result_id,
            coordinate_result,
        ):
            raise ValueError("ATLAS_COORDINATE_PREDECESSOR_IDENTITY_MISMATCH")
        return self.service.finalize(
            spec=spec,
            region=region,
            transition_evidence=transition_evidence,
        )


@dataclass(frozen=True, slots=True)
class DecisionAssuranceOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: DecisionAssuranceService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
        )

    def run(
        self,
        *,
        spec: DecisionAssuranceSpec,
        evidence: tuple[DecisionComparisonEvidence, ...],
    ) -> DecisionAssuranceResult:
        return self.service.finalize(spec=spec, evidence=evidence)


@dataclass(frozen=True, slots=True)
class PropertySurvivalOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    dependence_service: EvidenceDependenceService
    survival_service: PropertySurvivalService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
        )
        if len(self.owner_contract.supporting_owners) != 1:
            raise ValueError("PROPERTY_STAGE_REQUIRES_EVIDENCE_DEPENDENCE_SUPPORT_OWNER")

    def run(
        self,
        *,
        spec: PropertySurvivalSpec,
        dependence_spec: EvidenceDependenceSpec,
        evidence: tuple[PropertySurvivalCellEvidence, ...],
        path_correspondences: tuple[PropertyPathCorrespondence, ...],
        decision_assurance_results: tuple[DecisionAssuranceResult, ...],
    ) -> FacePairedPropertySurvivalAssessment:
        if spec.dependence_spec != ObjectIdentity.from_record(
            dependence_spec.spec_id,
            dependence_spec,
        ):
            raise ValueError("PROPERTY_STAGE_DEPENDENCE_SPEC_IDENTITY_MISMATCH")
        dependence = self.dependence_service.assess(dependence_spec)
        return self.survival_service.finalize_face_paired(
            spec=spec,
            dependence_assessment=dependence,
            evidence=evidence,
            path_correspondences=path_correspondences,
            decision_assurance_results=decision_assurance_results,
        )


@dataclass(frozen=True, slots=True)
class PredictionIssueOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: MetatheoryPredictionIssueService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.PREDICTION_ISSUE,
        )

    def run(
        self,
        *,
        property_assessment: FacePairedPropertySurvivalAssessment,
        package: MetatheoryPredictionPackage,
        issue_clock_id: str,
        publication: ObjectIdentity,
        recovery: ObjectIdentity,
    ) -> MetatheoryPredictionIssueReceipt:
        if ObjectIdentity.from_record(
            property_assessment.assessment_id,
            property_assessment,
        ) not in set(package.development_parents):
            raise ValueError("PREDICTION_PROPERTY_PREDECESSOR_IDENTITY_MISMATCH")
        return self.service.issue(
            package=package,
            issue_clock_id=issue_clock_id,
            publication=publication,
            recovery=recovery,
        )


@dataclass(frozen=True, slots=True)
class TargetAcquisitionOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: MetatheoryTargetAcquisitionService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.TARGET_ACQUISITION,
        )

    def run(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue: MetatheoryPredictionIssueReceipt,
        sealed_targets: tuple[MetatheorySealedTarget, ...],
    ) -> MetatheoryTargetAcquisitionIndex:
        return self.service.finalize(
            package=package,
            issue=issue,
            sealed_targets=sealed_targets,
        )


@dataclass(frozen=True, slots=True)
class RevealAdjudicationOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: MetatheoryAdjudicationService

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
        )

    def run(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue: MetatheoryPredictionIssueReceipt,
        acquisition: MetatheoryTargetAcquisitionIndex,
        reveal: MetatheoryRevealAuthorizationBinding,
        spec: MetatheoryAdjudicationSpec,
        dependence_assessment: EvidenceDependenceAssessment,
        outcomes: tuple[MetatheoryTargetOutcome, ...],
        outcome_bindings: tuple[RevealedTargetOutcomeBinding, ...],
    ) -> CustodyBoundMetatheoryAdjudicationResult | MetatheoryAdjudicationStop:
        return self.service.adjudicate_custodied_outcomes(
            package=package,
            issue=issue,
            acquisition=acquisition,
            reveal=reveal,
            spec=spec,
            dependence_assessment=dependence_assessment,
            outcomes=outcomes,
            outcome_bindings=outcome_bindings,
        )


@dataclass(frozen=True, slots=True)
class ObstructionCloseoutOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: ObstructionAtlasBuilder

    def __post_init__(self) -> None:
        _require_role(
            self.owner_contract,
            MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        )

    def run(
        self,
        *,
        spec: ObstructionAtlasSpec,
        sources: tuple[ObstructionSourceBinding, ...],
    ) -> ObstructionAtlas | ObstructionAtlasBuildStop:
        return self.service.build(spec=spec, sources=sources)


@dataclass(frozen=True, slots=True)
class ReportOwnerRunner:
    owner_contract: MetatheoryStageOwnerContract
    service: MetatheoryReportService

    def __post_init__(self) -> None:
        _require_role(self.owner_contract, MetatheoryCampaignStageRole.REPORT)

    def run(
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
        return self.service.finalize(
            report_input=report_input,
            adjudication=adjudication,
            obstruction=obstruction,
            context=context,
            run_id=run_id,
            task_id=task_id,
            input_materialization_ids=input_materialization_ids,
            output_logical_artifact_ids=output_logical_artifact_ids,
            required_receipt_ids=required_receipt_ids,
        )


__all__ = [
    'AtlasQualificationOwnerRunner',
    'CoordinateConstructionOwnerRunner',
    'CoordinateEvaluationOwnerRunner',
    'DecisionAssuranceOwnerRunner',
    'ObstructionCloseoutOwnerRunner',
    'PredictionIssueOwnerRunner',
    'PropertySurvivalOwnerRunner',
    'ReportOwnerRunner',
    'RevealAdjudicationOwnerRunner',
    'ScientificSourceQualificationOwnerRunner',
    'SourcePipelineOwnerRunner',
    'TargetAcquisitionOwnerRunner',
]
