"""Installed public-route providers for the corrected metatheory stages."""

from __future__ import annotations

from dataclasses import fields, replace
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationResult
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeSpec, CoordinateChallengeResult, MetatheoryCoordinateConstructionProduct
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceResult
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.planning.metatheory_prediction import CustodyBoundMetatheoryAdjudicationResult, MetatheoryAdjudicationStop, MetatheoryPredictionIssueReceipt, MetatheoryPredictionPackage, MetatheoryRevealAuthorizationBinding, MetatheorySealedTarget, MetatheoryTargetAcquisitionIndex, MetatheoryTargetOutcome, RevealedTargetOutcomeBinding
from empirical_lawhood.planning.obstruction_atlas import ObstructionAtlasBuildStop, ObstructionAtlasSpec, ObstructionAtlas, ObstructionExpectedCell, ObstructionOperationalStatus, ObstructionSourceBinding
from empirical_lawhood.planning.property_survival import FacePairedPropertySurvivalAssessment
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceQualificationResult, ScientificSourceQualificationStop
from empirical_lawhood.planning.source_pipelines import SourcePipelineQualificationReceipt
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.atlas_qualification import AtlasQualificationService
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.coordinate_challenges import CoordinateChallengeService
from empirical_lawhood.runtime.coordinate_construction import MetatheoryCoordinateConstructionService
from empirical_lawhood.runtime.decision_assurance import DecisionAssuranceService
from empirical_lawhood.runtime.evidence_lineage import EvidenceDependenceService
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.metatheory_prediction import (
    MetatheoryAdjudicationService,
    MetatheoryPredictionIssueService,
    MetatheoryTargetAcquisitionService,
)
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryStageOwnerContract
from empirical_lawhood.runtime.metatheory_reporting import MetatheoryReportInput, MetatheoryReportService
from empirical_lawhood.runtime.obstruction_atlas import ObstructionAtlasBuilder
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.property_survival import PropertySurvivalService
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.scientific_source_qualification import (
    ScientificSourceQualificationService,
)

from .source_free_property_transport_contracts import source_free_property_transport_owner_contract
from .source_free_property_transport_owners import AtlasQualificationOwnerRunner, CoordinateConstructionOwnerRunner, CoordinateEvaluationOwnerRunner, DecisionAssuranceOwnerRunner, ObstructionCloseoutOwnerRunner, PredictionIssueOwnerRunner, PropertySurvivalOwnerRunner, ReportOwnerRunner, RevealAdjudicationOwnerRunner, ScientificSourceQualificationOwnerRunner, SourcePipelineOwnerRunner, TargetAcquisitionOwnerRunner
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE, SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES, SourceFreePropertyTransportAtlasQualificationConfig, SourceFreePropertyTransportCoordinateConstructionConfig, SourceFreePropertyTransportCoordinateEvaluationConfig, SourceFreePropertyTransportDecisionAssuranceConfig, SourceFreePropertyTransportObstructionCloseoutConfig, SourceFreePropertyTransportPredictionIssueConfig, SourceFreePropertyTransportPreparedMedium, SourceFreePropertyTransportPropertySurvivalConfig, SourceFreePropertyTransportReportConfig, SourceFreePropertyTransportRevealAdjudicationConfig, SourceFreePropertyTransportSealedOutcomeBundle, SourceFreePropertyTransportSourcePipelineConfig, SourceFreePropertyTransportSourceQualificationConfig, SourceFreePropertyTransportTargetAcquisitionConfig


SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


class _Clock:
    def __init__(self, clock_id: str, timestamp: str) -> None:
        self.clock_id = clock_id
        self.timestamp = timestamp

    def now_utc(self) -> str:
        return self.timestamp


_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    *SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES,
    SourceFreePropertyTransportPreparedMedium,
    SourcePipelineQualificationReceipt,
    ScientificSourceQualificationResult,
    ScientificSourceQualificationStop,
    MetatheoryCoordinateConstructionProduct,
    CoordinateChallengeSpec,
    CoordinateChallengeResult,
    AtlasQualificationResult,
    DecisionAssuranceResult,
    FacePairedPropertySurvivalAssessment,
    MetatheoryPredictionPackage,
    MetatheoryPredictionIssueReceipt,
    SourceFreePropertyTransportSealedOutcomeBundle,
    MetatheoryTargetAcquisitionIndex,
    CustodyBoundMetatheoryAdjudicationResult,
    MetatheoryAdjudicationStop,
    ObstructionAtlas,
    ObstructionAtlasBuildStop,
)
_RECORD_TYPE_BY_SCHEMA = {value.SCHEMA: value for value in _RECORD_TYPES}


def _read_records(context: TaskContext) -> tuple[CanonicalRecord, ...]:
    values: list[CanonicalRecord] = []
    for port in context.input_ports:
        try:
            record_type = _RECORD_TYPE_BY_SCHEMA[port.payload_schema]
        except KeyError as error:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_UNKNOWN_INPUT_SCHEMA") from error
        payload = port.read(port.size_bytes + 1)
        if len(payload) != port.size_bytes:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_INPUT_SIZE_MISMATCH")
        values.append(
            decode_canonical_bytes(payload, record_type, maximum_bytes=max(1, port.size_bytes))
        )
    return tuple(values)


def _one(records: tuple[CanonicalRecord, ...], kind: type[CanonicalRecord]) -> CanonicalRecord:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"SOURCE_FREE_PROPERTY_TRANSPORT_REQUIRES_EXACTLY_ONE_{kind.__name__}")
    return values[0]


def _optional(
    records: tuple[CanonicalRecord, ...],
    kind: type[CanonicalRecord],
) -> CanonicalRecord | None:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) > 1:
        raise ValueError(f"SOURCE_FREE_PROPERTY_TRANSPORT_REPEATS_{kind.__name__}")
    return values[0] if values else None


def _output(
    context: TaskContext,
    record: CanonicalRecord,
    *,
    schema: str | None = None,
) -> TaskOutputPayload:
    expected_schema = record.SCHEMA if schema is None else schema
    ports = tuple(
        value for value in context.output_ports if value.payload_schema == expected_schema
    )
    if len(ports) != 1 or record.SCHEMA != expected_schema:
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_OUTPUT_SCHEMA_MISMATCH")
    return TaskOutputPayload(output_id=ports[0].output_id, payload=record.canonical_bytes())


def _identity(record: CanonicalRecord) -> ObjectIdentity:
    for field_name in (
        "result_id",
        "assessment_id",
        "stop_id",
        "atlas_id",
        "receipt_id",
        "product_id",
        "index_id",
    ):
        value = getattr(record, field_name, None)
        if isinstance(value, str):
            return ObjectIdentity.from_record(value, record)
    raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PRODUCT_LACKS_IDENTITY")


def _check(role: MetatheoryCampaignStageRole) -> tuple[ReceiptCheck, ...]:
    return (
        ReceiptCheck(
            check_id=f"source-free-property-transport-owner.{role.value.lower()}",
            passed=True,
            reason_codes=(),
        ),
    )


class SourceFreePropertyTransportTaskRunner:
    """Dispatch one exact role to its registered scientific owner."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        role: MetatheoryCampaignStageRole,
        config: CanonicalRecord,
        registry: CapabilityRegistry,
    ) -> None:
        expected = SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role]
        if not isinstance(config, expected) or config.SCHEMA != manifest.config_schema:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_ROLE_MISMATCH")
        self.manifest = manifest
        self.role = role
        self.config = config
        self.registry = registry

    @property
    def owner_contract(self) -> MetatheoryStageOwnerContract:
        """Expose the exact coordinator contract without executing science."""

        return source_free_property_transport_owner_contract(role=self.role, registry=self.registry)

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _read_records(context)
        if _one(records, type(self.config)) != self.config:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_ISSUED_CONFIG_MISMATCH")
        if context.config.content_sha256 != self.config.fingerprint():
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_FINGERPRINT_MISMATCH")
        product, extra = self._execute_owner(context, records)
        outputs = tuple(
            sorted(
                (_output(context, product), *(_output(context, value) for value in extra)),
                key=lambda value: value.output_id,
            )
        )
        return RunnerResult(outputs=outputs, checks=_check(self.role))

    def _execute_owner(
        self,
        context: TaskContext,
        records: tuple[CanonicalRecord, ...],
    ) -> tuple[CanonicalRecord, tuple[CanonicalRecord, ...]]:
        contract = self.owner_contract
        config = self.config
        if isinstance(config, SourceFreePropertyTransportSourcePipelineConfig):
            medium = _one(records, SourceFreePropertyTransportPreparedMedium)
            if medium != config.prepared_medium:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PREPARED_MEDIUM_MISMATCH")
            clock = (
                None
                if config.decision_clock_id is None or config.decision_time_utc is None
                else _Clock(config.decision_clock_id, config.decision_time_utc)
            )
            return (
                SourcePipelineOwnerRunner(contract).run(
                    profile=config.profile,
                    source_manifest=config.source_manifest,
                    capability_registry=config.capability_registry,
                    dataset_registry=config.dataset_registry,
                    dataset_operation_request=config.dataset_operation_request,
                    execution_evidence=config.execution_evidence,
                    decision_clock=clock,
                ),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportSourceQualificationConfig):
            pipeline = _one(records, SourcePipelineQualificationReceipt)
            assert isinstance(pipeline, SourcePipelineQualificationReceipt)
            qualification_spec = replace(
                config.spec,
                source_pipeline_profile=pipeline.profile,
                source_pipeline_qualification=ObjectIdentity.from_record(
                    pipeline.receipt_id,
                    pipeline,
                ),
            )
            gate_evidence = tuple(
                replace(
                    value,
                    qualification_spec=ObjectIdentity.from_record(
                        qualification_spec.spec_id,
                        qualification_spec,
                    ),
                )
                for value in config.gate_evidence
            )
            return (
                ScientificSourceQualificationOwnerRunner(
                    contract,
                    ScientificSourceQualificationService(config.capability_registry),
                ).run(
                    spec=qualification_spec,
                    pipeline_receipt=pipeline,
                    gate_evidence=gate_evidence,
                ),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportCoordinateConstructionConfig):
            source = _optional(records, ScientificSourceQualificationResult)
            assert source is None or isinstance(source, ScientificSourceQualificationResult)
            coordinate_spec = (
                config.spec
                if source is None
                else replace(
                    config.spec,
                    source_qualification=ObjectIdentity.from_record(source.result_id, source),
                    source_qualification_not_applicable_reasons=(),
                )
            )
            nominations = tuple(
                replace(
                    value,
                    challenge_spec=ObjectIdentity.from_record(
                        coordinate_spec.spec_id,
                        coordinate_spec,
                    ),
                )
                for value in config.nominations
            )
            nomination_by_id = {value.nomination_id: value for value in nominations}
            evidence = tuple(
                replace(
                    value,
                    challenge_spec=ObjectIdentity.from_record(
                        coordinate_spec.spec_id,
                        coordinate_spec,
                    ),
                    nomination=ObjectIdentity.from_record(
                        value.nomination.object_id,
                        nomination_by_id[value.nomination.object_id],
                    ),
                )
                for value in config.evidence
            )
            construction = CoordinateConstructionOwnerRunner(
                contract,
                MetatheoryCoordinateConstructionService(config.capability_registry),
            ).run(
                spec=coordinate_spec,
                source_qualification=source,
                nominations=nominations,
                evidence=evidence,
            )
            return (
                construction,
                (coordinate_spec,),
            )
        if isinstance(config, SourceFreePropertyTransportCoordinateEvaluationConfig):
            construction_record = _one(records, MetatheoryCoordinateConstructionProduct)
            coordinate_spec_record = _one(records, CoordinateChallengeSpec)
            assert isinstance(construction_record, MetatheoryCoordinateConstructionProduct)
            assert isinstance(coordinate_spec_record, CoordinateChallengeSpec)
            return (
                CoordinateEvaluationOwnerRunner(
                    contract,
                    CoordinateChallengeService(config.capability_registry),
                ).run(spec=coordinate_spec_record, construction=construction_record),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportAtlasQualificationConfig):
            coordinate = _one(records, CoordinateChallengeResult)
            assert isinstance(coordinate, CoordinateChallengeResult)
            atlas_spec = replace(
                config.spec_template,
                coordinate_challenge_result=ObjectIdentity.from_record(
                    coordinate.result_id,
                    coordinate,
                ),
            )
            atlas_evidence = tuple(
                replace(
                    value,
                    qualification_spec=ObjectIdentity.from_record(
                        atlas_spec.spec_id,
                        atlas_spec,
                    ),
                )
                for value in config.transition_evidence
            )
            return (
                AtlasQualificationOwnerRunner(
                    contract,
                    AtlasQualificationService(config.capability_registry),
                ).run(
                    coordinate_result=coordinate,
                    spec=atlas_spec,
                    region=config.region,
                    transition_evidence=atlas_evidence,
                ),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportDecisionAssuranceConfig):
            _one(records, AtlasQualificationResult)
            return (
                DecisionAssuranceOwnerRunner(
                    contract,
                    DecisionAssuranceService(config.capability_registry),
                ).run(spec=config.spec, evidence=config.evidence),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportPropertySurvivalConfig):
            decision = _optional(records, DecisionAssuranceResult)
            assert decision is None or isinstance(decision, DecisionAssuranceResult)
            return (
                PropertySurvivalOwnerRunner(
                    contract,
                    EvidenceDependenceService(),
                    PropertySurvivalService(config.capability_registry),
                ).run(
                    spec=config.spec,
                    dependence_spec=config.dependence_spec,
                    evidence=config.evidence,
                    path_correspondences=config.path_correspondences,
                    decision_assurance_results=() if decision is None else (decision,),
                ),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportPredictionIssueConfig):
            assessment = _one(records, FacePairedPropertySurvivalAssessment)
            assert isinstance(assessment, FacePairedPropertySurvivalAssessment)
            package = replace(
                config.package_template,
                development_parents=(
                    ObjectIdentity.from_record(assessment.assessment_id, assessment),
                ),
            )
            issue = PredictionIssueOwnerRunner(
                contract,
                MetatheoryPredictionIssueService(),
            ).run(
                property_assessment=assessment,
                package=package,
                issue_clock_id=config.issue_clock_id,
                publication=config.publication,
                recovery=config.recovery,
            )
            return issue, (package,)
        if isinstance(config, SourceFreePropertyTransportTargetAcquisitionConfig):
            package_record = _one(records, MetatheoryPredictionPackage)
            issue_record = _one(records, MetatheoryPredictionIssueReceipt)
            assert isinstance(package_record, MetatheoryPredictionPackage)
            assert isinstance(issue_record, MetatheoryPredictionIssueReceipt)
            bundle = _generate_sealed_bundle(
                config=config,
                package=package_record,
                issue=issue_record,
            )
            sealed = (
                ()
                if not config.generator.target_present
                else (
                    MetatheorySealedTarget(
                        target_id=bundle.target_id,
                        acquired_physical_unit_ids=package_record.targets[0].physical_unit_ids,
                        sealed_artifact=ObjectIdentity.from_record(bundle.bundle_id, bundle),
                        acquisition_complete=True,
                    ),
                )
            )
            acquisition = TargetAcquisitionOwnerRunner(
                contract,
                MetatheoryTargetAcquisitionService(),
            ).run(package=package_record, issue=issue_record, sealed_targets=sealed)
            return acquisition, (bundle,)
        if isinstance(config, SourceFreePropertyTransportRevealAdjudicationConfig):
            package_record = _one(records, MetatheoryPredictionPackage)
            issue_record = _one(records, MetatheoryPredictionIssueReceipt)
            acquisition_record = _one(records, MetatheoryTargetAcquisitionIndex)
            bundle_record = _one(records, SourceFreePropertyTransportSealedOutcomeBundle)
            assert isinstance(package_record, MetatheoryPredictionPackage)
            assert isinstance(issue_record, MetatheoryPredictionIssueReceipt)
            assert isinstance(acquisition_record, MetatheoryTargetAcquisitionIndex)
            assert isinstance(bundle_record, SourceFreePropertyTransportSealedOutcomeBundle)
            return (
                _adjudicate(
                    contract=contract,
                    config=config,
                    package=package_record,
                    issue=issue_record,
                    acquisition=acquisition_record,
                    bundle=bundle_record,
                ),
                (),
            )
        if isinstance(config, SourceFreePropertyTransportObstructionCloseoutConfig):
            terminal = _decisive_terminal(records)
            return _build_obstruction(contract, config, terminal), ()
        if isinstance(config, SourceFreePropertyTransportReportConfig):
            obstruction = _one(records, ObstructionAtlas)
            assert isinstance(obstruction, ObstructionAtlas)
            if context.scientific_adjudication_context is None:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_REPORT_CONTEXT_MISSING")
            terminal = obstruction.source_bindings[0].terminal
            obstruction_identity = ObjectIdentity.from_record(obstruction.atlas_id, obstruction)
            report_input = MetatheoryReportInput(
                report_input_id=f"report-input.{context.run_id}.{context.task_id}",
                adjudication=terminal,
                obstruction=obstruction_identity,
                owner_products=tuple(
                    sorted((terminal, obstruction_identity), key=lambda value: value.object_id)
                ),
                grants_authority=False,
            )
            report = ReportOwnerRunner(contract, MetatheoryReportService()).run(
                report_input=report_input,
                adjudication=None,
                obstruction=obstruction,
                context=context.scientific_adjudication_context,
                run_id=context.run_id,
                task_id=context.task_id,
                input_materialization_ids=context.input_materialization_ids,
                output_logical_artifact_ids=tuple(
                    sorted(
                        value.logical_artifact_id
                        for value in context.output_ports
                        if value.logical_artifact_id is not None
                    )
                ),
                required_receipt_ids=context.dependency_receipt_ids,
            )
            return report, ()
        raise TypeError("SOURCE_FREE_PROPERTY_TRANSPORT_UNKNOWN_CONFIG_TYPE")


def _generate_sealed_bundle(
    *,
    config: SourceFreePropertyTransportTargetAcquisitionConfig,
    package: MetatheoryPredictionPackage,
    issue: MetatheoryPredictionIssueReceipt,
) -> SourceFreePropertyTransportSealedOutcomeBundle:
    if issue.prediction_package != ObjectIdentity.from_record(package.package_id, package):
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_TARGET_PACKAGE_ISSUE_MISMATCH")
    target = package.targets[0]
    digest = sha256(config.generator.seed.encode()).digest()
    outcomes: list[MetatheoryTargetOutcome] = []
    if config.generator.target_present:
        for cell in package.forecast_cells:
            categorical = None
            metric = None
            transition = None
            first_passage = None
            censored = False
            if cell.predictive_level is MetatheoryPredictiveLevel.CATEGORICAL:
                assert cell.categorical is not None
                categorical = (
                    cell.categorical.predicted_category_id
                    if digest[0] % 2 == 0
                    else cell.categorical.unsafe_observed_category_ids[0]
                )
            elif cell.predictive_level is MetatheoryPredictiveLevel.METRIC:
                assert cell.metric is not None
                metric = NamedDecimal(
                    value_id=f"observed.{cell.cell_id}",
                    value=(
                        (cell.metric.interval_lower.value + cell.metric.interval_upper.value)
                        / Decimal(2)
                        if digest[1] % 2 == 0
                        else cell.metric.interval_upper.value + Decimal(1)
                    ),
                    unit=cell.metric.interval_lower.unit,
                )
            else:
                assert cell.dynamical is not None
                if digest[2] % 2 == 0:
                    transition = cell.dynamical.transition_id
                    first_passage = NamedDecimal(
                        value_id=f"observed.{cell.cell_id}",
                        value=(
                            cell.dynamical.first_passage_lower.value
                            + cell.dynamical.first_passage_upper.value
                        )
                        / Decimal(2),
                        unit=cell.dynamical.first_passage_lower.unit,
                    )
                else:
                    censored = True
            outcomes.append(
                MetatheoryTargetOutcome(
                    outcome_id=f"outcome.{cell.cell_id}",
                    forecast_cell_id=cell.cell_id,
                    target_id=cell.target_id,
                    physical_unit_id=cell.physical_unit_id,
                    predictive_level=cell.predictive_level,
                    observed_category_id=categorical,
                    observed_metric=metric,
                    observed_transition_id=transition,
                    observed_first_passage=first_passage,
                    censored=censored,
                    evidence_links=(),
                )
            )
    return SourceFreePropertyTransportSealedOutcomeBundle(
        bundle_id=f"sealed-outcomes.{issue.receipt_id}",
        prediction_issue=ObjectIdentity.from_record(issue.receipt_id, issue),
        target_id=target.target_id,
        outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
        acquisition_complete=config.generator.target_present,
        outcomes_exposed=False,
        grants_authority=False,
    )


def _adjudicate(
    *,
    contract: MetatheoryStageOwnerContract,
    config: SourceFreePropertyTransportRevealAdjudicationConfig,
    package: MetatheoryPredictionPackage,
    issue: MetatheoryPredictionIssueReceipt,
    acquisition: MetatheoryTargetAcquisitionIndex,
    bundle: SourceFreePropertyTransportSealedOutcomeBundle,
) -> CustodyBoundMetatheoryAdjudicationResult | MetatheoryAdjudicationStop:
    if bundle.prediction_issue != ObjectIdentity.from_record(issue.receipt_id, issue):
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_SEALED_BUNDLE_ISSUE_MISMATCH")
    reveal = MetatheoryRevealAuthorizationBinding(
        binding_id=f"reveal-binding.{issue.receipt_id}",
        reveal_authority=config.reveal_authority,
        prediction_issue=ObjectIdentity.from_record(issue.receipt_id, issue),
        acquisition_index=ObjectIdentity.from_record(acquisition.index_id, acquisition),
        reveal_permitted=config.reveal_permitted,
        reason_codes=() if config.reveal_permitted else ("REVEAL_AUTHORITY_NOT_GRANTED",),
    )
    sealed = {value.target_id: value for value in acquisition.sealed_targets}
    forecasts = {value.cell_id: value for value in package.forecast_cells}
    bindings = []
    for outcome in bundle.outcomes:
        target = sealed[outcome.target_id]
        forecast = forecasts[outcome.forecast_cell_id]
        bindings.append(
            RevealedTargetOutcomeBinding(
                binding_id=f"outcome-binding.{outcome.outcome_id}",
                prediction_issue=ObjectIdentity.from_record(issue.receipt_id, issue),
                acquisition_index=ObjectIdentity.from_record(acquisition.index_id, acquisition),
                sealed_target=ObjectIdentity.from_record(target.target_id, target),
                sealed_artifact=target.sealed_artifact,
                forecast_cell=ObjectIdentity.from_record(forecast.cell_id, forecast),
                outcome=ObjectIdentity.from_record(outcome.outcome_id, outcome),
                target_id=outcome.target_id,
                physical_unit_id=outcome.physical_unit_id,
                predictive_level=outcome.predictive_level,
                reveal_authorization=ObjectIdentity.from_record(reveal.binding_id, reveal),
                revealed_view_publication=ObjectIdentity(
                    f"{config.revealed_publication_namespace}.{outcome.outcome_id}",
                    'empirical-lawhood/runtime/structural-transport/revealed-view-publication-reference',
                    "1.0.0",
                    sha256(
                        f"{config.revealed_publication_namespace}:{outcome.outcome_id}".encode()
                    ).hexdigest(),
                ),
                revealed_view_recovery=ObjectIdentity(
                    f"{config.revealed_recovery_namespace}.{outcome.outcome_id}",
                    'empirical-lawhood/runtime/structural-transport/revealed-view-recovery-reference',
                    "1.0.0",
                    sha256(
                        f"{config.revealed_recovery_namespace}:{outcome.outcome_id}".encode()
                    ).hexdigest(),
                ),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            )
        )
    spec = replace(
        config.spec_template,
        prediction_package=ObjectIdentity.from_record(package.package_id, package),
    )
    return RevealAdjudicationOwnerRunner(
        contract,
        MetatheoryAdjudicationService(config.scoring_registry),
    ).run(
        package=package,
        issue=issue,
        acquisition=acquisition,
        reveal=reveal,
        spec=spec,
        dependence_assessment=config.dependence_assessment,
        outcomes=bundle.outcomes,
        outcome_bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
    )


def _decisive_terminal(records: tuple[CanonicalRecord, ...]) -> CanonicalRecord:
    candidates = tuple(
        value
        for value in records
        if isinstance(
            value,
            (
                ScientificSourceQualificationResult,
                ScientificSourceQualificationStop,
                CoordinateChallengeResult,
                AtlasQualificationResult,
                FacePairedPropertySurvivalAssessment,
                CustodyBoundMetatheoryAdjudicationResult,
                MetatheoryAdjudicationStop,
            ),
        )
    )
    if len(candidates) != 1:
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_OBSTRUCTION_REQUIRES_ONE_DECISIVE_TERMINAL")
    return candidates[0]


def _terminal_dispositions(
    terminal: CanonicalRecord,
) -> tuple[tuple[MetatheoryCellDisposition, tuple[str, ...]], ...]:
    if isinstance(terminal, FacePairedPropertySurvivalAssessment):
        signature = terminal.property_assessment.signature
        reasons = tuple(
            sorted(
                {reason for cell in terminal.property_assessment.cells for reason in cell.reason_codes}
                | set(terminal.property_assessment.reason_codes)
            )
        )
        if signature.opposed_member_ids:
            return ((MetatheoryCellDisposition.OPPOSED, reasons),)
        if signature.unevaluable_member_ids:
            return ((MetatheoryCellDisposition.UNEVALUABLE, reasons),)
        return ((MetatheoryCellDisposition.SUPPORTED, ()),)
    if isinstance(terminal, CustodyBoundMetatheoryAdjudicationResult):
        values = {
            cell.disposition: tuple(
                sorted(
                    {
                        reason
                        for candidate in terminal.adjudication_result.cells
                        if candidate.disposition is cell.disposition
                        for reason in candidate.reason_codes
                    }
                )
            )
            for cell in terminal.adjudication_result.cells
        }
        return tuple(
            sorted(
                values.items(),
                key=lambda value: value[0].value,
            )
        )
    elif isinstance(terminal, (ScientificSourceQualificationStop, MetatheoryAdjudicationStop)):
        return ((MetatheoryCellDisposition.UNEVALUABLE, terminal.reason_codes),)
    elif isinstance(terminal, ScientificSourceQualificationResult):
        aggregate = terminal.disposition
        reasons = tuple(
            sorted({reason for result in terminal.gate_results for reason in result.reason_codes})
        )
    elif isinstance(terminal, CoordinateChallengeResult):
        aggregate = terminal.disposition
        reasons = tuple(sorted({reason for cell in terminal.cells for reason in cell.reason_codes}))
    elif isinstance(terminal, AtlasQualificationResult):
        aggregate = terminal.disposition
        reasons = tuple(sorted({reason for cell in terminal.cells for reason in cell.reason_codes}))
    else:
        aggregate = terminal.disposition  # type: ignore[attr-defined]
        reasons = tuple(getattr(terminal, "reason_codes", ()))
    value = {
        MetatheoryAggregateDisposition.SUPPORTED: MetatheoryCellDisposition.SUPPORTED,
        MetatheoryAggregateDisposition.OPPOSED: MetatheoryCellDisposition.OPPOSED,
        MetatheoryAggregateDisposition.MIXED: MetatheoryCellDisposition.OPPOSED,
        MetatheoryAggregateDisposition.UNEVALUABLE: MetatheoryCellDisposition.UNEVALUABLE,
        MetatheoryAggregateDisposition.PREREQUISITE_NONATTEMPT: (
            MetatheoryCellDisposition.UNEVALUABLE
        ),
    }[aggregate]
    return ((value, reasons),)


def _role_for_terminal(record: CanonicalRecord) -> MetatheoryCampaignStageRole:
    return {
        ScientificSourceQualificationResult: (
            MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION
        ),
        ScientificSourceQualificationStop: (
            MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION
        ),
        CoordinateChallengeResult: MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        AtlasQualificationResult: MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
        FacePairedPropertySurvivalAssessment: MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
        CustodyBoundMetatheoryAdjudicationResult: (MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION),
        MetatheoryAdjudicationStop: (MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION),
    }[type(record)]


def _build_obstruction(
    contract: MetatheoryStageOwnerContract,
    config: SourceFreePropertyTransportObstructionCloseoutConfig,
    terminal: CanonicalRecord,
) -> ObstructionAtlas | ObstructionAtlasBuildStop:
    identity = _identity(terminal)
    role = _role_for_terminal(terminal)
    dispositions = _terminal_dispositions(terminal)
    sources = tuple(
        ObstructionSourceBinding(
            source_id=f"source.{identity.object_id}.{disposition.value.lower()}",
            expected_cell_id=f"expected.{identity.object_id}.{disposition.value.lower()}",
            terminal=identity,
            stage=role,
            target_id=config.target_id,
            claim_kind=config.claim_kind,
            predictive_level=config.predictive_level,
            affected_operand_ids=config.affected_operand_ids,
            affected_property_ids=config.affected_property_ids,
            affected_coordinate_ids=config.affected_coordinate_ids,
            affected_action_ids=config.affected_action_ids,
            evidence_links=(),
            source_reason_codes=reasons,
            scientific_disposition=disposition,
            operational_status=ObstructionOperationalStatus.COMPLETE,
        )
        for disposition, reasons in dispositions
    )
    spec = ObstructionAtlasSpec(
        spec_id=f"obstruction-spec.{identity.object_id}",
        accepted_terminal_schemas=tuple(
            sorted(
                {
                    terminal.SCHEMA,
                    *(value.terminal_schema for value in config.mapping_registry.mappings),
                }
            )
        ),
        mapping_registry=config.mapping_registry,
        expected_cells=tuple(
            ObstructionExpectedCell(
                expected_cell_id=source.expected_cell_id,
                stage=source.stage,
                target_id=source.target_id,
                required=True,
            )
            for source in sources
        ),
        aggregation_policy_id="aggregation.source-free-property-transport.preserve-all",
        allow_unknown_as_unresolved=False,
        missing_expected_as_obstruction=True,
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_structural_evidence_ceiling=config.maximum_structural_evidence_ceiling,
    )
    return ObstructionCloseoutOwnerRunner(contract, ObstructionAtlasBuilder()).run(
        spec=spec,
        sources=sources,
    )


class SourceFreePropertyTransportCapabilityProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        role: MetatheoryCampaignStageRole,
        config: CanonicalRecord,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_MANIFEST_REGISTRY_MISMATCH")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.role = role
        self.config = config
        self.runner: TaskRunner = SourceFreePropertyTransportTaskRunner(
            manifest=manifest,
            role=role,
            config=config,
            registry=registry,
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_INPUT_MISMATCH")
        return (self.runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_ACCEPTS_NO_SOURCE_RECORDS")
        specifications = tuple(
            value
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for value in task.external_inputs
        )
        records: tuple[CanonicalRecord, ...] = (self.config,)
        if isinstance(self.config, SourceFreePropertyTransportSourcePipelineConfig):
            records = (*records, self.config.prepared_medium)
        by_schema = {value.SCHEMA: value for value in records}
        if len(specifications) != len(records):
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_EXTERNAL_INPUT_ROSTER_MISMATCH")
        payloads = []
        for specification in specifications:
            expected_schema = specification.expected_payload_schema
            if expected_schema is None:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_EXTERNAL_INPUT_SCHEMA_REQUIRED")
            try:
                record = by_schema[expected_schema]
            except KeyError as error:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_EXTERNAL_INPUT_SCHEMA_MISMATCH") from error
            if specification.expected_content_sha256 not in {
                None,
                record.fingerprint(),
            } or specification.expected_media_type not in {None, SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE}:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_EXTERNAL_INPUT_IDENTITY_MISMATCH")
            visibility = specification.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
            access = specification.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND
            config_id = getattr(record, "config_id", None)
            medium_id = getattr(record, "medium_id", None)
            record_id = config_id if isinstance(config_id, str) else medium_id
            if not isinstance(record_id, str):
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_EXTERNAL_INPUT_LACKS_IDENTITY")
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            payloads.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=specification.logical_artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
                    payload=record.canonical_bytes(),
                    visibility_ceiling=visibility,
                    outcome_access=access,
                    parent_visibility_ceilings=(visibility,),
                    lineage_parents=(parent,),
                    logical_content_sha256=record.fingerprint(),
                )
            )
        return tuple(sorted(payloads, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_REGISTRY_MISMATCH")
        del execution_plan
        record_types = tuple(
            value for value in _RECORD_TYPES if value.SCHEMA in self.manifest.output_schema_ids
        )
        if ScientificAdjudicationRecord.SCHEMA in self.manifest.output_schema_ids:
            record_types = (*record_types, ScientificAdjudicationRecord)
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(cast(Any, record_type)))),
            )
            for record_type in record_types
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_PROVIDER_REGISTRY_MISMATCH")
        if self.role is not MetatheoryCampaignStageRole.REPORT:
            return None
        if execution_plan is None:
            output_id = "report"
        else:
            outputs = tuple(
                value
                for task in execution_plan.tasks
                if task.capability.capability_key == self.manifest.capability_key
                for value in task.outputs
                if value.payload_schema == ScientificAdjudicationRecord.SCHEMA
            )
            if len(outputs) != 1:
                raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_REPORT_OUTPUT_LOCATOR_MISMATCH")
            output_id = outputs[0].output_id
        assert isinstance(self.config, SourceFreePropertyTransportReportConfig)
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=output_id,
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=1_000_000,
            fixture_scope_id=self.config.fixture_scope_id,
            plumbing_only=True,
        )


def source_free_property_transport_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    role: MetatheoryCampaignStageRole,
    config: CanonicalRecord,
) -> CampaignRuntimeProvider:
    return SourceFreePropertyTransportCapabilityProvider(
        registry=registry,
        manifest=manifest,
        role=role,
        config=config,
    )


__all__ = [
    "SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE",
    'SourceFreePropertyTransportCapabilityProvider',
    'SourceFreePropertyTransportTaskRunner',
    'source_free_property_transport_provider',
]
