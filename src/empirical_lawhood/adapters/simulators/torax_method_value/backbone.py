"Translate TORAX method value evidence into the response-law and atlas route."

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.adapters.geometry.services import AtlasAssembler
from empirical_lawhood.adapters.methods import LocalLinearIdentifier
from empirical_lawhood.adapters.methods.contracts import CandidateFamilyLedger, ComponentUncertaintyFamilyAssessment, IdentificationDataset, LawIdentificationConfig, LawQualificationBatch, LawQualificationBatchCoordinate, LawQualificationBatchCoordinateDeclaration, LawQualificationCoordinateDisposition, ComponentUncertaintyCandidateAssessment, ComponentQualificationAssessment, ReceiverCriterion, StructuralCoefficientTolerance
from empirical_lawhood.adapters.methods.evidence_projection import IdentificationEvidenceProjector, TaggedIdentificationProjectionPlan, ObservationTagBinding, ObservationValuePartition, project_response_method_identification_dataset
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.adapters.methods.identification import build_response_method_identification_service
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.simulators.torax_native import build_native_torax_action_word
from empirical_lawhood.adapters.simulators.torax_native.contracts import NativeToraxAction, NativeToraxEpisodeDisposition, NativeToraxEpisode, NativeToraxTrajectory, NativeToraxView
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.identification import LawMethodKind, LawQualificationResult
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal, QuantityBound
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.geometry import (
    AtlasAssemblyObstruction,
    AtlasDomainCell,
    QualificationBatchAtlasAssemblySpec,
)
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    ExternalEvidencePayload,
    IdentificationEvidenceDomain,
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
    IdentificationManifestObservation,
    NestedCoordinateKind,
    NestedEvidenceCoordinate,
    ObservationActionDeliveryBinding,
    ObservationDisposition,
    QualificationScopeSpec,
)
from empirical_lawhood.runtime.artifacts import ArtifactManifest
from empirical_lawhood.runtime.route_conformance import PUBLIC_ROUTE_STAGE_ORDER, PublicRouteConformanceStage

from .contracts import ToraxMethodValueArm, ToraxMethodValueBackboneArmReceipt, ToraxMethodValueBackboneProjectionConfig, ToraxMethodValueBackboneStageDisposition, ToraxMethodValueBackboneStageStatus, ToraxMethodValueExperimentSpec, ToraxMethodValueObservationActionBinding, ToraxMethodValueResponseMethodActionCompatibilityReceipt, ToraxMethodValueTaskRole, ToraxMethodValueTaskSpec


@dataclass(frozen=True, slots=True)
class ToraxMethodValueBackboneEvidenceCell:
    task: ToraxMethodValueTaskSpec
    action: NativeToraxAction
    view: NativeToraxView
    preparation_id: str
    trajectory: NativeToraxTrajectory
    episode: NativeToraxEpisode
    artifact_manifest: ArtifactManifest


@dataclass(frozen=True, slots=True)
class ToraxMethodValueBackboneArmExecution:
    projection_config: ToraxMethodValueBackboneProjectionConfig
    manifest: IdentificationEvidenceManifest
    projection: IdentificationEvidenceProjection
    action_compatibility: ToraxMethodValueResponseMethodActionCompatibilityReceipt
    dataset: object
    law_config: LawIdentificationConfig
    family: ComponentUncertaintyFamilyAssessment
    profile: ComponentQualificationAssessment
    qualification_result: LawQualificationResult
    qualification_batch: LawQualificationBatch
    atlas_spec: QualificationBatchAtlasAssemblySpec
    atlas_obstruction: AtlasAssemblyObstruction
    receipt: ToraxMethodValueBackboneArmReceipt


class _CompactSource:
    def __init__(self, values: dict[str, tuple[NamedDecimal, ...]]) -> None:
        self._values = {
            observation_id: {value.value_id: value for value in row}
            for observation_id, row in values.items()
        }

    def read_compact_values(
        self,
        manifest: IdentificationEvidenceManifest,
        observation: IdentificationManifestObservation,
        required_value_ids: tuple[str, ...],
    ) -> tuple[NamedDecimal, ...]:
        del manifest
        values = self._values[observation.observation_id]
        return tuple(values[value_id] for value_id in required_value_ids)


class _RetainingFamilyAssembler(CandidateFamilyAssembler):
    def __init__(self) -> None:
        object.__setattr__(self, "family", None)

    def assemble(
        self,
        ledger: CandidateFamilyLedger,
        assessments: tuple[ComponentUncertaintyCandidateAssessment, ...],
    ) -> ComponentUncertaintyFamilyAssessment:
        self.family = CandidateFamilyAssembler().assemble(ledger, assessments)
        return self.family


def _named(value_id: str, value: Decimal, unit: str) -> NamedDecimal:
    return NamedDecimal(value_id=value_id, value=value, unit=unit)


def _values(cell: ToraxMethodValueBackboneEvidenceCell) -> tuple[NamedDecimal, ...]:
    trajectory = cell.trajectory
    episode = cell.episode
    if (
        episode.disposition is not NativeToraxEpisodeDisposition.COMPLETE
        or episode.endpoint_delta_core_temperature_ev is None
        or episode.endpoint_core_edge_contrast_ev is None
    ):
        raise ValueError("TORAX method value compact values require a complete native episode")
    task = cell.task
    rows = (
        _named("action.proxy-heating-power", cell.action.realized_power_w, "W"),
        _named("denominator.chi-e", task.chi_e_m2_s, "m2/s"),
        _named("denominator.chi-i", task.chi_i_m2_s, "m2/s"),
        _named("denominator.core-density", task.core_electron_density_m3, "m-3"),
        _named("denominator.core-te", task.core_electron_temperature_ev, "eV"),
        _named("denominator.plasma-current", task.plasma_current_a, "A"),
        _named("denominator.source-location", task.source_radial_location, "1"),
        _named("denominator.source-width", task.source_width, "1"),
        _named(
            "denominator.target-delta-core-te",
            task.target_delta_core_temperature_ev,
            "eV",
        ),
        _named("history.core-edge-contrast", trajectory.core_edge_contrast_ev[0], "eV"),
        _named("history.core-te", trajectory.core_electron_temperature_ev[0], "eV"),
        _named(
            "history.radial-mean-te",
            trajectory.radial_mean_electron_temperature_ev[0],
            "eV",
        ),
        _named(
            "receiver.delta-core-te",
            episode.endpoint_delta_core_temperature_ev,
            "eV",
        ),
        _named(
            "receiver.endpoint-core-edge-contrast",
            episode.endpoint_core_edge_contrast_ev,
            "eV",
        ),
    )
    return tuple(sorted(rows, key=lambda value: value.value_id))


def _external_payload(manifest: ArtifactManifest) -> ExternalEvidencePayload:
    publication = manifest.publication
    if publication is None:
        raise ValueError("TORAX method value trajectory lacks its committed publication binding")
    materialization = manifest.materialization
    logical = manifest.logical
    artifact = ArtifactIdentity(
        artifact_id=logical.logical_artifact_id,
        role="native-torax-trajectory",
        payload_schema=logical.payload_schema,
        sha256=materialization.physical_sha256,
        media_type=logical.media_type,
        size_bytes=materialization.size_bytes,
    )
    return ExternalEvidencePayload(
        payload_id=artifact.artifact_id,
        artifact=artifact,
        external_root_contract_id=materialization.storage_root_id,
        relative_locator=materialization.relative_path,
        publication_receipt=ObjectIdentity.from_record(
            publication.publication_batch_id,
            publication,
        ),
        recovery_identity=ObjectIdentity.from_record(
            materialization.materialization_id,
            materialization,
        ),
    )


def _delivery(
    *,
    observation_id: str,
    action_word: OccurrenceActionWord,
) -> ObservationActionDeliveryBinding:
    occurrences = tuple(sorted(action_word.occurrences, key=lambda value: value.occurrence_id))
    return ObservationActionDeliveryBinding(
        binding_id=f"delivery.{observation_id}",
        action_word=ObjectIdentity.from_record(action_word.word_id, action_word),
        occurrence_ids=tuple(value.occurrence_id for value in occurrences),
        requested_event_ids=tuple(
            f"event.{value.occurrence_id}.requested" for value in occurrences
        ),
        accepted_event_ids=tuple(f"event.{value.occurrence_id}.accepted" for value in occurrences),
        applied_event_ids=tuple(f"event.{value.occurrence_id}.applied" for value in occurrences),
        realized_event_ids=tuple(f"event.{value.occurrence_id}.realized" for value in occurrences),
    )


def _projection_config(
    spec: ToraxMethodValueExperimentSpec,
    arm: ToraxMethodValueArm,
    cells: tuple[ToraxMethodValueBackboneEvidenceCell, ...],
    system: SystemSpec,
) -> ToraxMethodValueBackboneProjectionConfig:
    selected_task_ids = {value.task.task_id for value in cells}
    calibration = tuple(
        sorted(
            value.task_id
            for value in spec.tasks_for(ToraxMethodValueTaskRole.DEVELOPMENT_COMMON)
            if value.task_id in selected_task_ids
        )
    )
    held_out = tuple(sorted(selected_task_ids - set(calibration)))
    return ToraxMethodValueBackboneProjectionConfig(
        config_id=f"config.torax-method-value.backbone-projection.{arm.value.lower()}",
        arm=arm,
        chart_id="chart.torax-native-proxy-power",
        calibration_task_ids=calibration,
        held_out_task_ids=held_out,
        denominator_value_ids=system.relation.denominator_quantity_ids,
        history_value_ids=system.relation.history_quantity_ids,
        action_value_ids=system.relation.action_quantity_ids,
        receiver_value_ids=system.relation.receiver_quantity_ids,
        include_only_complete_native_episodes=True,
        action_delivery_lowering_supported=False,
        reference_tolerance_ev=spec.reference_tolerance_ev,
        frozen=True,
    )


def _law_config(
    spec: ToraxMethodValueExperimentSpec,
    arm: ToraxMethodValueArm,
    projection_config: ToraxMethodValueBackboneProjectionConfig,
) -> LawIdentificationConfig:
    action_id = "action.proxy-heating-power"
    terms = ("intercept", f"linear.{action_id}")
    slope_tolerance = spec.reference_tolerance_ev / Decimal("300000")
    structural = []
    for receiver_id in projection_config.receiver_value_ids:
        for term_id in terms:
            structural.append(
                StructuralCoefficientTolerance(
                    tolerance_id=(f"tolerance.{receiver_id}.{term_id.replace('.', '-')}"),
                    receiver_quantity_id=receiver_id,
                    term_id=term_id,
                    native_unit="eV" if term_id == "intercept" else "eV/W",
                    zero_absolute_tolerance=(
                        spec.reference_tolerance_ev if term_id == "intercept" else slope_tolerance
                    ),
                    maximum_refinement_difference=(
                        spec.reference_tolerance_ev if term_id == "intercept" else slope_tolerance
                    ),
                )
            )
    criteria = tuple(
        ReceiverCriterion(
            criterion_id=f"criterion.{receiver_id}",
            receiver_quantity_id=receiver_id,
            native_unit="eV",
            maximum_held_out_rmse=spec.reference_tolerance_ev,
            maximum_wrong_action_effect=spec.reference_tolerance_ev,
            minimum_baseline_improvement=spec.reference_tolerance_ev,
            aleatoric_uncertainty_bound=Decimal(0),
            aleatoric_bound_method_id="deterministic-torax-zero-noise",
            observation_uncertainty_bound=spec.reference_tolerance_ev,
            observation_bound_method_id="frozen-torax-method-value-reference-tolerance",
        )
        for receiver_id in projection_config.receiver_value_ids
    )
    return LawIdentificationConfig(
        config_id=f"config.torax-method-value.local-linear.{arm.value.lower()}",
        method_key="baseline.local-linear",
        method_version="1.0.0",
        method_kind=LawMethodKind.LOCAL_LINEAR,
        chart_id=projection_config.chart_id,
        feature_quantity_ids=(action_id,),
        receiver_quantity_ids=projection_config.receiver_value_ids,
        receiver_criteria=criteria,
        action_bounds=(
            QuantityBound(
                bound_id="bound.action.proxy-heating-power",
                quantity_id=action_id,
                native_unit="W",
                lower=Decimal("500000"),
                upper=Decimal("1100000"),
            ),
        ),
        structural_tolerances=tuple(sorted(structural, key=lambda value: value.tolerance_id)),
        mapping_assumption_ids=(
            "direct-torax-four-stage-action-identity",
            "exact-action-link-retained-in-compatibility-receipt",
            "finite-frozen-torax-method-value-development-roster",
            "numerical-views-nested-not-independent",
        ),
        polynomial_degree=1,
        minimum_recurrence_units=2,
        minimum_action_levels=3,
        uncertainty_confidence_level=Decimal("0.95"),
        maximum_residual_history_correlation=Decimal("0.1"),
        causal_strength=CausalStrength.SIMULATOR_INTERVENTION,
        representation_kind=LawRepresentationKind.FINITE_ACTION_OPERATOR,
    )


def _batch_disposition(result: LawQualificationResult) -> LawQualificationCoordinateDisposition:
    return {
        ScientificStatus.NOT_SUPPORTED: LawQualificationCoordinateDisposition.NOT_SUPPORTED,
        ScientificStatus.NOT_TESTED: (
            LawQualificationCoordinateDisposition.PREREQUISITE_OBSTRUCTION
        ),
        ScientificStatus.MIXED: LawQualificationCoordinateDisposition.MIXED,
        ScientificStatus.PARTIAL: LawQualificationCoordinateDisposition.PARTIAL,
        ScientificStatus.UNEVALUABLE: LawQualificationCoordinateDisposition.UNEVALUABLE,
    }[result.scientific_status]


def _stage_dispositions(
    *,
    arm: ToraxMethodValueArm,
    dataset: IdentificationDataset,
    family: ComponentUncertaintyFamilyAssessment,
    profile: ComponentQualificationAssessment,
    result: LawQualificationResult,
    obstruction: AtlasAssemblyObstruction,
) -> tuple[ToraxMethodValueBackboneStageDisposition, ...]:
    records = {
        PublicRouteConformanceStage.OBSERVATION: ObjectIdentity.from_record(
            dataset.dataset_id, dataset
        ),
        PublicRouteConformanceStage.CANDIDATE_FAMILY: ObjectIdentity.from_record(
            family.assessment_id, family
        ),
        PublicRouteConformanceStage.COMPONENT_QUALIFICATION: ObjectIdentity.from_record(
            profile.assessment_id, profile
        ),
    }
    stages = []
    for stage in PUBLIC_ROUTE_STAGE_ORDER:
        disposition_id = f"stage.torax-method-value-backbone.{arm.value.lower()}.{stage.value.lower()}"
        if stage in records:
            stages.append(
                ToraxMethodValueBackboneStageDisposition(
                    disposition_id=disposition_id,
                    stage=stage,
                    status=ToraxMethodValueBackboneStageStatus.COMPLETE,
                    record=records[stage],
                    reason_codes=(),
                )
            )
        elif stage is PublicRouteConformanceStage.LAW_QUALIFICATION:
            stages.append(
                ToraxMethodValueBackboneStageDisposition(
                    disposition_id=disposition_id,
                    stage=stage,
                    status=ToraxMethodValueBackboneStageStatus.OBSTRUCTED,
                    record=ObjectIdentity.from_record(result.result_id, result),
                    reason_codes=result.reason_codes,
                )
            )
        elif stage is PublicRouteConformanceStage.ATLAS:
            stages.append(
                ToraxMethodValueBackboneStageDisposition(
                    disposition_id=disposition_id,
                    stage=stage,
                    status=ToraxMethodValueBackboneStageStatus.OBSTRUCTED,
                    record=ObjectIdentity.from_record(
                        obstruction.obstruction_id,
                        obstruction,
                    ),
                    reason_codes=obstruction.reason_codes,
                )
            )
        else:
            stages.append(
                ToraxMethodValueBackboneStageDisposition(
                    disposition_id=disposition_id,
                    stage=stage,
                    status=ToraxMethodValueBackboneStageStatus.CONDITION_FALSE,
                    record=None,
                    reason_codes=("ATLAS_ZERO_SUPPORTED_LAWS",),
                )
            )
    return tuple(stages)


def execute_backbone_arm(
    *,
    spec: ToraxMethodValueExperimentSpec,
    arm: ToraxMethodValueArm,
    system: SystemSpec,
    cells: tuple[ToraxMethodValueBackboneEvidenceCell, ...],
    source_qualification: ObjectIdentity,
    projection_capability: ObjectIdentity,
    projection_implementation: ObjectIdentity,
    payload_publisher: CandidatePayloadPublisher,
    payload_reader: CandidatePayloadReader,
) -> ToraxMethodValueBackboneArmExecution:
    "Qualify TORAX method value evidence through the identification, candidate, qualification, law and atlas owners."

    native_count = len(cells)
    complete_cells = tuple(
        sorted(
            (
                value
                for value in cells
                if value.episode.disposition is NativeToraxEpisodeDisposition.COMPLETE
            ),
            key=lambda value: (
                value.task.task_id,
                value.action.action_id,
                value.view.view_id,
            ),
        )
    )
    projection_config = _projection_config(spec, arm, complete_cells, system)
    payload_by_id = {
        value.payload_id: value
        for value in (_external_payload(cell.artifact_manifest) for cell in complete_cells)
    }
    action_words = {}
    coordinates = {}
    observations = []
    compact_values = {}
    exact_bindings = []
    common_ids = set(projection_config.calibration_task_ids)
    denominator_cell_id = f"cell.torax-method-value-development.{arm.value.lower()}"
    for cell in complete_cells:
        task_coordinate_id = f"coordinate.task.{arm.value.lower()}.{cell.task.task_id}"
        action_coordinate_id = (
            f"coordinate.action.{arm.value.lower()}.{cell.task.task_id}.{cell.action.action_label}"
        )
        view_coordinate_id = (
            f"coordinate.view.{arm.value.lower()}.{cell.task.task_id}."
            f"{cell.action.action_label}.{cell.view.view_id}"
        )
        coordinates[task_coordinate_id] = NestedEvidenceCoordinate(
            coordinate_id=task_coordinate_id,
            kind=NestedCoordinateKind.PREPARATION,
            parent_coordinate_id=None,
        )
        coordinates[action_coordinate_id] = NestedEvidenceCoordinate(
            coordinate_id=action_coordinate_id,
            kind=NestedCoordinateKind.REPEATED_DELIVERY,
            parent_coordinate_id=task_coordinate_id,
        )
        coordinates[view_coordinate_id] = NestedEvidenceCoordinate(
            coordinate_id=view_coordinate_id,
            kind=NestedCoordinateKind.QUALIFICATION_VIEW,
            parent_coordinate_id=action_coordinate_id,
        )
        word = build_native_torax_action_word(
            cell.action,
            preparation_id=cell.preparation_id,
        )
        action_words[word.word_id] = word
        observation_id = (
            f"observation.{arm.value.lower()}.{cell.task.task_id}."
            f"{cell.action.action_label}.{cell.view.view_id}"
        )
        delivery = _delivery(observation_id=observation_id, action_word=word)
        exact_bindings.append(
            ToraxMethodValueObservationActionBinding(
                binding_id=f"action-map.{observation_id}",
                observation_id=observation_id,
                delivery=delivery,
            )
        )
        payload = _external_payload(cell.artifact_manifest)
        observations.append(
            IdentificationManifestObservation(
                observation_id=observation_id,
                physical_unit_instance_id=(f"unit.{arm.value.lower()}.{cell.task.task_id}"),
                nested_coordinate_ids=tuple(
                    sorted(
                        (
                            task_coordinate_id,
                            action_coordinate_id,
                            view_coordinate_id,
                        )
                    )
                ),
                denominator_cell_id=denominator_cell_id,
                chart_id=projection_config.chart_id,
                split_id=("calibration" if cell.task.task_id in common_ids else "held-out"),
                role_id="primary",
                member_id=f"member.torax-method-value-development.{arm.value.lower()}",
                candidate_version_id=f"candidate-version.torax-method-value.{arm.value.lower()}",
                qualification_view_id=cell.view.view_id,
                receiver_id="receiver.torax-core-te-and-contrast",
                receiver_clock_id="torax-episode-clock",
                native_frame_id="torax-temperature",
                payload_id=payload.payload_id,
                payload_member_locator="trajectory",
                action_delivery=None,
                disposition=ObservationDisposition.COMPLETE,
                reason_codes=(),
            )
        )
        compact_values[observation_id] = _values(cell)
    observations_tuple = tuple(sorted(observations, key=lambda value: value.observation_id))
    independent_units = tuple(
        sorted({value.physical_unit_instance_id for value in observations_tuple})
    )
    scope = QualificationScopeSpec(
        scope_id=f"scope.torax-method-value-development.{arm.value.lower()}",
        claim_id=f"claim.torax-method-value-development.{arm.value.lower()}",
        population_id="population.torax-method-value-finite-development-roster",
        physical_unit_type_id=system.independent_unit.unit_id,
        independent_unit_instance_ids=independent_units,
        nested_coordinate_ids=tuple(sorted(coordinates)),
        aggregation_level_id=system.independent_unit.unit_id,
        uncertainty_unit_id=system.independent_unit.unit_id,
        locality_scope_id=system.system_id,
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        binding_id=f"claim-unit.torax-method-value-development.{arm.value.lower()}",
        claim_id=scope.claim_id,
        scope=ObjectIdentity.from_record(scope.scope_id, scope),
        independent_unit_instance_ids=independent_units,
        aggregation_level_id=scope.aggregation_level_id,
    )
    manifest = IdentificationEvidenceManifest(
        manifest_id=f"identification-manifest.torax-method-value-development.{arm.value.lower()}",
        system=ObjectIdentity.from_record(system.system_id, system),
        relation=system.relation,
        source_qualification=source_qualification,
        runtime_qualification=spec.runtime_qualification,
        observation_operator_qualification=source_qualification,
        evidence_world_id=system.world.world_id,
        evidence_domain=IdentificationEvidenceDomain.DEVELOPMENT,
        allowed_consumer_ids=("baseline.local-linear",),
        information_cutoff_id=f"cutoff.torax-method-value-development.{arm.value.lower()}",
        payloads=tuple(sorted(payload_by_id.values(), key=lambda value: value.payload_id)),
        coordinates=tuple(sorted(coordinates.values(), key=lambda value: value.coordinate_id)),
        action_words=tuple(sorted(action_words.values(), key=lambda value: value.word_id)),
        observations=observations_tuple,
        qualification_scopes=(scope,),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        parent_visibility_ceilings=(VisibilityCeiling.DEVELOPMENT_ONLY,),
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
    )
    action_compatibility = ToraxMethodValueResponseMethodActionCompatibilityReceipt(
        receipt_id=f"response-method-action-compatibility.torax-method-value.{arm.value.lower()}",
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        bindings=tuple(sorted(exact_bindings, key=lambda value: value.binding_id)),
        exact_action_words_retained=True,
        response_method_observation_links_retained=False,
        reason_codes=("RESPONSE_METHOD_ACTION_WORD_LINK_UNREPRESENTABLE",),
    )
    partition = ObservationValuePartition(
        partition_id=f"partition.torax-method-value-development.{arm.value.lower()}",
        denominator_value_ids=projection_config.denominator_value_ids,
        history_value_ids=projection_config.history_value_ids,
        action_value_ids=projection_config.action_value_ids,
        receiver_value_ids=projection_config.receiver_value_ids,
        tags=(),
    )
    plan = TaggedIdentificationProjectionPlan(
        plan_id=f"projection-plan.torax-method-value-development.{arm.value.lower()}",
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        allowed_consumer_id="baseline.local-linear",
        projection_capability=projection_capability,
        projection_config=ObjectIdentity.from_record(
            projection_config.config_id,
            projection_config,
        ),
        projection_implementation=projection_implementation,
        claim_unit_binding=binding,
        value_partition=partition,
        observation_tags=tuple(
            ObservationTagBinding(
                observation_id=value.observation_id,
                tags=(
                    "exact-action-map-retained-in-compatibility-receipt",
                    "native-complete-safety-valid",
                    "numerical-view-nested",
                ),
            )
            for value in observations_tuple
        ),
    )
    projection = IdentificationEvidenceProjector().project(
        system,
        manifest,
        plan,
        _CompactSource(compact_values),
    )
    dataset = project_response_method_identification_dataset(
        dataset_id=f"dataset.torax-method-value-development.{arm.value.lower()}",
        manifest=manifest,
        projection=projection,
        one_factor_exchanges=(),
    )
    law_config = _law_config(spec, arm, projection_config)
    capture = _RetainingFamilyAssembler()
    service = replace(
        build_response_method_identification_service(
            payload_publisher=payload_publisher,
            payload_reader=payload_reader,
        ),
        family_assembler=capture,
    )
    result = service.qualify_law(
        system,
        dataset,
        law_config,
        LocalLinearIdentifier(),
    )
    family = capture.family
    if family is None:
        raise RuntimeError("sole law service did not retain its candidate family")
    selected = family.assessments[0]
    profile = selected.profile_assessment
    if result.response_law is not None:
        raise RuntimeError("TORAX method value development unexpectedly produced a response law despite absent falsifiers")
    member_ids = result.qualification_trace.denominator_member_ids
    if len(member_ids) != 1:
        raise RuntimeError("TORAX method value law result changed its denominator-member cardinality")
    coordinate_id = f"coordinate.batch.torax-method-value.{arm.value.lower()}"
    domain_cell_id = f"domain-cell.torax-method-value.{arm.value.lower()}"
    declaration = LawQualificationBatchCoordinateDeclaration(
        coordinate_id=coordinate_id,
        domain_cell_id=domain_cell_id,
        chart_id=result.chart_id,
        denominator_cell_id=denominator_cell_id,
        denominator_member_id=member_ids[0],
    )
    coordinate = LawQualificationBatchCoordinate(
        coordinate_id=coordinate_id,
        domain_cell_id=domain_cell_id,
        chart_id=result.chart_id,
        denominator_cell_id=denominator_cell_id,
        denominator_member_id=member_ids[0],
        qualification_result=result,
        disposition=_batch_disposition(result),
        response_law=None,
        law_constituent_id=None,
        reason_codes=result.reason_codes,
        evidence_link_ids=tuple(value.link_id for value in result.evidence_links),
    )
    batch = LawQualificationBatch(
        batch_id=f"law-qualification-batch.torax-method-value.{arm.value.lower()}",
        system_id=system.system_id,
        world_id=system.world.world_id,
        declared_coordinates=(declaration,),
        coordinates=(coordinate,),
        overlaps=(),
        evidence_links=result.evidence_links,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
    )
    domain = AtlasDomainCell(
        domain_cell_id=domain_cell_id,
        chart_id=result.chart_id,
        denominator_cell_id=denominator_cell_id,
    )
    atlas_spec = QualificationBatchAtlasAssemblySpec(
        assembly_id=f"assembly.torax-method-value.{arm.value.lower()}",
        system=ObjectIdentity.from_record(system.system_id, system),
        qualification_batch=ObjectIdentity.from_record(batch.batch_id, batch),
        domain_cells=(domain,),
        transitions=(),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
    )
    atlas_result = AtlasAssembler().assemble_batch(
        system=system,
        batch=batch,
        spec=atlas_spec,
        evidence_links=result.evidence_links,
    )
    if not isinstance(atlas_result, AtlasAssemblyObstruction):
        raise RuntimeError("zero-law TORAX method value batch unexpectedly assembled an atlas")
    stages = _stage_dispositions(
        arm=arm,
        dataset=dataset,
        family=family,
        profile=profile,
        result=result,
        obstruction=atlas_result,
    )
    receipt = ToraxMethodValueBackboneArmReceipt(
        receipt_id=f"backbone-arm-receipt.torax-method-value.{arm.value.lower()}",
        experiment=ObjectIdentity.from_record(spec.experiment_id, spec),
        arm=arm,
        system=ObjectIdentity.from_record(system.system_id, system),
        projection_config=ObjectIdentity.from_record(
            projection_config.config_id,
            projection_config,
        ),
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        projection=ObjectIdentity.from_record(projection.projection_id, projection),
        action_compatibility=ObjectIdentity.from_record(
            action_compatibility.receipt_id,
            action_compatibility,
        ),
        dataset=ObjectIdentity.from_record(dataset.dataset_id, dataset),
        candidate_family_assessment=ObjectIdentity.from_record(
            family.assessment_id,
            family,
        ),
        qualification_profile_assessment=ObjectIdentity.from_record(
            profile.assessment_id,
            profile,
        ),
        qualification_result=ObjectIdentity.from_record(result.result_id, result),
        qualification_batch=ObjectIdentity.from_record(batch.batch_id, batch),
        atlas_obstruction=ObjectIdentity.from_record(
            atlas_result.obstruction_id,
            atlas_result,
        ),
        selected_task_ids=tuple(sorted({value.task.task_id for value in cells})),
        native_episode_count=native_count,
        projected_episode_count=len(complete_cells),
        excluded_invalid_episode_count=native_count - len(complete_cells),
        scientific_status=result.scientific_status,
        highest_supported_rung=result.highest_supported_rung,
        stages=stages,
        terminal=True,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    return ToraxMethodValueBackboneArmExecution(
        projection_config=projection_config,
        manifest=manifest,
        projection=projection,
        action_compatibility=action_compatibility,
        dataset=dataset,
        law_config=law_config,
        family=family,
        profile=profile,
        qualification_result=result,
        qualification_batch=batch,
        atlas_spec=atlas_spec,
        atlas_obstruction=atlas_result,
        receipt=receipt,
    )


__all__ = [
    'ToraxMethodValueBackboneArmExecution',
    'ToraxMethodValueBackboneEvidenceCell',
    'execute_backbone_arm',
]
