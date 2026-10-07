"""No-effect design builders for the Six-matrix response prospective reactive source law."""

from __future__ import annotations

from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_source_scientific_inputs import MatrixReactiveSourceAnalysisScientificInput, MatrixReactiveSourceRoutingScientificInput, reactive_source_history_scientific_inputs
from empirical_lawhood.adapters.methods.matrix_response_study.prospective_reactive_source_law import MatrixResponseProspectiveReactiveSourceLawFrozenScore, MatrixResponseProspectiveReactiveSourceLawMethodConfig, MatrixResponseProspectiveReactiveSourceLawProjection
from empirical_lawhood.adapters.simulators.six_matrix_response.executable_binding import PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING
from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID, SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, SixMatrixResponseProspectiveReactiveSourceLawTranche, build_six_matrix_response_reactive_source_slots, six_matrix_response_reactive_source_scientific_fingerprint
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignOwnerRole, LinkedCampaignProfile
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseQualificationConfig, ResponseAcquisitionView, ResponsePreparationUnit, ResponseDataSplitRole, ResponseExperimentExtensionSet
from empirical_lawhood.planning.prospective_config import ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.response_experiment_ports import NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding

from .contracts import MatrixResponseDesignBinding
from .causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig


def _owner(profile: LinkedCampaignProfile, role: LinkedCampaignOwnerRole) -> ObjectIdentity:
    return next(value.owner for value in profile.owners if value.role is role)


def build_matrix_response_study_reactive_source_source_config(
    tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche,
    *,
    design: MatrixResponseDesignBinding,
    routing_scientific_inputs: tuple[MatrixReactiveSourceRoutingScientificInput, ...],
    original_routing_census_sha256: str,
    frozen_score: MatrixResponseProspectiveReactiveSourceLawFrozenScore | None = None,
) -> SixMatrixResponseProspectiveReactiveSourceLawSourceConfig:
    source = design.source_config
    member = next(
        value
        for value in source.anisotropic_model.family_members
        if value.member_id == "six-matrix-response.member.mass-0p5.cross-coupling-1"
    )
    suffix = tranche.value.lower()
    return SixMatrixResponseProspectiveReactiveSourceLawSourceConfig(
        config_id=f"matrix-response-prospective-reactive-source-law.source-config.{suffix}",
        config_version="1.0.0",
        tranche=tranche,
        scientific_fingerprint=six_matrix_response_reactive_source_scientific_fingerprint(),
        seed_namespace_id=f"seed.matrix-response-prospective-reactive-source-law-{suffix}-complete-history",
        member=member,
        primary_view=source.primary_view,
        fine_view=source.secondary_view,
        slots=build_six_matrix_response_reactive_source_slots(tranche),
        scientific_history_inputs=reactive_source_history_scientific_inputs(
            tranche_ordinal=0 if tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else 1
        ),
        routing_scientific_inputs=routing_scientific_inputs,
        original_routing_census_sha256=original_routing_census_sha256,
        frozen_score=(
            None
            if frozen_score is None
            else ObjectIdentity.from_record(frozen_score.score_id, frozen_score)
        ),
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_source_method_config(
    source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    *,
    response_factor_config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    analysis_scientific_input: MatrixReactiveSourceAnalysisScientificInput,
    frozen_score: MatrixResponseProspectiveReactiveSourceLawFrozenScore | None = None,
) -> MatrixResponseProspectiveReactiveSourceLawMethodConfig:
    suffix = source.tranche.value.lower()
    return MatrixResponseProspectiveReactiveSourceLawMethodConfig(
        config_id=f"matrix-response-prospective-reactive-source-law.method-config.{suffix}",
        config_version="1.0.0",
        scientific_fingerprint=six_matrix_response_reactive_source_scientific_fingerprint(),
        source_config=ObjectIdentity.from_record(source.config_id, source),
        response_factor_config=response_factor_config,
        tranche=source.tranche,
        analysis_scientific_input=analysis_scientific_input,
        frozen_score=frozen_score,
        projection_task_prefix="matrix-response-prospective-reactive-source-law-project",
        aggregate_task_id="linked-method-identification",
        bootstrap_replicates=10_000,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _lineage(tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche) -> ProspectivePackageLineage:
    suffix = tranche.value.lower()
    condition = "PROSPECTIVE_REACTIVE_SOURCE_IMPLEMENTATION_FROZEN" if tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else "DEVELOPMENT_SCORE_FROZEN"
    node = ProspectivePackageLineageNode(
        package_id=f"package.matrix-response-prospective-reactive-source-law-{suffix}",
        package_label=f"matrix response prospective reactive source law {tranche.value} complete-history source tranche",
        kind=ProspectivePackageKind.EXCLUDED_QUALIFICATION,
        parent_package_ids=(),
        issue_condition_ids=(f"condition.{condition.lower().replace('_', '-')}",),
        maximum_evidence=EvidenceCeiling.ORDER_RELATION,
        current_linked_campaign_role=None,
        contains_action_selection_study=False,
        contains_controller=False,
        contains_compiler=False,
        contains_current_admission_or_controller_evaluation_record=False,
    )
    return ProspectivePackageLineage(
        lineage_id=f"lineage.matrix-response-prospective-reactive-source-law-{suffix}",
        shared_qualification_package_id=node.package_id,
        conditional_branch_id=f"branch.matrix-response-prospective-reactive-source-law-{suffix}-only",
        nodes=(node,),
        mutually_exclusive_with_branch_ids=(f"branch.matrix-response-prospective-reactive-source-law-{suffix}-rescue-forbidden",),
        route_predeclared_before_parent_issue=True,
        outcome_dependent_package_construction=tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.PROSPECTIVE_EVALUATION,
        grants_issue_or_execution_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _terminal_matrix(tranche: SixMatrixResponseProspectiveReactiveSourceLawTranche) -> ProspectiveTerminalMatrix:
    suffix = tranche.value.lower()
    return ProspectiveTerminalMatrix(
        matrix_id=f"terminal-matrix.matrix-response-prospective-reactive-source-law-{suffix}",
        selector_transitions=(ProspectiveSelectorTransition(
            precedence=1,
            condition="IMPLEMENTATION_FROZEN" if tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else "DEVELOPMENT_SCORE_FROZEN",
            transition=f"EXECUTE_PROSPECTIVE_REACTIVE_SOURCE_LAW_{tranche.value}",
        ),),
        terminal_rules=(ProspectiveTerminalRule(
            precedence=1,
            first_controlling_condition=f"{tranche.value}_TERMINAL",
            terminal_disposition="RETAIN_FROZEN_SOURCE_TERMINAL",
            highest_retained_object='MatrixResponseProspectiveReactiveSourceLawAggregate',
            downstream_consequence="PROSPECTIVE_EVALUATION_CONDITIONAL" if tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.DEVELOPMENT else "SUCCESSORS_SEPARATELY_PLANNED",
        ),),
        first_applicable_terminal_wins=True,
        valid_negative_mixed_partial_stopped_or_unevaluable_is_complete=True,
        postissue_alternate_rescue_allowed=False,
        grants_authority_or_followon_action=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_source_extension_set(
    *,
    evidence_profile: EvidenceProfileSelection,
    source_profile: SourcePipelineProfile,
    linked_profile: LinkedCampaignProfile,
    source_config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
    method_config: MatrixResponseProspectiveReactiveSourceLawMethodConfig,
) -> ResponseExperimentExtensionSet:
    evidence_identity = ObjectIdentity.from_record(evidence_profile.selection_id, evidence_profile)
    source_identity = ObjectIdentity.from_record(source_profile.profile_id, source_profile)
    linked_identity = ObjectIdentity.from_record(linked_profile.profile_id, linked_profile)
    method_identity = ObjectIdentity.from_record(method_config.config_id, method_config)
    if (
        linked_profile.evidence_profile_selection != evidence_identity
        or linked_profile.source_profile != source_identity
        or linked_profile.projection_config != method_identity
        or linked_profile.method_config != method_identity
        or method_config.source_config != ObjectIdentity.from_record(source_config.config_id, source_config)
    ):
        raise ValueError("matrix response prospective reactive source law strict root lineage differs")
    units = tuple(
        ResponsePreparationUnit(
            physical_independent_unit_id=slot.physical_independent_unit_id,
            preparation_coordinate_id=f"coordinate.matrix-response-prospective-reactive-source-law-{source_config.tranche.value.lower()}-{slot.family_id.removeprefix('six-matrix-response.history.')}",
            preparation_instance_id=slot.preparation_instance_id,
            preparation_sha256=source_config.scientific_fingerprint,
            adapter_realization_id=f"seed.matrix-response-prospective-reactive-source-law-{source_config.tranche.value.lower()}.h{slot.history_index:03d}",
            split_role=ResponseDataSplitRole.DEVELOPMENT if slot.history_index < 128 else ResponseDataSplitRole.EVALUATION,
        )
        for slot in source_config.slots
    )
    groups: list[ResponseAcquisitionGroup] = []
    views: list[ResponseAcquisitionView] = []
    for slot in source_config.slots:
        pairs = [(slot.primary_acquisition_group_id, slot.primary_view_id, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID)]
        if slot.fine_acquisition_group_id is not None and slot.fine_view_id is not None:
            pairs.append((slot.fine_acquisition_group_id, slot.fine_view_id, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID))
        for group, view, member in pairs:
            groups.append(ResponseAcquisitionGroup(group, slot.physical_independent_unit_id, slot.preparation_instance_id, (view,)))
            views.append(ResponseAcquisitionView(view, group, slot.physical_independent_unit_id, member, None))
    physical_ids = tuple(value.physical_independent_unit_id for value in units)
    qualification_config = ResponseQualificationConfig(
        config_id=f"law-qualification-config.matrix-response-prospective-reactive-source-law-{source_config.tranche.value.lower()}",
        evidence_profile_selection=evidence_identity,
        source_pipeline_profile=source_identity,
        linked_campaign_profile=linked_identity,
        action_chart=None,
        package_lineage=_lineage(source_config.tranche),
        terminal_matrix=_terminal_matrix(source_config.tranche),
        physical_units=units,
        acquisition_groups=tuple(sorted(groups, key=lambda value: value.acquisition_group_id)),
        nested_views=tuple(sorted(views, key=lambda value: value.view_id)),
        identification_admission_physical_independent_unit_ids=physical_ids,
        development_unit_ids=physical_ids[:128],
        evaluation_unit_ids=physical_ids[128:],
        numerical_member_ids=tuple(sorted((SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID))),
        projection_config=method_identity,
        method_config=method_identity,
        qualification_profile=linked_profile.qualification_profile,
        law_finalizer_owner=_owner(linked_profile, LinkedCampaignOwnerRole.SOLE_LAW_FINALIZER),
        batch_atlas_owner=_owner(linked_profile, LinkedCampaignOwnerRole.BATCH_ATLAS),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    if set(linked_profile.rosters.physical_independent_unit_ids) != set(physical_ids):
        raise ValueError("matrix response prospective reactive source law linked physical roster differs")
    return ResponseExperimentExtensionSet(
        extension_set_id=f"extension-set.matrix-response-prospective-reactive-source-law-{source_config.tranche.value.lower()}",
        evidence_profile_selection=evidence_identity,
        source_pipeline_profile=source_identity,
        linked_campaign_profile=linked_identity,
        identification_config=qualification_config,
        admission_config=None,
        prospective_evaluation_config=None,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_source_substrate_binding(source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig) -> ResponseSubstrateBinding:
    suffix = source.tranche.value.lower()
    return ResponseSubstrateBinding(
        binding_id=f"substrate-binding.matrix-response-prospective-reactive-source-law-{suffix}",
        medium_id="medium.six-matrix-response-six-matrix-complete-history",
        interaction_kind=NativeInteractionKind.READ_ONLY_ACQUISITION,
        provider_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_key,
        provider_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_version,
        provider_implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.implementation_sha256,
        installed_executable_binding=ObjectIdentity.from_record(PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING.binding_id, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING),
        source_profile_schema=SourcePipelineProfile.SCHEMA,
        native_request_schema=SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest.SCHEMA,
        native_object_schema=SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA,
        native_episode_schema=None,
        native_projection_schema=MatrixResponseProspectiveReactiveSourceLawProjection.SCHEMA,
        native_config=ObjectIdentity.from_record(source.config_id, source),
        receivers=(NativeReceiverContract(
            receiver_id="receiver.matrix-response-prospective-reactive-source-law-complete-phase-space",
            quantity_id="quantity.matrix-response-prospective-reactive-source-law-positions-momenta-innovations-parameters",
            native_unit="heterogeneous-native-phase-space",
            frame_id="frame.six-matrix-response-six-matrix-hermitian",
            direction_id="direction.six-matrix-response-anonymous-receiver",
            clock_ids=("clock.six-matrix-response-native-integration-step",),
        ),),
        clocks=(NativeClockContract(
            clock_id="clock.six-matrix-response-native-integration-step",
            native_unit="integration-step",
            frame_id="frame.six-matrix-response-native-integration-time",
        ),),
        action_contract=None,
        external_source_contacted=False,
        scientific_verdict_constructed=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    'build_matrix_response_study_reactive_source_extension_set', 'build_matrix_response_study_reactive_source_method_config',
    'build_matrix_response_study_reactive_source_source_config', 'build_matrix_response_study_reactive_source_substrate_binding',
]
