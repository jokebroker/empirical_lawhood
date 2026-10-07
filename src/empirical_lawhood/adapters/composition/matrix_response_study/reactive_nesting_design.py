"""Exact outcome-blind reactive entrance parameterised design binding.

This module only joins already-authored strict profiles to the fixed matrix response reactive entrance
source and method records.  It performs no source contact, issue, execution,
reveal, or scientific reduction.
"""

from __future__ import annotations

from hashlib import sha256
from decimal import Decimal

from empirical_lawhood.adapters.methods.matrix_response_study.reactive_entrance_source import MatrixResponseReactiveEntranceImplementationQualification, MatrixResponseReactiveEntranceMethodConfig, MatrixResponseReactiveEntranceProjection
from empirical_lawhood.adapters.methods.matrix_response_study.reactive_nesting import MatrixResponseRNSourceCompatibilityReceipt
from empirical_lawhood.adapters.simulators.six_matrix_response.executable_binding import REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING
from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import REACTIVE_ENTRANCE_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance import REACTIVE_ENTRANCE_BRIDGE_SEED_RULE, REACTIVE_ENTRANCE_FINE_MEMBER_ID, REACTIVE_ENTRANCE_FINE_STEPS, REACTIVE_ENTRANCE_HDF5_SCHEMA, REACTIVE_ENTRANCE_PRIMARY_MEMBER_ID, REACTIVE_ENTRANCE_PRIMARY_STEPS, REACTIVE_ENTRANCE_RECEIVER_CADENCE, REACTIVE_ENTRANCE_SOURCE_SEED_RULE, SixMatrixResponseReactiveEntrancePrecursorRequest, SixMatrixResponseReactiveEntrancePrecursorResult, SixMatrixResponseReactiveEntranceSourceConfig, build_six_matrix_response_reactive_entrance_slots
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import replay_selected_co_anneal_event
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignOwnerRole, LinkedCampaignProfile
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseQualificationConfig, ResponseAcquisitionView, ResponsePreparationUnit, ResponseDataSplitRole, ResponseExperimentExtensionSet
from empirical_lawhood.planning.prospective_config import ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.response_experiment_ports import NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding

from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance_scientific_inputs import MatrixReactiveEntranceScientificInputs

from .contracts import MatrixResponseDesignBinding
from .causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from .shooting_design import MatrixResponseShootingStudyConfig


def _owner(profile: LinkedCampaignProfile, role: LinkedCampaignOwnerRole) -> ObjectIdentity:
    return next(value.owner for value in profile.owners if value.role is role)


def build_matrix_response_study_reactive_entrance_source_config(
    *,
    design: MatrixResponseDesignBinding,
    shooting: MatrixResponseShootingStudyConfig,
    plan_sha256: str,
    source_cohort: ObjectIdentity,
    scientific_inputs: MatrixReactiveEntranceScientificInputs,
) -> SixMatrixResponseReactiveEntranceSourceConfig:
    """Reconstruct the exact step-816 reactive entrance source config without source effects."""

    if not isinstance(scientific_inputs, MatrixReactiveEntranceScientificInputs):
        raise ValueError("reactive-entrance design requires explicit original scientific inputs before replay")
    scientific_inputs.require_target_roots(
        plan_sha256=plan_sha256, source_cohort_sha256=source_cohort.object_fingerprint
    )
    source = design.source_config
    member = next(
        value for value in source.anisotropic_model.family_members if value.member_id == shooting.member_id
    )
    replay = replay_selected_co_anneal_event(
        member=member,
        numerical_view=source.primary_view,
        parent_rollout_id=shooting.parent_rollout_id,
        parent_report_sha256=shooting.parent_anisotropic_feasibility_report_sha256,
        target_x=shooting.target_alpha_tilde_x,
        target_y=shooting.target_alpha_tilde_y,
        seed_index=shooting.seed_index,
        scientific_seed_sha256=scientific_inputs.replay_scientific_seed_sha256,
        checkpoint_steps=(816,),
    )
    checkpoint = replay.checkpoints[0]
    scientific_inputs.require_target_checkpoint(checkpoint.combined_state_sha256)
    return SixMatrixResponseReactiveEntranceSourceConfig(
        config_id="matrix-response-reactive-entrance.source-config",
        config_version="1.0.0",
        plan_sha256=plan_sha256,
        source_cohort=source_cohort,
        checkpoint=checkpoint,
        member=member,
        primary_view=source.primary_view,
        half_view=source.secondary_view,
        slots=build_six_matrix_response_reactive_entrance_slots(checkpoint_sha256=checkpoint.combined_state_sha256, scientific_inputs=scientific_inputs),
        scientific_inputs=scientific_inputs,
        source_seed_rule_id=REACTIVE_ENTRANCE_SOURCE_SEED_RULE,
        bridge_seed_rule_id=REACTIVE_ENTRANCE_BRIDGE_SEED_RULE,
        fine_subset_rule_id="matrix-response-reactive-entrance.fine-subset.explicit-original-128-roster",
        primary_steps=REACTIVE_ENTRANCE_PRIMARY_STEPS,
        fine_steps=REACTIVE_ENTRANCE_FINE_STEPS,
        receiver_cadence_parent_steps=REACTIVE_ENTRANCE_RECEIVER_CADENCE,
        target_alpha_tilde_x=shooting.target_alpha_tilde_x,
        target_alpha_tilde_y=shooting.target_alpha_tilde_y,
        hdf5_payload_schema=REACTIVE_ENTRANCE_HDF5_SCHEMA,
        source_instance_count=1,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_entrance_method_config(
    *,
    response_factor_config: MatrixResponseCausalIntersectionResidenceStudyConfig,
    source_config: SixMatrixResponseReactiveEntranceSourceConfig,
    compatibility_receipt: MatrixResponseRNSourceCompatibilityReceipt,
    implementation_qualification: MatrixResponseReactiveEntranceImplementationQualification,
) -> MatrixResponseReactiveEntranceMethodConfig:
    """Bind the frozen source reducer and its pre-outcome qualification."""

    return MatrixResponseReactiveEntranceMethodConfig(
        config_id="matrix-response-reactive-entrance.method-config",
        config_version="1.0.0",
        source_config=ObjectIdentity.from_record(source_config.config_id, source_config),
        compatibility_receipt=compatibility_receipt,
        implementation_qualification=implementation_qualification,
        response_factor_config=response_factor_config,
        projection_task_prefix="matrix-response-reactive-entrance-project",
        aggregate_task_id="linked-method-identification",
        entry_start_step=256,
        entry_end_step=384,
        source_end_step=512,
        receiver_cadence_steps=16,
        direct_duration_steps=(16, 32, 128),
        hit_count_min=40,
        residence_16_steps_count_min=32,
        residence_16_steps_per_half_min=12,
        numerical_dual_entry_min=8,
        numerical_entry_time_difference_max=Decimal("0.032"),
        numerical_factor_agreement_min=Decimal("0.90"),
        confidence_level=Decimal("0.95"),
        source_instance_count=1,
        physical_independent_unit_count=512,
        numerical_view_count=128,
        no_top_up=True,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _lineage() -> ProspectivePackageLineage:
    root = ProspectivePackageLineageNode(
        package_id="package.matrix-response-reactive-entrance-source-qualification",
        package_label="Six-matrix response reactive entrance reactive-entrance source qualification",
        kind=ProspectivePackageKind.EXCLUDED_QUALIFICATION,
        parent_package_ids=(),
        issue_condition_ids=("condition.matrix-response-reactive-entrance-source-compatibility-qualified",),
        maximum_evidence=EvidenceCeiling.ORDER_RELATION,
        current_linked_campaign_role=None,
        contains_action_selection_study=False,
        contains_controller=False,
        contains_compiler=False,
        contains_current_admission_or_controller_evaluation_record=False,
    )
    return ProspectivePackageLineage(
        lineage_id="lineage.matrix-response-reactive-entrance-reactive-entrance",
        shared_qualification_package_id=root.package_id,
        conditional_branch_id="branch.matrix-response-reactive-entrance-only",
        nodes=(root,),
        mutually_exclusive_with_branch_ids=("branch.matrix-response-reactive-entrance-rescue-forbidden",),
        route_predeclared_before_parent_issue=True,
        outcome_dependent_package_construction=False,
        grants_issue_or_execution_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _terminal_matrix() -> ProspectiveTerminalMatrix:
    return ProspectiveTerminalMatrix(
        matrix_id="terminal-matrix.matrix-response-reactive-entrance-reactive-entrance",
        selector_transitions=(
            ProspectiveSelectorTransition(
                precedence=1,
                condition="SOURCE_COMPATIBILITY_QUALIFIED",
                transition="EXECUTE_FIXED_REACTIVE_ENTRANCE",
            ),
        ),
        terminal_rules=(
            ProspectiveTerminalRule(
                precedence=1,
                first_controlling_condition="REACTIVE_ENTRANCE_TERMINAL",
                terminal_disposition="RETAIN_EXACT_FIVE_WAY_TERMINAL",
                highest_retained_object='MatrixResponseReactiveEntranceAggregate',
                downstream_consequence="SEPARATE_CONDITIONAL_REACTIVE_NESTING_ISSUE",
            ),
        ),
        first_applicable_terminal_wins=True,
        valid_negative_mixed_partial_stopped_or_unevaluable_is_complete=True,
        postissue_alternate_rescue_allowed=False,
        grants_authority_or_followon_action=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_entrance_extension_set(
    *,
    evidence_profile: EvidenceProfileSelection,
    source_profile: SourcePipelineProfile,
    linked_profile: LinkedCampaignProfile,
    source_config: SixMatrixResponseReactiveEntranceSourceConfig,
    method_config: MatrixResponseReactiveEntranceMethodConfig,
) -> ResponseExperimentExtensionSet:
    """Build the exact 512-unit/640-acquisition reactive entrance extension."""

    evidence_identity = ObjectIdentity.from_record(
        evidence_profile.selection_id,
        evidence_profile,
    )
    source_profile_identity = ObjectIdentity.from_record(source_profile.profile_id, source_profile)
    linked_identity = ObjectIdentity.from_record(linked_profile.profile_id, linked_profile)
    method_identity = ObjectIdentity.from_record(method_config.config_id, method_config)
    if (
        linked_profile.evidence_profile_selection != evidence_identity
        or linked_profile.source_profile != source_profile_identity
        or linked_profile.projection_config != method_identity
        or linked_profile.method_config != method_identity
        or method_config.source_config
        != ObjectIdentity.from_record(source_config.config_id, source_config)
    ):
        raise ValueError("matrix response reactive entrance strict roots/config lineage differs")

    units = tuple(
        ResponsePreparationUnit(
            physical_independent_unit_id=slot.physical_independent_unit_id,
            preparation_coordinate_id="coordinate.matrix-response-reactive-entrance-step-816",
            preparation_instance_id=slot.preparation_instance_id,
            preparation_sha256=source_config.checkpoint.combined_state_sha256,
            adapter_realization_id=f"seed.matrix-response-reactive-entrance.r{slot.slot_index:04d}",
            split_role=(
                ResponseDataSplitRole.DEVELOPMENT
                if slot.slot_index < 256
                else ResponseDataSplitRole.EVALUATION
            ),
        )
        for slot in source_config.slots
    )
    views: list[ResponseAcquisitionView] = []
    groups: list[ResponseAcquisitionGroup] = []
    for slot in source_config.slots:
        view_pairs = [
            (
                slot.primary_acquisition_group_id,
                slot.primary_view_id,
                REACTIVE_ENTRANCE_PRIMARY_MEMBER_ID,
            )
        ]
        if slot.fine_acquisition_group_id is not None and slot.fine_view_id is not None:
            view_pairs.append(
                (
                    slot.fine_acquisition_group_id,
                    slot.fine_view_id,
                    REACTIVE_ENTRANCE_FINE_MEMBER_ID,
                )
            )
        for group_id, view_id, member_id in view_pairs:
            views.append(
                ResponseAcquisitionView(
                    view_id=view_id,
                    acquisition_group_id=group_id,
                    physical_independent_unit_id=slot.physical_independent_unit_id,
                    numerical_member_id=member_id,
                    action_word_id=None,
                )
            )
            groups.append(
                ResponseAcquisitionGroup(
                    acquisition_group_id=group_id,
                    physical_independent_unit_id=slot.physical_independent_unit_id,
                    preparation_instance_id=slot.preparation_instance_id,
                    view_ids=(view_id,),
                )
            )
    physical_unit_ids = tuple(value.physical_independent_unit_id for value in units)
    qualification_config = ResponseQualificationConfig(
        config_id="law-qualification-config.matrix-response-reactive-entrance-reactive-entrance",
        evidence_profile_selection=evidence_identity,
        source_pipeline_profile=source_profile_identity,
        linked_campaign_profile=linked_identity,
        action_chart=None,
        package_lineage=_lineage(),
        terminal_matrix=_terminal_matrix(),
        physical_units=units,
        acquisition_groups=tuple(sorted(groups, key=lambda value: value.acquisition_group_id)),
        nested_views=tuple(sorted(views, key=lambda value: value.view_id)),
        identification_admission_physical_independent_unit_ids=physical_unit_ids,
        development_unit_ids=physical_unit_ids[:256],
        evaluation_unit_ids=physical_unit_ids[256:],
        numerical_member_ids=tuple(sorted((REACTIVE_ENTRANCE_FINE_MEMBER_ID, REACTIVE_ENTRANCE_PRIMARY_MEMBER_ID))),
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
    if set(linked_profile.rosters.physical_independent_unit_ids) != set(physical_unit_ids):
        raise ValueError("matrix response reactive entrance linked physical-unit roster differs")
    if set(linked_profile.rosters.denominator_member_ids) != set(qualification_config.numerical_member_ids):
        raise ValueError("matrix response reactive entrance linked numerical-member roster differs")
    return ResponseExperimentExtensionSet(
        extension_set_id="extension-set.matrix-response-reactive-entrance-reactive-entrance",
        evidence_profile_selection=evidence_identity,
        source_pipeline_profile=source_profile_identity,
        linked_campaign_profile=linked_identity,
        identification_config=qualification_config,
        admission_config=None,
        prospective_evaluation_config=None,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_matrix_response_study_reactive_entrance_substrate_binding(
    source_config: SixMatrixResponseReactiveEntranceSourceConfig,
) -> ResponseSubstrateBinding:
    """Bind reactive entrance source semantics to the installed neutral acquisition seam."""

    return ResponseSubstrateBinding(
        binding_id="substrate-binding.matrix-response-reactive-entrance-reactive-entrance",
        medium_id="medium.six-matrix-response-six-matrix-conditional-wiener",
        interaction_kind=NativeInteractionKind.READ_ONLY_ACQUISITION,
        provider_key=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_key,
        provider_version=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_version,
        provider_implementation_sha256=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.implementation_sha256,
        installed_executable_binding=ObjectIdentity.from_record(
            REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING.binding_id,
            REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING,
        ),
        source_profile_schema=SourcePipelineProfile.SCHEMA,
        native_request_schema=SixMatrixResponseReactiveEntrancePrecursorRequest.SCHEMA,
        native_object_schema=SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA,
        native_episode_schema=None,
        native_projection_schema=MatrixResponseReactiveEntranceProjection.SCHEMA,
        native_config=ObjectIdentity.from_record(source_config.config_id, source_config),
        receivers=(
            NativeReceiverContract(
                receiver_id="receiver.matrix-response-reactive-entrance-complete-phase-space",
                quantity_id="quantity.matrix-response-reactive-entrance-positions-momenta-innovations-parameters",
                native_unit="heterogeneous-native-phase-space",
                frame_id="frame.six-matrix-response-six-matrix-hermitian",
                direction_id="direction.six-matrix-response-anonymous-receiver",
                clock_ids=("clock.six-matrix-response-native-integration-step",),
            ),
        ),
        clocks=(
            NativeClockContract(
                clock_id="clock.six-matrix-response-native-integration-step",
                native_unit="integration-step",
                frame_id="frame.six-matrix-response-native-integration-time",
            ),
        ),
        action_contract=None,
        external_source_contacted=False,
        scientific_verdict_constructed=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def matrix_response_study_reactive_entrance_preparation_roster_sha256(source_config: SixMatrixResponseReactiveEntranceSourceConfig) -> str:
    """Compact audit digest for the exact physical/acquisition/view roster."""

    return sha256(
        b"\0".join(
            (
                slot.physical_independent_unit_id
                + "\0"
                + slot.primary_acquisition_group_id
                + "\0"
                + slot.primary_view_id
                + "\0"
                + (slot.fine_acquisition_group_id or "")
                + "\0"
                + (slot.fine_view_id or "")
            ).encode("utf-8")
            for slot in source_config.slots
        )
    ).hexdigest()


__all__ = [
    'build_matrix_response_study_reactive_entrance_extension_set',
    'build_matrix_response_study_reactive_entrance_method_config',
    'build_matrix_response_study_reactive_entrance_source_config',
    'build_matrix_response_study_reactive_entrance_substrate_binding',
    'matrix_response_study_reactive_entrance_preparation_roster_sha256',
]
