'Outcome-blind authoring root for the fresh ambient pressure superconductor material control control act.'

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
)
from empirical_lawhood.planning.formal_gaps import FormalGapCoverage, FormalGapRegister
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import (
    SourceMaterializationConfig,
    SourceReadMode,
)

from .gauge_covariant_response_authoring import build_gauge_covariant_response_authoring_bundle
from .gauge_covariant_response_contracts import GaugeCovariantResponseConfig
from .material_source_design_contracts import MaterialSourceDesignConfig
from .material_control_candidate import material_control_candidate_catalog, material_control_template
from .material_control_contracts import MATERIAL_CONTROL_METHOD_CAPABILITY_KEY, MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY, MaterialControlConfig, MaterialControlControlBundleQualification, decode_material_control_config_bytes
from .multiband_strong_coupling_response import qualify_multiband_bridge
from .material_control_registration import material_control_metadata_task_budget, material_control_registry, material_control_task_budget
from .material_control_runtime import SOURCE_CONTRACT_ARTIFACT_ID, material_control_protocol
from .material_control_solver import MATERIAL_CONTROL_WORKFLOW_PROFILES
from .material_control_source import expected_material_control_control_bundle

MATERIAL_CONTROL_SCOPE_ID = 'scope.ambient-pressure-superconductor-material-control-control-science-freeze'


def _study_budget(config: MaterialControlConfig) -> ResourceBudget:
    workflow = material_control_task_budget(config)
    metadata = material_control_metadata_task_budget(config)
    workflow_count = 12
    metadata_count = 9
    return ResourceBudget(
        cpu_cores=workflow.cpu_cores,
        memory_bytes=workflow.memory_bytes,
        gpu_devices=workflow.gpu_devices,
        wall_time_seconds=(
            workflow_count * workflow.wall_time_seconds
            + metadata_count * metadata.wall_time_seconds
        ),
        source_scan_bytes=(
            workflow_count * workflow.source_scan_bytes
            + metadata_count * metadata.source_scan_bytes
        ),
        output_bytes=(
            workflow_count * workflow.output_bytes
            + metadata_count * metadata.output_bytes
        ),
    )


def _entry_package(
    *, draft: StudyDraft, register: FormalGapRegister, coverage: FormalGapCoverage
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f'binding.ambient-pressure-superconductor-material-control.entry-requirement.{requirement.value.lower().replace("_", "-")}',),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("simulation-and-freeze-authority-separate",)
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(
            ExperimentEntryRequirement, key=lambda value: value.value
        )
    )
    checklist = ExperimentEntryChecklist(
        checklist_id='entry-checklist.ambient-pressure-superconductor-material-control',
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=coverage.applicability[0].evidence_world,
        execution_route_id='route.ambient-pressure-superconductor-material-control-generic-campaign',
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(
            sorted(ExperimentEntryTransition, key=lambda value: value.value)
        ),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id='experiment-entry-package.ambient-pressure-superconductor-material-control',
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class MaterialControlAuthoringBundle:
    config: MaterialControlConfig
    config_payload: bytes
    source_contract: MaterialControlControlBundleQualification
    source_config: SourceMaterializationConfig
    system: object
    experiment: object
    model_set: object
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    qualification: MaterializationQualificationReceipt
    inventory: FormalGapSourceCapabilityInventory
    formal_coverage: FormalGapCoverage
    entry_package: ExperimentEntryPackage
    package: StudyDefinition
    draft: StudyDraft
    context: CandidateCompilationContext

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        values = {
            self.config.payload_sha256: self.config_payload,
            self.source_contract.fingerprint(): self.source_contract.canonical_bytes(),
            self.source_config.fingerprint(): self.source_config.canonical_bytes(),
            self.qualification.fingerprint(): self.qualification.canonical_bytes(),
        }
        return tuple(values[key] for key in sorted(values))


def build_material_control_authoring_bundle(
    *,
    base_config: GaugeCovariantResponseConfig,
    base_payload: bytes,
    material_source_design_config: MaterialSourceDesignConfig,
    material_source_design_payload: bytes,
    control_input_archive_sha256: str,
    config: MaterialControlConfig,
    config_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> MaterialControlAuthoringBundle:
    if (
        decode_material_control_config_bytes(config_payload, expected_parents=config.predecessors)
        != config
    ):
        raise ValueError('material control authoring config differs from its exact bytes')
    if config.implementation_sha256 != implementation_sha256:
        raise ValueError('material control config differs from its implementation closure')
    if config.control_input_archive_sha256 != control_input_archive_sha256:
        raise ValueError('material control config differs from its exact control-input archive')
    if config.workflow_profile_ids != tuple(
        value.profile_id for value in MATERIAL_CONTROL_WORKFLOW_PROFILES
    ):
        raise ValueError('material control config differs from its closed workflow profile set')
    if (
        qualify_multiband_bridge(
            implementation_sha256=implementation_sha256
        ).fingerprint()
        != config.bridge_qualification_sha256
    ):
        raise ValueError('material control config differs from its method-qualified gauge covariant response bridge')
    base = build_gauge_covariant_response_authoring_bundle(
        base_config=material_source_design_config,
        base_payload=material_source_design_payload,
        config=base_config,
        config_payload=base_payload,
        register=register,
        implementation_sha256=implementation_sha256,
    )
    budget = _study_budget(config)
    base_system = cast(Any, base.system)
    policy = replace(
        base_system.authority_policy,
        policy_id='policy.ambient-pressure-superconductor-material-control-public-local-nonactuating',
        scope_ids=(MATERIAL_CONTROL_SCOPE_ID,),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        budget_ceiling=budget,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    system = replace(
        base_system,
        system_id='system.ambient-pressure-superconductor-material-control-control-science-freeze',
        label='ambient pressure superconductor material control public calibration control and science-freeze act',
        authority_policy=policy,
        numerical_views=tuple(
            value
            for value in base_system.numerical_views
            if value.view_id
            in {"view.pbe-efficiency-base", "view.pbe-precision-refined"}
        ),
    )
    base_experiment = cast(Any, base.experiment)
    controls = tuple(
        replace(
            control,
            control_id=f'control.ambient-pressure-superconductor-material-control-{index + 1}',
            capability_key=(
                MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY if index == 0 else MATERIAL_CONTROL_METHOD_CAPABILITY_KEY
            ),
        )
        for index, control in enumerate(base_experiment.controls)
    )
    falsifiers = tuple(
        replace(
            falsifier,
            falsifier_id=f'falsifier.ambient-pressure-superconductor-material-control-{index + 1}',
            capability_key=MATERIAL_CONTROL_METHOD_CAPABILITY_KEY,
        )
        for index, falsifier in enumerate(base_experiment.obligations.falsifiers)
    )
    obligations = replace(
        base_experiment.obligations,
        obligations_id='obligations.ambient-pressure-superconductor-material-control-control-science-freeze',
        falsifiers=falsifiers,
    )
    claim = replace(
        base_experiment.claims[0],
        claim_id='claim.ambient-pressure-superconductor-material-control-calibration-method-control-readiness',
        proposition=(
            "The exact public Pb/MgB2 and negative-control workflows recover their declared "
            'classes, the conditional material-gauge covariant response bridge is valid on both positive controls, '
            'and constructive search search conformance permits a target-blind material control science freeze.'
        ),
        estimand=(
            'Noncompensating calibration control recovery and method/search readiness; no room-temperature '
            "material or discovery claim."
        ),
        promotion_rule=(
            'At most response guided exploration execution readiness. Low-temperature Pb/MgB2 recovery cannot promote a '
            "300 K superconducting-material claim."
        ),
        numerical_view_ids=("view.pbe-efficiency-base", "view.pbe-precision-refined"),
    )
    experiment = replace(
        base_experiment,
        experiment_id='experiment.ambient-pressure-superconductor-material-control-control-science-freeze',
        system_id=system.system_id,
        claims=(claim,),
        controls=controls,
        obligations=obligations,
        authority_policy_id=policy.policy_id,
    )
    base_model = cast(Any, base.model_set)
    model_set = replace(
        base_model,
        model_set_id='model-set.ambient-pressure-superconductor-material-control-low-temperature-controls',
        member_view_ids=("view.pbe-efficiency-base", "view.pbe-precision-refined"),
        plausibility_rule=(
            'Retain exactly the material control PBE efficiency-base and PBE precision-refined views; '
            "the unexecuted PBEsol design view is not a member of this control act."
        ),
        support_rule=(
            'Pb and MgB2 validate the material bridge below their own Tc only; development atlas materials '
            "remain uncontacted and any 300 K claim requires a fresh material-specific state."
        ),
        validity=experiment.obligations.validity,
    )
    node = CampaignNode(
        node_id='campaign-node.ambient-pressure-superconductor-material-control',
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id, experiment
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    campaign = CampaignSpec(
        campaign_id='campaign.ambient-pressure-superconductor-material-control-control-science-freeze',
        objective=(
            'Execute exact public calibration workflows, material-gauge covariant response and search-world conformance, then '
            'freeze response guided exploration science only if every predeclared material control gate passes.'
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=(claim.claim_id,),
        budget=budget,
        authority_policy=ObjectIdentity.from_record(policy.policy_id, policy),
        decision_rights=(
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-material-control-freeze',
                action=AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                decision_maker_id=policy.delegate_id,
                authority_policy_id=policy.policy_id,
                delegated=True,
            ),
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-material-control-reveal',
                action=AuthorityAction.EVALUATOR_REVEAL,
                decision_maker_id=policy.delegate_id,
                authority_policy_id=policy.policy_id,
                delegated=True,
            ),
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-material-control-simulation',
                action=AuthorityAction.SIMULATION_EXECUTION,
                decision_maker_id=policy.delegate_id,
                authority_policy_id=policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
        predecessor_campaign_ids=('campaign.ambient-pressure-superconductor-gauge-covariant-response-staged',),
    )
    source_contract = expected_material_control_control_bundle(
        base_source_sha256=config.predecessors.development_source_sha256
    )
    if source_contract.fingerprint() != config.control_bundle_sha256:
        raise ValueError('material control source contract differs from frozen config')
    source_bytes = source_contract.canonical_bytes()
    source_config = SourceMaterializationConfig(
        config_id='source-config.ambient-pressure-superconductor-material-control-control-bundle',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=source_contract.fingerprint(),
        expected_size_bytes=len(source_bytes),
        maximum_bytes=1024 * 1024,
        payload_schema=MaterialControlControlBundleQualification.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    registry = material_control_registry(config=config, implementation_sha256=implementation_sha256)
    protocol = material_control_protocol(registry=registry, config=config)
    template = material_control_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        config=config,
        source_contract=source_contract,
    )
    catalog = material_control_candidate_catalog(registry=registry, template=template)
    source_identity = ObjectIdentity.from_record(
        SOURCE_CONTRACT_ARTIFACT_ID, source_contract
    )
    observer = ObjectIdentity(
        object_id='observer.ambient-pressure-superconductor-material-control-control-source-inspector',
        object_schema='empirical-lawhood/material/public-control-readiness/source-inspector',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id='qualification.ambient-pressure-superconductor-material-control-control-source',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        materialization=source_identity,
        content_sha256=source_contract.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observer,
        numerical_view_ids=source_contract.science_view_ids,
        native_unit_ids=tuple(
            sorted({value.native_unit for value in system.quantities})
        ),
        frame_ids=tuple(
            sorted({value.coordinate_frame for value in system.quantities})
        ),
        clock_ids=tuple(value.clock_id for value in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-material-control-system',
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    model_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-material-control-model-set',
        object_identity=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        materialization_sha256=model_set.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    draft = StudyDraft(
        draft_id='draft.ambient-pressure-superconductor-material-control-control-science-freeze',
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            'Do the exact calibration control, material-gauge covariant response and constructive search intersections justify freezing the '
            'already identified response guided exploration design before any development atlas target outcome is accessed?'
        ),
        alternative_ids=tuple(
            sorted(
                (
                    'alternative.ambient-pressure-superconductor-material-control-control-or-method-stop',
                    'alternative.ambient-pressure-superconductor-material-control-resource-stop',
                    'alternative.ambient-pressure-superconductor-material-control-science-freeze',
                )
            )
        ),
        design_origin=DesignOrigin(
            origin_id='origin.ambient-pressure-superconductor-material-control-gauge-covariant-response-predeclared-study',
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(
                sorted((design_input.input_id, model_input.input_id))
            ),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        development_unit_ids=base.draft.development_unit_ids,
        evaluation_unit_ids=base.draft.evaluation_unit_ids,
        development_seed_ids=base.draft.development_seed_ids,
        evaluation_seed_ids=base.draft.evaluation_seed_ids,
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                implementation_sha256=manifest.implementation_sha256,
            )
            for manifest in registry.capabilities
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=SOURCE_CONTRACT_ARTIFACT_ID,
                role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                evidence_world_id=system.world.world_id,
                materialization=source_identity,
                content_sha256=source_contract.fingerprint(),
                source_config_sha256=source_config.fingerprint(),
                observation_operator=observer,
                numerical_view_ids=source_contract.science_view_ids,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id, qualification
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=budget,
    )
    coverage = replace(
        base.formal_coverage,
        coverage_id='formal-gap-coverage.ambient-pressure-superconductor-material-control',
        denominator_id=system.system_id,
        candidate_act_id=draft.draft_id,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id='formal-source-inventory.ambient-pressure-superconductor-material-control',
        denominator_id=system.system_id,
        evidence_world=coverage.applicability[0].evidence_world,
        source_materializations=(source_identity,),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=(),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=tuple(
            sorted(value.gap_id for value in register.gaps)
        ),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    entry = _entry_package(draft=draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id='programme-authoring-package.ambient-pressure-superconductor-material-control',
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id='context.ambient-pressure-superconductor-material-control',
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        implementation_sha256=implementation_sha256,
    )
    return MaterialControlAuthoringBundle(
        config=config,
        config_payload=config_payload,
        source_contract=source_contract,
        source_config=source_config,
        system=system,
        experiment=experiment,
        model_set=model_set,
        campaign=campaign,
        protocol=protocol,
        registry=registry,
        template=template,
        catalog=catalog,
        qualification=qualification,
        inventory=inventory,
        formal_coverage=coverage,
        entry_package=entry,
        package=package,
        draft=draft,
        context=context,
    )


__all__ = ["MATERIAL_CONTROL_SCOPE_ID", 'MaterialControlAuthoringBundle', 'build_material_control_authoring_bundle']
