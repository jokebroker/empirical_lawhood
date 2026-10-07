'Outcome-blind ambient pressure superconductor material source design authoring root on the standard campaign platform.'

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ViewModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import FormalGapSourceCapabilityInventory
from empirical_lawhood.planning.formal_gaps import (
    FormalGapApplicability,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .material_source_design_candidate import material_source_design_candidate_catalog, material_source_design_template
from .material_source_design_contracts import MaterialSourceDesignConfig, MaterialSourceDesignSourceQualification, SplitPartition
from .material_source_design_design import build_material_roster
from .material_source_design_registration import material_source_design_registry
from .material_source_design_runtime import SOURCE_CONTRACT_ARTIFACT_ID, material_source_design_protocol
from .material_source_design_source import expected_material_source_design_source_qualification
from .material_source_design_system import material_source_design_experiment, material_source_design_system, study_budget


def material_source_design_model_set(system: SystemSpec, experiment: ExperimentSpec) -> ViewModelSetSpec:
    return ViewModelSetSpec(
        model_set_id='model-set.ambient-pressure-superconductor-material-source-design-three-frozen-views',
        target_world_id=system.world.world_id,
        member_view_ids=tuple(view.view_id for view in system.numerical_views),
        model_relation_ids=('relation.ambient-pressure-superconductor-material-source-design-view-design-intersection',),
        plausibility_rule="Retain the three predeclared base/refined/independent design views.",
        support_rule='material source design qualifies sources and design only; no view contains a material outcome.',
        uncertainty_set_ids=('uncertainty.ambient-pressure-superconductor-material-source-design-calibration-fields-open',),
        validity=experiment.obligations.validity,
        intersection_semantics=ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def material_source_design_campaign(system: SystemSpec, experiment: ExperimentSpec, config: MaterialSourceDesignConfig) -> CampaignSpec:
    node = CampaignNode(
        node_id='campaign-node.ambient-pressure-superconductor-material-source-design-source-design-audit-audit',
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id='campaign.ambient-pressure-superconductor-material-source-design-source-design-audit-audit',
        objective=(
            "Qualify the exact public source roster and determine whether the complete "
            'ambient pressure superconductor development design/provider basis can be frozen without target contact.'
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(claim.claim_id for claim in experiment.claims),
        budget=study_budget(config),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id, system.authority_policy
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-material-source-design-nonactuating-audit',
                action=AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def material_source_design_formal_coverage(
    *, register: FormalGapRegister, denominator_id: str, draft_id: str
) -> FormalGapCoverage:
    applicability = tuple(
        FormalGapApplicability(
            gap_id=gap.gap_id,
            evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
            present_operand_ids=(),
            satisfied_prerequisite_ids=(),
            independent_unit_ids=(),
            independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
            numerical_view_ids=(),
            available_estimator_family_ids=(),
            available_control_ids=(),
            multiplicity_family_ids=(),
            requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
            denominator_applicable=False,
            resource_envelope_satisfied=True,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for gap in register.gaps
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
            readiness_reason=None,
            reason_codes=('outside-material-source-design-source-and-design-audit',),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for gap in register.gaps
    )
    return FormalGapCoverage(
        coverage_id='formal-gap-coverage.ambient-pressure-superconductor-material-source-design-source-design-audit',
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *, draft: StudyDraft, register: FormalGapRegister, coverage: FormalGapCoverage
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f'binding.ambient-pressure-superconductor-material-source-design{requirement.value.lower()}',),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("execution-and-reveal-authority-separate",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id='entry-checklist.ambient-pressure-superconductor-material-source-design-source-design-audit',
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id='route.ambient-pressure-superconductor-material-source-design-generic-campaign',
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id='experiment-entry-package.ambient-pressure-superconductor-material-source-design-source-design-audit',
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignAuthoringBundle:
    config: MaterialSourceDesignConfig
    config_payload: bytes
    source_contract: MaterialSourceDesignSourceQualification
    source_config: SourceMaterializationConfig
    system: SystemSpec
    experiment: ExperimentSpec
    model_set: ViewModelSetSpec
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


def build_material_source_design_authoring_bundle(
    *,
    config: MaterialSourceDesignConfig,
    config_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> MaterialSourceDesignAuthoringBundle:
    source_contract = expected_material_source_design_source_qualification()
    source_bytes = source_contract.canonical_bytes()
    source_config = SourceMaterializationConfig(
        config_id='source-config.ambient-pressure-superconductor-material-source-design-public-roster',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=source_contract.fingerprint(),
        expected_size_bytes=len(source_bytes),
        maximum_bytes=1024 * 1024,
        payload_schema=MaterialSourceDesignSourceQualification.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    system = material_source_design_system(config)
    experiment = material_source_design_experiment(system, config)
    model_set = material_source_design_model_set(system, experiment)
    campaign = material_source_design_campaign(system, experiment, config)
    registry = material_source_design_registry(config=config, implementation_sha256=implementation_sha256)
    protocol = material_source_design_protocol(registry=registry, config=config)
    template = material_source_design_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        config=config,
        source_contract=source_contract,
    )
    catalog = material_source_design_candidate_catalog(registry=registry, template=template)
    source_identity = ObjectIdentity.from_record(SOURCE_CONTRACT_ARTIFACT_ID, source_contract)
    observation_operator = ObjectIdentity(
        object_id='observer.ambient-pressure-superconductor-material-source-design-physical-source-inspector',
        object_schema='empirical-lawhood/material/source-design-closure/source-inspector',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id='qualification.ambient-pressure-superconductor-material-source-design-source-contract',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        materialization=source_identity,
        content_sha256=source_contract.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
        native_unit_ids=tuple(sorted({quantity.native_unit for quantity in system.quantities})),
        frame_ids=tuple(sorted({quantity.coordinate_frame for quantity in system.quantities})),
        clock_ids=tuple(clock.clock_id for clock in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-material-source-design-system',
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    model_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-material-source-design-model-set',
        object_identity=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        materialization_sha256=model_set.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    roster = build_material_roster()
    development_units = tuple(
        sorted(
            family.family_id
            for family in roster.families
            if family.partition is not SplitPartition.SEALED_PROSPECTIVE
        )
    )
    evaluation_units = tuple(
        sorted(
            family.family_id
            for family in roster.families
            if family.partition is SplitPartition.SEALED_PROSPECTIVE
        )
    )
    draft = StudyDraft(
        draft_id='draft.ambient-pressure-superconductor-material-source-design-source-design-audit-audit',
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            'Do exact source custody and the complete executable ambient pressure superconductor development design '
            "both close before any protected target outcome is contacted?"
        ),
        alternative_ids=(
            'alternative.ambient-pressure-superconductor-material-source-design-design-freezes',
            'alternative.ambient-pressure-superconductor-material-source-design-design-provider-gap',
            'alternative.ambient-pressure-superconductor-material-source-design-source-not-qualified',
        ),
        design_origin=DesignOrigin(
            origin_id='origin.ambient-pressure-superconductor-material-source-design-owner-predeclared-a2',
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(sorted((design_input.input_id, model_input.input_id))),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=tuple(sorted((design_input, model_input), key=lambda value: value.input_id)),
        development_unit_ids=development_units,
        evaluation_unit_ids=evaluation_units,
        development_seed_ids=('seed.ambient-pressure-superconductor-material-source-design-policy',),
        evaluation_seed_ids=('seed.ambient-pressure-superconductor-material-source-design-sealed-prospective-sealed',),
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
                observation_operator=observation_operator,
                numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id, qualification
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=study_budget(config),
    )
    coverage = material_source_design_formal_coverage(
        register=register,
        denominator_id=system.system_id,
        draft_id=draft.draft_id,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id='formal-source-inventory.ambient-pressure-superconductor-material-source-design-source-design-audit',
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=(source_identity,),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=(),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=tuple(sorted(gap.gap_id for gap in register.gaps)),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    entry = _entry_package(draft=draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id='programme-authoring-package.ambient-pressure-superconductor-material-source-design-source-design-audit',
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id='context.ambient-pressure-superconductor-material-source-design-source-design-audit',
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        implementation_sha256=implementation_sha256,
    )
    return MaterialSourceDesignAuthoringBundle(
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


__all__ = [
    'MaterialSourceDesignAuthoringBundle',
    'material_source_design_campaign',
    'material_source_design_formal_coverage',
    'material_source_design_model_set',
    'build_material_source_design_authoring_bundle',
]
