'Outcome-blind ambient pressure superconductor excluded solver control authoring root on the standard campaign platform.'

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ViewModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_gaps import (
    FormalGapApplicability,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.formal_analysis import FormalGapSourceCapabilityInventory
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode
from empirical_lawhood.kernel.worlds import EvidenceUnitScope

from .candidate import excluded_solver_control_candidate_catalog, excluded_solver_control_template
from .contracts import ExcludedSolverControlConfig, ExcludedSolverControlSourceManifest, SOURCE_MANIFEST_SOURCE_ID
from .registration import excluded_solver_control_registry
from .runtime import excluded_solver_control_protocol
from .source import expected_excluded_solver_control_source_manifest
from .system import CONTROL_UNIT_IDS, excluded_solver_control_experiment, excluded_solver_control_system, study_budget


def excluded_solver_control_model_set(system: SystemSpec, experiment: ExperimentSpec) -> ViewModelSetSpec:
    return ViewModelSetSpec(
        model_set_id='model-set.ambient-pressure-superconductor-excluded-solver-control-qe76-single-view',
        target_world_id=system.world.world_id,
        member_view_ids=tuple(value.view_id for value in system.numerical_views),
        model_relation_ids=('relation.ambient-pressure-superconductor-excluded-solver-control-single-view-identity',),
        plausibility_rule="Retain the one predeclared excluded Pb numerical view.",
        support_rule='excluded solver control qualifies workflow execution only; it cannot enter material order relation--controller use.',
        uncertainty_set_ids=('uncertainty.ambient-pressure-superconductor-excluded-solver-control-single-view-limited',),
        validity=experiment.obligations.validity,
        intersection_semantics=ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def excluded_solver_control_campaign(system: SystemSpec, experiment: ExperimentSpec, config: ExcludedSolverControlConfig) -> CampaignSpec:
    node = CampaignNode(
        node_id='campaign-node.ambient-pressure-superconductor-excluded-solver-control-pb-control',
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id='campaign.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke',
        objective=(
            "Qualify one exact local QE solver route using the excluded, truth-known Pb "
            'SCF control without contacting an ambient pressure superconductor target.'
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=study_budget(config),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id, system.authority_policy
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-excluded-solver-control-simulation',
                action=AuthorityAction.SIMULATION_EXECUTION,
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


def excluded_solver_control_formal_coverage(
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
            reason_codes=('outside-excluded-solver-control-excluded-solver-smoke',),
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
        coverage_id='formal-gap-coverage.ambient-pressure-superconductor-excluded-solver-control-pb',
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
            object_ids=(f'binding.ambient-pressure-superconductor-excluded-solver-control{requirement.value.lower()}',),
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
        checklist_id='entry-checklist.ambient-pressure-superconductor-excluded-solver-control-pb',
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id='route.ambient-pressure-superconductor-excluded-solver-control-generic-campaign',
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id='experiment-entry-package.ambient-pressure-superconductor-excluded-solver-control-pb',
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class ExcludedSolverControlAuthoringBundle:
    config: ExcludedSolverControlConfig
    config_payload: bytes
    source_manifest: ExcludedSolverControlSourceManifest
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
            self.source_manifest.fingerprint(): self.source_manifest.canonical_bytes(),
            self.source_config.fingerprint(): self.source_config.canonical_bytes(),
            self.qualification.fingerprint(): self.qualification.canonical_bytes(),
        }
        return tuple(values[key] for key in sorted(values))


def build_excluded_solver_control_authoring_bundle(
    *,
    config: ExcludedSolverControlConfig,
    config_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> ExcludedSolverControlAuthoringBundle:
    source_manifest = expected_excluded_solver_control_source_manifest(config)
    source_config = SourceMaterializationConfig(
        config_id='source-config.ambient-pressure-superconductor-excluded-solver-control-solver-environment',
        source_id=SOURCE_MANIFEST_SOURCE_ID,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=source_manifest.fingerprint(),
        expected_size_bytes=len(source_manifest.canonical_bytes()),
        maximum_bytes=1024 * 1024,
        payload_schema=ExcludedSolverControlSourceManifest.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    system = excluded_solver_control_system(config)
    experiment = excluded_solver_control_experiment(system, config)
    model_set = excluded_solver_control_model_set(system, experiment)
    campaign = excluded_solver_control_campaign(system, experiment, config)
    registry = excluded_solver_control_registry(config=config, implementation_sha256=implementation_sha256)
    protocol = excluded_solver_control_protocol(registry=registry, config=config)
    template = excluded_solver_control_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        config=config,
        source_manifest=source_manifest,
    )
    catalog = excluded_solver_control_candidate_catalog(registry=registry, template=template)
    source_identity = ObjectIdentity.from_record(
        SOURCE_MANIFEST_SOURCE_ID,
        source_manifest,
    )
    observation_operator = ObjectIdentity(
        object_id='observer.ambient-pressure-superconductor-excluded-solver-control-held-asset-inspector',
        object_schema='empirical-lawhood/material/excluded-lead-solver-control/held-asset-inspector',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id='qualification.ambient-pressure-superconductor-excluded-solver-control-solver-environment',
        source_id=SOURCE_MANIFEST_SOURCE_ID,
        materialization=source_identity,
        content_sha256=source_manifest.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        native_unit_ids=tuple(sorted({value.native_unit for value in system.quantities})),
        frame_ids=tuple(sorted({value.coordinate_frame for value in system.quantities})),
        clock_ids=tuple(value.clock_id for value in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-excluded-solver-control-system',
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    model_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-excluded-solver-control-model-set',
        object_identity=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        materialization_sha256=model_set.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    draft_id = 'draft.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke'
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the exact fixed QE profile complete and expose the required typed operands "
            "for the excluded Pb SCF control?"
        ),
        alternative_ids=(
            'alternative.ambient-pressure-superconductor-excluded-solver-control-control-fails',
            'alternative.ambient-pressure-superconductor-excluded-solver-control-control-passes',
            'alternative.ambient-pressure-superconductor-excluded-solver-control-resource-path-inadequate',
        ),
        design_origin=DesignOrigin(
            origin_id='origin.ambient-pressure-superconductor-excluded-solver-control-owner-predeclared-a1',
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(sorted((design_input.input_id, model_input.input_id))),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=tuple(sorted((design_input, model_input), key=lambda value: value.input_id)),
        development_unit_ids=CONTROL_UNIT_IDS,
        evaluation_unit_ids=('evaluation-unit.ambient-pressure-superconductor-excluded-solver-control-pb-control-001',),
        development_seed_ids=('seed.ambient-pressure-superconductor-excluded-solver-control-deterministic',),
        evaluation_seed_ids=('seed.ambient-pressure-superconductor-excluded-solver-control-evaluator',),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in registry.capabilities
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=SOURCE_MANIFEST_SOURCE_ID,
                role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                evidence_world_id=system.world.world_id,
                materialization=source_identity,
                content_sha256=source_manifest.fingerprint(),
                source_config_sha256=source_config.fingerprint(),
                observation_operator=observation_operator,
                numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id, qualification
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=study_budget(config),
    )
    coverage = excluded_solver_control_formal_coverage(
        register=register,
        denominator_id=system.system_id,
        draft_id=draft.draft_id,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id='formal-source-inventory.ambient-pressure-superconductor-excluded-solver-control-scf-smoke',
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
        package_id='programme-authoring-package.ambient-pressure-superconductor-excluded-solver-control-pb',
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id='context.ambient-pressure-superconductor-excluded-solver-control-pb',
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        implementation_sha256=implementation_sha256,
    )
    return ExcludedSolverControlAuthoringBundle(
        config=config,
        config_payload=config_payload,
        source_manifest=source_manifest,
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
    'ExcludedSolverControlAuthoringBundle',
    'excluded_solver_control_campaign',
    'excluded_solver_control_formal_coverage',
    'excluded_solver_control_model_set',
    'build_excluded_solver_control_authoring_bundle',
]
