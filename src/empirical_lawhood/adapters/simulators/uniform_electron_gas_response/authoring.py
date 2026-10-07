"""Outcome-blind authoring root and formal coverage for uniform electron gas transverse receiver screen."""

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

from .candidate import uniform_electron_gas_transverse_screen_candidate_catalog, uniform_electron_gas_transverse_screen_template
from .contracts import CONFIG_SCHEMA, CONFIG_VERSION, UniformElectronGasTransverseScreenConfig
from .registration import uniform_electron_gas_transverse_screen_registry
from .runtime import CONFIG_SOURCE_ID, MEDIUM_ARTIFACT_ID, uniform_electron_gas_transverse_screen_protocol
from .system import TRUTH_SEED_IDS, TRUTH_UNIT_IDS, uniform_electron_gas_transverse_screen_experiment, uniform_electron_gas_transverse_screen_system, study_budget


def _config_identity(config: UniformElectronGasTransverseScreenConfig) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=MEDIUM_ARTIFACT_ID,
        object_schema=CONFIG_SCHEMA,
        object_version=CONFIG_VERSION,
        object_fingerprint=config.payload_sha256,
    )


def uniform_electron_gas_transverse_screen_model_set(system: SystemSpec, experiment: ExperimentSpec) -> ViewModelSetSpec:
    return ViewModelSetSpec(
        model_set_id="model-set.uniform-electron-gas-transverse-screen-base-refined",
        target_world_id=system.world.world_id,
        member_view_ids=tuple(value.view_id for value in system.numerical_views),
        model_relation_ids=("relation.uniform-electron-gas-response-base-refined-view-intersection",),
        plausibility_rule="Retain both predeclared numerical views regardless of result.",
        support_rule="receiver admission is the exact intersection; no view average may repair a failed gate.",
        uncertainty_set_ids=("uncertainty.uniform-electron-gas-response-exact-view-plurality",),
        validity=experiment.obligations.validity,
        intersection_semantics=ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def uniform_electron_gas_transverse_screen_campaign(
    system: SystemSpec, experiment: ExperimentSpec, config: UniformElectronGasTransverseScreenConfig
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.uniform-electron-gas-transverse-screen-conformance",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.uniform-electron-gas-transverse-screen-conformance-closeout",
        objective=(
            "Qualify the frozen transverse receiver method in an analytic reference world "
            "and close target acts with exact source-prerequisite nonattempts."
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
                decision_right_id="decision-right.uniform-electron-gas-transverse-screen-conformance-execution",
                action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
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


def uniform_electron_gas_transverse_screen_formal_coverage(
    *, config: UniformElectronGasTransverseScreenConfig, register: FormalGapRegister, denominator_id: str, draft_id: str
) -> FormalGapCoverage:
    selected = set(config.reference_conformance_gaps)
    units = tuple(f"acquisition-{value}" for value in TRUTH_UNIT_IDS)
    views = config.views
    applicability = []
    assignments = []
    for gap in register.gaps:
        tested = gap.gap_id in selected
        applicability.append(
            FormalGapApplicability(
                gap_id=gap.gap_id,
                evidence_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
                present_operand_ids=(gap.required_operand_ids if tested else ()),
                satisfied_prerequisite_ids=(gap.support_prerequisite_ids if tested else ()),
                independent_unit_ids=(units if tested else ()),
                independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
                numerical_view_ids=(views if tested else ()),
                available_estimator_family_ids=(gap.estimator_family_ids if tested else ()),
                available_control_ids=(gap.control_ids if tested else ()),
                multiplicity_family_ids=((gap.multiplicity_family_id,) if tested else ()),
                requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                denominator_applicable=tested,
                resource_envelope_satisfied=True,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
        if tested:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
                    readiness_reason=None,
                    reason_codes=(),
                    selected_estimator_family_id=gap.estimator_family_ids[0],
                    selected_control_ids=gap.control_ids,
                    selected_multiplicity_family_id=gap.multiplicity_family_id,
                    obligation_ids=(f"formal-obligation.{gap.gap_id}",),
                    output_ids=("method-conformance",),
                    adjudication_owner_ids=("evaluator-reveal",),
                )
            )
        else:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=(FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR),
                    readiness_reason=None,
                    reason_codes=("OUTSIDE_FROZEN_TRANSVERSE_SCREEN_DENOMINATOR",),
                    selected_estimator_family_id=None,
                    selected_control_ids=(),
                    selected_multiplicity_family_id=None,
                    obligation_ids=(),
                    output_ids=(),
                    adjudication_owner_ids=(),
                )
            )
    return FormalGapCoverage(
        coverage_id="formal-gap-coverage.uniform-electron-gas-transverse-screen-conformance",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=draft_id,
        applicability=tuple(applicability),
        assignments=tuple(assignments),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *, draft: StudyDraft, register: FormalGapRegister, coverage: FormalGapCoverage
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.uniform-electron-gas-transverse-screen.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id="entry-checklist.uniform-electron-gas-transverse-screen-conformance",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        execution_route_id="route.uniform-electron-gas-transverse-screen-conformance",
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id="experiment-entry-package.uniform-electron-gas-transverse-screen-conformance",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class UniformElectronGasTransverseScreenAuthoringBundle:
    config: UniformElectronGasTransverseScreenConfig
    config_payload: bytes
    system: SystemSpec
    experiment: ExperimentSpec
    model_set: ViewModelSetSpec
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    qualification: MaterializationQualificationReceipt
    formal_coverage: FormalGapCoverage
    entry_package: ExperimentEntryPackage
    package: StudyDefinition
    draft: StudyDraft
    context: CandidateCompilationContext


def build_uniform_electron_gas_transverse_screen_authoring_bundle(
    *,
    config: UniformElectronGasTransverseScreenConfig,
    config_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> UniformElectronGasTransverseScreenAuthoringBundle:
    system = uniform_electron_gas_transverse_screen_system(config)
    experiment = uniform_electron_gas_transverse_screen_experiment(system, config)
    model_set = uniform_electron_gas_transverse_screen_model_set(system, experiment)
    campaign = uniform_electron_gas_transverse_screen_campaign(system, experiment, config)
    registry = uniform_electron_gas_transverse_screen_registry(config=config, implementation_sha256=implementation_sha256)
    protocol = uniform_electron_gas_transverse_screen_protocol(registry=registry, config=config)
    template = uniform_electron_gas_transverse_screen_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        config=config,
    )
    catalog = uniform_electron_gas_transverse_screen_candidate_catalog(registry=registry, template=template)
    config_identity = _config_identity(config)
    observation_operator = ObjectIdentity(
        object_id="observer.uniform-electron-gas-transverse-screen",
        object_schema='empirical-lawhood/simulators/uniform-electron-gas-response/observation-operator',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.uniform-electron-gas-transverse-screen-config",
        source_id=CONFIG_SOURCE_ID,
        materialization=config_identity,
        content_sha256=config.payload_sha256,
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=config.views,
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
        input_id="design-input.uniform-electron-gas-transverse-screen-system",
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    model_input = DesignInputRecord(
        input_id="design-input.uniform-electron-gas-transverse-screen-model-set",
        object_identity=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        materialization_sha256=model_set.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    draft_id = "draft.uniform-electron-gas-transverse-screen-conformance-closeout"
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the frozen transverse receiver method recover all truth-known cases, "
            "and which target acts are prohibited by the terminal source disposition?"
        ),
        alternative_ids=(
            "alternative.uniform-electron-gas-transverse-screen-method-conforms",
            "alternative.uniform-electron-gas-transverse-screen-method-fails",
            "alternative.uniform-electron-gas-transverse-screen-source-prerequisite-stop",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.uniform-electron-gas-transverse-screen-owner-predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(sorted((design_input.input_id, model_input.input_id))),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=tuple(sorted((design_input, model_input), key=lambda value: value.input_id)),
        development_unit_ids=TRUTH_UNIT_IDS,
        evaluation_unit_ids=tuple(f"oracle-{value}" for value in TRUTH_UNIT_IDS),
        development_seed_ids=TRUTH_SEED_IDS,
        evaluation_seed_ids=("seed.uniform-electron-gas-transverse-screen-conformance-oracle",),
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
                source_id=CONFIG_SOURCE_ID,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                evidence_world_id=system.world.world_id,
                materialization=config_identity,
                content_sha256=config.payload_sha256,
                source_config_sha256=config.payload_sha256,
                observation_operator=observation_operator,
                numerical_view_ids=config.views,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id, qualification
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=study_budget(config),
    )
    formal_coverage = uniform_electron_gas_transverse_screen_formal_coverage(
        config=config,
        register=register,
        denominator_id=system.system_id,
        draft_id=draft.draft_id,
    )
    entry = _entry_package(draft=draft, register=register, coverage=formal_coverage)
    package = StudyDefinition(
        package_id="programme-authoring-package.uniform-electron-gas-transverse-screen-conformance-closeout",
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id="context.uniform-electron-gas-transverse-screen-conformance-closeout",
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        implementation_sha256=implementation_sha256,
    )
    return UniformElectronGasTransverseScreenAuthoringBundle(
        config=config,
        config_payload=config_payload,
        system=system,
        experiment=experiment,
        model_set=model_set,
        campaign=campaign,
        protocol=protocol,
        registry=registry,
        template=template,
        catalog=catalog,
        qualification=qualification,
        formal_coverage=formal_coverage,
        entry_package=entry,
        package=package,
        draft=draft,
        context=context,
    )


__all__ = [
    "UniformElectronGasTransverseScreenAuthoringBundle",
    "build_uniform_electron_gas_transverse_screen_authoring_bundle",
    "uniform_electron_gas_transverse_screen_campaign",
    "uniform_electron_gas_transverse_screen_formal_coverage",
    "uniform_electron_gas_transverse_screen_model_set",
]
