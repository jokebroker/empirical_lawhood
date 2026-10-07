'Outcome-blind gauge covariant response authoring root on the standard campaign platform.'

from __future__ import annotations

from dataclasses import dataclass, replace

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

from .gauge_covariant_response_candidate import gauge_covariant_response_candidate_catalog, gauge_covariant_response_template
from .gauge_covariant_response_contracts import GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY, GaugeCovariantResponseConfig, GaugeCovariantResponseSourceQualification
from .gauge_covariant_response_registration import gauge_covariant_response_conditional_registry, gauge_covariant_response_development_registry
from .gauge_covariant_response_runtime import SOURCE_CONTRACT_ARTIFACT_ID, gauge_covariant_response_conditional_protocols, gauge_covariant_response_development_protocol
from .gauge_covariant_response_source import expected_gauge_covariant_response_source_qualification
from .material_source_design_authoring import build_material_source_design_authoring_bundle
from .material_source_design_contracts import MaterialSourceDesignConfig
from .material_source_design_system import task_budget

GAUGE_COVARIANT_RESPONSE_SCOPE_ID = 'scope.ambient-pressure-superconductor-gauge-covariant-response-staged'


def _study_budget(config: GaugeCovariantResponseConfig) -> ResourceBudget:
    one = task_budget(config)
    return ResourceBudget(
        cpu_cores=one.cpu_cores,
        memory_bytes=one.memory_bytes,
        gpu_devices=one.gpu_devices,
        wall_time_seconds=10 * one.wall_time_seconds,
        source_scan_bytes=10 * one.source_scan_bytes,
        output_bytes=10 * one.output_bytes,
    )


def _entry_package(
    *, draft: StudyDraft, register: FormalGapRegister, coverage: FormalGapCoverage
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f'binding.ambient-pressure-superconductor-gauge-covariant-response-material-linked-receiver{requirement.value.lower()}',),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("execution-and-reveal-authority-separate",)
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
        checklist_id='entry-checklist.ambient-pressure-superconductor-gauge-covariant-response-development',
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=coverage.applicability[0].evidence_world,
        execution_route_id='route.ambient-pressure-superconductor-gauge-covariant-response-generic-campaign',
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
        package_id='experiment-entry-package.ambient-pressure-superconductor-gauge-covariant-response-development',
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseAuthoringBundle:
    config: GaugeCovariantResponseConfig
    config_payload: bytes
    source_contract: GaugeCovariantResponseSourceQualification
    source_config: SourceMaterializationConfig
    system: object
    experiment: object
    model_set: object
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    conditional_protocols: tuple[ProtocolTemplate, ...]
    registry: CapabilityRegistry
    conditional_registry: CapabilityRegistry
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


def build_gauge_covariant_response_authoring_bundle(
    *,
    base_config: MaterialSourceDesignConfig,
    base_payload: bytes,
    config: GaugeCovariantResponseConfig,
    config_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> GaugeCovariantResponseAuthoringBundle:
    # Import the immutable source design audit ontology and finite roster through its public
    # authoring constructor.  New gauge covariant response science is added below under new identities.
    base = build_material_source_design_authoring_bundle(
        config=base_config,
        config_payload=base_payload,
        register=register,
        implementation_sha256=implementation_sha256,
    )
    budget = _study_budget(config)
    policy = replace(
        base.system.authority_policy,
        policy_id='policy.ambient-pressure-superconductor-gauge-covariant-response-public-local-nonactuating',
        scope_ids=(GAUGE_COVARIANT_RESPONSE_SCOPE_ID,),
        budget_ceiling=budget,
    )
    system = replace(
        base.system,
        system_id='system.ambient-pressure-superconductor-gauge-covariant-response-staged-gauge-covariant-response-development',
        label='ambient pressure superconductor gauge covariant response method qualification and staged development gate',
        authority_policy=policy,
    )
    controls = tuple(
        replace(
            control,
            control_id=f'control.ambient-pressure-superconductor-gauge-covariant-response-{index + 1}',
            capability_key=(
                GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY if index == 0 else GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY
            ),
        )
        for index, control in enumerate(base.experiment.controls)
    )
    falsifiers = tuple(
        replace(
            falsifier,
            falsifier_id=f'falsifier.ambient-pressure-superconductor-gauge-covariant-response-{index + 1}',
            capability_key=(
                GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY if index == 0 else GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY
            ),
        )
        for index, falsifier in enumerate(base.experiment.obligations.falsifiers)
    )
    obligations = replace(
        base.experiment.obligations,
        obligations_id='obligations.ambient-pressure-superconductor-gauge-covariant-response-staged',
        falsifiers=falsifiers,
    )
    claim = replace(
        base.experiment.claims[0],
        claim_id='claim.ambient-pressure-superconductor-gauge-covariant-response-design-frozen-and-material-control-gated',
        proposition=(
            'The gauge covariant response source/gauge covariant response/provider basis freezes without target contact, after which '
            'material control either freezes complete controls or emits a typed source/method stop.'
        ),
        estimand=(
            'Exact design closure, strict-gauge covariant response method conformance and noncompensating material control '
            "control-source readiness."
        ),
        promotion_rule='At most method/design readiness until material control science freeze succeeds.',
    )
    experiment = replace(
        base.experiment,
        experiment_id='experiment.ambient-pressure-superconductor-gauge-covariant-response-staged',
        system_id=system.system_id,
        claims=(claim,),
        controls=controls,
        obligations=obligations,
        authority_policy_id=policy.policy_id,
    )
    model_set = replace(
        base.model_set,
        model_set_id='model-set.ambient-pressure-superconductor-gauge-covariant-response-method-and-design',
        support_rule=(
            'gauge covariant response fixtures qualify method behavior only; material promotion requires explicit '
            'DFT/Wannier/pairing compatibility and a successful material control science freeze.'
        ),
        validity=experiment.obligations.validity,
    )
    node = CampaignNode(
        node_id='campaign-node.ambient-pressure-superconductor-gauge-covariant-response-development',
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id, experiment
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    campaign = CampaignSpec(
        campaign_id='campaign.ambient-pressure-superconductor-gauge-covariant-response-staged',
        objective=(
            'Freeze the additive gauge covariant response strict-gauge covariant response and staged provider basis, execute the material control '
            "control gate, and preserve typed downstream nonattempts and closeout on stop."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=(claim.claim_id,),
        budget=budget,
        authority_policy=ObjectIdentity.from_record(policy.policy_id, policy),
        decision_rights=(
            DecisionRight(
                decision_right_id='decision-right.ambient-pressure-superconductor-gauge-covariant-response-nonactuating-development',
                action=AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
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
        predecessor_campaign_ids=(base.campaign.campaign_id,),
    )

    source_contract = expected_gauge_covariant_response_source_qualification(
        base_source_sha256=config.base_source_sha256
    )
    source_bytes = source_contract.canonical_bytes()
    source_config = SourceMaterializationConfig(
        config_id='source-config.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-response-extension',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=source_contract.fingerprint(),
        expected_size_bytes=len(source_bytes),
        maximum_bytes=1024 * 1024,
        payload_schema=GaugeCovariantResponseSourceQualification.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    registry = gauge_covariant_response_development_registry(
        config=config, implementation_sha256=implementation_sha256
    )
    conditional_registry = gauge_covariant_response_conditional_registry(
        config=config, implementation_sha256=implementation_sha256
    )
    protocol = gauge_covariant_response_development_protocol(registry=registry, config=config)
    conditional_protocols = gauge_covariant_response_conditional_protocols(
        registry=conditional_registry, config=config
    )
    template = gauge_covariant_response_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        config=config,
        source_contract=source_contract,
    )
    catalog = gauge_covariant_response_candidate_catalog(registry=registry, template=template)

    source_identity = ObjectIdentity.from_record(
        SOURCE_CONTRACT_ARTIFACT_ID, source_contract
    )
    observation_operator = ObjectIdentity(
        object_id='observer.ambient-pressure-superconductor-gauge-covariant-response-physical-source-inspector',
        object_schema='empirical-lawhood/material/staged-source-design/source-inspector',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id='qualification.ambient-pressure-superconductor-gauge-covariant-response-source-contract',
        source_id=SOURCE_CONTRACT_ARTIFACT_ID,
        materialization=source_identity,
        content_sha256=source_contract.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
        native_unit_ids=tuple(
            sorted({quantity.native_unit for quantity in system.quantities})
        ),
        frame_ids=tuple(
            sorted({quantity.coordinate_frame for quantity in system.quantities})
        ),
        clock_ids=tuple(clock.clock_id for clock in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-gauge-covariant-response-system',
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    model_input = DesignInputRecord(
        input_id='design-input.ambient-pressure-superconductor-gauge-covariant-response-model-set',
        object_identity=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        materialization_sha256=model_set.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    draft = StudyDraft(
        draft_id='draft.ambient-pressure-superconductor-gauge-covariant-response-staged',
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            'Does the additive gauge covariant response method/provider basis close before target contact, and do '
            'the exact available operands permit material control science freeze?'
        ),
        alternative_ids=tuple(
            sorted(
                (
                    'alternative.ambient-pressure-superconductor-gauge-covariant-response-material-control-source-stop',
                    'alternative.ambient-pressure-superconductor-gauge-covariant-response-design-gap',
                    'alternative.ambient-pressure-superconductor-gauge-covariant-response-development-continues',
                )
            )
        ),
        design_origin=DesignOrigin(
            origin_id='origin.ambient-pressure-superconductor-gauge-covariant-response-owner-issued-amendment',
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
                observation_operator=observation_operator,
                numerical_view_ids=tuple(
                    view.view_id for view in system.numerical_views
                ),
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
        coverage_id='formal-gap-coverage.ambient-pressure-superconductor-gauge-covariant-response',
        denominator_id=system.system_id,
        candidate_act_id=draft.draft_id,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id='formal-source-inventory.ambient-pressure-superconductor-gauge-covariant-response',
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
            sorted(gap.gap_id for gap in register.gaps)
        ),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    entry = _entry_package(draft=draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id='programme-authoring-package.ambient-pressure-superconductor-gauge-covariant-response',
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id='context.ambient-pressure-superconductor-gauge-covariant-response',
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=tuple(
            sorted((design_input, model_input), key=lambda value: value.input_id)
        ),
        implementation_sha256=implementation_sha256,
    )
    return GaugeCovariantResponseAuthoringBundle(
        config=config,
        config_payload=config_payload,
        source_contract=source_contract,
        source_config=source_config,
        system=system,
        experiment=experiment,
        model_set=model_set,
        campaign=campaign,
        protocol=protocol,
        conditional_protocols=conditional_protocols,
        registry=registry,
        conditional_registry=conditional_registry,
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


__all__ = ['GAUGE_COVARIANT_RESPONSE_SCOPE_ID', 'GaugeCovariantResponseAuthoringBundle', 'build_gauge_covariant_response_authoring_bundle']
