"""Strict programme authoring root for one SDCB-SC phase."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from empirical_lawhood.adapters.methods import standard_formal_method_catalog
from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
)
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
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
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalGapApplicability,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate, StandardCandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .codecs import MATERIAL_CORPUS_TABLE_SCHEMA
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MaterialFamilyDiscoveryPhase, MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION, SourceQualification
from .protocol import CORPUS_INPUT_ID, SOURCE_MANIFEST_INPUT_ID, SOURCE_QUALIFICATION_INPUT_ID, material_family_candidate_catalog, material_family_discovery_study_template, material_family_protocol, world_configs
from .records import MaterialFamilyDiscoveryWorldBuildConfig
from .registration import material_family_capability_registry
from .system import NUMERICAL_VIEW_ID, study_budget, material_family_experiment, material_family_system


def _identity(
    *,
    object_id: str,
    schema: str,
    fingerprint: str,
) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=fingerprint,
    )


def _phase_slug(phase: MaterialFamilyDiscoveryPhase) -> str:
    return phase.value.lower()


def material_family_campaign(
    *,
    family_config: MaterialFamilyConfig,
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    slug = _phase_slug(family_config.phase)
    node = CampaignNode(
        node_id=f"campaign-node.material-family-discovery-{slug}",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    rights = [
        DecisionRight(
            decision_right_id=f"decision-right.material-family-discovery-{slug}-execute",
            action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
            decision_maker_id=system.authority_policy.delegate_id,
            authority_policy_id=system.authority_policy.policy_id,
            delegated=True,
        )
    ]
    if family_config.phase is MaterialFamilyDiscoveryPhase.EVALUATION:
        rights.append(
            DecisionRight(
                decision_right_id='decision-right.material-family-discovery-evaluation-reveal',
                action=AuthorityAction.EVALUATOR_REVEAL,
                decision_maker_id=system.authority_policy.delegator_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=False,
            )
        )
    return CampaignSpec(
        campaign_id=f"campaign.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        objective=(
            "Execute the matched held-family first-discovery policy roster with "
            "causal batch commitments and family-level adjudication."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=study_budget(),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=tuple(sorted(rights, key=lambda value: value.decision_right_id)),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def material_family_formal_coverage(
    *,
    register: FormalGapRegister,
    denominator_id: str,
    draft_id: str,
    phase: MaterialFamilyDiscoveryPhase,
) -> FormalGapCoverage:
    "Keep the benchmark hierarchy distinct from metatheory measurement through controller use claims."

    applicability = tuple(
        FormalGapApplicability(
            gap_id=gap.gap_id,
            evidence_world=FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
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
            reason_codes=('benchmark-comparison-is-not-a-measurement-through-controller-use-law-claim',),
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
        coverage_id=(f"formal-gap-coverage.material-family-discovery-{_phase_slug(phase)}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"),
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
    phase: MaterialFamilyDiscoveryPhase,
) -> ExperimentEntryPackage:
    slug = _phase_slug(phase)
    authority_reasons = ["local-nonactuating-execution-authorization-required"]
    if phase is MaterialFamilyDiscoveryPhase.EVALUATION:
        authority_reasons.append("separate-evaluator-reveal-authorization-required")
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.material-family-discovery.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                tuple(sorted(authority_reasons))
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id=f"entry-checklist.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
        execution_route_id='route.material-family-discovery-trusted-local',
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=f"experiment-entry-package.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryAuthoringBundle:
    family_config: MaterialFamilyConfig
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig
    policies: tuple[DiscoveryPolicyConfig, ...]
    source_manifest: MaterialSourceManifest
    source_qualification: SourceQualification
    corpus_payload: bytes
    corpus_sha256: str
    world_configs: tuple[MaterialFamilyDiscoveryWorldBuildConfig, ...]
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    registry: CapabilityRegistry
    protocol: ProtocolTemplate
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    source_configs: tuple[SourceMaterializationConfig, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    design_inputs: tuple[DesignInputRecord, ...]
    draft: StudyDraft
    formal_coverage: FormalGapCoverage
    inventory: FormalGapSourceCapabilityInventory
    formal_methods: FormalMethodCatalog
    entry_package: ExperimentEntryPackage
    package: StudyDefinition
    context: CandidateCompilationContext
    standard_context: StandardCandidateCompilationContext

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        payloads = (
            self.family_config.canonical_bytes(),
            self.adjudication_config.canonical_bytes(),
            self.source_manifest.canonical_bytes(),
            self.source_qualification.canonical_bytes(),
            self.corpus_payload,
            *(value.canonical_bytes() for value in self.policies),
            *(value.canonical_bytes() for value in self.world_configs),
            *(value.canonical_bytes() for value in self.source_configs),
            *(value.canonical_bytes() for value in self.qualifications),
        )
        by_hash = {sha256(value).hexdigest(): value for value in payloads}
        return tuple(by_hash[key] for key in sorted(by_hash))


def build_material_family_authoring_bundle(
    *,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
    source_manifest: MaterialSourceManifest,
    source_qualification: SourceQualification,
    corpus_payload: bytes,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> MaterialFamilyDiscoveryAuthoringBundle:
    corpus_sha = sha256(corpus_payload).hexdigest()
    if family_config.source_manifest_sha256 != source_manifest.fingerprint():
        raise ValueError("SC authoring source manifest differs")
    if family_config.source_qualification_sha256 != source_qualification.fingerprint():
        raise ValueError("SC authoring source qualification differs")
    if adjudication_config.family_config_sha256 != family_config.fingerprint():
        raise ValueError("SC authoring adjudication config differs")
    if {value.fingerprint() for value in policies} != set(family_config.policy_config_sha256s):
        raise ValueError("SC authoring policy roster differs")
    if not corpus_payload:
        raise ValueError("SC authoring corpus payload is absent")

    system = material_family_system()
    experiment = material_family_experiment(
        system=system,
        family_config=family_config,
        adjudication_config=adjudication_config,
    )
    campaign = material_family_campaign(
        family_config=family_config,
        system=system,
        experiment=experiment,
    )
    registry = material_family_capability_registry(
        phase=family_config.phase,
        implementation_sha256=implementation_sha256,
    )
    protocol = material_family_protocol(
        experiment=experiment,
        registry=registry,
        family_config=family_config,
        adjudication_config=adjudication_config,
        policies=policies,
    )
    template = material_family_discovery_study_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        family_config=family_config,
        source_manifest=source_manifest,
        source_qualification=source_qualification,
        corpus_sha256=corpus_sha,
        corpus_size_bytes=len(corpus_payload),
        policies=policies,
    )
    catalog = material_family_candidate_catalog(registry=registry, template=template)
    worlds = world_configs(family_config)
    source_definitions = (
        (
            SOURCE_MANIFEST_INPUT_ID,
            SourceMaterializationRole.OBSERVATION_STREAM,
            source_manifest.canonical_bytes(),
            MaterialSourceManifest.SCHEMA,
            "application/json",
            ObjectIdentity.from_record(source_manifest.manifest_id, source_manifest),
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            SourceReadMode.ORDINARY_BOUNDED,
        ),
        (
            SOURCE_QUALIFICATION_INPUT_ID,
            SourceMaterializationRole.CALIBRATION,
            source_qualification.canonical_bytes(),
            SourceQualification.SCHEMA,
            "application/json",
            ObjectIdentity.from_record(
                source_qualification.qualification_id,
                source_qualification,
            ),
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.OUTCOME_VISIBLE,
            SourceReadMode.ORDINARY_BOUNDED,
        ),
        (
            CORPUS_INPUT_ID,
            SourceMaterializationRole.PREPARED_MEDIUM,
            corpus_payload,
            MATERIAL_CORPUS_TABLE_SCHEMA,
            "application/vnd.apache.arrow.file",
            _identity(
                object_id=CORPUS_INPUT_ID,
                schema=MATERIAL_CORPUS_TABLE_SCHEMA,
                fingerprint=corpus_sha,
            ),
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.OUTCOME_VISIBLE,
            SourceReadMode.CHUNKED_HIGH_VOLUME,
        ),
    )
    source_configs = tuple(
        SourceMaterializationConfig(
            config_id=f"source-config.{source_id}",
            source_id=source_id,
            role=role,
            content_sha256=sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
            maximum_bytes=max(len(payload), 1024**2),
            payload_schema=schema,
            media_type=media_type,
            read_mode=read_mode,
            outcome_access=access,
            visibility_ceiling=visibility,
        )
        for (
            source_id,
            role,
            payload,
            schema,
            media_type,
            _materialization,
            access,
            visibility,
            read_mode,
        ) in source_definitions
    )
    source_config_by_id = {value.source_id: value for value in source_configs}
    observer = _identity(
        object_id=f"observer.material-family-discovery-qualified-material-corpus-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        schema='empirical-lawhood/material/family-discovery/corpus-observer',
        fingerprint=implementation_sha256,
    )
    unit_ids = tuple(sorted({value.native_unit for value in system.quantities}))
    frame_ids = tuple(sorted({value.coordinate_frame for value in system.quantities}))
    qualifications = tuple(
        MaterializationQualificationReceipt(
            receipt_id=f"qualification.{source_id}.{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
            source_id=source_id,
            materialization=materialization,
            content_sha256=sha256(payload).hexdigest(),
            evidence_world_id=system.world.world_id,
            observation_operator=observer,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            native_unit_ids=unit_ids,
            frame_ids=frame_ids,
            clock_ids=(system.clocks[0].clock_id,),
            receiver_semantics_id=system.relation.relation_id,
            validity_contract_id=experiment.obligations.validity.validity_id,
            uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            outcome_access=access,
            visibility_ceiling=visibility,
        )
        for (
            source_id,
            _role,
            payload,
            _schema,
            _media_type,
            materialization,
            access,
            visibility,
            _read_mode,
        ) in source_definitions
    )
    qualification_by_id = {value.source_id: value for value in qualifications}
    source_refs = tuple(
        SourceMaterializationRef(
            source_id=source_id,
            role=role,
            evidence_world_id=system.world.world_id,
            materialization=materialization,
            content_sha256=sha256(payload).hexdigest(),
            source_config_sha256=source_config_by_id[source_id].fingerprint(),
            observation_operator=observer,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            qualification_receipt=ObjectIdentity.from_record(
                qualification_by_id[source_id].receipt_id,
                qualification_by_id[source_id],
            ),
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        )
        for (
            source_id,
            role,
            payload,
            _schema,
            _media_type,
            materialization,
            _access,
            _visibility,
            _read_mode,
        ) in source_definitions
    )
    cutoff = experiment.information_cutoffs[0]
    design_inputs = tuple(
        sorted(
            (
                DesignInputRecord(
                    input_id='design-input.material-family-discovery-family-config',
                    object_identity=ObjectIdentity.from_record(
                        family_config.config_id, family_config
                    ),
                    materialization_sha256=family_config.fingerprint(),
                    information_cutoff=cutoff,
                    role=DesignInputRole.DEVELOPMENT_TUNING,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    operator_id="human.project-owner",
                ),
                DesignInputRecord(
                    input_id='design-input.material-family-discovery-adjudication-config',
                    object_identity=ObjectIdentity.from_record(
                        adjudication_config.config_id, adjudication_config
                    ),
                    materialization_sha256=adjudication_config.fingerprint(),
                    information_cutoff=cutoff,
                    role=DesignInputRole.CLAIM_DERIVATION,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    operator_id="human.project-owner",
                ),
            ),
            key=lambda value: value.input_id,
        )
    )
    slug = _phase_slug(family_config.phase)
    draft_id = f"draft.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the local-law boundary explorer outperform pinned constrained "
            "BoTorch UCB on matched held-family first discovery, or satisfy the "
            "frozen query/false-promotion Pareto alternative?"
        ),
        alternative_ids=(
            'alternative.material-family-discovery-comparison-not-supported',
            'alternative.material-family-discovery-comparison-pareto-supported',
            'alternative.material-family-discovery-comparison-superiority-supported',
            'alternative.material-family-discovery-unevaluable',
        ),
        design_origin=DesignOrigin(
            origin_id=f"origin.material-family-discovery-{slug}-predeclared-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=design_inputs,
        development_unit_ids=family_config.development_family_ids,
        evaluation_unit_ids=family_config.evaluation_family_ids,
        development_seed_ids=(f"seed.material-family-discovery-{slug}-policies-20260814",),
        evaluation_seed_ids=('seed.material-family-discovery-evaluation-bootstrap-2026081401',),
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
        source_materializations=tuple(sorted(source_refs, key=lambda value: value.source_id)),
        resource_ceiling=study_budget(),
    )
    formal_coverage = material_family_formal_coverage(
        register=register,
        denominator_id=system.system_id,
        draft_id=draft_id,
        phase=family_config.phase,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=(f"formal-source-inventory.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"),
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in source_refs),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=(),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=tuple(sorted(value.gap_id for value in register.gaps)),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    entry = _entry_package(
        draft=draft,
        register=register,
        coverage=formal_coverage,
        phase=family_config.phase,
    )
    package = StudyDefinition(
        package_id=f"programme-authoring-package.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id=f"context.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        registry=registry,
        templates=(template,),
        qualifications=tuple(sorted(qualifications, key=lambda value: value.receipt_id)),
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    formal_methods = standard_formal_method_catalog(register)
    standard_context = StandardCandidateCompilationContext(
        context_id=f"standard-context.material-family-discovery-{slug}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        base=context,
        formal_methods=formal_methods,
        source_inventories=(inventory,),
    )
    return MaterialFamilyDiscoveryAuthoringBundle(
        family_config=family_config,
        adjudication_config=adjudication_config,
        policies=policies,
        source_manifest=source_manifest,
        source_qualification=source_qualification,
        corpus_payload=corpus_payload,
        corpus_sha256=corpus_sha,
        world_configs=worlds,
        system=system,
        experiment=experiment,
        campaign=campaign,
        registry=registry,
        protocol=protocol,
        template=template,
        catalog=catalog,
        source_configs=tuple(sorted(source_configs, key=lambda value: value.source_id)),
        qualifications=tuple(sorted(qualifications, key=lambda value: value.receipt_id)),
        design_inputs=design_inputs,
        draft=draft,
        formal_coverage=formal_coverage,
        inventory=inventory,
        formal_methods=formal_methods,
        entry_package=entry,
        package=package,
        context=context,
        standard_context=standard_context,
    )


__all__ = [
    'MaterialFamilyDiscoveryAuthoringBundle',
    'build_material_family_authoring_bundle',
    'material_family_campaign',
    'material_family_formal_coverage',
]
