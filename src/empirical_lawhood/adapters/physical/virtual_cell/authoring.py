"""Standard outcome-blind authoring root for the 2025 VCC Tier-L0 act."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from empirical_lawhood.adapters.methods import standard_formal_method_catalog
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
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
from empirical_lawhood.runtime.source_resolution import (
    SourceMaterializationConfig,
    SourceReadMode,
)

from .config import (
    CAPABILITY_VERSION,
    build_virtual_cell_2025_pipeline_config,
)
from .contracts import LeaderboardSnapshot, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest, VirtualCellSourceObject
from .materialization import TargetFeatureMaterializationReceipt
from .protocol import (
    SOURCE_MANIFEST_INPUT_ID,
    TARGET_FEATURE_INPUT_ID,
    TEST_ROSTER_INPUT_ID,
    virtual_cell_candidate_catalog,
    virtual_cell_development_protocol,
    virtual_cell_development_template,
    virtual_cell_protocol,
    virtual_cell_template,
)
from .records import TARGET_FEATURE_TABLE_SCHEMA, TEST_ROSTER_TEXT_SCHEMA
from .registry import virtual_cell_capability_keys, virtual_cell_capability_registry
from .system import DEVELOPMENT_UNIT_IDS, EVALUATION_UNIT_IDS, NUMERICAL_VIEW_ID, study_budget, virtual_cell_2025_experiment, virtual_cell_2025_system


_DRAFT_ID = "draft.virtual-cell-2025-tier-l0-sealed-benchmark"
_DEVELOPMENT_DRAFT_ID = "draft.virtual-cell-2025-tier-l0-development"


def virtual_cell_campaign(
    system: SystemSpec, experiment: ExperimentSpec
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.virtual-cell-2025-tier-l0-benchmark",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id, experiment
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.virtual-cell-2025-tier-l0-sealed-benchmark",
        objective=(
            "Freeze one validation-selected response-law prediction before the shared "
            "test reveal, then evaluate it once with the exact 2025 scorer."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=study_budget(),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.virtual-cell-2025-freeze",
                action=AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
            DecisionRight(
                decision_right_id="decision-right.virtual-cell-2025-test-reveal",
                action=AuthorityAction.EVALUATOR_REVEAL,
                decision_maker_id=system.authority_policy.delegator_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=False,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def virtual_cell_development_campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.virtual-cell-2025-tier-l0-development",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id, experiment
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.virtual-cell-2025-tier-l0-development",
        objective=(
            "Run the validation-only Tier-L0 tournament and freeze one final-test "
            "prediction without requesting or opening the sealed final response."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=study_budget(),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.virtual-cell-2025-development-freeze",
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


def virtual_cell_formal_coverage(
    *,
    register: FormalGapRegister,
    denominator_id: str,
    candidate_act_id: str = _DRAFT_ID,
    coverage_id: str = "formal-gap-coverage.virtual-cell-2025-tier-l0",
) -> FormalGapCoverage:
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
            # The standard formal compiler currently requires its law qualification envelope
            # even when every gap is denominator-inapplicable.  The experiment
            # and world independently retain NON_PROMOTABLE as the actual claim.
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
            reason_codes=('competition-prediction-is-not-a-measurement-through-controller-use-law-claim',),
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
        coverage_id=coverage_id,
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=denominator_id,
        candidate_act_id=candidate_act_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
    id_suffix: str,
    requires_reveal: bool,
) -> ExperimentEntryPackage:
    authority_reasons = tuple(
        sorted(
            {
                "eligibility-license-and-ip-decision-required-before-submission",
                "submission-and-upload-authority-not-held",
                *(
                    ("separate-evaluator-reveal-authority-required",)
                    if requires_reveal
                    else ()
                ),
            }
        )
    )
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.virtual-cell-2025.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                authority_reasons
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
        checklist_id=f"entry-checklist.virtual-cell-2025-{id_suffix}",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
        execution_route_id="route.virtual-cell-2025-trusted-local",
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
        package_id=f"experiment-entry-package.virtual-cell-2025-{id_suffix}",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def _identity(
    *,
    object_id: str,
    schema: str,
    fingerprint: str,
) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version=CAPABILITY_VERSION,
        object_fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _SourceDefinition:
    source_id: str
    role: SourceMaterializationRole
    payload: bytes
    payload_schema: str
    media_type: str
    materialization: ObjectIdentity
    source_outcome_access: OutcomeAccess


@dataclass(frozen=True, slots=True)
class VirtualCellAuthoringBundle:
    config: ProvenanceBoundVirtualCellPipelineConfig
    config_payload: bytes
    source_manifest: VirtualCellSourceManifest
    test_source: VirtualCellSourceObject
    target_feature_payload: bytes
    test_roster_payload: bytes
    leaderboard_snapshot: LeaderboardSnapshot
    feature_materialization_receipt: TargetFeatureMaterializationReceipt
    source_configs: tuple[SourceMaterializationConfig, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    development_campaign: CampaignSpec
    protocol: ProtocolTemplate
    development_protocol: ProtocolTemplate
    registry: CapabilityRegistry
    template: StudyTemplate
    development_template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    inventory: FormalGapSourceCapabilityInventory
    formal_methods: FormalMethodCatalog
    formal_coverage: FormalGapCoverage
    development_formal_coverage: FormalGapCoverage
    entry_package: ExperimentEntryPackage
    development_entry_package: ExperimentEntryPackage
    package: StudyDefinition
    development_package: StudyDefinition
    draft: StudyDraft
    development_draft: StudyDraft
    context: CandidateCompilationContext
    standard_context: StandardCandidateCompilationContext

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        values = {
            sha256(payload).hexdigest(): payload
            for payload in (
                self.config_payload,
                self.source_manifest.canonical_bytes(),
                self.test_source.canonical_bytes(),
                self.target_feature_payload,
                self.test_roster_payload,
                self.leaderboard_snapshot.canonical_bytes(),
                *(value.canonical_bytes() for value in self.source_configs),
                *(value.canonical_bytes() for value in self.qualifications),
            )
        }
        return tuple(values[key] for key in sorted(values))


def build_virtual_cell_authoring_bundle(
    *,
    source_manifest: VirtualCellSourceManifest,
    target_feature_payload: bytes,
    test_roster_payload: bytes,
    leaderboard_snapshot: LeaderboardSnapshot,
    feature_materialization_receipt: TargetFeatureMaterializationReceipt,
    register: FormalGapRegister,
    implementation_sha256: str,
    pipeline_config: ProvenanceBoundVirtualCellPipelineConfig | None = None,
) -> VirtualCellAuthoringBundle:
    config = (
        build_virtual_cell_2025_pipeline_config(
            source_manifest=source_manifest,
            feature_receipt=feature_materialization_receipt,
            leaderboard=leaderboard_snapshot,
        )
        if pipeline_config is None
        else pipeline_config
    )
    if source_manifest.fingerprint() != config.source_manifest_sha256:
        raise ValueError("VCC authoring source manifest differs from the frozen config")
    if sha256(target_feature_payload).hexdigest() != config.target_feature_sha256:
        raise ValueError("VCC authoring target features differ from the frozen config")
    if leaderboard_snapshot.fingerprint() != config.leaderboard_snapshot_sha256:
        raise ValueError("VCC authoring leaderboard differs from the frozen config")
    if (
        feature_materialization_receipt.arrow_sha256 != config.target_feature_sha256
        or feature_materialization_receipt.feature_provenance_sha256
        != config.target_feature_provenance_sha256
        or feature_materialization_receipt.build_receipt_sha256
        != config.target_feature_build_receipt_sha256
        or feature_materialization_receipt.source_manifest_sha256
        != source_manifest.fingerprint()
    ):
        raise ValueError("VCC target-feature materialization receipt differs")
    test_sources = tuple(value for value in source_manifest.objects if value.sealed)
    if len(test_sources) != 1:
        raise ValueError("VCC source manifest lacks one exact sealed test object")
    test_source = test_sources[0]
    roster_source = next(
        value
        for value in source_manifest.objects
        if value.split.value == "TEST" and value.role == "prefix/target-roster"
    )
    if (
        len(test_roster_payload) != roster_source.size_bytes
        or sha256(test_roster_payload).hexdigest() != roster_source.sha256
    ):
        raise ValueError("VCC test roster bytes differ from sealed source custody")

    system = virtual_cell_2025_system()
    experiment = virtual_cell_2025_experiment(system, config, test_source=test_source)
    campaign = virtual_cell_campaign(system, experiment)
    development_campaign = virtual_cell_development_campaign(system, experiment)
    registry = virtual_cell_capability_registry(
        {key: implementation_sha256 for key in virtual_cell_capability_keys()}
    )
    protocol = virtual_cell_protocol(registry=registry, config=config)
    development_protocol = virtual_cell_development_protocol(
        registry=registry,
        config=config,
    )
    template = virtual_cell_template(
        experiment=experiment,
        protocol=protocol,
        registry=registry,
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_sha256=sha256(target_feature_payload).hexdigest(),
        target_feature_size_bytes=len(target_feature_payload),
        test_roster_sha256=sha256(test_roster_payload).hexdigest(),
        test_roster_size_bytes=len(test_roster_payload),
        leaderboard_snapshot_sha256=leaderboard_snapshot.fingerprint(),
        leaderboard_snapshot_size_bytes=len(leaderboard_snapshot.canonical_bytes()),
    )
    development_template = virtual_cell_development_template(
        experiment=experiment,
        protocol=development_protocol,
        full_protocol=protocol,
        registry=registry,
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_sha256=sha256(target_feature_payload).hexdigest(),
        target_feature_size_bytes=len(target_feature_payload),
        test_roster_sha256=sha256(test_roster_payload).hexdigest(),
        test_roster_size_bytes=len(test_roster_payload),
        leaderboard_snapshot_sha256=leaderboard_snapshot.fingerprint(),
        leaderboard_snapshot_size_bytes=len(leaderboard_snapshot.canonical_bytes()),
    )
    catalog = virtual_cell_candidate_catalog(
        registry=registry,
        templates=(development_template, template),
    )
    definitions = (
        _SourceDefinition(
            source_id=SOURCE_MANIFEST_INPUT_ID,
            role=SourceMaterializationRole.PREPARED_MEDIUM,
            payload=source_manifest.canonical_bytes(),
            payload_schema=VirtualCellSourceManifest.SCHEMA,
            media_type="application/json",
            materialization=ObjectIdentity.from_record(
                source_manifest.manifest_id,
                source_manifest,
            ),
            source_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        _SourceDefinition(
            source_id=TARGET_FEATURE_INPUT_ID,
            role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
            payload=target_feature_payload,
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            media_type="application/vnd.apache.arrow.file",
            materialization=_identity(
                object_id=config.target_feature_artifact_id,
                schema=TARGET_FEATURE_TABLE_SCHEMA,
                fingerprint=sha256(target_feature_payload).hexdigest(),
            ),
            source_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        _SourceDefinition(
            source_id=TEST_ROSTER_INPUT_ID,
            role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
            payload=test_roster_payload,
            payload_schema=TEST_ROSTER_TEXT_SCHEMA,
            media_type="text/csv",
            materialization=_identity(
                object_id=TEST_ROSTER_INPUT_ID,
                schema=TEST_ROSTER_TEXT_SCHEMA,
                fingerprint=sha256(test_roster_payload).hexdigest(),
            ),
            source_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )
    observation_operator = _identity(
        object_id="observer.virtual-cell-2025-qualified-source",
        schema='empirical-lawhood/physical/virtual-cell/source-observer',
        fingerprint=implementation_sha256,
    )
    source_configs = tuple(
        SourceMaterializationConfig(
            config_id=f"source-config.{value.source_id}",
            source_id=value.source_id,
            role=value.role,
            content_sha256=sha256(value.payload).hexdigest(),
            expected_size_bytes=len(value.payload),
            maximum_bytes=max(len(value.payload), 1024**2),
            payload_schema=value.payload_schema,
            media_type=value.media_type,
            read_mode=SourceReadMode.ORDINARY_BOUNDED,
            outcome_access=value.source_outcome_access,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        for value in definitions
    )
    source_configs_by_id = {value.source_id: value for value in source_configs}
    qualifications = tuple(
        MaterializationQualificationReceipt(
            receipt_id=f"qualification.{value.source_id}",
            source_id=value.source_id,
            materialization=value.materialization,
            content_sha256=sha256(value.payload).hexdigest(),
            evidence_world_id=system.world.world_id,
            observation_operator=observation_operator,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
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
        for value in definitions
    )
    qualifications_by_id = {value.source_id: value for value in qualifications}
    source_refs = tuple(
        SourceMaterializationRef(
            source_id=value.source_id,
            role=value.role,
            evidence_world_id=system.world.world_id,
            materialization=value.materialization,
            content_sha256=sha256(value.payload).hexdigest(),
            source_config_sha256=source_configs_by_id[value.source_id].fingerprint(),
            observation_operator=observation_operator,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            qualification_receipt=ObjectIdentity.from_record(
                qualifications_by_id[value.source_id].receipt_id,
                qualifications_by_id[value.source_id],
            ),
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        )
        for value in definitions
    )
    cutoff = experiment.information_cutoffs[0]
    design_inputs = tuple(
        sorted(
            (
                DesignInputRecord(
                    input_id="design-input.virtual-cell-2025-frozen-config",
                    object_identity=ObjectIdentity.from_record(
                        config.config_id, config
                    ),
                    materialization_sha256=config.fingerprint(),
                    information_cutoff=cutoff,
                    role=DesignInputRole.DEVELOPMENT_TUNING,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    operator_id="human.project-owner",
                ),
                DesignInputRecord(
                    input_id="design-input.virtual-cell-2025-source-system",
                    object_identity=ObjectIdentity.from_record(
                        system.system_id, system
                    ),
                    materialization_sha256=system.fingerprint(),
                    information_cutoff=cutoff,
                    role=DesignInputRole.READINESS_METADATA,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    operator_id="human.project-owner",
                ),
                DesignInputRecord(
                    input_id="design-input.virtual-cell-2025-target-features",
                    object_identity=ObjectIdentity.from_record(
                        feature_materialization_receipt.receipt_id,
                        feature_materialization_receipt,
                    ),
                    materialization_sha256=feature_materialization_receipt.fingerprint(),
                    information_cutoff=cutoff,
                    role=DesignInputRole.DEVELOPMENT_TUNING,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    operator_id="human.project-owner",
                ),
            ),
            key=lambda value: value.input_id,
        )
    )
    draft = StudyDraft(
        draft_id=_DRAFT_ID,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Which frozen Tier-L0 response-law method best predicts the official 2025 "
            "held-target transcript responses under validation-only selection?"
        ),
        alternative_ids=(
            "alternative.virtual-cell-2025-baseline-retained",
            "alternative.virtual-cell-2025-no-predictive-signal",
            "alternative.virtual-cell-2025-rapo-l-selected",
            "alternative.virtual-cell-2025-unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.virtual-cell-2025-owner-predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=design_inputs,
        development_unit_ids=DEVELOPMENT_UNIT_IDS,
        evaluation_unit_ids=EVALUATION_UNIT_IDS,
        development_seed_ids=("seed.virtual-cell-2025-development",),
        evaluation_seed_ids=("seed.virtual-cell-2025-evaluator",),
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
        source_materializations=tuple(
            sorted(source_refs, key=lambda value: value.source_id)
        ),
        resource_ceiling=study_budget(),
    )
    development_draft = replace(
        draft,
        draft_id=_DEVELOPMENT_DRAFT_ID,
        question=(
            "Which validation-selected Tier-L0 response-law method should be frozen "
            "for the still-sealed 2025 final targets?"
        ),
        design_origin=replace(
            draft.design_origin,
            origin_id="origin.virtual-cell-2025-development-predeclared",
        ),
        campaign=development_campaign,
        dag_template_key=development_template.template_key,
        capability_selections=tuple(
            value
            for value in draft.capability_selections
            if value.capability_key
            in {step.capability_key for step in development_protocol.steps}
        ),
    )
    coverage = virtual_cell_formal_coverage(
        register=register,
        denominator_id=system.system_id,
    )
    development_coverage = virtual_cell_formal_coverage(
        register=register,
        denominator_id=system.system_id,
        candidate_act_id=_DEVELOPMENT_DRAFT_ID,
        coverage_id=("formal-gap-coverage.virtual-cell-2025-tier-l0-development"),
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.virtual-cell-2025-tier-l0",
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
        denominator_inapplicable_gap_ids=tuple(
            sorted(gap.gap_id for gap in register.gaps)
        ),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    entry_package = _entry_package(
        draft=draft,
        register=register,
        coverage=coverage,
        id_suffix="tier-l0",
        requires_reveal=True,
    )
    development_entry_package = _entry_package(
        draft=development_draft,
        register=register,
        coverage=development_coverage,
        id_suffix="tier-l0-development",
        requires_reveal=False,
    )
    package = StudyDefinition(
        package_id="programme-authoring-package.virtual-cell-2025-tier-l0",
        draft=draft,
        entry_package=entry_package,
    )
    development_package = StudyDefinition(
        package_id=(
            "programme-authoring-package.virtual-cell-2025-tier-l0-development"
        ),
        draft=development_draft,
        entry_package=development_entry_package,
    )
    context = CandidateCompilationContext(
        context_id="context.virtual-cell-2025-tier-l0",
        registry=registry,
        templates=tuple(
            sorted(
                (development_template, template),
                key=lambda value: value.template_key,
            )
        ),
        qualifications=tuple(
            sorted(qualifications, key=lambda value: value.receipt_id)
        ),
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    formal_methods = standard_formal_method_catalog(register)
    standard_context = StandardCandidateCompilationContext(
        context_id="standard-context.virtual-cell-2025-tier-l0",
        base=context,
        formal_methods=formal_methods,
        source_inventories=(inventory,),
    )
    return VirtualCellAuthoringBundle(
        config=config,
        config_payload=config.canonical_bytes(),
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_payload=target_feature_payload,
        test_roster_payload=test_roster_payload,
        leaderboard_snapshot=leaderboard_snapshot,
        feature_materialization_receipt=feature_materialization_receipt,
        source_configs=tuple(sorted(source_configs, key=lambda value: value.source_id)),
        qualifications=tuple(
            sorted(qualifications, key=lambda value: value.receipt_id)
        ),
        system=system,
        experiment=experiment,
        campaign=campaign,
        development_campaign=development_campaign,
        protocol=protocol,
        development_protocol=development_protocol,
        registry=registry,
        template=template,
        development_template=development_template,
        catalog=catalog,
        inventory=inventory,
        formal_methods=formal_methods,
        formal_coverage=coverage,
        development_formal_coverage=development_coverage,
        entry_package=entry_package,
        development_entry_package=development_entry_package,
        package=package,
        development_package=development_package,
        draft=draft,
        development_draft=development_draft,
        context=context,
        standard_context=standard_context,
    )


__all__ = [
    "VirtualCellAuthoringBundle",
    "build_virtual_cell_authoring_bundle",
    "virtual_cell_campaign",
    "virtual_cell_development_campaign",
    "virtual_cell_formal_coverage",
]
