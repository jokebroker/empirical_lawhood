"""Production-owned public authoring composition for metatheory R1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.physical_scale_morphism.authoring import build_physical_scale_morphism_authoring_bundle
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope, WorldKind
from empirical_lawhood.planning.campaigns import CampaignNode
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition, ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodBinding,
    FormalMethodCatalog,
    FormalMethodRole,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalDomain,
    FormalGapApplicability,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
    FormalGapSpec,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, MaterializationQualificationReceipt, StudyDraft, SourceMaterializationRef
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateScientificGraph, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, required_candidate_obligation_ids
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration

from .executable_binding import SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_REGISTRATIONS
from .extension_bundle import SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS
from .source_free_property_transport_inputs import SourceFreePropertyTransportCaseInputs
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE, SourceFreePropertyTransportSourcePipelineConfig


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportAuthoringBundle:
    inputs: SourceFreePropertyTransportCaseInputs
    authoring: ExecutableStudyDefinition
    standard_context: StandardCandidateCompilationContext
    catalog: CandidateCapabilityCatalog
    decoder_registrations: tuple[StudyExtensionDecoderRegistration, ...]


def _portfolio_world(draft: StudyDraft) -> FormalGapEvidenceWorld:
    assert draft.system is not None
    return {
        WorldKind.ANALYTIC_REFERENCE: FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        WorldKind.NUMERICAL_SIMULATOR: FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        WorldKind.PHYSICAL_EXPERIMENT: FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
    }[draft.system.world.kind]


def _entry_package_for_draft(draft: StudyDraft) -> ExperimentEntryPackage:
    source = ObjectIdentity(
        object_id="source.source-free-property-transport.formal-gap",
        object_schema='empirical-lawhood/methods/structural-transport/synthetic-input/formal-gap-source',
        object_version="1.0.0",
        object_fingerprint=sha256(b"source-free-property-transport-formal-gap-source").hexdigest(),
    )
    world = _portfolio_world(draft)
    gaps = tuple(
        FormalGapSpec(
            gap_id=f"gap.source-free-property-transport.{domain.value.lower()}",
            domain=domain,
            question_family=f"Source-free metatheory {domain.value.lower()} conformance.",
            required_operand_ids=("operand.source-free-property-transport.synthetic",),
            compatible_evidence_worlds=(world,),
            minimum_independent_units=1,
            minimum_numerical_views=0,
            estimator_family_ids=("estimator.source-free-property-transport",),
            control_ids=("control.source-free-property-transport",),
            decisive_falsifier_ids=("falsifier.source-free-property-transport",),
            support_prerequisite_ids=("support.source-free-property-transport",),
            multiplicity_family_id="multiplicity.source-free-property-transport",
            maximum_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            provenance_source_ids=(source.object_id,),
        )
        for domain in FormalDomain
    )
    register = FormalGapRegister(
        register_id="formal-gap-register.source-free-property-transport",
        register_version="1.0.0",
        provenance_sources=(source,),
        gaps=tuple(sorted(gaps, key=lambda value: value.gap_id)),
    )
    applicability = tuple(
        FormalGapApplicability(
            gap_id=gap.gap_id,
            evidence_world=world,
            present_operand_ids=(),
            satisfied_prerequisite_ids=(),
            independent_unit_ids=(),
            independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
            numerical_view_ids=(),
            available_estimator_family_ids=(),
            available_control_ids=(),
            multiplicity_family_ids=(),
            requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
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
            reason_codes=("CONTRACT_CONFORMANCE_DENOMINATOR",),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for gap in register.gaps
    )
    coverage = FormalGapCoverage(
        coverage_id="formal-gap-coverage.source-free-property-transport",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id="denominator.source-free-property-transport.contract-conformance",
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.source-free-property-transport.{requirement.value.lower()}",),
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
        checklist_id=f"entry-checklist.{draft.draft_id}",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=world,
        execution_route_id="route.source-free-property-transport.public-issue-compilation-execution",
        durability_disposition_id="durability.source-free-property-transport.receipt-first",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=f"experiment-entry-package.{draft.draft_id}",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def _standard_authoring(
    draft: StudyDraft,
    base_context: CandidateCompilationContext,
    *,
    formal_method_capability_key: str,
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    assert draft.system is not None and draft.experiment is not None
    entry = _entry_package_for_draft(draft)
    template = base_context.template(draft.dag_template_key)
    assert template is not None
    graph_binding = template.coverage.bindings[0]
    evaluator = base_context.registry.resolve(formal_method_capability_key, "1.0.0")
    control_id = draft.experiment.controls[0].control_id
    register = replace(
        entry.register,
        gaps=tuple(
            replace(
                value,
                estimator_family_ids=("estimator.source-free-property-transport",),
                control_ids=(control_id,),
                multiplicity_family_id="multiplicity.source-free-property-transport",
            )
            for value in entry.register.gaps
        ),
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.source-free-property-transport",
        denominator_id=draft.system.system_id,
        evidence_world=entry.checklist.portfolio_world,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in draft.source_materializations),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=("operand.source-free-property-transport.synthetic",),
        satisfied_prerequisite_ids=("support.source-free-property-transport",),
        independent_unit_ids=(draft.development_unit_ids[0],),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(),
        available_estimator_family_ids=("estimator.source-free-property-transport",),
        available_control_ids=(control_id,),
        multiplicity_family_ids=("multiplicity.source-free-property-transport",),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
            readiness_reason=None,
            reason_codes=(),
            selected_estimator_family_id="estimator.source-free-property-transport",
            selected_control_ids=(control_id,),
            selected_multiplicity_family_id="multiplicity.source-free-property-transport",
            obligation_ids=(graph_binding.obligation_id,),
            output_ids=(graph_binding.required_output_id,),
            adjudication_owner_ids=(graph_binding.proof_owner_node_id,),
        )
        for gap in register.gaps
    )
    coverage = replace(
        entry.coverage,
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=draft.system.system_id,
        applicability=applicability,
        assignments=assignments,
    )
    checklist = replace(
        entry.checklist,
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
    )
    entry = replace(entry, register=register, coverage=coverage, checklist=checklist)
    authoring = StudyDefinition(
        package_id=f"programme-authoring-package.{draft.draft_id}",
        draft=draft,
        entry_package=entry,
    )
    gap_ids = tuple(value.gap_id for value in register.gaps)
    bindings = tuple(
        sorted(
            (
                FormalMethodBinding(
                    family_id=family_id,
                    role=role,
                    capability_key=evaluator.capability_key,
                    capability_version=evaluator.capability_version,
                    implementation_sha256=evaluator.implementation_sha256,
                    supported_gap_ids=gap_ids,
                    input_schema_id=evaluator.input_schema_ids[0],
                    output_schema_id=evaluator.output_schema_ids[0],
                    maximum_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                )
                for role, family_id in (
                    (FormalMethodRole.ESTIMATOR, "estimator.source-free-property-transport"),
                    (FormalMethodRole.MULTIPLICITY, "multiplicity.source-free-property-transport"),
                )
            ),
            key=lambda value: value.binding_id,
        )
    )
    return authoring, StandardCandidateCompilationContext(
        context_id=f"standard-context.{draft.draft_id}",
        base=base_context,
        formal_methods=FormalMethodCatalog(
            catalog_id=f"formal-method-catalog.{draft.draft_id}",
            bindings=bindings,
        ),
        source_inventories=(inventory,),
    )


def _coverage(
    draft: StudyDraft,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    assert draft.experiment is not None
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation in step.obligation_ids
    }
    bindings = []
    for obligation in required_candidate_obligation_ids(
        draft.experiment,
        protocol,
    ):
        node_id, output_id = owners.get(obligation, ("metatheory-report", "result"))
        bindings.append(
            ObligationCoverageBinding(
                obligation_id=obligation,
                proof_owner_node_id=node_id,
                required_output_id=output_id,
                contributor_edge_ids=incoming[node_id],
            )
        )
    return ObligationCoverage(
        coverage_id=f"coverage.{draft.draft_id}",
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def _role_for_config(config: CanonicalRecord) -> MetatheoryCampaignStageRole:
    return next(
        role
        for role, record_type in SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE.items()
        if isinstance(config, record_type)
    )


def build_source_free_property_transport_authoring(
    inputs: SourceFreePropertyTransportCaseInputs,
    *,
    implementation_sha256: str | None = None,
) -> SourceFreePropertyTransportAuthoringBundle:
    source_config = next(
        value for value in inputs.configs if isinstance(value, SourceFreePropertyTransportSourcePipelineConfig)
    )
    token = source_config.config_id.split(".")[2]
    implementation = implementation_sha256 or sha256(b'source-free-property-transport-authoring').hexdigest()
    base = build_physical_scale_morphism_authoring_bundle(implementation_sha256=implementation)
    prepared_medium = source_config.prepared_medium
    qualification = replace(
        base.qualification,
        receipt_id=f"qualification.source-free-property-transport.{token}",
        source_id=prepared_medium.medium_id,
        materialization=ObjectIdentity.from_record(
            prepared_medium.medium_id,
            prepared_medium,
        ),
        content_sha256=prepared_medium.fingerprint(),
    )
    source_materialization = replace(
        base.draft.source_materializations[0],
        source_id=prepared_medium.medium_id,
        materialization=ObjectIdentity.from_record(
            prepared_medium.medium_id,
            prepared_medium,
        ),
        content_sha256=prepared_medium.fingerprint(),
        source_config_sha256=source_config.fingerprint(),
        qualification_receipt=ObjectIdentity.from_record(
            qualification.receipt_id,
            qualification,
        ),
    )
    assert isinstance(qualification, MaterializationQualificationReceipt)
    assert isinstance(source_materialization, SourceMaterializationRef)
    claim = replace(
        base.experiment.claims[0],
        claim_id=f"claim.source-free-property-transport.{token}.contract-conformance",
        proposition="The installed real-owner metatheory route preserves its typed contracts.",
        estimand="Source-free owner-routing and custody contract conformance only.",
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    experiment = replace(
        base.experiment,
        experiment_id=f"experiment.source-free-property-transport.{token}",
        claims=(claim,),
    )
    old_experiment = ObjectIdentity.from_record(
        base.experiment.experiment_id,
        base.experiment,
    )
    new_experiment = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    nodes = tuple(
        replace(value, object_identity=new_experiment)
        if value.object_identity == old_experiment
        else value
        for value in base.campaign.nodes
    )
    assert all(isinstance(value, CampaignNode) for value in nodes)
    campaign = replace(
        base.campaign,
        campaign_id=f"campaign.source-free-property-transport.{token}",
        objective="Run the bounded source-free real-owner metatheory conformance vertical.",
        target_claim_ids=(claim.claim_id,),
        nodes=nodes,
        evidence_state=tuple(
            new_experiment if value == old_experiment else value
            for value in base.campaign.evidence_state
        ),
    )
    active_keys = {value.capability_key for value in inputs.compilation.protocol.steps}
    registrations = tuple(
        value
        for value in SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_REGISTRATIONS
        if value.manifest.capability_key in active_keys
    )
    registry = CapabilityRegistry(
        registry_id=f"registry.source-free-property-transport.{token}.selected",
        capabilities=tuple(
            sorted((value.manifest for value in registrations), key=lambda value: value.registry_id)
        ),
    )
    draft = replace(
        base.draft,
        draft_id=f"draft.source-free-property-transport.{token}",
        question="Does the installed source-free route invoke each declared metatheory owner?",
        alternative_ids=(
            "alternative.source-free-property-transport.owner-route-conforms",
            "alternative.source-free-property-transport.owner-route-refuses",
        ),
        experiment=experiment,
        campaign=campaign,
        dag_template_key=f"executable-source-free-property-transport.{token}",
        capability_selections=tuple(
            sorted(
                (
                    CapabilitySelection(
                        capability_key=value.capability_key,
                        capability_version=value.capability_version,
                        implementation_sha256=value.implementation_sha256,
                    )
                    for value in registry.capabilities
                ),
                key=lambda value: value.selection_id,
            )
        ),
        source_materializations=(source_materialization,),
    )
    template = StudyTemplate(
        template_key=draft.dag_template_key,
        template_version="1.0.0",
        protocol=inputs.compilation.protocol,
        graph=inputs.graph,
        coverage=_coverage(
            draft,
            inputs.compilation.protocol,
            inputs.graph,
        ),
    )
    catalog = CandidateCapabilityCatalog(
        catalog_id=f"catalog.source-free-property-transport.{token}",
        registrations=tuple(sorted(registrations, key=lambda value: value.registration_id)),
        templates=(template,),
    )
    context = CandidateCompilationContext(
        context_id=f"context.source-free-property-transport.{token}",
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=base.context.known_design_inputs,
        implementation_sha256=implementation,
    )
    authoring, standard_context = _standard_authoring(
        draft,
        context,
        formal_method_capability_key=registry.capabilities[0].capability_key,
    )
    decoder_by_role = dict(SOURCE_FREE_PROPERTY_TRANSPORT_DECODER_REGISTRATIONS)
    proposals = []
    decoders = []
    for index, config in enumerate(inputs.configs, start=1):
        role = _role_for_config(config)
        decoder = decoder_by_role[role]
        namespace = f"source-free-property-transport-{token}-{index:02d}"
        proposals.append(
            ProposedStudyExtension(
                extension_id=f"extension.{namespace}",
                namespace_id=namespace,
                payload=ObjectIdentity.from_record(getattr(config, "config_id"), config),
                payload_size_bytes=len(config.canonical_bytes()),
                decoder_key=decoder.decoder_key,
                decoder_version=decoder.decoder_version,
                decoder_config_sha256=decoder.config_sha256,
                required_for_activation=True,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            )
        )
        decoders.append(decoder)
    extension_set = ProposedStudyExtensionSet(
        extension_set_id=f"extensions.source-free-property-transport.{token}",
        authoring_package=ObjectIdentity.from_record(authoring.package_id, authoring),
        namespace_roster=tuple(sorted(value.namespace_id for value in proposals)),
        extensions=tuple(sorted(proposals, key=lambda value: value.extension_id)),
    )
    wrapper = ExecutableStudyDefinition(
        package_id=f"programme-authoring-package.source-free-property-transport.{token}",
        base=authoring,
        extension_set=extension_set,
    )
    return SourceFreePropertyTransportAuthoringBundle(
        inputs=inputs,
        authoring=wrapper,
        standard_context=standard_context,
        catalog=catalog,
        decoder_registrations=tuple(decoders),
    )


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportCandidateContextProvider:
    """Exact no-contact context port for one frozen source-free authoring root."""

    bundle: SourceFreePropertyTransportAuthoringBundle

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft != self.bundle.authoring.base.draft:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CANDIDATE_DRAFT_MISMATCH")
        source = CandidateSourceResolution(
            qualifications=self.bundle.standard_context.base.qualifications,
            receipts=(),
            diagnostics=(),
        )
        return CandidateContextResolution(
            context=self.bundle.standard_context.base,
            diagnostics=(),
            source_resolution=source,
        )

    def resolve_standard(
        self,
        package: StudyDefinition,
    ) -> StandardCandidateContextResolution:
        if package != self.bundle.authoring.base:
            raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_STANDARD_AUTHORING_MISMATCH")
        source = CandidateSourceResolution(
            qualifications=self.bundle.standard_context.base.qualifications,
            receipts=(),
            diagnostics=(),
        )
        return StandardCandidateContextResolution(
            context=self.bundle.standard_context,
            diagnostics=(),
            source_resolution=source,
        )


__all__ = [
    'SourceFreePropertyTransportAuthoringBundle',
    'SourceFreePropertyTransportCandidateContextProvider',
    'build_source_free_property_transport_authoring',
]
