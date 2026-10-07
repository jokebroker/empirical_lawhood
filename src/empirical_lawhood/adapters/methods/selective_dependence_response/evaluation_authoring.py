"""Standard sealed-evaluation authoring root for one selective dependence response target."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.formal_analysis import standard_formal_method_catalog
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.plans import ScientificInputRole
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .authoring import SelectiveDependenceResponseTargetPlanningProjection, build_target_campaign, build_target_experiment, build_target_system
from .composition import SelectiveDependenceResponseStageComposition, compose_target_evaluation
from .evaluation_protocol import SelectiveDependenceResponseEvaluationBinding


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseEvaluationAuthoringAct:
    """Pure authority-bound, issue-pending standard evaluation package."""

    projection: SelectiveDependenceResponseTargetPlanningProjection
    binding: SelectiveDependenceResponseEvaluationBinding
    stage_composition: SelectiveDependenceResponseStageComposition
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    source_configs: tuple[SourceMaterializationConfig, ...]
    source_records: tuple[CanonicalRecord, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    design_inputs: tuple[DesignInputRecord, ...]
    inventory: FormalGapSourceCapabilityInventory
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    draft: StudyDraft
    package: StudyDefinition
    context: StandardCandidateCompilationContext
    compilation: StudyCompilationReport

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        records: tuple[CanonicalRecord, ...] = (
            *self.stage_composition.configs,
            *self.source_configs,
            *self.source_records,
            *self.qualifications,
        )
        return tuple(value.canonical_bytes() for value in records)

    @property
    def candidate_composition(self) -> SelectiveDependenceResponseStageComposition:
        return replace(
            self.stage_composition,
            catalog=self.catalog,
            known_design_inputs=self.design_inputs,
            formal_source_inventories=(self.inventory,),
        )


def _complete_coverage(
    template: StudyTemplate,
    experiment: ExperimentSpec,
    *,
    target_slug: str,
) -> StudyTemplate:
    bindings = {value.obligation_id: value for value in template.coverage.bindings}
    evaluator = next(
        value
        for value in template.protocol.steps
        if value.step_id == "reveal-and-adjudicate-target"
    )
    incoming = tuple(
        sorted(
            value.edge_id
            for value in template.graph.edges
            if value.consumer_node_id == evaluator.step_id
        )
    )
    for obligation_id in required_candidate_obligation_ids(
        experiment,
        template.protocol,
    ):
        bindings.setdefault(
            obligation_id,
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=evaluator.step_id,
                required_output_id=evaluator.outputs[0].output_id,
                contributor_edge_ids=incoming,
            ),
        )
    return replace(
        template,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.selective-dependence-response.{target_slug}.evaluation.standard",
            bindings=tuple(sorted(bindings.values(), key=lambda value: value.obligation_id)),
        ),
    )


def _graph_records(binding: SelectiveDependenceResponseEvaluationBinding) -> dict[str, CanonicalRecord]:
    return {
        "input.construct-review": binding.construct_review,
        "input.development-completion": binding.development_completion,
        "input.evaluation-package": binding.evaluation_package,
        "input.execution-authority": binding.execution_authority,
        "input.programme-forecast-completion": binding.study_completion,
        "input.reveal-authority": binding.reveal_authority,
        "input.source-qualification": binding.source_qualification,
        "input.target-analysis-freeze": binding.analysis_freeze,
        "input.target-design": binding.design,
    }


def _record_id(record: CanonicalRecord) -> str:
    for name in (
        "package_id",
        "envelope_id",
        "authority_id",
        "qualification_id",
        "freeze_id",
        "ledger_id",
        "attestation_id",
        "design_id",
    ):
        value = getattr(record, name, None)
        if isinstance(value, str):
            return value
    raise TypeError("evaluation authoring record lacks a stable identity")


def _sources(
    *,
    binding: SelectiveDependenceResponseEvaluationBinding,
    stage: SelectiveDependenceResponseStageComposition,
    template: StudyTemplate,
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> tuple[
    tuple[SourceMaterializationConfig, ...],
    tuple[CanonicalRecord, ...],
    tuple[MaterializationQualificationReceipt, ...],
    tuple[SourceMaterializationRef, ...],
]:
    records = _graph_records(binding)
    external = {value.input_id: value for value in template.graph.external_inputs}
    if set(records) != set(external):
        raise ValueError("evaluation authoring record roster differs from graph")
    source_ids = {
        value.input_id
        for value in external.values()
        if value.scientific_role is ScientificInputRole.MODEL
    }
    if source_ids != {"input.target-design"}:
        raise ValueError("evaluation formal source roster differs")
    observer = ObjectIdentity(
        object_id=f"observer.selective-dependence-response.{binding.target_slug}.evaluation-authoring",
        object_schema='empirical-lawhood/methods/selective-dependence-response/evaluation-authoring-observer',
        object_version="1.0.0",
        object_fingerprint=stage.implementation_sha256,
    )
    view_ids = tuple(value.view_id for value in system.numerical_views)
    native_units = tuple(sorted({value.native_unit for value in system.quantities}))
    frames = tuple(sorted({value.coordinate_frame for value in system.quantities}))
    clocks = tuple(value.clock_id for value in system.clocks)
    configs = []
    qualifications = []
    refs = []
    for input_id in sorted(source_ids):
        record = records[input_id]
        spec = external[input_id]
        config = SourceMaterializationConfig(
            config_id=f"source-config.selective-dependence-response.{binding.target_slug}.evaluation.{input_id}",
            source_id=input_id,
            role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
            content_sha256=record.fingerprint(),
            expected_size_bytes=len(record.canonical_bytes()),
            maximum_bytes=spec.maximum_size_bytes,
            payload_schema=record.SCHEMA,
            media_type=spec.media_type,
            read_mode=SourceReadMode.ORDINARY_BOUNDED,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        identity = ObjectIdentity.from_record(_record_id(record), record)
        qualification = MaterializationQualificationReceipt(
            receipt_id=(f"qualification.selective-dependence-response.{binding.target_slug}.evaluation.{input_id}"),
            source_id=input_id,
            materialization=identity,
            content_sha256=record.fingerprint(),
            evidence_world_id=system.world.world_id,
            observation_operator=observer,
            numerical_view_ids=view_ids,
            native_unit_ids=native_units,
            frame_ids=frames,
            clock_ids=clocks,
            receiver_semantics_id=system.relation.relation_id,
            validity_contract_id=experiment.obligations.validity.validity_id,
            uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        refs.append(
            SourceMaterializationRef(
                source_id=input_id,
                role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                evidence_world_id=system.world.world_id,
                materialization=identity,
                content_sha256=record.fingerprint(),
                source_config_sha256=config.fingerprint(),
                observation_operator=observer,
                numerical_view_ids=view_ids,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id,
                    qualification,
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            )
        )
        configs.append(config)
        qualifications.append(qualification)
    return (
        tuple(configs),
        tuple(records[value] for value in sorted(records)),
        tuple(qualifications),
        tuple(refs),
    )


def _design_inputs(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    binding: SelectiveDependenceResponseEvaluationBinding,
    cutoff: InformationCutoff,
) -> tuple[DesignInputRecord, ...]:
    records = _graph_records(binding)
    records["input.contamination-ledger"] = projection.binding.contamination_ledger
    values = []
    for input_id, record in records.items():
        access = (
            OutcomeAccess.OUTCOME_BLIND
            if input_id in {"input.execution-authority", "input.reveal-authority"}
            else getattr(record, "outcome_access", OutcomeAccess.OUTCOME_BLIND)
        )
        visible = access is OutcomeAccess.DEVELOPMENT_VISIBLE
        physical_ids: tuple[str, ...] = ()
        if input_id == "input.source-qualification":
            physical_ids = projection.binding.preparation.canary_unit_ids
        elif input_id == "input.development-completion":
            physical_ids = projection.binding.preparation.development_unit_ids
        values.append(
            DesignInputRecord(
                input_id=f"design-input.selective-dependence-response.{binding.target_slug}.evaluation.{input_id}",
                object_identity=ObjectIdentity.from_record(_record_id(record), record),
                materialization_sha256=record.fingerprint(),
                information_cutoff=cutoff,
                role=(
                    DesignInputRole.DEVELOPMENT_TUNING
                    if visible
                    else DesignInputRole.READINESS_METADATA
                ),
                outcome_access=access,
                visibility_ceiling=(
                    VisibilityCeiling.DEVELOPMENT_ONLY
                    if visible
                    else (
                        VisibilityCeiling.OUTCOME_VISIBLE
                        if input_id == "input.contamination-ledger"
                        else VisibilityCeiling.PROSPECTIVE
                    )
                ),
                operator_id=(
                    "service.selective-dependence-response-development-custody" if visible else "human.project-owner"
                ),
                physical_unit_ids=physical_ids,
            )
        )
    return tuple(sorted(values, key=lambda value: value.input_id))


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
    target_slug: str,
) -> ExperimentEntryPackage:
    applicability = derive_formal_gap_applicability(register, inventory)
    coverage = FormalGapCoverage(
        coverage_id=f"formal-gap-coverage.selective-dependence-response.{target_slug}.evaluation",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=tuple(
            FormalGapCoverageAssignment(
                gap_id=value.gap_id,
                disposition=FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE,
                readiness_reason=ReadinessStatus.COMPUTABILITY_BOUNDARY,
                reason_codes=("FORMAL_LOWERING_NOT_IN_BOUNDED_SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION",),
                selected_estimator_family_id=None,
                selected_control_ids=(),
                selected_multiplicity_family_id=None,
                obligation_ids=(),
                output_ids=(),
                adjudication_owner_ids=(),
            )
            for value in register.gaps
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.selective-dependence-response.{target_slug}.evaluation.{requirement.value.lower()}",),
            readiness=ReadinessStatus.READY,
            reason_codes=(),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id=f"entry-checklist.selective-dependence-response.{target_slug}.evaluation",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id="route.standard-candidate-issue-compile-execute",
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=f"experiment-entry-package.selective-dependence-response.{target_slug}.evaluation",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def build_target_evaluation_authoring_act(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    binding: SelectiveDependenceResponseEvaluationBinding,
    implementation_sha256: str,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseEvaluationAuthoringAct:
    """Compile one exact sealed-evaluation graph without issuing or executing it."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if binding.target_id != projection.binding.target_id or (
        binding.target_slug != projection.target_slug
    ):
        raise ValueError("evaluation authoring projection crosses targets")
    if (
        binding.design != projection.binding.design
        or binding.analysis_freeze != projection.binding.analysis_freeze
        or binding.construct_review != projection.binding.construct_review
    ):
        raise ValueError("evaluation authoring target lineage differs")
    if (
        binding.evaluation_package.evaluation_implementation_sha256 != implementation_sha256
        or binding.development_completion.development_implementation_sha256 != implementation_sha256
        or binding.study_completion.study_implementation_sha256 != implementation_sha256
    ):
        raise ValueError("evaluation authoring implementation identity differs")
    stage = compose_target_evaluation(
        binding,
        implementation_sha256=implementation_sha256,
    )
    system = build_target_system(projection)
    experiment = build_target_experiment(projection, system)
    campaign = build_target_campaign(projection, system, experiment)
    template = _complete_coverage(
        stage.catalog.templates[0],
        experiment,
        target_slug=binding.target_slug,
    )
    catalog = CandidateCapabilityCatalog(
        catalog_id=f"selective-dependence-response-{binding.target_slug}-evaluation-authoring-catalog",
        registrations=stage.catalog.registrations,
        templates=(template,),
    )
    source_configs, source_records, qualifications, source_refs = _sources(
        binding=binding,
        stage=stage,
        template=template,
        system=system,
        experiment=experiment,
    )
    design_inputs = _design_inputs(
        projection=projection,
        binding=binding,
        cutoff=experiment.information_cutoffs[0],
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=f"formal-source-inventory.selective-dependence-response.{binding.target_slug}.evaluation",
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in source_refs),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=binding.evaluation_package.evaluation_complete_unit_ids,
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        available_estimator_family_ids=(),
        available_control_ids=tuple(value.control_id for value in experiment.controls),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=tuple(value.gap_id for value in register.gaps),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    draft = StudyDraft(
        draft_id=f"draft.selective-dependence-response.{binding.target_slug}.evaluation",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question="; ".join(value.proposition for value in experiment.claims),
        alternative_ids=(
            f"alternative.selective-dependence-response.{binding.target_slug}.mixed",
            f"alternative.selective-dependence-response.{binding.target_slug}.not-supported",
            f"alternative.selective-dependence-response.{binding.target_slug}.supported",
            f"alternative.selective-dependence-response.{binding.target_slug}.unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id=f"origin.selective-dependence-response.{binding.target_slug}.evaluation-visible-parent",
            kind=DesignOriginKind.PROSPECTIVE_NOMINATION,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            nomination=ObjectIdentity.from_record(
                projection.binding.contamination_ledger.ledger_id,
                projection.binding.contamination_ledger,
            ),
            requests_fresh_child=True,
        ),
        design_inputs=design_inputs,
        development_unit_ids=experiment.reveal_barrier.development_unit_ids,
        evaluation_unit_ids=binding.evaluation_package.evaluation_complete_unit_ids,
        development_seed_ids=tuple(
            f"seed.{value}" for value in experiment.reveal_barrier.development_unit_ids
        ),
        evaluation_seed_ids=tuple(
            f"seed.{value}" for value in binding.evaluation_package.evaluation_complete_unit_ids
        ),
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
            for value in stage.registry.capabilities
        ),
        source_materializations=source_refs,
        resource_ceiling=system.authority_policy.budget_ceiling,
    )
    entry = _entry_package(
        draft=draft,
        register=register,
        inventory=inventory,
        target_slug=binding.target_slug,
    )
    package = StudyDefinition(
        package_id=(f"programme-authoring-package.selective-dependence-response.{binding.target_slug}.evaluation"),
        draft=draft,
        entry_package=entry,
    )
    base = CandidateCompilationContext(
        context_id=f"candidate-context.selective-dependence-response.{binding.target_slug}.evaluation",
        registry=stage.registry,
        templates=(template,),
        qualifications=qualifications,
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id=f"standard-context.selective-dependence-response.{binding.target_slug}.evaluation",
        base=base,
        formal_methods=standard_formal_method_catalog(register),
        source_inventories=(inventory,),
    )
    payload = package.canonical_bytes()
    compilation = compile_study_candidate(
        authoring_package=package,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(payload),
            raw_materialization_sha256=sha256(b"application/json\x00" + payload).hexdigest(),
        ),
        context=context,
    )
    return SelectiveDependenceResponseEvaluationAuthoringAct(
        projection=projection,
        binding=binding,
        stage_composition=stage,
        system=system,
        experiment=experiment,
        campaign=campaign,
        source_configs=source_configs,
        source_records=source_records,
        qualifications=qualifications,
        design_inputs=design_inputs,
        inventory=inventory,
        template=template,
        catalog=catalog,
        draft=draft,
        package=package,
        context=context,
        compilation=compilation,
    )


__all__ = [
    'SelectiveDependenceResponseEvaluationAuthoringAct',
    "build_target_evaluation_authoring_act",
]
