# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Small truth-known experiment-entry fixtures shared by issue/rehearsal tests."""

from __future__ import annotations

from dataclasses import replace

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope, WorldKind
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
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
from empirical_lawhood.planning.study_authoring import StudyDraft
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateCompilationContext,
    StandardCandidateCompilationContext,
)


def _portfolio_world(draft: StudyDraft) -> FormalGapEvidenceWorld:
    assert draft.system is not None
    return {
        WorldKind.ANALYTIC_REFERENCE: FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        WorldKind.NUMERICAL_SIMULATOR: FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        WorldKind.PHYSICAL_EXPERIMENT: FormalGapEvidenceWorld.RETROSPECTIVE_DATASET,
    }[draft.system.world.kind]


def entry_package_for_draft(draft: StudyDraft) -> ExperimentEntryPackage:
    """Return a complete non-promotable fixture with denominator-inapplicable gaps."""

    source = ObjectIdentity(
        object_id="source.formal-gap-fixture",
        object_schema='empirical-lawhood/testing/fixtures/formal-gap-source',
        object_version="1.0.0",
        object_fingerprint="a" * 64,
    )
    world = _portfolio_world(draft)
    gaps = tuple(
        FormalGapSpec(
            gap_id=f"gap.{domain.value.lower()}.fixture",
            domain=domain,
            question_family=f"Truth-known {domain.value.lower()} fixture.",
            required_operand_ids=("operand.fixture",),
            compatible_evidence_worlds=(world,),
            minimum_independent_units=1,
            minimum_numerical_views=0,
            estimator_family_ids=("estimator.fixture",),
            control_ids=("control.fixture",),
            decisive_falsifier_ids=("falsifier.fixture",),
            support_prerequisite_ids=("support.fixture",),
            multiplicity_family_id="multiplicity.fixture",
            maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
            provenance_source_ids=(source.object_id,),
        )
        for domain in FormalDomain
    )
    register = FormalGapRegister(
        register_id="formal-gap-register.fixture",
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
            reason_codes=("DENOMINATOR_OBJECT_UNDEFINED",),
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
        coverage_id="formal-gap-coverage.fixture",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id="denominator.fixture",
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.{requirement.value.lower().replace('_', '-')}",),
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
        checklist_id="experiment-entry-checklist.fixture",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=world,
        execution_route_id="route.truth-known-fixture",
        durability_disposition_id="durability.synthetic-nonpromotable",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id="experiment-entry-package.fixture",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def standard_authoring_for_draft(
    draft: StudyDraft,
    base_context: CandidateCompilationContext,
    *,
    formal_method_capability_key: str = "reference.evaluate",
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    """Wrap one truth-known draft in the mandatory standard authoring contract."""

    assert draft.system is not None
    assert draft.experiment is not None
    entry = entry_package_for_draft(draft)
    template = base_context.template(draft.dag_template_key)
    assert template is not None
    graph_binding = template.coverage.bindings[0]
    evaluator = base_context.registry.resolve(formal_method_capability_key, "1.0.0")
    control_id = draft.experiment.controls[0].control_id
    gaps = tuple(
        replace(
            value,
            estimator_family_ids=("estimator.fixture",),
            control_ids=(control_id,),
            multiplicity_family_id="multiplicity.fixture",
        )
        for value in entry.register.gaps
    )
    register = replace(entry.register, gaps=gaps)
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.reference",
        denominator_id=draft.system.system_id,
        evidence_world=entry.checklist.portfolio_world,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in draft.source_materializations),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=("operand.fixture",),
        satisfied_prerequisite_ids=("support.fixture",),
        independent_unit_ids=(draft.development_unit_ids[0],),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(),
        available_estimator_family_ids=("estimator.fixture",),
        available_control_ids=(control_id,),
        multiplicity_family_ids=("multiplicity.fixture",),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
            readiness_reason=None,
            reason_codes=(),
            selected_estimator_family_id="estimator.fixture",
            selected_control_ids=(control_id,),
            selected_multiplicity_family_id="multiplicity.fixture",
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
    entry = replace(
        entry,
        register=register,
        coverage=coverage,
        checklist=checklist,
    )
    authoring = StudyDefinition(
        package_id="programme-authoring-package.reference",
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
                    maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                )
                for role, family_id in (
                    (FormalMethodRole.ESTIMATOR, "estimator.fixture"),
                    (FormalMethodRole.MULTIPLICITY, "multiplicity.fixture"),
                )
            ),
            key=lambda value: value.binding_id,
        )
    )
    context = StandardCandidateCompilationContext(
        context_id="standard-context.reference",
        base=base_context,
        formal_methods=FormalMethodCatalog(
            catalog_id="formal-method-catalog.reference",
            bindings=bindings,
        ),
        source_inventories=(inventory,),
    )
    return authoring, context


__all__ = ["entry_package_for_draft", "standard_authoring_for_draft"]
