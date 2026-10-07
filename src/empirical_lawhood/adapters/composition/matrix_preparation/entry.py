"""Bounded development entry declarations using the shared formal-gap owners.

The declared operand population is the enclosing draft's exposed development
roster. No evaluation physical-unit roster is invented to satisfy entry.
"""

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
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
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.adapters.methods.matrix_preparation.closeout import PreparationDevelopmentResult
from empirical_lawhood.adapters.methods.matrix_preparation.records import PreparationTaskAssessmentReport
from empirical_lawhood.adapters.methods.matrix_preparation.extension_bundle import METHOD_CAPABILITIES
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT


def preparation_entry(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    science: ObjectIdentity,
    *,
    prefix: str = DEVELOPMENT,
    minimum_units: int = 64,
    operand_description: str = "Finite preparation {domain} operands on 128 exposed roots: native secant, observable action chart, causal feature support and five-readout dynamics; no prospective support.",
    estimator: str = "fixed-development-chain",
    uncertainty: str = "whole-root-development-uncertainty",
    ceiling: EvidenceCeiling = EvidenceCeiling.LOCAL_LAW,
    evaluator: CapabilityManifest = METHOD_CAPABILITIES[-1],
    input_schema: str = PreparationTaskAssessmentReport.SCHEMA,
    output_schema: str = PreparationDevelopmentResult.SCHEMA,
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    assert draft.system is not None and draft.experiment is not None
    world = FormalGapEvidenceWorld.NUMERICAL_SIMULATOR
    gaps = tuple(
        FormalGapSpec(
            f"{prefix}.gap.{domain.value.lower()}",
            domain,
            operand_description.format(domain=domain.value.lower()),
            (f"{prefix}.native-operands",),
            (world,),
            minimum_units,
            2,
            (f"{prefix}.{estimator}",),
            (draft.experiment.controls[0].control_id,),
            (f"{prefix}.falsifier",),
            (f"{prefix}.fixed-roster",),
            f"{prefix}.{uncertainty}",
            ceiling,
            (science.object_id,),
        )
        for domain in FormalDomain
    )
    register = FormalGapRegister(
        f"{prefix}.formal-register",
        "1.0.0",
        (science,),
        tuple(sorted(gaps, key=lambda value: value.gap_id)),
    )
    inventory = FormalGapSourceCapabilityInventory(
        f"{prefix}.source-inventory",
        draft.system.system_id,
        world,
        tuple(
            sorted(
                (value.materialization for value in draft.source_materializations),
                key=lambda v: v.object_id,
            )
        ),
        (f"{prefix}.native-operands",),
        (f"{prefix}.fixed-roster",),
        draft.development_unit_ids,
        EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        tuple(value.view_id for value in draft.system.numerical_views),
        (f"{prefix}.{estimator}",),
        (draft.experiment.controls[0].control_id,),
        (f"{prefix}.{uncertainty}",),
        (),
        (),
        ceiling,
        OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    template = context.template(draft.dag_template_key)
    assert template is not None
    terminal = next(
        value
        for value in template.coverage.bindings
        if value.obligation_id == f"{prefix}.single-terminal"
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap.gap_id,
            FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
            None,
            (),
            f"{prefix}.{estimator}",
            (draft.experiment.controls[0].control_id,),
            f"{prefix}.{uncertainty}",
            (terminal.obligation_id,),
            (terminal.required_output_id,),
            (terminal.proof_owner_node_id,),
        )
        for gap in gaps
    )
    coverage = FormalGapCoverage(
        f"{prefix}.formal-coverage",
        ObjectIdentity.from_record(register.register_id, register),
        draft.system.system_id,
        draft.draft_id,
        applicability,
        tuple(sorted(assignments, key=lambda value: value.gap_id)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    requirements = tuple(
        ExperimentEntryRequirementBinding(
            requirement,
            (f"{prefix}.entry.{requirement.value.lower()}",),
            ReadinessStatus.AUTHORITY_REQUIRED
            if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else ReadinessStatus.READY,
            ("EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
            if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else (),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        f"{prefix}.entry-checklist",
        ObjectIdentity.from_record(draft.draft_id, draft),
        ObjectIdentity.from_record(register.register_id, register),
        ObjectIdentity.from_record(coverage.coverage_id, coverage),
        world,
        f"{prefix}.public-entry",
        f"{prefix}.receipt-first-recovery",
        requirements,
        tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        tuple(sorted(ExperimentTerminalClass, key=lambda value: value.value)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    authoring = StudyDefinition(
        f"{prefix}.authoring-base",
        draft,
        ExperimentEntryPackage(f"{prefix}.entry", register, coverage, checklist),
    )
    methods = tuple(
        FormalMethodBinding(
            key,
            role,
            evaluator.capability_key,
            evaluator.capability_version,
            evaluator.implementation_sha256,
            tuple(sorted(gap.gap_id for gap in gaps)),
            input_schema,
            output_schema,
            ceiling,
            OutcomeAccess.EVALUATOR_REVEAL,
        )
        for role, key in (
            (FormalMethodRole.ESTIMATOR, f"{prefix}.{estimator}"),
            (FormalMethodRole.MULTIPLICITY, f"{prefix}.{uncertainty}"),
        )
    )
    return authoring, StandardCandidateCompilationContext(
        f"{prefix}.standard-context",
        context,
        FormalMethodCatalog(
            f"{prefix}.formal-method-catalog",
            tuple(sorted(methods, key=lambda value: value.binding_id)),
        ),
        (inventory,),
    )
