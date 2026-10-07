"""Static capability binding for the selective dependence response FiPy target."""

from __future__ import annotations

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseInspectedPilotInventory

from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponsePhase, SelectiveDependenceResponseTargetPanel
from empirical_lawhood.adapters.methods.selective_dependence_response.forecast import SelectiveDependenceResponseDevelopmentBundle, SelectiveDependenceResponseDevelopmentLineage
from empirical_lawhood.adapters.methods.selective_dependence_response.development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope
from empirical_lawhood.adapters.methods.selective_dependence_response.evaluation import SelectiveDependenceResponseEvaluationPackage
from empirical_lawhood.adapters.methods.selective_dependence_response.evaluation_protocol import SelectiveDependenceResponseEvaluationBinding
from empirical_lawhood.adapters.methods.selective_dependence_response.method import contamination_ledger, method_question_freeze
from empirical_lawhood.adapters.methods.selective_dependence_response.method_completion import SelectiveDependenceResponseMethodCompletionEnvelope
from empirical_lawhood.adapters.methods.selective_dependence_response.study_completion import SelectiveDependenceResponseStudyForecastCompletionEnvelope
from empirical_lawhood.adapters.methods.selective_dependence_response.target_protocol import SelectiveDependenceResponseTargetBinding
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.study_issue import StudyOperationAuthority

from .contracts import FipyReactionDiffusionResponseFiPyDesign, FipyReactionDiffusionResponseFiPySourceQualification
from .design import fipy_analysis_freeze, fipy_design, fipy_preparation_freeze
from .development import freeze_fipy_development
from .evaluation import evaluate_fipy_target
from .preparation_inputs import fipy_preparation_input
from .runtime import execute_fipy_complete_unit, qualify_fipy_source, verify_fipy_source_binding


def _qualify(
    design: CanonicalRecord,
) -> tuple[CanonicalRecord, SelectiveDependenceResponseTargetPanel]:
    if not isinstance(design, FipyReactionDiffusionResponseFiPyDesign):
        raise ValueError("FiPy binding received another design")
    return qualify_fipy_source(
        design,
        canary_unit_ids=fipy_preparation_freeze().canary_unit_ids,
    )


def _execute(
    design: CanonicalRecord, complete_unit_id: str, phase: SelectiveDependenceResponsePhase
) -> SelectiveDependenceResponseCompleteUnitResult:
    if not isinstance(design, FipyReactionDiffusionResponseFiPyDesign):
        raise ValueError("FiPy binding received another design")
    return execute_fipy_complete_unit(design, complete_unit_id=complete_unit_id, phase=phase, preparation_input=fipy_preparation_input(design, complete_unit_id, phase))


def fipy_target_binding(
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    *,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope | None = None,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetBinding:
    contamination = contamination_ledger(inspected_pilot_inventory=inspected_pilot_inventory, inspected_pilot_export=inspected_pilot_export)
    design = fipy_design()
    preparation = fipy_preparation_freeze()
    question = method_question_freeze()
    analysis_freeze = fipy_analysis_freeze()

    def analyze(
        panel: SelectiveDependenceResponseTargetPanel, lineage: SelectiveDependenceResponseDevelopmentLineage
    ) -> SelectiveDependenceResponseDevelopmentBundle:
        (
            parent,
            forecast,
            law,
            signature,
            comparators,
            power,
            separation,
            challenge,
            denominator,
        ) = freeze_fipy_development(
            panel=panel,
            design=design,
            preparation=preparation,
            question=question,
            contamination=contamination,
            construct_review=construct_review,
            analysis_freeze=analysis_freeze,
            lineage=lineage,
        )
        return SelectiveDependenceResponseDevelopmentBundle(
            bundle_id="bundle.fipy.development",
            target_id=design.target_id,
            parent=parent,
            law=law,
            signature=signature,
            comparators=comparators,
            power=power,
            separation=separation,
            challenge=challenge,
            denominator=denominator,
            forecast=forecast,
            evaluation_eligible=parent.eligible_for_evaluation and forecast is not None,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )

    return SelectiveDependenceResponseTargetBinding(
        target_id=design.target_id,
        target_slug="fipy",
        provider_key="selective-dependence-response.fipy-development-provider",
        design=design,
        design_id=design.design_id,
        design_type=FipyReactionDiffusionResponseFiPyDesign,
        preparation=preparation,
        analysis_freeze=analysis_freeze,
        method_question=question,
        contamination_ledger=contamination,
        construct_review=construct_review,
        method_completion=method_completion,
        source_qualification_type=FipyReactionDiffusionResponseFiPySourceQualification,
        qualify_source=_qualify,
        execute_unit=_execute,
        analyze_development=analyze,
    )


def fipy_evaluation_binding(
    *,
    development_completion: SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    development: SelectiveDependenceResponseDevelopmentBundle,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    source_qualification: FipyReactionDiffusionResponseFiPySourceQualification,
    evaluation_package: SelectiveDependenceResponseEvaluationPackage,
    study_completion: SelectiveDependenceResponseStudyForecastCompletionEnvelope,
    execution_authority: StudyOperationAuthority,
    reveal_authority: StudyOperationAuthority,
) -> SelectiveDependenceResponseEvaluationBinding:
    design = fipy_design()
    analysis_freeze = fipy_analysis_freeze()

    def verify_source(design_value: CanonicalRecord, source_value: CanonicalRecord) -> None:
        if not isinstance(design_value, FipyReactionDiffusionResponseFiPyDesign) or not isinstance(
            source_value, FipyReactionDiffusionResponseFiPySourceQualification
        ):
            raise ValueError("FiPy evaluation source binding type differs")
        verify_fipy_source_binding(design_value, source_value)

    return SelectiveDependenceResponseEvaluationBinding(
        target_id=design.target_id,
        target_slug="fipy",
        provider_key="selective-dependence-response.fipy-evaluation-provider",
        design=design,
        design_type=FipyReactionDiffusionResponseFiPyDesign,
        analysis_freeze=analysis_freeze,
        development_completion=development_completion,
        development=development,
        construct_review=construct_review,
        source_qualification=source_qualification,
        source_qualification_type=FipyReactionDiffusionResponseFiPySourceQualification,
        evaluation_package=evaluation_package,
        study_completion=study_completion,
        execution_authority=execution_authority,
        reveal_authority=reveal_authority,
        verify_source_binding=verify_source,
        execute_unit=_execute,
        evaluate_target=evaluate_fipy_target,
    )


__all__ = ["fipy_evaluation_binding", "fipy_target_binding"]
