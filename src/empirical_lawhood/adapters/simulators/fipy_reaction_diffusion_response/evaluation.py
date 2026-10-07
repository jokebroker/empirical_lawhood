"""Frozen target binding for selective dependence response FiPy evaluation."""

from __future__ import annotations

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis_design import SelectiveDependenceResponseTargetAnalysisFreeze
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponseTargetPanel
from empirical_lawhood.adapters.methods.selective_dependence_response.evaluation import SelectiveDependenceResponseEvaluationPackage, SelectiveDependenceResponseSealedIndex, SelectiveDependenceResponseTargetEvaluationBundle, evaluate_target
from empirical_lawhood.adapters.methods.selective_dependence_response.forecast import SelectiveDependenceResponseDevelopmentBundle
from empirical_lawhood.kernel.provenance import ObjectIdentity


def evaluate_fipy_target(
    *,
    package: SelectiveDependenceResponseEvaluationPackage,
    development: SelectiveDependenceResponseDevelopmentBundle,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    sealed_index: SelectiveDependenceResponseSealedIndex,
    panel: SelectiveDependenceResponseTargetPanel,
    reveal_authority: ObjectIdentity,
    evaluator_task_id: str,
    evaluator_attempt_id: str,
    input_receipt_ids: tuple[str, ...],
    input_materialization_ids: tuple[str, ...],
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> SelectiveDependenceResponseTargetEvaluationBundle:
    return evaluate_target(
        package=package,
        development=development,
        construct_review=construct_review,
        sealed_index=sealed_index,
        panel=panel,
        reveal_authority=reveal_authority,
        evaluator_task_id=evaluator_task_id,
        evaluator_attempt_id=evaluator_attempt_id,
        input_receipt_ids=input_receipt_ids,
        input_materialization_ids=input_materialization_ids,
        analysis_freeze=analysis_freeze,
        source_family_id="fipy",
        solver_family_id="fipy-scipy-linear-lu",
    )


__all__ = ["evaluate_fipy_target"]
