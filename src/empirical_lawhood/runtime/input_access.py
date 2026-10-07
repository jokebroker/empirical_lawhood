"""Identical planned/verified access for completed private model dependencies.

This does not grant reveal authority or downgrade inherited outcome access.
The execution owner authenticates dependency receipts before opening inputs.
Only an explicitly model-labelled, prospective internal edge can exercise the
separate frozen-model permission; raw outcomes and external slots cannot.
"""

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from .capabilities import input_access_allowed
from .plans import ProtocolExecutionTask, ExecutionTask, ScientificInputRole


def planned_input_access_allowed(
    task: ProtocolExecutionTask,
    artifact_id: str,
    access: OutcomeAccess,
    visibility: VisibilityCeiling | None,
    *,
    reveal_barrier_authorized: bool = False,
) -> bool:
    model_edges = (
        tuple(
            edge
            for edge in task.scientific_inputs
            if edge.operational_logical_artifact_id == artifact_id
        )
        if isinstance(task, ExecutionTask)
        else ()
    )
    frozen_model = (
        visibility is VisibilityCeiling.PROSPECTIVE
        and bool(model_edges)
        and all(
            edge.scientific_role is ScientificInputRole.MODEL
            and edge.producer_task_id in task.dependency_task_ids
            and edge.external_input_id is None
            and edge.outcome_access is OutcomeAccess.EVALUATOR_REVEAL
            and edge.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
            for edge in model_edges
        )
    )
    return input_access_allowed(
        task.capability,
        access,
        reveal_barrier_authorized=reveal_barrier_authorized,
        frozen_model_authorized=frozen_model,
    )
