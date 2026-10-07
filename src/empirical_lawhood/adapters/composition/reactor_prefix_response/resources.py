"""Finite native costs: five units, four branches per unit, no retries or JIT."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.methods.reactor_prefix_response.design import PREFIX
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import BRANCHES, native_task_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificStage

from .authoring import BUDGET, SCIENCE_TASK


def reactor_resource_envelope(
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
    *,
    run_id: str = PREFIX,
    native_config=None,
) -> ExecutionResourceEnvelopeSpec:
    branches = BRANCHES if native_config is None else native_config.branches
    source_tasks = tuple(
        native_task_id(*branch, config=native_config) for branch in branches
    )
    expected = {SCIENCE_TASK, *source_tasks}
    if {s.step_id for s in protocol.steps} != expected or len(protocol.steps) != 21:
        raise ValueError(
            "reactor resource envelope requires exactly twenty branches and one owner chain"
        )
    by_task = {native_task_id(*b, config=native_config): b[0] for b in branches}
    cells = []
    for step in protocol.steps:
        scenario = by_task.get(step.step_id)
        native = scenario is not None
        if (
            step.maximum_attempts != 1
            or (step.stage is ScientificStage.PREPARE) != native
        ):
            raise ValueError("reactor native stage or retry policy differs")
        unit = None if scenario is None else f"unit.{scenario.replace('_', '-')}"
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                run_id,
                unit,
                None if unit is None else f"{unit}.initial-state",
                1,
                1 if native else 0,
                1 if native else 0,
                native,
                NonTimeResourceBudget(
                    f"budget.{step.step_id}",
                    budget.cpu_cores,
                    budget.memory_bytes,
                    budget.output_bytes,
                    0,
                    budget.source_scan_bytes,
                ),
                ProgressLivenessContract(
                    f"progress.{step.step_id}",
                    ProgressHeartbeat.SCHEMA,
                    "reactor-completed-rk4-updates",
                    Decimal(60),
                    True,
                    True,
                )
                if native
                else None,
                None,
                None,
            )
        )
    return ExecutionResourceEnvelopeSpec(
        f"{run_id}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(run_id, 20, 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    "NATIVE_NUMERICAL_FAILURE",
                    "NATIVE_OBSERVER_FAILURE",
                    "WORKER_PROGRESS_STALLED",
                    *(r.value for r in OperationalFailureClass),
                }
            )
        ),
        None,
        None,
        BUDGET.cpu_cores,
        BUDGET.memory_bytes,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "protocol": protocol.fingerprint(),
                    "native_branches": 20,
                    "native_updates": 600,
                    "decisions": 40,
                    "independent_units": 5,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
