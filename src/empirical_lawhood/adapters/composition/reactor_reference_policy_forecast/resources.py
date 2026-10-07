"""Exact full-batch costs: 42 units, two nested views, no retries or JIT."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BRANCHES, native_task_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificStage
from empirical_lawhood.adapters.methods.reactor_reference_policy_forecast.science import BUDGET, PREFIX
from .authoring import SCIENCE_TASK, SOURCE_TASKS, TERMINAL_TASK


def reactor_resource_envelope(
    protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    expected = {SCIENCE_TASK, TERMINAL_TASK, *SOURCE_TASKS}
    if {s.step_id for s in protocol.steps} != expected or len(protocol.steps) != 86:
        raise ValueError(
            "reactor resource envelope requires 84 full batches, qualification and separate reveal"
        )
    by_task = {native_task_id(*b): b[0] for b in BRANCHES}
    cells = []
    for step in protocol.steps:
        scenario = by_task.get(step.step_id)
        native = scenario is not None
        if step.maximum_attempts != 1 or (step.stage is ScientificStage.PREPARE) != native:
            raise ValueError("reactor native stage or retry policy differs")
        unit = scenario
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                PREFIX,
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
                    "reactor-completed-native-decisions",
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
        f"{PREFIX}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(PREFIX, 84, 0),),
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
                    "native_branches": 84,
                    "native_updates": 3628800,
                    "decisions": 241920,
                    "independent_units": 42,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
