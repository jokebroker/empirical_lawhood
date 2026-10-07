"""Closed task census, per-worker CPU reservations and production resource envelope."""

from decimal import Decimal
from hashlib import sha256
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe, CONFIRMATION_ROOT_COUNT, NATIVE_BENCHMARK_ARMS
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_runner import ROOT_TASKS, NATIVE_BENCHMARK_TASKS
from empirical_lawhood.adapters.simulators.reactor_causal_response.config import NATIVE_TASKS
from empirical_lawhood.adapters.methods.reactor_causal_response.science import PREFIX

# CPU is reserved without refunds: no outcome-dependent allocation or rerun.
# Docker's separately bounded verifier process is included in native task costs.
TASK_CPU_SECONDS = tuple(
    sorted(
        [
            *(
                (
                    task,
                    1200
                    if role in ("fit", "nomination")
                    else 4500
                    if role == "confirmation"
                    else 300,
                )
                for task, role, _ in NATIVE_TASKS
            ),
            *((task, 3000) for task in ROOT_TASKS),
            *((task, 2000) for task in NATIVE_BENCHMARK_TASKS),
            ("empirical.discovery", 3600),
            ("empirical.calibration", 1200),
            ("empirical.qualification", 2400),
            ("empirical.contribution", 1200),
            ("empirical.adjudication", 300),
        ]
    )
)


NATIVE_BATCH_RESERVATION = (
    368 + CONFIRMATION_ROOT_COUNT * 10 + len(NATIVE_BENCHMARK_ARMS) * (5 + 52)
)


def allocation_fits(recipe: EmpiricalRecipe) -> bool:
    return (
        sum(seconds for _, seconds in TASK_CPU_SECONDS) <= recipe.aggregate_cpu_seconds
        and NATIVE_BATCH_RESERVATION <= recipe.maximum_batches
    )


def empirical_resource_envelope(
    protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    if {s.step_id for s in protocol.steps} != dict(TASK_CPU_SECONDS).keys() or len(
        protocol.steps
    ) != 114:
        raise ValueError("empirical resource envelope requires the complete 114-task census")
    native = {task: task.removeprefix("empirical.") for task, _, _ in NATIVE_TASKS}
    cells = []
    for step in protocol.steps:
        unit = native.get(step.step_id)
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                PREFIX,
                unit,
                None if unit is None else f"{unit}.initial-state",
                1,
                int(unit is not None),
                int(unit is not None),
                unit is not None,
                NonTimeResourceBudget(
                    f"budget.{step.step_id}",
                    budget.cpu_cores,
                    budget.memory_bytes,
                    budget.output_bytes,
                    0,
                    budget.source_scan_bytes,
                ),
                None
                if unit is None
                else ProgressLivenessContract(
                    f"progress.{step.step_id}",
                    ProgressHeartbeat.SCHEMA,
                    "reactor-completed-native-decisions",
                    Decimal(60),
                    True,
                    True,
                ),
                None,
                None,
            )
        )
    return ExecutionResourceEnvelopeSpec(
        f"{PREFIX}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(PREFIX, len(NATIVE_TASKS), 0),),
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
        4,
        32 * 1024**3,
        (),
        (),
        sha256(
            canonical_json_bytes(
                dict(
                    protocol=protocol.fingerprint(),
                    grouped_native_tasks=len(NATIVE_TASKS),
                    native_verifier_tasks=len(NATIVE_BENCHMARK_ARMS),
                    task_cpu_seconds=TASK_CPU_SECONDS,
                    scientific_batch_cap=EmpiricalRecipe().maximum_batches,
                    native_verifier_counts_require_separate_closure=True,
                )
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
