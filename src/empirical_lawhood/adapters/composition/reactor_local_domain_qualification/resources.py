"""Closed task census, per-worker CPU reservations and production resource envelope."""

from decimal import Decimal
from hashlib import sha256
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import LocalQualificationDesign, PREFIX, SOURCE_TASKS

TASK_CPU_SECONDS = tuple(
    sorted(
        (
            *((task, 1800) for task in SOURCE_TASKS),
            ("local.calibration", 1200),
            ("local.qualification", 7200),
            ("local.adjudication", 300),
        )
    )
)
NATIVE_BATCH_RESERVATION = 64 * (3 + 93) * 2


def allocation_fits(recipe: LocalQualificationDesign) -> bool:
    return (
        sum(s for _, s in TASK_CPU_SECONDS) <= recipe.cpu_seconds
        and NATIVE_BATCH_RESERVATION <= recipe.native_call_cap
    )


def local_resource_envelope(
    protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    if {s.step_id for s in protocol.steps} != dict(TASK_CPU_SECONDS).keys() or len(
        protocol.steps
    ) != 67:
        raise ValueError("local resource envelope requires the complete 67-task census")
    native = {task: task.removeprefix("local.") for task in SOURCE_TASKS}
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
        (ChildResourceTokenLimit(PREFIX, len(SOURCE_TASKS), 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    "NATIVE_NUMERICAL_FAILURE",
                    "NATIVE_OBSERVER_FAILURE",
                    "WORKER_PROGRESS_STALLED",
                    "RESOURCE_COMPUTABILITY_UNAVAILABLE",
                    "RETRY_BUDGET_EXHAUSTED",
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
                    grouped_native_tasks=len(SOURCE_TASKS),
                    native_batch_reservation=NATIVE_BATCH_RESERVATION,
                    task_cpu_seconds=TASK_CPU_SECONDS,
                    scientific_batch_cap=12288,
                    all_views_nested_within_preassigned_root=True,
                )
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
