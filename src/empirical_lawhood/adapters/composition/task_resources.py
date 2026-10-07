"""Shared progress-based reservations; scientific task assignments remain adapter-owned."""

from __future__ import annotations

from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate


def progress_resource_envelope(
    *,
    prefix: str,
    design: CanonicalRecord,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
    cpu_seconds: int,
    native_preparations: tuple[tuple[str, str, str], ...],
    progress_counter: str,
    heartbeat_seconds: D,
    maximum_concurrency: int,
    maximum_memory_bytes: int,
    terminal_failure_codes: tuple[str, ...],
) -> ExecutionResourceEnvelopeSpec:
    """Assemble existing resource owners from an explicit effect-bearing census."""
    if sum(s.resource_budget.wall_time_seconds for s in protocol.steps) > cpu_seconds:
        raise ValueError("task CPU reservations exceed the frozen ceiling")
    native_by_task = {task: (unit, preparation) for task, unit, preparation in native_preparations}
    if len(native_by_task) != len(native_preparations) or not set(native_by_task) <= {
        s.step_id for s in protocol.steps
    }:
        raise ValueError("native resource census contains duplicate or absent tasks")
    cells = []
    for step in protocol.steps:
        assignment = native_by_task.get(step.step_id)
        native = assignment is not None
        root, preparation = (None, None) if assignment is None else assignment
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                prefix,
                root,
                preparation,
                1,
                int(native),
                int(native),
                native,
                NonTimeResourceBudget(
                    f"budget.{step.step_id}",
                    budget.cpu_cores,
                    budget.memory_bytes,
                    budget.output_bytes,
                    0,
                    budget.source_scan_bytes,
                ),
                None
                if not native
                else ProgressLivenessContract(
                    f"progress.{step.step_id}",
                    ProgressHeartbeat.SCHEMA,
                    progress_counter,
                    heartbeat_seconds,
                    True,
                    True,
                ),
                None,
                None,
            )
        )
    return ExecutionResourceEnvelopeSpec(
        f"{prefix}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda v: v.cell_id)),
        (ChildResourceTokenLimit(prefix, len(native_by_task), 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    *terminal_failure_codes,
                    *(v.value for v in OperationalFailureClass),
                }
            )
        ),
        None,
        None,
        maximum_concurrency,
        maximum_memory_bytes,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {"protocol": protocol.fingerprint(), "design": design.fingerprint()}
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
