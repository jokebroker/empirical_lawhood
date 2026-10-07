"""Preparation's exact task resource envelope, with no elapsed-time scientific stop."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.native_source import NativeLawQualificationExperiment
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificStage
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, PreparationSourceConfig
from empirical_lawhood.adapters.methods.matrix_preparation.topology import preparation_task_declarations
from empirical_lawhood.adapters.simulators.matrix_preparation.continuation import preparation_continuation
from .authoring import PreparationAuthoringBundle
from .science import DEVELOPMENT_BUDGET


def build_preparation_resource_envelope(
    bundle: PreparationAuthoringBundle,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    source = next(r for r in bundle.payloads if isinstance(r, PreparationSourceConfig))
    extension = next(r for r in bundle.payloads if isinstance(r, NativeLawQualificationExperiment))
    continuation = preparation_continuation(bundle.payloads)
    roots = source.roots if continuation is None else continuation.missing_roots
    native = {f"{r.root_id}.native": r for r in roots}
    retained = set() if continuation is None else set(continuation.retained_task_ids)
    run_id = DEVELOPMENT if continuation is None else continuation.run_id
    projections = {f"{r.root_id}.project.r{v}": r for r in source.roots for v in (1, 2)}
    if len(protocol.steps) != 397 - len(retained) or {s.step_id for s in protocol.steps} != {
        d.task_id for d in preparation_task_declarations() if d.task_id not in retained
    }:
        raise ValueError("preparation resources require the exact complete production task roster")
    cells = []
    for step in protocol.steps:
        segment = native.get(step.step_id)
        root = segment if segment else projections.get(step.step_id)
        unit = None if root is None else root.physical_unit_id
        is_native = segment is not None
        if step.maximum_attempts != 1 or (step.stage is ScientificStage.PREPARE) != is_native:
            raise ValueError(
                "preparation resource identity changes its native effect or attempt policy"
            )
        b = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                run_id,
                unit,
                None if unit is None else f"{unit}.instance",
                1,
                int(is_native),
                int(is_native),
                is_native,
                NonTimeResourceBudget(
                    f"budget.{step.step_id}",
                    b.cpu_cores,
                    b.memory_bytes,
                    b.output_bytes,
                    0,
                    b.source_scan_bytes,
                ),
                ProgressLivenessContract(
                    f"progress.{step.step_id}",
                    ProgressHeartbeat.SCHEMA,
                    f"{DEVELOPMENT}.completed-native-updates",
                    Decimal(600),
                    True,
                    True,
                )
                if is_native
                else None,
                None,
                None,
            )
        )
    return ExecutionResourceEnvelopeSpec(
        f"{run_id}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(run_id, len(native), 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    "NATIVE_NUMERICAL_FAILURE",
                    "NATIVE_OBSERVER_FAILURE",
                    "RESOURCE_COMPUTABILITY_UNAVAILABLE",
                    "WORKER_PROGRESS_STALLED",
                    *(r.value for r in OperationalFailureClass),
                }
            )
        ),
        None,
        None,
        DEVELOPMENT_BUDGET.cpu_cores,
        DEVELOPMENT_BUDGET.memory_bytes,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "carrier": extension.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": len(native),
                    "retry_tokens": 0,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
