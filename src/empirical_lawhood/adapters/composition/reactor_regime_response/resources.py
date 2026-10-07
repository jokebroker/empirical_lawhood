"Bounded B/C/D task reservations under the whole regime-response study ceiling."

from __future__ import annotations

from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.adapters.methods.reactor_regime_response.config import PREFIX, ROOTS, ReactorRegimeResponseDesign
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import RegimeCContinuation


def phase_cpu_reservations(phase: str) -> tuple[tuple[str, int], ...]:
    if phase == "D":
        roots = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
        reservations = [
            (f"regime.{kind}.{root}", seconds)
            for root in roots
            for kind, seconds in (
                ("prepare", 60), ("seal", 30), ("assignment", 15),
                ("action", 360), ("d-seal", 15), ("d-reveal", 15),
            )
        ]
        reservations.extend((
            ("regime.d-plan", 300), ("regime.d-cohort", 1200),
            ("regime.d-adjudication", 300),
        ))
        return tuple(sorted(reservations))
    if phase not in ("B", "C"):
        raise ValueError("only B/C/D phase reservations are declared")
    roles = ("fit", "nomination") if phase == "B" else ("calibration", "qualification")
    roots = tuple(root for root, role, _, _ in ROOTS if role in roles)
    budgets = [
        (f"regime.{kind}.{root}", seconds)
        for root in roots
        for kind, seconds in (("prepare", 120), ("seal", 60), ("assay", 360))
    ]
    budgets.extend(
        (("regime.fit", 2400), ("regime.nomination", 5400),
         ("regime.adjudication", 300)) if phase == "B"
        else (
            ("regime.calibration", 2400),
            ("regime.qualification", 3600),
            ("regime.law-qualification", 2400),
            ("regime.adjudication", 300),
        )
    )
    return tuple(sorted(budgets))


def phase_resource_envelope(
    phase: str,
    design: ReactorRegimeResponseDesign,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
    continuation: RegimeCContinuation | None = None,
) -> ExecutionResourceEnvelopeSpec:
    reservations = dict(phase_cpu_reservations(phase))
    if continuation is not None:
        if phase != "C":
            raise ValueError("only C may use retained-task resource credits")
        reservations = {key: value for key, value in reservations.items()
                        if key not in continuation.completed_task_ids}
    if {step.step_id for step in protocol.steps} != set(reservations):
        raise ValueError("phase resource envelope lacks a full task census")
    if (
        sum(reservations.values()) > design.acquisition_cpu_seconds
        or sum(value.startswith("regime.assay.") for value in reservations) !=
        (48 if phase == "B" else 64 if continuation is not None else 96 if phase == "C" else 0)
    ):
        raise ValueError("phase CPU or root reservation exceeds frozen design")
    cells = []
    for step in protocol.steps:
        root = step.step_id.removeprefix("regime.prepare.") if step.step_id.startswith("regime.prepare.") else None
        if root is None and step.step_id.startswith("regime.assay."):
            root = step.step_id.removeprefix("regime.assay.")
        if root is None and step.step_id.startswith("regime.action."):
            root = step.step_id.removeprefix("regime.action.")
        native = root is not None
        budget = step.resource_budget
        cells.append(ExecutionResourceTaskCellSpec(
            f"cell.{step.step_id}", step.step_id, PREFIX,
            root, None if root is None else f"{root}.initial-state",
            1, int(native), int(native), native,
            NonTimeResourceBudget(
                f"budget.{step.step_id}", budget.cpu_cores,
                budget.memory_bytes, budget.output_bytes, 0, budget.source_scan_bytes,
            ),
            None if not native else ProgressLivenessContract(
                f"progress.{step.step_id}", ProgressHeartbeat.SCHEMA,
                "reactor-completed-native-decisions", D(60), True, True,
            ),
            None, None,
        ))
    return ExecutionResourceEnvelopeSpec(
        f"{PREFIX}.phase-{phase.lower()}.resources", issued_extensions,
        tuple(sorted(cells, key=lambda value: value.cell_id)),
        (ChildResourceTokenLimit(PREFIX, sum(step.step_id.startswith(("regime.prepare.", "regime.assay.", "regime.action.")) for step in protocol.steps), 0),),
        (), (),
        tuple(sorted({
            "NATIVE_NUMERICAL_FAILURE", "NATIVE_OBSERVER_FAILURE",
            "WORKER_PROGRESS_STALLED", "RESOURCE_COMPUTABILITY_UNAVAILABLE",
            "RETRY_BUDGET_EXHAUSTED",
            *(value.value for value in OperationalFailureClass),
        })),
        None, None, 4, 32 * 1024**3, (), (),
        sha256(canonical_json_bytes({
            "phase": phase, "protocol": protocol.fingerprint(),
            "retained_c_continuation": None if continuation is None else continuation.fingerprint(),
            "cpu_reservations": tuple(sorted(reservations.items())),
            "native_call_ceiling": (
                design.phase_B_native_call_cap if phase == "B" else
                design.phase_C_native_call_cap if phase == "C" else
                design.phase_D_native_call_cap
            ),
            "whole_study_cpu_ceiling": design.acquisition_cpu_seconds,
            "whole_study_wall_ceiling": design.acquisition_wall_seconds,
        })).hexdigest(),
        True, True, True, True,
    )
