"""development's exact task resource envelope, with no elapsed-time scientific stop."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.native_source import NativeLawQualificationExperiment
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificStage
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID, ResponseGeometryDevelopmentNativeConfig, development_segments

from .development_authoring import ResponseGeometryDevelopmentAuthoringBundle
from .development_design import DEVELOPMENT_BUDGET


def build_response_geometry_development_resource_envelope(
    bundle: ResponseGeometryDevelopmentAuthoringBundle, protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    source = next(r for r in bundle.payloads if isinstance(r, ResponseGeometryDevelopmentNativeConfig))
    extension = next(r for r in bundle.payloads if isinstance(r, NativeLawQualificationExperiment))
    native = {s.task_id: s for s in development_segments()}
    projections = {
        f"{r.root_id}.project.r{refinement}": r for r in source.roots for refinement in (1, 2)
    }
    methods = {
        f"{DEVELOPMENT_PANEL_ID}.{role}.{context}"
        for context in ("assembling", "prepared")
        for role in ("fit", "calibrate", "support", "assess")
    }
    if len(protocol.steps) != 2953 or {s.step_id for s in protocol.steps} != (
        set(native) | set(projections) | methods | {f"{DEVELOPMENT_PANEL_ID}.evaluate"}
    ):
        raise ValueError("development resources require the exact complete production task roster")
    cells = []
    for step in protocol.steps:
        segment = native.get(step.step_id)
        root = segment.root if segment else projections.get(step.step_id)
        unit = None if root is None else source.physical_unit_id(root)
        is_native = segment is not None
        if step.maximum_attempts != 1 or (step.stage is ScientificStage.PREPARE) != is_native:
            raise ValueError("development resource identity changes its native effect or attempt policy")
        b = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                DEVELOPMENT_PANEL_ID,
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
                    f"{DEVELOPMENT_PANEL_ID}.completed-native-updates",
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
        f"{DEVELOPMENT_PANEL_ID}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(DEVELOPMENT_PANEL_ID, len(native), 0),),
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
