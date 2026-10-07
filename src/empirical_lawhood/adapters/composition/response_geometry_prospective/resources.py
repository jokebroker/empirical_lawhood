"""Exact assay resource declarations; construction grants no issue or effect authority."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import derive_source_qualification_topology

from .design import BUDGET, ResponseGeometryAssayAuthoringBundle


def build_response_geometry_assay_resource_envelope(
    bundle: ResponseGeometryAssayAuthoringBundle,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    """Bind the compiled production roster without inventing physical aggregate units."""
    carrier = next(
        record for record in bundle.payloads if isinstance(record, FreshSourceQualificationExperiment)
    )
    topology = derive_source_qualification_topology(carrier)
    expected = {stage.task_id: stage for stage in topology.stages}
    if len(protocol.steps) != 865 or {step.step_id for step in protocol.steps} != set(expected):
        raise ValueError("assay resources require the exact compiled 865-task roster")
    native = {segment.segment_id: segment for segment in carrier.segments}
    views = {view.view_id: view for view in carrier.views}
    units = {unit.physical_independent_unit_id: unit for unit in carrier.physical_units}
    child = "response-geometry-assay"
    cells = []
    for step in protocol.steps:
        stage = expected[step.step_id]
        if (
            step.maximum_attempts != 1
            or step.stage != stage.stage
            or step.capability_key != stage.owner.object_id
        ):
            raise ValueError("assay resource task changes its owner, stage or single attempt")
        source = step.step_id in native
        operand = native.get(step.step_id) or views.get(step.step_id)
        unit = None if operand is None else units[operand.physical_independent_unit_id]
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                child,
                None if unit is None else unit.physical_independent_unit_id,
                None if unit is None else unit.preparation_instance_id,
                1,
                int(source),
                int(source),
                source,
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
                    "response-geometry-assay.completed-native-updates",
                    Decimal(600),
                    True,
                    True,
                )
                if source
                else None,
                None,
                None,
            )
        )
    cells.sort(key=lambda cell: cell.cell_id)
    if sum(cell.physical_execution_cost for cell in cells) != 792:
        raise ValueError("assay resources change the native segment count")
    return ExecutionResourceEnvelopeSpec(
        f"{child}.resources",
        issued_extensions,
        tuple(cells),
        (ChildResourceTokenLimit(child, 792, 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    "NATIVE_NUMERICAL_FAILURE",
                    "NATIVE_OBSERVER_FAILURE",
                    "RESOURCE_COMPUTABILITY_UNAVAILABLE",
                    "WORKER_PROGRESS_STALLED",
                    *(reason.value for reason in OperationalFailureClass),
                }
            )
        ),
        None,
        None,
        BUDGET.cpu_cores,
        min(
            BUDGET.memory_bytes,
            sum(
                sorted((cell.resource_budget.memory_bytes for cell in cells), reverse=True)[
                    : BUDGET.cpu_cores
                ]
            ),
        ),
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "carrier": carrier.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": 792,
                    "retry_tokens": 0,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
