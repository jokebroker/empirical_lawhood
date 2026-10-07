"causal response prediction exact non-time reservations; a separate elapsed budget controls active time."

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import derive_source_qualification_topology
from empirical_lawhood.adapters.simulators.causal_response.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from .authoring import BUDGET, PREFIX, CausalResponseAuthoringBundle


def build_causal_response_resource_envelope(
    bundle: CausalResponseAuthoringBundle,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    carrier = next(r for r in bundle.payloads if isinstance(r, FreshSourceQualificationExperiment))
    template = bundle.standard_context.base.templates[0].protocol
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(SOURCE_BINDING.binding_id)
    expand = getattr(factory, "expand_parameterised_protocol")
    exact = expand(records=bundle.payloads, template=template)
    if protocol != exact:
        raise ValueError("causal response prediction resources require the exact factory-expanded protocol")
    stages = derive_source_qualification_topology(carrier).stages
    expected = {s.task_id: s for s in stages}
    if len(protocol.steps) != 3073 or {s.step_id for s in protocol.steps} != set(expected):
        raise ValueError("causal response prediction resources change the complete 3073-task roster")
    native = {s.segment_id: s for s in carrier.segments}
    views = {v.view_id: v for v in carrier.views}
    units = {u.physical_independent_unit_id: u for u in carrier.physical_units}
    cells = []
    for step in protocol.steps:
        source = step.step_id in native
        operand = native.get(step.step_id) or views.get(step.step_id)
        unit = None if operand is None else units[operand.physical_independent_unit_id]
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                PREFIX,
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
                    f"{PREFIX}.completed-native-updates",
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
    cells.sort(key=lambda c: c.cell_id)
    if sum(c.physical_execution_cost for c in cells) != 2944:
        raise ValueError("causal response prediction resources change the native segment census")
    return ExecutionResourceEnvelopeSpec(
        f"{PREFIX}.resources",
        issued_extensions,
        tuple(cells),
        (ChildResourceTokenLimit(PREFIX, 2944, 0),),
        (),
        (),
        tuple(
            sorted(
                {
                    "CAMPAIGN_ELAPSED_BUDGET_INSUFFICIENT_FOR_PROJECTED_TASK",
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
        BUDGET.cpu_cores,
        min(
            BUDGET.memory_bytes,
            sum(
                sorted((c.resource_budget.memory_bytes for c in cells), reverse=True)[
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
                    "native_execution_tokens": 2944,
                    "retry_tokens": 0,
                    "ceiling": "NON_TIME_ENVELOPE_SEPARATE_CAUSAL_RESPONSE_PREDICTION_CUMULATIVE_ELAPSED_BINDING_REQUIRED",
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
