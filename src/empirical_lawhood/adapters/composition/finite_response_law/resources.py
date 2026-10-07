"""Bind the declared native census to existing resource and liveness records."""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment, ProspectiveRetainedSourceUse
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import derive_source_qualification_topology
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.simulators.finite_response_law.executable_binding import native_bindings
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig, native_invocations
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.control_provider import control_tasks
from .design import native_budget


def native_resource_envelope(
    bundle: ResponseGeometryAssayAuthoringBundle, protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    carrier = next(r for r in bundle.payloads if isinstance(r, PredecessorBoundSourceQualificationExperiment))
    source = next(r for r in bundle.payloads if isinstance(r, FiniteResponseLawNativeConfig))
    budget = native_budget(source)
    source_binding, _, _ = native_bindings(source)
    if isinstance(carrier, ProspectiveRetainedSourceUse):
        from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation.executable_binding import SOURCE_BINDING

        source_binding = SOURCE_BINDING
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(source_binding.binding_id)
    exact = getattr(factory, "expand_parameterised_protocol")(
        records=bundle.payloads, template=bundle.standard_context.base.templates[0].protocol
    )
    expected = {s.task_id for s in derive_source_qualification_topology(carrier).stages}
    controls = tuple(r for r in bundle.payloads if type(r) is FiniteResponseLawControlConfig)
    if controls:
        if len(controls) != 1 or controls[0].source != source:
            raise ValueError("Independent evaluation resource census substitutes its causal control configuration")
        expected.update(control_tasks(controls[0]))
        expected.update(f"{r.stage_unit}.seal-control" for r in controls[0].source.roots)
        from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig, reveal_tasks

        expected.update(reveal_tasks(FiniteResponseLawEvaluationRevealConfig(controls[0])))
    if protocol != exact or {s.step_id for s in protocol.steps} != expected:
        raise ValueError("FLH resources require the full exact factory-expanded protocol")
    native = {s.segment_id: s for s in carrier.segments}
    views = {v.view_id: v for v in carrier.views}
    units = {u.physical_independent_unit_id: u for u in carrier.physical_units}
    prefix = f"{PROGRAMME}.{source.stage}"
    cells = []
    for step in protocol.steps:
        is_native = step.step_id in native
        operand = native.get(step.step_id) or views.get(step.step_id)
        unit = None if operand is None else units[operand.physical_independent_unit_id]
        b = step.resource_budget
        if step.maximum_attempts != 1:
            raise ValueError("FLH has no native retry or replacement budget")
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                prefix,
                None if unit is None else unit.physical_independent_unit_id,
                None if unit is None else unit.preparation_instance_id,
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
                    f"{PROGRAMME}.completed-native-updates",
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
    if (
        sum(c.physical_execution_cost for c in cells)
        != len(native_invocations(source))
        - (
            len(carrier.retained_predecessors)
            if isinstance(carrier, ProspectiveRetainedSourceUse)
            else 0
        )
        or sum(s.resource_budget.output_bytes for s in protocol.steps) > budget.output_bytes
        or sum(s.resource_budget.source_scan_bytes for s in protocol.steps)
        > budget.source_scan_bytes
        or sum(s.resource_budget.wall_time_seconds for s in protocol.steps) > 69120
    ):
        raise ValueError("FLH exact full census exceeds its frozen programme budget")
    return ExecutionResourceEnvelopeSpec(
        f"{prefix}.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (ChildResourceTokenLimit(prefix, len(native), 0),),
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
        budget.cpu_cores,
        min(
            budget.memory_bytes,
            sum(
                sorted((c.resource_budget.memory_bytes for c in cells), reverse=True)[
                    : budget.cpu_cores
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
                    "native_execution_tokens": len(native),
                    "retry_tokens": 0,
                    "ceiling": "NON_TIME_ENVELOPE_SEPARATE_CUMULATIVE_ELAPSED_BINDING_REQUIRED",
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
