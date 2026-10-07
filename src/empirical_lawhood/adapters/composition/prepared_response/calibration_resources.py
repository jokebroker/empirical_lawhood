"fresh calibration exact non-time resource envelope; measured elapsed admission stays separate."

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.methods.prepared_response.calibration_provider import PreparedResponseCalibrationCalibrationTask
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import CALIBRATION_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.prepared_response.policy_native import CALIBRATION_POLICIES, prepared_response_calibration_native_invocations
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.plans import ProtocolTemplate

from .calibration_design import CALIBRATION_BUDGET, PREFIX


def build_prepared_response_calibration_resource_envelope(
    records: tuple[CanonicalRecord, ...],
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    sources = tuple(value for value in records if isinstance(value, PreparedNativeSpec))
    if len(sources) != 1 or sources[0].stage != 'calibration':
        raise ValueError("prepared fresh calibration resources require one exact fresh calibration native spec")
    source = sources[0]
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(
        CALIBRATION_SOURCE_BINDING.binding_id
    )
    exact = factory.expand_parameterised_protocol(records=records, template=protocol)
    if protocol != replace(exact, template_id=protocol.template_id):
        raise ValueError("prepared fresh calibration resources require the exact factory-expanded protocol")
    native = {
        invocation.task_id: invocation for invocation in prepared_response_calibration_native_invocations(source)
    }
    policies = {
        f"{root.root_id}.policy.{policy}.decision": root
        for root in source.roots
        for policy in CALIBRATION_POLICIES
    }
    projections = {
        f"{root.root_id}.policy.{policy}.project.r{refinement}": root
        for root in source.roots
        for policy in CALIBRATION_POLICIES
        for refinement in (1, 2)
    }
    expected = set(native) | set(policies) | set(projections)
    expected.add(PreparedResponseCalibrationCalibrationTask.TASK_ID)
    if len(protocol.steps) != 10177 or {value.step_id for value in protocol.steps} != expected:
        raise ValueError("prepared fresh calibration resources change the complete 10177-task roster")
    cells = []
    for step in protocol.steps:
        invocation = native.get(step.step_id)
        root = (
            invocation.root
            if invocation is not None
            else policies.get(step.step_id) or projections.get(step.step_id)
        )
        unit_id = None if root is None else root.physical_unit_id
        is_native = invocation is not None
        if step.maximum_attempts != 1:
            raise ValueError("prepared fresh calibration resources change the no-retry contract")
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                PREFIX,
                unit_id,
                None if unit_id is None else f"{unit_id}.instance",
                1,
                int(is_native),
                int(is_native),
                is_native,
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
                if is_native
                else None,
                None,
                None,
            )
        )
    cells.sort(key=lambda value: value.cell_id)
    if sum(value.physical_execution_cost for value in cells) != 7872:
        raise ValueError("prepared fresh calibration resources change the native task census")
    return ExecutionResourceEnvelopeSpec(
        f"{PREFIX}.resources",
        issued_extensions,
        tuple(cells),
        (ChildResourceTokenLimit(PREFIX, 7872, 0),),
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
                    *(value.value for value in OperationalFailureClass),
                }
            )
        ),
        None,
        None,
        CALIBRATION_BUDGET.cpu_cores,
        min(
            CALIBRATION_BUDGET.memory_bytes,
            sum(
                sorted(
                    (value.resource_budget.memory_bytes for value in cells), reverse=True
                )[: CALIBRATION_BUDGET.cpu_cores]
            ),
        ),
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "source": source.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": 7872,
                    "retry_tokens": 0,
                    "ceiling": "NON_TIME_ONLY_MEASURED_CUMULATIVE_ADMISSION_REQUIRED",
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )


__all__ = ['build_prepared_response_calibration_resource_envelope']
