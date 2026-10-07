"Exact deadline-free resource envelope for the preparation-policy development graph."

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayAuthoringBundle
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, preparation_policy_native_invocations
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityProviderFactory
from empirical_lawhood.runtime.plans import ProtocolTemplate

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from .design import native_budget


def preparation_policy_resource_envelope(
    bundle: ResponseGeometryAssayAuthoringBundle,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    source = next(record for record in bundle.payloads if type(record) is FiniteResponseLawPreparationPolicyNativeConfig)
    carrier = next(
        record for record in bundle.payloads if type(record) is PredecessorBoundSourceQualificationExperiment
    )
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(SOURCE_BINDING.binding_id)
    if not isinstance(factory, ExecutableCapabilityProviderFactory):
        raise TypeError("preparation-policy development source binding lacks its installed provider factory")
    exact = getattr(factory, "expand_parameterised_protocol")(
        records=bundle.payloads, template=bundle.standard_context.base.templates[0].protocol
    )
    if protocol != exact or len(protocol.steps) != 4_129:
        raise ValueError("preparation-policy development resources require the exact factory-expanded full graph")
    native_ids = {value.task_id for value in preparation_policy_native_invocations(source)}
    unit_by_segment = {
        segment.segment_id: segment.physical_independent_unit_id for segment in carrier.segments
    }
    preparation_by_unit = {
        unit.physical_independent_unit_id: unit.preparation_instance_id
        for unit in carrier.physical_units
    }
    cells = []
    child_id = f"{PROGRAMME}.preparation-policy"
    for step in protocol.steps:
        native = step.step_id in native_ids
        budget = step.resource_budget
        cells.append(
            ExecutionResourceTaskCellSpec(
                f"cell.{step.step_id}",
                step.step_id,
                child_id,
                unit_by_segment.get(step.step_id),
                preparation_by_unit.get(unit_by_segment.get(step.step_id, "")),
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
                ProgressLivenessContract(
                    f"progress.{step.step_id}",
                    ProgressHeartbeat.SCHEMA,
                    f"{PROGRAMME}.completed-native-updates",
                    Decimal(600),
                    True,
                    True,
                )
                if native
                else None,
                None,
                None,
            )
        )
    limit = native_budget(source)  # type: ignore[arg-type]
    if (
        sum(cell.physical_execution_cost for cell in cells) != 4_104
        or sum(cell.resource_budget.output_bytes for cell in cells) > limit.output_bytes
        or sum(cell.resource_budget.source_byte_limit for cell in cells) > limit.source_scan_bytes
    ):
        raise ValueError("preparation-policy development full graph exceeds its declared non-time resource census")
    return ExecutionResourceEnvelopeSpec(
        f"{PROGRAMME}.preparation-policy.resources",
        issued_extensions,
        tuple(sorted(cells, key=lambda value: value.cell_id)),
        (ChildResourceTokenLimit(child_id, 4_104, 0),),
        (),
        (),
        tuple(
            sorted(
                {
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
        8,
        8 * 1024**3,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "carrier": carrier.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": 4_104,
                    "retry_tokens": 0,
                    "elapsed_time_has_no_control_effect": True,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
