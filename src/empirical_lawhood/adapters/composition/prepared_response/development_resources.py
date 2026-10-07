"""dependent refinement's exact non-time resource envelope; elapsed admission stays separate.

The envelope closes the full source/projection/fit/selection census without
claiming that the complete campaign fits its cumulative elapsed budget.  A
measured canary must separately supply ``CampaignElapsedReservationPlan``
projections before issue.
"""

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget, ProgressHeartbeat, ProgressLivenessContract
from dataclasses import replace

from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificStage
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.executable_binding import DEVELOPMENT_SOURCE_BINDING
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import prepared_static_native_invocations
from empirical_lawhood.adapters.methods.prepared_response.development_provider import PreparedResponseDevelopmentSelectionTask
from empirical_lawhood.adapters.methods.prepared_response.models import STRUCTURES

from .development_design import DEVELOPMENT_BUDGET


PREFIX = "prepared-response.dependent-refinement"


def build_prepared_response_development_resource_envelope(
    records: tuple[CanonicalRecord, ...],
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    "Bind all 8,713 dependent refinement tasks and 8,448 native effects to one envelope."

    sources = tuple(value for value in records if isinstance(value, PreparedNativeSpec))
    if len(sources) != 1 or sources[0].stage != 'development':
        raise ValueError("prepared dependent refinement resources require one exact dependent refinement native spec")
    source = sources[0]
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(
        DEVELOPMENT_SOURCE_BINDING.binding_id
    )
    expand = getattr(factory, "expand_parameterised_protocol")
    exact = expand(records=records, template=protocol)
    # Expansion is idempotent because it replaces the prototype steps with the
    # exact graph.  Comparing the result prevents a caller from trimming or
    # changing task resources after candidate compilation.
    if protocol != replace(exact, template_id=protocol.template_id):
        raise ValueError("prepared dependent refinement resources require the exact factory-expanded protocol")

    native = {
        invocation.task_id: invocation
        for invocation in prepared_static_native_invocations(source)
    }
    projections = {
        f"{root.root_id}.project.r{refinement}": root
        for root in source.roots
        for refinement in (1, 2)
    }
    expected_ids = set(native) | set(projections)
    expected_ids.update(
        f"{PREFIX}.fit.{context}.{structure}"
        for context in ("assembling", "prepared")
        for structure in STRUCTURES
    )
    expected_ids.add(PreparedResponseDevelopmentSelectionTask.TASK_ID)
    if len(protocol.steps) != 8713 or {step.step_id for step in protocol.steps} != expected_ids:
        raise ValueError("prepared dependent refinement resources change the complete 8713-task roster")

    cells: list[ExecutionResourceTaskCellSpec] = []
    for step in protocol.steps:
        invocation = native.get(step.step_id)
        root = invocation.root if invocation is not None else projections.get(step.step_id)
        unit_id = None if root is None else root.physical_unit_id
        is_native = invocation is not None
        if step.maximum_attempts != 1 or (step.stage is ScientificStage.PREPARE) != is_native:
            raise ValueError("prepared dependent refinement resource identity changes native effect or retry policy")
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
    if sum(value.physical_execution_cost for value in cells) != 8448:
        raise ValueError("prepared dependent refinement resources change the native task census")
    return ExecutionResourceEnvelopeSpec(
        f"{PREFIX}.resources",
        issued_extensions,
        tuple(cells),
        (ChildResourceTokenLimit(PREFIX, 8448, 0),),
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
        DEVELOPMENT_BUDGET.cpu_cores,
        min(
            DEVELOPMENT_BUDGET.memory_bytes,
            sum(
                sorted(
                    (value.resource_budget.memory_bytes for value in cells), reverse=True
                )[: DEVELOPMENT_BUDGET.cpu_cores]
            ),
        ),
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "source": source.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": 8448,
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


__all__ = ['build_prepared_response_development_resource_envelope']
