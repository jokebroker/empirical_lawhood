"""Declared reactor task assignments for the shared resource assembler."""

from decimal import Decimal as D
from empirical_lawhood.adapters.composition.task_resources import progress_resource_envelope
from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import ClassicalDesign, PREFIX
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.plans import ProtocolTemplate


def resource_envelope(
    design: ClassicalDesign, protocol: ProtocolTemplate, issued_extensions: ObjectIdentity
) -> ExecutionResourceEnvelopeSpec:
    prefixes = ("classical.prepare.", "classical.assay.", "classical.action.")
    native = tuple(
        (s.step_id, s.step_id.removeprefix(p), f"{s.step_id.removeprefix(p)}.initial-state")
        for s in protocol.steps
        for p in prefixes
        if s.step_id.startswith(p)
    )
    return progress_resource_envelope(
        prefix=PREFIX,
        design=design,
        protocol=protocol,
        issued_extensions=issued_extensions,
        cpu_seconds=design.cpu_seconds,
        native_preparations=native,
        progress_counter="reactor-completed-native-decisions",
        heartbeat_seconds=D(60),
        maximum_concurrency=4,
        maximum_memory_bytes=32 * 1024**3,
        terminal_failure_codes=(
            "NATIVE_NUMERICAL_FAILURE",
            "NATIVE_OBSERVER_FAILURE",
            "WORKER_PROGRESS_STALLED",
            "RESOURCE_COMPUTABILITY_UNAVAILABLE",
            "RETRY_BUDGET_EXHAUSTED",
        ),
    )
