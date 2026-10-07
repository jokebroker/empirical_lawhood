"""Explicit native census on the existing V3 progress envelope."""

from typing import Any


from decimal import Decimal
from empirical_lawhood.adapters.composition.task_resources import progress_resource_envelope


def resource_envelope(stage: Any, protocol: Any, issued_extensions: Any) -> Any:
    native = tuple(
        (s.step_id, s.step_id.split(".", 2)[2], f"{s.step_id}.native-branch-census")
        for s in protocol.steps
        if s.step_id.startswith(("ap.prefix.", "ap.parents.", "ap.assay."))
    )
    return progress_resource_envelope(
        prefix=stage.config_id,
        design=stage,
        protocol=protocol,
        issued_extensions=issued_extensions,
        cpu_seconds=stage.cpu_seconds,
        native_preparations=native,
        progress_counter="applicability-completed-native-intervals",
        heartbeat_seconds=Decimal(60),
        maximum_concurrency=4,
        maximum_memory_bytes=32 * 1024**3,
        terminal_failure_codes=(
            "NATIVE_NUMERICAL_FAILURE",
            "NATIVE_OBSERVER_FAILURE",
            "WORKER_PROGRESS_STALLED",
            "RESOURCE_COMPUTABILITY_UNAVAILABLE",
        ),
    )
