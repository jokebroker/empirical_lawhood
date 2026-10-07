"""Registered remote/container task executor behind the common TaskExecutor port."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityPermission
from empirical_lawhood.runtime.execution import (
    ExecutorEnforcementCapability,
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)


class ExternalExecutionBackendKind(StrEnum):
    REMOTE_REGISTERED = "REMOTE_REGISTERED"
    OCI_REGISTERED = "OCI_REGISTERED"


class ExternalExecutionError(RuntimeError):
    """A registered external execution boundary failed closed."""


class ExternalWorkerUnavailable(ExternalExecutionError):
    """The registered worker did not return a typed result."""


_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 0,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
    OutcomeAccess.EVALUATOR_REVEAL: 2,
    OutcomeAccess.EVALUATION_REVEALED: 2,
    OutcomeAccess.PRIVILEGED_TRUTH: 3,
}


@dataclass(frozen=True, slots=True)
class RegisteredExecutionBackend(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/infrastructure/registered-execution-backend'

    backend_id: str
    kind: ExternalExecutionBackendKind
    transport_id: str
    transport_implementation_sha256: str
    allowed_capability_ids: tuple[str, ...]
    maximum_timeout_seconds: int
    maximum_output_bytes: int
    enabled: bool
    network_enabled: bool
    image_digest: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("backend_id", self.backend_id),
            ("transport_id", self.transport_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.transport_implementation_sha256,
            field_name="transport_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.allowed_capability_ids,
            field_name="allowed_capability_ids",
            allow_empty=False,
        )
        if self.maximum_timeout_seconds <= 0 or self.maximum_output_bytes <= 0:
            raise ValueError("external backend compute ceilings must be positive")
        if self.kind is ExternalExecutionBackendKind.REMOTE_REGISTERED:
            if not self.network_enabled or self.image_digest is not None:
                raise ValueError("remote backend requires network and has no local image")
        else:
            if self.image_digest is None or not self.image_digest.startswith("sha256:"):
                raise ValueError("OCI backend requires an exact image digest")
            validate_sha256(self.image_digest.removeprefix("sha256:"), field_name="image_digest")


class ExternalWorkerTransport(Protocol):
    transport_id: str
    implementation_sha256: str

    def execute(
        self,
        backend: RegisteredExecutionBackend,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult: ...


class RegisteredExternalTaskExecutor:
    """Run a frozen task through one pre-installed transport implementation."""

    # This compatibility surface does not yet enforce the complete local
    # resource/input/scratch/stream contract and is refused by production
    # campaign composition.
    enforcement_capability = ExecutorEnforcementCapability(
        capability_id="registered-external-task-executor",
        cpu_time_limit=False,
        address_space_limit=False,
        wall_time_limit=False,
        source_scan_limit=False,
        scratch_limit=False,
        output_limit=False,
        network_isolation=False,
    )

    def __init__(
        self,
        backend: RegisteredExecutionBackend,
        transport: ExternalWorkerTransport,
    ) -> None:
        if (
            transport.transport_id != backend.transport_id
            or transport.implementation_sha256 != backend.transport_implementation_sha256
        ):
            raise ExternalExecutionError("external transport identity differs from registry")
        self.backend = backend
        self.transport = transport

    def execute(
        self,
        runner: TaskRunner,
        context: TaskContext,
        *,
        timeout_seconds: int,
    ) -> RunnerResult:
        self._validate_request(runner, context, timeout_seconds)
        try:
            result = self.transport.execute(
                self.backend,
                runner,
                context,
                timeout_seconds=timeout_seconds,
            )
        except ExternalWorkerUnavailable:
            raise
        except (OSError, TimeoutError) as error:
            raise ExternalWorkerUnavailable("registered external worker is unavailable") from error
        self._validate_result(result, runner, context)
        return result

    def _validate_request(
        self,
        runner: TaskRunner,
        context: TaskContext,
        timeout_seconds: int,
    ) -> None:
        manifest = runner.manifest
        if not self.backend.enabled:
            raise ExternalExecutionError("external execution backend is disabled")
        if manifest.registry_id not in self.backend.allowed_capability_ids:
            raise ExternalExecutionError("capability is absent from the external backend allowlist")
        if timeout_seconds <= 0 or timeout_seconds > self.backend.maximum_timeout_seconds:
            raise ExternalExecutionError("task timeout exceeds the external backend ceiling")
        if timeout_seconds > manifest.resource_ceiling.wall_time_seconds:
            raise ExternalExecutionError("task timeout exceeds the capability resource ceiling")
        self._validate_task_context(manifest, context)
        if (
            self.backend.kind is ExternalExecutionBackendKind.OCI_REGISTERED
            and self.backend.network_enabled
            and not manifest.requires_network
        ):
            raise ExternalExecutionError("OCI backend leaks undeclared worker network access")

    @staticmethod
    def _validate_task_context(
        manifest: CapabilityManifest,
        context: TaskContext,
    ) -> None:
        if context.config.config_schema != manifest.config_schema:
            raise ExternalExecutionError("task configuration schema differs from capability")
        if not set(context.permissions).issubset(manifest.permissions):
            raise ExternalExecutionError("task context leaks capability permissions")
        if (
            _OUTCOME_ACCESS_RANK[context.outcome_access]
            > _OUTCOME_ACCESS_RANK[manifest.maximum_outcome_access]
        ):
            raise ExternalExecutionError("task context leaks outcome access")
        if CapabilityPermission.COMMAND_ACTUATOR in context.permissions:
            raise ExternalExecutionError("generic external executor cannot command actuators")
        output_schemas = {port.payload_schema for port in context.output_ports}
        if not output_schemas.issubset(manifest.output_schema_ids):
            raise ExternalExecutionError("task output schema differs from capability")

    def _validate_result(
        self,
        result: RunnerResult,
        runner: TaskRunner,
        context: TaskContext,
    ) -> None:
        if not isinstance(result, RunnerResult):
            raise ExternalExecutionError("external worker returned an untyped result")
        outputs = tuple(
            output for output in result.outputs if isinstance(output, TaskOutputPayload)
        )
        if len(outputs) != len(result.outputs):
            raise ExternalExecutionError(
                "external worker streaming is disabled until equivalence is enforced"
            )
        output_bytes = sum(len(output.payload) for output in outputs)
        if output_bytes > min(
            self.backend.maximum_output_bytes,
            runner.manifest.resource_ceiling.output_bytes,
        ):
            raise ExternalExecutionError("external worker exceeded its output ceiling")
        expected_output_ids = tuple(port.output_id for port in context.output_ports)
        if tuple(output.output_id for output in result.outputs) != expected_output_ids:
            raise ExternalExecutionError("external worker output ports differ from ExecutionPlan")
