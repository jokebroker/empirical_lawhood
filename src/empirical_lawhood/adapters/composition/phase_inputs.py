"""Exact phase input bindings over the installed dependency/custody helpers."""

from typing import TypeVar, Protocol
from empirical_lawhood.runtime.task_records import (
    dependency as dependency,
    config_input as config_input,
    contracts as contracts,
    verify_registry as verify_registry,
    artifact_identity,
    canonical_task_result,
    external_config as _external_config,
)
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import ExternalInputPayload
from empirical_lawhood.runtime.execution import RunnerResult
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.execution import TaskContext, WorkerInputKind
from empirical_lawhood.runtime.artifacts import ArtifactManifest


class UpstreamOperand(Protocol):
    @property
    def key(self) -> str: ...
    @property
    def artifact(self) -> ArtifactIdentity: ...


class PhaseInputs(Protocol):
    @property
    def upstream(self) -> tuple[UpstreamOperand, ...]: ...


R = TypeVar("R", bound=CanonicalRecord)


def upstream(context: TaskContext, phase: PhaseInputs, key: str, kind: type[R]) -> R:
    source = next(r for r in phase.upstream if r.key == key)
    ports = tuple(
        p
        for p in context.input_ports
        if p.kind is WorkerInputKind.EXTERNAL
        and p.binding.artifact_id == source.artifact.artifact_id
        and p.payload_schema == kind.SCHEMA
    )
    if len(ports) != 1:
        raise ValueError("phase consumer lacks its exact compiled upstream input")
    raw = ports[0].read()
    record = decode_canonical_bytes(raw, kind, maximum_bytes=256 * 1024**2)
    if len(raw) != source.artifact.size_bytes or record.fingerprint() != source.artifact.sha256:
        raise ValueError("phase upstream differs from its frozen receipted bytes")
    return record


def artifact(manifest: ArtifactManifest) -> ArtifactIdentity:
    return artifact_identity(manifest, role="classical-receipted-operand")


def result(context: TaskContext, records: tuple[CanonicalRecord, ...]) -> RunnerResult:
    return canonical_task_result(
        context, records, check_id="classical-exact-custody-and-scientific-owner"
    )


def external_config(
    plan: ProtocolExecutionPlan, capability: CapabilityManifest, config: CanonicalRecord
) -> tuple[ExternalInputPayload, ...]:
    return _external_config(plan, capability, config, config_id=config.config_id)  # type: ignore[attr-defined]
