"""Exact bounded task inputs and canonical outputs over existing custody ports."""

from dataclasses import fields
from typing import Protocol, TypeVar
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactProfile,
    CanonicalTaskReceipt,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    TaskContext,
    RunnerResult,
    TaskOutputPayload,
    WorkerInputKind,
    WorkerInputBinding,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract, ExternalInputPayload

R = TypeVar("R", bound=CanonicalRecord)


class DependencyCustodyReader(Protocol):
    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]: ...


def validate_dependency(
    binding: WorkerInputBinding, manifest: ArtifactManifest, receipt: CanonicalTaskReceipt
) -> None:
    from empirical_lawhood.kernel.status import OperationalStatus

    if (
        binding.artifact_id != manifest.logical.logical_artifact_id
        or binding.materialization_id != manifest.materialization.materialization_id
        or binding.payload_schema != manifest.logical.payload_schema
        or binding.size_bytes != manifest.materialization.size_bytes
        or manifest.logical not in receipt.output_logical_artifacts
        or manifest.materialization not in receipt.output_materializations
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
    ):
        raise ValueError("dependency receipt/port identity differs")


def dependency(
    context: TaskContext,
    custody: DependencyCustodyReader,
    kind: type[R],
    task: str,
    *,
    maximum_bytes: int = 256 * 1024**2,
) -> tuple[R, ArtifactManifest, CanonicalTaskReceipt]:
    """Resolve the declared producer before authenticating its exact operand.

    Select by the compiled receipt/materialization binding before opening a
    port. Other same-schema siblings are never read or reauthenticated for this
    lookup. The selected record still passes all custody and digest checks.
    """
    parents = tuple(r for r in context.dependency_receipts if r.task_id == task)
    if len(parents) != 1:
        raise ValueError("input has no unique compiled parent receipt")
    ports = tuple(
        p
        for p in context.input_ports
        if p.kind is WorkerInputKind.DEPENDENCY
        and p.payload_schema == kind.SCHEMA
        and p.materialization_id in parents[0].output_materialization_ids
    )
    if len(ports) != 1:
        raise ValueError("input has no unique compiled parent/schema operand")
    port = ports[0]
    manifest, receipt = custody.read_dependency(context, port.binding)
    validate_dependency(port.binding, manifest, receipt)
    if receipt.task_id != task or receipt.receipt_id != parents[0].receipt_id:
        raise ValueError("custody substituted the compiled producer receipt")
    record = decode_canonical_bytes(port.read(), kind, maximum_bytes=maximum_bytes)
    if record.fingerprint() != manifest.logical.content_sha256:
        raise ValueError("operand differs from immutable custody")
    return record, manifest, receipt


def artifact_identity(manifest: ArtifactManifest, *, role: str) -> ArtifactIdentity:
    logical = manifest.logical
    return ArtifactIdentity(
        logical.logical_artifact_id,
        role,
        logical.payload_schema,
        logical.content_sha256,
        logical.media_type,
        manifest.materialization.size_bytes,
    )


def canonical_task_result(
    context: TaskContext, records: tuple[CanonicalRecord, ...], *, check_id: str
) -> RunnerResult:
    by_schema = {record.SCHEMA: record for record in records}
    if (
        len(by_schema) != len(records)
        or len(records) != len(context.output_ports)
        or {port.payload_schema for port in context.output_ports} != set(by_schema)
    ):
        raise ValueError("task output contract differs")
    return RunnerResult(
        tuple(
            TaskOutputPayload(port.output_id, by_schema[port.payload_schema].canonical_bytes())
            for port in context.output_ports
        ),
        (ReceiptCheck(check_id, True, ()),),
    )


def config_input(context: TaskContext, config: R) -> None:
    ports = tuple(port for port in context.input_ports if port.payload_schema == config.SCHEMA)
    if (
        context.config.content_sha256 != config.fingerprint()
        or len(ports) != 1
        or decode_canonical_bytes(ports[0].read(), type(config), maximum_bytes=4 * 1024**2)
        != config
    ):
        raise ValueError("runtime configuration differs from issue")


def external_config(
    plan: ProtocolExecutionPlan, capability: CapabilityManifest, config: CanonicalRecord, *, config_id: str
) -> tuple[ExternalInputPayload, ...]:
    specs = {
        spec.logical_artifact_id: spec
        for task in plan.tasks
        if task.capability.capability_key == capability.capability_key
        for spec in task.external_inputs
        if spec.expected_payload_schema == config.SCHEMA
    }
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(config_id, config),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    records = []
    for key, spec in sorted(specs.items()):
        if spec.expected_content_sha256 != config.fingerprint():
            raise ValueError("external config changed its canonical identity")
        records.append(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=key,
                payload_schema=config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                payload=config.canonical_bytes(),
                visibility_ceiling=parent.visibility_ceiling,
                outcome_access=parent.outcome_access,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=config.fingerprint(),
            )
        )
    return tuple(records)


def contracts(
    capability: CapabilityManifest, records: tuple[type[CanonicalRecord], ...]
) -> tuple[CapabilityOutputSemanticContract, ...]:
    return tuple(
        CapabilityOutputSemanticContract.from_manifest(
            capability,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            record_version=record.VERSION,
            top_level_keys=("schema", "value", "version"),
            value_keys=tuple(sorted(f.name for f in fields(record))),  # type: ignore[arg-type]
        )
        for record in records
    )


def verify_registry(registry: CapabilityRegistry, capability: CapabilityManifest) -> str:
    if registry.resolve(capability.capability_key, capability.capability_version) != capability:
        raise ValueError("installed capability differs")
    return registry.fingerprint()
