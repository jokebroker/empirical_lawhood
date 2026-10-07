"""Authenticate one complete static root before reducing any native operand."""

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedNativeSpec, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import prepared_static_native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativePhaseData
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native


@dataclass(frozen=True, slots=True)
class PreparedAuthenticatedRoot:
    identities: tuple[ObjectIdentity, ...]
    common: PreparedCommonStart | None
    data: dict[str, tuple[PreparedNativePhaseData, PreparedNativePhaseData] | None]


def authenticate_prepared_root(
    spec: PreparedNativeSpec,
    root: PreparedRoot,
    inputs: tuple[tuple[PreparedNativeTaskResult, bytes], ...],
) -> PreparedAuthenticatedRoot:
    if root not in spec.roots:
        raise ValueError("prepared projection is outside its frozen root census")
    expected = {
        task.task_id: task
        for task in prepared_static_native_invocations(spec, root=root)
    }
    if (
        type(inputs) is not tuple
        or len(inputs) != len(expected)
        or {result.invocation.task_id for result, _ in inputs} != set(expected)
    ):
        raise ValueError("prepared projection requires its complete native root census")
    records = {result.invocation.task_id: result for result, _ in inputs}
    identities = {
        key: ObjectIdentity.from_record(result.result_id, result) for key, result in records.items()
    }
    data = {}
    for result, payload in inputs:
        task = expected[result.invocation.task_id]
        if result.invocation != task or result.predecessors != tuple(
            identities[key] for key in task.dependency_task_ids
        ):
            raise ValueError("prepared projection source identities do not form the frozen lineage")
        data[task.task_id] = decode_prepared_task_native(result, payload)
    common = records[f"{root.root_id}.prefix.native"].common_start
    if any(result.common_start != common for result in records.values()):
        raise ValueError("prepared root changes its single frozen common-start record")
    return PreparedAuthenticatedRoot(
        tuple(sorted(identities.values(), key=lambda value: value.object_id)), common, data
    )
