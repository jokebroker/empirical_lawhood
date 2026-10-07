"""Classical record role/check markers over shared task custody assembly."""

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactManifest
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import ExternalInputPayload
from empirical_lawhood.runtime.task_records import (
    artifact_identity,
    canonical_task_result,
    external_config as _external_config,
    config_input as config_input,
    contracts as contracts,
    dependency as dependency,
    verify_registry as verify_registry,
)


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
