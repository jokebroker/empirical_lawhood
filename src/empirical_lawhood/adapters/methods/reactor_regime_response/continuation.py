"""Exact retained operands for the owner-directed C resource continuation.

Completed preparation, prediction and assay tasks become external inputs;
their original receipts remain the only acquisition/forecast evidence.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.time import validate_utc_timestamp
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactManifest, ArtifactProfile, CanonicalTaskReceipt
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ExecutionPlan
from empirical_lawhood.runtime.providers import ExternalInputPayload, ExternalInputSource, MAX_EXTERNAL_INPUT_CHUNK_BYTES
from empirical_lawhood.runtime.study_issue import IssuedExecutableStudyManifest

from .config import PREFIX, ROOTS
from .resource_contract import RegimePhaseAllocation
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from .preassay_readout import RegimePreassayReadout


def completed_c_slots() -> tuple[tuple[str, str, str], ...]:
    slots: list[tuple[str, str, str]] = []
    for root, role, _, _ in ROOTS:
        if role not in ("calibration", "qualification"):
            continue
        slots.extend((
            (f"regime.prepare.{root}", "causal", RegimeCausalPreparation.SCHEMA),
            (f"regime.prepare.{root}", "private", RegimePrivatePreparation.SCHEMA),
        ))
        if role == "calibration":
            slots.extend((
                (f"regime.seal.{root}", "science", RegimePredictionSeal.SCHEMA),
                (f"regime.seal.{root}", "preassay", RegimePreassayReadout.SCHEMA),
                (f"regime.assay.{root}", "assay", RegimeAssayPanel.SCHEMA),
            ))
    return tuple(sorted(slots))


@dataclass(frozen=True, slots=True)
class RegimeRetainedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-retained-input'
    task_id: str
    output_id: str
    manifest: ArtifactManifest
    receipt: CanonicalTaskReceipt

    def __post_init__(self) -> None:
        logical, material = self.manifest.logical, self.manifest.materialization
        if (
            (self.task_id, self.output_id, logical.payload_schema) not in completed_c_slots()
            or self.receipt.task_id != self.task_id
            or self.receipt.run_id != f"{PREFIX}-phase-c"
            or self.receipt.operational_status is not OperationalStatus.SUCCEEDED
            or self.manifest.publication is None
            or logical not in self.receipt.output_logical_artifacts
            or material not in self.receipt.output_materializations
            or logical.content_sha256 != material.physical_sha256
            or logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or logical.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("retained C operand lacks exact original successful custody")

    @property
    def artifact(self) -> ArtifactIdentity:
        logical = self.manifest.logical
        return ArtifactIdentity(logical.logical_artifact_id, "retained-reactor-c-operand",
                                logical.payload_schema, logical.content_sha256,
                                logical.media_type, self.manifest.materialization.size_bytes)


@dataclass(frozen=True, slots=True)
class RegimeCContinuation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-c-continuation'
    continuation_id: str
    original_execution: ObjectIdentity
    original_issue: ObjectIdentity
    original_allocation: ObjectIdentity
    original_started_at_utc: str
    inputs: tuple[RegimeRetainedInput, ...]

    def __post_init__(self) -> None:
        validate_utc_timestamp(self.original_started_at_utc)
        if (
            self.continuation_id != f"{PREFIX}.retained-calibration-qualification-continuation"
            or self.original_execution.object_schema != ExecutionPlan.SCHEMA
            or self.original_execution.object_id != f"execution.{PREFIX}-phase-c"
            or self.original_issue.object_schema != IssuedExecutableStudyManifest.SCHEMA
            or self.original_allocation.object_schema != RegimePhaseAllocation.SCHEMA
            or tuple((row.task_id, row.output_id, row.manifest.logical.payload_schema)
                     for row in self.inputs) != completed_c_slots()
            or len({row.artifact.artifact_id for row in self.inputs}) != 288
            or len({row.receipt.implementation_commit for row in self.inputs}) != 1
            or len({(row.task_id, row.receipt.fingerprint()) for row in self.inputs}) != 160
        ):
            raise ValueError("C continuation changes its 160-task/288-output retention census")

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.task_id for row in self.inputs}))


class RegimeRetainedInputsPort(Protocol):
    @property
    def continuation(self) -> RegimeCContinuation | None: ...

    def create_source(self, declaration: RegimeRetainedInput) -> ExternalInputSource: ...


@dataclass(frozen=True, slots=True)
class NoRegimeRetainedInputs:
    continuation: None = None

    def create_source(self, declaration: RegimeRetainedInput) -> ExternalInputSource:
        raise ValueError("this phase has no retained C operands")


def retained_input_map(port: RegimeRetainedInputsPort) -> dict[str, RegimeRetainedInput]:
    return {} if port.continuation is None else {
        row.artifact.artifact_id: row for row in port.continuation.inputs
    }


def retained_payloads(plan: ProtocolExecutionPlan, port: RegimeRetainedInputsPort) -> tuple[ExternalInputPayload, ...]:
    """The method provider owns each streamed external operand exactly once."""
    if port.continuation is None:
        return ()
    if set(port.continuation.completed_task_ids).intersection(task.task_id for task in plan.tasks):
        raise ValueError("C continuation would reacquire a completed task")
    specs = {item.logical_artifact_id: item for task in plan.tasks for item in task.external_inputs}
    values = []
    for row in port.continuation.inputs:
        artifact, logical = row.artifact, row.manifest.logical
        spec = specs.get(artifact.artifact_id)
        if (spec is None or spec.expected_content_sha256 != artifact.sha256
                or spec.expected_payload_schema != artifact.payload_schema
                or spec.expected_media_type != artifact.media_type
                or spec.expected_visibility_ceiling is not logical.visibility_ceiling
                or spec.expected_outcome_access is not logical.outcome_access):
            raise ValueError("retained C operand changed after candidate freeze")
        parent = ArtifactLineageParent(ObjectIdentity.from_record(row.receipt.receipt_id, row.receipt),
                                       logical.visibility_ceiling, logical.outcome_access)
        values.append(ExternalInputPayload(
            artifact.artifact_id, artifact.payload_schema, ArtifactProfile.CANONICAL_JSON,
            artifact.media_type, port.create_source(row), artifact.size_bytes,
            artifact.sha256, artifact.size_bytes, min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
            logical.visibility_ceiling, logical.outcome_access,
            (parent.visibility_ceiling,), (parent,), artifact.sha256,
        ))
    return tuple(values)
