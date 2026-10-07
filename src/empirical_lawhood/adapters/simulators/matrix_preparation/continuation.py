"""Additive custody and completion declaration for the cancelled development run.

The native experiment configuration is unchanged. The new execution imports all
51 committed roots and performs native work for the exact 77-root complement.
These records describe custody; they confer no execution or scientific authority.
"""

from dataclasses import dataclass
from typing import ClassVar
import re

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.matrix_preparation.retained_exports import PreparationRetainedSourceExport
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from .contracts import CANONICAL_MEDIA_TYPE, DEVELOPMENT, NATIVE_SCHEMA, PreparationNativeResult, PreparationRoot, PreparationSourceConfig, native_updates_for_root, preparation_roots


CONTINUATION_RUN = "matrix-preparation-adequacy.native-completion"
NATIVE_OUTPUT_SCHEMAS = {
    "native-observations": NATIVE_SCHEMA,
    "native-result": PreparationNativeResult.SCHEMA,
    "stage-envelope": LinkedCampaignStageEnvelope.SCHEMA,
}


def retained_development_roots() -> tuple[PreparationRoot, ...]:
    return tuple(PreparationRoot("assembling", i) for i in range(51))


@dataclass(frozen=True, slots=True)
class PreparationNativeCustodyInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-native-custody-input'
    root: PreparationRoot
    output_id: str
    artifact: ArtifactIdentity
    relative_path: str
    task_receipt: ObjectIdentity
    manifest_sha256: str
    publication_commit_sha256: str

    source_export: PreparationRetainedSourceExport | None = None

    def __post_init__(self) -> None:
        if type(self.source_export) is not PreparationRetainedSourceExport:
            raise ValueError("retained completion input requires a separately verified original-source export and current target custody before work")
        self.source_export.validate_target(
            self.artifact, self.relative_path, self.task_receipt,
            self.manifest_sha256, self.publication_commit_sha256,
            (self.root.physical_unit_id,),
        )
        validate_relative_locator(self.relative_path)
        validate_sha256(self.manifest_sha256, field_name="manifest_sha256")
        validate_sha256(self.publication_commit_sha256, field_name="publication_commit_sha256")
        binary = self.output_id == "native-observations"
        suffix = ".h5" if binary else ".canonical.json"
        if (
            self.root not in retained_development_roots()
            or self.output_id not in NATIVE_OUTPUT_SCHEMAS
            or self.artifact.artifact_id != f"artifact.{DEVELOPMENT}.{self.slot_id}"
            or self.artifact.payload_schema != NATIVE_OUTPUT_SCHEMAS[self.output_id]
            or self.artifact.media_type
            != ("application/x-hdf5" if binary else CANONICAL_MEDIA_TYPE)
            or not 0 < self.artifact.size_bytes <= (98 * 1024**2 if binary else 512 * 1024)
            or self.task_receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
            or self.task_receipt.object_version != "1.0.0"
            or self.task_receipt.object_id != f"receipt.{DEVELOPMENT}.{self.task_id}.attempt-001"
            or self.relative_path
            != f"runs/{DEVELOPMENT}/outputs/{self.task_id}/{self.output_id}{suffix}"
        ):
            raise ValueError("development continuation substitutes a retained native custody slot")

    @property
    def task_id(self) -> str:
        return f"{self.root.root_id}.native"

    @property
    def slot_id(self) -> str:
        return f"{self.task_id}.{self.output_id}"

    @property
    def imported_artifact_id(self) -> str:
        # This is a new materialization identity, matching the unchanged method
        # input convention. Original artifact and receipt remain lineage parents.
        return f"artifact.{CONTINUATION_RUN}.{self.slot_id}"


@dataclass(frozen=True, slots=True)
class PreparationDevelopmentContinuation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-development-continuation'
    config_id: str
    source_config: ObjectIdentity
    native_inputs: tuple[PreparationNativeCustodyInput, ...]
    custody_audit_sha256: str
    cancellation_sha256: str
    implementation_plan: ObjectIdentity
    source_commit: str
    source_run_id: str = DEVELOPMENT
    run_id: str = CONTINUATION_RUN
    grants_authority: bool = False
    fresh_independent_units: int = 0
    maximum_elapsed_seconds: int = 86400
    deadline_mechanism: str = "PUBLIC_MAINTENANCE_INTERRUPTION_WITH_EXACT_PROCESS_AND_PLAN"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.custody_audit_sha256, field_name="custody_audit_sha256")
        validate_sha256(self.cancellation_sha256, field_name="cancellation_sha256")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("development continuation requires an explicit source commit")
        require_sorted_unique_ids(
            self.native_inputs, attribute="slot_id", field_name="native_inputs"
        )
        expected = tuple(
            sorted(
                f"{root.root_id}.native.{output}"
                for root in retained_development_roots()
                for output in NATIVE_OUTPUT_SCHEMAS
            )
        )
        if (
            tuple(row.slot_id for row in self.native_inputs) != expected
            or self.source_config.object_schema != PreparationSourceConfig.SCHEMA
            or self.source_config.object_id != f"{DEVELOPMENT}.source-config"
            or self.implementation_plan.object_schema != 'empirical-lawhood/document/markdown'
            or self.implementation_plan.object_id != f"{CONTINUATION_RUN}.implementation-plan"
            or self.source_run_id != DEVELOPMENT
            or self.run_id != CONTINUATION_RUN
            or self.grants_authority is not False
            or type(self.fresh_independent_units) is not int
            or self.fresh_independent_units != 0
            or type(self.maximum_elapsed_seconds) is not int
            or self.maximum_elapsed_seconds != 86400
            or self.deadline_mechanism
            != "PUBLIC_MAINTENANCE_INTERRUPTION_WITH_EXACT_PROCESS_AND_PLAN"
        ):
            raise ValueError("development continuation changes its fixed custody, plan or census")
        for root in retained_development_roots():
            rows = tuple(row for row in self.native_inputs if row.root == root)
            if len({row.task_receipt for row in rows}) != 1:
                raise ValueError("retained native outputs must share their original task receipt")

    @property
    def retained_task_ids(self) -> tuple[str, ...]:
        return tuple(f"{r.root_id}.native" for r in retained_development_roots())

    @property
    def missing_roots(self) -> tuple[PreparationRoot, ...]:
        retained = set(retained_development_roots())
        return tuple(r for r in preparation_roots() if r not in retained)

    @property
    def maximum_native_updates(self) -> int:
        return sum(native_updates_for_root(r) for r in self.missing_roots)

    def validate_source(self, source: PreparationSourceConfig) -> None:
        if self.source_config != ObjectIdentity.from_record(source.config_id, source):
            raise ValueError("retained native results require their unchanged source configuration")


def preparation_continuation(
    records: tuple[CanonicalRecord, ...],
) -> PreparationDevelopmentContinuation | None:
    selected = tuple(r for r in records if isinstance(r, PreparationDevelopmentContinuation))
    if len(selected) > 1:
        raise ValueError("development continuation requires one exact declaration")
    return selected[0] if selected else None
