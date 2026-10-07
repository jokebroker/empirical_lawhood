"""Exact retained development measurements for a separate, analysis-only correction.

These declarations grant no new issue or execution authority. The original
source, roots, split, projections and scientific definitions remain bound;
only the nine uncompleted method tasks belong to the proposed new run.
"""

from dataclasses import dataclass
from typing import ClassVar
import re

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord, require_sorted_unique_ids, validate_relative_locator,
    validate_sha256, validate_stable_id,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import CANONICAL_MEDIA_TYPE
from .development_projection import DEVELOPMENT_DATA_SCHEMA, ResponseGeometryDevelopmentViewReport
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig


RETAINED_ANALYSIS_SOURCE_RUN = "response-geometry-development.native-completion"
RETAINED_ANALYSIS_RUN = "response-geometry-development.method-completion"
RETAINED_ANALYSIS_PREFIX = "response-geometry-development"
DEVELOPMENT_PROJECTION_SCHEMAS = {
    "report": ResponseGeometryDevelopmentViewReport.SCHEMA,
    "data": DEVELOPMENT_DATA_SCHEMA,
    "stage": LinkedCampaignStageEnvelope.SCHEMA,
}


def response_geometry_development_projection_input_slots() -> tuple[str, ...]:
    return tuple(sorted(f"{RETAINED_ANALYSIS_PREFIX}.{context}.r{root:02d}.project.r{view}.{output}"
        for context in ("assembling", "prepared") for root in range(64)
        for view in (1, 2) for output in DEVELOPMENT_PROJECTION_SCHEMAS))


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentProjectionCustodyInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-projection-custody-input'
    slot_id: str
    artifact: ArtifactIdentity
    relative_path: str
    task_receipt: ObjectIdentity
    manifest_sha256: str
    publication_commit_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.slot_id, field_name="slot_id")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.manifest_sha256, field_name="manifest_sha256")
        validate_sha256(self.publication_commit_sha256, field_name="publication_commit_sha256")
        task, output = self.slot_id.rsplit(".", 1)
        if (self.slot_id not in response_geometry_development_projection_input_slots()
                or self.artifact.artifact_id != f"artifact.{RETAINED_ANALYSIS_SOURCE_RUN}.{self.slot_id}"
                or self.artifact.payload_schema != DEVELOPMENT_PROJECTION_SCHEMAS[output]
                or not 0 < self.artifact.size_bytes <= 16 * 1024**2
                or self.artifact.media_type != ("application/x-hdf5" if output == "data" else CANONICAL_MEDIA_TYPE)
                or self.task_receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
                or self.task_receipt.object_id != f"receipt.{RETAINED_ANALYSIS_SOURCE_RUN}.{task}.attempt-001"
                or self.relative_path != f"runs/{RETAINED_ANALYSIS_SOURCE_RUN}/outputs/{task}/"
                    + ("data.h5" if output == "data" else f"{output}.canonical.json")):
            raise ValueError("development continuation substitutes a retained projection/custody identity")

    @property
    def imported_artifact_id(self) -> str:
        # Public input persistence publishes a new materialization. Its logical
        # record carries the original artifact/receipt as lineage, rather than
        # assigning changed lineage to the old immutable logical identity.
        return f"retained.{RETAINED_ANALYSIS_RUN}.{self.slot_id}"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAnalysisContinuationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-analysis-continuation-config'
    config_id: str
    qualification: ResponseGeometryDevelopmentQualificationConfig
    source_qualification_artifact: ArtifactIdentity
    projection_inputs: tuple[ResponseGeometryDevelopmentProjectionCustodyInput, ...]
    terminal_inventory_sha256: str
    failure_diagnosis_sha256: str
    owner_amendment: ObjectIdentity | None
    source_commit: str
    source_run_id: str = RETAINED_ANALYSIS_SOURCE_RUN
    run_id: str = RETAINED_ANALYSIS_RUN
    maximum_native_updates: int = 0
    grants_authority: bool = False
    input_materialization_rule: str = "NEW_IMPORT_ID_WITH_ORIGINAL_ARTIFACT_AND_RECEIPT_PARENTS"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.terminal_inventory_sha256, field_name="terminal_inventory_sha256")
        validate_sha256(self.failure_diagnosis_sha256, field_name="failure_diagnosis_sha256")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("development analysis requires an explicit source commit")
        require_sorted_unique_ids(self.projection_inputs, attribute="slot_id", field_name="projection_inputs")
        if (tuple(row.slot_id for row in self.projection_inputs) != response_geometry_development_projection_input_slots()
                or self.source_qualification_artifact.artifact_id != f"config-artifact.{self.qualification.config_id}"
                or self.source_qualification_artifact.payload_schema != self.qualification.SCHEMA
                or self.source_qualification_artifact.sha256 != self.qualification.fingerprint()
                or self.source_qualification_artifact.size_bytes != len(self.qualification.canonical_bytes())
                or self.source_run_id != RETAINED_ANALYSIS_SOURCE_RUN or self.run_id != RETAINED_ANALYSIS_RUN
                or type(self.maximum_native_updates) is not int or self.maximum_native_updates != 0
                or self.grants_authority is not False
                or self.input_materialization_rule != "NEW_IMPORT_ID_WITH_ORIGINAL_ARTIFACT_AND_RECEIPT_PARENTS"
                or (self.owner_amendment is not None and (
                    self.owner_amendment.object_id != f"{RETAINED_ANALYSIS_RUN}.amendment"
                    or self.owner_amendment.object_schema != 'empirical-lawhood/document/markdown'))):
            raise ValueError("development analysis correction changes its exact retained panel or authority ceiling")

    @property
    def method_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted((f"{RETAINED_ANALYSIS_PREFIX}.evaluate", *(f"{RETAINED_ANALYSIS_PREFIX}.{role}.{context}"
            for context in ("assembling", "prepared") for role in ("fit", "calibrate", "support", "assess")))))

    def inputs_for_task(self, task_id: str) -> tuple[ResponseGeometryDevelopmentProjectionCustodyInput, ...]:
        if task_id not in self.method_task_ids:
            raise ValueError("development analysis correction contains no acquisition/projection task")
        if task_id == f"{RETAINED_ANALYSIS_PREFIX}.evaluate":
            return ()
        role, context = task_id.rsplit(".", 2)[-2:]
        roots = {"fit": tuple(range(16)), "calibrate": tuple(range(16, 32)),
                 "support": tuple(range(32)), "assess": (*range(16), *range(32, 64))}[role]
        slots = {f"{RETAINED_ANALYSIS_PREFIX}.{context}.r{root:02d}.project.r{view}.{output}"
                 for root in roots for view in (1, 2) for output in DEVELOPMENT_PROJECTION_SCHEMAS}
        return tuple(row for row in self.projection_inputs if row.slot_id in slots)

    def artifact_id(self, run_id: str, slot_id: str, *, task_id: str | None = None) -> str:
        if run_id != self.run_id:
            raise ValueError("development retained measurements are bound to one proposed correction run")
        suffix = ".assessment" if task_id is not None and ".assess." in task_id else ""
        if slot_id == self.config_id:
            return slot_id + suffix
        retained = next((row for row in self.projection_inputs if row.slot_id == slot_id), None)
        if retained is not None:
            return retained.imported_artifact_id + suffix
        if slot_id.startswith("config-artifact."):
            return slot_id
        return f"artifact.{run_id}.{slot_id}"
