"""Identity context for rechecking current protected preparation outputs."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.kernel.evidence import OutcomeAccess

PREPARATION_PROTECTED_OUTCOMES=(OutcomeAccess.EVALUATION_SEALED,OutcomeAccess.EVALUATOR_REVEAL,OutcomeAccess.EVALUATION_REVEALED)


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityResultAccess(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/result-access"
    run_id: str
    issued_study: ObjectIdentity
    execution_authority: ObjectIdentity
    reveal_authority: ObjectIdentity

    def __post_init__(self):
        validate_stable_id(self.run_id)
        if (self.execution_authority.object_schema != "empirical-lawhood/planning/study-operation-authority"
            or self.reveal_authority.object_schema != self.execution_authority.object_schema):
            raise ValueError("preparation result access requires current programme operation grants")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityRetainedResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/retained-result"
    # Payload text preserves the exact current family schema without importing
    # a learned fitter or depending on a historical compatibility decoder.
    record_payload: str
    manifest: ArtifactManifest
    receipt: CanonicalTaskReceipt
    result_access: PreparationApplicabilityResultAccess

    def __post_init__(self):
        from hashlib import sha256
        from empirical_lawhood.kernel.status import OperationalStatus
        raw=self.record_payload.encode("utf-8")
        if (len(raw)>16*1024**2 or self.manifest.publication is None
            or self.manifest.logical.outcome_access not in PREPARATION_PROTECTED_OUTCOMES
            or self.manifest.logical not in self.receipt.output_logical_artifacts
            or self.manifest.materialization not in self.receipt.output_materializations
            or self.manifest.materialization.physical_sha256!=sha256(raw).hexdigest()
            or self.manifest.materialization.size_bytes!=len(raw)
            or self.receipt.task_id!="ap.report" or self.receipt.operational_status is not OperationalStatus.SUCCEEDED
            or not self.receipt.checks or self.result_access.run_id!=self.receipt.run_id):
            raise ValueError("retained preparation result changes its exact report/task publication")
