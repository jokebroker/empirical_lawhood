"""Closed current publications for original F, exposure inspection and Q transfer."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationReport
from empirical_lawhood.adapters.methods.preparation_applicability.conformance import PreparationApplicabilityNativeConformance
from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PreparationApplicabilityResultAccess, PREPARATION_PROTECTED_OUTCOMES
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids, public_historical_exposure
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_strings, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt, ArtifactWriter


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityExposureInspection(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/exposure-inspection"
    inspection_id: str
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]
    inspected_sources: tuple[ArtifactIdentity, ...]
    prior_allocations: tuple[MatrixAllocation, ...] = ()

    def __post_init__(self):
        validate_stable_id(self.inspection_id)
        require_sorted_unique_strings(self.excluded_unit_ids, field_name="excluded_unit_ids", allow_empty=False)
        require_sorted_unique_strings(self.excluded_seed_ids, field_name="excluded_seed_ids", allow_empty=False)
        if not self.inspected_sources or len(self.inspected_sources) > 256:
            raise ValueError("exposure inspection requires bounded explicitly inspected source identities")
        known_units,known_seeds=public_historical_exposure()
        expected_units=tuple(sorted(set(known_units).union(root.root_id for allocation in self.prior_allocations for root in allocation.roots)))
        expected_seeds=tuple(sorted(set(known_seeds).union(seed for allocation in self.prior_allocations for seed in effective_seed_ids(allocation))))
        if self.excluded_unit_ids!=expected_units or self.excluded_seed_ids!=expected_seeds:
            raise ValueError("exposure inspection omits known public or supplied actual effective allocations")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityImportReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/import-receipt"
    receipt_id: str
    key: str
    payload_manifest: ArtifactManifest

    def __post_init__(self):
        validate_stable_id(self.receipt_id)
        if self.key not in ("lower", "exposure", "conformance") or self.payload_manifest.publication is None:
            raise ValueError("import receipt requires a committed current input publication")


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityPublishedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/published-input"
    key: str
    record: OriginalFiniteResponseLaw | ConstructedPreparationReport | PreparationApplicabilityExposureInspection | PreparationApplicabilityNativeConformance
    manifest: ArtifactManifest
    receipt: CanonicalTaskReceipt | PreparationApplicabilityImportReceipt
    receipt_manifest: ArtifactManifest | None = None
    result_access: PreparationApplicabilityResultAccess | None = None

    def __post_init__(self):
        kinds = {"lower": OriginalFiniteResponseLaw, "qualification": ConstructedPreparationReport,
                 "exposure": PreparationApplicabilityExposureInspection,"conformance": PreparationApplicabilityNativeConformance}
        logical, physical = self.manifest.logical, self.manifest.materialization
        raw = self.record.canonical_bytes()
        if (self.key not in kinds or not isinstance(self.record, kinds[self.key])
            or self.manifest.publication is None
            or logical.payload_schema != self.record.SCHEMA
            or physical.physical_sha256 != self.record.fingerprint()
            or physical.size_bytes != len(raw)):
            raise ValueError("current input differs from its exact committed publication")
        if self.key == "qualification":
            if (not isinstance(self.receipt, CanonicalTaskReceipt) or self.receipt_manifest is not None
                or self.result_access is None or self.result_access.run_id != self.receipt.run_id
                or self.record.phase != "Q"
                or logical.outcome_access not in PREPARATION_PROTECTED_OUTCOMES
                or logical not in self.receipt.output_logical_artifacts
                or physical not in self.receipt.output_materializations
                or self.receipt.operational_status is not OperationalStatus.SUCCEEDED
                or not self.receipt.checks or self.receipt.task_id != "ap.report"):
                raise ValueError("Q requires its actual successful exact report task receipt")
        elif (self.result_access is not None or logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
              or not isinstance(self.receipt, PreparationApplicabilityImportReceipt)
              or self.receipt.key != self.key or self.receipt.payload_manifest != self.manifest
              or self.receipt_manifest is None or self.receipt_manifest.publication is None
              or self.receipt_manifest.logical.payload_schema != self.receipt.SCHEMA
              or self.receipt_manifest.materialization.physical_sha256 != self.receipt.fingerprint()
              or self.receipt_manifest.materialization.size_bytes != len(self.receipt.canonical_bytes())):
            raise ValueError("imported original F/exposure needs its separate committed import receipt")

    def authenticate(self, writer: ArtifactWriter, receipt_store, *, result_access_authenticator=None) -> None:
        if self.key == "qualification":
            if result_access_authenticator is None:
                raise PermissionError("current Q publication requires actual issued run and reveal validation")
            result_access_authenticator(context=self.result_access, receipt=self.receipt,
                logical=self.manifest.logical, writer=writer)
            retained = receipt_store.read_by_receipt_id(self.receipt.run_id, self.receipt.task_id, self.receipt.receipt_id)
            if retained != self.receipt:
                raise ValueError("current Q receipt is absent or differs from committed custody")
        else:
            writer.verify_manifest(self.receipt_manifest)
        writer.verify_manifest(self.manifest)

    def require_stage(self, stage) -> None:
        if self.key == "conformance":
            return
        if self.key == "exposure":
            inspection = self.record
            if (stage.exposure.inspection != ObjectIdentity.from_record(inspection.inspection_id, inspection)
                or stage.exposure.custody_receipt != ObjectIdentity.from_record(self.receipt.receipt_id, self.receipt)
                or stage.exposure.excluded_unit_ids != inspection.excluded_unit_ids
                or stage.exposure.excluded_seed_ids != inspection.excluded_seed_ids):
                raise ValueError("stage substitutes its authenticated exposure inspection")
            stage.exposure.check(stage.allocation)
            return
        prior = next(value for value in stage.upstream if value.key == self.key)
        a = prior.artifact
        if (a.artifact_id != self.manifest.logical.logical_artifact_id
            or a.payload_schema != self.record.SCHEMA
            or a.sha256 != self.record.fingerprint() or a.size_bytes != len(self.record.canonical_bytes())
            or a.media_type != self.manifest.logical.media_type
            or prior.source_run_id != (self.receipt.run_id if isinstance(self.receipt, CanonicalTaskReceipt) else self.receipt.receipt_id)
            or prior.custody_receipt != ObjectIdentity.from_record(self.receipt.receipt_id, self.receipt)):
            raise ValueError("stage substitutes its exact current upstream publication")
