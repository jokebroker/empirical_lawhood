"""Custody and separate prospective qualification and revealed terminal records."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import CandidateFamilyLedger, LawCandidateEvidence
from empirical_lawhood.adapters.methods.reactor_prefix_response.batch_calibration import ReactorForecastCalibration
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BRANCHES, native_task_id
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from .law import ReactorForecastPayload
from .science import PREFIX

QUALIFY = "forecast-qualification"
REVEAL = "forecast-adjudication"


@dataclass(frozen=True, slots=True)
class ReactorBatchCustody(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-reference-policy-forecast/reactor-batch-custody'
    # Ordered by the exact assigned unit/view roster, not artifact lexical order.
    manifests: tuple[ArtifactManifest, ...]
    receipts: tuple[CanonicalTaskReceipt, ...]

    def __post_init__(self) -> None:
        if (
            tuple(r.task_id for r in self.receipts) != tuple(native_task_id(*b) for b in BRANCHES)
            or len(self.manifests) != 84
        ):
            raise ValueError("batch custody must account for exactly 84 assigned native tasks")
        if (
            len({r.run_id for r in self.receipts}) != 1
            or len({r.implementation_commit for r in self.receipts}) != 1
        ):
            raise ValueError("batch custody cannot mix runs or implementation commits")
        for manifest, receipt in zip(self.manifests, self.receipts, strict=True):
            if (
                receipt.operational_status is not OperationalStatus.SUCCEEDED
                or receipt.output_materializations != (manifest.materialization,)
                or receipt.output_logical_artifacts != (manifest.logical,)
                or manifest.publication is None
                or manifest.logical.payload_schema != 'empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-trace'
                or manifest.logical.outcome_access is not OutcomeAccess.EVALUATION_SEALED
                or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            ):
                raise ValueError("batch custody changes native receipt/publication/visibility")


@dataclass(frozen=True, slots=True)
class ReactorForecastResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-reference-policy-forecast/reactor-forecast-result'
    design: ObjectIdentity
    calibration: ReactorForecastCalibration
    custody: ReactorBatchCustody
    payload: ReactorForecastPayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult

    def __post_init__(self) -> None:
        expected = tuple(
            t.object_fingerprint for s in self.calibration.unit_scores for t in s.traces
        )
        if (
            expected != tuple(m.logical.content_sha256 for m in self.custody.manifests)
            or self.payload.design != self.design
            or self.payload.bounds != self.calibration.bounds
            or self.payload.calibration
            != ObjectIdentity.from_record(f"{PREFIX}.calibration", self.calibration)
            or self.family.dataset_or_projection != self.payload.calibration
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
        ):
            raise ValueError("forecast result substitutes its exact qualified operands")

    @property
    def scientific_status(self) -> ScientificStatus:
        return self.qualification.scientific_status
