"Finite response-law qualification runtime inputs and committed outputs; existing owners decide qualification."

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import JointUncertaintyFamilyAssessment
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.candidate_payloads import (
    CandidatePayloadPublicationReceipt,
)

from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
from .calibration_operands import BOUNDARIES, PREDICTION_SCHEMA
from .calibration_records import FiniteResponseLawAssignedBoundaryCalibration, FiniteResponseLawBoundaryCalibration
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .science import PROGRAMME

CALIBRATE = f"{PROGRAMME}.calibration.calibrate"
FREEZE = f"{PROGRAMME}.calibration.freeze-method-inputs"
ADJUDICATE = f"{PROGRAMME}.calibration.adjudicate"
QUALIFY = f"{PROGRAMME}.calibration.qualify"
READOUT_SCHEMA = 'empirical-lawhood/methods/finite-response-law/calibration-readout'
REPORT_SCHEMA = 'empirical-lawhood/methods/finite-response-law/frozen-development-report'
COEFFICIENT_SCHEMA = (
    'empirical-lawhood/methods/finite-response-law/frozen-development-coefficients'
)
MANIFEST_SCHEMA = 'empirical-lawhood/methods/finite-response-law/frozen-development-manifest'


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-method-config'
    )
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawCalibrationConfig]] = FiniteResponseLawCalibrationConfig
    NATIVE_EVALUATION_TYPE: ClassVar[type[CanonicalRecord]] = (
        FiniteResponseLawCalibrationNativeEvaluation
    )
    config_id: str
    native_source: FiniteResponseLawCalibrationConfig
    native_evaluation: ArtifactIdentity
    native_receipt: ArtifactIdentity
    expected_native_receipt: ObjectIdentity
    development_report: ArtifactIdentity
    coefficients: ArtifactIdentity
    development_manifest: ArtifactIdentity

    def __post_init__(self) -> None:
        if (
            self.config_id != f"{PROGRAMME}.calibration.method-config"
            or type(self.native_source) is not self.SOURCE_TYPE
            or self.native_evaluation.payload_schema
            != self.NATIVE_EVALUATION_TYPE.SCHEMA
            or self.native_receipt.payload_schema != CanonicalTaskReceipt.SCHEMA
            or self.expected_native_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
            or self.native_receipt.sha256
            != self.expected_native_receipt.object_fingerprint
            or self.development_report.payload_schema != REPORT_SCHEMA
            or self.coefficients.payload_schema != COEFFICIENT_SCHEMA
            or self.development_manifest.payload_schema != MANIFEST_SCHEMA
            or len({a.artifact_id for a in self.inputs}) != 5
            or any(
                a.extensions or not 0 < a.size_bytes <= 8 * 1024**2 for a in self.inputs
            )
            or sum(a.size_bytes for a in self.inputs) > 16 * 1024**2
        ):
            raise ValueError(
                "Finite response-law qualification method changes its native/development input contracts"
            )

    @property
    def inputs(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            sorted(
                (
                    self.native_evaluation,
                    self.native_receipt,
                    self.development_report,
                    self.coefficients,
                    self.development_manifest,
                ),
                key=lambda a: a.artifact_id,
            )
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationMethodConfig(FiniteResponseLawCalibrationMethodConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-method-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawCalibrationConfig]] = FiniteResponseLawAssignedCalibrationConfig
    NATIVE_EVALUATION_TYPE: ClassVar[type[CanonicalRecord]] = (
        FiniteResponseLawAssignedCalibrationNativeEvaluation
    )
    native_source: FiniteResponseLawAssignedCalibrationConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-report'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawCalibrationMethodConfig]] = (
        FiniteResponseLawCalibrationMethodConfig
    )
    BOUNDARY_TYPE: ClassVar[type[FiniteResponseLawBoundaryCalibration]] = FiniteResponseLawBoundaryCalibration
    report_id: str
    config: ObjectIdentity
    prediction_artifact: ArtifactIdentity
    readout_artifact: ArtifactIdentity
    boundaries: tuple[FiniteResponseLawBoundaryCalibration, ...]
    joint_opportunities: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            self.report_id != f"{CALIBRATE}.result"
            or self.config.object_schema != self.CONFIG_TYPE.SCHEMA
            or self.prediction_artifact.payload_schema != PREDICTION_SCHEMA
            or self.readout_artifact.payload_schema != READOUT_SCHEMA
            or tuple(b.boundary for b in self.boundaries) != BOUNDARIES
            or any(type(b) is not self.BOUNDARY_TYPE for b in self.boundaries)
            or len({b.root_ids for b in self.boundaries}) != 1
            or any(
                b.prediction_artifact != self.prediction_artifact
                for b in self.boundaries
            )
            or tuple(b for b, _ in self.joint_opportunities) != BOUNDARIES
            or any(type(v) is not bool for _, v in self.joint_opportunities)
            or len({b.native_evaluation for b in self.boundaries}) != 1
            or len({b.native_task_receipt for b in self.boundaries}) != 1
        ):
            raise ValueError(
                "Finite response-law calibration report changes its committed four-boundary census"
            )

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.report_id, self)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationReport(FiniteResponseLawCalibrationReport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-report'
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawCalibrationMethodConfig]] = (
        FiniteResponseLawAssignedCalibrationMethodConfig
    )
    BOUNDARY_TYPE: ClassVar[type[FiniteResponseLawBoundaryCalibration]] = (
        FiniteResponseLawAssignedBoundaryCalibration
    )
    boundaries: tuple[FiniteResponseLawAssignedBoundaryCalibration, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawQualificationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-qualification-report'
    )
    CALIBRATION_TYPE: ClassVar[type[FiniteResponseLawCalibrationReport]] = FiniteResponseLawCalibrationReport
    report_id: str
    config: ObjectIdentity
    calibration: FiniteResponseLawCalibrationReport
    families: tuple[JointUncertaintyFamilyAssessment, ...]
    qualifications: tuple[LawQualificationResult, ...]
    publications: tuple[CandidatePayloadPublicationReceipt, ...]

    def __post_init__(self) -> None:
        if (
            self.report_id != f"{QUALIFY}.result"
            or type(self.calibration) is not self.CALIBRATION_TYPE
            or self.config != self.calibration.config
            or len(self.families) != 4
            or len(self.qualifications) != 4
            or len(self.publications) != 4
        ):
            raise ValueError(
                "Finite response-law qualification report requires every declared boundary result"
            )
        for boundary, calibration, family, result, publication in zip(
            BOUNDARIES,
            self.calibration.boundaries,
            self.families,
            self.qualifications,
            self.publications,
            strict=True,
        ):
            if (
                family.ledger.method_key != f"{PROGRAMME}.{boundary}-law"
                or len(family.assessments) != 1
                or family.assessments[0].candidate_evidence.payload_publication
                != publication
                or family.ledger.dataset_or_projection != calibration.identity
                or result.dataset_or_projection != calibration.identity
                or result.candidate_family_assessment
                != ObjectIdentity.from_record(family.assessment_id, family)
                or result.axis_map
                != ObjectIdentity.from_record(
                    family.ledger.axis_map.axis_map_id, family.ledger.axis_map
                )
            ):
                raise ValueError(
                    "Finite response-law qualification report detaches a sole-service result or publication"
                )

    @property
    def eligible_for_prospective_evaluation(self) -> bool:
        return (
            all(
                q.scientific_status is ScientificStatus.SUPPORTED
                for q in self.qualifications[:2]
            )
            and dict(self.calibration.joint_opportunities)["composed"]
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedQualificationReport(FiniteResponseLawQualificationReport):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-qualification-report'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CALIBRATION_TYPE: ClassVar[type[FiniteResponseLawCalibrationReport]] = (
        FiniteResponseLawAssignedCalibrationReport
    )
    calibration: FiniteResponseLawAssignedCalibrationReport
