"""Exact retained projection custody for nonacquiring finite completion."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_retention import FiniteResponseLawEvaluationRetention
from ..assigned_native_records import FiniteResponseLawAssignedEvaluationViewObservation
from ..control_records import FiniteResponseLawControlConfig
from ..native_records import FiniteResponseLawCalibrationViewObservation
from ..science import PROGRAMME

FREEZE = f"{PROGRAMME}.calibration.freeze-retained-inputs"
AGGREGATE = f"{PROGRAMME}.native-evaluate"
ADJUDICATE = f"{PROGRAMME}.native-adjudicate"
PREFIX = f"{PROGRAMME}.calibration.retained-completion"
PROSPECTIVE_EVALUATION_COMPLETION_PREFIX = f"{PROGRAMME}.prospective-evaluation.retained-completion"
MAX_INPUT_BYTES = 8 * 1024**2


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRetainedProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/completion/finite-response-law-retained-projection'
    task_id: str
    report: ArtifactIdentity
    receipt: ArtifactIdentity
    expected_receipt: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        if (
            self.report.payload_schema
            not in (
                FiniteResponseLawCalibrationViewObservation.SCHEMA,
                FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
            )
            or self.receipt.payload_schema != CanonicalTaskReceipt.SCHEMA
            or self.expected_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
            or self.receipt.sha256 != self.expected_receipt.object_fingerprint
            or self.report.artifact_id == self.receipt.artifact_id
            or any(
                a.extensions or not 0 < a.size_bytes <= MAX_INPUT_BYTES
                for a in (self.report, self.receipt)
            )
        ):
            raise ValueError("Retained projection lacks its exact report/receipt identities")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRetainedCompletionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/completion/finite-response-law-retained-completion-config'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawCalibrationConfig]] = FiniteResponseLawCalibrationConfig
    VIEW_SCHEMA: ClassVar[str] = FiniteResponseLawCalibrationViewObservation.SCHEMA
    PREFIX_ID: ClassVar[str] = PREFIX
    config_id: str
    native_source: FiniteResponseLawCalibrationConfig
    retained_run_id: str
    retained_implementation_commit: str
    projections: tuple[FiniteResponseLawRetainedProjection, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.retained_run_id, field_name="retained_run_id")
        expected = tuple(
            f"{r.root_id}.flh-project.r{v}" for r in self.native_source.roots for v in (1, 2)
        )
        if (
            type(self.native_source) is not self.SOURCE_TYPE
            or self.config_id != f"{self.PREFIX_ID}.config"
            or any(row.report.payload_schema != self.VIEW_SCHEMA for row in self.projections)
            or len(self.retained_implementation_commit) != 40
            or any(c not in "0123456789abcdef" for c in self.retained_implementation_commit)
            or tuple(p.task_id for p in self.projections) != expected
            or len({a.artifact_id for a in self.inputs}) != 4 * len(self.native_source.roots)
            or sum(a.size_bytes for a in self.inputs) > 64 * 1024**2
            or len(self.canonical_bytes()) > 512 * 1024
        ):
            raise ValueError("Retained completion changes its original root/view custody census")

    @property
    def study_prefix(self) -> str:
        return self.PREFIX_ID

    @property
    def freeze_task_id(self) -> str:
        return (
            f"{PROSPECTIVE_EVALUATION_COMPLETION_PREFIX}.freeze-retained-inputs"
            if type(self) is FiniteResponseLawAssignedRetainedCompletionConfig
            else FREEZE
        )

    @property
    def aggregate_task_id(self) -> str:
        return (
            f"{PROSPECTIVE_EVALUATION_COMPLETION_PREFIX}.native-evaluate"
            if type(self) is FiniteResponseLawAssignedRetainedCompletionConfig
            else AGGREGATE
        )

    @property
    def adjudicate_task_id(self) -> str:
        return (
            f"{PROSPECTIVE_EVALUATION_COMPLETION_PREFIX}.native-adjudicate"
            if type(self) is FiniteResponseLawAssignedRetainedCompletionConfig
            else ADJUDICATE
        )

    @property
    def inputs(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            sorted(
                (a for p in self.projections for a in (p.report, p.receipt)),
                key=lambda a: a.artifact_id,
            )
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedRetainedCompletionConfig(FiniteResponseLawRetainedCompletionConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/completion/finite-response-law-assigned-retained-completion-config'
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawCalibrationConfig]] = FiniteResponseLawAssignedEvaluationConfig
    VIEW_SCHEMA: ClassVar[str] = FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA
    PREFIX_ID: ClassVar[str] = PROSPECTIVE_EVALUATION_COMPLETION_PREFIX
    native_source: FiniteResponseLawAssignedEvaluationConfig
    control: FiniteResponseLawControlConfig
    retention: FiniteResponseLawEvaluationRetention

    def __post_init__(self) -> None:
        FiniteResponseLawRetainedCompletionConfig.__post_init__(self)
        if (
            type(self.control) is not FiniteResponseLawControlConfig
            or type(self.retention) is not FiniteResponseLawEvaluationRetention
            or self.control.source != self.native_source
            or self.retention.source != self.native_source
            or self.retained_run_id == self.retention.donor_run_id
        ):
            raise ValueError("Independent evaluation completion loses its original source, control or donor")
