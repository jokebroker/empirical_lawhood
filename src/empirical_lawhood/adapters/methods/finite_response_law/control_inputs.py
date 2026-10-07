"Authenticate the existing finite response-law qualification owners before constructing any finite response-law evaluation consumer."

from hashlib import sha256

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from .control_records import FiniteResponseLawControlConfig
from .method_records import FiniteResponseLawAssignedQualificationReport, FiniteResponseLawQualificationReport, QUALIFY
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
from .calibration_operands import authenticated_evaluation


def authenticated_control_inputs(
    config: FiniteResponseLawControlConfig,
    inputs: dict[str, bytes],
) -> tuple[FiniteResponseLawQualificationReport, FiniteResponseLawCalibrationNativeEvaluation]:
    """Reject contradictory custody; never substitute a synthetic qualification."""
    if set(inputs) != {a.artifact_id for a in config.inputs}:
        raise ValueError("Finite response-law evaluation prior input roster differs from its frozen declaration")
    assigned = type(config.source) is FiniteResponseLawAssignedEvaluationConfig
    report_type = FiniteResponseLawAssignedQualificationReport if assigned else FiniteResponseLawQualificationReport
    native_type = FiniteResponseLawAssignedCalibrationNativeEvaluation if assigned else FiniteResponseLawCalibrationNativeEvaluation
    for artifact in config.inputs:
        raw = inputs[artifact.artifact_id]
        if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
            raise ValueError("Finite response-law evaluation prior input bytes differ from their declared identity")
    receipt = decode_canonical_bytes(
        inputs[config.qualification_receipt.artifact_id],
        CanonicalTaskReceipt,
        maximum_bytes=8 * 1024**2,
    )
    if (
        ObjectIdentity.from_record(receipt.receipt_id, receipt)
        != config.expected_qualification_receipt
        or receipt.task_id != QUALIFY
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
    ):
        raise ValueError("Finite response-law evaluation lacks its actual sole-qualification receipt")
    artifact = config.qualification
    logical = tuple(
        a for a in receipt.output_logical_artifacts if a.payload_schema == artifact.payload_schema
    )
    physical = tuple(
        a for a in receipt.output_materializations if a.logical_artifact_id == artifact.artifact_id
    )
    if (
        len(logical) != 1
        or len(physical) != 1
        or logical[0].logical_artifact_id != artifact.artifact_id
        or logical[0].content_sha256 != artifact.sha256
        or logical[0].visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        or logical[0].outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        or physical[0].physical_sha256 != artifact.sha256
        or physical[0].size_bytes != artifact.size_bytes
        or physical[0].compression != "none"
        or physical[0].partition_selector is not None
    ):
        raise ValueError("Finite response-law evaluation prior qualification is detached from its authoritative receipt")
    report = decode_canonical_bytes(
        inputs[artifact.artifact_id], report_type, maximum_bytes=8 * 1024**2
    )
    prior = decode_canonical_bytes(
        inputs[config.calibration_native.artifact_id],
        native_type,
        maximum_bytes=8 * 1024**2,
    )
    native_receipt = decode_canonical_bytes(
        inputs[config.calibration_native_receipt.artifact_id],
        CanonicalTaskReceipt,
        maximum_bytes=8 * 1024**2,
    )
    native = authenticated_evaluation(
        inputs[config.calibration_native.artifact_id],
        config.calibration_native,
        native_receipt,
        expected_receipt=config.expected_calibration_native_receipt,
        source=prior.config.projection.native_spec,
    )
    if (
        not report.eligible_for_prospective_evaluation
        or any(
            b.native_evaluation != ObjectIdentity.from_record(native.evaluation_id, native)
            or b.native_task_receipt != config.expected_calibration_native_receipt
            for b in report.calibration.boundaries
        )
        or config.source.science != native.config.projection.native_spec.science
    ):
        raise ValueError("Finite response-law evaluation changes the frozen science or lacks genuine primary eligibility")
    return report, native
