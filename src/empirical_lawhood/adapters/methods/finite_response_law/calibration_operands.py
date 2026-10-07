"Authenticated finite response-law qualification operands for the registered qualification task.\n\nNo fitting, source acquisition, task orchestration or terminal verdict lives\nhere. The provider commits these outputs through its existing runtime ports.\n"

import json
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any

import numpy as np

from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt

from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
from .assigned_prediction import AssignedPrediction, before_parent
from .calibration_analysis import _validate_prediction, calibration_scores
from .calibration_panel import CalibrationPanel, calibration_panel
from .calibration_readout import calibration_usability, two_future_diagnostics
from .calibration_records import FiniteResponseLawAssignedBoundaryCalibration, FiniteResponseLawBoundaryCalibration, boundary_calibration
from .frozen_package import frozen_development
from .law_payloads import uncalibrated_lower_prediction
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .science import PROGRAMME

BOUNDARIES = ("lower", "composed", "cached", "direct")
PREDICTION_SCHEMA = 'empirical-lawhood/methods/finite-response-law/calibration-predictions'


def authenticated_evaluation(
    raw: bytes,
    artifact: ArtifactIdentity,
    receipt: CanonicalTaskReceipt,
    *,
    expected_receipt: ObjectIdentity,
    source: FiniteResponseLawCalibrationConfig | FiniteResponseLawAssignedCalibrationConfig,
) -> FiniteResponseLawCalibrationNativeEvaluation | FiniteResponseLawAssignedCalibrationNativeEvaluation:
    """Reject missing/contradictory custody before constructing any root score.

    A successful completion task may contain authenticated failed native roots.
    Its scientific measurement status is deliberately not an all-roots gate.
    """
    evaluation_type = (
        FiniteResponseLawAssignedCalibrationNativeEvaluation
        if type(source) is FiniteResponseLawAssignedCalibrationConfig
        else FiniteResponseLawCalibrationNativeEvaluation
        if type(source) is FiniteResponseLawCalibrationConfig
        else None
    )
    if (
        evaluation_type is None
        or ObjectIdentity.from_record(receipt.receipt_id, receipt) != expected_receipt
        or receipt.task_id != f"{PROGRAMME}.native-evaluate"
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or artifact.payload_schema != evaluation_type.SCHEMA
        or len(raw) != artifact.size_bytes
        or sha256(raw).hexdigest() != artifact.sha256
    ):
        raise ValueError(
            "Finite response-law native completion custody is unaccounted or contradictory"
        )
    logical = tuple(
        v
        for v in receipt.output_logical_artifacts
        if v.payload_schema == artifact.payload_schema
    )
    physical = tuple(
        v
        for v in receipt.output_materializations
        if v.logical_artifact_id == artifact.artifact_id
    )
    if (
        len(logical) != 1
        or len(physical) != 1
        or logical[0].logical_artifact_id != artifact.artifact_id
        or logical[0].content_sha256 != artifact.sha256
        or logical[0].visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        or logical[0].outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        or physical[0].physical_sha256 != artifact.sha256
        or physical[0].size_bytes != len(raw)
        or physical[0].compression != "none"
        or physical[0].partition_selector is not None
    ):
        raise ValueError(
            "Finite response-law native artifact is detached from its authenticated completion receipt"
        )
    evaluation = decode_canonical_bytes(raw, evaluation_type, maximum_bytes=8 * 1024**2)
    if evaluation.config.projection.native_spec != source:
        raise ValueError("Finite response-law calibration native root/assignment specification differs")
    return evaluation


@dataclass(frozen=True)
class CalibrationOperands:
    """In-process output bundle; persisted scientific records have their own schemas."""

    native_evaluation: (
        FiniteResponseLawCalibrationNativeEvaluation | FiniteResponseLawAssignedCalibrationNativeEvaluation
    )
    panel: CalibrationPanel
    predictions: dict[str, AssignedPrediction]
    prediction_bytes: bytes
    prediction_artifact: ArtifactIdentity
    boundaries: (
        tuple[FiniteResponseLawBoundaryCalibration, ...]
        | tuple[FiniteResponseLawAssignedBoundaryCalibration, ...]
    )
    readout: dict[str, Any]


def restore_calibration_operands(
    *,
    evaluation: FiniteResponseLawCalibrationNativeEvaluation
    | FiniteResponseLawAssignedCalibrationNativeEvaluation,
    prediction_bytes: bytes,
    prediction_artifact: ArtifactIdentity,
    readout_bytes: bytes,
    readout_artifact: ArtifactIdentity,
    boundaries: tuple[FiniteResponseLawBoundaryCalibration, ...]
    | tuple[FiniteResponseLawAssignedBoundaryCalibration, ...],
) -> CalibrationOperands:
    """Read committed task outputs without fitting or recalibrating at qualification."""
    for raw, artifact in (
        (prediction_bytes, prediction_artifact),
        (readout_bytes, readout_artifact),
    ):
        if (
            len(raw) > 8 * 1024**2
            or len(raw) != artifact.size_bytes
            or sha256(raw).hexdigest() != artifact.sha256
        ):
            raise ValueError("Committed calibration bytes differ from their artifact")
    if tuple(b.boundary for b in boundaries) != BOUNDARIES or any(
        b.prediction_artifact != prediction_artifact
        or b.native_evaluation
        != ObjectIdentity.from_record(evaluation.evaluation_id, evaluation)
        for b in boundaries
    ):
        raise ValueError(
            "Committed calibration detaches its predictions/native evaluation"
        )
    panel = calibration_panel(evaluation)
    boundary_type = (
        FiniteResponseLawAssignedBoundaryCalibration
        if type(evaluation) is FiniteResponseLawAssignedCalibrationNativeEvaluation
        else FiniteResponseLawBoundaryCalibration
        if type(evaluation) is FiniteResponseLawCalibrationNativeEvaluation
        else None
    )
    if boundary_type is None or any(
        type(boundary) is not boundary_type or boundary.root_ids != panel.root_ids
        for boundary in boundaries
    ):
        raise ValueError(
            "Committed calibration changes its assigned physical-root cohort"
        )
    expected = {"assigned_parent_indices", "predicted_handoff"} | {
        f"{b}_{field}" for b in BOUNDARIES for field in ("mean", "sigma", "supported")
    }
    from .array_transport import decode_npz_transport

    array_bytes = (
        decode_npz_transport(prediction_bytes, PREDICTION_SCHEMA)
        if prediction_artifact.media_type == "application/json"
        else prediction_bytes
    )
    with np.load(BytesIO(array_bytes), allow_pickle=False) as archive:
        infos = archive.zip.infolist()
        if (
            len(infos) != len(expected)
            or set(archive.files) != expected
            or sum(i.file_size for i in infos) > 1024**2
        ):
            raise ValueError("Committed predictions change their bounded array census")
        parents = archive["assigned_parent_indices"]
        handoff = archive["predicted_handoff"]
        if (
            parents.dtype != np.int64
            or not np.array_equal(parents, panel.parent_indices)
            or handoff.dtype != np.float64
            or handoff.shape != (32, 24)
        ):
            raise ValueError("Committed predictions change their assigned-parent axes")
        predictions = {
            b: AssignedPrediction(
                *(archive[f"{b}_{f}"] for f in ("mean", "sigma", "supported"))
            )
            for b in BOUNDARIES
        }
        for prediction in predictions.values():
            _validate_prediction(prediction)
    readout = json.loads(readout_bytes)
    if not isinstance(readout, dict) or set(readout) != {
        "two_future_diagnostics",
        "usability",
    }:
        raise ValueError("Committed calibration readout changes its declared products")
    return CalibrationOperands(
        evaluation,
        panel,
        predictions,
        prediction_bytes,
        prediction_artifact,
        boundaries,
        readout,
    )


def calibration_operands(
    *,
    native_bytes: bytes,
    native_artifact: ArtifactIdentity,
    native_receipt: CanonicalTaskReceipt,
    expected_native_receipt: ObjectIdentity,
    source: FiniteResponseLawCalibrationConfig | FiniteResponseLawAssignedCalibrationConfig,
    development_report_bytes: bytes,
    coefficient_bytes: bytes,
    development_report: ArtifactIdentity,
    coefficients: ArtifactIdentity,
    development_manifest: ArtifactIdentity,
    prediction_artifact_id: str = f"{PROGRAMME}.calibration.predictions",
) -> CalibrationOperands:
    """Run the frozen four-boundary calculation once on the full assigned census.

    The manifest and development input identities are supplied by the issued
    method configuration. Its provider authenticates the manifest bytes and
    pre-calibration lineage; this function authenticates the operands it reads.
    """
    evaluation = authenticated_evaluation(
        native_bytes,
        native_artifact,
        native_receipt,
        expected_receipt=expected_native_receipt,
        source=source,
    )
    points, widths = frozen_development(
        report_bytes=development_report_bytes,
        coefficient_bytes=coefficient_bytes,
        report_identity=development_report,
        coefficient_identity=coefficients,
    )
    panel = calibration_panel(evaluation)
    pre = before_parent(
        points,
        {b: widths[b] for b in BOUNDARIES[1:]},
        panel.prefix,
        panel.parent_indices,
    )
    predictions = {
        "lower": uncalibrated_lower_prediction(
            points, widths["lower"], panel.handoff[:, :, 0]
        ),
        "composed": pre.composed,
        "cached": pre.cached,
        "direct": pre.direct,
    }
    arrays = {
        "assigned_parent_indices": panel.parent_indices,
        "predicted_handoff": pre.predicted_interface,
    }
    for boundary, prediction in predictions.items():
        for field in ("mean", "sigma", "supported"):
            arrays[f"{boundary}_{field}"] = getattr(prediction, field)
    encoded = BytesIO()
    np.savez_compressed(encoded, allow_pickle=False, **arrays)
    from .array_transport import encode_npz_transport

    raw = encode_npz_transport(encoded.getvalue(), PREDICTION_SCHEMA)
    artifact = ArtifactIdentity(
        prediction_artifact_id,
        "frozen-calibration-predictions",
        PREDICTION_SCHEMA,
        sha256(raw).hexdigest(),
        "application/json",
        len(raw),
    )
    native_id = ObjectIdentity.from_record(evaluation.evaluation_id, evaluation)
    records = []
    usability = {}
    for boundary, prediction in predictions.items():
        score = calibration_scores(panel, prediction)
        records.append(
            boundary_calibration(
                score,
                prediction,
                calibration_id=f"{PROGRAMME}.calibration.{boundary}.calibration",
                boundary=boundary,
                development_manifest=development_manifest,
                frozen_coefficients=coefficients,
                native_evaluation=native_id,
                native_task_receipt=expected_native_receipt,
                prediction_artifact=artifact,
                root_ids=panel.root_ids,
            )
        )
        usability[boundary] = calibration_usability(prediction, score.q, panel.root_ids, request_seeds=panel.request_seeds)
    return CalibrationOperands(
        evaluation,
        panel,
        predictions,
        raw,
        artifact,
        tuple(records),
        {
            "two_future_diagnostics": two_future_diagnostics(panel, predictions),
            "usability": usability,
        },
    )
