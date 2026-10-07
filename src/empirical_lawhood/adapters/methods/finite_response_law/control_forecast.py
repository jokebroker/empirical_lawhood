"Native-prefix projection and the complete request-blind finite response-law evaluation forecast task."

from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawEvaluationTaskResult, decode_task_native
from .control_plan import FiniteResponseLawControlLawContext
from .control_prediction import predict_control_table
from .control_records import FiniteResponseLawControlConfig, FiniteResponseLawRootForecast, FiniteResponseLawUnavailableControlBoundary
from .evaluation_native_records import FiniteResponseLawEvaluationInterface
from .method_records import FiniteResponseLawQualificationReport
from .native_projection import _calibration_interface
from .science import PARENTS


def forecast_root(
    *,
    config: FiniteResponseLawControlConfig,
    report: FiniteResponseLawQualificationReport,
    contexts: tuple[FiniteResponseLawControlLawContext, ...],
    prefix: FiniteResponseLawEvaluationTaskResult,
    prefix_bytes: bytes,
    prefix_artifacts: tuple[ArtifactIdentity, ...],
    payload_reader: CandidatePayloadReader,
) -> FiniteResponseLawRootForecast:
    """No request, actual handoff, or future input exists at this boundary."""
    source = config.source
    if (
        prefix.invocation.phase != "prefix"
        or prefix.invocation.root not in source.roots
        or prefix.invocation.source != ObjectIdentity.from_record(source.spec_id, source)
        or report.fingerprint() != config.qualification.sha256
        or not report.eligible_for_prospective_evaluation
    ):
        raise ValueError("Finite response-law evaluation prediction changes the issued prefix or frozen qualification")
    pair = decode_task_native(prefix, prefix_bytes)
    # This is the sole existing causal instrument, using PRIMARY input only.
    interface = cast(
        FiniteResponseLawEvaluationInterface | None,
        _calibration_interface(
            None if pair is None or not prefix.native_complete else pair[0],
            prefix,
        ),
    )
    boundaries = tuple(b.boundary for b in report.calibration.boundaries)
    supported = {
        b
        for b in ("composed", "cached", "direct")
        if report.qualifications[boundaries.index(b)].scientific_status
        is ScientificStatus.SUPPORTED
    }
    by_qualification = {
        ObjectIdentity.from_record(c.qualification.result_id, c.qualification): c for c in contexts
    }
    if len(by_qualification) != len(contexts) or len(contexts) != len(supported):
        raise ValueError("Finite response-law evaluation prediction context omits or invents a qualified boundary")
    tables = []
    unavailable = []
    for boundary in ("cached", "composed", "direct"):
        qualification = report.qualifications[boundaries.index(boundary)]
        if boundary not in supported:
            unavailable.append(FiniteResponseLawUnavailableControlBoundary(boundary, qualification))
            continue
        context = by_qualification[
            ObjectIdentity.from_record(qualification.result_id, qualification)
        ]
        tables.append(
            predict_control_table(
                context=context,
                report=report,
                boundary=boundary,
                root_id=prefix.invocation.root.stage_unit,
                prefix=None if interface is None else interface.values,
                prefix_source=ObjectIdentity.from_record(prefix.result_id, prefix),
                prefix_artifacts=prefix_artifacts,
                parent_index=PARENTS.index(prefix.invocation.root.assigned_parent),
                payload_reader=payload_reader,
            )
        )
    return FiniteResponseLawRootForecast(
        prefix.invocation.root,
        ObjectIdentity.from_record(config.config_id, config),
        ObjectIdentity.from_record(report.report_id, report),
        prefix,
        interface,
        tuple(tables),
        tuple(unavailable),
    )
