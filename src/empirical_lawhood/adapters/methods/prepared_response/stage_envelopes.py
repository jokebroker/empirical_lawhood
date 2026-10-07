"Deterministic fresh calibration stage envelopes shared by method and source providers."

from empirical_lawhood.adapters.simulators.prepared_response.policy_native import PreparedResponseCalibrationNativeTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import envelope
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope

from .policy_decision import PreparedParentDecision


def prepared_parent_decision_envelope(
    result: PreparedParentDecision,
) -> LinkedCampaignStageEnvelope:
    return envelope(
        result.decision_id,
        result,
        result.decision_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        result.disposition == "DECIDED",
        result.reason_codes,
    )


def prepared_response_calibration_native_stage_envelope(
    result: PreparedResponseCalibrationNativeTaskResult,
) -> LinkedCampaignStageEnvelope:
    reasons = (
        (result.unentered_reason,)
        if result.unentered_reason is not None
        else tuple(sorted({
            view.delivery.reason for view in result.native_pair.views
            if view.delivery.reason is not None
        }))
        if result.native_pair is not None
        else ()
    )
    return envelope(
        result.invocation.task_id,
        result,
        result.result_id,
        LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
        result.native_complete,
        reasons,
    )
