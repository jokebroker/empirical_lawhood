"Outcome-blind construction of the exact current tokamak control replication action chart."

from __future__ import annotations

from empirical_lawhood.adapters.simulators.gym_torax_native.action_word import GYM_TORAX_FUTURE_IP_ACTION_WORD_ID, GYM_TORAX_LOWER_IP_ACTION_WORD_ID, GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID, GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID, build_gym_torax_action_word_chart
from empirical_lawhood.kernel.action_contracts import ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.prospective_config import ActionWordRecordedSemanticCompatibility, ProspectiveActionSemanticRole, ProspectiveSequentialActionWordChart


GYM_TORAX_ACTION_CHART_ID = 'action-chart.tokamak-control.four-word-current'

_ROLE_BY_WORD_ID = {
    GYM_TORAX_FUTURE_IP_ACTION_WORD_ID: ProspectiveActionSemanticRole.FUTURE_CAUSAL_FALSIFIER,
    GYM_TORAX_LOWER_IP_ACTION_WORD_ID: ProspectiveActionSemanticRole.ACTIVE,
    GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID: ProspectiveActionSemanticRole.NATIVE_HOLD,
    GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID: (ProspectiveActionSemanticRole.WRONG_SIGN_FALSIFIER),
}


def build_gym_torax_prospective_action_chart(
    *,
    historical_compatibility: tuple[ActionWordRecordedSemanticCompatibility, ...] | None = None,
) -> ProspectiveSequentialActionWordChart:
    """Bind explicit externally supplied provenance without inventing history."""

    if not isinstance(historical_compatibility, tuple) or len(historical_compatibility) != 4 or any(
        not isinstance(value, ActionWordRecordedSemanticCompatibility)
        for value in historical_compatibility
    ):
        raise ValueError("prospective action chart requires four explicit external historical semantic mappings")
    words = build_gym_torax_action_word_chart()
    first = words[0]
    current_words = {ObjectIdentity.from_record(word.word_id, word): word for word in words}
    if set(current_words) != {value.current_action_word for value in historical_compatibility}:
        raise ValueError("historical semantic mappings must bind the exact current target action bytes")
    if any(
        value.semantic_role is not _ROLE_BY_WORD_ID[current_words[value.current_action_word].word_id]
        for value in historical_compatibility
    ):
        raise ValueError("historical semantic mappings change a physical action role")
    return ProspectiveSequentialActionWordChart(
        chart_id=GYM_TORAX_ACTION_CHART_ID,
        action_words=words,
        historical_compatibility=historical_compatibility,
        common_denominator_id=first.denominator_id,
        common_retained_history_id=first.retained_history_id,
        common_receiver_id=first.receiver_id,
        common_horizon_id=first.horizon_id,
        required_support_status=ActionWordSupportStatus.SUPPORTED,
        required_reason_codes=(),
        source_qualification_required_before_issue=True,
        source_qualification_claimed_complete=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


__all__ = ['GYM_TORAX_ACTION_CHART_ID', 'build_gym_torax_prospective_action_chart']
