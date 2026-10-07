"""Finite words shared by ordinary and constructed applicability wrappers.

This roster imports declarations only. Neither acquisition nor a learned-policy
provider is needed to interpret a signed response panel.
"""

from empirical_lawhood.adapters.simulators.prepared_response.contracts import (
    PreparedForceWord,
    prepared_words,
)

WORDS: tuple[PreparedForceWord, ...] = tuple(
    word for word in prepared_words() if word.direction_index in (0, 1)
)

from empirical_lawhood.adapters.methods.preparation_applicability.records import MENU
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULES

SCHEDULES = tuple(next(schedule for schedule in PREPARATION_POLICY_SCHEDULES if schedule.schedule_id == name) for name in MENU)
