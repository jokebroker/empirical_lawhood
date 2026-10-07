"""Strict generic controller dispositions for the sole current route.

Scientific HOLD is never an operational fallback.  The only HOLD commitment is
an independently qualified measured native action carried by a compiled
programme.  Operational recovery and termination remain separate states.
"""

from __future__ import annotations

from enum import StrEnum


class ScientificCommitmentKind(StrEnum):
    ACTION = "ACTION"
    MEASURED_HOLD = "MEASURED_HOLD"
    NONATTEMPT = "NONATTEMPT"


class OperationalDeliveryState(StrEnum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    DELIVERY_RECOVERY = "DELIVERY_RECOVERY"
    TERMINATED = "TERMINATED"


__all__ = ["OperationalDeliveryState", "ScientificCommitmentKind"]
