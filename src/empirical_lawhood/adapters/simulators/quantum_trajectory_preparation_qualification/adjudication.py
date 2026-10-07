"""Noncompensating preparation development/evaluation adjudication."""

from __future__ import annotations

from typing import Mapping, cast

from .types import Verdict


def development_handoff(
    result: Mapping[str, object],
    *,
    identities: Mapping[str, str],
    development_closeout_sha256: str,
) -> dict[str, object] | None:
    selected = result.get("selected_burnin")
    if not isinstance(selected, (int, float)):
        return None
    value = float(selected)
    return {
        "denominator_id": "strong",
        "selected_burnin": value,
        "sentinel_burnin": 2.0 * value,
        "window_endpoints": [value, value + 100.0, 2.0 * value, 2.0 * value + 100.0],
        "development_verdict": "STABLE_PREPARATION_WINDOW_SELECTED",
        "source_identity": identities["source_identity"],
        "config_identity": identities["config_identity"],
        "implementation_identity": identities["implementation_identity"],
        "development_closeout_sha256": development_closeout_sha256,
    }


def evaluation_verdict(result: Mapping[str, object]) -> str:
    intervals = result.get("intervals")
    memory = result.get("memory")
    if not isinstance(intervals, list) or not isinstance(memory, list):
        return Verdict.EVALUATION_UNEVALUABLE.value
    if len(intervals) != 200 or len(memory) != 294:
        return Verdict.EVALUATION_UNEVALUABLE.value
    if not all(isinstance(row, Mapping) for row in (*intervals, *memory)):
        return Verdict.EVALUATION_UNEVALUABLE.value
    interval_rows = cast(list[Mapping[str, object]], intervals)
    memory_rows = cast(list[Mapping[str, object]], memory)
    if any(row.get("validity") != "VALID" for row in interval_rows) or any(
        row.get("validity") != "VALID" for row in memory_rows
    ):
        return Verdict.EVALUATION_UNEVALUABLE.value
    if any(row.get("passed") is not True for row in interval_rows):
        return Verdict.FRESH_STATIONARITY_STOP.value
    if any(row.get("material") is True for row in memory_rows):
        return Verdict.PREPARATION_MEMORY_STOP.value
    return Verdict.QUALIFIED.value


__all__ = ["development_handoff", "evaluation_verdict"]
