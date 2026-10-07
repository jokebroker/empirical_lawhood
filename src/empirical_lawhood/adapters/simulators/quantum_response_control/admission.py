"""Noncompensating nine-gate admission and finite one-step reachability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from .contracts import Action


class GateKind(StrEnum):
    TARGET = "TARGET"
    PHYSICAL_SINK = "PHYSICAL_SINK"
    EFFORT = "EFFORT"
    OBSERVATION_VALIDITY = "OBSERVATION_VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    BASELINE_PRESERVATION = "BASELINE_PRESERVATION"
    DYNAMICS = "DYNAMICS"
    REACHABILITY = "REACHABILITY"
    AUTHORITY = "AUTHORITY"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"


class AdmissionStatus(StrEnum):
    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class GateResult:
    kind: GateKind
    status: GateStatus
    margin: float | None
    reason_codes: tuple[str, ...]
    evidence_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status is GateStatus.PASS and self.reason_codes:
            raise ValueError("passing admission gate retains failure reasons")
        if self.status is not GateStatus.PASS and not self.reason_codes:
            raise ValueError("nonpassing admission gate requires reasons")


@dataclass(frozen=True, slots=True)
class CandidateAdmission:
    cell_id: str
    chart_id: str
    action: Action
    model_member_id: str
    support_passed: bool
    atlas_gap: bool
    gates: tuple[GateResult, ...]
    status: AdmissionStatus
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReachabilityAssessment:
    cell_id: str
    action: Action
    admitted: bool
    native_action_available: bool
    accepted_realizable: bool
    viable_direction: bool
    latency_passed: bool
    history_conflict: bool
    model_members_passed: bool
    authority_passed: bool
    reachable: bool
    reason_codes: tuple[str, ...]


def intersect_gates(
    *,
    cell_id: str,
    chart_id: str,
    action: Action,
    model_member_id: str,
    support_passed: bool,
    atlas_gap: bool,
    gates: Sequence[GateResult],
) -> CandidateAdmission:
    kinds = [gate.kind for gate in gates]
    if set(kinds) != set(GateKind) or len(kinds) != len(set(kinds)):
        raise ValueError("admission does not contain exactly the nine gates")
    reasons: list[str] = []
    if not support_passed:
        reasons.append("outside-support")
    if atlas_gap:
        reasons.append("atlas-gap")
    failed = [gate for gate in gates if gate.status is GateStatus.FAIL]
    unresolved = [gate for gate in gates if gate.status is GateStatus.UNEVALUABLE]
    for gate in (*failed, *unresolved):
        reasons.extend(gate.reason_codes)
    if not support_passed or atlas_gap or failed:
        status = AdmissionStatus.REJECTED
    elif unresolved:
        status = AdmissionStatus.UNEVALUABLE
    else:
        status = AdmissionStatus.ADMITTED
    return CandidateAdmission(
        cell_id=cell_id,
        chart_id=chart_id,
        action=action,
        model_member_id=model_member_id,
        support_passed=support_passed,
        atlas_gap=atlas_gap,
        gates=tuple(sorted(gates, key=lambda gate: gate.kind.value)),
        status=status,
        reason_codes=tuple(sorted(set(reasons))),
    )


def finite_reachability(
    admission: CandidateAdmission,
    *,
    native_action_available: bool,
    accepted_realizable: bool,
    viable_direction: bool,
    latency_passed: bool,
    history_conflict: bool,
    model_members_passed: bool,
    authority_passed: bool,
) -> ReachabilityAssessment:
    checks: Mapping[str, bool] = {
        "not-admitted": admission.status is AdmissionStatus.ADMITTED,
        "native-action-unavailable": native_action_available,
        "action-not-realizable": accepted_realizable,
        "response-direction-not-viable": viable_direction,
        "latency-deadline-missed": latency_passed,
        "action-history-conflict": not history_conflict,
        "model-member-disagreement": model_members_passed,
        "authority-missing": authority_passed,
    }
    reasons = tuple(sorted(reason for reason, passed in checks.items() if not passed))
    return ReachabilityAssessment(
        cell_id=admission.cell_id,
        action=admission.action,
        admitted=admission.status is AdmissionStatus.ADMITTED,
        native_action_available=native_action_available,
        accepted_realizable=accepted_realizable,
        viable_direction=viable_direction,
        latency_passed=latency_passed,
        history_conflict=history_conflict,
        model_members_passed=model_members_passed,
        authority_passed=authority_passed,
        reachable=not reasons,
        reason_codes=reasons,
    )


def gate(
    kind: GateKind,
    passed: bool | None,
    *,
    margin: float | None = None,
    reason: str,
    evidence_identities: Sequence[str] = (),
) -> GateResult:
    status = (
        GateStatus.PASS
        if passed is True
        else GateStatus.FAIL
        if passed is False
        else GateStatus.UNEVALUABLE
    )
    return GateResult(
        kind=kind,
        status=status,
        margin=margin,
        reason_codes=() if passed is True else (reason,),
        evidence_identities=tuple(sorted(set(evidence_identities))),
    )


__all__ = [
    "AdmissionStatus",
    "CandidateAdmission",
    "GateKind",
    "GateResult",
    "GateStatus",
    "ReachabilityAssessment",
    "finite_reachability",
    "gate",
    "intersect_gates",
]
