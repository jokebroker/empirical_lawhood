"""Typed, non-promotable FreeGSNKE source/action qualification result."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetEpisodeStatus, FreeGsnkeTargetProcessResponse


class FreeGsnkeSourceQualificationDisposition(StrEnum):
    PASS = "PASS"
    LINEARIZATION_DOMAIN_STOP = "LINEARIZATION_DOMAIN_STOP"
    DYNAMIC_GS_CONVERGENCE_STOP = "DYNAMIC_GS_CONVERGENCE_STOP"
    SOURCE_ACTION_STOP = "SOURCE_ACTION_STOP"
    OPERATIONAL_UNEVALUABLE = "OPERATIONAL_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class FreeGsnkeSourceQualificationConfig(CanonicalRecord):
    """Exact excluded qualification roster frozen before branch execution."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-source-qualification-config'

    config_id: str
    qualification_id: str
    source_binding: ObjectIdentity
    saved_preparation: ObjectIdentity
    requests: tuple[ObjectIdentity, ...]
    required_branch_ids: tuple[str, ...]
    excluded_from_target_evidence: bool
    target_outcome_access_count_at_freeze: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        if self.source_binding.object_schema != FreeGsnkeSourceBinding.SCHEMA:
            raise ValueError("FreeGSNKE qualification source identity differs")
        if self.saved_preparation.object_schema != FreeGsnkeSavedPreparation.SCHEMA:
            raise ValueError("FreeGSNKE qualification saved preparation differs")
        require_sorted_unique_ids(
            self.requests,
            attribute="object_id",
            field_name="requests",
        )
        require_sorted_unique_strings(
            self.required_branch_ids,
            field_name="required_branch_ids",
        )
        if not self.requests or len(self.requests) != len(self.required_branch_ids):
            raise ValueError("FreeGSNKE qualification branch/request roster differs")
        if not self.excluded_from_target_evidence or self.target_outcome_access_count_at_freeze:
            raise ValueError("FreeGSNKE source qualification crossed target evidence")


@dataclass(frozen=True, slots=True)
class FreeGsnkeSourceQualificationResult(CanonicalRecord):
    """Complete issued-branch accounting with adverse source stops retained."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-source-qualification-result'

    result_id: str
    qualification: ObjectIdentity
    responses: tuple[ObjectIdentity, ...]
    disposition: FreeGsnkeSourceQualificationDisposition
    planned_branch_count: int
    observed_complete_branch_count: int
    adverse_branch_count: int
    operational_failure_branch_count: int
    linearization_domain_stop_count: int
    dynamic_gs_convergence_stop_count: int
    other_adverse_count: int
    all_issued_branches_accounted: bool
    reason_codes: tuple[str, ...]
    excluded_from_target_evidence: bool
    target_evidence_unit_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.qualification.object_schema != FreeGsnkeSourceQualificationConfig.SCHEMA:
            raise ValueError("FreeGSNKE qualification result identity differs")
        require_sorted_unique_ids(
            self.responses,
            attribute="object_id",
            field_name="responses",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        counts = (
            self.planned_branch_count,
            self.observed_complete_branch_count,
            self.adverse_branch_count,
            self.operational_failure_branch_count,
            self.linearization_domain_stop_count,
            self.dynamic_gs_convergence_stop_count,
            self.other_adverse_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("FreeGSNKE qualification counts must be nonnegative")
        if (
            self.planned_branch_count != len(self.responses)
            or self.planned_branch_count
            != self.observed_complete_branch_count + self.adverse_branch_count
            or self.adverse_branch_count
            != (
                self.operational_failure_branch_count
                + self.linearization_domain_stop_count
                + self.dynamic_gs_convergence_stop_count
                + self.other_adverse_count
            )
            or not self.all_issued_branches_accounted
        ):
            raise ValueError("FreeGSNKE qualification branch accounting differs")
        if self.disposition is FreeGsnkeSourceQualificationDisposition.PASS:
            if self.adverse_branch_count or self.reason_codes:
                raise ValueError("passing FreeGSNKE qualification retains an adverse branch")
        elif not self.reason_codes:
            raise ValueError("adverse FreeGSNKE qualification lacks a typed reason")
        if (
            not self.excluded_from_target_evidence
            or self.target_evidence_unit_count
            or self.outcome_access
            not in {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ):
            raise ValueError("FreeGSNKE qualification result crossed target evidence")


def reduce_freegsnke_source_qualification(
    *,
    config: FreeGsnkeSourceQualificationConfig,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    responses: tuple[FreeGsnkeTargetProcessResponse, ...],
) -> FreeGsnkeSourceQualificationResult:
    """Reduce the exact issued roster without converting a stop into nonentry."""

    requests_by_id = {value.request_id: value for value in requests}
    responses_by_request = {value.request.object_id: value for value in responses}
    expected_ids = {value.object_id for value in config.requests}
    if (
        len(requests_by_id) != len(requests)
        or len(responses_by_request) != len(responses)
        or set(requests_by_id) != expected_ids
        or set(responses_by_request) != expected_ids
        or len({value.phase for value in requests}) != 1
        or not {value.phase for value in requests}
        <= {FreeGsnkePhase.SCOUT, FreeGsnkePhase.EVALUATION}
        or {value.branch_id for value in requests} != set(config.required_branch_ids)
        or any(
            ObjectIdentity.from_record(value.request_id, value) not in config.requests
            for value in requests
        )
    ):
        raise ValueError("FreeGSNKE qualification issued roster is incomplete or substituted")
    for request in requests:
        response = responses_by_request[request.request_id]
        if response.request != ObjectIdentity.from_record(request.request_id, request):
            raise ValueError("FreeGSNKE qualification response changed its request")

    error_codes = tuple(
        value.episode.error_code for value in responses if value.episode.error_code is not None
    )
    complete = sum(
        value.episode.status is FreeGsnkeTargetEpisodeStatus.OBSERVED_COMPLETE
        for value in responses
    )
    operational = sum(
        value.episode.status
        in {
            FreeGsnkeTargetEpisodeStatus.WORKER_FAILED,
            FreeGsnkeTargetEpisodeStatus.WORKER_TIMED_OUT,
        }
        for value in responses
    )
    domain = sum(
        value.episode.error_code
        in {
            "freegsnke-linearization-domain-policy-stop",
            "freegsnke-linearization-domain-stop",
        }
        for value in responses
    )
    convergence = sum(
        value.episode.error_code == "freegsnke-dynamic-gs-convergence-stop" for value in responses
    )
    adverse = len(responses) - complete
    other = adverse - operational - domain - convergence
    if operational:
        disposition = FreeGsnkeSourceQualificationDisposition.OPERATIONAL_UNEVALUABLE
    elif domain:
        disposition = FreeGsnkeSourceQualificationDisposition.LINEARIZATION_DOMAIN_STOP
    elif convergence:
        disposition = FreeGsnkeSourceQualificationDisposition.DYNAMIC_GS_CONVERGENCE_STOP
    elif adverse:
        disposition = FreeGsnkeSourceQualificationDisposition.SOURCE_ACTION_STOP
    else:
        disposition = FreeGsnkeSourceQualificationDisposition.PASS
    reason_codes = tuple(sorted(set(error_codes)))
    return FreeGsnkeSourceQualificationResult(
        result_id=f"source-qualification-result.{config.qualification_id}",
        qualification=ObjectIdentity.from_record(config.config_id, config),
        responses=tuple(
            sorted(
                (ObjectIdentity.from_record(value.response_id, value) for value in responses),
                key=lambda value: value.object_id,
            )
        ),
        disposition=disposition,
        planned_branch_count=len(config.requests),
        observed_complete_branch_count=complete,
        adverse_branch_count=adverse,
        operational_failure_branch_count=operational,
        linearization_domain_stop_count=domain,
        dynamic_gs_convergence_stop_count=convergence,
        other_adverse_count=other,
        all_issued_branches_accounted=True,
        reason_codes=reason_codes,
        excluded_from_target_evidence=True,
        target_evidence_unit_count=0,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if requests[0].phase is FreeGsnkePhase.SCOUT
            else OutcomeAccess.EVALUATION_REVEALED
        ),
    )


__all__ = [
    'FreeGsnkeSourceQualificationConfig',
    'FreeGsnkeSourceQualificationDisposition',
    'FreeGsnkeSourceQualificationResult',
    "reduce_freegsnke_source_qualification",
]
