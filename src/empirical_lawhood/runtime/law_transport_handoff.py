"Authenticated measurement through local law donor-law handoff into a separately issued target."

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

from .conditional_children import FrozenParentInputBinding, bind_frozen_parent_input
from .response_experiment import ResponseExperimentObstruction, ResponseExperimentStageRole, ResponseStageTerminal
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus
from .candidate_compiler import DraftStudyCandidate, ExecutableStudyCandidate


LAW_TRANSPORT_INPUT_IDS = (
    'donor-measurement-through-law-qualification-terminal',
    "frozen-response-law",
    "forecast-method-spec",
    "forecast-method-config",
    "frozen-property-forecasts",
)

_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


@dataclass(frozen=True, slots=True)
class AuthenticatedLawTransportHandoff:
    """Ephemeral validated view; issued bindings remain the canonical custody."""

    target_candidate: ObjectIdentity
    scientific_graph_sha256: str
    donor_terminal: ResponseStageTerminal
    response_law: ResponseLaw
    forecast_method_spec: CanonicalRecord
    forecast_method_config: CanonicalRecord
    frozen_forecasts: CanonicalRecord
    bindings: tuple[FrozenParentInputBinding, ...]


def bind_law_transport_handoff(
    *,
    target_candidate: DraftStudyCandidate | ExecutableStudyCandidate,
    donor_terminal: CanonicalRecord,
    response_law: CanonicalRecord,
    forecast_method_spec: CanonicalRecord,
    forecast_method_config: CanonicalRecord,
    frozen_forecasts: CanonicalRecord,
) -> tuple[FrozenParentInputBinding, ...]:
    """Bind exact donor and forecast bytes to predeclared target input slots."""

    records = (
        donor_terminal,
        response_law,
        forecast_method_spec,
        forecast_method_config,
        frozen_forecasts,
    )
    record_ids = tuple(_record_id(value) for value in records)
    bindings = tuple(
        bind_frozen_parent_input(
            candidate=target_candidate,
            external_input_id=input_id,
            parent_record_id=record_id,
            parent_record=record,
        )
        for input_id, record_id, record in zip(
            LAW_TRANSPORT_INPUT_IDS,
            record_ids,
            records,
            strict=True,
        )
    )
    scientific_graph = (
        target_candidate.scientific_graph
        if isinstance(target_candidate, DraftStudyCandidate)
        else target_candidate.base_candidate.base_candidate.scientific_graph
    )
    authenticate_law_transport_handoff(
        target_candidate=ObjectIdentity.from_record(
            target_candidate.candidate_id,
            target_candidate,
        ),
        scientific_graph_sha256=scientific_graph.fingerprint(),
        bindings=bindings,
        forecast_method_spec_type=type(forecast_method_spec),
        forecast_method_config_type=type(forecast_method_config),
        frozen_forecasts_type=type(frozen_forecasts),
    )
    return tuple(sorted(bindings, key=lambda value: value.external_input_id))


def _record_id(value: CanonicalRecord) -> str:
    for attribute in (
        "terminal_id",
        "law_id",
        "spec_id",
        "config_id",
        "plan_id",
        "roster_issue_id",
        "forecast_id",
        "bundle_id",
        "record_id",
    ):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str):
            return candidate
    raise ValueError("law-transport record has no stable canonical identity field")


def _decode(
    binding: FrozenParentInputBinding,
    record_type: type[_RecordT],
    maximum_bytes: int,
) -> _RecordT:
    return binding.decode_parent(record_type, maximum_bytes=maximum_bytes)


def authenticate_law_transport_handoff(
    *,
    target_candidate: ObjectIdentity,
    scientific_graph_sha256: str,
    bindings: tuple[FrozenParentInputBinding, ...],
    forecast_method_spec_type: type[_RecordT],
    forecast_method_config_type: type[_RecordT],
    frozen_forecasts_type: type[_RecordT],
    maximum_input_bytes: int = 1_000_000,
) -> AuthenticatedLawTransportHandoff:
    """Reconstruct only from issued bindings; no donor acquisition port exists."""

    validate_sha256(scientific_graph_sha256, field_name="scientific_graph_sha256")
    if maximum_input_bytes < 1:
        raise ValueError("law-transport input bound must be positive")
    if tuple(sorted(value.external_input_id for value in bindings)) != tuple(
        sorted(LAW_TRANSPORT_INPUT_IDS)
    ):
        raise ValueError("law-transport handoff has missing or extra target inputs")
    if any(
        value.candidate != target_candidate
        or value.scientific_graph_sha256 != scientific_graph_sha256
        for value in bindings
    ):
        raise ValueError("law-transport target issue predecessor was substituted")
    by_input = {value.external_input_id: value for value in bindings}
    terminal = _decode(
        by_input['donor-measurement-through-law-qualification-terminal'],
        ResponseStageTerminal,
        maximum_input_bytes,
    )
    law = _decode(
        by_input["frozen-response-law"],
        ResponseLaw,
        maximum_input_bytes,
    )
    method_spec = _decode(
        by_input["forecast-method-spec"],
        forecast_method_spec_type,
        maximum_input_bytes,
    )
    method_config = _decode(
        by_input["forecast-method-config"],
        forecast_method_config_type,
        maximum_input_bytes,
    )
    forecasts = _decode(
        by_input["frozen-property-forecasts"],
        frozen_forecasts_type,
        maximum_input_bytes,
    )
    if (
        terminal.stage.role is not ResponseExperimentStageRole.IDENTIFICATION_REVEAL_AND_ADJUDICATION
        or not terminal.stage.applicable
        or terminal.operational_status is not OperationalStatus.SUCCEEDED
        or terminal.scientific_status is not ScientificStatus.SUPPORTED
        or terminal.obstruction is not ResponseExperimentObstruction.NONE
        or not terminal.condition_satisfied
    ):
        raise ValueError("law-transport donor is not a supported measurement through local law terminal")
    law_identity = ObjectIdentity.from_record(_record_id(law), law)
    if getattr(terminal, "scientific_product", None) != law_identity:
        raise ValueError("donor measurement through local law terminal does not own the frozen response law")
    if getattr(forecasts, "method_spec", None) != ObjectIdentity.from_record(
        _record_id(method_spec),
        method_spec,
    ):
        raise ValueError("frozen forecasts substitute their method specification")
    if getattr(forecasts, "method_config", None) != ObjectIdentity.from_record(
        _record_id(method_config),
        method_config,
    ):
        raise ValueError("frozen forecasts substitute their method configuration")
    if getattr(forecasts, "source_law", None) != law_identity:
        raise ValueError("frozen forecasts were not derived from the authenticated donor law")
    return AuthenticatedLawTransportHandoff(
        target_candidate=target_candidate,
        scientific_graph_sha256=scientific_graph_sha256,
        donor_terminal=terminal,
        response_law=law,
        forecast_method_spec=method_spec,
        forecast_method_config=method_config,
        frozen_forecasts=forecasts,
        bindings=tuple(sorted(bindings, key=lambda value: value.external_input_id)),
    )


__all__ = [
    'AuthenticatedLawTransportHandoff',
    "LAW_TRANSPORT_INPUT_IDS",
    'authenticate_law_transport_handoff',
    'bind_law_transport_handoff',
]
