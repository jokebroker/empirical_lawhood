"""Bounded, strict decoding for canonical uniform electron gas panel records."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
import json
from typing import Any, BinaryIO, Mapping, Sequence

from .contracts import ActionCurrentRow, BranchInput, ClosureDiagnosticInput, EvidenceLane, GaugeControlRow, MAXIMUM_PANEL_BYTES, UniformElectronGasTransverseScreenConfig, TransversePanel, Vector3


def _keys(value: Mapping[str, object], expected: Sequence[str], *, field_name: str) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{field_name} fields differ: "
            f"missing={sorted(set(expected) - set(value))}, "
            f"unknown={sorted(set(value) - set(expected))}"
        )


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return value


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an exact integer")
    return value


def _decimal(value: object, *, field_name: str) -> Decimal:
    document = _mapping(value, field_name=field_name)
    _keys(document, ("decimal",), field_name=field_name)
    text = document["decimal"]
    if not isinstance(text, str):
        raise ValueError(f"{field_name}.decimal must be a string")
    result = Decimal(text)
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _optional_decimal(value: object, *, field_name: str) -> Decimal | None:
    return None if value is None else _decimal(value, field_name=field_name)


def _vector(value: object, *, field_name: str) -> Vector3:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-vector")
    result = tuple(
        _decimal(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    return (result[0], result[1], result[2])


def _record(value: object, *, schema: str, field_name: str) -> Mapping[str, object]:
    document = _mapping(value, field_name=field_name)
    _keys(document, ("schema", "value", "version"), field_name=field_name)
    if document["schema"] != schema or document["version"] != "1.0.0":
        raise ValueError(f"{field_name} schema/version differs")
    return _mapping(document["value"], field_name=f"{field_name}.value")


def _field_names(record_type: type[Any]) -> tuple[str, ...]:
    return tuple(value.name for value in fields(record_type))


def _decode_row(value: object) -> ActionCurrentRow:
    body = _record(value, schema=ActionCurrentRow.SCHEMA, field_name="action row")
    _keys(body, _field_names(ActionCurrentRow), field_name="action row value")
    reason = body["reason_code"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason_code must be string or null")
    return ActionCurrentRow(
        row_id=_string(body["row_id"], field_name="row_id"),
        q_over_kf=_decimal(body["q_over_kf"], field_name="q_over_kf"),
        u=_decimal(body["u"], field_name="u"),
        requested_A_T=_vector(body["requested_A_T"], field_name="requested_A_T"),
        accepted_A_T=_vector(body["accepted_A_T"], field_name="accepted_A_T"),
        applied_A_T=_vector(body["applied_A_T"], field_name="applied_A_T"),
        realized_A_T=_vector(body["realized_A_T"], field_name="realized_A_T"),
        current_density=_vector(body["current_density"], field_name="current_density"),
        current_error_bound_A_m2=_decimal(
            body["current_error_bound_A_m2"], field_name="current_error_bound_A_m2"
        ),
        requested_clock=_integer(body["requested_clock"], field_name="requested_clock"),
        accepted_clock=_integer(body["accepted_clock"], field_name="accepted_clock"),
        applied_clock=_integer(body["applied_clock"], field_name="applied_clock"),
        receiver_clock=_integer(body["receiver_clock"], field_name="receiver_clock"),
        accepted=_boolean(body["accepted"], field_name="accepted"),
        valid=_boolean(body["valid"], field_name="valid"),
        reason_code=reason,
    )


def _decode_control(value: object) -> GaugeControlRow:
    body = _record(value, schema=GaugeControlRow.SCHEMA, field_name="gauge control")
    _keys(body, _field_names(GaugeControlRow), field_name="gauge control value")
    return GaugeControlRow(
        control_id=_string(body["control_id"], field_name="control_id"),
        q_over_kf=_decimal(body["q_over_kf"], field_name="q_over_kf"),
        diamagnetic_normal=_decimal(body["diamagnetic_normal"], field_name="diamagnetic_normal"),
        paramagnetic_normal=_decimal(body["paramagnetic_normal"], field_name="paramagnetic_normal"),
        ward_residual=_decimal(body["ward_residual"], field_name="ward_residual"),
        wrong_polarization_current_A_m2=_decimal(
            body["wrong_polarization_current_A_m2"],
            field_name="wrong_polarization_current_A_m2",
        ),
        converged=_boolean(body["converged"], field_name="converged"),
        limit_order_supported=_boolean(
            body["limit_order_supported"], field_name="limit_order_supported"
        ),
    )


def _decode_panel(value: object) -> TransversePanel:
    body = _record(value, schema=TransversePanel.SCHEMA, field_name="transverse panel")
    _keys(body, _field_names(TransversePanel), field_name="panel value")
    rows = body["rows"]
    controls = body["controls"]
    if not isinstance(rows, list) or not isinstance(controls, list):
        raise ValueError("panel rows and controls must be arrays")
    baseline = body["baseline_preserved"]
    if baseline is not None and not isinstance(baseline, bool):
        raise ValueError("baseline_preserved must be boolean or null")
    return TransversePanel(
        panel_id=_string(body["panel_id"], field_name="panel_id"),
        source_id=_string(body["source_id"], field_name="source_id"),
        branch_id=_string(body["branch_id"], field_name="branch_id"),
        view_id=_string(body["view_id"], field_name="view_id"),
        evidence_lane=EvidenceLane(_string(body["evidence_lane"], field_name="evidence_lane")),
        evidence_world=_string(body["evidence_world"], field_name="evidence_world"),
        independent_unit_id=_string(body["independent_unit_id"], field_name="independent_unit_id"),
        deterministic=_boolean(body["deterministic"], field_name="deterministic"),
        r_s=_decimal(body["r_s"], field_name="r_s"),
        temperature_K=_decimal(body["temperature_K"], field_name="temperature_K"),
        q_direction=_vector(body["q_direction"], field_name="q_direction"),
        action_direction=_vector(body["action_direction"], field_name="action_direction"),
        rows=tuple(_decode_row(item) for item in rows),
        controls=tuple(_decode_control(item) for item in controls),
        order_zero=_optional_decimal(body["order_zero"], field_name="order_zero"),
        order_at_max_action=_optional_decimal(
            body["order_at_max_action"], field_name="order_at_max_action"
        ),
        baseline_preserved=baseline,
        source_semantics_valid=_boolean(
            body["source_semantics_valid"], field_name="source_semantics_valid"
        ),
    )


def _load(payload: bytes, *, maximum_bytes: int) -> object:
    if not isinstance(payload, bytes) or len(payload) > maximum_bytes:
        raise ValueError("source payload is absent or oversized")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source payload is not strict JSON") from error


def decode_branch_input_bytes(payload: bytes, config: UniformElectronGasTransverseScreenConfig) -> BranchInput:
    """Decode one exact canonical, complete truth/target-shaped panel input."""

    document = _load(payload, maximum_bytes=MAXIMUM_PANEL_BYTES)
    body = _record(document, schema=BranchInput.SCHEMA, field_name="branch input")
    _keys(body, _field_names(BranchInput), field_name="branch input value")
    panels_value = body["panels"]
    if not isinstance(panels_value, list):
        raise ValueError("branch input panels must be an array")
    value = BranchInput(
        input_id=_string(body["input_id"], field_name="input_id"),
        case_id=_string(body["case_id"], field_name="case_id"),
        panels=tuple(_decode_panel(item) for item in panels_value),
        authority_valid=_boolean(body["authority_valid"], field_name="authority_valid"),
    )
    if value.canonical_bytes() != payload:
        raise ValueError("branch input must use exact canonical JSON bytes")
    if tuple(panel.view_id for panel in value.panels) != config.views or any(
        panel.r_s != config.r_s
        or panel.temperature_K != config.temperature_K
        or len(panel.rows) != len(config.q_over_kf) * len(config.u_values)
        or len(panel.controls) != len(config.q_over_kf)
        for panel in value.panels
    ):
        raise ValueError("branch input cardinality/denominator differs from frozen config")
    return value


def decode_closure_input_bytes(payload: bytes) -> ClosureDiagnosticInput:
    document = _load(payload, maximum_bytes=MAXIMUM_PANEL_BYTES)
    body = _record(
        document,
        schema=ClosureDiagnosticInput.SCHEMA,
        field_name="closure diagnostic input",
    )
    _keys(
        body,
        _field_names(ClosureDiagnosticInput),
        field_name="closure diagnostic value",
    )
    value = ClosureDiagnosticInput(
        input_id=_string(body["input_id"], field_name="input_id"),
        source_id=_string(body["source_id"], field_name="source_id"),
        branch_id=_string(body["branch_id"], field_name="branch_id"),
        r_s=_decimal(body["r_s"], field_name="r_s"),
        temperature_K=_decimal(body["temperature_K"], field_name="temperature_K"),
        supplied_tc_K=_decimal(body["supplied_tc_K"], field_name="supplied_tc_K"),
        stiffness_closure=_string(body["stiffness_closure"], field_name="stiffness_closure"),
    )
    if value.canonical_bytes() != payload:
        raise ValueError("closure input must use exact canonical JSON bytes")
    return value


def read_preopened_bytes(reader: BinaryIO, *, maximum_bytes: int) -> bytes:
    """Read only from a preopened bounded stream and reject trailing bytes."""

    if maximum_bytes <= 0 or maximum_bytes > MAXIMUM_PANEL_BYTES:
        raise ValueError("preopened source byte bound is invalid")
    payload = reader.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("preopened source exceeds its byte bound")
    return payload


__all__ = [
    "decode_branch_input_bytes",
    "decode_closure_input_bytes",
    "read_preopened_bytes",
]
