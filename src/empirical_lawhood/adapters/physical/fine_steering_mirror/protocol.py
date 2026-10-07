"""Strict loader for the development-frozen FSM protocol."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from empirical_lawhood.adapters._bounded_files import AdapterFileBoundError, read_bounded_regular

from .contracts import FineSteeringMirrorObservationOperatorSpec, FineSteeringMirrorProtocolSpec


MAX_FSM_PROTOCOL_BYTES = 1024 * 1024


class FineSteeringMirrorProtocolError(ValueError):
    "The protocol file differs from the statically supported contract."


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FineSteeringMirrorProtocolError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise FineSteeringMirrorProtocolError(f"{name} keys must be strings")
    return cast(dict[str, object], value)


def _decimal(value: object, *, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as error:
        raise FineSteeringMirrorProtocolError(f"{name} must be a decimal string") from error


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FineSteeringMirrorProtocolError(f"{name} must be an integer")
    return value


def load_protocol(path: Path) -> tuple[FineSteeringMirrorProtocolSpec, dict[str, str]]:
    "Load the one registered fine-steering mirror protocol without dynamic dispatch."

    try:
        payload = read_bounded_regular(path, maximum_bytes=MAX_FSM_PROTOCOL_BYTES)
        text = payload.decode("utf-8")
    except (AdapterFileBoundError, UnicodeDecodeError) as error:
        raise FineSteeringMirrorProtocolError("FSM protocol violates its bounded UTF-8 contract") from error
    raw = _mapping(yaml.safe_load(text), name="protocol")
    if raw.get("schema") != "empirical-lawhood/physical/fine-steering-mirror/frozen-protocol":
        raise FineSteeringMirrorProtocolError("unsupported FSM protocol schema")
    if raw.get("status") != "FROZEN_BEFORE_EVALUATION_RECEIVER_ACQUISITION":
        raise FineSteeringMirrorProtocolError("FSM protocol is not frozen at the receiver boundary")
    source = _mapping(raw.get("source_bindings"), name="source_bindings")
    members_raw = _mapping(source.get("acquired_members"), name="acquired_members")
    members = {name: str(value) for name, value in members_raw.items()}
    expected_names = {
        *(f"u_{level}mv_train.npy" for level in (100, 200, 300)),
        *(f"y_{level}mv_train.npy" for level in (100, 200, 300)),
        *(f"u_{level}mv_test.npy" for level in (100, 200, 300)),
    }
    if set(members) != expected_names:
        raise FineSteeringMirrorProtocolError("acquired source member set differs from the frozen split")
    observation = _mapping(raw.get("observation_operator"), name="observation_operator")
    band = observation.get("frequency_band_hz")
    if not isinstance(band, list) or len(band) != 2:
        raise FineSteeringMirrorProtocolError("frequency_band_hz must have two entries")
    relation = _mapping(raw.get("relational_identity"), name="relational_identity")
    gates = _mapping(raw.get("frozen_gates"), name="frozen_gates")
    if gates.get("wrong_time") != "UNEVALUABLE_PHASE_INVARIANT_PEAK_ESTIMAND":
        raise FineSteeringMirrorProtocolError("wrong-time limitation must remain explicit")
    operator = FineSteeringMirrorObservationOperatorSpec(
        operator_id=str(observation["operator_id"]),
        sample_count=_integer(observation["sample_count"], name="sample_count"),
        sampling_frequency_hz=_decimal(
            observation["sampling_frequency_hz"], name="sampling_frequency_hz"
        ),
        input_channels=_integer(observation["input_channels"], name="input_channels"),
        output_channels=_integer(observation["output_channels"], name="output_channels"),
        realizations_per_block=_integer(
            observation["realizations_per_block"], name="realizations_per_block"
        ),
        periods_per_realization=_integer(
            observation["periods_per_realization"], name="periods_per_realization"
        ),
        frequency_band_hz=(
            _decimal(band[0], name="frequency_band_hz[0]"),
            _decimal(band[1], name="frequency_band_hz[1]"),
        ),
        period_aggregation=str(observation["period_aggregation"]),
        matrix_estimator=str(observation["matrix_estimator"]),
        receiver_estimator=str(observation["receiver_estimator"]),
    )
    if relation.get("primary_receiver_index_zero_based") != 1:
        raise FineSteeringMirrorProtocolError("fine-steering mirror primary receiver must remain probe 2")
    protocol = FineSteeringMirrorProtocolSpec(
        protocol_id=str(raw["protocol_id"]),
        source_config_sha256=str(source["source_config_sha256"]),
        source_audit_sha256=str(source["source_audit_receipt_sha256"]),
        development_acquisition_sha256=str(source["development_acquisition_receipt_sha256"]),
        observation_operator=operator,
        action_levels_volts=(Decimal("0.1"), Decimal("0.2"), Decimal("0.3")),
        primary_receiver_index=1,
        maximum_input_matrix_condition=_decimal(
            gates["maximum_input_matrix_condition"],
            name="maximum_input_matrix_condition",
        ),
        maximum_period_relative_difference=_decimal(
            gates["maximum_period_relative_difference"],
            name="maximum_period_relative_difference",
        ),
        maximum_within_action_peak_range_hz=_decimal(
            gates["maximum_within_action_peak_range_hz"],
            name="maximum_within_action_peak_range_hz",
        ),
        maximum_heldout_absolute_error_hz=_decimal(
            gates["maximum_heldout_absolute_error_hz"],
            name="maximum_heldout_absolute_error_hz",
        ),
        minimum_primary_wrong_action_advantage_hz=_decimal(
            gates["minimum_primary_wrong_action_rmse_advantage_hz"],
            name="minimum_primary_wrong_action_rmse_advantage_hz",
        ),
        maximum_primary_slope_hz_per_v=_decimal(
            gates["maximum_primary_slope_hz_per_v"],
            name="maximum_primary_slope_hz_per_v",
        ),
        minimum_law_qualification_evaluation_blocks_per_action=_integer(
            gates["law_qualification_minimum_evaluation_blocks_per_action"],
            name="law_qualification_minimum_evaluation_blocks_per_action",
        ),
        minimum_law_qualification_denominator_exchanges=_integer(
            gates["law_qualification_minimum_one_factor_denominator_exchanges"],
            name="law_qualification_minimum_one_factor_denominator_exchanges",
        ),
        expected_development_sha256=tuple(
            sorted(digest for name, digest in members.items() if name.endswith("_train.npy"))
        ),
    )
    return protocol, members
