"""Forty-millisecond native FAIR-MAST core-temperature endpoint reduction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import (
    MAST_ENDPOINT_HORIZON_S,
    ELECTRONVOLT_J,
    MastEndpointResult,
    MastEndpointState,
    MastEvent,
    MastEventDisposition,
)


@dataclass(frozen=True, slots=True)
class MastCoreTemperatureMeasurement(CanonicalRecord):
    """One native ``t_e_core`` sample; source uncertainty is optional, never invented."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/mast-archive/mast-core-temperature-measurement'
    )

    measurement_id: str
    coordinate_s: Decimal
    temperature_ev: Decimal | None
    uncertainty_ev: Decimal | None
    diagnostic_available: bool
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.measurement_id, field_name="measurement_id")
        validate_decimal(self.coordinate_s, field_name="coordinate_s")
        if self.temperature_ev is not None:
            validate_decimal(self.temperature_ev, field_name="temperature_ev", minimum=Decimal(0))
        if self.uncertainty_ev is not None:
            validate_decimal(self.uncertainty_ev, field_name="uncertainty_ev", minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (
            self.diagnostic_available and self.temperature_ev is not None and not self.reason_codes
        ):
            raise ValueError("core-temperature validity differs from its source state")
        if not self.valid and self.uncertainty_ev is not None:
            raise ValueError("invalid core-temperature sample cannot carry uncertainty")


@dataclass(frozen=True, slots=True)
class MastEndpointRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-endpoint-rule'

    rule_id: str
    horizon_s: Decimal
    clock_tolerance_s: Decimal
    receiver_signal_id: str
    no_response_tolerance_ev: Decimal
    reduction_id: str
    development_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.rule_id, field_name="rule_id")
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        validate_stable_id(self.receiver_signal_id, field_name="receiver_signal_id")
        for name, value in (
            ("horizon_s", self.horizon_s),
            ("clock_tolerance_s", self.clock_tolerance_s),
            ("no_response_tolerance_ev", self.no_response_tolerance_ev),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.horizon_s != MAST_ENDPOINT_HORIZON_S:
            raise ValueError("endpoint rule must preserve the frozen 40 ms horizon")
        if self.receiver_signal_id != "t-e-core":
            raise ValueError("primary MAST receiver must remain native t_e_core")
        if self.reduction_id != "native-t-e-core" or not self.development_only:
            raise ValueError("endpoint rule must preserve the native development receiver")


def decode_mast_endpoint_rule(payload: bytes) -> MastEndpointRule:
    return decode_canonical_bytes(payload, MastEndpointRule, maximum_bytes=64 * 1024)


def reduce_mast_endpoint(
    *,
    event: MastEvent,
    reference: MastCoreTemperatureMeasurement | None,
    endpoint: MastCoreTemperatureMeasurement | None,
    rule: MastEndpointRule,
) -> MastEndpointResult:
    if event.disposition not in {MastEventDisposition.ACCEPTED, MastEventDisposition.CANDIDATE}:
        raise ValueError("endpoint reduction requires an eligible event")
    if event.t0_s is None:
        raise ValueError("eligible event has no t0")
    identity = ObjectIdentity.from_record(event.event_id, event)
    if reference is None or endpoint is None:
        return _unevaluable(
            event,
            identity,
            rule,
            MastEndpointState.DIAGNOSTIC_UNAVAILABLE,
            "DIAGNOSTIC_UNAVAILABLE",
        )
    if not reference.diagnostic_available or not endpoint.diagnostic_available:
        return _unevaluable(
            event,
            identity,
            rule,
            MastEndpointState.DIAGNOSTIC_UNAVAILABLE,
            "DIAGNOSTIC_UNAVAILABLE",
        )
    if not reference.valid or not endpoint.valid:
        return _unevaluable(
            event,
            identity,
            rule,
            MastEndpointState.INVALID_MEASUREMENT,
            "INVALID_TEMPERATURE_MEASUREMENT",
        )
    target = event.t0_s + rule.horizon_s
    if (
        reference.coordinate_s >= event.t0_s
        or abs(endpoint.coordinate_s - target) > rule.clock_tolerance_s
    ):
        return _unevaluable(
            event, identity, rule, MastEndpointState.OUT_OF_WINDOW, "ENDPOINT_CLOCK_OUT_OF_WINDOW"
        )
    before = reference.temperature_ev
    after = endpoint.temperature_ev
    if before is None or after is None:  # narrowed by the validity contract
        raise AssertionError("valid native core-temperature sample lost its value")
    delta = after - before
    uncertainty = (
        reference.uncertainty_ev + endpoint.uncertainty_ev
        if reference.uncertainty_ev is not None and endpoint.uncertainty_ev is not None
        else None
    )
    state = (
        MastEndpointState.NO_PHYSICAL_RESPONSE
        if abs(delta) <= rule.no_response_tolerance_ev
        else MastEndpointState.EVALUABLE
    )
    return MastEndpointResult(
        endpoint_id=f"mast-endpoint-{event.shot_id}",
        shot_id=event.shot_id,
        event=identity,
        horizon_s=rule.horizon_s,
        delta_te_core_ev=delta,
        delta_te_core_j=delta * ELECTRONVOLT_J,
        uncertainty_ev=uncertainty,
        uncertainty_j=uncertainty * ELECTRONVOLT_J if uncertainty is not None else None,
        measurement_uncertainty_available=uncertainty is not None,
        state=state,
        native_unit="eV",
        si_unit="J",
        evaluable=True,
        reference_coordinate_s=reference.coordinate_s,
        endpoint_coordinate_s=endpoint.coordinate_s,
        reason_codes=(),
    )


def _unevaluable(
    event: MastEvent,
    identity: ObjectIdentity,
    rule: MastEndpointRule,
    state: MastEndpointState,
    reason: str,
) -> MastEndpointResult:
    return MastEndpointResult(
        endpoint_id=f"mast-endpoint-{event.shot_id}",
        shot_id=event.shot_id,
        event=identity,
        horizon_s=rule.horizon_s,
        delta_te_core_ev=None,
        delta_te_core_j=None,
        uncertainty_ev=None,
        uncertainty_j=None,
        measurement_uncertainty_available=False,
        state=state,
        native_unit="eV",
        si_unit="J",
        evaluable=False,
        reference_coordinate_s=None,
        endpoint_coordinate_s=None,
        reason_codes=(reason,),
    )


__all__ = [
    "MastEndpointRule",
    "MastCoreTemperatureMeasurement",
    "decode_mast_endpoint_rule",
    "reduce_mast_endpoint",
]
