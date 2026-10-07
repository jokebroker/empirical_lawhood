# SPDX-License-Identifier: MPL-2.0
"""Constructor refusals and frozen valid-byte controls for native clocks."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from enum import StrEnum
import json
from pathlib import Path

import pytest

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord, CanonicalizationError
from empirical_lawhood.kernel.time import (
    AvailabilitySpec, CausalPhase, ClockCoordinate, ClockLabelSemantics,
    ClockProjectionStatus, ClockRelationKind, ClockRelationSpec,
    ClockSpec, ClockTransport, ClockTransportAvailability, ClockTransportKind,
    ClockTransportMonotonicity, CoordinateOrigin, HoldSemantics, InformationCutoff,
    SamplingSemantics,
)


def _clock() -> ClockSpec:
    return ClockSpec("test-clock", "Test clock", "s", "test-start", SamplingSemantics.REGULAR,
                     HoldSemantics.ZERO_ORDER, ClockLabelSemantics.INTERVAL_END, Decimal("0.1"), Decimal("0.001"))


def _coordinate() -> ClockCoordinate:
    return ClockCoordinate("source-clock", Decimal("5"), "s", "source-frame", CoordinateOrigin.ABSOLUTE)


def _transport() -> ClockTransport:
    evaluator = ExecutableReference(
        "transport-evaluator", "test.transport", "1.0.0", "affine-transport",
        ArtifactIdentity("transport-source", "implementation", "empirical-lawhood/tests/transport-source", "1" * 64,
                         "application/json", 100), SafePayloadFormat.CANONICAL_JSON,
        "empirical-lawhood/tests/clock-input", "empirical-lawhood/tests/clock-output", True,
    )
    return ClockTransport(
        "test-transport", "source-clock", "target-clock", "s", "s", "source-frame", "target-frame",
        CoordinateOrigin.ABSOLUTE, CoordinateOrigin.ABSOLUTE, ClockTransportKind.AFFINE_BOUNDED,
        ClockTransportAvailability.AVAILABLE, Decimal("2"), Decimal("1"), Decimal("0.01"),
        Decimal("0"), Decimal("10"), ClockTransportMonotonicity.STRICTLY_INCREASING, evaluator, (),
    )


def positive_records() -> dict[str, CanonicalRecord]:
    records: dict[str, CanonicalRecord] = {}
    for field, enum in (("sampling", SamplingSemantics), ("hold", HoldSemantics), ("label_semantics", ClockLabelSemantics)):
        for value in enum:
            changes = {field: value}
            if field == "sampling" and value is not SamplingSemantics.REGULAR:
                changes["nominal_period"] = None
            records[f"clock.{field}.{value.value}"] = replace(_clock(), **changes)
    for phase in CausalPhase:
        records[f"availability.phase.{phase.value}"] = AvailabilitySpec("test-clock", phase, OutcomeAccess.OUTCOME_BLIND, Decimal("5"))
        for inclusive in (False, True):
            records[f"cutoff.{phase.value}.{inclusive}"] = InformationCutoff("test-cutoff", "test-clock", phase, Decimal("5"), inclusive)
    for access in OutcomeAccess:
        records[f"availability.access.{access.value}"] = AvailabilitySpec("test-clock", CausalPhase.PRE_ACTION, access, Decimal("5"))
    for kind in ClockRelationKind:
        records[f"relation.{kind.value}"] = ClockRelationSpec("test-relation", "source-clock", "target-clock", kind,
                                                            None if kind is ClockRelationKind.IDENTITY else Decimal("1"),
                                                            Decimal("0"), "test-evidence")
    for origin in CoordinateOrigin:
        records[f"coordinate.{origin.value}"] = replace(_coordinate(), origin=origin)
        records[f"transport.source_origin.{origin.value}"] = replace(_transport(), source_origin=origin)
        records[f"transport.target_origin.{origin.value}"] = replace(_transport(), target_origin=origin)
    exact = replace(_transport(), kind=ClockTransportKind.AFFINE_EXACT, tolerance=Decimal("0"))
    identity = replace(exact, kind=ClockTransportKind.IDENTITY, target_clock_id="source-clock", target_coordinate_frame="source-frame",
                       scale=Decimal("1"), offset=Decimal("0"))
    unavailable = replace(_transport(), availability=ClockTransportAvailability.UNAVAILABLE, scale=None, offset=None,
                          reason_codes=("CLOCK_UNAVAILABLE",))
    for name, transport in (("bounded", _transport()), ("exact", exact), ("identity", identity), ("unavailable", unavailable)):
        records[f"transport.{name}"] = transport
        records[f"projection.{name}"] = transport.project(_coordinate())
    return records


@pytest.mark.parametrize("name", tuple(positive_records()))
def test_valid_clock_records_keep_frozen_bytes_and_decoder_agreement(name: str) -> None:
    record = positive_records()[name]
    expected = json.loads((Path(__file__).parent / "fixtures/clock-canonical.json").read_text())[name].encode()
    assert record.canonical_bytes() == expected
    decoded = decode_canonical_bytes(expected, type(record), maximum_bytes=len(expected))
    assert decoded == record
    assert decoded.canonical_bytes() == expected


@pytest.mark.parametrize(
    ("record", "field", "enum"),
    (("clock", "sampling", SamplingSemantics), ("clock", "hold", HoldSemantics),
     ("clock", "label_semantics", ClockLabelSemantics), ("availability", "phase", CausalPhase),
     ("availability", "outcome_access", OutcomeAccess), ("cutoff", "phase", CausalPhase),
     ("relation", "kind", ClockRelationKind), ("coordinate", "origin", CoordinateOrigin),
     ("transport", "source_origin", CoordinateOrigin), ("transport", "target_origin", CoordinateOrigin),
     ("transport", "kind", ClockTransportKind), ("transport", "availability", ClockTransportAvailability),
     ("transport", "monotonicity", ClockTransportMonotonicity), ("projection", "status", ClockProjectionStatus)),
)
@pytest.mark.parametrize("invalid_kind", ("string", "other-enum", "integer", "boolean"))
def test_enum_values_refuse_before_direct_constructor_decisions(record: str, field: str, enum: type[StrEnum], invalid_kind: str) -> None:
    records = {
        "clock": _clock(),
        "availability": AvailabilitySpec("test-clock", CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal("5")),
        "cutoff": InformationCutoff("test-cutoff", "test-clock", CausalPhase.PRE_ACTION, Decimal("5")),
        "relation": ClockRelationSpec("test-relation", "source-clock", "target-clock", ClockRelationKind.FIXED_DELAY, Decimal("1"), Decimal("0"), "test-evidence"),
        "coordinate": _coordinate(), "transport": _transport(), "projection": _transport().project(_coordinate()),
    }
    valid = getattr(records[record], field)
    wrong_enum = StrEnum("WrongEnum", {"SAME_TEXT": valid.value})
    invalid = {"string": valid.value, "other-enum": wrong_enum.SAME_TEXT, "integer": 1, "boolean": True}[invalid_kind]
    with pytest.raises(ValueError, match=field):
        replace(records[record], **{field: invalid})
    document = records[record].to_document()
    document["value"][field] = "NOT_AN_ENUM_VALUE"
    with pytest.raises(CanonicalizationError, match=field):
        decode_canonical_bytes((json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(), type(records[record]), maximum_bytes=4096)


@pytest.mark.parametrize("invalid", ("false", "true", "", 0, 1, None, object()))
def test_cutoff_inclusive_flag_refuses_truthy_and_falsey_nonbooleans(invalid: object) -> None:
    with pytest.raises(ValueError, match="includes_coordinate"):
        InformationCutoff("test-cutoff", "test-clock", CausalPhase.PRE_ACTION, Decimal("5"), invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("inclusive", (False, True))
def test_equal_coordinate_decision_retains_real_boolean_semantics(inclusive: bool) -> None:
    cutoff = InformationCutoff("test-cutoff", "test-clock", CausalPhase.PRE_ACTION, Decimal("5"), inclusive)
    availability = AvailabilitySpec("test-clock", CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal("5"))
    assert cutoff.allows(availability) is inclusive
    assert cutoff.allows(replace(availability, available_at=Decimal("4")))
    assert not cutoff.allows(replace(availability, available_at=Decimal("6")))
    assert not cutoff.allows(replace(availability, outcome_access=OutcomeAccess.EVALUATION_SEALED))
    assert not cutoff.allows(replace(availability, clock_id="other-clock"))


@pytest.mark.parametrize("invalid", ("PRE_ACTION", 1, True))
def test_phase_order_refuses_untyped_other_phase(invalid: object) -> None:
    with pytest.raises(ValueError, match="other"):
        CausalPhase.PRE_ACTION.precedes_or_equals(invalid)  # type: ignore[arg-type]
    assert CausalPhase.PRE_ACTION.precedes_or_equals(CausalPhase.RECEIVER)
    assert not CausalPhase.RECEIVER.precedes_or_equals(CausalPhase.PRE_ACTION)
