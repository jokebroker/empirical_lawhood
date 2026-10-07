# SPDX-License-Identifier: MPL-2.0
# Adapted from icf-yolo: synthetic shared-core contract regressions.
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from decimal import Decimal
from typing import ClassVar

import pytest

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.quantities import QuantitySpec
from empirical_lawhood.kernel.serialization import (
    MAX_RELATIVE_LOCATOR_BYTES,
    MAX_RELATIVE_LOCATOR_SEGMENT_BYTES,
    MAX_RELATIVE_LOCATOR_SEGMENTS,
    CanonicalizationError,
    CanonicalRecord,
    canonical_json_bytes,
    validate_document_shape,
    validate_relative_locator,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec


@dataclass(frozen=True, slots=True)
class GoldenRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/golden-record'
    amount: Decimal
    ids: tuple[str, ...]


def test_canonical_bytes_and_identity_match_independent_literal_contract():
    record = GoldenRecord(Decimal("1.2500"), ("a", "b"))
    expected = (b'{"schema":"empirical-lawhood/tests/golden-record","value":'
                b'{"amount":{"decimal":"1.25"},"ids":["a","b"]},"version":"1.0.0"}\n')
    assert record.canonical_bytes() == expected
    assert record.fingerprint() == "be2d5b79c930afe792f3f8ddf50a2ea4b8ff6fbf6dd20cc97a911d85b263c686"
    assert decode_canonical_bytes(expected, GoldenRecord, maximum_bytes=len(expected)) == record


@pytest.mark.parametrize("value", [Decimal(v) for v in ("0", "-0.00", "1.23456789", "1e20", "-1e-20")])
def test_decimal_canonicalization_is_stable(value: Decimal) -> None:
    assert canonical_json_bytes(value) == canonical_json_bytes(Decimal(str(value)))


@pytest.mark.parametrize("values", [set(), {0}, {-1000, 1, 1000}])
def test_frozenset_canonicalization_is_order_independent(values: set[int]) -> None:
    forward = frozenset(values)
    reverse = frozenset(reversed(sorted(values)))
    assert canonical_json_bytes(forward) == canonical_json_bytes(reverse)


def test_binary_float_is_rejected() -> None:
    with pytest.raises(CanonicalizationError, match="use Decimal"):
        canonical_json_bytes(0.1)


@pytest.mark.parametrize("value", ["Upper", "has space", "", "_leading"])
def test_unstable_semantic_ids_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="stable identifier"):
        validate_stable_id(value)


def test_schema_identity_is_independent_of_format_revision() -> None:
    assert validate_schema("empirical-lawhood/kernel/example")
    assert validate_schema("empirical-lawhood/kernel/example-receiver")
    assert GoldenRecord.VERSION == "1.0.0"
    for schema in (
        "empirical-lawhood/kernel/example/v0",
        "empirical-lawhood/kernel/example/v12",
        "empirical-lawhood/kernel/example/latest",
        "empirical-lawhood/kernel//example",
    ):
        with pytest.raises(ValueError, match="invalid schema"):
            validate_schema(schema)


def test_relative_locator_accepts_exact_utf8_and_segment_boundaries() -> None:
    exact_total = "/".join((*(("a" * 225,) * 8), "b" * 240))
    exact_segment = f"{'é' * 127}a"
    exact_segments = "/".join("a" for _ in range(MAX_RELATIVE_LOCATOR_SEGMENTS))

    assert len(exact_total.encode("utf-8")) == MAX_RELATIVE_LOCATOR_BYTES
    assert len(exact_segment.encode("utf-8")) == MAX_RELATIVE_LOCATOR_SEGMENT_BYTES
    assert validate_relative_locator(exact_total) == exact_total
    assert validate_relative_locator(exact_segment) == exact_segment
    assert validate_relative_locator(exact_segments) == exact_segments
    assert validate_relative_locator("datasets/café/β.json") == "datasets/café/β.json"


@pytest.mark.parametrize(
    ("locator", "message"),
    (
        ("", "nonempty"),
        ("/absolute", "absolute"),
        ("//server/share", "absolute"),
        (".", "current"),
        ("./dataset", "current"),
        ("dataset/./file", "current"),
        ("..", "parent"),
        ("../escape", "parent"),
        ("dataset/../escape", "parent"),
        ("dataset//file", "empty"),
        ("dataset/", "empty"),
        (r"dataset\file", "POSIX"),
        ("dataset/control\nfile", "control"),
        ("datasets/cafe\u0301/file", "NFC-normalized"),
    ),
)
def test_relative_locator_rejects_hostile_forms(locator: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_relative_locator(locator)


def test_relative_locator_enforces_total_segment_and_segment_count_bounds() -> None:
    over_total = "/".join((*(("a" * 225,) * 8), "b" * 241))
    over_segment = "é" * 128
    over_segments = "/".join("a" for _ in range(MAX_RELATIVE_LOCATOR_SEGMENTS + 1))

    with pytest.raises(ValueError, match="byte limit"):
        validate_relative_locator(over_total)
    with pytest.raises(ValueError, match="segment exceeds"):
        validate_relative_locator(over_segment)
    with pytest.raises(ValueError, match="segment-count"):
        validate_relative_locator(over_segments)


def test_fingerprint_changes_when_scientific_identity_changes(
    numerical_system: SystemSpec,
) -> None:
    quantity = numerical_system.quantities[0]
    changed = replace(quantity, coordinate_frame="different-terminal")
    assert quantity.fingerprint() != changed.fingerprint()


def test_fingerprint_is_cached_on_an_immutable_record(
    numerical_system: SystemSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shared reference-world fixture has already fingerprinted its fields.
    quantity = replace(numerical_system.quantities[0])
    calls = 0
    original = CanonicalRecord.canonical_bytes

    def counted(record: CanonicalRecord) -> bytes:
        nonlocal calls
        calls += record is quantity
        return original(record)

    monkeypatch.setattr(CanonicalRecord, "canonical_bytes", counted)
    first = quantity.fingerprint()
    second = quantity.fingerprint()

    assert first == second
    assert calls == 1
    assert "_canonical_fingerprint" not in quantity.to_document()["value"]


def test_public_records_are_deeply_immutable(numerical_system: SystemSpec) -> None:
    quantity: QuantitySpec = numerical_system.quantities[0]
    with pytest.raises(FrozenInstanceError):
        quantity.native_unit = "kW"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        numerical_system.quantities = ()  # type: ignore[misc]
    assert isinstance(numerical_system.quantities, tuple)
    assert isinstance(numerical_system.world.available_outcome_access, frozenset)


@dataclass(slots=True)
class MutableRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/mutable-record'

    value: str


def test_mutable_canonical_record_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be frozen"):
        MutableRecord("unsafe").canonical_bytes()


def test_document_shape_rejects_unknown_fields_and_versions(
    numerical_system: SystemSpec,
) -> None:
    document = numerical_system.quantities[0].to_document()
    fields = frozenset(document["value"])  # type: ignore[arg-type]
    assert (
        validate_document_shape(
            document,
            expected_schema=QuantitySpec.SCHEMA,
            expected_version=QuantitySpec.VERSION,
            field_names=fields,
        )
        == document["value"]
    )
    unknown = {**document, "unexpected": True}
    with pytest.raises(CanonicalizationError, match="envelope fields"):
        validate_document_shape(
            unknown,
            expected_schema=QuantitySpec.SCHEMA,
            expected_version=QuantitySpec.VERSION,
            field_names=fields,
        )
    with pytest.raises(CanonicalizationError, match="version"):
        validate_document_shape(
            document,
            expected_schema=QuantitySpec.SCHEMA,
            expected_version="2.0.0",
            field_names=fields,
        )


def test_bounded_canonical_decoder_requires_exact_bytes(
    numerical_system: SystemSpec,
) -> None:
    quantity = numerical_system.quantities[0]
    payload = quantity.canonical_bytes()

    assert (
        decode_canonical_bytes(
            payload,
            QuantitySpec,
            maximum_bytes=len(payload),
        )
        == quantity
    )
    with pytest.raises(CanonicalizationError, match="exceeds"):
        decode_canonical_bytes(
            payload,
            QuantitySpec,
            maximum_bytes=len(payload) - 1,
        )
    with pytest.raises(CanonicalizationError, match="not canonical"):
        decode_canonical_bytes(
            payload.rstrip(b"\n") + b" \n",
            QuantitySpec,
            maximum_bytes=len(payload) + 1,
        )


def test_bounded_canonical_decoder_rejects_duplicate_keys(
    numerical_system: SystemSpec,
) -> None:
    payload = numerical_system.quantities[0].canonical_bytes()
    duplicate = payload.replace(
        b'{"schema":',
        b'{"schema":"empirical-lawhood/kernel/quantity-spec","schema":',
        1,
    )

    with pytest.raises(CanonicalizationError, match="duplicate"):
        decode_canonical_bytes(
            duplicate,
            QuantitySpec,
            maximum_bytes=len(duplicate),
        )


def test_recursive_encoding_matches_established_json_for_shared_records() -> None:
    from dataclasses import dataclass
    from decimal import Decimal
    import json
    from typing import ClassVar
    from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_value

    @dataclass(frozen=True)
    class Sample(CanonicalRecord):
        SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/recursive-encoding'
        value: object

    child = Sample(("unicode-λ", Decimal("-0.00"), frozenset(("a", "z")), True, 1))
    root = Sample((child, child, {"z": child, "a": None}))
    expected = (
        json.dumps(canonical_value(root), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode()
    assert root.canonical_bytes() == expected
    assert root.canonical_bytes() == expected


def test_frozen_record_does_not_cache_through_mutable_mapping() -> None:
    from dataclasses import dataclass
    from hashlib import sha256
    from types import MappingProxyType
    from typing import ClassVar
    from empirical_lawhood.kernel.serialization import CanonicalRecord

    @dataclass(frozen=True)
    class Sample(CanonicalRecord):
        SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/mapping-encoding'
        value: object

    mapping = {"value": 1}
    child = Sample(MappingProxyType(mapping))
    root = Sample((child, child))
    before = root.canonical_bytes()
    first = root.fingerprint()
    mapping["value"] = 2
    after = root.canonical_bytes()
    assert before != after
    assert root.fingerprint() != first
    assert root.fingerprint() == sha256(after).hexdigest()


def test_encoding_cache_is_independent_of_decimal_context() -> None:
    from dataclasses import dataclass
    from decimal import Decimal, localcontext
    from hashlib import sha256
    from typing import ClassVar

    @dataclass(frozen=True)
    class Sample(CanonicalRecord):
        SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/context-encoding'
        value: Decimal

    value = Sample(Decimal("1.2345678901234567890123456789"))
    with localcontext() as context:
        context.prec = 10
        low = value.canonical_bytes()
        assert value.fingerprint() == sha256(low).hexdigest()
        context.prec = 28
        high = value.canonical_bytes()
        assert value.fingerprint() == sha256(high).hexdigest()
        assert low == high
        context.prec = 10
        assert value.canonical_bytes() == low


@dataclass(frozen=True, slots=True)
class RepeatedScalar(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/repeated-scalar'
    value: bool | int

    def __post_init__(self) -> None:
        if type(self.value) is int and self.value < 0:
            raise ValueError("negative scalar")


@dataclass(frozen=True, slots=True)
class RepeatedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/repeated-root'
    children: tuple[RepeatedScalar, ...]


def test_repeated_record_decoder_keeps_types_and_rejects_second_bad_occurrence() -> None:
    import json

    original = RepeatedRoot((RepeatedScalar(True), RepeatedScalar(1), RepeatedScalar(True)))
    payload = original.canonical_bytes()
    decoded = decode_canonical_bytes(payload, RepeatedRoot, maximum_bytes=10000)
    assert decoded == original
    assert tuple(type(c.value) for c in decoded.children) == (bool, int, bool)
    for mutation in ("field", "negative", "version"):
        document = json.loads(payload)
        child = document["value"]["children"][2]
        if mutation == "field":
            child["value"]["unrecognized"] = True
        elif mutation == "negative":
            child["value"]["value"] = -1
        else:
            child["version"] = "2.0.0"
        bad = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(CanonicalizationError):
            decode_canonical_bytes(bad, RepeatedRoot, maximum_bytes=10000)
        assert decode_canonical_bytes(payload, RepeatedRoot, maximum_bytes=10000) == original
    duplicate = payload.replace(b'"value":true', b'"value":true,"value":true')
    with pytest.raises(CanonicalizationError, match="duplicate"):
        decode_canonical_bytes(duplicate, RepeatedRoot, maximum_bytes=10000)


def test_shared_encoding_concurrent_decimal_contexts_stay_coherent() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from decimal import localcontext
    from hashlib import sha256
    import sys

    @dataclass(frozen=True, slots=True)
    class Value(CanonicalRecord):
        SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/threaded-context'
        value: Decimal

    shared = Value(Decimal("1.2345678901234567890123456789"))

    def encode(precision: int) -> None:
        with localcontext() as context:
            context.prec = precision
            expected = Value(shared.value).canonical_bytes()
            expected_digest = sha256(expected).hexdigest()
            for _ in range(250):
                assert shared.canonical_bytes() == expected
                assert shared.fingerprint() == expected_digest

    interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.000001)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(encode, (7, 10, 18, 28) * 8))
    finally:
        sys.setswitchinterval(interval)


@pytest.fixture
def numerical_system():
    from tests.runtime_platform.conftest import build_protocol_fixture
    return build_protocol_fixture().system
