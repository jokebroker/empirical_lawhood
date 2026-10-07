# SPDX-License-Identifier: MPL-2.0
"""Parse-local reuse cannot erase syntax occurrences or retain mutable records."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import ClassVar

import pytest
import yaml

from empirical_lawhood.api import codecs
from empirical_lawhood.kernel.serialization import CanonicalRecord


@dataclass(frozen=True)
class Leaf(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/parse-leaf"
    calls: ClassVar[int] = 0
    amount: int

    def __post_init__(self) -> None:
        type(self).calls += 1


@dataclass(frozen=True)
class MutableLeaf(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/parse-mutable-leaf"
    backing_kind: ClassVar[str] = "mapping"
    calls: ClassVar[int] = 0
    amount: int

    def __post_init__(self) -> None:
        type(self).calls += 1
        backing = MappingProxyType({"value": self.amount}) if self.backing_kind == "mapping" else [self.amount]
        object.__setattr__(self, "amount", backing)


@dataclass(frozen=True)
class Root(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/parse-root"
    children: tuple[Leaf | MutableLeaf, ...]


def document(child_type: type[CanonicalRecord] = Leaf) -> dict[str, object]:
    leaf = {"schema": child_type.SCHEMA, "version": "1.0.0", "value": {"amount": 1}}
    return {"schema": Root.SCHEMA, "version": "1.0.0", "value": {"children": [leaf, leaf]}}


def load(doc: dict[str, object], media: str = "application/json") -> Root:
    text = json.dumps(doc) if media == "application/json" else yaml.safe_dump(doc)
    return codecs.loads_registered_authoring(text, media_type=media, root_schemas={Root.SCHEMA: Root})


@pytest.mark.parametrize("media", ["application/json", "application/yaml"])
def test_repeated_immutable_records_are_validated_once_per_parse(media: str) -> None:
    doc = document()
    # Separate mapping occurrences also avoid introducing prohibited YAML anchors.
    doc = json.loads(json.dumps(doc))
    Leaf.calls = 0
    first = load(doc, media)
    assert Leaf.calls == 1
    assert first.children[0] is first.children[1]
    second = load(doc, media)
    assert Leaf.calls == 2
    assert second.children[0] is not first.children[0]
    doc["value"]["children"][0]["value"]["amount"] = 2  # type: ignore[index]
    third = load(doc, media)
    assert third.children[0].amount == 2
    assert third.children[1].amount == 1
    assert codecs._authoring_parse.get() is None


@pytest.mark.parametrize("backing_kind", ["mapping", "list"])
def test_mutable_backing_container_is_not_reused(backing_kind: str) -> None:
    MutableLeaf.backing_kind = backing_kind
    MutableLeaf.calls = 0
    result = load(document(MutableLeaf))
    assert MutableLeaf.calls == 2
    assert result.children[0] is not result.children[1]


def test_integer_boolean_nodes_are_not_equivalent_and_errors_reset() -> None:
    doc = json.loads(json.dumps(document()))
    doc["value"]["children"][1]["value"]["amount"] = True
    with pytest.raises(codecs.AuthoringCodecError, match="integer"):
        load(doc)
    assert codecs._authoring_parse.get() is None
    assert load(document()).children[0].amount == 1


def test_node_budget_counts_repeated_occurrences(monkeypatch: pytest.MonkeyPatch) -> None:
    text = json.dumps(document())
    raw = codecs._parse_json(text)
    # Find the exact accepted count independently, then deny one node fewer.
    count = 0
    stack = [raw]
    while stack:
        value = stack.pop()
        count += 1
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    monkeypatch.setattr(codecs, "MAX_AUTHORING_NODES", count - 1)
    with pytest.raises(codecs.AuthoringCodecError, match="node limit"):
        load(document())
    assert codecs._authoring_parse.get() is None


def test_direct_mutable_document_decoding_has_no_cache() -> None:
    doc = document()
    first = codecs.decode_record(doc, Root)
    doc["value"]["children"][0]["value"]["amount"] = 3  # type: ignore[index]
    second = codecs.decode_record(doc, Root)
    assert first.children[0].amount == 1
    assert second.children[0].amount == 3
