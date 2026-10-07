# SPDX-License-Identifier: MPL-2.0
"""Freeze shared values and retained boundary-specific decoder policies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from typing import ClassVar

import pytest
import yaml

from empirical_lawhood.api import codecs
from empirical_lawhood.infrastructure import dataset_projection as projection
from empirical_lawhood.kernel import decoding
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import RevealBarrierSpec
from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveEvaluationBarrier, RetrospectiveExperimentSpec
from empirical_lawhood.kernel.serialization import CanonicalRecord, CanonicalizationError
from empirical_lawhood.kernel.time import CausalPhase


@dataclass(frozen=True, slots=True)
class Child(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/decoder-policy-child"
    amount: Decimal

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("amount must be nonnegative")


@dataclass(frozen=True, slots=True)
class Corpus(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/decoder-policy-corpus"
    name: str
    count: int
    enabled: bool
    phase: CausalPhase
    pair: tuple[int, str]
    children: tuple[Child, ...]
    names: frozenset[str]
    optional: Child | None


@dataclass(frozen=True, slots=True)
class BarrierUnion(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/barrier-union"
    barrier: RevealBarrierSpec | RetrospectiveEvaluationBarrier


def _record() -> Corpus:
    return Corpus("synthetic", 1, True, CausalPhase.PRE_ACTION, (1, "a"),
                  (Child(Decimal("1.25")), Child(Decimal("2"))), frozenset(("a", "b")), None)


def _bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _typed(document: object, kind: type[CanonicalRecord]) -> tuple[CanonicalRecord, ...]:
    return (decoding.decode_canonical_record(document, kind), codecs.decode_record(document, kind),
            projection._decode_record(document, kind, where="document"))


def test_nested_common_forms_keep_values_and_canonical_bytes() -> None:
    original = _record()
    document = original.to_document()
    for decoded in _typed(document, Corpus):
        assert decoded == original
        assert decoded.canonical_bytes() == original.canonical_bytes()
    payload = original.canonical_bytes()
    assert decoding.decode_canonical_bytes(payload, Corpus, maximum_bytes=len(payload)) == original
    assert projection.decode_canonical_record(payload, Corpus, maximum_bytes=len(payload)) == original
    for media_type, text in (("application/json", payload.decode()), ("application/yaml", yaml.safe_dump(document))):
        assert codecs.loads_registered_authoring(text, media_type=media_type, root_schemas={Corpus.SCHEMA: Corpus}) == original


@pytest.mark.parametrize("mutation", ("unknown", "schema", "revision", "semantic", "decimal", "enum", "count", "enabled", "pair", "union"))
def test_invalid_nested_corpus_keeps_field_locations_and_refusal_categories(mutation: str) -> None:
    document = _record().to_document()
    child = document["value"]["children"][0]
    field = "document.value.children[0]"
    if mutation == "unknown":
        child["value"]["extra"] = True
    elif mutation == "schema":
        child["schema"] = "empirical-lawhood/tests/wrong-child"
    elif mutation == "revision":
        child["version"] = "2.0.0"
    elif mutation == "semantic":
        child["value"]["amount"]["decimal"] = "-1"
    elif mutation == "decimal":
        child["value"]["amount"]["decimal"] = "NaN"
        field += ".value.amount"
    else:
        field = "document.value." + mutation
        if mutation == "enum":
            document["value"]["phase"] = "WRONG_PHASE"
            field = "document.value.phase"
        elif mutation == "count":
            document["value"]["count"] = True
        elif mutation == "enabled":
            document["value"]["enabled"] = 1
        elif mutation == "pair":
            document["value"]["pair"] = [1]
        else:
            document["value"]["optional"] = 1
            field = "document.value.optional"
    with pytest.raises(codecs.AuthoringCodecError) as api:
        codecs.decode_record(document, Corpus)
    assert api.value.field == field
    for decode in (lambda: decoding.decode_canonical_record(document, Corpus),
                   lambda: projection._decode_record(document, Corpus, where="document")):
        with pytest.raises(CanonicalizationError):
            decode()
    assert all(record == _record() for record in _typed(_record().to_document(), Corpus))


def test_nested_shape_errors_have_an_authoring_location_absent_from_kernel_error() -> None:
    document = _record().to_document()
    document["value"]["children"][0]["value"]["extra"] = True
    with pytest.raises(codecs.AuthoringCodecError) as api:
        codecs.decode_record(document, Corpus)
    with pytest.raises(CanonicalizationError) as kernel:
        decoding.decode_canonical_record(document, Corpus)
    assert api.value.field == "document.value.children[0]"
    assert not hasattr(kernel.value, "field")
    assert "document.value.children[0]" not in str(kernel.value)


def test_projection_diagnostics_are_an_existing_distinct_contract() -> None:
    document = _record().to_document()
    document["value"]["children"] = "not-a-list"
    with pytest.raises(CanonicalizationError, match="must be a sequence"):
        decoding.decode_canonical_record(document, Corpus)
    with pytest.raises(CanonicalizationError, match="must be an array"):
        projection._decode_record(document, Corpus, where="document")
    document = _record().to_document()
    document["value"]["phase"] = "WRONG_PHASE"
    with pytest.raises(CanonicalizationError, match="unknown enum"):
        decoding.decode_canonical_record(document, Corpus)
    with pytest.raises(CanonicalizationError, match="unsupported enum"):
        projection._decode_record(document, Corpus, where="document")


def test_broader_document_and_authoring_decimal_acceptance_stays_separate() -> None:
    document = _record().to_document()
    document["value"]["children"][0]["value"]["amount"]["decimal"] = "1.25e0"
    assert all(record == _record() for record in _typed(document, Corpus))
    assert codecs.loads_registered_authoring(json.dumps(document), media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus}) == _record()
    with pytest.raises(CanonicalizationError, match="spelling"):
        decoding.decode_canonical_bytes(_bytes(document), Corpus, maximum_bytes=4096)
    with pytest.raises(CanonicalizationError, match="exact canonical JSON"):
        projection.decode_canonical_record(_bytes(document), Corpus, maximum_bytes=4096)


def test_mutable_documents_and_separate_failed_requests_are_revalidated() -> None:
    document = _record().to_document()
    for record in _typed(document, Corpus):
        assert record.count == 1
    document["value"]["count"] = 2
    assert all(record.count == 2 for record in _typed(document, Corpus))
    document["value"]["count"] = False
    for decode in (decoding.decode_canonical_record, codecs.decode_record,
                   lambda document, kind: projection._decode_record(document, kind, where="document")):
        with pytest.raises(ValueError):
            decode(document, Corpus)
    document["value"]["count"] = 3
    assert all(record.count == 3 for record in _typed(document, Corpus))


def test_explicit_retrospective_union_option_keeps_its_distinct_schema() -> None:
    barrier = RetrospectiveEvaluationBarrier(
        barrier_id="historical-barrier", evaluation_cohort_id="historical-cohort",
        evaluation_manifest_sha256="1" * 64, development_unit_ids=("analysis-unit",),
        sealed_outcome_artifact_ids=("label-port",), evaluation_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        fresh_evidence=False, historical_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    original = BarrierUnion(barrier)
    for decoded in _typed(original.to_document(), BarrierUnion):
        assert decoded == original
        assert type(decoded.barrier) is RetrospectiveEvaluationBarrier
    for decoder in (decoding.decode_canonical_record, codecs.decode_record,
                    lambda document, kind: projection._decode_record(document, kind, where="document")):
        with pytest.raises(ValueError):
            decoder(barrier.to_document(), RevealBarrierSpec)
    assert codecs.ROOT_SCHEMAS[RetrospectiveExperimentSpec.SCHEMA] is RetrospectiveExperimentSpec


@pytest.mark.parametrize("decoder", ("kernel", "projection", "api-json", "api-yaml"))
def test_duplicate_keys_refuse_at_each_outer_boundary(decoder: str) -> None:
    if decoder == "api-yaml":
        text = yaml.safe_dump(_record().to_document()) + "schema: duplicate\n"
        with pytest.raises(codecs.AuthoringCodecError, match="duplicate"):
            codecs.loads_registered_authoring(text, media_type="application/yaml", root_schemas={Corpus.SCHEMA: Corpus})
        return
    payload = _record().canonical_bytes().replace(b'"count":1', b'"count":1,"count":1')
    if decoder == "kernel":
        call = lambda: decoding.decode_canonical_bytes(payload, Corpus, maximum_bytes=4096)
    elif decoder == "projection":
        call = lambda: projection.decode_canonical_record(payload, Corpus, maximum_bytes=4096)
    else:
        call = lambda: codecs.loads_registered_authoring(payload.decode(), media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus})
    with pytest.raises(ValueError, match="duplicate"):
        call()


def test_existing_authoring_registry_byte_node_and_depth_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _record().canonical_bytes().decode()
    for registry in ({}, {"wrong-schema": Corpus}):
        with pytest.raises(ValueError, match="registry"):
            codecs.loads_registered_authoring(text, media_type="application/json", root_schemas=registry)
    with pytest.raises(codecs.AuthoringCodecError, match="byte limit"):
        codecs.loads_registered_authoring(text, media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus}, maximum_bytes=10)
    with monkeypatch.context() as patch:
        patch.setattr(codecs, "MAX_AUTHORING_NODES", 10)
        with pytest.raises(codecs.AuthoringCodecError, match="node limit"):
            codecs.loads_registered_authoring(text, media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus})
    nested = "[" * 128 + "0" + "]" * 128
    with pytest.raises(codecs.AuthoringCodecError, match="nesting limit"):
        codecs.loads_registered_authoring(nested, media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus})
    assert codecs.loads_registered_authoring(text, media_type="application/json", root_schemas={Corpus.SCHEMA: Corpus}) == _record()
    with pytest.raises(CanonicalizationError, match="byte limit"):
        projection.decode_canonical_record(text.encode(), Corpus, maximum_bytes=10)
    with pytest.raises(CanonicalizationError, match="nesting limit"):
        projection._parse_json(nested.encode(), maximum_bytes=4096)
    assert projection._parse_json(("[" * 127 + "0" + "]" * 127).encode(), maximum_bytes=4096)


@pytest.mark.parametrize("yaml_text", ("value: &shared 1", "value: *shared", "value: !!python/object:builtins.object {}"))
def test_authoring_yaml_alias_anchor_and_tag_policy_is_retained(yaml_text: str) -> None:
    with pytest.raises(codecs.AuthoringCodecError):
        codecs.loads_registered_authoring(yaml_text, media_type="application/yaml", root_schemas={Corpus.SCHEMA: Corpus})


def test_all_typed_owners_share_only_immutable_annotation_metadata() -> None:
    assert projection.record_annotations is codecs.record_annotations is decoding.record_annotations
    metadata = decoding.record_annotations(Corpus)
    assert metadata["count"] is int
    with pytest.raises(TypeError):
        metadata["count"] = bool  # type: ignore[index]
    document = _record().to_document()
    assert projection._decode_record(document, Corpus, where="document").count == 1
    document["value"]["count"] = False
    with pytest.raises(CanonicalizationError, match="integer"):
        projection._decode_record(document, Corpus, where="document")
    document["value"]["count"] = 2
    assert projection._decode_record(document, Corpus, where="document").count == 2
