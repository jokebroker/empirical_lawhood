"""Exact Arrow IPC codecs for row-oriented SC scientific payloads."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from typing import cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    CandidateView,
    DiscoveryObservation,
    LabelState,
    ObservationOrigin,
    PolicyDecisionKind,
)
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.runtime.artifacts import TabularFieldContract, TabularPayloadContract

from .contracts import SourceQualification
from .records import PolicyHistoryPrefix
from .worlds import (
    FEATURE_SCHEMA_ID,
    FEATURE_WIDTH,
    MaterialCandidate,
    MaterialCorpus,
    MaterialSearchWorld,
)


MATERIAL_CORPUS_TABLE_SCHEMA = 'empirical-lawhood/reference-worlds/material-family-discovery/material-corpus-table'
WORLD_POLICY_TABLE_SCHEMA = 'empirical-lawhood/reference-worlds/material-family-discovery/world-policy-table'
WORLD_TRUTH_TABLE_SCHEMA = 'empirical-lawhood/reference-worlds/material-family-discovery/world-truth-table'
POLICY_HISTORY_TABLE_SCHEMA = 'empirical-lawhood/reference-worlds/material-family-discovery/policy-history-table'
_FEATURE_DTYPE = np.dtype("<f4")
_FEATURE_BYTES = FEATURE_WIDTH * _FEATURE_DTYPE.itemsize


class MaterialFamilyDiscoveryTableError(ValueError):
    """Exact SC table profile or semantic validation failure."""


def _metadata(
    *,
    payload_schema: str,
    logical_types: dict[str, str],
    units: dict[str, str],
    frames: dict[str, str],
    clocks: dict[str, str],
    keys: tuple[str, ...],
) -> dict[bytes, bytes]:
    def encoded(value: object) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    return {
        b"empirical_lawhood_payload_schema": payload_schema.encode("utf-8"),
        b"empirical_lawhood_logical_types": encoded(logical_types),
        b"empirical_lawhood_units": encoded(units),
        b"empirical_lawhood_frames": encoded(frames),
        b"empirical_lawhood_clocks": encoded(clocks),
        b"empirical_lawhood_keys": encoded(list(keys)),
    }


def _schema(
    payload_schema: str,
    fields: tuple[pa.Field, ...],
    logical_types: dict[str, str],
    units: dict[str, str],
    frames: dict[str, str],
    keys: tuple[str, ...],
    clocks: dict[str, str] | None = None,
) -> pa.Schema:
    return pa.schema(
        fields,
        metadata=_metadata(
            payload_schema=payload_schema,
            logical_types=logical_types,
            units=units,
            frames=frames,
            clocks=clocks or {},
            keys=keys,
        ),
    )


def material_corpus_arrow_schema() -> pa.Schema:
    names = (
        "candidate_id",
        "formula_sha256",
        "features_le_f32",
        "stratum_id",
        "family_id",
        "family_label",
        "tc_kelvin_decimal",
        "label_state",
        "source_row_count",
        "ambiguous_family",
    )
    return _schema(
        MATERIAL_CORPUS_TABLE_SCHEMA,
        (
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("formula_sha256", pa.string(), nullable=False),
            pa.field("features_le_f32", pa.binary(), nullable=False),
            pa.field("stratum_id", pa.string(), nullable=False),
            pa.field("family_id", pa.string(), nullable=True),
            pa.field("family_label", pa.string(), nullable=True),
            pa.field("tc_kelvin_decimal", pa.string(), nullable=True),
            pa.field("label_state", pa.string(), nullable=False),
            pa.field("source_row_count", pa.int32(), nullable=False),
            pa.field("ambiguous_family", pa.bool_(), nullable=False),
        ),
        logical_types={
            "candidate_id": "canonical-material-id",
            "formula_sha256": "canonical-formula-sha256",
            "features_le_f32": FEATURE_SCHEMA_ID,
            "stratum_id": "element-count-stratum",
            "family_id": "nims-exact-str3-family-id",
            "family_label": "nims-exact-str3-normalized-label",
            "tc_kelvin_decimal": "exact-decimal-critical-temperature",
            "label_state": "measured-explicit-negative-unlabelled",
            "source_row_count": "nested-source-row-count",
            "ambiguous_family": "cross-row-family-conflict-flag",
        },
        units={**{name: "1" for name in names}, "tc_kelvin_decimal": "K"},
        frames={name: "nims-supercon-220808" for name in names},
        keys=("candidate_id",),
    )


def world_policy_arrow_schema() -> pa.Schema:
    names = (
        "candidate_id",
        "stratum_id",
        "features_le_f32",
        "initial_label_state",
        "initial_tc_kelvin_decimal",
        "eligible_for_query",
    )
    return _schema(
        WORLD_POLICY_TABLE_SCHEMA,
        (
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("stratum_id", pa.string(), nullable=False),
            pa.field("features_le_f32", pa.binary(), nullable=False),
            pa.field("initial_label_state", pa.string(), nullable=True),
            pa.field("initial_tc_kelvin_decimal", pa.string(), nullable=True),
            pa.field("eligible_for_query", pa.bool_(), nullable=False),
        ),
        logical_types={
            "candidate_id": "canonical-material-id",
            "stratum_id": "element-count-stratum",
            "features_le_f32": FEATURE_SCHEMA_ID,
            "initial_label_state": "policy-visible-initial-label-state",
            "initial_tc_kelvin_decimal": "policy-visible-exact-decimal-critical-temperature",
            "eligible_for_query": "frozen-policy-eligibility",
        },
        units={**{name: "1" for name in names}, "initial_tc_kelvin_decimal": "K"},
        frames={name: "matched-world-policy-view" for name in names},
        keys=("candidate_id",),
    )


def world_truth_arrow_schema() -> pa.Schema:
    names = (
        "candidate_id",
        "family_id",
        "family_label",
        "tc_kelvin_decimal",
        "label_state",
        "is_target_family",
    )
    return _schema(
        WORLD_TRUTH_TABLE_SCHEMA,
        (
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("family_id", pa.string(), nullable=True),
            pa.field("family_label", pa.string(), nullable=True),
            pa.field("tc_kelvin_decimal", pa.string(), nullable=True),
            pa.field("label_state", pa.string(), nullable=False),
            pa.field("is_target_family", pa.bool_(), nullable=False),
        ),
        logical_types={
            "candidate_id": "canonical-material-id",
            "family_id": "evaluator-only-nims-exact-str3-family-id",
            "family_label": "evaluator-only-nims-exact-str3-label",
            "tc_kelvin_decimal": "receiver-exact-decimal-critical-temperature",
            "label_state": "receiver-label-state",
            "is_target_family": "evaluator-only-family-discovery-truth",
        },
        units={**{name: "1" for name in names}, "tc_kelvin_decimal": "K"},
        frames={name: "sealed-world-receiver-truth" for name in names},
        keys=("candidate_id",),
    )


def policy_history_arrow_schema() -> pa.Schema:
    names = (
        "world_id",
        "policy_id",
        "round_index",
        "decision_id",
        "decision_kind",
        "requested_order",
        "candidate_id",
        "information_query",
        "observation_state",
        "tc_kelvin_decimal",
        "query_cost_decimal",
        "hold_reason_id",
    )
    return _schema(
        POLICY_HISTORY_TABLE_SCHEMA,
        (
            pa.field("world_id", pa.string(), nullable=False),
            pa.field("policy_id", pa.string(), nullable=False),
            pa.field("round_index", pa.int32(), nullable=False),
            pa.field("decision_id", pa.string(), nullable=False),
            pa.field("decision_kind", pa.string(), nullable=False),
            pa.field("requested_order", pa.int32(), nullable=False),
            pa.field("candidate_id", pa.string(), nullable=True),
            pa.field("information_query", pa.bool_(), nullable=False),
            pa.field("observation_state", pa.string(), nullable=True),
            pa.field("tc_kelvin_decimal", pa.string(), nullable=True),
            pa.field("query_cost_decimal", pa.string(), nullable=False),
            pa.field("hold_reason_id", pa.string(), nullable=True),
        ),
        logical_types={
            "world_id": "opaque-independent-family-world-id",
            "policy_id": "frozen-policy-config-id",
            "round_index": "causal-query-round",
            "decision_id": "immutable-pre-reveal-commitment-id",
            "decision_kind": "query-or-hold",
            "requested_order": "within-committed-batch-order",
            "candidate_id": "canonical-material-id",
            "information_query": "information-not-promotion-flag",
            "observation_state": "post-commitment-receiver-state",
            "tc_kelvin_decimal": "post-commitment-exact-decimal-critical-temperature",
            "query_cost_decimal": "declared-query-cost",
            "hold_reason_id": "typed-hold-reason",
        },
        units={
            **{name: "1" for name in names},
            "tc_kelvin_decimal": "K",
            "query_cost_decimal": "declared-query-cost-unit",
        },
        frames={name: "policy-private-causal-history" for name in names},
        clocks={"round_index": 'clock.material-family-discovery-query-round'},
        keys=("decision_id", "requested_order"),
    )


def _contract(
    schema: pa.Schema, payload_schema: str, keys: tuple[str, ...]
) -> TabularPayloadContract:
    metadata = schema.metadata or {}
    logical = json.loads(metadata[b"empirical_lawhood_logical_types"])
    units = json.loads(metadata[b"empirical_lawhood_units"])
    frames = json.loads(metadata[b"empirical_lawhood_frames"])
    clocks = json.loads(metadata[b"empirical_lawhood_clocks"])
    return TabularPayloadContract(
        payload_schema=payload_schema,
        fields=tuple(
            TabularFieldContract(
                name=field.name,
                physical_type=str(field.type),
                logical_type=logical[field.name],
                nullable=field.nullable,
                native_unit=units[field.name],
                coordinate_frame=frames[field.name],
                clock_id=clocks.get(field.name),
            )
            for field in schema
        ),
        primary_key_fields=keys,
    )


def table_payload_contracts() -> tuple[TabularPayloadContract, ...]:
    return tuple(
        sorted(
            (
                _contract(
                    material_corpus_arrow_schema(), MATERIAL_CORPUS_TABLE_SCHEMA, ("candidate_id",)
                ),
                _contract(
                    policy_history_arrow_schema(),
                    POLICY_HISTORY_TABLE_SCHEMA,
                    ("decision_id", "requested_order"),
                ),
                _contract(
                    world_policy_arrow_schema(), WORLD_POLICY_TABLE_SCHEMA, ("candidate_id",)
                ),
                _contract(world_truth_arrow_schema(), WORLD_TRUTH_TABLE_SCHEMA, ("candidate_id",)),
            ),
            key=lambda value: value.payload_schema,
        )
    )


def _feature_bytes(values: tuple[float, ...]) -> bytes:
    if len(values) != FEATURE_WIDTH:
        raise MaterialFamilyDiscoveryTableError("SC feature vector width differs")
    array = np.asarray(values, dtype=_FEATURE_DTYPE)
    if not np.all(np.isfinite(array)):
        raise MaterialFamilyDiscoveryTableError("SC feature vector contains nonfinite values")
    return array.tobytes(order="C")


def _features(payload: bytes) -> tuple[float, ...]:
    if len(payload) != _FEATURE_BYTES:
        raise MaterialFamilyDiscoveryTableError("SC encoded feature width differs")
    values = np.frombuffer(payload, dtype=_FEATURE_DTYPE)
    if not np.all(np.isfinite(values)):
        raise MaterialFamilyDiscoveryTableError("SC encoded feature contains nonfinite values")
    return tuple(float(value) for value in values)


def _write(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return cast(bytes, sink.getvalue().to_pybytes())


def _read(payload: bytes, expected: pa.Schema, *, maximum_rows: int) -> pa.Table:
    try:
        reader = pa.ipc.open_file(pa.BufferReader(payload))
        table = reader.read_all()
    except (pa.ArrowException, OSError, ValueError) as error:
        raise MaterialFamilyDiscoveryTableError("SC Arrow IPC payload is invalid") from error
    if not table.schema.equals(expected, check_metadata=True):
        raise MaterialFamilyDiscoveryTableError("SC Arrow IPC schema differs")
    if table.num_rows <= 0 or table.num_rows > maximum_rows:
        raise MaterialFamilyDiscoveryTableError("SC Arrow IPC row count exceeds its contract")
    return table


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise MaterialFamilyDiscoveryTableError("SC table decimal is invalid") from error
    if not result.is_finite():
        raise MaterialFamilyDiscoveryTableError("SC table decimal is nonfinite")
    return result


def encode_material_corpus(corpus: MaterialCorpus) -> bytes:
    rows = tuple(
        {
            "candidate_id": value.candidate_id,
            "formula_sha256": value.formula_sha256,
            "features_le_f32": _feature_bytes(value.features),
            "stratum_id": value.stratum_id,
            "family_id": value.family_id,
            "family_label": value.family_label,
            "tc_kelvin_decimal": (None if value.tc_kelvin is None else str(value.tc_kelvin)),
            "label_state": value.label_state.value,
            "source_row_count": value.source_row_count,
            "ambiguous_family": value.ambiguous_family,
        }
        for value in corpus.candidates
    )
    return _write(pa.Table.from_pylist(rows, schema=material_corpus_arrow_schema()))


def decode_material_corpus(
    payload: bytes,
    *,
    qualification: SourceQualification,
    corpus_id: str = 'corpus.material-family-discovery-nims-220808',
) -> MaterialCorpus:
    table = _read(payload, material_corpus_arrow_schema(), maximum_rows=40_000)
    candidates = []
    family_labels: dict[str, str] = {}
    for row in table.to_pylist():
        family_id = row["family_id"]
        family_label = row["family_label"]
        if (family_id is None) != (family_label is None):
            raise MaterialFamilyDiscoveryTableError("SC corpus family ID/label nullability differs")
        if family_id is not None:
            validate_stable_id(family_id, field_name="family_id")
            prior = family_labels.setdefault(family_id, family_label)
            if prior != family_label:
                raise MaterialFamilyDiscoveryTableError("SC corpus family label differs within identity")
        try:
            state = LabelState(row["label_state"])
        except ValueError as error:
            raise MaterialFamilyDiscoveryTableError("SC corpus label state is invalid") from error
        candidates.append(
            MaterialCandidate(
                candidate_id=row["candidate_id"],
                formula_sha256=row["formula_sha256"],
                features=_features(row["features_le_f32"]),
                stratum_id=row["stratum_id"],
                family_id=family_id,
                family_label=family_label,
                tc_kelvin=_decimal(row["tc_kelvin_decimal"]),
                label_state=state,
                source_row_count=row["source_row_count"],
                ambiguous_family=row["ambiguous_family"],
            )
        )
    return MaterialCorpus(
        corpus_id=corpus_id,
        candidates=tuple(candidates),
        family_labels=tuple(sorted(family_labels.items())),
        qualification=qualification,
    )


def encode_world_policy(world: MaterialSearchWorld) -> bytes:
    initial = {value.candidate_id: value for value in world.initial_observations}
    pool = set(world.candidate_pool_ids)
    rows = tuple(
        {
            "candidate_id": value.candidate_id,
            "stratum_id": value.stratum_id,
            "features_le_f32": _feature_bytes(value.features),
            "initial_label_state": (
                None
                if value.candidate_id not in initial
                else initial[value.candidate_id].state.value
            ),
            "initial_tc_kelvin_decimal": (
                None
                if value.candidate_id not in initial
                else str(initial[value.candidate_id].tc_kelvin)
            ),
            "eligible_for_query": value.candidate_id in pool,
        }
        for value in world.policy_candidates
    )
    return _write(pa.Table.from_pylist(rows, schema=world_policy_arrow_schema()))


def decode_world_policy(
    payload: bytes,
) -> tuple[tuple[CandidateView, ...], tuple[DiscoveryObservation, ...], tuple[str, ...]]:
    table = _read(payload, world_policy_arrow_schema(), maximum_rows=40_000)
    candidates = []
    initial = []
    pool_ids = []
    observed_ids = []
    for row in table.to_pylist():
        candidate_id = row["candidate_id"]
        observed_ids.append(candidate_id)
        candidates.append(
            CandidateView(
                candidate_id=candidate_id,
                stratum_id=row["stratum_id"],
                features=_features(row["features_le_f32"]),
            )
        )
        if row["eligible_for_query"]:
            if (
                row["initial_label_state"] is not None
                or row["initial_tc_kelvin_decimal"] is not None
            ):
                raise MaterialFamilyDiscoveryTableError("eligible SC candidate carries an initial outcome")
            pool_ids.append(candidate_id)
        else:
            try:
                state = LabelState(row["initial_label_state"])
            except (TypeError, ValueError) as error:
                raise MaterialFamilyDiscoveryTableError("SC initial label state is invalid") from error
            tc = _decimal(row["initial_tc_kelvin_decimal"])
            initial.append(
                DiscoveryObservation(
                    candidate_id=candidate_id,
                    origin=ObservationOrigin.INITIAL_LABEL,
                    state=state,
                    tc_kelvin=tc,
                    query_cost=Decimal(0),
                    round_index=None,
                )
            )
    if tuple(observed_ids) != tuple(sorted(set(observed_ids))):
        raise MaterialFamilyDiscoveryTableError("SC policy table candidate IDs are not sorted and unique")
    return tuple(candidates), tuple(initial), tuple(pool_ids)


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryTruthRow:
    candidate_id: str
    family_id: str | None
    family_label: str | None
    tc_kelvin: Decimal | None
    label_state: LabelState
    is_target_family: bool


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryWorldTruth:
    rows: tuple[MaterialFamilyDiscoveryTruthRow, ...]
    target_family_id: str
    target_family_label: str

    @property
    def by_id(self) -> dict[str, MaterialFamilyDiscoveryTruthRow]:
        return {value.candidate_id: value for value in self.rows}

    @property
    def target_candidate_ids(self) -> frozenset[str]:
        return frozenset(value.candidate_id for value in self.rows if value.is_target_family)


def encode_world_truth(world: MaterialSearchWorld) -> bytes:
    rows = tuple(
        {
            "candidate_id": value.candidate_id,
            "family_id": value.family_id,
            "family_label": value.family_label,
            "tc_kelvin_decimal": None if value.tc_kelvin is None else str(value.tc_kelvin),
            "label_state": value.label_state.value,
            "is_target_family": value.candidate_id in world.target_candidate_ids,
        }
        for value in world.candidate_truth
    )
    return _write(pa.Table.from_pylist(rows, schema=world_truth_arrow_schema()))


def decode_world_truth(payload: bytes) -> MaterialFamilyDiscoveryWorldTruth:
    table = _read(payload, world_truth_arrow_schema(), maximum_rows=10_000)
    rows = []
    ids = []
    targets: set[tuple[str, str]] = set()
    for row in table.to_pylist():
        candidate_id = row["candidate_id"]
        ids.append(candidate_id)
        family_id = row["family_id"]
        family_label = row["family_label"]
        if (family_id is None) != (family_label is None):
            raise MaterialFamilyDiscoveryTableError("SC truth family ID/label nullability differs")
        if row["is_target_family"]:
            if family_id is None or family_label is None:
                raise MaterialFamilyDiscoveryTableError("SC target truth lacks family identity")
            targets.add((family_id, family_label))
        try:
            state = LabelState(row["label_state"])
        except ValueError as error:
            raise MaterialFamilyDiscoveryTableError("SC truth label state is invalid") from error
        tc = _decimal(row["tc_kelvin_decimal"])
        if state in {LabelState.MEASURED, LabelState.EXPLICIT_NEGATIVE} and tc is None:
            raise MaterialFamilyDiscoveryTableError("SC measured truth lacks Tc")
        if state in {LabelState.UNLABELLED, LabelState.INVALID} and tc is not None:
            raise MaterialFamilyDiscoveryTableError("SC unlabelled/invalid truth carries Tc")
        rows.append(
            MaterialFamilyDiscoveryTruthRow(
                candidate_id=candidate_id,
                family_id=family_id,
                family_label=family_label,
                tc_kelvin=tc,
                label_state=state,
                is_target_family=row["is_target_family"],
            )
        )
    if tuple(ids) != tuple(sorted(set(ids))):
        raise MaterialFamilyDiscoveryTableError("SC truth candidate IDs are not sorted and unique")
    if len(targets) != 1:
        raise MaterialFamilyDiscoveryTableError("SC truth must contain exactly one target family")
    target_family_id, target_family_label = next(iter(targets))
    return MaterialFamilyDiscoveryWorldTruth(tuple(rows), target_family_id, target_family_label)


def encode_policy_history(prefix: PolicyHistoryPrefix) -> bytes:
    observations = {value.candidate_id: value for value in prefix.observations}
    rows = []
    for decision in prefix.decisions:
        if decision.kind is PolicyDecisionKind.HOLD:
            rows.append(
                {
                    "world_id": prefix.world_id,
                    "policy_id": prefix.policy_id,
                    "round_index": decision.round_index,
                    "decision_id": decision.decision_id,
                    "decision_kind": decision.kind.value,
                    "requested_order": -1,
                    "candidate_id": None,
                    "information_query": False,
                    "observation_state": None,
                    "tc_kelvin_decimal": None,
                    "query_cost_decimal": "0",
                    "hold_reason_id": decision.hold_reason_id,
                }
            )
            continue
        for order, candidate_id in enumerate(decision.requested_candidate_ids):
            observation = observations[candidate_id]
            rows.append(
                {
                    "world_id": prefix.world_id,
                    "policy_id": prefix.policy_id,
                    "round_index": decision.round_index,
                    "decision_id": decision.decision_id,
                    "decision_kind": decision.kind.value,
                    "requested_order": order,
                    "candidate_id": candidate_id,
                    "information_query": candidate_id in decision.information_query_ids,
                    "observation_state": observation.state.value,
                    "tc_kelvin_decimal": (
                        None if observation.tc_kelvin is None else str(observation.tc_kelvin)
                    ),
                    "query_cost_decimal": str(observation.query_cost),
                    "hold_reason_id": None,
                }
            )
    return _write(pa.Table.from_pylist(rows, schema=policy_history_arrow_schema()))


def validate_policy_history_table(payload: bytes, prefix: PolicyHistoryPrefix) -> None:
    table = _read(payload, policy_history_arrow_schema(), maximum_rows=1_000)
    expected = encode_policy_history(prefix)
    observed = _write(table)
    if observed != expected:
        raise MaterialFamilyDiscoveryTableError("SC policy history table differs from its canonical prefix")


__all__ = [
    "MATERIAL_CORPUS_TABLE_SCHEMA",
    "POLICY_HISTORY_TABLE_SCHEMA",
    'MaterialFamilyDiscoveryTableError',
    'MaterialFamilyDiscoveryTruthRow',
    'MaterialFamilyDiscoveryWorldTruth',
    "WORLD_POLICY_TABLE_SCHEMA",
    "WORLD_TRUTH_TABLE_SCHEMA",
    "decode_material_corpus",
    "decode_world_policy",
    "decode_world_truth",
    "encode_material_corpus",
    "encode_policy_history",
    "encode_world_policy",
    "encode_world_truth",
    "material_corpus_arrow_schema",
    "policy_history_arrow_schema",
    "table_payload_contracts",
    "validate_policy_history_table",
    "world_policy_arrow_schema",
    "world_truth_arrow_schema",
]
