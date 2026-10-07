# SPDX-License-Identifier: MPL-2.0
# Adapted from empirical-lawhood: real SQLite transactions; synthetic dataset identities.
from __future__ import annotations


import hashlib




import json


import sqlite3






import threading


from contextlib import contextmanager


from dataclasses import replace


from pathlib import Path


from typing import Any, Generator


import pytest






from sqlalchemy import Engine




import empirical_lawhood.infrastructure.sql.dataset_repository as dataset_repository_module


from empirical_lawhood.infrastructure import (
    decode_dataset_catalog_snapshot,
    decode_unified_catalog_snapshot,
    encode_dataset_catalog_snapshot,
    encode_unified_catalog_snapshot,
)






from empirical_lawhood.infrastructure.dataset_projection import (
    decode_dataset_record_json as strict_decode_dataset_record_json,
)


from empirical_lawhood.infrastructure.sql import (
    DatasetCatalogCorruption,
    DatasetCatalogQueryWorkLimitExceeded,
    DatasetCatalogWriteConflict,
    SQLiteDatasetRepository,
    create_catalog_engine,
    create_read_only_catalog_engine,
    dataset_catalog_query_preflight_connection as strict_dataset_query_preflight,
    upgrade_catalog,
)


from empirical_lawhood.infrastructure.sql.schema import (
    dataset_external_identifier,
    dataset_family,
    dataset_projection_state,
)






from empirical_lawhood.kernel.serialization import CanonicalizationError


from empirical_lawhood.planning.datasets import (
    DatasetFamily,
    ExternalIdentifier,
    ExternalIdentifierKind,
    IdentityState,
)


from empirical_lawhood.runtime.catalog import (
    CatalogSnapshot,
)






from empirical_lawhood.runtime.datasets import (
    DatasetCatalogProjectionAnchor,
    DatasetCatalogProjectionState,
    DatasetCatalogQuery,
    DatasetCatalogQueryBoundary,
    DatasetCatalogSnapshot,
    DatasetCatalogWorkLimit,
    UnifiedCatalogSnapshot,
)


@pytest.fixture
def catalog_engine(tmp_path: Path) -> Generator[Engine, None, None]:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _identifier(value: str = "collection-one") -> ExternalIdentifier:
    return ExternalIdentifier(
        kind=ExternalIdentifierKind.PROVIDER_COLLECTION,
        namespace="provider.example",
        value=value,
    )


def _family(
    family_id: str = "family.example",
    *,
    external_value: str = "collection-one",
) -> DatasetFamily:
    return DatasetFamily(
        family_id=family_id,
        canonical_name=f"Dataset {family_id}",
        provider_id="provider.example",
        external_identifiers=(_identifier(external_value),),
        description="Dataset repository fixture.",
        keywords=("fixture",),
        first_observation_id=None,
        latest_observation_id=None,
        identity_state=IdentityState.FAMILY_RESOLVED,
        reason_codes=(),
    )


def _snapshot(*families: DatasetFamily) -> DatasetCatalogSnapshot:
    return DatasetCatalogSnapshot(
        tuple(sorted(families, key=lambda value: value.family_id)), (), (), (), (), ()
    )


def _projection_revision(snapshot: DatasetCatalogSnapshot) -> int:
    return (
        len(snapshot.families)
        + len(snapshot.releases)
        + len(snapshot.observations)
        + len(snapshot.materializations)
        + len(snapshot.acquisition_attempts)
        + len(snapshot.bindings)
        + sum(len(record.external_identifiers) for record in snapshot.families)
        + sum(len(record.external_identifiers) for record in snapshot.releases)
        + sum(len(record.provider_object_ids) for record in snapshot.observations)
    )


def _projection_anchor(
    snapshot: DatasetCatalogSnapshot,
    *,
    revision: int | None = None,
) -> DatasetCatalogProjectionAnchor:
    state = DatasetCatalogProjectionState(
        snapshot_schema=snapshot.SCHEMA,
        snapshot_fingerprint=snapshot.fingerprint(),
        family_count=len(snapshot.families),
        release_count=len(snapshot.releases),
        observation_count=len(snapshot.observations),
        materialization_count=len(snapshot.materializations),
        acquisition_attempt_count=len(snapshot.acquisition_attempts),
        binding_count=len(snapshot.bindings),
        canonical_byte_count=len(snapshot.canonical_bytes()),
        projection_revision=(_projection_revision(snapshot) if revision is None else revision),
    )
    return DatasetCatalogProjectionAnchor(
        receipt_id=f"receipt.dataset-projection.{snapshot.fingerprint()[:16]}",
        receipt_schema='empirical-lawhood/testing/fixtures/dataset-projection-receipt',
        receipt_sha256=hashlib.sha256(
            b"trusted-dataset-projection-receipt\0" + snapshot.canonical_bytes()
        ).hexdigest(),
        projection_state=state,
    )


@contextmanager
def _read_only_repository(engine: Engine) -> Generator[SQLiteDatasetRepository, None, None]:
    database = engine.url.database
    assert database is not None
    read_only_engine = create_read_only_catalog_engine(Path(database))
    try:
        yield SQLiteDatasetRepository(read_only_engine)
    finally:
        read_only_engine.dispose()


def _query(
    snapshot: DatasetCatalogSnapshot,
    *,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
    max_records_examined: int = 100,
    max_query_steps: int = 100_000,
    limit: int = 10,
    trusted_anchor: DatasetCatalogProjectionAnchor | None = None,
) -> DatasetCatalogQuery:
    anchor = trusted_anchor or _projection_anchor(snapshot)
    boundary = DatasetCatalogQueryBoundary(
        snapshot_fingerprint=snapshot.fingerprint(),
        projection_state_fingerprint=anchor.projection_state.fingerprint(),
        projection_anchor_fingerprint=anchor.fingerprint(),
        work_limit=DatasetCatalogWorkLimit(
            max_records_examined=max_records_examined,
            max_query_steps=max_query_steps,
        ),
        family_ids=(),
        release_ids=(),
        provider_ids=(),
        external_identifiers=external_identifiers,
        custody_states=(),
        acquisition_states=(),
        experiment_spec_ids=(),
        roles=(),
    )
    return DatasetCatalogQuery(boundary=boundary, limit=limit, cursor=None)


def test_dataset_and_unified_projection_codecs_require_exact_canonical_bytes() -> None:
    datasets = _snapshot(_family())
    unified = UnifiedCatalogSnapshot(CatalogSnapshot.empty(), datasets)

    dataset_payload = encode_dataset_catalog_snapshot(datasets)
    unified_payload = encode_unified_catalog_snapshot(unified)
    assert decode_dataset_catalog_snapshot(dataset_payload) == datasets
    assert decode_unified_catalog_snapshot(unified_payload) == unified

    with pytest.raises(CanonicalizationError, match="exact canonical JSON"):
        decode_dataset_catalog_snapshot(dataset_payload + b" ")
    document = json.loads(dataset_payload)
    document["value"]["unexpected"] = []
    hostile = (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()
    with pytest.raises(CanonicalizationError, match="fields differ"):
        decode_dataset_catalog_snapshot(hostile)
    with pytest.raises(CanonicalizationError, match="duplicate JSON mapping key"):
        decode_dataset_catalog_snapshot(b'{"schema":"one","schema":"two"}\n')
    deeply_nested = (b"[" * 2_000) + b"0" + (b"]" * 2_000)
    with pytest.raises(CanonicalizationError, match="nesting limit"):
        decode_dataset_catalog_snapshot(deeply_nested)


def test_insert_or_verify_is_idempotent_but_conflicting_identity_fails(
    catalog_engine: Engine,
) -> None:
    repository = SQLiteDatasetRepository(catalog_engine)
    snapshot = _snapshot(_family())

    repository.append_snapshot(snapshot)
    repository.append_snapshot(snapshot)
    assert repository.snapshot() == snapshot
    assert repository.integrity_check() == ()

    conflicting = _snapshot(
        replace(snapshot.families[0], description="Different immutable content.")
    )
    with pytest.raises(DatasetCatalogWriteConflict, match="different content"):
        repository.append_snapshot(conflicting)
    assert repository.snapshot() == snapshot


@pytest.mark.parametrize(
    "values",
    (
        {"record_json": "{}\n"},
        {"provider_id": "provider.substituted"},
        {"record_fingerprint": b"f" * 32},
    ),
)
def test_snapshot_fails_closed_on_any_unauthenticated_row_update(
    catalog_engine: Engine,
    values: dict[str, object],
) -> None:
    repository = SQLiteDatasetRepository(catalog_engine)
    repository.append_snapshot(_snapshot(_family()))
    with catalog_engine.begin() as connection:
        connection.execute(dataset_family.update().values(**values))

    with pytest.raises(DatasetCatalogCorruption, match="projection is dirty"):
        repository.snapshot()
    assert repository.integrity_check() == ("dataset-record-corrupt",)


def test_external_identifier_query_is_exact_and_normalized_rows_are_verified(
    catalog_engine: Engine,
) -> None:
    repository = SQLiteDatasetRepository(catalog_engine)
    snapshot = _snapshot(_family())
    repository.append_snapshot(snapshot)

    with pytest.raises(DatasetCatalogCorruption, match="query preflight"):
        repository.query(_query(snapshot))
    with _read_only_repository(catalog_engine) as reader:
        exact = reader.query(
            _query(snapshot, external_identifiers=(_identifier("collection-one"),))
        )
        missing = reader.query(_query(snapshot, external_identifiers=(_identifier("collection"),)))
    assert exact.records == snapshot.families
    assert missing.records == ()
    assert exact.records_examined == 1
    assert 0 < exact.query_steps <= exact.boundary.work_limit.max_query_steps

    with catalog_engine.begin() as connection:
        connection.execute(
            dataset_external_identifier.update().values(value="collection-substituted")
        )
    with pytest.raises(DatasetCatalogCorruption, match="projection is dirty"):
        repository.family(snapshot.families[0].family_id)
    with pytest.raises(DatasetCatalogCorruption, match="projection is dirty"):
        repository.snapshot()


def test_projection_state_revision_and_query_boundary_are_exact(
    catalog_engine: Engine,
) -> None:
    repository = SQLiteDatasetRepository(catalog_engine)
    first = _snapshot(_family("family.a", external_value="collection-a"))
    repository.append_snapshot(first)
    with catalog_engine.connect() as connection:
        first_state = connection.execute(dataset_projection_state.select()).one()._mapping
    assert first_state["projection_valid"] == 1
    assert first_state["projection_revision"] > 0

    repository.append_snapshot(first)
    with catalog_engine.connect() as connection:
        unchanged = connection.execute(dataset_projection_state.select()).one()._mapping
    assert unchanged["projection_revision"] == first_state["projection_revision"]

    repository.append_snapshot(_snapshot(_family("family.b", external_value="collection-b")))
    combined = _snapshot(
        _family("family.a", external_value="collection-a"),
        _family("family.b", external_value="collection-b"),
    )
    with catalog_engine.connect() as connection:
        advanced = connection.execute(dataset_projection_state.select()).one()._mapping
    assert advanced["projection_valid"] == 1
    assert advanced["projection_revision"] > first_state["projection_revision"]
    assert advanced["family_count"] == 2
    with _read_only_repository(catalog_engine) as reader:
        assert reader.query(_query(combined)).records == combined.families
        state = reader.projection_state(_query(combined).boundary.work_limit)
        assert state.snapshot_fingerprint == combined.fingerprint()
        assert state.projection_revision == advanced["projection_revision"]
        assert state.family_count == 2
        assert state.record_count == 2

        with pytest.raises(DatasetCatalogQueryWorkLimitExceeded):
            reader.projection_state(
                DatasetCatalogWorkLimit(
                    max_records_examined=1,
                    max_query_steps=1,
                )
            )

    with catalog_engine.begin() as connection:
        connection.execute(dataset_projection_state.update().values(canonical_byte_count=0))
    with _read_only_repository(catalog_engine) as reader:
        with pytest.raises(DatasetCatalogCorruption, match="state metadata"):
            reader.query(_query(combined))


def test_query_read_transaction_blocks_concurrent_row_substitution(
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SQLiteDatasetRepository(catalog_engine)
    snapshot = _snapshot(_family())
    repository.append_snapshot(snapshot)
    database = catalog_engine.url.database
    assert database is not None
    commit_started = threading.Event()
    writer_done = threading.Event()
    writer_errors: list[BaseException] = []
    writer_threads: list[threading.Thread] = []

    def substitute_row() -> None:
        connection = sqlite3.connect(database, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE dataset_family SET provider_id = ? WHERE family_id = ?",
                ("provider.substituted", "family.example"),
            )
            commit_started.set()
            connection.commit()
        except BaseException as error:
            writer_errors.append(error)
        finally:
            connection.close()
            writer_done.set()

    def start_substitution(connection: Any) -> Any:
        result = strict_dataset_query_preflight(connection)
        thread = threading.Thread(target=substitute_row)
        writer_threads.append(thread)
        thread.start()
        assert commit_started.wait(timeout=2)
        return result

    def decode_while_writer_waits(record_json: str, record_type: type[Any]) -> Any:
        assert commit_started.is_set()
        assert not writer_done.is_set()
        return strict_decode_dataset_record_json(record_json, record_type)

    monkeypatch.setattr(
        dataset_repository_module,
        "dataset_catalog_query_preflight_connection",
        start_substitution,
    )
    monkeypatch.setattr(
        dataset_repository_module,
        "decode_dataset_record_json",
        decode_while_writer_waits,
    )
    with _read_only_repository(catalog_engine) as reader:
        page = reader.query(_query(snapshot))
    assert page.records == snapshot.families
    assert len(writer_threads) == 1
    writer_threads[0].join(timeout=5)
    assert writer_done.is_set()
    assert writer_errors == []
    monkeypatch.setattr(
        dataset_repository_module,
        "decode_dataset_record_json",
        strict_decode_dataset_record_json,
    )
    with pytest.raises(DatasetCatalogCorruption, match="projection is dirty"):
        repository.snapshot()
