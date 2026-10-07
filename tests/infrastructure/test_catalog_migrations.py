# SPDX-License-Identifier: MPL-2.0
# Adapted from icf-yolo: synthetic shared-core contract regressions.
from __future__ import annotations


from pathlib import Path

import pytest

from alembic import command

from alembic.config import Config

from sqlalchemy import event, inspect

from sqlalchemy.exc import IntegrityError

from empirical_lawhood.infrastructure.sql import (
    create_catalog_engine,
    create_read_only_catalog_engine,
    dataset_catalog_query_preflight,
    unified_catalog_query_preflight,
    upgrade_catalog,
)

from empirical_lawhood.infrastructure.sql.database import ALEMBIC_HEAD, SCHEMA_VERSION

from empirical_lawhood.infrastructure.sql.migrations.catalog_tables import CATALOG_TABLE_NAMES

from empirical_lawhood.infrastructure.sql.migrations.dataset_tables import DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS, DATASET_PROJECTION_DIRTY_TRIGGER_NAMES, DATASET_SCHEMA_OBJECT_NAMES, DATASET_TABLE_NAMES

from empirical_lawhood.infrastructure.sql.schema import (
    dataset_external_identifier,
    dataset_family,
)




def _migration_config(connection: object) -> Config:
    config = Config()
    migrations = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "empirical_lawhood"
        / "infrastructure"
        / "sql"
        / "migrations"
    )
    config.set_main_option("script_location", str(migrations))
    config.attributes["connection"] = connection
    return config

def _upgrade(engine: object, revision: str) -> None:
    with engine.connect() as connection:  # type: ignore[attr-defined]
        command.upgrade(_migration_config(connection), revision)

def _schema_rows(engine: object) -> tuple[tuple[object, ...], ...]:
    with engine.connect() as connection:  # type: ignore[attr-defined]
        return tuple(
            tuple(row)
            for row in connection.exec_driver_sql(
                "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name, tbl_name"
            )
        )

def _present_dataset_schema_objects(engine: object) -> tuple[tuple[str, str], ...]:
    placeholders = ", ".join("?" for _ in DATASET_SCHEMA_OBJECT_NAMES)
    with engine.connect() as connection:  # type: ignore[attr-defined]
        return tuple(
            (str(row[0]), str(row[1]))
            for row in connection.exec_driver_sql(
                "SELECT type, name FROM sqlite_schema "
                f"WHERE name IN ({placeholders}) ORDER BY type, name",
                DATASET_SCHEMA_OBJECT_NAMES,
            )
        )

def _family_values(
    *,
    family_id: str = "family.example",
    fingerprint_byte: bytes = b"f",
    keywords_json: str = "[]\n",
) -> dict[str, object]:
    return {
        "family_id": family_id,
        "record_schema": 'empirical-lawhood/planning/dataset-family',
        "record_fingerprint": fingerprint_byte * 32,
        "record_json": (
            '{"schema":"empirical-lawhood/planning/dataset-family","value":{},"version":"1.0.0"}\n'
        ),
        "canonical_name": "Example family",
        "provider_id": "provider.example",
        "description": "Bounded metadata only.",
        "keywords_json": keywords_json,
        "first_observation_id": None,
        "latest_observation_id": None,
        "identity_state": "FAMILY_RESOLVED",
        "reason_codes_json": "[]\n",
    }

def test_initial_catalog_installs_all_science_and_dataset_tables(tmp_path: Path) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    _upgrade(engine, ALEMBIC_HEAD)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == SCHEMA_VERSION
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == ALEMBIC_HEAD
        assert frozenset(inspect(connection).get_table_names()) == CATALOG_TABLE_NAMES | frozenset(DATASET_TABLE_NAMES) | {"alembic_version"}
    engine.dispose()


def test_initialized_catalog_repeat_head_is_idempotent(tmp_path: Path) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO storage_root "
            "(storage_root_id, logical_name, canonical_path, mount_contract_schema, verification_status) VALUES (?, ?, ?, ?, ?)",
            ("storage.initial", "initial sentinel", "/external/initial", "empirical-lawhood/testing/fixtures/storage-root", "VERIFIED"),
        )
    first_schema = _schema_rows(engine)
    for _ in range(2):
        upgrade_catalog(engine)
        assert _schema_rows(engine) == first_schema
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT logical_name FROM storage_root WHERE storage_root_id = ?", ("storage.initial",)).scalar_one() == "initial sentinel"
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM storage_root WHERE storage_root_id = ?", ("storage.initial",)).scalar_one() == 1
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == SCHEMA_VERSION
    engine.dispose()


def test_initial_catalog_refuses_preexisting_schema_without_partial_adoption(tmp_path: Path) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE dataset_family (pk INTEGER PRIMARY KEY)")
    before = _schema_rows(engine)
    with engine.connect() as connection, pytest.raises(RuntimeError, match="refuses pre-existing schema objects: table dataset_family"):
        command.upgrade(_migration_config(connection), ALEMBIC_HEAD)
    assert _schema_rows(engine) == before
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 0
        assert inspect(connection).get_table_names() == ["dataset_family"]
    engine.dispose()


def test_initial_catalog_rolls_back_nth_ddl_and_retries_cleanly(tmp_path: Path) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    dataset_ddl_count = 0

    def fail_third_dataset_ddl(_connection: object, _cursor: object, statement: str, _parameters: object, _context: object, _executemany: bool) -> None:
        nonlocal dataset_ddl_count
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith(("CREATE TABLE DATASET_", "CREATE TABLE ACQUISITION_")):
            dataset_ddl_count += 1
            if dataset_ddl_count == 3:
                raise RuntimeError("injected third dataset DDL failure")

    event.listen(engine, "before_cursor_execute", fail_third_dataset_ddl)
    try:
        with engine.connect() as connection, pytest.raises(RuntimeError, match="injected third dataset DDL failure"):
            command.upgrade(_migration_config(connection), ALEMBIC_HEAD)
    finally:
        event.remove(engine, "before_cursor_execute", fail_third_dataset_ddl)
    assert dataset_ddl_count == 3
    assert _schema_rows(engine) == ()
    assert _present_dataset_schema_objects(engine) == ()
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 0
    upgrade_catalog(engine)
    assert len(_present_dataset_schema_objects(engine)) == len(DATASET_SCHEMA_OBJECT_NAMES)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == SCHEMA_VERSION
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == ALEMBIC_HEAD
    engine.dispose()


def test_initial_catalog_refuses_nonempty_schema_revision(tmp_path: Path) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA user_version=1")
    with engine.connect() as connection, pytest.raises(RuntimeError, match="requires an empty schema user_version 0; observed 1"):
        command.upgrade(_migration_config(connection), ALEMBIC_HEAD)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 1
        assert inspect(connection).get_table_names() == []
    engine.dispose()


def test_dataset_query_preflight_rejects_same_name_noop_trigger(tmp_path: Path) -> None:
    database_path = tmp_path / "catalog.sqlite3"
    write_engine = create_catalog_engine(f"sqlite+pysqlite:///{database_path}")
    upgrade_catalog(write_engine)
    trigger_name = "trg_dataset_family_projection_dirty_after_update"
    with write_engine.begin() as connection:
        connection.exec_driver_sql(f"DROP TRIGGER {trigger_name}")
        connection.exec_driver_sql(
            f"CREATE TRIGGER {trigger_name} AFTER UPDATE ON dataset_family BEGIN SELECT 1; END"
        )
    write_engine.dispose()

    read_engine = create_read_only_catalog_engine(database_path)
    preflight = dataset_catalog_query_preflight(read_engine)
    unified = unified_catalog_query_preflight(read_engine)
    assert preflight.present_trigger_names == tuple(sorted(DATASET_PROJECTION_DIRTY_TRIGGER_NAMES))
    assert preflight.present_trigger_definitions != (DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS)
    assert not preflight.passed
    assert not unified.passed
    read_engine.dispose()

def test_dataset_projection_enforces_canonical_json_unique_identity_and_foreign_keys(
    tmp_path: Path,
) -> None:
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    family_values = _family_values(keywords_json='["example"]\n')
    with engine.begin() as connection:
        result = connection.execute(dataset_family.insert().values(**family_values))
        assert result.inserted_primary_key is not None
        family_pk = int(result.inserted_primary_key[0])
        connection.execute(
            dataset_external_identifier.insert().values(
                identifier_fingerprint=b"i" * 32,
                owner_kind="FAMILY",
                family_pk=family_pk,
                release_pk=None,
                observation_pk=None,
                identifier_kind="ACCESSION",
                namespace="provider.example",
                value="example-accession",
            )
        )

    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(dataset_family.insert().values(**family_values))

    noncanonical = dict(family_values)
    noncanonical.update(
        family_id="family.noncanonical",
        record_fingerprint=b"n" * 32,
        record_json=(
            '{ "schema":"empirical-lawhood/planning/dataset-family","value":{},"version":"1.0.0"}\n'
        ),
    )
    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(dataset_family.insert().values(**noncanonical))

    noncanonical_array = dict(family_values)
    noncanonical_array.update(
        family_id="family.noncanonical-array",
        record_fingerprint=b"a" * 32,
        keywords_json='[ "example" ]\n',
    )
    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(dataset_family.insert().values(**noncanonical_array))

    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(
            dataset_external_identifier.insert().values(
                identifier_fingerprint=b"x" * 32,
                owner_kind="FAMILY",
                family_pk=999_999,
                release_pk=None,
                observation_pk=None,
                identifier_kind="ACCESSION",
                namespace="provider.example",
                value="missing-owner",
            )
        )
    engine.dispose()
