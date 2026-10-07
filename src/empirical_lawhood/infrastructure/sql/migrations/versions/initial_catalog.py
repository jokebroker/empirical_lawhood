"""Initialize the complete bounded catalog and dataset projections atomically."""

from __future__ import annotations

from alembic import op

from empirical_lawhood.infrastructure.sql.migrations.catalog_tables import CATALOG_METADATA
from empirical_lawhood.infrastructure.sql.migrations.dataset_tables import (
    DATASET_METADATA,
    DATASET_TABLES,
    DATASET_PROJECTION_DIRTY_TRIGGER_SQL,
    EMPTY_DATASET_SNAPSHOT_CANONICAL_BYTES,
    EMPTY_DATASET_SNAPSHOT_FINGERPRINT,
    EMPTY_DATASET_SNAPSHOT_SCHEMA,
    dataset_projection_state,
)

revision = "initial_catalog"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    user_version = int(connection.exec_driver_sql("PRAGMA user_version").scalar_one())
    if user_version != 0:
        raise RuntimeError(
            "catalog initialization requires an empty schema user_version 0; "
            f"observed {user_version}"
        )
    collisions = tuple(
        (str(row[0]), str(row[1]))
        for row in connection.exec_driver_sql(
            "SELECT type, name FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
            "AND name != 'alembic_version' ORDER BY type, name"
        )
    )
    if collisions:
        names = ", ".join(f"{kind} {name}" for kind, name in collisions)
        raise RuntimeError("catalog initialization refuses pre-existing schema objects: " + names)
    CATALOG_METADATA.create_all(bind=connection, checkfirst=False)
    DATASET_METADATA.create_all(
        bind=connection,
        tables=DATASET_TABLES,
        checkfirst=False,
    )
    for trigger_sql in DATASET_PROJECTION_DIRTY_TRIGGER_SQL:
        connection.exec_driver_sql(trigger_sql)
    connection.execute(
        dataset_projection_state.insert().values(
            state_id=1,
            snapshot_schema=EMPTY_DATASET_SNAPSHOT_SCHEMA,
            snapshot_fingerprint=EMPTY_DATASET_SNAPSHOT_FINGERPRINT,
            family_count=0,
            release_count=0,
            observation_count=0,
            materialization_count=0,
            acquisition_attempt_count=0,
            binding_count=0,
            canonical_byte_count=EMPTY_DATASET_SNAPSHOT_CANONICAL_BYTES,
            projection_valid=1,
            projection_revision=0,
        )
    )
    op.execute("PRAGMA user_version=1")


def downgrade() -> None:
    raise RuntimeError("catalog downgrades are intentionally unsupported")
