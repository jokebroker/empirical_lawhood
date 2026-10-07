"""SQLite engine policy, fixed production path, migrations and schema audit."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from math import gcd
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool, StaticPool

from .schema import DATASET_TABLE_NAMES, PROJECTION_TABLE_NAMES
from .migrations.dataset_tables import DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS, DATASET_PROJECTION_DIRTY_TRIGGER_NAMES, normalize_dataset_projection_trigger_sql

APPLICATION_ID: Final = 0x454C4157  # ASCII "ELAW"
BUSY_TIMEOUT_MS: Final = 5_000
SCHEMA_VERSION: Final = 1
ALEMBIC_HEAD: Final = "initial_catalog"
PRODUCTION_RELATIVE_PATH: Final = Path(".empirical-lawhood/experiment_catalog.sqlite3")
DEFAULT_SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT: Final = 10_000_000
MAX_SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT: Final = 100_000_000
DEFAULT_SCHEMA_AUDIT_DIAGNOSTIC_ROW_LIMIT: Final = 10_000
MAX_SCHEMA_AUDIT_DIAGNOSTIC_ROW_LIMIT: Final = 100_000
DEFAULT_SCHEMA_AUDIT_DIAGNOSTIC_BYTE_LIMIT: Final = 1_000_000
MAX_SCHEMA_AUDIT_DIAGNOSTIC_BYTE_LIMIT: Final = 10_000_000
_SCHEMA_AUDIT_PROGRESS_INTERVAL: Final = 100
_ALLOWED_BLOB_COLUMNS: Final = frozenset(
    {
        ("acquisition_attempt", "authorization_sha256"),
        ("acquisition_attempt", "expected_physical_sha256"),
        ("acquisition_attempt", "preview_sha256"),
        ("acquisition_attempt", "record_fingerprint"),
        ("artifact", "physical_sha256"),
        ("dataset_external_identifier", "identifier_fingerprint"),
        ("dataset_family", "record_fingerprint"),
        ("dataset_materialization", "logical_decoder_sha256"),
        ("dataset_materialization", "logical_sha256"),
        ("dataset_materialization", "physical_sha256"),
        ("dataset_materialization", "record_fingerprint"),
        ("dataset_materialization", "selector_sha256"),
        ("dataset_materialization", "verifier_sha256"),
        ("dataset_materialization", "verification_policy_sha256"),
        ("dataset_observation", "adapter_sha256"),
        ("dataset_observation", "canonical_request_sha256"),
        ("dataset_observation", "pagination_token_sha256"),
        ("dataset_observation", "record_fingerprint"),
        ("dataset_observation", "response_sha256"),
        ("dataset_projection_state", "snapshot_fingerprint"),
        ("dataset_release", "expected_physical_sha256"),
        ("dataset_release", "local_selector_sha256"),
        ("dataset_release", "manifest_sha256"),
        ("dataset_release", "record_fingerprint"),
        ("experiment_dataset_binding", "binding_receipt_sha256"),
        ("experiment_dataset_binding", "record_fingerprint"),
        ("experiment_dataset_binding", "selector_sha256"),
        ("experiment_dataset_binding", "transform_implementation_sha256"),
        ("experiment_dataset_binding", "transform_reference_sha256"),
        ("logical_artifact", "content_sha256"),
        ("receipt", "sha256"),
        ("scientific_object", "fingerprint"),
        ("workflow_run", "plan_fingerprint"),
    }
)
_FORBIDDEN_PAYLOAD_COLUMNS: Final = frozenset(
    {"array", "body", "figure", "model_bytes", "payload", "receipt_body", "row_data"}
)
_CATALOG_QUERY_TABLES: Final = (
    "alembic_version",
    "artifact",
    "knowledge_edge",
    "metric_definition",
    "metric_observation",
    "receipt",
    "scientific_object",
    "storage_root",
)
_DATASET_CATALOG_QUERY_TABLES: Final = tuple(sorted(DATASET_TABLE_NAMES))
_DATASET_CATALOG_QUERY_TRIGGERS: Final = tuple(sorted(DATASET_PROJECTION_DIRTY_TRIGGER_NAMES))
_DATASET_CATALOG_QUERY_TRIGGER_DEFINITIONS: Final = DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS


class CatalogSchemaAuditLimitReason(str, Enum):
    """Stable reason for an explicit schema audit stopping before a verdict."""

    SQLITE_VM_STEPS = "SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT_EXCEEDED"
    DIAGNOSTIC_ROWS = "SCHEMA_AUDIT_DIAGNOSTIC_ROW_LIMIT_EXCEEDED"
    DIAGNOSTIC_BYTES = "SCHEMA_AUDIT_DIAGNOSTIC_BYTE_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class CatalogSchemaAuditLimits:
    """Independent work and diagnostic ceilings for an explicit schema audit."""

    sqlite_vm_step_limit: int = DEFAULT_SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT
    diagnostic_row_limit: int = DEFAULT_SCHEMA_AUDIT_DIAGNOSTIC_ROW_LIMIT
    diagnostic_byte_limit: int = DEFAULT_SCHEMA_AUDIT_DIAGNOSTIC_BYTE_LIMIT

    def __post_init__(self) -> None:
        for field_name, value, maximum in (
            (
                "sqlite_vm_step_limit",
                self.sqlite_vm_step_limit,
                MAX_SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT,
            ),
            (
                "diagnostic_row_limit",
                self.diagnostic_row_limit,
                MAX_SCHEMA_AUDIT_DIAGNOSTIC_ROW_LIMIT,
            ),
            (
                "diagnostic_byte_limit",
                self.diagnostic_byte_limit,
                MAX_SCHEMA_AUDIT_DIAGNOSTIC_BYTE_LIMIT,
            ),
        ):
            if value <= 0 or value > maximum:
                raise ValueError(f"{field_name} must be in [1, {maximum}]")


class CatalogSchemaAuditLimitExceeded(RuntimeError):
    """Raised when an explicit audit cannot reach a complete bounded verdict."""

    def __init__(
        self,
        *,
        reason: CatalogSchemaAuditLimitReason,
        phase: str,
        limit: int,
        observed: int,
    ) -> None:
        self.reason = reason
        self.reason_code = reason.value
        self.phase = phase
        self.limit = limit
        self.observed = observed
        super().__init__(f"catalog schema audit stopped: {reason.value} during {phase}")


class _SchemaAuditDiagnosticBudget:
    """Bound decoded audit diagnostics before retaining them in an audit result."""

    def __init__(self, limits: CatalogSchemaAuditLimits) -> None:
        self._limits = limits
        self.rows = 0
        self.bytes = 0

    def consume_row(self, phase: str, *values: object) -> None:
        self.rows += 1
        if self.rows > self._limits.diagnostic_row_limit:
            raise CatalogSchemaAuditLimitExceeded(
                reason=CatalogSchemaAuditLimitReason.DIAGNOSTIC_ROWS,
                phase=phase,
                limit=self._limits.diagnostic_row_limit,
                observed=self.rows,
            )
        self.consume_bytes(phase, *values)

    def consume_bytes(self, phase: str, *values: object) -> None:
        # NUL separators make the retained-value accounting unambiguous and
        # conservatively include a byte of structure per decoded value.
        self.bytes += sum(len(str(value).encode("utf-8")) + 1 for value in values)
        if self.bytes > self._limits.diagnostic_byte_limit:
            raise CatalogSchemaAuditLimitExceeded(
                reason=CatalogSchemaAuditLimitReason.DIAGNOSTIC_BYTES,
                phase=phase,
                limit=self._limits.diagnostic_byte_limit,
                observed=self.bytes,
            )


@dataclass(frozen=True, slots=True)
class CatalogQueryPreflight:
    """Constant-schema-work identity check for public bounded catalog reads."""

    application_id: int
    user_version: int
    alembic_revision: str
    present_table_names: tuple[str, ...]
    query_only: bool
    foreign_keys_enabled: bool

    @property
    def passed(self) -> bool:
        return (
            self.application_id == APPLICATION_ID
            and self.user_version == SCHEMA_VERSION
            and self.alembic_revision == ALEMBIC_HEAD
            and self.present_table_names == _CATALOG_QUERY_TABLES
            and self.query_only
            and self.foreign_keys_enabled
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogQueryPreflight:
    """Constant-schema-work identity check for bounded dataset pages."""

    application_id: int
    user_version: int
    alembic_revision: str
    present_table_names: tuple[str, ...]
    present_trigger_names: tuple[str, ...]
    present_trigger_definitions: tuple[tuple[str, str], ...]
    query_only: bool
    foreign_keys_enabled: bool

    @property
    def passed(self) -> bool:
        return (
            self.application_id == APPLICATION_ID
            and self.user_version == SCHEMA_VERSION
            and self.alembic_revision == ALEMBIC_HEAD
            and self.present_table_names == _DATASET_CATALOG_QUERY_TABLES
            and self.present_trigger_names == _DATASET_CATALOG_QUERY_TRIGGERS
            and self.present_trigger_definitions == _DATASET_CATALOG_QUERY_TRIGGER_DEFINITIONS
            and self.query_only
            and self.foreign_keys_enabled
        )


@dataclass(frozen=True, slots=True)
class UnifiedCatalogQueryPreflight:
    """Exact fixed-work identity for both science and dataset projections."""

    science: CatalogQueryPreflight
    datasets: DatasetCatalogQueryPreflight

    @property
    def passed(self) -> bool:
        return self.science.passed and self.datasets.passed


@dataclass(frozen=True, slots=True)
class CatalogSchemaAudit:
    table_names: tuple[str, ...]
    projection_table_names: tuple[str, ...]
    blob_columns: tuple[str, ...]
    forbidden_payload_columns: tuple[str, ...]
    application_id: int
    user_version: int
    alembic_revision: str
    journal_mode: str
    auto_vacuum_mode: int
    busy_timeout_ms: int
    foreign_keys_enabled: bool
    integrity_result: str
    foreign_key_violation_count: int
    page_count: int
    page_size: int

    @property
    def passed(self) -> bool:
        return (
            set(self.projection_table_names) == set(PROJECTION_TABLE_NAMES)
            and not self.forbidden_payload_columns
            and set(self.blob_columns)
            == {f"{table}.{column}" for table, column in _ALLOWED_BLOB_COLUMNS}
            and self.application_id == APPLICATION_ID
            and self.user_version == SCHEMA_VERSION
            and self.alembic_revision == ALEMBIC_HEAD
            and self.journal_mode == "delete"
            and self.auto_vacuum_mode == 2
            and self.busy_timeout_ms == BUSY_TIMEOUT_MS
            and self.foreign_keys_enabled
            and self.integrity_result == "ok"
            and self.foreign_key_violation_count == 0
        )


def _configure_connection_local(dbapi_connection: sqlite3.Connection) -> None:
    """Set connection-local safety policy; these values do not rewrite SQLite."""

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def _configure_sqlite_read_write(dbapi_connection: Any, _connection_record: Any) -> None:
    if not isinstance(dbapi_connection, sqlite3.Connection):
        raise TypeError("platform catalog requires the sqlite3 DBAPI")
    _configure_connection_local(dbapi_connection)
    cursor = dbapi_connection.cursor()
    try:
        application_id = int(cursor.execute("PRAGMA application_id").fetchone()[0])
        if application_id not in (0, APPLICATION_ID):
            raise ValueError("catalog has another application identity; use an empty public catalog")
        # These PRAGMAs are persistent database policy and therefore belong only
        # on explicitly write-capable engines.
        cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
        cursor.execute(f"PRAGMA application_id={APPLICATION_ID}")
    finally:
        cursor.close()


def _configure_sqlite_read_only(dbapi_connection: Any, _connection_record: Any) -> None:
    if not isinstance(dbapi_connection, sqlite3.Connection):
        raise TypeError("platform catalog requires the sqlite3 DBAPI")
    _configure_connection_local(dbapi_connection)
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA query_only=ON")
        if int(cursor.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise RuntimeError("SQLite read-only query policy was not established")
    finally:
        cursor.close()


def create_catalog_engine(
    database_url: str = "sqlite+pysqlite:///:memory:",
) -> Engine:
    if not database_url.startswith("sqlite+pysqlite:///"):
        raise ValueError("platform catalog only supports SQLite pysqlite URLs")
    in_memory = database_url.endswith(":memory:")
    options: dict[str, object] = {
        "future": True,
        "connect_args": {"check_same_thread": False},
    }
    if in_memory:
        options["poolclass"] = StaticPool
    engine = create_engine(database_url, **options)
    event.listen(engine, "connect", _configure_sqlite_read_write)
    return engine


def create_read_only_catalog_engine(database_path: Path) -> Engine:
    """Open one existing SQLite catalog with URI `mode=ro` and no repair path."""

    if database_path.is_symlink():
        raise ValueError("read-only catalog cannot be a symlink")
    resolved = database_path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    sqlite_uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro"

    def connect_read_only() -> sqlite3.Connection:
        return sqlite3.connect(
            sqlite_uri,
            uri=True,
            check_same_thread=False,
        )

    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        creator=connect_read_only,
        poolclass=NullPool,
    )
    event.listen(engine, "connect", _configure_sqlite_read_only)
    return engine


def _migration_config() -> Config:
    config = Config()
    migrations = Path(__file__).resolve().parent / "migrations"
    config.set_main_option("script_location", str(migrations))
    return config


def upgrade_catalog(engine: Engine) -> None:
    config = _migration_config()
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def production_catalog_path(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    if not (root / ".git").exists():
        raise ValueError("production catalog root must be the exact Git worktree")
    runtime_directory = root / PRODUCTION_RELATIVE_PATH.parent
    if runtime_directory.exists() and runtime_directory.is_symlink():
        raise ValueError("repository-local catalog directory cannot be symlinked")
    return root / PRODUCTION_RELATIVE_PATH


def initialize_production_catalog(repo_root: Path) -> Engine:
    path = production_catalog_path(repo_root)
    path.parent.mkdir(mode=0o700, parents=False, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError("repository-local catalog file cannot be symlinked")
    engine = create_catalog_engine(f"sqlite+pysqlite:///{path}")
    upgrade_catalog(engine)
    audit = schema_audit(engine)
    if not audit.passed:
        engine.dispose()
        raise RuntimeError(f"catalog schema audit failed: {audit}")
    return engine


def catalog_query_preflight(engine: Engine) -> CatalogQueryPreflight:
    """Check query identity without row-scanning integrity diagnostics.

    Full integrity and foreign-key scans remain available through ``schema_audit``
    for explicit catalog-check/doctor workflows. A page request performs only
    fixed schema/identity reads before its two independently capped statements.
    """

    table_placeholders = ", ".join("?" for _ in _CATALOG_QUERY_TABLES)
    with engine.connect() as connection:
        application_id = int(str(connection.exec_driver_sql("PRAGMA application_id").scalar_one()))
        user_version = int(str(connection.exec_driver_sql("PRAGMA user_version").scalar_one()))
        query_only = bool(int(str(connection.exec_driver_sql("PRAGMA query_only").scalar_one())))
        foreign_keys_enabled = bool(
            int(str(connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()))
        )
        present_table_names = tuple(
            sorted(
                str(row[0])
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_schema "
                    f"WHERE type = 'table' AND name IN ({table_placeholders})",
                    _CATALOG_QUERY_TABLES,
                )
            )
        )
        revision = str(
            connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
        )
    return CatalogQueryPreflight(
        application_id=application_id,
        user_version=user_version,
        alembic_revision=revision,
        present_table_names=present_table_names,
        query_only=query_only,
        foreign_keys_enabled=foreign_keys_enabled,
    )


def dataset_catalog_query_preflight_connection(
    connection: Connection,
) -> DatasetCatalogQueryPreflight:
    """Verify dataset query identity on the exact page-work connection."""

    if not isinstance(connection, Connection):
        raise TypeError("dataset catalog preflight requires a SQLAlchemy Connection")
    table_placeholders = ", ".join("?" for _ in _DATASET_CATALOG_QUERY_TABLES)
    application_id = int(str(connection.exec_driver_sql("PRAGMA application_id").scalar_one()))
    user_version = int(str(connection.exec_driver_sql("PRAGMA user_version").scalar_one()))
    query_only = bool(int(str(connection.exec_driver_sql("PRAGMA query_only").scalar_one())))
    foreign_keys_enabled = bool(
        int(str(connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()))
    )
    present_table_names = tuple(
        sorted(
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_schema "
                f"WHERE type = 'table' AND name IN ({table_placeholders})",
                _DATASET_CATALOG_QUERY_TABLES,
            )
        )
    )
    trigger_rows = connection.exec_driver_sql(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' ORDER BY name LIMIT ?",
        (len(_DATASET_CATALOG_QUERY_TRIGGERS) + 1,),
    )
    present_trigger_definitions_list: list[tuple[str, str]] = []
    for raw_name, raw_sql in trigger_rows:
        name = str(raw_name)
        try:
            normalized_sql = normalize_dataset_projection_trigger_sql(raw_sql)
        except (TypeError, UnicodeError, ValueError):
            normalized_sql = "<invalid>"
        present_trigger_definitions_list.append((name, normalized_sql))
    present_trigger_definitions = tuple(sorted(present_trigger_definitions_list))
    present_trigger_names = tuple(name for name, _ in present_trigger_definitions)
    revision = str(
        connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
    )
    return DatasetCatalogQueryPreflight(
        application_id=application_id,
        user_version=user_version,
        alembic_revision=revision,
        present_table_names=present_table_names,
        present_trigger_names=present_trigger_names,
        present_trigger_definitions=present_trigger_definitions,
        query_only=query_only,
        foreign_keys_enabled=foreign_keys_enabled,
    )


def dataset_catalog_query_preflight(engine: Engine) -> DatasetCatalogQueryPreflight:
    """Verify the exact public dataset schema identity with constant bounded work."""

    with engine.connect() as connection:
        return dataset_catalog_query_preflight_connection(connection)


def unified_catalog_query_preflight_connection(
    connection: Connection,
) -> UnifiedCatalogQueryPreflight:
    """Verify unified query identity on the exact page-work connection."""

    if not isinstance(connection, Connection):
        raise TypeError("unified catalog preflight requires a SQLAlchemy Connection")

    science_placeholders = ", ".join("?" for _ in _CATALOG_QUERY_TABLES)
    dataset_placeholders = ", ".join("?" for _ in _DATASET_CATALOG_QUERY_TABLES)
    application_id = int(str(connection.exec_driver_sql("PRAGMA application_id").scalar_one()))
    user_version = int(str(connection.exec_driver_sql("PRAGMA user_version").scalar_one()))
    query_only = bool(int(str(connection.exec_driver_sql("PRAGMA query_only").scalar_one())))
    foreign_keys_enabled = bool(
        int(str(connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()))
    )
    revision = str(
        connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
    )
    science_tables = tuple(
        sorted(
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_schema "
                f"WHERE type = 'table' AND name IN ({science_placeholders})",
                _CATALOG_QUERY_TABLES,
            )
        )
    )
    dataset_tables = tuple(
        sorted(
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_schema "
                f"WHERE type = 'table' AND name IN ({dataset_placeholders})",
                _DATASET_CATALOG_QUERY_TABLES,
            )
        )
    )
    dataset_trigger_rows = connection.exec_driver_sql(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' ORDER BY name LIMIT ?",
        (len(_DATASET_CATALOG_QUERY_TRIGGERS) + 1,),
    )
    dataset_trigger_definitions_list: list[tuple[str, str]] = []
    for raw_name, raw_sql in dataset_trigger_rows:
        name = str(raw_name)
        try:
            normalized_sql = normalize_dataset_projection_trigger_sql(raw_sql)
        except (TypeError, UnicodeError, ValueError):
            normalized_sql = "<invalid>"
        dataset_trigger_definitions_list.append((name, normalized_sql))
    dataset_trigger_definitions = tuple(sorted(dataset_trigger_definitions_list))
    dataset_triggers = tuple(name for name, _ in dataset_trigger_definitions)
    return UnifiedCatalogQueryPreflight(
        science=CatalogQueryPreflight(
            application_id=application_id,
            user_version=user_version,
            alembic_revision=revision,
            present_table_names=science_tables,
            query_only=query_only,
            foreign_keys_enabled=foreign_keys_enabled,
        ),
        datasets=DatasetCatalogQueryPreflight(
            application_id=application_id,
            user_version=user_version,
            alembic_revision=revision,
            present_table_names=dataset_tables,
            present_trigger_names=dataset_triggers,
            present_trigger_definitions=dataset_trigger_definitions,
            query_only=query_only,
            foreign_keys_enabled=foreign_keys_enabled,
        ),
    )


def unified_catalog_query_preflight(engine: Engine) -> UnifiedCatalogQueryPreflight:
    """Verify both bounded public query surfaces without scanning data rows."""

    with engine.connect() as connection:
        return unified_catalog_query_preflight_connection(connection)


def schema_audit(
    engine: Engine,
    *,
    limits: CatalogSchemaAuditLimits = CatalogSchemaAuditLimits(),
) -> CatalogSchemaAudit:
    """Return a complete audit or raise a typed limit stop before any verdict.

    The progress handler is cumulative across reflection, integrity, foreign-key
    and policy statements. Diagnostic rows and decoded diagnostic bytes have
    separate cumulative ceilings; no truncated audit object is ever returned.
    """

    diagnostics = _SchemaAuditDiagnosticBudget(limits)
    table_names: list[str] = []
    blob_columns: list[str] = []
    forbidden: list[str] = []
    foreign_key_violation_count = 0
    work_steps = 0
    work_exhausted = False
    progress_interval = gcd(limits.sqlite_vm_step_limit, _SCHEMA_AUDIT_PROGRESS_INTERVAL)

    def enforce_work_limit() -> int:
        nonlocal work_steps, work_exhausted
        work_steps += progress_interval
        if work_steps >= limits.sqlite_vm_step_limit:
            work_exhausted = True
            return 1
        return 0

    with engine.connect() as connection:
        driver_connection = connection.connection.driver_connection
        set_progress_handler = getattr(driver_connection, "set_progress_handler", None)
        if set_progress_handler is None:
            raise RuntimeError("catalog schema audit backend cannot enforce a SQLite work limit")
        set_progress_handler(enforce_work_limit, progress_interval)
        phase = "table inventory"
        try:
            table_result = connection.exec_driver_sql(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
                "ORDER BY name"
            )
            for row in table_result:
                table_name = str(row[0])
                diagnostics.consume_row(phase, table_name)
                table_names.append(table_name)

            phase = "column inventory"
            for table_name in table_names:
                column_result = connection.exec_driver_sql(
                    "SELECT name, type FROM pragma_table_xinfo(?) ORDER BY cid",
                    (table_name,),
                )
                for row in column_result:
                    column_name = str(row[0])
                    type_name = str(row[1]).upper()
                    diagnostics.consume_row(phase, table_name, column_name, type_name)
                    if "BLOB" in type_name:
                        blob_columns.append(f"{table_name}.{column_name}")
                    if column_name in _FORBIDDEN_PAYLOAD_COLUMNS:
                        forbidden.append(f"{table_name}.{column_name}")

            def audit_scalar(statement: str, scalar_phase: str) -> object:
                nonlocal phase
                phase = scalar_phase
                value = connection.exec_driver_sql(statement).scalar_one()
                diagnostics.consume_bytes(phase, value)
                return value

            revision = audit_scalar("SELECT version_num FROM alembic_version", "revision")
            phase = "integrity diagnostics"
            integrity_rows: list[str] = []
            for row in connection.exec_driver_sql("PRAGMA integrity_check"):
                value = str(row[0])
                diagnostics.consume_row(phase, value)
                integrity_rows.append(value)
            integrity = (
                "ok"
                if tuple(integrity_rows) == ("ok",)
                else (
                    integrity_rows[0] if len(integrity_rows) == 1 else "multiple integrity failures"
                )
            )

            phase = "foreign-key diagnostics"
            for row in connection.exec_driver_sql("PRAGMA foreign_key_check"):
                diagnostics.consume_row(phase, *tuple(row))
                foreign_key_violation_count += 1

            application_id = audit_scalar("PRAGMA application_id", "application ID")
            user_version = audit_scalar("PRAGMA user_version", "user version")
            journal_mode = audit_scalar("PRAGMA journal_mode", "journal mode")
            auto_vacuum_mode = audit_scalar("PRAGMA auto_vacuum", "auto-vacuum mode")
            busy_timeout_ms = audit_scalar("PRAGMA busy_timeout", "busy timeout")
            foreign_keys_enabled = audit_scalar("PRAGMA foreign_keys", "foreign-key policy")
            page_count = audit_scalar("PRAGMA page_count", "page count")
            page_size = audit_scalar("PRAGMA page_size", "page size")
        except OperationalError as error:
            if work_exhausted:
                raise CatalogSchemaAuditLimitExceeded(
                    reason=CatalogSchemaAuditLimitReason.SQLITE_VM_STEPS,
                    phase=phase,
                    limit=limits.sqlite_vm_step_limit,
                    observed=work_steps,
                ) from error
            raise
        finally:
            set_progress_handler(None, 0)

    table_name_tuple = tuple(table_names)
    return CatalogSchemaAudit(
        table_names=table_name_tuple,
        projection_table_names=tuple(
            sorted(set(table_name_tuple).intersection(PROJECTION_TABLE_NAMES))
        ),
        blob_columns=tuple(sorted(blob_columns)),
        forbidden_payload_columns=tuple(sorted(forbidden)),
        application_id=int(str(application_id)),
        user_version=int(str(user_version)),
        alembic_revision=str(revision),
        journal_mode=str(journal_mode).lower(),
        auto_vacuum_mode=int(str(auto_vacuum_mode)),
        busy_timeout_ms=int(str(busy_timeout_ms)),
        foreign_keys_enabled=bool(int(str(foreign_keys_enabled))),
        integrity_result=str(integrity),
        foreign_key_violation_count=foreign_key_violation_count,
        page_count=int(str(page_count)),
        page_size=int(str(page_size)),
    )
