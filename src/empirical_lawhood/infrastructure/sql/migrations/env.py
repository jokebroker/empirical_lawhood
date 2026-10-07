from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from empirical_lawhood.infrastructure.sql.schema import metadata


def run_migrations_offline() -> None:
    url = context.config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if not isinstance(connection, Connection):
        raise RuntimeError("catalog migration requires an injected SQLAlchemy connection")
    if connection.in_transaction():
        raise RuntimeError("catalog migration requires a fresh non-transactional connection")
    context.configure(
        connection=connection,
        target_metadata=metadata,
        transactional_ddl=True,
        transaction_per_migration=True,
    )
    # Python 3.11's sqlite3 default transaction mode does not begin a database
    # transaction for DDL.  Issue BEGIN explicitly so tables, indexes, PRAGMA
    # user_version and Alembic's revision row commit or roll back as one unit.
    connection.exec_driver_sql("BEGIN IMMEDIATE")
    try:
        with context.begin_transaction():
            context.run_migrations()
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
