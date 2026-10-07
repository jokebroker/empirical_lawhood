"""Single-writer operational scheduler projection for local execution."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, Engine, alias, and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError, OperationalError

from empirical_lawhood.kernel.serialization import validate_sha256, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.execution import (
    OperationalAttempt,
    TaskAttemptDisposition,
    TaskBlockReason,
)

from .schema import lease, run_event, task_attempt, workflow_run


class OperationalWriteConflict(RuntimeError):
    pass


DEFAULT_OPERATIONAL_QUERY_WORK_LIMIT = 1_000_000
MAX_OPERATIONAL_QUERY_WORK_LIMIT = 10_000_000
DEFAULT_ATTEMPT_HISTORY_LIMIT = 100
MAX_ATTEMPT_HISTORY_LIMIT = 1_000
_SQLITE_PROGRESS_INTERVAL = 100
_OPERATIONAL_QUERY_TABLES = ("lease", "task_attempt", "workflow_run")


class OperationalQueryWorkLimitExceeded(RuntimeError):
    """Raised when a bounded operational read exhausts its SQLite VM budget."""

    def __init__(self, *, work_limit: int, work_steps: int) -> None:
        self.work_limit = work_limit
        self.work_steps = work_steps
        super().__init__("operational query exceeded its bounded SQLite work instruction limit")


@dataclass(frozen=True, slots=True)
class OperationalAttemptCursor:
    """Stable append-only position for one run's task/attempt ordering."""

    task_id: str
    attempt_id: str
    attempt_pk: int
    snapshot_max_pk: int

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="cursor.task_id")
        validate_stable_id(self.attempt_id, field_name="cursor.attempt_id")
        if self.attempt_pk <= 0:
            raise ValueError("attempt cursor primary key must be positive")
        if self.snapshot_max_pk < self.attempt_pk:
            raise ValueError("attempt cursor snapshot precedes its page position")


@dataclass(frozen=True, slots=True)
class OperationalAttemptPage:
    attempts: tuple[OperationalAttempt, ...]
    has_more: bool
    next_cursor: OperationalAttemptCursor | None
    query_work_limit: int
    query_work_steps: int


@dataclass(frozen=True, slots=True)
class OperationalQueryPreflight:
    present_table_names: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.present_table_names == _OPERATIONAL_QUERY_TABLES


def _digest(value: str) -> bytes:
    return bytes.fromhex(value)


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row._mapping  # type: ignore[no-any-return]


class SQLiteOperationalRepository:
    """Mutable operational state; never a source of scientific truth."""

    def __init__(
        self,
        engine: Engine,
        *,
        query_work_limit: int = DEFAULT_OPERATIONAL_QUERY_WORK_LIMIT,
    ) -> None:
        if query_work_limit <= 0 or query_work_limit > MAX_OPERATIONAL_QUERY_WORK_LIMIT:
            raise ValueError(
                f"operational query work limit must be in [1, {MAX_OPERATIONAL_QUERY_WORK_LIMIT}]"
            )
        self.engine = engine
        self.query_work_limit = query_work_limit
        self._writer_thread = threading.get_ident()

    @contextmanager
    def _bounded_connection(self) -> Iterator[tuple[Connection, list[int]]]:
        """Yield one connection with a repository-owned cumulative VM ceiling."""

        work_state = [0, 0]

        def enforce_work_limit() -> int:
            work_state[0] += _SQLITE_PROGRESS_INTERVAL
            if work_state[0] > self.query_work_limit:
                work_state[1] = 1
                return 1
            return 0

        with self.engine.connect() as connection:
            driver_connection = connection.connection.driver_connection
            set_progress_handler = getattr(driver_connection, "set_progress_handler", None)
            if set_progress_handler is None:
                raise RuntimeError("operational query backend cannot enforce a SQLite work limit")
            set_progress_handler(enforce_work_limit, _SQLITE_PROGRESS_INTERVAL)
            try:
                yield connection, work_state
            except OperationalError as error:
                if work_state[1]:
                    raise OperationalQueryWorkLimitExceeded(
                        work_limit=self.query_work_limit,
                        work_steps=work_state[0],
                    ) from error
                raise
            finally:
                set_progress_handler(None, 0)

    def _assert_writer(self) -> None:
        if threading.get_ident() != self._writer_thread:
            raise OperationalWriteConflict("operational writes require the owner thread")

    @staticmethod
    def _run_pk(connection: Connection, run_id: str) -> int:
        return int(
            connection.execute(
                select(workflow_run.c.pk).where(workflow_run.c.run_id == run_id)
            ).scalar_one()
        )

    @staticmethod
    def _attempt_pk(connection: Connection, attempt_id: str) -> int:
        return int(
            connection.execute(
                select(task_attempt.c.pk).where(task_attempt.c.attempt_id == attempt_id)
            ).scalar_one()
        )

    @staticmethod
    def _append_event(
        connection: Connection,
        *,
        run_pk: int,
        run_id: str,
        event_kind: str,
        reason_code: str | None = None,
    ) -> None:
        latest = connection.execute(
            select(func.max(run_event.c.sequence)).where(run_event.c.workflow_run_pk == run_pk)
        ).scalar_one()
        sequence = 0 if latest is None else int(latest) + 1
        connection.execute(
            run_event.insert().values(
                event_id=f"event.{run_id}.{sequence:06d}",
                workflow_run_pk=run_pk,
                sequence=sequence,
                event_kind=event_kind,
                reason_code=reason_code,
            )
        )

    def register_run(self, run_id: str, plan_sha256: str) -> None:
        self._assert_writer()
        validate_stable_id(run_id, field_name="run_id")
        validate_sha256(plan_sha256, field_name="plan_sha256")
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(workflow_run.c.pk, workflow_run.c.plan_fingerprint).where(
                    workflow_run.c.run_id == run_id
                )
            ).one_or_none()
            if existing is not None:
                if bytes(existing.plan_fingerprint) != _digest(plan_sha256):
                    raise OperationalWriteConflict("run ID is bound to another execution plan")
                return
            result = connection.execute(
                workflow_run.insert().values(
                    run_id=run_id,
                    plan_fingerprint=_digest(plan_sha256),
                    state=OperationalStatus.PENDING.value,
                )
            )
            primary_key = result.inserted_primary_key
            if not primary_key or primary_key[0] is None:
                raise RuntimeError("workflow run insert returned no primary key")
            self._append_event(
                connection,
                run_pk=int(primary_key[0]),
                run_id=run_id,
                event_kind="RUN_REGISTERED",
            )

    def run_status(self, run_id: str) -> OperationalStatus:
        validate_stable_id(run_id, field_name="run_id")
        with self._bounded_connection() as (connection, _work_state):
            value = connection.execute(
                select(workflow_run.c.state).where(workflow_run.c.run_id == run_id)
            ).scalar_one()
        return OperationalStatus(value)

    def query_preflight(self) -> OperationalQueryPreflight:
        """Confirm the fixed status-query table set under the same VM ceiling."""

        placeholders = ", ".join("?" for _ in _OPERATIONAL_QUERY_TABLES)
        with self._bounded_connection() as (connection, _work_state):
            present = tuple(
                sorted(
                    str(row[0])
                    for row in connection.exec_driver_sql(
                        "SELECT name FROM sqlite_schema "
                        f"WHERE type = 'table' AND name IN ({placeholders})",
                        _OPERATIONAL_QUERY_TABLES,
                    )
                )
            )
        return OperationalQueryPreflight(present_table_names=present)

    @staticmethod
    def _attempt_from_row(row: Mapping[str, Any], *, ordinal: int) -> OperationalAttempt:
        return OperationalAttempt(
            attempt_id=row["attempt_id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            ordinal=ordinal,
            disposition=TaskAttemptDisposition(row["state"]),
            reason_code=row["reason_code"],
            lease_id=row["lease_id"],
            lease_expires_epoch_seconds=row["expires_epoch_seconds"],
        )

    def current_attempts(self, run_id: str) -> tuple[OperationalAttempt, ...]:
        """Return only the latest immutable attempt for each task in one run."""

        validate_stable_id(run_id, field_name="run_id")
        lease_alias = alias(lease, name="current_attempt_lease")
        current = (
            select(
                task_attempt.c.task_id,
                func.max(task_attempt.c.pk).label("current_attempt_pk"),
                func.count(task_attempt.c.pk).label("attempt_ordinal"),
            )
            .join(workflow_run, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
            .where(workflow_run.c.run_id == run_id)
            .group_by(task_attempt.c.task_id)
            .subquery("current_attempt_by_task")
        )
        statement = (
            select(
                task_attempt,
                workflow_run.c.run_id,
                current.c.attempt_ordinal,
                lease_alias.c.lease_id,
                lease_alias.c.expires_epoch_seconds,
            )
            .join(workflow_run, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
            .join(current, task_attempt.c.pk == current.c.current_attempt_pk)
            .outerjoin(lease_alias, lease_alias.c.task_attempt_pk == task_attempt.c.pk)
            .order_by(task_attempt.c.task_id)
        )
        with self._bounded_connection() as (connection, _work_state):
            rows = tuple(map(_row_mapping, connection.execute(statement)))
        return tuple(
            self._attempt_from_row(row, ordinal=int(row["attempt_ordinal"])) for row in rows
        )

    def attempt_history_page(
        self,
        run_id: str,
        *,
        limit: int = DEFAULT_ATTEMPT_HISTORY_LIMIT,
        cursor: OperationalAttemptCursor | None = None,
    ) -> OperationalAttemptPage:
        """Read an explicit limit+1 page in stable task/append order."""

        validate_stable_id(run_id, field_name="run_id")
        if limit <= 0 or limit > MAX_ATTEMPT_HISTORY_LIMIT:
            raise ValueError(f"attempt history limit must be in [1, {MAX_ATTEMPT_HISTORY_LIMIT}]")
        lease_alias = alias(lease, name="history_attempt_lease")
        predicates: list[Any] = [workflow_run.c.run_id == run_id]
        if cursor is not None:
            predicates.append(
                or_(
                    task_attempt.c.task_id > cursor.task_id,
                    and_(
                        task_attempt.c.task_id == cursor.task_id,
                        task_attempt.c.pk > cursor.attempt_pk,
                    ),
                )
            )
        starting_ordinal = 0
        with self._bounded_connection() as (connection, work_state):
            snapshot_max_pk = (
                cursor.snapshot_max_pk
                if cursor is not None
                else int(
                    connection.execute(
                        select(func.coalesce(func.max(task_attempt.c.pk), 0))
                        .join(
                            workflow_run,
                            task_attempt.c.workflow_run_pk == workflow_run.c.pk,
                        )
                        .where(workflow_run.c.run_id == run_id)
                    ).scalar_one()
                )
            )
            predicates.append(task_attempt.c.pk <= snapshot_max_pk)
            if cursor is not None:
                anchor = connection.execute(
                    select(
                        task_attempt.c.task_id,
                        task_attempt.c.attempt_id,
                        task_attempt.c.pk,
                        func.count().over().label("attempt_ordinal"),
                    )
                    .join(workflow_run, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
                    .where(
                        workflow_run.c.run_id == run_id,
                        task_attempt.c.task_id == cursor.task_id,
                        task_attempt.c.pk <= cursor.attempt_pk,
                    )
                    .order_by(task_attempt.c.pk.desc())
                    .limit(1)
                ).one_or_none()
                if (
                    anchor is None
                    or str(anchor.task_id) != cursor.task_id
                    or str(anchor.attempt_id) != cursor.attempt_id
                    or int(anchor.pk) != cursor.attempt_pk
                ):
                    raise ValueError("attempt history cursor is not an exact run position")
                starting_ordinal = int(anchor.attempt_ordinal)
            statement = (
                select(
                    task_attempt,
                    workflow_run.c.run_id,
                    lease_alias.c.lease_id,
                    lease_alias.c.expires_epoch_seconds,
                )
                .join(workflow_run, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
                .outerjoin(lease_alias, lease_alias.c.task_attempt_pk == task_attempt.c.pk)
                .where(*predicates)
                .order_by(task_attempt.c.task_id, task_attempt.c.pk)
                .limit(limit + 1)
            )
            raw_rows = tuple(map(_row_mapping, connection.execute(statement)))
            work_steps = work_state[0]
        has_more = len(raw_rows) > limit
        page_rows = raw_rows[:limit]
        attempts: list[OperationalAttempt] = []
        previous_task_id = cursor.task_id if cursor is not None else None
        ordinal = starting_ordinal
        for row in page_rows:
            task_id = str(row["task_id"])
            if task_id == previous_task_id:
                ordinal += 1
            else:
                previous_task_id = task_id
                ordinal = 1
            attempts.append(self._attempt_from_row(row, ordinal=ordinal))
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = OperationalAttemptCursor(
                task_id=str(last["task_id"]),
                attempt_id=str(last["attempt_id"]),
                attempt_pk=int(last["pk"]),
                snapshot_max_pk=snapshot_max_pk,
            )
        return OperationalAttemptPage(
            attempts=tuple(attempts),
            has_more=has_more,
            next_cursor=next_cursor,
            query_work_limit=self.query_work_limit,
            query_work_steps=work_steps,
        )

    def set_run_status(self, run_id: str, status: OperationalStatus) -> None:
        self._assert_writer()
        validate_stable_id(run_id, field_name="run_id")
        with self.engine.begin() as connection:
            run_pk = self._run_pk(connection, run_id)
            current = OperationalStatus(
                connection.execute(
                    select(workflow_run.c.state).where(workflow_run.c.pk == run_pk)
                ).scalar_one()
            )
            if current is status:
                return
            connection.execute(
                update(workflow_run).where(workflow_run.c.pk == run_pk).values(state=status.value)
            )
            self._append_event(
                connection,
                run_pk=run_pk,
                run_id=run_id,
                event_kind=f"RUN_{status.value}",
            )

    def attempts(self, run_id: str) -> tuple[OperationalAttempt, ...]:
        validate_stable_id(run_id, field_name="run_id")
        lease_alias = alias(lease, name="attempt_lease")
        statement = (
            select(
                task_attempt,
                workflow_run.c.run_id,
                func.row_number()
                .over(
                    partition_by=task_attempt.c.task_id,
                    order_by=task_attempt.c.pk,
                )
                .label("attempt_ordinal"),
                lease_alias.c.lease_id,
                lease_alias.c.expires_epoch_seconds,
            )
            .join(workflow_run, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
            .outerjoin(lease_alias, lease_alias.c.task_attempt_pk == task_attempt.c.pk)
            .where(workflow_run.c.run_id == run_id)
            .order_by(task_attempt.c.task_id, task_attempt.c.pk)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement)
            return tuple(
                OperationalAttempt(
                    attempt_id=row["attempt_id"],
                    run_id=row["run_id"],
                    task_id=row["task_id"],
                    ordinal=int(row["attempt_ordinal"]),
                    disposition=TaskAttemptDisposition(row["state"]),
                    reason_code=row["reason_code"],
                    lease_id=row["lease_id"],
                    lease_expires_epoch_seconds=row["expires_epoch_seconds"],
                )
                for row in map(_row_mapping, rows)
            )

    def start_attempt(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_id: str,
        lease_id: str,
        owner_id: str,
        expires_epoch_seconds: int,
    ) -> None:
        self._assert_writer()
        for name, value in (
            ("run_id", run_id),
            ("task_id", task_id),
            ("attempt_id", attempt_id),
            ("lease_id", lease_id),
            ("owner_id", owner_id),
        ):
            validate_stable_id(value, field_name=name)
        try:
            with self.engine.begin() as connection:
                run_pk = self._run_pk(connection, run_id)
                result = connection.execute(
                    task_attempt.insert().values(
                        attempt_id=attempt_id,
                        workflow_run_pk=run_pk,
                        task_id=task_id,
                        state=TaskAttemptDisposition.RUNNING.value,
                        reason_code=None,
                    )
                )
                primary_key = result.inserted_primary_key
                if not primary_key or primary_key[0] is None:
                    raise RuntimeError("task attempt insert returned no primary key")
                connection.execute(
                    lease.insert().values(
                        lease_id=lease_id,
                        task_attempt_pk=int(primary_key[0]),
                        owner_id=owner_id,
                        expires_epoch_seconds=expires_epoch_seconds,
                    )
                )
                self._append_event(
                    connection,
                    run_pk=run_pk,
                    run_id=run_id,
                    event_kind="TASK_ATTEMPT_STARTED",
                )
        except IntegrityError as error:
            raise OperationalWriteConflict("attempt or lease identity already exists") from error

    def _finish_attempt(
        self,
        attempt_id: str,
        disposition: TaskAttemptDisposition,
        reason_code: str | None,
    ) -> None:
        self._assert_writer()
        validate_stable_id(attempt_id, field_name="attempt_id")
        with self.engine.begin() as connection:
            attempt_pk = self._attempt_pk(connection, attempt_id)
            run_row = connection.execute(
                select(workflow_run.c.pk, workflow_run.c.run_id)
                .join(task_attempt, task_attempt.c.workflow_run_pk == workflow_run.c.pk)
                .where(task_attempt.c.pk == attempt_pk)
            ).one()
            connection.execute(delete(lease).where(lease.c.task_attempt_pk == attempt_pk))
            updated = connection.execute(
                update(task_attempt)
                .where(
                    task_attempt.c.pk == attempt_pk,
                    task_attempt.c.state == TaskAttemptDisposition.RUNNING.value,
                )
                .values(state=disposition.value, reason_code=reason_code)
            )
            if updated.rowcount != 1:
                raise OperationalWriteConflict("attempt is not in RUNNING state")
            self._append_event(
                connection,
                run_pk=int(run_row.pk),
                run_id=str(run_row.run_id),
                event_kind=f"TASK_ATTEMPT_{disposition.value}",
                reason_code=reason_code,
            )

    def complete_attempt(self, attempt_id: str) -> None:
        self._finish_attempt(attempt_id, TaskAttemptDisposition.SUCCEEDED, None)

    def fail_attempt(self, attempt_id: str, reason_code: str) -> None:
        validate_stable_id(reason_code, field_name="reason_code")
        self._finish_attempt(attempt_id, TaskAttemptDisposition.FAILED, reason_code)

    def block_attempt(
        self,
        attempt_id: str,
        reason: TaskBlockReason,
    ) -> None:
        if not isinstance(reason, TaskBlockReason):
            raise TypeError("blocked attempts require a typed TaskBlockReason")
        self._finish_attempt(
            attempt_id,
            TaskAttemptDisposition.BLOCKED,
            reason.value,
        )

    def block_task(
        self,
        run_id: str,
        task_id: str,
        reason: TaskBlockReason,
    ) -> None:
        self._assert_writer()
        if not isinstance(reason, TaskBlockReason):
            raise TypeError("blocked tasks require a typed TaskBlockReason")
        reason_code = reason.value
        for name, value in (("run_id", run_id), ("task_id", task_id)):
            validate_stable_id(value, field_name=name)
        with self.engine.begin() as connection:
            run_pk = self._run_pk(connection, run_id)
            prior_count = int(
                connection.execute(
                    select(func.count(task_attempt.c.pk)).where(
                        task_attempt.c.workflow_run_pk == run_pk,
                        task_attempt.c.task_id == task_id,
                    )
                ).scalar_one()
            )
            ordinal = prior_count + 1
            attempt_id = f"{run_id}.{task_id}.block-{ordinal:03d}"
            connection.execute(
                task_attempt.insert().values(
                    attempt_id=attempt_id,
                    workflow_run_pk=run_pk,
                    task_id=task_id,
                    state=TaskAttemptDisposition.BLOCKED.value,
                    reason_code=reason_code,
                )
            )
            self._append_event(
                connection,
                run_pk=run_pk,
                run_id=run_id,
                event_kind="TASK_BLOCKED",
                reason_code=reason_code,
            )
