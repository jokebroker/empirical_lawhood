# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.runtime_platform.support import ProtocolFixture
from tests.runtime_platform.test_execution_scheduler import (
    CrashAfterReceipt,
    _plane,
    _recovery,
    _runner,
    _runtime,
)
from tests.runtime_platform.test_study_package_preservation import _authorized_inputs, _compile_authorized

from empirical_lawhood.infrastructure.execution import InjectedSchedulerCrash
from empirical_lawhood.infrastructure.recovery import (
    ExternalRunRecoveryStore,
    RunRecoveryError,
    decode_task_recovery_event,
    decode_run_recovery_index,
)
from empirical_lawhood.infrastructure.sql import (
    SQLiteOperationalRepository,
    create_catalog_engine,
    upgrade_catalog,
)
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.serialization import CanonicalizationError
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.execution import TaskAttemptDisposition
from empirical_lawhood.runtime.plans import CandidateExecutionPlan
from empirical_lawhood.runtime.recovery import CandidateRunRecoveryIndex, TaskRecoveryEvent, build_run_recovery_index


def _delete_test_projection(
    repository: SQLiteOperationalRepository,
    database_path: Path,
) -> None:
    repository.engine.dispose()
    database_path.unlink()
    for suffix in ("-shm", "-wal"):
        database_path.with_name(f"{database_path.name}{suffix}").unlink(missing_ok=True)


def test_recovery_index_round_trips_and_rejects_unknown_fields(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    index = scheduler.recovery_index
    assert decode_run_recovery_index(index.canonical_bytes()) == index
    document = json.loads(index.canonical_bytes())
    document["value"]["unexpected"] = True
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(CanonicalizationError):
        decode_run_recovery_index(payload)
    repository.engine.dispose()


def test_exact_edge_recovery_index_persists_as_additive(
    tmp_path: Path,
) -> None:
    issue_root = tmp_path / "issue"
    issue_root.mkdir()
    values = _authorized_inputs(issue_root)
    _package, run_plan, execution_plan, _parity = _compile_authorized(
        values,
        run_plan_id="run.exact-edge-recovery",
    )
    assert isinstance(execution_plan, CandidateExecutionPlan)
    index = build_run_recovery_index(
        execution_plan,
        run_id=run_plan.run_plan_id,
        wave_id="wave.exact-edge-recovery",
        authority_identities=(execution_plan.candidate,),
    )
    assert isinstance(index, CandidateRunRecoveryIndex)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    store = ExternalRunRecoveryStore(_plane(recovery_root))

    store.freeze(index)

    assert (
        store.read_index(
            index.index_relative_path,
            expected_schema=CandidateRunRecoveryIndex.SCHEMA,
        )
        == index
    )
    with pytest.raises(RunRecoveryError, match="artifact violates"):
        store.read_index(
            index.index_relative_path,
            expected_schema='empirical-lawhood/runtime/protocol-run-recovery-index',
        )


def test_total_sqlite_loss_rebuilds_terminal_state_with_zero_executor_calls(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane, _receipts, first_runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    first = scheduler.execute(run_id, protocol_fixture.execution_plan)
    expected_attempts = tuple(
        (attempt.attempt_id, attempt.task_id, attempt.disposition)
        for attempt in repository.attempts(run_id)
    )
    assert first.status is OperationalStatus.SUCCEEDED
    database = tmp_path / "catalog.sqlite3"
    _delete_test_projection(repository, database)

    resumed, rebuilt, _plane, _receipts, recovery_runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    recovered = resumed.execute(run_id, protocol_fixture.execution_plan)

    assert recovered == first
    assert (
        tuple(
            (attempt.attempt_id, attempt.task_id, attempt.disposition)
            for attempt in rebuilt.attempts(run_id)
        )
        == expected_attempts
    )
    assert all(not runner.contexts for runner in recovery_runners)
    assert all(len(runner.contexts) == 1 for runner in first_runners)
    rebuilt.engine.dispose()


def test_terminal_recovery_uses_bounded_index_lineage_and_commits_receipts(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id

    result = scheduler.execute(run_id, protocol_fixture.execution_plan)
    terminal = scheduler.recovery_store.read_terminal(scheduler.recovery_index)

    assert terminal is not None
    expected_receipts = {
        receipt.attempt_id: (receipt.receipt_id, receipt.fingerprint())
        for receipt in result.receipts
    }
    assert {
        attempt.attempt_id: (attempt.receipt_id, attempt.receipt_sha256)
        for attempt in terminal.attempts
        if attempt.receipt_id is not None
    } == expected_receipts

    manifest_path = tmp_path / (
        f"{scheduler.recovery_index.terminal_event_relative_path}.manifest.json"
    )
    manifest = json.loads(manifest_path.read_bytes())
    lineage = manifest["value"]["logical"]["value"]["lineage_parents"]
    assert len(lineage) == 1
    assert lineage[0]["value"]["identity"]["value"]["object_id"] == (
        scheduler.recovery_index.recovery_index_id
    )
    repository.engine.dispose()


def test_recovery_event_is_strict_and_unsupported_schemas_are_source_bound(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    scheduler, repository, _plane, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    scheduler.execute(run_id, protocol_fixture.execution_plan)
    candidate = scheduler.recovery_index.tasks[0].attempts[0]
    payload = tmp_path.joinpath(candidate.event_relative_path).read_bytes()

    event = decode_task_recovery_event(payload)
    assert isinstance(event, TaskRecoveryEvent)
    assert event.failure is None
    document = json.loads(payload)
    document["schema"] = 'empirical-lawhood/runtime/candidate-task-recovery-event'
    old_schema_payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with pytest.raises(CanonicalizationError, match="schema is unsupported"):
        decode_task_recovery_event(old_schema_payload)
    repository.engine.dispose()


def test_post_receipt_pre_sql_crash_survives_total_projection_loss_without_repeat(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
) -> None:
    crash = CrashAfterReceipt("freeze")
    scheduler, repository, _plane, _receipts, first_runners = _runtime(
        tmp_path,
        protocol_fixture,
        failure_injector=crash,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    with pytest.raises(InjectedSchedulerCrash, match="freeze"):
        scheduler.execute(run_id, protocol_fixture.execution_plan)
    assert len(_runner(first_runners, "reference.freeze").contexts) == 1
    _delete_test_projection(repository, tmp_path / "catalog.sqlite3")

    resumed, rebuilt, _plane, _receipts, recovery_runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    recovered = resumed.execute(run_id, protocol_fixture.execution_plan)

    assert recovered.status is OperationalStatus.SUCCEEDED
    assert not _runner(recovery_runners, "reference.freeze").contexts
    assert len(_runner(first_runners, "reference.freeze").contexts) == 1
    assert all(
        attempt.disposition is TaskAttemptDisposition.SUCCEEDED
        for attempt in rebuilt.attempts(run_id)
    )
    rebuilt.engine.dispose()


@pytest.mark.parametrize("mutation", ("missing", "substituted", "sidecar-tampered"))
def test_terminal_recovery_fails_closed_on_receipt_custody_mutation(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    mutation: str,
) -> None:
    scheduler, repository, _plane, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    run_id = protocol_fixture.run_plan.run_plan_id
    result = scheduler.execute(run_id, protocol_fixture.execution_plan)
    receipt = result.receipts[0]
    relative_path = f"runs/{run_id}/receipts/{receipt.task_id}/{receipt.attempt_id}.json"
    receipt_path = tmp_path / relative_path
    sidecar_path = tmp_path / f"{relative_path}.manifest.json"
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "substituted":
        payload = bytearray(receipt_path.read_bytes())
        payload[len(payload) // 2] ^= 1
        receipt_path.write_bytes(payload)
    else:
        payload = bytearray(sidecar_path.read_bytes())
        payload[len(payload) // 2] ^= 1
        sidecar_path.write_bytes(payload)
    _delete_test_projection(repository, tmp_path / "catalog.sqlite3")

    resumed, rebuilt, _plane, _receipts, recovery_runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    with pytest.raises(RunRecoveryError, match="terminal"):
        resumed.execute(run_id, protocol_fixture.execution_plan)
    assert all(not runner.contexts for runner in recovery_runners)
    rebuilt.engine.dispose()


def test_recovery_probes_scale_linearly_and_never_scan_directories(
    tmp_path: Path,
    protocol_fixture: ProtocolFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scheduler, base_repository, plane, _receipts, _runners = _runtime(
        tmp_path,
        protocol_fixture,
    )
    base_repository.engine.dispose()
    base = protocol_fixture.execution_plan
    expanded = replace(
        base,
        tasks=tuple(replace(task, maximum_attempts=20) for task in base.tasks),
    )
    small_index, store = _recovery(
        plane,
        base,
        authority=protocol_fixture.run_plan.authorization,
        wave_id="wave.scaling-small",
    )
    large_index, _ = _recovery(
        plane,
        expanded,
        authority=protocol_fixture.run_plan.authorization,
        wave_id="wave.scaling-large",
    )
    store.freeze(small_index)
    store.freeze(large_index)
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda _self: (_ for _ in ()).throw(AssertionError("directory scan")),
    )
    monkeypatch.setattr(
        Path,
        "glob",
        lambda _self, _pattern: (_ for _ in ()).throw(AssertionError("directory scan")),
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda _self, _pattern: (_ for _ in ()).throw(AssertionError("directory scan")),
    )

    reports = []
    for name, plan, index in (
        ("small", base, small_index),
        ("large", expanded, large_index),
    ):
        engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / f'{name}.sqlite3'}")
        upgrade_catalog(engine)
        repository = SQLiteOperationalRepository(engine)
        reports.append(
            store.reconcile(
                index,
                plan,
                repository,
                ExternalTaskReceiptStore(plane),
            )
        )
        engine.dispose()

    small, large = reports
    assert small.receipt_probe_count == small.candidate_count
    assert large.receipt_probe_count == large.candidate_count
    assert small.event_probe_count == small.candidate_count + len(base.tasks)
    assert large.event_probe_count == large.candidate_count + len(expanded.tasks)
    assert large.candidate_count == 10 * small.candidate_count
