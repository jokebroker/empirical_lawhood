"""Stop a bound local service without stranding any native task attempt."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from datetime import datetime, timezone
import hashlib
import os
import select
import signal
import time
from pathlib import Path
import platform
from typing import Protocol

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.bounded_process import run_bounded_command
from empirical_lawhood.infrastructure.execution_resource_envelopes import ExternalExecutionResourceEnvelopeStore
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.execution_envelope import RunExecutionResourceEnvelope
from empirical_lawhood.runtime.maintenance import ContinuationBoundInterruptionAuthority, InterruptionAuthority, MaintenanceProcessStopAuthority, MaintenanceReceiptBoundary, MaintenanceStopAuthority, maintenance_receipt_boundary


class MaintenanceServiceControl(Protocol):
    def require_running(self, authority: MaintenanceStopAuthority) -> None: ...

    def stop(self, authority: MaintenanceStopAuthority) -> None: ...


class LocalUserServiceControl:
    """Fixed systemctl argv, exact unit/ExecStart/PID/start-time binding."""

    @staticmethod
    def _properties(service_name: str) -> dict[str, str]:
        result = run_bounded_command(
            [
                "systemctl",
                "--user",
                "show",
                service_name,
                "-p",
                "MainPID",
                "-p",
                "ExecStart",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "Restart",
            ],
            timeout_seconds=10,
            maximum_stdout_bytes=64 * 1024,
            maximum_stderr_bytes=16 * 1024,
        )
        if result.returncode:
            raise ValueError("maintenance service inspection failed")
        return dict(
            line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line
        )

    def require_running(self, authority: MaintenanceStopAuthority) -> None:
        state = self._properties(authority.service_name)
        if (
            state.get("MainPID") != str(authority.main_pid)
            or state.get("ActiveState") != "active"
            or state.get("SubState") != "running"
            or state.get("Restart") != "no"
            or hashlib.sha256(state.get("ExecStart", "").encode()).hexdigest()
            != authority.service_exec_start_sha256
        ):
            raise ValueError(
                "maintenance service identity changed or is not running without restart"
            )
        # procfs reports size zero and changing counters; only PID start time
        # is stable. Bound the stream directly instead of requiring file-size
        # stability, which is appropriate for immutable artifact files.
        with Path(f"/proc/{authority.main_pid}/stat").open("rb") as stream:
            process_bytes = stream.read(16385)
        if len(process_bytes) > 16384:
            raise ValueError("maintenance process identity exceeds its bound")
        process_stat = process_bytes.decode()
        tail = process_stat.rsplit(") ", 1)[1].split()
        if int(tail[19]) != authority.main_start_ticks:
            raise ValueError("maintenance process PID was reused")

    def stop(self, authority: MaintenanceStopAuthority) -> None:
        self.require_running(authority)
        result = run_bounded_command(
            ["systemctl", "--user", "stop", authority.service_name],
            timeout_seconds=30,
            maximum_stdout_bytes=16 * 1024,
            maximum_stderr_bytes=16 * 1024,
        )
        if result.returncode:
            raise ValueError("maintenance service stop did not complete")
        state = self._properties(authority.service_name)
        if state.get("MainPID") != "0" or state.get("ActiveState") not in {"inactive", "failed"}:
            raise ValueError("maintenance service remains active after stop")


class MaintenanceProcessControl(Protocol):
    def require_running(self, authority: MaintenanceProcessStopAuthority) -> None: ...

    def stop(self, authority: MaintenanceProcessStopAuthority) -> None: ...


class LocalProcessTreeControl:
    """Freeze a bounded same-owner tree, then stop through stable Linux pidfds."""

    @staticmethod
    def _syscall(number: int, *arguments: object) -> int:
        # Linux x86-64 and asm-generic (aarch64) UAPI numbers. This environment's
        # Python/libc lack pidfd wrappers, so use the documented syscall route:
        # https://man7.org/linux/man-pages/man2/pidfd_open.2.html
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
            raise ValueError("direct process stopping requires supported Linux pidfds")
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = int(syscall(ctypes.c_long(number), *arguments))
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    @classmethod
    def _open_pidfd(cls, pid: int) -> int:
        return cls._syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))

    @classmethod
    def _signal_pidfd(cls, fd: int, signum: int) -> None:
        cls._syscall(
            424, ctypes.c_int(fd), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0)
        )

    @staticmethod
    def _proc_bytes(pid: int, member: str, maximum_bytes: int = 16384) -> bytes:
        with Path(f"/proc/{pid}/{member}").open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValueError("maintenance process inspection exceeds its bound")
        return payload

    @classmethod
    def _state(cls, pid: int, *, require_owner: bool = True) -> tuple[int, int, str]:
        if require_owner and Path(f"/proc/{pid}").stat().st_uid != os.geteuid():
            raise ValueError("maintenance process has another owner")
        tail = cls._proc_bytes(pid, "stat").decode().rsplit(") ", 1)[1].split()
        return int(tail[19]), int(tail[1]), tail[0]

    def require_running(self, authority: MaintenanceProcessStopAuthority) -> None:
        # A controller cannot stop its own ancestry and still complete custody.
        ancestor = os.getpid()
        for _ in range(128):
            if ancestor == authority.main_pid:
                raise ValueError("maintenance target contains the calling process")
            if ancestor <= 1:
                break
            ancestor = self._state(ancestor, require_owner=False)[1]
        else:
            raise ValueError("maintenance ancestry exceeds its bound")
        start, _, state = self._state(authority.main_pid)
        if start != authority.main_start_ticks:
            raise ValueError("maintenance process PID was reused")
        if state in {"Z", "X"}:
            raise ValueError("maintenance process is not running")
        if (
            hashlib.sha256(self._proc_bytes(authority.main_pid, "cmdline", 65536)).hexdigest()
            != authority.command_line_sha256
        ):
            raise ValueError("maintenance process command identity changed")

    def _group_frozen(self, pid: int) -> bool:
        with os.scandir(f"/proc/{pid}/task") as threads:
            for count, thread in enumerate(threads, start=1):
                if count > 2048:
                    raise ValueError("maintenance thread census exceeds its bound")
                if self._state(int(thread.name))[2] not in {"T", "t"}:
                    return False
        return True

    def _children(self, pid: int, maximum_processes: int) -> tuple[int, ...]:
        children: set[int] = set()
        # Linux tracks children per thread; reading only the leader misses workers
        # launched by the scheduler's thread pool. The whole parent is frozen here.
        with os.scandir(f"/proc/{pid}/task") as threads:
            for count, thread in enumerate(threads, start=1):
                if count > 2048:
                    raise ValueError("maintenance thread census exceeds its bound")
                if not thread.name.isdecimal():
                    raise ValueError("maintenance thread identity is invalid")
                payload = self._proc_bytes(
                    pid, f"task/{thread.name}/children", maximum_processes * 24
                )
                children.update(int(value) for value in payload.split())
                if len(children) + 1 > maximum_processes:
                    raise ValueError("maintenance process census exceeds owner authority")
        return tuple(sorted(children))

    def stop(self, authority: MaintenanceProcessStopAuthority) -> None:
        self.require_running(authority)
        frozen: list[tuple[int, bool]] = []
        deadline = time.monotonic() + 10
        try:
            pending: list[tuple[int, int, int | None]] = [
                (authority.main_pid, authority.main_start_ticks, None)
            ]
            seen: set[int] = set()
            while pending:
                pid, expected_start, parent = pending.pop()
                if pid in seen:
                    raise ValueError("maintenance process tree repeats a process")
                seen.add(pid)
                if len(seen) > authority.maximum_processes:
                    raise ValueError("maintenance process census exceeds owner authority")
                fd = self._open_pidfd(pid)
                try:
                    start, observed_parent, state = self._state(pid)
                    if start != expected_start or (
                        parent is not None and observed_parent != parent
                    ):
                        raise ValueError("maintenance process tree identity changed")
                    if state in {"Z", "X"}:
                        raise ValueError("maintenance process exited during tree binding")
                    self._signal_pidfd(fd, signal.SIGSTOP)
                    frozen.append((fd, state not in {"T", "t"}))
                except BaseException:
                    os.close(fd)
                    raise
                while not self._group_frozen(pid):
                    if time.monotonic() >= deadline:
                        raise ValueError("maintenance process did not freeze")
                    time.sleep(0.005)
                if pid == authority.main_pid:
                    self.require_running(authority)
                for child in self._children(pid, authority.maximum_processes):
                    child_start, child_parent, _ = self._state(child)
                    if child_parent != pid:
                        raise ValueError("maintenance descendant was reparented")
                    pending.append((child, child_start, pid))
                if len(pending) + len(seen) > authority.maximum_processes:
                    raise ValueError("maintenance process census exceeds owner authority")
            # Every process is now frozen under a stable handle; no live parent
            # can create another child between census and termination.
            for fd, _ in reversed(frozen):
                self._signal_pidfd(fd, signal.SIGKILL)
            poller = select.poll()
            remaining = {fd for fd, _ in frozen}
            for fd in remaining:
                poller.register(fd, select.POLLIN)
            while remaining:
                if time.monotonic() >= deadline:
                    raise ValueError("maintenance process stop did not complete")
                for fd, _ in poller.poll(20):
                    remaining.discard(fd)
                    poller.unregister(fd)
        finally:
            # A failed pre-stop census must not leave any process suspended.
            for fd, resume in reversed(frozen):
                try:
                    if resume:
                        self._signal_pidfd(fd, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)


@dataclass(frozen=True, slots=True)
class MaintenanceStopObservation:
    stopped: bool
    ready: bool
    completed_tasks: int
    unstarted_tasks: int
    checkpoint_relative_path: str | None


@dataclass(frozen=True, slots=True)
class MaintenanceInterruptionObservation:
    stopped: bool
    completed_tasks: int
    interrupted_tasks: int
    unstarted_tasks: int
    checkpoint_relative_path: str | None


def _write_once(root: GuardedExternalRoot, relative_path: str, payload: bytes) -> None:
    path = root.resolve(relative_path, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = root.resolve(relative_path, for_write=True)
    if path.exists():
        if read_bounded_bytes(path, maximum_bytes=256 * 1024 * 1024) != payload:
            raise ValueError("maintenance record identity conflicts with retained bytes")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("maintenance record write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ReceiptBoundaryStopper:
    """The same exclusive store lock spans inspection, receipt proof and stop.

    Scheduler reservation precedes its SQLite lease and worker launch. Holding
    this lock therefore closes the check-to-launch race without changing the
    running source, frozen attempt ceiling, evidence or recovery index.
    """

    def __init__(
        self,
        plane: ExternalArtifactPlane,
        *,
        service: MaintenanceServiceControl | None = None,
        process: MaintenanceProcessControl | None = None,
    ) -> None:
        self.plane = plane
        self.store = ExternalExecutionResourceEnvelopeStore(plane.root)
        self.receipts = ExternalTaskReceiptStore(plane)
        self.service = service or LocalUserServiceControl()
        self.process = process or LocalProcessTreeControl()

    def interrupt(
        self,
        authority: ContinuationBoundInterruptionAuthority | InterruptionAuthority,
        *,
        confirmed: bool,
    ) -> MaintenanceInterruptionObservation:
        """Preserve and stop an issued run; never refund or retry its attempts.

        All original bytes and the original resource envelope remain in place.
        Receipt authentication and the independent file inventory follow the
        stop, so an active worker is not held waiting during a long audit.
        """
        bound = (
            authority.service_stop
            if isinstance(authority, ContinuationBoundInterruptionAuthority)
            else authority.stop
        )
        if isinstance(bound, MaintenanceProcessStopAuthority):
            self.process.require_running(bound)
        else:
            self.service.require_running(bound)
        envelope_id = f"run-resource-envelope.{bound.run_id}"
        for suffix in ("json", "lock"):
            if not self.plane.root.resolve(
                self.store._relative_path(envelope_id, suffix), for_write=False
            ).is_file():
                raise FileNotFoundError("maintenance requires an already initialized run")
        with self.store._locked(envelope_id) as path:
            envelope, _, _ = self.store._read(path, envelope_id)
            if envelope.execution_plan != bound.execution_plan:
                raise ValueError("maintenance authority names another execution plan")
            completed = tuple(
                spec.task_id
                for spec, cell in zip(envelope.spec.task_cells, envelope.cells, strict=True)
                if cell.first_valid_receipt is not None
            )
            unstarted = tuple(
                spec.task_id
                for spec, cell in zip(envelope.spec.task_cells, envelope.cells, strict=True)
                if not cell.attempt_ids
            )
            unfinished = tuple(
                (spec.task_id, cell.current_attempt_id, cell.state.value)
                for spec, cell in zip(envelope.spec.task_cells, envelope.cells, strict=True)
                if cell.attempt_ids and cell.first_valid_receipt is None
            )
            if len(unfinished) > authority.maximum_open_attempts:
                raise ValueError("maintenance open-attempt census exceeds owner authority")
            if not confirmed:
                return MaintenanceInterruptionObservation(
                    False, len(completed), len(unfinished), len(unstarted), None
                )
            prefix = f"runs/{bound.run_id}/maintenance/{bound.stop_id}"
            _write_once(
                self.plane.root,
                f"{prefix}/interruption-authority.json",
                authority.canonical_bytes(),
            )
            _write_once(
                self.plane.root, f"{prefix}/interrupted-envelope.json", envelope.canonical_bytes()
            )
            intent = {
                "schema": 'empirical-lawhood/infrastructure/maintenance-interruption-intent',
                "authority_sha256": authority.fingerprint(),
                "original_envelope_sha256": envelope.fingerprint(),
                "completed_task_ids": completed,
                "unfinished_attempts": unfinished,
                "unstarted_task_ids": unstarted,
                "native_retries_authorized_in_original_run": 0,
                "files_deleted_by_this_operation": 0,
            }
            _write_once(
                self.plane.root, f"{prefix}/interruption-intent.json", canonical_json_bytes(intent)
            )
            if isinstance(bound, MaintenanceProcessStopAuthority):
                self.process.stop(bound)
            else:
                self.service.stop(bound)
            observed, _, _ = self.store._read(path, envelope_id)
            if observed != envelope:
                raise ValueError("maintenance envelope changed while its reservation lock was held")
            result_path = f"{prefix}/interrupted.json"
            _write_once(
                self.plane.root,
                result_path,
                canonical_json_bytes(
                    {
                        **intent,
                        "schema": 'empirical-lawhood/infrastructure/maintenance-interruption-result',
                        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
                        "original_envelope_unchanged": True,
                        "completed_receipt_authentication": "REQUIRED_AFTER_STOP",
                        "scientific_adjudication_created": False,
                    }
                ),
            )
            return MaintenanceInterruptionObservation(
                True, len(completed), len(unfinished), len(unstarted), result_path
            )

    def try_stop(
        self, authority: MaintenanceStopAuthority, *, confirmed: bool
    ) -> MaintenanceStopObservation:
        self.service.require_running(authority)
        envelope_id = f"run-resource-envelope.{authority.run_id}"
        for suffix in ("json", "lock"):
            existing = self.plane.root.resolve(
                self.store._relative_path(envelope_id, suffix), for_write=False
            )
            if not existing.is_file():
                raise FileNotFoundError("maintenance requires an already initialized run")
        with self.store._locked(envelope_id) as path:
            envelope, _, _ = self.store._read(path, envelope_id)
            if envelope.execution_plan != authority.execution_plan:
                raise ValueError("maintenance authority names another execution plan")
            boundary = maintenance_receipt_boundary(authority.run_id, envelope)
            if boundary is None:
                completed = sum(cell.first_valid_receipt is not None for cell in envelope.cells)
                unstarted = sum(not cell.attempt_ids for cell in envelope.cells)
                return MaintenanceStopObservation(False, False, completed, unstarted, None)
            if not confirmed:
                return MaintenanceStopObservation(
                    False,
                    True,
                    len(boundary.completed_receipts),
                    len(boundary.unstarted_task_ids),
                    None,
                )
            self._authenticate_receipts(boundary, envelope)
            self.service.require_running(authority)
            prefix = f"runs/{authority.run_id}/maintenance/{authority.stop_id}"
            _write_once(self.plane.root, f"{prefix}/authority.json", authority.canonical_bytes())
            _write_once(self.plane.root, f"{prefix}/boundary.json", boundary.canonical_bytes())
            _write_once(self.plane.root, f"{prefix}/envelope.json", envelope.canonical_bytes())
            self.service.stop(authority)
            observed, _, _ = self.store._read(path, envelope_id)
            if observed != envelope:
                raise ValueError("maintenance envelope changed while its reservation lock was held")
            result_path = f"{prefix}/stopped.json"
            _write_once(
                self.plane.root,
                result_path,
                canonical_json_bytes(
                    {
                        "schema": 'empirical-lawhood/infrastructure/maintenance-stop-result',
                        "authority_sha256": authority.fingerprint(),
                        "boundary_sha256": boundary.fingerprint(),
                        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
                        "native_retries_authorized": 0,
                        "all_launched_tasks_have_durable_receipts": True,
                        "original_envelope_unchanged": True,
                    }
                ),
            )
            return MaintenanceStopObservation(
                True,
                True,
                len(boundary.completed_receipts),
                len(boundary.unstarted_task_ids),
                result_path,
            )

    def _authenticate_receipts(
        self, boundary: MaintenanceReceiptBoundary, envelope: RunExecutionResourceEnvelope
    ) -> None:
        expected = {value.object_id: value for value in boundary.completed_receipts}
        for spec, cell in zip(envelope.spec.task_cells, envelope.cells, strict=True):
            if cell.first_valid_receipt is None:
                continue
            assert cell.current_attempt_id is not None
            receipt = self.receipts.read(boundary.run_id, spec.task_id, cell.current_attempt_id)
            if (
                receipt is None
                or receipt.operational_status is not OperationalStatus.SUCCEEDED
                or ObjectIdentity.from_record(receipt.receipt_id, receipt)
                != expected[receipt.receipt_id]
            ):
                raise ValueError("maintenance boundary lacks its exact durable task receipt")
