"""Bounded operational exports, separate from scientific products and custody.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any
from uuid import uuid4

from .bounded_io import bounded_file_sha256, read_bounded_bytes
from .bounded_process import BoundedProcessError, run_bounded_command
from .atomic_files import rename_noreplace


MAX_REPORT_BYTES = 1024 * 1024
MAX_INPUT_COPY_BYTES = 64 * 1024
MAX_RECORD_BYTES = 64 * 1024
MAX_DIAGNOSTICS_BYTES = 32 * 1024
MAX_EVENTS = 32


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: object, maximum: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for text in json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False).iterencode(value):
        chunk = text.encode("ascii")
        size += len(chunk)
        if size + 1 > maximum:
            raise ValueError("operational record exceeds its byte limit")
        chunks.append(chunk)
    return b"".join(chunks) + b"\n"


def runtime_identity() -> dict[str, object]:
    """Observe this executing package, never infer source from the user's cwd."""
    distributions: dict[str, str | None] = {}
    for name in ("empirical-lawhood", "numpy", "scipy", "brian2", "torax", "gymtorax", "pybamm", "cantera", "fipy", "grid2op", "lightsim2grid"):
        try:
            distributions[name] = version(name)
        except PackageNotFoundError:
            distributions[name] = None
    module = Path(__file__).absolute()
    source: dict[str, object] = {"executing_module": str(module), "checkout": None, "commit": None, "tracked_changes": None}
    candidate = module.parents[3]
    if (candidate / "src/empirical_lawhood/infrastructure/development_attempts.py") == module and (candidate / ".git").exists():
        try:
            observation = run_bounded_command(["git", "-C", str(candidate), "rev-parse", "HEAD"], timeout_seconds=2, maximum_stdout_bytes=41, maximum_stderr_bytes=4096)
            commit = observation.stdout
            if observation.returncode == 0 and len(commit) == 41 and all(c in b"0123456789abcdef" for c in commit[:40]):
                source.update(checkout=str(candidate), commit=commit.decode().strip())
                changes = [run_bounded_command(["git", "-C", str(candidate), "diff", "--quiet", *scope], timeout_seconds=2, maximum_stdout_bytes=0, maximum_stderr_bytes=4096).returncode for scope in ([], ["--cached"])]
                source["tracked_changes"] = any(code == 1 for code in changes) if all(code in (0, 1) for code in changes) else None
        except (OSError, BoundedProcessError):
            pass
    return {"python_version": sys.version.split()[0], "python_executable": sys.executable,
            "installed_distribution_versions": distributions, "source": source,
            "qualification": "NOT_ESTABLISHED", "source_bound_clean_execution": False}


def safe_failure(error: BaseException, *, stage: str) -> dict[str, object]:
    """Keep classification, not arbitrary exception text, locals or sealed data."""
    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    contract_code = None
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(type(current).__name__[:100])
        if type(current).__module__ == "empirical_lawhood.api.configuration" and type(current).__name__ == "ConfigurationError":
            contract_code = getattr(current, "code", None)
        current = current.__cause__ or current.__context__
    effective = chain[-1] if chain else "Exception"
    return {"exception_type": chain[0], "cause_types": chain[1:], "contract_code": contract_code, "stage": stage,
            "diagnostic": f"{effective} during {stage}; inspect the terminal diagnostic and declared input contract.",
            "next_action": "Preserve this attempt; correct the input or storage and select a new development attempt. No automatic rerun."}


class DevelopmentAttempt:
    """Own one fresh directory and bounded atomic operational files."""

    def __init__(self, destination: Path, operation: str, *, native: bool = False) -> None:
        self.path = Path(os.path.abspath(destination))
        self._fd = -1
        self._installed: dict[str, str] = {}
        self._events: list[bytes] = []
        self._report: bytes | None = None
        self._report_error: Exception | None = None
        self.failure_exception: BaseException | None = None
        self.closed = False
        self.record: dict[str, Any] = {
            "schema": "empirical-lawhood/operational/development-attempt", "version": "1.0.0",
            "attempt_id": str(uuid4()), "operation": operation, "started_at_utc": _utc(),
            "ended_at_utc": None, "state": "RUNNING", "exit_code": None,
            "stage": "argument_validation", "last_completed_stage": "invocation_created",
            "native_contact": "unknown" if native else "none", "operation_completed": False,
            "scientific_adjudication": "NOT_PERFORMED", "inputs": [], "products": [],
            "events_dropped": 0, "runtime": runtime_identity(),
            "handler_entered": False,
        }
        parent_fd = os.open(self.path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                for part in self.path.parent.parts[1:]:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    os.close(parent_fd)
                    parent_fd = next_fd
                os.mkdir(self.path.name, mode=0o700, dir_fd=parent_fd)
                self._fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            self._write("invocation.json", _json_bytes(self.record, MAX_RECORD_BYTES), MAX_RECORD_BYTES)
            self.event("invocation_created")
        except BaseException:
            self.close()
            raise

    def _check_directory(self) -> None:
        actual = os.stat(self.path, follow_symlinks=False)
        held = os.fstat(self._fd)
        if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (held.st_dev, held.st_ino) or held.st_nlink == 0:
            raise OSError("attempt directory identity changed")

    def _write(self, name: str, payload: bytes, maximum: int) -> None:
        if len(payload) > maximum:
            raise ValueError("attempt product exceeds its byte limit")
        self._check_directory()
        expected = self._installed.get(name)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
        except FileNotFoundError:
            if expected is not None:
                raise OSError("attempt product disappeared") from None
        else:
            try:
                observed = os.fstat(descriptor)
                if expected is None or not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
                    raise FileExistsError("protected attempt product already exists")
                data = bytearray()
                while len(data) <= maximum:
                    chunk = os.read(descriptor, min(65536, maximum - len(data) + 1))
                    if not chunk:
                        break
                    data.extend(chunk)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise FileExistsError("attempt product was changed by another writer")
            finally:
                os.close(descriptor)
        temporary = f".write-{uuid4().hex}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._fd)
        try:
            try:
                offset = 0
                while offset < len(payload):
                    count = os.write(descriptor, payload[offset:offset + 65536])
                    if count <= 0:
                        raise OSError("attempt write made no progress")
                    offset += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._check_directory()
            if expected is None:
                rename_noreplace(temporary, name, source_dir_fd=self._fd, destination_dir_fd=self._fd)
            else:
                os.replace(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
            self._installed[name] = hashlib.sha256(payload).hexdigest()
            os.fsync(self._fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self._fd)
            except FileNotFoundError:
                pass

    def event(self, stage: str) -> None:
        self.record["stage"] = stage[:100]
        if len(self._events) >= MAX_EVENTS:
            self.record["events_dropped"] += 1
            return
        item = _json_bytes({"at_utc": _utc(), "stage": stage[:100]}, 1024)
        self._events.append(item)
        self._write("diagnostics.jsonl", b"".join(self._events), MAX_DIAGNOSTICS_BYTES)

    def capture_input(self, role: str, path: Path, *, maximum_bytes: int, copy: bool, expected_sha256: str | None = None) -> Path:
        payload = read_bounded_bytes(path, maximum_bytes=maximum_bytes) if copy else None
        count, digest = (len(payload), hashlib.sha256(payload).hexdigest()) if payload is not None else bounded_file_sha256(path, maximum_bytes=maximum_bytes)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("development input changed after no-contact validation")
        row = {"role": role, "locator": str(path.absolute()), "sha256": digest,
               "bytes": count, "accepted": None, "used_snapshot": copy,
               "identity_scope": "consumed_snapshot" if copy else "observed_before_operation"}
        existing = next((item for item in self.record["inputs"] if item["role"] == role and item["locator"] == row["locator"]), None)
        if existing is None:
            self.record["inputs"].append(row)
        else:
            if existing["sha256"] != digest:
                raise ValueError("development input changed after its recorded identity")
            existing.update(row)
            row = existing
        if copy:
            assert payload is not None
            if len(payload) > MAX_INPUT_COPY_BYTES:
                raise ValueError("input copy exceeds the operational byte limit")
            suffix = path.suffix.lower() if path.suffix.lower() in {".yaml", ".yml"} else ".json"
            snapshots = [item["retained_file"] for item in self.record["inputs"] if "retained_file" in item]
            name = row.get("retained_file") or ("input" if not snapshots else f"input-{len(snapshots):03d}") + suffix
            self._write(name, payload, MAX_INPUT_COPY_BYTES)
            row["retained_file"] = name
        self._write("invocation.json", _json_bytes(self.record, MAX_RECORD_BYTES), MAX_RECORD_BYTES)
        return self.path / row["retained_file"] if copy else path

    def operation_completed(self) -> None:
        self.record["operation_completed"] = True
        self.record["last_completed_stage"] = "operation_completed"
        if self.record["native_contact"] == "unknown":
            self.record["native_contact"] = "operation_reported_complete"
        for row in self.record["inputs"]:
            if row["used_snapshot"]:
                row["accepted"] = True

    def report(self, rendered: str) -> None:
        self.operation_completed()
        if len(rendered) + 1 > MAX_REPORT_BYTES:
            self._report_error = ValueError("returned report exceeds the operational export byte limit")
        else:
            self._report = rendered.encode("ascii") + b"\n"

    def finish(self, exit_code: int, error: BaseException | None = None) -> None:
        if self.closed:
            return
        self.record.update(ended_at_utc=_utc(), exit_code=exit_code,
                           state="SUCCEEDED" if exit_code == 0 else "FAILED")
        if error is None and self._report_error is not None:
            error = self._report_error
            self.record.update(state="EXPORT_FAILED", exit_code=1, stage="result_export")
        if exit_code and error is None:
            error = RuntimeError("CLI returned a nonzero exit without an exception")
        failure = None if error is None else safe_failure(error, stage=self.record["stage"])
        try:
            if self._report is not None:
                self.event("result_export")
                self._write("report.json", self._report, MAX_REPORT_BYTES)
                self.record["products"].append({"path": "report.json", "role": "returned_route_report", "complete": True,
                                               "bytes": len(self._report), "sha256": hashlib.sha256(self._report).hexdigest()})
            if failure is not None:
                self.record["failure"] = failure
                self._write("failure.json", _json_bytes(failure, 16 * 1024), 16 * 1024)
            summary = (f"# Development attempt\n\nOperation: `{self.record['operation']}`\n\n"
                       f"State: **{self.record['state']}**; exit: {self.record['exit_code']}.\n\n"
                       f"Stage: `{self.record['stage']}`; operation completed: {self.record['operation_completed']}.\n\n"
                       "This operational record supplies no scientific adjudication, qualification or issue authority.\n\n"
                       "See [invocation](invocation.json) and [bounded diagnostics](diagnostics.jsonl).\n\n")
            if self._report is not None:
                summary += "The command's unchanged returned report is in [report.json](report.json).\n\n"
            if failure is not None:
                summary += f"Failure classification: `{failure['exception_type']}`. See [failure](failure.json).\n\n{failure['next_action']}\n"
            self._write("report.md", summary.encode(), 16 * 1024)
            self.event("terminal")
            self._write("invocation.json", _json_bytes(self.record, MAX_RECORD_BYTES), MAX_RECORD_BYTES)
        except BaseException as secondary:
            self.record.update(state="EXPORT_FAILED", exit_code=130 if isinstance(secondary, KeyboardInterrupt) else exit_code if exit_code else 1)
            self.record["export_failure"] = safe_failure(secondary, stage="result_export")
            if failure is not None:
                self.record["failure"] = failure
            for name, value in (("failure.json", self.record), ("invocation.json", self.record)):
                try:
                    self._write(name, _json_bytes(value, MAX_RECORD_BYTES), MAX_RECORD_BYTES)
                except BaseException:
                    pass
            raise

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
