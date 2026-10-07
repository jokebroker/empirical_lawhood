"""Bounded public preflight for the retained full-batch reactor source port."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import os

from empirical_lawhood.adapters._bounded_files import AdapterFileBoundError, read_bounded_contained
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .batch_design import BATCH_SOURCE_SHA256, ReactorBatchSource, validate_nominal_spec
from .contracts import PARAMS_SHA256, PLANT_SHA256, PUBLIC_SCENARIOS_SHA256
from .forecast import REFERENCE_SHA256

SOURCE_MEMBERS = (
    ("tests/plant.py", 32768, PLANT_SHA256),
    ("environment/spec/plant_params.json", 4096, PARAMS_SHA256),
    ("environment/spec/scenarios_public.json", 8192, PUBLIC_SCENARIOS_SHA256),
    ("solution/controller.py", 16384, REFERENCE_SHA256),
)


_MAX_SOURCE_FILES = 128
_MAX_SOURCE_MEMBER_BYTES = 4 * 1024**2
_MAX_SOURCE_BYTES = 16 * 1024**2
# Count entries and reopened relative-directory components, bounding both breadth
# and deep traversal work independently of the number of accepted source files.
_MAX_SOURCE_TRAVERSAL_WORK = 8192


class _ReactorSourceSnapshot:
    """Operation-local captured bytes; this is not an atomic filesystem snapshot."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("source root must be an absolute, real local directory")
        self.root = root
        self.payloads: dict[str, bytes] = {}
        self.total_bytes = 0

    def read_member(self, relative: str, maximum_bytes: int) -> bytes:
        raw = self.payloads.get(relative)
        if raw is not None:
            if len(raw) > maximum_bytes:
                raise AdapterFileBoundError("adapter input exceeds its byte limit")
            return raw
        remaining = _MAX_SOURCE_BYTES - self.total_bytes
        raw = read_bounded_contained(
            self.root, self.root / relative, maximum_bytes=min(maximum_bytes, remaining)
        )
        self.total_bytes += len(raw)
        self.payloads[relative] = raw
        return raw

    def census(self) -> tuple[tuple[str, str], ...]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.root.anchor, flags)
        try:
            for part in self.root.parts[1:]:
                following = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = following
            before = os.fstat(descriptor)
            pending = [Path()]
            members: list[str] = []
            work = 0
            while pending:
                relative_directory = pending.pop()
                directory = os.dup(descriptor)
                try:
                    for part in relative_directory.parts:
                        work += 1
                        if work > _MAX_SOURCE_TRAVERSAL_WORK:
                            raise ValueError("REACTOR_SOURCE_TRAVERSAL_TOO_LARGE")
                        following = os.open(part, flags, dir_fd=directory)
                        os.close(directory)
                        directory = following
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            work += 1
                            if work > _MAX_SOURCE_TRAVERSAL_WORK:
                                raise ValueError("REACTOR_SOURCE_TRAVERSAL_TOO_LARGE")
                            relative = relative_directory / entry.name
                            if entry.is_symlink():
                                raise ValueError("REACTOR_SOURCE_CENSUS_ESCAPES_ROOT")
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(relative)
                            else:
                                # Special files must reach the contained reader's
                                # descriptor type check, never a blocking open.
                                if len(members) == _MAX_SOURCE_FILES:
                                    raise ValueError("REACTOR_SOURCE_CENSUS_TOO_LARGE")
                                members.append(relative.as_posix())
                finally:
                    os.close(directory)
            census: list[tuple[str, str]] = []
            for relative in sorted(members):
                try:
                    raw = self.read_member(relative, _MAX_SOURCE_MEMBER_BYTES)
                except AdapterFileBoundError as error:
                    raise ValueError(f"REACTOR_SOURCE_CENSUS_INVALID: {error}") from error
                census.append((relative, sha256(raw).hexdigest()))
            after = self.root.stat(follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("REACTOR_SOURCE_CENSUS_ROOT_CHANGED")
            return tuple(census)
        except OSError as error:
            raise ValueError("REACTOR_SOURCE_CENSUS_ESCAPES_ROOT") from error
        finally:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ReactorBatchInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-input'
    config_id: str
    source_sha256: str = BATCH_SOURCE_SHA256
    horizon_s: Decimal = Decimal(28800)
    sample_dt_s: Decimal = Decimal(10)
    plant_timesteps_s: tuple[Decimal, Decimal] = (Decimal(1), Decimal("0.5"))
    independent_unit: str = "assigned-scenario-seed"
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if not self.config_id.startswith("empirical-lawhood-reactor-"):
            raise ValueError("reactor input requires a target-owned config identity")
        for name, field in self.__dataclass_fields__.items():
            if (
                name not in ("SCHEMA", "config_id")
                and getattr(self, name) != field.default
            ):
                raise ValueError(f"reactor batch input changes the retained {name}")


def load_batch_source(
    source_root: Path | None, *, _snapshot: _ReactorSourceSnapshot | None = None
) -> ReactorBatchSource:
    """Authenticate and decode the held source without executing its Python."""
    if source_root is None:
        raise ValueError(
            "SOURCE_ROOT_REQUIRED: provide the upstream tree containing "
            "tests/plant.py, environment/spec/plant_params.json, "
            "environment/spec/scenarios_public.json and solution/controller.py"
        )
    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
    ):
        raise ValueError("source root must be an absolute, real local directory")
    snapshot = _snapshot or _ReactorSourceSnapshot(source_root)
    if snapshot.root != source_root:
        raise ValueError("REACTOR_SOURCE_SNAPSHOT_ROOT_MISMATCH")
    payloads: list[str] = []
    for member, maximum_bytes, digest in SOURCE_MEMBERS:
        try:
            raw = snapshot.read_member(member, maximum_bytes)
        except AdapterFileBoundError as error:
            reason = "SOURCE_MEMBER_TOO_LARGE" if "byte limit" in str(error) else "SOURCE_MEMBER_REQUIRED"
            raise ValueError(f"{reason}: {member}") from error
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"SOURCE_MEMBER_PIN_MISMATCH: {member}")
        payloads.append(raw.decode("utf-8"))
    source = ReactorBatchSource(*payloads)
    validate_nominal_spec(source)
    return source


def check_batch_input(
    config: ReactorBatchInput, source_root: Path | None
) -> dict[str, object]:
    """Select the retained batch provider without issue or native contact."""

    from .batch_design import ReactorBatchConfig, assigned_scenarios
    from .reactor_binding import inspect_reactor_binding

    source = load_batch_source(source_root)
    if source.fingerprint() != config.source_sha256:
        raise ValueError("SOURCE_BUNDLE_PIN_MISMATCH")
    native = ReactorBatchConfig(config.config_id, assigned_scenarios())
    binding = inspect_reactor_binding("batch", native, source)
    return {
        "config_id": config.config_id,
        "native_config_sha256": native.fingerprint(),
        "source_sha256": source.fingerprint(),
        "accepted_source_members": tuple(member for member, _, _ in SOURCE_MEMBERS),
        "horizon_s": str(config.horizon_s),
        "sample_dt_s": str(config.sample_dt_s),
        "plant_timesteps_s": tuple(str(v) for v in config.plant_timesteps_s),
        "independent_unit": config.independent_unit,
        "evidence_role": config.evidence_role,
        **binding,
        "native_tasks_executed": 0,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = [
    "SOURCE_MEMBERS",
    'ReactorBatchInput',
    "check_batch_input",
    "load_batch_source",
]
