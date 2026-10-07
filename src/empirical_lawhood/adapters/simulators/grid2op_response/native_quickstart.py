"""Target-owned, development-only Grid2Op held-input quick start."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import Parser
from hashlib import sha256
from importlib import metadata, resources
from pathlib import Path, PurePosixPath
from typing import ClassVar
from zipfile import ZipFile

from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateTargetPhase
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import Grid2OpActionBranchRequest, Grid2OpActionKind, Grid2OpEpisodeRequest, Grid2OpNativeAction
from .runtime import Grid2OpOfflineEnvironmentAdapter, execute_grid2op_episode
from .source import build_held_grid2op_source_binding


def _relative_source_member(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
    ):
        raise ValueError("Grid2Op source member must be a bounded relative POSIX path")


@dataclass(frozen=True, slots=True)
class Grid2OpDevelopmentInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-development-input'

    config_id: str
    binding_id: str
    dataset_relative_path: str
    grid2op_wheel_relative_path: str
    backend_wheel_relative_path: str
    native_chronic_id: str
    environment_seed: int
    receiver_horizon_steps: tuple[int, ...]
    fresh_target_identity_disjoint: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_stable_id(self.binding_id)
        if not self.config_id.startswith(
            "empirical-lawhood-"
        ) or not self.binding_id.startswith("source.empirical-lawhood-"):
            raise ValueError("Grid2Op quick start needs new target-owned identities")
        for value in (
            self.dataset_relative_path,
            self.grid2op_wheel_relative_path,
            self.backend_wheel_relative_path,
        ):
            _relative_source_member(value)
        validate_stable_id(self.native_chronic_id)
        if self.environment_seed < 0:
            raise ValueError("Grid2Op native seed must be nonnegative")
        if self.receiver_horizon_steps not in ((1, 3), (1, 3, 6)):
            raise ValueError("Grid2Op native horizons must follow the source chart")
        if type(self.fresh_target_identity_disjoint) is not bool:
            raise TypeError("Grid2Op disjointness declaration must be boolean")


def _observation_operator_sha256() -> str:
    """Bind the two installed target modules that define native observation."""

    package = resources.files(__package__)
    digest = sha256()
    for name in ("analysis.py", "runtime.py"):
        payload = package.joinpath(name).read_bytes()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _held_path(root: Path, relative: str, *, directory: bool) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Grid2Op source member escapes the held source root")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Grid2Op held source member is absent: {relative}")
    return path


def _require_wheel_identity(path: Path, *, distribution: str, version: str) -> None:
    with ZipFile(path) as archive:
        members = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        if len(members) != 1 or archive.getinfo(members[0]).file_size > 256 * 1024:
            raise ValueError(
                f"Grid2Op held {distribution} wheel metadata is absent or unbounded"
            )
        fields = Parser().parsestr(archive.read(members[0]).decode("utf-8"))
    observed_name = fields.get("Name", "").lower().replace("-", "_")
    if (
        observed_name != distribution.lower().replace("-", "_")
        or fields.get("Version") != version
    ):
        raise ValueError(
            f"Grid2Op held {distribution} wheel identity differs from the source chart"
        )


def run_native_development_check(
    config: Grid2OpDevelopmentInput, *, source_root: Path
) -> dict[str, object]:
    """Run hold and one line-disconnect branch on one operator-held chronic."""

    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("Grid2Op held source root is absent or relative")
    dataset = _held_path(source_root, config.dataset_relative_path, directory=True)
    grid2op_wheel = _held_path(
        source_root, config.grid2op_wheel_relative_path, directory=False
    )
    backend_wheel = _held_path(
        source_root, config.backend_wheel_relative_path, directory=False
    )
    if not (dataset / "chronics" / config.native_chronic_id).is_dir():
        raise ValueError("Grid2Op selected native chronic is absent")
    _require_wheel_identity(grid2op_wheel, distribution="grid2op", version="1.12.5")
    _require_wheel_identity(
        backend_wheel, distribution="lightsim2grid", version="0.13.1"
    )
    for distribution, expected in (("grid2op", "1.12.5"), ("lightsim2grid", "0.13.1")):
        if metadata.version(distribution) != expected:
            raise ValueError(
                f"installed {distribution} version differs from the source chart"
            )
    source = build_held_grid2op_source_binding(
        binding_id=config.binding_id,
        grid2op_version="1.12.5",
        grid2op_wheel=grid2op_wheel,
        backend_version="0.13.1",
        backend_wheel=backend_wheel,
        dataset_path=dataset,
        native_chronic_ids=(config.native_chronic_id,),
        observation_operator_sha256=_observation_operator_sha256(),
        fresh_target_identity_disjoint=config.fresh_target_identity_disjoint,
    )
    chronic = source.chronic_bindings[0]
    branches = tuple(
        sorted(
            (
                Grid2OpActionBranchRequest(
                    branch_id=f"branch.empirical-lawhood-{name}",
                    action=Grid2OpNativeAction(
                        action_id=name,
                        kind=kind,
                        target_native_ids=targets,
                        integer_values=values,
                        decimal_values=(),
                        canonical_description_bits=64,
                    ),
                    receiver_horizon_steps=config.receiver_horizon_steps,
                )
                for name, kind, targets, values in (
                    ("hold", Grid2OpActionKind.HOLD, (), ()),
                    (
                        "disconnect-line-000",
                        Grid2OpActionKind.SET_LINE_STATUS,
                        ("line-000",),
                        (-1,),
                    ),
                )
            ),
            key=lambda value: value.branch_id,
        )
    )
    request = Grid2OpEpisodeRequest(
        request_id=f"request.empirical-lawhood-grid2op-development-{config.native_chronic_id}",
        unit_id=chronic.chronic_id,
        phase=IndependentSubstrateTargetPhase.DEVELOPMENT,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        chronic=ObjectIdentity.from_record(chronic.chronic_id, chronic),
        environment_seed=config.environment_seed,
        initial_step=0,
        branches=branches,
        receiver_gauge_ids=("thermal", "topology"),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    port = Grid2OpOfflineEnvironmentAdapter(source=source, dataset_path=dataset)
    try:
        result = execute_grid2op_episode(port=port, source=source, request=request)
    finally:
        port.close()
    traces = tuple(
        {
            "branch_id": trace.branch.object_id,
            "reset_state_sha256": trace.reset_state_sha256,
            "steps": tuple(
                {
                    "horizon_step": step.horizon_step,
                    "timestamp_utc": step.timestamp_utc,
                    "requested_action_code": step.requested_action_code,
                    "accepted_action_code": step.accepted_action_code,
                    "applied_action_code": step.applied_action_code,
                    "realized_action_code": step.realized_action_code,
                    "disconnected_line_count": step.disconnected_line_count,
                    "maximum_rho": str(step.maximum_rho)
                    if step.maximum_rho is not None
                    else None,
                    "connected_component_count": step.connected_component_count,
                    "observation_valid": step.observation_valid,
                    "reason_codes": step.reason_codes,
                }
                for step in trace.steps
            ),
        }
        for trace in result.traces
    )
    return {
        "config_id": config.config_id,
        "source_binding_sha256": source.fingerprint(),
        "source_identity_disjoint": source.fresh_target_identity_disjoint,
        "development_only": True,
        "independent_units": result.scientific_unit_count,
        "nested_action_branches": len(result.traces),
        "native_observation_complete": all(
            step.observation_valid for trace in result.traces for step in trace.steps
        ),
        "traces": traces,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['Grid2OpDevelopmentInput', "run_native_development_check"]
