"""Bounded held-source inventory for the fresh independent substrate grounding-only Grid2Op target."""

from __future__ import annotations

import bz2
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .contracts import Grid2OpChronicBinding, Grid2OpSourceBinding


_EXOGENOUS_FILENAMES = (
    "hazards.csv.bz2",
    "load_p.csv.bz2",
    "load_q.csv.bz2",
    "maintenance.csv.bz2",
    "prod_p.csv.bz2",
    "prod_v.csv.bz2",
)


class Grid2OpSourceQualificationDisposition(StrEnum):
    PASS = "PASS"
    SOURCE_IDENTITY_MISMATCH = "SOURCE_IDENTITY_MISMATCH"
    ACTION_SURFACE_INCOMPLETE = "ACTION_SURFACE_INCOMPLETE"
    RESET_OR_REALIZATION_INCOMPLETE = "RESET_OR_REALIZATION_INCOMPLETE"
    OPERATIONAL_UNEVALUABLE = "OPERATIONAL_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class Grid2OpSourceQualification(CanonicalRecord):
    """Excluded source readiness follow-up qualification; never a target evidence unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-source-qualification'

    qualification_id: str
    target_source: ObjectIdentity
    qualification_source: ObjectIdentity
    qualification_native_chronic_id: str
    supported_action_kind_ids: tuple[str, ...]
    target_action_kind_ids: tuple[str, ...]
    hold_observed: bool
    exact_reset_observed: bool
    requested_accepted_applied_realized_distinct: bool
    realized_action_observable: bool
    backend_step_count: int
    backend_error_count: int
    excluded_from_target_evidence: bool
    target_evidence_unit_count: int
    disposition: Grid2OpSourceQualificationDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        if (
            self.target_source.object_schema != Grid2OpSourceBinding.SCHEMA
            or self.qualification_source.object_schema != Grid2OpSourceBinding.SCHEMA
        ):
            raise ValueError("Grid2Op qualification source identity differs")
        require_sorted_unique_strings(
            self.supported_action_kind_ids,
            field_name="supported_action_kind_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.target_action_kind_ids,
            field_name="target_action_kind_ids",
            allow_empty=False,
        )
        if not set(self.target_action_kind_ids) <= set(self.supported_action_kind_ids):
            raise ValueError("Grid2Op target action chart exceeds the qualified source")
        if "hold" not in self.target_action_kind_ids:
            raise ValueError("Grid2Op target action chart lacks mandatory hold")
        if self.backend_step_count <= 0 or self.backend_error_count < 0:
            raise ValueError("Grid2Op qualification step accounting is invalid")
        if (
            not self.excluded_from_target_evidence
            or self.target_evidence_unit_count
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("Grid2Op source qualification crossed target evidence")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        passed = all(
            (
                self.hold_observed,
                self.exact_reset_observed,
                self.requested_accepted_applied_realized_distinct,
                self.realized_action_observable,
                self.backend_error_count == 0,
                not self.reason_codes,
            )
        )
        if passed != (self.disposition is Grid2OpSourceQualificationDisposition.PASS):
            raise ValueError("Grid2Op qualification disposition is not fact-derived")
        if self.disposition is not Grid2OpSourceQualificationDisposition.PASS and (
            not self.reason_codes or self.disposition.value not in self.reason_codes
        ):
            raise ValueError("stopped Grid2Op qualification lacks its controlling reason")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, *, selected_names: set[str] | None = None) -> str:
    digest = sha256()
    members = tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_file() and (selected_names is None or path.name in selected_names)
        )
    )
    if not members:
        raise ValueError("Grid2Op chronic has no bounded source members")
    for path in members:
        relative = path.name.encode("utf-8")
        content = _sha256_file(path).encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _timestep_seconds(path: Path) -> int:
    value = path.read_text(encoding="ascii").strip()
    parsed = datetime.strptime(value, "%H:%M").replace(tzinfo=UTC)
    seconds = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    if seconds <= 0:
        raise ValueError("Grid2Op chronic timestep is not positive")
    return seconds


def _maximum_steps(path: Path) -> int:
    with bz2.open(path, "rt", encoding="utf-8", errors="strict", newline="") as source:
        rows = sum(1 for _line in source)
    # One header and one reset observation precede the maximum action-step count.
    maximum = rows - 2
    if maximum <= 0:
        raise ValueError("Grid2Op chronic has no executable steps")
    return maximum


def _chronic_binding(root: Path, native_id: str) -> Grid2OpChronicBinding:
    if root.name != native_id or not root.is_dir():
        raise ValueError("Grid2Op chronic locator differs from its native identity")
    timestep = _timestep_seconds(root / "time_interval.info")
    start = datetime.strptime(
        (root / "start_datetime.info").read_text(encoding="ascii").strip(),
        "%Y-%m-%d %H:%M",
    ).replace(tzinfo=UTC)
    initial = start + timedelta(seconds=timestep)
    exogenous_names = {value for value in _EXOGENOUS_FILENAMES if (root / value).is_file()}
    required = {"load_p.csv.bz2", "load_q.csv.bz2", "prod_p.csv.bz2", "prod_v.csv.bz2"}
    if not required <= exogenous_names:
        raise ValueError("Grid2Op chronic lacks its exact exogenous state files")
    forecast_names = {
        "load_p_forecasted.csv.bz2",
        "load_q_forecasted.csv.bz2",
        "prod_p_forecasted.csv.bz2",
        "prod_v_forecasted.csv.bz2",
    }
    return Grid2OpChronicBinding(
        chronic_id=f"chronic.grid2op.{native_id}",
        native_chronic_id=native_id,
        content_sha256=_tree_sha256(root),
        initial_timestamp_utc=initial.strftime("%Y-%m-%dT%H:%M:%SZ"),
        timestep_seconds=timestep,
        maximum_steps=_maximum_steps(root / "load_p.csv.bz2"),
        exogenous_event_sha256=_tree_sha256(root, selected_names=exogenous_names),
        forecasts_available=all((root / value).is_file() for value in forecast_names),
    )


def build_held_grid2op_source_binding(
    *,
    binding_id: str,
    grid2op_version: str,
    grid2op_wheel: Path,
    backend_version: str,
    backend_wheel: Path,
    dataset_path: Path,
    native_chronic_ids: tuple[str, ...],
    observation_operator_sha256: str,
    fresh_target_identity_disjoint: bool,
) -> Grid2OpSourceBinding:
    """Inventory exact held bytes without importing or executing Grid2Op."""

    validate_sha256(
        observation_operator_sha256,
        field_name="observation_operator_sha256",
    )
    require_sorted_unique_strings(
        native_chronic_ids,
        field_name="native_chronic_ids",
        allow_empty=False,
    )
    for path in (grid2op_wheel, backend_wheel, dataset_path):
        if not path.is_absolute() or not path.exists():
            raise ValueError("Grid2Op held source locator is absent or relative")
    chronics_root = dataset_path / "chronics"
    chronics = tuple(
        _chronic_binding(chronics_root / native_id, native_id) for native_id in native_chronic_ids
    )
    return Grid2OpSourceBinding(
        binding_id=binding_id,
        grid2op_version=grid2op_version,
        package_source_sha256=_sha256_file(grid2op_wheel),
        environment_name=dataset_path.name,
        backend_class_id="lightsim2grid.LightSimBackend",
        backend_version=backend_version,
        backend_source_sha256=_sha256_file(backend_wheel),
        grid_sha256=_sha256_file(dataset_path / "grid.json"),
        rules_class_id="grid2op.Rules.DefaultRules",
        parameters_sha256=_sha256_file(dataset_path / "config.py"),
        action_class_id="grid2op.Action.TopologyAndDispatchAction",
        observation_class_id="grid2op.Observation.CompleteObservation",
        observation_operator_sha256=observation_operator_sha256,
        chronic_handler_class_id="grid2op.Chronics.Multifolder",
        chronic_bindings=chronics,
        reward_ignored=True,
        source_network_required=False,
        fresh_target_identity_disjoint=fresh_target_identity_disjoint,
    )


__all__ = [
    'Grid2OpSourceQualificationDisposition',
    'Grid2OpSourceQualification',
    "build_held_grid2op_source_binding",
]
