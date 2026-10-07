"""Bounded science for the public MAST-U static archive-to-FreeGSNKE test."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Final, cast
import zipfile

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import ArtifactIdentity

from .contracts import StaticTransportCloseout, StaticTransportDisposition, StaticTransportFacePrediction, StaticTransportFaceResult, StaticTransportGateReceipt, StaticTransportPredictionPackage


EXPERIMENT_ID: Final = "mast-public-static-freegsnke-metatheory-validation"
TRACE_ARCHIVE_SHA256: Final = "31c1ffacde84db094c91d143d027927746f2778bd5a2bb921140de5a07349bee"
MACHINE_ARCHIVE_SHA256: Final = "28294edad722037a3597e665ad4d1ae419a14b9422a2262689ddba7ac350710a"
SHOT_MEMBERS: Final = {
    "45292": (
        "MAST-U_validation_odr/data/MAST-U_shot_45292.pickle",
        "d2bc37eac5340bab0f23dc76e6ab824fec823e921b29b2e6996378ceda9fa5e2",
    ),
    "45425": (
        "MAST-U_validation_odr/data/MAST-U_shot_45425.pickle",
        "8768d670328f78ca8dc8722967c13e22c34bf383019571aa2f5843884a4abec4",
    ),
}
MACHINE_MEMBERS: Final = {
    "active": (
        "MAST-U_active_coils_nonsym.pickle",
        "bc38cf8641e36e8bbba9a7349c0017a6042357aa586b8a7fcbede90846c88d40",
    ),
    "active-symmetric": (
        "MAST-U_active_coils.pickle",
        "5c5cc1da422fdf43f04639c85e97936dc4a68b00cfdda5a1078d426970e84c94",
    ),
    "limiter": (
        "MAST-U_limiter.pickle",
        "e1903bf8b6931e31909b31c9a6ac8a69921b2b28c526cd28c45303a914e2af7f",
    ),
    "probes": (
        "MAST-U_magnetic_probes.pickle",
        "d7c1814d67ef94e7b194aa4bb50b43e1a572ee41208396b170b4aac7c72e9f76",
    ),
    "passive": (
        "MAST-U_passive_coils.pickle",
        "76ae3e588318e60fab2a8b42ef4337e92786507541667992495156bf0c28b85b",
    ),
    "wall": (
        "MAST-U_wall.pickle",
        "e1903bf8b6931e31909b31c9a6ac8a69921b2b28c526cd28c45303a914e2af7f",
    ),
}
SLICE_TIMES_S: Final = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
COORDINATES: Final = ("ACTIVE_ONLY", "FULL_NATIVE", "SYMMETRY_COLLAPSED")
VIEWS: Final = {"baseline-65": 65, "refined-129": 129}
FACES: Final = (
    "boundary.magnetic-axis",
    "boundary.primary-x-point",
    "boundary.separatrix",
    "boundary.strike-point",
    "category.equilibrium-topology",
    "composition.independent-comparator",
    "representation.normalized-poloidal-flux",
    "response.magnetic-flux-or-pickup",
)
SOLVER_TOLERANCE: Final = 1e-6
COORDINATE_RESOLUTION_M: Final = 1e-12
FLUX_RESOLUTION: Final = 1e-12


class StaticTransportError(RuntimeError):
    """Typed experiment-local operational/contract failure."""


@dataclass(frozen=True, slots=True)
class StaticCellSummary:
    coordinate_id: str
    view_id: str
    valid: NDArray[np.bool_]
    relative_change: NDArray[np.float64]
    runtime_s: NDArray[np.float64]
    psi_normalized: NDArray[np.float64]
    axis_rz_m: NDArray[np.float64]
    xpoint_rz_m: NDArray[np.float64]
    xpoint_flux_mismatch: NDArray[np.float64]
    topology_code: NDArray[np.str_]
    midplane_r_m: NDArray[np.float64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise StaticTransportError(f"archive is absent or unsafe: {path}")
    if sha256_file(path) != expected_sha256:
        raise StaticTransportError(f"archive digest differs: {path}")


def _member_bytes(
    archive: Path,
    archive_sha256: str,
    member: str,
    member_sha256: str,
) -> bytes:
    _verify_archive(archive, archive_sha256)
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member)
        if info.file_size > 100_000_000 or info.flag_bits & 0x1:
            raise StaticTransportError("selected archive member violates its bound")
        payload = source.read(info)
    if hashlib.sha256(payload).hexdigest() != member_sha256:
        raise StaticTransportError(f"member digest differs: {member}")
    return payload


def publish_bytes(path: Path, payload: bytes) -> None:
    """No-replace publication with byte-exact idempotent recovery."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise StaticTransportError(f"no-replace publication conflict: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise StaticTransportError(f"ambiguous partial publication: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def publish_record(path: Path, record: object) -> None:
    canonical_bytes = getattr(record, "canonical_bytes", None)
    if canonical_bytes is None:
        raise TypeError("published record must be canonical")
    publish_bytes(path, cast(bytes, canonical_bytes()))


def artifact_identity(
    path: Path,
    *,
    artifact_id: str,
    role: str,
    payload_schema: str,
    media_type: str,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=artifact_id,
        role=role,
        payload_schema=payload_schema,
        sha256=sha256_file(path),
        media_type=media_type,
        size_bytes=path.stat().st_size,
    )


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[np.generic]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue())
    return output.getvalue()


def _read_npz(path: Path) -> dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def fair_metadata_canary(metadata_root: Path) -> StaticTransportGateReceipt:
    """Inspect FAIR-MAST metadata only; never open a signal value object."""

    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    shots_path = metadata_root / "shots.parquet"
    sources_path = metadata_root / "sources.parquet"
    if not shots_path.is_file() or not sources_path.is_file():
        raise StaticTransportError("pinned FAIR-MAST metadata snapshot is absent")
    shots = parquet.read_table(shots_path, columns=["shot_id", "facility"])
    sources = parquet.read_table(sources_path, columns=["shot_id", "name", "quality"])
    shot_ids = {str(value) for value in shots.column("shot_id").to_pylist()}
    source_rows = sources.to_pylist()
    source_names = {str(row["name"]) for row in source_rows}
    quality_values = {str(row["quality"]) for row in source_rows}
    # These required declarations are absent globally from the pinned schema.
    required = {
        "active_pf_order_unit_sign": False,
        "equilibrium_clock_and_validity": False,
        "geometry_identity": False,
        "magnetic_receiver_unit_and_clock": False,
        "passive_current_representation": False,
        "profile_preparation_semantics": False,
        "source_quality_qualified": quality_values == {"Checked"},
    }
    eligible = len(shot_ids) if all(required.values()) else 0
    reasons = (
        "FAIR_SOURCE_UNQUALIFIED",
        "GEOMETRY_IDENTITY_ABSENT",
        "PASSIVE_CURRENT_REPRESENTATION_ABSENT",
        "PROFILE_PREPARATION_SEMANTICS_ABSENT",
        "RECEIVER_UNIT_CLOCK_UNQUALIFIED",
        "SOURCE_QUALITY_NOT_CHECKED",
    )
    return StaticTransportGateReceipt(
        receipt_id="receipt.mast-static.fair-metadata-canary",
        gate_id="gate.f0",
        disposition=StaticTransportDisposition.STOPPED,
        reason_codes=tuple(sorted(reasons)),
        facts={
            "eligible_shot_count": eligible,
            "minimum_required_shot_count": 24,
            "receiver_value_read_count": 0,
            "route_selected": "ROUTE_U_26M5_JMPP",
            "shot_metadata_row_count": shots.num_rows,
            "source_metadata_row_count": sources.num_rows,
            "source_names": tuple(sorted(source_names)),
            "required_declarations": required,
        },
        artifacts=tuple(
            sorted(
                (
                    artifact_identity(
                        shots_path,
                        artifact_id="artifact.fair-mast.shots-metadata",
                        role="source-metadata",
                        payload_schema='empirical-lawhood/physical/mast-freegsnke-static-transport/archive-shot-metadata',
                        media_type="application/vnd.apache.parquet",
                    ),
                    artifact_identity(
                        sources_path,
                        artifact_id="artifact.fair-mast.sources-metadata",
                        role="source-metadata",
                        payload_schema='empirical-lawhood/physical/mast-freegsnke-static-transport/archive-source-metadata',
                        media_type="application/vnd.apache.parquet",
                    ),
                ),
                key=lambda value: value.artifact_id,
            )
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _shot_payload(shot_id: str, *, trace_archive: Path) -> dict[str, object]:
    member, digest = SHOT_MEMBERS[shot_id]
    payload = _member_bytes(trace_archive, TRACE_ARCHIVE_SHA256, member, digest)
    loaded = pickle.loads(payload)  # noqa: S301 - exact trusted public release, owner accepted
    if not isinstance(loaded, dict) or int(loaded.get("shot_number", -1)) != int(shot_id):
        raise StaticTransportError("selected shot payload differs from its identity")
    return cast(dict[str, object], loaded)


def _slice_indices(value: Mapping[str, object]) -> NDArray[np.int64]:
    times = np.asarray(value["efit_times"], dtype=np.float64)
    indices = []
    for requested in SLICE_TIMES_S:
        matches = np.flatnonzero(times == requested)
        if len(matches) != 1:
            raise StaticTransportError(f"shot lacks exact slice time: {requested}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def preparation_arrays(value: Mapping[str, object]) -> dict[str, NDArray[np.generic]]:
    """Select preparation fields only; this function names no receiver field."""

    indices = _slice_indices(value)
    grid = cast(Mapping[str, object], value["efit_grid"])
    profiles = cast(Mapping[str, object], value["efit_1D_current_profiles"])
    currents = cast(Mapping[str, object], value["efit_currents"])
    plasma = cast(Mapping[str, object], value["efit_plasma_current"])
    other = cast(Mapping[str, object], value["efit_other_input_output_parameters"])
    labels = np.asarray(currents["coil_name"], dtype="U32")
    current_output = np.asarray(currents["currents_output"], dtype=np.float64)[indices]
    arrays: dict[str, NDArray[np.generic]] = {
        "alpha": np.asarray(profiles["pprime_coeffs"], dtype=np.float64)[indices],
        "alpha_logic": np.asarray(profiles["pprime_flatedge_logical"], dtype=np.int8)[indices],
        "beta": np.asarray(profiles["ffprime_coeffs"], dtype=np.float64)[indices],
        "beta_logic": np.asarray(profiles["ffprime_flatedge_logical"], dtype=np.int8)[indices],
        "coil_labels": labels,
        "currents_output_a": current_output,
        "fvac_t_m": np.asarray(other["bvacradiusproduct_input"], dtype=np.float64)[indices],
        "grid": np.asarray(
            [
                float(np.asarray(grid["rmin"])[0]),
                float(np.asarray(grid["rmax"])[0]),
                float(np.asarray(grid["zmin"])[0]),
                float(np.asarray(grid["zmax"])[0]),
            ],
            dtype=np.float64,
        ),
        "plasma_current_a": np.asarray(plasma["plasma_current_output"], dtype=np.float64)[indices],
        "slice_indices": indices,
        "slice_times_s": np.asarray(SLICE_TIMES_S, dtype=np.float64),
    }
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "coil_labels"):
        raise StaticTransportError("preparation contains a non-finite selected value")
    if current_output.shape != (len(SLICE_TIMES_S), len(labels)):
        raise StaticTransportError("current preparation shape differs")
    return arrays


def receiver_arrays(value: Mapping[str, object]) -> dict[str, NDArray[np.generic]]:
    """Select the predeclared reconstruction receivers at evaluator reveal."""

    indices = _slice_indices(value)
    flux = cast(Mapping[str, object], value["efit_2D_poloidal_flux"])
    other = cast(Mapping[str, object], value["efit_other_input_output_parameters"])
    psi = 2 * np.pi * np.asarray(flux["psi"], dtype=np.float64)[indices]
    xpoints = np.stack(
        (
            np.asarray(flux["x_points_r"], dtype=np.float64)[indices],
            np.asarray(flux["x_points_z"], dtype=np.float64)[indices],
        ),
        axis=-1,
    )
    strikes = np.stack(
        (
            np.asarray(flux["r_strikepoints"], dtype=np.float64)[indices],
            np.asarray(flux["z_strikepoints"], dtype=np.float64)[indices],
        ),
        axis=-1,
    )
    for points in (xpoints, strikes):
        valid = (
            np.isfinite(points).all(axis=-1)
            & (points[..., 0] >= 0.06)
            & (points[..., 0] <= 2.0)
            & (points[..., 1] >= -2.2)
            & (points[..., 1] <= 2.2)
        )
        points[~valid] = np.nan
    arrays: dict[str, NDArray[np.generic]] = {
        "axis_rz_m": np.stack(
            (
                np.asarray(other["magnetic_axis_r"], dtype=np.float64)[indices],
                np.asarray(other["magnetic_axis_z"], dtype=np.float64)[indices],
            ),
            axis=-1,
        ),
        "midplane_r_m": np.stack(
            (
                np.asarray(flux["r_midplane_in"], dtype=np.float64)[indices],
                np.asarray(flux["r_midplane_out"], dtype=np.float64)[indices],
            ),
            axis=-1,
        ),
        "psi_axis_wb": 2 * np.pi * np.asarray(flux["psi_axis"], dtype=np.float64)[indices],
        "psi_boundary_wb": 2 * np.pi * np.asarray(flux["psi_boundary"], dtype=np.float64)[indices],
        "psi_total_wb": psi,
        "slice_times_s": np.asarray(SLICE_TIMES_S, dtype=np.float64),
        "strikepoints_rz_m": strikes,
        "xpoints_rz_m": xpoints,
    }
    required = ("axis_rz_m", "midplane_r_m", "psi_axis_wb", "psi_boundary_wb", "psi_total_wb")
    if any(not np.isfinite(arrays[name]).all() for name in required):
        raise StaticTransportError("required receiver field contains non-finite data")
    return arrays


def materialize_shot(
    shot_id: str,
    *,
    role: str,
    destination: Path,
    trace_archive: Path,
) -> ArtifactIdentity:
    if role not in {"preparation", "receiver"}:
        raise ValueError("shot role must be preparation or receiver")
    value = _shot_payload(shot_id, trace_archive=trace_archive)
    arrays = preparation_arrays(value) if role == "preparation" else receiver_arrays(value)
    publish_bytes(destination, _deterministic_npz_bytes(arrays))
    return artifact_identity(
        destination,
        artifact_id=f"artifact.mastu-shot-{shot_id}.{role}",
        role=f"mastu-{role}",
        payload_schema=f'empirical-lawhood/physical/mast-freegsnke-static-transport/{role}-arrays',
        media_type="application/x-npz",
    )


def _machine_configuration(machine_archive: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for role, (member, expected) in MACHINE_MEMBERS.items():
        payload = _member_bytes(machine_archive, MACHINE_ARCHIVE_SHA256, member, expected)
        values[role] = pickle.loads(payload)  # noqa: S301 - exact trusted public release
    return values


def _build_tokamak(machine: Mapping[str, object], coordinate_id: str) -> Any:
    from freegsnke import build_machine  # type: ignore[import-untyped]

    active_role = "active-symmetric" if coordinate_id == "SYMMETRY_COLLAPSED" else "active"
    return build_machine.tokamak(
        active_coils_data=machine[active_role],
        passive_coils_data=machine["passive"],
        limiter_data=machine["limiter"],
        wall_data=machine["wall"],
        magnetic_probe_data=machine["probes"],
    )


def _set_currents(
    tokamak: Any,
    machine: Mapping[str, object],
    coordinate_id: str,
    labels: list[str],
    values: NDArray[np.float64],
) -> None:
    lower_labels = [value.lower() for value in labels]
    if coordinate_id == "SYMMETRY_COLLAPSED":
        for name in cast(Mapping[str, object], machine["active-symmetric"]):
            source = "p1" if name == "Solenoid" else name.lower()
            indices = [index for index, label in enumerate(lower_labels[:24]) if source in label]
            if not indices:
                raise StaticTransportError(f"symmetric active mapping absent: {name}")
            polarity = float(np.sign(values[indices[0]]))
            tokamak[name].current = polarity * float(np.mean(np.abs(values[indices])))
    else:
        for name in tokamak.coils_list[:23]:
            source = "p1" if name == "Solenoid" else name.lower()
            if source not in lower_labels:
                raise StaticTransportError(f"nonsymmetric active mapping absent: {name}")
            tokamak[name].current = float(values[lower_labels.index(source)])
    passive = cast(list[dict[str, object]], machine["passive"])
    for coil in passive:
        source = str(coil.get("efitGroup", coil.get("element", ""))).lower()
        current = 0.0
        if coordinate_id != "ACTIVE_ONLY" and source in lower_labels:
            current = float(values[lower_labels.index(source)]) * float(
                cast(Any, coil["current_multiplier"])
            )
        tokamak[str(coil["name"])].current = current


def _primary_xpoint(eq: Any) -> tuple[NDArray[np.float64], float, str]:
    points = np.asarray(eq.xpt, dtype=np.float64)
    span = abs(float(eq.psi_axis) - float(eq.psi_bndry))
    mismatch = np.abs(points[:, 2] - float(eq.psi_bndry)) / span
    order = np.lexsort((points[:, 1], points[:, 0], mismatch))
    selected = int(order[0])
    point = points[selected, :2]
    tolerance = float(mismatch[selected])
    if tolerance > 1e-3:
        return point, tolerance, "UNQUALIFIED"
    return point, tolerance, "LOWER_SINGLE_NULL" if point[1] < 0 else "UPPER_SINGLE_NULL"


def _one_static_cell(
    preparation: Mapping[str, NDArray[np.generic]],
    machine: Mapping[str, object],
    coordinate_id: str,
    view_id: str,
    index: int,
) -> dict[str, NDArray[np.generic] | float | str | bool]:
    from freegsnke import GSstaticsolver, equilibrium_update, jtor_update
    from scipy.interpolate import RegularGridInterpolator

    size = VIEWS[view_id]
    grid = np.asarray(preparation["grid"], dtype=np.float64)
    tokamak = _build_tokamak(machine, coordinate_id)
    labels = np.asarray(preparation["coil_labels"]).astype(str).tolist()
    currents = np.asarray(preparation["currents_output_a"], dtype=np.float64)[index]
    _set_currents(tokamak, machine, coordinate_id, labels, currents)
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=float(grid[0]),
        Rmax=float(grid[1]),
        Zmin=float(grid[2]),
        Zmax=float(grid[3]),
        nx=size,
        ny=size,
        psi=None,
    )
    profiles = jtor_update.Lao85(
        eq=eq,
        Ip=float(np.asarray(preparation["plasma_current_a"])[index]),
        fvac=float(np.asarray(preparation["fvac_t_m"])[index]),
        alpha=np.asarray(preparation["alpha"], dtype=np.float64)[index],
        beta=np.asarray(preparation["beta"], dtype=np.float64)[index],
        alpha_logic=bool(np.asarray(preparation["alpha_logic"])[index]),
        beta_logic=bool(np.asarray(preparation["beta_logic"])[index]),
    )
    solver = GSstaticsolver.NKGSsolver(eq)
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=SOLVER_TOLERANCE,
            max_solving_iterations=100,
            verbose=False,
        )
    runtime_s = time.monotonic() - started
    psi = 2 * np.pi * np.asarray(eq.psi(), dtype=np.float64)
    axis = 2 * np.pi * float(eq.psi_axis)
    boundary = 2 * np.pi * float(eq.psi_bndry)
    normalized = (psi - axis) / (boundary - axis)
    source_r = np.linspace(grid[0], grid[1], 65)
    source_z = np.linspace(grid[2], grid[3], 65)
    r_mesh, z_mesh = np.meshgrid(source_r, source_z, indexing="ij")
    common = RegularGridInterpolator(
        (np.asarray(eq.R_1D), np.asarray(eq.Z_1D)),
        normalized,
        bounds_error=True,
    )(np.stack((r_mesh, z_mesh), axis=-1))
    point, mismatch, topology = _primary_xpoint(eq)
    midplane = np.asarray(eq.innerOuterSeparatrix(0), dtype=np.float64)
    relative_change = float(solver.relative_change)
    valid = bool(
        np.isfinite(common).all()
        and np.isfinite(point).all()
        and np.isfinite(midplane).all()
        and math.isfinite(relative_change)
        and relative_change <= SOLVER_TOLERANCE
        and not bool(eq.intersectsWall())
        and not bool(eq.flag_limiter)
        and topology != "UNQUALIFIED"
    )
    return {
        "axis_rz_m": np.asarray(eq.magneticAxis()[:2], dtype=np.float64),
        "midplane_r_m": midplane,
        "psi_normalized": common,
        "relative_change": relative_change,
        "runtime_s": runtime_s,
        "topology_code": topology,
        "valid": valid,
        "xpoint_flux_mismatch": mismatch,
        "xpoint_rz_m": point,
    }


def run_static_target(
    preparation_path: Path, destination: Path, *, machine_archive: Path
) -> ArtifactIdentity:
    preparation = _read_npz(preparation_path)
    machine = _machine_configuration(machine_archive)
    arrays: dict[str, NDArray[np.generic]] = {
        "slice_times_s": np.asarray(SLICE_TIMES_S, dtype=np.float64)
    }
    for coordinate in COORDINATES:
        for view in sorted(VIEWS):
            cells = [
                _one_static_cell(preparation, machine, coordinate, view, index)
                for index in range(len(SLICE_TIMES_S))
            ]
            prefix = f"{coordinate.lower()}__{view.replace('-', '_')}"
            for key in cells[0]:
                arrays[f"{prefix}__{key}"] = np.asarray([cell[key] for cell in cells])
    publish_bytes(destination, _deterministic_npz_bytes(arrays))
    return artifact_identity(
        destination,
        artifact_id=f"artifact.{destination.stem}",
        role="fresh-freegsnke-static-output",
        payload_schema='empirical-lawhood/physical/mast-freegsnke-static-transport/static-output-arrays',
        media_type="application/x-npz",
    )


def _cell(
    outputs: Mapping[str, NDArray[np.generic]], coordinate: str, view: str
) -> StaticCellSummary:
    prefix = f"{coordinate.lower()}__{view.replace('-', '_')}"
    return StaticCellSummary(
        coordinate_id=coordinate,
        view_id=view,
        valid=np.asarray(outputs[f"{prefix}__valid"], dtype=np.bool_),
        relative_change=np.asarray(outputs[f"{prefix}__relative_change"], dtype=np.float64),
        runtime_s=np.asarray(outputs[f"{prefix}__runtime_s"], dtype=np.float64),
        psi_normalized=np.asarray(outputs[f"{prefix}__psi_normalized"], dtype=np.float64),
        axis_rz_m=np.asarray(outputs[f"{prefix}__axis_rz_m"], dtype=np.float64),
        xpoint_rz_m=np.asarray(outputs[f"{prefix}__xpoint_rz_m"], dtype=np.float64),
        xpoint_flux_mismatch=np.asarray(
            outputs[f"{prefix}__xpoint_flux_mismatch"], dtype=np.float64
        ),
        topology_code=np.asarray(outputs[f"{prefix}__topology_code"], dtype=str),
        midplane_r_m=np.asarray(outputs[f"{prefix}__midplane_r_m"], dtype=np.float64),
    )


def static_output_validity(outputs_path: Path) -> dict[str, bool]:
    outputs = _read_npz(outputs_path)
    return {
        coordinate: all(
            bool(value)
            for view in sorted(VIEWS)
            for value in _cell(outputs, coordinate, view).valid
        )
        for coordinate in COORDINATES
    }


def _source_topology(receivers: Mapping[str, NDArray[np.generic]]) -> NDArray[np.str_]:
    points = np.asarray(receivers["xpoints_rz_m"], dtype=np.float64)
    labels = []
    for value in points:
        valid = value[np.isfinite(value).all(axis=1)]
        if len(valid) != 1:
            labels.append("UNQUALIFIED")
        else:
            labels.append("LOWER_SINGLE_NULL" if valid[0, 1] < 0 else "UPPER_SINGLE_NULL")
    return np.asarray(labels, dtype=str)


def face_point_estimates(
    receivers_path: Path,
    outputs_path: Path,
) -> dict[str, dict[str, dict[str, object]]]:
    receivers = _read_npz(receivers_path)
    outputs = _read_npz(outputs_path)
    source_psi = np.asarray(receivers["psi_total_wb"], dtype=np.float64)
    source_axis_flux = np.asarray(receivers["psi_axis_wb"], dtype=np.float64)
    source_boundary_flux = np.asarray(receivers["psi_boundary_wb"], dtype=np.float64)
    source_normalized = (source_psi - source_axis_flux[:, None, None]) / (
        source_boundary_flux - source_axis_flux
    )[:, None, None]
    source_axis = np.asarray(receivers["axis_rz_m"], dtype=np.float64)
    source_xpoints = np.asarray(receivers["xpoints_rz_m"], dtype=np.float64)
    source_midplane = np.asarray(receivers["midplane_r_m"], dtype=np.float64)
    source_topology = _source_topology(receivers)
    result: dict[str, dict[str, dict[str, object]]] = {}
    for coordinate in COORDINATES:
        result[coordinate] = {}
        for view in sorted(VIEWS):
            cell = _cell(outputs, coordinate, view)
            source_primary = np.full((len(SLICE_TIMES_S), 2), np.nan)
            for index, points in enumerate(source_xpoints):
                valid = points[np.isfinite(points).all(axis=1)]
                if len(valid) == 1:
                    source_primary[index] = valid[0]
            psi_delta = cell.psi_normalized - source_normalized
            result[coordinate][view] = {
                "axis_distance_m": np.linalg.norm(cell.axis_rz_m - source_axis, axis=1).tolist(),
                "complete_cell": cell.valid.tolist(),
                "flux_max_abs_1": np.max(np.abs(psi_delta), axis=(1, 2)).tolist(),
                "flux_rms_1": np.sqrt(np.mean(psi_delta**2, axis=(1, 2))).tolist(),
                "midplane_max_abs_m": np.max(
                    np.abs(cell.midplane_r_m - source_midplane), axis=1
                ).tolist(),
                "relative_change": cell.relative_change.tolist(),
                "runtime_s": cell.runtime_s.tolist(),
                "source_topology": source_topology.tolist(),
                "target_topology": cell.topology_code.tolist(),
                "topology_match": (cell.topology_code == source_topology).tolist(),
                "xpoint_distance_m": np.linalg.norm(
                    cell.xpoint_rz_m - source_primary, axis=1
                ).tolist(),
                "xpoint_source_complete": np.isfinite(source_primary).all(axis=1).tolist(),
            }
    return result


def _maximum(values: object) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise StaticTransportError("no finite development metric values")
    return float(np.max(finite))


def build_prediction_package(
    development_receivers: Path,
    development_outputs: Path,
    *,
    evaluator_implementation_sha256: str,
) -> tuple[StaticTransportPredictionPackage, dict[str, object]]:
    metrics = face_point_estimates(development_receivers, development_outputs)
    full_refined = metrics["FULL_NATIVE"]["refined-129"]
    output_arrays = _read_npz(development_outputs)
    full_base_cell = _cell(output_arrays, "FULL_NATIVE", "baseline-65")
    full_refined_cell = _cell(output_arrays, "FULL_NATIVE", "refined-129")
    if not all(full_base_cell.valid) or not all(full_refined_cell.valid):
        raise StaticTransportError("development FULL_NATIVE target is incomplete")
    view_disagreement = {
        "boundary.magnetic-axis": float(
            np.max(np.linalg.norm(full_base_cell.axis_rz_m - full_refined_cell.axis_rz_m, axis=1))
        ),
        "boundary.primary-x-point": float(
            np.max(
                np.linalg.norm(full_base_cell.xpoint_rz_m - full_refined_cell.xpoint_rz_m, axis=1)
            )
        ),
        "representation.normalized-poloidal-flux": float(
            np.max(
                np.sqrt(
                    np.mean(
                        (full_base_cell.psi_normalized - full_refined_cell.psi_normalized) ** 2,
                        axis=(1, 2),
                    )
                )
            )
        ),
    }
    metric_inputs = {
        "boundary.magnetic-axis": ("m", full_refined["axis_distance_m"]),
        "boundary.primary-x-point": ("m", full_refined["xpoint_distance_m"]),
        "representation.normalized-poloidal-flux": ("1", full_refined["flux_rms_1"]),
    }
    predictions: list[StaticTransportFacePrediction] = []
    for face in FACES:
        if face in metric_inputs:
            unit, values = metric_inputs[face]
            if face == "boundary.primary-x-point" and not all(
                cast(list[bool], full_refined["xpoint_source_complete"])
            ):
                predictions.append(
                    StaticTransportFacePrediction(
                        face_id=face,
                        predictive_level="METRIC",
                        applicable=False,
                        metric_unit=unit,
                        development_maximum=None,
                        development_view_disagreement_maximum=None,
                        categorical_prediction=None,
                        source_uncertainty_qualified=False,
                        decisive_falsifier="OPERAND_ABSENT",
                        limitation_codes=("DEVELOPMENT_SOURCE_XPOINT_INCOMPLETE",),
                    )
                )
                continue
            predictions.append(
                StaticTransportFacePrediction(
                    face_id=face,
                    predictive_level="METRIC",
                    applicable=True,
                    metric_unit=unit,
                    development_maximum=format(_maximum(values), ".17g"),
                    development_view_disagreement_maximum=format(view_disagreement[face], ".17g"),
                    categorical_prediction=None,
                    source_uncertainty_qualified=False,
                    decisive_falsifier="protected-shot-maximum-exceeds-issued-envelope",
                    limitation_codes=("SOURCE_OBSERVATION_UNCERTAINTY_UNKNOWN",),
                )
            )
        elif face == "category.equilibrium-topology":
            topology_complete = all(
                value != "UNQUALIFIED" for value in cast(list[str], full_refined["source_topology"])
            )
            predictions.append(
                StaticTransportFacePrediction(
                    face_id=face,
                    predictive_level="CATEGORICAL",
                    applicable=topology_complete,
                    metric_unit="category",
                    development_maximum=None,
                    development_view_disagreement_maximum=None,
                    categorical_prediction=(
                        "EXACT_SOURCE_SLICE_LABEL" if topology_complete else None
                    ),
                    source_uncertainty_qualified=True,
                    decisive_falsifier=(
                        "any-complete-protected-slice-category-mismatch"
                        if topology_complete
                        else "OPERAND_ABSENT"
                    ),
                    limitation_codes=(
                        () if topology_complete else ("DEVELOPMENT_SOURCE_TOPOLOGY_INCOMPLETE",)
                    ),
                )
            )
        else:
            predictions.append(
                StaticTransportFacePrediction(
                    face_id=face,
                    predictive_level="METRIC",
                    applicable=False,
                    metric_unit="unbound",
                    development_maximum=None,
                    development_view_disagreement_maximum=None,
                    categorical_prediction=None,
                    source_uncertainty_qualified=False,
                    decisive_falsifier="OPERAND_ABSENT",
                    limitation_codes=("SOURCE_QUALIFIED_OPERAND_ABSENT",),
                )
            )
    package = StaticTransportPredictionPackage(
        prediction_id="prediction.mast-static-transport.route-u",
        experiment_id=EXPERIMENT_ID,
        selected_route="ROUTE_U_26M5_JMPP",
        source_shot_id="45292",
        target_shot_id="45425",
        source_member_sha256=SHOT_MEMBERS["45292"][1],
        target_member_sha256=SHOT_MEMBERS["45425"][1],
        machine_archive_sha256=MACHINE_ARCHIVE_SHA256,
        slice_times_s=tuple(format(value, ".17g") for value in SLICE_TIMES_S),
        coordinate_ids=tuple(sorted(COORDINATES)),
        numerical_view_ids=tuple(sorted(VIEWS)),
        predictions=tuple(sorted(predictions, key=lambda value: value.face_id)),
        target_preparation_contact_count=0,
        target_receiver_contact_count=0,
        target_execution_count=0,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        claim_ceiling=(
            "ONE_PROTECTED_MASTU_SHOT_STATIC_PROPERTY_EXISTENCE_WITH_SHARED_PREPARATION_LINEAGE"
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    dossier: dict[str, object] = {
        "development_metrics": metrics,
        "prediction_fingerprint": package.fingerprint(),
        "view_disagreement": view_disagreement,
    }
    return package, dossier


def _result_for_metric(
    prediction: StaticTransportFacePrediction,
    target_maximum: float,
) -> StaticTransportFaceResult:
    development = float(cast(str, prediction.development_maximum))
    disagreement = float(cast(str, prediction.development_view_disagreement_maximum))
    resolution = COORDINATE_RESOLUTION_M if prediction.metric_unit == "m" else FLUX_RESOLUTION
    descriptive_limit = development + disagreement + resolution
    inside = target_maximum <= descriptive_limit
    # Unknown source uncertainty is controlling; descriptive envelope is not promoted.
    return StaticTransportFaceResult(
        face_id=prediction.face_id,
        disposition=StaticTransportDisposition.UNEVALUABLE,
        predictive_level="METRIC",
        complete_unit_numerator=1,
        complete_unit_denominator=1,
        target_value=format(target_maximum, ".17g"),
        issued_limit=None,
        point_estimate_inside_development_envelope=inside,
        falsifier_triggered=None,
        reason_codes=("SOURCE_OBSERVATION_UNCERTAINTY_UNKNOWN",),
        evidence_ceiling="DESCRIPTIVE_POINT_ESTIMATE_ONLY",
    )


def evaluate_closeout(
    prediction: StaticTransportPredictionPackage,
    target_receivers: Path,
    target_outputs: Path,
    *,
    route_receipt: ArtifactIdentity,
    development_artifacts: tuple[ArtifactIdentity, ...],
    target_artifacts: tuple[ArtifactIdentity, ...],
) -> tuple[StaticTransportCloseout, dict[str, object]]:
    metrics = face_point_estimates(target_receivers, target_outputs)
    full = metrics["FULL_NATIVE"]["refined-129"]
    results: list[StaticTransportFaceResult] = []
    prediction_by_face = {value.face_id: value for value in prediction.predictions}
    for face in FACES:
        item = prediction_by_face[face]
        if not item.applicable:
            results.append(
                StaticTransportFaceResult(
                    face_id=face,
                    disposition=StaticTransportDisposition.NOT_APPLICABLE,
                    predictive_level=item.predictive_level,
                    complete_unit_numerator=0,
                    complete_unit_denominator=1,
                    target_value=None,
                    issued_limit=None,
                    point_estimate_inside_development_envelope=None,
                    falsifier_triggered=None,
                    reason_codes=("SOURCE_QUALIFIED_OPERAND_ABSENT",),
                    evidence_ceiling="OPERAND_ABSENT",
                )
            )
        elif face == "boundary.magnetic-axis":
            results.append(_result_for_metric(item, _maximum(full["axis_distance_m"])))
        elif face == "boundary.primary-x-point":
            if not all(cast(list[bool], full["xpoint_source_complete"])):
                results.append(
                    StaticTransportFaceResult(
                        face_id=face,
                        disposition=StaticTransportDisposition.UNEVALUABLE,
                        predictive_level="METRIC",
                        complete_unit_numerator=0,
                        complete_unit_denominator=1,
                        target_value=None,
                        issued_limit=None,
                        point_estimate_inside_development_envelope=None,
                        falsifier_triggered=None,
                        reason_codes=("AMBIGUOUS_SOURCE_XPOINT_SLOT",),
                        evidence_ceiling="SOURCE_RECONSTRUCTION_LABEL_UNQUALIFIED",
                    )
                )
            else:
                results.append(_result_for_metric(item, _maximum(full["xpoint_distance_m"])))
        elif face == "representation.normalized-poloidal-flux":
            results.append(_result_for_metric(item, _maximum(full["flux_rms_1"])))
        elif face == "category.equilibrium-topology":
            complete = [
                source != "UNQUALIFIED" and bool(match)
                for source, match in zip(
                    cast(list[str], full["source_topology"]),
                    cast(list[bool], full["topology_match"]),
                    strict=True,
                )
            ]
            source_complete = all(
                value != "UNQUALIFIED" for value in cast(list[str], full["source_topology"])
            )
            disposition = (
                StaticTransportDisposition.SUPPORTED
                if source_complete and all(complete)
                else StaticTransportDisposition.OPPOSED
                if source_complete
                else StaticTransportDisposition.UNEVALUABLE
            )
            reasons = (
                ()
                if disposition is StaticTransportDisposition.SUPPORTED
                else ("PROTECTED_TOPOLOGY_MISMATCH",)
                if disposition is StaticTransportDisposition.OPPOSED
                else ("AMBIGUOUS_SOURCE_TOPOLOGY_LABEL",)
            )
            results.append(
                StaticTransportFaceResult(
                    face_id=face,
                    disposition=disposition,
                    predictive_level="CATEGORICAL",
                    complete_unit_numerator=int(source_complete),
                    complete_unit_denominator=1,
                    target_value=None,
                    issued_limit=None,
                    point_estimate_inside_development_envelope=None,
                    falsifier_triggered=(not all(complete)) if source_complete else None,
                    reason_codes=reasons,
                    evidence_ceiling="ONE_PROTECTED_MASTU_SHOT_CATEGORY_EXISTENCE",
                )
            )
    entered_metric_keys = ("axis_distance_m", "flux_rms_1")
    full_maxima = {key: _maximum(full[key]) for key in entered_metric_keys}
    no_worse = True
    strictly_better = False
    challenger_summary: dict[str, object] = {}
    for challenger in ("ACTIVE_ONLY", "SYMMETRY_COLLAPSED"):
        challenger_values = metrics[challenger]["refined-129"]
        targetable = all(cast(list[bool], challenger_values["complete_cell"]))
        if not targetable:
            challenger_summary[challenger] = {
                "disposition": "COORDINATE_UNTARGETABLE",
                "complete_cell": challenger_values["complete_cell"],
            }
            continue
        maxima = {key: _maximum(challenger_values[key]) for key in entered_metric_keys}
        comparisons = {key: full_maxima[key] <= maxima[key] for key in entered_metric_keys}
        no_worse = no_worse and all(comparisons.values())
        strictly_better = strictly_better or any(
            full_maxima[key] < maxima[key] for key in entered_metric_keys
        )
        challenger_summary[challenger] = {
            "disposition": "TARGETABLE",
            "maxima": maxima,
            "full_no_worse": comparisons,
        }
    challenger_summary["DESCRIPTIVE_SUMMARY"] = {
        "full_native_no_worse_on_complete_point_estimates": no_worse,
        "full_native_strictly_better_on_at_least_one_point_estimate": strictly_better,
    }
    coordinate_disposition = StaticTransportDisposition.UNEVALUABLE
    coordinate_reasons = (
        "ENTERED_METRIC_FACE_UNEVALUABLE",
        "SOURCE_OBSERVATION_UNCERTAINTY_UNKNOWN",
    )
    closeout = StaticTransportCloseout(
        closeout_id="closeout.mast-static-freegsnke-metatheory-validation",
        experiment_id=EXPERIMENT_ID,
        prediction_package_fingerprint=prediction.fingerprint(),
        route_receipt=route_receipt,
        development_artifacts=tuple(
            sorted(development_artifacts, key=lambda value: value.artifact_id)
        ),
        target_artifacts=tuple(sorted(target_artifacts, key=lambda value: value.artifact_id)),
        face_results=tuple(sorted(results, key=lambda value: value.face_id)),
        coordinate_disposition=coordinate_disposition,
        coordinate_reason_codes=coordinate_reasons,
        achieved_dependence_class="SHARED_ARCHIVE_PREPARATION_AND_ANALYSIS_LINEAGE",
        shared_lineage=True,
        operational_status="COMPLETED",
        claim_ceiling=(
            "ONE_PROTECTED_MASTU_SHOT_STATIC_PROPERTY_EXISTENCE_NO_PREVALENCE_OR_INDEPENDENT_RECURRENCE"
        ),
        response_law_constructed=False,
        admission_or_controller_evaluation_constructed=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    analysis: dict[str, object] = {
        "challenger_summary": challenger_summary,
        "coordinate_disposition": coordinate_disposition.value,
        "face_metrics": metrics,
        "full_native_target_maxima": full_maxima,
    }
    return closeout, analysis
