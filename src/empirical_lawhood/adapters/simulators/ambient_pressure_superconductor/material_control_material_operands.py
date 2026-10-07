'Typed reduction of EPW/Wannier outputs to the conditional material-gauge covariant response bridge.\n\nThe reducer never accepts a band-gap distribution as an orbital self-energy by\nassertion.  For anisotropic controls it reconstructs the EPW fine-Fermi-shell\nk/band ordering from the exact binary ``egnv`` operand, diagonalizes the held\nWannier Hamiltonian at those k points, and fits the local self-energy in that\nsame Wannier gauge.  Failure of energy alignment, fit, position-operator,\nMatsubara-decay, or Migdal-parameter proxy is a validity stop.  The Migdal\nquantity is an approximation-validity diagnostic, not a rigorous vertex bound.\n'

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from math import pi
from pathlib import Path
import re
import struct
import tarfile
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .material_source_design_design import build_material_roster
from .material_control_contracts import MultibandStrongCouplingMaterialCompatibility, MaterialGaugeCovariantCompatibilityDisposition, MaterialGaugeCovariantViewDisposition, MultibandStrongCouplingMaterialViewResult
from .material_response_reduction import MaterialEliashbergSlice, WannierHopping, WannierMaterialSpec, evaluate_material_multiband_strong_coupling_response
from .material_control_solver import MaterialControlWorkflowProfile


MAXIMUM_MEMBER_BYTES: Final = 2 * 1024**3
PAIRING_PROJECTION_LIMIT: Final = Decimal("0.15")
POSITION_OMISSION_LIMIT: Final = Decimal("0.10")
MATSUBARA_DECAY_PROXY_LIMIT: Final = Decimal("0.05")
MIGDAL_PARAMETER_LIMIT: Final = Decimal("0.10")
RYDBERG_TO_EV: Final = 13.605_693_122_994

ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WannierHamiltonianOperand:
    payload_sha256: str
    orbital_count: int
    hoppings: tuple[WannierHopping, ...]


@dataclass(frozen=True, slots=True)
class PositionOperand:
    payload_sha256: str
    orbital_centres_reduced: tuple[tuple[float, float, float], ...]
    offdiagonal_relative_upper: float


@dataclass(frozen=True, slots=True)
class FineShellPoint:
    reduced_k: tuple[float, float, float]
    energies_eV: tuple[float, ...]
    reference_energy_eV: float


@dataclass(frozen=True, slots=True)
class MaterialOperandReduction:
    spec: WannierMaterialSpec | None
    compatibility: MultibandStrongCouplingMaterialCompatibility


def unresolved_material_operands(
    *, profile: MaterialControlWorkflowProfile, raw_archive_sha256: str, reason_code: str
) -> MaterialOperandReduction:
    """Return a typed validity stop when exact material reduction cannot be decoded."""

    compatibility = MultibandStrongCouplingMaterialCompatibility(
        compatibility_id=(
            f"compatibility.material-control-{profile.structure_id.removeprefix('structure.')}-"
            f"{profile.view_id.removeprefix('view.')}-gauge-covariant-response-unresolved"
        ),
        structure_id=profile.structure_id,
        view_id=profile.view_id,
        normal_hamiltonian_sha256=raw_archive_sha256,
        eliashberg_state_sha256=raw_archive_sha256,
        orbital_centres_sha256=raw_archive_sha256,
        common_wannier_gauge=False,
        local_self_energy_supported=False,
        phonon_spectrum_available=False,
        migdal_parameter_proxy_resolved=False,
        static_transverse_pair_decoupling_supported=False,
        static_coulomb_gauge_supported=True,
        native_to_si_map_complete=True,
        pairing_projection_relative_residual=Decimal("1e99"),
        pairing_projection_relative_limit=PAIRING_PROJECTION_LIMIT,
        hamiltonian_truncation_eV=Decimal("1e99"),
        hamiltonian_truncation_limit_eV=Decimal("0"),
        position_matrix_omission_relative_upper=Decimal("1e99"),
        position_matrix_omission_relative_limit=POSITION_OMISSION_LIMIT,
        matsubara_decay_proxy_relative_upper=Decimal("1e99"),
        matsubara_decay_proxy_relative_limit=MATSUBARA_DECAY_PROXY_LIMIT,
        migdal_parameter_relative_upper=Decimal("1e99"),
        migdal_parameter_relative_limit=MIGDAL_PARAMETER_LIMIT,
        disposition=MaterialGaugeCovariantCompatibilityDisposition.UNRESOLVED,
        material_promotion_authorized=False,
        reason_codes=(reason_code,),
    )
    return MaterialOperandReduction(spec=None, compatibility=compatibility)


def _decimal(value: float) -> Decimal:
    return Decimal("1e99") if not np.isfinite(value) else Decimal(str(value))


def _archive_members(path: Path) -> dict[str, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError('material-gauge covariant response raw archive is unavailable')
    result: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.issym() or member.islnk():
                continue
            if member.size > MAXIMUM_MEMBER_BYTES:
                raise ValueError('material-gauge covariant response raw member exceeds its bound')
            name = member.name
            if not (
                name.endswith(("_hr.dat", "_r.dat", ".wout"))
                or "/egnv" in name
                or ".imag_iso_" in name
                or ".imag_aniso_" in name
                or name.endswith(".phdos")
                or "matdyn.stdout" in name
            ):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('material-gauge covariant response raw member cannot be read')
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError('material-gauge covariant response raw member size differs')
            result[name] = payload
    return result


def _one_member(members: dict[str, bytes], suffix: str) -> tuple[str, bytes]:
    values = tuple((name, payload) for name, payload in members.items() if name.endswith(suffix))
    if len(values) != 1:
        raise ValueError(f'material-gauge covariant response archive requires exactly one {suffix} member')
    return values[0]


def parse_wannier_hr(payload: bytes) -> WannierHamiltonianOperand:
    lines = payload.decode("ascii").splitlines()
    if len(lines) < 4:
        raise ValueError("Wannier HR payload is truncated")
    orbital_count = int(lines[1].strip())
    translation_count = int(lines[2].strip())
    if orbital_count < 2 or translation_count <= 0:
        raise ValueError("Wannier HR dimensions are invalid")
    cursor = 3
    degeneracies: list[int] = []
    while len(degeneracies) < translation_count:
        degeneracies.extend(int(value) for value in lines[cursor].split())
        cursor += 1
    if len(degeneracies) != translation_count or any(value <= 0 for value in degeneracies):
        raise ValueError("Wannier HR degeneracies are invalid")
    expected = translation_count * orbital_count * orbital_count
    data = lines[cursor:]
    if len(data) != expected:
        raise ValueError("Wannier HR matrix row count differs")
    hoppings: list[WannierHopping] = []
    block = orbital_count * orbital_count
    for index, line in enumerate(data):
        fields = line.split()
        if len(fields) != 7:
            raise ValueError("Wannier HR row shape differs")
        degeneracy = degeneracies[index // block]
        hoppings.append(
            WannierHopping(
                translation=(int(fields[0]), int(fields[1]), int(fields[2])),
                source_orbital=int(fields[3]) - 1,
                target_orbital=int(fields[4]) - 1,
                real_eV=float(fields[5]) / degeneracy,
                imag_eV=float(fields[6]) / degeneracy,
            )
        )
    return WannierHamiltonianOperand(
        payload_sha256=sha256(payload).hexdigest(),
        orbital_count=orbital_count,
        hoppings=tuple(hoppings),
    )


def parse_wannier_position(
    payload: bytes,
    *,
    orbital_count: int,
    lattice_vectors_A: RealMatrix,
) -> PositionOperand:
    lines = payload.decode("ascii").splitlines()
    if len(lines) < 4 or int(lines[1]) != orbital_count:
        raise ValueError("Wannier position operand orbital count differs")
    translation_count = int(lines[2])
    rows = lines[3:]
    if len(rows) != translation_count * orbital_count * orbital_count:
        raise ValueError("Wannier position row count differs")
    centres_A = np.zeros((orbital_count, 3), dtype=np.float64)
    offdiagonal_norm_sq = 0.0
    for line in rows:
        fields = line.split()
        if len(fields) != 11:
            raise ValueError("Wannier position row shape differs")
        translation = tuple(int(value) for value in fields[:3])
        source = int(fields[3]) - 1
        target = int(fields[4]) - 1
        vector = np.asarray(
            [
                complex(float(fields[5 + 2 * axis]), float(fields[6 + 2 * axis]))
                for axis in range(3)
            ],
            dtype=np.complex128,
        )
        if translation == (0, 0, 0) and source == target:
            centres_A[source] = vector.real
        else:
            offdiagonal_norm_sq += float(np.vdot(vector, vector).real)
    inverse = np.linalg.inv(lattice_vectors_A.T)
    reduced_values = tuple(inverse @ centre for centre in centres_A)
    reduced = tuple(
        (
            float(value[0] % 1.0),
            float(value[1] % 1.0),
            float(value[2] % 1.0),
        )
        for value in reduced_values
    )
    characteristic_A = float(min(np.linalg.norm(value) for value in lattice_vectors_A))
    relative = (offdiagonal_norm_sq**0.5) / max(characteristic_A, 1.0e-15)
    return PositionOperand(
        payload_sha256=sha256(payload).hexdigest(),
        orbital_centres_reduced=reduced,
        offdiagonal_relative_upper=relative,
    )


def parse_epw_egnv(payload: bytes) -> tuple[FineShellPoint, ...]:
    """Decode EPW 6.1's little-endian direct-access stream operand."""

    stream = BytesIO(payload)

    def unpack(format_: str) -> tuple[int | float, ...]:
        size = struct.calcsize(format_)
        block = stream.read(size)
        if len(block) != size:
            raise ValueError("EPW egnv stream is truncated")
        return struct.unpack(format_, block)

    nkftot, _nk1, _nk2, _nk3, nks = (int(value) for value in unpack("<5i"))
    header = unpack("<i5d")
    band_count = int(header[0])
    # EPW writes ``ef0`` and the following eigenvalues in Rydberg.  Its reader
    # converts both to eV only after reopening this stream; do the same before
    # comparing them with the Wannier Hamiltonian, whose native unit is eV.
    reference_energy_eV = float(header[2]) * RYDBERG_TO_EV
    if nkftot <= 0 or nks <= 0 or band_count < 2:
        raise ValueError("EPW egnv dimensions are invalid")
    points: list[tuple[int, FineShellPoint]] = []
    for _index in range(nkftot):
        row = unpack("<4d2i")
        coordinates = (float(row[1]), float(row[2]), float(row[3]))
        shell_index = int(row[5])
        energies = tuple(
            float(unpack("<d")[0]) * RYDBERG_TO_EV - reference_energy_eV for _ in range(band_count)
        )
        if shell_index > 0:
            points.append(
                (
                    shell_index,
                    FineShellPoint(
                        reduced_k=coordinates,
                        energies_eV=energies,
                        reference_energy_eV=reference_energy_eV,
                    ),
                )
            )
    if stream.read(1) or len(points) != nks:
        raise ValueError("EPW egnv shell count or stream length differs")
    points.sort(key=lambda value: value[0])
    if tuple(index for index, _value in points) != tuple(range(1, nks + 1)):
        raise ValueError("EPW egnv shell indices are not contiguous")
    return tuple(value for _index, value in points)


def _hamiltonian_at_k(
    operand: WannierHamiltonianOperand,
    reduced_k: tuple[float, float, float],
) -> ComplexMatrix:
    result = np.zeros((operand.orbital_count, operand.orbital_count), dtype=np.complex128)
    for hopping in operand.hoppings:
        phase = 2.0 * pi * sum(reduced_k[axis] * hopping.translation[axis] for axis in range(3))
        result[hopping.source_orbital, hopping.target_orbital] += hopping.value_eV * np.exp(
            1j * phase
        )
    return result


def _numeric_rows(payload: bytes) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for line in payload.splitlines():
        if not line.strip() or line.lstrip().startswith((b"#", b"!")):
            continue
        try:
            row = tuple(float(value.replace(b"D", b"E")) for value in line.split())
        except ValueError:
            continue
        if row:
            rows.append(row)
    return tuple(rows)


def _temperature_from_member(name: str) -> float | None:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", name)
    return None if match is None else float(match.group(1))


def _select_gap_member(
    members: dict[str, bytes], *, prefix: str, temperature_K: float, anisotropic: bool
) -> tuple[str, bytes]:
    marker = f"{prefix}.imag_{('aniso' if anisotropic else 'iso')}_"
    candidates = tuple(
        (name, payload)
        for name, payload in members.items()
        if marker in name and "gap0" not in name
    )
    matching = tuple(
        value
        for value in candidates
        if (temperature := _temperature_from_member(value[0])) is not None
        and abs(temperature - temperature_K) < 1.0e-6
    )
    if len(matching) != 1:
        raise ValueError('material-gauge covariant response archive lacks its exact control-temperature gap member')
    return matching[0]


def _isotropic_slices(
    payload: bytes, *, orbital_count: int
) -> tuple[tuple[MaterialEliashbergSlice, ...], float]:
    rows = _numeric_rows(payload)
    if not rows or any(len(row) < 3 for row in rows):
        raise ValueError("isotropic Eliashberg rows are unavailable")
    slices = tuple(
        MaterialEliashbergSlice(
            omega_eV=row[0],
            z_orbital=tuple(row[1] for _ in range(orbital_count)),
            phi_eV_orbital=tuple(row[1] * abs(row[2]) for _ in range(orbital_count)),
        )
        for row in rows
    )
    return slices, 0.0


def _anisotropic_slices(
    payload: bytes,
    *,
    hamiltonian: WannierHamiltonianOperand,
    shell: tuple[FineShellPoint, ...],
) -> tuple[tuple[MaterialEliashbergSlice, ...], float, float]:
    rows = _numeric_rows(payload)
    expected_per_frequency = sum(
        sum(abs(energy) < 0.2 for energy in point.energies_eV) for point in shell
    )
    if expected_per_frequency <= 0 or len(rows) % expected_per_frequency:
        raise ValueError("anisotropic Eliashberg row count differs from the fine shell")
    weights: list[tuple[float, ...]] = []
    energy_residuals: list[float] = []
    for point in shell:
        values, vectors = np.linalg.eigh(_hamiltonian_at_k(hamiltonian, point.reduced_k))
        values -= point.reference_energy_eV
        for band, energy in enumerate(point.energies_eV):
            if abs(energy) >= 0.2:
                continue
            nearest = int(np.argmin(np.abs(values - energy)))
            energy_residuals.append(abs(float(values[nearest]) - energy))
            weights.append(tuple(float(value) for value in np.abs(vectors[:, nearest]) ** 2))
    design = np.asarray(weights, dtype=np.float64)
    if design.shape != (expected_per_frequency, hamiltonian.orbital_count):
        raise ValueError("anisotropic Wannier projection matrix shape differs")
    slices: list[MaterialEliashbergSlice] = []
    relative_residual = 0.0
    for offset in range(0, len(rows), expected_per_frequency):
        block = rows[offset : offset + expected_per_frequency]
        omega = block[0][0]
        if any(len(row) < 4 or abs(row[0] - omega) > 1.0e-9 for row in block):
            raise ValueError("anisotropic Eliashberg frequency block differs")
        observed_z = np.asarray([row[2] for row in block], dtype=np.float64)
        observed_phi = np.asarray([row[2] * abs(row[3]) for row in block], dtype=np.float64)
        z_fit, *_ = np.linalg.lstsq(design, observed_z, rcond=None)
        phi_fit, *_ = np.linalg.lstsq(design, observed_phi, rcond=None)
        z_fit = np.maximum(z_fit, 1.0)
        phi_fit = np.maximum(phi_fit, 0.0)
        predicted = design @ phi_fit
        denominator = max(float(np.linalg.norm(observed_phi)), 1.0e-15)
        relative_residual = max(
            relative_residual,
            float(np.linalg.norm(predicted - observed_phi)) / denominator,
        )
        slices.append(
            MaterialEliashbergSlice(
                omega_eV=omega,
                z_orbital=tuple(float(value) for value in z_fit),
                phi_eV_orbital=tuple(float(value) for value in phi_fit),
            )
        )
    return tuple(slices), relative_residual, max(energy_residuals, default=float("inf"))


def _matsubara_decay_proxy(slices: tuple[MaterialEliashbergSlice, ...]) -> float:
    terms = np.asarray(
        [
            sum(value * value for value in slice_.phi_eV_orbital) / slice_.omega_eV**4
            for slice_ in slices
        ],
        dtype=np.float64,
    )
    if len(terms) < 4 or float(np.sum(terms)) <= 0:
        return float("inf")
    # A positive decreasing 1/omega^4 asymptotic estimate nominates whether a
    # response-level retained-slice convergence check is worth attempting.
    # It is not itself promoted as a rigorous response-tail bound.
    tail = float(terms[-1] * max(len(terms), 1) / 3.0)
    return tail / float(np.sum(terms))


def _minimum_fermi_pocket_depth_eV(
    hamiltonian: WannierHamiltonianOperand,
    *,
    chemical_potential_eV: float,
) -> float | None:
    """Return a conservative sampled electronic scale for the Migdal proxy.

    Full bandwidth can hide a shallow Fermi pocket in a multiband material.
    The proxy therefore uses the smallest distance from the chemical potential
    to either sampled edge of every band that crosses it.  Absence of a
    resolved crossing is a validity stop, not a fallback to a larger scale.
    """

    by_band: list[list[float]] = [[] for _ in range(hamiltonian.orbital_count)]
    grid = tuple(index / 10.0 for index in range(10))
    for kx in grid:
        for ky in grid:
            for kz in grid:
                values = np.linalg.eigvalsh(_hamiltonian_at_k(hamiltonian, (kx, ky, kz)))
                for band, value in enumerate(values):
                    by_band[band].append(float(value))
    depths = tuple(
        min(chemical_potential_eV - min(values), max(values) - chemical_potential_eV)
        for values in by_band
        if min(values) < chemical_potential_eV < max(values)
    )
    return None if not depths else min(depths)


def _max_phonon_eV(members: dict[str, bytes]) -> float | None:
    pattern = re.compile(
        rb"freq\s*\([^)]*\)\s*=\s*[-+0-9.EeDd]+\s*\[THz\]\s*=\s*"
        rb"([-+0-9.EeDd]+)\s*\[cm-1\]",
        re.IGNORECASE,
    )
    values: list[float] = []
    for name, payload in members.items():
        if "matdyn.stdout" not in name:
            continue
        values.extend(abs(float(value.replace(b"D", b"E"))) for value in pattern.findall(payload))
    return None if not values else max(values) * 1.239_841_984e-4


def reduce_material_operands(
    *,
    raw_archive_path: Path,
    profile: MaterialControlWorkflowProfile,
    fermi_energy_eV: Decimal,
    electron_phonon_lambda: Decimal,
) -> MaterialOperandReduction:
    if profile.structure_id not in {'structure.calibration-pb-fcc', 'structure.calibration-mgb2-alb2'}:
        raise ValueError('material-gauge covariant response reduction is reserved for positive controls')
    members = _archive_members(raw_archive_path)
    _hr_name, hr_payload = _one_member(members, f'{profile.prefix}_hr.dat')
    _position_name, position_payload = _one_member(members, f'{profile.prefix}_r.dat')
    hamiltonian = parse_wannier_hr(hr_payload)
    structure = next(
        value
        for value in build_material_roster().structures
        if value.structure_id == profile.structure_id
    )
    lattice = np.asarray(
        [[float(component) for component in vector] for vector in structure.lattice_vectors_A],
        dtype=np.float64,
    )
    position = parse_wannier_position(
        position_payload,
        orbital_count=hamiltonian.orbital_count,
        lattice_vectors_A=lattice,
    )
    is_mgb2 = profile.structure_id == 'structure.calibration-mgb2-alb2'
    temperature_K = 20.0 if is_mgb2 else 4.0
    gap_name, gap_payload = _select_gap_member(
        members,
        prefix=profile.prefix,
        temperature_K=temperature_K,
        anisotropic=is_mgb2,
    )
    projection_residual = 0.0
    energy_alignment_residual = 0.0
    material_fermi_eV = float(fermi_energy_eV)
    if is_mgb2:
        egnv_values = tuple(payload for name, payload in members.items() if name.endswith("/egnv"))
        if len(egnv_values) != 1:
            raise ValueError('anisotropic material-gauge covariant response reduction requires one egnv operand')
        shell = parse_epw_egnv(egnv_values[0])
        material_fermi_eV = shell[0].reference_energy_eV
        slices, projection_residual, energy_alignment_residual = _anisotropic_slices(
            gap_payload,
            hamiltonian=hamiltonian,
            shell=shell,
        )
    else:
        slices, projection_residual = _isotropic_slices(
            gap_payload,
            orbital_count=hamiltonian.orbital_count,
        )
    decay_proxy = _matsubara_decay_proxy(slices)
    fermi_pocket_depth_eV = _minimum_fermi_pocket_depth_eV(
        hamiltonian,
        chemical_potential_eV=material_fermi_eV,
    )
    maximum_phonon_eV = _max_phonon_eV(members)
    phonon_spectrum_available = maximum_phonon_eV is not None
    migdal_parameter = (
        float("inf")
        if maximum_phonon_eV is None or fermi_pocket_depth_eV is None
        else float(electron_phonon_lambda) * maximum_phonon_eV / max(fermi_pocket_depth_eV, 1.0e-15)
    )
    common_gauge = energy_alignment_residual <= 1.0e-4
    local_self_energy = projection_residual <= float(PAIRING_PROJECTION_LIMIT)
    position_bound = position.offdiagonal_relative_upper
    transverse_decoupling = all(
        (
            common_gauge,
            local_self_energy,
            position_bound <= float(POSITION_OMISSION_LIMIT),
        )
    )
    pass_flags = all(
        (
            common_gauge,
            local_self_energy,
            phonon_spectrum_available,
            transverse_decoupling,
            position_bound <= float(POSITION_OMISSION_LIMIT),
            decay_proxy <= float(MATSUBARA_DECAY_PROXY_LIMIT),
            migdal_parameter <= float(MIGDAL_PARAMETER_LIMIT),
        )
    )
    reasons: list[str] = []
    for passed, reason in (
        (common_gauge, "reason.common-wannier-gauge-unresolved"),
        (local_self_energy, "reason.orbital-local-self-energy-fit-unresolved"),
        (phonon_spectrum_available, "reason.phonon-spectrum-operand-missing"),
        (
            transverse_decoupling,
            "reason.static-transverse-pair-decoupling-condition-unresolved",
        ),
        (
            position_bound <= float(POSITION_OMISSION_LIMIT),
            "reason.position-matrix-omission-bound-exceeded",
        ),
        (
            decay_proxy <= float(MATSUBARA_DECAY_PROXY_LIMIT),
            "reason.matsubara-decay-proxy-exceeded",
        ),
        (
            migdal_parameter <= float(MIGDAL_PARAMETER_LIMIT),
            "reason.migdal-parameter-proxy-exceeded",
        ),
    ):
        if not passed:
            reasons.append(reason)
    if pass_flags:
        reasons.append('reason.all-material-gauge-covariant-response-validity-operands-resolved')
    compatibility = MultibandStrongCouplingMaterialCompatibility(
        compatibility_id=(
            f"compatibility.material-control-{profile.structure_id.removeprefix('structure.')}-"
            f"{profile.view_id.removeprefix('view.')}-gauge-covariant-response"
        ),
        structure_id=profile.structure_id,
        view_id=profile.view_id,
        normal_hamiltonian_sha256=hamiltonian.payload_sha256,
        eliashberg_state_sha256=sha256(gap_payload).hexdigest(),
        orbital_centres_sha256=position.payload_sha256,
        common_wannier_gauge=common_gauge,
        local_self_energy_supported=local_self_energy,
        phonon_spectrum_available=phonon_spectrum_available,
        migdal_parameter_proxy_resolved=(migdal_parameter <= float(MIGDAL_PARAMETER_LIMIT)),
        static_transverse_pair_decoupling_supported=transverse_decoupling,
        static_coulomb_gauge_supported=True,
        native_to_si_map_complete=True,
        pairing_projection_relative_residual=_decimal(projection_residual),
        pairing_projection_relative_limit=PAIRING_PROJECTION_LIMIT,
        hamiltonian_truncation_eV=Decimal("0"),
        hamiltonian_truncation_limit_eV=Decimal("0"),
        position_matrix_omission_relative_upper=_decimal(position_bound),
        position_matrix_omission_relative_limit=POSITION_OMISSION_LIMIT,
        matsubara_decay_proxy_relative_upper=_decimal(decay_proxy),
        matsubara_decay_proxy_relative_limit=MATSUBARA_DECAY_PROXY_LIMIT,
        migdal_parameter_relative_upper=_decimal(migdal_parameter),
        migdal_parameter_relative_limit=MIGDAL_PARAMETER_LIMIT,
        disposition=(
            MaterialGaugeCovariantCompatibilityDisposition.PASS
            if pass_flags
            else MaterialGaugeCovariantCompatibilityDisposition.UNRESOLVED
        ),
        material_promotion_authorized=False,
        reason_codes=tuple(sorted(reasons)),
    )
    if not pass_flags:
        return MaterialOperandReduction(spec=None, compatibility=compatibility)
    max_rx = max(abs(value.translation[0]) for value in hamiltonian.hoppings)
    minimum_nx = 12 if "efficiency" in profile.view_id else 16
    nx = max(minimum_nx, 2 * max_rx + 6)
    twist_values = (0.25, 0.75) if "efficiency" in profile.view_id else (0.125, 0.375, 0.625, 0.875)
    twists = tuple(sorted((0.0, ky, kz) for ky in twist_values for kz in twist_values))
    volume_m3 = abs(float(np.linalg.det(lattice))) * 1.0e-30
    transverse_length_m = float(np.linalg.norm(lattice[1])) * 1.0e-10
    spec = WannierMaterialSpec(
        nx=nx,
        orbital_centres_reduced=position.orbital_centres_reduced,
        hoppings=hamiltonian.hoppings,
        eliashberg_slices=slices,
        transverse_twists=twists,
        temperature_K=temperature_K,
        primitive_cell_volume_m3=volume_m3,
        transverse_lattice_vector_m=transverse_length_m,
        chemical_potential_eV=material_fermi_eV,
    )
    return MaterialOperandReduction(spec=spec, compatibility=compatibility)


def evaluate_material_view(
    *, reduction: MaterialOperandReduction, profile: MaterialControlWorkflowProfile
) -> MultibandStrongCouplingMaterialViewResult:
    compatibility = reduction.compatibility
    if reduction.spec is None:
        return MultibandStrongCouplingMaterialViewResult(
            result_id=f"result.material-control-{profile.profile_id.removeprefix('workflow.')}-material-gauge-covariant-response",
            structure_id=profile.structure_id,
            view_id=profile.view_id,
            compatibility_sha256=compatibility.fingerprint(),
            compatibility_passed=False,
            temperature_K=Decimal("20") if "mgb2" in profile.profile_id else Decimal("4"),
            mode_count=3,
            gauge_covariance_residual_eV_per_cell=Decimal("0"),
            signed_current_residual_eV_per_cell=Decimal("0"),
            decomposition_residual_eV_per_cell=Decimal("0"),
            finite_q_fit_residual_eV_per_cell=Decimal("0"),
            finite_q_fit_relative_limit=Decimal("0.10"),
            intercept_eV_per_cell=Decimal("0"),
            raw_normal_intercept_eV_per_cell=Decimal("0"),
            inverse_penetration_depth_sq_per_m2=Decimal("0"),
            finite_difference_convergence_relative=Decimal("0"),
            spatial_grid_convergence_relative=Decimal("0"),
            matsubara_convergence_relative=Decimal("0"),
            uncertainty_relative_upper=Decimal("0"),
            stiffness_lower_per_m2=Decimal("0"),
            stiffness_upper_per_m2=Decimal("0"),
            disposition=MaterialGaugeCovariantViewDisposition.VALIDITY_UNRESOLVED,
            material_promotion_authorized=False,
            reason_codes=compatibility.reason_codes,
        )
    result = evaluate_material_multiband_strong_coupling_response(
        reduction.spec,
        step=1.0e-4,
        current_amplitude=8.0e-4,
    )
    step_view = evaluate_material_multiband_strong_coupling_response(
        reduction.spec,
        step=2.0e-4,
        current_amplitude=8.0e-4,
    )
    retained_count = max(4, 3 * len(reduction.spec.eliashberg_slices) // 4)
    truncated_spec = replace(
        reduction.spec,
        eliashberg_slices=reduction.spec.eliashberg_slices[:retained_count],
    )
    matsubara_view = evaluate_material_multiband_strong_coupling_response(
        truncated_spec,
        step=1.0e-4,
        current_amplitude=8.0e-4,
    )
    maximum_translation = max(abs(value.translation[0]) for value in reduction.spec.hoppings)
    base_nx = max(8, 2 * maximum_translation + 2, reduction.spec.nx - 4)
    spatial_view = evaluate_material_multiband_strong_coupling_response(
        replace(reduction.spec, nx=base_nx),
        step=1.0e-4,
        current_amplitude=8.0e-4,
    )
    maximum_signed = max(value.signed_current_residual_eV_per_cell for value in result.modes)
    maximum_decomposition = max(value.decomposition_residual_eV_per_cell for value in result.modes)
    relative_limit = Decimal("0.10")
    point = result.inverse_penetration_depth_sq_per_m2
    finite_difference_relative = abs(
        result.intercept_eV_per_cell - step_view.intercept_eV_per_cell
    ) / max(abs(result.intercept_eV_per_cell), 1.0e-30)
    matsubara_relative = abs(
        result.intercept_eV_per_cell - matsubara_view.intercept_eV_per_cell
    ) / max(abs(result.intercept_eV_per_cell), 1.0e-30)
    spatial_relative = abs(result.intercept_eV_per_cell - spatial_view.intercept_eV_per_cell) / max(
        abs(result.intercept_eV_per_cell), 1.0e-30
    )
    numerical_relative = max(
        0.01,
        result.finite_q_fit_residual_eV_per_cell / max(abs(result.intercept_eV_per_cell), 1.0e-30),
        finite_difference_relative,
        spatial_relative,
        matsubara_relative,
        float(compatibility.position_matrix_omission_relative_upper),
        float(compatibility.pairing_projection_relative_residual),
        float(compatibility.migdal_parameter_relative_upper),
    )
    radius = abs(point) * numerical_relative
    lower = point - radius
    upper = point + radius
    numerical_checks = (
        (
            result.gauge_covariance_residual_eV_per_cell <= 1.0e-9,
            'reason.material-gauge-covariant-response-gauge-covariance-fail',
        ),
        (maximum_signed <= 1.0e-8, 'reason.material-gauge-covariant-response-signed-current-fail'),
        (
            maximum_decomposition <= 1.0e-7,
            'reason.material-gauge-covariant-response-dia-para-decomposition-fail',
        ),
        (
            result.finite_q_fit_residual_eV_per_cell
            <= float(relative_limit) * abs(result.intercept_eV_per_cell),
            'reason.material-gauge-covariant-response-finite-q-fit-fail',
        ),
        (
            finite_difference_relative <= 0.05,
            'reason.material-gauge-covariant-response-finite-difference-convergence-fail',
        ),
        (
            spatial_relative <= 0.10,
            'reason.material-gauge-covariant-response-spatial-grid-convergence-fail',
        ),
        (
            matsubara_relative <= 0.05,
            'reason.material-gauge-covariant-response-matsubara-convergence-fail',
        ),
        (lower > 0.0, 'reason.material-gauge-covariant-response-positive-interval-fail'),
    )
    numerical_pass = all(passed for passed, _reason in numerical_checks)
    reasons = tuple(
        sorted(
            ('reason.material-gauge-covariant-response-view-numerics-pass',)
            if numerical_pass
            else tuple(reason for passed, reason in numerical_checks if not passed)
        )
    )
    return MultibandStrongCouplingMaterialViewResult(
        result_id=f"result.material-control-{profile.profile_id.removeprefix('workflow.')}-material-gauge-covariant-response",
        structure_id=profile.structure_id,
        view_id=profile.view_id,
        compatibility_sha256=compatibility.fingerprint(),
        compatibility_passed=True,
        temperature_K=_decimal(reduction.spec.temperature_K),
        mode_count=len(result.modes),
        gauge_covariance_residual_eV_per_cell=_decimal(
            result.gauge_covariance_residual_eV_per_cell
        ),
        signed_current_residual_eV_per_cell=_decimal(maximum_signed),
        decomposition_residual_eV_per_cell=_decimal(maximum_decomposition),
        finite_q_fit_residual_eV_per_cell=_decimal(result.finite_q_fit_residual_eV_per_cell),
        finite_q_fit_relative_limit=relative_limit,
        intercept_eV_per_cell=_decimal(result.intercept_eV_per_cell),
        raw_normal_intercept_eV_per_cell=_decimal(abs(result.raw_normal_intercept_eV_per_cell)),
        inverse_penetration_depth_sq_per_m2=_decimal(point),
        finite_difference_convergence_relative=_decimal(finite_difference_relative),
        spatial_grid_convergence_relative=_decimal(spatial_relative),
        matsubara_convergence_relative=_decimal(matsubara_relative),
        uncertainty_relative_upper=_decimal(numerical_relative),
        stiffness_lower_per_m2=_decimal(lower),
        stiffness_upper_per_m2=_decimal(upper),
        disposition=(
            MaterialGaugeCovariantViewDisposition.PASS
            if numerical_pass
            else MaterialGaugeCovariantViewDisposition.NUMERICAL_FAIL
        ),
        material_promotion_authorized=False,
        reason_codes=reasons,
    )


__all__ = [
    "FineShellPoint",
    "MaterialOperandReduction",
    "PositionOperand",
    "WannierHamiltonianOperand",
    "evaluate_material_view",
    "parse_epw_egnv",
    "parse_wannier_hr",
    "parse_wannier_position",
    "reduce_material_operands",
    "unresolved_material_operands",
]
