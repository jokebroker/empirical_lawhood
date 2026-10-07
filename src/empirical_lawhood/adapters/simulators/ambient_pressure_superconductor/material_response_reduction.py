'Material reduction for the conditional material control multiband gauge covariant response method.\n\nThis module is deliberately narrower than a general electromagnetic-response\nsolver.  It evaluates a finite-temperature, finite-wave-vector transverse\nresponse for an exact Wannier tight-binding operand and an orbital-local\nEliashberg self-energy.  A caller may treat the result as material evidence\nonly when a separate compatibility record establishes the common Wannier\ngauge, bounds the omitted non-local position matrix elements, and bounds the\npairing projection and response-level numerical convergence.  A separate\nMigdal-parameter proxy screens the local-self-energy approximation regime; it\nis not represented as a rigorous transverse-vertex bound.\n\nThe vector potential is introduced on every hopping by the exact straight-bond\nPeierls line integral.  The anomalous self-energy transforms at both endpoints\nof a pair, so a lattice pure-gauge test exercises the assembled material\noperator rather than a post-hoc scalar correction.  The finite Matsubara\ncontact term is removed only by an exact denominator-matched normal comparator.\n'

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .multiband_strong_coupling_response import ELEMENTARY_CHARGE_C, EV_TO_J, HBAR_J_S, KB_EV_PER_K, MU0_H_PER_M


ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
ReducedPosition = tuple[float, float, float]
Translation = tuple[int, int, int]
Twist = tuple[float, float, float]

_HERMITICITY_LIMIT_EV: Final = 1.0e-9


@dataclass(frozen=True, slots=True)
class WannierHopping:
    """One directed Wannier Hamiltonian matrix element ``H_ij(R)``."""

    translation: Translation
    source_orbital: int
    target_orbital: int
    real_eV: float
    imag_eV: float

    def __post_init__(self) -> None:
        if len(self.translation) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.translation
        ):
            raise ValueError("Wannier translation must contain three integers")
        if self.source_orbital < 0 or self.target_orbital < 0:
            raise ValueError("Wannier orbital indices must be nonnegative")
        if not np.isfinite(self.real_eV) or not np.isfinite(self.imag_eV):
            raise ValueError("Wannier hopping must be finite")

    @property
    def value_eV(self) -> complex:
        return complex(self.real_eV, self.imag_eV)


@dataclass(frozen=True, slots=True)
class MaterialEliashbergSlice:
    """One positive-frequency self-energy in the Hamiltonian's Wannier gauge."""

    omega_eV: float
    z_orbital: tuple[float, ...]
    phi_eV_orbital: tuple[float, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.omega_eV) or self.omega_eV <= 0.0:
            raise ValueError("positive finite Matsubara frequency required")
        if not self.z_orbital or len(self.z_orbital) != len(self.phi_eV_orbital):
            raise ValueError("Z and phi must cover one common nonempty orbital set")
        if any(not np.isfinite(value) or value < 1.0 for value in self.z_orbital):
            raise ValueError("orbital Z values must be finite and at least one")
        if any(not np.isfinite(value) or value < 0.0 for value in self.phi_eV_orbital):
            raise ValueError("orbital phi values must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class WannierMaterialSpec:
    """Bounded finite-q material denominator with transverse twist quadrature."""

    nx: int
    orbital_centres_reduced: tuple[ReducedPosition, ...]
    hoppings: tuple[WannierHopping, ...]
    eliashberg_slices: tuple[MaterialEliashbergSlice, ...]
    transverse_twists: tuple[Twist, ...]
    temperature_K: float
    primitive_cell_volume_m3: float
    transverse_lattice_vector_m: float
    chemical_potential_eV: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 8:
            raise ValueError("three-mode finite-q inference requires nx >= 8")
        orbital_count = len(self.orbital_centres_reduced)
        if orbital_count < 2:
            raise ValueError("material bridge requires at least two Wannier orbitals")
        for centre in self.orbital_centres_reduced:
            if len(centre) != 3 or any(not np.isfinite(value) for value in centre):
                raise ValueError("Wannier centres must be finite reduced coordinates")
        if not self.hoppings:
            raise ValueError("material bridge requires a nonempty Wannier Hamiltonian")
        if any(
            value.source_orbital >= orbital_count or value.target_orbital >= orbital_count
            for value in self.hoppings
        ):
            raise ValueError("Wannier hopping references an absent orbital")
        if any(abs(value.translation[0]) * 2 >= self.nx for value in self.hoppings):
            raise ValueError("nx aliases a retained Wannier hopping translation")
        if not self.eliashberg_slices or any(
            len(value.z_orbital) != orbital_count for value in self.eliashberg_slices
        ):
            raise ValueError("every self-energy slice must cover every Wannier orbital")
        frequencies = tuple(value.omega_eV for value in self.eliashberg_slices)
        if frequencies != tuple(sorted(set(frequencies))):
            raise ValueError("Matsubara frequencies must be strictly increasing")
        if not self.transverse_twists:
            raise ValueError("transverse Brillouin-zone quadrature cannot be empty")
        if self.transverse_twists != tuple(sorted(set(self.transverse_twists))):
            raise ValueError("transverse twists must be sorted and unique")
        if any(
            len(value) != 3
            or any(
                not np.isfinite(component) or component < 0.0 or component >= 1.0
                for component in value
            )
            for value in self.transverse_twists
        ):
            raise ValueError("Brillouin-zone twists must lie in [0, 1)")
        for field_name in (
            "temperature_K",
            "primitive_cell_volume_m3",
            "transverse_lattice_vector_m",
        ):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{field_name} must be positive and finite')
        if not np.isfinite(self.chemical_potential_eV):
            raise ValueError("chemical potential must be finite")
        _assert_hermitian(self)

    @property
    def orbital_count(self) -> int:
        return len(self.orbital_centres_reduced)

    @property
    def normal_size(self) -> int:
        return self.nx * self.orbital_count


@dataclass(frozen=True, slots=True)
class MaterialR2Mode:
    mode: int
    q_reduced: float
    total_eV_per_cell: float
    diamagnetic_eV_per_cell: float
    paramagnetic_eV_per_cell: float
    decomposition_residual_eV_per_cell: float
    signed_current_residual_eV_per_cell: float


@dataclass(frozen=True, slots=True)
class MaterialR2Result:
    intercept_eV_per_cell: float
    raw_normal_intercept_eV_per_cell: float
    slope_eV_per_cell: float
    finite_q_fit_residual_eV_per_cell: float
    london_kernel_J_per_m3_A2: float
    inverse_penetration_depth_sq_per_m2: float
    gauge_covariance_residual_eV_per_cell: float
    modes: tuple[MaterialR2Mode, ...]


def _index(spec: WannierMaterialSpec, x: int, orbital: int) -> int:
    return (x % spec.nx) * spec.orbital_count + orbital


def _line_integral_y(*, start_x: float, delta_x: float, delta_y: float, q: float) -> float:
    """Exact reduced-coordinate integral of ``A_y=cos(q*x)`` on a straight bond."""

    if abs(delta_x) <= 1.0e-14:
        return delta_y * cos(q * start_x)
    return delta_y * (sin(q * (start_x + delta_x)) - sin(q * start_x)) / (q * delta_x)


def _normal_hamiltonian(
    spec: WannierMaterialSpec,
    *,
    amplitude: float,
    mode: int,
    twist: Twist,
    gauge_chi: RealVector | None = None,
) -> ComplexMatrix:
    result = np.zeros((spec.normal_size, spec.normal_size), dtype=np.complex128)
    q = 2.0 * pi * mode / spec.nx
    kx_offset, ky, kz = twist
    for x in range(spec.nx):
        for hopping in spec.hoppings:
            source = _index(spec, x, hopping.source_orbital)
            target = _index(spec, x + hopping.translation[0], hopping.target_orbital)
            source_centre = spec.orbital_centres_reduced[hopping.source_orbital]
            target_centre = spec.orbital_centres_reduced[hopping.target_orbital]
            delta_x = hopping.translation[0] + target_centre[0] - source_centre[0]
            delta_y = hopping.translation[1] + target_centre[1] - source_centre[1]
            delta_z = hopping.translation[2] + target_centre[2] - source_centre[2]
            start_x = x + source_centre[0]
            peierls = amplitude * _line_integral_y(
                start_x=start_x,
                delta_x=delta_x,
                delta_y=delta_y,
                q=q,
            )
            phase = peierls + 2.0 * pi * (
                kx_offset * delta_x / spec.nx + ky * delta_y + kz * delta_z
            )
            if gauge_chi is not None:
                phase += gauge_chi[target] - gauge_chi[source]
            result[source, target] += hopping.value_eV * np.exp(1j * phase)
    result -= spec.chemical_potential_eV * np.eye(spec.normal_size, dtype=np.complex128)
    return result


def _assert_hermitian(spec: WannierMaterialSpec) -> None:
    for twist in (spec.transverse_twists[0], spec.transverse_twists[-1]):
        matrix = _normal_hamiltonian(spec, amplitude=0.0, mode=1, twist=twist)
        residual = float(np.max(np.abs(matrix - matrix.conjugate().T)))
        if residual > _HERMITICITY_LIMIT_EV:
            raise ValueError("Wannier Hamiltonian is not Hermitian in the retained supercell")


def _nambu_inverse(
    spec: WannierMaterialSpec,
    slice_: MaterialEliashbergSlice,
    *,
    amplitude: float,
    mode: int,
    twist: Twist,
    gauge_chi: RealVector | None = None,
) -> ComplexMatrix:
    electron = _normal_hamiltonian(
        spec,
        amplitude=amplitude,
        mode=mode,
        twist=twist,
        gauge_chi=gauge_chi,
    )
    # The y/z directions have already been Fourier-reduced to twists.  The
    # transpose of the full real-space operator therefore lives in the
    # opposite twist block.  Its transpose reverses both the physical Peierls
    # link and the pure-gauge link; changing either amplitude a second time
    # would double-reverse the field.
    opposite_twist: Twist = (
        (-twist[0]) % 1.0,
        (-twist[1]) % 1.0,
        (-twist[2]) % 1.0,
    )
    hole = _normal_hamiltonian(
        spec,
        amplitude=amplitude,
        mode=mode,
        twist=opposite_twist,
        gauge_chi=gauge_chi,
    ).T
    z = np.empty(spec.normal_size, dtype=np.float64)
    phi = np.empty(spec.normal_size, dtype=np.complex128)
    for x in range(spec.nx):
        for orbital in range(spec.orbital_count):
            index = _index(spec, x, orbital)
            z[index] = slice_.z_orbital[orbital]
            pair_phase = 0.0 if gauge_chi is None else -2.0 * gauge_chi[index]
            phi[index] = slice_.phi_eV_orbital[orbital] * np.exp(1j * pair_phase)
    frequency = 1j * slice_.omega_eV * np.diag(z)
    pairing = np.diag(phi)
    return np.block(
        [
            [frequency - electron, -pairing],
            [-pairing.conjugate(), frequency + hole],
        ]
    )


def _grand_potential_eV(
    spec: WannierMaterialSpec,
    *,
    amplitude: float,
    mode: int,
    gauge_chi: RealVector | None = None,
) -> float:
    total = 0.0
    for twist in spec.transverse_twists:
        for slice_ in spec.eliashberg_slices:
            operator = _nambu_inverse(
                spec,
                slice_,
                amplitude=amplitude,
                mode=mode,
                twist=twist,
                gauge_chi=gauge_chi,
            )
            sign, logabsdet = np.linalg.slogdet(operator)
            if sign == 0 or not np.isfinite(logabsdet):
                raise ValueError("singular or nonfinite material Nambu operator")
            total += float(logabsdet)
    return -KB_EV_PER_K * spec.temperature_K * total / len(spec.transverse_twists)


def material_normal_comparator(spec: WannierMaterialSpec) -> WannierMaterialSpec:
    return WannierMaterialSpec(
        nx=spec.nx,
        orbital_centres_reduced=spec.orbital_centres_reduced,
        hoppings=spec.hoppings,
        eliashberg_slices=tuple(
            MaterialEliashbergSlice(
                omega_eV=value.omega_eV,
                z_orbital=value.z_orbital,
                phi_eV_orbital=tuple(0.0 for _ in value.phi_eV_orbital),
            )
            for value in spec.eliashberg_slices
        ),
        transverse_twists=spec.transverse_twists,
        temperature_K=spec.temperature_K,
        primitive_cell_volume_m3=spec.primitive_cell_volume_m3,
        transverse_lattice_vector_m=spec.transverse_lattice_vector_m,
        chemical_potential_eV=spec.chemical_potential_eV,
    )


def material_gauge_residual_eV_per_cell(spec: WannierMaterialSpec) -> float:
    chi = np.empty(spec.normal_size, dtype=np.float64)
    for x in range(spec.nx):
        value = 0.13 * sin(2.0 * pi * x / spec.nx)
        for orbital in range(spec.orbital_count):
            chi[_index(spec, x, orbital)] = value
    reference = _grand_potential_eV(spec, amplitude=0.0, mode=1)
    transformed = _grand_potential_eV(spec, amplitude=0.0, mode=1, gauge_chi=chi)
    return abs(transformed - reference) / spec.nx


def _raw_mode(
    spec: WannierMaterialSpec,
    *,
    mode: int,
    step: float,
    current_amplitude: float,
) -> MaterialR2Mode:
    if mode <= 0 or mode >= spec.nx // 2:
        raise ValueError("finite-q mode is outside the resolved long-wavelength branch")
    if not np.isfinite(step) or step <= 0.0 or step >= 0.01:
        raise ValueError("curvature step is outside the declared weak-field range")
    if current_amplitude <= 2.0 * step or current_amplitude >= 0.01:
        raise ValueError("signed-current amplitude must exceed two steps and remain weak")

    zero = _grand_potential_eV(spec, amplitude=0.0, mode=mode)
    plus = _grand_potential_eV(spec, amplitude=step, mode=mode)
    minus = _grand_potential_eV(spec, amplitude=-step, mode=mode)
    total = (plus + minus - 2.0 * zero) / (step * step)

    diamagnetic = 0.0
    paramagnetic = 0.0
    for twist in spec.transverse_twists:
        for slice_ in spec.eliashberg_slices:
            operator_zero = _nambu_inverse(
                spec,
                slice_,
                amplitude=0.0,
                mode=mode,
                twist=twist,
            )
            operator_plus = _nambu_inverse(
                spec,
                slice_,
                amplitude=step,
                mode=mode,
                twist=twist,
            )
            operator_minus = _nambu_inverse(
                spec,
                slice_,
                amplitude=-step,
                mode=mode,
                twist=twist,
            )
            first = (operator_plus - operator_minus) / (2.0 * step)
            second = (operator_plus + operator_minus - 2.0 * operator_zero) / (step * step)
            inverse = np.linalg.inv(operator_zero)
            factor = -KB_EV_PER_K * spec.temperature_K / len(spec.transverse_twists)
            diamagnetic += factor * float(np.trace(inverse @ second).real)
            paramagnetic -= factor * float(np.trace(inverse @ first @ inverse @ first).real)

    def current(at: float) -> float:
        high = _grand_potential_eV(spec, amplitude=at + step, mode=mode)
        low = _grand_potential_eV(spec, amplitude=at - step, mode=mode)
        return (high - low) / (2.0 * step)

    normalization = float(spec.nx)
    return MaterialR2Mode(
        mode=mode,
        q_reduced=2.0 * pi * mode / spec.nx,
        total_eV_per_cell=total / normalization,
        diamagnetic_eV_per_cell=diamagnetic / normalization,
        paramagnetic_eV_per_cell=paramagnetic / normalization,
        decomposition_residual_eV_per_cell=(
            abs(total - diamagnetic - paramagnetic) / normalization
        ),
        signed_current_residual_eV_per_cell=(
            abs(current(current_amplitude) + current(-current_amplitude)) / normalization
        ),
    )


def material_mode(
    spec: WannierMaterialSpec,
    *,
    mode: int,
    step: float = 2.0e-4,
    current_amplitude: float = 8.0e-4,
) -> MaterialR2Mode:
    result, _raw_normal = _material_mode_with_normal(
        spec,
        mode=mode,
        step=step,
        current_amplitude=current_amplitude,
    )
    return result


def _material_mode_with_normal(
    spec: WannierMaterialSpec,
    *,
    mode: int,
    step: float,
    current_amplitude: float,
) -> tuple[MaterialR2Mode, MaterialR2Mode]:
    paired = _raw_mode(
        spec,
        mode=mode,
        step=step,
        current_amplitude=current_amplitude,
    )
    normal = _raw_mode(
        material_normal_comparator(spec),
        mode=mode,
        step=step,
        current_amplitude=current_amplitude,
    )
    total = paired.total_eV_per_cell - normal.total_eV_per_cell
    diamagnetic = paired.diamagnetic_eV_per_cell - normal.diamagnetic_eV_per_cell
    paramagnetic = paired.paramagnetic_eV_per_cell - normal.paramagnetic_eV_per_cell
    return MaterialR2Mode(
        mode=mode,
        q_reduced=paired.q_reduced,
        total_eV_per_cell=total,
        diamagnetic_eV_per_cell=diamagnetic,
        paramagnetic_eV_per_cell=paramagnetic,
        decomposition_residual_eV_per_cell=abs(total - diamagnetic - paramagnetic),
        signed_current_residual_eV_per_cell=max(
            paired.signed_current_residual_eV_per_cell,
            normal.signed_current_residual_eV_per_cell,
        ),
    ), normal


def evaluate_material_multiband_strong_coupling_response(
    spec: WannierMaterialSpec,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    step: float = 2.0e-4,
    current_amplitude: float = 8.0e-4,
) -> MaterialR2Result:
    if len(modes) < 3 or modes != tuple(sorted(set(modes))):
        raise ValueError("material finite-q inference requires at least three sorted modes")
    observations: list[MaterialR2Mode] = []
    raw_normal_observations: list[MaterialR2Mode] = []
    for mode in modes:
        observation, raw_normal = _material_mode_with_normal(
            spec,
            mode=mode,
            step=step,
            current_amplitude=current_amplitude,
        )
        observations.append(observation)
        raw_normal_observations.append(raw_normal)
    x = np.asarray([value.q_reduced**2 for value in observations], dtype=np.float64)
    y = np.asarray([value.total_eV_per_cell for value in observations], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    raw_normal_y = np.asarray(
        [value.total_eV_per_cell for value in raw_normal_observations],
        dtype=np.float64,
    )
    raw_normal_coefficients, *_ = np.linalg.lstsq(design, raw_normal_y, rcond=None)
    intercept = float(coefficients[0])
    # Every sampled mode is A_y=A0*cos(qx), q>0.  Its spatial mean-square is
    # A0^2/2, so the curvature with respect to A0 is K*V/2.  Restore that
    # factor before converting the phase curvature to the SI London kernel.
    energy_density = 2.0 * intercept * EV_TO_J / spec.primitive_cell_volume_m3
    gauge_factor = ELEMENTARY_CHARGE_C * spec.transverse_lattice_vector_m / HBAR_J_S
    london_kernel = energy_density * gauge_factor * gauge_factor
    return MaterialR2Result(
        intercept_eV_per_cell=intercept,
        raw_normal_intercept_eV_per_cell=float(raw_normal_coefficients[0]),
        slope_eV_per_cell=float(coefficients[1]),
        finite_q_fit_residual_eV_per_cell=float(np.max(np.abs(y - fitted))),
        london_kernel_J_per_m3_A2=london_kernel,
        inverse_penetration_depth_sq_per_m2=MU0_H_PER_M * london_kernel,
        gauge_covariance_residual_eV_per_cell=material_gauge_residual_eV_per_cell(spec),
        modes=tuple(observations),
    )


__all__ = [
    "MaterialEliashbergSlice",
    "MaterialR2Mode",
    "MaterialR2Result",
    "WannierHopping",
    "WannierMaterialSpec",
    'evaluate_material_multiband_strong_coupling_response',
    "material_gauge_residual_eV_per_cell",
    "material_mode",
    "material_normal_comparator",
]
