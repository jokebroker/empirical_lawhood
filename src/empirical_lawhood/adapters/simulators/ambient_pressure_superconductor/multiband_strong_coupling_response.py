'Multiband strong-coupling bridge for the ambient pressure superconductor material gauge covariant response receiver.\n\nThe gauge covariant response lattice fixture intentionally qualified only a single-band static BCS\nmethod.  This additive module supplies the missing material bridge needed by\nMgB2-like controls.  It evaluates the fermionic effective action of a local,\norbital-resolved Eliashberg self-energy in a finite periodic Wannier lattice.\nThe electromagnetic field enters every hopping through a Peierls phase and\nthe anomalous self-energy transforms with charge ``2e``.  These two operations\nmake pure-gauge covariance a property of the constructed operator rather than\nan after-the-fact correction.\n\nThe implementation is a bounded conformance and material-reduction method.  A\nmaterial use must separately prove that the normal Hamiltonian, Eliashberg\nself-energy and orbital transformation share one Wannier gauge.  It must not\nconsume band-resolved gaps without that compatibility map, and it does not\ninfer a transition temperature from a stiffness calculation.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import cos, pi, sin
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .material_control_contracts import BridgeDisposition, MultibandStrongCouplingBridgeQualification


KB_EV_PER_K: Final = 8.617_333_262_145e-5
ELEMENTARY_CHARGE_C: Final = 1.602_176_634e-19
HBAR_J_S: Final = 1.054_571_817e-34
MU0_H_PER_M: Final = 1.256_637_062_12e-6
EV_TO_J: Final = ELEMENTARY_CHARGE_C

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LocalEliashbergSlice:
    """One positive-Matsubara local self-energy in a fixed orbital gauge."""

    omega_eV: float
    z_orbital: tuple[float, ...]
    phi_eV_orbital: tuple[float, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.omega_eV) or self.omega_eV <= 0.0:
            raise ValueError("positive finite Matsubara frequency required")
        if not self.z_orbital or len(self.z_orbital) != len(self.phi_eV_orbital):
            raise ValueError("Z and phi must cover the same nonempty orbital set")
        if any(not np.isfinite(value) or value < 1.0 for value in self.z_orbital):
            raise ValueError("causal quasiparticle renormalization requires finite Z >= 1")
        if any(not np.isfinite(value) or value < 0.0 for value in self.phi_eV_orbital):
            raise ValueError("this bridge requires finite nonnegative real gap amplitudes")


@dataclass(frozen=True, slots=True)
class MultibandLatticeSpec:
    'Small periodic two-dimensional Wannier representation for gauge covariant response checks.'

    nx: int
    ny: int
    hopping_x_eV: tuple[float, ...]
    hopping_y_eV: tuple[float, ...]
    chemical_potential_eV: tuple[float, ...]
    onsite_hybridization_eV: float
    temperature_K: float
    cell_volume_m3: float
    transverse_bond_length_m: float
    eliashberg_slices: tuple[LocalEliashbergSlice, ...]

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("finite-q bridge requires at least a 3 by 3 periodic lattice")
        band_count = len(self.hopping_x_eV)
        if band_count < 2:
            raise ValueError("multiband bridge requires at least two orbitals")
        if len(self.hopping_y_eV) != band_count or len(self.chemical_potential_eV) != band_count:
            raise ValueError("normal-state operands must cover every orbital")
        normal_values = (
            *self.hopping_x_eV,
            *self.hopping_y_eV,
            *self.chemical_potential_eV,
            self.onsite_hybridization_eV,
        )
        if any(not np.isfinite(value) for value in normal_values):
            raise ValueError("normal-state lattice operands must be finite")
        if not np.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("positive finite temperature required")
        if not np.isfinite(self.cell_volume_m3) or self.cell_volume_m3 <= 0.0:
            raise ValueError("positive finite cell volume required")
        if not np.isfinite(self.transverse_bond_length_m) or self.transverse_bond_length_m <= 0.0:
            raise ValueError("positive finite transverse bond length required")
        if not self.eliashberg_slices:
            raise ValueError("at least one positive Matsubara slice required")
        if any(len(value.z_orbital) != band_count for value in self.eliashberg_slices):
            raise ValueError("every Eliashberg slice must cover every orbital")
        frequencies = tuple(value.omega_eV for value in self.eliashberg_slices)
        if frequencies != tuple(sorted(set(frequencies))):
            raise ValueError("Matsubara frequencies must be strictly increasing")

    @property
    def orbital_count(self) -> int:
        return len(self.hopping_x_eV)

    @property
    def site_count(self) -> int:
        return self.nx * self.ny


@dataclass(frozen=True, slots=True)
class MultibandStrongCouplingCurvature:
    """One finite-q response including its noncompensating checks."""

    mode: int
    q_lattice: float
    total_eV_per_cell: float
    diamagnetic_eV_per_cell: float
    paramagnetic_eV_per_cell: float
    decomposition_residual_eV_per_cell: float
    signed_current_residual_eV_per_cell: float


@dataclass(frozen=True, slots=True)
class MultibandStrongCouplingExtrapolation:
    """Small-q transverse intercept and its SI receiver conversion."""

    intercept_eV_per_cell: float
    slope_eV_per_cell: float
    fit_residual_eV_per_cell: float
    london_kernel_J_per_m3_A2: float
    inverse_penetration_depth_sq_per_m2: float


def _index(spec: MultibandLatticeSpec, x: int, y: int, orbital: int) -> int:
    return ((x % spec.nx) * spec.ny + (y % spec.ny)) * spec.orbital_count + orbital


def _transverse_phase(spec: MultibandLatticeSpec, x: int, amplitude: float, mode: int) -> float:
    return amplitude * cos(2.0 * pi * mode * x / spec.nx)


def _normal_hamiltonian(
    spec: MultibandLatticeSpec,
    *,
    amplitude: float,
    mode: int,
    gauge_chi: RealVector | None = None,
) -> ComplexMatrix:
    size = spec.site_count * spec.orbital_count
    result = np.zeros((size, size), dtype=np.complex128)
    for x in range(spec.nx):
        for y in range(spec.ny):
            for orbital in range(spec.orbital_count):
                here = _index(spec, x, y, orbital)
                result[here, here] = -spec.chemical_potential_eV[orbital]
                for dx, dy, hopping, transverse in (
                    (1, 0, spec.hopping_x_eV[orbital], False),
                    (0, 1, spec.hopping_y_eV[orbital], True),
                ):
                    neighbour = _index(spec, x + dx, y + dy, orbital)
                    phase = _transverse_phase(spec, x, amplitude, mode) if transverse else 0.0
                    if gauge_chi is not None:
                        phase += gauge_chi[neighbour] - gauge_chi[here]
                    value = -hopping * np.exp(1j * phase)
                    result[here, neighbour] += value
                    result[neighbour, here] += value.conjugate()
            for left in range(spec.orbital_count):
                for right in range(left + 1, spec.orbital_count):
                    left_index = _index(spec, x, y, left)
                    right_index = _index(spec, x, y, right)
                    result[left_index, right_index] = spec.onsite_hybridization_eV
                    result[right_index, left_index] = spec.onsite_hybridization_eV
    return result


def _nambu_inverse(
    spec: MultibandLatticeSpec,
    slice_: LocalEliashbergSlice,
    *,
    amplitude: float,
    mode: int,
    gauge_chi: RealVector | None = None,
) -> ComplexMatrix:
    electron = _normal_hamiltonian(
        spec,
        amplitude=amplitude,
        mode=mode,
        gauge_chi=gauge_chi,
    )
    hole = electron.T
    normal_size = electron.shape[0]
    z_diagonal = np.empty(normal_size, dtype=np.float64)
    phi_diagonal = np.empty(normal_size, dtype=np.complex128)
    for x in range(spec.nx):
        for y in range(spec.ny):
            for orbital in range(spec.orbital_count):
                index = _index(spec, x, y, orbital)
                z_diagonal[index] = slice_.z_orbital[orbital]
                pair_phase = 0.0 if gauge_chi is None else -2.0 * gauge_chi[index]
                phi_diagonal[index] = slice_.phi_eV_orbital[orbital] * np.exp(1j * pair_phase)
    frequency = 1j * slice_.omega_eV * np.diag(z_diagonal)
    pairing = np.diag(phi_diagonal)
    return np.block(
        [
            [frequency - electron, -pairing],
            [-pairing.conjugate(), frequency + hole],
        ]
    )


def _grand_potential_eV(
    spec: MultibandLatticeSpec,
    *,
    amplitude: float,
    mode: int,
    gauge_chi: RealVector | None = None,
) -> float:
    """Return the field-dependent fermionic grand potential up to a constant."""

    total_logdet = 0.0
    for slice_ in spec.eliashberg_slices:
        operator = _nambu_inverse(
            spec,
            slice_,
            amplitude=amplitude,
            mode=mode,
            gauge_chi=gauge_chi,
        )
        sign, logabsdet = np.linalg.slogdet(operator)
        if sign == 0 or not np.isfinite(logabsdet):
            raise ValueError("singular or nonfinite Nambu operator")
        total_logdet += float(logabsdet)
    return -KB_EV_PER_K * spec.temperature_K * total_logdet


def gauge_covariance_residual_eV(
    spec: MultibandLatticeSpec,
    *,
    gauge_amplitude: float = 0.17,
) -> float:
    """Compare a zero-field operator with a lattice pure-gauge transform."""

    chi = np.empty(spec.site_count * spec.orbital_count, dtype=np.float64)
    for x in range(spec.nx):
        for y in range(spec.ny):
            value = gauge_amplitude * sin(2.0 * pi * x / spec.nx) * cos(2.0 * pi * y / spec.ny)
            for orbital in range(spec.orbital_count):
                chi[_index(spec, x, y, orbital)] = value
    reference = _grand_potential_eV(spec, amplitude=0.0, mode=1)
    transformed = _grand_potential_eV(
        spec,
        amplitude=0.0,
        mode=1,
        gauge_chi=chi,
    )
    return abs(transformed - reference) / spec.site_count


def _raw_transverse_curvature(
    spec: MultibandLatticeSpec,
    *,
    mode: int,
    step: float = 2.0e-4,
    current_amplitude: float = 2.0e-3,
) -> MultibandStrongCouplingCurvature:
    """Evaluate an unsubtracted finite-q response at a finite Matsubara cutoff."""

    if mode <= 0 or mode >= spec.nx // 2 + 1:
        raise ValueError("mode must be a nonzero resolved transverse wave number")
    if not np.isfinite(step) or step <= 0.0 or step >= 0.01:
        raise ValueError("Peierls finite-difference step is outside the weak-field domain")
    if current_amplitude <= 2.0 * step or current_amplitude >= 0.01:
        raise ValueError("signed-current amplitude must exceed two steps and remain weak")
    omega_zero = _grand_potential_eV(spec, amplitude=0.0, mode=mode)
    omega_plus = _grand_potential_eV(spec, amplitude=step, mode=mode)
    omega_minus = _grand_potential_eV(spec, amplitude=-step, mode=mode)
    total = (omega_plus + omega_minus - 2.0 * omega_zero) / (step * step)

    diamagnetic = 0.0
    paramagnetic = 0.0
    for slice_ in spec.eliashberg_slices:
        operator_zero = _nambu_inverse(spec, slice_, amplitude=0.0, mode=mode)
        operator_plus = _nambu_inverse(spec, slice_, amplitude=step, mode=mode)
        operator_minus = _nambu_inverse(spec, slice_, amplitude=-step, mode=mode)
        first = (operator_plus - operator_minus) / (2.0 * step)
        second = (operator_plus + operator_minus - 2.0 * operator_zero) / (step * step)
        inverse = np.linalg.inv(operator_zero)
        factor = -KB_EV_PER_K * spec.temperature_K
        diamagnetic += factor * float(np.trace(inverse @ second).real)
        paramagnetic -= factor * float(np.trace(inverse @ first @ inverse @ first).real)

    normalization = float(spec.site_count)
    total /= normalization
    diamagnetic /= normalization
    paramagnetic /= normalization

    def current(at: float) -> float:
        high = _grand_potential_eV(spec, amplitude=at + step, mode=mode)
        low = _grand_potential_eV(spec, amplitude=at - step, mode=mode)
        return (high - low) / (2.0 * step * normalization)

    return MultibandStrongCouplingCurvature(
        mode=mode,
        q_lattice=2.0 * pi * mode / spec.nx,
        total_eV_per_cell=total,
        diamagnetic_eV_per_cell=diamagnetic,
        paramagnetic_eV_per_cell=paramagnetic,
        decomposition_residual_eV_per_cell=abs(total - diamagnetic - paramagnetic),
        signed_current_residual_eV_per_cell=abs(
            current(current_amplitude) + current(-current_amplitude)
        ),
    )


def transverse_curvature(
    spec: MultibandLatticeSpec,
    *,
    mode: int,
    step: float = 2.0e-4,
    current_amplitude: float = 2.0e-3,
) -> MultibandStrongCouplingCurvature:
    """Evaluate the normal-referenced gauge-closed finite-q response.

    A finite Matsubara representation leaves a common ultraviolet contact
    term in both paired and normal calculations.  Subtracting the exact
    denominator-matched normal comparator is the conserving regularization:
    it restores normal cancellation without changing the superconducting
    low-frequency contribution or compensating any material gate.
    """

    paired = _raw_transverse_curvature(
        spec,
        mode=mode,
        step=step,
        current_amplitude=current_amplitude,
    )
    normal = _raw_transverse_curvature(
        normal_comparator(spec),
        mode=mode,
        step=step,
        current_amplitude=current_amplitude,
    )
    total = paired.total_eV_per_cell - normal.total_eV_per_cell
    diamagnetic = paired.diamagnetic_eV_per_cell - normal.diamagnetic_eV_per_cell
    paramagnetic = paired.paramagnetic_eV_per_cell - normal.paramagnetic_eV_per_cell
    return MultibandStrongCouplingCurvature(
        mode=mode,
        q_lattice=paired.q_lattice,
        total_eV_per_cell=total,
        diamagnetic_eV_per_cell=diamagnetic,
        paramagnetic_eV_per_cell=paramagnetic,
        decomposition_residual_eV_per_cell=abs(total - diamagnetic - paramagnetic),
        signed_current_residual_eV_per_cell=max(
            paired.signed_current_residual_eV_per_cell,
            normal.signed_current_residual_eV_per_cell,
        ),
    )


def small_q_extrapolation(
    spec: MultibandLatticeSpec,
    *,
    modes: tuple[int, ...] = (1, 2),
    step: float = 2.0e-4,
    current_amplitude: float = 2.0e-3,
) -> tuple[MultibandStrongCouplingExtrapolation, tuple[MultibandStrongCouplingCurvature, ...]]:
    """Fit the transverse response against q^2 and convert its intercept to SI."""

    if len(modes) < 2 or modes != tuple(sorted(set(modes))):
        raise ValueError("small-q extrapolation requires sorted unique modes")
    observations = tuple(
        transverse_curvature(
            spec,
            mode=mode,
            step=step,
            current_amplitude=current_amplitude,
        )
        for mode in modes
    )
    return _fit_small_q(spec, observations)


def raw_small_q_extrapolation(
    spec: MultibandLatticeSpec,
    *,
    modes: tuple[int, ...] = (1, 2),
    step: float = 2.0e-4,
    current_amplitude: float = 2.0e-3,
) -> tuple[MultibandStrongCouplingExtrapolation, tuple[MultibandStrongCouplingCurvature, ...]]:
    """Expose the unsubtracted finite-cutoff response as a diagnostic.

    This is deliberately distinct from the denominator-matched ultraviolet
    subtraction in :func:`small_q_extrapolation`; the regularized normal zero
    is therefore not misrepresented as an independent Ward-identity test.
    """

    if len(modes) < 2 or modes != tuple(sorted(set(modes))):
        raise ValueError("small-q extrapolation requires sorted unique modes")
    observations = tuple(
        _raw_transverse_curvature(
            spec,
            mode=mode,
            step=step,
            current_amplitude=current_amplitude,
        )
        for mode in modes
    )
    return _fit_small_q(spec, observations)


def _fit_small_q(
    spec: MultibandLatticeSpec,
    observations: tuple[MultibandStrongCouplingCurvature, ...],
) -> tuple[MultibandStrongCouplingExtrapolation, tuple[MultibandStrongCouplingCurvature, ...]]:
    x = np.asarray([value.q_lattice**2 for value in observations], dtype=np.float64)
    y = np.asarray([value.total_eV_per_cell for value in observations], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = float(np.max(np.abs(y - fitted)))
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    # All sampled transverse modes are nonzero cosines, whose spatial
    # mean-square is A0^2/2.  Their amplitude curvature is therefore K*V/2.
    energy_density = 2.0 * intercept * EV_TO_J / spec.cell_volume_m3
    gauge_factor = ELEMENTARY_CHARGE_C * spec.transverse_bond_length_m / HBAR_J_S
    london_kernel = energy_density * gauge_factor * gauge_factor
    return (
        MultibandStrongCouplingExtrapolation(
            intercept_eV_per_cell=intercept,
            slope_eV_per_cell=slope,
            fit_residual_eV_per_cell=residual,
            london_kernel_J_per_m3_A2=london_kernel,
            inverse_penetration_depth_sq_per_m2=MU0_H_PER_M * london_kernel,
        ),
        observations,
    )


def normal_comparator(spec: MultibandLatticeSpec) -> MultibandLatticeSpec:
    """Return an exact normal-state comparator without changing the denominator."""

    slices = tuple(
        LocalEliashbergSlice(
            omega_eV=value.omega_eV,
            z_orbital=value.z_orbital,
            phi_eV_orbital=tuple(0.0 for _ in value.phi_eV_orbital),
        )
        for value in spec.eliashberg_slices
    )
    return MultibandLatticeSpec(
        nx=spec.nx,
        ny=spec.ny,
        hopping_x_eV=spec.hopping_x_eV,
        hopping_y_eV=spec.hopping_y_eV,
        chemical_potential_eV=spec.chemical_potential_eV,
        onsite_hybridization_eV=spec.onsite_hybridization_eV,
        temperature_K=spec.temperature_K,
        cell_volume_m3=spec.cell_volume_m3,
        transverse_bond_length_m=spec.transverse_bond_length_m,
        eliashberg_slices=slices,
    )


def qualify_multiband_bridge(*, implementation_sha256: str) -> MultibandStrongCouplingBridgeQualification:
    """Run a two-orbital local-self-energy transverse-response fixture."""

    temperature_K = 20.0

    def fixture(matsubara_count: int) -> MultibandLatticeSpec:
        slices = tuple(
            LocalEliashbergSlice(
                omega_eV=(2 * index + 1) * pi * KB_EV_PER_K * temperature_K,
                z_orbital=(1.8, 1.3),
                phi_eV_orbital=(0.008, 0.003),
            )
            for index in range(matsubara_count)
        )
        return MultibandLatticeSpec(
            nx=10,
            ny=3,
            hopping_x_eV=(1.0, 0.4),
            hopping_y_eV=(0.7, 0.3),
            chemical_potential_eV=(0.0, 0.0),
            onsite_hybridization_eV=0.08,
            temperature_K=temperature_K,
            cell_volume_m3=3.0e-29,
            transverse_bond_length_m=3.0e-10,
            eliashberg_slices=slices,
        )

    base_spec = fixture(32)
    spec = fixture(48)
    paired, observations = small_q_extrapolation(
        spec,
        modes=(1, 2, 3),
        step=2.5e-4,
        current_amplitude=1.0e-3,
    )
    base, _ = small_q_extrapolation(
        base_spec,
        modes=(1, 2, 3),
        step=2.5e-4,
        current_amplitude=1.0e-3,
    )
    step_view, _ = small_q_extrapolation(
        spec,
        modes=(1, 2, 3),
        step=5.0e-4,
        current_amplitude=2.0e-3,
    )
    normal, _normal_observations = small_q_extrapolation(
        normal_comparator(spec),
        modes=(1, 2, 3),
        step=2.5e-4,
        current_amplitude=1.0e-3,
    )
    raw_normal, _ = raw_small_q_extrapolation(
        normal_comparator(spec),
        modes=(1, 2, 3),
        step=2.5e-4,
        current_amplitude=1.0e-3,
    )
    gauge_residual = gauge_covariance_residual_eV(spec)
    signed_residual = max(value.signed_current_residual_eV_per_cell for value in observations)
    decomposition_residual = max(value.decomposition_residual_eV_per_cell for value in observations)
    values = {
        "gauge": Decimal(str(gauge_residual)),
        "signed": Decimal(str(signed_residual)),
        "decomposition": Decimal(str(decomposition_residual)),
        "finite_q": Decimal(str(paired.fit_residual_eV_per_cell)),
        "normal": Decimal(str(abs(normal.intercept_eV_per_cell))),
        "raw_normal": Decimal(str(abs(raw_normal.intercept_eV_per_cell))),
        "paired": Decimal(str(paired.intercept_eV_per_cell)),
        "inverse_lambda_sq": Decimal(str(paired.inverse_penetration_depth_sq_per_m2)),
        "matsubara": Decimal(
            str(
                abs(paired.intercept_eV_per_cell - base.intercept_eV_per_cell)
                / max(abs(paired.intercept_eV_per_cell), 1.0e-30)
            )
        ),
        "step": Decimal(
            str(
                abs(paired.intercept_eV_per_cell - step_view.intercept_eV_per_cell)
                / max(abs(paired.intercept_eV_per_cell), 1.0e-30)
            )
        ),
    }
    passed = (
        values["gauge"] <= Decimal("1e-10")
        and values["signed"] <= Decimal("1e-9")
        and values["decomposition"] <= Decimal("1e-8")
        and values["finite_q"] <= Decimal("0.1") * values["paired"]
        and values["normal"] <= Decimal("1e-12")
        and values["matsubara"] <= Decimal("0.05")
        and values["step"] <= Decimal("0.05")
        and values["paired"] > 0
        and values["inverse_lambda_sq"] > 0
    )
    return MultibandStrongCouplingBridgeQualification(
        qualification_id='qualification.ambient-pressure-superconductor-gauge-covariant-response-multiband-strong-coupling',
        implementation_sha256=implementation_sha256,
        formalism_source_ids=tuple(
            sorted(
                (
                    'source.ambient-pressure-superconductor.gauge-covariant-response-hiorth-2603-10955v1',
                    'source.ambient-pressure-superconductor.gauge-covariant-response-watanabe-2501-13722v2',
                    'source.ambient-pressure-superconductor.epw-tutorial04-page-20260530',
                )
            )
        ),
        equation_ids=tuple(
            sorted(
                (
                    "equation.nambu-logdet-effective-action",
                    "equation.peierls-finite-q-transverse-curvature",
                    "equation.peierls-dia-para-decomposition",
                )
            )
        ),
        compatibility_map_ids=tuple(
            sorted(
                (
                    "compatibility.epw-band-self-energy-to-wannier-orbital-gauge",
                    "compatibility.peierls-curvature-to-london-kernel-si",
                )
            )
        ),
        validity_requirement_ids=tuple(
            sorted(
                (
                    "validity.local-self-energy-in-common-wannier-gauge",
                    "validity.migdal-parameter-proxy-below-predeclared-limit",
                    "validity.static-local-pair-field-transverse-decoupling-condition",
                    "validity.static-transverse-coulomb-gauge-only",
                )
            )
        ),
        orbital_count=spec.orbital_count,
        matsubara_count=len(spec.eliashberg_slices),
        temperature_K=Decimal(str(spec.temperature_K)),
        gauge_covariance_residual_eV_per_cell=values["gauge"],
        signed_current_residual_eV_per_cell=values["signed"],
        decomposition_residual_eV_per_cell=values["decomposition"],
        finite_q_fit_residual_eV_per_cell=values["finite_q"],
        normal_intercept_eV_per_cell=values["normal"],
        raw_normal_intercept_eV_per_cell=values["raw_normal"],
        matsubara_convergence_relative=values["matsubara"],
        finite_difference_convergence_relative=values["step"],
        paired_intercept_eV_per_cell=values["paired"],
        inverse_penetration_depth_sq_per_m2=values["inverse_lambda_sq"],
        disposition=(
            BridgeDisposition.CONDITIONAL_METHOD_PASS if passed else BridgeDisposition.METHOD_FAIL
        ),
        material_promotion_authorized=False,
        limitations=tuple(
            sorted(
                (
                    "limitation.bridge-fixture-is-method-evidence-not-material-evidence",
                    "limitation.material-use-requires-common-wannier-gauge-and-projection-residual",
                    "limitation.no-tc-inference-from-stiffness",
                    "limitation.nonlocal-self-energy-requires-a-follow-up-vertex-contract",
                    "limitation.not-a-general-longitudinal-nonlinear-cfop-vertex",
                    "limitation.normal-zero-is-predeclared-uv-subtraction-not-independent-ward-test",
                )
            )
        ),
    )


__all__ = [
    "LocalEliashbergSlice",
    "MultibandLatticeSpec",
    'MultibandStrongCouplingCurvature',
    'MultibandStrongCouplingExtrapolation',
    "gauge_covariance_residual_eV",
    "normal_comparator",
    "qualify_multiband_bridge",
    "raw_small_q_extrapolation",
    "small_q_extrapolation",
    "transverse_curvature",
]
