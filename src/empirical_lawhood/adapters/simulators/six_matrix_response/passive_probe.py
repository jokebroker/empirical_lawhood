"""Nonbackreacting passive matrix probes for the bounded Matrix transient response assay."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from .model import ComplexArray, hermitian_part, hermiticity_residual
from .operator_shuffle_scientific_inputs import MatrixResponseOperatorShuffleScientificInputs, require_operator_shuffle_scientific_inputs
from .shooting import traceless_hermitian_basis
from .spectral import adjoint_laplacian_basis

RealArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceProbeRoster:
    """In-memory seed-derived field roster; the config owns its identity."""

    field_ids: tuple[str, ...]
    fields: ComplexArray
    coordinates: RealArray
    development_field_count: int
    seed_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.fields)
        coordinates = np.asarray(self.coordinates)
        if (
            len(self.field_ids) != 12
            or values.shape != (12, 4, 4)
            or values.dtype != np.dtype("complex128")
            or coordinates.shape != (12, 15)
            or coordinates.dtype != np.dtype("float64")
            or self.development_field_count != 6
        ):
            raise ValueError("Matrix transient response probe roster geometry differs")
        if tuple(sorted(self.field_ids)) != self.field_ids or len(set(self.field_ids)) != 12:
            raise ValueError("Matrix transient response probe field IDs differ")
        if len(self.seed_sha256) != 64:
            raise ValueError("Matrix transient response probe seed identity differs")
        int(self.seed_sha256, 16)
        if not np.isfinite(values).all() or not np.isfinite(coordinates).all():
            raise ValueError("Matrix transient response probe roster is nonfinite")
        if hermiticity_residual(values) > 1e-12:
            raise ValueError("Matrix transient response probe roster is not Hermitian")
        traces = np.trace(values, axis1=-2, axis2=-1)
        if float(np.max(np.abs(traces))) > 1e-12:
            raise ValueError("Matrix transient response probe roster is not traceless")
        gram = coordinates @ coordinates.T
        if float(np.max(np.abs(gram - np.eye(12)))) > 1e-12:
            raise ValueError("Matrix transient response probe roster is not orthonormal")
        basis = traceless_hermitian_basis(4)
        reconstructed = np.asarray(np.tensordot(coordinates, basis, axes=([-1], [0])), dtype="<c16")
        if not np.allclose(values, reconstructed, rtol=0.0, atol=1e-12):
            raise ValueError("Matrix transient response probe fields do not match their frozen coordinates")
        copied_fields = np.ascontiguousarray(values, dtype="<c16")
        copied_fields.setflags(write=False)
        copied_coordinates = np.ascontiguousarray(coordinates, dtype="<f8")
        copied_coordinates.setflags(write=False)
        object.__setattr__(self, "fields", copied_fields)
        object.__setattr__(self, "coordinates", copied_coordinates)


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceProbePropagation:
    """One complete nonbackreacting propagation over an explicit Y path."""

    start_step: int
    timestep: float
    kappas: tuple[float, ...]
    states: ComplexArray
    identity_states: ComplexArray
    maximum_hermiticity_residual: float
    maximum_trace_residual: float
    maximum_relative_norm_increase: float
    maximum_identity_relative_drift: float

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        identities = np.asarray(self.identity_states)
        if self.start_step < 0 or self.timestep <= 0 or self.kappas != (0.25, 0.5, 1.0):
            raise ValueError("Matrix transient response propagation timing/kappa identity differs")
        if (
            states.ndim != 5
            or not 1 <= states.shape[0] <= 12
            or states.shape[1] != 3
            or states.shape[-2:] != (4, 4)
            or identities.shape != (3, states.shape[2], 4, 4)
            or states.dtype != np.dtype("complex128")
            or identities.dtype != np.dtype("complex128")
        ):
            raise ValueError("Matrix transient response propagated probe geometry differs")
        metrics = (
            self.maximum_hermiticity_residual,
            self.maximum_trace_residual,
            self.maximum_relative_norm_increase,
            self.maximum_identity_relative_drift,
        )
        if not np.isfinite(states).all() or not np.isfinite(identities).all():
            raise ValueError("Matrix transient response propagated probes are nonfinite")
        if not np.isfinite(metrics).all() or min(metrics) < 0:
            raise ValueError("Matrix transient response propagation diagnostics differ")
        copied_states = np.ascontiguousarray(states, dtype="<c16")
        copied_states.setflags(write=False)
        copied_identities = np.ascontiguousarray(identities, dtype="<c16")
        copied_identities.setflags(write=False)
        object.__setattr__(self, "states", copied_states)
        object.__setattr__(self, "identity_states", copied_identities)

    @property
    def end_step(self) -> int:
        return self.start_step + int(self.states.shape[2]) - 1


def probe_roster_seed_sha256(
    *, config_fingerprint: str, rule_id: str, scientific_seed: int
) -> str:
    """Identify the probe RNG allocation without drawing the fields."""

    if len(config_fingerprint) != 64:
        raise ValueError("Matrix transient response config fingerprint must be SHA-256")
    int(config_fingerprint, 16)
    if type(scientific_seed) is not int or not 0 <= scientific_seed < 2**256:
        raise ValueError("passive probe scientific seed must be a 256-bit unsigned integer")
    return scientific_seed.to_bytes(32, "big").hex()


def derive_probe_roster(
    *, config_fingerprint: str, rule_id: str, scientific_seed: int,
    field_id_prefix: str = "matrix-transient-response.probe.field",
) -> SixMatrixResponseTransientControlledInvarianceProbeRoster:
    """Derive the twelve fields before any event path is read."""

    digest = bytes.fromhex(
        probe_roster_seed_sha256(
            config_fingerprint=config_fingerprint,
            rule_id=rule_id,
            scientific_seed=scientific_seed,
        )
    )
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    draw = np.asarray(rng.standard_normal((15, 12)), dtype=np.float64)
    orthonormal, triangular = np.linalg.qr(draw, mode="reduced")
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    coordinates = np.asarray((orthonormal * signs).T, dtype=np.float64)
    basis = traceless_hermitian_basis(4)
    fields = np.asarray(np.tensordot(coordinates, basis, axes=([-1], [0])), dtype="<c16")
    return SixMatrixResponseTransientControlledInvarianceProbeRoster(
        field_ids=tuple(f"{field_id_prefix}-{index:02d}" for index in range(12)),
        fields=fields,
        coordinates=coordinates,
        development_field_count=6,
        seed_sha256=digest.hex(),
    )


def direct_double_commutator(*, triplet: ComplexArray, probe: ComplexArray) -> ComplexArray:
    """Evaluate sum_a [Y_a,[Y_a,probe]] without the spectral builder."""

    matrices = np.asarray(triplet)
    value = np.asarray(probe)
    if (
        matrices.shape != (3, 4, 4)
        or value.shape != (4, 4)
        or matrices.dtype != np.dtype("complex128")
        or value.dtype != np.dtype("complex128")
        or not np.isfinite(matrices).all()
        or not np.isfinite(value).all()
    ):
        raise ValueError("Matrix transient response direct commutator inputs differ")
    output = np.zeros((4, 4), dtype="<c16")
    for matrix in matrices:
        first = matrix @ value - value @ matrix
        output += matrix @ first - first @ matrix
    return np.ascontiguousarray(output, dtype="<c16")


def _rk4_probe_step(
    *,
    probe: ComplexArray,
    y_start: ComplexArray,
    y_end: ComplexArray,
    kappa: float,
    timestep: float,
) -> ComplexArray:
    midpoint = np.asarray((y_start + y_end) * 0.5, dtype="<c16")

    def drift(triplet: ComplexArray, value: ComplexArray) -> ComplexArray:
        return np.asarray(-kappa * direct_double_commutator(triplet=triplet, probe=value))

    k1 = drift(y_start, probe)
    k2 = drift(midpoint, np.asarray(probe + 0.5 * timestep * k1, dtype="<c16"))
    k3 = drift(midpoint, np.asarray(probe + 0.5 * timestep * k2, dtype="<c16"))
    k4 = drift(y_end, np.asarray(probe + timestep * k3, dtype="<c16"))
    return hermitian_part(probe + (timestep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def propagate_passive_probes(
    *,
    y_path: ComplexArray,
    start_step: int,
    timestep: float,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
    kappas: tuple[float, ...] = (0.25, 0.5, 1.0),
    field_indices: tuple[int, ...] = tuple(range(12)),
) -> SixMatrixResponseTransientControlledInvarianceProbePropagation:
    """Propagate the frozen fields without mutating or feeding back into the path."""

    path = np.asarray(y_path)
    if (
        path.ndim != 4
        or path.shape[1:] != (3, 4, 4)
        or path.shape[0] < 2
        or path.dtype != np.dtype("complex128")
        or start_step < 0
        or timestep <= 0
        or kappas != (0.25, 0.5, 1.0)
        or not field_indices
        or field_indices != tuple(sorted(set(field_indices)))
        or min(field_indices) < 0
        or max(field_indices) >= 12
        or not np.isfinite(path).all()
        or hermiticity_residual(path) > 1e-12
    ):
        raise ValueError("Matrix transient response passive-probe path differs")
    state_count = path.shape[0]
    fields = roster.fields[np.asarray(field_indices)]
    states = np.empty((len(field_indices), 3, state_count, 4, 4), dtype="<c16")
    states[:, :, 0] = fields[:, None]
    identity = np.eye(4, dtype="<c16") / 2.0
    identities = np.empty((3, state_count, 4, 4), dtype="<c16")
    identities[:, 0] = identity
    maximum_norm_increase = 0.0
    for offset in range(1, state_count):
        y_start = path[offset - 1]
        y_end = path[offset]
        previous_batch = np.concatenate(
            (states[:, :, offset - 1], identities[None, :, offset - 1]), axis=0
        )
        current_batch = _rk4_probe_batch_step(
            probes=previous_batch,
            y_start=y_start,
            y_end=y_end,
            kappas=np.asarray(kappas, dtype=np.float64)[None, :, None, None],
            timestep=timestep,
        )
        states[:, :, offset] = current_batch[:-1]
        identities[:, offset] = current_batch[-1]
        for field_index in range(len(field_indices)):
            for kappa_index in range(len(kappas)):
                previous = states[field_index, kappa_index, offset - 1]
                current = states[field_index, kappa_index, offset]
                before = max(float(np.linalg.norm(previous)), np.finfo(np.float64).tiny)
                maximum_norm_increase = max(
                    maximum_norm_increase,
                    (float(np.linalg.norm(current)) - before) / before,
                )
    traces = np.trace(states, axis1=-2, axis2=-1)
    identity_drift = np.linalg.norm(identities - identity, axis=(-2, -1)) / np.linalg.norm(identity)
    return SixMatrixResponseTransientControlledInvarianceProbePropagation(
        start_step=start_step,
        timestep=timestep,
        kappas=kappas,
        states=states,
        identity_states=identities,
        maximum_hermiticity_residual=hermiticity_residual(states),
        maximum_trace_residual=float(np.max(np.abs(traces))),
        maximum_relative_norm_increase=max(0.0, maximum_norm_increase),
        maximum_identity_relative_drift=float(np.max(identity_drift)),
    )


def _rk4_probe_batch_step(
    *,
    probes: ComplexArray,
    y_start: ComplexArray,
    y_end: ComplexArray,
    kappas: RealArray,
    timestep: float,
) -> ComplexArray:
    """Apply the scalar operation order to a validated field/kappa stack."""

    midpoint = np.asarray((y_start + y_end) * 0.5, dtype="<c16")

    def drift(triplet: ComplexArray, values: ComplexArray) -> ComplexArray:
        output = np.zeros_like(values)
        for matrix in triplet:
            first = matrix @ values - values @ matrix
            output += matrix @ first - first @ matrix
        return np.asarray(-kappas * output, dtype="<c16")

    k1 = drift(y_start, probes)
    k2 = drift(midpoint, np.asarray(probes + 0.5 * timestep * k1, dtype="<c16"))
    k3 = drift(midpoint, np.asarray(probes + 0.5 * timestep * k2, dtype="<c16"))
    k4 = drift(y_end, np.asarray(probes + timestep * k3, dtype="<c16"))
    return hermitian_part(probes + (timestep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def passive_probe_step(
    *, probes: ComplexArray, y_start: ComplexArray, y_end: ComplexArray, timestep: float
) -> ComplexArray:
    """Stream one native interval using the existing ordered passive RK4.

    The bounded stack contains the twelve fields and identity, each at all three
    kappas. It supplies no whole-path diagnostics or scientific classification.
    """
    for name, value, shape in (
        ("probes", probes, (13, 3, 4, 4)),
        ("y_start", y_start, (3, 4, 4)),
        ("y_end", y_end, (3, 4, 4)),
    ):
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype("complex128"):
            raise ValueError(f"streamed passive {name} has another geometry")
        if not np.isfinite(array).all() or hermiticity_residual(array) > 1e-12:
            raise ValueError(f"streamed passive {name} must be finite and Hermitian")
    if timestep not in {0.001, 0.0005, 0.00025}:
        raise ValueError("streamed passive timestep is outside the qualified views")
    return _rk4_probe_batch_step(
        probes=probes,
        y_start=y_start,
        y_end=y_end,
        kappas=np.array([0.25, 0.5, 1.0])[None, :, None, None],
        timestep=timestep,
    )


def passive_transfer_step(
    *, fields: ComplexArray, y_start: ComplexArray, y_end: ComplexArray, timestep: float
) -> ComplexArray:
    """Propagate a complete traceless HS basis at kappa=1 for a 32-tick map."""
    for name, value, shape in (
        ("fields", fields, (15, 4, 4)),
        ("y_start", y_start, (3, 4, 4)),
        ("y_end", y_end, (3, 4, 4)),
    ):
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype("complex128"):
            raise ValueError(f"passive transfer {name} has another geometry")
        if not np.isfinite(array).all() or hermiticity_residual(array) > 1e-12:
            raise ValueError(f"passive transfer {name} must be finite and Hermitian")
    if timestep not in {0.001, 0.0005, 0.00025}:
        raise ValueError("passive transfer timestep is outside the qualified views")
    return _rk4_probe_batch_step(
        probes=fields[:, None],
        y_start=y_start,
        y_end=y_end,
        kappas=np.ones((1, 1, 1, 1)),
        timestep=timestep,
    )[:, 0]


def passive_observer_step(
    *,
    probes: ComplexArray,
    transfers: tuple[ComplexArray, ...],
    y_start: ComplexArray,
    y_end: ComplexArray,
    timestep: float,
) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
    """Advance assay's probe and one/two transfer cohorts in one ordered RK4 batch.

    Each field retains the scalar operation order and its own kappa. Sharing
    the batch shares no scientific state, origin or numerical approximation.
    The separate probe/transfer primitives remain independent reference paths.
    """
    if len(transfers) not in {1, 2}:
        raise ValueError("passive observer must retain one or two live transfer origins")
    for name, value, shape in (
        ("probes", probes, (13, 3, 4, 4)),
        ("y_start", y_start, (3, 4, 4)),
        ("y_end", y_end, (3, 4, 4)),
        *(("transfer", value, (15, 4, 4)) for value in transfers),
    ):
        if value.shape != shape or value.dtype != np.dtype("complex128"):
            raise ValueError(f"passive observer {name} has another geometry")
        if not np.isfinite(value).all() or hermiticity_residual(value) > 1e-12:
            raise ValueError(f"passive observer {name} must be finite and Hermitian")
    if timestep not in {0.001, 0.0005, 0.00025}:
        raise ValueError("passive observer timestep is outside the qualified views")
    fields = np.concatenate((probes.reshape(39, 4, 4), *transfers))
    kappas = np.concatenate((np.tile([0.25, 0.5, 1.0], 13), np.ones(15 * len(transfers))))
    current = _rk4_probe_batch_step(
        probes=fields,
        y_start=y_start,
        y_end=y_end,
        kappas=kappas[:, None, None],
        timestep=timestep,
    )
    return current[:39].reshape(13, 3, 4, 4), tuple(
        current[39 + 15 * index : 54 + 15 * index] for index in range(len(transfers))
    )


def traceless_operator(triplet: ComplexArray) -> RealArray:
    """Represent the installed adjoint Laplacian on the real traceless HS basis."""

    matrices = np.asarray(triplet)
    if matrices.shape != (3, 4, 4) or matrices.dtype != np.dtype("complex128"):
        raise ValueError("Matrix transient response traceless operator triplet differs")
    ambient = adjoint_laplacian_basis(matrices)
    basis = traceless_hermitian_basis(4)
    columns = np.stack(tuple(value.reshape(16, order="F") for value in basis), axis=1)
    reduced = columns.conj().T @ ambient @ columns
    imaginary = float(np.max(np.abs(reduced.imag)))
    if imaginary > 1e-10:
        raise FloatingPointError("Matrix transient response traceless operator is materially complex")
    output = np.asarray((reduced.real + reduced.real.T) * 0.5, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(output)
    if float(np.min(eigenvalues)) < -1e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise FloatingPointError("Matrix transient response traceless operator is materially nonpositive")
    return output


def encode_traceless_probe(probe: ComplexArray) -> RealArray:
    value = np.asarray(probe)
    if value.shape != (4, 4) or value.dtype != np.dtype("complex128"):
        raise ValueError("Matrix transient response probe encoding geometry differs")
    basis = traceless_hermitian_basis(4)
    return np.asarray([float(np.vdot(item, value).real) for item in basis], dtype=np.float64)


def decode_traceless_probe(coordinates: RealArray) -> ComplexArray:
    values = np.asarray(coordinates)
    if (
        values.shape != (15,)
        or values.dtype != np.dtype("float64")
        or not np.isfinite(values).all()
    ):
        raise ValueError("Matrix transient response probe decoding geometry differs")
    return np.asarray(np.tensordot(values, traceless_hermitian_basis(4), axes=([0], [0])))


def heat_predict_probe(
    *, operator: RealArray, probe: ComplexArray, kappa: float, horizon_time: float
) -> ComplexArray:
    generator = np.asarray(operator)
    if (
        generator.shape != (15, 15)
        or generator.dtype != np.dtype("float64")
        or not np.isfinite(generator).all()
        or kappa <= 0
        or horizon_time <= 0
    ):
        raise ValueError("Matrix transient response heat-prediction inputs differ")
    heat_map = _heat_map(generator.tobytes(), kappa, horizon_time)
    predicted = heat_map @ encode_traceless_probe(probe)
    return decode_traceless_probe(np.asarray(predicted, dtype=np.float64))


@lru_cache(maxsize=128)
def _heat_map(operator_bytes: bytes, kappa: float, horizon_time: float) -> RealArray:
    """Share pure matrix arithmetic by exact inputs, with bounded immutable storage."""

    operator = np.frombuffer(operator_bytes, dtype=np.float64).reshape(15, 15)
    value = np.asarray(expm(-kappa * horizon_time * operator), dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(15, 15)


def normalized_increment_loss(
    *, predicted: ComplexArray, observed: ComplexArray, initial: ComplexArray, floor: float
) -> tuple[float, bool]:
    if floor <= 0:
        raise ValueError("Matrix transient response numerical floor must be positive")
    change_squared = float(np.linalg.norm(observed - initial) ** 2)
    resolved = change_squared >= floor**2
    numerator = float(np.linalg.norm(predicted - observed) ** 2)
    return numerator / (change_squared + floor**2), resolved


def radius_only_operator(operator: RealArray) -> RealArray:
    value = np.asarray(operator)
    if value.shape != (15, 15) or value.dtype != np.dtype("float64"):
        raise ValueError("Matrix transient response radius operator geometry differs")
    return np.eye(15, dtype=np.float64) * (float(np.trace(value)) / 15.0)


def derive_shuffled_operators(
    *, operator: RealArray, config_fingerprint: str, trajectory_id: str, origin_step: int,
    scientific_inputs: MatrixResponseOperatorShuffleScientificInputs,
) -> tuple[RealArray, ...]:
    inputs = require_operator_shuffle_scientific_inputs(
        scientific_inputs, current_config_sha256=config_fingerprint,
        current_trajectory_id=trajectory_id,
    )
    seeds = inputs.seeds_for_origin(origin_step)
    value = np.asarray(operator)
    if value.shape != (15, 15) or value.dtype != np.dtype("float64") or origin_step < 0:
        raise ValueError("Matrix transient response shuffled-operator inputs differ")
    outputs: list[RealArray] = []
    for index in range(32):
        digest = bytes.fromhex(seeds[index])
        rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
        draw = np.asarray(rng.standard_normal((15, 15)), dtype=np.float64)
        orthogonal, triangular = np.linalg.qr(draw)
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        orthogonal = orthogonal * signs
        shuffled = np.asarray(orthogonal.T @ value @ orthogonal, dtype=np.float64)
        outputs.append(shuffled)
    return tuple(outputs)


def hs_overlap(left: ComplexArray, right: ComplexArray) -> float:
    denominator = max(
        float(np.linalg.norm(left)) * float(np.linalg.norm(right)),
        np.finfo(np.float64).tiny,
    )
    return float(abs(np.vdot(left, right)) / denominator)


def normalized_probe_difference(left: ComplexArray, right: ComplexArray) -> float:
    denominator = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-15)
    return float(np.linalg.norm(left - right) / denominator)


__all__ = [
    'SixMatrixResponseTransientControlledInvarianceProbePropagation',
    'SixMatrixResponseTransientControlledInvarianceProbeRoster',
    'decode_traceless_probe',
    'derive_probe_roster',
    'derive_shuffled_operators',
    'direct_double_commutator',
    'encode_traceless_probe',
    'heat_predict_probe',
    'hs_overlap',
    'normalized_increment_loss',
    'normalized_probe_difference',
    'passive_probe_step',
    'passive_transfer_step',
    'propagate_passive_probes',
    'radius_only_operator',
    'traceless_operator',
]
