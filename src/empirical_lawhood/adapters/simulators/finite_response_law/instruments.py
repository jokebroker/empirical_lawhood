"""The frozen baseline-reference sketch and its strictly backward rate.

Saved native tensors end here. This instrument does not acquire trajectories,
fit a model, infer a phase variable or alter the instantaneous native state.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame, prepared_observation_history, prepared_receiver
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, hermiticity_residual
from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import _hessian_image

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawReferenceInstrument(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-reference-instrument'
    instrument_id: str = "finite-response-law.baseline-x-force-hessian-backward-rate"
    q: int = 2
    mass_x: Decimal = Decimal("0.5")
    mass_y: Decimal = Decimal("0.5")
    gamma: Decimal = Decimal(1)
    alpha_tilde_x_ratio: tuple[int, int] = (2, 3)
    alpha_tilde_y_ratio: tuple[int, int] = (22, 3)
    snapshot_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 9, 10)
    reference_tick_duration: Decimal = Decimal("0.001")
    backward_ticks: int = 32
    feature_ids: tuple[str, ...] = (
        "x-port-1",
        "momentum-port-1",
        "x-squared-hs-norm",
        "y-squared-hs-norm",
        "momentum-x-squared-hs-norm",
        "momentum-y-squared-hs-norm",
        "x-port-2",
        "momentum-port-2",
        "reference-force-port-1",
        "reference-force-port-2",
        "reference-hessian-11",
        "reference-hessian-12",
        "reference-hessian-21",
        "reference-hessian-22",
        "reference-hessian-full-residual-1",
        "reference-hessian-full-residual-2",
        "backward-rate-force-1",
        "backward-rate-force-2",
        "backward-rate-hessian-11",
        "backward-rate-hessian-12",
        "backward-rate-hessian-21",
        "backward-rate-hessian-22",
        "backward-rate-full-residual-1",
        "backward-rate-full-residual-2",
    )
    feature_units: tuple[str, ...] = (
        "native-hs-position",
        "native-hs-momentum",
        "native-hs-position-squared",
        "native-hs-position-squared",
        "native-hs-momentum-squared",
        "native-hs-momentum-squared",
        "native-hs-position",
        "native-hs-momentum",
        "native-hs-force",
        "native-hs-force",
        *("native-hs-force-per-position",) * 6,
        "native-hs-force-per-native-time",
        "native-hs-force-per-native-time",
        *("native-hs-force-per-position-per-native-time",) * 6,
    )

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                type(getattr(self, name)) is not type(field.default)
                or getattr(self, name) != field.default
            ):
                raise ValueError(f'Finite response-law reference instrument differs from frozen recipe: {name}')
        if any(type(v) is not int for v in self.snapshot_channels):
            raise ValueError("Finite response-law snapshot channel identities must be integers")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.instrument_id, self)

    @property
    def parameters(self) -> SixMatrixParameters:
        return SixMatrixParameters(
            self.q,
            float(self.mass_x),
            float(self.mass_y),
            float(self.gamma),
            self.alpha_tilde_x_ratio[0] / self.alpha_tilde_x_ratio[1],
            self.alpha_tilde_y_ratio[0] / self.alpha_tilde_y_ratio[1],
        )


REFERENCE_INSTRUMENT = FiniteResponseLawReferenceInstrument()


def reference_sketch(positions: ComplexArray, frame: PreparedPortFrame) -> Array:
    """Evaluate the fixed reference operator at actual saved positions.

    The existing HVP primitive has these same fixed baseline parameters. Its
    directions occupy X only; residual norms retain both native sectors.
    """
    if (
        positions.shape != (2, 3, 4, 4)
        or positions.dtype != np.dtype("complex128")
        or not np.isfinite(positions).all()
        or hermiticity_residual(positions) > 1e-12
    ):
        raise ValueError("Finite response-law reference sketch requires finite Hermitian q=2 saved positions")
    gradient = analytic_gradient_terms(positions, REFERENCE_INSTRUMENT.parameters).total
    force = -prepared_receiver(frame, gradient[0])
    basis = np.zeros((2, 2, 3, 4, 4), dtype=np.complex128)
    basis[:, 0] = frame.modes
    images = np.stack([_hessian_image(positions, direction) for direction in basis])
    projected = np.einsum("iabc,jabc->ij", frame.modes.conj(), images[:, 0]).real
    residual = images - np.einsum("ij,iabcd->jabcd", projected, basis)
    result = np.r_[force, projected.ravel(), np.linalg.norm(residual.reshape(2, -1), axis=1)]
    if not np.isfinite(result).all():
        raise ValueError("Finite response-law reference operator is unresolved")
    return np.frombuffer(result.tobytes(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCompactInterface:
    instrument: ObjectIdentity
    frame_cutoff_tick: int
    cutoff_tick: int
    backward_tick: int
    values: Array

    def __post_init__(self) -> None:
        if (
            self.instrument != REFERENCE_INSTRUMENT.identity
            or type(self.frame_cutoff_tick) is not int
            or self.frame_cutoff_tick != 4096
            or type(self.cutoff_tick) is not int
            or self.cutoff_tick not in (4096, 4368)
            or type(self.backward_tick) is not int
            or self.backward_tick != self.cutoff_tick - 32
            or self.values.shape != (24,)
            or self.values.dtype != np.dtype("float64")
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("Finite response-law compact interface changes its recipe, causal cutoffs or geometry")
        object.__setattr__(self, "values", np.frombuffer(self.values.tobytes(), dtype=np.float64))

    def features(self, family: str) -> Array:
        dimensions = {"SNAPSHOT": 8, "SNAPSHOT_AND_REFERENCE_SKETCH": 16, "SNAPSHOT_REFERENCE_SKETCH_AND_RATE": 24}
        if family not in dimensions:
            raise ValueError("Finite response-law feature family is outside the frozen three-family grid")
        return self.values[: dimensions[family]]


def compact_interface(
    *,
    frame: PreparedPortFrame,
    ticks: tuple[int, ...],
    positions: ComplexArray,
    momenta: ComplexArray,
    cutoff_tick: int,
) -> FiniteResponseLawCompactInterface:
    """Reduce the authenticated 31-sample prefix/handoff causal window.

    The caller authenticates source and numerical-view lineage before this pure
    reduction. The instrument requires the actual current and t-32 samples; it
    neither fills missing states nor accepts observations after the cutoff.
    """
    if frame.cutoff_tick != 4096 or cutoff_tick not in (4096, 4368):
        raise ValueError("Finite response-law Tier 1 instrument requires its prepared prefix or +272 handoff")
    history = prepared_observation_history(
        frame=frame, ticks=ticks, positions=positions, momenta=momenta, cutoff_tick=cutoff_tick
    )
    backward_tick = cutoff_tick - REFERENCE_INSTRUMENT.backward_ticks
    backward_index = ticks.index(backward_tick)
    current = reference_sketch(positions[-1], frame)
    previous = reference_sketch(positions[backward_index], frame)
    elapsed = float(
        REFERENCE_INSTRUMENT.reference_tick_duration * REFERENCE_INSTRUMENT.backward_ticks
    )
    values = np.r_[
        history[-1, list(REFERENCE_INSTRUMENT.snapshot_channels)],
        current,
        (current - previous) / elapsed,
    ]
    return FiniteResponseLawCompactInterface(
        REFERENCE_INSTRUMENT.identity, frame.cutoff_tick, cutoff_tick, backward_tick, values
    )
