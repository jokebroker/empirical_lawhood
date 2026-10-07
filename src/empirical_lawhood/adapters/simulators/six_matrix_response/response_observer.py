"""Causal, restartable Response geometry passive instrument at the native owner.

This observer carries a bounded prefix and pending fixed-origin forecasts;
it never has access to a future path or makes a law/admission verdict.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal

from .contracts import SixMatrixResponseModelFamilyMember
from .model import ComplexArray, SixMatrixState, hermiticity_residual
from .passive_probe import SixMatrixResponseTransientControlledInvarianceProbeRoster, heat_predict_probe, normalized_increment_loss, passive_probe_step, passive_observer_step, radius_only_operator, traceless_operator
from .simulation import fast_receiver
from .shooting import traceless_hermitian_basis
from .spectral import spectral_receiver


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeFactorMetrics(CanonicalRecord):
    """Original native normalized metrics, with undefined values retained."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-factor-metrics'

    phi: Decimal | None
    closure_ratio: Decimal | None
    kernel_ratio: Decimal | None

    def __post_init__(self) -> None:
        for field in ("phi", "closure_ratio", "kernel_ratio"):
            value = getattr(self, field)
            if value is not None:
                validate_decimal(value, field_name=field, minimum=Decimal(0))

    @property
    def known(self) -> bool:
        return (
            self.phi is not None
            and self.closure_ratio is not None
            and self.kernel_ratio is not None
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeGeometrySample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-geometry-sample'

    reference_tick: int
    x: ResponseGeometryNativeFactorMetrics
    y: ResponseGeometryNativeFactorMetrics

    def __post_init__(self) -> None:
        if (
            type(self.reference_tick) is not int
            or self.reference_tick < 0
            or self.reference_tick % 16
        ):
            raise ValueError(
                "Response geometry structural samples require the fixed 16-reference-tick lattice"
            )


def geometry_sample(
    *, state: SixMatrixState, member: SixMatrixResponseModelFamilyMember, refinement: int
) -> ResponseGeometryNativeGeometrySample:
    """Measure the historical native phi/closure/kernel operands at one causal tick.

    This is a new record for the unchanged native normalization, not the historic
    full-path projector. It imports no historical episode, selection or finalizer.
    """
    if type(refinement) is not int or refinement not in {1, 2, 4} or state.q != 2:
        raise ValueError("Response geometry structural sample requires a declared q=2 view")
    if state.step_index % (16 * refinement) or not state.finite:
        raise ValueError("Response geometry structural sample clock/state differs")
    fast = fast_receiver(state=state, member=member)
    spectra = {
        value.sector: value
        for value in spectral_receiver(
            receiver_prefix="response-geometry.geometry", q=2, positions=state.positions
        )
    }
    outputs = []
    for name, alpha, radius, closure in (
        ("X", state.alpha_tilde_x, float(fast.radius_x), float(fast.closure_residual_x)),
        ("Y", state.alpha_tilde_y, float(fast.radius_y), float(fast.closure_residual_y)),
    ):
        denominator = (alpha / 2) ** 2 * 0.75
        phi = (
            float(np.sqrt(max(radius, 0) / denominator))
            if denominator > np.finfo(float).tiny
            else None
        )
        scale = abs(alpha / 3) * np.sqrt(4 * max(radius, 0) / 3)
        closure_ratio = closure / max(float(scale), np.finfo(float).tiny)
        values = spectra[name].eigenvalues
        # Preserve the old conservative zero-spectrum kernel-ratio convention.
        ratio = float(values[3] / values[4]) if float(values[4]) > np.finfo(float).eps else 1.0
        metrics = tuple(
            Decimal(str(value)) if value is not None and np.isfinite(value) else None
            for value in (phi, closure_ratio, ratio)
        )
        outputs.append(ResponseGeometryNativeFactorMetrics(*metrics))
    return ResponseGeometryNativeGeometrySample(state.step_index // refinement, outputs[0], outputs[1])


def _encode(value: npt.NDArray[Any]) -> str:
    return base64.b64encode(value.tobytes(order="C")).decode("ascii")


def _decode(payload: str, shape: tuple[int, ...], *, real: bool = False) -> npt.NDArray[Any]:
    dtype = np.dtype("<f8" if real else "<c16")
    size = int(np.prod(shape)) * dtype.itemsize
    if len(payload) != 4 * ((size + 2) // 3):
        raise ValueError("Response geometry observer array has another bounded byte geometry")
    raw = base64.b64decode(payload, validate=True)
    if len(raw) != size or base64.b64encode(raw).decode("ascii") != payload:
        raise ValueError("Response geometry observer array must use canonical base64")
    value = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(value).all() or (not real and hermiticity_residual(value) > 1e-12):
        raise ValueError("Response geometry observer array must be finite and Hermitian")
    return value


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeProbeForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-probe-forecast'

    origin_native_step: int
    initial_base64: str
    operator_base64: str
    y_base64: str | None = None
    y_velocity_base64: str | None = None

    def __post_init__(self) -> None:
        if type(self.origin_native_step) is not int or self.origin_native_step < 0:
            raise ValueError("Response geometry forecast origin must be a native step")
        _decode(self.initial_base64, (12, 3, 4, 4))
        operator = _decode(self.operator_base64, (15, 15), real=True)
        if np.linalg.norm(operator - operator.T) > 1e-10:
            raise ValueError("Response geometry passive forecast operator must be symmetric")
        if np.linalg.eigvalsh(operator)[0] < -1e-10 * max(1.0, float(np.linalg.norm(operator, 2))):
            raise ValueError("Response geometry passive forecast operator must be nonnegative")
        if (self.y_base64 is None) != (self.y_velocity_base64 is None):
            raise ValueError("Response geometry causal drift requires both origin Y and momentum")
        if self.y_base64 is not None and self.y_velocity_base64 is not None:
            _decode(self.y_base64, (3, 4, 4))
            _decode(self.y_velocity_base64, (3, 4, 4))


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeTransferOrigin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-transfer-origin'

    origin_native_step: int
    fields_base64: str

    def __post_init__(self) -> None:
        if type(self.origin_native_step) is not int or self.origin_native_step < 0:
            raise ValueError("Response geometry transfer origin must be a native step")
        _decode(self.fields_base64, (15, 4, 4))


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeProbeCheckpoint(CanonicalRecord):
    """Native observer continuation only; outer source checkpoint owns identity.

    It must be composed with positions/momenta, RNG/bridge, force occurrence and
    the rolling structural/mode window. This record alone cannot resume a source.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-probe-checkpoint'

    native_step: int
    refinement: int
    probes_base64: str
    y_base64: str
    pending: tuple[ResponseGeometryNativeProbeForecast, ...]
    prefix_hermiticity: Decimal
    prefix_trace: Decimal
    prefix_identity_drift: Decimal
    prefix_norm_increase: Decimal
    transfers: tuple[ResponseGeometryNativeTransferOrigin, ...]

    def __post_init__(self) -> None:
        if type(self.native_step) is not int or self.native_step < 0:
            raise ValueError("Response geometry observer checkpoint requires a completed native step")
        if type(self.refinement) is not int or self.refinement not in {1, 2, 4}:
            raise ValueError("Response geometry observer checkpoint has another numerical view")
        _decode(self.probes_base64, (13, 3, 4, 4))
        _decode(self.y_base64, (3, 4, 4))
        origins = tuple(value.origin_native_step for value in self.pending)
        latest = (self.native_step // (16 * self.refinement)) * 16 * self.refinement
        expected = tuple(step for step in (latest - 16 * self.refinement, latest) if step >= 0)
        if origins != expected:
            raise ValueError("Response geometry observer checkpoint lost or changed pending forecast clocks")
        if tuple(value.origin_native_step for value in self.transfers) != expected:
            raise ValueError("Response geometry observer checkpoint lost or changed pending transfer clocks")
        for field in (
            "prefix_hermiticity",
            "prefix_trace",
            "prefix_identity_drift",
            "prefix_norm_increase",
        ):
            validate_decimal(getattr(self, field), field_name=field, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class PassiveReadout:
    origin_tick: int
    evidence_ready_tick: int
    geometry_loss: float | None
    radius_loss: float | None
    loss_ratio: float | None
    known: bool
    passes: bool | None
    reason: str | None
    propagator_base64: str
    transfer_resolved: bool
    drift_loss: float | None


class ResponseGeometryNativePassiveObserver:
    """Advance in native order, producing one mature readout every 16 ticks."""

    def __init__(
        self,
        *,
        roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
        y: ComplexArray,
        refinement: int,
        y_velocity: ComplexArray | None = None,
    ) -> None:
        if type(refinement) is not int or refinement not in {1, 2, 4}:
            raise ValueError("Response geometry observer numerical refinement differs")
        self.refinement = refinement
        self.native_step = 0
        self.probes = np.empty((13, 3, 4, 4), dtype="<c16")
        self.probes[:12] = roster.fields[:, None]
        self.probes[12] = np.eye(4, dtype="<c16") / 2
        self.y = _decode(_encode(y), (3, 4, 4))
        self.maxima = np.zeros(4, dtype=np.float64)
        self.pending: list[ResponseGeometryNativeProbeForecast] = []
        self.transfers: list[tuple[int, ComplexArray]] = []
        self._launch(y_velocity)

    def _launch(self, y_velocity: ComplexArray | None) -> None:
        self.pending.append(
            ResponseGeometryNativeProbeForecast(
                self.native_step,
                _encode(self.probes[:12]),
                _encode(traceless_operator(self.y)),
                _encode(self.y) if y_velocity is not None else None,
                _encode(y_velocity) if y_velocity is not None else None,
            )
        )
        self.transfers.append((self.native_step, traceless_hermitian_basis(4)))

    def checkpoint(self) -> ResponseGeometryNativeProbeCheckpoint:
        return ResponseGeometryNativeProbeCheckpoint(
            self.native_step,
            self.refinement,
            _encode(self.probes),
            _encode(self.y),
            tuple(self.pending),
            prefix_hermiticity=Decimal(str(float(self.maxima[0]))),
            prefix_trace=Decimal(str(float(self.maxima[1]))),
            prefix_identity_drift=Decimal(str(float(self.maxima[2]))),
            prefix_norm_increase=Decimal(str(float(self.maxima[3]))),
            transfers=tuple(
                ResponseGeometryNativeTransferOrigin(step, _encode(fields)) for step, fields in self.transfers
            ),
        )

    @classmethod
    def restore(cls, checkpoint: ResponseGeometryNativeProbeCheckpoint) -> ResponseGeometryNativePassiveObserver:
        value = cls.__new__(cls)
        value.refinement = checkpoint.refinement
        value.native_step = checkpoint.native_step
        value.probes = _decode(checkpoint.probes_base64, (13, 3, 4, 4))
        value.y = _decode(checkpoint.y_base64, (3, 4, 4))
        value.pending = list(checkpoint.pending)
        value.transfers = [
            (origin.origin_native_step, _decode(origin.fields_base64, (15, 4, 4)))
            for origin in checkpoint.transfers
        ]
        value.maxima = np.array(
            [
                float(checkpoint.prefix_hermiticity),
                float(checkpoint.prefix_trace),
                float(checkpoint.prefix_identity_drift),
                float(checkpoint.prefix_norm_increase),
            ]
        )
        return value

    def advance(
        self, *, y: ComplexArray, native_step: int, y_velocity: ComplexArray | None = None
    ) -> PassiveReadout | None:
        if type(native_step) is not int or native_step != self.native_step + 1:
            raise ValueError("Response geometry passive observer requires consecutive native intervals")
        current, advanced_transfers = passive_observer_step(
            probes=self.probes,
            transfers=tuple(fields for _, fields in self.transfers),
            y_start=self.y,
            y_end=y,
            timestep=0.001 / self.refinement,
        )
        transfers = list(zip((step for step, _ in self.transfers), advanced_transfers, strict=True))
        if not np.isfinite(current).all():
            raise FloatingPointError("Response geometry passive observation became nonfinite")
        before = np.maximum(
            np.linalg.norm(self.probes[:12], axis=(-2, -1)), np.finfo(np.float64).tiny
        )
        growth = (np.linalg.norm(current[:12], axis=(-2, -1)) - before) / before
        errors = np.array(
            [
                hermiticity_residual(current),
                np.max(np.abs(np.trace(current[:12], axis1=-2, axis2=-1))),
                np.max(np.linalg.norm(current[12] - np.eye(4) / 2, axis=(-2, -1))),
                max(0.0, float(np.max(growth))),
            ]
        )
        self.maxima = np.maximum(self.maxima, errors)
        # The primitive already validates Y. Preserve an immutable byte-exact
        # snapshot without base64 serialization on every native interval.
        self.probes, self.y, self.native_step = (
            current,
            np.frombuffer(y.tobytes(), dtype="<c16").reshape(3, 4, 4),
            native_step,
        )
        self.transfers = transfers
        readout = None
        if native_step % (16 * self.refinement) == 0:
            if (
                self.pending
                and native_step - self.pending[0].origin_native_step == 32 * self.refinement
            ):
                readout = self._readout(self.pending.pop(0))
            self._launch(y_velocity)
        return readout

    def _readout(self, forecast: ResponseGeometryNativeProbeForecast) -> PassiveReadout:
        initial = _decode(forecast.initial_base64, (12, 3, 4, 4))
        operator = _decode(forecast.operator_base64, (15, 15), real=True)
        losses: list[list[float]] = [[], []]
        all_resolved = True
        candidates = (operator, radius_only_operator(operator))
        for field in range(12):
            for index, kappa in enumerate((0.25, 0.5, 1.0)):
                for method, candidate in enumerate(candidates):
                    prediction = heat_predict_probe(
                        operator=candidate,
                        probe=initial[field, index],
                        kappa=kappa,
                        horizon_time=0.032,
                    )
                    loss, resolved = normalized_increment_loss(
                        predicted=prediction,
                        observed=self.probes[field, index],
                        initial=initial[field, index],
                        floor=1e-12,
                    )
                    all_resolved &= resolved
                    losses[method].append(loss)
        geometry, radius = (float(np.mean(values)) for values in losses)
        ratio = geometry / radius if radius > 1e-300 else None
        reason = None
        if np.max(self.maxima) > 1e-10:
            reason = "PREFIX_INVARIANT_UNRESOLVED"
        elif not all_resolved or ratio is None or not np.isfinite([geometry, radius, ratio]).all():
            reason = "NORMALIZATION_UNRESOLVED"
        known = reason is None
        drift_loss = None
        if (
            forecast.y_base64 is not None
            and forecast.y_velocity_base64 is not None
            and all_resolved
        ):
            # A failed comparison model is an unknown comparison, not a
            # retroactive failure of the measured native passive response.
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    drift = causal_drift_prediction(
                        y=_decode(forecast.y_base64, (3, 4, 4)),
                        y_velocity=_decode(forecast.y_velocity_base64, (3, 4, 4)),
                        initial=initial,
                        refinement=self.refinement,
                    )
                    drift_losses = [
                        normalized_increment_loss(
                            predicted=drift[field, kappa],
                            observed=self.probes[field, kappa],
                            initial=initial[field, kappa],
                            floor=1e-12,
                        )[0]
                        for field in range(12)
                        for kappa in range(3)
                    ]
                if np.isfinite(drift_losses).all():
                    drift_loss = float(np.mean(drift_losses))
            except FloatingPointError:
                pass
        origin, fields = self.transfers.pop(0)
        if origin != forecast.origin_native_step:
            raise ValueError("Response geometry transfer and forecast origins differ")
        propagator = np.einsum("lab,jab->lj", traceless_hermitian_basis(4).conj(), fields).real
        transfer_resolved = bool(
            np.isfinite(propagator).all() and np.linalg.norm(propagator, 2) <= 1 + 1e-10
        )
        return PassiveReadout(
            forecast.origin_native_step // self.refinement,
            self.native_step // self.refinement,
            geometry if all_resolved else None,
            radius if all_resolved else None,
            ratio if all_resolved else None,
            known,
            bool(geometry <= 0.25 and ratio is not None and ratio <= 0.75) if known else None,
            reason,
            _encode(propagator),
            transfer_resolved,
            drift_loss,
        )


def causal_drift_prediction(
    *, y: ComplexArray, y_velocity: ComplexArray, initial: ComplexArray, refinement: int
) -> ComplexArray:
    """Predict from Y+s*Pi_Y at the origin only; no future path is an input."""
    if type(refinement) is not int or refinement not in {1, 2, 4}:
        raise ValueError("Response geometry causal drift refinement differs")
    origin = _decode(_encode(y), (3, 4, 4))
    velocity = _decode(_encode(y_velocity), (3, 4, 4))
    fields = np.empty((13, 3, 4, 4), dtype="<c16")
    fields[:12] = _decode(_encode(initial), (12, 3, 4, 4))
    fields[12] = np.eye(4, dtype="<c16") / 2
    dt = 0.001 / refinement
    for step in range(32 * refinement):
        fields = passive_probe_step(
            probes=fields,
            y_start=np.asarray(origin + step * dt * velocity, dtype=np.complex128),
            y_end=np.asarray(origin + (step + 1) * dt * velocity, dtype=np.complex128),
            timestep=dt,
        )
    return fields[:12]
