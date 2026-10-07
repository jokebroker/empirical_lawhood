"""Bounded current-root operands for the fixed exploratory radial bridge.

This is the minimum P08 kernel/outer-fold producer used by readiness. It does
not implement the full transfer report, inner uncertainty report, calibration,
qualification, or native acquisition. The caller authenticates the explicit
current array artifacts and allocation before calling these pure functions.
"""

from dataclasses import dataclass, field, fields
from decimal import Decimal
from types import MappingProxyType
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from .fitting import Array, Normalizer
from .original_f import OriginalFiniteResponseLaw

RADIAL_STIFFNESS = (88 / 9) ** 2
TRAINING_SCHEDULES = tuple(range(1, 9))
TRAINING_TIMES = tuple(range(2, 26))


@dataclass(frozen=True, slots=True)
class TransientBridgeSpec(CanonicalRecord):
    """Closed numerical recipe; root/seed allocations are separate operands."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/transient-bridge-spec"
    root_cohorts: tuple[tuple[str, int], ...] = (("q2", 8), ("cir1", 16))
    channels: tuple[str, ...] = ("displacement", "velocity", "backward32-displacement", "backward32-velocity")
    ticks: tuple[int, ...] = tuple(range(0, 401, 16))
    timestep: Decimal = Decimal("0.001")
    friction: Decimal = Decimal(1)
    stiffness_ratio: tuple[int, int] = (88, 9)
    amplitude: Decimal = Decimal("0.125")
    backward_seconds: Decimal = Decimal("0.032")
    kernel_rms_floor: Decimal = Decimal("0.000000000001")
    modal_rcond: Decimal = Decimal("0.0000000001")
    ridge: Decimal = Decimal(10)
    normalizer_rms_floor: Decimal = Decimal("0.00000001")
    schedules: tuple[int, ...] = TRAINING_SCHEDULES
    times: tuple[int, ...] = TRAINING_TIMES

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) != item.default for item in fields(self)):
            raise ValueError("Transient bridge changes its closed exploratory recipe")


@dataclass(frozen=True, slots=True)
class TransientBridgeInput(CanonicalRecord):
    """Current array and allocation lineage, independent of original F provenance."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/transient-bridge-input"
    operand_id: str
    allocation: ObjectIdentity
    original_f: ObjectIdentity
    trajectory: ArtifactIdentity
    native_panel: ArtifactIdentity
    root_ids: tuple[str, ...]
    root_cohorts: tuple[str, ...]
    spec: TransientBridgeSpec = field(default=TransientBridgeSpec(), kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        if (
            self.original_f.object_schema != OriginalFiniteResponseLaw.SCHEMA
            or self.original_f.object_version != OriginalFiniteResponseLaw.VERSION
            or len(self.root_ids) != 24
            or len(set(self.root_ids)) != 24
            or self.root_cohorts != ("q2",) * 8 + ("cir1",) * 16
            or self.trajectory.artifact_id == self.native_panel.artifact_id
            or self.spec != TransientBridgeSpec()
        ):
            raise ValueError("Transient bridge current input lineage or ordered root roles differ")
        for root_id in self.root_ids:
            validate_stable_id(root_id, field_name="root_id")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.operand_id, self)


def _array_input(value: Array, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"UNEVALUABLE: transient bridge {name} axes, dtype or finite population differ")


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    copied = np.array(value, copy=True)
    copied.flags.writeable = False
    return copied


def mechanical_kernel(stiffness: float = RADIAL_STIFFNESS) -> tuple[Array, Array]:
    """Exact fixed-kernel evolution; zero stiffness is the declared comparator."""
    if type(stiffness) not in (int, float) or stiffness not in (0.0, RADIAL_STIFFNESS):
        raise ValueError("Undeclared transient mechanical kernel")
    from scipy.linalg import expm

    evolution = expm(np.array([[0.0, 1.0, 0.0], [-stiffness, -1.0, 1.0], [0.0, 0.0, 0.0]]) * 0.001)
    state = np.zeros((9, 401, 2), dtype=np.float64)
    waveform = np.zeros((9, 400), dtype=np.float64)
    for schedule in range(1, 9):
        duration = 128 if schedule in (1, 2, 5, 6) else 256
        recovery = 16 if schedule in (1, 3, 5, 7) else 144
        sign = -1 if schedule < 5 else 1
        ramp, dwell = 3 * duration // 8, duration // 4
        start = 400 - duration - recovery
        pulse = np.concatenate((np.arange(1, ramp + 1) / ramp, np.ones(dwell), 1 - np.arange(1, ramp + 1) / ramp))
        waveform[schedule, start:start + duration] = sign * 0.125 * pulse
        for tick in range(400):
            state[schedule, tick + 1] = (evolution @ np.concatenate((state[schedule, tick], waveform[schedule, tick:tick + 1])))[:2]
    values = state[:, np.arange(0, 401, 16)]
    rates = np.zeros_like(values)
    rates[:, 2:] = (values[:, 2:] - values[:, :-2]) / 0.032
    return _readonly(np.concatenate((values, rates), axis=-1)), _readonly(waveform)


@dataclass(frozen=True, slots=True)
class BridgeFit:
    normalizer: Normalizer
    kernel_scale: Array
    operator: Array
    rank: int
    residual_mse: float

    def predict(self, prefix: Array, kernel: Array) -> Array:
        if not 0 < len(prefix) <= 96:
            raise ValueError("Transient bridge prediction root count exceeds its bound")
        _array_input(prefix, (len(prefix), 24), "prefix")
        _array_input(kernel, (9, 26, 4), "kernel")
        coefficients = affine_prediction(self.operator, self.normalizer.apply(prefix)).reshape(-1, 4, 24)
        return np.einsum("stk,rkj->rstj", kernel / self.kernel_scale, coefficients)


def _selection(selection: tuple[int, ...], allowed: tuple[int, ...], name: str) -> NDArray[np.int64]:
    if not isinstance(selection, tuple) or not selection or tuple(sorted(set(selection))) != selection or any(type(item) is not int or item not in allowed for item in selection):
        raise ValueError(f"Transient bridge {name} selection is empty, repeated or undeclared")
    return np.asarray(selection, dtype=np.int64)


def fit_bridge(
    prefix: Array,
    delta: Array,
    kernel: Array,
    *,
    schedules: tuple[int, ...] = TRAINING_SCHEDULES,
    times: tuple[int, ...] = TRAINING_TIMES,
) -> BridgeFit:
    """Use only supplied training roots and declared schedule/time cells."""
    if not 0 < len(prefix) <= 24:
        raise ValueError("Transient bridge training root count exceeds its bound")
    _array_input(prefix, (len(prefix), 24), "training prefix")
    _array_input(delta, (len(prefix), 9, 26, 24), "training delta")
    _array_input(kernel, (9, 26, 4), "kernel")
    schedule_index = _selection(schedules, TRAINING_SCHEDULES, "schedule")
    time_index = _selection(times, TRAINING_TIMES, "time")
    selected = kernel[schedule_index][:, time_index]
    scale = np.maximum(np.sqrt(np.mean(selected ** 2, axis=(0, 1))), 1e-12)
    design = (selected / scale).reshape(-1, 4)
    target = delta[:, schedule_index][:, :, time_index]
    target_matrix = target.transpose(1, 2, 0, 3).reshape(-1, len(prefix) * 24)
    weights, _, rank, _ = np.linalg.lstsq(design, target_matrix, rcond=1e-10)
    if rank != 4:
        raise ValueError("UNEVALUABLE: transient modal design does not have rank four")
    coefficients = weights.reshape(4, len(prefix), 24).transpose(1, 0, 2).reshape(len(prefix), 96)
    normalizer = Normalizer.fit(prefix)
    operator = fit_affine_operator(normalizer.apply(prefix), coefficients, ridge=10)
    return BridgeFit(
        Normalizer(_readonly(normalizer.center), _readonly(normalizer.scale)),
        _readonly(scale),
        _readonly(operator),
        int(rank),
        float(np.mean((design @ weights - target_matrix) ** 2)),
    )


def fit_hold(prefix: Array, handoff: Array, query: Array) -> Array:
    """Excluded-root native HOLD forecast with the same ridge10 normalizer."""
    if not 0 < len(prefix) <= 24 or not 0 < len(query) <= 96:
        raise ValueError("Transient HOLD root count exceeds its bound")
    _array_input(prefix, (len(prefix), 24), "HOLD training prefix")
    _array_input(handoff, (len(prefix), 24), "HOLD training handoff")
    _array_input(query, (len(query), 24), "HOLD query")
    normalizer = Normalizer.fit(prefix)
    operator = fit_affine_operator(normalizer.apply(prefix), handoff, ridge=10)
    return affine_prediction(operator, normalizer.apply(query))


@dataclass(frozen=True, slots=True)
class TransientBridgeFoldOperands:
    input: TransientBridgeInput
    arrays: Mapping[str, NDArray[np.generic]]
    fits: tuple[Mapping[str, object], ...]


def current_root_fold_arrays(
    *,
    current_input: TransientBridgeInput,
    original_f: OriginalFiniteResponseLaw,
    primary_prefix: Array,
    native_handoff: Array,
    trajectories: Array,
) -> TransientBridgeFoldOperands:
    """Produce four radial.all outer fits and reusable readiness operands.

    Ordering comes from current_input.root_ids/roles, not archival roster names.
    Fold assignments are root index modulo four. Every normalizer, HOLD fit and
    modal target uses only the eighteen training roots. Complete measured
    trajectories/handoffs remain supplied current artifacts; none are simulated
    or reconstructed as observations. The output HOLD forecast is explicitly a
    current excluded-root fit, not an archival U2 prediction.
    """
    if current_input.original_f != original_f.identity:
        raise ValueError("Transient bridge original F differs from its current operand identity")
    _array_input(primary_prefix, (24, 24), "primary prefix")
    _array_input(native_handoff, (24, 9, 24, 2), "native handoff")
    _array_input(trajectories, (24, 9, 2, 26, 24), "trajectories")
    if not np.allclose(trajectories[:, :, :, -1].transpose(0, 1, 3, 2), native_handoff, rtol=0, atol=1e-12):
        raise ValueError("Current trajectory endpoints differ from measured handoff")
    if not np.allclose(trajectories[:, :, 0, 0], primary_prefix[:, None], rtol=0, atol=1e-12):
        raise ValueError("Current trajectories change the same-root primary prefix")
    lower_scale = np.asarray(original_f.scale, dtype=np.float64)
    delta = (trajectories[:, :, 0] - trajectories[:, :1, 0]) / lower_scale
    kernel, waveform = mechanical_kernel()
    folds = np.arange(24, dtype=np.int64) % 4
    forecast = np.empty((24, 9, 26, 24), dtype=np.float64)
    hold = np.empty((24, 24), dtype=np.float64)
    fits = []
    for fold in range(4):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        fit = fit_bridge(primary_prefix[train], delta[train], kernel)
        forecast[test] = fit.predict(primary_prefix[test], kernel)
        hold[test] = fit_hold(primary_prefix[train], native_handoff[train, 0, :, 0], primary_prefix[test])
        fits.append(MappingProxyType({
            "model": "radial.all",
            "fold": fold,
            "training_roots": _readonly(train),
            "test_roots": _readonly(test),
            "normalizer_center": fit.normalizer.center,
            "normalizer_scale": fit.normalizer.scale,
            "kernel_scale": fit.kernel_scale,
            "operator": fit.operator,
            "kernel_rank": fit.rank,
            "training_modal_projection_mse": fit.residual_mse,
        }))
    native = hold[:, None] + forecast[:, :, -1] * lower_scale
    arrays = MappingProxyType({name: _readonly(value) for name, value in {
        "folds": folds,
        "truth_delta": delta[:, :, -1],
        "radial.kernel": kernel,
        "radial.waveform": waveform,
        "radial.all.forecast_delta": forecast,
        "radial.all.hold_native": hold,
        "radial.all.native": native,
    }.items()})
    return TransientBridgeFoldOperands(current_input, arrays, tuple(fits))
