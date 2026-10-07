"""Complete-interval composition of native Response geometry action/observer checkpoints."""

from dataclasses import dataclass
import json
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .response_assay import ResponseGeometryNativeForcePulse
from .response_observer import ResponseGeometryNativeGeometrySample, ResponseGeometryNativeProbeCheckpoint, _decode
from .simulation import SixMatrixResponseCheckpoint


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeCheckpoint(CanonicalRecord):
    """One nested continuation, never an authorization to repeat an effect.

    The native clock is global within the root. Checkpoints occur only after a
    complete BAOAB interval; its two force kicks and observer update are indivisible.
    The enclosing issued request/receipt authenticates all bytes and RNG purposes.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-checkpoint'

    checkpoint_id: str
    native: SixMatrixResponseCheckpoint
    passive: ResponseGeometryNativeProbeCheckpoint
    structural_window: tuple[ResponseGeometryNativeGeometrySample, ...]
    x_window_base64: str
    frozen_mode_base64: str | None
    parent_id: str | None
    parent_origin_tick: int | None
    pulse: ResponseGeometryNativeForcePulse | None
    invocation_positions_base64: str | None
    bridge_rng_state_json: str
    pending_noises_base64: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        if self.native.q != 2 or self.native.step_index != self.passive.native_step:
            raise ValueError("Response geometry native and observer continuation clocks differ")
        if self.native.receivers:
            raise ValueError(
                "Response geometry retains its explicit rolling window, not an unbounded receiver prefix"
            )
        refinement = self.passive.refinement
        tick = self.native.step_index // refinement
        latest = (tick // 16) * 16
        expected = tuple(range(max(0, latest - 240), latest + 1, 16))
        if tuple(value.reference_tick for value in self.structural_window) != expected:
            raise ValueError("Response geometry checkpoint lost or changed the causal structural window")
        _decode(self.x_window_base64, (len(expected), 3, 4, 4))
        if self.frozen_mode_base64 is not None:
            mode = _decode(self.frozen_mode_base64, (3, 4, 4))
            if abs(float(np.vdot(mode, mode).real) - 1) > 1e-12:
                raise ValueError("Response geometry checkpoint force mode must have unit HS norm")
        if (self.parent_id is None) != (self.parent_origin_tick is None):
            raise ValueError("Response geometry parent identity and origin must appear together")
        if self.parent_id is not None:
            validate_stable_id(self.parent_id, field_name="parent_id")
            if (
                type(self.parent_origin_tick) is not int
                or not 240 <= self.parent_origin_tick <= tick
            ):
                raise ValueError("Response geometry checkpoint parent origin is not in the causal prefix")
            if self.frozen_mode_base64 is None:
                raise ValueError("Response geometry parent branch requires its frozen pre-parent mode")
        if (self.pulse is None) != (self.invocation_positions_base64 is None):
            raise ValueError("Response geometry pulse and invocation receiver origin must appear together")
        if self.pulse is not None:
            if (
                self.parent_origin_tick is None
                or not self.parent_origin_tick <= self.pulse.invocation_tick <= tick
            ):
                raise ValueError("Response geometry checkpoint pulse is outside the parent continuation")
            assert self.invocation_positions_base64 is not None
            _decode(self.invocation_positions_base64, (2, 3, 4, 4))
        for payload in (self.native.rng_state_json, self.bridge_rng_state_json):
            if len(payload) > 1024:
                raise ValueError("Response geometry checkpoint RNG state exceeds its bounded encoding")
            document = json.loads(payload)
            if not isinstance(document, dict):
                raise ValueError("Response geometry checkpoint RNG state must be a mapping")
            if json.dumps(document, sort_keys=True, separators=(",", ":")) != payload:
                raise ValueError("Response geometry checkpoint RNG state must be canonical JSON")
            if document.get("bit_generator") != "PCG64DXSM":
                raise ValueError("Response geometry checkpoint RNG algorithm must be PCG64DXSM")
            generator = np.random.Generator(np.random.PCG64DXSM(0))
            generator.bit_generator.state = document
        remainder = self.native.step_index % refinement
        expected_pending = refinement - remainder if remainder else 0
        if len(self.pending_noises_base64) != expected_pending:
            raise ValueError("Response geometry checkpoint lost or changed pending refined innovations")
        for noise in self.pending_noises_base64:
            _decode(noise, (2, 3, 4, 4))
        native_positions = _decode(self.native.positions_base64, (2, 3, 4, 4))
        _decode(self.native.momenta_base64, (2, 3, 4, 4))
        observer_y = _decode(self.passive.y_base64, (3, 4, 4))
        if not np.array_equal(native_positions[1], observer_y):
            raise ValueError("Response geometry checkpoint observer is detached from its native state")
        if len(self.canonical_bytes()) > 128 * 1024:
            raise ValueError("Response geometry checkpoint exceeds its declared 128-KiB allocation")
