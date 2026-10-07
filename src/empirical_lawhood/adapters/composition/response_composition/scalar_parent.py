# SPDX-License-Identifier: MPL-2.0

"""Closed response composition scalar interchange for receipt-authenticated source qualification development.

The original absolute channels, rather than pre-subtracted responses or summary
pass counts, are inputs. The consumer owns HOLD subtraction and preservation.
"""

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.response_composition.development import ResponseCompositionDevelopmentSpec, develop_context, paired_response
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
)

if TYPE_CHECKING:
    from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
    from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativeCheckpoint

SHAPES = {
    "features": (32, 2, 6, 200),
    "original_channels": (32, 2, 5, 9, 5, 5),
    "parent_work": (32, 2, 5),
}


def prepared_checkpoint_features(
    checkpoint: 'PreparedNativeCheckpoint',
    frame: 'PreparedPortFrame',
    cutoff_tick: int,
) -> np.ndarray:
    """Compose the original 192 history and eight native mechanism scalars.

    Callers authenticate the checkpoint and frozen pre-parent frame before
    entry. The native endpoint and its last causal history sample must agree.
    This reduction performs no native update, fit, publication or issue.
    """
    from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_mechanism_sketch, prepared_observation_history
    from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
    from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint

    q = _decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4))
    p = _decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4))
    history = prepared_observation_history(
        frame=frame,
        ticks=tuple(checkpoint.history_ticks),
        positions=q,
        momenta=p,
        cutoff_tick=cutoff_tick,
    )
    state, _ = state_from_checkpoint(checkpoint.native)
    if (
        checkpoint.native.step_index != cutoff_tick * checkpoint.passive.refinement
        or not np.array_equal(q[-1], state.positions)
        or not np.array_equal(p[-1], state.momenta)
    ):
        raise ValueError(
            "causal history endpoint differs from authenticated checkpoint"
        )
    return np.r_[history.ravel(), prepared_mechanism_sketch(state, frame)]


@dataclass(frozen=True, slots=True)
class ResponseCompositionScalarParent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/response-composition/response-composition-scalar-parent'
    original_root_ids: tuple[str, ...]
    development_spec: ResponseCompositionDevelopmentSpec
    arrays_base64: tuple[tuple[str, str], ...]
    evidence_role: str = "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
    encoding: str = "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_FIXED_SHAPES"

    def __post_init__(self) -> None:
        if type(self.development_spec) is not ResponseCompositionDevelopmentSpec:
            raise ValueError("RESPONSE_COMPOSITION_DEVELOPMENT_SPEC_REQUIRED")
        require_sorted_unique_strings(
            self.original_root_ids, field_name="original_root_ids"
        )
        if (
            len(self.original_root_ids) != 32
            or any(
                not root.endswith(f".qualification.{context}.r{index:03d}")
                for root, (context, index) in zip(
                    self.original_root_ids,
                    (
                        (context, index)
                        for context in ("assembling", "prepared")
                        for index in range(16)
                    ),
                    strict=True,
                )
            )
            or tuple(key for key, _ in self.arrays_base64) != tuple(SHAPES)
            or self.encoding != "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_FIXED_SHAPES"
            or self.evidence_role != "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError("RESPONSE_COMPOSITION_SCALAR_PARENT_CONTRACT_MISMATCH")
        self.arrays()

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {}
        for key, text in self.arrays_base64:
            shape = SHAPES[key]
            expected = int(np.prod(shape)) * 8
            if len(text) != 4 * ((expected + 2) // 3):
                raise ValueError("RESPONSE_COMPOSITION_SCALAR_PARENT_ARRAY_SIZE_INVALID")
            raw = base64.b64decode(text, validate=True)
            if len(raw) != expected or base64.b64encode(raw).decode() != text:
                raise ValueError("RESPONSE_COMPOSITION_SCALAR_PARENT_ARRAY_ENCODING_INVALID")
            array = np.frombuffer(raw, dtype="<f8").reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError("RESPONSE_COMPOSITION_SCALAR_PARENT_UNEVALUABLE_NONFINITE")
            arrays[key] = array
        if np.any(arrays["original_channels"][..., 2:] < 0) or np.any(
            arrays["parent_work"] < 0
        ):
            raise ValueError("RESPONSE_COMPOSITION_SCALAR_PARENT_NEGATIVE_PRESERVATION_OR_WORK")
        return arrays


def analyze_scalar_parent(parent: ResponseCompositionScalarParent) -> dict[str, object]:
    """Use the retained operator after the caller has authenticated authority."""
    arrays = parent.arrays()
    channels = arrays["original_channels"]
    response = paired_response(channels[..., :2])
    preservation = (
        (channels[..., 2] <= 0.125)
        & (channels[..., 3] <= 0.05)
        & (channels[..., 4] <= 0.05)
    ).all(axis=(1, 3, 4))
    preservation &= (arrays["parent_work"] <= 32).all(axis=1)
    return {
        context: develop_context(
            arrays["features"][offset : offset + 16],
            response[offset : offset + 16],
            preservation[offset : offset + 16],
        )
        for context, offset in (("assembling", 0), ("prepared", 16))
    }


__all__ = [
    "SHAPES",
    'ResponseCompositionScalarParent',
    "analyze_scalar_parent",
    "prepared_checkpoint_features",
]
