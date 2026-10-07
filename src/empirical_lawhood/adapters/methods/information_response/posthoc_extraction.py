# SPDX-License-Identifier: MPL-2.0
"""Reconstruct original outcome-visible information-response native operands.

Authenticate the original two-context, 32-root-per-context inventory, frozen
model bank, readout arrays and native result/payload hashes before entry under
separate analysis authority. Prefix frames precede preparation; handoff features
include only history through the actual parent handoff. The extractor opens no
files, integrates no source and neither fits nor publishes a qualified law.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_handoff
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_mechanism_sketch, prepared_observation_history
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .models import InformationResponseModelBank

CONTEXTS = ("assembling", "prepared")
PARENTS = (
    "hold",
    "y-negative-128",
    "y-positive-128",
    "y-negative-256",
    "y-positive-256",
)
HORIZONS = (64, 128, 192, 256, 320)


def _future_branches(
    record: bytes, payload: bytes, task: str
) -> tuple[dict[str, Any], ...]:
    typed = decode_canonical_bytes(
        record, PreparedNativeTaskResult, maximum_bytes=16 * 1024**2
    )
    if typed.invocation.task_id != task or not typed.native_complete:
        raise ValueError("Retained future differs from its complete native task")
    pair = decode_prepared_task_native(typed, payload)
    assert pair is not None
    return tuple(
        {"ticks": branch.ticks, "positions": branch.positions} for branch in pair
    )


def _receiver(modes: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("pabc,...abc->...p", modes.conj(), values))


def extract_retained_arrays(
    old: dict[str, Any],
    bank: InformationResponseModelBank,
    native_inputs: Mapping[str, tuple[bytes, bytes]],
) -> dict[str, Any]:
    """Reproduce every retained handoff forecast and paired endpoint gain.

    Supply separately verified original-source exports binding original hashes
    and interpretation to current task identities and target custody, with
    authenticated original readout arrays, model bank and canonical
    native result/payload pairs. Parent-period features remain permissible only
    through their actual handoff cutoff. The returned arrays are exposed-data
    audit operands, not permission for a fresh scientific experiment.
    """
    required = {
        task
        for context in CONTEXTS
        for index in range(32)
        for root in (f"{CAMPAIGN}.prospective-evaluation.{context}.r{index:03d}",)
        for parent in PARENTS
        for task in (
            f"{root}.{parent}.parent.native",
            *(f"{root}.{parent}.{purpose}.{CAMPAIGN}.word.a16.d{axis}.{sign}.native"
              for purpose in ("common-response", "prospective-task")
              for axis in (0, 1) for sign in ("negative", "positive")),
        )
    }
    if not required.issubset(native_inputs):
        raise ValueError("INFORMATION_RESPONSE_RETAINED_VERIFIED_TARGET_EXPORT_REQUIRED")
    history = np.empty((2, 32, 5, 2, 16, 12))
    sketch = np.empty((2, 32, 5, 2, 8))
    prefix_history = np.empty((2, 32, 2, 16, 12))
    prefix_sketch = np.empty((2, 32, 2, 8))
    translation = np.empty((2, 32, 5, 2, 2))
    endpoints = np.empty((2, 32, 2, 5, 2, 4, 5, 2))
    for ci, context in enumerate(CONTEXTS):
        for ri in range(32):
            root = f"{CAMPAIGN}.prospective-evaluation.{context}.r{ri:03d}"
            for pi, parent in enumerate(PARENTS):
                record, payload = native_inputs[f"{root}.{parent}.parent.native"]
                typed = decode_canonical_bytes(
                    record, PreparedNativeTaskResult, maximum_bytes=16 * 1024**2
                )
                pair = decode_prepared_task_native(typed, payload)
                common = typed.common_start
                assert (
                    pair is not None
                    and common is not None
                    and (common.frame is not None)
                )
                if pi == 0:
                    for vi, checkpoint in enumerate(common.checkpoints):
                        state, _ = state_from_checkpoint(checkpoint.native)
                        prefix_history[ci, ri, vi] = prepared_observation_history(
                            frame=common.frame,
                            ticks=checkpoint.history_ticks,
                            positions=_decode(
                                checkpoint.history_positions_base64, (31, 2, 3, 4, 4)
                            ),
                            momenta=_decode(
                                checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)
                            ),
                            cutoff_tick=common.root.landmark,
                        )
                        prefix_sketch[ci, ri, vi] = prepared_mechanism_sketch(
                            state, common.frame
                        )
                for vi, data in enumerate(pair):
                    observed = project_prepared_handoff(
                        common,
                        bind_prepared_native_handoff(data),
                        instrument_tier="I1",
                    )
                    history[ci, ri, pi, vi] = observed.history
                    sketch[ci, ri, pi, vi] = observed.sketch
                    translation[ci, ri, pi, vi] = observed.preparent_displacement
                predicted = bank.contexts[ci].predict(
                    history[ci, ri, pi], sketch[ci, ri, pi], (pi, pi)
                )
                np.testing.assert_allclose(
                    predicted, old["forecasts"][ci, ri, :, pi], rtol=0, atol=0
                )
                for fi, future in enumerate(("common-response", "prospective-task")):
                    for wi, (axis, sign) in enumerate(
                        (
                            (0, "negative"),
                            (0, "positive"),
                            (1, "negative"),
                            (1, "positive"),
                        )
                    ):
                        task = f"{root}.{parent}.{future}.{CAMPAIGN}.word.a16.d{axis}.{sign}.native"
                        rec, data = native_inputs[task]
                        raw = _future_branches(rec, data, task)
                        for vi, branch in enumerate(raw):
                            ix = [
                                int(
                                    np.flatnonzero(
                                        branch["ticks"] == common.root.handoff + t
                                    )[0]
                                )
                                for t in HORIZONS
                            ]
                            endpoints[ci, ri, vi, pi, fi, wi] = _receiver(
                                common.frame.modes,
                                branch["positions"][ix, 0] - branch["positions"][0, 0],
                            )
    reconstructed = np.stack(
        (
            (endpoints[..., 1, :, :] - endpoints[..., 0, :, :]) / 2,
            (endpoints[..., 3, :, :] - endpoints[..., 2, :, :]) / 2,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(reconstructed, old["gains"], rtol=0, atol=2e-12)
    values = {k: old[k] for k in ("gains", "forecasts", "parent_work", "force_work")}
    values.update(
        history=history,
        sketch=sketch,
        prefix_history=prefix_history,
        prefix_sketch=prefix_sketch,
        translation=translation,
        endpoints=endpoints,
    )
    return values
