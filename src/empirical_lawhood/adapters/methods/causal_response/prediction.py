"""Causal prediction attachment: only frozen coefficients and the parent handoff."""

from decimal import Decimal

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_handoff
from empirical_lawhood.adapters.simulators.causal_response.contracts import CausalResponseNativeConfig
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from .models import central_gain
from .records import CausalResponseCommittedPrediction


def predict_parent(
    config: CausalResponseNativeConfig, result: PreparedNativeTaskResult, payload: bytes
) -> CausalResponseCommittedPrediction:
    identity = ObjectIdentity.from_record
    gain: tuple[Decimal | None, ...] = (None,) * 40
    disposition = "NOT_A_HANDOFF" if result.invocation.phase != "parent" else "UNAVAILABLE"
    if result.invocation.phase == "parent" and result.native_complete:
        pair = decode_prepared_task_native(result, payload)
        common = result.common_start
        if pair is not None and common is not None and common.frame is not None:
            context = next(
                c for c in config.model_bank.contexts if c.recipe.context == common.root.context
            )
            model = context.predictor()
            try:
                measured = [
                    project_prepared_handoff(
                        common,
                        bind_prepared_native_handoff(v),
                        instrument_tier="I1" if model.structure == "mechanism-i1" else "I0",
                    )
                    for v in pair
                ]
                history = np.stack([m.history for m in measured])
                sketch = None
                if model.structure == "mechanism-i1":
                    sketches = [m.sketch for m in measured]
                    if any(s is None for s in sketches):
                        raise ValueError("causal response prediction mechanism measurement is unavailable")
                    sketch = np.stack([s for s in sketches if s is not None])
                values = central_gain(model.predict(history, sketch))
                gain = tuple(Decimal(str(float(v))) for v in values.ravel())
                disposition = "COMMITTED_AT_HANDOFF"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
                # A scientifically unresolved prediction remains assigned. No refit/retry.
                disposition = "UNAVAILABLE"
    return CausalResponseCommittedPrediction(
        identity(config.spec_id, config),
        identity(result.result_id, result),
        result.invocation,
        identity(config.model_bank.bank_id, config.model_bank),
        disposition,
        gain,
    )
