"""Causal prediction attachment: only frozen coefficients and the parent handoff."""

from decimal import Decimal

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_handoff
from empirical_lawhood.adapters.simulators.information_response.contracts import InformationResponseNativeConfig
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS
from .records import InformationResponseCommittedPrediction


def predict_parent(
    config: InformationResponseNativeConfig, result: PreparedNativeTaskResult, payload: bytes
) -> InformationResponseCommittedPrediction:
    identity = ObjectIdentity.from_record
    gain: tuple[Decimal | None, ...] = (None,) * 280
    disposition = "NOT_A_HANDOFF" if result.invocation.phase != "parent" else "UNAVAILABLE"
    if result.invocation.phase == "parent" and result.native_complete:
        pair = decode_prepared_task_native(result, payload)
        common = result.common_start
        if pair is not None and common is not None and common.frame is not None:
            context = next(
                c for c in config.model_bank.contexts if c.recipe.context == common.root.context
            )
            try:
                measured = [
                    project_prepared_handoff(
                        common, bind_prepared_native_handoff(v), instrument_tier="I1"
                    )
                    for v in pair
                ]
                history = np.stack([m.history for m in measured])
                sketches = [m.sketch for m in measured]
                if any(s is None for s in sketches):
                    raise ValueError("information response prediction required mechanism measurement is unavailable")
                sketch = np.stack([s for s in sketches if s is not None])
                parent = PARENTS.index(str(result.invocation.parent))
                values = context.predict(history, sketch, (parent, parent))
                gain = tuple(Decimal(str(float(v))) for v in values.ravel())
                disposition = "COMMITTED_AT_HANDOFF"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
                # A scientifically unresolved prediction remains assigned. No refit/retry.
                disposition = "UNAVAILABLE"
    return InformationResponseCommittedPrediction(
        identity(config.spec_id, config),
        identity(result.result_id, result),
        result.invocation,
        identity(config.model_bank.bank_id, config.model_bank),
        disposition,
        gain,
    )
