"causal response prediction's explicit exposed-source qualification compatibility map for retained scalar labels."

import numpy as np

from empirical_lawhood.adapters.methods.prepared_response.projection import _future_lineage
from empirical_lawhood.adapters.simulators.prepared_response.instruments import _observation_window
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff, PreparedNativePhaseData
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from .models import Array


def exposed_source_qualification_transition_labels(
    common: PreparedCommonStart,
    handoff: PreparedNativeHandoff,
    future: PreparedNativePhaseData,
) -> Array:
    _future_lineage(common, handoff, future)
    if common.root.stage != 'qualification' or future.delivery.disposition != "COMPLETE":
        raise ValueError("causal response prediction retained training projection requires complete original source qualification data")
    frame, checkpoint = common.frame, handoff.checkpoint
    assert frame is not None
    positions = np.concatenate(
        (_decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4)), future.positions[1:])
    )
    momenta = np.concatenate(
        (_decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)), future.momenta[1:])
    )
    ticks = (*checkpoint.history_ticks, *(int(t) for t in future.ticks[1:]))
    if tuple(future.ticks) != tuple(range(common.root.handoff, common.root.handoff + 321, 16)):
        raise ValueError("causal response prediction retained transition labels change the declared sample clock")
    return np.stack(
        [
            _observation_window(
                frame,
                ticks[row : row + 31],
                positions[row : row + 31],
                momenta[row : row + 31],
                last_only=True,
            )[0]
            for row in range(21)
        ]
    )
