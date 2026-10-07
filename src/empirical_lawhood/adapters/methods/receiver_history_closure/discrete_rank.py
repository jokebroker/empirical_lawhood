"""Family record assembly for the shared rigorous threshold-rank method."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.certified_rank import (
    binary64_matrix_snapshot,
    certify_rank,
    residual_certified_rank_bracket,
)
from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryDiscreteRankBracket,
    ReceiverHistoryRankCompatibility,
    ReceiverHistoryStructuralRankStep,
)
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .history import sampled_history_matrix

FloatArray = npt.NDArray[np.float64]


def discrete_rank_bracket(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate: ReceiverHistoryCoordinateLabel,
    structural: ReceiverHistoryStructuralRankStep,
    *,
    matrix: FloatArray | None = None,
) -> ReceiverHistoryDiscreteRankBracket:
    if coordinate.coordinate_id != structural.coordinate_id:
        raise ValueError("receiver-history structural/discrete coordinate identity differs")
    matrix = binary64_matrix_snapshot(
        sampled_history_matrix(config, descriptor, coordinate.depth)
        if matrix is None
        else matrix
    )
    if matrix.shape != (8 * (coordinate.depth + 1), descriptor.scale_cells):
        raise ValueError("receiver-history discrete-rank matrix shape differs")
    reconstruction = certify_rank(matrix, precision_bits=256)
    lower = reconstruction.lower
    upper = reconstruction.upper
    residual = reconstruction.residual
    separation = None
    if lower != upper:
        compatibility = ReceiverHistoryRankCompatibility.RANK_UNRESOLVED
    elif upper > structural.structural_rank:
        compatibility = ReceiverHistoryRankCompatibility.NUMERICAL_CANCELLATION
    elif upper == structural.structural_rank:
        compatibility = ReceiverHistoryRankCompatibility.COMPATIBLE
    else:
        compatibility = ReceiverHistoryRankCompatibility.SAMPLING_ALIAS
    matrix_digest = sha256(
        canonical_json_bytes(
            {
                "dtype": str(matrix.dtype),
                "shape": matrix.shape,
                "sha256": sha256(matrix.tobytes(order="C")).hexdigest(),
            }
        )
    ).hexdigest()
    return ReceiverHistoryDiscreteRankBracket(
        coordinate_id=coordinate.coordinate_id,
        lower_rank=lower,
        upper_rank=upper,
        state_count=descriptor.scale_cells,
        precision_bits=reconstruction.precision_bits,
        residual_norm=residual,
        separation_ratio=separation,
        matrix_sha256=matrix_digest,
        compatibility=compatibility,
        algorithm_id=config.discrete_rank_algorithm_id,
    )


__all__ = ["discrete_rank_bracket", "residual_certified_rank_bracket"]
