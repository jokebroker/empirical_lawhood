"""Independent evaluator-side task convention check; no model/charter helper calls."""

import math

import numpy as np
import numpy.typing as npt


def check_prepared_endpoint(
    *,
    absolute_displacement: npt.NDArray[np.float64],
    preparent_velocity: tuple[float, float],
    target_draw: int,
    horizon_ticks: int,
    epsilon: float,
    distance_multiple: int,
) -> tuple[bool, float]:
    """Two measured views must attain the same pre-parent-defined native target.

    This checks target attainment only. Delivery, preservation, numerical
    agreement, admission and observation validity remain separate operands.
    """
    points = np.asarray(absolute_displacement)
    if (
        points.shape != (2, 2)
        or points.dtype != np.dtype("float64")
        or not np.isfinite(points).all()
        or len(preparent_velocity) != 2
        or not all(math.isfinite(v) for v in preparent_velocity)
        or type(target_draw) is not int
        or target_draw not in range(9)
        or type(horizon_ticks) is not int
        or horizon_ticks not in (192, 256, 320)
        or epsilon not in (1 / 32, 1 / 16, 3 / 32, 1 / 8)
        or type(distance_multiple) is not int
        or distance_multiple not in (4, 6)
    ):
        raise ValueError("endpoint checker lacks the complete two-view native task operands")
    directions = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1))
    x, y = directions[target_draw]
    length = math.hypot(x, y)
    scale = epsilon * distance_multiple / length if length else 0.0
    coefficient = -math.expm1(-0.001 * (272 + horizon_ticks))
    center = (
        preparent_velocity[0] * coefficient + x * scale,
        preparent_velocity[1] * coefficient + y * scale,
    )
    loss = max(abs(float(points[v, j]) - center[j]) for v in range(2) for j in range(2))
    return loss <= epsilon, loss
