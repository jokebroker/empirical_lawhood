"""Closed-prefix native measurements without changing the pinned plant or callbacks.

The worker retains references to the native arrays only at completed callback
boundaries. No truth crosses the online observation barrier. A caller must close
and join the worker before copying the measured prefix for sealed evaluation.
"""

from types import FrameType
import numpy as np

from .batch_bridge import ReactorBatchBridge


class ReactorPrefixBridge(ReactorBatchBridge):
    _prefix_columns: tuple[np.ndarray, ...] | None = None
    _prefix_count: int = 0

    def _retain_boundary(self, frame: FrameType, time_s: float) -> None:
        values = frame.f_locals
        names = (
            "time_s",
            "t_reactor",
            "t_jacket",
            "dosed_grid",
            "conv_grid",
            "n_a_grid",
            "feed_grid",
            "tj_cmd_grid",
        )
        count = values.get("grid_index")
        expected = int(time_s / float(self._plant_dt_s)) + 1
        columns = tuple(values.get(name) for name in names)
        if (
            type(count) is not int
            or count != expected
            or count < self._prefix_count
            or any(
                not isinstance(a, np.ndarray)
                or a.dtype != np.dtype("float64")
                or a.ndim != 1
                or len(a) < count
                for a in columns
            )
        ):
            raise ValueError("native stopped-prefix instrumentation differs from its pinned grid")
        self._prefix_columns = columns  # type: ignore[assignment]
        self._prefix_count = count

    @property
    def closed_prefix_grid(self) -> np.ndarray:
        if not self._closed or (self._thread is not None and self._thread.is_alive()):
            raise ValueError("native truth is unavailable before the branch closes")
        if self._prefix_columns is None:
            return np.empty((0, 8), dtype=np.float64)
        grid = np.column_stack([a[: self._prefix_count] for a in self._prefix_columns])
        if not np.isfinite(grid).all() or not np.array_equal(
            grid[:, 0], np.arange(self._prefix_count) * float(self._plant_dt_s)
        ):
            raise ValueError("native prefix has a nonfinite value or incomplete clock")
        return grid
