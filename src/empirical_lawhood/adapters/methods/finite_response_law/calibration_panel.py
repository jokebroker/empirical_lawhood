"""Fresh assigned-parent operands from the complete authenticated native census.

The singleton parent axis denotes the one acquired parent, not an invented
five-parent panel. This reducer neither fits nor qualifies a law.
"""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import _assigned_stage_unit
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .paired_assay import paired_native_assay
from .science import PARENTS, root_seed

Array = NDArray[np.float64]
FIXED_CALIBRATION_ROOT_IDS = tuple(f"calibration.r{r:03d}" for r in range(32))


def valid_calibration_root_ids(root_ids: tuple[str, ...]) -> bool:
    """One ordered 32-root physical cohort, old witness or assigned namespace."""

    if root_ids == FIXED_CALIBRATION_ROOT_IDS:
        return True
    if len(root_ids) != 32 or not _assigned_stage_unit(root_ids[0], "calibration", 32):
        return False
    namespace = root_ids[0].rsplit(".r", 1)[0]
    return root_ids == tuple(f"{namespace}.r{r:03d}" for r in range(32))


@dataclass(frozen=True, slots=True)
class CalibrationPanel:
    y: Array  # root, acquired parent (1), pair (4), quantity (8), future (2), view (2)
    observed: NDArray[np.bool_]
    valid: NDArray[np.bool_]
    prefix: Array  # root, feature (24), view (2); unavailable remains NaN
    handoff: Array
    parent_work: Array  # root, acquired parent (1), view (2)
    parent_indices: NDArray[np.int64]
    root_ids: tuple[str, ...]
    request_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        shape = (32, 1, 4, 8, 2, 2)
        if (
            self.y.shape != shape
            or self.y.dtype != np.float64
            or any(
                a.shape != shape or a.dtype != np.bool_
                for a in (self.observed, self.valid)
            )
            or any(
                a.shape != (32, 24, 2) or a.dtype != np.float64
                for a in (self.prefix, self.handoff)
            )
            or self.parent_work.shape != (32, 1, 2)
            or self.parent_work.dtype != np.float64
            or self.parent_indices.shape != (32,)
            or self.parent_indices.dtype != np.int64
            or ((self.parent_indices < 0) | (self.parent_indices >= 5)).any()
            or not valid_calibration_root_ids(self.root_ids)
            or self.request_seeds and (len(self.request_seeds) != 32 or any(type(s) is not int or not 0 <= s < 2**128 for s in self.request_seeds))
            or np.any(self.valid & (~self.observed | ~np.isfinite(self.y)))
        ):
            raise ValueError(
                "Fresh calibration operands change their assigned-root census or axes"
            )
        for name in (
            "y",
            "observed",
            "valid",
            "prefix",
            "handoff",
            "parent_work",
            "parent_indices",
        ):
            value = getattr(self, name)
            # Own immutable bytes, not a writable alias of a caller's array.
            object.__setattr__(
                self,
                name,
                np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape),
            )


def calibration_panel(evaluation: FiniteResponseLawCalibrationNativeEvaluation) -> CalibrationPanel:
    """Retain failed/missing roots and unfavorable measured work/preservation."""
    source = evaluation.config.projection.native_spec
    y = np.full((32, 1, 4, 8, 2, 2), np.nan)
    observed = np.zeros(y.shape, dtype=bool)
    valid = np.zeros(y.shape, dtype=bool)
    prefix, handoff = (np.full((32, 24, 2), np.nan) for _ in range(2))
    work = np.full((32, 1, 2), np.nan)
    for r, root in enumerate(source.roots):
        views = tuple(v for v in evaluation.views if v.root == root)
        for v, view in enumerate(views):
            if view.prefix_interface is not None:
                prefix[r, :, v] = view.prefix_interface.values
            if view.handoff_interface is not None:
                handoff[r, :, v] = view.handoff_interface.values
            if view.parent_absolute_density_work is not None:
                work[r, 0, v] = float(view.parent_absolute_density_work)
        for k, (magnitude, direction) in enumerate(((8, 0), (8, 1), (16, 0), (16, 1))):
            assay = paired_native_assay(
                views,
                parent=root.assigned_parent,
                word=PreparedForceWord(Decimal(magnitude), direction, 1),
            )
            for readout in assay.readouts:
                f, v = int(readout.purpose[-1]) - 1, readout.refinement - 1
                for j, value in enumerate(readout.values):
                    if value is not None:
                        y[r, 0, k, j, f, v] = float(value)
                        observed[r, 0, k, j, f, v] = True
                        valid[r, 0, k, j, f, v] = readout.source_valid
    return CalibrationPanel(
        y,
        observed,
        valid,
        prefix,
        handoff,
        work,
        np.asarray(
            [PARENTS.index(root.assigned_parent) for root in source.roots],
            dtype=np.int64,
        ),
        tuple(root.stage_unit for root in source.roots),
        tuple(root_seed(root, "calibration-request") for root in source.roots),
    )
