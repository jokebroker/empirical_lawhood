"""Bounded numerical development check for a strictly decoded RC model.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .contracts import ResistorCapacitorLadderModelConfig
from .matrix_exponential import solve_matrix_exponential
from .refinement import solve_backward_euler

_MAX_CONFIG_BYTES = 64 * 1024


def check_native_model(config_path: Path) -> dict[str, object]:
    """Compute two native views of one model without issuing a campaign."""

    model = decode_canonical_bytes(
        read_bounded_bytes(config_path, maximum_bytes=_MAX_CONFIG_BYTES),
        ResistorCapacitorLadderModelConfig,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    exact = solve_matrix_exponential(model)
    reference = solve_backward_euler(model, maximum_step_seconds=0.001)
    defect = float(np.max(np.abs(exact.voltages_volts - reference.voltages_volts)))
    return {
        "config_id": model.config_id,
        "cells": model.scale_cells,
        "horizon_s": str(model.output_times_seconds[-1]),
        "receiver_unit": "V",
        "boundary_current_unit": "A",
        "maximum_backward_euler_defect_v": defect,
        "final_cell_voltages_v": [float(value) for value in exact.voltages_volts[-1]],
        "native_numerical_check_executed": True,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ["check_native_model"]
