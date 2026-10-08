# SPDX-License-Identifier: MPL-2.0
"""Refuse frozen exposed RC descriptor drift before expensive software tests."""

from __future__ import annotations

import json
import os
import platform

import numpy as np
from threadpoolctl import threadpool_info

from empirical_lawhood.adapters.simulator_morphism_challenges.descriptors import (
    deterministic_development_seed, development_unit_ids, family_from_unit_id, generate_descriptor,
)
from empirical_lawhood.adapters.simulator_morphism_challenges.numeric_inputs import original_numeric_definition


def main() -> None:
    environment = {name: os.environ.get(name) for name in (
        "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    )}
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "environment": environment, "threadpools": threadpool_info()}), flush=True)
    np.show_runtime()
    fingerprints = []
    for unit in development_unit_ids():
        for scale in (16, 32, 64):
            descriptor = generate_descriptor(unit_id=unit, family=family_from_unit_id(unit),
                                             scale_cells=scale, seed=deterministic_development_seed(unit))
            # The existing owner retains exact seed, descriptor and fibre hashes.
            # This is exposed input verification with no outcome or authority.
            original_numeric_definition(descriptor)
            fingerprints.append({"unit": unit, "scale": scale, "sha256": descriptor.fingerprint()})
    print(json.dumps({"status": "PASSED", "exact_frozen_descriptors": fingerprints,
                      "native_execution_performed": False, "scientific_qualification": False}), flush=True)


if __name__ == "__main__":
    main()
