# SPDX-License-Identifier: MPL-2.0
"""Ambient-pressure scientific contracts with explicit native source loading."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    'ExcludedSolverControlAuthoringBundle': "authoring",
    'ExcludedSolverControlConfig': "contracts",
    'ExcludedSolverControlConfigDecoder': "registration",
    'ExcludedSolverControlRuntimeProvider': "runtime",
    'ExcludedSolverControlSourceManifest': "contracts",
    "FixedQEExecutor": "solver",
    "MaterialSolverExecutor": "solver",
    "SyntheticMaterialSolverExecutor": "solver",
    'excluded_solver_control_candidate_catalog': "candidate",
    'build_excluded_solver_control_authoring_bundle': "authoring",
    "config_decoders": "registration",
    'expected_excluded_solver_control_source_manifest': "source",
    'inspect_excluded_solver_control_external_assets': "source",
    "load_config": "contracts",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{module_name}'), name)
