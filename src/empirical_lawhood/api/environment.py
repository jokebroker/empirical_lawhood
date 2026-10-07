# SPDX-License-Identifier: MPL-2.0
"""Noncontact dependency observations for the documented native environments."""

from __future__ import annotations

from enum import Enum
from importlib import metadata
import platform

from .models import DoctorDependencySummary


class EnvironmentRoute(str, Enum):
    REACTOR = "reactor"
    PREPARED_RESPONSE = 'prepared-response'
    OPEN_SIMULATORS = "open-simulators"
    REACTION_RESPONSE = 'reaction-response'
    GRID2OP = "grid2op"
    RC_CHALLENGES = "rc-challenges"
    FINITE_RESPONSE_LAW = "finite-response-law"
    PREPARATION_APPLICABILITY = "preparation-applicability"


# These are the documented native contracts, not the wider package installation
# ranges. Observing their metadata cannot establish source or scientific readiness.
_VERSIONS = {
    EnvironmentRoute.REACTOR: (("numpy", "2.4.6"),),
    EnvironmentRoute.PREPARED_RESPONSE: (("numpy", "2.4.6"),),
    EnvironmentRoute.OPEN_SIMULATORS: (
        ("gymtorax", "1.1.1"), ("torax", "1.4.2"),
        ("jax", "0.10.2"), ("jaxlib", "0.10.2"),
        ("numpy", "2.4.6"), ("scipy", "1.17.1"),
        ("xarray", "2026.7.0"), ("pybamm", "26.6.2.0"),
    ),
    EnvironmentRoute.REACTION_RESPONSE: (("cantera", "3.2.0"), ("fipy", "4.0.3")),
    EnvironmentRoute.RC_CHALLENGES: (("numpy", "2.4.6"), ("scipy", "1.17.1")),
    EnvironmentRoute.FINITE_RESPONSE_LAW: (("numpy", "2.4.6"), ("scipy", "1.17.1")),
    EnvironmentRoute.PREPARATION_APPLICABILITY: (("numpy", "2.4.6"), ("scipy", "1.17.1")),
    EnvironmentRoute.GRID2OP: (("grid2op", "1.12.5"), ("lightsim2grid", "0.13.1")),
}


def inspect_route_dependencies(
    route: EnvironmentRoute | None,
) -> tuple[DoctorDependencySummary, ...]:
    """Read distribution metadata only; never import or initialize a simulator."""

    selected = (("gymtorax", "1.1.1"), ("torax", "1.4.2")) if route is None else _VERSIONS[route]
    observations: list[tuple[str, str | None, str]] = []
    if route is not None:
        python = platform.python_version() if platform.python_implementation() == "CPython" else None
        observations.append(("cpython", python, "3.11.14"))
    for distribution, expected in selected:
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = None
        observations.append((distribution, installed, expected))
    return tuple(
        DoctorDependencySummary(
            dependency_id=name,
            available=installed is not None,
            required_for_actions=("gym-torax-substrate" if route is None else route.value,),
            reason_codes=(
                (f"DEPENDENCY_{name.upper()}_ABSENT",) if installed is None else
                (f"DEPENDENCY_{name.upper()}_VERSION_MISMATCH",) if installed != expected else ()
            ),
            installed_version=installed,
            required_version=expected,
            version_matches=installed == expected,
        )
        for name, installed, expected in observations
    )
