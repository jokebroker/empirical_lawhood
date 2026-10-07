"""Load the three pinned public reactor inputs from installed package resources.

SPDX-License-Identifier: MPL-2.0
"""

from importlib import resources

from .panel import ReactorSourceBundle


_RESOURCE_ROOT = resources.files("empirical_lawhood").joinpath(
    "_vendor", "terminal_bench_science"
)


def load_packaged_reactor_source() -> ReactorSourceBundle:
    """Load only the three packaged upstream bytes, then check their exact pins."""
    return ReactorSourceBundle(
        _RESOURCE_ROOT.joinpath("tests", "plant.py").read_text(encoding="utf-8"),
        _RESOURCE_ROOT.joinpath("environment", "spec", "plant_params.json").read_text(
            encoding="utf-8"
        ),
        _RESOURCE_ROOT.joinpath(
            "environment", "spec", "scenarios_public.json"
        ).read_text(encoding="utf-8"),
    )
