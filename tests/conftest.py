# SPDX-License-Identifier: MPL-2.0
"""Portable tests and explicit, mandatory integration profiles.

The numerical worker setup is adapted from the source project's test harness.
Synthetic fixtures exercise software contracts and confer no research authority.
"""

from __future__ import annotations

import importlib.metadata
import os

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set native worker limits before collecting numerical test modules."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        help="Require every selected integration check; missing inputs/dependencies fail the job.",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("native"):
        for distribution in marker.args:
            try:
                importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pytest.skip(f"optional native distribution {distribution!r} is not installed")


class _RequiredChecks:
    @staticmethod
    def _require_result(report) -> None:
        if report.skipped:
            reason = str(report.longrepr)
            report.outcome = "failed"
            report.longrepr = f"Required release-profile check skipped: {reason}"

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        self._require_result(outcome.get_result())

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector):
        outcome = yield
        self._require_result(outcome.get_result())


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--fail-on-skip"):
        config.pluginmanager.register(_RequiredChecks(), "required-release-checks")
