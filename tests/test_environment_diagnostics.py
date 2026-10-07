# SPDX-License-Identifier: MPL-2.0
"""Distribution observations report absence/drift without importing a native stack."""

from importlib import metadata
import json

import pytest
from typer.testing import CliRunner

from empirical_lawhood.api.environment import EnvironmentRoute, inspect_route_dependencies
from empirical_lawhood.cli.app import app


@pytest.mark.parametrize("installed", (None, "2.0.0", "2.4.6"))
@pytest.mark.parametrize("route", (EnvironmentRoute.REACTOR, EnvironmentRoute.PREPARED_RESPONSE))
def test_exact_route_versions_and_absence_are_not_conflated(monkeypatch, installed, route):
    def version(name):
        assert name == "numpy"
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        return installed
    monkeypatch.setattr(metadata, "version", version)
    values = inspect_route_dependencies(route)
    numpy = next(value for value in values if value.dependency_id == "numpy")
    assert numpy.available is (installed is not None)
    assert numpy.version_matches is (installed == "2.4.6")
    assert numpy.installed_version == installed
    assert numpy.required_version == "2.4.6"
    assert bool(numpy.reason_codes) is (installed != "2.4.6")


def test_cli_selected_route_is_visible_and_unknown_route_is_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["doctor", "--route", "reactor", "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)["payload"]
    assert payload["environment_route"] == "reactor"
    assert {row["dependency_id"] for row in payload["optional_dependencies"]} == {"cpython", "numpy"}
    assert payload["operating_system"] and payload["architecture"]
    assert list(tmp_path.iterdir()) == []
    invalid = CliRunner().invoke(app, ["doctor", "--route", "unknown", "--format", "json"])
    assert invalid.exit_code == 2, invalid.output
