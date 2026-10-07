"""CLI prerequisites have one owner and exact installed-tree agreement.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
from typer.main import get_command

from empirical_lawhood.cli.app import app
from empirical_lawhood.cli.introspection import canonical_click_tree
from empirical_lawhood.cli.metadata import COMMAND_METADATA, GROUP_METADATA
from empirical_lawhood.cli.reference_contract import (
    CliReferenceContractError,
    join_command_metadata,
)
from scripts.generate_cli_reference import render


ROOT = Path(__file__).resolve().parents[1]


def test_reference_retains_exact_existing_prose_and_every_leaf_has_prerequisites():
    facts = canonical_click_tree(get_command(app))
    entries = join_command_metadata(facts, GROUP_METADATA, COMMAND_METADATA)
    for entry in entries:
        if entry.fact.command_kind == "command":
            assert entry.metadata.native_software
            assert entry.metadata.native_status
    assert render() == (ROOT / "docs/cli.md").read_text()


@pytest.mark.parametrize("changed", ("missing", "extra", "duplicate"))
def test_metadata_refuses_tree_disagreement(changed):
    commands = COMMAND_METADATA
    if changed == "missing":
        commands = commands[1:]
    elif changed == "extra":
        commands += (replace(commands[0], command="campaign absent-command"),)
    else:
        commands += (commands[0],)
    with pytest.raises(CliReferenceContractError):
        join_command_metadata(canonical_click_tree(get_command(app)), GROUP_METADATA, commands)


@pytest.mark.parametrize(
    ("native", "status"),
    ((None, "status"), ("software", None), ("", "status"), ("software", "  ")),
)
def test_native_status_pair_refuses_both_missing_directions_and_empty_fields(native, status):
    commands = (replace(COMMAND_METADATA[0], native_software=native, native_status=status), *COMMAND_METADATA[1:])
    with pytest.raises(CliReferenceContractError, match="incomplete native"):
        join_command_metadata(canonical_click_tree(get_command(app)), GROUP_METADATA, commands)


@pytest.mark.parametrize("script", ("generate_extension_bundle_aggregate.py", "generate_executable_binding_aggregate.py"))
@pytest.mark.parametrize("elsewhere", (False, True))
def test_generator_check_from_checkout_and_unrelated_directory(tmp_path, script, elsewhere):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--check"],
        cwd=tmp_path if elsewhere else ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
