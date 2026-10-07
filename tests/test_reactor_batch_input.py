"""The shipped full-batch selector selects the provider or refuses before contact."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_input import ReactorBatchInput, check_batch_input
from empirical_lawhood.api.codecs import load_registered_authoring

CONFIG = Path(__file__).parents[1] / "experiments/reactor-response/batch-input.json"
PACKAGED = (
    Path(__file__).parents[1] / "src/empirical_lawhood/_vendor/terminal_bench_science"
)


def test_shipped_batch_config_and_missing_upstream_refuse_before_contact(
    tmp_path: Path,
) -> None:
    config = load_registered_authoring(
        CONFIG,
        root_schemas={ReactorBatchInput.SCHEMA: ReactorBatchInput},
        maximum_bytes=16 * 1024,
    )
    with pytest.raises(ValueError, match="SOURCE_ROOT_REQUIRED"):
        check_batch_input(config, None)
    with pytest.raises(ValueError, match="solution/controller.py"):
        check_batch_input(config, PACKAGED)
    with pytest.raises(ValueError, match="horizon_s"):
        replace(config, horizon_s=1)
    with pytest.raises(ValueError, match="evidence_role"):
        replace(config, evidence_role="PROSPECTIVE")

    root = tmp_path / "upstream"
    for member in (
        "tests/plant.py",
        "environment/spec/plant_params.json",
        "environment/spec/scenarios_public.json",
    ):
        destination = root / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PACKAGED / member).read_bytes())
    (root / "tests/plant.py").write_bytes(b"not the pinned native source")
    with pytest.raises(ValueError, match="SOURCE_MEMBER_PIN_MISMATCH: tests/plant.py"):
        check_batch_input(config, root)


@pytest.mark.held
def test_authentic_batch_source_selects_registered_runner_without_contact() -> None:
    source_name = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    if not source_name:
        pytest.skip("authentic held reactor source is not mounted")
    config = load_registered_authoring(
        CONFIG,
        root_schemas={ReactorBatchInput.SCHEMA: ReactorBatchInput},
        maximum_bytes=16 * 1024,
    )
    result = check_batch_input(config, Path(source_name))
    assert result["binding_selected"] == (
        "binding.terminal-bench-science-batch.reactor-batch"
    )
    assert result["provider_built"] is True
    assert result["runner_selected"] == 'ReactorBatchRunner'
    assert result["missing_port_keys"] == ()
    assert result["native_contact"] is False
    assert result["native_tasks_executed"] == 0
