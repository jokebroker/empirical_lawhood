"""Focused installed version and native input/output checks for retained independent-substrate solvers."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponsePhase
from empirical_lawhood.adapters.simulators.cantera_reaction_response.design import cantera_design, cantera_preparation_freeze
from empirical_lawhood.adapters.methods.selective_dependence_response.preparation_inputs import SelectiveDependenceResponsePreparationInput
from empirical_lawhood.adapters.simulators.cantera_reaction_response.preparation_inputs import DRAW_VARIABLE_IDS as CANTERA_DRAW_VARIABLE_IDS, cantera_preparation_input
from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.preparation_inputs import DRAW_VARIABLE_IDS as FIPY_DRAW_VARIABLE_IDS, fipy_preparation_input
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.cantera_reaction_response.runtime import execute_cantera_complete_unit
from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.design import fipy_design, fipy_preparation_freeze
from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.runtime import execute_fipy_complete_unit
from empirical_lawhood.adapters.simulators.reaction_diffusion_development_input import ReactionDiffusionDevelopmentInput
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]


def _fixture_preparation(design, unit_id: str, full_seed: int) -> SelectiveDependenceResponsePreparationInput:
    return SelectiveDependenceResponsePreparationInput(
        complete_unit_id=unit_id,
        target_id=design.target_id,
        target_design=ObjectIdentity.from_record(design.design_id, design),
        phase="CANARY",
        roster_index=0,
        full_seed=full_seed,
        draw_variable_ids=CANTERA_DRAW_VARIABLE_IDS if design.target_id.startswith("target.cantera") else FIPY_DRAW_VARIABLE_IDS,
    )


def _shipped_config(slug: str) -> ReactionDiffusionDevelopmentInput:
    bundle = {"cantera": "reactor-flow-response", "fipy": "reaction-diffusion-response"}[slug]
    return decode_canonical_bytes(
        (ROOT / "experiments" / bundle / "config.json").read_bytes(),
        ReactionDiffusionDevelopmentInput,
        maximum_bytes=32 * 1024,
    )


@pytest.mark.native('cantera')
def test_cantera_native_canary_binds_version_action_and_receivers() -> None:
    cantera = pytest.importorskip("cantera")
    design = _shipped_config("cantera").design
    assert cantera.__version__ == design.cantera_version == "3.2.0"
    unit_id = cantera_preparation_freeze().canary_unit_ids[0]
    result = execute_cantera_complete_unit(
        design, complete_unit_id=unit_id, phase=SelectiveDependenceResponsePhase.CANARY,
        preparation_input=cantera_preparation_input(design, unit_id, SelectiveDependenceResponsePhase.CANARY),
    )
    assert result.complete_unit_id == unit_id
    assert len(result.conditions) == 32
    assert result.nested_conditions_count_as_units is False
    assert all(not condition.stopped for condition in result.conditions)
    assert {observation.receiver_id for condition in result.conditions for observation in condition.receivers} >= {"temperature", "element-error"}
    assert {condition.action.action_id for condition in result.conditions}


@pytest.mark.native('fipy')
def test_fipy_native_canary_binds_version_action_and_receivers() -> None:
    fipy = pytest.importorskip("fipy")
    design = _shipped_config("fipy").design
    assert fipy.__version__ == design.fipy_version == "4.0.3"
    unit_id = fipy_preparation_freeze().canary_unit_ids[0]
    result = execute_fipy_complete_unit(
        design, complete_unit_id=unit_id, phase=SelectiveDependenceResponsePhase.CANARY,
        preparation_input=fipy_preparation_input(design, unit_id, SelectiveDependenceResponsePhase.CANARY),
    )
    assert result.complete_unit_id == unit_id
    assert len(result.conditions) == 32
    assert result.nested_conditions_count_as_units is False
    assert all(not condition.stopped for condition in result.conditions)
    assert {observation.receiver_id for condition in result.conditions for observation in condition.receivers} >= {"downstream-mean", "balance-error"}
    assert {condition.action.action_id for condition in result.conditions}


@pytest.mark.native('fipy')
def test_fipy_source_mass_balance_and_outside_action_refusal() -> None:
    pytest.importorskip("fipy")
    from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.runtime import _condition, _preparation

    design = _shipped_config("fipy").design
    preparation = _preparation(_fixture_preparation(design, "unit.public-fipy-science-check", 318843877258697519703146587715971708824))
    common = {
        "denominator_id": "diffusivity-high",
        "history_id": "cleared-checkpoint",
        "horizon_id": "long",
    }
    hold, _ = _condition(design, preparation, action_id="source-hold", **common)
    inject, _ = _condition(design, preparation, action_id="source-inject", **common)
    outside, _ = _condition(design, preparation, action_id="source-outside", **common)
    assert not any(value.stopped for value in (hold, inject, outside))
    hold_receivers = {item.receiver_id: item.value for item in hold.receivers}
    inject_receivers = {item.receiver_id: item.value for item in inject.receivers}
    outside_receivers = {item.receiver_id: item.value for item in outside.receivers}
    assert hold_receivers == outside_receivers
    assert hold_receivers["field-mass"] == 0
    assert 0 < inject_receivers["field-mass"] < inject.action.realized_value
    assert inject_receivers["downstream-mean"] > 0
    assert inject_receivers["balance-error"] < design.maximum_balance_error
    cell_width = design.domain_length / design.mesh_cells
    source_cells = sum(
        Decimal("0.15") <= (index + Decimal("0.5")) * cell_width <= Decimal("0.25")
        for index in range(design.mesh_cells)
    )
    expected_mass = design.action_rates[2] * source_cells * cell_width * design.action_duration_seconds
    assert inject.action.realized_value == expected_mass
    assert outside.action.requested_value < 0
    assert outside.action.accepted_value == outside.action.applied_value == outside.action.realized_value == 0
    assert outside.action.acceptance_state == "rejected-to-hold"
    assert inject.action.realized_clock == design.action_duration_seconds
    assert inject.action.realized_unit == "field-mass"


@pytest.mark.native('cantera')
def test_cantera_flow_integral_temperature_response_and_outside_refusal() -> None:
    pytest.importorskip("cantera")
    from empirical_lawhood.adapters.simulators.cantera_reaction_response.runtime import _condition, _preparation

    design = _shipped_config("cantera").design
    preparation = _preparation(_fixture_preparation(design, "unit.public-cantera-science-check", 78329071824047932297957754850949812751))
    common = {
        "denominator_id": "heat-loss-low",
        "history_id": "cold-checkpoint",
        "horizon_id": "short",
    }
    hold, _ = _condition(design, preparation, action_id="flow-hold", **common)
    high, _ = _condition(design, preparation, action_id="flow-high", **common)
    outside, _ = _condition(design, preparation, action_id="flow-outside", **common)
    assert not any(value.stopped for value in (hold, high, outside))
    hold_receivers = {item.receiver_id: item.value for item in hold.receivers}
    high_receivers = {item.receiver_id: item.value for item in high.receivers}
    outside_receivers = {item.receiver_id: item.value for item in outside.receivers}
    assert outside_receivers == hold_receivers
    assert high.action.applied_value > hold.action.applied_value
    assert high.action.realized_value == pytest.approx(
        high.action.applied_value * design.horizon_seconds[0], abs=Decimal("1e-12")
    )
    assert high_receivers["temperature"] < hold_receivers["temperature"]
    assert high_receivers["element-error"] < design.maximum_element_error
    assert outside.action.requested_value > outside.action.accepted_value
    assert outside.action.accepted_value == hold.action.accepted_value
    assert outside.action.acceptance_state == "rejected-to-hold"
    assert high.action.realized_unit == "kilogram"
    assert high.action.realized_clock == design.horizon_seconds[0]


@pytest.mark.native('fipy')
def test_independent_substrate_native_quickstart_refuses_wrong_solver_before_native_run(tmp_path) -> None:
    pytest.importorskip("fipy")
    design = replace(
        fipy_design(),
        design_id="empirical-lawhood-fipy-version-refusal-design",
        fipy_version="0.0.1",
    )
    config = ReactionDiffusionDevelopmentInput(
        config_id="empirical-lawhood-fipy-version-refusal",
        independent_unit_id="unit.empirical-lawhood-fipy-version-refusal-001",
        design=design,
        preparation_input=_fixture_preparation(design, "unit.empirical-lawhood-fipy-version-refusal-001", 0),
    )
    path = tmp_path / "config.json"
    path.write_bytes(config.canonical_bytes())
    result = CliRunner().invoke(
        app, ["campaign", 'reaction-response-native-check', "--config", str(path)]
    )
    assert result.exit_code == 3
    assert "installed FiPy version differs" in result.output
    assert "native_result_sha256" not in result.output


def test_independent_substrate_native_quickstart_refuses_unbounded_or_changed_chart() -> None:
    with pytest.raises(ValueError, match="resource bound"):
        ReactionDiffusionDevelopmentInput(
            config_id="empirical-lawhood-fipy-unbounded",
            independent_unit_id="unit.empirical-lawhood-fipy-unbounded-001",
            design=replace(
                fipy_design(),
                design_id="empirical-lawhood-fipy-unbounded-design",
                mesh_cells=121,
            ),
        )
    with pytest.raises(ValueError, match="axes differ"):
        ReactionDiffusionDevelopmentInput(
            config_id="empirical-lawhood-cantera-wrong-chart",
            independent_unit_id="unit.empirical-lawhood-cantera-wrong-chart-001",
            design=replace(
                cantera_design(),
                design_id="empirical-lawhood-cantera-wrong-chart-design",
                action_ids=("flow-high", "flow-hold", "flow-low", "wrong-action"),
            ),
        )
