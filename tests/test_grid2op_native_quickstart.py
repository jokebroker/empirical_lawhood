"""Held Grid2Op source, prospective boundary and native precontact checks."""

from __future__ import annotations

import bz2
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateScientificDesignBasis
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateTargetPhase
from empirical_lawhood.adapters.simulators.grid2op_response.contracts import Grid2OpActionBranchRequest, Grid2OpActionKind, Grid2OpEpisodeRequest, Grid2OpNativeAction
from empirical_lawhood.adapters.simulators.grid2op_response.design import build_grid2op_target_design
from empirical_lawhood.adapters.simulators.grid2op_response.native_quickstart import Grid2OpDevelopmentInput
from empirical_lawhood.adapters.simulators.grid2op_response.runtime import execute_grid2op_episode
from empirical_lawhood.adapters.simulators.grid2op_response.source import build_held_grid2op_source_binding
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "experiments/grid-response-inputs/config.json"


def test_shipped_grid2op_selection_refuses_missing_held_source_before_contact(
    monkeypatch,
) -> None:
    import empirical_lawhood.adapters.simulators.grid2op_response.native_quickstart as quickstart

    def unexpected_import(_distribution: str) -> str:
        raise AssertionError("simulator version checked before held source precontact")

    monkeypatch.setattr(quickstart.metadata, "version", unexpected_import)
    record = decode_canonical_bytes(
        SHIPPED.read_bytes(), Grid2OpDevelopmentInput, maximum_bytes=16 * 1024
    )
    assert record.fresh_target_identity_disjoint is False
    result = CliRunner().invoke(
        app,
        [
            "campaign",
            "grid2op-native-check",
            "--config",
            str(SHIPPED),
            "--source-root",
            str(ROOT / "absent-held-grid2op-source"),
        ],
    )
    assert result.exit_code == 3
    assert "held source root is absent" in result.output
    assert "source_binding_sha256" not in result.output


def test_grid2op_held_binding_records_non_disjoint_development_and_refuses_prospective(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "datasets" / "l2rpn_case14_sandbox"
    chronic = dataset / "chronics" / "researcher-chronic-001"
    chronic.mkdir(parents=True)
    (dataset / "grid.json").write_bytes(b"{}\n")
    (dataset / "config.py").write_bytes(b"# bounded fixture\n")
    (chronic / "time_interval.info").write_text("00:05\n", encoding="ascii")
    (chronic / "start_datetime.info").write_text("2026-01-01 00:00\n", encoding="ascii")
    for name in ("load_p", "load_q", "prod_p", "prod_v"):
        (chronic / f"{name}.csv.bz2").write_bytes(bz2.compress(b"head\n1\n2\n3\n"))
    wheel = tmp_path / "grid2op.whl"
    backend = tmp_path / "lightsim2grid.whl"
    wheel.write_bytes(b"source-wheel-fixture")
    backend.write_bytes(b"backend-wheel-fixture")
    source = build_held_grid2op_source_binding(
        binding_id="source.empirical-lawhood-grid2op-fixture",
        grid2op_version="1.12.5",
        grid2op_wheel=wheel,
        backend_version="0.13.1",
        backend_wheel=backend,
        dataset_path=dataset,
        native_chronic_ids=("researcher-chronic-001",),
        observation_operator_sha256="1" * 64,
        fresh_target_identity_disjoint=False,
    )
    assert source.fresh_target_identity_disjoint is False
    assert source.chronic_bindings[0].timestep_seconds == 300
    assert source.chronic_bindings[0].maximum_steps == 2
    with pytest.raises(ValueError, match="disjoint held source"):
        build_grid2op_target_design(
            source=source,
            common_design_basis=ObjectIdentity(
                object_id="basis.empirical-lawhood-grid2op-fixture",
                object_schema=IndependentSubstrateScientificDesignBasis.SCHEMA,
                object_version="1.0.0",
                object_fingerprint="0" * 64,
            ),
        )
    selected_chronic = source.chronic_bindings[0]
    hold = Grid2OpActionBranchRequest(
        branch_id="branch.empirical-lawhood-grid2op-hold-fixture",
        action=Grid2OpNativeAction(
            action_id="hold",
            kind=Grid2OpActionKind.HOLD,
            target_native_ids=(),
            integer_values=(),
            decimal_values=(),
            canonical_description_bits=8,
        ),
        receiver_horizon_steps=(1,),
    )
    request = Grid2OpEpisodeRequest(
        request_id="request.empirical-lawhood-grid2op-sealed-fixture",
        unit_id=selected_chronic.chronic_id,
        phase=IndependentSubstrateTargetPhase.EVALUATION,
        source_binding=ObjectIdentity.from_record(source.binding_id, source),
        chronic=ObjectIdentity.from_record(
            selected_chronic.chronic_id, selected_chronic
        ),
        environment_seed=1,
        initial_step=0,
        branches=(hold,),
        receiver_gauge_ids=("thermal", "topology"),
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )

    class CountingPort:
        calls = 0

        def source_binding(self):
            return source

        def execute_branch(self, **_kwargs):
            self.calls += 1
            raise AssertionError("sealed source contacted")

    port = CountingPort()
    with pytest.raises(ValueError, match="sealed execution requires a disjoint"):
        execute_grid2op_episode(port=port, source=source, request=request)
    assert port.calls == 0


def test_grid2op_selection_rejects_path_escape() -> None:
    record = decode_canonical_bytes(
        SHIPPED.read_bytes(), Grid2OpDevelopmentInput, maximum_bytes=16 * 1024
    )
    from dataclasses import replace

    with pytest.raises(ValueError, match="bounded relative"):
        replace(record, dataset_relative_path="../held-source")
