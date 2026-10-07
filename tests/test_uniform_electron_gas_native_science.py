"""uniform electron gas SI transverse response, slab, action-stage and refusal checks."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from math import cosh, pi
from pathlib import Path

import pytest
from scipy.constants import elementary_charge, hbar, m_e, mu_0, physical_constants
from typer.testing import CliRunner

from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.native_quickstart import UniformElectronGasAnalyticReferenceConfig, build_analytic_development_rows, run_analytic_development_check
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.physics import finite_q_estimate, penetration_depth_from_kernel, row_map, ueg_scales
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes


def _config() -> UniformElectronGasAnalyticReferenceConfig:
    return decode_canonical_bytes(
        Path("experiments/electron-gas-response/config.json").read_bytes(),
        UniformElectronGasAnalyticReferenceConfig,
        maximum_bytes=32 * 1024,
    )


def test_ueg_si_scale_and_finite_q_intercept_against_free_electron_formula() -> None:
    config = _config()
    scales = ueg_scales(config.r_s)
    a0 = physical_constants["Bohr radius"][0]
    expected_density = 3.0 / (4.0 * pi * (4 * a0) ** 3)
    expected_kf = (3.0 * pi**2 * expected_density) ** (1.0 / 3.0)
    expected_ef = hbar**2 * expected_kf**2 / (2.0 * m_e)
    assert float(scales.density_m3) == pytest.approx(expected_density, rel=1e-14)
    assert float(scales.k_f_m1) == pytest.approx(expected_kf, rel=1e-14)
    assert float(scales.e_f_eV) == pytest.approx(
        expected_ef / elementary_charge, rel=1e-14
    )

    rows = build_analytic_development_rows(config)
    assert len(rows) == 20
    mapped = row_map(rows)
    expected_k0 = 1.0 / (mu_0 * float(config.reference_penetration_depth_m) ** 2)
    for q in config.q_over_kf:
        estimate = finite_q_estimate(
            panel_id=config.config_id,
            q_over_kf=q,
            rows=mapped,
            config=config,
            action_direction=config.action_direction,
        )
        assert float(estimate.kernel_full) == pytest.approx(
            expected_k0 * (1 + 4 * float(q) ** 2), rel=1e-14
        )
        assert estimate.locality_pass and estimate.even_remainder_pass
        assert estimate.zero_offset_pass
    report = run_analytic_development_check(config)
    assert report["independent_units"] == 1
    assert report["nested_q_u_conditions"] == 20
    assert float(report["kernel_intercept_A_T_m3"]) == pytest.approx(
        expected_k0, rel=1e-14
    )
    assert float(report["fitted_penetration_depth_m"]) == pytest.approx(1e-7, rel=1e-14)
    assert float(report["slab_centre_field_ratio"]) == pytest.approx(
        1 / cosh(2.5), rel=1e-14
    )
    assert report["campaign_candidate_compiled"] is False


def test_realized_vector_potential_controls_kernel_not_requested_or_applied() -> None:
    config = _config()
    rows = build_analytic_development_rows(config)
    q = config.q_over_kf[0]
    original = finite_q_estimate(
        panel_id=config.config_id,
        q_over_kf=q,
        rows=row_map(rows),
        config=config,
        action_direction=config.action_direction,
    )
    changed = tuple(
        replace(
            row,
            realized_A_T=tuple(component / 2 for component in row.realized_A_T),
        )
        if row.q_over_kf == q
        else row
        for row in rows
    )
    assert changed[0].requested_A_T == changed[0].applied_A_T
    assert changed[0].realized_A_T != changed[0].applied_A_T
    estimate = finite_q_estimate(
        panel_id=config.config_id,
        q_over_kf=q,
        rows=row_map(changed),
        config=config,
        action_direction=config.action_direction,
    )
    assert estimate.kernel_full == 2 * original.kernel_full
    assert (
        changed[0].requested_clock,
        changed[0].accepted_clock,
        changed[0].applied_clock,
        changed[0].receiver_clock,
    ) == (0, 1, 2, 3)


def test_wrong_sign_current_cannot_become_a_penetration_depth() -> None:
    config = _config()
    q = config.q_over_kf[0]
    rows = build_analytic_development_rows(config)
    wrong_sign = tuple(
        replace(row, current_density=tuple(-value for value in row.current_density))
        if row.q_over_kf == q
        else row
        for row in rows
    )
    estimate = finite_q_estimate(
        panel_id=config.config_id,
        q_over_kf=q,
        rows=row_map(wrong_sign),
        config=config,
        action_direction=config.action_direction,
    )
    assert estimate.kernel_full < 0
    with pytest.raises(ValueError, match="positive stiffness"):
        penetration_depth_from_kernel(estimate.kernel_full)


def test_nontransverse_or_overfield_input_refuses_before_a_result(
    tmp_path: Path,
) -> None:
    config = _config()
    with pytest.raises(ValueError, match="not orthogonal"):
        replace(config, q_direction=config.action_direction)
    with pytest.raises(ValueError, match="clocks must be ordered"):
        replace(config, receiver_clock=config.applied_clock)
    overfield = replace(config, field_ceiling_T=Decimal("1e-9"))
    with pytest.raises(ValueError, match="exceeds the predeclared ceiling"):
        run_analytic_development_check(overfield)
    path = tmp_path / "overfield.json"
    path.write_bytes(overfield.canonical_bytes())
    result = CliRunner().invoke(
        app, ["campaign", 'electron-gas-reference-check', "--config", str(path)]
    )
    assert result.exit_code == 3
    assert "exceeds the predeclared ceiling" in result.output
    assert "kernel_intercept_A_T_m3" not in result.output
