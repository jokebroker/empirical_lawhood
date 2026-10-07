# SPDX-License-Identifier: MPL-2.0
"""Independent discrete probability checks for the fixed-sample diagnostic."""

from fractions import Fraction
from hashlib import sha256
import json
from math import comb
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.methods.finite_response_law.power import (
    fixed_sample_power_report,
    joint_success_risk_power,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import PLAN_SHA256
from empirical_lawhood.cli.app import app
from empirical_lawhood.cli import platform
from empirical_lawhood.cli.metadata import COMMAND_METADATA, invocation_contracts
from empirical_lawhood.planning.paired_power import paired_binary_power


def _category_probability(n, a, b, pa, pb):
    return comb(n, a) * comb(n - a, b) * pa**a * pb**b * (1 - pa - pb) ** (n - a - b)


@pytest.mark.parametrize(
    ("ps", "pf", "required", "maximum", "expected"),
    (
        (1, 0, 52, 2, 1),
        (0, 0, 52, 2, 0),
        (0, 1, 0, 2, 0),
        (0, 1, 0, 64, 1),
        (0, 0, 0, 0, 1),
    ),
)
def test_joint_power_limiting_categories(ps, pf, required, maximum, expected):
    assert joint_success_risk_power(64, required, maximum, ps, pf) == expected


def test_small_joint_event_and_conditional_zero_risk_identity():
    # At n=4, S>=2 and F=0 has probability 9/32 for category weights 1/2,1/4,1/4.
    assert joint_success_risk_power(4, 2, 0, 0.5, 0.25) == pytest.approx(
        Fraction(9, 32)
    )
    for success in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        marginal = sum(
            comb(8, k) * success**k * (1 - success) ** (8 - k) for k in range(5, 9)
        )
        assert joint_success_risk_power(8, 5, 0, float(success), 0) == pytest.approx(
            float(marginal)
        )


@pytest.mark.parametrize(
    "arguments",
    (
        (0, 0, 0, 0.5, 0),
        (65, 52, 2, 0.5, 0),
        (True, 0, 0, 0.5, 0),
        (64, 65, 2, 0.5, 0),
        (64, 52, -1, 0.5, 0),
        (64, 52, 2, 0.9, 0.2),
        (64, 52, 2, float("nan"), 0),
    ),
)
def test_invalid_or_unbounded_design_is_refused(arguments):
    with pytest.raises(ValueError, match="invalid bounded"):
        joint_success_risk_power(*arguments)


def test_complete_grid_against_rational_multinomial_enumeration():
    report = fixed_sample_power_report()
    assert (
        report["n"],
        report["required_successes"],
        report["maximum_false_admissions"],
    ) == (64, 52, 2)
    assert [(c["p_success"], c["p_false"]) for c in report["multinomial_cases"]] == [
        (s, f) for s in (0.80, 0.85, 0.90) for f in (0, 0.01, 0.02, 0.05)
    ]
    for case in report["multinomial_cases"]:
        ps, pf = Fraction(str(case["p_success"])), Fraction(str(case["p_false"]))
        joint = sum(
            _category_probability(64, s, f, ps, pf)
            for s in range(52, 65)
            for f in range(min(2, 64 - s) + 1)
        )
        success = sum(comb(64, s) * ps**s * (1 - ps) ** (64 - s) for s in range(52, 65))
        risk = sum(comb(64, f) * pf**f * (1 - pf) ** (64 - f) for f in range(3))
        assert case["joint_use_risk_power"] == pytest.approx(float(joint), abs=2e-14)
        assert case["success_marginal_power"] == pytest.approx(
            float(success), abs=2e-14
        )
        assert case["risk_marginal_power"] == pytest.approx(float(risk), abs=2e-14)
        assert (
            case["joint_use_risk_power"]
            <= min(case["success_marginal_power"], case["risk_marginal_power"]) + 2e-14
        )
        assert (
            case["joint_use_risk_power"]
            >= max(0, case["success_marginal_power"] + case["risk_marginal_power"] - 1)
            - 2e-14
        )


def test_paired_mapping_against_exact_rational_rejection_region():
    paired = fixed_sample_power_report()["paired_use_sensitivity"]
    assert [(c["p_improve"], c["p_deteriorate"]) for c in paired] == [
        (0.10, 0.0),
        (0.15, 0.05),
        (0.20, 0.05),
        (0.25, 0.05),
        (0.30, 0.10),
    ]
    for case in paired:
        better, worse = (
            Fraction(str(case["p_improve"])),
            Fraction(str(case["p_deteriorate"])),
        )
        expected = Fraction(0)
        for b in range(65):
            for c in range(65 - b):
                # Exact one-sided conditional binomial tail <= 1/20.
                if b > c and 20 * sum(
                    comb(b + c, k) for k in range(b, b + c + 1)
                ) <= 2 ** (b + c):
                    expected += _category_probability(64, b, c, better, worse)
        assert case["exact_paired_added_use_power"] == pytest.approx(
            float(expected), abs=3e-14
        )
    assert paired_binary_power(64, 0, 0, 0.05) == 0
    symmetric = paired_binary_power(64, 0, 0.4, 0.05)
    assert 0 < symmetric <= 0.05
    tails = [Fraction(0), Fraction(0)]
    for b in range(65):
        for c in range(65 - b):
            weight = _category_probability(64, b, c, Fraction(1, 5), Fraction(1, 5))
            for direction, wins in enumerate((b, c)):
                if 20 * sum(comb(b + c, k) for k in range(wins, b + c + 1)) <= 2 ** (
                    b + c
                ):
                    tails[direction] += weight
    assert tails[0] == tails[1]
    assert symmetric == pytest.approx(float(tails[0]), abs=3e-14)


def test_prediction_normal_boundary_and_limits_are_distinct():
    report = fixed_sample_power_report()
    values = report["prediction_comparison_sensitivity"]
    assert [v["mean_paired_improvement_over_root_sd"] for v in values] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert [v["normal_approximation_one_sided_power"] for v in values] == sorted(
        v["normal_approximation_one_sided_power"] for v in values
    )
    # Effect 0.2 gives z=-0.04485...; the standardized threshold is near zero.
    assert values[1]["normal_approximation_one_sided_power"] == pytest.approx(
        0.4821120125881396
    )
    assert "10% relevance gate" in report["limits"]
    assert "max(0,p+i-1)" in report["limits"] and "min(p,i)" in report["limits"]
    assert "No identified joint information-power distribution" in report["limits"]
    assert "no nomination" in report["evidence_ceiling"]


def test_specification_identity_and_no_contact_cli(tmp_path, monkeypatch):
    protocol = (
        Path(__file__).parents[1]
        / "src/empirical_lawhood/adapters/methods/finite_response_law/specification.md"
    )
    assert sha256(protocol.read_bytes()).hexdigest() == PLAN_SHA256

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic attempted source or application contact")

    monkeypatch.setattr(platform, "create_cli_api", forbidden)
    monkeypatch.setattr(platform, "create_inspection_api", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["campaign", "response-composition-power"])
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report == fixed_sample_power_report()
    assert report["specification_sha256"] == PLAN_SHA256
    assert report["native_contact"] is False and report["outcome_reads"] is False
    assert report["campaign_issued"] is False
    assert list(tmp_path.iterdir()) == []
    assert (
        result.stdout
        == CliRunner().invoke(app, ["campaign", "response-composition-power"]).stdout
    )
    metadata = next(
        c
        for c in COMMAND_METADATA
        if c.command == "campaign response-composition-power"
    )
    assert "without source, outcome or native contact" in metadata.effects
    assert metadata.native_software == "no native contact at this verb"
    assert metadata.native_status == "portable inspection or local validation"
    assert metadata.clean_commit == "no"
    assert invocation_contracts()[metadata.command].output == "JSON report. No --format"
