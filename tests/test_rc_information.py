"""Independent analytical RC controls; no simulator or scientific campaign.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import replace
from decimal import Decimal, ROUND_UP, getcontext, localcontext
import math

import mpmath as mp
import pytest

from empirical_lawhood.adapters.methods.rc_information.checker import (
    check_rc_information_bound,
)
from empirical_lawhood.adapters.methods.rc_information.contracts import (
    RCInformationConfig,
    RCInformationInstanceKind,
)
from empirical_lawhood.api.rc_information import (
    RCInformationReport,
    calculate_rc_information,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes


def baseline(scale: str = "1") -> RCInformationConfig:
    # An explicitly synthetic common predictor-input identity; no source or
    # qualification receipt is required by this analytical calculation.
    return RCInformationConfig(
        config_id="rc-information.baseline",
        instance_kind=RCInformationInstanceKind.BASELINE,
        voltage_scale_volts=Decimal(scale),
        equal_branch_capacitance_farads=Decimal("0.001"),
        time_constants_seconds=(Decimal(1), Decimal(4)),
        initial_state_a_v0=(Decimal(1), Decimal(1)),
        initial_state_b_v0=(Decimal("0.5"), Decimal("1.5")),
        predictor_input_a_sha256="a" * 64,
        predictor_input_b_sha256="a" * 64,
    )


def test_baseline_matches_independent_mean_response_and_stationary_condition():
    report = calculate_rc_information(baseline())
    result = report.result
    # Independent binary evaluation of the analytical branch equations.
    expected_time = 4 * math.log(4) / 3
    expected_a = (math.exp(-expected_time) + math.exp(-expected_time / 4)) / 2
    expected_b = (
        0.5 * math.exp(-expected_time) + 1.5 * math.exp(-expected_time / 4)
    ) / 2
    assert float(result.maximizing_time_approx_seconds) == pytest.approx(
        expected_time, rel=1e-14
    )
    assert float(result.homogeneous_response_a_approx_volts) == pytest.approx(
        expected_a, rel=1e-14
    )
    assert float(result.homogeneous_response_b_approx_volts) == pytest.approx(
        expected_b, rel=1e-14
    )
    assert float(result.maximum_response_separation_approx_volts) == pytest.approx(
        abs(expected_a - expected_b), rel=1e-14
    )
    assert float(
        result.identical_input_minimax_lower_bound_approx_volts
    ) == pytest.approx(abs(expected_a - expected_b) / 2, rel=1e-14)
    derivative = math.exp(-expected_time) - math.exp(-expected_time / 4) / 4
    assert derivative == pytest.approx(0, abs=1e-15)
    assert result.maximizing_time_expression == "4*ln(4)/3 s"
    assert result.maximum_separation_expression == "V0/4*(4**(-1/3)-4**(-4/3))"
    assert report.check.passed
    assert result.analytical_construction is True
    assert result.numerical_interval_certified is False


def test_positive_voltage_scale_changes_the_actual_bound_without_source_edits():
    one = calculate_rc_information(baseline("1"))
    three = calculate_rc_information(baseline("3"))
    assert three.result.config != one.result.config
    assert (
        three.result.maximizing_time_approx_seconds
        == one.result.maximizing_time_approx_seconds
    )
    assert float(
        three.result.maximum_response_separation_approx_volts
    ) == pytest.approx(
        3 * float(one.result.maximum_response_separation_approx_volts), rel=1e-14
    )
    assert float(
        three.result.identical_input_minimax_lower_bound_approx_volts
    ) == pytest.approx(
        3 * float(one.result.identical_input_minimax_lower_bound_approx_volts),
        rel=1e-14,
    )


def test_swapped_states_preserve_bound_and_swap_responses():
    original = calculate_rc_information(baseline())
    swapped_config = replace(
        baseline(),
        config_id="rc-information.swapped",
        instance_kind=RCInformationInstanceKind.VARIANT,
        initial_state_a_v0=baseline().initial_state_b_v0,
        initial_state_b_v0=baseline().initial_state_a_v0,
    )
    swapped = calculate_rc_information(swapped_config)
    assert (
        swapped.result.homogeneous_response_a_approx_volts
        == original.result.homogeneous_response_b_approx_volts
    )
    assert (
        swapped.result.homogeneous_response_b_approx_volts
        == original.result.homogeneous_response_a_approx_volts
    )
    assert (
        swapped.result.maximum_response_separation_approx_volts
        == original.result.maximum_response_separation_approx_volts
    )
    assert (
        swapped.result.identical_input_minimax_lower_bound_approx_volts
        == original.result.identical_input_minimax_lower_bound_approx_volts
    )


@pytest.mark.parametrize("degeneracy", ("equal-time-constants", "identical-states"))
def test_indistinguishable_responses_have_valid_zero_bound(degeneracy):
    changes = (
        {"time_constants_seconds": (Decimal(2), Decimal(2))}
        if degeneracy == "equal-time-constants"
        else {"initial_state_b_v0": baseline().initial_state_a_v0}
    )
    config = replace(
        baseline(),
        config_id=f"rc-information.{degeneracy}",
        instance_kind=RCInformationInstanceKind.VARIANT,
        **changes,
    )
    report = calculate_rc_information(config)
    assert report.result.maximizing_time_approx_seconds == 0
    assert report.result.maximum_response_separation_approx_volts == 0
    assert report.result.identical_input_minimax_lower_bound_approx_volts == 0
    assert report.check.passed


def test_time_constant_variant_preserves_independent_equation_agreement():
    config = replace(
        baseline(),
        config_id="rc-information.time-variant",
        instance_kind=RCInformationInstanceKind.VARIANT,
        time_constants_seconds=(Decimal(2), Decimal(8)),
    )
    report = calculate_rc_information(config)
    expected_time = 8 * math.log(4) / 3
    expected_separation = (
        math.exp(-expected_time / 8) - math.exp(-expected_time / 2)
    ) / 4
    assert float(report.result.maximizing_time_approx_seconds) == pytest.approx(
        expected_time, rel=1e-14
    )
    assert float(
        report.result.maximum_response_separation_approx_volts
    ) == pytest.approx(expected_separation, rel=1e-14)
    assert report.input.instance_kind is RCInformationInstanceKind.VARIANT


def test_initial_state_variant_changes_separation_with_same_initial_receiver():
    original = calculate_rc_information(baseline())
    config = replace(
        baseline(),
        config_id="rc-information.state-variant",
        instance_kind=RCInformationInstanceKind.VARIANT,
        initial_state_b_v0=(Decimal(0), Decimal(2)),
    )
    report = calculate_rc_information(config)
    assert float(
        report.result.maximum_response_separation_approx_volts
    ) == pytest.approx(
        2 * float(original.result.maximum_response_separation_approx_volts), rel=1e-14
    )


def test_equal_capacitance_scale_cancels_when_time_constants_are_fixed():
    config = baseline()
    original = calculate_rc_information(config)
    scaled = calculate_rc_information(
        replace(config, equal_branch_capacitance_farads=Decimal("0.004"))
    )
    assert scaled.result.config != original.result.config
    assert (
        scaled.result.maximum_response_separation_approx_volts
        == original.result.maximum_response_separation_approx_volts
    )


def test_changed_baseline_states_or_time_constants_require_variant_identity():
    with pytest.raises(ValueError, match="VARIANT"):
        replace(baseline(), time_constants_seconds=(Decimal(2), Decimal(8)))
    with pytest.raises(ValueError, match="VARIANT"):
        replace(baseline(), initial_state_b_v0=(Decimal(1), Decimal(1)))


def test_different_predictor_inputs_refuse_the_identical_input_argument():
    with pytest.raises(ValueError, match="same predictor input"):
        replace(baseline(), predictor_input_b_sha256="b" * 64)


def test_different_initial_receiver_means_refuse_the_common_input_argument():
    with pytest.raises(ValueError, match="equal initial receiver"):
        replace(
            baseline(),
            instance_kind=RCInformationInstanceKind.VARIANT,
            initial_state_b_v0=(Decimal(1), Decimal(2)),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"voltage_scale_volts": Decimal(0)},
        {"equal_branch_capacitance_farads": Decimal("-1")},
        {"time_constants_seconds": (Decimal(1), Decimal(0))},
        {"initial_state_a_v0": (Decimal("NaN"), Decimal(1))},
        {"instance_kind": "BASELINE"},
        {"voltage_unit": "mV"},
        {"receiver_id": "other-receiver"},
    ),
)
def test_invalid_operands_refuse_at_the_constructor(changes):
    with pytest.raises(ValueError):
        replace(baseline(), **changes)


def test_checker_rejects_a_fabricated_bound_and_wrong_input_identity(monkeypatch):
    config = baseline()
    report = calculate_rc_information(config)

    # An independent checker must not obtain its oracle from the producer.
    def refuse_calculation(*args, **kwargs):
        raise AssertionError("checker called the analytical producer")

    monkeypatch.setattr(
        "empirical_lawhood.adapters.methods.rc_information.analytic.evaluate_rc_information",
        refuse_calculation,
    )
    assert check_rc_information_bound(config, report.result).passed
    fabricated = replace(
        report.result, identical_input_minimax_lower_bound_approx_volts=Decimal(1)
    )
    assert not check_rc_information_bound(config, fabricated).passed
    wrong_input = replace(config, config_id="rc-information.other-input")
    assert not check_rc_information_bound(
        wrong_input, report.result
    ).input_identity_matches


def test_checker_rejects_nonmaximal_time_and_wrong_response():
    config = baseline()
    result = calculate_rc_information(config).result
    check = check_rc_information_bound(
        config, replace(result, maximizing_time_approx_seconds=Decimal(0))
    )
    assert not check.stationary_or_degenerate_maximum
    wrong_response = replace(result, homogeneous_response_a_approx_volts=Decimal(0))
    assert not check_rc_information_bound(
        config, wrong_response
    ).numerical_equations_agree


def test_small_voltage_scale_does_not_hide_a_zeroed_bound():
    config = baseline("1e-100")
    result = calculate_rc_information(config).result
    assert result.identical_input_minimax_lower_bound_approx_volts > 0
    check = check_rc_information_bound(
        config,
        replace(result, identical_input_minimax_lower_bound_approx_volts=Decimal(0)),
    )
    assert not check.identical_input_bound_agrees


def test_checker_rejects_a_changed_exact_expression_and_report_substitution():
    config = baseline()
    report = calculate_rc_information(config)
    changed = replace(
        report.result, minimax_lower_bound_expression="maximum_response_separation"
    )
    assert not check_rc_information_bound(config, changed).analytical_expressions_agree
    with pytest.raises(ValueError, match="result/check binding"):
        replace(report, result=changed)


def test_common_prediction_triangle_bound_has_a_saturating_midpoint():
    result = calculate_rc_information(baseline()).result
    left = float(result.homogeneous_response_a_approx_volts)
    right = float(result.homogeneous_response_b_approx_volts)
    bound = abs(left - right) / 2
    for prediction in (-10, left, right, 0, (left + right) / 2, 10):
        assert max(abs(prediction - left), abs(prediction - right)) >= bound - 1e-15
    assert max(
        abs((left + right) / 2 - left), abs((left + right) / 2 - right)
    ) == pytest.approx(bound, abs=1e-15)


def test_decimal_and_mpmath_contexts_are_restored_and_do_not_choose_the_result():
    config = baseline()
    expected = calculate_rc_information(config)
    with localcontext() as context, mp.workdps(17):
        context.prec = 7
        context.rounding = ROUND_UP
        before = (
            context.prec,
            context.rounding,
            context.Emin,
            context.Emax,
            dict(context.flags),
            dict(context.traps),
            mp.mp.dps,
        )
        observed = calculate_rc_information(config)
        after = (
            getcontext().prec,
            getcontext().rounding,
            getcontext().Emin,
            getcontext().Emax,
            dict(getcontext().flags),
            dict(getcontext().traps),
            mp.mp.dps,
        )
        assert after == before
    assert observed == expected
    assert observed.canonical_bytes() == expected.canonical_bytes()


def test_report_exact_bytes_round_trip_without_sample_or_qualification_claims():
    report = calculate_rc_information(baseline())
    payload = report.canonical_bytes()
    assert (
        decode_canonical_bytes(payload, RCInformationReport, maximum_bytes=len(payload))
        == report
    )
    document = report.to_document()
    assert "sample_count" not in document["value"]
    assert "confidence_interval" not in document["value"]
