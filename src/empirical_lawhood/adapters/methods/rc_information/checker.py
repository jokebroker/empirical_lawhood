"""Independent equation, stationary-point and identical-input bound checks.

SPDX-License-Identifier: MPL-2.0
"""

from decimal import Decimal

import mpmath as mp

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import (
    RCInformationBoundCheck,
    RCInformationConfig,
    RCInformationInstanceKind,
    RCInformationResult,
)


def check_rc_information_bound(
    config: RCInformationConfig, result: RCInformationResult
) -> RCInformationBoundCheck:
    """Check the supplied result without invoking its numerical producer.

    Each branch solves C*dV/dt = -C*V/tau for the homogeneous response.
    The difference has zero initial mean, tends to zero, and has a unique
    positive stationary point when nondegenerate. The triangle inequality
    gives max(|p-y_a|, |p-y_b|) >= |y_a-y_b|/2 for every common prediction p.
    mpmath agreement is a numerical cross-check, not interval certification.
    """
    if not isinstance(config, RCInformationConfig) or not isinstance(
        result, RCInformationResult
    ):
        raise TypeError("RC bound checker requires typed config and result")
    identity_matches = result.config == ObjectIdentity.from_record(
        config.config_id, config
    )
    with mp.workdps(110):

        def number(value: Decimal) -> mp.mpf:
            return mp.mpf(str(value))

        tau_one, tau_two = map(number, config.time_constants_seconds)
        scale = number(config.voltage_scale_volts)
        state_a = tuple(map(number, config.initial_state_a_v0))
        state_b = tuple(map(number, config.initial_state_b_v0))
        coefficient_one = scale * (state_a[0] - state_b[0]) / 2
        coefficient_two = scale * (state_a[1] - state_b[1]) / 2
        time = number(result.maximizing_time_approx_seconds)
        tolerance = (
            mp.mpf("1e-70")
            * abs(scale)
            * max(mp.mpf(1), *(abs(value) for value in (*state_a, *state_b)))
        )
        initial_difference = coefficient_one + coefficient_two
        derivative = (
            -coefficient_one * mp.exp(-time / tau_one) / tau_one
            - coefficient_two * mp.exp(-time / tau_two) / tau_two
        )
        degenerate = tau_one == tau_two or coefficient_one == 0
        if degenerate:
            stationary = time == 0 and abs(initial_difference) <= tolerance
        else:
            # Independently solve the branch derivative equation; do not reuse
            # the producer's maximizing-time helper or Decimal arithmetic.
            root = mp.log(-coefficient_one * tau_two / (coefficient_two * tau_one)) / (
                1 / tau_one - 1 / tau_two
            )
            stationary = (
                time > 0
                and abs(time - root) <= mp.mpf("1e-70") * abs(root)
                and abs(derivative) <= tolerance / min(tau_one, tau_two)
            )
        response_a = (
            scale
            * sum(
                state * mp.exp(-time / tau)
                for state, tau in zip(state_a, (tau_one, tau_two), strict=True)
            )
            / 2
        )
        response_b = (
            scale
            * sum(
                state * mp.exp(-time / tau)
                for state, tau in zip(state_b, (tau_one, tau_two), strict=True)
            )
            / 2
        )
        separation = abs(response_a - response_b)
        equations_agree = all(
            (
                abs(number(result.homogeneous_response_a_approx_volts) - response_a)
                <= tolerance,
                abs(number(result.homogeneous_response_b_approx_volts) - response_b)
                <= tolerance,
                abs(
                    number(result.maximum_response_separation_approx_volts) - separation
                )
                <= tolerance,
            )
        )
        bound_agrees = (
            config.predictor_input_a_sha256 == config.predictor_input_b_sha256
            and abs(initial_difference) <= tolerance
            and abs(
                number(result.identical_input_minimax_lower_bound_approx_volts)
                - separation / 2
            )
            <= tolerance
        )
        if degenerate:
            expressions = ("0", "0", "0")
        elif config.instance_kind is RCInformationInstanceKind.BASELINE:
            expressions = (
                "4*ln(4)/3 s",
                "V0/4*(4**(-1/3)-4**(-4/3))",
                "V0/8*(4**(-1/3)-4**(-4/3))",
            )
        else:
            expressions = (
                "ln(tau2/tau1)/(1/tau1-1/tau2)",
                "V0*abs((a1-b1)/2)*abs(exp(-t_max/tau1)-exp(-t_max/tau2))",
                "maximum_response_separation/2",
            )
        expressions_agree = expressions == (
            result.maximizing_time_expression,
            result.maximum_separation_expression,
            result.minimax_lower_bound_expression,
        )
    return RCInformationBoundCheck(
        result=ObjectIdentity.from_record(f"{config.config_id}.result", result),
        checker_precision_digits=110,
        input_identity_matches=identity_matches,
        stationary_or_degenerate_maximum=bool(stationary),
        numerical_equations_agree=bool(equations_agree),
        identical_input_bound_agrees=bool(bound_agrees),
        analytical_expressions_agree=expressions_agree,
    )
