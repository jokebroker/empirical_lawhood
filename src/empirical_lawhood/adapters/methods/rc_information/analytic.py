"""Decimal evaluation of the two-state RC analytical illustration.

SPDX-License-Identifier: MPL-2.0
"""

from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import (
    RCInformationConfig,
    RCInformationInstanceKind,
    RCInformationResult,
)


def evaluate_rc_information(config: RCInformationConfig) -> RCInformationResult:
    """Evaluate the exact mean-response difference on t >= 0.

    Common forcing cancels between cases. Their equal initial receiver means
    make the two difference coefficients opposites. The sole interior
    stationary point, or zero in a degenerate instance, is therefore the
    global maximum of the absolute response difference.
    """
    if not isinstance(config, RCInformationConfig):
        raise TypeError("RC information calculation requires its typed config")
    with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN)):
        tau_one, tau_two = config.time_constants_seconds
        difference = (config.initial_state_a_v0[0] - config.initial_state_b_v0[0]) / 2
        if tau_one == tau_two or difference == 0:
            time = Decimal(0)
        else:
            time = (tau_two / tau_one).ln() / (1 / tau_one - 1 / tau_two)
        decay = ((-time / tau_one).exp(), (-time / tau_two).exp())
        response_a = (
            config.voltage_scale_volts
            * sum(
                state * attenuation
                for state, attenuation in zip(
                    config.initial_state_a_v0, decay, strict=True
                )
            )
            / 2
        )
        response_b = (
            config.voltage_scale_volts
            * sum(
                state * attenuation
                for state, attenuation in zip(
                    config.initial_state_b_v0, decay, strict=True
                )
            )
            / 2
        )
        separation = abs(
            config.voltage_scale_volts * difference * (decay[0] - decay[1])
        )
        if tau_one == tau_two or difference == 0:
            time_expression = "0"
            separation_expression = "0"
            bound_expression = "0"
        elif config.instance_kind is RCInformationInstanceKind.BASELINE:
            time_expression = "4*ln(4)/3 s"
            separation_expression = "V0/4*(4**(-1/3)-4**(-4/3))"
            bound_expression = "V0/8*(4**(-1/3)-4**(-4/3))"
        else:
            time_expression = "ln(tau2/tau1)/(1/tau1-1/tau2)"
            separation_expression = (
                "V0*abs((a1-b1)/2)*abs(exp(-t_max/tau1)-exp(-t_max/tau2))"
            )
            bound_expression = "maximum_response_separation/2"
        return RCInformationResult(
            config=ObjectIdentity.from_record(config.config_id, config),
            maximizing_time_approx_seconds=time,
            homogeneous_response_a_approx_volts=response_a,
            homogeneous_response_b_approx_volts=response_b,
            maximum_response_separation_approx_volts=separation,
            identical_input_minimax_lower_bound_approx_volts=separation / 2,
            maximizing_time_expression=time_expression,
            maximum_separation_expression=separation_expression,
            minimax_lower_bound_expression=bound_expression,
        )
