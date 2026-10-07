"Method-value authority composition over the substrate-general TORAX system."

from __future__ import annotations

from empirical_lawhood.adapters.simulators.torax_native import build_native_torax_system
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.worlds import WorldKind

from .contracts import ToraxMethodValueExperimentSpec


def build_method_value_torax_system(spec: ToraxMethodValueExperimentSpec) -> SystemSpec:
    "Bind the frozen method-value authority without contaminating the native adapter."

    if spec.experiment_id != "torax-receiver-conditioned-io-method-value":
        raise ValueError("Method-value system composition requires the terminal authority identity")
    budget = ResourceBudget(
        cpu_cores=16,
        memory_bytes=64 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=86_400,
        source_scan_bytes=0,
        output_bytes=64 * 1024**3,
    )
    authority = AuthorityPolicy(
        policy_id="authority-policy.torax-native",
        delegator_id="human.project-owner",
        delegate_id='operator.torax-expression-locus-explicit-clock-method-value',
        scope_ids=("experiment.torax-receiver-conditioned-io-method-value",),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=(
            "exact-method-value-phase-authority",
            "external-custody",
            "non-actuating",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=budget,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return build_native_torax_system(spec.views, authority_policy=authority)


__all__ = ['build_method_value_torax_system']
