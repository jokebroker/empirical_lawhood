"The regime-response study's joint local response relation and scalar native action."

from dataclasses import replace

from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe
from empirical_lawhood.adapters.methods.reactor_causal_response.science import empirical_system
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind
from empirical_lawhood.kernel.systems import SystemSpec

from .config import PREFIX

METHOD = f"{PREFIX}.local-response"
RECEIVERS = ("reactor-peak-temperature", "reactor-feed-cooling")
FIXED_JACKET = "reactor-fixed-jacket"
POLICY_DENOMINATOR = "reactor-regime-preparation-policy-version"
UNIT = "reactor-assigned-scenario-seed"
HORIZON = "reactor-local-ten-second"
CHART = "reactor-regime-response-fixed-jacket-feed-chart"


def regime_system() -> SystemSpec:
    """Keep the jacket as an explicit denominator, never a zero action word."""
    base = empirical_system(EmpiricalRecipe())
    temperature = next(
        value for value in base.quantities
        if value.quantity_id == "reactor-peak-temperature"
    )
    jacket_action = next(
        value for value in base.quantities
        if value.quantity_id == "reactor-jacket-command"
    )
    denominator = next(
        value for value in base.quantities
        if value.quantity_id == "empirical-policy-version"
    )
    cooling = replace(
        temperature,
        quantity_id=RECEIVERS[1],
        label="Paired zero-feed minus candidate ten-second peak temperature",
    )
    jacket = replace(
        jacket_action,
        quantity_id=FIXED_JACKET,
        label="Fixed absolute native jacket command",
        kind=QuantityKind.DENOMINATOR,
        availability=denominator.availability,
    )
    policy_denominator = replace(
        denominator,
        quantity_id=POLICY_DENOMINATOR,
        label="Frozen exploration-unshifted or selected C/P preparation policy version",
    )
    world = replace(base.world, world_id=f"{PREFIX}.world")
    envelope = replace(
        base.computability_envelopes[0],
        envelope_id=f"{PREFIX}.computability",
        represented_effect_ids=("local-ten-second-paired-response",),
        required_structure_ids=("causal-domains", "matched-input-views"),
        computable_structure_ids=("causal-domains", "matched-input-views"),
    )
    relation = replace(
        base.relation,
        relation_id=f"{PREFIX}.relation",
        denominator_quantity_ids=tuple(sorted((POLICY_DENOMINATOR, FIXED_JACKET))),
        history_quantity_ids=tuple(
            value for value in base.relation.history_quantity_ids
            if value not in {"action-id", *(f"feature-{index:02d}" for index in range(18, 23))}
        ),
        action_quantity_ids=("reactor-feed",),
        receiver_quantity_ids=tuple(sorted(RECEIVERS)),
        horizon=replace(base.relation.horizon, horizon_id=HORIZON),
    )
    return replace(
        base,
        system_id=f"{PREFIX}.system",
        label="Frozen local paired-cooling reactor response law",
        world=world,
        relation=relation,
        quantities=tuple(sorted((
            *(value for value in base.quantities if value.quantity_id not in (
                "reactor-jacket-command", "reactor-end-conversion", "action-id",
                "empirical-policy-version",
                *(f"feature-{index:02d}" for index in range(18, 23)),
            )),
            cooling, jacket, policy_denominator,
        ), key=lambda value: value.quantity_id)),
        ports=tuple(value for value in base.ports if value.quantity_id == "reactor-feed"),
        computability_envelopes=(envelope,),
        numerical_views=tuple(
            replace(
                value,
                world_id=world.world_id,
                boundary_condition_ids=(CHART,),
                computability_envelope_id=envelope.envelope_id,
            )
            for value in base.numerical_views
        ),
        authority_policy=replace(
            base.authority_policy,
            policy_id=f"{PREFIX}.authority",
            scope_ids=(PREFIX,),
            allowed_actions=frozenset((
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
            )),
            required_gate_ids=(
                f"{PREFIX}.exact-route-gate",
                f"{PREFIX}.typed-authority-gate",
            ),
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        ),
    )
