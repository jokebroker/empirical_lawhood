"""Joint 120 s intervention relation with explicitly nested response windows."""

from dataclasses import replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.reactor_selected_action_response.science import classical_system
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import CHART as CHART, RECEIVER
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.authority import ResourceBudget
from .config import PREFIX
from .selection import HORIZON_ID

CLOCK = "reactor-clock"
CUTOFF = "reactor-frontier-precommitted-causal-callback"
UNIT = "reactor-assigned-scenario-seed"
METHOD = f"{PREFIX}.bounded-response"
RECEIVERS = ("reactor-peak-temperature", RECEIVER)
HORIZON = HORIZON_ID
LOCAL_CLOCK = "reactor-frontier-local-guard-clock"
LOCAL_FRAME = "reactor-frontier-local-guard-window"


def frontier_system() -> SystemSpec:
    base = classical_system()
    quantities = tuple(
        replace(
            q,
            quantity_id=RECEIVER,
            label="Zero-reference minus pulse peak over the payload's explicit response window",
        )
        if q.quantity_id == "reactor-feed-cooling"
        else q
        for q in base.quantities
    )
    prototype = next(q for q in quantities if q.quantity_id == "domain-feature-00")
    quantities += (
        replace(prototype, quantity_id="frontier-context", label="Frozen exploration-unshifted context index"),
    )
    world = replace(base.world, world_id=f"{PREFIX}.world")
    envelope = replace(
        base.computability_envelopes[0],
        envelope_id=f"{PREFIX}.computability",
        represented_effect_ids=("finite-pulse-nested-peak-response-and-120s-safety",),
    )
    relation = replace(
        base.relation,
        relation_id=f"{PREFIX}.relation",
        receiver_quantity_ids=tuple(sorted(RECEIVERS)),
        history_quantity_ids=tuple(
            sorted((*base.relation.history_quantity_ids, "frontier-context"))
        ),
        horizon=replace(base.relation.horizon, horizon_id=HORIZON, duration=D(120)),
    )
    return replace(
        base,
        system_id=f"{PREFIX}.system",
        label="Finite empirical reactor pulse frontier",
        world=world,
        quantities=tuple(sorted(quantities, key=lambda q: q.quantity_id)),
        relation=relation,
        computability_envelopes=(envelope,),
        numerical_views=tuple(
            replace(
                v,
                world_id=world.world_id,
                boundary_condition_ids=(CHART,),
                computability_envelope_id=envelope.envelope_id,
            )
            for v in base.numerical_views
        ),
        authority_policy=replace(
            base.authority_policy,
            policy_id=f"{PREFIX}.authority",
            scope_ids=(PREFIX,),
            budget_ceiling=ResourceBudget(4, 32 * 1024**3, 0, 108 * 3600, 512 * 1024**3, 128 * 1024**3),
            required_gate_ids=(f"{PREFIX}.exact-route-gate", f"{PREFIX}.typed-authority-gate"),
        ),
    )
