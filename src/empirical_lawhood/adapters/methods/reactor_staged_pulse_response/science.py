"""Native local/induced-history relations with explicitly named raw receiver maps."""

from dataclasses import replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.reactor_selected_action_response.science import classical_system
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import CHART
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.quantities import ResponseDirection
from .config import Coordinate, PREFIX, ClassicalStage
from .measurement import receiver_ids
from .prediction import HORIZON

METHOD = f"{PREFIX}.bounded-response"
CUTOFF = "reactor-staged-pulse-response-pre-response-native-callback"
UNIT = "reactor-assigned-scenario-seed"
CLOCK = "reactor-clock"
LOCAL_CLOCK = "reactor-staged-pulse-response-local-clock"
LOCAL_FRAME = "reactor-staged-pulse-response-local-window"


def stage_system(stage: ClassicalStage) -> SystemSpec:
    return staged_pulse_reactor_system(
        stage.coordinates[-1] if stage.block == "staged-sequence-comparison" else stage.coordinates[0]
    )


def staged_pulse_reactor_system(coordinate: Coordinate) -> SystemSpec:
    base = classical_system()
    stem = f"{PREFIX}.{coordinate.kind}"
    response = next(q for q in base.quantities if q.quantity_id == "reactor-feed-cooling")
    context = next(q for q in base.quantities if q.quantity_id == "domain-feature-00")
    labels = {
        "c": "Raw zero-reference minus action peak in the specified local window",
        "c1": "Raw first 10-second counterfactual cooling",
        "c2": "Raw conditional second 10-second counterfactual cooling",
        "g": "Raw zero-reference minus actual 240-second episode peak",
        "s": "Actual action guard peak temperature",
        "s2": "Actual second 120-second guard peak temperature",
    }
    quantities = (
        tuple(
            q
            for q in base.quantities
            if q.quantity_id not in ("reactor-feed-cooling", "reactor-peak-temperature")
        )
        + tuple(
            replace(
                response,
                quantity_id=k,
                label=labels[k],
                response_direction=ResponseDirection.LOWER_IS_BETTER
                if k.startswith("s")
                else ResponseDirection.HIGHER_IS_BETTER,
            )
            for k in receiver_ids(coordinate)
        )
        + (
            replace(
                context,
                quantity_id="classical-context",
                label="Frozen preparation or exact induced-history context",
            ),
            replace(
                context,
                quantity_id="initial-temperature",
                label="Permitted observation at the first native cutoff",
                dimension=response.dimension,
                native_unit=response.native_unit,
            ),
            replace(
                context,
                quantity_id="first-callback-time",
                label="First native cutoff",
                dimension="time",
                native_unit="s",
            ),
        )
    )
    world = replace(base.world, world_id=f"{PREFIX}.world")
    horizon = D(240 if coordinate.kind == "baseline" else 120)
    relation = replace(
        base.relation,
        relation_id=f"{stem}.relation",
        receiver_quantity_ids=tuple(sorted(receiver_ids(coordinate))),
        history_quantity_ids=tuple(
            sorted(
                (
                    *base.relation.history_quantity_ids,
                    "classical-context",
                    "initial-temperature",
                    "first-callback-time",
                )
            )
        ),
        horizon=replace(base.relation.horizon, horizon_id=HORIZON, duration=horizon),
    )
    envelope = replace(
        base.computability_envelopes[0],
        envelope_id=f"{stem}.computability",
        represented_effect_ids=("finite-native-feed-response-with-explicit-receiver-map",),
    )
    return replace(
        base,
        system_id=f"{stem}.system",
        label="Classical EL finite reactor service relation",
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
            budget_ceiling=ResourceBudget(
                4, 32 * 1024**3, 0, 54 * 3600, 512 * 1024**3, 96 * 1024**3
            ),
            required_gate_ids=(f"{PREFIX}.exact-route-gate", f"{PREFIX}.typed-authority-gate"),
        ),
    )
