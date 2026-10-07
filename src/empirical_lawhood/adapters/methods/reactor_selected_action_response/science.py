"""Same native operands, fresh evidence world and selected response-bound claim."""

from dataclasses import replace

from empirical_lawhood.adapters.methods.reactor_regime_response.science import CHART as CHART, HORIZON as HORIZON, RECEIVERS as RECEIVERS, UNIT as UNIT, regime_system
from empirical_lawhood.kernel.systems import SystemSpec
from .config import PREFIX

METHOD = f"{PREFIX}.bounded-response"
CUTOFF = "reactor-classical-selected-causal-callback"
CLOCK = "reactor-clock"


def classical_system() -> SystemSpec:
    base = regime_system()
    world = replace(base.world, world_id=f"{PREFIX}.world")
    prototype = next(q for q in base.quantities if q.quantity_id == "feature-00")
    history = tuple(
        replace(
            prototype,
            quantity_id=f"domain-feature-{i:02d}",
            label=f"Frozen eligibility coordinate {i:02d}",
        )
        for i in range(23)
    )
    history += tuple(
        replace(
            prototype,
            quantity_id=key,
            label=key.replace("-", " "),
            dimension="temperature",
            native_unit="K",
        )
        for key in ("observed-temperature", "observed-jacket")
    )
    quantities = tuple(
        sorted(
            (*(q for q in base.quantities if not q.quantity_id.startswith("feature-")), *history),
            key=lambda q: q.quantity_id,
        )
    )
    return replace(
        base,
        system_id=f"{PREFIX}.system",
        label="Selected finite reactor cooling bound and prospective use",
        world=world,
        quantities=quantities,
        relation=replace(
            base.relation,
            relation_id=f"{PREFIX}.relation",
            history_quantity_ids=tuple(
                sorted(
                    (
                        *(
                            q
                            for q in base.relation.history_quantity_ids
                            if not q.startswith("feature-")
                        ),
                        *(q.quantity_id for q in history),
                    )
                )
            ),
        ),
        numerical_views=tuple(replace(v, world_id=world.world_id) for v in base.numerical_views),
        authority_policy=replace(
            base.authority_policy,
            policy_id=f"{PREFIX}.authority",
            scope_ids=(PREFIX,),
            required_gate_ids=(f"{PREFIX}.exact-route-gate", f"{PREFIX}.typed-authority-gate"),
        ),
    )
