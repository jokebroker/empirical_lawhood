"Explicit local-domain qualification scope reusing the installed native quantity/clock grammar."

from dataclasses import replace
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.adapters.methods.reactor_causal_response.science import empirical_system, CLOCK as CLOCK, UNIT as UNIT, RECEIVERS as RECEIVERS, UNITS as UNITS
from empirical_lawhood.adapters.methods.reactor_causal_response.config import EmpiricalRecipe
from .config import PREFIX as PREFIX, ROOTS

CHART = "reactor-local-measured-words"
SUPPORT = "reactor-local-discovered-support"
CUTOFF = "reactor-local-causal-callback"
METHOD = f"{PREFIX}.local-response"
QUALIFICATION_UNITS = tuple(r for r, role, _, _ in ROOTS if role == "qualification")
BUDGET = ResourceBudget(1, 8 * 1024**3, 0, 43200, 256 * 1024**3, 256 * 1024**3)
PROTOCOL_BUDGET = ResourceBudget(4, 32 * 1024**3, 0, 129600, 256 * 1024**3, 256 * 1024**3)
ASSUMPTIONS = tuple(
    sorted(
        (
            "fresh-assigned-simulation-roots",
            "frozen-causal-domain-recognition",
            "fixed-exploration-and-local-assay-population",
            "conditional-on-causal-domain-contact",
            "no-cross-domain-simultaneous-coverage",
            "no-physical-or-private-panel-transport",
        )
    )
)


def local_system(receiver: int | None = None) -> SystemSpec:
    # Reuse units, clocks, native action quantities and preparation semantics.
    # Replace scientific scope, policy and support explicitly; no R2 law or
    # qualification result is read or transported into this follow-up.
    base = empirical_system(EmpiricalRecipe())
    world = replace(
        base.world,
        world_id=f"{PREFIX}.world",
        label="Pinned reactor: fresh conditional local-law assays",
        represented_physics=(
            "native-actuator-stages",
            "native-delayed-noisy-observer",
            "native-rk4",
        ),
        maximum_evidence=EvidenceCeiling.LOCAL_LAW,
        available_outcome_access=base.world.available_outcome_access
        | {OutcomeAccess.DEVELOPMENT_VISIBLE},
    )
    compute = replace(
        base.computability_envelopes[0],
        envelope_id=f"{PREFIX}.computability",
        represented_effect_ids=("local-ten-second-response",),
        required_structure_ids=("causal-domains", "matched-input-views"),
        computable_structure_ids=("causal-domains", "matched-input-views"),
    )
    policy = replace(
        base.authority_policy,
        policy_id=f"{PREFIX}.authority",
        scope_ids=(PREFIX,),
        required_gate_ids=(
            f"{PREFIX}.exact-production-proof",
            f"{PREFIX}.typed-execution-authority",
        ),
        budget_ceiling=PROTOCOL_BUDGET,
    )
    relation = replace(
        base.relation,
        relation_id=f"{PREFIX}.relation"
        if receiver is None
        else f"{PREFIX}.receiver-{receiver}.relation",
        receiver_quantity_ids=tuple(sorted(RECEIVERS))
        if receiver is None
        else (RECEIVERS[receiver],),
        horizon=replace(base.relation.horizon, horizon_id=f"{PREFIX}.horizon"),
    )
    return replace(
        base,
        system_id=f"{PREFIX}.system"
        if receiver is None
        else f"{PREFIX}.receiver-{receiver}.system",
        label="Frozen distinct local reactor response laws",
        world=world,
        relation=relation,
        authority_policy=policy,
        computability_envelopes=(compute,),
        numerical_views=tuple(
            replace(
                view,
                world_id=world.world_id,
                boundary_condition_ids=(SUPPORT,),
                computability_envelope_id=compute.envelope_id,
            )
            for view in base.numerical_views
        ),
    )
