"""Exact seed-parameterized draw body from the installed reactor source binding."""

from decimal import Decimal
import random
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorAssignedScenario

from empirical_lawhood.adapters.methods.reactor_causal_response.config import ROLES


def scenario(role: str, index: int) -> ReactorAssignedScenario:
    count, base = next((n, b) for r, n, b in ROLES if r == role)
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("undeclared root")
    seed = base + index
    split = "heldout" if role in ("qualification", "confirmation") else "calibration"
    return draw_scenario(f"reactor-empirical-{role}-{index:03d}", split, seed)


def draw_scenario(unit_id: str, split: str, seed: int) -> ReactorAssignedScenario:
    """Same native preparation distribution, with an explicit fresh unit and seed."""
    if type(seed) is not int or seed < 0 or split not in ("calibration", "heldout"):
        raise ValueError("invalid assigned preparation identity")
    rng = random.Random(seed)

    def uniform(low: float, high: float) -> Decimal:
        return Decimal(f"{rng.uniform(low, high):.12f}")

    c1 = uniform(0.4, 1)
    c2 = uniform(0.4, 1)
    c3 = uniform(0.4, float(c2))
    t1 = uniform(0, 16000)
    t2 = uniform(float(t1) + 1800, 19800)
    t3 = uniform(float(t2) + 1800, 21600)
    fouling = uniform(0, min(0.4, 1 - 0.3 / float(min(c1, c2, c3))))
    # Round down at the combined cooling/fouling boundary.
    fouling = min(fouling, 1 - Decimal(".3") / min(c1, c2, c3))
    return ReactorAssignedScenario(
        unit_id,
        split,
        seed,
        (c1, c2, c3),
        (t1, t2, t3),
        fouling,
        (uniform(3000, 9000), uniform(10800, 18000)),
        uniform(-14, 14),
        uniform(1800, 7200),
        uniform(0.7, 1.3),
        uniform(-0.6, 1.2),
        tuple(
            NamedDecimal(key, uniform(1 - band, 1 + band), "1")
            for key, band in (
                ("dh", 0.12),
                ("ea", 0.02),
                ("jacket_tau", 0.12),
                ("k0", 0.12),
                ("rho_cp", 0.06),
                ("ua0", 0.12),
            )
        ),
    )


def roster() -> tuple[ReactorAssignedScenario, ...]:
    return tuple(scenario(role, i) for role, n, _ in ROLES for i in range(n))


def reject_overlap(retained: tuple[tuple[str, int], ...]) -> None:
    roots, seeds = {x[0] for x in retained}, {x[1] for x in retained}
    if any(s.unit_id in roots or s.seed in seeds for s in roster()):
        raise ValueError("retained acquisition overlaps assigned root/seed")
