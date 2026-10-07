"""Fresh fixed roots and prescribed perturbation; old batch identities are unchanged."""

from dataclasses import dataclass
from decimal import Decimal
import random
from typing import ClassVar
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256 as BATCH_SOURCE_SHA256, ReactorAssignedScenario as ReactorAssignedScenario, ReactorBatchSource as ReactorBatchSource

PULSE_ONSET = 7200
PULSE_RETURN = 7800
RECOVERY_END = 9000
PULSE_JACKET_K = Decimal("-1")


def assigned_scenarios() -> tuple[ReactorAssignedScenario, ...]:
    """Frozen seed schedule; no native execution, exposed outcomes or private rows."""
    scenarios = []
    for split, count, seed_base in (("calibration", 32, 61000), ("heldout", 10, 72000)):
        for index in range(count):
            seed = seed_base + index
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
            scenarios.append(
                ReactorAssignedScenario(
                    f"reactor-history-{split}-{index:03d}",
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
            )
    return tuple(scenarios)


@dataclass(frozen=True, slots=True)
class ReactorBatchConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-matched-replay-forecast/reactor-batch-config'
    config_id: str
    scenarios: tuple[ReactorAssignedScenario, ...]
    plant_timesteps_s: tuple[Decimal, ...] = (Decimal(1), Decimal(".5"))
    horizon_s: Decimal = Decimal(28800)
    sample_dt_s: Decimal = Decimal(10)
    numpy_version: str = "2.4.6"
    python_version: str = "3.11.14"
    source_bundle_sha256: str = BATCH_SOURCE_SHA256
    pulse_onset_s: Decimal = Decimal(PULSE_ONSET)
    pulse_return_s: Decimal = Decimal(PULSE_RETURN)
    pulse_jacket_k: Decimal = PULSE_JACKET_K
    recovery_end_s: Decimal = Decimal(RECOVERY_END)
    command_tape_rule: str = "native-donor-prescribed-before-pulse-outcomes"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.scenarios != assigned_scenarios()
            or self.pulse_onset_s != PULSE_ONSET
            or self.pulse_return_s != PULSE_RETURN
            or self.pulse_jacket_k != PULSE_JACKET_K
            or self.recovery_end_s != RECOVERY_END
            or self.command_tape_rule != "native-donor-prescribed-before-pulse-outcomes"
            or self.plant_timesteps_s != (Decimal(1), Decimal(".5"))
            or self.horizon_s != 28800
            or self.sample_dt_s != 10
            or self.numpy_version != "2.4.6"
            or self.python_version != "3.11.14"
            or self.source_bundle_sha256 != BATCH_SOURCE_SHA256
        ):
            raise ValueError("batch configuration changes its preassigned units, views or runtime")


BRANCHES = tuple((s.unit_id, "paired") for s in assigned_scenarios())


def native_task_id(unit_id: str, view: str = "paired") -> str:
    if (unit_id, view) not in BRANCHES:
        raise ValueError("undeclared history unit")
    return f"history.{unit_id}.{view}"
