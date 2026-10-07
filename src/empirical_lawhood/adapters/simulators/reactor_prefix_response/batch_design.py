"""Preassigned public-envelope units for fresh absolute forecast qualification."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import random
from typing import ClassVar

from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal, validate_stable_id
from .contracts import PARAMS_SHA256, PLANT_SHA256, PUBLIC_SCENARIOS_SHA256
from .forecast import REFERENCE_SHA256

# Current public-schema identity of the unchanged four pinned upstream members.
BATCH_SOURCE_SHA256 = "2fc284e231c6e8ce1cdca06190e9df6516f928951a045a8553a5af653886eef4"


@dataclass(frozen=True, slots=True)
class ReactorBatchSource(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-source'
    plant_source: str
    plant_params: str
    public_scenarios: str
    reference_controller: str

    def __post_init__(self) -> None:
        for value, maximum, digest in (
            (self.plant_source, 32768, PLANT_SHA256),
            (self.plant_params, 4096, PARAMS_SHA256),
            (self.public_scenarios, 8192, PUBLIC_SCENARIOS_SHA256),
            (self.reference_controller, 16384, REFERENCE_SHA256),
        ):
            data = value.encode()
            if len(data) > maximum or sha256(data).hexdigest() != digest:
                raise ValueError("batch source changes an exact pinned upstream artifact")


@dataclass(frozen=True, slots=True)
class ReactorAssignedScenario(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-assigned-scenario'
    unit_id: str
    split: str
    seed: int
    cooling_factors: tuple[Decimal, Decimal, Decimal]
    cooling_onsets_s: tuple[Decimal, Decimal, Decimal]
    fouling_fraction: Decimal
    fouling_ramp_s: tuple[Decimal, Decimal]
    feed_temperature_offset_k: Decimal
    feed_temperature_onset_s: Decimal
    kinetic_multiplier: Decimal
    volume_exponent: Decimal
    plant_draw: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        if (
            self.split not in {"calibration", "heldout"}
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("assigned scenario requires its declared split and nonnegative seed")
        for value in (
            *self.cooling_factors,
            *self.cooling_onsets_s,
            self.fouling_fraction,
            *self.fouling_ramp_s,
            self.feed_temperature_offset_k,
            self.feed_temperature_onset_s,
            self.kinetic_multiplier,
            self.volume_exponent,
        ):
            validate_decimal(value, field_name="public-envelope-coordinate")
        if (
            len(self.cooling_factors) != 3
            or len(self.cooling_onsets_s) != 3
            or any(not Decimal(".4") <= v <= 1 for v in self.cooling_factors)
            or any(not 0 <= v <= 21600 for v in self.cooling_onsets_s)
            or any(b - a < 1800 for a, b in zip(self.cooling_onsets_s, self.cooling_onsets_s[1:]))
            or sum(b > a for a, b in zip((Decimal(1), *self.cooling_factors), self.cooling_factors))
            > 1
            or not 0 <= self.fouling_fraction <= Decimal(".4")
            or min(self.cooling_factors) * (1 - self.fouling_fraction) < Decimal(".3")
            or len(self.fouling_ramp_s) != 2
            or not 3000 <= self.fouling_ramp_s[0] <= 9000
            or not 10800 <= self.fouling_ramp_s[1] <= 18000
            or not -14 <= self.feed_temperature_offset_k <= 14
            or not 1800 <= self.feed_temperature_onset_s <= 7200
            or not Decimal(".7") <= self.kinetic_multiplier <= Decimal("1.3")
            or not Decimal("-.6") <= self.volume_exponent <= Decimal("1.2")
        ):
            raise ValueError("assigned scenario lies outside the published fault envelope")
        bands = {
            "dh": ".12",
            "ea": ".02",
            "jacket_tau": ".12",
            "k0": ".12",
            "rho_cp": ".06",
            "ua0": ".12",
        }
        if tuple(v.value_id for v in self.plant_draw) != tuple(sorted(bands)) or any(
            v.unit != "1" or abs(v.value - 1) > Decimal(bands[v.value_id]) for v in self.plant_draw
        ):
            raise ValueError("assigned scenario changes the published plant draw bands")

    def native_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.unit_id,
            "seed": self.seed,
            "cooling_capacity_factor": float(self.cooling_factors[0]),
            "cooling_onset_s": float(self.cooling_onsets_s[0]),
            "cooling_capacity_factor2": float(self.cooling_factors[1]),
            "cooling_onset2_s": float(self.cooling_onsets_s[1]),
            "cooling_capacity_factor3": float(self.cooling_factors[2]),
            "cooling_onset3_s": float(self.cooling_onsets_s[2]),
            "fouling_loss_fraction": float(self.fouling_fraction),
            "fouling_ramp_s": [float(v) for v in self.fouling_ramp_s],
            "feed_temp_offset_k": float(self.feed_temperature_offset_k),
            "feed_temp_onset_s": float(self.feed_temperature_onset_s),
            "k0_plant_multiplier": float(self.kinetic_multiplier),
            "ua_volume_exponent": float(self.volume_exponent),
            "plant_draw": {v.value_id: float(v.value) for v in self.plant_draw},
        }


def assigned_scenarios() -> tuple[ReactorAssignedScenario, ...]:
    """Frozen seed schedule; no native execution, exposed outcomes or private rows."""
    scenarios = []
    for split, count, seed_base in (("calibration", 32, 41000), ("heldout", 10, 52000)):
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
                    f"reactor-{split}-{index:03d}",
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


def validate_nominal_spec(source: ReactorBatchSource) -> None:
    spec = json.loads(source.plant_params)
    if spec["limits"]["horizon_s"] != 28800 or spec["timing"]["sample_dt_s"] != 10:
        raise ValueError("batch source changes the frozen horizon or callback clock")


@dataclass(frozen=True, slots=True)
class ReactorBatchConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-config'
    config_id: str
    scenarios: tuple[ReactorAssignedScenario, ...]
    plant_timesteps_s: tuple[Decimal, ...] = (Decimal(1), Decimal(".5"))
    horizon_s: Decimal = Decimal(28800)
    sample_dt_s: Decimal = Decimal(10)
    numpy_version: str = "2.4.6"
    python_version: str = "3.11.14"
    source_bundle_sha256: str = BATCH_SOURCE_SHA256

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.scenarios != assigned_scenarios()
            or self.plant_timesteps_s != (Decimal(1), Decimal(".5"))
            or self.horizon_s != 28800
            or self.sample_dt_s != 10
            or self.numpy_version != "2.4.6"
            or self.python_version != "3.11.14"
            or self.source_bundle_sha256 != BATCH_SOURCE_SHA256
        ):
            raise ValueError("batch configuration changes its preassigned units, views or runtime")


BRANCHES = tuple((s.unit_id, view) for s in assigned_scenarios() for view in ("native", "refined"))


def native_task_id(unit_id: str, view: str) -> str:
    if (unit_id, view) not in BRANCHES:
        raise ValueError("undeclared reactor full-batch unit/view")
    return f"batch.{unit_id}.{view}"
