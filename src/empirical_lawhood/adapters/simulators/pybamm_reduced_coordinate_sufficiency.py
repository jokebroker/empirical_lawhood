"""PyBaMM battery reduced observation finite reduced-coordinate sufficiency experiment.

This module owns the frozen battery reduced observation denominator/history chart,
preparation rosters, source acquisition, direct exact-restart sentinel, reduced
coordinates, donor prediction and pure adjudication.  It deliberately does
not import the immutable battery electrothermal or battery exact restart semantic implementations.

The scientific unit is a freshly generated ``(initial SOC, initial
temperature)`` preparation.  Denominators, histories, numerical views,
receiver clocks and chart coordinates are nested observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
from hashlib import sha256
import json
import math
import time
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


BATTERY_REDUCED_OBSERVATION_PLAN_ID = "battery-reduced-observation-sufficiency"
BATTERY_REDUCED_OBSERVATION_DESIGN_ID = "design.battery-reduced-observation.sufficiency"
BATTERY_REDUCED_OBSERVATION_PROVIDER_KEY = "open-sim.battery-reduced-observation"
BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION = "26.6.2.0"
BATTERY_REDUCED_OBSERVATION_PARAMETER_SET = "Chen2020"
BATTERY_REDUCED_OBSERVATION_EXTERNAL_ROOT = "runs/battery-reduced-observation-sufficiency"
BATTERY_REDUCED_OBSERVATION_ROUTE_ID = "LOSSLESS-TARGET-DIRECT"
BATTERY_REDUCED_OBSERVATION_MINIMUM_FREE_BYTES = 5 * 1024**3
BATTERY_REDUCED_OBSERVATION_MAXIMUM_JSON_BYTES = 64 * 1024**2
BATTERY_REDUCED_OBSERVATION_CAPACITY_FLOOR_AH = Decimal("1e-10")
BATTERY_REDUCED_OBSERVATION_TEMPERATURE_FLOOR_K = Decimal("1e-8")
BATTERY_REDUCED_OBSERVATION_VOLTAGE_FLOOR_V = Decimal("1e-8")
BATTERY_REDUCED_OBSERVATION_CAPACITY_MATERIALITY_AH = Decimal("0.001")
BATTERY_REDUCED_OBSERVATION_TEMPERATURE_MATERIALITY_K = Decimal("0.005")
BATTERY_REDUCED_OBSERVATION_VOLTAGE_MATERIALITY_V = Decimal("0.005")
BATTERY_REDUCED_OBSERVATION_SUPPORT_RADIUS = Decimal("1")
BATTERY_REDUCED_OBSERVATION_SCALE_FLOOR_FACTOR = Decimal("1e-12")
BATTERY_REDUCED_OBSERVATION_FAMILY_ALPHA = Decimal("0.010")
_Q = Decimal("0.0000001")


class BatteryReducedObservationStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"
    RESERVE = "RESERVE"


class BatteryReducedObservationDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    DIRECT_RESTART_FAILURE = "DIRECT_RESTART_FAILURE"
    IDENTITY_INVALID = "IDENTITY_INVALID"
    RECEIVER_INVALID = "RECEIVER_INVALID"
    TERMINATED = "TERMINATED"


class BatteryReducedObservationVerdict(StrEnum):
    EXACT_REFERENCE_DRIFT = "BATTERY_REDUCED_OBSERVATION_EXACT_REFERENCE_DRIFT"
    UNIFORM_SUFFICIENCY = "BATTERY_REDUCED_OBSERVATION_REDUCED_CHART_UNIFORMLY_PREDICTIVE_ON_FROZEN_SUPPORT"
    DENOMINATOR_CONDITIONAL = "BATTERY_REDUCED_OBSERVATION_REDUCED_CHART_DENOMINATOR_CONDITIONAL"
    HISTORY_CONDITIONAL = "BATTERY_REDUCED_OBSERVATION_REDUCED_CHART_HISTORY_CONDITIONAL"
    LOSS_RECURRENT = "BATTERY_REDUCED_OBSERVATION_REDUCED_COORDINATE_LOSS_RECURRENT"
    MIXED = "BATTERY_REDUCED_OBSERVATION_REDUCED_COORDINATE_MIXED"
    PARTIAL = "BATTERY_REDUCED_OBSERVATION_REDUCED_COORDINATE_PARTIAL"
    UNEVALUABLE = "BATTERY_REDUCED_OBSERVATION_REDUCED_COORDINATE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-numerical-view'

    view_id: str
    solver: str
    output_interval_s: int
    rtol: Decimal
    atol: Decimal
    spatial_points: int

    def __post_init__(self) -> None:
        expected = {
            "CASADI": (
                "view.pybamm.casadi-coarse",
                10,
                Decimal("1e-5"),
                Decimal("1e-6"),
                20,
            ),
            "IDAKLU": (
                "view.pybamm.idaklu-refined",
                1,
                Decimal("1e-7"),
                Decimal("1e-8"),
                30,
            ),
        }.get(self.solver)
        if (
            expected is None
            or (
                self.view_id,
                self.output_interval_s,
                self.rtol,
                self.atol,
                self.spatial_points,
            )
            != expected
        ):
            raise ValueError("battery reduced observation numerical view differs from the frozen chart")


def numerical_views() -> tuple[BatteryReducedObservationNumericalView, ...]:
    return (
        BatteryReducedObservationNumericalView(
            view_id="view.pybamm.casadi-coarse",
            solver="CASADI",
            output_interval_s=10,
            rtol=Decimal("1e-5"),
            atol=Decimal("1e-6"),
            spatial_points=20,
        ),
        BatteryReducedObservationNumericalView(
            view_id="view.pybamm.idaklu-refined",
            solver="IDAKLU",
            output_interval_s=1,
            rtol=Decimal("1e-7"),
            atol=Decimal("1e-8"),
            spatial_points=30,
        ),
    )


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationDenominator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-denominator'

    denominator_id: str
    model: str
    thermal: str
    view: BatteryReducedObservationNumericalView
    source_version: str = BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION
    parameter_set: str = BATTERY_REDUCED_OBSERVATION_PARAMETER_SET

    def __post_init__(self) -> None:
        expected = (
            f"denominator.pybamm.{self.model.lower()}.{self.thermal.lower()}."
            f"{self.view.view_id.removeprefix('view.pybamm.')}"
        )
        if (
            self.denominator_id != expected
            or self.model not in {"SPME", "DFN"}
            or self.thermal not in {"ISOTHERMAL", "LUMPED"}
            or self.source_version != BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION
            or self.parameter_set != BATTERY_REDUCED_OBSERVATION_PARAMETER_SET
        ):
            raise ValueError("battery reduced observation denominator differs from the frozen chart")


def denominators() -> tuple[BatteryReducedObservationDenominator, ...]:
    values = []
    for model in ("SPME", "DFN"):
        for thermal in ("ISOTHERMAL", "LUMPED"):
            for view in numerical_views():
                values.append(
                    BatteryReducedObservationDenominator(
                        denominator_id=(
                            f"denominator.pybamm.{model.lower()}."
                            f"{thermal.lower()}."
                            f"{view.view_id.removeprefix('view.pybamm.')}"
                        ),
                        model=model,
                        thermal=thermal,
                        view=view,
                    )
                )
    return tuple(sorted(values, key=lambda item: item.denominator_id))


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationHistory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-history'

    history_id: str
    word_id: str
    current_delta_a: tuple[Decimal, ...]
    ambient_delta_k: tuple[Decimal, ...]
    checkpoint_s: int
    end_s: int

    def __post_init__(self) -> None:
        validate_stable_id(self.history_id, field_name="history_id")
        validate_stable_id(self.word_id, field_name="word_id")
        if (
            len(self.current_delta_a) != 5
            or len(self.ambient_delta_k) != 5
            or self.checkpoint_s not in {300, 450}
            or self.end_s - self.checkpoint_s != 300
        ):
            raise ValueError("battery reduced observation history differs from the frozen chart")

    def values_at(self, clock_s: int) -> tuple[Decimal, Decimal]:
        boundaries = (0, 150, 300, 450, 600, 900)
        if not 0 <= clock_s < 900:
            raise ValueError("clock leaves the frozen action chart")
        index = next(index for index, stop in enumerate(boundaries[1:]) if clock_s < stop)
        return self.current_delta_a[index], self.ambient_delta_k[index]


def histories() -> tuple[BatteryReducedObservationHistory, ...]:
    zero = (Decimal(0),) * 5
    rows = (
        (
            "i-contiguous-matched-end",
            (0, 1, 1, 0, 0),
            zero,
            450,
            750,
        ),
        ("i-plus-600", (1, 1, 1, 1, 0), zero, 300, 600),
        ("i-split", (1, 0, 1, 0, 0), zero, 450, 750),
        ("i-then-t", (1, 0, 0, 0, 0), (0, Decimal("0.5"), 0, 0, 0), 300, 600),
        ("t-then-i", (0, 1, 0, 0, 0), (Decimal("0.5"), 0, 0, 0, 0), 300, 600),
    )
    return tuple(
        BatteryReducedObservationHistory(
            history_id=f"history.battery-reduced-observation.{token}.{checkpoint}.{end}",
            word_id=f"word.battery-electrothermal.{token}",
            current_delta_a=tuple(Decimal(value) for value in current),
            ambient_delta_k=tuple(Decimal(value) for value in ambient),
            checkpoint_s=checkpoint,
            end_s=end,
        )
        for token, current, ambient, checkpoint, end in rows
    )


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationChart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-chart'

    chart_id: str
    coordinate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        require_sorted_unique_strings(
            tuple(sorted(self.coordinate_ids)),
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.coordinate_ids) not in {4, 5, 6, 7}:
            raise ValueError("battery reduced observation chart dimension is outside the frozen roster")


def charts() -> tuple[BatteryReducedObservationChart, ...]:
    base = (
        "soc-coordinate",
        "cumulative-discharged-capacity-ah",
        "mean-temperature-k",
        "terminal-voltage-v",
    )
    return (
        BatteryReducedObservationChart("chart.battery-reduced-observation.gauge", base),
        BatteryReducedObservationChart(
            "chart.battery-reduced-observation.electrolyte",
            (*base, "electrolyte-concentration-contrast"),
        ),
        BatteryReducedObservationChart(
            "chart.battery-reduced-observation.electrochemical",
            (
                *base,
                "electrolyte-concentration-contrast",
                "negative-particle-surface-stoichiometry-contrast",
            ),
        ),
        BatteryReducedObservationChart(
            "chart.battery-reduced-observation.electrothermal",
            (
                *base,
                "electrolyte-concentration-contrast",
                "negative-particle-surface-stoichiometry-contrast",
                "total-heating",
            ),
        ),
    )


_STRATA = (
    ('low-state-of-charge-cool', Decimal("0.25"), Decimal("0.40"), Decimal("291.15"), Decimal("293.15")),
    ('low-state-of-charge-warm', Decimal("0.25"), Decimal("0.40"), Decimal("295.15"), Decimal("297.15")),
    ('middle-state-of-charge-cool', Decimal("0.45"), Decimal("0.60"), Decimal("291.15"), Decimal("293.15")),
    ('middle-state-of-charge-warm', Decimal("0.45"), Decimal("0.60"), Decimal("295.15"), Decimal("297.15")),
    ('high-state-of-charge-cool', Decimal("0.65"), Decimal("0.80"), Decimal("291.15"), Decimal("293.15")),
    ('high-state-of-charge-warm', Decimal("0.65"), Decimal("0.80"), Decimal("295.15"), Decimal("297.15")),
)

_FROZEN_STRATUM_HASH_OPERANDS = (
    (115, 49, 45, 108, 111, 119, 45, 99, 111, 111, 108),
    (115, 50, 45, 108, 111, 119, 45, 119, 97, 114, 109),
    (115, 51, 45, 109, 105, 100, 100, 108, 101, 45, 99, 111, 111, 108),
    (115, 52, 45, 109, 105, 100, 100, 108, 101, 45, 119, 97, 114, 109),
    (115, 53, 45, 104, 105, 103, 104, 45, 99, 111, 111, 108),
    (115, 54, 45, 104, 105, 103, 104, 45, 119, 97, 114, 109),
)


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-preparation'

    unit_id: str
    stage: BatteryReducedObservationStage
    stratum_id: str
    initial_soc: Decimal
    initial_temperature_k: Decimal
    reserve: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        validate_decimal(self.initial_soc, field_name="initial_soc", minimum=Decimal("0.25"))
        validate_decimal(
            self.initial_temperature_k,
            field_name="initial_temperature_k",
            minimum=Decimal("291.15"),
        )
        if self.initial_soc > Decimal("0.80") or self.initial_temperature_k > Decimal("297.15"):
            raise ValueError("battery reduced observation preparation leaves the frozen support")
        if self.reserve is not (self.stage is BatteryReducedObservationStage.RESERVE):
            raise ValueError("battery reduced observation reserve flag differs from stage")


def _fraction(seed: int, *tokens: object) -> Decimal:
    payload = b":".join((str(seed).encode(), *(token if isinstance(token, bytes) else str(token).encode() for token in tokens)))
    numerator = int.from_bytes(sha256(payload).digest()[:8], "big")
    return Decimal(numerator) / Decimal(2**64)


def _stratified_roster(
    *,
    stage: BatteryReducedObservationStage,
    per_stratum: int,
    seed: int,
    token: str,
) -> tuple[BatteryReducedObservationPreparation, ...]:
    values: list[BatteryReducedObservationPreparation] = []
    for stratum_index, (stratum_id, soc_lo, soc_hi, temp_lo, temp_hi) in enumerate(
        _STRATA,
        start=1,
    ):
        ranks = list(range(per_stratum))
        ranks.sort(key=lambda index: _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "perm", index))
        for index in range(per_stratum):
            soc_fraction = (
                Decimal(index) + _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "soc", index)
            ) / Decimal(per_stratum)
            temp_fraction = (
                Decimal(ranks[index]) + _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "temperature", index)
            ) / Decimal(per_stratum)
            values.append(
                BatteryReducedObservationPreparation(
                    unit_id=(f"unit.battery-reduced-observation.{token}.s{stratum_index:02d}.{index + 1:02d}"),
                    stage=stage,
                    stratum_id=stratum_id,
                    initial_soc=(soc_lo + (soc_hi - soc_lo) * soc_fraction).quantize(
                        _Q, rounding=ROUND_HALF_EVEN
                    ),
                    initial_temperature_k=(temp_lo + (temp_hi - temp_lo) * temp_fraction).quantize(
                        _Q, rounding=ROUND_HALF_EVEN
                    ),
                    reserve=stage is BatteryReducedObservationStage.RESERVE,
                )
            )
    return tuple(sorted(values, key=lambda item: item.unit_id))


def qualification_preparations() -> tuple[BatteryReducedObservationPreparation, ...]:
    seed = 20_260_729_101
    values = []
    for index, stratum_index in enumerate((0, 5), start=1):
        stratum_id, soc_lo, soc_hi, temp_lo, temp_hi = _STRATA[stratum_index]
        values.append(
            BatteryReducedObservationPreparation(
                unit_id=f"unit.battery-reduced-observation.qualification.{index:02d}",
                stage=BatteryReducedObservationStage.QUALIFICATION,
                stratum_id=stratum_id,
                initial_soc=(soc_lo + (soc_hi - soc_lo) * _fraction(seed, index, "soc")).quantize(
                    _Q,
                    rounding=ROUND_HALF_EVEN,
                ),
                initial_temperature_k=(
                    temp_lo + (temp_hi - temp_lo) * _fraction(seed, index, "temperature")
                ).quantize(_Q, rounding=ROUND_HALF_EVEN),
                reserve=False,
            )
        )
    return tuple(values)


def development_preparations() -> tuple[BatteryReducedObservationPreparation, ...]:
    return _stratified_roster(
        stage=BatteryReducedObservationStage.DEVELOPMENT,
        per_stratum=2,
        seed=20_260_729_102,
        token="development",
    )


def evaluation_preparations() -> tuple[BatteryReducedObservationPreparation, ...]:
    return _stratified_roster(
        stage=BatteryReducedObservationStage.EVALUATION,
        per_stratum=2,
        seed=20_260_729_103,
        token="evaluation",
    )


def reserve_preparations() -> tuple[BatteryReducedObservationPreparation, ...]:
    return _stratified_roster(
        stage=BatteryReducedObservationStage.RESERVE,
        per_stratum=1,
        seed=20_260_729_104,
        token="reserve",
    )


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-design'

    design_id: str
    qualification_units: tuple[BatteryReducedObservationPreparation, ...]
    development_units: tuple[BatteryReducedObservationPreparation, ...]
    evaluation_units: tuple[BatteryReducedObservationPreparation, ...]
    reserve_units: tuple[BatteryReducedObservationPreparation, ...]
    denominators: tuple[BatteryReducedObservationDenominator, ...]
    histories: tuple[BatteryReducedObservationHistory, ...]
    charts: tuple[BatteryReducedObservationChart, ...]
    formal_gap_count: int
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        if (
            self.design_id != BATTERY_REDUCED_OBSERVATION_DESIGN_ID
            or len(self.qualification_units) != 2
            or len(self.development_units) != 12
            or len(self.evaluation_units) != 12
            or len(self.reserve_units) != 6
            or len(self.denominators) != 8
            or len(self.histories) != 5
            or len(self.charts) != 4
            or self.formal_gap_count != 48
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
        ):
            raise ValueError("battery reduced observation design differs from the frozen experiment")
        require_sorted_unique_ids(
            self.denominators,
            attribute="denominator_id",
            field_name="denominators",
        )
        all_units = (
            *self.qualification_units,
            *self.development_units,
            *self.evaluation_units,
            *self.reserve_units,
        )
        require_sorted_unique_strings(
            tuple(sorted(unit.unit_id for unit in all_units)),
            field_name="unit_ids",
        )
        coordinates = tuple((unit.initial_soc, unit.initial_temperature_k) for unit in all_units)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("battery reduced observation preparation coordinates collide")

    @property
    def nonreserve_record_count(self) -> int:
        return (
            (
                len(self.qualification_units)
                + len(self.development_units)
                + len(self.evaluation_units)
            )
            * len(self.denominators)
            * len(self.histories)
        )

    @property
    def nonreserve_operation_count(self) -> int:
        return 2 * self.nonreserve_record_count


def build_design() -> BatteryReducedObservationDesign:
    return BatteryReducedObservationDesign(
        design_id=BATTERY_REDUCED_OBSERVATION_DESIGN_ID,
        qualification_units=qualification_preparations(),
        development_units=development_preparations(),
        evaluation_units=evaluation_preparations(),
        reserve_units=reserve_preparations(),
        denominators=denominators(),
        histories=histories(),
        charts=charts(),
        formal_gap_count=48,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


def units_for_stage(design: BatteryReducedObservationDesign, stage: BatteryReducedObservationStage) -> tuple[BatteryReducedObservationPreparation, ...]:
    return {
        BatteryReducedObservationStage.QUALIFICATION: design.qualification_units,
        BatteryReducedObservationStage.DEVELOPMENT: design.development_units,
        BatteryReducedObservationStage.EVALUATION: design.evaluation_units,
        BatteryReducedObservationStage.RESERVE: design.reserve_units,
    }[stage]


def intended_keys(
    design: BatteryReducedObservationDesign,
    stage: BatteryReducedObservationStage,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (unit.unit_id, denominator.denominator_id, history.history_id)
        for unit in units_for_stage(design, stage)
        for denominator in design.denominators
        for history in design.histories
    )


def validate_repository_config(document: Mapping[str, object]) -> None:
    design = build_design()
    expected = {
        "campaign_token": "BATTERY-REDUCED-OBSERVATION-PYBAMM-REDUCED-COORDINATE-SUFFICIENCY",
        "candidate_charts": [
            {
                "chart_id": chart.chart_id,
                "coordinates": list(chart.coordinate_ids),
            }
            for chart in design.charts
        ],
        "claim_ceiling": (
            "FINITE_DENOMINATOR_HISTORY_LOCAL_LAW_PREDICTIVE_SUFFICIENCY_OR_OBSTRUCTION"
        ),
        "denominator_ids": [denominator.denominator_id for denominator in design.denominators],
        "descriptive_resampling": {
            "family_alpha": "0.010",
            "method": "PREPARATION_STRATIFIED_EXACT_BOOTSTRAP_ENUMERATION",
            "multiplicity": ("BONFERRONI_OVER_320_SELECTED_CHART_RECEIVER_STATISTICS"),
            "replicates": 4_096,
            "seed": "NOT_APPLICABLE_EXACT_ENUMERATION",
        },
        "design_id": BATTERY_REDUCED_OBSERVATION_DESIGN_ID,
        "external_root": BATTERY_REDUCED_OBSERVATION_EXTERNAL_ROOT,
        "formal_gap_count": 48,
        "histories": [
            [history.word_id, history.checkpoint_s, history.end_s] for history in design.histories
        ],
        "local_support_radius": "1",
        "materiality": {
            "capacity_ah": "0.001",
            "maximum_temperature_k": "0.005",
            "mean_temperature_k": "0.005",
            "terminal_voltage_v": "0.005",
        },
        "native_threads_per_worker": 1,
        "parameter_set": BATTERY_REDUCED_OBSERVATION_PARAMETER_SET,
        "predictor": {
            "distance": "NONCOMPENSATING_MAX_DEVELOPMENT_MAD",
            "donor_count": 1,
            "donor_scope": "SAME_DENOMINATOR_AND_HISTORY",
            "evaluation_catalogue": "FROZEN_DEVELOPMENT_ONLY",
            "gauge": "STRICT_FUTURE_CHECKPOINT_RELATIVE_INCREMENT",
            "tie_break": "LEXICAL_DEVELOPMENT_UNIT_ID",
        },
        "receiver_floors": {
            "capacity_ah": "1e-10",
            "maximum_temperature_k": "1e-8",
            "mean_temperature_k": "1e-8",
            "terminal_voltage_v": "1e-8",
        },
        "rosters": {
            "development_count": 12,
            "development_seed": 20_260_729_102,
            "evaluation_count": 12,
            "evaluation_seed": 20_260_729_103,
            "qualification_count": 2,
            "qualification_seed": 20_260_729_101,
            "reserve_count": 6,
            "reserve_seed": 20_260_729_104,
        },
        "route_id": BATTERY_REDUCED_OBSERVATION_ROUTE_ID,
        "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/config',
        "source_version": BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION,
        "version": "1.0.0",
    }
    if dict(document) != expected:
        raise ValueError("battery reduced observation repository config differs from the frozen chart")


def design_calculation() -> dict[str, object]:
    design = build_design()
    rosters = {stage.value: len(units_for_stage(design, stage)) for stage in BatteryReducedObservationStage}
    stage_records = {stage.value: len(intended_keys(design, stage)) for stage in BatteryReducedObservationStage}
    return {
        "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/outcome-blind-design-calculation',
        "version": "1.0.0",
        "value": {
            "design_id": design.design_id,
            "design_sha256": design.fingerprint(),
            "roster_counts": rosters,
            "stage_record_counts": stage_records,
            "nonreserve_record_count": design.nonreserve_record_count,
            "nonreserve_operation_count": design.nonreserve_operation_count,
            "maximum_reserve_operation_count": 2 * stage_records[BatteryReducedObservationStage.RESERVE.value],
            "chart_dimensions": {
                chart.chart_id: len(chart.coordinate_ids) for chart in design.charts
            },
            "outcome_access": OutcomeAccess.OUTCOME_BLIND.value,
        },
    }


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("battery reduced observation produced a nonfinite scalar")
    return Decimal(repr(float(value)))


def _hex_array(values: np.ndarray | Sequence[float]) -> list[str]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("battery reduced observation produced a nonfinite numeric payload")
    return [float(value).hex() for value in array]


def decode_hex(values: Sequence[str]) -> np.ndarray:
    array = np.asarray([float.fromhex(value) for value in values], dtype=np.float64)
    if [float(value).hex() for value in array] != list(values):
        raise ValueError("battery reduced observation lossless numeric round trip changed")
    return array


def _lookup(
    design: BatteryReducedObservationDesign,
    unit_id: str,
    denominator_id: str,
    history_id: str,
) -> tuple[BatteryReducedObservationPreparation, BatteryReducedObservationDenominator, BatteryReducedObservationHistory]:
    units = (
        *design.qualification_units,
        *design.development_units,
        *design.evaluation_units,
        *design.reserve_units,
    )
    try:
        unit = next(item for item in units if item.unit_id == unit_id)
        denominator = next(
            item for item in design.denominators if item.denominator_id == denominator_id
        )
        history = next(item for item in design.histories if item.history_id == history_id)
    except StopIteration as error:
        raise ValueError("battery reduced observation acquisition key leaves the frozen design") from error
    return unit, denominator, history


def _source_options(denominator: BatteryReducedObservationDenominator) -> dict[str, str]:
    values = {
        "thermal": denominator.thermal.lower(),
        "cell geometry": "pouch",
    }
    if denominator.thermal == "ISOTHERMAL":
        values["calculate heat source for isothermal models"] = "true"
    return values


def _inputs(
    unit: BatteryReducedObservationPreparation,
    current_delta_a: Decimal,
    ambient_delta_k: Decimal,
) -> dict[str, float]:
    return {
        "Current function [A]": float(current_delta_a),
        "Ambient temperature [K]": float(unit.initial_temperature_k + ambient_delta_k),
    }


def _runtime_objects(
    unit: BatteryReducedObservationPreparation,
    denominator: BatteryReducedObservationDenominator,
) -> tuple[Any, Any]:
    import pybamm  # type: ignore[import-untyped]

    if pybamm.__version__ != BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION:
        raise RuntimeError("installed PyBaMM differs from the frozen battery reduced observation source")
    model_class = pybamm.lithium_ion.SPMe if denominator.model == "SPME" else pybamm.lithium_ion.DFN
    model = model_class(options=_source_options(denominator))
    parameters = pybamm.ParameterValues(BATTERY_REDUCED_OBSERVATION_PARAMETER_SET)
    parameters["Current function [A]"] = "[input]"
    parameters["Ambient temperature [K]"] = "[input]"
    parameters["Initial temperature [K]"] = float(unit.initial_temperature_k)
    if denominator.view.solver == "CASADI":
        solver = pybamm.CasadiSolver(
            mode="safe",
            rtol=float(denominator.view.rtol),
            atol=float(denominator.view.atol),
        )
    else:
        solver = pybamm.IDAKLUSolver(
            rtol=float(denominator.view.rtol),
            atol=float(denominator.view.atol),
        )
    points = denominator.view.spatial_points
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        solver=solver,
        var_pts={
            "x_n": points,
            "x_s": points,
            "x_p": points,
            "r_n": points,
            "r_p": points,
        },
    )
    return pybamm, simulation


def _step_prefix(
    unit: BatteryReducedObservationPreparation,
    denominator: BatteryReducedObservationDenominator,
    history: BatteryReducedObservationHistory,
) -> tuple[Any, Any, Any]:
    pybamm, simulation = _runtime_objects(unit, denominator)
    initial = _inputs(unit, *history.values_at(0))
    simulation.build(initial_soc=float(unit.initial_soc), inputs=initial)
    solution = None
    for start, stop in zip((0, 150, 300), (150, 300, 450), strict=True):
        if start >= history.checkpoint_s:
            break
        duration = min(stop, history.checkpoint_s) - start
        values = _inputs(unit, *history.values_at(start))
        solution = simulation.step(
            duration,
            t_eval=np.asarray([0.0, float(duration)], dtype=np.float64),
            starting_solution=solution,
            inputs=values,
        )
    if solution is None or str(solution.termination) != "final time":
        raise RuntimeError("prefix did not reach the checkpoint")
    return pybamm, simulation, solution


def _scalar_series(solution: Any, variable: str, times: np.ndarray) -> np.ndarray:
    values = np.asarray(solution[variable](times), dtype=np.float64)
    return values.reshape(-1)


def _field_series(solution: Any, variable: str, times: np.ndarray) -> np.ndarray:
    values = np.asarray(solution[variable](times), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values.reshape(-1, times.size)


def _receiver_values(solution: Any, times: np.ndarray) -> dict[str, np.ndarray]:
    temperature = _field_series(solution, "Cell temperature [K]", times)
    return {
        "capacity_ah": _scalar_series(solution, "Discharge capacity [A.h]", times),
        "mean_temperature_k": _scalar_series(
            solution,
            "Volume-averaged cell temperature [K]",
            times,
        ),
        "maximum_temperature_k": np.max(temperature, axis=0),
        "terminal_voltage_v": _scalar_series(solution, "Terminal voltage [V]", times),
    }


def _checkpoint_coordinates(
    solution: Any,
    *,
    unit: BatteryReducedObservationPreparation,
    checkpoint_s: int,
) -> dict[str, str]:
    times = np.asarray([float(checkpoint_s)], dtype=np.float64)
    receivers = _receiver_values(solution, times)
    electrolyte = _field_series(
        solution,
        "Electrolyte concentration [mol.m-3]",
        times,
    )
    stoichiometry = _field_series(
        solution,
        "Negative particle surface stoichiometry",
        times,
    )
    total_heating = _scalar_series(
        solution,
        "Volume-averaged total heating [W.m-3]",
        times,
    )
    capacity = float(receivers["capacity_ah"][0])
    values = {
        "soc-coordinate": float(unit.initial_soc) - capacity / 5.0,
        "cumulative-discharged-capacity-ah": capacity,
        "mean-temperature-k": float(receivers["mean_temperature_k"][0]),
        "terminal-voltage-v": float(receivers["terminal_voltage_v"][0]),
        "electrolyte-concentration-contrast": float(
            np.max(electrolyte[:, 0]) - np.min(electrolyte[:, 0])
        ),
        "negative-particle-surface-stoichiometry-contrast": float(
            np.max(stoichiometry[:, 0]) - np.min(stoichiometry[:, 0])
        ),
        "total-heating": float(total_heating[0]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("checkpoint coordinate is nonfinite")
    return {key: value.hex() for key, value in sorted(values.items())}


def _future_clock(
    denominator: BatteryReducedObservationDenominator,
    history: BatteryReducedObservationHistory,
) -> np.ndarray:
    interval = denominator.view.output_interval_s
    return np.arange(
        history.checkpoint_s + interval,
        history.end_s + interval,
        interval,
        dtype=np.float64,
    )


def _encode_trace(solution: Any, times: np.ndarray) -> dict[str, object]:
    receivers = _receiver_values(solution, times)
    return {
        "time_s_hex": _hex_array(times),
        **{f"{name}_hex": _hex_array(values) for name, values in sorted(receivers.items())},
        "termination": str(solution.termination),
    }


def _checkpoint_receiver(solution: Any, checkpoint_s: int) -> dict[str, str]:
    values = _receiver_values(
        solution,
        np.asarray([float(checkpoint_s)], dtype=np.float64),
    )
    return {key: float(value[0]).hex() for key, value in sorted(values.items())}


def _action_audit(
    solution: Any,
    *,
    unit: BatteryReducedObservationPreparation,
    history: BatteryReducedObservationHistory,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    for start, stop in zip(
        (0, 150, 300),
        (150, 300, 450),
        strict=True,
    ):
        if start >= history.checkpoint_s:
            break
        end = min(stop, history.checkpoint_s)
        current, ambient_delta = history.values_at(start)
        requested = {
            "current_delta_a": str(current),
            "ambient_delta_k": str(ambient_delta),
        }
        applied = _inputs(unit, current, ambient_delta)
        sample_s = float(start + (end - start) / 2)
        times = np.asarray([sample_s], dtype=np.float64)
        realized_current = float(_scalar_series(solution, "Current [A]", times)[0])
        realized_ambient = float(_scalar_series(solution, "Ambient temperature [K]", times)[0])
        valid = math.isclose(
            realized_current,
            float(current),
            rel_tol=0,
            abs_tol=1e-10,
        ) and math.isclose(
            realized_ambient,
            float(unit.initial_temperature_k + ambient_delta),
            rel_tol=0,
            abs_tol=1e-8,
        )
        rows.append(
            {
                "start_s": start,
                "end_s": end,
                "strict_interior_sample_s": sample_s,
                "requested": requested,
                "accepted": {name: float(value).hex() for name, value in sorted(applied.items())},
                "applied": {name: float(value).hex() for name, value in sorted(applied.items())},
                "realized": {
                    "current_a": realized_current.hex(),
                    "ambient_temperature_k": realized_ambient.hex(),
                },
                "realization_valid": valid,
            }
        )
    return rows, all(bool(row["realization_valid"]) for row in rows)


def _hold_realization(
    solution: Any,
    *,
    unit: BatteryReducedObservationPreparation,
    clock: np.ndarray,
) -> tuple[dict[str, object], bool]:
    current = _scalar_series(solution, "Current [A]", clock)
    ambient = _scalar_series(solution, "Ambient temperature [K]", clock)
    valid = bool(
        np.allclose(current, 0.0, rtol=0, atol=1e-10)
        and np.allclose(
            ambient,
            float(unit.initial_temperature_k),
            rtol=0,
            atol=1e-8,
        )
    )
    return (
        {
            "requested": {
                "current_delta_a": "0",
                "ambient_delta_k": "0",
            },
            "accepted": {
                name: float(value).hex()
                for name, value in sorted(_inputs(unit, Decimal(0), Decimal(0)).items())
            },
            "applied": {
                name: float(value).hex()
                for name, value in sorted(_inputs(unit, Decimal(0), Decimal(0)).items())
            },
            "realized_current_a_hex": _hex_array(current),
            "realized_ambient_temperature_k_hex": _hex_array(ambient),
            "realization_valid": valid,
        },
        valid,
    )


def _starting_solution(
    pybamm: Any,
    *,
    checkpoint_s: int,
    state: np.ndarray,
    model: Any,
    inputs: Mapping[str, float],
) -> Any:
    return pybamm.Solution(
        [np.asarray([float(checkpoint_s)], dtype=np.float64)],
        [state.reshape(-1, 1)],
        [model],
        [{name: np.asarray([float(value)], dtype=np.float64) for name, value in inputs.items()}],
        termination="final time",
        check_solution=False,
    )


def _trace_error(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, float]:
    if left["time_s_hex"] != right["time_s_hex"]:
        raise ValueError("battery reduced observation trace clocks differ")
    return {
        key: float(
            np.max(
                np.abs(
                    decode_hex(left[f"{key}_hex"])  # type: ignore[arg-type]
                    - decode_hex(right[f"{key}_hex"])  # type: ignore[arg-type]
                )
            )
        )
        for key in (
            "capacity_ah",
            "mean_temperature_k",
            "maximum_temperature_k",
            "terminal_voltage_v",
        )
    }


def _state_layout_sha256(model: Any) -> str:
    rows = []
    for variable, slices in sorted(
        model.y_slices.items(),
        key=lambda item: str(item[0]),
    ):
        rows.append((str(variable), tuple((value.start, value.stop) for value in slices)))
    return sha256(stable_json_bytes(rows)).hexdigest()


def acquire_cell(
    *,
    design: BatteryReducedObservationDesign,
    unit_id: str,
    denominator_id: str,
    history_id: str,
    source_identity_sha256: str,
    implementation_sha256: str,
    outcome_access: OutcomeAccess,
) -> tuple[dict[str, object], dict[str, object]]:
    """Acquire one native capture and independent direct restart continuation.

    Outcome visibility is a bound record property; callers enforce the
    development/evaluation authority and storage boundary.
    """

    validate_sha256(source_identity_sha256, field_name="source_identity_sha256")
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    unit, denominator, history = _lookup(
        design,
        unit_id,
        denominator_id,
        history_id,
    )
    return acquire_preparation(
        unit=unit,
        denominator=denominator,
        history=history,
        source_identity_sha256=source_identity_sha256,
        implementation_sha256=implementation_sha256,
        outcome_access=outcome_access,
    )


def acquire_preparation(
    *,
    unit: BatteryReducedObservationPreparation,
    denominator: BatteryReducedObservationDenominator,
    history: BatteryReducedObservationHistory,
    source_identity_sha256: str,
    implementation_sha256: str,
    outcome_access: OutcomeAccess,
) -> tuple[dict[str, object], dict[str, object]]:
    """Acquire a fresh typed preparation without the historical design roster."""

    validate_sha256(source_identity_sha256, field_name="source_identity_sha256")
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if (
        not isinstance(unit, BatteryReducedObservationPreparation)
        or not isinstance(denominator, BatteryReducedObservationDenominator)
        or not isinstance(history, BatteryReducedObservationHistory)
    ):
        raise TypeError("battery reduced observation preparation, denominator or history has another type")
    if not isinstance(outcome_access, OutcomeAccess):
        raise TypeError("battery reduced observation outcome-access stage has another type")
    started = time.monotonic()
    key = {
        "unit_id": unit.unit_id,
        "denominator_id": denominator.denominator_id,
        "history_id": history.history_id,
    }
    try:
        pybamm, source_simulation, prefix = _step_prefix(unit, denominator, history)
        state = np.asarray(prefix.last_state.y, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("checkpoint state is nonfinite")
        source_inputs = {
            name: float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
            for name, value in prefix.last_state.all_inputs[-1].items()
        }
        state_hex = _hex_array(state)
        state_sha256 = sha256(
            stable_json_bytes(
                {
                    "state_hex": state_hex,
                    "source_inputs": {
                        name: float(value).hex() for name, value in sorted(source_inputs.items())
                    },
                }
            )
        ).hexdigest()
        checkpoint = _checkpoint_receiver(prefix, history.checkpoint_s)
        coordinates = _checkpoint_coordinates(
            prefix,
            unit=unit,
            checkpoint_s=history.checkpoint_s,
        )
        prefix_action_rows, prefix_action_valid = _action_audit(
            prefix,
            unit=unit,
            history=history,
        )
        if not prefix_action_valid:
            raise RuntimeError("prefix action realization differs")
        clock = _future_clock(denominator, history)
        relative = clock - history.checkpoint_s
        hold = _inputs(unit, Decimal(0), Decimal(0))
        native_solution = source_simulation.step(
            history.end_s - history.checkpoint_s,
            t_eval=np.concatenate((np.asarray([0.0], dtype=np.float64), relative)),
            starting_solution=prefix,
            inputs=hold,
        )
        available_clock = clock[clock <= float(np.max(native_solution.t)) + 1e-8]
        if available_clock.size == 0:
            raise RuntimeError("native continuation terminated before first future clock")
        native_trace = _encode_trace(native_solution, available_clock)
        native_complete = bool(
            str(native_solution.termination) == "final time"
            and math.isclose(
                float(np.max(native_solution.t)),
                float(history.end_s),
                rel_tol=0,
                abs_tol=1e-8,
            )
        )
        hold_realization, hold_valid = _hold_realization(
            native_solution,
            unit=unit,
            clock=available_clock,
        )
        if not hold_valid:
            raise RuntimeError("native hold realization differs")
        capture: dict[str, object] = {
            "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/native-state-capture',
            "version": "1.0.0",
            "value": {
                **key,
                "stage": unit.stage.value,
                "stratum_id": unit.stratum_id,
                "initial_soc": str(unit.initial_soc),
                "initial_temperature_k": str(unit.initial_temperature_k),
                "checkpoint_s": history.checkpoint_s,
                "end_s": history.end_s,
                "source_identity_sha256": source_identity_sha256,
                "implementation_sha256": implementation_sha256,
                "state_hex": state_hex,
                "state_payload_sha256": state_sha256,
                "state_layout_sha256": _state_layout_sha256(source_simulation.built_model),
                "source_inputs_hex": {
                    name: float(value).hex() for name, value in sorted(source_inputs.items())
                },
                "checkpoint_receiver_hex": checkpoint,
                "checkpoint_coordinates_hex": coordinates,
                "prefix_action_rows": prefix_action_rows,
                "continuation_hold_realization": hold_realization,
                "native_trace": native_trace,
                "disposition": (
                    BatteryReducedObservationDisposition.COMPLETE.value
                    if native_complete
                    else BatteryReducedObservationDisposition.TERMINATED.value
                ),
                "reason_codes": (
                    [] if native_complete else ["SOURCE_TERMINATED_BEFORE_FROZEN_HORIZON"]
                ),
                "outcome_access": outcome_access.value,
                "runtime_seconds": time.monotonic() - started,
            },
        }
        if not native_complete:
            return (
                capture,
                {
                    "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/exact-state-continuation-result',
                    "version": "1.0.0",
                    "value": {
                        **key,
                        "route_id": BATTERY_REDUCED_OBSERVATION_ROUTE_ID,
                        "capture_sha256": canonical_sha256(capture),
                        "disposition": BatteryReducedObservationDisposition.TERMINATED.value,
                        "reason_codes": ["SOURCE_TERMINATION_PRECLUDES_DIRECT_RESTART_SENTINEL"],
                        "outcome_access": outcome_access.value,
                        "runtime_seconds": 0.0,
                    },
                },
            )
    except Exception as error:
        message = str(error).lower()
        if any(token in message for token in ("terminated", "termination", "event")):
            failure_disposition = BatteryReducedObservationDisposition.TERMINATED
        elif isinstance(error, KeyError):
            failure_disposition = BatteryReducedObservationDisposition.RECEIVER_INVALID
        else:
            failure_disposition = BatteryReducedObservationDisposition.SOURCE_FAILURE
        failure: dict[str, object] = {
            "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/native-state-capture',
            "version": "1.0.0",
            "value": {
                **key,
                "stage": unit.stage.value,
                "stratum_id": unit.stratum_id,
                "source_identity_sha256": source_identity_sha256,
                "implementation_sha256": implementation_sha256,
                "disposition": failure_disposition.value,
                "reason_codes": [f"PYBAMM_{type(error).__name__.upper()}"],
                "error_summary": str(error)[:500],
                "outcome_access": outcome_access.value,
                "runtime_seconds": time.monotonic() - started,
            },
        }
        r3_skipped: dict[str, object] = {
            "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/exact-state-continuation-result',
            "version": "1.0.0",
            "value": {
                **key,
                "route_id": BATTERY_REDUCED_OBSERVATION_ROUTE_ID,
                "disposition": BatteryReducedObservationDisposition.DIRECT_RESTART_FAILURE.value,
                "reason_codes": ["CAPTURE_NOT_COMPLETE"],
                "outcome_access": outcome_access.value,
                "runtime_seconds": 0.0,
            },
        }
        return failure, r3_skipped

    r3_started = time.monotonic()
    try:
        capture_value = capture["value"]
        assert isinstance(capture_value, dict)
        state = decode_hex(capture_value["state_hex"])
        if _hex_array(state) != capture_value["state_hex"]:
            raise ValueError("direct restart state round trip changed")
        pybamm, target_simulation = _runtime_objects(unit, denominator)
        hold = _inputs(unit, Decimal(0), Decimal(0))
        target_simulation.build(initial_soc=float(unit.initial_soc), inputs=hold)
        target_model = target_simulation.built_model
        if _state_layout_sha256(target_model) != capture_value["state_layout_sha256"]:
            raise ValueError("direct restart target state layout differs")
        source_inputs = {
            name: float.fromhex(value) for name, value in capture_value["source_inputs_hex"].items()
        }
        start = _starting_solution(
            pybamm,
            checkpoint_s=history.checkpoint_s,
            state=state,
            model=target_model,
            inputs=source_inputs,
        )
        clock = _future_clock(denominator, history)
        solution = target_simulation.step(
            history.end_s - history.checkpoint_s,
            t_eval=np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    clock - history.checkpoint_s,
                )
            ),
            starting_solution=start,
            inputs=hold,
        )
        trace = _encode_trace(solution, clock)
        if trace["termination"] != "final time":
            raise RuntimeError("direct restart continuation terminated")
        route_hold_realization, route_hold_valid = _hold_realization(
            solution,
            unit=unit,
            clock=clock,
        )
        if not route_hold_valid:
            raise RuntimeError("direct restart hold realization differs")
        native_trace = capture_value["native_trace"]
        errors = _trace_error(trace, native_trace)
        floors = {
            "capacity_ah": float(BATTERY_REDUCED_OBSERVATION_CAPACITY_FLOOR_AH),
            "mean_temperature_k": float(BATTERY_REDUCED_OBSERVATION_TEMPERATURE_FLOOR_K),
            "maximum_temperature_k": float(BATTERY_REDUCED_OBSERVATION_TEMPERATURE_FLOOR_K),
            "terminal_voltage_v": float(BATTERY_REDUCED_OBSERVATION_VOLTAGE_FLOOR_V),
        }
        closed = all(errors[name] <= floors[name] for name in floors)
        result: dict[str, object] = {
            "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/exact-state-continuation-result',
            "version": "1.0.0",
            "value": {
                **key,
                "route_id": BATTERY_REDUCED_OBSERVATION_ROUTE_ID,
                "capture_sha256": canonical_sha256(capture),
                "trace": trace,
                "continuation_hold_realization": route_hold_realization,
                "maximum_receiver_errors": errors,
                "receiver_closed": closed,
                "state_bit_exact": True,
                "disposition": BatteryReducedObservationDisposition.COMPLETE.value,
                "reason_codes": [] if closed else ["DIRECT_RESTART_RECEIVER_FLOOR_EXCEEDED"],
                "outcome_access": outcome_access.value,
                "runtime_seconds": time.monotonic() - r3_started,
            },
        }
    except Exception as error:
        result = {
            "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/exact-state-continuation-result',
            "version": "1.0.0",
            "value": {
                **key,
                "route_id": BATTERY_REDUCED_OBSERVATION_ROUTE_ID,
                "capture_sha256": canonical_sha256(capture),
                "disposition": BatteryReducedObservationDisposition.DIRECT_RESTART_FAILURE.value,
                "reason_codes": [f"DIRECT_RESTART_{type(error).__name__.upper()}"],
                "error_summary": str(error)[:500],
                "outcome_access": outcome_access.value,
                "runtime_seconds": time.monotonic() - r3_started,
            },
        }
    return capture, result


def canonical_sha256(document: object) -> str:
    return sha256(stable_json_bytes(document)).hexdigest()


def _record_value(document: Mapping[str, object]) -> Mapping[str, object]:
    value = document.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("battery reduced observation record value is not a mapping")
    return value


def receiver_response(capture: Mapping[str, object]) -> dict[str, np.ndarray]:
    value = capture["value"]
    if not isinstance(value, Mapping) or value.get("disposition") != "COMPLETE":
        raise ValueError("battery reduced observation response requires a complete capture")
    checkpoint = value["checkpoint_receiver_hex"]
    trace = value["native_trace"]
    if not isinstance(checkpoint, Mapping) or not isinstance(trace, Mapping):
        raise ValueError("battery reduced observation capture receiver payload is invalid")
    return {
        name: decode_hex(trace[f"{name}_hex"]) - float.fromhex(str(checkpoint[name]))
        for name in (
            "capacity_ah",
            "mean_temperature_k",
            "maximum_temperature_k",
            "terminal_voltage_v",
        )
    }


def coordinate_vector(
    capture: Mapping[str, object],
    chart: BatteryReducedObservationChart,
) -> np.ndarray:
    value = capture["value"]
    if not isinstance(value, Mapping):
        raise ValueError("battery reduced observation capture value is invalid")
    coordinates = value.get("checkpoint_coordinates_hex")
    if not isinstance(coordinates, Mapping):
        raise ValueError("battery reduced observation capture lacks checkpoint coordinates")
    result = np.asarray(
        [float.fromhex(str(coordinates[name])) for name in chart.coordinate_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("battery reduced observation coordinate vector is nonfinite")
    return result


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("battery reduced observation scaling requires at least two development units")
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    floor = 1e-12 * np.maximum(1.0, np.abs(center))
    scale = np.maximum(mad, floor)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("battery reduced observation scaling is nonfinite")
    return center, scale


def max_scaled_distance(
    left: np.ndarray,
    right: np.ndarray,
    scale: np.ndarray,
) -> float:
    if left.shape != right.shape or left.shape != scale.shape:
        raise ValueError("battery reduced observation chart distance shapes differ")
    return float(np.max(np.abs(left - right) / scale))


def receiver_errors(
    actual: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
) -> dict[str, float]:
    values = {}
    for name in (
        "capacity_ah",
        "mean_temperature_k",
        "maximum_temperature_k",
        "terminal_voltage_v",
    ):
        if actual[name].shape != predicted[name].shape:
            raise ValueError("battery reduced observation receiver response clocks differ")
        values[name] = float(np.max(np.abs(actual[name] - predicted[name])))
    return values


def materially_adequate(errors: Mapping[str, float]) -> bool:
    return (
        errors["capacity_ah"] <= float(BATTERY_REDUCED_OBSERVATION_CAPACITY_MATERIALITY_AH)
        and errors["mean_temperature_k"] <= float(BATTERY_REDUCED_OBSERVATION_TEMPERATURE_MATERIALITY_K)
        and errors["maximum_temperature_k"] <= float(BATTERY_REDUCED_OBSERVATION_TEMPERATURE_MATERIALITY_K)
        and errors["terminal_voltage_v"] <= float(BATTERY_REDUCED_OBSERVATION_VOLTAGE_MATERIALITY_V)
    )


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationScaleRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-scale-row'

    denominator_id: str
    history_id: str
    chart_id: str
    coordinate_id: str
    center: Decimal
    scale: Decimal


@dataclass(frozen=True, slots=True)
class BatteryReducedObservationPredictionRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-reduced-observation-prediction-row'

    target_unit_id: str
    donor_unit_id: str | None
    denominator_id: str
    history_id: str
    chart_id: str
    distance: Decimal | None
    inside_support: bool
    capacity_error_ah: Decimal | None
    mean_temperature_error_k: Decimal | None
    maximum_temperature_error_k: Decimal | None
    terminal_voltage_error_v: Decimal | None
    adequate: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.target_unit_id,
            self.denominator_id,
            self.history_id,
            self.chart_id,
        ):
            validate_stable_id(value, field_name="prediction_identity")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def build_prediction_rows(
    *,
    development_captures: Sequence[Mapping[str, object]],
    target_captures: Sequence[Mapping[str, object]],
    chart: BatteryReducedObservationChart,
    leave_one_out: bool,
) -> tuple[tuple[BatteryReducedObservationScaleRow, ...], tuple[BatteryReducedObservationPredictionRow, ...]]:
    """Build source-local nearest-donor predictions under one chart.

    Scaling and donors are always derived only from the development captures.
    Evaluation targets never enter centers, scales or the donor catalogue.
    """

    def key(document: Mapping[str, object]) -> tuple[str, str, str]:
        value = document["value"]
        assert isinstance(value, Mapping)
        return (
            str(value["unit_id"]),
            str(value["denominator_id"]),
            str(value["history_id"]),
        )

    complete_development = [
        item
        for item in development_captures
        if _record_value(item).get("disposition") == BatteryReducedObservationDisposition.COMPLETE.value
    ]
    development_by_cell: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for capture in complete_development:
        _, denominator_id, history_id = key(capture)
        development_by_cell.setdefault((denominator_id, history_id), []).append(capture)
    scale_rows: list[BatteryReducedObservationScaleRow] = []
    rows: list[BatteryReducedObservationPredictionRow] = []
    for (denominator_id, history_id), donors in sorted(development_by_cell.items()):
        donors.sort(key=lambda item: key(item)[0])
        matrix = np.vstack([coordinate_vector(item, chart) for item in donors])
        center, scale = robust_scale(matrix)
        for coordinate_id, center_value, scale_value in zip(
            chart.coordinate_ids,
            center,
            scale,
            strict=True,
        ):
            scale_rows.append(
                BatteryReducedObservationScaleRow(
                    denominator_id=denominator_id,
                    history_id=history_id,
                    chart_id=chart.chart_id,
                    coordinate_id=coordinate_id,
                    center=_decimal(float(center_value)),
                    scale=_decimal(float(scale_value)),
                )
            )
        local_targets = [
            item
            for item in target_captures
            if _record_value(item).get("denominator_id") == denominator_id
            and _record_value(item).get("history_id") == history_id
        ]
        for target in sorted(local_targets, key=key):
            target_unit_id, _, _ = key(target)
            if _record_value(target).get("disposition") != BatteryReducedObservationDisposition.COMPLETE.value:
                rows.append(
                    BatteryReducedObservationPredictionRow(
                        target_unit_id=target_unit_id,
                        donor_unit_id=None,
                        denominator_id=denominator_id,
                        history_id=history_id,
                        chart_id=chart.chart_id,
                        distance=None,
                        inside_support=False,
                        capacity_error_ah=None,
                        mean_temperature_error_k=None,
                        maximum_temperature_error_k=None,
                        terminal_voltage_error_v=None,
                        adequate=None,
                        reason_codes=("TARGET_CAPTURE_INCOMPLETE",),
                    )
                )
                continue
            target_vector = coordinate_vector(target, chart)
            ranked = []
            for donor in donors:
                donor_unit_id, _, _ = key(donor)
                if leave_one_out and donor_unit_id == target_unit_id:
                    continue
                ranked.append(
                    (
                        max_scaled_distance(
                            target_vector,
                            coordinate_vector(donor, chart),
                            scale,
                        ),
                        donor_unit_id,
                        donor,
                    )
                )
            if not ranked:
                rows.append(
                    BatteryReducedObservationPredictionRow(
                        target_unit_id=target_unit_id,
                        donor_unit_id=None,
                        denominator_id=denominator_id,
                        history_id=history_id,
                        chart_id=chart.chart_id,
                        distance=None,
                        inside_support=False,
                        capacity_error_ah=None,
                        mean_temperature_error_k=None,
                        maximum_temperature_error_k=None,
                        terminal_voltage_error_v=None,
                        adequate=None,
                        reason_codes=("DONOR_POOL_EMPTY",),
                    )
                )
                continue
            distance, donor_unit_id, donor = min(ranked, key=lambda item: item[:2])
            inside = distance <= float(BATTERY_REDUCED_OBSERVATION_SUPPORT_RADIUS)
            if not inside:
                rows.append(
                    BatteryReducedObservationPredictionRow(
                        target_unit_id=target_unit_id,
                        donor_unit_id=donor_unit_id,
                        denominator_id=denominator_id,
                        history_id=history_id,
                        chart_id=chart.chart_id,
                        distance=_decimal(distance),
                        inside_support=False,
                        capacity_error_ah=None,
                        mean_temperature_error_k=None,
                        maximum_temperature_error_k=None,
                        terminal_voltage_error_v=None,
                        adequate=None,
                        reason_codes=("OUTSIDE_FROZEN_LOCAL_SUPPORT",),
                    )
                )
                continue
            errors = receiver_errors(receiver_response(target), receiver_response(donor))
            adequate = materially_adequate(errors)
            rows.append(
                BatteryReducedObservationPredictionRow(
                    target_unit_id=target_unit_id,
                    donor_unit_id=donor_unit_id,
                    denominator_id=denominator_id,
                    history_id=history_id,
                    chart_id=chart.chart_id,
                    distance=_decimal(distance),
                    inside_support=True,
                    capacity_error_ah=_decimal(errors["capacity_ah"]),
                    mean_temperature_error_k=_decimal(errors["mean_temperature_k"]),
                    maximum_temperature_error_k=_decimal(errors["maximum_temperature_k"]),
                    terminal_voltage_error_v=_decimal(errors["terminal_voltage_v"]),
                    adequate=adequate,
                    reason_codes=() if adequate else ("SUPPORTED_PREDICTION_LOSS",),
                )
            )
    return tuple(scale_rows), tuple(rows)


def _row_max_error(row: BatteryReducedObservationPredictionRow) -> float:
    values = (
        (row.capacity_error_ah, BATTERY_REDUCED_OBSERVATION_CAPACITY_MATERIALITY_AH),
        (row.mean_temperature_error_k, BATTERY_REDUCED_OBSERVATION_TEMPERATURE_MATERIALITY_K),
        (row.maximum_temperature_error_k, BATTERY_REDUCED_OBSERVATION_TEMPERATURE_MATERIALITY_K),
        (row.terminal_voltage_error_v, BATTERY_REDUCED_OBSERVATION_VOLTAGE_MATERIALITY_V),
    )
    if any(value is None for value, _ in values):
        return math.inf
    return max(float(value / floor) for value, floor in values if value is not None)


def chart_score(
    rows: Sequence[BatteryReducedObservationPredictionRow],
    chart: BatteryReducedObservationChart,
) -> tuple[int, int, float, float, int, str]:
    outside = sum(not row.inside_support for row in rows)
    supported = [row for row in rows if row.inside_support and row.adequate is not None]
    lossy = sum(row.adequate is False for row in supported)
    errors = np.asarray([_row_max_error(row) for row in supported], dtype=np.float64)
    maximum = float(np.max(errors)) if errors.size else math.inf
    q95 = float(np.quantile(errors, 0.95)) if errors.size else math.inf
    return outside, lossy, maximum, q95, len(chart.coordinate_ids), chart.chart_id


def select_chart(
    development_captures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def serialize_score(
        value: tuple[int, int, float, float, int, str],
    ) -> list[int | float | str]:
        return [
            value[0],
            value[1],
            value[2] if math.isfinite(value[2]) else "Infinity",
            value[3] if math.isfinite(value[3]) else "Infinity",
            value[4],
            value[5],
        ]

    analyses = []
    for chart in charts():
        scales, rows = build_prediction_rows(
            development_captures=development_captures,
            target_captures=development_captures,
            chart=chart,
            leave_one_out=True,
        )
        score = chart_score(rows, chart)
        analyses.append((chart, scales, rows, score))
    perfect = [item for item in analyses if item[3][0] == 0 and item[3][1] == 0]
    if perfect:
        selected = min(
            perfect,
            key=lambda item: (len(item[0].coordinate_ids), item[0].chart_id),
        )
        selection_rule = "LOWEST_DIMENSION_ZERO_OUTSIDE_ZERO_LOSS"
    else:
        selected = min(analyses, key=lambda item: item[3])
        selection_rule = "LEXICOGRAPHIC_FROZEN_SCORE"
    chart, scales, rows, score = selected
    return {
        "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/frozen-development-selection',
        "version": "1.0.0",
        "value": {
            "selected_chart": chart.to_document(),
            "selected_chart_id": chart.chart_id,
            "selection_rule": selection_rule,
            "selected_score": serialize_score(score),
            "all_chart_scores": {item[0].chart_id: serialize_score(item[3]) for item in analyses},
            "scales": [row.to_document() for row in scales],
            "development_predictions": [row.to_document() for row in rows],
            "development_capture_set_sha256": canonical_sha256(
                sorted(canonical_sha256(item) for item in development_captures)
            ),
            "outcome_access": OutcomeAccess.DEVELOPMENT_VISIBLE.value,
            "maximum_evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        },
    }


def selected_chart_from_handoff(handoff: Mapping[str, object]) -> BatteryReducedObservationChart:
    value = handoff.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("battery reduced observation handoff is malformed")
    selected_id = value.get("selected_chart_id")
    try:
        return next(chart for chart in charts() if chart.chart_id == selected_id)
    except StopIteration as error:
        raise ValueError("battery reduced observation handoff selected an unfrozen chart") from error


def adjudicate(
    *,
    development_captures: Sequence[Mapping[str, object]],
    evaluation_captures: Sequence[Mapping[str, object]],
    evaluation_r3_results: Sequence[Mapping[str, object]],
    handoff: Mapping[str, object],
) -> dict[str, object]:
    chart = selected_chart_from_handoff(handoff)
    _, rows = build_prediction_rows(
        development_captures=development_captures,
        target_captures=evaluation_captures,
        chart=chart,
        leave_one_out=False,
    )
    exact_complete = [
        result
        for result in evaluation_r3_results
        if _record_value(result).get("disposition") == BatteryReducedObservationDisposition.COMPLETE.value
    ]
    exact_drift = any(
        not bool(_record_value(result).get("receiver_closed")) for result in exact_complete
    )
    intended = len(build_design().evaluation_units) * 8 * 5
    complete_rows = [row for row in rows if row.adequate is not None]
    all_complete = (
        len(rows) == intended and len(complete_rows) == intended and len(exact_complete) == intended
    )
    inside = [row for row in complete_rows if row.inside_support]
    loss = [row for row in inside if row.adequate is False]
    outside = [row for row in rows if not row.inside_support]
    if exact_drift:
        verdict = BatteryReducedObservationVerdict.EXACT_REFERENCE_DRIFT
    elif all_complete and not outside and not loss:
        verdict = BatteryReducedObservationVerdict.UNIFORM_SUFFICIENCY
    elif not complete_rows:
        verdict = BatteryReducedObservationVerdict.UNEVALUABLE
    else:
        denominator_rows = {
            denominator: [row for row in rows if row.denominator_id == denominator]
            for denominator in sorted({row.denominator_id for row in rows})
        }
        history_rows = {
            history: [row for row in rows if row.history_id == history]
            for history in sorted({row.history_id for row in rows})
        }
        denominator_good = {
            key: len(local) == 60
            and all(row.inside_support and row.adequate is True for row in local)
            for key, local in denominator_rows.items()
        }
        denominator_lossy = {
            key: len(local) == 60
            and all(row.inside_support and row.adequate is not None for row in local)
            and any(row.adequate is False for row in local)
            for key, local in denominator_rows.items()
        }
        history_good = {
            key: len(local) == 96
            and all(row.inside_support and row.adequate is True for row in local)
            for key, local in history_rows.items()
        }
        history_lossy = {
            key: len(local) == 96
            and all(row.inside_support and row.adequate is not None for row in local)
            and any(row.adequate is False for row in local)
            for key, local in history_rows.items()
        }
        if any(denominator_good.values()) and any(denominator_lossy.values()):
            verdict = BatteryReducedObservationVerdict.DENOMINATOR_CONDITIONAL
        elif any(history_good.values()) and any(history_lossy.values()):
            verdict = BatteryReducedObservationVerdict.HISTORY_CONDITIONAL
        elif all(
            local
            and any(row.inside_support for row in local)
            and any(row.inside_support and row.adequate is False for row in local)
            for local in denominator_rows.values()
        ):
            verdict = BatteryReducedObservationVerdict.LOSS_RECURRENT
        elif all_complete and not outside and loss:
            verdict = BatteryReducedObservationVerdict.MIXED
        else:
            verdict = BatteryReducedObservationVerdict.PARTIAL
    return {
        "schema": 'empirical-lawhood/simulators/battery-reduced-observation-prediction/reduced-coordinate-adjudication',
        "version": "1.0.0",
        "value": {
            "verdict": verdict.value,
            "selected_chart_id": chart.chart_id,
            "intended_record_count": intended,
            "prediction_row_count": len(rows),
            "complete_prediction_count": len(complete_rows),
            "inside_support_count": len(inside),
            "outside_support_count": len(outside),
            "supported_loss_count": len(loss),
            "exact_complete_count": len(exact_complete),
            "exact_drift_count": sum(
                not bool(_record_value(result).get("receiver_closed")) for result in exact_complete
            ),
            "predictions": [row.to_document() for row in rows],
            "handoff_sha256": canonical_sha256(handoff),
            "outcome_access": OutcomeAccess.EVALUATION_REVEALED.value,
            "maximum_evidence_ceiling": EvidenceCeiling.LOCAL_LAW.value,
        },
    }


def document_bytes(document: Mapping[str, object]) -> bytes:
    payload = stable_json_bytes(document)
    if len(payload) > BATTERY_REDUCED_OBSERVATION_MAXIMUM_JSON_BYTES:
        raise ValueError("battery reduced observation record exceeds the bounded JSON limit")
    json.loads(payload)
    return payload


def stable_json_bytes(document: object) -> bytes:
    """Encode an already JSON-shaped artifact deterministically.

    Scientific binary values are float-hex strings before this boundary.
    Operational duration/resource fields may be finite JSON numbers.
    """

    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "BATTERY_REDUCED_OBSERVATION_DESIGN_ID",
    "BATTERY_REDUCED_OBSERVATION_EXTERNAL_ROOT",
    "BATTERY_REDUCED_OBSERVATION_MAXIMUM_JSON_BYTES",
    "BATTERY_REDUCED_OBSERVATION_MINIMUM_FREE_BYTES",
    "BATTERY_REDUCED_OBSERVATION_PLAN_ID",
    "BATTERY_REDUCED_OBSERVATION_PROVIDER_KEY",
    "BATTERY_REDUCED_OBSERVATION_ROUTE_ID",
    "BATTERY_REDUCED_OBSERVATION_SOURCE_VERSION",
    'BatteryReducedObservationChart',
    'BatteryReducedObservationDesign',
    'BatteryReducedObservationDisposition',
    'BatteryReducedObservationHistory',
    'BatteryReducedObservationPreparation',
    'BatteryReducedObservationStage',
    'BatteryReducedObservationVerdict',
    "acquire_cell",
    "acquire_preparation",
    "adjudicate",
    "build_design",
    "build_prediction_rows",
    "canonical_sha256",
    "chart_score",
    "charts",
    "decode_hex",
    "denominators",
    "design_calculation",
    "document_bytes",
    "histories",
    "intended_keys",
    "materially_adequate",
    "receiver_response",
    "robust_scale",
    "select_chart",
    "selected_chart_from_handoff",
    "stable_json_bytes",
    "units_for_stage",
    "validate_repository_config",
]
