"""Bounded PyBaMM battery electrothermal electrothermal hierarchy experiment.

This adapter owns the source-local denominator chart, finite word algebra,
state capture/restart semantics, donor construction, and pure adjudication.
Storage, issue, execution authority, and reveal remain script/infrastructure
concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import StrEnum
from hashlib import sha256
import math
import statistics
import time
from collections.abc import Iterable
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.formal_gaps import FormalGapRegister
from empirical_lawhood.adapters.simulators._battery_electrothermal_bootstrap_seeds import CLOSURE_SEED_COMMITMENTS


BATTERY_ELECTROTHERMAL_SOURCE_VERSION = "26.6.2.0"
BATTERY_ELECTROTHERMAL_PARAMETER_SET = "Chen2020"
BATTERY_ELECTROTHERMAL_DESIGN_ID = "design.battery-electrothermal.hierarchy"
BATTERY_ELECTROTHERMAL_PROVIDER_KEY = "open-sim.battery-electrothermal-hierarchy"
BATTERY_ELECTROTHERMAL_HORIZON_S = 900
BATTERY_ELECTROTHERMAL_BOUNDARIES_S = (0, 150, 300, 450, 600, 900)
BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES = 100_000
BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED = 2_026_072_8085
BATTERY_ELECTROTHERMAL_FAMILY_ALPHA = Decimal("0.010")
BATTERY_ELECTROTHERMAL_EVALUATION_CANDIDATES = (24, 36, 48, 72, 96)
BATTERY_ELECTROTHERMAL_PRIMARY_N = 24
BATTERY_ELECTROTHERMAL_EXTERNAL_ROOT = "runs/battery-electrothermal-hierarchy"
BATTERY_ELECTROTHERMAL_MINIMUM_FREE_BYTES = 256 * 1024**3
_Q = Decimal("0.0000001")


class BatteryElectrothermalStage(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class BatteryElectrothermalUnitRole(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION_PRIMARY = "EVALUATION_PRIMARY"
    EVALUATION_RESERVE = "EVALUATION_RESERVE"


class BatteryElectrothermalModel(StrEnum):
    SPME = "SPME"
    DFN = "DFN"


class BatteryElectrothermalThermal(StrEnum):
    ISOTHERMAL = "ISOTHERMAL"
    LUMPED = "LUMPED"
    X_FULL = "X_FULL"


class BatteryElectrothermalSolver(StrEnum):
    CASADI = "CASADI"
    IDAKLU = "IDAKLU"


class BatteryElectrothermalEpisodeDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    SCIENTIFIC_PARTIAL = "SCIENTIFIC_PARTIAL"
    SIMULATOR_TERMINATED = "SIMULATOR_TERMINATED"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    TECHNICAL_OBSERVATION_FAILURE = "TECHNICAL_OBSERVATION_FAILURE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    UNEVALUABLE_OPERAND = "UNEVALUABLE_OPERAND"


class BatteryElectrothermalFormalDisposition(StrEnum):
    DIRECT_PRIMARY = "DIRECT_PRIMARY"
    DIRECT_SECONDARY = "DIRECT_SECONDARY"
    FINITE_PROXY_ONLY = "FINITE_PROXY_ONLY"
    ADMISSION_NOT_ENTERED_TYPED_EXCLUSION = "ADMISSION_NOT_ENTERED_TYPED_EXCLUSION"
    OPERAND_ABSENT_TYPED_EXCLUSION = "OPERAND_ABSENT_TYPED_EXCLUSION"


class BatteryElectrothermalEffect(StrEnum):
    RAW_ORDER = "RAW_ORDER"
    ADJUSTED_ORDER = "ADJUSTED_ORDER"
    RAW_REPETITION = "RAW_REPETITION"
    MATCHED_REPETITION = "MATCHED_REPETITION"


class BatteryElectrothermalCoordinate(StrEnum):
    CAPACITY = "CAPACITY"
    TEMPERATURE = "TEMPERATURE"
    VOLTAGE = "VOLTAGE"


class BatteryElectrothermalEffectClass(StrEnum):
    MATERIAL_POSITIVE = "MATERIAL_POSITIVE"
    MATERIAL_NEGATIVE = "MATERIAL_NEGATIVE"
    NULL_EQUIVALENT_FINITE = "NULL_EQUIVALENT_FINITE"
    NONZERO_SUBMATERIAL = "NONZERO_SUBMATERIAL"
    UNRESOLVED = "UNRESOLVED"
    PARTIAL = "PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


class BatteryElectrothermalExactClass(StrEnum):
    CLOSED = "EXACT_STATE_CLOSED"
    OPPOSED = "EXACT_STATE_OPPOSED"
    UNEVALUABLE = "EXACT_STATE_UNEVALUABLE"


class BatteryElectrothermalReducedClass(StrEnum):
    LOSSY = "REDUCED_STATE_MATERIALLY_LOSSY"
    ADEQUATE = "REDUCED_STATE_ADEQUATE_FINITE"
    MIXED = "REDUCED_STATE_MIXED"
    UNEVALUABLE = "REDUCED_STATE_UNEVALUABLE"


class BatteryElectrothermalOrderVerdict(StrEnum):
    FULL = "BATTERY_ELECTROTHERMAL_ORDER_FULL_HIERARCHY_RECURRENT"
    THERMAL = "BATTERY_ELECTROTHERMAL_ORDER_THERMAL_CONDITIONAL"
    MODEL = "BATTERY_ELECTROTHERMAL_ORDER_MODEL_CONDITIONAL"
    VIEW = "BATTERY_ELECTROTHERMAL_ORDER_VIEW_CONDITIONAL"
    TIMING_ONLY = "BATTERY_ELECTROTHERMAL_RAW_ORDER_TIMING_ONLY"
    UNRESOLVED = "BATTERY_ELECTROTHERMAL_ORDER_UNRESOLVED"
    PARTIAL = "BATTERY_ELECTROTHERMAL_ORDER_PARTIAL"
    UNEVALUABLE = "BATTERY_ELECTROTHERMAL_ORDER_UNEVALUABLE"


class BatteryElectrothermalRepetitionVerdict(StrEnum):
    FULL = "BATTERY_ELECTROTHERMAL_REPETITION_FULL_HIERARCHY_RECURRENT"
    THERMAL = "BATTERY_ELECTROTHERMAL_REPETITION_THERMAL_CONDITIONAL"
    MODEL = "BATTERY_ELECTROTHERMAL_REPETITION_MODEL_CONDITIONAL"
    VIEW = "BATTERY_ELECTROTHERMAL_REPETITION_VIEW_CONDITIONAL"
    END_TIME_ONLY = "BATTERY_ELECTROTHERMAL_RAW_REPETITION_END_TIME_ONLY"
    UNRESOLVED = "BATTERY_ELECTROTHERMAL_REPETITION_UNRESOLVED"
    PARTIAL = "BATTERY_ELECTROTHERMAL_REPETITION_PARTIAL"
    UNEVALUABLE = "BATTERY_ELECTROTHERMAL_REPETITION_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-numerical-view'

    view_id: str
    solver: BatteryElectrothermalSolver
    nominal_output_interval_s: Decimal
    rtol: Decimal
    atol: Decimal
    spatial_points: int

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        expected = {
            BatteryElectrothermalSolver.CASADI: (
                "view.pybamm.casadi-coarse",
                Decimal(10),
                Decimal("1e-5"),
                Decimal("1e-6"),
                20,
            ),
            BatteryElectrothermalSolver.IDAKLU: (
                "view.pybamm.idaklu-refined",
                Decimal(1),
                Decimal("1e-7"),
                Decimal("1e-8"),
                30,
            ),
        }[self.solver]
        if (
            self.view_id,
            self.nominal_output_interval_s,
            self.rtol,
            self.atol,
            self.spatial_points,
        ) != expected:
            raise ValueError("battery electrothermal numerical view differs from the frozen chart")


def numerical_views() -> tuple[BatteryElectrothermalNumericalView, ...]:
    return tuple(
        sorted(
            (
                BatteryElectrothermalNumericalView(
                    view_id="view.pybamm.casadi-coarse",
                    solver=BatteryElectrothermalSolver.CASADI,
                    nominal_output_interval_s=Decimal(10),
                    rtol=Decimal("1e-5"),
                    atol=Decimal("1e-6"),
                    spatial_points=20,
                ),
                BatteryElectrothermalNumericalView(
                    view_id="view.pybamm.idaklu-refined",
                    solver=BatteryElectrothermalSolver.IDAKLU,
                    nominal_output_interval_s=Decimal(1),
                    rtol=Decimal("1e-7"),
                    atol=Decimal("1e-8"),
                    spatial_points=30,
                ),
            ),
            key=lambda item: item.view_id,
        )
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalDenominator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-denominator'

    denominator_id: str
    model: BatteryElectrothermalModel
    thermal: BatteryElectrothermalThermal
    view: BatteryElectrothermalNumericalView
    source_version: str
    parameter_set: str
    cell_geometry: str
    calculate_isothermal_heat_source: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.denominator_id, field_name="denominator_id")
        expected_id = (
            f"denominator.pybamm.{self.model.value.lower()}."
            f"{self.thermal.value.lower().replace('_', '-')}."
            f"{self.view.view_id.removeprefix('view.pybamm.')}"
        )
        if (
            self.denominator_id != expected_id
            or self.source_version != BATTERY_ELECTROTHERMAL_SOURCE_VERSION
            or self.parameter_set != BATTERY_ELECTROTHERMAL_PARAMETER_SET
            or self.cell_geometry != "pouch"
            or self.calculate_isothermal_heat_source is not (self.thermal is BatteryElectrothermalThermal.ISOTHERMAL)
        ):
            raise ValueError("battery electrothermal denominator differs from the frozen chart")


def denominators(*, include_x_full: bool = True) -> tuple[BatteryElectrothermalDenominator, ...]:
    values: list[BatteryElectrothermalDenominator] = []
    thermals = (
        BatteryElectrothermalThermal.ISOTHERMAL,
        BatteryElectrothermalThermal.LUMPED,
        *((BatteryElectrothermalThermal.X_FULL,) if include_x_full else ()),
    )
    for model in (BatteryElectrothermalModel.SPME, BatteryElectrothermalModel.DFN):
        for thermal in thermals:
            for view in numerical_views():
                denominator_id = (
                    f"denominator.pybamm.{model.value.lower()}."
                    f"{thermal.value.lower().replace('_', '-')}."
                    f"{view.view_id.removeprefix('view.pybamm.')}"
                )
                values.append(
                    BatteryElectrothermalDenominator(
                        denominator_id=denominator_id,
                        model=model,
                        thermal=thermal,
                        view=view,
                        source_version=BATTERY_ELECTROTHERMAL_SOURCE_VERSION,
                        parameter_set=BATTERY_ELECTROTHERMAL_PARAMETER_SET,
                        cell_geometry="pouch",
                        calculate_isothermal_heat_source=(thermal is BatteryElectrothermalThermal.ISOTHERMAL),
                    )
                )
    return tuple(sorted(values, key=lambda item: item.denominator_id))


def validate_repository_config(document: Mapping[str, object]) -> None:
    """Reject any repository campaign config that differs from the bounded chart."""

    expected: dict[str, object] = {
        "bootstrap": {
            "replicates": BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES,
            "seed": BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED,
            "stratified_by": "preparation_stratum",
        },
        "candidate_evaluation_n": list(BATTERY_ELECTROTHERMAL_EVALUATION_CANDIDATES),
        "claim_ceiling": "LOCAL_LAW",
        "denominator_chart": {
            "models": ["SPME", "DFN"],
            "numerical_views": [
                "view.pybamm.casadi-coarse",
                "view.pybamm.idaklu-refined",
            ],
            "thermal_closures": ["ISOTHERMAL", "LUMPED", "X_FULL"],
            "x_full_entry_rule": "paired_pre_outcome_qualification_only",
        },
        "design_id": BATTERY_ELECTROTHERMAL_DESIGN_ID,
        "external_root": BATTERY_ELECTROTHERMAL_EXTERNAL_ROOT,
        "formal_gap_count": 48,
        "horizon_s": BATTERY_ELECTROTHERMAL_HORIZON_S,
        "materiality": {
            "capacity_ah": "0.002",
            "temperature_k": "0.005",
            "voltage_v": "0.005",
        },
        "multiplicity": {
            "family_alpha": "0.010",
            "families": [
                "RAW_ORDER",
                "ADJUSTED_ORDER",
                "RAW_REPETITION",
                "MATCHED_REPETITION",
                "REDUCED_CLOSURE",
            ],
            "method": "bonferroni_within_frozen_family",
        },
        "parameter_set": BATTERY_ELECTROTHERMAL_PARAMETER_SET,
        "primary_word_count": len(BATTERY_ELECTROTHERMAL_WORD_IDS),
        "reduced_state": [
            "soc_coordinate",
            "volume_averaged_cell_temperature_k",
            "terminal_voltage_v",
        ],
        "restart_histories": [
            [word_id, checkpoint_s, end_s] for word_id, checkpoint_s, end_s in BATTERY_ELECTROTHERMAL_RESTART_HISTORIES
        ],
        "schema": 'empirical-lawhood/simulators/battery-electrothermal-response/config',
        "selected_evaluation_n": BATTERY_ELECTROTHERMAL_PRIMARY_N,
        "source_version": BATTERY_ELECTROTHERMAL_SOURCE_VERSION,
        "version": "1.0.0",
    }
    if dict(document) != expected:
        raise ValueError("battery electrothermal repository config differs from the bounded chart")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalActionWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-action-word'

    word_id: str
    current_delta_a_by_interval: tuple[Decimal, ...]
    ambient_delta_k_by_interval: tuple[Decimal, ...]
    role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        if len(self.current_delta_a_by_interval) != 5 or len(self.ambient_delta_k_by_interval) != 5:
            raise ValueError("battery electrothermal words bind exactly five half-open intervals")
        for value in (
            *self.current_delta_a_by_interval,
            *self.ambient_delta_k_by_interval,
        ):
            validate_decimal(value, field_name="action_delta")
        if not self.role:
            raise ValueError("battery electrothermal word role is required")

    def values_at(self, clock_s: int) -> tuple[Decimal, Decimal]:
        if not 0 <= clock_s <= BATTERY_ELECTROTHERMAL_HORIZON_S:
            raise ValueError("battery electrothermal word clock leaves the horizon")
        if clock_s == BATTERY_ELECTROTHERMAL_HORIZON_S:
            return Decimal(0), Decimal(0)
        index = next(index for index, stop in enumerate(BATTERY_ELECTROTHERMAL_BOUNDARIES_S[1:]) if clock_s < stop)
        return (
            self.current_delta_a_by_interval[index],
            self.ambient_delta_k_by_interval[index],
        )


def _word(
    token: str,
    current: Sequence[str | int],
    ambient: Sequence[str | int],
    role: str,
) -> BatteryElectrothermalActionWord:
    return BatteryElectrothermalActionWord(
        word_id=f"word.battery-electrothermal.{token}",
        current_delta_a_by_interval=tuple(Decimal(value) for value in current),
        ambient_delta_k_by_interval=tuple(Decimal(value) for value in ambient),
        role=role,
    )


def action_words() -> tuple[BatteryElectrothermalActionWord, ...]:
    z = (0, 0, 0, 0, 0)
    values = (
        _word("hold", z, z, "baseline"),
        _word("i-early-150", (1, 0, 0, 0, 0), z, "isolated-early-current"),
        _word("i-late-150", (0, 1, 0, 0, 0), z, "isolated-late-current"),
        _word("t-early-150", z, ("0.5", 0, 0, 0, 0), "isolated-early-ambient"),
        _word("t-late-150", z, (0, "0.5", 0, 0, 0), "isolated-late-ambient"),
        _word("i-then-t", (1, 0, 0, 0, 0), (0, "0.5", 0, 0, 0), "electrothermal-order"),
        _word("t-then-i", (0, 1, 0, 0, 0), ("0.5", 0, 0, 0, 0), "electrothermal-order"),
        _word(
            "i-contiguous-early-300",
            (1, 1, 0, 0, 0),
            z,
            "electrothermal-repetition-comparator",
        ),
        _word("i-split", (1, 0, 1, 0, 0), z, "electrothermal-repetition"),
        _word(
            "i-contiguous-matched-end",
            (0, 1, 1, 0, 0),
            z,
            "matched-end-repetition-comparator",
        ),
        _word("i-plus-600", (1, 1, 1, 1, 0), z, "closure-prefix"),
        _word("i-minus-300", (-1, -1, 0, 0, 0), z, "signed-falsifier"),
        _word("t-plus-300", z, ("0.5", "0.5", 0, 0, 0), "thermal-calibration"),
        _word(
            "it-simultaneous-300",
            (1, 1, 0, 0, 0),
            ("0.5", "0.5", 0, 0, 0),
            "simultaneous-calibration",
        ),
        _word("future-i-plus", (0, 0, 0, 0, 1), z, "causal-falsifier"),
    )
    return tuple(sorted(values, key=lambda item: item.word_id))


BATTERY_ELECTROTHERMAL_WORD_IDS = tuple(item.word_id for item in action_words())
BATTERY_ELECTROTHERMAL_QUALIFICATION_EXTREME_WORD_IDS = tuple(
    sorted(
        (
            "word.battery-electrothermal.hold",
            "word.battery-electrothermal.i-minus-300",
            "word.battery-electrothermal.it-simultaneous-300",
            "word.battery-electrothermal.i-plus-600",
            "word.battery-electrothermal.future-i-plus",
        )
    )
)
BATTERY_ELECTROTHERMAL_RESTART_HISTORIES = (
    ("word.battery-electrothermal.i-plus-600", 300, 600),
    ("word.battery-electrothermal.i-then-t", 300, 600),
    ("word.battery-electrothermal.t-then-i", 300, 600),
    ("word.battery-electrothermal.i-split", 450, 750),
    ("word.battery-electrothermal.i-contiguous-matched-end", 450, 750),
)
BATTERY_ELECTROTHERMAL_QUALIFICATION_WORD_IDS = tuple(
    sorted(
        {
            *BATTERY_ELECTROTHERMAL_QUALIFICATION_EXTREME_WORD_IDS,
            *(item[0] for item in BATTERY_ELECTROTHERMAL_RESTART_HISTORIES),
        }
    )
)


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalPreparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-preparation'

    unit_id: str
    stage: BatteryElectrothermalStage
    role: BatteryElectrothermalUnitRole
    stratum_id: str
    initial_soc: Decimal
    initial_temperature_k: Decimal
    reserve: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        validate_decimal(
            self.initial_soc,
            field_name="initial_soc",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.initial_temperature_k,
            field_name="initial_temperature_k",
            minimum=Decimal(250),
        )
        if self.reserve is not (self.role is BatteryElectrothermalUnitRole.EVALUATION_RESERVE):
            raise ValueError("battery electrothermal reserve flag differs from the unit role")


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


def _fraction(seed: int, *tokens: object) -> Decimal:
    digest = sha256(b":".join((str(seed).encode(), *(item if isinstance(item, bytes) else str(item).encode() for item in tokens)))).digest()
    numerator = int.from_bytes(digest[:8], "big")
    return Decimal(numerator) / Decimal(2**64)


def _roster(
    *,
    stage: BatteryElectrothermalStage,
    role: BatteryElectrothermalUnitRole,
    per_stratum: int,
    seed: int,
    token: str,
) -> tuple[BatteryElectrothermalPreparation, ...]:
    values: list[BatteryElectrothermalPreparation] = []
    for stratum_index, (stratum_id, soc_lo, soc_hi, temp_lo, temp_hi) in enumerate(
        _STRATA,
        start=1,
    ):
        temp_ranks = list(range(per_stratum))
        temp_ranks.sort(key=lambda index: _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "perm", index))
        for index in range(per_stratum):
            soc_fraction = (
                Decimal(index) + _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "soc", index)
            ) / Decimal(per_stratum)
            temp_fraction = (
                Decimal(temp_ranks[index]) + _fraction(seed, token, bytes(_FROZEN_STRATUM_HASH_OPERANDS[stratum_index - 1]), "temp", index)
            ) / Decimal(per_stratum)
            soc = (soc_lo + (soc_hi - soc_lo) * soc_fraction).quantize(
                _Q,
                rounding=ROUND_HALF_EVEN,
            )
            temperature = (temp_lo + (temp_hi - temp_lo) * temp_fraction).quantize(
                _Q, rounding=ROUND_HALF_EVEN
            )
            values.append(
                BatteryElectrothermalPreparation(
                    unit_id=(f"unit.battery-electrothermal.{token}.s{stratum_index:02d}.{index + 1:02d}"),
                    stage=stage,
                    role=role,
                    stratum_id=stratum_id,
                    initial_soc=soc,
                    initial_temperature_k=temperature,
                    reserve=role is BatteryElectrothermalUnitRole.EVALUATION_RESERVE,
                )
            )
    return tuple(sorted(values, key=lambda item: item.unit_id))


def qualification_preparations() -> tuple[BatteryElectrothermalPreparation, ...]:
    return _roster(
        stage=BatteryElectrothermalStage.QUALIFICATION,
        role=BatteryElectrothermalUnitRole.QUALIFICATION,
        per_stratum=2,
        seed=2_026_072_8081,
        token="qualification",
    )


def development_preparations() -> tuple[BatteryElectrothermalPreparation, ...]:
    return _roster(
        stage=BatteryElectrothermalStage.DEVELOPMENT,
        role=BatteryElectrothermalUnitRole.DEVELOPMENT,
        per_stratum=2,
        seed=2_026_072_8082,
        token="development",
    )


def evaluation_preparations(
    n: int = BATTERY_ELECTROTHERMAL_PRIMARY_N,
) -> tuple[BatteryElectrothermalPreparation, ...]:
    if n not in BATTERY_ELECTROTHERMAL_EVALUATION_CANDIDATES:
        raise ValueError("battery electrothermal evaluation n is not a predeclared candidate")
    return _roster(
        stage=BatteryElectrothermalStage.EVALUATION,
        role=BatteryElectrothermalUnitRole.EVALUATION_PRIMARY,
        per_stratum=n // 6,
        seed=2_026_072_8083,
        token="evaluation",
    )


def reserve_preparations() -> tuple[BatteryElectrothermalPreparation, ...]:
    return _roster(
        stage=BatteryElectrothermalStage.EVALUATION,
        role=BatteryElectrothermalUnitRole.EVALUATION_RESERVE,
        per_stratum=1,
        seed=2_026_072_8084,
        token="evaluation-reserve",
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalFormalAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-formal-assignment'

    gap_id: str
    disposition: BatteryElectrothermalFormalDisposition
    reason: str

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        if not self.reason:
            raise ValueError("battery electrothermal formal assignment reason is required")


_FORMAL_DIRECT_PRIMARY = frozenset(
    {
        "gap.algebra.cross-context-recurrence",
        "gap.algebra.identity-inverse-repetition",
        "gap.algebra.order-commutator",
        "gap.algebra.quotient-lumpability",
        "gap.algebra.sequential-composition",
        "gap.dynamics.hysteresis-return",
        "gap.dynamics.recurrence-stationarity",
        "gap.dynamics.state-closure-memory",
        "gap.geometry.receiver-fibers",
    }
)
_FORMAL_DIRECT_SECONDARY = frozenset(
    {
        "gap.algebra.falsifier-preservation",
        "gap.algebra.signed-opposition",
        "gap.algebra.simultaneous-composition",
        "gap.calculus.numerical-view-convergence",
        "gap.calculus.uncertainty-propagation",
        "gap.dynamics.causal-cones-clock-transport",
        "gap.dynamics.delay-relaxation",
        "gap.geometry.boundary-strata",
        "gap.geometry.support-charts-atlas",
    }
)
_FORMAL_FINITE_PROXY = frozenset(
    {
        "gap.algebra.homogeneity",
        "gap.calculus.dose-scaling",
        "gap.calculus.mixed-port-volterra",
        "gap.calculus.odd-even-susceptibility",
        "gap.calculus.time-scaling",
        "gap.dynamics.local-evolution-operator",
        "gap.dynamics.stability-transient",
        "gap.geometry.metric-structure",
    }
)
_FORMAL_ADMISSION_EXCLUSIONS = frozenset(
    {
        "gap.dynamics.controllability-observability",
        "gap.dynamics.preservation-barriers",
        "gap.geometry.admission-margins",
        "gap.geometry.decision-quotients",
        "gap.geometry.reachability-viability",
    }
)


def formal_assignments(
    register: FormalGapRegister,
) -> tuple[BatteryElectrothermalFormalAssignment, ...]:
    values: list[BatteryElectrothermalFormalAssignment] = []
    for gap in register.gaps:
        if gap.gap_id in _FORMAL_DIRECT_PRIMARY:
            disposition = BatteryElectrothermalFormalDisposition.DIRECT_PRIMARY
            reason = "predeclared denominator-local primary finite test"
        elif gap.gap_id in _FORMAL_DIRECT_SECONDARY:
            disposition = BatteryElectrothermalFormalDisposition.DIRECT_SECONDARY
            reason = "predeclared denominator-local secondary finite test"
        elif gap.gap_id in _FORMAL_FINITE_PROXY:
            disposition = BatteryElectrothermalFormalDisposition.FINITE_PROXY_ONLY
            reason = "finite proxy without continuum or intrinsic-object claim"
        elif gap.gap_id in _FORMAL_ADMISSION_EXCLUSIONS:
            disposition = BatteryElectrothermalFormalDisposition.ADMISSION_NOT_ENTERED_TYPED_EXCLUSION
            reason = "battery electrothermal stops at local law and constructs no admission object"
        else:
            disposition = BatteryElectrothermalFormalDisposition.OPERAND_ABSENT_TYPED_EXCLUSION
            reason = "required typed formal operand is absent"
        values.append(
            BatteryElectrothermalFormalAssignment(
                gap_id=gap.gap_id,
                disposition=disposition,
                reason=reason,
            )
        )
    result = tuple(sorted(values, key=lambda item: item.gap_id))
    if len(result) != 48:
        raise ValueError("battery electrothermal must disposition all 48 formal gaps")
    return result


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-design'

    design_id: str
    qualification_units: tuple[BatteryElectrothermalPreparation, ...]
    development_units: tuple[BatteryElectrothermalPreparation, ...]
    evaluation_units: tuple[BatteryElectrothermalPreparation, ...]
    reserve_units: tuple[BatteryElectrothermalPreparation, ...]
    proposed_denominators: tuple[BatteryElectrothermalDenominator, ...]
    words: tuple[BatteryElectrothermalActionWord, ...]
    formal_assignments: tuple[BatteryElectrothermalFormalAssignment, ...]
    evaluation_n: int
    bootstrap_replicates: int
    bootstrap_seed: int
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        if (
            self.design_id != BATTERY_ELECTROTHERMAL_DESIGN_ID
            or len(self.qualification_units) != 12
            or len(self.development_units) != 12
            or len(self.evaluation_units) != self.evaluation_n
            or len(self.reserve_units) != 6
            or len(self.proposed_denominators) != 12
            or len(self.words) != 15
            or len(self.formal_assignments) != 48
            or self.evaluation_n not in BATTERY_ELECTROTHERMAL_EVALUATION_CANDIDATES
            or self.bootstrap_replicates != BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES
            or self.bootstrap_seed != BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
        ):
            raise ValueError("battery electrothermal design differs from the frozen scientific scope")
        all_units: tuple[BatteryElectrothermalPreparation, ...] = (
            *self.qualification_units,
            *self.development_units,
            *self.evaluation_units,
            *self.reserve_units,
        )
        require_sorted_unique_strings(
            tuple(sorted(item.unit_id for item in all_units)),
            field_name="all_units",
        )
        coordinates = tuple((item.initial_soc, item.initial_temperature_k) for item in all_units)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("battery electrothermal preparation coordinates collide")


def build_design(
    register: FormalGapRegister,
    *,
    evaluation_n: int = BATTERY_ELECTROTHERMAL_PRIMARY_N,
) -> BatteryElectrothermalDesign:
    return BatteryElectrothermalDesign(
        design_id=BATTERY_ELECTROTHERMAL_DESIGN_ID,
        qualification_units=qualification_preparations(),
        development_units=development_preparations(),
        evaluation_units=evaluation_preparations(evaluation_n),
        reserve_units=reserve_preparations(),
        proposed_denominators=denominators(include_x_full=True),
        words=action_words(),
        formal_assignments=formal_assignments(register),
        evaluation_n=evaluation_n,
        bootstrap_replicates=BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES,
        bootstrap_seed=BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalPlanningRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-planning-row'

    family_id: str
    view_id: str
    coordinate: BatteryElectrothermalCoordinate
    pilot_mean: Decimal
    pilot_paired_sd: Decimal
    inflated_planning_sd: Decimal
    planning_class: str

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        validate_stable_id(self.view_id, field_name="view_id")
        for name, value in (
            ("pilot_mean", self.pilot_mean),
            ("pilot_paired_sd", self.pilot_paired_sd),
            ("inflated_planning_sd", self.inflated_planning_sd),
        ):
            validate_decimal(
                value, field_name=name, minimum=Decimal(0) if name != "pilot_mean" else None
            )
        if self.planning_class not in {"effect_target", "null_precision_target"}:
            raise ValueError("battery electrothermal planning class is invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalPlanningCalculation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-planning-calculation'

    calculation_id: str
    predecessor_index_sha256: str
    source_record_sha256: str
    rows: tuple[BatteryElectrothermalPlanningRow, ...]
    candidate_power_minima: tuple[tuple[int, Decimal, Decimal, bool], ...]
    selected_n: int
    four_denominator_word_episode_count: int
    six_denominator_word_episode_count: int
    four_denominator_restart_record_count: int
    six_denominator_restart_record_count: int
    provisional_output_bytes: int
    provisional_wall_seconds: Decimal
    available_external_bytes: int
    host_logical_cpu_count: int
    host_physical_cpu_count: int
    host_memory_bytes: int
    source_feasible_model_thermal_ids: tuple[str, ...]
    calculated_at_utc: str
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.calculation_id, field_name="calculation_id")
        validate_sha256(
            self.predecessor_index_sha256,
            field_name="predecessor_index_sha256",
        )
        validate_sha256(self.source_record_sha256, field_name="source_record_sha256")
        require_sorted_unique_strings(
            self.source_feasible_model_thermal_ids,
            field_name="source_feasible_model_thermal_ids",
            allow_empty=False,
        )
        if (
            len(self.rows) != 12
            or self.selected_n not in BATTERY_ELECTROTHERMAL_EVALUATION_CANDIDATES
            or self.provisional_output_bytes <= 0
            or self.provisional_wall_seconds <= 0
            or self.available_external_bytes <= BATTERY_ELECTROTHERMAL_MINIMUM_FREE_BYTES
            or self.host_logical_cpu_count <= 0
            or self.host_physical_cpu_count <= 0
            or self.host_memory_bytes <= 0
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery electrothermal planning calculation is incomplete")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalStageConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-stage-config'

    config_id: str
    design: ObjectIdentity
    stage: BatteryElectrothermalStage
    units: tuple[BatteryElectrothermalPreparation, ...]
    primary_unit_ids: tuple[str, ...]
    reserve_unit_ids: tuple[str, ...]
    denominators: tuple[BatteryElectrothermalDenominator, ...]
    words: tuple[BatteryElectrothermalActionWord, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        require_sorted_unique_ids(
            self.denominators,
            attribute="denominator_id",
            field_name="denominators",
        )
        require_sorted_unique_ids(self.words, attribute="word_id", field_name="words")
        require_sorted_unique_strings(
            self.primary_unit_ids,
            field_name="primary_unit_ids",
        )
        require_sorted_unique_strings(
            self.reserve_unit_ids,
            field_name="reserve_unit_ids",
        )
        unit_ids = {item.unit_id for item in self.units}
        if (
            set(self.primary_unit_ids) | set(self.reserve_unit_ids) != unit_ids
            or set(self.primary_unit_ids) & set(self.reserve_unit_ids)
            or len(self.denominators) not in {8, 12}
        ):
            raise ValueError("battery electrothermal stage config unit or denominator chart is invalid")
        expected_access = {
            BatteryElectrothermalStage.QUALIFICATION: OutcomeAccess.DEVELOPMENT_VISIBLE,
            BatteryElectrothermalStage.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            BatteryElectrothermalStage.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.stage]
        expected_ceiling = (
            EvidenceCeiling.LOCAL_LAW
            if self.stage is BatteryElectrothermalStage.EVALUATION
            else EvidenceCeiling.NON_PROMOTABLE
        )
        if (
            self.outcome_access is not expected_access
            or self.maximum_evidence_ceiling is not expected_ceiling
        ):
            raise ValueError("battery electrothermal stage visibility or claim ceiling is invalid")


def build_stage_config(
    design: BatteryElectrothermalDesign,
    *,
    stage: BatteryElectrothermalStage,
    include_x_full: bool,
) -> BatteryElectrothermalStageConfig:
    entered = denominators(include_x_full=include_x_full)
    all_words = {item.word_id: item for item in design.words}
    if stage is BatteryElectrothermalStage.QUALIFICATION:
        units = design.qualification_units
        primary = tuple(item.unit_id for item in units)
        reserve: tuple[str, ...] = ()
        word_ids = BATTERY_ELECTROTHERMAL_QUALIFICATION_WORD_IDS
    elif stage is BatteryElectrothermalStage.DEVELOPMENT:
        units = design.development_units
        primary = tuple(item.unit_id for item in units)
        reserve = ()
        word_ids = BATTERY_ELECTROTHERMAL_WORD_IDS
    else:
        units = tuple(
            sorted((*design.evaluation_units, *design.reserve_units), key=lambda x: x.unit_id)
        )
        primary = tuple(item.unit_id for item in design.evaluation_units)
        reserve = tuple(item.unit_id for item in design.reserve_units)
        word_ids = BATTERY_ELECTROTHERMAL_WORD_IDS
    words = tuple(sorted((all_words[item] for item in word_ids), key=lambda x: x.word_id))
    suffix = "six-tier" if include_x_full else "four-tier"
    return BatteryElectrothermalStageConfig(
        config_id=f"config.battery-electrothermal.{stage.value.lower()}.{suffix}",
        design=ObjectIdentity.from_record(design.design_id, design),
        stage=stage,
        units=units,
        primary_unit_ids=tuple(sorted(primary)),
        reserve_unit_ids=tuple(sorted(reserve)),
        denominators=entered,
        words=words,
        outcome_access={
            BatteryElectrothermalStage.QUALIFICATION: OutcomeAccess.DEVELOPMENT_VISIBLE,
            BatteryElectrothermalStage.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            BatteryElectrothermalStage.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[stage],
        maximum_evidence_ceiling=(
            EvidenceCeiling.LOCAL_LAW
            if stage is BatteryElectrothermalStage.EVALUATION
            else EvidenceCeiling.NON_PROMOTABLE
        ),
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalActionLedgerRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-action-ledger-row'

    request_clock_s: int
    requested_current_delta_a: Decimal
    accepted_current_delta_a: Decimal
    applied_current_delta_a: Decimal
    realized_current_delta_a: Decimal
    requested_ambient_delta_k: Decimal
    accepted_ambient_delta_k: Decimal
    applied_ambient_delta_k: Decimal
    realized_ambient_delta_k: Decimal

    def __post_init__(self) -> None:
        if self.request_clock_s not in BATTERY_ELECTROTHERMAL_BOUNDARIES_S:
            raise ValueError("battery electrothermal action row clock differs")
        if not (
            self.requested_current_delta_a
            == self.accepted_current_delta_a
            == self.applied_current_delta_a
            == self.realized_current_delta_a
            and self.requested_ambient_delta_k
            == self.accepted_ambient_delta_k
            == self.applied_ambient_delta_k
            == self.realized_ambient_delta_k
        ):
            raise ValueError("battery electrothermal action realization is not exact")


def action_ledger(word: BatteryElectrothermalActionWord) -> tuple[BatteryElectrothermalActionLedgerRow, ...]:
    return tuple(
        BatteryElectrothermalActionLedgerRow(
            request_clock_s=clock,
            requested_current_delta_a=word.values_at(clock)[0],
            accepted_current_delta_a=word.values_at(clock)[0],
            applied_current_delta_a=word.values_at(clock)[0],
            realized_current_delta_a=word.values_at(clock)[0],
            requested_ambient_delta_k=word.values_at(clock)[1],
            accepted_ambient_delta_k=word.values_at(clock)[1],
            applied_ambient_delta_k=word.values_at(clock)[1],
            realized_ambient_delta_k=word.values_at(clock)[1],
        )
        for clock in BATTERY_ELECTROTHERMAL_BOUNDARIES_S
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalCheckpointState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-checkpoint-state'

    clock_s: int
    state_y: tuple[Decimal, ...]
    state_layout_sha256: str

    def __post_init__(self) -> None:
        if self.clock_s not in {300, 450} or not self.state_y:
            raise ValueError("battery electrothermal checkpoint state is invalid")
        validate_sha256(self.state_layout_sha256, field_name="state_layout_sha256")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalDiagnosticSeries(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-diagnostic-series'

    diagnostic_id: str
    native_unit: str
    reduction: str
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.diagnostic_id, field_name="diagnostic_id")
        if not self.native_unit or not self.reduction:
            raise ValueError("battery electrothermal diagnostic metadata is required")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalCapturedEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-captured-episode'

    episode_id: str
    config: ObjectIdentity
    unit: ObjectIdentity
    denominator: ObjectIdentity
    word: ObjectIdentity
    time_s: tuple[Decimal, ...]
    discharge_capacity_ah: tuple[Decimal, ...]
    mean_temperature_k: tuple[Decimal, ...]
    maximum_temperature_k: tuple[Decimal, ...]
    terminal_voltage_v: tuple[Decimal, ...]
    source_current_a: tuple[Decimal, ...]
    source_ambient_temperature_k: tuple[Decimal, ...]
    diagnostics: tuple[BatteryElectrothermalDiagnosticSeries, ...]
    absent_diagnostic_ids: tuple[str, ...]
    temperature_field_shape: tuple[int, int] | None
    temperature_field_k: tuple[Decimal, ...]
    checkpoint_states: tuple[BatteryElectrothermalCheckpointState, ...]
    action_rows: tuple[BatteryElectrothermalActionLedgerRow, ...]
    state_vector_length: int
    termination: str
    disposition: BatteryElectrothermalEpisodeDisposition
    reason_codes: tuple[str, ...]
    source_build_seconds: Decimal
    runtime_seconds: Decimal
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.absent_diagnostic_ids,
            field_name="absent_diagnostic_ids",
        )
        lengths = {
            len(item)
            for item in (
                self.time_s,
                self.discharge_capacity_ah,
                self.mean_temperature_k,
                self.maximum_temperature_k,
                self.terminal_voltage_v,
                self.source_current_a,
                self.source_ambient_temperature_k,
            )
        }
        if len(lengths) != 1:
            raise ValueError("battery electrothermal primary receiver arrays differ in length")
        if self.temperature_field_shape is None:
            if self.temperature_field_k:
                raise ValueError("battery electrothermal temperature field lacks a shape")
        elif math.prod(self.temperature_field_shape) != len(self.temperature_field_k):
            raise ValueError("battery electrothermal temperature field shape differs")
        if self.disposition is BatteryElectrothermalEpisodeDisposition.COMPLETE:
            if (
                not self.time_s
                or self.time_s[0] != 0
                or self.time_s[-1] != BATTERY_ELECTROTHERMAL_HORIZON_S
                or tuple(item.clock_s for item in self.checkpoint_states) != (300, 450)
                or len(self.action_rows) != len(BATTERY_ELECTROTHERMAL_BOUNDARIES_S)
                or self.termination != "final time"
                or self.reason_codes
                or self.state_vector_length <= 0
                or self.source_build_seconds < 0
            ):
                raise ValueError("complete battery electrothermal episode is incomplete")


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("battery electrothermal produced a nonfinite scalar")
    return Decimal(f"{value:.15g}")


def _source_options(denominator: BatteryElectrothermalDenominator) -> dict[str, str]:
    values = {
        "thermal": {
            BatteryElectrothermalThermal.ISOTHERMAL: "isothermal",
            BatteryElectrothermalThermal.LUMPED: "lumped",
            BatteryElectrothermalThermal.X_FULL: "x-full",
        }[denominator.thermal],
        "cell geometry": "pouch",
    }
    if denominator.thermal is BatteryElectrothermalThermal.ISOTHERMAL:
        values["calculate heat source for isothermal models"] = "true"
    return values


def _inputs(
    unit: BatteryElectrothermalPreparation,
    current_delta_a: Decimal,
    ambient_delta_k: Decimal,
) -> dict[str, float]:
    return {
        "Current function [A]": float(current_delta_a),
        "Ambient temperature [K]": float(unit.initial_temperature_k + ambient_delta_k),
    }


def _runtime_objects(
    unit: BatteryElectrothermalPreparation,
    denominator: BatteryElectrothermalDenominator,
) -> tuple[Any, Any]:
    import pybamm  # type: ignore[import-untyped]

    if pybamm.__version__ != BATTERY_ELECTROTHERMAL_SOURCE_VERSION:
        raise RuntimeError("installed PyBaMM differs from the battery electrothermal source")
    model_class = (
        pybamm.lithium_ion.SPMe if denominator.model is BatteryElectrothermalModel.SPME else pybamm.lithium_ion.DFN
    )
    model = model_class(options=_source_options(denominator))
    parameters = pybamm.ParameterValues(BATTERY_ELECTROTHERMAL_PARAMETER_SET)
    parameters["Current function [A]"] = "[input]"
    parameters["Ambient temperature [K]"] = "[input]"
    parameters["Initial temperature [K]"] = float(unit.initial_temperature_k)
    if denominator.view.solver is BatteryElectrothermalSolver.CASADI:
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


def _state_layout_sha256(solution: Any) -> str:
    model = solution.all_models[-1]
    layout = tuple(
        sorted(
            (
                str(variable),
                tuple(
                    (
                        int(item.start or 0),
                        int(item.stop or 0),
                        int(item.step or 1),
                    )
                    for item in (slices if isinstance(slices, list) else [slices])
                ),
            )
            for variable, slices in model.y_slices.items()
        )
    )
    return sha256(canonical_json_bytes(layout)).hexdigest()


def _step_schedule(
    *,
    unit: BatteryElectrothermalPreparation,
    denominator: BatteryElectrothermalDenominator,
    word: BatteryElectrothermalActionWord,
    stop_s: int = BATTERY_ELECTROTHERMAL_HORIZON_S,
) -> tuple[Any, Any, Mapping[int, Any], Decimal]:
    build_started = time.monotonic()
    pybamm, simulation = _runtime_objects(unit, denominator)
    current, ambient = word.values_at(0)
    simulation.build(
        initial_soc=float(unit.initial_soc),
        inputs=_inputs(unit, current, ambient),
    )
    source_build_seconds = _decimal(time.monotonic() - build_started)
    checkpoints: dict[int, Any] = {}
    interval = float(denominator.view.nominal_output_interval_s)
    for start, stop in zip(
        BATTERY_ELECTROTHERMAL_BOUNDARIES_S[:-1],
        BATTERY_ELECTROTHERMAL_BOUNDARIES_S[1:],
        strict=True,
    ):
        if start >= stop_s:
            break
        segment_stop = min(stop, stop_s)
        current, ambient = word.values_at(start)
        duration = segment_stop - start
        relative = np.arange(0.0, duration + interval, interval, dtype=np.float64)
        relative[-1] = float(duration)
        simulation.step(
            duration,
            t_eval=np.unique(relative),
            inputs=_inputs(unit, current, ambient),
        )
        if segment_stop in {300, 450}:
            checkpoints[segment_stop] = simulation.solution.last_state
        if segment_stop == stop_s:
            break
    return pybamm, simulation, checkpoints, source_build_seconds


def _sample_clock(interval: Decimal, *, start: int = 0, stop: int = 900) -> np.ndarray:
    values = np.arange(start, stop + float(interval), float(interval), dtype=np.float64)
    values[-1] = float(stop)
    return np.unique(values)


def _variable(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = np.asarray(solution[name](times), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"battery electrothermal source variable {name!r} is nonfinite")
    return values


def _scalar_series(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = _variable(solution, name, times).reshape(-1)
    if values.size != times.size:
        raise RuntimeError(f"battery electrothermal source variable {name!r} is not scalar by clock")
    return values


def _field_by_time(solution: Any, name: str, times: np.ndarray) -> np.ndarray:
    values = _variable(solution, name, times)
    if values.ndim == 1:
        if values.size != times.size:
            raise RuntimeError(f"battery electrothermal source field {name!r} has an invalid shape")
        return values.reshape(1, -1)
    if values.shape[-1] == times.size:
        return values.reshape(-1, times.size)
    if values.shape[0] == times.size:
        return np.moveaxis(values, 0, -1).reshape(-1, times.size)
    raise RuntimeError(f"battery electrothermal source field {name!r} lacks a clock axis")


def _diagnostic(
    diagnostic_id: str,
    unit: str,
    reduction: str,
    values: np.ndarray,
) -> BatteryElectrothermalDiagnosticSeries:
    return BatteryElectrothermalDiagnosticSeries(
        diagnostic_id=diagnostic_id,
        native_unit=unit,
        reduction=reduction,
        values=tuple(_decimal(float(item)) for item in values),
    )


def acquire_episode(
    *,
    config: BatteryElectrothermalStageConfig,
    unit_id: str,
    denominator_id: str,
    word_id: str,
    episode_suffix: str,
) -> BatteryElectrothermalCapturedEpisode:
    """Acquire one complete word while preserving valid partial outcomes."""

    validate_stable_id(episode_suffix, field_name="episode_suffix")
    unit = next(item for item in config.units if item.unit_id == unit_id)
    denominator = next(
        item for item in config.denominators if item.denominator_id == denominator_id
    )
    word = next(item for item in config.words if item.word_id == word_id)
    started = time.monotonic()
    disposition = BatteryElectrothermalEpisodeDisposition.COMPLETE
    reasons: set[str] = set()
    termination = "not-started"
    times = np.asarray([], dtype=np.float64)
    arrays: dict[str, np.ndarray] = {}
    diagnostics: list[BatteryElectrothermalDiagnosticSeries] = []
    absent_diagnostics: set[str] = set()
    temperature_field = np.asarray([], dtype=np.float64)
    temperature_shape: tuple[int, int] | None = None
    checkpoint_records: tuple[BatteryElectrothermalCheckpointState, ...] = ()
    state_length = 0
    source_build_seconds = Decimal(0)
    try:
        _, simulation, checkpoints, source_build_seconds = _step_schedule(
            unit=unit,
            denominator=denominator,
            word=word,
        )
        solution = simulation.solution
        termination = str(solution.termination)
        times = _sample_clock(denominator.view.nominal_output_interval_s)
        arrays["capacity"] = _scalar_series(
            solution,
            "Discharge capacity [A.h]",
            times,
        )
        arrays["temperature"] = _scalar_series(
            solution,
            "Volume-averaged cell temperature [K]",
            times,
        )
        arrays["voltage"] = _scalar_series(solution, "Terminal voltage [V]", times)
        source_current = _scalar_series(solution, "Current [A]", times)
        source_ambient = _scalar_series(solution, "Ambient temperature [K]", times)
        arrays["current"] = source_current
        arrays["ambient"] = source_ambient

        cell_temperature = _field_by_time(solution, "Cell temperature [K]", times)
        arrays["maximum_temperature"] = np.max(cell_temperature, axis=0)
        if denominator.thermal is BatteryElectrothermalThermal.X_FULL:
            temperature_field = cell_temperature
            temperature_shape = (
                int(cell_temperature.shape[0]),
                int(cell_temperature.shape[1]),
            )

        scalar_diagnostics = (
            (
                "diagnostic.total-heating",
                "Volume-averaged total heating [W.m-3]",
                "W.m-3",
            ),
            (
                "diagnostic.ohmic-heating",
                "Volume-averaged Ohmic heating [W.m-3]",
                "W.m-3",
            ),
            (
                "diagnostic.irreversible-heating",
                "Volume-averaged irreversible electrochemical heating [W.m-3]",
                "W.m-3",
            ),
            (
                "diagnostic.reversible-heating",
                "Volume-averaged reversible heating [W.m-3]",
                "W.m-3",
            ),
            (
                "diagnostic.negative-reaction-overpotential",
                "X-averaged negative electrode reaction overpotential [V]",
                "V",
            ),
            (
                "diagnostic.positive-reaction-overpotential",
                "X-averaged positive electrode reaction overpotential [V]",
                "V",
            ),
        )
        for diagnostic_id, variable_name, native_unit in scalar_diagnostics:
            try:
                values = _scalar_series(solution, variable_name, times)
            except (KeyError, RuntimeError, ValueError):
                absent_diagnostics.add(diagnostic_id)
                continue
            diagnostics.append(
                _diagnostic(
                    diagnostic_id,
                    native_unit,
                    "source-volume-or-x-average",
                    values,
                )
            )
        field_diagnostics = (
            (
                "electrolyte-concentration",
                "Electrolyte concentration [mol.m-3]",
                "mol.m-3",
            ),
            (
                "negative-particle-surface-stoichiometry",
                "Negative particle surface stoichiometry",
                "1",
            ),
            (
                "positive-particle-surface-stoichiometry",
                "Positive particle surface stoichiometry",
                "1",
            ),
        )
        for token, variable_name, native_unit in field_diagnostics:
            try:
                values = _field_by_time(solution, variable_name, times)
            except (KeyError, RuntimeError, ValueError):
                absent_diagnostics.update(
                    (
                        f"diagnostic.{token}-minimum",
                        f"diagnostic.{token}-maximum",
                        f"diagnostic.{token}-contrast",
                    )
                )
                continue
            diagnostics.extend(
                (
                    _diagnostic(
                        f"diagnostic.{token}-minimum",
                        native_unit,
                        "spatial-minimum",
                        np.min(values, axis=0),
                    ),
                    _diagnostic(
                        f"diagnostic.{token}-maximum",
                        native_unit,
                        "spatial-maximum",
                        np.max(values, axis=0),
                    ),
                    _diagnostic(
                        f"diagnostic.{token}-contrast",
                        native_unit,
                        "spatial-maximum-minus-minimum",
                        np.max(values, axis=0) - np.min(values, axis=0),
                    ),
                )
            )

        checkpoint_values: list[BatteryElectrothermalCheckpointState] = []
        for clock in (300, 450):
            checkpoint = checkpoints[clock]
            values = np.asarray(checkpoint.y, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise RuntimeError("battery electrothermal checkpoint state is nonfinite")
            layout_sha256 = _state_layout_sha256(checkpoint)
            checkpoint_values.append(
                BatteryElectrothermalCheckpointState(
                    clock_s=clock,
                    state_y=tuple(_decimal(float(item)) for item in values),
                    state_layout_sha256=layout_sha256,
                )
            )
        checkpoint_records = tuple(checkpoint_values)
        state_length = len(checkpoint_records[0].state_y)
        if any(len(item.state_y) != state_length for item in checkpoint_records):
            raise RuntimeError("battery electrothermal state vector length changes by checkpoint")

        # Boundary samples can belong to the preceding source segment. Exact
        # realization is checked at strict segment-interior output clocks.
        for start, stop in zip(
            BATTERY_ELECTROTHERMAL_BOUNDARIES_S[:-1],
            BATTERY_ELECTROTHERMAL_BOUNDARIES_S[1:],
            strict=True,
        ):
            interior = (times > start) & (times < stop)
            if not np.any(interior):
                continue
            expected_current, expected_ambient = word.values_at(start)
            if not np.allclose(
                source_current[interior],
                float(expected_current),
                rtol=0,
                atol=1e-10,
            ) or not np.allclose(
                source_ambient[interior],
                float(unit.initial_temperature_k + expected_ambient),
                rtol=0,
                atol=1e-8,
            ):
                disposition = BatteryElectrothermalEpisodeDisposition.DELIVERY_INVALID
                reasons.add("SOURCE_INPUT_REALIZATION_MISMATCH")
        if termination != "final time" or not math.isclose(
            float(times[-1]),
            BATTERY_ELECTROTHERMAL_HORIZON_S,
            abs_tol=1e-8,
        ):
            disposition = BatteryElectrothermalEpisodeDisposition.SIMULATOR_TERMINATED
            reasons.add("SIMULATOR_TERMINATED_BEFORE_HORIZON")
    except Exception as error:
        if times.size:
            disposition = BatteryElectrothermalEpisodeDisposition.SCIENTIFIC_PARTIAL
        else:
            disposition = BatteryElectrothermalEpisodeDisposition.TECHNICAL_OBSERVATION_FAILURE
        reasons.add(f"PYBAMM_{type(error).__name__.upper()}")
        termination = f"error:{type(error).__name__}"

    def encode(name: str) -> tuple[Decimal, ...]:
        return tuple(_decimal(float(item)) for item in arrays.get(name, ()))

    token = (
        f"{unit_id.removeprefix('unit.battery-electrothermal.')}."
        f"{denominator_id.removeprefix('denominator.pybamm.')}."
        f"{word_id.removeprefix('word.battery-electrothermal.')}.{episode_suffix}"
    )
    return BatteryElectrothermalCapturedEpisode(
        episode_id=f"episode.battery-electrothermal.{token}",
        config=ObjectIdentity.from_record(config.config_id, config),
        unit=ObjectIdentity.from_record(unit.unit_id, unit),
        denominator=ObjectIdentity.from_record(
            denominator.denominator_id,
            denominator,
        ),
        word=ObjectIdentity.from_record(word.word_id, word),
        time_s=tuple(_decimal(float(item)) for item in times),
        discharge_capacity_ah=encode("capacity"),
        mean_temperature_k=encode("temperature"),
        maximum_temperature_k=encode("maximum_temperature"),
        terminal_voltage_v=encode("voltage"),
        source_current_a=encode("current"),
        source_ambient_temperature_k=encode("ambient"),
        diagnostics=tuple(sorted(diagnostics, key=lambda item: item.diagnostic_id)),
        absent_diagnostic_ids=tuple(sorted(absent_diagnostics)),
        temperature_field_shape=temperature_shape,
        temperature_field_k=tuple(_decimal(float(item)) for item in temperature_field.reshape(-1)),
        checkpoint_states=checkpoint_records,
        action_rows=action_ledger(word),
        state_vector_length=state_length,
        termination=termination,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
        source_build_seconds=source_build_seconds,
        runtime_seconds=_decimal(time.monotonic() - started),
        outcome_access=config.outcome_access,
    )


def checkpoint_state(
    episode: BatteryElectrothermalCapturedEpisode,
    clock_s: int,
) -> BatteryElectrothermalCheckpointState:
    try:
        return next(item for item in episode.checkpoint_states if item.clock_s == clock_s)
    except StopIteration as error:
        raise ValueError("battery electrothermal episode lacks the requested checkpoint") from error


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalDonorEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-donor-entry'

    episode_id: str
    unit_id: str
    denominator_id: str
    word_id: str
    checkpoint_s: int
    soc: Decimal
    mean_temperature_k: Decimal
    terminal_voltage_v: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("unit_id", self.unit_id),
            ("denominator_id", self.denominator_id),
            ("word_id", self.word_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.checkpoint_s not in {300, 450}:
            raise ValueError("battery electrothermal donor checkpoint is invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalDonorCatalogue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/battery-electrothermal-donor-catalogue'

    catalogue_id: str
    development_config: ObjectIdentity
    entries: tuple[BatteryElectrothermalDonorEntry, ...]
    medians: tuple[tuple[str, int, str, Decimal], ...]
    scales: tuple[tuple[str, int, str, Decimal], ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.catalogue_id, field_name="catalogue_id")
        if (
            not self.entries
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery electrothermal donor catalogue is incomplete")


def _episode_reduced_coordinates(
    config: BatteryElectrothermalStageConfig,
    episode: BatteryElectrothermalCapturedEpisode,
    clock_s: int,
) -> tuple[Decimal, Decimal, Decimal]:
    unit = next(item for item in config.units if item.unit_id == episode.unit.object_id)
    try:
        index = episode.time_s.index(Decimal(clock_s))
    except ValueError as error:
        raise ValueError("battery electrothermal episode lacks the reduced-state clock") from error
    return (
        unit.initial_soc - episode.discharge_capacity_ah[index] / Decimal(5),
        episode.mean_temperature_k[index],
        episode.terminal_voltage_v[index],
    )


def episode_reduced_coordinates(
    config: BatteryElectrothermalStageConfig,
    episode: BatteryElectrothermalCapturedEpisode,
    clock_s: int,
) -> tuple[Decimal, Decimal, Decimal]:
    """Return the frozen reduced coordinates used for donor matching."""

    return _episode_reduced_coordinates(config, episode, clock_s)


def build_donor_catalogue(
    config: BatteryElectrothermalStageConfig,
    episodes: Iterable[BatteryElectrothermalCapturedEpisode],
) -> BatteryElectrothermalDonorCatalogue:
    if config.stage is not BatteryElectrothermalStage.DEVELOPMENT:
        raise ValueError("battery electrothermal donor catalogue requires development")
    expected = {
        (unit_id, denominator.denominator_id, word.word_id)
        for unit_id in config.primary_unit_ids
        for denominator in config.denominators
        for word in config.words
    }
    entries: list[BatteryElectrothermalDonorEntry] = []
    seen: set[tuple[str, str, str]] = set()
    medians: list[tuple[str, int, str, Decimal]] = []
    scales: list[tuple[str, int, str, Decimal]] = []
    coordinate_ids = ("soc", "temperature", "voltage")
    for episode in episodes:
        if episode.disposition is not BatteryElectrothermalEpisodeDisposition.COMPLETE:
            continue
        key = (
            episode.unit.object_id,
            episode.denominator.object_id,
            episode.word.object_id,
        )
        if key in seen:
            raise ValueError("battery electrothermal donor episode key is duplicated")
        seen.add(key)
        for checkpoint_s in (300, 450):
            coordinates = _episode_reduced_coordinates(
                config,
                episode,
                checkpoint_s,
            )
            entries.append(
                BatteryElectrothermalDonorEntry(
                    episode_id=episode.episode_id,
                    unit_id=episode.unit.object_id,
                    denominator_id=episode.denominator.object_id,
                    word_id=episode.word.object_id,
                    checkpoint_s=checkpoint_s,
                    soc=coordinates[0],
                    mean_temperature_k=coordinates[1],
                    terminal_voltage_v=coordinates[2],
                )
            )
    if seen != expected:
        raise ValueError("battery electrothermal donor catalogue lacks a complete development chart")
    for denominator in config.denominators:
        for checkpoint_s in (300, 450):
            local_entries = [
                item
                for item in entries
                if (
                    item.denominator_id == denominator.denominator_id
                    and item.checkpoint_s == checkpoint_s
                )
            ]
            matrix = np.asarray(
                [
                    [
                        float(item.soc),
                        float(item.mean_temperature_k),
                        float(item.terminal_voltage_v),
                    ]
                    for item in local_entries
                ],
                dtype=np.float64,
            )
            median = np.median(matrix, axis=0)
            mad = np.median(np.abs(matrix - median), axis=0)
            mad[mad == 0] = 1.0
            medians.extend(
                (
                    denominator.denominator_id,
                    checkpoint_s,
                    coordinate,
                    _decimal(float(value)),
                )
                for coordinate, value in zip(coordinate_ids, median, strict=True)
            )
            scales.extend(
                (
                    denominator.denominator_id,
                    checkpoint_s,
                    coordinate,
                    _decimal(float(value)),
                )
                for coordinate, value in zip(coordinate_ids, mad, strict=True)
            )
    return BatteryElectrothermalDonorCatalogue(
        catalogue_id="donor-catalogue.battery-electrothermal.development",
        development_config=ObjectIdentity.from_record(config.config_id, config),
        entries=tuple(
            sorted(
                entries,
                key=lambda item: (
                    item.denominator_id,
                    item.checkpoint_s,
                    item.unit_id,
                    item.word_id,
                ),
            )
        ),
        medians=tuple(sorted(medians)),
        scales=tuple(sorted(scales)),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def select_donor(
    catalogue: BatteryElectrothermalDonorCatalogue,
    *,
    denominator_id: str,
    checkpoint_s: int,
    target: tuple[Decimal, Decimal, Decimal],
) -> BatteryElectrothermalDonorEntry:
    coordinate_ids = ("soc", "temperature", "voltage")
    median = np.asarray(
        [
            float(value)
            for coordinate in coordinate_ids
            for denominator, clock, name, value in catalogue.medians
            if (denominator == denominator_id and clock == checkpoint_s and name == coordinate)
        ],
        dtype=np.float64,
    )
    scale = np.asarray(
        [
            float(value)
            for coordinate in coordinate_ids
            for denominator, clock, name, value in catalogue.scales
            if (denominator == denominator_id and clock == checkpoint_s and name == coordinate)
        ],
        dtype=np.float64,
    )
    if median.size != 3 or scale.size != 3:
        raise ValueError("battery electrothermal donor scale is absent for the denominator/checkpoint")
    target_values = np.asarray([float(item) for item in target], dtype=np.float64)
    ranked: list[tuple[float, str, str, BatteryElectrothermalDonorEntry]] = []
    for item in catalogue.entries:
        if item.denominator_id != denominator_id or item.checkpoint_s != checkpoint_s:
            continue
        values = np.asarray(
            [
                float(item.soc),
                float(item.mean_temperature_k),
                float(item.terminal_voltage_v),
            ],
            dtype=np.float64,
        )
        distance = float(
            np.sum(((values - median) / scale - (target_values - median) / scale) ** 2)
        )
        ranked.append((distance, item.unit_id, item.word_id, item))
    if not ranked:
        raise ValueError("battery electrothermal donor pool is empty")
    return min(ranked, key=lambda value: value[:3])[3]


def _fabricated_solution(
    pybamm: Any,
    template_state: Any,
    state_y: Sequence[Decimal],
    checkpoint_s: int,
) -> Any:
    values = np.asarray([float(item) for item in state_y], dtype=np.float64).reshape(-1, 1)
    return pybamm.Solution(
        [np.asarray([float(checkpoint_s)], dtype=np.float64)],
        [values],
        [template_state.all_models[-1]],
        [template_state.all_inputs[-1]],
        termination="final time",
        check_solution=False,
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalRestartResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-restart-result'

    restart_id: str
    config: ObjectIdentity
    unit_id: str
    denominator_id: str
    history_word_id: str
    checkpoint_s: int
    continuation_end_s: int
    donor_unit_id: str
    donor_word_id: str
    time_s: tuple[Decimal, ...]
    actual_capacity_ah: tuple[Decimal, ...]
    exact_capacity_ah: tuple[Decimal, ...]
    reduced_capacity_ah: tuple[Decimal, ...]
    actual_temperature_k: tuple[Decimal, ...]
    exact_temperature_k: tuple[Decimal, ...]
    reduced_temperature_k: tuple[Decimal, ...]
    actual_maximum_temperature_k: tuple[Decimal, ...]
    exact_maximum_temperature_k: tuple[Decimal, ...]
    reduced_maximum_temperature_k: tuple[Decimal, ...]
    actual_voltage_v: tuple[Decimal, ...]
    exact_voltage_v: tuple[Decimal, ...]
    reduced_voltage_v: tuple[Decimal, ...]
    exact_capacity_error_ah: Decimal | None
    exact_temperature_error_k: Decimal | None
    exact_maximum_temperature_error_k: Decimal | None
    exact_voltage_error_v: Decimal | None
    reduced_capacity_error_ah: Decimal | None
    reduced_temperature_error_k: Decimal | None
    reduced_maximum_temperature_error_k: Decimal | None
    reduced_voltage_error_v: Decimal | None
    reduced_operational_error: Decimal | None
    exact_restart_complete: bool
    reduced_restart_complete: bool
    reason_codes: tuple[str, ...]
    runtime_seconds: Decimal
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("restart_id", self.restart_id),
            ("unit_id", self.unit_id),
            ("denominator_id", self.denominator_id),
            ("history_word_id", self.history_word_id),
            ("donor_unit_id", self.donor_unit_id),
            ("donor_word_id", self.donor_word_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.checkpoint_s not in {300, 450}
            or self.continuation_end_s - self.checkpoint_s != 300
            or self.runtime_seconds < 0
        ):
            raise ValueError("battery electrothermal restart clock or runtime is invalid")
        actual_arrays = (
            self.actual_capacity_ah,
            self.actual_temperature_k,
            self.actual_maximum_temperature_k,
            self.actual_voltage_v,
        )
        exact_arrays = (
            self.exact_capacity_ah,
            self.exact_temperature_k,
            self.exact_maximum_temperature_k,
            self.exact_voltage_v,
        )
        reduced_arrays = (
            self.reduced_capacity_ah,
            self.reduced_temperature_k,
            self.reduced_maximum_temperature_k,
            self.reduced_voltage_v,
        )
        if self.exact_restart_complete and (
            not self.time_s
            or any(len(item) != len(self.time_s) for item in (*actual_arrays, *exact_arrays))
            or any(
                item is None
                for item in (
                    self.exact_capacity_error_ah,
                    self.exact_temperature_error_k,
                    self.exact_maximum_temperature_error_k,
                    self.exact_voltage_error_v,
                )
            )
        ):
            raise ValueError("complete battery electrothermal exact restart is incomplete")
        if self.reduced_restart_complete and (
            not self.time_s
            or any(len(item) != len(self.time_s) for item in (*actual_arrays, *reduced_arrays))
            or any(
                item is None
                for item in (
                    self.reduced_capacity_error_ah,
                    self.reduced_temperature_error_k,
                    self.reduced_maximum_temperature_error_k,
                    self.reduced_voltage_error_v,
                    self.reduced_operational_error,
                )
            )
        ):
            raise ValueError("complete battery electrothermal reduced restart is incomplete")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalRestartSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-restart-summary'

    restart_id: str
    unit_id: str
    denominator_id: str
    history_word_id: str
    checkpoint_s: int
    exact_capacity_error_ah: Decimal | None
    exact_temperature_error_k: Decimal | None
    exact_maximum_temperature_error_k: Decimal | None
    exact_voltage_error_v: Decimal | None
    reduced_operational_error: Decimal | None
    exact_restart_complete: bool
    reduced_restart_complete: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("restart_id", self.restart_id),
            ("unit_id", self.unit_id),
            ("denominator_id", self.denominator_id),
            ("history_word_id", self.history_word_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.checkpoint_s not in {300, 450}:
            raise ValueError("battery electrothermal restart summary checkpoint is invalid")


def summarize_restart(result: BatteryElectrothermalRestartResult) -> BatteryElectrothermalRestartSummary:
    return BatteryElectrothermalRestartSummary(
        restart_id=result.restart_id,
        unit_id=result.unit_id,
        denominator_id=result.denominator_id,
        history_word_id=result.history_word_id,
        checkpoint_s=result.checkpoint_s,
        exact_capacity_error_ah=result.exact_capacity_error_ah,
        exact_temperature_error_k=result.exact_temperature_error_k,
        exact_maximum_temperature_error_k=result.exact_maximum_temperature_error_k,
        exact_voltage_error_v=result.exact_voltage_error_v,
        reduced_operational_error=result.reduced_operational_error,
        exact_restart_complete=result.exact_restart_complete,
        reduced_restart_complete=result.reduced_restart_complete,
        reason_codes=result.reason_codes,
    )


def acquire_restart_result(
    *,
    config: BatteryElectrothermalStageConfig,
    evaluation_episode: BatteryElectrothermalCapturedEpisode,
    donor_episode: BatteryElectrothermalCapturedEpisode,
    checkpoint_s: int,
    continuation_end_s: int,
    restart_suffix: str,
) -> BatteryElectrothermalRestartResult:
    """Run actual, exact serialized-state, and reduced donor continuations."""

    validate_stable_id(restart_suffix, field_name="restart_suffix")
    started = time.monotonic()
    if (
        evaluation_episode.denominator.object_id != donor_episode.denominator.object_id
        or evaluation_episode.disposition is not BatteryElectrothermalEpisodeDisposition.COMPLETE
        or donor_episode.disposition is not BatteryElectrothermalEpisodeDisposition.COMPLETE
        or checkpoint_s not in {300, 450}
        or continuation_end_s - checkpoint_s != 300
    ):
        raise ValueError("battery electrothermal restart operands are incompatible")
    unit = next(item for item in config.units if item.unit_id == evaluation_episode.unit.object_id)
    denominator = next(
        item
        for item in config.denominators
        if item.denominator_id == evaluation_episode.denominator.object_id
    )
    word = next(item for item in config.words if item.word_id == evaluation_episode.word.object_id)
    exact_state = checkpoint_state(evaluation_episode, checkpoint_s)
    donor_state = checkpoint_state(donor_episode, checkpoint_s)
    if exact_state.state_layout_sha256 != donor_state.state_layout_sha256:
        raise ValueError("battery electrothermal donor state layout differs within a denominator")
    reasons: set[str] = set()
    exact_complete = reduced_complete = False
    clock = _sample_clock(
        denominator.view.nominal_output_interval_s,
        start=checkpoint_s,
        stop=continuation_end_s,
    )
    relative = clock - checkpoint_s
    values: dict[str, np.ndarray] = {}
    exact_errors: dict[str, Decimal | None] = {
        "capacity": None,
        "temperature": None,
        "maximum_temperature": None,
        "voltage": None,
    }
    reduced_errors = dict(exact_errors)
    reduced_operational_error = None
    try:
        pybamm, actual_simulation, checkpoints, _ = _step_schedule(
            unit=unit,
            denominator=denominator,
            word=word,
            stop_s=checkpoint_s,
        )
        prefix = checkpoints[checkpoint_s]
        actual_solution = actual_simulation.step(
            continuation_end_s - checkpoint_s,
            t_eval=relative,
            starting_solution=prefix,
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        values["actual_capacity"] = _scalar_series(
            actual_solution,
            "Discharge capacity [A.h]",
            clock,
        )
        values["actual_temperature"] = _scalar_series(
            actual_solution,
            "Volume-averaged cell temperature [K]",
            clock,
        )
        values["actual_voltage"] = _scalar_series(
            actual_solution,
            "Terminal voltage [V]",
            clock,
        )
        values["actual_maximum_temperature"] = np.max(
            _field_by_time(actual_solution, "Cell temperature [K]", clock),
            axis=0,
        )

        _, exact_simulation = _runtime_objects(unit, denominator)
        exact_simulation.build(
            initial_soc=float(unit.initial_soc),
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        exact_start = _fabricated_solution(
            pybamm,
            prefix,
            exact_state.state_y,
            checkpoint_s,
        )
        exact_solution = exact_simulation.step(
            continuation_end_s - checkpoint_s,
            t_eval=relative,
            starting_solution=exact_start,
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        values["exact_capacity"] = _scalar_series(
            exact_solution,
            "Discharge capacity [A.h]",
            clock,
        )
        values["exact_temperature"] = _scalar_series(
            exact_solution,
            "Volume-averaged cell temperature [K]",
            clock,
        )
        values["exact_voltage"] = _scalar_series(
            exact_solution,
            "Terminal voltage [V]",
            clock,
        )
        values["exact_maximum_temperature"] = np.max(
            _field_by_time(exact_solution, "Cell temperature [K]", clock),
            axis=0,
        )
        exact_complete = str(exact_solution.termination) == "final time"

        _, reduced_simulation = _runtime_objects(unit, denominator)
        reduced_simulation.build(
            initial_soc=float(unit.initial_soc),
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        reduced_start = _fabricated_solution(
            pybamm,
            prefix,
            donor_state.state_y,
            checkpoint_s,
        )
        reduced_solution = reduced_simulation.step(
            continuation_end_s - checkpoint_s,
            t_eval=relative,
            starting_solution=reduced_start,
            inputs=_inputs(unit, Decimal(0), Decimal(0)),
        )
        values["reduced_capacity"] = _scalar_series(
            reduced_solution,
            "Discharge capacity [A.h]",
            clock,
        )
        values["reduced_temperature"] = _scalar_series(
            reduced_solution,
            "Volume-averaged cell temperature [K]",
            clock,
        )
        values["reduced_voltage"] = _scalar_series(
            reduced_solution,
            "Terminal voltage [V]",
            clock,
        )
        values["reduced_maximum_temperature"] = np.max(
            _field_by_time(reduced_solution, "Cell temperature [K]", clock),
            axis=0,
        )
        reduced_complete = str(reduced_solution.termination) == "final time"

        for coordinate in (
            "capacity",
            "temperature",
            "maximum_temperature",
            "voltage",
        ):
            exact_errors[coordinate] = _decimal(
                float(
                    np.max(np.abs(values[f"exact_{coordinate}"] - values[f"actual_{coordinate}"]))
                )
            )
            reduced_errors[coordinate] = _decimal(
                float(
                    np.max(np.abs(values[f"reduced_{coordinate}"] - values[f"actual_{coordinate}"]))
                )
            )
        assert reduced_errors["capacity"] is not None
        assert reduced_errors["temperature"] is not None
        assert reduced_errors["voltage"] is not None
        reduced_operational_error = max(
            reduced_errors["capacity"] / Decimal("0.001"),
            reduced_errors["temperature"] / Decimal("0.005"),
            reduced_errors["voltage"] / Decimal("0.005"),
        )
    except Exception as error:
        reasons.add(f"RESTART_{type(error).__name__.upper()}")
    if not exact_complete:
        reasons.add("EXACT_RESTART_INCOMPLETE")
    if not reduced_complete:
        reasons.add("REDUCED_RESTART_INCOMPLETE")

    def encode(name: str) -> tuple[Decimal, ...]:
        return tuple(_decimal(float(item)) for item in values.get(name, ()))

    token = (
        f"{evaluation_episode.unit.object_id.removeprefix('unit.battery-electrothermal.')}."
        f"{denominator.denominator_id.removeprefix('denominator.pybamm.')}."
        f"{word.word_id.removeprefix('word.battery-electrothermal.')}.{checkpoint_s}.{restart_suffix}"
    )
    return BatteryElectrothermalRestartResult(
        restart_id=f"restart.battery-electrothermal.{token}",
        config=ObjectIdentity.from_record(config.config_id, config),
        unit_id=unit.unit_id,
        denominator_id=denominator.denominator_id,
        history_word_id=word.word_id,
        checkpoint_s=checkpoint_s,
        continuation_end_s=continuation_end_s,
        donor_unit_id=donor_episode.unit.object_id,
        donor_word_id=donor_episode.word.object_id,
        time_s=tuple(_decimal(float(item)) for item in clock),
        actual_capacity_ah=encode("actual_capacity"),
        exact_capacity_ah=encode("exact_capacity"),
        reduced_capacity_ah=encode("reduced_capacity"),
        actual_temperature_k=encode("actual_temperature"),
        exact_temperature_k=encode("exact_temperature"),
        reduced_temperature_k=encode("reduced_temperature"),
        actual_maximum_temperature_k=encode("actual_maximum_temperature"),
        exact_maximum_temperature_k=encode("exact_maximum_temperature"),
        reduced_maximum_temperature_k=encode("reduced_maximum_temperature"),
        actual_voltage_v=encode("actual_voltage"),
        exact_voltage_v=encode("exact_voltage"),
        reduced_voltage_v=encode("reduced_voltage"),
        exact_capacity_error_ah=exact_errors["capacity"],
        exact_temperature_error_k=exact_errors["temperature"],
        exact_maximum_temperature_error_k=exact_errors["maximum_temperature"],
        exact_voltage_error_v=exact_errors["voltage"],
        reduced_capacity_error_ah=reduced_errors["capacity"],
        reduced_temperature_error_k=reduced_errors["temperature"],
        reduced_maximum_temperature_error_k=reduced_errors["maximum_temperature"],
        reduced_voltage_error_v=reduced_errors["voltage"],
        reduced_operational_error=reduced_operational_error,
        exact_restart_complete=exact_complete,
        reduced_restart_complete=reduced_complete,
        reason_codes=tuple(sorted(reasons)),
        runtime_seconds=_decimal(time.monotonic() - started),
        outcome_access=config.outcome_access,
    )


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalExactTolerance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-exact-tolerance'

    denominator_id: str
    history_word_id: str
    checkpoint_s: int
    capacity_ah: Decimal
    temperature_k: Decimal
    maximum_temperature_k: Decimal
    voltage_v: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.denominator_id, field_name="denominator_id")
        validate_stable_id(self.history_word_id, field_name="history_word_id")
        if (
            self.checkpoint_s not in {300, 450}
            or self.capacity_ah < Decimal("1e-10")
            or self.temperature_k < Decimal("1e-8")
            or self.maximum_temperature_k < Decimal("1e-8")
            or self.voltage_v < Decimal("1e-8")
            or self.capacity_ah > Decimal("1e-6")
            or self.temperature_k > Decimal("1e-5")
            or self.maximum_temperature_k > Decimal("1e-5")
            or self.voltage_v > Decimal("1e-5")
        ):
            raise ValueError("battery electrothermal exact tolerance is outside its frozen bounds")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalResourceEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-resource-envelope'

    envelope_id: str
    entered_denominator_ids: tuple[str, ...]
    worker_process_count: int
    maximum_episode_output_bytes: int
    maximum_restart_output_bytes: int
    projected_output_bytes: int
    wall_time_ceiling_seconds: int
    memory_ceiling_bytes: int
    minimum_free_external_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_strings(
            self.entered_denominator_ids,
            field_name="entered_denominator_ids",
            allow_empty=False,
        )
        if (
            self.worker_process_count not in {1, 2, 4}
            or self.maximum_episode_output_bytes <= 0
            or self.maximum_restart_output_bytes <= 0
            or self.projected_output_bytes <= 0
            or self.wall_time_ceiling_seconds <= 0
            or self.memory_ceiling_bytes < 16 * 1024**3
            or self.minimum_free_external_bytes != BATTERY_ELECTROTHERMAL_MINIMUM_FREE_BYTES
        ):
            raise ValueError("battery electrothermal resource envelope is invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-qualification'

    qualification_id: str
    source_version: str
    source_record_sha256: str
    proposed_denominator_ids: tuple[str, ...]
    entered_denominator_ids: tuple[str, ...]
    excluded_denominator_ids: tuple[str, ...]
    state_layouts: tuple[tuple[str, int, str], ...]
    receiver_array_lengths: tuple[tuple[str, int], ...]
    exact_tolerances: tuple[BatteryElectrothermalExactTolerance, ...]
    numerical_floors: tuple[tuple[str, str, Decimal], ...]
    maximum_episode_size_bytes: int
    maximum_restart_size_bytes: int
    maximum_source_build_seconds: Decimal
    maximum_episode_runtime_seconds: Decimal
    maximum_restart_runtime_seconds: Decimal
    maximum_observed_worker_rss_bytes: int
    resource_envelope: BatteryElectrothermalResourceEnvelope
    recovery_verified_without_reacquisition: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_sha256(self.source_record_sha256, field_name="source_record_sha256")
        require_sorted_unique_strings(
            self.proposed_denominator_ids,
            field_name="proposed_denominator_ids",
        )
        require_sorted_unique_strings(
            self.entered_denominator_ids,
            field_name="entered_denominator_ids",
        )
        require_sorted_unique_strings(
            self.excluded_denominator_ids,
            field_name="excluded_denominator_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            set(self.entered_denominator_ids) | set(self.excluded_denominator_ids)
            != set(self.proposed_denominator_ids)
            or set(self.entered_denominator_ids) & set(self.excluded_denominator_ids)
            or len(self.entered_denominator_ids) not in {8, 12}
            or tuple(item[0] for item in self.receiver_array_lengths)
            != self.entered_denominator_ids
            or any(item[1] <= 0 for item in self.receiver_array_lengths)
            or self.maximum_source_build_seconds < 0
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("battery electrothermal qualification disposition is invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-freeze'

    freeze_id: str
    plan_sha256: str
    design: ObjectIdentity
    planning_calculation: ObjectIdentity
    qualification: ObjectIdentity
    development_config: ObjectIdentity
    evaluation_config: ObjectIdentity
    implementation_sha256: str
    config_file_sha256: str
    formal_register: ObjectIdentity
    entered_denominator_ids: tuple[str, ...]
    exact_tolerances: tuple[BatteryElectrothermalExactTolerance, ...]
    resource_envelope: BatteryElectrothermalResourceEnvelope
    frozen_at_utc: str
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for name, value in (
            ("plan_sha256", self.plan_sha256),
            ("implementation_sha256", self.implementation_sha256),
            ("config_file_sha256", self.config_file_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_strings(
            self.entered_denominator_ids,
            field_name="entered_denominator_ids",
        )
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
        ):
            raise ValueError("battery electrothermal freeze visibility or ceiling is invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalUnitEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-unit-effect'

    effect_id: str
    unit_id: str
    stratum_id: str
    denominator_id: str
    effect: BatteryElectrothermalEffect
    coordinate: BatteryElectrothermalCoordinate
    value: Decimal | None
    complete_bundle: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.effect_id, field_name="effect_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.complete_bundle is not (self.value is not None):
            raise ValueError("battery electrothermal unit effect completion differs from its value")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalEffectEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-effect-estimate'

    estimate_id: str
    denominator_id: str
    effect: BatteryElectrothermalEffect
    coordinate: BatteryElectrothermalCoordinate
    intended_unit_count: int
    complete_unit_count: int
    mean: Decimal | None
    simultaneous_lower: Decimal | None
    simultaneous_upper: Decimal | None
    row_alpha: Decimal
    classification: BatteryElectrothermalEffectClass
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not 0 <= self.complete_unit_count <= self.intended_unit_count:
            raise ValueError("battery electrothermal effect counts are invalid")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalClosureEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-closure-estimate'

    estimate_id: str
    denominator_id: str
    history_word_id: str
    checkpoint_s: int
    intended_unit_count: int
    complete_unit_count: int
    exact_class: BatteryElectrothermalExactClass
    maximum_exact_capacity_error_ah: Decimal | None
    maximum_exact_temperature_error_k: Decimal | None
    maximum_exact_voltage_error_v: Decimal | None
    reduced_median: Decimal | None
    reduced_median_lower: Decimal | None
    reduced_median_upper: Decimal | None
    reduced_exceedance_proportion: Decimal | None
    reduced_proportion_lower: Decimal | None
    reduced_proportion_upper: Decimal | None
    reduced_class: BatteryElectrothermalReducedClass
    row_alpha: Decimal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalPairwiseSensitivity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-pairwise-sensitivity'

    sensitivity_id: str
    left_denominator_id: str
    right_denominator_id: str
    effect: BatteryElectrothermalEffect
    coordinate: BatteryElectrothermalCoordinate
    complete_unit_count: int
    mean_difference: Decimal | None
    lower_95: Decimal | None
    upper_95: Decimal | None
    interpretation: str


@dataclass(frozen=True, slots=True)
class BatteryElectrothermalAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/battery-electrothermal-adjudication'

    adjudication_id: str
    freeze: ObjectIdentity
    evaluation_config: ObjectIdentity
    unit_effects: tuple[BatteryElectrothermalUnitEffect, ...]
    effect_estimates: tuple[BatteryElectrothermalEffectEstimate, ...]
    closure_estimates: tuple[BatteryElectrothermalClosureEstimate, ...]
    pairwise_sensitivities: tuple[BatteryElectrothermalPairwiseSensitivity, ...]
    order_verdict: str
    repetition_verdict: str
    exact_closure_verdict: str
    reduced_closure_verdict: str
    complete_episode_count: int
    intended_episode_count: int
    complete_restart_count: int
    intended_restart_count: int
    revealed_at_utc: str
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or not self.effect_estimates
            or not self.closure_estimates
        ):
            raise ValueError("battery electrothermal adjudication is incomplete")


def _receiver_functional(
    episode: BatteryElectrothermalCapturedEpisode,
    *,
    coordinate: BatteryElectrothermalCoordinate,
    window_start_s: int,
    window_end_s: int,
) -> Decimal:
    times = np.asarray([float(item) for item in episode.time_s], dtype=np.float64)
    selected = (times > window_start_s) & (times <= window_end_s)
    if not np.any(selected):
        raise ValueError("battery electrothermal receiver window is empty")
    if coordinate is BatteryElectrothermalCoordinate.CAPACITY:
        values = np.asarray(
            [float(item) for item in episode.discharge_capacity_ah],
            dtype=np.float64,
        )
        result = float(np.mean(values[selected]))
    elif coordinate is BatteryElectrothermalCoordinate.TEMPERATURE:
        values = np.asarray(
            [float(item) for item in episode.mean_temperature_k],
            dtype=np.float64,
        )
        result = float(np.max(values[selected]))
    else:
        values = np.asarray(
            [float(item) for item in episode.terminal_voltage_v],
            dtype=np.float64,
        )
        result = float(np.min(values[selected]))
    return _decimal(result)


def build_unit_effects(
    config: BatteryElectrothermalStageConfig,
    episodes: Sequence[BatteryElectrothermalCapturedEpisode],
    *,
    unit_ids: tuple[str, ...] | None = None,
    denominator_ids: tuple[str, ...] | None = None,
) -> tuple[BatteryElectrothermalUnitEffect, ...]:
    if config.stage is not BatteryElectrothermalStage.EVALUATION:
        raise ValueError("battery electrothermal primary effects require evaluation")
    by_key = {
        (
            item.unit.object_id,
            item.denominator.object_id,
            item.word.object_id,
        ): item
        for item in episodes
    }
    selected_unit_ids = set(unit_ids or config.primary_unit_ids)
    selected_denominator_ids = set(
        denominator_ids or tuple(item.denominator_id for item in config.denominators)
    )
    units = {item.unit_id: item for item in config.units if item.unit_id in selected_unit_ids}
    definitions: Mapping[
        BatteryElectrothermalEffect,
        tuple[int, int, Mapping[str, int]],
    ] = {
        BatteryElectrothermalEffect.RAW_ORDER: (
            300,
            600,
            {
                "word.battery-electrothermal.i-then-t": 1,
                "word.battery-electrothermal.t-then-i": -1,
            },
        ),
        BatteryElectrothermalEffect.ADJUSTED_ORDER: (
            300,
            600,
            {
                "word.battery-electrothermal.i-then-t": 1,
                "word.battery-electrothermal.i-early-150": -1,
                "word.battery-electrothermal.t-late-150": -1,
                "word.battery-electrothermal.t-then-i": -1,
                "word.battery-electrothermal.t-early-150": 1,
                "word.battery-electrothermal.i-late-150": 1,
            },
        ),
        BatteryElectrothermalEffect.RAW_REPETITION: (
            300,
            600,
            {
                "word.battery-electrothermal.i-split": 1,
                "word.battery-electrothermal.i-contiguous-early-300": -1,
            },
        ),
        BatteryElectrothermalEffect.MATCHED_REPETITION: (
            450,
            750,
            {
                "word.battery-electrothermal.i-split": 1,
                "word.battery-electrothermal.i-contiguous-matched-end": -1,
            },
        ),
    }
    values: list[BatteryElectrothermalUnitEffect] = []
    for unit_id, unit in sorted(units.items()):
        for denominator in config.denominators:
            if denominator.denominator_id not in selected_denominator_ids:
                continue
            hold = by_key.get((unit_id, denominator.denominator_id, "word.battery-electrothermal.hold"))
            for effect, (window_start, window_end, coefficients) in definitions.items():
                for coordinate in BatteryElectrothermalCoordinate:
                    reasons: set[str] = set()
                    value: Decimal | None = None
                    required = [
                        by_key.get((unit_id, denominator.denominator_id, word_id))
                        for word_id in coefficients
                    ]
                    if (
                        hold is not None
                        and hold.disposition is BatteryElectrothermalEpisodeDisposition.COMPLETE
                        and all(
                            item is not None and item.disposition is BatteryElectrothermalEpisodeDisposition.COMPLETE
                            for item in required
                        )
                    ):
                        hold_value = _receiver_functional(
                            hold,
                            coordinate=coordinate,
                            window_start_s=window_start,
                            window_end_s=window_end,
                        )
                        total = Decimal(0)
                        for word_id, coefficient in coefficients.items():
                            episode = by_key[(unit_id, denominator.denominator_id, word_id)]
                            functional = _receiver_functional(
                                episode,
                                coordinate=coordinate,
                                window_start_s=window_start,
                                window_end_s=window_end,
                            )
                            total += Decimal(coefficient) * (functional - hold_value)
                        value = total
                    else:
                        reasons.add("MATCHED_WORD_OR_HOLD_BUNDLE_INCOMPLETE")
                    token = (
                        f"{unit_id.removeprefix('unit.battery-electrothermal.')}."
                        f"{denominator.denominator_id.removeprefix('denominator.pybamm.')}."
                        f"{effect.value.lower()}.{coordinate.value.lower()}"
                    )
                    values.append(
                        BatteryElectrothermalUnitEffect(
                            effect_id=f"effect.battery-electrothermal.{token}",
                            unit_id=unit_id,
                            stratum_id=unit.stratum_id,
                            denominator_id=denominator.denominator_id,
                            effect=effect,
                            coordinate=coordinate,
                            value=value,
                            complete_bundle=value is not None,
                            reason_codes=tuple(sorted(reasons)),
                        )
                    )
    return tuple(sorted(values, key=lambda item: item.effect_id))


_MATERIALITY = {
    BatteryElectrothermalCoordinate.CAPACITY: Decimal("0.002"),
    BatteryElectrothermalCoordinate.TEMPERATURE: Decimal("0.005"),
    BatteryElectrothermalCoordinate.VOLTAGE: Decimal("0.005"),
}


def _bootstrap_means(
    rows: Sequence[BatteryElectrothermalUnitEffect],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    grouped: dict[str, np.ndarray] = {}
    for stratum in sorted({item.stratum_id for item in rows}, key=lambda stratum: bytes(_FROZEN_STRATUM_HASH_OPERANDS[tuple(row[0] for row in _STRATA).index(stratum)]) if stratum in {row[0] for row in _STRATA} else stratum.encode()):
        grouped[stratum] = np.asarray(
            [
                float(item.value)
                for item in rows
                if item.stratum_id == stratum and item.value is not None
            ],
            dtype=np.float64,
        )
    rng = np.random.default_rng(seed)
    pieces = []
    counts = []
    for stratum, values in grouped.items():
        if values.size == 0:
            raise ValueError(f"battery electrothermal bootstrap stratum {stratum} is empty")
        indices = rng.integers(0, values.size, size=(replicates, values.size))
        pieces.append(values[indices].sum(axis=1))
        counts.append(values.size)
    return np.asarray(
        np.sum(np.stack(pieces), axis=0) / sum(counts),
        dtype=np.float64,
    )


def _seed_for(*tokens: str) -> int:
    # Fixed closure words retain their numerical streams under descriptive IDs.
    if len(tokens) == 3 and tokens[2] == "closure":
        commitment = CLOSURE_SEED_COMMITMENTS.get((tokens[0], tokens[1]))
        if commitment is not None:
            return commitment
    digest = sha256(":".join(tokens).encode()).digest()
    return BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED + int.from_bytes(digest[:4], "big")


def _classify_effect(
    lower: Decimal,
    upper: Decimal,
    coordinate: BatteryElectrothermalCoordinate,
) -> BatteryElectrothermalEffectClass:
    boundary = _MATERIALITY[coordinate]
    if lower > boundary:
        return BatteryElectrothermalEffectClass.MATERIAL_POSITIVE
    if upper < -boundary:
        return BatteryElectrothermalEffectClass.MATERIAL_NEGATIVE
    if lower >= -boundary and upper <= boundary:
        return BatteryElectrothermalEffectClass.NULL_EQUIVALENT_FINITE
    if lower > 0 or upper < 0:
        return BatteryElectrothermalEffectClass.NONZERO_SUBMATERIAL
    return BatteryElectrothermalEffectClass.UNRESOLVED


def estimate_effects(
    config: BatteryElectrothermalStageConfig,
    unit_effects: Sequence[BatteryElectrothermalUnitEffect],
) -> tuple[BatteryElectrothermalEffectEstimate, ...]:
    intended = len(config.primary_unit_ids)
    threshold = max(20, math.ceil(Decimal("0.80") * intended))
    row_alpha = BATTERY_ELECTROTHERMAL_FAMILY_ALPHA / Decimal(len(config.denominators) * 3)
    estimates: list[BatteryElectrothermalEffectEstimate] = []
    for denominator in config.denominators:
        for effect in BatteryElectrothermalEffect:
            for coordinate in BatteryElectrothermalCoordinate:
                rows = [
                    item
                    for item in unit_effects
                    if (
                        item.denominator_id == denominator.denominator_id
                        and item.effect is effect
                        and item.coordinate is coordinate
                        and item.value is not None
                    )
                ]
                reasons: set[str] = set()
                mean = lower = upper = None
                if len(rows) < threshold:
                    classification = BatteryElectrothermalEffectClass.PARTIAL if rows else BatteryElectrothermalEffectClass.UNEVALUABLE
                    reasons.add("INSUFFICIENT_COMPLETE_MATCHED_PREPARATIONS")
                else:
                    bootstrap = _bootstrap_means(
                        rows,
                        replicates=BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES,
                        seed=_seed_for(
                            denominator.denominator_id,
                            effect.value,
                            coordinate.value,
                        ),
                    )
                    alpha = float(row_alpha)
                    complete_values = [float(item.value) for item in rows if item.value is not None]
                    mean = _decimal(statistics.mean(complete_values))
                    lower = _decimal(float(np.quantile(bootstrap, alpha / 2, method="linear")))
                    upper = _decimal(
                        float(
                            np.quantile(
                                bootstrap,
                                1 - alpha / 2,
                                method="linear",
                            )
                        )
                    )
                    classification = _classify_effect(lower, upper, coordinate)
                estimates.append(
                    BatteryElectrothermalEffectEstimate(
                        estimate_id=(
                            f"estimate.battery-electrothermal."
                            f"{denominator.denominator_id.removeprefix('denominator.pybamm.')}."
                            f"{effect.value.lower()}.{coordinate.value.lower()}"
                        ),
                        denominator_id=denominator.denominator_id,
                        effect=effect,
                        coordinate=coordinate,
                        intended_unit_count=intended,
                        complete_unit_count=len(rows),
                        mean=mean,
                        simultaneous_lower=lower,
                        simultaneous_upper=upper,
                        row_alpha=row_alpha,
                        classification=classification,
                        reason_codes=tuple(sorted(reasons)),
                    )
                )
    return tuple(sorted(estimates, key=lambda item: item.estimate_id))


def estimate_closure(
    config: BatteryElectrothermalStageConfig,
    restarts: Sequence[BatteryElectrothermalRestartResult | BatteryElectrothermalRestartSummary],
    tolerances: Sequence[BatteryElectrothermalExactTolerance],
) -> tuple[BatteryElectrothermalClosureEstimate, ...]:
    intended = len(config.primary_unit_ids)
    threshold = max(20, math.ceil(Decimal("0.80") * intended))
    row_alpha = BATTERY_ELECTROTHERMAL_FAMILY_ALPHA / Decimal(
        len(config.denominators) * len(BATTERY_ELECTROTHERMAL_RESTART_HISTORIES) * 2
    )
    unit_strata = {item.unit_id: item.stratum_id for item in config.units}
    values: list[BatteryElectrothermalClosureEstimate] = []
    for denominator in config.denominators:
        for word_id, checkpoint_s, _ in BATTERY_ELECTROTHERMAL_RESTART_HISTORIES:
            rows = [
                item
                for item in restarts
                if (
                    item.denominator_id == denominator.denominator_id
                    and item.history_word_id == word_id
                    and item.checkpoint_s == checkpoint_s
                    and item.unit_id in config.primary_unit_ids
                )
            ]
            complete = [
                item
                for item in rows
                if (
                    item.exact_restart_complete
                    and item.reduced_restart_complete
                    and item.reduced_operational_error is not None
                )
            ]
            tolerance = next(
                item
                for item in tolerances
                if (
                    item.denominator_id == denominator.denominator_id
                    and item.history_word_id == word_id
                    and item.checkpoint_s == checkpoint_s
                )
            )
            reasons: set[str] = set()
            exact_capacity = (
                max(
                    item.exact_capacity_error_ah
                    for item in complete
                    if item.exact_capacity_error_ah is not None
                )
                if complete
                else None
            )
            exact_temperature = (
                max(
                    item.exact_temperature_error_k
                    for item in complete
                    if item.exact_temperature_error_k is not None
                )
                if complete
                else None
            )
            exact_voltage = (
                max(
                    item.exact_voltage_error_v
                    for item in complete
                    if item.exact_voltage_error_v is not None
                )
                if complete
                else None
            )
            if len(rows) != intended or any(not item.exact_restart_complete for item in rows):
                exact_class = BatteryElectrothermalExactClass.UNEVALUABLE
                reasons.add("NOT_ALL_EXACT_BRANCHES_COMPLETE")
            elif (
                exact_capacity is not None
                and exact_temperature is not None
                and exact_voltage is not None
                and exact_capacity <= tolerance.capacity_ah
                and exact_temperature <= tolerance.temperature_k
                and exact_voltage <= tolerance.voltage_v
                and all(
                    item.exact_maximum_temperature_error_k is not None
                    and item.exact_maximum_temperature_error_k <= tolerance.maximum_temperature_k
                    for item in rows
                )
            ):
                exact_class = BatteryElectrothermalExactClass.CLOSED
            else:
                exact_class = BatteryElectrothermalExactClass.OPPOSED
                reasons.add("EXACT_RESTART_EXCEEDS_FROZEN_TOLERANCE")

            median = median_lower = median_upper = None
            proportion = proportion_lower = proportion_upper = None
            if len(complete) < threshold:
                reduced_class = BatteryElectrothermalReducedClass.UNEVALUABLE
                reasons.add("INSUFFICIENT_COMPLETE_THREE_BRANCH_RECORDS")
            else:
                grouped: dict[str, np.ndarray] = {}
                for stratum in sorted({unit_strata[item.unit_id] for item in complete}, key=lambda stratum: bytes(_FROZEN_STRATUM_HASH_OPERANDS[tuple(row[0] for row in _STRATA).index(stratum)]) if stratum in {row[0] for row in _STRATA} else stratum.encode()):
                    grouped[stratum] = np.asarray(
                        [
                            float(item.reduced_operational_error)
                            for item in complete
                            if unit_strata[item.unit_id] == stratum
                            and item.reduced_operational_error is not None
                        ],
                        dtype=np.float64,
                    )
                rng = np.random.default_rng(
                    _seed_for(
                        denominator.denominator_id,
                        word_id,
                        "closure",
                    )
                )
                samples = []
                for stratum, group in grouped.items():
                    if group.size == 0:
                        raise ValueError(f"battery electrothermal closure stratum {stratum} is empty")
                    indices = rng.integers(
                        0,
                        group.size,
                        size=(BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES, group.size),
                    )
                    samples.append(group[indices])
                bootstrap_matrix = np.concatenate(samples, axis=1)
                medians = np.median(bootstrap_matrix, axis=1)
                proportions = np.mean(bootstrap_matrix > 1, axis=1)
                observed = np.asarray(
                    [
                        float(item.reduced_operational_error)
                        for item in complete
                        if item.reduced_operational_error is not None
                    ],
                    dtype=np.float64,
                )
                alpha = float(row_alpha)
                median = _decimal(float(np.median(observed)))
                median_lower = _decimal(float(np.quantile(medians, alpha / 2, method="linear")))
                median_upper = _decimal(float(np.quantile(medians, 1 - alpha / 2, method="linear")))
                proportion = _decimal(float(np.mean(observed > 1)))
                proportion_lower = _decimal(
                    float(np.quantile(proportions, alpha / 2, method="linear"))
                )
                proportion_upper = _decimal(
                    float(
                        np.quantile(
                            proportions,
                            1 - alpha / 2,
                            method="linear",
                        )
                    )
                )
                if median_lower > 1 and proportion_lower >= Decimal("0.80"):
                    reduced_class = BatteryElectrothermalReducedClass.LOSSY
                elif median_upper < 1 and proportion_upper <= Decimal("0.10"):
                    reduced_class = BatteryElectrothermalReducedClass.ADEQUATE
                else:
                    reduced_class = BatteryElectrothermalReducedClass.MIXED
            values.append(
                BatteryElectrothermalClosureEstimate(
                    estimate_id=(
                        f"closure.battery-electrothermal."
                        f"{denominator.denominator_id.removeprefix('denominator.pybamm.')}."
                        f"{word_id.removeprefix('word.battery-electrothermal.')}.{checkpoint_s}"
                    ),
                    denominator_id=denominator.denominator_id,
                    history_word_id=word_id,
                    checkpoint_s=checkpoint_s,
                    intended_unit_count=intended,
                    complete_unit_count=len(complete),
                    exact_class=exact_class,
                    maximum_exact_capacity_error_ah=exact_capacity,
                    maximum_exact_temperature_error_k=exact_temperature,
                    maximum_exact_voltage_error_v=exact_voltage,
                    reduced_median=median,
                    reduced_median_lower=median_lower,
                    reduced_median_upper=median_upper,
                    reduced_exceedance_proportion=proportion,
                    reduced_proportion_lower=proportion_lower,
                    reduced_proportion_upper=proportion_upper,
                    reduced_class=reduced_class,
                    row_alpha=row_alpha,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
    return tuple(sorted(values, key=lambda item: item.estimate_id))


def pairwise_sensitivities(
    config: BatteryElectrothermalStageConfig,
    unit_effects: Sequence[BatteryElectrothermalUnitEffect],
) -> tuple[BatteryElectrothermalPairwiseSensitivity, ...]:
    by_denominator = {item.denominator_id: item for item in config.denominators}
    pairs: set[tuple[str, str]] = set()
    for left in config.denominators:
        for right in config.denominators:
            if left.denominator_id >= right.denominator_id:
                continue
            differences = sum(
                (
                    left.model is not right.model,
                    left.thermal is not right.thermal,
                    left.view.view_id != right.view.view_id,
                )
            )
            if differences == 1:
                pairs.add((left.denominator_id, right.denominator_id))
    by_key = {
        (
            item.unit_id,
            item.denominator_id,
            item.effect,
            item.coordinate,
        ): item
        for item in unit_effects
        if item.value is not None
    }
    values: list[BatteryElectrothermalPairwiseSensitivity] = []
    for left_id, right_id in sorted(pairs):
        left = by_denominator[left_id]
        right = by_denominator[right_id]
        axis = (
            "model"
            if left.model is not right.model
            else ("thermal" if left.thermal is not right.thermal else "numerical-view")
        )
        for effect in BatteryElectrothermalEffect:
            for coordinate in BatteryElectrothermalCoordinate:
                paired = []
                strata = []
                for unit in config.units:
                    if unit.unit_id not in config.primary_unit_ids:
                        continue
                    left_row = by_key.get((unit.unit_id, left_id, effect, coordinate))
                    right_row = by_key.get((unit.unit_id, right_id, effect, coordinate))
                    if left_row is None or right_row is None:
                        continue
                    assert left_row.value is not None and right_row.value is not None
                    paired.append(float(right_row.value - left_row.value))
                    strata.append(unit.stratum_id)
                mean = lower = upper = None
                if paired:
                    synthetic = tuple(
                        BatteryElectrothermalUnitEffect(
                            effect_id=f"effect.synthetic.{index:04d}",
                            unit_id=f"unit.synthetic.{index:04d}",
                            stratum_id=stratum,
                            denominator_id=right_id,
                            effect=effect,
                            coordinate=coordinate,
                            value=_decimal(value),
                            complete_bundle=True,
                            reason_codes=(),
                        )
                        for index, (value, stratum) in enumerate(zip(paired, strata, strict=True))
                    )
                    bootstrap = _bootstrap_means(
                        synthetic,
                        replicates=BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES,
                        seed=_seed_for(
                            left_id,
                            right_id,
                            effect.value,
                            coordinate.value,
                        ),
                    )
                    mean = _decimal(statistics.mean(paired))
                    lower = _decimal(float(np.quantile(bootstrap, 0.025, method="linear")))
                    upper = _decimal(float(np.quantile(bootstrap, 0.975, method="linear")))
                token = sha256(f"{left_id}:{right_id}".encode()).hexdigest()[:12]
                values.append(
                    BatteryElectrothermalPairwiseSensitivity(
                        sensitivity_id=(
                            f"sensitivity.battery-electrothermal.{axis}.{token}."
                            f"{effect.value.lower()}.{coordinate.value.lower()}"
                        ),
                        left_denominator_id=left_id,
                        right_denominator_id=right_id,
                        effect=effect,
                        coordinate=coordinate,
                        complete_unit_count=len(paired),
                        mean_difference=mean,
                        lower_95=lower,
                        upper_95=upper,
                        interpretation=(
                            f"right-minus-left paired {axis} sensitivity; "
                            "secondary and not multiplicity-promoting"
                        ),
                    )
                )
    return tuple(sorted(values, key=lambda item: item.sensitivity_id))


def _denominator_axes(
    denominator: BatteryElectrothermalDenominator,
) -> Mapping[str, str]:
    return {
        "thermal": denominator.thermal.value,
        "model": denominator.model.value,
        "view": denominator.view.view_id,
    }


def _axis_conditional(
    config: BatteryElectrothermalStageConfig,
    estimates: Sequence[BatteryElectrothermalEffectEstimate],
    *,
    axis: str,
) -> bool:
    denominator_by_id = {item.denominator_id: item for item in config.denominators}
    for coordinate in BatteryElectrothermalCoordinate:
        rows = [item for item in estimates if item.coordinate is coordinate]
        classes_by_level: dict[str, set[BatteryElectrothermalEffectClass]] = {}
        for item in rows:
            level = _denominator_axes(denominator_by_id[item.denominator_id])[axis]
            classes_by_level.setdefault(level, set()).add(item.classification)
        if (
            classes_by_level
            and all(len(classes) == 1 for classes in classes_by_level.values())
            and len({next(iter(classes)) for classes in classes_by_level.values()}) > 1
        ):
            return True
    return False


def _effect_panel_verdict(
    config: BatteryElectrothermalStageConfig,
    estimates: Sequence[BatteryElectrothermalEffectEstimate],
    *,
    raw_effect: BatteryElectrothermalEffect,
    controlled_effect: BatteryElectrothermalEffect,
    panel: str,
) -> str:
    controlled = [item for item in estimates if item.effect is controlled_effect]
    raw = [item for item in estimates if item.effect is raw_effect]
    classes = {item.classification for item in controlled}
    if BatteryElectrothermalEffectClass.UNEVALUABLE in classes:
        return (
            BatteryElectrothermalOrderVerdict.UNEVALUABLE.value
            if panel == "order"
            else BatteryElectrothermalRepetitionVerdict.UNEVALUABLE.value
        )
    if BatteryElectrothermalEffectClass.PARTIAL in classes:
        return (
            BatteryElectrothermalOrderVerdict.PARTIAL.value
            if panel == "order"
            else BatteryElectrothermalRepetitionVerdict.PARTIAL.value
        )
    if all(
        item.classification is BatteryElectrothermalEffectClass.NULL_EQUIVALENT_FINITE for item in controlled
    ) and any(
        item.classification
        in {
            BatteryElectrothermalEffectClass.MATERIAL_POSITIVE,
            BatteryElectrothermalEffectClass.MATERIAL_NEGATIVE,
            BatteryElectrothermalEffectClass.NONZERO_SUBMATERIAL,
        }
        for item in raw
    ):
        return (
            BatteryElectrothermalOrderVerdict.TIMING_ONLY.value
            if panel == "order"
            else BatteryElectrothermalRepetitionVerdict.END_TIME_ONLY.value
        )
    if BatteryElectrothermalEffectClass.UNRESOLVED in classes:
        return (
            BatteryElectrothermalOrderVerdict.UNRESOLVED.value
            if panel == "order"
            else BatteryElectrothermalRepetitionVerdict.UNRESOLVED.value
        )
    if all(
        len({item.classification for item in controlled if item.coordinate is coordinate}) == 1
        for coordinate in BatteryElectrothermalCoordinate
    ):
        return BatteryElectrothermalOrderVerdict.FULL.value if panel == "order" else BatteryElectrothermalRepetitionVerdict.FULL.value
    for axis in ("thermal", "model", "view"):
        if _axis_conditional(config, controlled, axis=axis):
            if panel == "order":
                return {
                    "thermal": BatteryElectrothermalOrderVerdict.THERMAL.value,
                    "model": BatteryElectrothermalOrderVerdict.MODEL.value,
                    "view": BatteryElectrothermalOrderVerdict.VIEW.value,
                }[axis]
            return {
                "thermal": BatteryElectrothermalRepetitionVerdict.THERMAL.value,
                "model": BatteryElectrothermalRepetitionVerdict.MODEL.value,
                "view": BatteryElectrothermalRepetitionVerdict.VIEW.value,
            }[axis]
    return (
        BatteryElectrothermalOrderVerdict.UNRESOLVED.value
        if panel == "order"
        else BatteryElectrothermalRepetitionVerdict.UNRESOLVED.value
    )


def _closure_axis_conditional(
    config: BatteryElectrothermalStageConfig,
    closure: Sequence[BatteryElectrothermalClosureEstimate],
    *,
    axis: str,
) -> bool:
    denominator_by_id = {item.denominator_id: item for item in config.denominators}
    classes_by_level: dict[str, set[BatteryElectrothermalReducedClass]] = {}
    for item in closure:
        level = _denominator_axes(denominator_by_id[item.denominator_id])[axis]
        classes_by_level.setdefault(level, set()).add(item.reduced_class)
    return (
        bool(classes_by_level)
        and all(len(classes) == 1 for classes in classes_by_level.values())
        and len({next(iter(classes)) for classes in classes_by_level.values()}) > 1
    )


def terminal_closure_verdicts(
    closure: Sequence[BatteryElectrothermalClosureEstimate],
    *,
    config: BatteryElectrothermalStageConfig | None = None,
) -> tuple[str, str]:
    exact = {item.exact_class for item in closure}
    exact_evaluable = exact - {BatteryElectrothermalExactClass.UNEVALUABLE}
    if exact == {BatteryElectrothermalExactClass.CLOSED}:
        exact_verdict = "battery electrothermal_EXACT_STATE_CLOSED_ALL_ENTERED"
    elif exact == {BatteryElectrothermalExactClass.UNEVALUABLE}:
        exact_verdict = "battery electrothermal_EXACT_STATE_UNEVALUABLE"
    elif BatteryElectrothermalExactClass.UNEVALUABLE in exact and exact_evaluable:
        exact_verdict = "battery electrothermal_EXACT_STATE_PARTIAL"
    elif BatteryElectrothermalExactClass.OPPOSED in exact and BatteryElectrothermalExactClass.CLOSED in exact:
        exact_verdict = "battery electrothermal_EXACT_STATE_DENOMINATOR_CONDITIONAL"
    else:
        exact_verdict = "battery electrothermal_EXACT_STATE_OPPOSED"
    reduced = {item.reduced_class for item in closure}
    reduced_evaluable = reduced - {BatteryElectrothermalReducedClass.UNEVALUABLE}
    if reduced == {BatteryElectrothermalReducedClass.LOSSY}:
        reduced_verdict = "battery electrothermal_REDUCED_STATE_LOSS_RECURRENT"
    elif reduced == {BatteryElectrothermalReducedClass.ADEQUATE}:
        reduced_verdict = "battery electrothermal_REDUCED_STATE_ADEQUATE_ALL_ENTERED"
    elif reduced == {BatteryElectrothermalReducedClass.UNEVALUABLE}:
        reduced_verdict = "battery electrothermal_REDUCED_STATE_UNEVALUABLE"
    elif BatteryElectrothermalReducedClass.UNEVALUABLE in reduced and reduced_evaluable:
        reduced_verdict = "battery electrothermal_REDUCED_STATE_PARTIAL"
    elif config is not None and any(
        _closure_axis_conditional(config, closure, axis=axis)
        for axis in ("thermal", "model", "view")
    ):
        axis = next(
            axis
            for axis in ("thermal", "model", "view")
            if _closure_axis_conditional(config, closure, axis=axis)
        )
        reduced_verdict = {
            "thermal": "battery electrothermal_REDUCED_STATE_THERMAL_CONDITIONAL",
            "model": "battery electrothermal_REDUCED_STATE_MODEL_CONDITIONAL",
            "view": "battery electrothermal_REDUCED_STATE_VIEW_CONDITIONAL",
        }[axis]
    else:
        reduced_verdict = "battery electrothermal_REDUCED_STATE_MIXED"
    return exact_verdict, reduced_verdict


def adjudicate(
    *,
    freeze: BatteryElectrothermalFreeze,
    config: BatteryElectrothermalStageConfig,
    episodes: Sequence[BatteryElectrothermalCapturedEpisode],
    restarts: Sequence[BatteryElectrothermalRestartResult],
    revealed_at_utc: str,
) -> BatteryElectrothermalAdjudication:
    unit_effects = build_unit_effects(config, episodes)
    return adjudicate_summaries(
        freeze=freeze,
        config=config,
        unit_effects=tuple(unit_effects),
        restarts=restarts,
        complete_episode_count=sum(
            item.disposition is BatteryElectrothermalEpisodeDisposition.COMPLETE
            for item in episodes
            if item.unit.object_id in config.primary_unit_ids
        ),
        complete_restart_count=sum(
            item.exact_restart_complete and item.reduced_restart_complete
            for item in restarts
            if item.unit_id in config.primary_unit_ids
        ),
        revealed_at_utc=revealed_at_utc,
    )


def adjudicate_summaries(
    *,
    freeze: BatteryElectrothermalFreeze,
    config: BatteryElectrothermalStageConfig,
    unit_effects: Sequence[BatteryElectrothermalUnitEffect],
    restarts: Sequence[BatteryElectrothermalRestartResult | BatteryElectrothermalRestartSummary],
    complete_episode_count: int,
    complete_restart_count: int,
    revealed_at_utc: str,
) -> BatteryElectrothermalAdjudication:
    effect_estimates = estimate_effects(config, unit_effects)
    closure = estimate_closure(config, restarts, freeze.exact_tolerances)
    pairwise = pairwise_sensitivities(config, unit_effects)
    exact_verdict, reduced_verdict = terminal_closure_verdicts(
        closure,
        config=config,
    )
    return BatteryElectrothermalAdjudication(
        adjudication_id="adjudication.battery-electrothermal.primary",
        freeze=ObjectIdentity.from_record(freeze.freeze_id, freeze),
        evaluation_config=ObjectIdentity.from_record(config.config_id, config),
        unit_effects=tuple(unit_effects),
        effect_estimates=effect_estimates,
        closure_estimates=closure,
        pairwise_sensitivities=pairwise,
        order_verdict=_effect_panel_verdict(
            config,
            effect_estimates,
            raw_effect=BatteryElectrothermalEffect.RAW_ORDER,
            controlled_effect=BatteryElectrothermalEffect.ADJUSTED_ORDER,
            panel="order",
        ),
        repetition_verdict=_effect_panel_verdict(
            config,
            effect_estimates,
            raw_effect=BatteryElectrothermalEffect.RAW_REPETITION,
            controlled_effect=BatteryElectrothermalEffect.MATCHED_REPETITION,
            panel="repetition",
        ),
        exact_closure_verdict=exact_verdict,
        reduced_closure_verdict=reduced_verdict,
        complete_episode_count=complete_episode_count,
        intended_episode_count=(
            len(config.primary_unit_ids) * len(config.denominators) * len(config.words)
        ),
        complete_restart_count=complete_restart_count,
        intended_restart_count=(
            len(config.primary_unit_ids) * len(config.denominators) * len(BATTERY_ELECTROTHERMAL_RESTART_HISTORIES)
        ),
        revealed_at_utc=revealed_at_utc,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


__all__ = [
    'BatteryElectrothermalActionWord',
    'BatteryElectrothermalAdjudication',
    'BatteryElectrothermalCapturedEpisode',
    'BatteryElectrothermalClosureEstimate',
    'BatteryElectrothermalDenominator',
    'BatteryElectrothermalDesign',
    'BatteryElectrothermalDonorCatalogue',
    'BatteryElectrothermalDonorEntry',
    'BatteryElectrothermalEpisodeDisposition',
    'BatteryElectrothermalExactTolerance',
    'BatteryElectrothermalFormalAssignment',
    'BatteryElectrothermalFreeze',
    'BatteryElectrothermalNumericalView',
    'BatteryElectrothermalPlanningCalculation',
    'BatteryElectrothermalPlanningRow',
    'BatteryElectrothermalPreparation',
    'BatteryElectrothermalQualification',
    'BatteryElectrothermalResourceEnvelope',
    'BatteryElectrothermalRestartResult',
    'BatteryElectrothermalRestartSummary',
    'BatteryElectrothermalStage',
    'BatteryElectrothermalStageConfig',
    'BatteryElectrothermalUnitEffect',
    'BatteryElectrothermalUnitRole',
    "BATTERY_ELECTROTHERMAL_BOOTSTRAP_REPLICATES",
    "BATTERY_ELECTROTHERMAL_BOOTSTRAP_SEED",
    "BATTERY_ELECTROTHERMAL_DESIGN_ID",
    "BATTERY_ELECTROTHERMAL_EXTERNAL_ROOT",
    "BATTERY_ELECTROTHERMAL_MINIMUM_FREE_BYTES",
    "BATTERY_ELECTROTHERMAL_PRIMARY_N",
    "BATTERY_ELECTROTHERMAL_QUALIFICATION_EXTREME_WORD_IDS",
    "BATTERY_ELECTROTHERMAL_QUALIFICATION_WORD_IDS",
    "BATTERY_ELECTROTHERMAL_RESTART_HISTORIES",
    "BATTERY_ELECTROTHERMAL_SOURCE_VERSION",
    "acquire_episode",
    "episode_reduced_coordinates",
    "acquire_restart_result",
    "action_words",
    "adjudicate",
    "adjudicate_summaries",
    "build_design",
    "build_donor_catalogue",
    "build_stage_config",
    "build_unit_effects",
    "denominators",
    "formal_assignments",
    "numerical_views",
    "select_donor",
    "summarize_restart",
    "validate_repository_config",
]
