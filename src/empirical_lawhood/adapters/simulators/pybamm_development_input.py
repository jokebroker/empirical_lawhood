# SPDX-License-Identifier: MPL-2.0
"""Disclosed battery response, exact restart and reduced-observation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators import pybamm_electrothermal_hierarchy as battery_electrothermal
from empirical_lawhood.adapters.simulators import pybamm_exact_restart as battery_restart
from empirical_lawhood.adapters.simulators import pybamm_reduced_coordinate_sufficiency as battery_reduced_observation
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class PyBaMMDevelopmentInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/py-ba-mm-development-input'

    config_id: str
    denominator_id: str
    electrothermal_unit_id: str
    restart_unit_id: str
    reduced_observation_unit_ids: tuple[str, str, str]
    initial_soc: Decimal
    initial_temperature_k: Decimal
    reduced_observation_initial_socs: tuple[Decimal, Decimal, Decimal]
    reduced_observation_initial_temperatures_k: tuple[Decimal, Decimal, Decimal]
    restart_history_id: str
    reduced_observation_history_id: str
    reduced_observation_chart_id: str

    def __post_init__(self) -> None:
        for value in (
            self.config_id,
            self.denominator_id,
            self.electrothermal_unit_id,
            self.restart_unit_id,
            self.restart_history_id,
            self.reduced_observation_history_id,
            self.reduced_observation_chart_id,
            *self.reduced_observation_unit_ids,
        ):
            validate_stable_id(value)
        if not self.config_id.startswith("empirical-lawhood-"):
            raise ValueError("PyBaMM config needs a target-owned identity")
        unit_ids = (self.electrothermal_unit_id, self.restart_unit_id, *self.reduced_observation_unit_ids)
        require_sorted_unique_strings(tuple(sorted(unit_ids)), field_name="unit_ids")
        if not all(value.startswith("unit.empirical-lawhood-") for value in unit_ids):
            raise ValueError("PyBaMM units need target-owned identities")
        if self.denominator_id != "denominator.pybamm.spme.isothermal.casadi-coarse":
            raise ValueError("bounded PyBaMM starter selects SPMe/isothermal/CasADi")
        if self.restart_history_id != "history.battery-exact-restart.i-plus-600.300.600":
            raise ValueError("Exact-restart starter requires the 300 s current checkpoint")
        if self.reduced_observation_history_id != "history.battery-reduced-observation.i-plus-600.300.600":
            raise ValueError("Reduced-observation starter requires the 300 s checkpoint-relative hold")
        if self.reduced_observation_chart_id != "chart.battery-reduced-observation.gauge":
            raise ValueError(
                "Reduced-observation starter chart is a disclosed development illustration"
            )
        for name, value, lower, upper in (
            ("initial_soc", self.initial_soc, Decimal("0.25"), Decimal("0.80")),
            (
                "initial_temperature_k",
                self.initial_temperature_k,
                Decimal("291.15"),
                Decimal("297.15"),
            ),
            *(
                ("reduced_observation_initial_soc", value, Decimal("0.25"), Decimal("0.80"))
                for value in self.reduced_observation_initial_socs
            ),
            *(
                (
                    "reduced_observation_initial_temperature_k",
                    value,
                    Decimal("291.15"),
                    Decimal("297.15"),
                )
                for value in self.reduced_observation_initial_temperatures_k
            ),
        ):
            validate_decimal(value, field_name=name, minimum=lower)
            if value > upper:
                raise ValueError(f"{name} leaves the bounded preparation range")
        if (
            len(
                set(
                    zip(
                        self.reduced_observation_initial_socs,
                        self.reduced_observation_initial_temperatures_k,
                        strict=True,
                    )
                )
            )
            != 3
        ):
            raise ValueError("Reduced-observation donor and target preparations must be distinct")
        if any(
            not (
                Decimal("0.25") <= soc <= Decimal("0.40")
                and Decimal("291.15") <= temperature <= Decimal("293.15")
            )
            for soc, temperature in zip(
                self.reduced_observation_initial_socs[:2],
                self.reduced_observation_initial_temperatures_k[:2],
                strict=True,
            )
        ) or not (
            Decimal("0.65") <= self.reduced_observation_initial_socs[2] <= Decimal("0.80")
            and Decimal("295.15")
            <= self.reduced_observation_initial_temperatures_k[2]
            <= Decimal("297.15")
        ):
            raise ValueError("Reduced-observation disclosed donor/target strata differ")


def run_native_development_check(
    config: PyBaMMDevelopmentInput,
) -> dict[str, object]:
    """Execute three distinct source-native contracts without issuing a campaign."""

    import pybamm

    if pybamm.__version__ != battery_electrothermal.BATTERY_ELECTROTHERMAL_SOURCE_VERSION:
        raise ValueError("installed PyBaMM version differs from the source design")
    denominator = next(
        item
        for item in battery_electrothermal.denominators(include_x_full=False)
        if item.denominator_id == config.denominator_id
    )
    words = {item.word_id: item for item in battery_electrothermal.action_words()}
    unit_b = battery_electrothermal.BatteryElectrothermalPreparation(
        config.electrothermal_unit_id,
        battery_electrothermal.BatteryElectrothermalStage.DEVELOPMENT,
        battery_electrothermal.BatteryElectrothermalUnitRole.DEVELOPMENT,
        'middle-state-of-charge-cool',
        config.initial_soc,
        config.initial_temperature_k,
        False,
    )
    simulations = {}
    for token in ("hold", "i-early-150", "future-i-plus"):
        _, simulation, _, _ = battery_electrothermal._step_schedule(
            unit=unit_b,
            denominator=denominator,
            word=words[f"word.battery-electrothermal.{token}"],
            stop_s=150,
        )
        if simulation.solution.termination != "final time":
            raise RuntimeError(f"Electrothermal {token} did not reach 150 s")
        simulations[token] = simulation
    capacity = lambda token: float(
        simulations[token].solution["Discharge capacity [A.h]"](150)
    )
    voltage = lambda token: float(
        simulations[token].solution["Terminal voltage [V]"](150)
    )
    if not (
        abs(capacity("hold")) <= 1e-10
        and abs(capacity("i-early-150") - 150 / 3600) <= 1e-8
        and abs(capacity("future-i-plus") - capacity("hold")) <= 1e-10
        and voltage("i-early-150") < voltage("hold") - 0.01
        and abs(voltage("future-i-plus") - voltage("hold")) <= 1e-7
    ):
        raise ValueError(
            "Electrothermal action integral, voltage response or future cutoff differs"
        )

    unit_c = battery_restart.BatteryExactRestartPreparation(
        config.restart_unit_id,
        battery_restart.BatteryExactRestartStage.LOCALIZATION,
        battery_restart.BatteryExactRestartUnitRole.LOCALIZATION,
        'middle-state-of-charge-cool',
        config.initial_soc,
        config.initial_temperature_k,
        False,
    )
    history_c = next(
        item for item in battery_restart.histories() if item.history_id == config.restart_history_id
    )
    pybamm_module, source, prefix, _ = battery_restart._step_prefix(
        unit=unit_c, denominator=denominator, history=history_c
    )
    state = battery_restart.decode_hex(
        battery_restart._hex_values(np.asarray(prefix.all_ys[0], dtype=np.float64).reshape(-1))
    )
    source_inputs = {
        name: float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
        for name, value in prefix.all_inputs[-1].items()
    }
    hold_inputs = battery_restart._inputs(unit_c, Decimal(0), Decimal(0))
    clock = np.asarray([300.0, 450.0, 600.0])
    native = source.step(
        300, t_eval=clock - 300, starting_solution=prefix, inputs=hold_inputs
    )
    _, rebuilt = battery_restart._runtime_objects(unit_c, denominator)
    rebuilt.build(initial_soc=float(unit_c.initial_soc), inputs=hold_inputs)
    starting = battery_restart._starting_solution(
        pybamm_module,
        checkpoint_s=300,
        state=state,
        model=rebuilt.built_model,
        inputs=source_inputs,
    )
    reconstructed = rebuilt.step(
        300, t_eval=clock - 300, starting_solution=starting, inputs=hold_inputs
    )
    defects = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(native[name](clock))
                    - np.asarray(reconstructed[name](clock))
                )
            )
        )
        for name in (
            "Discharge capacity [A.h]",
            "Volume-averaged cell temperature [K]",
            "Terminal voltage [V]",
        )
    }
    if reconstructed.termination != "final time" or any(
        defects[name] > tolerance
        for name, tolerance in (
            ("Discharge capacity [A.h]", 1e-10),
            ("Volume-averaged cell temperature [K]", 1e-8),
            ("Terminal voltage [V]", 1e-8),
        )
    ):
        raise ValueError("Exact-restart lossless restart differs from native same-object hold")

    denominator_d = next(
        item
        for item in battery_reduced_observation.denominators()
        if item.denominator_id == config.denominator_id
    )
    history_d = next(
        item for item in battery_reduced_observation.histories() if item.history_id == config.reduced_observation_history_id
    )
    implementation_sha256 = sha256(Path(battery_reduced_observation.__file__).read_bytes()).hexdigest()
    captures = []
    for index, unit_id in enumerate(config.reduced_observation_unit_ids):
        unit_d = battery_reduced_observation.BatteryReducedObservationPreparation(
            unit_id=unit_id,
            stage=battery_reduced_observation.BatteryReducedObservationStage.DEVELOPMENT,
            stratum_id=('low-state-of-charge-cool' if index < 2 else 'high-state-of-charge-warm'),
            initial_soc=config.reduced_observation_initial_socs[index],
            initial_temperature_k=config.reduced_observation_initial_temperatures_k[index],
            reserve=False,
        )
        capture, restart = battery_reduced_observation.acquire_preparation(
            unit=unit_d,
            denominator=denominator_d,
            history=history_d,
            source_identity_sha256=config.fingerprint(),
            implementation_sha256=implementation_sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        if (
            capture["value"]["disposition"] != "COMPLETE"
            or restart["value"]["disposition"] != "COMPLETE"
        ):
            raise RuntimeError("Reduced-observation native capture or hold continuation stopped")
        if capture["value"]["checkpoint_s"] != 300 or capture["value"]["end_s"] != 600:
            raise ValueError("Reduced-observation checkpoint-relative clock differs")
        if (
            not restart["value"]["receiver_closed"]
            or not restart["value"]["state_bit_exact"]
        ):
            raise ValueError("Reduced-observation exact comparator did not close")
        captures.append(capture)
    chart = next(item for item in battery_reduced_observation.charts() if item.chart_id == config.reduced_observation_chart_id)
    scale, predictions = battery_reduced_observation.build_prediction_rows(
        development_captures=captures[:2],
        target_captures=captures[2:],
        chart=chart,
        leave_one_out=False,
    )
    if (
        len(predictions) != 1
        or predictions[0].target_unit_id == predictions[0].donor_unit_id
    ):
        raise ValueError("Reduced-observation did not select a distinct donor")
    prediction = predictions[0]
    return {
        "config_id": config.config_id,
        "source_version": pybamm.__version__,
        "denominator_id": denominator.denominator_id,
        "independent_preparations": 5,
        "nested_electrothermal_words": 3,
        "nested_restart_routes": 2,
        "electrothermal_response": {
            "unit_id": unit_b.unit_id,
            "horizon_s": 150,
            "hold_capacity_ah": capacity("hold"),
            "driven_capacity_ah": capacity("i-early-150"),
            "future_capacity_ah": capacity("future-i-plus"),
            "hold_voltage_v": voltage("hold"),
            "driven_voltage_v": voltage("i-early-150"),
        },
        "exact_restart": {
            "unit_id": unit_c.unit_id,
            "checkpoint_s": 300,
            "continuation_end_s": 600,
            "source_current_a": source_inputs["Current function [A]"],
            "maximum_restart_defects": defects,
        },
        "reduced_observation": {
            "unit_ids": list(config.reduced_observation_unit_ids),
            "checkpoint_s": 300,
            "continuation_end_s": 600,
            "chart_id": chart.chart_id,
            "coordinate_count": len(scale),
            "target_unit_id": prediction.target_unit_id,
            "donor_unit_id": prediction.donor_unit_id,
            "distance": str(prediction.distance),
            "inside_support": prediction.inside_support,
            "reason_codes": list(prediction.reason_codes),
        },
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['PyBaMMDevelopmentInput', "run_native_development_check"]
