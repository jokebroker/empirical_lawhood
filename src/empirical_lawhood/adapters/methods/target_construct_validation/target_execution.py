"""Native simulator execution for frozen target construct validation complete units."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import math
import time
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .target_freeze import TargetConstructValidationTargetDesignFreeze


class TargetConstructValidationExecutionPhase(StrEnum):
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationActionRealization(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-action-realization'

    realization_id: str
    native_action_id: str
    requested_value: Decimal
    accepted_value: Decimal
    applied_value: Decimal
    realized_value: Decimal
    requested_clock: Decimal
    accepted_clock: Decimal
    applied_clock: Decimal
    realized_clock: Decimal
    native_unit: str
    acceptance_rule: str

    def __post_init__(self) -> None:
        for name in ("realization_id", "native_action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "requested_value",
            "accepted_value",
            "applied_value",
            "realized_value",
            "requested_clock",
            "accepted_clock",
            "applied_clock",
            "realized_clock",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.acceptance_rule, field_name="acceptance_rule")
        if not (
            self.requested_clock <= self.accepted_clock <= self.applied_clock <= self.realized_clock
        ):
            raise ValueError("action realization clocks are not causal")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationReceiverValue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-receiver-value'

    receiver_id: str
    value: Decimal
    native_unit: str
    finite: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if not self.finite:
            raise ValueError("nonfinite receiver values require a typed stopped condition")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNativeConditionResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-native-condition-result'

    condition_id: str
    denominator_value_id: str
    history_value_id: str
    native_action_id: str
    horizon_value_id: str
    action: TargetConstructValidationActionRealization
    receivers: tuple[TargetConstructValidationReceiverValue, ...]
    solver_step_count: int
    stopped: bool
    stop_code: str | None

    def __post_init__(self) -> None:
        for name in (
            "condition_id",
            "denominator_value_id",
            "history_value_id",
            "native_action_id",
            "horizon_value_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.receivers,
            attribute="receiver_id",
            field_name="receivers",
        )
        if self.action.native_action_id != self.native_action_id:
            raise ValueError("condition and action ledger differ")
        if self.solver_step_count <= 0:
            raise ValueError("condition must report positive solver work")
        if self.stopped != (self.stop_code is not None):
            raise ValueError("condition stop status and code differ")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCompleteUnitResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-complete-unit-result'

    result_id: str
    target_id: str
    phase: TargetConstructValidationExecutionPhase
    complete_unit_id: str
    seed: int
    target_design: ObjectIdentity
    conditions: tuple[TargetConstructValidationNativeConditionResult, ...]
    condition_ids_sha256: str
    elapsed_seconds: Decimal
    simulator_version: str
    solver_id: str
    generator_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("result_id", "target_id", "complete_unit_id", "solver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.seed < 0:
            raise ValueError("complete-unit seed must be nonnegative")
        require_sorted_unique_ids(
            self.conditions,
            attribute="condition_id",
            field_name="conditions",
        )
        validate_nonempty(self.simulator_version, field_name="simulator_version")
        validate_decimal(self.elapsed_seconds, field_name="elapsed_seconds")
        expected_digest = sha256(
            ("\n".join(value.condition_id for value in self.conditions) + "\n").encode("ascii")
        ).hexdigest()
        if self.condition_ids_sha256 != expected_digest:
            raise ValueError("complete-unit condition digest differs")
        if self.generator_response_count != len(self.conditions):
            raise ValueError("generator response count is not condition-derived")
        expected_access = {
            TargetConstructValidationExecutionPhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            TargetConstructValidationExecutionPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            TargetConstructValidationExecutionPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("complete-unit phase and outcome access differ")


_FROZEN_COMPLETE_UNIT_SEEDS = {
    'unit.cantera-stirred-reactor.canary.001': 2038722229,
    'unit.cantera-stirred-reactor.canary.002': 3288842631,
    'unit.cantera-stirred-reactor.development.001': 3787014210,
    'unit.cantera-stirred-reactor.development.002': 802212272,
    'unit.cantera-stirred-reactor.development.003': 2415435086,
    'unit.cantera-stirred-reactor.development.004': 745175017,
    'unit.cantera-stirred-reactor.development.005': 2039086118,
    'unit.cantera-stirred-reactor.development.006': 1011841609,
    'unit.cantera-stirred-reactor.development.007': 2001662239,
    'unit.cantera-stirred-reactor.development.008': 152087206,
    'unit.cantera-stirred-reactor.development.009': 114491280,
    'unit.cantera-stirred-reactor.development.010': 473600295,
    'unit.cantera-stirred-reactor.development.011': 3134989680,
    'unit.cantera-stirred-reactor.development.012': 5666336,
    'unit.cantera-stirred-reactor.development.013': 3146521396,
    'unit.cantera-stirred-reactor.development.014': 3627542165,
    'unit.cantera-stirred-reactor.development.015': 3366604821,
    'unit.cantera-stirred-reactor.development.016': 788701232,
    'unit.cantera-stirred-reactor.development.017': 354923907,
    'unit.cantera-stirred-reactor.development.018': 759421926,
    'unit.cantera-stirred-reactor.development.019': 720406300,
    'unit.cantera-stirred-reactor.development.020': 356767006,
    'unit.cantera-stirred-reactor.development.021': 70439956,
    'unit.cantera-stirred-reactor.development.022': 725334882,
    'unit.cantera-stirred-reactor.development.023': 1451518372,
    'unit.cantera-stirred-reactor.development.024': 2841740288,
    'unit.cantera-stirred-reactor.evaluation.001': 864851497,
    'unit.cantera-stirred-reactor.evaluation.002': 1028420286,
    'unit.cantera-stirred-reactor.evaluation.003': 3066373070,
    'unit.cantera-stirred-reactor.evaluation.004': 1122831187,
    'unit.cantera-stirred-reactor.evaluation.005': 306937153,
    'unit.cantera-stirred-reactor.evaluation.006': 3588898156,
    'unit.cantera-stirred-reactor.evaluation.007': 281067622,
    'unit.cantera-stirred-reactor.evaluation.008': 4166642581,
    'unit.cantera-stirred-reactor.evaluation.009': 646197988,
    'unit.cantera-stirred-reactor.evaluation.010': 2968200328,
    'unit.cantera-stirred-reactor.evaluation.011': 4051666416,
    'unit.cantera-stirred-reactor.evaluation.012': 1444815768,
    'unit.cantera-stirred-reactor.evaluation.013': 1470222219,
    'unit.cantera-stirred-reactor.evaluation.014': 1086521280,
    'unit.cantera-stirred-reactor.evaluation.015': 295342877,
    'unit.cantera-stirred-reactor.evaluation.016': 4279233278,
    'unit.cantera-stirred-reactor.evaluation.017': 1288246675,
    'unit.cantera-stirred-reactor.evaluation.018': 4159931950,
    'unit.cantera-stirred-reactor.evaluation.019': 4282848706,
    'unit.cantera-stirred-reactor.evaluation.020': 2631167745,
    'unit.cantera-stirred-reactor.evaluation.021': 353330204,
    'unit.cantera-stirred-reactor.evaluation.022': 114222647,
    'unit.cantera-stirred-reactor.evaluation.023': 739117758,
    'unit.cantera-stirred-reactor.evaluation.024': 4076515505,
    'unit.cantera-stirred-reactor.evaluation.025': 3898430447,
    'unit.cantera-stirred-reactor.evaluation.026': 1340398538,
    'unit.cantera-stirred-reactor.evaluation.027': 459521916,
    'unit.cantera-stirred-reactor.evaluation.028': 315246847,
    'unit.cantera-stirred-reactor.evaluation.029': 1332535736,
    'unit.cantera-stirred-reactor.evaluation.030': 1816131908,
    'unit.cantera-stirred-reactor.evaluation.031': 1715071642,
    'unit.cantera-stirred-reactor.evaluation.032': 1753273060,
    'unit.cantera-stirred-reactor.reserve.001': 4058432228,
    'unit.cantera-stirred-reactor.reserve.002': 2443394722,
    'unit.cantera-stirred-reactor.reserve.003': 3455208436,
    'unit.cantera-stirred-reactor.reserve.004': 736216140,
    'unit.cantera-stirred-reactor.reserve.005': 510413315,
    'unit.cantera-stirred-reactor.reserve.006': 160363955,
    'unit.cantera-stirred-reactor.reserve.007': 614900666,
    'unit.cantera-stirred-reactor.reserve.008': 798769802,
    'unit.fipy-source-diffusion.canary.001': 3073166223,
    'unit.fipy-source-diffusion.canary.002': 2170074350,
    'unit.fipy-source-diffusion.development.001': 3994264723,
    'unit.fipy-source-diffusion.development.002': 1713455518,
    'unit.fipy-source-diffusion.development.003': 1596987497,
    'unit.fipy-source-diffusion.development.004': 2504551103,
    'unit.fipy-source-diffusion.development.005': 960871577,
    'unit.fipy-source-diffusion.development.006': 1442594408,
    'unit.fipy-source-diffusion.development.007': 488907029,
    'unit.fipy-source-diffusion.development.008': 705249956,
    'unit.fipy-source-diffusion.development.009': 4168315881,
    'unit.fipy-source-diffusion.development.010': 3453092975,
    'unit.fipy-source-diffusion.development.011': 393820248,
    'unit.fipy-source-diffusion.development.012': 2030502234,
    'unit.fipy-source-diffusion.development.013': 2175796624,
    'unit.fipy-source-diffusion.development.014': 140303180,
    'unit.fipy-source-diffusion.development.015': 374612148,
    'unit.fipy-source-diffusion.development.016': 3474007460,
    'unit.fipy-source-diffusion.development.017': 2458363505,
    'unit.fipy-source-diffusion.development.018': 2277646525,
    'unit.fipy-source-diffusion.development.019': 2745775813,
    'unit.fipy-source-diffusion.development.020': 95824055,
    'unit.fipy-source-diffusion.development.021': 3310620390,
    'unit.fipy-source-diffusion.development.022': 3616583616,
    'unit.fipy-source-diffusion.development.023': 3023090196,
    'unit.fipy-source-diffusion.development.024': 1099108441,
    'unit.fipy-source-diffusion.evaluation.001': 1617539427,
    'unit.fipy-source-diffusion.evaluation.002': 1735635673,
    'unit.fipy-source-diffusion.evaluation.003': 3571770893,
    'unit.fipy-source-diffusion.evaluation.004': 1002781243,
    'unit.fipy-source-diffusion.evaluation.005': 1488697714,
    'unit.fipy-source-diffusion.evaluation.006': 3168103771,
    'unit.fipy-source-diffusion.evaluation.007': 2298732260,
    'unit.fipy-source-diffusion.evaluation.008': 215196895,
    'unit.fipy-source-diffusion.evaluation.009': 2969714290,
    'unit.fipy-source-diffusion.evaluation.010': 2036807923,
    'unit.fipy-source-diffusion.evaluation.011': 3271755162,
    'unit.fipy-source-diffusion.evaluation.012': 1117305869,
    'unit.fipy-source-diffusion.evaluation.013': 1065077928,
    'unit.fipy-source-diffusion.evaluation.014': 2930727105,
    'unit.fipy-source-diffusion.evaluation.015': 1941623034,
    'unit.fipy-source-diffusion.evaluation.016': 1426684755,
    'unit.fipy-source-diffusion.evaluation.017': 1829118446,
    'unit.fipy-source-diffusion.evaluation.018': 3761726020,
    'unit.fipy-source-diffusion.evaluation.019': 4008131610,
    'unit.fipy-source-diffusion.evaluation.020': 526684395,
    'unit.fipy-source-diffusion.evaluation.021': 4108494102,
    'unit.fipy-source-diffusion.evaluation.022': 1557940005,
    'unit.fipy-source-diffusion.evaluation.023': 1876653411,
    'unit.fipy-source-diffusion.evaluation.024': 4190178941,
    'unit.fipy-source-diffusion.evaluation.025': 925548694,
    'unit.fipy-source-diffusion.evaluation.026': 1884182966,
    'unit.fipy-source-diffusion.evaluation.027': 3293460823,
    'unit.fipy-source-diffusion.evaluation.028': 3694399696,
    'unit.fipy-source-diffusion.evaluation.029': 2228068531,
    'unit.fipy-source-diffusion.evaluation.030': 2377959061,
    'unit.fipy-source-diffusion.evaluation.031': 2462250283,
    'unit.fipy-source-diffusion.evaluation.032': 2839541160,
    'unit.fipy-source-diffusion.reserve.001': 2538891864,
    'unit.fipy-source-diffusion.reserve.002': 1917947213,
    'unit.fipy-source-diffusion.reserve.003': 1043986201,
    'unit.fipy-source-diffusion.reserve.004': 4279238301,
    'unit.fipy-source-diffusion.reserve.005': 1621229318,
    'unit.fipy-source-diffusion.reserve.006': 1623337689,
    'unit.fipy-source-diffusion.reserve.007': 2124358830,
    'unit.fipy-source-diffusion.reserve.008': 2621348302,
}


def _seed(unit_id: str) -> int:
    if unit_id in _FROZEN_COMPLETE_UNIT_SEEDS:
        return _FROZEN_COMPLETE_UNIT_SEEDS[unit_id]
    raise ValueError("complete unit has no reviewed numerical seed commitment; name-derived fallback is unavailable")


def _decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("simulator returned a nonfinite value")
    return Decimal(repr(float(value)))


def _cantera_condition(
    *,
    unit_id: str,
    denominator_id: str,
    history_id: str,
    action_id: str,
    seed: int,
) -> tuple[TargetConstructValidationNativeConditionResult, str]:
    import cantera as ct  # type: ignore[import-not-found]

    residence = 0.10 if denominator_id.endswith("0p10s") else 0.20
    initial_temperature = 1100.0 if history_id.endswith("1100k") else 950.0
    perturbation = float(np.random.default_rng(seed).normal(0.0, 0.25))
    requested_multiplier = {
        "action.low-flow": 0.75,
        "action.hold": 1.0,
        "action.high-flow": 1.25,
    }[action_id]
    accepted_multiplier = min(1.25, max(0.75, requested_multiplier))

    upstream_gas = ct.Solution("gri30.yaml")
    upstream_gas.TPX = 300.0, ct.one_atm, "CH4:0.05,O2:0.21,N2:0.74"
    reactor_gas = ct.Solution("gri30.yaml")
    reactor_gas.TPX = (
        initial_temperature + perturbation,
        ct.one_atm,
        "CO2:0.10,H2O:0.20,N2:0.70",
    )
    upstream = ct.Reservoir(upstream_gas, clone=True)
    reactor = ct.IdealGasReactor(reactor_gas, energy="on", volume=1.0e-3, clone=True)
    downstream = ct.Reservoir(reactor_gas, clone=True)
    base_mdot = reactor.mass / residence
    applied_mdot = accepted_multiplier * base_mdot
    controller = ct.MassFlowController(upstream, reactor, mdot=applied_mdot)
    ct.PressureController(reactor, downstream, primary=controller, K=1.0e-5)
    network = ct.ReactorNet([reactor])
    horizon = 5.0 * residence
    network.advance(horizon)
    realized_mdot = float(controller.mass_flow_rate)
    thermo = reactor.phase
    receivers = tuple(
        sorted(
            (
                TargetConstructValidationReceiverValue(
                    receiver_id="receiver.co2-mole-fraction",
                    value=_decimal(float(thermo["CO2"].X[0])),
                    native_unit="mole-fraction",
                    finite=True,
                ),
                TargetConstructValidationReceiverValue(
                    receiver_id="receiver.temperature",
                    value=_decimal(float(reactor.T)),
                    native_unit="kelvin",
                    finite=True,
                ),
            ),
            key=lambda value: value.receiver_id,
        )
    )
    condition_id = f"condition.{unit_id}.{denominator_id}.{history_id}.{action_id}"
    return (
        TargetConstructValidationNativeConditionResult(
            condition_id=condition_id,
            denominator_value_id=denominator_id,
            history_value_id=history_id,
            native_action_id=action_id,
            horizon_value_id="horizon.five-residence-times",
            action=TargetConstructValidationActionRealization(
                realization_id=f"realization.{condition_id}",
                native_action_id=action_id,
                requested_value=_decimal(requested_multiplier),
                accepted_value=_decimal(accepted_multiplier),
                applied_value=_decimal(applied_mdot),
                realized_value=_decimal(realized_mdot),
                requested_clock=Decimal("0"),
                accepted_clock=Decimal("0"),
                applied_clock=Decimal("0"),
                realized_clock=_decimal(horizon),
                native_unit="flow-multiplier-then-kilogram-per-second",
                acceptance_rule="Clip requested flow multiplier to the frozen [0.75,1.25] chart.",
            ),
            receivers=receivers,
            solver_step_count=max(1, int(network.solver_stats.get("steps", 1))),
            stopped=False,
            stop_code=None,
        ),
        ct.__version__,
    )


def _fipy_condition(
    *,
    unit_id: str,
    denominator_id: str,
    history_id: str,
    action_id: str,
    seed: int,
) -> tuple[TargetConstructValidationNativeConditionResult, str]:
    import fipy  # type: ignore
    from fipy import (
        CellVariable,
        DiffusionTerm,
        Grid1D,
        ImplicitSourceTerm,
        LinearLUSolver,
        TransientTerm,
    )

    diffusivity = 0.02 if denominator_id.endswith("0p02") else 0.08
    requested_rate = {
        "action.low-source": 0.25,
        "action.hold": 0.50,
        "action.high-source": 0.75,
    }[action_id]
    accepted_rate = min(0.75, max(0.25, requested_rate))
    unit_factor = 1.0 + float(np.random.default_rng(seed).normal(0.0, 1.0e-4))
    applied_rate = accepted_rate * unit_factor
    nx = 50
    length = 1.0
    mesh = Grid1D(nx=nx, Lx=length)
    x = np.asarray(mesh.cellCenters[0], dtype=float)
    if history_id == "history.residual-gaussian":
        initial = 0.10 * np.exp(-(((x - 0.35) / 0.10) ** 2))
    else:
        initial = np.zeros_like(x)
    field = CellVariable(mesh=mesh, value=initial, hasOld=True)
    field.constrain(0.0, mesh.facesRight)
    source = CellVariable(mesh=mesh, value=0.0)
    equation = TransientTerm() == (
        DiffusionTerm(coeff=diffusivity) - ImplicitSourceTerm(coeff=0.10) + source
    )
    dt = 0.01
    steps = 50
    pulse_steps = 10
    source_mask = x < 0.20
    realized_integral = 0.0
    solver = LinearLUSolver(tolerance=1.0e-10, iterations=1000)
    for step in range(steps):
        field.updateOld()
        source.setValue(applied_rate if step < pulse_steps else 0.0, where=source_mask)
        equation.solve(var=field, dt=dt, solver=solver)
        if step < pulse_steps:
            realized_integral += float(np.sum(np.asarray(source.value) * mesh.cellVolumes)) * dt
    values = np.asarray(field.value, dtype=float)
    downstream_mean = float(np.mean(values[x >= 0.70]))
    dx = length / nx
    outward_flux = float(diffusivity * values[-1] / (0.5 * dx))
    receivers = tuple(
        sorted(
            (
                TargetConstructValidationReceiverValue(
                    receiver_id="receiver.downstream-mean",
                    value=_decimal(downstream_mean),
                    native_unit="field-amplitude",
                    finite=True,
                ),
                TargetConstructValidationReceiverValue(
                    receiver_id="receiver.right-outward-flux",
                    value=_decimal(outward_flux),
                    native_unit="field-flux",
                    finite=True,
                ),
            ),
            key=lambda value: value.receiver_id,
        )
    )
    condition_id = f"condition.{unit_id}.{denominator_id}.{history_id}.{action_id}"
    return (
        TargetConstructValidationNativeConditionResult(
            condition_id=condition_id,
            denominator_value_id=denominator_id,
            history_value_id=history_id,
            native_action_id=action_id,
            horizon_value_id="horizon.post-source-0p5s",
            action=TargetConstructValidationActionRealization(
                realization_id=f"realization.{condition_id}",
                native_action_id=action_id,
                requested_value=_decimal(requested_rate),
                accepted_value=_decimal(accepted_rate),
                applied_value=_decimal(applied_rate),
                realized_value=_decimal(realized_integral),
                requested_clock=Decimal("0"),
                accepted_clock=Decimal("0"),
                applied_clock=Decimal("0"),
                realized_clock=Decimal("0.5"),
                native_unit="source-rate-then-integrated-source",
                acceptance_rule="Clip requested source rate to the frozen [0.25,0.75] chart.",
            ),
            receivers=receivers,
            solver_step_count=steps,
            stopped=False,
            stop_code=None,
        ),
        fipy.__version__,
    )


def execute_complete_unit(
    design: TargetConstructValidationTargetDesignFreeze,
    *,
    phase: TargetConstructValidationExecutionPhase,
    complete_unit_id: str,
) -> TargetConstructValidationCompleteUnitResult:
    rosters = {
        TargetConstructValidationExecutionPhase.CANARY: design.task.canary_unit_ids,
        TargetConstructValidationExecutionPhase.DEVELOPMENT: design.task.development_complete_unit_ids,
        TargetConstructValidationExecutionPhase.EVALUATION: design.task.evaluation_complete_unit_ids,
    }
    if complete_unit_id not in rosters[phase]:
        raise ValueError("complete unit lies outside the frozen phase roster")
    seed = _seed(complete_unit_id)
    started = time.monotonic()
    conditions = []
    versions: set[str] = set()
    runner = _cantera_condition if design.task.package_name == "cantera" else _fipy_condition
    for denominator_id, history_id, action_id in (
        (denominator, history, action)
        for denominator in design.task.denominator_value_ids
        for history in design.task.history_value_ids
        for action in design.task.native_action_value_ids
    ):
        condition, version = runner(
            unit_id=complete_unit_id,
            denominator_id=denominator_id,
            history_id=history_id,
            action_id=action_id,
            seed=seed,
        )
        conditions.append(condition)
        versions.add(version)
    if len(versions) != 1:
        raise ValueError("simulator version changed within a complete unit")
    values = tuple(sorted(conditions, key=lambda value: value.condition_id))
    elapsed = time.monotonic() - started
    return TargetConstructValidationCompleteUnitResult(
        result_id=f"complete-unit-result.{complete_unit_id}",
        target_id=design.task.target_id,
        phase=phase,
        complete_unit_id=complete_unit_id,
        seed=seed,
        target_design=ObjectIdentity.from_record(design.freeze_id, design),
        conditions=values,
        condition_ids_sha256=sha256(
            ("\n".join(value.condition_id for value in values) + "\n").encode("ascii")
        ).hexdigest(),
        elapsed_seconds=_decimal(elapsed),
        simulator_version=next(iter(versions)),
        solver_id=design.task.solver_id,
        generator_response_count=len(values),
        outcome_access=(
            OutcomeAccess.EVALUATION_SEALED
            if phase is TargetConstructValidationExecutionPhase.EVALUATION
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        ),
    )


__all__ = [
    'TargetConstructValidationActionRealization',
    'TargetConstructValidationCompleteUnitResult',
    'TargetConstructValidationExecutionPhase',
    'TargetConstructValidationNativeConditionResult',
    'TargetConstructValidationReceiverValue',
    "execute_complete_unit",
]
