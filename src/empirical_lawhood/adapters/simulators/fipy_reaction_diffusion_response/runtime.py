"""FiPy complete-unit execution for a signed reaction--diffusion field."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis import build_panel
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseActionRealization, SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseConditionResult, SelectiveDependenceResponseContextDecision, SelectiveDependenceResponseDisposition, SelectiveDependenceResponsePhase, SelectiveDependenceResponseReceiverObservation, SelectiveDependenceResponseTargetPanel, digest_ids
from empirical_lawhood.adapters.methods.selective_dependence_response.inference import noncompensating_disposition
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal

from empirical_lawhood.adapters.methods.selective_dependence_response.preparation_inputs import SelectiveDependenceResponsePreparationInput

from .preparation_inputs import DRAW_VARIABLE_IDS, fipy_preparation_input, validate_reviewed_preparation_input
from .contracts import FipyReactionDiffusionResponseFiPyDesign, FipyReactionDiffusionResponseFiPySourceQualification


def _decimal(value: float) -> Decimal:
    return Decimal(format(float(value), ".12g"))


def _rng(preparation_input: SelectiveDependenceResponsePreparationInput) -> np.random.Generator:
    if type(preparation_input) is not SelectiveDependenceResponsePreparationInput or preparation_input.draw_variable_ids != DRAW_VARIABLE_IDS:
        raise ValueError("native preparation requires explicit full numerical inputs in the exact variable draw order")
    validate_reviewed_preparation_input(preparation_input)
    return np.random.default_rng(preparation_input.full_seed)


def _preparation(preparation_input: SelectiveDependenceResponsePreparationInput) -> dict[str, float]:
    rng = _rng(preparation_input)
    return {
        "background-amplitude": rng.uniform(0.0, 0.0010),
        "residual-amplitude": rng.uniform(0.035, 0.045),
        "residual-centre": rng.uniform(0.27, 0.35),
        "residual-width": rng.uniform(0.045, 0.070),
    }


def _condition(
    design: FipyReactionDiffusionResponseFiPyDesign,
    preparation: dict[str, float],
    *,
    denominator_id: str,
    history_id: str,
    action_id: str,
    horizon_id: str,
) -> tuple[SelectiveDependenceResponseConditionResult, tuple[Decimal, ...]]:
    import fipy as fp  # type: ignore[import-untyped]

    length = float(design.domain_length)
    dx = length / design.mesh_cells
    mesh = fp.Grid1D(nx=design.mesh_cells, dx=dx)
    x = np.asarray(mesh.cellCenters[0], dtype=float)
    values = np.zeros(design.mesh_cells, dtype=float)
    if history_id == "residual-checkpoint":
        values = preparation["background-amplitude"] * (1.0 - x)
        values += preparation["residual-amplitude"] * np.exp(
            -0.5 * ((x - preparation["residual-centre"]) / preparation["residual-width"]) ** 2
        )
    field = fp.CellVariable(mesh=mesh, value=values, hasOld=True)
    field.constrain(0.0, mesh.facesRight)
    source_region = (x >= 0.15) & (x <= 0.25)
    requested_by_action = {
        "source-remove": float(design.action_rates[0]),
        "source-hold": float(design.action_rates[1]),
        "source-inject": float(design.action_rates[2]),
        "source-outside": float(design.outside_action_rate),
    }
    requested = requested_by_action[action_id]
    outside = action_id == "source-outside"
    accepted = 0.0 if outside else requested
    diffusivity = {
        "diffusivity-low": float(design.diffusivities[0]),
        "diffusivity-high": float(design.diffusivities[1]),
    }[denominator_id]
    horizon = {
        "short": float(design.horizon_seconds[0]),
        "long": float(design.horizon_seconds[1]),
    }[horizon_id]
    dt = float(design.timestep_seconds)
    action_duration = float(design.action_duration_seconds)
    source = fp.CellVariable(mesh=mesh, value=0.0)
    equation = (
        fp.TransientTerm()
        == fp.DiffusionTerm(coeff=diffusivity) - (float(design.decay_rate) * field) + source
    )
    initial_mass = float(np.sum(values) * dx)
    source_integral = 0.0
    decay_integral = 0.0
    boundary_outflow_integral = 0.0
    peak = float(np.max(values))
    elapsed = 0.0
    steps = 0
    stopped = False
    stop_code: str | None = None
    margins: tuple[Decimal, ...]
    try:
        while elapsed < horizon - dt / 2:
            step = min(dt, horizon - elapsed)
            field.updateOld()
            applied = accepted if elapsed < action_duration - dt / 2 else 0.0
            source.setValue(applied, where=source_region)
            previous_mass = float(np.sum(np.asarray(field.value)) * dx)
            equation.solve(var=field, dt=step)
            current = np.asarray(field.value, dtype=float)
            current_mass = float(np.sum(current) * dx)
            applied_mass_rate = applied * float(np.sum(source_region)) * dx
            source_integral += applied_mass_rate * step
            decay_integral += float(design.decay_rate) * (previous_mass + current_mass) * step / 2
            outward_flux = max(0.0, diffusivity * current[-1] / (dx / 2))
            boundary_outflow_integral += outward_flux * step
            peak = max(peak, float(np.max(current)))
            elapsed += step
            steps += 1
            if steps > design.maximum_solver_steps:
                raise RuntimeError("solver-step-ceiling")
        final = np.asarray(field.value, dtype=float)
        final_mass = float(np.sum(final) * dx)
        boundary_flux = max(0.0, diffusivity * final[-1] / (dx / 2))
        balance_error = abs(
            final_mass
            - (initial_mass + source_integral - decay_integral - boundary_outflow_integral)
        )
        observed = {
            "balance-error": balance_error,
            "boundary-flux": boundary_flux,
            "downstream-mean": float(np.mean(final[x >= 0.75])),
            "field-mass": final_mass,
            "field-maximum": float(np.max(final)),
            "field-minimum": float(np.min(final)),
        }
        margins = (
            _decimal(float(design.maximum_balance_error) - balance_error),
            _decimal(float(design.maximum_boundary_flux) - boundary_flux),
            _decimal(
                min(
                    observed["downstream-mean"] - float(design.target_downstream_lower),
                    float(design.target_downstream_upper) - observed["downstream-mean"],
                )
            ),
            _decimal(observed["field-minimum"] + float(design.nonnegativity_tolerance)),
            _decimal(float(design.maximum_peak_field) - observed["field-maximum"]),
        )
    except Exception as error:  # FiPy failure is retained, never imputed.
        stopped = True
        stop_code = f"fipy-{type(error).__name__.lower()}"
        observed = {
            "balance-error": 0.0,
            "boundary-flux": 0.0,
            "downstream-mean": 0.0,
            "field-mass": 0.0,
            "field-maximum": 0.0,
            "field-minimum": 0.0,
        }
        margins = ()
    condition_id = f"condition.{denominator_id}.{history_id}.{action_id}.{horizon_id}"
    realized_duration = min(horizon, action_duration)
    source_region_length = float(np.sum(source_region)) * dx
    action = SelectiveDependenceResponseActionRealization(
        realization_id=f"realization.{condition_id}",
        action_id=action_id,
        requested_value=_decimal(requested),
        requested_unit="field-amplitude-per-second",
        accepted_value=_decimal(accepted),
        accepted_unit="field-amplitude-per-second",
        applied_value=_decimal(accepted),
        applied_unit="field-amplitude-per-second",
        realized_value=_decimal(accepted * source_region_length * realized_duration),
        realized_unit="field-mass",
        requested_clock=Decimal("0"),
        accepted_clock=Decimal("0"),
        applied_clock=Decimal("0"),
        realized_clock=_decimal(realized_duration),
        acceptance_state="rejected-to-hold" if outside else "accepted",
    )
    receivers = tuple(
        SelectiveDependenceResponseReceiverObservation(
            observation_id=f"observation.{condition_id}.{receiver_id}",
            receiver_id=receiver_id,
            value=_decimal(value),
            native_unit={
                "balance-error": "field-mass",
                "boundary-flux": "field-flux",
                "downstream-mean": "field-amplitude",
                "field-mass": "field-mass",
                "field-maximum": "field-amplitude",
                "field-minimum": "field-amplitude",
            }[receiver_id],
            valid=not stopped,
        )
        for receiver_id, value in sorted(observed.items())
    )
    margin_ids = (
        "balance-error-ceiling",
        "boundary-flux-ceiling",
        "downstream-target-band",
        "nonnegativity-tolerance",
        "peak-field-ceiling",
    )
    gate_margins = tuple(
        NamedDecimal(value_id=margin_id, value=value, unit="native-margin")
        for margin_id, value in zip(margin_ids, margins, strict=True)
    )
    return (
        SelectiveDependenceResponseConditionResult(
            condition_id=condition_id,
            denominator_id=denominator_id,
            history_id=history_id,
            action_id=action_id,
            horizon_id=horizon_id,
            support_state="outside-support" if outside else "inside-support",
            action=action,
            receivers=receivers,
            gate_margins=gate_margins,
            fibre_admitted=(
                None if stopped else (False if outside else all(value >= 0 for value in margins))
            ),
            disposition=SelectiveDependenceResponseDisposition.UNEVALUABLE,
            stopped=stopped,
            stop_code=stop_code,
        ),
        margins,
    )


def execute_fipy_complete_unit(
    design: FipyReactionDiffusionResponseFiPyDesign,
    *,
    complete_unit_id: str,
    phase: SelectiveDependenceResponsePhase,
    preparation_input: SelectiveDependenceResponsePreparationInput | None = None,
) -> SelectiveDependenceResponseCompleteUnitResult:
    if type(preparation_input) is not SelectiveDependenceResponsePreparationInput:
        raise ValueError("native complete unit requires explicit numerical preparation inputs before solver work")
    preparation_input.validate_context(
        complete_unit_id=complete_unit_id,
        target_id=design.target_id,
        target_design=ObjectIdentity.from_record(design.design_id, design),
        phase=phase.value,
        draw_variable_ids=DRAW_VARIABLE_IDS,
    )
    preparation = _preparation(preparation_input)
    raw: dict[
        tuple[str, str, str], dict[str, tuple[SelectiveDependenceResponseConditionResult, tuple[Decimal, ...]]]
    ] = {}
    for denominator_id in design.denominator_ids:
        for history_id in design.history_ids:
            for horizon_id in design.horizon_ids:
                key = denominator_id, history_id, horizon_id
                raw[key] = {}
                for action_id in design.action_ids:
                    raw[key][action_id] = _condition(
                        design,
                        preparation,
                        denominator_id=denominator_id,
                        history_id=history_id,
                        action_id=action_id,
                        horizon_id=horizon_id,
                    )
    conditions = []
    context_decisions = []
    for (denominator_id, history_id, horizon_id), branches in raw.items():
        hold = branches["source-hold"][1]
        active = {
            action_id: margins
            for action_id, (_, margins) in branches.items()
            if action_id not in {"source-hold", "source-outside"}
        }
        fibre = noncompensating_disposition(
            active_action_gate_margins=active,
            hold_gate_margins=hold or None,
        )
        admitted_active_action_ids = tuple(
            sorted(
                action_id
                for action_id, margins in active.items()
                if margins and all(value >= 0 for value in margins)
            )
        )
        active_fibres_complete = all(bool(margins) for margins in active.values())
        hold_viable = None if not hold else all(value >= 0 for value in hold)
        context_decisions.append(
            SelectiveDependenceResponseContextDecision(
                decision_id=f"decision.{denominator_id}.{history_id}.{horizon_id}",
                denominator_id=denominator_id,
                history_id=history_id,
                horizon_id=horizon_id,
                admitted_active_action_ids=admitted_active_action_ids,
                active_fibres_complete=active_fibres_complete,
                hold_action_id="source-hold",
                hold_viable=hold_viable,
                disposition=fibre,
            )
        )
        for action_id, (condition, margins) in branches.items():
            if condition.stopped:
                disposition = SelectiveDependenceResponseDisposition.UNEVALUABLE
            elif action_id == "source-outside":
                disposition = SelectiveDependenceResponseDisposition.NONATTEMPT
            elif action_id == "source-hold":
                disposition = (
                    SelectiveDependenceResponseDisposition.HOLD_ONLY
                    if hold and all(value >= 0 for value in hold)
                    else SelectiveDependenceResponseDisposition.NONATTEMPT
                )
            elif margins and all(value >= 0 for value in margins):
                disposition = SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
            elif fibre is SelectiveDependenceResponseDisposition.HOLD_ONLY:
                disposition = SelectiveDependenceResponseDisposition.HOLD_ONLY
            else:
                disposition = SelectiveDependenceResponseDisposition.NONATTEMPT
            conditions.append(
                SelectiveDependenceResponseConditionResult(
                    condition_id=condition.condition_id,
                    denominator_id=condition.denominator_id,
                    history_id=condition.history_id,
                    action_id=condition.action_id,
                    horizon_id=condition.horizon_id,
                    support_state=condition.support_state,
                    action=condition.action,
                    receivers=condition.receivers,
                    gate_margins=condition.gate_margins,
                    fibre_admitted=condition.fibre_admitted,
                    disposition=disposition,
                    stopped=condition.stopped,
                    stop_code=condition.stop_code,
                )
            )
    ordered = tuple(sorted(conditions, key=lambda value: value.condition_id))
    access = (
        OutcomeAccess.EVALUATION_SEALED
        if phase is SelectiveDependenceResponsePhase.EVALUATION
        else OutcomeAccess.DEVELOPMENT_VISIBLE
    )
    preparation_values = tuple(
        NamedDecimal(
            value_id=key,
            value=_decimal(value),
            unit="field-preparation-unit",
        )
        for key, value in sorted(preparation.items())
    )
    return SelectiveDependenceResponseCompleteUnitResult(
        result_id=f"result.{complete_unit_id}",
        target_id=design.target_id,
        complete_unit_id=complete_unit_id,
        phase=phase,
        preparation_values=preparation_values,
        conditions=ordered,
        context_decisions=tuple(sorted(context_decisions, key=lambda value: value.decision_id)),
        expected_condition_ids_sha256=digest_ids(tuple(value.condition_id for value in ordered)),
        source_version=design.fipy_version,
        solver_id="fipy-scipy-linear-lu",
        nested_conditions_count_as_units=False,
        outcome_access=access,
    )


def qualify_fipy_source(
    design: FipyReactionDiffusionResponseFiPyDesign,
    *,
    canary_unit_ids: tuple[str, ...],
) -> tuple[FipyReactionDiffusionResponseFiPySourceQualification, SelectiveDependenceResponseTargetPanel]:
    preparation_inputs = tuple(
        fipy_preparation_input(design, unit_id, SelectiveDependenceResponsePhase.CANARY)
        for unit_id in canary_unit_ids
    )
    import fipy

    units = tuple(
        execute_fipy_complete_unit(
            design,
            complete_unit_id=unit_id,
            phase=SelectiveDependenceResponsePhase.CANARY,
            preparation_input=preparation_input,
        )
        for unit_id, preparation_input in zip(canary_unit_ids, preparation_inputs, strict=True)
    )
    if not units:
        raise ValueError("FiPy source qualification requires excluded canaries")
    unit = units[0]
    replay = execute_fipy_complete_unit(
        design,
        complete_unit_id=unit.complete_unit_id,
        phase=SelectiveDependenceResponsePhase.CANARY,
        preparation_input=fipy_preparation_input(design, unit.complete_unit_id, SelectiveDependenceResponsePhase.CANARY),
    )
    valid = all(not condition.stopped for value in units for condition in value.conditions)
    reset = unit.canonical_bytes() == replay.canonical_bytes()
    balance_ok = all(
        observation.value <= design.maximum_balance_error
        for value in units
        for condition in value.conditions
        for observation in condition.receivers
        if observation.receiver_id == "balance-error"
    )
    actions = {
        condition.action.requested_value for value in units for condition in value.conditions
    }
    version_matches = fipy.__version__ == design.fipy_version
    signed_source_observable = any(value < 0 for value in actions) and any(
        value > 0 for value in actions
    )
    realized_integral_observable = all(
        condition.action.realized_unit != condition.action.applied_unit
        for value in units
        for condition in value.conditions
    )
    source_ready = all(
        (
            valid,
            reset,
            balance_ok,
            version_matches,
            signed_source_observable,
            realized_integral_observable,
        )
    )
    panel = build_panel(
        panel_id="panel.fipy.canary",
        target_id=design.target_id,
        complete_units=units,
    )
    qualification = FipyReactionDiffusionResponseFiPySourceQualification(
        qualification_id="qualification.fipy.source",
        design_id=design.design_id,
        observed_fipy_version=fipy.__version__,
        version_matches_design=version_matches,
        equation_constructed=valid,
        signed_source_observable=signed_source_observable,
        realized_integral_observable=realized_integral_observable,
        reset_reproducible=reset,
        balance_qualified=balance_ok,
        source_ready=source_ready,
        canary_panel=ObjectIdentity.from_record(panel.panel_id, panel),
        canary_result_identities=tuple(
            sorted(
                (ObjectIdentity.from_record(value.result_id, value) for value in units),
                key=lambda value: value.object_id,
            )
        ),
        excluded_complete_unit_count=len(units),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    return qualification, panel


def verify_fipy_source_binding(
    design: FipyReactionDiffusionResponseFiPyDesign,
    qualification: FipyReactionDiffusionResponseFiPySourceQualification,
) -> None:
    """Verify source identity without executing or observing a response."""

    import fipy

    if qualification.design_id != design.design_id:
        raise ValueError("FiPy source qualification binds another design")
    if fipy.__version__ != design.fipy_version:
        raise ValueError("FiPy package version drifted after source qualification")
    if not qualification.source_ready:
        raise ValueError("FiPy source qualification is not ready")


__all__ = [
    "execute_fipy_complete_unit",
    "qualify_fipy_source",
    "verify_fipy_source_binding",
]
