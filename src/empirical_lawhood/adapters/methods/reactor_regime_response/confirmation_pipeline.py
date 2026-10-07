"""Use frozen B forecasts for C calibration and all-assigned qualification."""

from __future__ import annotations

from math import isfinite

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .config import ROOTS
from .control_math import WordForecast, forecast_chart
from .measured_panel import CONTEXTS, MeasuredContext, measured_contexts
from .model_records import RegimeFitPackage
from .nomination_pipeline import _valid_word
from .nomination_records import RegimeNominationPackage
from .prediction_seal import seal_confirmation_predictions
from .preparation_evidence import preparation_evidence
from .qualification_math import SelectedRootOperand
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation

ConfirmationRow = tuple[
    RegimeCausalPreparation,
    RegimePrivatePreparation,
    RegimePredictionSeal,
    RegimeAssayPanel,
]


def validate_confirmation_rows(
    role: str,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    rows: tuple[ConfirmationRow, ...],
) -> None:
    expected = tuple(root for root, assigned, _, _ in ROOTS if assigned == role)
    if (
        role not in ("calibration", "qualification")
        or tuple(causal.root for causal, _, _, _ in rows) != expected
        or nomination.fit_package != ObjectIdentity.from_record(fit.package_id, fit)
    ):
        raise ValueError("C changed its fit/nomination or all-assigned root denominator")
    for causal, private, seal, assay in rows:
        root = causal.root
        causal_id = ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
        if (
            (causal.role, private.role, seal.role, assay.role) != (role,) * 4
            or (causal.seed, private.seed, seal.seed, assay.seed) != (causal.seed,) * 4
            or private.causal_preparation != causal_id
            or seal != seal_confirmation_predictions(causal, fit, nomination)
            or assay.causal_preparation != causal_id
            or assay.private_preparation
            != ObjectIdentity.from_record(f"{root}.private-preparation", private)
            or assay.sealed_prediction
            != ObjectIdentity.from_record(f"{root}.prediction-seal", seal)
        ):
            raise ValueError("C changed a persisted causal/prediction/assay receipt identity")


def selected_operand(
    row: ConfirmationRow,
    nomination: RegimeNominationPackage,
    *,
    q_temperature: float,
    q_cooling: float,
) -> SelectedRootOperand | None:
    """Recover the primary law from sealed forecasts, never a rival's intervals."""
    route = nomination.selected_route
    if route is None:
        return None
    causal, private, seal, assay = row
    selected = next(item for item in nomination.routes if item.route == route)
    if selected.coefficient_model_id is None or selected.absolute_model_id is None:
        raise ValueError("selected C route has no frozen primary models")
    contexts = {item.context: item for item in measured_contexts(causal, assay)}
    chart: MeasuredContext = contexts[route]
    evidence = preparation_evidence(causal, private, route)
    arrays = seal.predictions.unpack()
    ids = {model_id: index for index, model_id in enumerate(seal.candidate_ids)}
    index = CONTEXTS.index(route)
    coefficient_id = f"coefficient.{selected.coefficient_model_id}".lower()
    absolute_id = f"absolute.{selected.absolute_model_id}".lower()
    if coefficient_id not in ids or absolute_id not in ids:
        raise ValueError("C seal lacks its nominated primary coefficient/absolute fit")
    k, absolute_prediction_index = ids[coefficient_id], ids[absolute_id]
    prediction_present = bool(arrays["present"][k, index] and arrays["present"][absolute_prediction_index, index])
    forecasts: tuple[WordForecast, WordForecast, WordForecast] | None = None
    if prediction_present:
        masses = tuple(float(value) for value in arrays["masses"][index])
        projection = tuple(bool(value) for value in arrays["projection_valid"][index])
        measured_match = (
            chart.nominal_mass_kg is not None
            and chart.refined_mass_kg is not None
            and all(
                abs(projected - actual) <= 1e-6
                for actuals in (chart.nominal_mass_kg, chart.refined_mass_kg)
                for projected, actual in zip(masses, actuals, strict=True)
            )
        )
        supported = bool(arrays["support"][k, index] and arrays["support"][absolute_prediction_index, index])
        if all(isfinite(value) for value in masses) and masses[0] == 0 < masses[1] < masses[2]:
            values = forecast_chart(
                float(arrays["forecast"][absolute_prediction_index, index]),
                float(arrays["forecast"][k, index]),
                masses,  # type: ignore[arg-type]
                q_temperature,
                q_cooling,
                (supported,) * 3,
                tuple(
                    bool(valid and measured_match and _valid_word(assay, route, word))
                    for word, valid in enumerate(projection)
                ),  # type: ignore[arg-type]
            )
            forecasts = values  # type: ignore[assignment]
    grid_max = (
        max(evidence.nominal_max_K, evidence.refined_max_K)
        if evidence.nominal_max_K is not None and evidence.refined_max_K is not None
        else None
    )
    return SelectedRootOperand(
        causal.root,
        route,
        chart,
        forecasts,
        evidence.prefix_valid,
        grid_max,
    )
