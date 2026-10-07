"""Seal all B nomination predictions from fit-root models before assays."""

from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING

import numpy as np

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .causal_contexts import CausalContext, causal_contexts
from .information_models import causal_inputs
from .measured_panel import CONTEXTS
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .records import RegimeCausalPreparation, RegimePredictionSeal
from .release import require_scientific_release

if TYPE_CHECKING:
    from .law_terminal import RegimeJointLawResult
    from .qualification_pipeline import RegimeQualificationPackage


def _support(lower: tuple[D, ...], upper: tuple[D, ...], values: tuple[float, ...]) -> bool:
    return all(float(lo) <= value <= float(hi) for lo, hi, value in zip(lower, upper, values, strict=True))


def seal_fit_acquisition(preparation: RegimeCausalPreparation) -> RegimePredictionSeal:
    if preparation.role != "fit":
        raise ValueError("only fit roots can use an acquisition-only seal")
    contexts = causal_contexts(preparation)
    return RegimePredictionSeal(
        preparation.root,
        preparation.role,
        preparation.seed,
        ObjectIdentity.from_record(f"{preparation.root}.causal-preparation", preparation),
        None,
        tuple(row.input_sha256 for row in contexts),
        (),
        LocalArrayPayload.pack({}),
        "FIT_ACQUISITION_ONLY",
    )


def _base_feature(context: CausalContext, mask: str) -> tuple[float, ...] | None:
    return context.current if mask == "current" else context.history


def seal_nomination_predictions(
    preparation: RegimeCausalPreparation,
    fit_package: RegimeFitPackage,
) -> RegimePredictionSeal:
    if preparation.role != "nomination":
        raise ValueError("nomination seal changed the assigned root role")
    return _seal_predictions(
        preparation,
        fit_package,
        ObjectIdentity.from_record(fit_package.package_id, fit_package),
    )


def seal_confirmation_predictions(
    preparation: RegimeCausalPreparation,
    fit_package: RegimeFitPackage,
    nomination: RegimeNominationPackage,
) -> RegimePredictionSeal:
    if (
        preparation.role not in ("calibration", "qualification")
        or nomination.fit_package
        != ObjectIdentity.from_record(fit_package.package_id, fit_package)
    ):
        raise ValueError("confirmation seal changed its B nominee or assigned role")
    return _seal_predictions(
        preparation,
        fit_package,
        ObjectIdentity.from_record(nomination.package_id, nomination),
    )


def seal_prospective_predictions(
    preparation: RegimeCausalPreparation,
    fit_package: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    qualification: RegimeQualificationPackage,
    law: RegimeJointLawResult,
) -> RegimePredictionSeal:
    require_scientific_release(nomination, qualification, law)
    if (
        preparation.role != "prospective"
        or nomination.fit_package
        != ObjectIdentity.from_record(fit_package.package_id, fit_package)
        or qualification.fit_package != nomination.fit_package
        or qualification.nomination
        != ObjectIdentity.from_record(nomination.package_id, nomination)
    ):
        raise ValueError("D seal changed its released C law/route or assigned role")
    return _seal_predictions(
        preparation,
        fit_package,
        ObjectIdentity.from_record(qualification.package_id, qualification),
    )


def _seal_predictions(
    preparation: RegimeCausalPreparation,
    fit_package: RegimeFitPackage,
    model_package: ObjectIdentity,
) -> RegimePredictionSeal:
    contexts = causal_contexts(preparation)
    if tuple(row.name for row in contexts) != CONTEXTS:
        raise ValueError("nomination prediction lost its eight fixed contexts")
    info_inputs = {row.context: row for row in causal_inputs(preparation)}
    forecast_by_id: dict[str, np.ndarray] = {}
    support_by_id: dict[str, np.ndarray] = {}
    present_by_id: dict[str, np.ndarray] = {}

    def add(model_id: str, fit: object | None, values: tuple[tuple[float, ...] | None, ...]) -> None:
        if model_id in forecast_by_id:
            raise ValueError("prediction roster reused a model identity")
        forecast = np.zeros(8, dtype=np.float64)
        support = np.zeros(8, dtype=np.uint8)
        present = np.zeros(8, dtype=np.uint8)
        if fit is not None:
            from .frozen_models import FrozenFit

            if not isinstance(fit, FrozenFit):
                raise TypeError("nomination model is not its frozen fit record")
            model = fit.to_fit()
            for index, feature in enumerate(values):
                if feature is None:
                    continue
                array = np.asarray(feature, dtype=np.float64)[None, :]
                prediction = float(model.predict(array)[0])
                if not np.isfinite(prediction):
                    raise ValueError("frozen nomination model produced a nonfinite forecast")
                forecast[index] = prediction
                support[index] = int(_support(fit.support_lower, fit.support_upper, feature))
                present[index] = 1
        forecast_by_id[model_id] = forecast
        support_by_id[model_id] = support
        present_by_id[model_id] = present

    for item in fit_package.coefficients:
        add(
            f"coefficient.{item.model_id}".lower(),
            item.fitted,
            tuple(_base_feature(row, item.mask) for row in contexts),
        )
    for item in fit_package.absolute:
        add(
            f"absolute.{item.model_id}".lower(),
            item.fitted,
            tuple(_base_feature(row, item.mask) for row in contexts),
        )
    for info_item in fit_package.information:
        for arm, fit in zip(info_item.arms, info_item.fits, strict=True):
            values: list[tuple[float, ...] | None] = [None] * 8
            if info_item.pair_id == "S_H":
                for index in range(4):
                    values[index] = _base_feature(contexts[index], arm)
            else:
                context_name = "p_q" if arm.startswith("p") else "c_q"
                prepared = info_inputs.get(context_name)
                if prepared is not None:
                    values[CONTEXTS.index(context_name)] = (
                        prepared.full if arm.endswith("full") else prepared.masked
                    )
            add(f"information.{fit.fit_id}", fit, tuple(values))
    ids = tuple(sorted(forecast_by_id))
    if not ids:
        raise ValueError("nomination has no prefit prediction candidates")
    masses = np.zeros((8, 3), dtype=np.float64)
    projection_valid = np.zeros((8, 3), dtype=np.uint8)
    callbacks = np.full(8, -1, dtype=np.int32)
    for index, context in enumerate(contexts):
        if context.callback is not None and context.input_sha256 is not None:
            callbacks[index] = context.callback
        if context.projected_masses_kg is not None:
            masses[index] = context.projected_masses_kg
            projection_valid[index] = context.projection_valid
    return RegimePredictionSeal(
        preparation.root,
        preparation.role,
        preparation.seed,
        ObjectIdentity.from_record(f"{preparation.root}.causal-preparation", preparation),
        model_package,
        tuple(row.input_sha256 for row in contexts),
        ids,
        LocalArrayPayload.pack(
            {
                "forecast": np.stack(tuple(forecast_by_id[model_id] for model_id in ids)),
                "support": np.stack(tuple(support_by_id[model_id] for model_id in ids)),
                "present": np.stack(tuple(present_by_id[model_id] for model_id in ids)),
                "masses": masses,
                "projection_valid": projection_valid,
                "context_callback": callbacks,
            }
        ),
        "FROZEN_PREDICTIONS",
    )
