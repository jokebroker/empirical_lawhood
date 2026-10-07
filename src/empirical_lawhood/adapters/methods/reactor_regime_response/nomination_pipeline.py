"""Role-separated nomination from sixteen persisted root assay panels."""

from __future__ import annotations

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .causal_contexts import causal_contexts
from .config import ROOTS
from .information_models import InformationInput, causal_inputs, score_information_nomination
from .measured_panel import MeasuredContext, measured_contexts
from .model_records import RegimeFitPackage
from .model_selection import score_absolute_nomination, score_nomination_roster
from .nomination_records import NominationInformation, NominationRoute, NominationScore, RegimeNominationPackage
from .prediction_seal import seal_nomination_predictions
from .preparation_evidence import preparation_evidence
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from .route_nomination import NominationRoot, ROUTES, RouteCandidate, nominate_route, select_route


def _valid_word(assay: RegimeAssayPanel, route: str, word: int) -> bool:
    """A native feasibility flag cannot be inferred from point predictions."""
    slots = {(name, view, action): status for name, _, view, action, status in assay.slots}
    arrays = assay.arrays.unpack()
    for view in (0, 1):
        if slots[(route, view, word)] != "MEASURED":
            return False
        stem = f"{route}_v{view}_a{word}"
        if any(f"{stem}_{name}" not in arrays for name in ("grid", "exposure", "valid")):
            return False
        grid = arrays[f"{stem}_grid"]
        exposure = arrays[f"{stem}_exposure"]
        if (
            grid.shape != (21 if view else 11, 8)
            or exposure.shape != (20 if view else 10, 4)
            or arrays[f"{stem}_valid"].tolist() != [1]
            or not np.isfinite(grid).all()
            or not np.isfinite(exposure).all()
            or np.any(grid[:, 3] < 0)
            or np.any(grid[:, 3] > 287.3 + 1e-6)
            or np.any(exposure[:, 2] < 0)
        ):
            return False
    return True


def build_nomination_package(
    fit_package: RegimeFitPackage,
    rows: tuple[
        tuple[
            RegimeCausalPreparation,
            RegimePrivatePreparation,
            RegimePredictionSeal,
            RegimeAssayPanel,
        ], ...
    ],
) -> RegimeNominationPackage:
    expected = tuple(root for root, role, _, _ in ROOTS if role == "nomination")
    if tuple(causal.root for causal, _, _, _ in rows) != expected:
        raise ValueError("nomination lacks exactly sixteen assigned independent roots")
    all_contexts: list[MeasuredContext] = []
    all_inputs: list[InformationInput] = []
    causal_ids = []
    assay_ids = []
    by_route: dict[str, list[NominationRoot]] = {route: [] for route in ROUTES}
    for causal, private, seal, assay in rows:
        root = causal.root
        causal_id = ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
        seal_id = ObjectIdentity.from_record(f"{root}.prediction-seal", seal)
        if (
            (causal.role, private.role, seal.role, assay.role)
            != ("nomination",) * 4
            or (causal.seed, private.seed, seal.seed, assay.seed)
            != (causal.seed,) * 4
            or private.causal_preparation != causal_id
            or seal.causal_preparation != causal_id
            or seal.model_package
            != ObjectIdentity.from_record(fit_package.package_id, fit_package)
            or seal != seal_nomination_predictions(causal, fit_package)
            or assay.causal_preparation != causal_id
            or assay.private_preparation
            != ObjectIdentity.from_record(f"{root}.private-preparation", private)
            or assay.sealed_prediction != seal_id
        ):
            raise ValueError("nomination changed a fit-only seal or receipt identity")
        contexts = measured_contexts(causal, assay)
        causal_by_context = {item.name: item for item in causal_contexts(causal)}
        chart_by_context = {item.context: item for item in contexts}
        all_contexts.extend(contexts)
        all_inputs.extend(causal_inputs(causal))
        causal_ids.append(causal_id)
        assay_ids.append(ObjectIdentity.from_record(f"{root}.assay-panel", assay))
        for route in ROUTES:
            chart = chart_by_context[route]
            context = causal_by_context[route]
            preparation = preparation_evidence(causal, private, route)
            by_route[route].append(
                NominationRoot(
                    root,
                    route,
                    chart,
                    context.projected_masses_kg,
                    context.projection_valid,
                    tuple(_valid_word(assay, route, word) for word in range(3)),  # type: ignore[arg-type]
                    preparation.prefix_valid,
                    preparation.nominal_max_K,
                    preparation.refined_max_K,
                )
            )
    contexts_tuple = tuple(all_contexts)
    inputs_tuple = tuple(all_inputs)
    scored_coefficient = score_nomination_roster(
        tuple(item.to_candidate() for item in fit_package.coefficients), contexts_tuple
    )
    scored_absolute = score_absolute_nomination(
        tuple(item.to_candidate() for item in fit_package.absolute), contexts_tuple
    )
    information = []
    for pair_id in ("S_H", "S_I", "S_Q"):
        fitted = tuple(
            item.to_fitted() for item in fit_package.information if item.pair_id == pair_id
        )
        selected, reasons = score_information_nomination(
            pair_id, fitted, inputs_tuple, contexts_tuple
        )
        information.append(NominationInformation.from_selected(pair_id, selected, reasons))
    candidates: list[RouteCandidate | None] = []
    route_records = []
    for route in ROUTES:
        try:
            chosen = nominate_route(
                tuple(by_route[route]), scored_coefficient, scored_absolute
            )
        except ValueError as error:
            if str(error) not in (
                "route has no finite nomination candidate",
                "selected route has no fitted empirical absolute baseline",
            ):
                raise
            chosen = None
            route_records.append(NominationRoute.from_candidate(route, None, str(error)))
        else:
            route_records.append(NominationRoute.from_candidate(route, chosen))
        candidates.append(chosen)
    selected_route = select_route(tuple(candidates))  # type: ignore[arg-type]
    return RegimeNominationPackage(
        "reactor-regime-response-nomination-package",
        ObjectIdentity.from_record(fit_package.package_id, fit_package),
        tuple(causal_ids),
        tuple(assay_ids),
        tuple(NominationScore.from_candidate(item) for item in scored_coefficient),
        tuple(NominationScore.from_candidate(item) for item in scored_absolute),
        tuple(information),
        tuple(route_records),  # type: ignore[arg-type]
        None if selected_route is None else selected_route.route,
        () if selected_route is not None else ("NO_FINITE_NOMINATED_ROUTE",),
    )
