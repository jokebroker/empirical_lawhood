"""Fit-root-only scientific reduction after sealed native assay custody."""

from __future__ import annotations

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .config import ROOTS
from .information_models import InformationInput, FittedInformationSpec, causal_inputs, fit_information_roster
from .measured_panel import MeasuredContext, measured_contexts
from .model_records import RegimeFitPackage
from .model_selection import fit_absolute_candidates, fit_candidate_roster
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal


def build_fit_package(
    rows: tuple[
        tuple[RegimeCausalPreparation, RegimePredictionSeal, RegimeAssayPanel], ...
    ],
) -> RegimeFitPackage:
    """No nomination/calibration/qualification result can reach any fit."""
    expected = tuple(root for root, role, _, _ in ROOTS if role == "fit")
    if tuple(causal.root for causal, _, _ in rows) != expected:
        raise ValueError("fit package lacks its 32 assigned independent roots")
    all_contexts: list[MeasuredContext] = []
    all_inputs: list[InformationInput] = []
    causal_ids = []
    assay_ids = []
    for causal, seal, assay in rows:
        root = causal.root
        if (
            (seal.root, assay.root, seal.role, assay.role)
            != (root, root, "fit", "fit")
            or seal.causal_preparation
            != ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
            or assay.sealed_prediction
            != ObjectIdentity.from_record(f"{root}.prediction-seal", seal)
            or assay.causal_preparation != seal.causal_preparation
        ):
            raise ValueError("fit root changed its sealed causal/assay identity")
        all_contexts.extend(measured_contexts(causal, assay))
        all_inputs.extend(causal_inputs(causal))
        causal_ids.append(ObjectIdentity.from_record(f"{root}.causal-preparation", causal))
        assay_ids.append(ObjectIdentity.from_record(f"{root}.assay-panel", assay))
    contexts = tuple(all_contexts)
    inputs = tuple(all_inputs)
    coefficient = fit_candidate_roster(contexts)
    absolute = fit_absolute_candidates(contexts)
    information: list[FittedInformationSpec] = []
    failures: list[str] = []
    for pair_id in ("S_H", "S_I", "S_Q"):
        fitted, reasons = fit_information_roster(pair_id, inputs, contexts)
        information.extend(fitted)
        failures.extend(f"{pair_id}:{reason}" for reason in reasons)
    return RegimeFitPackage.build(
        tuple(causal_ids),
        tuple(assay_ids),
        coefficient,
        absolute,
        tuple(information),
        tuple(failures),
    )
