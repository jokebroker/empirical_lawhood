"""Saved C contrast losses from pre-assay forecasts and fresh root charts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .confirmation_pipeline import ConfirmationRow
from .information_math import RootComparisonLosses
from .measured_panel import CONTEXTS, MeasuredContext, measured_contexts
from .model_records import RegimeFitPackage
from .model_selection import Candidate, nominate
from .nomination_records import NominationScore, RegimeNominationPackage

BASELINE = ("early", "middle", "late", "prepared_t0")


def _scored_candidates(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
) -> tuple[Candidate, ...]:
    scores = {item.model_id: item for item in nomination.coefficient_scores}
    if set(scores) != {item.model_id for item in fit.coefficients}:
        raise ValueError("fresh R comparison lost a nominated K/S/L score")
    result = []
    for item in fit.coefficients:
        score: NominationScore = scores[item.model_id]
        candidate = item.to_candidate()
        result.append(
            replace(
                candidate,
                nomination_root_losses_K2={root: float(value) for root, value in score.root_losses_K2},
                nomination_mean_loss_K2=(
                    None if score.mean_loss_K2 is None else float(score.mean_loss_K2)
                ),
                nomination_leaf_contacts=score.leaf_contacts,
                reasons=score.reasons,
            )
        )
    return tuple(result)


def _nominated_model_id(candidates: tuple[Candidate, ...]) -> str | None:
    try:
        return nominate(candidates).model_id
    except ValueError as error:
        if str(error) != "no finite nominated coefficient model":
            raise
        return None


def _information_ids(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    pair_id: str,
) -> dict[str, str]:
    choice = next(item for item in nomination.information if item.pair_id == pair_id)
    if choice.spec_id is None:
        return {}
    matching = tuple(
        item for item in fit.information
        if item.pair_id == pair_id and item.to_fitted().spec.spec_id == choice.spec_id
    )
    if len(matching) != 1:
        raise ValueError("fresh information comparison substituted its fitted smooth pair")
    item = matching[0]
    return {
        arm: f"information.{model.fit_id}"
        for arm, model in zip(item.arms, item.fits, strict=True)
    }


def root_comparison_losses(
    row: ConfirmationRow,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
) -> RootComparisonLosses:
    causal, _, seal, assay = row
    if causal.role != "qualification":
        raise ValueError("five fresh contrasts require a qualification root")
    charts = {item.context: item for item in measured_contexts(causal, assay)}
    candidates = _scored_candidates(fit, nomination)
    local = _nominated_model_id(
        tuple(item for item in candidates if item.mask == "history" and item.family == "local")
    )
    rival = _nominated_model_id(
        tuple(item for item in candidates if item.mask == "history" and item.family != "local")
    )
    info_h = _information_ids(fit, nomination, "S_H")
    info_i = _information_ids(fit, nomination, "S_I")
    info_q = _information_ids(fit, nomination, "S_Q")
    arrays = seal.predictions.unpack()
    ids = {model_id: index for index, model_id in enumerate(seal.candidate_ids)}

    def prediction(model_id: str | None, context: str) -> float | None:
        if model_id is None or model_id not in ids:
            return None
        index = ids[model_id]
        column = CONTEXTS.index(context)
        if not arrays["present"][index, column]:
            return None
        return float(arrays["forecast"][index, column])

    def loss(model_id: str | None, input_context: str, target_context: str) -> float | None:
        chart: MeasuredContext = charts[target_context]
        value = prediction(model_id, input_context)
        if (
            value is None
            or chart.nominal_mass_kg is None
            or chart.nominal_cooling_K is None
        ):
            return None
        masses = np.asarray(chart.nominal_mass_kg[1:], dtype=np.float64)
        cooling = np.asarray(chart.nominal_cooling_K[1:], dtype=np.float64)
        return float(np.mean((value * masses - cooling) ** 2))

    def baseline_loss(model_id: str | None) -> float | None:
        values = tuple(loss(model_id, context, context) for context in BASELINE)
        return (
            None if any(value is None for value in values)
            else float(np.mean(tuple(float(value) for value in values if value is not None)))
        )

    def source_contact(names: tuple[str, ...]) -> bool:
        return all(
            charts[name].callback is not None
            and charts[name].nominal_mass_kg is not None
            and charts[name].nominal_cooling_K is not None
            for name in names
        )

    def source_valid(names: tuple[str, ...]) -> bool:
        return all(charts[name].valid for name in names)

    baseline_ok = source_contact(BASELINE)
    probe_preparation_common_anchor_contact = source_contact(("prepared_t0", "p_q"))
    common_anchor_preparation_pair_contact = source_contact(("prepared_t0", "c_q", "p_q"))
    p_q_ok = source_contact(("p_q",))
    source_flags = (
        source_valid(BASELINE),
        source_valid(BASELINE),
        source_valid(("prepared_t0", "c_q", "p_q")),
        source_valid(("prepared_t0", "c_q", "p_q")),
        source_valid(("p_q",)),
    )
    return RootComparisonLosses(
        causal.root,
        baseline_ok,
        common_anchor_preparation_pair_contact,
        p_q_ok,
        baseline_loss(None if rival is None else f"coefficient.{rival}".lower()),
        baseline_loss(None if local is None else f"coefficient.{local}".lower()),
        baseline_loss(info_h.get("current")),
        baseline_loss(info_h.get("history")),
        loss(info_i.get("c_masked"), "c_q", "prepared_t0"),
        loss(info_i.get("c_full"), "c_q", "prepared_t0"),
        loss(info_i.get("p_masked"), "p_q", "prepared_t0"),
        loss(info_i.get("p_full"), "p_q", "prepared_t0"),
        loss(info_q.get("p_masked"), "p_q", "p_q"),
        loss(info_q.get("p_full"), "p_q", "p_q"),
        all(source_flags),
        True,
        probe_preparation_common_anchor_contact,
        source_flags,
    )
