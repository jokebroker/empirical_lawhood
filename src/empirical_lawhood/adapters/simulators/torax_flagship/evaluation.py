"""Coarse response ordering with preparation-first numerical aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import (
    ToraxAction,
    ToraxMatchedPanel,
    ToraxPanelDisposition,
    ToraxPreparationEnsemble,
)


class ToraxPropertyStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    REVERSED = "REVERSED"
    ASSUMPTION_SENSITIVE = "ASSUMPTION_SENSITIVE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ToraxEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-evaluation-config'

    config_id: str
    materiality_threshold_ev: Decimal
    primary_view_id: str
    sensitivity_view_id: str
    development_only: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("primary_view_id", self.primary_view_id),
            ("sensitivity_view_id", self.sensitivity_view_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.materiality_threshold_ev,
            field_name="materiality_threshold_ev",
            minimum=Decimal(0),
        )
        if (
            self.primary_view_id != "torax-primary-view"
            or self.sensitivity_view_id != "torax-sensitivity-view"
        ):
            raise ValueError("TORAX evaluation changed the two qualified numerical views")
        if not self.development_only:
            raise ValueError("Phase 7 TORAX evaluation config is development-only")


@dataclass(frozen=True, slots=True)
class ToraxPanelEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-panel-evaluation'

    evaluation_id: str
    panel: ObjectIdentity
    preparation: ObjectIdentity
    theta_id: str
    view_id: str
    up_over_down: ToraxPropertyStatus
    down_hold_up_order: ToraxPropertyStatus
    up_minus_down_ev: Decimal | None
    materiality_threshold_ev: Decimal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("theta_id", self.theta_id),
            ("view_id", self.view_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.materiality_threshold_ev,
            field_name="materiality_threshold_ev",
            minimum=Decimal(0),
        )
        if self.up_minus_down_ev is not None:
            validate_decimal(self.up_minus_down_ev, field_name="up_minus_down_ev")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        unevaluable = self.up_over_down is ToraxPropertyStatus.UNEVALUABLE
        if unevaluable != (self.up_minus_down_ev is None):
            raise ValueError("TORAX panel property status differs from its response contrast")
        if unevaluable != bool(self.reason_codes):
            raise ValueError("TORAX panel reasons differ from evaluability")


@dataclass(frozen=True, slots=True)
class ToraxPreparationEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-preparation-evaluation'

    evaluation_id: str
    preparation: ObjectIdentity
    panel_evaluations: tuple[ToraxPanelEvaluation, ...]
    up_over_down: ToraxPropertyStatus
    down_hold_up_order: ToraxPropertyStatus
    independent_preparation_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        require_sorted_unique_ids(
            self.panel_evaluations,
            attribute="evaluation_id",
            field_name="panel_evaluations",
        )
        if self.independent_preparation_count != 0:
            raise ValueError("views/Theta members cannot inflate preparation count")
        if any(value.preparation != self.preparation for value in self.panel_evaluations):
            raise ValueError("view aggregation crosses preparation identities")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class ToraxAssumptionEnsembleEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-assumption-ensemble-evaluation'

    evaluation_id: str
    ensemble: ObjectIdentity
    member_evaluations: tuple[ToraxPreparationEvaluation, ...]
    up_over_down: ToraxPropertyStatus
    down_hold_up_order: ToraxPropertyStatus
    independent_preparation_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        require_sorted_unique_ids(
            self.member_evaluations,
            attribute="evaluation_id",
            field_name="member_evaluations",
        )
        if not self.member_evaluations:
            raise ValueError("Theta aggregation requires member evaluations")
        if self.independent_preparation_count != 1:
            raise ValueError("one MAST-derived ensemble is one numerical preparation")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def evaluate_torax_panel(
    panel: ToraxMatchedPanel,
    *,
    materiality_threshold_ev: Decimal,
) -> ToraxPanelEvaluation:
    validate_decimal(
        materiality_threshold_ev,
        field_name="materiality_threshold_ev",
        minimum=Decimal(0),
    )
    panel_identity = ObjectIdentity.from_record(panel.panel_id, panel)
    if panel.disposition is not ToraxPanelDisposition.COMPLETE:
        return ToraxPanelEvaluation(
            evaluation_id=f"evaluation.{panel.panel_id}",
            panel=panel_identity,
            preparation=panel.preparation,
            theta_id=panel.theta_id,
            view_id=panel.view.view_id,
            up_over_down=ToraxPropertyStatus.UNEVALUABLE,
            down_hold_up_order=ToraxPropertyStatus.UNEVALUABLE,
            up_minus_down_ev=None,
            materiality_threshold_ev=materiality_threshold_ev,
            reason_codes=("MATCHED_PANEL_INCOMPLETE",),
        )
    values = {value.action.action: value.endpoint_delta_te_core_ev for value in panel.rollouts}
    if any(value is None for value in values.values()):  # constructor should prevent this
        raise AssertionError("complete panel lacks endpoint values")
    down = values[ToraxAction.DOWN]
    hold = values[ToraxAction.HOLD]
    up = values[ToraxAction.UP]
    assert down is not None and hold is not None and up is not None
    contrast = up - down
    up_status = (
        ToraxPropertyStatus.SUPPORTED
        if contrast > materiality_threshold_ev
        else ToraxPropertyStatus.REVERSED
    )
    order_status = (
        ToraxPropertyStatus.SUPPORTED
        if down < hold < up and contrast > materiality_threshold_ev
        else ToraxPropertyStatus.REVERSED
    )
    return ToraxPanelEvaluation(
        evaluation_id=f"evaluation.{panel.panel_id}",
        panel=panel_identity,
        preparation=panel.preparation,
        theta_id=panel.theta_id,
        view_id=panel.view.view_id,
        up_over_down=up_status,
        down_hold_up_order=order_status,
        up_minus_down_ev=contrast,
        materiality_threshold_ev=materiality_threshold_ev,
        reason_codes=(),
    )


def evaluate_torax_panel_from_config(
    panel: ToraxMatchedPanel,
    *,
    config: ToraxEvaluationConfig,
) -> ToraxPanelEvaluation:
    if panel.view.view_id not in {config.primary_view_id, config.sensitivity_view_id}:
        raise ValueError("TORAX panel uses an undeclared numerical view")
    return evaluate_torax_panel(
        panel,
        materiality_threshold_ev=config.materiality_threshold_ev,
    )


def aggregate_torax_views(
    evaluations: tuple[ToraxPanelEvaluation, ...],
) -> ToraxPreparationEvaluation:
    require_sorted_unique_ids(evaluations, attribute="evaluation_id", field_name="evaluations")
    if len(evaluations) != 2:
        raise ValueError("TORAX preparation aggregation requires primary and sensitivity views")
    if len({value.preparation for value in evaluations}) != 1:
        raise ValueError("TORAX view evaluations cross preparations")
    roles = {value.view_id for value in evaluations}
    if roles != {"torax-primary-view", "torax-sensitivity-view"}:
        raise ValueError("TORAX view set differs from the frozen pair")
    preparation = evaluations[0].preparation
    up_status = _aggregate_property(tuple(value.up_over_down for value in evaluations))
    order_status = _aggregate_property(tuple(value.down_hold_up_order for value in evaluations))
    reasons = (
        ("NUMERICAL_VIEW_UNEVALUABLE",)
        if ToraxPropertyStatus.UNEVALUABLE in {up_status, order_status}
        else (
            ("NUMERICAL_VIEW_SENSITIVE",)
            if ToraxPropertyStatus.ASSUMPTION_SENSITIVE in {up_status, order_status}
            else ()
        )
    )
    return ToraxPreparationEvaluation(
        evaluation_id=f"preparation-evaluation.{preparation.object_id}",
        preparation=preparation,
        panel_evaluations=evaluations,
        up_over_down=up_status,
        down_hold_up_order=order_status,
        independent_preparation_count=0,
        reason_codes=reasons,
    )


def aggregate_torax_theta(
    ensemble: ToraxPreparationEnsemble,
    evaluations: tuple[ToraxPreparationEvaluation, ...],
) -> ToraxAssumptionEnsembleEvaluation:
    require_sorted_unique_ids(evaluations, attribute="evaluation_id", field_name="evaluations")
    expected = {value.preparation_id for value in ensemble.preparations}
    actual = {value.preparation.object_id for value in evaluations}
    if actual != expected:
        raise ValueError("Theta evaluations do not cover the exact preparation ensemble")
    up_status = _aggregate_property(tuple(value.up_over_down for value in evaluations))
    order_status = _aggregate_property(tuple(value.down_hold_up_order for value in evaluations))
    reasons = (
        ("THETA_MEMBER_UNEVALUABLE",)
        if ToraxPropertyStatus.UNEVALUABLE in {up_status, order_status}
        else (
            ("THETA_ASSUMPTION_SENSITIVE",)
            if ToraxPropertyStatus.ASSUMPTION_SENSITIVE in {up_status, order_status}
            else ()
        )
    )
    return ToraxAssumptionEnsembleEvaluation(
        evaluation_id=f"ensemble-evaluation.{ensemble.ensemble_id}",
        ensemble=ObjectIdentity.from_record(ensemble.ensemble_id, ensemble),
        member_evaluations=evaluations,
        up_over_down=up_status,
        down_hold_up_order=order_status,
        independent_preparation_count=1,
        reason_codes=reasons,
    )


def _aggregate_property(statuses: tuple[ToraxPropertyStatus, ...]) -> ToraxPropertyStatus:
    if ToraxPropertyStatus.UNEVALUABLE in statuses:
        return ToraxPropertyStatus.UNEVALUABLE
    if len(set(statuses)) > 1:
        return ToraxPropertyStatus.ASSUMPTION_SENSITIVE
    return statuses[0]


__all__ = [
    "ToraxEvaluationConfig",
    "ToraxPanelEvaluation",
    "ToraxAssumptionEnsembleEvaluation",
    "ToraxPreparationEvaluation",
    "ToraxPropertyStatus",
    "aggregate_torax_views",
    "aggregate_torax_theta",
    "evaluate_torax_panel",
    "evaluate_torax_panel_from_config",
]
