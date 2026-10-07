"""Finite, target-neutral selective dependence response comparator and separation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections import Counter
from typing import Callable, ClassVar, Mapping

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS, SelectiveDependenceResponseCaseState, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseExchangeForecast


class SelectiveDependenceResponseComparatorKind(StrEnum):
    ACTION_ONLY = "ACTION_ONLY"
    DEFAULT_SAFE_HOLD = "DEFAULT_SAFE_HOLD"
    DENOMINATOR_BLIND = "DENOMINATOR_BLIND"
    DEVELOPMENT_MAJORITY = "DEVELOPMENT_MAJORITY"
    HISTORY_BLIND = "HISTORY_BLIND"
    RECEIVER_BLIND = "RECEIVER_BLIND"
    RESPONSE_ONLY_ADMISSION = "RESPONSE_ONLY_ADMISSION"
    SATURATED_FINITE_LOOKUP = "SATURATED_FINITE_LOOKUP"
    TARGET_NATIVE_MECHANISTIC = "TARGET_NATIVE_MECHANISTIC"
    TAU_BLIND = "TAU_BLIND"


SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS = tuple(sorted(SelectiveDependenceResponseComparatorKind, key=lambda value: value.value))


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCategoricalPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-categorical-prediction'

    prediction_id: str
    cell_id: str
    response_state: SelectiveDependenceResponseCaseState
    fibre_admitted: bool | None
    disposition: SelectiveDependenceResponseDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        if (self.fibre_admitted is None) != (self.disposition is SelectiveDependenceResponseDisposition.UNEVALUABLE):
            raise ValueError("categorical fibre and disposition evaluability differ")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseComparatorExchangePrediction(CanonicalRecord):
    """One comparator answer to a frozen exchange; ``None`` is abstention."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-exchange-prediction'

    prediction_id: str
    exchange_id: str
    expectation: SelectiveDependenceResponseExchangeExpectation | None

    def __post_init__(self) -> None:
        for name in ("prediction_id", "exchange_id"):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseComparatorEncoding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-encoding'

    encoding_id: str
    target_id: str
    kind: SelectiveDependenceResponseComparatorKind
    dependency_role_ids: tuple[str, ...]
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...]
    exchange_predictions: tuple[SelectiveDependenceResponseComparatorExchangePrediction, ...]
    out_of_cell_rule: str
    development_unit_ids_sha256: str
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("encoding_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        expected_roles = tuple(
            role for role in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS if role in self.dependency_role_ids
        )
        if self.dependency_role_ids != expected_roles:
            raise ValueError("comparator dependencies are unknown or out of canonical order")
        require_sorted_unique_ids(
            self.predictions,
            attribute="prediction_id",
            field_name="predictions",
        )
        if len({value.cell_id for value in self.predictions}) != len(self.predictions):
            raise ValueError("comparator contains duplicate cell predictions")
        require_sorted_unique_ids(
            self.exchange_predictions,
            attribute="prediction_id",
            field_name="exchange_predictions",
        )
        if len({value.exchange_id for value in self.exchange_predictions}) != len(
            self.exchange_predictions
        ):
            raise ValueError("comparator contains duplicate exchange predictions")
        validate_nonempty(self.out_of_cell_rule, field_name="out_of_cell_rule")
        validate_sha256(
            self.development_unit_ids_sha256,
            field_name="development_unit_ids_sha256",
        )
        if self.evaluation_outcome_count:
            raise ValueError("comparator fitting cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("comparator encoding must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseComparatorSeparation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-separation'

    separation_id: str
    comparator_id: str
    differing_cell_ids: tuple[str, ...]
    relevant_cell_ids: tuple[str, ...]
    differing_exchange_ids: tuple[str, ...]
    relevant_exchange_ids: tuple[str, ...]
    separated: bool

    def __post_init__(self) -> None:
        for name in ("separation_id", "comparator_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.differing_cell_ids, field_name="differing_cell_ids")
        require_sorted_unique_strings(self.relevant_cell_ids, field_name="relevant_cell_ids")
        if not set(self.differing_cell_ids).issubset(self.relevant_cell_ids):
            raise ValueError("a differing cell lies outside the comparator challenge")
        for name in ("differing_exchange_ids", "relevant_exchange_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not set(self.differing_exchange_ids).issubset(self.relevant_exchange_ids):
            raise ValueError("a differing exchange lies outside the comparator challenge")
        if self.separated != bool(self.differing_cell_ids or self.differing_exchange_ids):
            raise ValueError("comparator separation flag is not data-derived")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseForecastSeparationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-forecast-separation-report'

    report_id: str
    target_id: str
    separations: tuple[SelectiveDependenceResponseComparatorSeparation, ...]
    exact_comparator_kinds: tuple[SelectiveDependenceResponseComparatorKind, ...]
    all_claim_relevant_comparators_separated: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("report_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.separations,
            attribute="separation_id",
            field_name="separations",
        )
        if self.exact_comparator_kinds != SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS:
            raise ValueError("forecast separation does not cover the exact comparator family")
        if len(self.separations) != len(SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS):
            raise ValueError("forecast separation comparator count differs")
        expected = all(value.separated for value in self.separations)
        if self.all_claim_relevant_comparators_separated != expected:
            raise ValueError("separation aggregate is not data-derived")
        if self.evaluation_outcome_count:
            raise ValueError("pre-issue separation cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("separation report must remain development visible")


def compare_forecast_to_comparators(
    *,
    target_id: str,
    forecast: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    exchange_forecast: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    comparators: tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    relevant_cells_by_kind: Mapping[SelectiveDependenceResponseComparatorKind, tuple[str, ...]],
    relevant_exchanges_by_kind: Mapping[SelectiveDependenceResponseComparatorKind, tuple[str, ...]],
) -> SelectiveDependenceResponseForecastSeparationReport:
    """Require exact prediction disagreement on each comparator's named challenge."""

    if tuple(sorted((value.kind for value in comparators), key=lambda value: value.value)) != (
        SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS
    ):
        raise ValueError("exactly one encoding per comparator kind is required")
    if set(relevant_cells_by_kind) != set(SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS):
        raise ValueError("relevant-cell mapping does not cover the comparator family")
    if set(relevant_exchanges_by_kind) != set(SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS):
        raise ValueError("relevant-exchange mapping does not cover the comparator family")
    primary = {
        value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
        for value in forecast
    }
    if len(primary) != len(forecast):
        raise ValueError("forecast contains duplicate cell predictions")
    primary_exchanges = {value.exchange_id: value.expectation for value in exchange_forecast}
    if len(primary_exchanges) != len(exchange_forecast):
        raise ValueError("forecast contains duplicate exchange predictions")
    separations = []
    for comparator in comparators:
        alternate = {
            value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
            for value in comparator.predictions
        }
        relevant = tuple(sorted(relevant_cells_by_kind[comparator.kind]))
        if any(cell_id not in primary or cell_id not in alternate for cell_id in relevant):
            raise ValueError("a named comparison cell lacks a prediction")
        differing = tuple(cell_id for cell_id in relevant if primary[cell_id] != alternate[cell_id])
        alternate_exchanges = {
            value.exchange_id: value.expectation for value in comparator.exchange_predictions
        }
        relevant_exchanges = tuple(sorted(relevant_exchanges_by_kind[comparator.kind]))
        if any(
            exchange_id not in primary_exchanges or exchange_id not in alternate_exchanges
            for exchange_id in relevant_exchanges
        ):
            raise ValueError("a named comparison exchange lacks a prediction")
        differing_exchanges = tuple(
            exchange_id
            for exchange_id in relevant_exchanges
            if primary_exchanges[exchange_id] != alternate_exchanges[exchange_id]
        )
        separations.append(
            SelectiveDependenceResponseComparatorSeparation(
                separation_id=(
                    f"separation.{target_id}.{comparator.kind.value.lower().replace('_', '-')}"
                ),
                comparator_id=comparator.encoding_id,
                differing_cell_ids=differing,
                relevant_cell_ids=relevant,
                differing_exchange_ids=differing_exchanges,
                relevant_exchange_ids=relevant_exchanges,
                separated=bool(differing or differing_exchanges),
            )
        )
    ordered = tuple(sorted(separations, key=lambda value: value.separation_id))
    return SelectiveDependenceResponseForecastSeparationReport(
        report_id=f"report.{target_id}.forecast-separation",
        target_id=target_id,
        separations=ordered,
        exact_comparator_kinds=SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS,
        all_claim_relevant_comparators_separated=all(value.separated for value in ordered),
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def comparator_relevant_cells(
    *,
    forecast: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    hold_action_id: str,
) -> dict[SelectiveDependenceResponseComparatorKind, tuple[str, ...]]:
    """Name each comparator's scientific challenge without consulting it.

    The mapping is derived from the frozen primary forecast and the named
    exchange roster, never from whichever cells happen to differ from a fitted
    comparator.  An empty roster is retained as an unseparated challenge and
    therefore prevents evaluation issue.
    """

    values = {
        value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
        for value in forecast
    }
    if len(values) != len(forecast) or not values:
        raise ValueError("comparator relevance requires unique nonempty forecast cells")
    roles = {cell_id: _cell_roles(cell_id) for cell_id in values}
    if any(
        exchange.left_cell_id not in values or exchange.right_cell_id not in values
        for exchange in exchange_forecasts
    ):
        raise ValueError("a named exchange lies outside the primary forecast")

    def varying_cells(omitted_index: int) -> tuple[str, ...]:
        grouped: dict[tuple[str, ...], list[str]] = {}
        for cell_id, coordinates in roles.items():
            key = tuple(
                coordinate for index, coordinate in enumerate(coordinates) if index != omitted_index
            )
            grouped.setdefault(key, []).append(cell_id)
        return tuple(
            sorted(
                cell_id
                for cell_ids in grouped.values()
                if len({values[cell_id] for cell_id in cell_ids}) > 1
                for cell_id in cell_ids
            )
        )

    action_only_groups: dict[str, list[str]] = {}
    for cell_id, coordinates in roles.items():
        action_only_groups.setdefault(coordinates[2], []).append(cell_id)
    action_only = tuple(
        sorted(
            cell_id
            for cell_ids in action_only_groups.values()
            if len({values[cell_id] for cell_id in cell_ids}) > 1
            for cell_id in cell_ids
        )
    )
    exchange_cells_by_role = {
        role_id: tuple(
            sorted(
                {
                    cell_id
                    for exchange in exchange_forecasts
                    if exchange.coordinate_role_id == role_id
                    for cell_id in (exchange.left_cell_id, exchange.right_cell_id)
                }
            )
        )
        for role_id in ("D", "H", "tau")
    }
    role_blind = {
        "D": tuple(
            cell_id for cell_id in varying_cells(0) if cell_id in exchange_cells_by_role["D"]
        ),
        "H": tuple(
            cell_id for cell_id in varying_cells(1) if cell_id in exchange_cells_by_role["H"]
        ),
        "tau": tuple(
            cell_id for cell_id in varying_cells(3) if cell_id in exchange_cells_by_role["tau"]
        ),
    }
    response_only = tuple(
        sorted(
            cell_id
            for cell_id, (state, fibre_admitted, _) in values.items()
            if roles[cell_id][2] != hold_action_id
            and state
            not in {
                SelectiveDependenceResponseCaseState.NEUTRAL,
                SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT,
                SelectiveDependenceResponseCaseState.UNEVALUABLE,
            }
            and fibre_admitted is False
        )
    )
    default_hold = tuple(
        sorted(
            cell_id
            for cell_id, (_, fibre_admitted, _) in values.items()
            if roles[cell_id][2] == hold_action_id and fibre_admitted is False
        )
    )
    all_cells = tuple(sorted(values))
    return {
        SelectiveDependenceResponseComparatorKind.ACTION_ONLY: action_only,
        SelectiveDependenceResponseComparatorKind.DEFAULT_SAFE_HOLD: default_hold,
        SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND: role_blind["D"],
        SelectiveDependenceResponseComparatorKind.DEVELOPMENT_MAJORITY: all_cells,
        SelectiveDependenceResponseComparatorKind.HISTORY_BLIND: role_blind["H"],
        SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND: varying_cells(4),
        SelectiveDependenceResponseComparatorKind.RESPONSE_ONLY_ADMISSION: response_only,
        SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP: all_cells,
        SelectiveDependenceResponseComparatorKind.TARGET_NATIVE_MECHANISTIC: all_cells,
        SelectiveDependenceResponseComparatorKind.TAU_BLIND: role_blind["tau"],
    }


def comparator_relevant_exchanges(
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...],
) -> dict[SelectiveDependenceResponseComparatorKind, tuple[str, ...]]:
    """Name exchange questions made unavailable by each reduced comparator."""

    by_role = {
        role_id: tuple(
            sorted(
                value.exchange_id
                for value in exchange_forecasts
                if value.coordinate_role_id == role_id
                and value.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            )
        )
        for role_id in ("D", "H", "R", "tau")
    }
    all_exchanges = tuple(sorted(value.exchange_id for value in exchange_forecasts))
    non_action_active = tuple(
        sorted(
            value.exchange_id
            for value in exchange_forecasts
            if value.coordinate_role_id != "A"
            and value.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
        )
    )
    empty: tuple[str, ...] = ()
    return {
        SelectiveDependenceResponseComparatorKind.ACTION_ONLY: non_action_active,
        SelectiveDependenceResponseComparatorKind.DEFAULT_SAFE_HOLD: empty,
        SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND: by_role["D"],
        SelectiveDependenceResponseComparatorKind.DEVELOPMENT_MAJORITY: all_exchanges,
        SelectiveDependenceResponseComparatorKind.HISTORY_BLIND: by_role["H"],
        SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND: by_role["R"],
        SelectiveDependenceResponseComparatorKind.RESPONSE_ONLY_ADMISSION: empty,
        SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP: all_exchanges,
        SelectiveDependenceResponseComparatorKind.TARGET_NATIVE_MECHANISTIC: empty,
        SelectiveDependenceResponseComparatorKind.TAU_BLIND: by_role["tau"],
    }


def _cell_roles(cell_id: str) -> tuple[str, str, str, str, str]:
    parts = cell_id.split(".")
    if len(parts) != 6 or parts[0] != "cell" or not parts[5].startswith("receiver-"):
        raise ValueError("categorical cell does not expose exact D/H/A/tau/R coordinates")
    return parts[1], parts[2], parts[3], parts[4], parts[5].removeprefix("receiver-")


def _modal(
    values: list[tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]],
) -> tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]:
    counts = Counter(values)
    return sorted(
        counts,
        key=lambda value: (
            -counts[value],
            value[0].value,
            "none" if value[1] is None else str(value[1]),
            value[2].value,
        ),
    )[0]


def fit_comparator_family(
    *,
    target_id: str,
    forecast: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    exchange_forecast: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    development_unit_ids_sha256: str,
    hold_action_id: str,
    mechanistic_prediction: Callable[
        [str, str, str, str, str],
        tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition],
    ],
) -> tuple[SelectiveDependenceResponseComparatorEncoding, ...]:
    """Fit the exact ten comparators without evaluation-outcome access."""

    values = {
        item.cell_id: (item.response_state, item.fibre_admitted, item.disposition)
        for item in forecast
    }
    if len(values) != len(forecast):
        raise ValueError("forecast cells repeat")
    roles = {cell_id: _cell_roles(cell_id) for cell_id in values}

    def grouped(
        indices: tuple[int, ...],
    ) -> dict[tuple[str, ...], tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]]:
        buckets: dict[
            tuple[str, ...], list[tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]]
        ] = {}
        for cell_id, coordinates in roles.items():
            key = tuple(coordinates[index] for index in indices)
            buckets.setdefault(key, []).append(values[cell_id])
        return {key: _modal(items) for key, items in buckets.items()}

    all_majority = _modal(list(values.values()))
    grouped_predictions = {
        SelectiveDependenceResponseComparatorKind.ACTION_ONLY: (grouped((2,)), (2,)),
        SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND: (grouped((1, 2, 3, 4)), (1, 2, 3, 4)),
        SelectiveDependenceResponseComparatorKind.HISTORY_BLIND: (grouped((0, 2, 3, 4)), (0, 2, 3, 4)),
        SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND: (grouped((0, 1, 2, 3)), (0, 1, 2, 3)),
        SelectiveDependenceResponseComparatorKind.TAU_BLIND: (grouped((0, 1, 2, 4)), (0, 1, 2, 4)),
    }
    dependency_roles = {
        SelectiveDependenceResponseComparatorKind.ACTION_ONLY: ("A",),
        SelectiveDependenceResponseComparatorKind.DEFAULT_SAFE_HOLD: SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND: ("A", "H", "R", "tau"),
        SelectiveDependenceResponseComparatorKind.DEVELOPMENT_MAJORITY: (),
        SelectiveDependenceResponseComparatorKind.HISTORY_BLIND: ("A", "D", "R", "tau"),
        SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND: ("A", "D", "H", "tau"),
        SelectiveDependenceResponseComparatorKind.RESPONSE_ONLY_ADMISSION: SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP: SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        SelectiveDependenceResponseComparatorKind.TARGET_NATIVE_MECHANISTIC: SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        SelectiveDependenceResponseComparatorKind.TAU_BLIND: ("A", "D", "H", "R"),
    }
    encodings = []
    for kind in SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS:
        predictions = []
        for cell_id, coordinates in roles.items():
            state, fibre_admitted, disposition = values[cell_id]
            if kind in grouped_predictions:
                lookup, indices = grouped_predictions[kind]
                state, fibre_admitted, disposition = lookup[
                    tuple(coordinates[index] for index in indices)
                ]
            elif kind is SelectiveDependenceResponseComparatorKind.DEVELOPMENT_MAJORITY:
                state, fibre_admitted, disposition = all_majority
            elif kind is SelectiveDependenceResponseComparatorKind.DEFAULT_SAFE_HOLD:
                if coordinates[2] == hold_action_id:
                    fibre_admitted = True
                    disposition = SelectiveDependenceResponseDisposition.HOLD_ONLY
            elif kind is SelectiveDependenceResponseComparatorKind.RESPONSE_ONLY_ADMISSION:
                if coordinates[2] == hold_action_id:
                    fibre_admitted = True
                    disposition = SelectiveDependenceResponseDisposition.HOLD_ONLY
                elif state not in {
                    SelectiveDependenceResponseCaseState.NEUTRAL,
                    SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT,
                    SelectiveDependenceResponseCaseState.UNEVALUABLE,
                }:
                    fibre_admitted = True
                    disposition = SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
            elif kind is SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP:
                state, fibre_admitted, disposition = (
                    SelectiveDependenceResponseCaseState.UNEVALUABLE,
                    None,
                    SelectiveDependenceResponseDisposition.UNEVALUABLE,
                )
            elif kind is SelectiveDependenceResponseComparatorKind.TARGET_NATIVE_MECHANISTIC:
                state, fibre_admitted, disposition = mechanistic_prediction(*coordinates)
            predictions.append(
                SelectiveDependenceResponseCategoricalPrediction(
                    prediction_id=(
                        f"prediction.{target_id}.{kind.value.lower().replace('_', '-')}.{cell_id}"
                    ),
                    cell_id=cell_id,
                    response_state=state,
                    fibre_admitted=fibre_admitted,
                    disposition=disposition,
                )
            )
        ordered_predictions = tuple(sorted(predictions, key=lambda value: value.prediction_id))
        exchange_predictions = []
        for exchange in exchange_forecast:
            expectation: SelectiveDependenceResponseExchangeExpectation | None = exchange.expectation
            if kind in {
                SelectiveDependenceResponseComparatorKind.DEVELOPMENT_MAJORITY,
                SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP,
            }:
                expectation = None
            elif (
                kind is SelectiveDependenceResponseComparatorKind.ACTION_ONLY
                and exchange.coordinate_role_id != "A"
                and exchange.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            ):
                expectation = SelectiveDependenceResponseExchangeExpectation.INVARIANT
            elif (
                kind is SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND
                and exchange.coordinate_role_id == "D"
                and exchange.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            ):
                expectation = SelectiveDependenceResponseExchangeExpectation.INVARIANT
            elif (
                kind is SelectiveDependenceResponseComparatorKind.HISTORY_BLIND
                and exchange.coordinate_role_id == "H"
                and exchange.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            ):
                expectation = SelectiveDependenceResponseExchangeExpectation.INVARIANT
            elif (
                kind is SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND
                and exchange.coordinate_role_id == "R"
                and exchange.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            ):
                expectation = SelectiveDependenceResponseExchangeExpectation.INVARIANT
            elif (
                kind is SelectiveDependenceResponseComparatorKind.TAU_BLIND
                and exchange.coordinate_role_id == "tau"
                and exchange.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            ):
                expectation = SelectiveDependenceResponseExchangeExpectation.INVARIANT
            exchange_predictions.append(
                SelectiveDependenceResponseComparatorExchangePrediction(
                    prediction_id=(
                        f"prediction.{target_id}.{kind.value.lower().replace('_', '-')}."
                        f"{exchange.exchange_id}"
                    ),
                    exchange_id=exchange.exchange_id,
                    expectation=expectation,
                )
            )
        encodings.append(
            SelectiveDependenceResponseComparatorEncoding(
                encoding_id=f"encoding.{target_id}.{kind.value.lower().replace('_', '-')}",
                target_id=target_id,
                kind=kind,
                dependency_role_ids=tuple(
                    role for role in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS if role in dependency_roles[kind]
                ),
                predictions=ordered_predictions,
                exchange_predictions=tuple(
                    sorted(exchange_predictions, key=lambda value: value.prediction_id)
                ),
                out_of_cell_rule=(
                    "Refuse unseen preparation keys."
                    if kind is SelectiveDependenceResponseComparatorKind.SATURATED_FINITE_LOOKUP
                    else "Apply the frozen categorical encoding only inside named support."
                ),
                development_unit_ids_sha256=development_unit_ids_sha256,
                evaluation_outcome_count=0,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            )
        )
    return tuple(sorted(encodings, key=lambda value: value.encoding_id))


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_COMPARATOR_KINDS",
    'SelectiveDependenceResponseCategoricalPrediction',
    'SelectiveDependenceResponseComparatorExchangePrediction',
    'SelectiveDependenceResponseComparatorEncoding',
    'SelectiveDependenceResponseComparatorKind',
    'SelectiveDependenceResponseComparatorSeparation',
    'SelectiveDependenceResponseForecastSeparationReport',
    "compare_forecast_to_comparators",
    "comparator_relevant_cells",
    "comparator_relevant_exchanges",
    "fit_comparator_family",
]
