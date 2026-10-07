"Executable outcome-blind scientific grammar comparator fitting and complete-unit panel scoring.\n\nThe target adapter owns conversion from native outcomes to these finite\ncategorical cases.  This module then applies the common independent substrate grounding comparator\ngrammar without opening source artifacts, pooling targets, selecting on\nevaluation outcomes, or recreating structural recurrence.  All development lookup contents are\nencoded in the claim-bearing comparator record and therefore count toward its\nexact canonical description length.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .independent_substrate_grounding import IndependentSubstrateComparatorEncoding, IndependentSubstrateComparatorKind, IndependentSubstrateComparatorScore, IndependentSubstrateLookupCell, IndependentSubstrateRestrictivenessAxis


class IndependentSubstrateForecastPanelPhase(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class IndependentSubstrateCategoricalForecastCase(CanonicalRecord):
    """One nested forecast case belonging to one complete physical unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-categorical-forecast-case'

    case_id: str
    complete_unit_id: str
    forecast_id: str
    forecast_code: int
    key_codes: tuple[int, ...]
    denominator_key_positions: tuple[int, ...]
    receiver_key_positions: tuple[int, ...]
    legal_state_codes: tuple[int, ...]
    observed_state_codes: tuple[int, ...]
    hold_state_codes: tuple[int, ...]
    admit_or_act_state_codes: tuple[int, ...]
    unsafe_admission_state_codes: tuple[int, ...]
    unsafe_observed: bool
    phase: IndependentSubstrateForecastPanelPhase

    def __post_init__(self) -> None:
        for name in ("case_id", "complete_unit_id", "forecast_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.forecast_code < 0 or not self.key_codes:
            raise ValueError("forecast case requires nonnegative codes and a key")
        if any(value < 0 for value in self.key_codes):
            raise ValueError("forecast case key codes must be nonnegative")
        if self.key_codes[0] != self.forecast_code:
            raise ValueError("forecast code must be the first cell-key coordinate")
        for name in (
            "denominator_key_positions",
            "receiver_key_positions",
            "legal_state_codes",
            "observed_state_codes",
            "hold_state_codes",
            "admit_or_act_state_codes",
            "unsafe_admission_state_codes",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must contain nonnegative codes")
        if not all(
            (
                self.legal_state_codes,
                self.observed_state_codes,
                self.hold_state_codes,
                self.admit_or_act_state_codes,
            )
        ):
            raise ValueError("forecast case state rosters must be nonempty")
        legal = set(self.legal_state_codes)
        for name in (
            "observed_state_codes",
            "hold_state_codes",
            "admit_or_act_state_codes",
            "unsafe_admission_state_codes",
        ):
            if not set(getattr(self, name)).issubset(legal):
                raise ValueError(f"{name} lies outside the forecast alphabet")
        positions = (
            *self.denominator_key_positions,
            *self.receiver_key_positions,
        )
        if len(positions) != len(set(positions)) or any(
            value <= 0 or value >= len(self.key_codes) for value in positions
        ):
            raise ValueError("forecast case D/R key positions overlap or lie outside key")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateCategoricalForecastPanel(CanonicalRecord):
    """All predeclared forecast cases over one exact complete-unit roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-categorical-forecast-panel'

    panel_id: str
    phase: IndependentSubstrateForecastPanelPhase
    cases: tuple[IndependentSubstrateCategoricalForecastCase, ...]
    complete_unit_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    nested_cases_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        require_sorted_unique_strings(
            self.complete_unit_ids,
            field_name="complete_unit_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        if self.complete_unit_ids_sha256 != _digest_ids(self.complete_unit_ids):
            raise ValueError("forecast panel complete-unit digest differs")
        if not self.cases or any(value.phase is not self.phase for value in self.cases):
            raise ValueError("forecast panel is empty or crosses phases")
        if {value.complete_unit_id for value in self.cases} != set(self.complete_unit_ids):
            raise ValueError("forecast panel case/unit roster differs")
        by_unit: dict[str, set[str]] = {}
        for case in self.cases:
            by_unit.setdefault(case.complete_unit_id, set()).add(case.forecast_id)
        forecast_rosters = {tuple(sorted(value)) for value in by_unit.values()}
        if len(forecast_rosters) != 1:
            raise ValueError("forecast panel does not retain every forecast per unit")
        by_forecast: dict[str, list[IndependentSubstrateCategoricalForecastCase]] = {}
        for case in self.cases:
            by_forecast.setdefault(case.forecast_id, []).append(case)
        for forecast_cases in by_forecast.values():
            reference = forecast_cases[0]
            if any(
                (
                    value.forecast_code,
                    value.legal_state_codes,
                    value.hold_state_codes,
                    value.admit_or_act_state_codes,
                    value.unsafe_admission_state_codes,
                    value.denominator_key_positions,
                    value.receiver_key_positions,
                    len(value.key_codes),
                )
                != (
                    reference.forecast_code,
                    reference.legal_state_codes,
                    reference.hold_state_codes,
                    reference.admit_or_act_state_codes,
                    reference.unsafe_admission_state_codes,
                    reference.denominator_key_positions,
                    reference.receiver_key_positions,
                    len(reference.key_codes),
                )
                for value in forecast_cases
            ):
                raise ValueError("forecast panel changes one frozen forecast schema")
        if self.nested_cases_count_as_units:
            raise ValueError("nested forecast cases cannot inflate replication")
        expected_access = {
            IndependentSubstrateForecastPanelPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            IndependentSubstrateForecastPanelPhase.EVALUATION: OutcomeAccess.EVALUATOR_REVEAL,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("forecast panel outcome access differs")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateForecastPrediction(CanonicalRecord):
    """One externally produced method emission for one evaluation case."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-forecast-prediction'

    prediction_id: str
    case_id: str
    emitted_state_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.case_id, field_name="case_id")
        if (
            not self.emitted_state_codes
            or tuple(sorted(set(self.emitted_state_codes))) != self.emitted_state_codes
            or any(value < 0 for value in self.emitted_state_codes)
        ):
            raise ValueError("forecast prediction states must be nonempty/nonnegative")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateScoredComparator(CanonicalRecord):
    """One exact encoding and its all-case evaluation score."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-scored-comparator'

    result_id: str
    encoding: IndependentSubstrateComparatorEncoding
    score: IndependentSubstrateComparatorScore

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if (
            self.score.encoding
            != ObjectIdentity.from_record(self.encoding.encoding_id, self.encoding)
            or self.score.scientific_description_bits != self.encoding.scientific_description_bits
        ):
            raise ValueError("scored comparator encoding/score identity differs")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateComparatorAdjudication(CanonicalRecord):
    """Target-local anti-tautology decision under exact outcome-blind scientific grammar precedence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-comparator-adjudication'

    adjudication_id: str
    evaluation_panel: ObjectIdentity
    scored_comparators: tuple[IndependentSubstrateScoredComparator, ...]
    best_non_structural_recurrence_kind: IndependentSubstrateComparatorKind
    structural_recurrence_zero_unsafe_false_admission: bool
    structural_recurrence_primary_exact_match: bool
    structural_recurrence_strictly_sharper_than_wildcard: bool
    structural_recurrence_beats_best_non_structural_recurrence: bool
    comparator_tied_or_won: bool
    structural_recurrence_less_safe_or_exact_than_comparator: bool
    disposition: IndependentSubstrateRestrictivenessAxis
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        if self.evaluation_panel.object_schema != IndependentSubstrateCategoricalForecastPanel.SCHEMA:
            raise ValueError("comparator adjudication panel identity differs")
        require_sorted_unique_ids(
            self.scored_comparators,
            attribute="result_id",
            field_name="scored_comparators",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        by_kind = {value.encoding.kind: value for value in self.scored_comparators}
        if len(by_kind) != len(self.scored_comparators) or set(by_kind) != set(IndependentSubstrateComparatorKind):
            raise ValueError("comparator adjudication does not cover the exact roster")
        structural_recurrence = by_kind[IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE].score
        wildcard = by_kind[IndependentSubstrateComparatorKind.WILDCARD_ALL].score
        non_structural_recurrence = tuple(
            value for kind, value in by_kind.items() if kind is not IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE
        )
        best = min(
            non_structural_recurrence,
            key=lambda value: (value.score.lexicographic_key, value.encoding.kind.value),
        )
        if self.best_non_structural_recurrence_kind is not best.encoding.kind:
            raise ValueError("comparator adjudication best comparator differs")
        zero_unsafe = structural_recurrence.unsafe_false_admission_count == 0
        exact = structural_recurrence.categorical_mismatch_count == 0
        sharper = structural_recurrence.prediction_set_cardinality < wildcard.prediction_set_cardinality
        beats = structural_recurrence.lexicographic_key < best.score.lexicographic_key
        tied_or_won = best.score.lexicographic_key <= structural_recurrence.lexicographic_key
        less_safe_or_exact = any(
            (value.score.unsafe_false_admission_count < structural_recurrence.unsafe_false_admission_count)
            or (
                value.score.unsafe_false_admission_count == structural_recurrence.unsafe_false_admission_count
                and value.score.categorical_mismatch_count < structural_recurrence.categorical_mismatch_count
            )
            for value in non_structural_recurrence
        )
        observed = (
            self.structural_recurrence_zero_unsafe_false_admission,
            self.structural_recurrence_primary_exact_match,
            self.structural_recurrence_strictly_sharper_than_wildcard,
            self.structural_recurrence_beats_best_non_structural_recurrence,
            self.comparator_tied_or_won,
            self.structural_recurrence_less_safe_or_exact_than_comparator,
        )
        expected = (
            zero_unsafe,
            exact,
            sharper,
            beats,
            tied_or_won,
            less_safe_or_exact,
        )
        if observed != expected:
            raise ValueError("comparator adjudication booleans are not score-derived")
        supported = all((zero_unsafe, exact, sharper, beats))
        disposition = (
            IndependentSubstrateRestrictivenessAxis.SUPPORTED
            if supported
            else IndependentSubstrateRestrictivenessAxis.OPPOSED
            if less_safe_or_exact
            else IndependentSubstrateRestrictivenessAxis.NOT_DISTINGUISHED
        )
        if self.disposition is not disposition:
            raise ValueError("comparator adjudication disposition is not score-derived")
        expected_reasons = {
            *(() if zero_unsafe else ('STRUCTURAL_RECURRENCE_UNSAFE_FALSE_ADMISSION',)),
            *(() if exact else ('STRUCTURAL_RECURRENCE_PRIMARY_EXACT_MATCH_FAILED',)),
            *(() if sharper else ('STRUCTURAL_RECURRENCE_NOT_SHARPER_THAN_WILDCARD',)),
            *(() if beats else ('COMPARATOR_TIED_OR_BEAT_STRUCTURAL_RECURRENCE',)),
        }
        if self.reason_codes != tuple(sorted(expected_reasons)):
            raise ValueError("comparator adjudication reasons are not score-derived")


def _forecast_schema(
    panel: IndependentSubstrateCategoricalForecastPanel,
) -> dict[int, IndependentSubstrateCategoricalForecastCase]:
    values: dict[int, IndependentSubstrateCategoricalForecastCase] = {}
    for case in panel.cases:
        previous = values.setdefault(case.forecast_code, case)
        if previous.forecast_id != case.forecast_id:
            raise ValueError("forecast integer code is reused across forecast identities")
    return values


def _encode_emission_table(values: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    encoded: list[int] = [len(values)]
    for forecast_code, states in sorted(values.items()):
        encoded.extend((forecast_code, len(states), *states))
    return tuple(encoded)


def _decode_emission_table(values: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    if not values:
        raise ValueError("comparator emission table is empty")
    expected = values[0]
    cursor = 1
    decoded: dict[int, tuple[int, ...]] = {}
    for _ in range(expected):
        if cursor + 2 > len(values):
            raise ValueError("comparator emission table is truncated")
        forecast_code, count = values[cursor : cursor + 2]
        cursor += 2
        if count < 1 or cursor + count > len(values):
            raise ValueError("comparator emission state table is truncated")
        states = values[cursor : cursor + count]
        cursor += count
        if forecast_code in decoded or tuple(sorted(set(states))) != states:
            raise ValueError("comparator emission table duplicates codes or states")
        decoded[forecast_code] = states
    if cursor != len(values):
        raise ValueError("comparator emission table contains trailing codes")
    return decoded


def _project_key(
    case: IndependentSubstrateCategoricalForecastCase,
    kind: IndependentSubstrateComparatorKind,
) -> tuple[int, ...]:
    removed: set[int] = set()
    if kind is IndependentSubstrateComparatorKind.DENOMINATOR_BLIND:
        removed.update(case.denominator_key_positions)
    if kind is IndependentSubstrateComparatorKind.RECEIVER_BLIND:
        removed.update(case.receiver_key_positions)
    return tuple(value for index, value in enumerate(case.key_codes) if index not in removed)


def _lookup_cells(
    *,
    namespace_id: str,
    kind: IndependentSubstrateComparatorKind,
    cases: tuple[IndependentSubstrateCategoricalForecastCase, ...],
) -> tuple[IndependentSubstrateLookupCell, ...]:
    grouped: dict[tuple[int, ...], set[int]] = {}
    for case in cases:
        grouped.setdefault(_project_key(case, kind), set()).update(case.observed_state_codes)
    slug = kind.value.lower().replace("_", "-")
    return tuple(
        IndependentSubstrateLookupCell(
            cell_id=f"lookup.{namespace_id}.{slug}.{index:06d}",
            key_codes=key,
            emitted_state_codes=tuple(sorted(states)),
        )
        for index, (key, states) in enumerate(sorted(grouped.items()))
    )


def fit_independent_substrate_comparator_encodings(
    *,
    namespace_id: str,
    development_panel: IndependentSubstrateCategoricalForecastPanel,
    structural_recurrence_method_parameter_codes: tuple[int, ...],
    denominator_dependency_ids: tuple[str, ...],
    receiver_dependency_ids: tuple[str, ...],
) -> tuple[IndependentSubstrateComparatorEncoding, ...]:
    """Fit only the predeclared development-derived comparator parameters."""

    validate_stable_id(namespace_id, field_name="namespace_id")
    if development_panel.phase is not IndependentSubstrateForecastPanelPhase.DEVELOPMENT:
        raise ValueError("comparator fitting requires development-visible cases")
    if any(value < 0 for value in structural_recurrence_method_parameter_codes):
        raise ValueError('structural recurrence method parameter codes must be nonnegative')
    require_sorted_unique_strings(
        denominator_dependency_ids,
        field_name="denominator_dependency_ids",
    )
    require_sorted_unique_strings(
        receiver_dependency_ids,
        field_name="receiver_dependency_ids",
    )
    schema = _forecast_schema(development_panel)
    fixed_tables: dict[IndependentSubstrateComparatorKind, dict[int, tuple[int, ...]]] = {
        IndependentSubstrateComparatorKind.WILDCARD_ALL: {
            code: case.legal_state_codes for code, case in schema.items()
        },
        IndependentSubstrateComparatorKind.ALWAYS_HOLD: {
            code: case.hold_state_codes for code, case in schema.items()
        },
        IndependentSubstrateComparatorKind.ALWAYS_ADMIT_OR_ACT: {
            code: case.admit_or_act_state_codes for code, case in schema.items()
        },
    }
    majority: dict[int, tuple[int, ...]] = {}
    for code in sorted(schema):
        counts = {state: 0 for state in schema[code].legal_state_codes}
        for case in development_panel.cases:
            if case.forecast_code == code:
                for state in case.observed_state_codes:
                    counts[state] += 1
        maximum = max(counts.values())
        majority[code] = tuple(state for state, count in sorted(counts.items()) if count == maximum)
    fixed_tables[IndependentSubstrateComparatorKind.DEVELOPMENT_MAJORITY] = majority

    values = []
    for kind in sorted(IndependentSubstrateComparatorKind, key=lambda value: value.value):
        slug = kind.value.lower().replace("_", "-")
        lookup = (
            _lookup_cells(
                namespace_id=namespace_id,
                kind=kind,
                cases=development_panel.cases,
            )
            if kind
            in {
                IndependentSubstrateComparatorKind.DENOMINATOR_BLIND,
                IndependentSubstrateComparatorKind.RECEIVER_BLIND,
                IndependentSubstrateComparatorKind.SATURATED_DEVELOPMENT_LOOKUP,
            }
            else ()
        )
        if kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE:
            parameters = structural_recurrence_method_parameter_codes
            out_of_cell = 'STRUCTURAL_RECURRENCE_PREDICTION_ISSUE'
        elif kind in fixed_tables:
            parameters = _encode_emission_table(fixed_tables[kind])
            out_of_cell = "FORECAST_EMISSION_TABLE"
        else:
            parameters = ()
            out_of_cell = "WILDCARD_ALL"
        denominator_dependencies = (
            ()
            if kind
            in {
                IndependentSubstrateComparatorKind.WILDCARD_ALL,
                IndependentSubstrateComparatorKind.ALWAYS_HOLD,
                IndependentSubstrateComparatorKind.ALWAYS_ADMIT_OR_ACT,
                IndependentSubstrateComparatorKind.DEVELOPMENT_MAJORITY,
                IndependentSubstrateComparatorKind.DENOMINATOR_BLIND,
            }
            else denominator_dependency_ids
        )
        receiver_dependencies = (
            ()
            if kind
            in {
                IndependentSubstrateComparatorKind.WILDCARD_ALL,
                IndependentSubstrateComparatorKind.ALWAYS_HOLD,
                IndependentSubstrateComparatorKind.ALWAYS_ADMIT_OR_ACT,
                IndependentSubstrateComparatorKind.DEVELOPMENT_MAJORITY,
                IndependentSubstrateComparatorKind.RECEIVER_BLIND,
            }
            else receiver_dependency_ids
        )
        values.append(
            IndependentSubstrateComparatorEncoding(
                encoding_id=f"encoding.{namespace_id}.{slug}",
                kind=kind,
                selected_parameter_codes=parameters,
                denominator_dependency_ids=denominator_dependencies,
                receiver_dependency_ids=receiver_dependencies,
                lookup_cells=lookup,
                out_of_cell_rule=out_of_cell,
            )
        )
    return tuple(sorted(values, key=lambda value: value.encoding_id))


def _emission_for_case(
    *,
    encoding: IndependentSubstrateComparatorEncoding,
    case: IndependentSubstrateCategoricalForecastCase,
    prediction_by_case: dict[str, IndependentSubstrateForecastPrediction],
) -> tuple[int, ...]:
    if encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE:
        try:
            emitted = prediction_by_case[case.case_id].emitted_state_codes
        except KeyError as error:
            raise ValueError('structural recurrence prediction omits an evaluation case') from error
    elif encoding.kind in {
        IndependentSubstrateComparatorKind.WILDCARD_ALL,
        IndependentSubstrateComparatorKind.ALWAYS_HOLD,
        IndependentSubstrateComparatorKind.ALWAYS_ADMIT_OR_ACT,
        IndependentSubstrateComparatorKind.DEVELOPMENT_MAJORITY,
    }:
        table = _decode_emission_table(encoding.selected_parameter_codes)
        try:
            emitted = table[case.forecast_code]
        except KeyError as error:
            raise ValueError("fixed comparator omits a frozen forecast") from error
    else:
        key = _project_key(case, encoding.kind)
        matches = tuple(value for value in encoding.lookup_cells if value.key_codes == key)
        emitted = (
            tuple(sorted({state for match in matches for state in match.emitted_state_codes}))
            if matches
            else case.legal_state_codes
        )
    if not emitted or not set(emitted).issubset(case.legal_state_codes):
        raise ValueError("comparator emitted an empty or illegal forecast set")
    return emitted


def score_independent_substrate_comparator(
    *,
    score_id: str,
    encoding: IndependentSubstrateComparatorEncoding,
    evaluation_panel: IndependentSubstrateCategoricalForecastPanel,
    structural_recurrence_predictions: tuple[IndependentSubstrateForecastPrediction, ...] = (),
) -> IndependentSubstrateComparatorScore:
    """Score one encoding on all cases while retaining complete-unit identity."""

    if evaluation_panel.phase is not IndependentSubstrateForecastPanelPhase.EVALUATION:
        raise ValueError("comparator scoring requires evaluator-revealed cases")
    require_sorted_unique_ids(
        structural_recurrence_predictions,
        attribute="prediction_id",
        field_name='structural_recurrence_predictions',
    )
    if len({value.case_id for value in structural_recurrence_predictions}) != len(structural_recurrence_predictions):
        raise ValueError('structural recurrence predictions repeat an evaluation case')
    prediction_by_case = {value.case_id: value for value in structural_recurrence_predictions}
    if encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE:
        if set(prediction_by_case) != {value.case_id for value in evaluation_panel.cases}:
            raise ValueError('structural recurrence prediction/evaluation case roster differs')
    elif structural_recurrence_predictions:
        raise ValueError('non-structural recurrence comparator cannot consume structural recurrence predictions')
    unsafe = 0
    mismatch = 0
    cardinality = 0
    for case in evaluation_panel.cases:
        emitted = _emission_for_case(
            encoding=encoding,
            case=case,
            prediction_by_case=prediction_by_case,
        )
        unsafe += int(
            case.unsafe_observed and bool(set(emitted) & set(case.unsafe_admission_state_codes))
        )
        mismatch += int(not set(case.observed_state_codes).issubset(set(emitted)))
        cardinality += len(emitted)
    return IndependentSubstrateComparatorScore(
        score_id=score_id,
        encoding=ObjectIdentity.from_record(encoding.encoding_id, encoding),
        unsafe_false_admission_count=unsafe,
        categorical_mismatch_count=mismatch,
        prediction_set_cardinality=cardinality,
        scientific_description_bits=encoding.scientific_description_bits,
    )


def adjudicate_independent_substrate_restrictiveness(
    *,
    adjudication_id: str,
    evaluation_panel: IndependentSubstrateCategoricalForecastPanel,
    scored_comparators: tuple[IndependentSubstrateScoredComparator, ...],
) -> IndependentSubstrateComparatorAdjudication:
    """Apply the nonaveraged outcome-blind scientific grammar restrictiveness conjunction for one target."""

    by_kind = {value.encoding.kind: value for value in scored_comparators}
    if len(by_kind) != len(scored_comparators) or set(by_kind) != set(IndependentSubstrateComparatorKind):
        raise ValueError("restrictiveness adjudication requires the exact comparator roster")
    structural_recurrence = by_kind[IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE].score
    wildcard = by_kind[IndependentSubstrateComparatorKind.WILDCARD_ALL].score
    non_structural_recurrence = tuple(
        value for kind, value in by_kind.items() if kind is not IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE
    )
    best = min(
        non_structural_recurrence,
        key=lambda value: (value.score.lexicographic_key, value.encoding.kind.value),
    )
    zero_unsafe = structural_recurrence.unsafe_false_admission_count == 0
    exact = structural_recurrence.categorical_mismatch_count == 0
    sharper = structural_recurrence.prediction_set_cardinality < wildcard.prediction_set_cardinality
    beats = structural_recurrence.lexicographic_key < best.score.lexicographic_key
    tied_or_won = best.score.lexicographic_key <= structural_recurrence.lexicographic_key
    less_safe_or_exact = any(
        (value.score.unsafe_false_admission_count < structural_recurrence.unsafe_false_admission_count)
        or (
            value.score.unsafe_false_admission_count == structural_recurrence.unsafe_false_admission_count
            and value.score.categorical_mismatch_count < structural_recurrence.categorical_mismatch_count
        )
        for value in non_structural_recurrence
    )
    supported = all((zero_unsafe, exact, sharper, beats))
    disposition = (
        IndependentSubstrateRestrictivenessAxis.SUPPORTED
        if supported
        else IndependentSubstrateRestrictivenessAxis.OPPOSED
        if less_safe_or_exact
        else IndependentSubstrateRestrictivenessAxis.NOT_DISTINGUISHED
    )
    reasons = {
        *(() if zero_unsafe else ('STRUCTURAL_RECURRENCE_UNSAFE_FALSE_ADMISSION',)),
        *(() if exact else ('STRUCTURAL_RECURRENCE_PRIMARY_EXACT_MATCH_FAILED',)),
        *(() if sharper else ('STRUCTURAL_RECURRENCE_NOT_SHARPER_THAN_WILDCARD',)),
        *(() if beats else ('COMPARATOR_TIED_OR_BEAT_STRUCTURAL_RECURRENCE',)),
    }
    return IndependentSubstrateComparatorAdjudication(
        adjudication_id=adjudication_id,
        evaluation_panel=ObjectIdentity.from_record(
            evaluation_panel.panel_id,
            evaluation_panel,
        ),
        scored_comparators=tuple(sorted(scored_comparators, key=lambda value: value.result_id)),
        best_non_structural_recurrence_kind=best.encoding.kind,
        structural_recurrence_zero_unsafe_false_admission=zero_unsafe,
        structural_recurrence_primary_exact_match=exact,
        structural_recurrence_strictly_sharper_than_wildcard=sharper,
        structural_recurrence_beats_best_non_structural_recurrence=beats,
        comparator_tied_or_won=tied_or_won,
        structural_recurrence_less_safe_or_exact_than_comparator=less_safe_or_exact,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    'IndependentSubstrateCategoricalForecastCase',
    'IndependentSubstrateCategoricalForecastPanel',
    'IndependentSubstrateComparatorAdjudication',
    'IndependentSubstrateForecastPanelPhase',
    'IndependentSubstrateForecastPrediction',
    'IndependentSubstrateScoredComparator',
    'adjudicate_independent_substrate_restrictiveness',
    'fit_independent_substrate_comparator_encodings',
    'score_independent_substrate_comparator',
]
