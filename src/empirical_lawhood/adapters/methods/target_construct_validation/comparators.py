"""Target-neutral categorical comparators and noncompensating scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Iterable

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.adapters.methods.scientific_description_code import scientific_description_octets

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .target_adjudication import TargetConstructValidationRestrictivenessAxis


TARGET_CONSTRUCT_VALIDATION_COMPARATOR_ROLE_IDS = ("A", "D", "H", "R", "tau")


class TargetConstructValidationComparatorKind(StrEnum):
    ALWAYS_ACT_OR_ADMIT = "always-act-or-admit"
    ALWAYS_HOLD = "always-hold"
    DENOMINATOR_BLIND = "denominator-blind"
    DEVELOPMENT_MAJORITY = "development-majority"
    HISTORY_BLIND = "history-blind"
    STRUCTURAL_RECURRENCE = "structural-recurrence"
    RECEIVER_BLIND = "receiver-blind"
    SATURATED_DEVELOPMENT_LOOKUP = "saturated-development-lookup"
    TARGET_NATIVE_BASELINE = "target-native-baseline"
    WILDCARD_ALL = "wildcard-all"


class TargetConstructValidationPanelPhase(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class TargetConstructValidationCaseOutcome(StrEnum):
    OBSERVED = "OBSERVED"
    ADVERSE = "ADVERSE"
    STOPPED = "STOPPED"
    MISSING = "MISSING"


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCategoricalForecastCase(CanonicalRecord):
    """One nested cell observation belonging to one complete unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-categorical-forecast-case'

    case_id: str
    complete_unit_id: str
    cell_id: str
    denominator_value_id: str
    history_value_id: str
    native_action_value_id: str
    receiver_value_id: str
    horizon_value_id: str
    requested_action_id: str
    accepted_action_id: str
    applied_action_id: str
    realized_action_id: str
    legal_state_ids: tuple[str, ...]
    observed_state_ids: tuple[str, ...]
    hold_state_ids: tuple[str, ...]
    admit_or_act_state_ids: tuple[str, ...]
    unsafe_admission_state_ids: tuple[str, ...]
    unsafe_observed: bool
    outcome: TargetConstructValidationCaseOutcome
    phase: TargetConstructValidationPanelPhase

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "complete_unit_id",
            "cell_id",
            "denominator_value_id",
            "history_value_id",
            "native_action_value_id",
            "receiver_value_id",
            "horizon_value_id",
            "requested_action_id",
            "accepted_action_id",
            "applied_action_id",
            "realized_action_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "legal_state_ids",
            "observed_state_ids",
            "hold_state_ids",
            "admit_or_act_state_ids",
            "unsafe_admission_state_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name in {"observed_state_ids", "unsafe_admission_state_ids"},
            )
        legal = set(self.legal_state_ids)
        for name in (
            "observed_state_ids",
            "hold_state_ids",
            "admit_or_act_state_ids",
            "unsafe_admission_state_ids",
        ):
            if not set(getattr(self, name)).issubset(legal):
                raise ValueError(f"{name} lies outside the target-native alphabet")
        observed = self.outcome in {
            TargetConstructValidationCaseOutcome.OBSERVED,
            TargetConstructValidationCaseOutcome.ADVERSE,
        }
        if observed != bool(self.observed_state_ids):
            raise ValueError("typed case outcome and observed states are inconsistent")

    def key(self, role_ids: tuple[str, ...]) -> tuple[str, ...]:
        values = {
            "A": self.native_action_value_id,
            "D": self.denominator_value_id,
            "H": self.history_value_id,
            "R": self.receiver_value_id,
            "tau": self.horizon_value_id,
        }
        return tuple(values[role_id] for role_id in role_ids)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCategoricalForecastPanel(CanonicalRecord):
    """Full Cartesian cell roster over exact complete units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-categorical-forecast-panel'

    panel_id: str
    phase: TargetConstructValidationPanelPhase
    cases: tuple[TargetConstructValidationCategoricalForecastCase, ...]
    cell_ids: tuple[str, ...]
    complete_unit_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    nested_cases_count_as_units: bool
    panel_limited_cases_retained: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        require_sorted_unique_strings(
            self.cell_ids,
            field_name="cell_ids",
            allow_empty=False,
        )
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
            raise ValueError("complete-unit roster digest differs")
        expected_pairs = {
            (unit_id, cell_id) for unit_id in self.complete_unit_ids for cell_id in self.cell_ids
        }
        observed_pairs = {(case.complete_unit_id, case.cell_id) for case in self.cases}
        if len(self.cases) != len(expected_pairs) or observed_pairs != expected_pairs:
            raise ValueError("panel must retain every cell for every complete unit")
        if any(case.phase is not self.phase for case in self.cases):
            raise ValueError("forecast panel crosses phases")
        if self.nested_cases_count_as_units:
            raise ValueError("nested cells cannot inflate complete-unit replication")
        if not self.panel_limited_cases_retained:
            raise ValueError("panel-limited cases must remain in the issued denominator")
        expected_access = {
            TargetConstructValidationPanelPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            TargetConstructValidationPanelPhase.EVALUATION: OutcomeAccess.EVALUATOR_REVEAL,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("panel phase and outcome access differ")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationComparatorPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-comparator-plan'

    plan_id: str
    target_id: str
    comparator_kinds: tuple[TargetConstructValidationComparatorKind, ...]
    key_role_ids: tuple[str, ...]
    target_native_baseline_id: str
    target_native_baseline_specification: ObjectIdentity
    primary_cell_ids: tuple[str, ...]
    evaluation_complete_unit_ids_sha256: str
    baseline_chosen_before_structural_recurrence_mapping: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("plan_id", "target_id", "target_native_baseline_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.comparator_kinds != tuple(
            sorted(TargetConstructValidationComparatorKind, key=lambda value: value.value)
        ):
            raise ValueError("comparator plan must contain the exact ten-member roster")
        if self.key_role_ids != TARGET_CONSTRUCT_VALIDATION_COMPARATOR_ROLE_IDS:
            raise ValueError("comparator keys must expose exact D/H/A/R/tau roles")
        require_sorted_unique_strings(
            self.primary_cell_ids,
            field_name="primary_cell_ids",
            allow_empty=False,
        )
        if not 6 <= len(self.primary_cell_ids) <= 24:
            raise ValueError("comparator plan must bind 6--24 primary cells")
        validate_sha256(
            self.evaluation_complete_unit_ids_sha256,
            field_name="evaluation_complete_unit_ids_sha256",
        )
        if not self.baseline_chosen_before_structural_recurrence_mapping:
            raise ValueError('target-native baseline must precede structural recurrence mapping')
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("comparator plan must freeze outcome-blindly")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationComparatorLookupCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-comparator-lookup-cell'

    lookup_id: str
    key_values: tuple[str, ...]
    emitted_state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.lookup_id, field_name="lookup_id")
        require_sorted_unique_strings(
            self.emitted_state_ids,
            field_name="emitted_state_ids",
        )
        if not self.key_values or any(not value for value in self.key_values):
            raise ValueError("comparator lookup key must be nonempty")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationComparatorEncoding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-comparator-encoding'

    encoding_id: str
    kind: TargetConstructValidationComparatorKind
    dependency_role_ids: tuple[str, ...]
    lookup_cells: tuple[TargetConstructValidationComparatorLookupCell, ...]
    development_complete_unit_ids_sha256: str
    out_of_cell_rule: str
    fit_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.encoding_id, field_name="encoding_id")
        if any(value not in TARGET_CONSTRUCT_VALIDATION_COMPARATOR_ROLE_IDS for value in self.dependency_role_ids):
            raise ValueError("comparator dependency names an unknown role")
        if (
            tuple(
                value for value in TARGET_CONSTRUCT_VALIDATION_COMPARATOR_ROLE_IDS if value in self.dependency_role_ids
            )
            != self.dependency_role_ids
        ):
            raise ValueError("comparator dependency roles are out of canonical order")
        require_sorted_unique_ids(
            self.lookup_cells,
            attribute="lookup_id",
            field_name="lookup_cells",
        )
        if len({value.key_values for value in self.lookup_cells}) != len(self.lookup_cells):
            raise ValueError("comparator encoding contains duplicate lookup keys")
        if any(
            len(value.key_values) != len(self.dependency_role_ids) for value in self.lookup_cells
        ):
            raise ValueError("comparator lookup key arity differs")
        validate_sha256(
            self.development_complete_unit_ids_sha256,
            field_name="development_complete_unit_ids_sha256",
        )
        if self.out_of_cell_rule != "WILDCARD_ALL_LEGAL_STATES":
            raise ValueError("comparators must wildcard outside development support")
        frozen_kinds = {
            TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE,
            TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE,
        }
        expected_access = (
            OutcomeAccess.OUTCOME_BLIND
            if self.kind in frozen_kinds
            else OutcomeAccess.DEVELOPMENT_VISIBLE
        )
        if self.fit_outcome_access is not expected_access:
            raise ValueError("comparator fit access differs from its scientific role")

    @property
    def scientific_description_bits(self) -> int:
        return 8 * scientific_description_octets(self)


def _dependency_roles(kind: TargetConstructValidationComparatorKind) -> tuple[str, ...]:
    if kind is TargetConstructValidationComparatorKind.DENOMINATOR_BLIND:
        return ("A", "H", "R", "tau")
    if kind is TargetConstructValidationComparatorKind.HISTORY_BLIND:
        return ("A", "D", "R", "tau")
    if kind is TargetConstructValidationComparatorKind.RECEIVER_BLIND:
        return ("A", "D", "H", "tau")
    return TARGET_CONSTRUCT_VALIDATION_COMPARATOR_ROLE_IDS


def _majority_states(cases: Iterable[TargetConstructValidationCategoricalForecastCase]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for case in cases:
        for state_id in case.observed_state_ids:
            counts[state_id] = counts.get(state_id, 0) + 1
    if not counts:
        return ()
    maximum = max(counts.values())
    return tuple(sorted(state_id for state_id, count in counts.items() if count == maximum))


def fit_development_encoding(
    panel: TargetConstructValidationCategoricalForecastPanel,
    *,
    kind: TargetConstructValidationComparatorKind,
) -> TargetConstructValidationComparatorEncoding:
    'Fit one non-structural recurrence/non-native comparator on development units only.'

    if panel.phase is not TargetConstructValidationPanelPhase.DEVELOPMENT:
        raise ValueError("comparator fitting requires a development panel")
    if kind in {
        TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE,
        TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE,
    }:
        raise ValueError('structural recurrence/native baseline must be supplied from pre-outcome freezes')
    roles = _dependency_roles(kind)
    groups: dict[tuple[str, ...], list[TargetConstructValidationCategoricalForecastCase]] = {}
    for case in panel.cases:
        groups.setdefault(case.key(roles), []).append(case)
    lookup: list[TargetConstructValidationComparatorLookupCell] = []
    for index, (key, cases) in enumerate(sorted(groups.items())):
        legal = tuple(sorted({state for case in cases for state in case.legal_state_ids}))
        if kind is TargetConstructValidationComparatorKind.WILDCARD_ALL:
            emitted = legal
        elif kind is TargetConstructValidationComparatorKind.ALWAYS_HOLD:
            emitted = tuple(sorted({state for case in cases for state in case.hold_state_ids}))
        elif kind is TargetConstructValidationComparatorKind.ALWAYS_ACT_OR_ADMIT:
            emitted = tuple(
                sorted({state for case in cases for state in case.admit_or_act_state_ids})
            )
        elif kind in {
            TargetConstructValidationComparatorKind.DEVELOPMENT_MAJORITY,
            TargetConstructValidationComparatorKind.DENOMINATOR_BLIND,
            TargetConstructValidationComparatorKind.HISTORY_BLIND,
            TargetConstructValidationComparatorKind.RECEIVER_BLIND,
        }:
            emitted = _majority_states(cases) or legal
        elif kind is TargetConstructValidationComparatorKind.SATURATED_DEVELOPMENT_LOOKUP:
            emitted = (
                tuple(sorted({state for case in cases for state in case.observed_state_ids}))
                or legal
            )
        else:  # pragma: no cover - closed enum exhaustiveness guard
            raise AssertionError(f"unsupported comparator kind: {kind}")
        lookup.append(
            TargetConstructValidationComparatorLookupCell(
                lookup_id=f"{panel.panel_id}.{kind.value.lower().replace('_', '-')}.{index}",
                key_values=key,
                emitted_state_ids=emitted,
            )
        )
    return TargetConstructValidationComparatorEncoding(
        encoding_id=f"{panel.panel_id}.{kind.value.lower().replace('_', '-')}.encoding",
        kind=kind,
        dependency_role_ids=roles,
        lookup_cells=tuple(sorted(lookup, key=lambda value: value.lookup_id)),
        development_complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
        out_of_cell_rule="WILDCARD_ALL_LEGAL_STATES",
        fit_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def build_frozen_encoding(
    *,
    encoding_id: str,
    kind: TargetConstructValidationComparatorKind,
    dependency_role_ids: tuple[str, ...],
    lookup_cells: tuple[TargetConstructValidationComparatorLookupCell, ...],
    development_complete_unit_ids_sha256: str,
) -> TargetConstructValidationComparatorEncoding:
    if kind not in {
        TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE,
        TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE,
    }:
        raise ValueError('only structural recurrence/native baseline are pre-outcome frozen encodings')
    return TargetConstructValidationComparatorEncoding(
        encoding_id=encoding_id,
        kind=kind,
        dependency_role_ids=dependency_role_ids,
        lookup_cells=lookup_cells,
        development_complete_unit_ids_sha256=development_complete_unit_ids_sha256,
        out_of_cell_rule="WILDCARD_ALL_LEGAL_STATES",
        fit_outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationComparatorScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-comparator-score'

    score_id: str
    encoding: ObjectIdentity
    evaluation_panel: ObjectIdentity
    complete_unit_ids_sha256: str
    complete_unit_count: int
    case_count: int
    unsafe_false_admission_count: int
    categorical_mismatch_count: int
    uncovered_case_count: int
    prediction_set_cardinality: int
    unevaluable_case_count: int
    scientific_description_bits: int

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        for name in (
            "complete_unit_count",
            "case_count",
            "unsafe_false_admission_count",
            "categorical_mismatch_count",
            "uncovered_case_count",
            "prediction_set_cardinality",
            "unevaluable_case_count",
            "scientific_description_bits",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.complete_unit_count == 0 or self.case_count == 0:
            raise ValueError("comparator score requires complete units and cases")

    @property
    def lexicographic_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.unsafe_false_admission_count,
            self.categorical_mismatch_count,
            self.uncovered_case_count,
            self.prediction_set_cardinality,
            self.scientific_description_bits,
        )


def score_comparator(
    panel: TargetConstructValidationCategoricalForecastPanel,
    encoding: TargetConstructValidationComparatorEncoding,
) -> TargetConstructValidationComparatorScore:
    if panel.phase is not TargetConstructValidationPanelPhase.EVALUATION:
        raise ValueError("comparator scoring requires an evaluation panel")
    table = {value.key_values: value.emitted_state_ids for value in encoding.lookup_cells}
    unsafe = mismatch = uncovered = cardinality = unevaluable = 0
    for case in panel.cases:
        emitted = table.get(case.key(encoding.dependency_role_ids), case.legal_state_ids)
        emitted = tuple(state for state in emitted if state in case.legal_state_ids)
        if not emitted:
            uncovered += 1
        cardinality += len(emitted)
        if case.outcome in {TargetConstructValidationCaseOutcome.MISSING, TargetConstructValidationCaseOutcome.STOPPED}:
            unevaluable += 1
            continue
        if not set(emitted).intersection(case.observed_state_ids):
            mismatch += 1
        if case.unsafe_observed and set(emitted).intersection(case.admit_or_act_state_ids):
            unsafe += 1
    return TargetConstructValidationComparatorScore(
        score_id=f"{panel.panel_id}.{encoding.kind.value.lower().replace('_', '-')}.score",
        encoding=ObjectIdentity.from_record(encoding.encoding_id, encoding),
        evaluation_panel=ObjectIdentity.from_record(panel.panel_id, panel),
        complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
        complete_unit_count=len(panel.complete_unit_ids),
        case_count=len(panel.cases),
        unsafe_false_admission_count=unsafe,
        categorical_mismatch_count=mismatch,
        uncovered_case_count=uncovered,
        prediction_set_cardinality=cardinality,
        unevaluable_case_count=unevaluable,
        scientific_description_bits=encoding.scientific_description_bits,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationScoredComparator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-scored-comparator'

    result_id: str
    encoding: TargetConstructValidationComparatorEncoding
    score: TargetConstructValidationComparatorScore

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.score.encoding != ObjectIdentity.from_record(
            self.encoding.encoding_id,
            self.encoding,
        ):
            raise ValueError("scored comparator encoding identity differs")
        if self.score.scientific_description_bits != self.encoding.scientific_description_bits:
            raise ValueError("comparator complexity is not encoding-derived")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationRestrictivenessResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-restrictiveness-result'

    result_id: str
    scored_comparators: tuple[TargetConstructValidationScoredComparator, ...]
    best_non_structural_recurrence_kind: TargetConstructValidationComparatorKind
    structural_recurrence_no_excess_unsafe_admission: bool
    structural_recurrence_strictly_improves_simple_exact_error: bool
    structural_recurrence_full_coverage: bool
    structural_recurrence_sharper_than_wildcard: bool
    native_baseline_did_not_beat_structural_recurrence: bool
    exact_complete_unit_inference_closed: bool
    disposition: TargetConstructValidationRestrictivenessAxis
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.scored_comparators,
            attribute="result_id",
            field_name="scored_comparators",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        by_kind = {value.encoding.kind: value.score for value in self.scored_comparators}
        if len(by_kind) != 10 or set(by_kind) != set(TargetConstructValidationComparatorKind):
            raise ValueError("restrictiveness result lacks the exact comparator roster")


def adjudicate_restrictiveness(
    scored: Iterable[TargetConstructValidationScoredComparator],
) -> TargetConstructValidationRestrictivenessResult:
    values = tuple(sorted(scored, key=lambda value: value.result_id))
    by_kind = {value.encoding.kind: value.score for value in values}
    if len(by_kind) != 10 or set(by_kind) != set(TargetConstructValidationComparatorKind):
        raise ValueError("COMPARATOR_FAMILY_INCOMPLETE")
    if len({value.complete_unit_ids_sha256 for value in by_kind.values()}) != 1:
        raise ValueError("comparators must score identical complete units")
    if len({value.evaluation_panel for value in by_kind.values()}) != 1:
        raise ValueError("comparators must score the identical evaluation panel")

    structural_recurrence = by_kind[TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE]
    wildcard = by_kind[TargetConstructValidationComparatorKind.WILDCARD_ALL]
    native = by_kind[TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE]
    non_structural_recurrence = {
        kind: score for kind, score in by_kind.items() if kind is not TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE
    }
    best_kind, best_score = min(
        non_structural_recurrence.items(),
        key=lambda value: (value[1].lexicographic_key, value[0].value),
    )
    simple_kinds = {
        TargetConstructValidationComparatorKind.ALWAYS_ACT_OR_ADMIT,
        TargetConstructValidationComparatorKind.ALWAYS_HOLD,
        TargetConstructValidationComparatorKind.DENOMINATOR_BLIND,
        TargetConstructValidationComparatorKind.DEVELOPMENT_MAJORITY,
        TargetConstructValidationComparatorKind.HISTORY_BLIND,
        TargetConstructValidationComparatorKind.RECEIVER_BLIND,
    }
    no_excess_unsafe = all(
        structural_recurrence.unsafe_false_admission_count <= by_kind[kind].unsafe_false_admission_count
        for kind in simple_kinds | {TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE}
    )
    improves_simple = all(
        structural_recurrence.categorical_mismatch_count < by_kind[kind].categorical_mismatch_count
        for kind in simple_kinds
    )
    full_coverage = structural_recurrence.uncovered_case_count == 0
    sharper = structural_recurrence.prediction_set_cardinality < wildcard.prediction_set_cardinality
    native_not_better = structural_recurrence.lexicographic_key <= native.lexicographic_key
    inference_closed = all(score.unevaluable_case_count == 0 for score in by_kind.values())
    less_safe_or_exact = any(
        score.unsafe_false_admission_count < structural_recurrence.unsafe_false_admission_count
        or (
            score.unsafe_false_admission_count == structural_recurrence.unsafe_false_admission_count
            and score.categorical_mismatch_count < structural_recurrence.categorical_mismatch_count
        )
        for score in non_structural_recurrence.values()
    )
    if not inference_closed:
        disposition = TargetConstructValidationRestrictivenessAxis.UNEVALUABLE
    elif less_safe_or_exact:
        disposition = TargetConstructValidationRestrictivenessAxis.OPPOSED
    elif all((no_excess_unsafe, improves_simple, full_coverage, sharper, native_not_better)):
        disposition = TargetConstructValidationRestrictivenessAxis.SUPPORTED
    else:
        disposition = TargetConstructValidationRestrictivenessAxis.NOT_DISTINGUISHED
    reasons = {
        *(() if no_excess_unsafe else ('STRUCTURAL_RECURRENCE_EXCESS_UNSAFE_ADMISSION',)),
        *(() if improves_simple else ('STRUCTURAL_RECURRENCE_DID_NOT_IMPROVE_EVERY_SIMPLE_COMPARATOR',)),
        *(() if full_coverage else ('STRUCTURAL_RECURRENCE_PRIMARY_COVERAGE_INCOMPLETE',)),
        *(() if sharper else ('STRUCTURAL_RECURRENCE_NOT_SHARPER_THAN_WILDCARD',)),
        *(() if native_not_better else ('TARGET_NATIVE_BASELINE_BEAT_STRUCTURAL_RECURRENCE',)),
        *(() if inference_closed else ("COMPLETE_UNIT_INFERENCE_UNEVALUABLE",)),
    }
    return TargetConstructValidationRestrictivenessResult(
        result_id="target-construct-validation.target-restrictiveness",
        scored_comparators=values,
        best_non_structural_recurrence_kind=best_kind,
        structural_recurrence_no_excess_unsafe_admission=no_excess_unsafe,
        structural_recurrence_strictly_improves_simple_exact_error=improves_simple,
        structural_recurrence_full_coverage=full_coverage,
        structural_recurrence_sharper_than_wildcard=sharper,
        native_baseline_did_not_beat_structural_recurrence=native_not_better,
        exact_complete_unit_inference_closed=inference_closed,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
    )
