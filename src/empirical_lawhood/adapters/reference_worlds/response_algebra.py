"""Ten truth-known response-algebra worlds with a truth-blind method surface."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.contracts import DataSplit
from empirical_lawhood.adapters.methods.response_algebra import (
    AffineTransition,
    DeliveredGeneratorRecord,
    DeliveredWordRecord,
    DiscreteTransition,
    GeneratorTransition,
    ReceiverEquivalenceCriterion,
    ResponseAlgebraMethodConfig,
    ResponseAlgebraMethodInput,
    ResponseAlgebraMethodResult,
    StateVector,
    StateViewRole,
    StateViewSpec,
    WordResponse,
)
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.response_algebra import ActionLetter, ActionStage, ActionStageValue, LetterActionWord, ActionWordMode, AlgebraRepresentation, PairScientificLabel, PortMateriality, ReceiverVisibility, SequentialComposition, SimultaneousComposition, StateSufficiency, TemporalComposition
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.response_algebra import ResponseAlgebraConformanceSpec


@dataclass(frozen=True, slots=True)
class TruthBlindResponseAlgebraInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/truth-blind-response-algebra-invocation'

    invocation_id: str
    method_input: ResponseAlgebraMethodInput
    method_config: ResponseAlgebraMethodConfig

    def __post_init__(self) -> None:
        validate_stable_id(self.invocation_id, field_name="invocation_id")
        if self.method_input.case_token not in self.invocation_id:
            raise ValueError("truth-blind invocation and case token differ")


@dataclass(frozen=True, slots=True)
class PrivilegedResponseAlgebraOracle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/privileged-response-algebra-oracle'

    oracle_id: str
    case_token: str
    truth_kind: str
    port_materiality: PortMateriality
    temporal_composition: TemporalComposition
    simultaneous_composition: SimultaneousComposition
    sequential_composition: SequentialComposition
    state_sufficiency: StateSufficiency
    representation: AlgebraRepresentation
    receiver_visibility: ReceiverVisibility
    pair_label: PairScientificLabel
    smooth_lie_eligible: bool
    false_positive_guard: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.oracle_id, field_name="oracle_id")
        validate_stable_id(self.case_token, field_name="case_token")
        validate_stable_id(self.truth_kind, field_name="truth_kind")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraCaseScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/response-algebra-case-score'

    score_id: str
    case_token: str
    passed: bool
    mismatched_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.case_token, field_name="case_token")
        require_sorted_unique_strings(self.mismatched_field_ids, field_name="mismatched_field_ids")
        if self.passed == bool(self.mismatched_field_ids):
            raise ValueError("case score pass flag and mismatches disagree")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraConformanceScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/response-algebra-conformance-score'

    score_id: str
    case_scores: tuple[ResponseAlgebraCaseScore, ...]
    passed_cases: int
    failed_cases: int
    false_positive_noncommutativity: int
    missed_material_noncommutativity: int
    minimum_required_passed_cases: int
    maximum_allowed_false_positives: int
    maximum_allowed_missed_material_cases: int
    gate_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        require_sorted_unique_ids(self.case_scores, attribute="score_id", field_name="case_scores")
        if self.passed_cases + self.failed_cases != len(self.case_scores):
            raise ValueError("conformance case counts do not close")
        expected = (
            self.passed_cases >= self.minimum_required_passed_cases
            and self.false_positive_noncommutativity <= self.maximum_allowed_false_positives
            and self.missed_material_noncommutativity <= self.maximum_allowed_missed_material_cases
        )
        if self.gate_passed != expected:
            raise ValueError("conformance gate flag differs from frozen thresholds")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraConformanceEvidenceSummary(CanonicalRecord):
    """Compact revealed summary; raw generated inputs remain external and separate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/response-algebra-conformance-evidence-summary'

    summary_id: str
    run_id: str
    conformance_spec_sha256: str
    execution_package_sha256: str
    authorization_sha256: str
    compiled_preview_sha256: str
    implementation_commit: str
    method_implementation_sha256: str
    results: tuple[ResponseAlgebraMethodResult, ...]
    oracle_sha256_by_case: tuple[tuple[str, str], ...]
    score: ResponseAlgebraConformanceScore
    scientific_ceiling: EvidenceCeiling
    claim_promotion_allowed: bool
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.run_id, field_name="run_id")
        for name, digest in (
            ("conformance_spec_sha256", self.conformance_spec_sha256),
            ("execution_package_sha256", self.execution_package_sha256),
            ("authorization_sha256", self.authorization_sha256),
            ("compiled_preview_sha256", self.compiled_preview_sha256),
            ("method_implementation_sha256", self.method_implementation_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.implementation_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_commit
        ):
            raise ValueError("summary implementation commit must be a Git SHA-1")
        require_sorted_unique_ids(self.results, attribute="result_id", field_name="results")
        oracle_tokens = tuple(value[0] for value in self.oracle_sha256_by_case)
        if tuple(sorted(set(oracle_tokens))) != oracle_tokens:
            raise ValueError("oracle fingerprints must have sorted unique case tokens")
        for case_token, digest in self.oracle_sha256_by_case:
            validate_stable_id(case_token, field_name="oracle_sha256_by_case")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("oracle fingerprint must be a lowercase SHA-256")
        result_tokens = tuple(value.case_token for value in self.results)
        score_tokens = tuple(value.case_token for value in self.score.case_scores)
        if result_tokens != oracle_tokens or result_tokens != score_tokens:
            raise ValueError("summary result, oracle and score case sets differ")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("truth-known method calibration must remain non-promotable")
        if self.claim_promotion_allowed:
            raise ValueError("method conformance cannot promote a substrate claim")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        if not {"llm", "rl"} <= set(self.prohibited_component_ids):
            raise ValueError("conformance summary must retain LLM and RL prohibitions")


@dataclass(frozen=True)
class _CaseDefinition:
    token: str
    truth_kind: str
    dimension: int
    primary_indices: tuple[int, ...]
    augmented_indices: tuple[int, ...] | None
    full_indices: tuple[int, ...] | None
    word_kind: str
    temporal_kind: str
    stochastic: bool = False
    smooth: bool = False
    hybrid: bool = False


_CASES = (
    _CaseDefinition("case-001", "disconnected-null", 2, (0, 1), None, None, "null", "stationary"),
    _CaseDefinition(
        "case-002", "lti-timing-counterfeit", 2, (0, 1), None, None, "lti", "stationary"
    ),
    _CaseDefinition(
        "case-003", "affine-commuting", 2, (0, 1), None, None, "commuting", "stationary"
    ),
    _CaseDefinition(
        "case-004", "nonlinear-commuting", 2, (0, 1), None, None, "nonlinear", "stationary"
    ),
    _CaseDefinition(
        "case-005",
        "heisenberg-smooth",
        3,
        (0, 1, 2),
        None,
        None,
        "heisenberg",
        "stationary",
        smooth=True,
    ),
    _CaseDefinition("case-006", "hidden-reservoir", 2, (1,), (0, 1), None, "hidden", "hidden"),
    _CaseDefinition("case-007", "time-varying-bath", 2, (0, 1), None, None, "commuting", "bath"),
    _CaseDefinition(
        "case-008",
        "stochastic-kernel",
        3,
        (0, 1, 2),
        None,
        None,
        "stochastic",
        "none",
        stochastic=True,
    ),
    _CaseDefinition(
        "case-009",
        "receiver-hidden-bracket",
        3,
        (0, 1),
        None,
        (0, 1, 2),
        "heisenberg",
        "stationary",
        smooth=True,
    ),
    _CaseDefinition(
        "case-010", "hybrid-threshold", 1, (0,), None, None, "hybrid", "hybrid", hybrid=True
    ),
)

_WORD_IDS = (
    "a-early",
    "a-late",
    "a-repeat",
    "a-then-b",
    "b-early",
    "b-late",
    "b-repeat",
    "b-then-a",
    "identity",
    "simultaneous-a-b",
)


def reference_response_algebra_conformance_spec() -> ResponseAlgebraConformanceSpec:
    return ResponseAlgebraConformanceSpec(
        conformance_id="response-algebra-method-conformance",
        case_tokens=tuple(value.token for value in _CASES),
        development_word_units_per_case=12,
        evaluation_word_units_per_case=12,
        development_temporal_units_per_case=18,
        evaluation_temporal_units_per_case=18,
        generator_units_per_split_per_case=18,
        master_seed=20260718,
        bootstrap_replicates=1000,
        bootstrap_seed_base=20261718,
        floor_upper=Decimal("0.001"),
        equivalence_width=Decimal("0.02"),
        minimum_material_effect=Decimal("0.08"),
        maximum_affine_closure_error=Decimal("0.02"),
        maximum_stationarity_error=Decimal("0.03"),
        minimum_wrong_horizon_ratio=Decimal("3"),
        minimum_stochastic_commutator=Decimal("0.08"),
        maximum_dose_scaling_relative_error=Decimal("0.08"),
        minimum_required_passed_cases=10,
        maximum_allowed_false_positives=0,
        maximum_allowed_missed_material_cases=0,
        prohibited_component_ids=("llm", "rl"),
    )


def _coordinate_ids(dimension: int) -> tuple[str, ...]:
    return tuple(
        ("reservoir", "x") if dimension == 2 else ("x",) if dimension == 1 else ("x", "y", "z")
    )


def _dose_id(dose: Decimal) -> str:
    return str(dose).replace("-", "m").replace(".", "p")


def _delivered_record_id(unit_id: str, word_id: str, dose: Decimal) -> str:
    return f"delivered.{unit_id}.{word_id}.dose-{_dose_id(dose)}"


def _stage(stage: ActionStage, dose: Decimal, clock: Decimal) -> ActionStageValue:
    return ActionStageValue(
        stage=stage,
        value=dose,
        native_unit="1",
        clock_id="reference-clock",
        clock_coordinate=clock,
    )


def _letter(letter_id: str, dose: Decimal, clock: Decimal) -> ActionLetter:
    return ActionLetter(
        letter_id=letter_id,
        port_id=f"port-{letter_id}",
        requested=_stage(ActionStage.REQUESTED, dose, clock),
        accepted=_stage(ActionStage.ACCEPTED, dose, clock),
        applied=_stage(ActionStage.APPLIED, dose, clock),
        realized=_stage(ActionStage.REALIZED, dose, clock),
        duration=Decimal("1"),
        duration_unit="s",
        support_id=f"support-{letter_id}",
    )


def _delivered_word(
    definition: _CaseDefinition,
    split: DataSplit,
    unit_id: str,
    family_word_id: str,
    dose: Decimal,
) -> DeliveredWordRecord:
    early = Decimal("0")
    late = Decimal("2")
    if family_word_id == "identity":
        mode = ActionWordMode.IDENTITY
        letters: tuple[ActionLetter, ...] = ()
    elif family_word_id == "a-early":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("a", dose, early),)
    elif family_word_id == "a-late":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("a", dose, late),)
    elif family_word_id == "b-early":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("b", dose, early),)
    elif family_word_id == "b-late":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("b", dose, late),)
    elif family_word_id == "a-repeat":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("a", dose, early), _letter("a", dose, late))
    elif family_word_id == "b-repeat":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("b", dose, early), _letter("b", dose, late))
    elif family_word_id == "a-then-b":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("a", dose, early), _letter("b", dose, late))
    elif family_word_id == "b-then-a":
        mode = ActionWordMode.SEQUENTIAL
        letters = (_letter("b", dose, early), _letter("a", dose, late))
    elif family_word_id == "simultaneous-a-b":
        mode = ActionWordMode.SIMULTANEOUS
        letters = (_letter("a", dose, early), _letter("b", dose, early))
    else:
        raise ValueError(family_word_id)
    record_id = _delivered_record_id(unit_id, family_word_id, dose)
    action_word = LetterActionWord(
        word_id=f"action-word.{record_id}",
        mode=mode,
        letters=letters,
        denominator_id="denominator-reference",
        retained_history_id=(
            "history-reservoir" if definition.word_kind == "hidden" else "history-zero"
        ),
        receiver_id="receiver-bundle",
        horizon_id="horizon-two-slot",
        prefix_support_ids=tuple(
            f"prefix.{record_id}.{index}" for index in range(len(letters) + 1)
        ),
        supported=True,
        reason_codes=(),
    )
    return DeliveredWordRecord(
        record_id=record_id,
        independent_unit_id=unit_id,
        split=split,
        family_word_id=family_word_id,
        dose=dose,
        delivered_word=action_word,
    )


def _delivered_words(
    definition: _CaseDefinition,
    spec: ResponseAlgebraConformanceSpec,
) -> tuple[DeliveredWordRecord, ...]:
    values = []
    for split in DataSplit:
        unit_count = (
            spec.development_word_units_per_case
            if split is DataSplit.CALIBRATION
            else spec.evaluation_word_units_per_case
        )
        for unit_index in range(unit_count):
            unit_id = f"{split.value.lower()}-unit-{unit_index:03d}"
            for word_id in _WORD_IDS:
                values.append(_delivered_word(definition, split, unit_id, word_id, Decimal("1")))
            if definition.smooth:
                for dose in (Decimal("0.05"), Decimal("0.1"), Decimal("0.2")):
                    for word_id in ("a-then-b", "b-then-a"):
                        values.append(_delivered_word(definition, split, unit_id, word_id, dose))
    return tuple(sorted(values, key=lambda value: value.record_id))


def _delivered_generator_record_id(
    unit_id: str,
    action_id: str,
    signed_dose: Decimal,
) -> str:
    return f"generator-delivered.{unit_id}.{action_id}.dose-{_dose_id(signed_dose)}"


def _delivered_generators(
    definition: _CaseDefinition,
    spec: ResponseAlgebraConformanceSpec,
) -> tuple[DeliveredGeneratorRecord, ...]:
    if not definition.smooth:
        return ()
    values = []
    for split in DataSplit:
        for unit_index in range(spec.generator_units_per_split_per_case):
            unit_id = f"{split.value.lower()}-generator-unit-{unit_index:03d}"
            for signed_dose in (
                Decimal("-0.2"),
                Decimal("-0.1"),
                Decimal("-0.05"),
                Decimal("0.05"),
                Decimal("0.1"),
                Decimal("0.2"),
            ):
                for action_id in ("a", "b"):
                    record_id = _delivered_generator_record_id(
                        unit_id,
                        action_id,
                        signed_dose,
                    )
                    values.append(
                        DeliveredGeneratorRecord(
                            record_id=record_id,
                            independent_unit_id=unit_id,
                            split=split,
                            action_id=action_id,
                            signed_dose=signed_dose,
                            delivered_letter=_letter(
                                action_id,
                                signed_dose,
                                Decimal("0"),
                            ),
                        )
                    )
    return tuple(sorted(values, key=lambda value: value.record_id))


def _view(
    view_id: str,
    role: StateViewRole,
    full_coordinate_ids: tuple[str, ...],
    indices: tuple[int, ...],
) -> StateViewSpec:
    coordinate_ids = tuple(full_coordinate_ids[index] for index in indices)
    if tuple(sorted(coordinate_ids)) != coordinate_ids:
        raise ValueError("reference state-view projection must retain canonical coordinate order")
    return StateViewSpec(
        view_id=view_id,
        role=role,
        coordinate_ids=coordinate_ids,
        native_units=tuple("1" for _value in coordinate_ids),
    )


def _state(
    vector_id: str,
    view: StateViewSpec,
    full_state: npt.NDArray[np.float64],
    full_coordinate_ids: tuple[str, ...],
) -> StateVector:
    index = {value: position for position, value in enumerate(full_coordinate_ids)}
    return StateVector(
        vector_id=vector_id,
        view_id=view.view_id,
        values=tuple(
            NamedDecimal(
                value_id=coordinate_id,
                value=Decimal(f"{float(full_state[index[coordinate_id]]):.15g}"),
                unit="1",
            )
            for coordinate_id in view.coordinate_ids
        ),
    )


def _flow_a(state: npt.NDArray[np.float64], dose: float) -> npt.NDArray[np.float64]:
    result = state.copy()
    result[0] += dose
    return result


def _flow_b(state: npt.NDArray[np.float64], dose: float) -> npt.NDArray[np.float64]:
    result = state.copy()
    result[1] += dose
    if result.size >= 3:
        result[2] += dose * state[0]
    return result


def _generic_word(
    definition: _CaseDefinition,
    word_id: str,
    initial: npt.NDArray[np.float64],
    dose: float,
) -> npt.NDArray[np.float64]:
    if definition.word_kind == "null":
        return initial.copy()
    if definition.word_kind == "lti":
        effects = {
            "identity": (0.0, 0.0),
            "a-early": (0.30, 0.0),
            "a-late": (0.16, 0.0),
            "b-early": (0.0, 0.42),
            "b-late": (0.0, 0.20),
            "a-repeat": (0.46, 0.0),
            "b-repeat": (0.0, 0.62),
            "a-then-b": (0.30, 0.20),
            "b-then-a": (0.16, 0.42),
            "simultaneous-a-b": (0.30, 0.42),
        }
        return initial + np.asarray(effects[word_id]) * dose
    if definition.word_kind in {"commuting", "nonlinear"}:
        a = np.asarray((0.25, 0.0)) * dose
        b = np.asarray((0.0, 0.30)) * dose
        values = {
            "identity": initial,
            "a-early": initial + a,
            "a-late": initial + a,
            "b-early": initial + b,
            "b-late": initial + b,
            "a-repeat": initial + 2.0 * a,
            "b-repeat": initial + 2.0 * b,
            "a-then-b": initial + a + b,
            "b-then-a": initial + a + b,
            "simultaneous-a-b": initial + a + b,
        }
        if definition.word_kind == "nonlinear":
            values["simultaneous-a-b"] = values["simultaneous-a-b"] + np.asarray(
                (0.20 * dose * dose, 0.0)
            )
        return values[word_id].copy()
    if definition.word_kind == "heisenberg":
        if word_id == "identity":
            return initial.copy()
        if word_id in {"a-early", "a-late"}:
            return _flow_a(initial, dose)
        if word_id in {"b-early", "b-late"}:
            return _flow_b(initial, dose)
        if word_id == "a-repeat":
            return _flow_a(_flow_a(initial, dose), dose)
        if word_id == "b-repeat":
            return _flow_b(_flow_b(initial, dose), dose)
        if word_id == "a-then-b":
            return _flow_b(_flow_a(initial, dose), dose)
        if word_id == "b-then-a":
            return _flow_a(_flow_b(initial, dose), dose)
        if word_id == "simultaneous-a-b":
            result = initial.copy()
            result[0] += dose
            result[1] += dose
            result[2] += dose * initial[0] + 0.5 * dose * dose
            return result
    if definition.word_kind == "hidden":

        def action_a(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            result = value.copy()
            result[1] += 0.25 + 0.35 * value[0]
            return result

        def action_b(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            result = value.copy()
            result[0] += 0.40
            result[1] += 0.12
            return result

        if word_id == "identity":
            return initial.copy()
        if word_id in {"a-early", "a-late"}:
            return action_a(initial)
        if word_id in {"b-early", "b-late"}:
            return action_b(initial)
        if word_id == "a-repeat":
            return action_a(action_a(initial))
        if word_id == "b-repeat":
            return action_b(action_b(initial))
        if word_id == "a-then-b":
            return action_b(action_a(initial))
        if word_id == "b-then-a":
            return action_a(action_b(initial))
        if word_id == "simultaneous-a-b":
            return action_a(initial) + action_b(initial) - initial
    if definition.word_kind == "hybrid":

        def action_a(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return value + 0.7

        def action_b(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray((-0.35,)) if value[0] >= 0.5 else value - 0.15

        if word_id == "identity":
            return initial.copy()
        if word_id in {"a-early", "a-late"}:
            return action_a(initial)
        if word_id in {"b-early", "b-late"}:
            return action_b(initial)
        if word_id == "a-repeat":
            return action_a(action_a(initial))
        if word_id == "b-repeat":
            return action_b(action_b(initial))
        if word_id == "a-then-b":
            return action_b(action_a(initial))
        if word_id == "b-then-a":
            return action_a(action_b(initial))
        if word_id == "simultaneous-a-b":
            return action_b(action_a(initial))
    raise ValueError(f"unsupported reference word: {definition.word_kind}/{word_id}")


def _stochastic_word(
    word_id: str,
    initial: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    a = np.asarray(((1, 0, 0), (0, 0, 1), (0, 1, 0)), dtype=np.float64)
    b = np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=np.float64)
    if word_id == "identity":
        return initial.copy()
    if word_id in {"a-early", "a-late"}:
        return initial @ a
    if word_id in {"b-early", "b-late"}:
        return initial @ b
    if word_id == "a-repeat":
        return initial @ a @ a
    if word_id == "b-repeat":
        return initial @ b @ b
    if word_id == "a-then-b":
        return initial @ a @ b
    if word_id == "b-then-a":
        return initial @ b @ a
    if word_id == "simultaneous-a-b":
        return initial @ ((a + b) / 2.0)
    raise ValueError(word_id)


def _initial_state(
    definition: _CaseDefinition,
    split: DataSplit,
    index: int,
    spec: ResponseAlgebraConformanceSpec,
) -> npt.NDArray[np.float64]:
    seed = spec.master_seed + int(definition.token[-3:]) * 100 + index
    if split is DataSplit.HELD_OUT:
        seed += 50_000
    generator = np.random.default_rng(seed)
    if definition.stochastic:
        value = np.zeros(3, dtype=np.float64)
        value[index % 3] = 1.0
        return value
    return generator.uniform(-0.45, 0.45, size=definition.dimension).astype(np.float64)


def _temporal_maps(
    definition: _CaseDefinition,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    dimension = definition.dimension
    first = np.eye(dimension, dtype=np.float64) * 0.78
    second = first.copy()
    first_bias = np.linspace(0.04, 0.08, dimension, dtype=np.float64)
    second_bias = first_bias.copy()
    if definition.temporal_kind == "hidden":
        first = np.asarray(((0.90, 0.0), (0.48, 0.72)), dtype=np.float64)
        second = first.copy()
        first_bias = np.asarray((0.03, 0.08))
        second_bias = first_bias.copy()
    elif definition.temporal_kind == "bath":
        second = np.eye(dimension, dtype=np.float64) * 1.08
        second_bias = np.linspace(-0.06, -0.02, dimension, dtype=np.float64)
    return first, first_bias, second, second_bias


def _case_views(definition: _CaseDefinition) -> tuple[StateViewSpec, ...]:
    coordinates = _coordinate_ids(definition.dimension)
    values = [
        _view(
            "receiver-view", StateViewRole.PRIMARY_RECEIVER, coordinates, definition.primary_indices
        )
    ]
    if definition.augmented_indices is not None:
        values.append(
            _view(
                "augmented-view",
                StateViewRole.AUGMENTED_STATE,
                coordinates,
                definition.augmented_indices,
            )
        )
    if definition.full_indices is not None:
        values.append(
            _view("full-state-view", StateViewRole.FULL_STATE, coordinates, definition.full_indices)
        )
    return tuple(sorted(values, key=lambda value: value.view_id))


def _word_responses(
    definition: _CaseDefinition,
    views: tuple[StateViewSpec, ...],
    spec: ResponseAlgebraConformanceSpec,
) -> tuple[WordResponse, ...]:
    coordinates = _coordinate_ids(definition.dimension)
    values: list[WordResponse] = []
    for split in DataSplit:
        unit_count = (
            spec.development_word_units_per_case
            if split is DataSplit.CALIBRATION
            else spec.evaluation_word_units_per_case
        )
        for unit_index in range(unit_count):
            unit_id = f"{split.value.lower()}-unit-{unit_index:03d}"
            initial = _initial_state(definition, split, unit_index, spec)
            for view in views:
                for word_id in _WORD_IDS:
                    final = (
                        _stochastic_word(word_id, initial)
                        if definition.stochastic
                        else _generic_word(definition, word_id, initial, 1.0)
                    )
                    prefix = f"{definition.token}.{unit_id}.{view.view_id}.{word_id}.dose-1000"
                    values.append(
                        WordResponse(
                            observation_id=f"word.{prefix}",
                            independent_unit_id=unit_id,
                            denominator_id="denominator-reference",
                            split=split,
                            numerical_view_id="fine-view",
                            state_view_id=view.view_id,
                            word_id=word_id,
                            dose=Decimal("1"),
                            delivered_word_record_id=_delivered_record_id(
                                unit_id,
                                word_id,
                                Decimal("1"),
                            ),
                            initial_state=_state(f"initial.{prefix}", view, initial, coordinates),
                            final_state=_state(f"final.{prefix}", view, final, coordinates),
                        )
                    )
                if (
                    definition.smooth
                    and view.role is not StateViewRole.PRIMARY_RECEIVER
                    or (definition.smooth and definition.full_indices is None)
                ):
                    for dose in (Decimal("0.05"), Decimal("0.1"), Decimal("0.2")):
                        for word_id in ("a-then-b", "b-then-a"):
                            final = _generic_word(
                                definition,
                                word_id,
                                initial,
                                float(dose),
                            )
                            dose_id = str(dose).replace(".", "p")
                            prefix = f"{definition.token}.{unit_id}.{view.view_id}.{word_id}.dose-{dose_id}"
                            values.append(
                                WordResponse(
                                    observation_id=f"word.{prefix}",
                                    independent_unit_id=unit_id,
                                    denominator_id="denominator-reference",
                                    split=split,
                                    numerical_view_id="fine-view",
                                    state_view_id=view.view_id,
                                    word_id=word_id,
                                    dose=dose,
                                    delivered_word_record_id=_delivered_record_id(
                                        unit_id,
                                        word_id,
                                        dose,
                                    ),
                                    initial_state=_state(
                                        f"initial.{prefix}", view, initial, coordinates
                                    ),
                                    final_state=_state(f"final.{prefix}", view, final, coordinates),
                                )
                            )
    return tuple(sorted(values, key=lambda value: value.observation_id))


def _affine_transitions(
    definition: _CaseDefinition,
    views: tuple[StateViewSpec, ...],
    spec: ResponseAlgebraConformanceSpec,
) -> tuple[AffineTransition, ...]:
    if definition.temporal_kind == "none":
        return ()
    coordinates = _coordinate_ids(definition.dimension)
    first, first_bias, second, second_bias = _temporal_maps(definition)
    values: list[AffineTransition] = []
    for split in DataSplit:
        unit_count = (
            spec.development_temporal_units_per_case
            if split is DataSplit.CALIBRATION
            else spec.evaluation_temporal_units_per_case
        )
        for unit_index in range(unit_count):
            unit_id = f"{split.value.lower()}-temporal-unit-{unit_index:03d}"
            start = _initial_state(definition, split, 100 + unit_index, spec)
            if definition.temporal_kind == "hybrid":
                middle = 0.72 * start + np.where(start >= 0, 0.24, -0.11)
                end = 0.72 * middle + np.where(middle >= 0, 0.24, -0.11)
            else:
                middle = first @ start + first_bias
                end = second @ middle + second_bias
            for view in views:
                for edge_id, start_anchor, end_anchor, duration, left, right in (
                    ("01", "t0", "t1", "1", start, middle),
                    ("02", "t0", "t2", "2", start, end),
                    ("12", "t1", "t2", "1", middle, end),
                ):
                    denominator = (
                        f"bath-{edge_id}"
                        if definition.temporal_kind == "bath"
                        else "denominator-reference"
                    )
                    prefix = f"{definition.token}.{unit_id}.{view.view_id}.{edge_id}"
                    values.append(
                        AffineTransition(
                            transition_id=f"temporal.{prefix}",
                            independent_unit_id=unit_id,
                            denominator_id=denominator,
                            split=split,
                            numerical_view_id="fine-view",
                            state_view_id=view.view_id,
                            start_anchor_id=start_anchor,
                            end_anchor_id=end_anchor,
                            duration=Decimal(duration),
                            duration_unit="s",
                            start_state=_state(f"start.{prefix}", view, left, coordinates),
                            end_state=_state(f"end.{prefix}", view, right, coordinates),
                        )
                    )
    return tuple(sorted(values, key=lambda value: value.transition_id))


def _discrete_transitions(definition: _CaseDefinition) -> tuple[DiscreteTransition, ...]:
    if not definition.stochastic:
        return ()
    a_end = (0, 2, 1)
    b_end = (1, 2, 0)
    values = []
    for split in DataSplit:
        for action_id, mapping in (("a", a_end), ("b", b_end)):
            for state_index, end_index in enumerate(mapping):
                for replicate in range(20):
                    prefix = f"{definition.token}.{split.value.lower()}.{action_id}.{state_index}.{replicate}"
                    values.append(
                        DiscreteTransition(
                            transition_id=f"discrete.{prefix}",
                            independent_unit_id=f"population-{split.value.lower()}-{replicate:03d}",
                            split=split,
                            action_id=action_id,
                            start_state_id=f"state-{state_index}",
                            end_state_id=f"state-{end_index}",
                        )
                    )
    return tuple(sorted(values, key=lambda value: value.transition_id))


def _generator_transitions(
    definition: _CaseDefinition,
    views: tuple[StateViewSpec, ...],
    spec: ResponseAlgebraConformanceSpec,
) -> tuple[GeneratorTransition, ...]:
    if not definition.smooth:
        return ()
    coordinates = _coordinate_ids(definition.dimension)
    target_view = next(
        (value for value in views if value.role is StateViewRole.FULL_STATE),
        next(value for value in views if value.role is StateViewRole.PRIMARY_RECEIVER),
    )
    values = []
    for split in DataSplit:
        for unit_index in range(spec.generator_units_per_split_per_case):
            unit_id = f"{split.value.lower()}-generator-unit-{unit_index:03d}"
            initial = _initial_state(definition, split, 300 + unit_index, spec)
            for dose in (
                Decimal("-0.2"),
                Decimal("-0.1"),
                Decimal("-0.05"),
                Decimal("0.05"),
                Decimal("0.1"),
                Decimal("0.2"),
            ):
                for action_id, flow in (("a", _flow_a), ("b", _flow_b)):
                    final = flow(initial, float(dose))
                    dose_id = str(dose).replace("-", "m").replace(".", "p")
                    prefix = f"{definition.token}.{unit_id}.{action_id}.{dose_id}"
                    values.append(
                        GeneratorTransition(
                            transition_id=f"generator.{prefix}",
                            independent_unit_id=unit_id,
                            split=split,
                            action_id=action_id,
                            signed_dose=dose,
                            delivered_generator_record_id=_delivered_generator_record_id(
                                unit_id,
                                action_id,
                                dose,
                            ),
                            state_view_id=target_view.view_id,
                            start_state=_state(
                                f"generator-start.{prefix}", target_view, initial, coordinates
                            ),
                            end_state=_state(
                                f"generator-end.{prefix}", target_view, final, coordinates
                            ),
                        )
                    )
    return tuple(sorted(values, key=lambda value: value.transition_id))


def _config(
    definition: _CaseDefinition,
    views: tuple[StateViewSpec, ...],
    spec: ResponseAlgebraConformanceSpec,
) -> ResponseAlgebraMethodConfig:
    coordinates = tuple(
        sorted({coordinate for view in views for coordinate in view.coordinate_ids})
    )
    units = {coordinate: "1" for coordinate in coordinates}
    primary = next(value for value in views if value.role is StateViewRole.PRIMARY_RECEIVER)
    augmented = next(
        (value for value in views if value.role is StateViewRole.AUGMENTED_STATE),
        None,
    )
    full = next((value for value in views if value.role is StateViewRole.FULL_STATE), None)
    return ResponseAlgebraMethodConfig(
        config_id=f"method-config.{definition.token}",
        method_key="response-algebra.direct-affine-finite",
        method_version="1.0.0",
        primary_numerical_view_id="fine-view",
        primary_state_view_id=primary.view_id,
        augmented_state_view_id=augmented.view_id if augmented is not None else None,
        full_state_view_id=full.view_id if full is not None else None,
        receiver_criteria=tuple(
            ReceiverEquivalenceCriterion(
                criterion_id=f"criterion.{coordinate}",
                coordinate_id=coordinate,
                native_unit=units[coordinate],
                floor_upper=spec.floor_upper,
                equivalence_width=spec.equivalence_width,
                materiality_lower=spec.minimum_material_effect,
            )
            for coordinate in coordinates
        ),
        confidence_level=Decimal("0.95"),
        bootstrap_replicates=spec.bootstrap_replicates,
        bootstrap_seed=spec.bootstrap_seed_base + int(definition.token[-3:]),
        affine_ridge=Decimal("0.000000000001"),
        maximum_affine_closure_error=spec.maximum_affine_closure_error,
        maximum_stationarity_error=spec.maximum_stationarity_error,
        minimum_wrong_horizon_ratio=spec.minimum_wrong_horizon_ratio,
        minimum_stochastic_commutator=spec.minimum_stochastic_commutator,
        maximum_dose_scaling_relative_error=spec.maximum_dose_scaling_relative_error,
        minimum_independent_units=8,
        require_exact_stage_value_equality=True,
        require_exact_stage_clock_equality=True,
    )


def truth_blind_response_algebra_invocations(
    spec: ResponseAlgebraConformanceSpec | None = None,
) -> tuple[TruthBlindResponseAlgebraInvocation, ...]:
    """Return generated inputs/configs without importing or embedding truth labels."""

    frozen = spec or reference_response_algebra_conformance_spec()
    if frozen.case_tokens != tuple(value.token for value in _CASES):
        raise ValueError("conformance spec case tokens differ from the static fixture registry")
    values = []
    for definition in _CASES:
        views = _case_views(definition)
        method_input = ResponseAlgebraMethodInput(
            case_token=definition.token,
            state_views=views,
            delivered_words=_delivered_words(definition, frozen),
            delivered_generators=_delivered_generators(definition, frozen),
            word_responses=_word_responses(definition, views, frozen),
            affine_transitions=_affine_transitions(definition, views, frozen),
            discrete_transitions=_discrete_transitions(definition),
            generator_transitions=_generator_transitions(definition, views, frozen),
            two_sided_actions_delivered=definition.smooth,
            local_state_closed=definition.temporal_kind not in {"bath", "hybrid"},
            hybrid_switching_observed=definition.hybrid,
            evaluator_reveal_attestation_id=f"evaluator-attestation.{definition.token}",
        )
        values.append(
            TruthBlindResponseAlgebraInvocation(
                invocation_id=f"invocation.{definition.token}",
                method_input=method_input,
                method_config=_config(definition, views, frozen),
            )
        )
    return tuple(sorted(values, key=lambda value: value.invocation_id))


def privileged_response_algebra_oracles() -> tuple[PrivilegedResponseAlgebraOracle, ...]:
    """Return scorer-only truth; method implementations must never import this function."""

    values = (
        (
            "case-001",
            "disconnected-null",
            PortMateriality.NULL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.BELOW_RESOLUTION,
            SequentialComposition.BELOW_RESOLUTION,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.NO_MATERIAL_RESPONSE,
            False,
            True,
        ),
        (
            "case-002",
            "lti-timing-counterfeit",
            PortMateriality.MATERIAL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.ADDITIVE_EQUIVALENT,
            SequentialComposition.LTI_OR_DRIFT_EXPLAINED,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.PATH_ORDER_EXPLAINED_BY_LTI_OR_DRIFT,
            False,
            True,
        ),
        (
            "case-003",
            "affine-commuting",
            PortMateriality.MATERIAL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.ADDITIVE_EQUIVALENT,
            SequentialComposition.COMMUTATOR_EQUIVALENT,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.MATERIAL_AFFINE_COMMUTATIVE,
            False,
            False,
        ),
        (
            "case-004",
            "nonlinear-commuting",
            PortMateriality.MATERIAL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.NONLINEAR_INTERACTION,
            SequentialComposition.COMMUTATOR_EQUIVALENT,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.NONLINEAR_COMMUTATIVE,
            False,
            False,
        ),
        (
            "case-005",
            "heisenberg-smooth",
            PortMateriality.MATERIAL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.NONLINEAR_INTERACTION,
            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_SMOOTH,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.SMOOTH_NONCOMMUTATIVE_LIE_LOCAL,
            True,
            False,
        ),
        (
            "case-006",
            "hidden-reservoir",
            PortMateriality.MATERIAL,
            TemporalComposition.HISTORY_AUGMENTED_CLOSURE,
            SimultaneousComposition.ADDITIVE_EQUIVALENT,
            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
            StateSufficiency.FINITE_HISTORY_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.GAUGE_DEPENDENT,
            PairScientificLabel.FINITE_NONCOMMUTATIVE_CLOSED,
            False,
            False,
        ),
        (
            "case-007",
            "time-varying-bath",
            PortMateriality.MATERIAL,
            TemporalComposition.NONSTATIONARY_COCYCLE,
            SimultaneousComposition.ADDITIVE_EQUIVALENT,
            SequentialComposition.COMMUTATOR_EQUIVALENT,
            StateSufficiency.TIME_VARYING_DENOMINATOR,
            AlgebraRepresentation.DETERMINISTIC_FINITE,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.MATERIAL_AFFINE_COMMUTATIVE,
            False,
            False,
        ),
        (
            "case-008",
            "stochastic-kernel",
            PortMateriality.MATERIAL,
            TemporalComposition.UNEVALUABLE,
            SimultaneousComposition.NONLINEAR_INTERACTION,
            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.STOCHASTIC_KERNEL,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.STOCHASTIC_NONCOMMUTATIVE,
            False,
            False,
        ),
        (
            "case-009",
            "receiver-hidden-bracket",
            PortMateriality.MATERIAL,
            TemporalComposition.STATIONARY_SEMIGROUP,
            SimultaneousComposition.ADDITIVE_EQUIVALENT,
            SequentialComposition.COMMUTATOR_EQUIVALENT,
            StateSufficiency.MARKOV_SUFFICIENT,
            AlgebraRepresentation.DETERMINISTIC_SMOOTH,
            ReceiverVisibility.BRACKET_HIDDEN_BY_PROJECTION,
            PairScientificLabel.MATERIAL_AFFINE_COMMUTATIVE,
            True,
            False,
        ),
        (
            "case-010",
            "hybrid-threshold",
            PortMateriality.MATERIAL,
            TemporalComposition.NONCLOSED,
            SimultaneousComposition.NONLINEAR_INTERACTION,
            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
            StateSufficiency.UNEVALUABLE,
            AlgebraRepresentation.HYBRID_SWITCHING,
            ReceiverVisibility.FAITHFUL_AT_TESTED_RESOLUTION,
            PairScientificLabel.HYBRID_NONCOMMUTATIVE,
            False,
            False,
        ),
    )
    return tuple(
        PrivilegedResponseAlgebraOracle(
            oracle_id=f"oracle.{value[0]}",
            case_token=value[0],
            truth_kind=value[1],
            port_materiality=value[2],
            temporal_composition=value[3],
            simultaneous_composition=value[4],
            sequential_composition=value[5],
            state_sufficiency=value[6],
            representation=value[7],
            receiver_visibility=value[8],
            pair_label=value[9],
            smooth_lie_eligible=value[10],
            false_positive_guard=value[11],
        )
        for value in values
    )


def score_response_algebra_conformance(
    results: tuple[ResponseAlgebraMethodResult, ...],
    oracles: tuple[PrivilegedResponseAlgebraOracle, ...],
    spec: ResponseAlgebraConformanceSpec | None = None,
) -> ResponseAlgebraConformanceScore:
    frozen = spec or reference_response_algebra_conformance_spec()
    require_sorted_unique_ids(results, attribute="result_id", field_name="results")
    require_sorted_unique_ids(oracles, attribute="oracle_id", field_name="oracles")
    result_by_token = {value.case_token: value for value in results}
    oracle_by_token = {value.case_token: value for value in oracles}
    if set(result_by_token) != set(oracle_by_token):
        raise ValueError("conformance result/oracle case sets differ")
    scores = []
    false_positives = 0
    misses = 0
    noncommutative = {
        SequentialComposition.MATERIAL_NONCOMMUTATIVE,
    }
    for case_token in sorted(result_by_token):
        result = result_by_token[case_token]
        oracle = oracle_by_token[case_token]
        observed = {
            "pair-label": result.pair_label,
            "port-materiality": result.signature.port_materiality,
            "receiver-visibility": result.signature.receiver_visibility,
            "representation": result.signature.representation,
            "sequential-composition": result.signature.sequential_composition,
            "simultaneous-composition": result.signature.simultaneous_composition,
            "smooth-lie-eligible": result.smooth_lie_eligible,
            "state-sufficiency": result.signature.state_sufficiency,
            "temporal-composition": result.signature.temporal_composition,
        }
        expected = {
            "pair-label": oracle.pair_label,
            "port-materiality": oracle.port_materiality,
            "receiver-visibility": oracle.receiver_visibility,
            "representation": oracle.representation,
            "sequential-composition": oracle.sequential_composition,
            "simultaneous-composition": oracle.simultaneous_composition,
            "smooth-lie-eligible": oracle.smooth_lie_eligible,
            "state-sufficiency": oracle.state_sufficiency,
            "temporal-composition": oracle.temporal_composition,
        }
        mismatches = tuple(sorted(key for key in expected if observed[key] != expected[key]))
        scores.append(
            ResponseAlgebraCaseScore(
                score_id=f"score.{case_token}",
                case_token=case_token,
                passed=not mismatches,
                mismatched_field_ids=mismatches,
            )
        )
        if (
            oracle.false_positive_guard
            and result.signature.sequential_composition in noncommutative
        ):
            false_positives += 1
        if (
            oracle.sequential_composition is SequentialComposition.MATERIAL_NONCOMMUTATIVE
            and result.signature.sequential_composition not in noncommutative
        ):
            misses += 1
    passed = sum(value.passed for value in scores)
    return ResponseAlgebraConformanceScore(
        score_id="response-algebra-conformance-score",
        case_scores=tuple(sorted(scores, key=lambda value: value.score_id)),
        passed_cases=passed,
        failed_cases=len(scores) - passed,
        false_positive_noncommutativity=false_positives,
        missed_material_noncommutativity=misses,
        minimum_required_passed_cases=frozen.minimum_required_passed_cases,
        maximum_allowed_false_positives=frozen.maximum_allowed_false_positives,
        maximum_allowed_missed_material_cases=frozen.maximum_allowed_missed_material_cases,
        gate_passed=(
            passed >= frozen.minimum_required_passed_cases
            and false_positives <= frozen.maximum_allowed_false_positives
            and misses <= frozen.maximum_allowed_missed_material_cases
        ),
    )
