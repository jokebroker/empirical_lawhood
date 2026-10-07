"Scientific records and pure computations for the truth-known response study."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Protocol

from empirical_lawhood.adapters.methods.contracts import DataSplit
from empirical_lawhood.adapters.methods.response_algebra import (
    ResponseAlgebraMethodInput,
    ResponseAlgebraMethodResult,
)
from empirical_lawhood.adapters.methods.response_algebra_protocol import (
    RESPONSE_ALGEBRA_EVALUATOR_KEY,
    ResponseAlgebraDevelopmentFreeze,
    ResponseAlgebraEvaluationOutput,
    calibration_only_response_algebra_input,
    evaluate_frozen_response_algebra,
    freeze_response_algebra_development,
)
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.response_algebra import (
    SignatureAxis,
    SignatureCriterion,
    StructuralClassDefinition,
    StructuralClassPrototype,
    TemporalComposition,
    prototype_class_definitions,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.response_algebra import (
    ActionAlphabetSpec,
    ActionWordDesignSpec,
    EquivalenceSpec,
    MultiplicityKind,
    MultiplicitySpec,
    NonadaptiveStoppingSpec,
    PlannedActionLetter,
    PlannedActionWord,
    ResponseAlgebraConformanceSpec,
    ResponseAlgebraProtocolSpec,
    StructuralClassHypothesis,
    WordExecutionMode,
)

from .response_algebra import (
    PrivilegedResponseAlgebraOracle,
    ResponseAlgebraConformanceScore,
    TruthBlindResponseAlgebraInvocation,
    privileged_response_algebra_oracles,
    reference_response_algebra_conformance_spec,
    score_response_algebra_conformance,
    truth_blind_response_algebra_invocations,
)


class _SplitRecord(Protocol):
    @property
    def independent_unit_id(self) -> str: ...

    @property
    def split(self) -> DataSplit: ...


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyConfig(CanonicalRecord):
    """Strict scientific config; it contains seeds and rules, never oracle labels."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-config'

    config_id: str
    response_algebra: ResponseAlgebraConformanceSpec
    response_algebra_implementation_sha256: str
    formalization_seed: int
    include_reference_action_case: bool
    include_reference_hold_case: bool
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(
            self.response_algebra_implementation_sha256,
            field_name="response_algebra_implementation_sha256",
        )
        if self.formalization_seed < 0:
            raise ValueError("truth-known formalization seed must be nonnegative")
        if not self.include_reference_action_case or not self.include_reference_hold_case:
            raise ValueError("truth-known qualification requires action and hold cases")
        if self.prohibited_component_ids != ("live-actuation", "llm", "rl"):
            raise ValueError("truth-known programme prohibitions differ from the frozen contract")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyDevelopmentCase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-development-case'

    case_id: str
    invocation: TruthBlindResponseAlgebraInvocation
    protocol: ResponseAlgebraProtocolSpec

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        if self.invocation.method_input.case_token != self.case_id:
            raise ValueError("Truth-known response study development case and invocation differ")
        if b'"split":"HELD_OUT"' in self.invocation.canonical_bytes():
            raise ValueError("Truth-known response study development case contains held-out records")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyDevelopmentBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-development-batch'

    batch_id: str
    cases: tuple[TruthKnownResponseStudyDevelopmentCase, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if len(self.cases) != 10:
            raise ValueError("Truth-known response study development batch requires ten cases")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudySealedEvaluationBatch(CanonicalRecord):
    """Full truth-blind inputs held behind the evaluation seal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-sealed-evaluation-batch'

    batch_id: str
    invocations: tuple[TruthBlindResponseAlgebraInvocation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.invocations,
            attribute="invocation_id",
            field_name="invocations",
        )
        if len(self.invocations) != 10:
            raise ValueError("Truth-known response study sealed evaluation batch requires ten cases")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyPrivilegedOracleBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-privileged-oracle-batch'

    batch_id: str
    oracles: tuple[PrivilegedResponseAlgebraOracle, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.oracles,
            attribute="oracle_id",
            field_name="oracles",
        )
        if len(self.oracles) != 10:
            raise ValueError("Truth-known response study oracle batch requires ten cases")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyFreezeBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-freeze-batch'

    batch_id: str
    freezes: tuple[ResponseAlgebraDevelopmentFreeze, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.freezes,
            attribute="freeze_id",
            field_name="freezes",
        )
        if len(self.freezes) != 10:
            raise ValueError("Truth-known response study freeze batch requires ten cases")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyMethodEvaluationBatch(CanonicalRecord):
    """Actual method outputs before privileged truth scoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-method-evaluation-batch'

    batch_id: str
    evaluations: tuple[ResponseAlgebraEvaluationOutput, ...]
    results: tuple[ResponseAlgebraMethodResult, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_ids(
            self.evaluations,
            attribute="evaluation_id",
            field_name="evaluations",
        )
        require_sorted_unique_ids(
            self.results,
            attribute="result_id",
            field_name="results",
        )
        if len(self.evaluations) != 10 or len(self.results) != 10:
            raise ValueError("Truth-known response study method evaluation requires ten cases")
        if (
            tuple(
                sorted(
                    (value.method_result for value in self.evaluations),
                    key=lambda value: value.result_id,
                )
            )
            != self.results
        ):
            raise ValueError("Truth-known response study method result projection differs from evaluations")


@dataclass(frozen=True, slots=True)
class TruthKnownResponseStudyAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/truth-known-response-study-adjudication'

    adjudication_id: str
    method_evaluation_sha256: str
    oracle_batch_sha256: str
    score: ResponseAlgebraConformanceScore
    claim_promotion_allowed: bool
    scientific_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_sha256(
            self.method_evaluation_sha256,
            field_name="method_evaluation_sha256",
        )
        validate_sha256(self.oracle_batch_sha256, field_name="oracle_batch_sha256")
        if self.claim_promotion_allowed:
            raise ValueError("Truth-known response study truth-known conformance cannot promote a scientific claim")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("Truth-known response study adjudication must remain non-promotable")


def default_truth_known_response_study_config(
    *,
    implementation_sha256: str,
    formalization_seed: int = 20260720,
) -> TruthKnownResponseStudyConfig:
    return TruthKnownResponseStudyConfig(
        config_id="truth-known-response-study",
        response_algebra=reference_response_algebra_conformance_spec(),
        response_algebra_implementation_sha256=implementation_sha256,
        formalization_seed=formalization_seed,
        include_reference_action_case=True,
        include_reference_hold_case=True,
        prohibited_component_ids=("live-actuation", "llm", "rl"),
    )


def _split_unit_ids(
    method_input: ResponseAlgebraMethodInput,
    split: DataSplit,
) -> tuple[str, ...]:
    records: tuple[_SplitRecord, ...] = (
        *method_input.delivered_words,
        *method_input.delivered_generators,
        *method_input.word_responses,
        *method_input.affine_transitions,
        *method_input.discrete_transitions,
        *method_input.generator_transitions,
    )
    return tuple(sorted({value.independent_unit_id for value in records if value.split is split}))


def _word_design(method_input: ResponseAlgebraMethodInput) -> ActionWordDesignSpec:
    letters = (
        PlannedActionLetter(
            letter_id="a",
            port_id="port-a",
            requested_value=Decimal("1"),
            native_unit="1",
            duration=Decimal("1"),
            duration_unit="s",
            clock_id="reference-clock",
            two_sided_partner_id=None,
        ),
        PlannedActionLetter(
            letter_id="b",
            port_id="port-b",
            requested_value=Decimal("1"),
            native_unit="1",
            duration=Decimal("1"),
            duration_unit="s",
            clock_id="reference-clock",
            two_sided_partner_id=None,
        ),
    )
    words = (
        PlannedActionWord("a-early", WordExecutionMode.SEQUENTIAL, ("a",), (Decimal("0"),)),
        PlannedActionWord("a-late", WordExecutionMode.SEQUENTIAL, ("a",), (Decimal("2"),)),
        PlannedActionWord(
            "a-repeat",
            WordExecutionMode.SEQUENTIAL,
            ("a", "a"),
            (Decimal("0"), Decimal("2")),
        ),
        PlannedActionWord(
            "a-then-b",
            WordExecutionMode.SEQUENTIAL,
            ("a", "b"),
            (Decimal("0"), Decimal("2")),
        ),
        PlannedActionWord("b-early", WordExecutionMode.SEQUENTIAL, ("b",), (Decimal("0"),)),
        PlannedActionWord("b-late", WordExecutionMode.SEQUENTIAL, ("b",), (Decimal("2"),)),
        PlannedActionWord(
            "b-repeat",
            WordExecutionMode.SEQUENTIAL,
            ("b", "b"),
            (Decimal("0"), Decimal("2")),
        ),
        PlannedActionWord(
            "b-then-a",
            WordExecutionMode.SEQUENTIAL,
            ("b", "a"),
            (Decimal("0"), Decimal("2")),
        ),
        PlannedActionWord("identity", WordExecutionMode.IDENTITY, (), ()),
        PlannedActionWord(
            "simultaneous-a-b",
            WordExecutionMode.SIMULTANEOUS,
            ("a", "b"),
            (Decimal("0"), Decimal("0")),
        ),
    )
    return ActionWordDesignSpec(
        design_id=f"ra3-design.{method_input.case_token}",
        alphabet=ActionAlphabetSpec(
            f"ra3-alphabet.{method_input.case_token}",
            letters,
        ),
        words=words,
        primary_pair=("a", "b"),
        receiver_ids=tuple(
            sorted(
                {
                    coordinate_id
                    for view in method_input.state_views
                    for coordinate_id in view.coordinate_ids
                }
            )
        ),
        horizon_ids=("horizon-two-slot",),
        development_unit_ids=_split_unit_ids(method_input, DataSplit.CALIBRATION),
        evaluation_unit_ids=_split_unit_ids(method_input, DataSplit.HELD_OUT),
        randomization_seed=20260718,
    )


def reference_response_algebra_protocol(
    invocation: TruthBlindResponseAlgebraInvocation,
) -> ResponseAlgebraProtocolSpec:
    """Freeze the shared prototype menu without consulting privileged truth."""

    config = invocation.method_config
    definitions = list(prototype_class_definitions())
    nomination_index = next(
        index
        for index, value in enumerate(definitions)
        if value.prototype
        is StructuralClassPrototype.TEMPORAL_COCYCLE__COMPOSITION_NULL_AT_RESOLUTION
    )
    original = definitions[nomination_index]
    stationary = SignatureCriterion(
        axis=SignatureAxis.TEMPORAL_COMPOSITION,
        allowed_values=(TemporalComposition.STATIONARY_SEMIGROUP.value,),
    )
    definitions[nomination_index] = StructuralClassDefinition(
        class_id="stationary-semigroup-composition-null-at-resolution",
        prototype=original.prototype,
        criteria=tuple(
            sorted(
                (
                    *(value for value in original.criteria if value.axis is not stationary.axis),
                    stationary,
                ),
                key=lambda value: value.axis,
            )
        ),
    )
    hypotheses = tuple(
        sorted(
            (
                StructuralClassHypothesis(
                    hypothesis_id=f"hypothesis.{definition.class_id}",
                    class_definition=definition,
                    primary=(
                        definition.prototype
                        is StructuralClassPrototype.TEMPORAL_COCYCLE__COMPOSITION_NULL_AT_RESOLUTION
                    ),
                    decisive_falsifier_ids=(
                        f"falsifier.{definition.class_id}.held-out-opposition",
                    ),
                )
                for definition in definitions
            ),
            key=lambda value: value.hypothesis_id,
        )
    )
    criteria = tuple(
        EquivalenceSpec(
            equivalence_id=f"equivalence.{invocation.method_input.case_token}.{value.coordinate_id}",
            receiver_id=value.coordinate_id,
            native_unit=value.native_unit,
            numerical_or_observation_floor_upper=value.floor_upper,
            equivalence_width=value.equivalence_width,
            constituent_materiality_lower=value.materiality_lower,
            confidence_level=config.confidence_level,
        )
        for value in config.receiver_criteria
    )
    development_ids = _split_unit_ids(invocation.method_input, DataSplit.CALIBRATION)
    evaluation_ids = _split_unit_ids(invocation.method_input, DataSplit.HELD_OUT)
    return ResponseAlgebraProtocolSpec(
        protocol_id=f"ra3-protocol.{invocation.method_input.case_token}",
        relation_id=f"ra3-relation.{invocation.method_input.case_token}",
        method_key=config.method_key,
        method_version=config.method_version,
        evaluator_key=RESPONSE_ALGEBRA_EVALUATOR_KEY,
        word_design=_word_design(invocation.method_input),
        equivalence_specs=criteria,
        class_hypotheses=hypotheses,
        multiplicity=MultiplicitySpec(
            multiplicity_id=f"ra3-multiplicity.{invocation.method_input.case_token}",
            kind=MultiplicityKind.PRIMARY_PAIR,
            family_ids=("pair-a-b",),
            error_rate=Decimal("0.05"),
        ),
        stopping=NonadaptiveStoppingSpec(
            stopping_id=f"ra3-stop.{invocation.method_input.case_token}",
            maximum_preparations=len(set(development_ids + evaluation_ids)),
            maximum_words_per_preparation=10,
            maximum_invalid_unit_fraction=Decimal("0"),
            maximum_retries=0,
        ),
        resource_budget=ResourceBudget(
            cpu_cores=4,
            memory_bytes=8 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3600,
            source_scan_bytes=4 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        prohibited_component_ids=("live-actuation", "llm", "rl"),
    )


def prepare_ra3_inputs(
    config: TruthKnownResponseStudyConfig,
) -> tuple[
    TruthKnownResponseStudyDevelopmentBatch,
    TruthKnownResponseStudySealedEvaluationBatch,
    TruthKnownResponseStudyPrivilegedOracleBatch,
]:
    """Compatibility composition over the three physically separate inputs."""

    return (
        prepare_ra3_development(config),
        prepare_ra3_sealed_evaluation(config),
        prepare_ra3_privileged_oracles(),
    )


def prepare_ra3_development(
    config: TruthKnownResponseStudyConfig,
) -> TruthKnownResponseStudyDevelopmentBatch:
    """Materialize only development-visible projections; never construct truth."""

    invocations = truth_blind_response_algebra_invocations(config.response_algebra)
    cases = tuple(
        TruthKnownResponseStudyDevelopmentCase(
            case_id=value.method_input.case_token,
            invocation=TruthBlindResponseAlgebraInvocation(
                invocation_id=value.invocation_id,
                method_input=calibration_only_response_algebra_input(value.method_input),
                method_config=value.method_config,
            ),
            protocol=reference_response_algebra_protocol(value),
        )
        for value in invocations
    )
    return TruthKnownResponseStudyDevelopmentBatch(batch_id="response-algebra-development", cases=cases)


def prepare_ra3_sealed_evaluation(
    config: TruthKnownResponseStudyConfig,
) -> TruthKnownResponseStudySealedEvaluationBatch:
    """Materialize truth-blind held-out method inputs behind the evaluation seal."""

    return TruthKnownResponseStudySealedEvaluationBatch(
        batch_id="response-algebra-sealed-evaluation",
        invocations=truth_blind_response_algebra_invocations(config.response_algebra),
    )


def prepare_ra3_privileged_oracles() -> TruthKnownResponseStudyPrivilegedOracleBatch:
    """Materialize only privileged labels in the custodian/evaluator lane."""

    return TruthKnownResponseStudyPrivilegedOracleBatch(
        batch_id="response-algebra-privileged-oracles",
        oracles=privileged_response_algebra_oracles(),
    )


def freeze_ra3_development(
    development: TruthKnownResponseStudyDevelopmentBatch,
    *,
    implementation_sha256: str,
) -> TruthKnownResponseStudyFreezeBatch:
    """Freeze all ten cases from development-only projections."""

    return TruthKnownResponseStudyFreezeBatch(
        batch_id="response-algebra-freezes",
        freezes=tuple(
            freeze_response_algebra_development(
                freeze_id=f"freeze.ra3.{case.case_id}",
                protocol=case.protocol,
                method_input=case.invocation.method_input,
                config=case.invocation.method_config,
                implementation_sha256=implementation_sha256,
            )
            for case in development.cases
        ),
    )


def evaluate_ra3_methods(
    development: TruthKnownResponseStudyDevelopmentBatch,
    sealed_evaluation: TruthKnownResponseStudySealedEvaluationBatch,
    freezes: TruthKnownResponseStudyFreezeBatch,
) -> TruthKnownResponseStudyMethodEvaluationBatch:
    """Run the actual shared estimator under exact frozen development choices."""

    development_by_case = {value.case_id: value for value in development.cases}
    freeze_by_case = {
        value.freeze_id.removeprefix("freeze.ra3."): value for value in freezes.freezes
    }
    evaluation_by_case = {
        value.method_input.case_token: value for value in sealed_evaluation.invocations
    }
    if not (set(development_by_case) == set(freeze_by_case) == set(evaluation_by_case)):
        raise ValueError("Truth-known response study development, freeze and evaluation case sets differ")
    evaluations = tuple(
        evaluate_frozen_response_algebra(
            protocol=development_by_case[case_id].protocol,
            method_input=evaluation_by_case[case_id].method_input,
            freeze=freeze_by_case[case_id],
        )
        for case_id in sorted(development_by_case)
    )
    results = tuple(
        sorted(
            (value.method_result for value in evaluations),
            key=lambda value: value.result_id,
        )
    )
    return TruthKnownResponseStudyMethodEvaluationBatch(
        batch_id="response-algebra-method-evaluation",
        evaluations=tuple(sorted(evaluations, key=lambda value: value.evaluation_id)),
        results=results,
    )


def adjudicate_ra3_methods(
    method_evaluation: TruthKnownResponseStudyMethodEvaluationBatch,
    oracle_batch: TruthKnownResponseStudyPrivilegedOracleBatch,
    config: TruthKnownResponseStudyConfig,
) -> TruthKnownResponseStudyAdjudication:
    """Score only after method results and privileged truth reach the evaluator."""

    return TruthKnownResponseStudyAdjudication(
        adjudication_id="response-algebra-adjudication",
        method_evaluation_sha256=method_evaluation.fingerprint(),
        oracle_batch_sha256=oracle_batch.fingerprint(),
        score=score_response_algebra_conformance(
            method_evaluation.results,
            oracle_batch.oracles,
            config.response_algebra,
        ),
        claim_promotion_allowed=False,
        scientific_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'TruthKnownResponseStudyAdjudication',
    'TruthKnownResponseStudyDevelopmentBatch',
    'TruthKnownResponseStudyDevelopmentCase',
    'TruthKnownResponseStudyFreezeBatch',
    'TruthKnownResponseStudyMethodEvaluationBatch',
    'TruthKnownResponseStudyPrivilegedOracleBatch',
    'TruthKnownResponseStudySealedEvaluationBatch',
    'TruthKnownResponseStudyConfig',
    "adjudicate_ra3_methods",
    'default_truth_known_response_study_config',
    "evaluate_ra3_methods",
    "freeze_ra3_development",
    "prepare_ra3_development",
    "prepare_ra3_inputs",
    "prepare_ra3_privileged_oracles",
    "prepare_ra3_sealed_evaluation",
    "reference_response_algebra_protocol",
]
