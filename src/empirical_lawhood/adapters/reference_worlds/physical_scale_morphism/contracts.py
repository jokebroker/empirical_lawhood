"""Truth-world identities and evaluator records for physical scale morphism."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS = (
    "boundary-topology",
    "composition-discrimination",
    "energy-mean-collision",
    "exact-receiver-composition",
    "future-closure",
    "guard-false-safe",
    "heterogeneous-local-support",
    "hidden-history",
    "hold-state-grammar",
    "leakage-firewall",
    "numerical-qualification",
    "power-and-precision",
    "saturation-coordinate",
    "scale-morphism",
    "unit-clock-grouping",
)


class PhysicalScaleMorphismTruthVariant(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    NOISY_REPEATED_BOARD = "NOISY_REPEATED_BOARD"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthInputDatum(CanonicalRecord):
    """One bounded truth-blind challenge datum generated outside the method."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-input-datum'

    datum_id: str
    numeric_value: Decimal | None
    categorical_value: str | None
    unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.datum_id, field_name="datum_id")
        if (self.numeric_value is None) == (self.categorical_value is None):
            raise ValueError("truth input datum must carry exactly one typed value")
        validate_nonempty(self.unit, field_name="unit")
        if self.numeric_value is not None:
            validate_decimal(self.numeric_value, field_name="numeric_value")
        else:
            assert self.categorical_value is not None
            validate_nonempty(self.categorical_value, field_name="categorical_value")
            if self.unit != "category":
                raise ValueError("categorical truth input must use the category unit")


def truth_input_sha256(values: tuple[PhysicalScaleMorphismTruthInputDatum, ...]) -> str:
    digest = sha256()
    for value in values:
        payload = value.canonical_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthSuiteConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-suite-config'

    config_id: str
    fixture_ids: tuple[str, ...]
    variants: tuple[PhysicalScaleMorphismTruthVariant, ...]
    validation_seeds: tuple[int, ...]
    fixtures_frozen_before_validation: bool
    truth_labels_hidden_from_method: bool
    protected_physical_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.fixture_ids != PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS:
            raise ValueError("truth suite lacks the exact frozen fixture roster")
        if self.variants != tuple(PhysicalScaleMorphismTruthVariant):
            raise ValueError("truth suite requires deterministic and noisy variants")
        if tuple(sorted(set(self.validation_seeds))) != self.validation_seeds:
            raise ValueError("truth validation seeds must be sorted and unique")
        if len(self.validation_seeds) != len(self.variants) or any(
            value < 0 for value in self.validation_seeds
        ):
            raise ValueError("truth validation seed roster differs")
        if not self.fixtures_frozen_before_validation or not self.truth_labels_hidden_from_method:
            raise ValueError("truth suite crossed its method/validation firewall")
        if self.protected_physical_outcome_count:
            raise ValueError("truth qualification cannot inspect physical outcomes")
        if self.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH:
            raise ValueError("truth suite config must bind the privileged evaluator world")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthCase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-case'

    case_id: str
    fixture_id: str
    variant: PhysicalScaleMorphismTruthVariant
    seed: int
    generator_id: str
    input_data: tuple[PhysicalScaleMorphismTruthInputDatum, ...]
    input_data_sha256: str
    physical_evidence_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        if self.fixture_id not in PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS:
            raise ValueError("truth case fixture is unknown")
        validate_stable_id(self.generator_id, field_name="generator_id")
        require_sorted_unique_ids(self.input_data, attribute="datum_id", field_name="input_data")
        if not self.input_data:
            raise ValueError("truth case cannot leave its challenge data inside the method")
        validate_sha256(self.input_data_sha256, field_name="input_data_sha256")
        if self.input_data_sha256 != truth_input_sha256(self.input_data):
            raise ValueError("truth-case input fingerprint is not datum-derived")
        if self.seed < 0 or self.physical_evidence_count:
            raise ValueError("truth case contains invalid seed/physical evidence")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthBlindObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-blind-observation'

    observation_id: str
    case_id: str
    truth_input_sha256: str
    observed_code_ids: tuple[str, ...]
    method_exception_code_ids: tuple[str, ...]
    privileged_label_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(self.case_id, field_name="case_id")
        validate_sha256(self.truth_input_sha256, field_name="truth_input_sha256")
        require_sorted_unique_strings(
            self.observed_code_ids, field_name="observed_code_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.method_exception_code_ids, field_name="method_exception_code_ids"
        )
        if self.privileged_label_access_count:
            raise ValueError("truth-blind method cannot access oracle labels")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthOracle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-oracle'

    oracle_id: str
    case_id: str
    truth_input_sha256: str
    expected_code_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.oracle_id, field_name="oracle_id")
        validate_stable_id(self.case_id, field_name="case_id")
        validate_sha256(self.truth_input_sha256, field_name="truth_input_sha256")
        require_sorted_unique_strings(
            self.expected_code_ids, field_name="expected_code_ids", allow_empty=False
        )
        if self.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH:
            raise ValueError("truth oracle must remain privileged")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthCaseScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-case-score'

    score_id: str
    case_id: str
    truth_input_sha256: str
    missing_code_ids: tuple[str, ...]
    unexpected_code_ids: tuple[str, ...]
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.case_id, field_name="case_id")
        validate_sha256(self.truth_input_sha256, field_name="truth_input_sha256")
        require_sorted_unique_strings(self.missing_code_ids, field_name="missing_code_ids")
        require_sorted_unique_strings(self.unexpected_code_ids, field_name="unexpected_code_ids")
        if self.passed != (not self.missing_code_ids and not self.unexpected_code_ids):
            raise ValueError("truth case pass is not derived from exact oracle comparison")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthMethodSuiteResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-method-suite-result'

    result_id: str
    config_id: str
    scores: tuple[PhysicalScaleMorphismTruthCaseScore, ...]
    passed_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    method_qualified: bool
    scientific_ceiling: EvidenceCeiling
    claim_promotion_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(self.scores, attribute="score_id", field_name="scores")
        for name in ("passed_case_ids", "failed_case_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        expected_passed = tuple(sorted(value.case_id for value in self.scores if value.passed))
        expected_failed = tuple(sorted(value.case_id for value in self.scores if not value.passed))
        if self.passed_case_ids != expected_passed or self.failed_case_ids != expected_failed:
            raise ValueError("truth suite case partition is not score-derived")
        expected_qualification = bool(self.scores) and not self.failed_case_ids
        if self.method_qualified != expected_qualification:
            raise ValueError("method qualification is not derived from every truth case")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("truth-world method result must remain nonpromotable")
        if self.claim_promotion_allowed:
            raise ValueError("truth-world conformance cannot promote physical claims")


__all__ = [
    "PHYSICAL_SCALE_MORPHISM_TRUTH_FIXTURE_IDS",
    'PhysicalScaleMorphismTruthBlindObservation',
    'PhysicalScaleMorphismTruthCase',
    'PhysicalScaleMorphismTruthCaseScore',
    'PhysicalScaleMorphismTruthInputDatum',
    'PhysicalScaleMorphismTruthMethodSuiteResult',
    'PhysicalScaleMorphismTruthOracle',
    'PhysicalScaleMorphismTruthSuiteConfig',
    'PhysicalScaleMorphismTruthVariant',
    "truth_input_sha256",
]
