"""Additive orchestration records for the simulator morphism challenges scientific DAGs.

The records in this module bind complete-unit payloads and phase transitions.
They contain no observer, generator, evaluator, filesystem, or scheduler code.
"""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_sha256,
    validate_schema,
    validate_stable_id,
)

from .contracts import SimulatorMorphismChallengeChallengeNomination, SimulatorMorphismChallengeDenominatorDescriptor, SimulatorMorphismChallengeDisorderFamily, SimulatorMorphismChallengeGeneratorOutcome, SimulatorMorphismChallengeHistoryRankForecast, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengePhase, SimulatorMorphismChallengeSeedRosterCommitment, SimulatorMorphismChallengeUnitAdjudication


def _sorted_unique(values: tuple[str, ...], *, field_name: str, allow_empty: bool = False) -> None:
    if values != tuple(sorted(set(values))) or (not allow_empty and not values):
        raise ValueError(f"{field_name} must be nonempty, sorted and unique")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeRequestedUnitLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-requested-unit-ledger'

    ledger_id: str
    phase: SimulatorMorphismChallengePhase
    config_sha256: str
    unit_ids: tuple[str, ...]
    scale_cells: tuple[int, ...]
    outcome_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        _sorted_unique(self.unit_ids, field_name="unit_ids")
        if self.scale_cells != tuple(sorted(set(self.scale_cells))):
            raise ValueError("scale_cells must be sorted and unique")
        if self.outcome_count != 0:
            raise ValueError("requested-unit ledger must precede outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeSeedEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-seed-entry'

    unit_id: str
    seed_hex: str
    seed_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        if re.fullmatch(r"[0-9a-f]{64}", self.seed_hex) is None:
            raise ValueError("simulator morphism challenges seed entry must contain exactly 32 encoded bytes")
        validate_sha256(self.seed_sha256, field_name="seed_sha256")
        from hashlib import sha256

        if sha256(bytes.fromhex(self.seed_hex)).hexdigest() != self.seed_sha256:
            raise ValueError("simulator morphism challenges seed entry digest differs")

    @property
    def seed_bytes(self) -> bytes:
        return bytes.fromhex(self.seed_hex)


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeSeedRoster(CanonicalRecord):
    """Custodian-only seed payload; never a method or report input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-seed-roster'

    roster_id: str
    entries: tuple[SimulatorMorphismChallengeSeedEntry, ...]
    outcome_count_at_generation: int

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        unit_ids = tuple(value.unit_id for value in self.entries)
        _sorted_unique(unit_ids, field_name="entries")
        if len(self.entries) != 36:
            raise ValueError("simulator morphism challenges evaluation roster requires 36 seed entries")
        if len({value.seed_sha256 for value in self.entries}) != len(self.entries):
            raise ValueError("simulator morphism challenges evaluation seeds must be unique")
        if self.outcome_count_at_generation != 0:
            raise ValueError("simulator morphism challenges seed roster must precede outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengePhaseCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-phase-closeout'

    closeout_id: str
    phase: SimulatorMorphismChallengePhase
    input_sha256s: tuple[str, ...]
    passed: bool
    reason_codes: tuple[str, ...]
    scientific_claim_assigned: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        _sorted_unique(self.input_sha256s, field_name="input_sha256s")
        for value in self.input_sha256s:
            validate_sha256(value, field_name="input_sha256s")
        _sorted_unique(self.reason_codes, field_name="reason_codes", allow_empty=self.passed)
        if self.passed and self.reason_codes:
            raise ValueError("passed phase closeout cannot carry stop reasons")
        if self.scientific_claim_assigned:
            raise ValueError("plumbing phase closeout cannot assign a scientific claim")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeArrayEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-array-entry'

    array_id: str
    offset: int
    element_count: int
    shape: tuple[int, ...]
    dtype_id: str
    native_unit: str
    coordinate_frame: str
    clock_id: str
    independent_unit_id: str
    row_key_schema: str
    content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "array_id",
            "dtype_id",
            "native_unit",
            "coordinate_frame",
            "clock_id",
            "independent_unit_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.row_key_schema)
        if self.offset < 0 or self.element_count < 0:
            raise ValueError("array slice bounds must be nonnegative")
        if not self.shape or any(value < 0 for value in self.shape):
            raise ValueError("array shape is invalid")
        product = 1
        for value in self.shape:
            product *= value
        if product != self.element_count or self.dtype_id != "float64":
            raise ValueError("array shape/count/dtype contract differs")
        validate_sha256(self.content_sha256, field_name="content_sha256")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeArrayManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-array-manifest'

    manifest_id: str
    unit_id: str
    payload_sha256: str
    logical_arrays_sha256: str
    entries: tuple[SimulatorMorphismChallengeArrayEntry, ...]

    def __post_init__(self) -> None:
        for name in ("manifest_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.payload_sha256, field_name="payload_sha256")
        validate_sha256(self.logical_arrays_sha256, field_name="logical_arrays_sha256")
        ids = tuple(value.array_id for value in self.entries)
        _sorted_unique(ids, field_name="entries")
        expected_offset = 0
        for value in self.entries:
            if value.independent_unit_id != self.unit_id:
                raise ValueError("array entry binds another independent unit")
            if value.offset != expected_offset:
                raise ValueError("array manifest slices must be contiguous")
            expected_offset += value.element_count


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDenominatorBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-denominator-bundle'

    bundle_id: str
    unit_id: str
    descriptors: tuple[SimulatorMorphismChallengeDenominatorDescriptor, ...]
    scientific_input: HistoryBudgetUnitScientificInput

    def __post_init__(self) -> None:
        if not isinstance(self.scientific_input, HistoryBudgetUnitScientificInput) or self.scientific_input.programme_ordinal != 0 or self.scientific_input.unit_id != self.unit_id:
            raise ValueError("denominator bundle requires exact original numeric input and current unit custody")
        if tuple(row.current_descriptor_sha256 for row in self.scientific_input.descriptor_inputs) != tuple(row.fingerprint() for row in self.descriptors):
            raise ValueError("denominator bundle differs from its complete current descriptor custody census")
        for descriptor, scientific_row in zip(self.descriptors, self.scientific_input.descriptor_inputs, strict=True):
            scientific_row.require_current_binding(programme_ordinal=0, unit_id=descriptor.unit_id, scale_cells=descriptor.scale_cells, source_seed_sha256=descriptor.seed_sha256, current_descriptor_sha256=descriptor.fingerprint())
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if any(value.unit_id != self.unit_id for value in self.descriptors):
            raise ValueError("denominator bundle mixes independent units")
        scales = tuple(value.scale_cells for value in self.descriptors)
        if scales not in {(16, 32, 64), (64, 128, 256)}:
            raise ValueError("denominator bundle scale roster differs")
        if len({value.seed_sha256 for value in self.descriptors}) != 1:
            raise ValueError("coupled scale views must share one seed block")
        if len({value.family for value in self.descriptors}) != 1:
            raise ValueError("coupled scale views must share one disorder family")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeObserverBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-observer-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    forecasts: tuple[SimulatorMorphismChallengeHistoryRankForecast, ...]
    nominations: tuple[SimulatorMorphismChallengeChallengeNomination, ...]
    arrays_manifest_sha256: str
    outcome_count_at_nomination: int

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("denominator_bundle_sha256", "arrays_manifest_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        forecast_ids = tuple(value.forecast_id for value in self.forecasts)
        nomination_ids = tuple(value.nomination_id for value in self.nominations)
        _sorted_unique(forecast_ids, field_name="forecasts")
        _sorted_unique(nomination_ids, field_name="nominations", allow_empty=True)
        if len(self.forecasts) != 3:
            raise ValueError("observer bundle must contain three coupled scale forecasts")
        if any(value.unit_id != self.unit_id for value in self.nominations):
            raise ValueError("observer bundle mixes independent units")
        if any(value.scale_cells not in {16, 32, 64, 128, 256} for value in self.nominations):
            raise ValueError("observer bundle nomination scale differs")
        if self.outcome_count_at_nomination != 0:
            raise ValueError("observer bundle cannot observe generator outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeHistoryBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-history-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    forecasts: tuple[SimulatorMorphismChallengeHistoryRankForecast, ...]
    arrays_manifest_sha256: str
    outcome_count: int

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("denominator_bundle_sha256", "arrays_manifest_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        ids = tuple(value.forecast_id for value in self.forecasts)
        _sorted_unique(ids, field_name="forecasts")
        if len(self.forecasts) != 3 or self.outcome_count != 0:
            raise ValueError("history bundle must contain three pre-outcome scale forecasts")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeNominationFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-nomination-freeze'

    freeze_id: str
    observer_bundle: SimulatorMorphismChallengeObserverBundle
    history_bundle_sha256: str
    method_freeze_sha256: str | None
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_sha256(self.history_bundle_sha256, field_name="history_bundle_sha256")
        if self.method_freeze_sha256 is not None:
            validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        if self.outcome_count_at_freeze != 0:
            raise ValueError("nomination freeze must precede generator outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeGeneratorBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-generator-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    observer_bundle_sha256: str
    outcomes: tuple[SimulatorMorphismChallengeGeneratorOutcome, ...]
    arrays_manifest_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "denominator_bundle_sha256",
            "observer_bundle_sha256",
            "arrays_manifest_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        outcome_ids = tuple(value.outcome_id for value in self.outcomes)
        _sorted_unique(outcome_ids, field_name="outcomes")
        if len(self.outcomes) != 3:
            raise ValueError("generator bundle must contain three coupled scale outcomes")
        if any(value.unit_id != self.unit_id for value in self.outcomes):
            raise ValueError("generator bundle mixes independent units")
        if any(value.outcome_access is not self.outcome_access for value in self.outcomes):
            raise ValueError("generator bundle outcome access differs")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeAdjudicationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-adjudication-bundle'

    bundle_id: str
    unit_id: str
    method_freeze_sha256: str | None
    adjudications: tuple[SimulatorMorphismChallengeUnitAdjudication, ...]
    raw_generator_payload_exposed: bool

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.method_freeze_sha256 is not None:
            validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        ids = tuple(value.adjudication_id for value in self.adjudications)
        _sorted_unique(ids, field_name="adjudications")
        if len(self.adjudications) != 3 or any(
            value.unit_id != self.unit_id for value in self.adjudications
        ):
            raise ValueError("adjudication bundle must contain one complete coupled unit")
        if self.raw_generator_payload_exposed:
            raise ValueError("bounded adjudication cannot expose raw generator payloads")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeCanaryReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-canary-report'

    report_id: str
    check_ids: tuple[str, ...]
    passed: bool
    reason_codes: tuple[str, ...]
    wall_time_seconds: Decimal
    peak_memory_bytes: int
    projected_evaluation_cpu_hours: Decimal
    projected_evaluation_wall_hours: Decimal
    projected_evaluation_output_bytes: int
    evaluation_roster_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        _sorted_unique(self.check_ids, field_name="check_ids")
        _sorted_unique(self.reason_codes, field_name="reason_codes", allow_empty=self.passed)
        for name in (
            "wall_time_seconds",
            "projected_evaluation_cpu_hours",
            "projected_evaluation_wall_hours",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            min(
                self.peak_memory_bytes,
                self.projected_evaluation_output_bytes,
                self.evaluation_roster_access_count,
            )
            < 0
        ):
            raise ValueError("canary resource/access counters must be nonnegative")
        if self.passed and self.reason_codes:
            raise ValueError("passed canary report cannot carry stop reasons")
        if self.evaluation_roster_access_count != 0:
            raise ValueError("canary phase cannot access the evaluation roster")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDevelopmentGateCount(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-development-gate-count'

    family: SimulatorMorphismChallengeDisorderFamily
    target_nomination_count: int
    sink_nomination_count: int

    @property
    def gate_count_id(self) -> str:
        return self.family.value

    def __post_init__(self) -> None:
        if self.target_nomination_count < 0 or self.sink_nomination_count < 0:
            raise ValueError("development gate nomination counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDevelopmentLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-development-ledger'

    ledger_id: str
    canary_report_sha256: str
    adjudications: tuple[SimulatorMorphismChallengeUnitAdjudication, ...]
    gate_counts: tuple[SimulatorMorphismChallengeDevelopmentGateCount, ...]
    requested_unit_count: int
    complete_unit_count: int
    evaluation_roster_access_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        validate_sha256(self.canary_report_sha256, field_name="canary_report_sha256")
        ids = tuple(value.adjudication_id for value in self.adjudications)
        _sorted_unique(ids, field_name="adjudications")
        if len(self.adjudications) != 18:
            raise ValueError("development ledger requires six complete three-scale units")
        counts = (
            self.requested_unit_count,
            self.complete_unit_count,
            self.evaluation_roster_access_count,
        )
        if min(counts) < 0 or self.requested_unit_count != 6:
            raise ValueError("development ledger denominator differs")
        if self.complete_unit_count > self.requested_unit_count:
            raise ValueError("development complete-unit count exceeds requests")
        if self.evaluation_roster_access_count != 0:
            raise ValueError("development cannot access evaluation seed bytes")
        gate_ids = tuple(value.gate_count_id for value in self.gate_counts)
        _sorted_unique(gate_ids, field_name="gate_counts")
        if len(self.gate_counts) != 3:
            raise ValueError("development ledger requires one gate count per family")
        _sorted_unique(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeDevelopmentGate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-development-gate'

    gate_id: str
    development_ledger_sha256: str
    issue_evaluation: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        validate_sha256(self.development_ledger_sha256, field_name="development_ledger_sha256")
        _sorted_unique(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=self.issue_evaluation,
        )
        if self.issue_evaluation == bool(self.reason_codes):
            raise ValueError("development gate decision and reasons conflict")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeEvaluationDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-evaluation-design-freeze'

    freeze_id: str
    method_freeze: SimulatorMorphismChallengeMethodFreeze
    evaluation_config_sha256: str
    seed_roster_commitment: SimulatorMorphismChallengeSeedRosterCommitment
    implementation_source_closure_sha256: str
    evaluation_outcome_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for name in ("evaluation_config_sha256", "implementation_source_closure_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.method_freeze.evaluation_config_sha256 != self.evaluation_config_sha256:
            raise ValueError("evaluation design and method freeze configs differ")
        if (
            self.method_freeze.seed_roster_commitment_sha256
            != self.seed_roster_commitment.fingerprint()
        ):
            raise ValueError("evaluation design and method freeze rosters differ")
        if self.evaluation_outcome_count != 0:
            raise ValueError("evaluation design freeze must precede outcomes")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeBootstrapInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-bootstrap-interval'

    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.lower, field_name="lower", minimum=Decimal(0))
        validate_decimal(self.upper, field_name="upper", minimum=Decimal(0))
        if self.lower > self.upper:
            raise ValueError("bootstrap interval endpoints are reversed")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeBootstrapCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-bootstrap-cell-summary'

    cell_id: str
    family: SimulatorMorphismChallengeDisorderFamily
    scale_cells: int
    requested_unit_count: int
    evaluable_k_full_count: int
    k_full_mean: Decimal | None
    k_full_bootstrap_95: SimulatorMorphismChallengeBootstrapInterval | None
    k_full_usable_resample_count: int
    b_full_mean: Decimal | None
    b_full_bootstrap_95: SimulatorMorphismChallengeBootstrapInterval | None
    b_full_usable_resample_count: int
    rank_curve_distance_mean: Decimal
    rank_curve_distance_bootstrap_95: SimulatorMorphismChallengeBootstrapInterval

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.scale_cells not in {64, 128, 256} or self.requested_unit_count != 12:
            raise ValueError("bootstrap cell denominator or scale differs")
        if not 0 <= self.evaluable_k_full_count <= self.requested_unit_count:
            raise ValueError("bootstrap evaluable count differs")
        for name in ("k_full_usable_resample_count", "b_full_usable_resample_count"):
            if not 0 <= getattr(self, name) <= 10_000:
                raise ValueError(f"{name} differs")
        optional_values = (
            self.k_full_mean,
            self.k_full_bootstrap_95,
            self.b_full_mean,
            self.b_full_bootstrap_95,
        )
        if self.evaluable_k_full_count == 0:
            if any(value is not None for value in optional_values) or any(
                value != 0
                for value in (
                    self.k_full_usable_resample_count,
                    self.b_full_usable_resample_count,
                )
            ):
                raise ValueError("right-censored bootstrap cell cannot report full-rank estimates")
        elif (
            any(value is None for value in optional_values)
            or min(
                self.k_full_usable_resample_count,
                self.b_full_usable_resample_count,
            )
            <= 0
        ):
            raise ValueError("evaluable bootstrap cell lacks full-rank estimates")
        if self.k_full_mean is not None:
            validate_decimal(
                self.k_full_mean,
                field_name="k_full_mean",
                minimum=Decimal(0),
            )
            if self.k_full_mean > 35:
                raise ValueError("bootstrap k_full mean exceeds the frozen history roster")
        if self.k_full_bootstrap_95 is not None and self.k_full_bootstrap_95.upper > 35:
            raise ValueError("bootstrap k_full interval exceeds the frozen history roster")
        if self.b_full_mean is not None:
            validate_decimal(
                self.b_full_mean,
                field_name="b_full_mean",
                minimum=Decimal(0),
            )
            if self.b_full_mean > 1:
                raise ValueError("bootstrap b_full mean exceeds one")
        if self.b_full_bootstrap_95 is not None and self.b_full_bootstrap_95.upper > 1:
            raise ValueError("bootstrap b_full interval exceeds one")
        validate_decimal(
            self.rank_curve_distance_mean,
            field_name="rank_curve_distance_mean",
            minimum=Decimal(0),
        )
        if self.rank_curve_distance_mean > 1 or self.rank_curve_distance_bootstrap_95.upper > 1:
            raise ValueError("rank-curve distance exceeds its normalized range")


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeBootstrapSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-bootstrap-summary'

    summary_id: str
    evaluation_config_sha256: str
    method_freeze_sha256: str
    cells: tuple[SimulatorMorphismChallengeBootstrapCellSummary, ...]
    resamples: int
    bootstrap_seed: int
    confidence_level: Decimal
    whole_seed_block_resampling: bool
    nested_scale_action_mode_resampling: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_sha256(self.evaluation_config_sha256, field_name="evaluation_config_sha256")
        validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        cell_ids = tuple(value.cell_id for value in self.cells)
        _sorted_unique(cell_ids, field_name="cells")
        expected = {(family, scale) for family in SimulatorMorphismChallengeDisorderFamily for scale in (64, 128, 256)}
        if (
            len(self.cells) != 9
            or {(value.family, value.scale_cells) for value in self.cells} != expected
        ):
            raise ValueError("bootstrap summary must contain the nine frozen cells")
        validate_decimal(
            self.confidence_level,
            field_name="confidence_level",
            minimum=Decimal(0),
        )
        if (
            self.resamples != 10_000
            or self.bootstrap_seed < 0
            or self.confidence_level != Decimal("0.95")
            or not self.whole_seed_block_resampling
            or self.nested_scale_action_mode_resampling
        ):
            raise ValueError("bootstrap execution contract differs")


__all__ = [name for name in globals() if name.startswith("SimulatorMorphismChallenge")]
