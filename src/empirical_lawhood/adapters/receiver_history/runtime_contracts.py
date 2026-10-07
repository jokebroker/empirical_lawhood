"""Additive orchestration records for the receiver-history scientific DAGs.

The records in this module bind complete-unit payloads and phase transitions.
They contain no observer, generator, evaluator, filesystem, or scheduler code.
"""

from __future__ import annotations

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

from .contracts import (
    ReceiverHistoryChallengeNomination,
    ReceiverHistoryChallengeGeometry,
    ReceiverHistoryConditioningStep,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryDiscreteRankBracket,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryEndpoint,
    ReceiverHistoryGeneratorOutcome,
    ReceiverHistoryHistoryRankForecast,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryOptimizationCertificate,
    ReceiverHistoryPhase,
    ReceiverHistoryPowerState,
    ReceiverHistorySeedRosterCommitment,
    ReceiverHistoryStructuralRankStep,
    ReceiverHistoryTargetedCoordinateAdjudication,
    ReceiverHistoryUntouchedCoordinateAdjudication,
    ReceiverHistoryUntouchedGeneratorOutcome,
    ReceiverHistoryPreparationMapManifest,
    ReceiverHistoryUnitAdjudication,
)


FLOAT64_ARRAY_PAYLOAD_SCHEMA = 'empirical-lawhood/receiver-history/float64-array-payload-npy'
INT64_ARRAY_PAYLOAD_SCHEMA = 'empirical-lawhood/receiver-history/int64-array-payload-npy'


def _sorted_unique(values: tuple[str, ...], *, field_name: str, allow_empty: bool = False) -> None:
    if values != tuple(sorted(set(values))) or (not allow_empty and not values):
        raise ValueError(f"{field_name} must be nonempty, sorted and unique")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryRequestedUnitLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-requested-unit-ledger'

    ledger_id: str
    phase: ReceiverHistoryPhase
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
class ReceiverHistorySeedEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-seed-entry'

    unit_id: str
    seed_hex: str
    seed_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        if re.fullmatch(r"[0-9a-f]{64}", self.seed_hex) is None:
            raise ValueError("receiver-history seed entry must contain exactly 32 encoded bytes")
        validate_sha256(self.seed_sha256, field_name="seed_sha256")
        from hashlib import sha256

        if sha256(bytes.fromhex(self.seed_hex)).hexdigest() != self.seed_sha256:
            raise ValueError("receiver-history seed entry digest differs")

    @property
    def seed_bytes(self) -> bytes:
        return bytes.fromhex(self.seed_hex)


@dataclass(frozen=True, slots=True)
class ReceiverHistorySeedRoster(CanonicalRecord):
    """Custodian-only seed payload; never a method or report input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-seed-roster'

    roster_id: str
    entries: tuple[ReceiverHistorySeedEntry, ...]
    outcome_count_at_generation: int

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        unit_ids = tuple(value.unit_id for value in self.entries)
        _sorted_unique(unit_ids, field_name="entries")
        if len(self.entries) not in {36, 90}:
            raise ValueError(
                "receiver-history evaluation roster requires an accepted seed-entry count"
            )
        if len({value.seed_sha256 for value in self.entries}) != len(self.entries):
            raise ValueError("receiver-history evaluation seeds must be unique")
        if self.outcome_count_at_generation != 0:
            raise ValueError("receiver-history seed roster must precede outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryPhaseCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-phase-closeout'

    closeout_id: str
    phase: ReceiverHistoryPhase
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
class ReceiverHistoryArrayEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-array-entry'

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
        if product != self.element_count or self.dtype_id not in {"float64", "int64"}:
            raise ValueError("array shape/count/dtype contract differs")
        validate_sha256(self.content_sha256, field_name="content_sha256")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryArrayManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-array-manifest'

    manifest_id: str
    unit_id: str
    payload_schema: str
    payload_sha256: str
    logical_arrays_sha256: str
    entries: tuple[ReceiverHistoryArrayEntry, ...]

    def __post_init__(self) -> None:
        for name in ("manifest_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.payload_sha256, field_name="payload_sha256")
        validate_sha256(self.logical_arrays_sha256, field_name="logical_arrays_sha256")
        if self.payload_schema not in {
            FLOAT64_ARRAY_PAYLOAD_SCHEMA,
            INT64_ARRAY_PAYLOAD_SCHEMA,
        }:
            raise ValueError("receiver-history array payload schema differs")
        ids = tuple(value.array_id for value in self.entries)
        _sorted_unique(ids, field_name="entries")
        expected_offset = 0
        for value in self.entries:
            if value.independent_unit_id != self.unit_id:
                raise ValueError("array entry binds another independent unit")
            if value.offset != expected_offset:
                raise ValueError("array manifest slices must be contiguous")
            expected_dtype = (
                "float64" if self.payload_schema == FLOAT64_ARRAY_PAYLOAD_SCHEMA else "int64"
            )
            if value.dtype_id != expected_dtype:
                raise ValueError("array entry dtype differs from its payload schema")
            expected_offset += value.element_count


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDenominatorBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-denominator-bundle'

    bundle_id: str
    unit_id: str
    preparation_substream_seed_hex: str
    preparation_substream_seed_sha256: str
    descriptors: tuple[ReceiverHistoryDenominatorDescriptor, ...]

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if re.fullmatch(r"[0-9a-f]{64}", self.preparation_substream_seed_hex) is None:
            raise ValueError("denominator bundle preparation substream must be 32 bytes")
        validate_sha256(
            self.preparation_substream_seed_sha256,
            field_name="preparation_substream_seed_sha256",
        )
        from hashlib import sha256

        if (
            sha256(bytes.fromhex(self.preparation_substream_seed_hex)).hexdigest()
            != self.preparation_substream_seed_sha256
        ):
            raise ValueError("denominator bundle preparation substream digest differs")
        if any(value.unit_id != self.unit_id for value in self.descriptors):
            raise ValueError("denominator bundle mixes independent units")
        scales = tuple(value.scale_cells for value in self.descriptors)
        if scales not in {(16, 32), (64, 128, 256)}:
            raise ValueError("denominator bundle scale roster differs")
        if len({value.seed_sha256 for value in self.descriptors}) != 1:
            raise ValueError("coupled scale views must share one seed block")
        if len({value.family for value in self.descriptors}) != 1:
            raise ValueError("coupled scale views must share one disorder family")

    @property
    def preparation_substream_seed(self) -> bytes:
        return bytes.fromhex(self.preparation_substream_seed_hex)


@dataclass(frozen=True, slots=True)
class ReceiverHistoryObserverBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-observer-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    forecasts: tuple[ReceiverHistoryHistoryRankForecast, ...]
    geometries: tuple[ReceiverHistoryChallengeGeometry, ...]
    nominations: tuple[ReceiverHistoryChallengeNomination, ...]
    optimization_certificates: tuple[ReceiverHistoryOptimizationCertificate, ...]
    arrays_manifest_sha256: str
    outcome_count_at_nomination: int

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("denominator_bundle_sha256", "arrays_manifest_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        forecast_ids = tuple(value.forecast_id for value in self.forecasts)
        nomination_ids = tuple(value.nomination_id for value in self.nominations)
        geometry_ids = tuple(value.nomination_id for value in self.geometries)
        certificate_ids = tuple(value.certificate_id for value in self.optimization_certificates)
        _sorted_unique(forecast_ids, field_name="forecasts")
        _sorted_unique(nomination_ids, field_name="nominations", allow_empty=True)
        _sorted_unique(geometry_ids, field_name="geometries", allow_empty=True)
        if geometry_ids != nomination_ids:
            raise ValueError("observer geometry and assessment rosters differ")
        _sorted_unique(
            certificate_ids,
            field_name="optimization_certificates",
            allow_empty=True,
        )
        if len(self.forecasts) not in {2, 3}:
            raise ValueError("observer bundle forecast roster differs")
        if any(value.unit_id != self.unit_id for value in self.nominations):
            raise ValueError("observer bundle mixes independent units")
        if any(value.unit_id != self.unit_id for value in self.geometries):
            raise ValueError("observer geometry bundle mixes independent units")
        if any(value.scale_cells not in {16, 32, 64, 128, 256} for value in self.nominations):
            raise ValueError("observer bundle nomination scale differs")
        if self.outcome_count_at_nomination != 0:
            raise ValueError("observer bundle cannot observe generator outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryHistoryBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-history-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    forecasts: tuple[ReceiverHistoryHistoryRankForecast, ...]
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...]
    structural_rank_steps: tuple[ReceiverHistoryStructuralRankStep, ...]
    discrete_rank_brackets: tuple[ReceiverHistoryDiscreteRankBracket, ...]
    conditioning_steps: tuple[ReceiverHistoryConditioningStep, ...]
    arrays_manifest_sha256: str
    outcome_count: int

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("denominator_bundle_sha256", "arrays_manifest_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        ids = tuple(value.forecast_id for value in self.forecasts)
        _sorted_unique(ids, field_name="forecasts")
        coordinate_ids = tuple(value.coordinate_id for value in self.coordinates)
        _sorted_unique(coordinate_ids, field_name="coordinates")
        for field_name, values in (
            ("structural_rank_steps", self.structural_rank_steps),
            ("discrete_rank_brackets", self.discrete_rank_brackets),
            ("conditioning_steps", self.conditioning_steps),
        ):
            if values:
                raise ValueError(f"{field_name} is excluded from the primary claim-bearing profile")
        if len(self.forecasts) not in {2, 3} or self.outcome_count != 0:
            raise ValueError("history bundle must contain the complete pre-outcome scale roster")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryNominationFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-nomination-freeze'

    freeze_id: str
    geometries: tuple[ReceiverHistoryChallengeGeometry, ...]
    certificate_request_keys: tuple[str, ...]
    observer_bundle_sha256: str
    untouched_bundle_sha256: str | None
    preparation_manifest_sha256: str | None
    history_bundle_sha256: str
    method_freeze_sha256: str | None
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_sha256(self.observer_bundle_sha256, field_name="observer_bundle_sha256")
        geometry_ids = tuple(value.nomination_id for value in self.geometries)
        _sorted_unique(geometry_ids, field_name="geometries", allow_empty=True)
        _sorted_unique(
            self.certificate_request_keys,
            field_name="certificate_request_keys",
            allow_empty=True,
        )
        validate_sha256(self.history_bundle_sha256, field_name="history_bundle_sha256")
        if (self.untouched_bundle_sha256 is None) != (self.preparation_manifest_sha256 is None):
            raise ValueError("nomination freeze untouched identities are partially present")
        if self.untouched_bundle_sha256 is not None:
            validate_sha256(self.untouched_bundle_sha256, field_name="untouched_bundle_sha256")
            assert self.preparation_manifest_sha256 is not None
            validate_sha256(
                self.preparation_manifest_sha256,
                field_name="preparation_manifest_sha256",
            )
        if self.method_freeze_sha256 is not None:
            validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        if self.outcome_count_at_freeze != 0:
            raise ValueError("nomination freeze must precede generator outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryGeneratorBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-generator-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    observer_bundle_sha256: str
    outcomes: tuple[ReceiverHistoryGeneratorOutcome, ...]
    untouched_outcomes: tuple[ReceiverHistoryUntouchedGeneratorOutcome, ...]
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
        untouched_ids = tuple(value.outcome_id for value in self.untouched_outcomes)
        _sorted_unique(outcome_ids, field_name="outcomes", allow_empty=True)
        _sorted_unique(untouched_ids, field_name="untouched_outcomes", allow_empty=True)
        if (len(self.outcomes), len(self.untouched_outcomes)) not in {
            (3, 0),
            (0, 3),
            (3, 3),
        }:
            raise ValueError("generator bundle cohort/scale roster differs")
        if any(value.unit_id != self.unit_id for value in self.outcomes):
            raise ValueError("generator bundle mixes independent units")
        if any(value.outcome_access is not self.outcome_access for value in self.outcomes):
            raise ValueError("generator bundle outcome access differs")
        if len(self.untouched_outcomes) not in {0, 3} or any(
            value.unit_id != self.unit_id or value.outcome_access is not self.outcome_access
            for value in self.untouched_outcomes
        ):
            raise ValueError("generator bundle untouched-outcome roster differs")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-bundle'

    bundle_id: str
    unit_id: str
    denominator_bundle_sha256: str
    preparation_manifest: ReceiverHistoryPreparationMapManifest
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...]
    float_arrays_manifest_sha256: str
    int_arrays_manifest_sha256: str
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "denominator_bundle_sha256",
            "float_arrays_manifest_sha256",
            "int_arrays_manifest_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.preparation_manifest.unit_id != self.unit_id:
            raise ValueError("untouched bundle mixes independent units")
        ids = tuple(value.coordinate_id for value in self.coordinates)
        _sorted_unique(ids, field_name="coordinates")
        if self.outcome_count_at_freeze != 0:
            raise ValueError("untouched bundle cannot observe outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryUntouchedFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-untouched-freeze'

    freeze_id: str
    unit_id: str
    untouched_bundle_sha256: str
    preparation_manifest_sha256: str
    method_freeze_sha256: str
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        for name in ("freeze_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "untouched_bundle_sha256",
            "preparation_manifest_sha256",
            "method_freeze_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.outcome_count_at_freeze != 0:
            raise ValueError("untouched freeze must precede generator outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryAdjudicationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-adjudication-bundle'

    bundle_id: str
    unit_id: str
    method_freeze_sha256: str | None
    history_bundle: ReceiverHistoryHistoryBundle | None
    adjudications: tuple[ReceiverHistoryUnitAdjudication, ...]
    targeted_adjudications: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...]
    untouched_adjudications: tuple[ReceiverHistoryUntouchedCoordinateAdjudication, ...]
    raw_generator_payload_exposed: bool

    def __post_init__(self) -> None:
        for name in ("bundle_id", "unit_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.method_freeze_sha256 is not None:
            validate_sha256(self.method_freeze_sha256, field_name="method_freeze_sha256")
        if self.history_bundle is not None and self.history_bundle.unit_id != self.unit_id:
            raise ValueError("adjudication bundle binds another history unit")
        ids = tuple(value.adjudication_id for value in self.adjudications)
        _sorted_unique(ids, field_name="adjudications", allow_empty=True)
        targeted_ids = tuple(value.adjudication_id for value in self.targeted_adjudications)
        _sorted_unique(targeted_ids, field_name="targeted_adjudications", allow_empty=True)
        untouched_ids = tuple(value.adjudication_id for value in self.untouched_adjudications)
        _sorted_unique(untouched_ids, field_name="untouched_adjudications", allow_empty=True)
        if len(self.adjudications) not in {0, 3} or any(
            value.unit_id != self.unit_id for value in self.adjudications
        ):
            raise ValueError("adjudication bundle must contain one complete coupled unit")
        if bool(self.adjudications) != bool(self.targeted_adjudications) or any(
            value.unit_id != self.unit_id for value in self.targeted_adjudications
        ):
            raise ValueError("adjudication bundle lacks its targeted coordinate ledger")
        if self.raw_generator_payload_exposed:
            raise ValueError("bounded adjudication cannot expose raw generator payloads")
        if any(value.unit_id != self.unit_id for value in self.untouched_adjudications):
            raise ValueError("adjudication bundle lacks its untouched cohort")
        if not self.targeted_adjudications and not self.untouched_adjudications:
            raise ValueError("adjudication bundle cannot be empty")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryTaskOutputMeasurement(CanonicalRecord):
    'Measured target-correlated resource payload bytes against one authored task-class ceiling.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-task-output-measurement'

    task_class_id: str
    measured_output_bytes: int
    output_ceiling_bytes: int
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.task_class_id, field_name="task_class_id")
        if self.measured_output_bytes < 0 or self.output_ceiling_bytes <= 0:
            raise ValueError("canary task output measurement is invalid")
        if self.passed != (self.measured_output_bytes <= self.output_ceiling_bytes):
            raise ValueError("canary task output disposition is not derived")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryCanaryReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-canary-report'

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
    task_output_measurements: tuple[ReceiverHistoryTaskOutputMeasurement, ...] = ()
    measured_cpu_time_seconds: Decimal | None = None

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
        if self.measured_cpu_time_seconds is not None:
            validate_decimal(
                self.measured_cpu_time_seconds,
                field_name="measured_cpu_time_seconds",
                minimum=Decimal(0),
            )
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
        measurement_ids = tuple(value.task_class_id for value in self.task_output_measurements)
        resource_block = self.report_id.startswith("receiver-history.resource-canary.block-")
        _sorted_unique(
            measurement_ids,
            field_name="task_output_measurements",
            allow_empty=not resource_block,
        )
        if resource_block and measurement_ids != (
            "adjudication",
            "descriptor",
            "generator",
            "history",
            "targeter",
        ):
            raise ValueError("resource canary task output roster differs")
        if resource_block:
            measured_cpu_time = self.measured_cpu_time_seconds
            if measured_cpu_time is None:
                raise ValueError("resource canary lacks measured CPU time")
            reduced_runtime_profile = 'reduced-action-menu-profile' in self.check_ids
            evaluation_units = Decimal(36 if reduced_runtime_profile else 90)
            parallel_tasks = Decimal(6 if reduced_runtime_profile else 4)
            if self.projected_evaluation_cpu_hours != (
                measured_cpu_time * evaluation_units / Decimal(3600)
            ):
                raise ValueError("resource canary CPU projection is not derived")
            if self.projected_evaluation_wall_hours != (
                self.wall_time_seconds * evaluation_units / (parallel_tasks * Decimal(3600))
            ):
                raise ValueError("resource canary wall projection is not derived")
            if self.projected_evaluation_output_bytes != (
                sum(value.measured_output_bytes for value in self.task_output_measurements)
                * int(evaluation_units)
            ):
                raise ValueError("resource canary output projection is not derived")
        if self.passed and any(not value.passed for value in self.task_output_measurements):
            raise ValueError("passed canary has a failed task output measurement")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDevelopmentGateCount(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-development-gate-count'

    family: ReceiverHistoryDisorderFamily
    target_nomination_count: int
    sink_nomination_count: int

    @property
    def gate_count_id(self) -> str:
        return self.family.value

    def __post_init__(self) -> None:
        if self.target_nomination_count < 0 or self.sink_nomination_count < 0:
            raise ValueError("development gate nomination counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryDevelopmentLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-development-ledger'

    ledger_id: str
    canary_report_sha256: str | None
    adjudications: tuple[ReceiverHistoryUnitAdjudication, ...]
    targeted_adjudications: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...]
    gate_counts: tuple[ReceiverHistoryDevelopmentGateCount, ...]
    requested_unit_count: int
    complete_unit_count: int
    evaluation_roster_access_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if self.canary_report_sha256 is not None:
            validate_sha256(self.canary_report_sha256, field_name="canary_report_sha256")
        ids = tuple(value.adjudication_id for value in self.adjudications)
        _sorted_unique(ids, field_name="adjudications")
        if len(self.adjudications) != 36:
            raise ValueError("development ledger requires twelve complete three-scale units")
        targeted_ids = tuple(value.adjudication_id for value in self.targeted_adjudications)
        _sorted_unique(targeted_ids, field_name="targeted_adjudications")
        if len(self.targeted_adjudications) != 576:
            raise ValueError("development ledger requires every endpoint-coordinate unit row")
        counts = (
            self.requested_unit_count,
            self.complete_unit_count,
            self.evaluation_roster_access_count,
        )
        if min(counts) < 0 or self.requested_unit_count != 12:
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
class ReceiverHistoryDevelopmentGate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-development-gate'

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
class ReceiverHistoryPowerCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-power-cell'

    cell_id: str
    endpoint: ReceiverHistoryEndpoint
    coordinate_id: str
    coordinate_label: str
    depth: int
    resolution_epsilon: Decimal
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    requested_unit_count: int
    valid_unit_count: int
    witness_unit_count: int
    informative_nonadverse_unit_count: int
    targetability_limited_unit_count: int
    power_state: ReceiverHistoryPowerState

    def __post_init__(self) -> None:
        for name in ("cell_id", "coordinate_id", "coordinate_label"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(
            self.resolution_epsilon,
            field_name="resolution_epsilon",
            minimum=Decimal(0),
        )
        counts = (
            self.valid_unit_count,
            self.witness_unit_count,
            self.informative_nonadverse_unit_count,
            self.targetability_limited_unit_count,
        )
        if (
            self.scale_cells not in {64, 128, 256}
            or not 0 <= self.depth <= 31
            or self.requested_unit_count != 4
            or min(counts) < 0
            or max(counts) > 4
            or self.witness_unit_count
            + self.informative_nonadverse_unit_count
            + self.targetability_limited_unit_count
            != self.valid_unit_count
        ):
            raise ValueError("receiver-history power-cell denominator differs")
        expected = (
            ReceiverHistoryPowerState.INVALID
            if self.valid_unit_count != 4
            else ReceiverHistoryPowerState.WITNESS_CAPABLE
            if self.witness_unit_count >= 3
            else ReceiverHistoryPowerState.NONADVERSITY_CAPABLE
            if self.informative_nonadverse_unit_count >= 3
            else ReceiverHistoryPowerState.MIXED_CAPABLE
            if self.witness_unit_count >= 2 and self.informative_nonadverse_unit_count >= 2
            else ReceiverHistoryPowerState.UNPOWERED
        )
        if self.power_state is not expected:
            raise ValueError("receiver-history power state is not mechanically derived")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryEndpointCoordinatePowerAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-endpoint-coordinate-power-atlas'

    atlas_id: str
    development_ledger_sha256: str
    cells: tuple[ReceiverHistoryPowerCell, ...]
    eligible_cell_ids: tuple[str, ...]
    eligible_question_ids: tuple[str, ...]
    invalid_cell_count: int
    outcome_count_at_freeze: int

    def __post_init__(self) -> None:
        validate_stable_id(self.atlas_id, field_name="atlas_id")
        validate_sha256(self.development_ledger_sha256, field_name="development_ledger_sha256")
        cell_ids = tuple(value.cell_id for value in self.cells)
        _sorted_unique(cell_ids, field_name="cells")
        if len(self.cells) != 135:
            raise ValueError("receiver-history power atlas requires all 135 typed cells")
        _sorted_unique(self.eligible_cell_ids, field_name="eligible_cell_ids", allow_empty=True)
        _sorted_unique(
            self.eligible_question_ids,
            field_name="eligible_question_ids",
            allow_empty=True,
        )
        capable = {
            value.cell_id
            for value in self.cells
            if value.power_state
            in {
                ReceiverHistoryPowerState.WITNESS_CAPABLE,
                ReceiverHistoryPowerState.NONADVERSITY_CAPABLE,
                ReceiverHistoryPowerState.MIXED_CAPABLE,
            }
        }
        if set(self.eligible_cell_ids) != capable:
            raise ValueError("receiver-history atlas mask is not mechanically cell-derived")
        invalid = sum(
            value.power_state is ReceiverHistoryPowerState.INVALID for value in self.cells
        )
        if self.invalid_cell_count != invalid or self.outcome_count_at_freeze != 0:
            raise ValueError("receiver-history atlas integrity fields differ")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryEvaluationDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-evaluation-design-freeze'

    freeze_id: str
    method_freeze: ReceiverHistoryMethodFreeze
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas
    evaluation_config_sha256: str
    continuation_predicate_id: str
    seed_roster_commitment: ReceiverHistorySeedRosterCommitment
    implementation_source_closure_sha256: str
    evaluation_outcome_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(self.continuation_predicate_id, field_name="continuation_predicate_id")
        for name in (
            "evaluation_config_sha256",
            "implementation_source_closure_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.evaluation_outcome_count != 0:
            raise ValueError("evaluation design freeze must precede outcomes")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryBootstrapInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-bootstrap-interval'

    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.lower, field_name="lower", minimum=Decimal(0))
        validate_decimal(self.upper, field_name="upper", minimum=Decimal(0))
        if self.lower > self.upper:
            raise ValueError("bootstrap interval endpoints are reversed")


@dataclass(frozen=True, slots=True)
class ReceiverHistoryBootstrapCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-bootstrap-cell-summary'

    cell_id: str
    family: ReceiverHistoryDisorderFamily
    scale_cells: int
    requested_unit_count: int
    evaluable_k_full_count: int
    k_full_mean: Decimal | None
    k_full_bootstrap_95: ReceiverHistoryBootstrapInterval | None
    k_full_usable_resample_count: int
    b_full_mean: Decimal | None
    b_full_bootstrap_95: ReceiverHistoryBootstrapInterval | None
    b_full_usable_resample_count: int
    rank_curve_distance_mean: Decimal
    rank_curve_distance_bootstrap_95: ReceiverHistoryBootstrapInterval

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.scale_cells not in {64, 128, 256} or self.requested_unit_count not in {12, 30}:
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
            if self.k_full_mean > 31:
                raise ValueError("bootstrap k_full mean exceeds the frozen history roster")
        if self.k_full_bootstrap_95 is not None and self.k_full_bootstrap_95.upper > 31:
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
class ReceiverHistoryBootstrapSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/receiver-history/receiver-history-bootstrap-summary'

    summary_id: str
    evaluation_config_sha256: str
    method_freeze_sha256: str
    cells: tuple[ReceiverHistoryBootstrapCellSummary, ...]
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
        expected = {
            (family, scale) for family in ReceiverHistoryDisorderFamily for scale in (64, 128, 256)
        }
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


__all__ = [name for name in globals() if name.startswith("ReceiverHistory")]
