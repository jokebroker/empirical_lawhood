"""Closed scientific records for the FAIR-MAST flagship archive surface."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


FAIR_MAST_SHOTS_METADATA_SHA256: Final = (
    "366d1c02b7accbb801fbf31adec7612a9b31a7ae12393f1cec189a9ff031d9fd"
)
FAIR_MAST_SOURCES_METADATA_SHA256: Final = (
    "0eb3cecb7e1037ace3b2ef55d73014f46256209af79991007176babfe036dcb3"
)
FAIR_MAST_INGESTION_COMMIT: Final = "ab435c799d892956fb042d55391f7d1be0c950e6"
FAIR_MAST_LICENSE_ID: Final = "cc-by-sa-4.0"
MAST_ENDPOINT_HORIZON_S: Final = Decimal("0.040")
ELECTRONVOLT_J: Final = Decimal("1.602176634e-19")
MAST_KNOWN_INSPECTED_SHOTS: Final = frozenset(
    {
        12303,
        12411,
        12745,
        13685,
        16335,
        16985,
        17900,
        18411,
        21425,
        21793,
        24045,
        24809,
        27194,
        27582,
        27795,
        28074,
        29250,
        29643,
        30037,
        30176,
        25095,
        25106,
        25165,
        25331,
    }
)


class MastCampaign(StrEnum):
    M7 = "M7"
    M8 = "M8"
    M9 = "M9"


class MastShotRole(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    CONSTRUCT_SELECTION = "CONSTRUCT_SELECTION"
    PROTECTED_EVALUATION = "PROTECTED_EVALUATION"


class MastActionLabel(StrEnum):
    DOWN = "DOWN"
    HOLD = "HOLD"
    UP = "UP"


class MastEventDisposition(StrEnum):
    CANDIDATE = "CANDIDATE"
    ACCEPTED = "ACCEPTED"
    EXCLUDED = "EXCLUDED"
    AMBIGUOUS = "AMBIGUOUS"


class MastFeatureRole(StrEnum):
    DENOMINATOR = "D"
    HISTORY = "H"
    RECEIVER = "R"


class MastEndpointState(StrEnum):
    EVALUABLE = "EVALUABLE"
    NO_PHYSICAL_RESPONSE = "NO_PHYSICAL_RESPONSE"
    DIAGNOSTIC_UNAVAILABLE = "DIAGNOSTIC_UNAVAILABLE"
    MAPPING_FAILED = "MAPPING_FAILED"
    OUT_OF_WINDOW = "OUT_OF_WINDOW"
    INVALID_MEASUREMENT = "INVALID_MEASUREMENT"


class MastMaterializationState(StrEnum):
    MATERIALIZED = "MATERIALIZED"
    IDEMPOTENT_REUSE = "IDEMPOTENT_REUSE"


@dataclass(frozen=True, slots=True)
class FairMastSourceDeclaration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-source-declaration'

    source_id: str
    source_level: str
    campaigns: tuple[MastCampaign, ...]
    shots_metadata_sha256: str
    sources_metadata_sha256: str
    ingestion_commit: str
    license_id: str
    anonymous_public_access: bool
    redistribution_review_required: bool
    expected_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_id != "fair-mast-level2-m7-m9":
            raise ValueError("FAIR-MAST source identity differs")
        if self.source_level != "LEVEL_2":
            raise ValueError("FAIR-MAST source level differs")
        if self.campaigns != tuple(MastCampaign):
            raise ValueError("FAIR-MAST flagship is restricted to M7--M9")
        for name, actual, expected in (
            ("shots_metadata_sha256", self.shots_metadata_sha256, FAIR_MAST_SHOTS_METADATA_SHA256),
            (
                "sources_metadata_sha256",
                self.sources_metadata_sha256,
                FAIR_MAST_SOURCES_METADATA_SHA256,
            ),
        ):
            validate_sha256(actual, field_name=name)
            if actual != expected:
                raise ValueError(f"{name} differs from the declared FAIR-MAST source identity")
        if self.ingestion_commit != FAIR_MAST_INGESTION_COMMIT:
            raise ValueError("FAIR-MAST ingestion commit differs")
        if self.license_id != FAIR_MAST_LICENSE_ID:
            raise ValueError("FAIR-MAST license differs")
        if not self.anonymous_public_access or not self.redistribution_review_required:
            raise ValueError("FAIR-MAST access/license qualification differs")
        require_sorted_unique_strings(
            self.expected_groups, field_name="expected_groups", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class FairMastSignalSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-signal-spec'

    signal_id: str
    native_unit: str
    semantic_role: str
    expected_group: str
    required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.signal_id, field_name="signal_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.semantic_role, field_name="semantic_role")
        validate_stable_id(self.expected_group, field_name="expected_group")
        expected = {
            "nbi-power": ("W", "realized-action"),
            "electron-temperature": ("eV", "receiver"),
            "electron-density": ("m^-3", "denominator"),
        }
        if self.signal_id not in expected:
            raise ValueError("signal is outside the closed MAST flagship gauge")
        if (self.native_unit, self.semantic_role) != expected[self.signal_id]:
            raise ValueError("MAST signal unit or role differs")


@dataclass(frozen=True, slots=True)
class MastShotAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-shot-assignment'

    assignment_id: str
    shot_id: int
    campaign: MastCampaign
    role: MastShotRole
    contamination_reason_codes: tuple[str, ...]
    challenge_steward_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.assignment_id, field_name="assignment_id")
        if self.shot_id <= 0 or self.assignment_id != f"mast-shot-{self.shot_id}":
            raise ValueError("MAST assignment must preserve the physical shot identity")
        require_sorted_unique_strings(
            self.contamination_reason_codes,
            field_name="contamination_reason_codes",
        )
        known_contaminated = self.shot_id in MAST_KNOWN_INSPECTED_SHOTS
        if known_contaminated and "KNOWN_INSPECTED" not in self.contamination_reason_codes:
            raise ValueError("known inspected shot contamination was not recorded")
        if self.role is MastShotRole.PROTECTED_EVALUATION:
            if self.challenge_steward_id is None:
                raise ValueError("protected MAST shots require a challenge steward")
            validate_stable_id(self.challenge_steward_id, field_name="challenge_steward_id")
            if self.contamination_reason_codes:
                raise ValueError("contaminated shots cannot enter the protected roster")
        elif self.challenge_steward_id is not None:
            raise ValueError("only protected shots may name a challenge steward")


@dataclass(frozen=True, slots=True)
class MastShotRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-shot-roster'

    roster_id: str
    assignments: tuple[MastShotAssignment, ...]
    issued: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        require_sorted_unique_ids(
            self.assignments, attribute="assignment_id", field_name="assignments"
        )
        shot_ids = tuple(value.shot_id for value in self.assignments)
        if len(set(shot_ids)) != len(shot_ids):
            raise ValueError("a MAST shot cannot cross roster roles")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("roster construction cannot reveal protected outcomes")
        if self.issued:
            raise ValueError("Phase 4 authoring cannot issue a protected roster")


@dataclass(frozen=True, slots=True)
class MastScalarSample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-scalar-sample'

    sample_id: str
    coordinate_s: Decimal
    value: Decimal | None
    native_unit: str
    clock_id: str
    valid: bool
    quality_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        validate_decimal(self.coordinate_s, field_name="coordinate_s")
        if self.value is not None:
            validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.clock_id, field_name="clock_id")
        require_sorted_unique_strings(self.quality_reason_codes, field_name="quality_reason_codes")
        if self.valid != (self.value is not None and not self.quality_reason_codes):
            raise ValueError("sample validity differs from value/quality state")


@dataclass(frozen=True, slots=True)
class MastActionTrace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-action-trace'

    shot_id: int
    campaign: MastCampaign
    nbi_power: tuple[MastScalarSample, ...]
    auxiliary_actuator: tuple[MastScalarSample, ...]

    def __post_init__(self) -> None:
        if self.shot_id <= 0:
            raise ValueError("shot_id must be positive")
        if len(self.nbi_power) < 4:
            raise ValueError("event construction requires at least four NBI samples")
        if any(value.native_unit != "W" for value in self.nbi_power):
            raise ValueError("NBI action trace must remain in watts")
        for name, values in (
            ("nbi_power", self.nbi_power),
            ("auxiliary_actuator", self.auxiliary_actuator),
        ):
            coordinates = tuple(value.coordinate_s for value in values)
            if tuple(sorted(set(coordinates))) != coordinates:
                raise ValueError(f"{name} coordinates must be sorted and unique")


@dataclass(frozen=True, slots=True)
class MastEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-event'

    event_id: str
    shot_id: int
    campaign: MastCampaign
    t0_s: Decimal | None
    action: MastActionLabel | None
    disposition: MastEventDisposition
    pre_power_w: Decimal | None
    post_power_w: Decimal | None
    delta_power_w: Decimal | None
    native_clock_id: str
    interpolation_record_id: str
    rule: ObjectIdentity
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.native_clock_id, field_name="native_clock_id")
        validate_stable_id(self.interpolation_record_id, field_name="interpolation_record_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        values = (self.t0_s, self.pre_power_w, self.post_power_w, self.delta_power_w)
        for index, value in enumerate(values):
            if value is not None:
                validate_decimal(value, field_name=f"event_value_{index}")
        if self.disposition in {MastEventDisposition.CANDIDATE, MastEventDisposition.ACCEPTED}:
            if self.action is None or any(value is None for value in values):
                raise ValueError("eligible MAST event requires action, time and power values")
        elif not self.reason_codes:
            raise ValueError("excluded/ambiguous MAST event requires reasons")


@dataclass(frozen=True, slots=True)
class MastFeatureValue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-feature-value'

    feature_id: str
    source_signal_id: str
    role: MastFeatureRole
    values: tuple[Decimal, ...]
    radial_coordinates: tuple[Decimal, ...]
    native_unit: str
    missing_mask: tuple[bool, ...]
    quality_reason_codes: tuple[str, ...]
    latest_source_coordinate_s: Decimal | None
    transform_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("feature_id", self.feature_id),
            ("source_signal_id", self.source_signal_id),
            ("transform_id", self.transform_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        if not self.values or len(self.values) != len(self.missing_mask):
            raise ValueError("feature values require an equally sized missingness mask")
        if self.radial_coordinates and len(self.radial_coordinates) != len(self.values):
            raise ValueError("feature radial coordinates differ from values")
        for decimal_value in (*self.values, *self.radial_coordinates):
            validate_decimal(decimal_value, field_name="feature_value")
        if self.latest_source_coordinate_s is not None:
            validate_decimal(
                self.latest_source_coordinate_s, field_name="latest_source_coordinate_s"
            )
        require_sorted_unique_strings(self.quality_reason_codes, field_name="quality_reason_codes")


@dataclass(frozen=True, slots=True)
class MastCausalState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-causal-state'

    state_id: str
    shot_id: int
    campaign: MastCampaign
    role: MastShotRole
    t0_s: Decimal
    cutoff_rule_id: str
    features: tuple[MastFeatureValue, ...]
    outcome_access: OutcomeAccess
    source_materialization: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        validate_decimal(self.t0_s, field_name="t0_s")
        validate_stable_id(self.cutoff_rule_id, field_name="cutoff_rule_id")
        require_sorted_unique_ids(self.features, attribute="feature_id", field_name="features")
        if {value.role for value in self.features} != set(MastFeatureRole):
            raise ValueError("MAST causal state must carry D, H and R roles")
        if any(
            value.latest_source_coordinate_s is not None
            and value.latest_source_coordinate_s >= self.t0_s
            for value in self.features
        ):
            raise ValueError("MAST causal state contains data at or after t0")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("causal state cannot carry revealed evaluation outcomes")
        if self.role is MastShotRole.PROTECTED_EVALUATION:
            if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
                raise ValueError("protected pre-action state must remain outcome-blind")

    def feature(self, feature_id: str) -> MastFeatureValue:
        for value in self.features:
            if value.feature_id == feature_id:
                return value
        raise KeyError(feature_id)


@dataclass(frozen=True, slots=True)
class MastEndpointResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-endpoint-result'

    endpoint_id: str
    shot_id: int
    event: ObjectIdentity
    horizon_s: Decimal
    delta_te_core_ev: Decimal | None
    delta_te_core_j: Decimal | None
    uncertainty_ev: Decimal | None
    uncertainty_j: Decimal | None
    measurement_uncertainty_available: bool
    state: MastEndpointState
    native_unit: str
    si_unit: str
    evaluable: bool
    reference_coordinate_s: Decimal | None
    endpoint_coordinate_s: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.endpoint_id, field_name="endpoint_id")
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        if self.horizon_s != MAST_ENDPOINT_HORIZON_S:
            raise ValueError("MAST endpoint horizon differs from the frozen 40 ms")
        for name, value in (
            ("delta_te_core_ev", self.delta_te_core_ev),
            ("delta_te_core_j", self.delta_te_core_j),
            ("uncertainty_ev", self.uncertainty_ev),
            ("uncertainty_j", self.uncertainty_j),
            ("reference_coordinate_s", self.reference_coordinate_s),
            ("endpoint_coordinate_s", self.endpoint_coordinate_s),
        ):
            if value is not None:
                validate_decimal(value, field_name=name)
        if self.native_unit != "eV" or self.si_unit != "J":
            raise ValueError("MAST endpoint units differ")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        typed_evaluable = self.state in {
            MastEndpointState.EVALUABLE,
            MastEndpointState.NO_PHYSICAL_RESPONSE,
        }
        if self.evaluable != typed_evaluable:
            raise ValueError("endpoint evaluability differs from its typed state")
        if self.evaluable and self.delta_te_core_ev is None:
            raise ValueError("evaluable endpoint requires a response value")
        if self.evaluable:
            delta_ev = self.delta_te_core_ev
            if delta_ev is None:  # constructor invariant, retained for static narrowing
                raise AssertionError("evaluable endpoint lost its response value")
            if self.delta_te_core_j != delta_ev * ELECTRONVOLT_J:
                raise ValueError("MAST endpoint SI conversion differs")
            if self.measurement_uncertainty_available != (self.uncertainty_ev is not None):
                raise ValueError("MAST uncertainty availability differs from its value")
            if self.uncertainty_ev is None:
                if self.uncertainty_j is not None:
                    raise ValueError("absent MAST uncertainty cannot carry an SI value")
            elif self.uncertainty_j != self.uncertainty_ev * ELECTRONVOLT_J:
                raise ValueError("MAST endpoint uncertainty SI conversion differs")
        elif self.delta_te_core_j is not None or self.uncertainty_j is not None:
            raise ValueError("unevaluable endpoint cannot carry SI response values")
        elif self.measurement_uncertainty_available:
            raise ValueError("unevaluable endpoint cannot claim measurement uncertainty")
        if not self.evaluable and not self.reason_codes:
            raise ValueError("unevaluable endpoint requires reasons")


@dataclass(frozen=True, slots=True)
class FairMastChunkReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-chunk-receipt'

    chunk_id: str
    shot_id: int
    signal_id: str
    chunk_index: int
    source_sha256: str
    raw_artifact: ArtifactIdentity
    raw_locator: str

    def __post_init__(self) -> None:
        validate_stable_id(self.chunk_id, field_name="chunk_id")
        validate_stable_id(self.signal_id, field_name="signal_id")
        if self.shot_id <= 0 or self.chunk_index < 0:
            raise ValueError("invalid FAIR-MAST chunk identity")
        validate_sha256(self.source_sha256, field_name="source_sha256")
        validate_relative_locator(self.raw_locator)
        if self.raw_artifact.sha256 != self.source_sha256:
            raise ValueError("external raw artifact differs from the source chunk")


@dataclass(frozen=True, slots=True)
class FairMastMaterializationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-materialization-receipt'

    receipt_id: str
    source: ObjectIdentity
    shot_id: int
    campaign: MastCampaign
    chunk_receipts: tuple[FairMastChunkReceipt, ...]
    canonical_artifact: ArtifactIdentity
    canonical_locator: str
    external_root: ObjectIdentity
    state: MastMaterializationState
    total_source_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.chunk_receipts, attribute="chunk_id", field_name="chunk_receipts"
        )
        if not self.chunk_receipts:
            raise ValueError("materialization receipt requires raw source chunks")
        validate_relative_locator(self.canonical_locator)
        if self.total_source_bytes != sum(
            value.raw_artifact.size_bytes for value in self.chunk_receipts
        ):
            raise ValueError("materialization byte count differs from raw artifacts")


@dataclass(frozen=True, slots=True)
class MastProtectedPreActionPackage(CanonicalRecord):
    """Projection deliberately incapable of carrying historical action or endpoint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/mast-protected-pre-action-package'

    package_id: str
    assignment: ObjectIdentity
    state: MastCausalState
    challenge_steward_id: str
    action_sealed: bool
    endpoint_sealed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        validate_stable_id(self.challenge_steward_id, field_name="challenge_steward_id")
        if self.state.role is not MastShotRole.PROTECTED_EVALUATION:
            raise ValueError("protected package requires a protected shot state")
        if self.state.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("protected package must remain outcome-blind")
        if not self.action_sealed or not self.endpoint_sealed:
            raise ValueError("historical action and endpoint must remain sealed")
