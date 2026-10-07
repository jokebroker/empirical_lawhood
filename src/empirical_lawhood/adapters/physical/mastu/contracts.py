"""Strict records for the outcome-visible MAST-U grounding reuse act."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


MASTU_SYSTEM_ID_DISCHARGES: Final = (47080, 47083, 47086, 47116, 47118, 47119)
MASTU_COUPLING_DISCHARGES: Final = (49297, 49298)
MASTU_FEEDBACK_DISCHARGES: Final = (47998, 48001)
MASTU_BOUNDARY_DISCHARGES: Final = (49303,)
MASTU_ALL_DISCHARGES: Final = tuple(
    sorted(
        (
            *MASTU_SYSTEM_ID_DISCHARGES,
            *MASTU_COUPLING_DISCHARGES,
            *MASTU_FEEDBACK_DISCHARGES,
            *MASTU_BOUNDARY_DISCHARGES,
        )
    )
)


class MastuDischargeRole(StrEnum):
    SYSTEM_IDENTIFICATION = "SYSTEM_IDENTIFICATION"
    DIRECTIONAL_COUPLING = "DIRECTIONAL_COUPLING"
    FEEDBACK_ALIGNMENT = "FEEDBACK_ALIGNMENT"
    BOUNDARY_CANDIDATE = "BOUNDARY_CANDIDATE"


@dataclass(frozen=True, slots=True)
class MastuDischargeUnit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-discharge-unit'

    unit_id: str
    discharge_id: int
    role: MastuDischargeRole
    geometry: str
    response_recurrence_unit: bool
    nested_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        if self.unit_id != f"mastu-{self.discharge_id}":
            raise ValueError("MAST-U unit ID differs from discharge identity")
        if self.discharge_id not in MASTU_ALL_DISCHARGES:
            raise ValueError("MAST-U discharge is outside the exact held set")
        if not isinstance(self.role, MastuDischargeRole):
            raise ValueError("unknown MAST-U discharge role")
        validate_nonempty(self.geometry, field_name="geometry")
        if self.response_recurrence_unit is not (
            self.role is MastuDischargeRole.SYSTEM_IDENTIFICATION
        ):
            raise ValueError("only system-identification discharges support recurrence")
        require_sorted_unique_strings(
            self.nested_observation_ids,
            field_name="nested_observation_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class MastuSignalRole(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-signal-role'

    signal_id: str
    semantic_label: str
    native_unit: str
    clock_semantic: str
    is_requested_action: bool
    is_receiver: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.signal_id, field_name="signal_id")
        validate_nonempty(self.semantic_label, field_name="semantic_label")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.clock_semantic, field_name="clock_semantic")
        if self.is_requested_action == self.is_receiver:
            raise ValueError("MAST-U signal must have exactly one action/receiver role")
        expected = {
            "d2-valve-flow-request": ("s^-1", True),
            "dalpha-hm10et": ("V", False),
            "front-ltar": ("m", False),
            "line-integrated-density": ("m^-2", False),
        }
        if self.signal_id not in expected:
            raise ValueError("MAST-U signal role is outside the frozen gauge")
        unit, action = expected[self.signal_id]
        if self.native_unit != unit or self.is_requested_action is not action:
            raise ValueError("MAST-U signal unit/action role differs")


@dataclass(frozen=True, slots=True)
class MastuGroundingReuseFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-grounding-reuse-freeze'

    run_id: str
    source_artifacts: tuple[ArtifactIdentity, ...]
    prior_result_artifacts: tuple[ArtifactIdentity, ...]
    adapter_implementation_sha256: str
    compatibility_implementation_sha256: str
    protocol_config_sha256: str
    discharges: tuple[MastuDischargeUnit, ...]
    signals: tuple[MastuSignalRole, ...]
    window_rule: str
    estimator_id: str
    random_seed: int
    snr_threshold: Decimal
    comparison_rule: str
    claim_ceiling: str
    new_independent_unit_count: int
    evaluation_sealed: bool
    outcome_visible_reuse: bool

    def __post_init__(self) -> None:
        if self.run_id != "mastu-retrospective-grounding-reuse":
            raise ValueError("MAST-U grounding reuse run ID differs")
        require_sorted_unique_ids(
            self.source_artifacts,
            attribute="artifact_id",
            field_name="source_artifacts",
        )
        if tuple(value.artifact_id for value in self.source_artifacts) != (
            "mastu-code-archive",
            "mastu-data-archive",
        ):
            raise ValueError("MAST-U source role set differs")
        require_sorted_unique_ids(
            self.prior_result_artifacts,
            attribute="artifact_id",
            field_name="prior_result_artifacts",
        )
        if len(self.prior_result_artifacts) < 3:
            raise ValueError("MAST-U reuse freeze requires prior table identities")
        for name, value in (
            ("adapter_implementation_sha256", self.adapter_implementation_sha256),
            ("compatibility_implementation_sha256", self.compatibility_implementation_sha256),
            ("protocol_config_sha256", self.protocol_config_sha256),
        ):
            validate_sha256(value, field_name=name)
        if tuple(value.discharge_id for value in self.discharges) != MASTU_ALL_DISCHARGES:
            raise ValueError("MAST-U reuse discharge set differs or is unsorted")
        if len({value.discharge_id for value in self.discharges}) != len(self.discharges):
            raise ValueError("MAST-U reuse discharges are duplicated")
        if tuple(value.signal_id for value in self.signals) != (
            "d2-valve-flow-request",
            "dalpha-hm10et",
            "front-ltar",
            "line-integrated-density",
        ):
            raise ValueError("MAST-U reuse action/receiver gauge differs")
        if sum(value.response_recurrence_unit for value in self.discharges) != 6:
            raise ValueError("MAST-U response recurrence requires exactly six discharges")
        validate_nonempty(self.window_rule, field_name="window_rule")
        if self.estimator_id != "direct-irregular-time-discrete-fourier-transform":
            raise ValueError("MAST-U reuse estimator differs")
        if self.random_seed != 20260710:
            raise ValueError("MAST-U reuse random seed differs")
        validate_decimal(self.snr_threshold, field_name="snr_threshold", minimum=Decimal(0))
        if self.snr_threshold != Decimal(5):
            raise ValueError("MAST-U reuse SNR threshold differs")
        if self.comparison_rule != "byte-and-numeric-table-equality-against-immutable-prior":
            raise ValueError("MAST-U prior comparison rule differs")
        if self.claim_ceiling != "RETROSPECTIVE_REUSE_ONLY":
            raise ValueError("MAST-U reuse claim ceiling differs")
        if self.new_independent_unit_count != 0:
            raise ValueError("MAST-U reuse cannot create new independent units")
        if self.evaluation_sealed or not self.outcome_visible_reuse:
            raise ValueError("MAST-U reuse must remain outcome-visible and unsealed")


@dataclass(frozen=True, slots=True)
class MastuTableComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-table-comparison'

    comparison_id: str
    table_id: str
    prior_sha256: str
    recomputed_sha256: str
    prior_row_count: int
    recomputed_row_count: int
    byte_equal: bool
    numeric_equal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        validate_stable_id(self.table_id, field_name="table_id")
        if self.comparison_id != f"comparison.{self.table_id}":
            raise ValueError("MAST-U comparison ID differs")
        validate_sha256(self.prior_sha256, field_name="prior_sha256")
        validate_sha256(self.recomputed_sha256, field_name="recomputed_sha256")
        if self.prior_row_count < 0 or self.recomputed_row_count < 0:
            raise ValueError("MAST-U comparison row counts must be nonnegative")
        if self.byte_equal and self.prior_sha256 != self.recomputed_sha256:
            raise ValueError("byte-equal MAST-U tables have different hashes")
        if self.numeric_equal and self.prior_row_count != self.recomputed_row_count:
            raise ValueError("numeric-equal MAST-U tables have different row counts")


@dataclass(frozen=True, slots=True)
class MastuComparisonErratum(CanonicalRecord):
    """Controlling correction when byte equality dominates a faulty secondary comparator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-comparison-erratum'

    erratum_id: str
    original_result: ArtifactIdentity
    original_comparison_table: ArtifactIdentity
    table_count: int
    byte_equal_table_count: int
    original_secondary_numeric_equal_count: int
    corrected_numeric_equal_count: int
    proof_rule: str
    controlling: bool

    def __post_init__(self) -> None:
        if self.erratum_id != "mastu-retrospective-grounding-reuse-comparison-erratum":
            raise ValueError("MAST-U comparison erratum ID differs")
        if self.table_count != 14 or self.byte_equal_table_count != 14:
            raise ValueError("MAST-U comparison erratum requires 14/14 byte equality")
        if self.original_secondary_numeric_equal_count != 3:
            raise ValueError("MAST-U comparison erratum source count differs")
        if self.corrected_numeric_equal_count != 14:
            raise ValueError("MAST-U corrected numeric equality count differs")
        if (
            self.proof_rule
            != "identical-csv-bytes-and-sha256-imply-identical-parsed-numeric-values"
        ):
            raise ValueError("MAST-U comparison erratum proof rule differs")
        if not self.controlling:
            raise ValueError("MAST-U comparison erratum must be controlling")


@dataclass(frozen=True, slots=True)
class MastuGroundingReuseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu/mastu-grounding-reuse-result'

    result_id: str
    freeze: ObjectIdentity
    table_artifacts: tuple[ArtifactIdentity, ...]
    comparisons: tuple[MastuTableComparison, ...]
    metrics: tuple[NamedDecimal, ...]
    operational_status: str
    scientific_status: str
    claim_ceiling: str
    response_recurrence_discharge_count: int
    held_distinct_discharge_count: int
    new_independent_unit_count: int
    fresh_mu_action_response: str
    fresh_mu_law_qualification: str

    def __post_init__(self) -> None:
        if self.result_id != "mastu-retrospective-grounding-reuse-result":
            raise ValueError("MAST-U reuse result ID differs")
        if self.freeze.object_schema != MastuGroundingReuseFreeze.SCHEMA:
            raise ValueError("MAST-U reuse result freeze identity differs")
        require_sorted_unique_ids(
            self.table_artifacts,
            attribute="artifact_id",
            field_name="table_artifacts",
        )
        if not self.table_artifacts:
            raise ValueError("MAST-U reuse result requires recomputed tables")
        if tuple(value.table_id for value in self.comparisons) != tuple(
            value.artifact_id.removeprefix("table.") for value in self.table_artifacts
        ):
            raise ValueError("MAST-U comparison/table roles differ")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        if self.operational_status != "PASS":
            raise ValueError("canonical MAST-U reuse result requires operational PASS")
        if self.scientific_status != "RETROSPECTIVE_REPRODUCED":
            raise ValueError("MAST-U reuse scientific status differs")
        if self.claim_ceiling != "RETROSPECTIVE_REUSE_ONLY":
            raise ValueError("MAST-U reuse result ceiling differs")
        if (
            self.response_recurrence_discharge_count != 6
            or self.held_distinct_discharge_count != 11
            or self.new_independent_unit_count != 0
        ):
            raise ValueError("MAST-U reuse independent-unit counts differ")
        if (
            self.fresh_mu_action_response != "NOT_SUPPORTED_ZERO_NEW_UNITS"
            or self.fresh_mu_law_qualification != "NOT_ATTEMPTED"
        ):
            raise ValueError("MAST-U reuse fresh proposition status differs")


__all__ = [
    "MASTU_ALL_DISCHARGES",
    "MASTU_BOUNDARY_DISCHARGES",
    "MASTU_COUPLING_DISCHARGES",
    "MASTU_FEEDBACK_DISCHARGES",
    "MASTU_SYSTEM_ID_DISCHARGES",
    "MastuComparisonErratum",
    "MastuDischargeRole",
    "MastuDischargeUnit",
    "MastuGroundingReuseFreeze",
    "MastuGroundingReuseResult",
    "MastuSignalRole",
    "MastuTableComparison",
]
