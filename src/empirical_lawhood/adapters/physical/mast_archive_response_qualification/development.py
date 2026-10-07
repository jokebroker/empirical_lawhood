"Bounded controls for the contaminated FAIR-MAST development slice."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp


# Frozen historical seed identities preserve the outcome-blind development sample.
_HISTORICAL_KDF_ROOT = "el-ec-rcj-io-p5-mw-v4"
_HISTORICAL_KDF_CHILD_GRAMMAR = "archive/v5/development-screen/{campaign_id}/{shot_id}"


@dataclass(frozen=True, slots=True)
class FairMastDevelopmentShotSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-development-shot-spec'

    shot_spec_id: str
    shot_id: int
    campaign_id: str
    nbi_start_time_s: Decimal
    nbi_end_time_s: Decimal
    rmp_coil: bool
    pellets: bool
    remote_command_values_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.shot_spec_id, field_name="shot_spec_id")
        for name in ("nbi_start_time_s", "nbi_end_time_s"):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.shot_spec_id != f"fair-mast-development-shot.{self.shot_id}"
            or self.shot_id <= 0
            or self.campaign_id not in {"M8", "M9"}
            or self.nbi_start_time_s < 0
            or self.nbi_end_time_s <= self.nbi_start_time_s
            or self.rmp_coil
            or self.pellets
        ):
            raise ValueError("FAIR-MAST development shot differs from the bounded development slice")


@dataclass(frozen=True, slots=True)
class FairMastDevelopmentSliceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-development-slice-spec'

    slice_id: str
    amendment: ObjectIdentity
    shots: tuple[FairMastDevelopmentShotSpec, ...]
    remote_value_array_paths: tuple[str, ...]
    held_stage0_array_paths: tuple[str, ...]
    maximum_remote_source_bytes: int
    new_receiver_acquisition_allowed: bool
    protected_roster_membership_allowed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.slice_id, field_name="slice_id")
        require_sorted_unique_ids(self.shots, attribute="shot_spec_id", field_name="shots")
        require_sorted_unique_strings(
            self.remote_value_array_paths,
            field_name="remote_value_array_paths",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.held_stage0_array_paths,
            field_name="held_stage0_array_paths",
            allow_empty=False,
        )
        if (
            tuple(value.shot_id for value in self.shots) != (27582, 29643)
            or tuple(value.remote_command_values_required for value in self.shots) != (True, False)
            or self.remote_value_array_paths
            != (
                "gas_injection/time",
                "gas_injection/valve_target_voltage",
                "pulse_schedule/i_plasma",
                "pulse_schedule/n_e_line",
                "pulse_schedule/time",
                "pulse_schedule/z_ref",
            )
            or self.maximum_remote_source_bytes != 16 * 1024 * 1024
            or self.new_receiver_acquisition_allowed
            or self.protected_roster_membership_allowed
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("FAIR-MAST development slice broadens its bounded authority")


@dataclass(frozen=True, slots=True)
class MastArchiveDevelopmentApproval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-development-approval'

    approval_id: str
    subject: ObjectIdentity
    approver: ObjectIdentity
    passed_gate_ids: tuple[str, ...]
    authorization_basis_sha256: str
    approved_at_utc: str
    codex_or_chat_is_approver_or_issuer: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.approval_id, field_name="approval_id")
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.authorization_basis_sha256,
            field_name="authorization_basis_sha256",
        )
        parse_utc_timestamp(self.approved_at_utc, field_name="approved_at_utc")
        if (
            self.codex_or_chat_is_approver_or_issuer
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("development approval assigns authority to Codex or sees outcomes")


@dataclass(frozen=True, slots=True)
class FairMastActionScreenSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-action-screen-spec'

    screen_id: str
    amendment: ObjectIdentity
    shots: tuple[FairMastDevelopmentShotSpec, ...]
    kdf_root: str
    kdf_child_grammar: str
    source_array_paths: tuple[str, ...]
    maximum_remote_source_bytes: int
    new_receiver_acquisition_allowed: bool
    protected_roster_membership_allowed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.screen_id, field_name="screen_id")
        require_sorted_unique_ids(self.shots, attribute="shot_spec_id", field_name="shots")
        require_sorted_unique_strings(
            self.source_array_paths,
            field_name="source_array_paths",
            allow_empty=False,
        )
        campaigns = tuple(value.campaign_id for value in self.shots)
        if (
            len(self.shots) != 20
            or campaigns.count("M8") != 10
            or campaigns.count("M9") != 10
            or any(not value.remote_command_values_required for value in self.shots)
            or self.kdf_root != _HISTORICAL_KDF_ROOT
            or self.kdf_child_grammar != _HISTORICAL_KDF_CHILD_GRAMMAR
            or self.source_array_paths
            != (
                "gas_injection/time",
                "gas_injection/valve_target_voltage",
                "pulse_schedule/i_plasma",
                "pulse_schedule/n_e_line",
                "pulse_schedule/time",
                "pulse_schedule/z_ref",
                "summary/power_nbi",
                "summary/time",
            )
            or self.maximum_remote_source_bytes != 256 * 1024 * 1024
            or self.new_receiver_acquisition_allowed
            or self.protected_roster_membership_allowed
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("FAIR-MAST action screen differs from its bounded KDF sample")


__all__ = [
    'FairMastActionScreenSpec',
    'FairMastDevelopmentShotSpec',
    'FairMastDevelopmentSliceSpec',
    'MastArchiveDevelopmentApproval',
]
