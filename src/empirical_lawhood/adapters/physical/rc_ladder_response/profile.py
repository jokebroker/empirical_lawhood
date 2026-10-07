"""Strict semantic profile for dossier-bound physical scale morphism RC-ladder bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)

from empirical_lawhood.adapters.methods.physical_scale_morphism.governance import PhysicalScaleMorphismConstructQualification, PhysicalScaleMorphismConstructTerminal


class RcLadderResponseStatusMeaning(StrEnum):
    CLIPPING = "CLIPPING"
    INTERLOCK = "INTERLOCK"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"
    NON_DELIVERY = "NON_DELIVERY"
    NORMAL = "NORMAL"
    OPERATOR_EVENT = "OPERATOR_EVENT"
    SATURATION = "SATURATION"
    SLEW_LIMIT = "SLEW_LIMIT"


@dataclass(frozen=True, slots=True)
class RcLadderResponseStatusCodeDefinition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-status-code-definition'

    code_id: str
    native_code: int
    meaning: RcLadderResponseStatusMeaning
    invalidates_observation: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.code_id, field_name="code_id")
        if not -(2**15) <= self.native_code < 2**15:
            raise ValueError("physical scale morphism status code must fit the frozen signed-int16 payload")
        expected_invalid = self.meaning in {
            RcLadderResponseStatusMeaning.INTERLOCK,
            RcLadderResponseStatusMeaning.INVALID_OBSERVATION,
            RcLadderResponseStatusMeaning.SATURATION,
        }
        if self.invalidates_observation != expected_invalid:
            raise ValueError("status invalidity is not derived from its semantic meaning")


@dataclass(frozen=True, slots=True)
class RcLadderResponsePhysicalSemanticProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-physical-semantic-profile'

    profile_id: str
    dossier: ObjectIdentity | None
    construct_qualification: ObjectIdentity | None
    allowed_scale_cells: tuple[int, ...]
    voltage_unit: str
    current_unit: str
    clock_unit: str
    gauge_id: str
    terminal_current_sign_id: str
    channel_rule_id: str
    environment_channel_ids: tuple[str, ...]
    environment_channel_units: tuple[str, ...]
    status_codebook: tuple[RcLadderResponseStatusCodeDefinition, ...]
    maximum_manifest_bytes: int
    maximum_member_bytes: int
    exact_dossier_bound: bool
    source_read_only: bool
    physical_action_supported: bool

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "gauge_id",
            "terminal_current_sign_id",
            "channel_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if tuple(sorted(set(self.allowed_scale_cells))) != self.allowed_scale_cells:
            raise ValueError("semantic-profile scale roster must be sorted and unique")
        if not self.allowed_scale_cells or not set(self.allowed_scale_cells).issubset({16, 32, 64}):
            raise ValueError("semantic-profile scale roster lies outside N16/N32/N64")
        if (self.voltage_unit, self.current_unit, self.clock_unit) != ("V", "A", "s"):
            raise ValueError("physical scale morphism semantic profile requires exact native SI units")
        require_sorted_unique_strings(
            self.environment_channel_ids,
            field_name="environment_channel_ids",
        )
        if len(self.environment_channel_units) != len(self.environment_channel_ids):
            raise ValueError("semantic-profile environment channel/unit rosters differ")
        for index, unit in enumerate(self.environment_channel_units):
            validate_nonempty(unit, field_name=f"environment_channel_units[{index}]")
        require_sorted_unique_ids(
            self.status_codebook, attribute="code_id", field_name="status_codebook"
        )
        native_codes = tuple(value.native_code for value in self.status_codebook)
        if len(set(native_codes)) != len(native_codes):
            raise ValueError("semantic profile repeats a native status code")
        normal = [
            value for value in self.status_codebook if value.meaning is RcLadderResponseStatusMeaning.NORMAL
        ]
        if len(normal) != 1:
            raise ValueError("semantic profile requires exactly one normal status code")
        if self.maximum_manifest_bytes <= 0 or self.maximum_member_bytes <= 0:
            raise ValueError("semantic-profile byte limits must be positive")
        if self.maximum_manifest_bytes > 16 * 1024 * 1024:
            raise ValueError("semantic-profile manifest limit exceeds the bounded decoder ceiling")
        if self.maximum_member_bytes > 128 * 1024 * 1024:
            raise ValueError("semantic-profile member limit exceeds the bounded decoder ceiling")
        if self.exact_dossier_bound:
            if self.dossier is None or self.construct_qualification is None:
                raise ValueError("exact semantic profile lacks dossier/construct identities")
            if self.dossier.object_schema != 'empirical-lawhood/methods/physical-scale-morphism/independent-apparatus-dossier':
                raise ValueError("semantic profile dossier identity has the wrong schema")
            if self.construct_qualification.object_schema != (
                'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-construct-qualification'
            ):
                raise ValueError("semantic profile construct identity has the wrong schema")
        elif self.dossier is not None or self.construct_qualification is not None:
            raise ValueError("draft semantic profile cannot impersonate a dossier-bound profile")
        if not self.source_read_only or self.physical_action_supported:
            raise ValueError("physical scale morphism physical archive profile is read-only and nonactuating")


def draft_semantic_profile() -> RcLadderResponsePhysicalSemanticProfile:
    """Return the synthetic-only draft profile before independent construct review."""

    return RcLadderResponsePhysicalSemanticProfile(
        profile_id="physical-scale-morphism-synthetic-draft-profile",
        dossier=None,
        construct_qualification=None,
        allowed_scale_cells=(16,),
        voltage_unit="V",
        current_unit="A",
        clock_unit="s",
        gauge_id="node-to-ground",
        terminal_current_sign_id="positive-current-into-ladder",
        channel_rule_id="rc-node-voltage-terminal-current",
        environment_channel_ids=("temperature-celsius",),
        environment_channel_units=("degC",),
        status_codebook=(
            RcLadderResponseStatusCodeDefinition(
                code_id="status.normal",
                native_code=0,
                meaning=RcLadderResponseStatusMeaning.NORMAL,
                invalidates_observation=False,
            ),
        ),
        maximum_manifest_bytes=4 * 1024 * 1024,
        maximum_member_bytes=128 * 1024 * 1024,
        exact_dossier_bound=False,
        source_read_only=True,
        physical_action_supported=False,
    )


def derive_dossier_bound_semantic_profile(
    *,
    profile_id: str,
    construct_qualification: PhysicalScaleMorphismConstructQualification,
    gauge_id: str,
    terminal_current_sign_id: str,
    channel_rule_id: str,
    environment_channel_ids: tuple[str, ...],
    environment_channel_units: tuple[str, ...],
    status_codebook: tuple[RcLadderResponseStatusCodeDefinition, ...],
) -> RcLadderResponsePhysicalSemanticProfile:
    """Bind a native codebook only after an actually qualified IP-6 record."""

    if (
        construct_qualification.terminal
        is not PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_QUALIFIED
        or not construct_qualification.source_profile_derivation_allowed
    ):
        raise ValueError("dossier-bound source profile requires qualified construct independence")
    return RcLadderResponsePhysicalSemanticProfile(
        profile_id=profile_id,
        dossier=construct_qualification.dossier,
        construct_qualification=ObjectIdentity.from_record(
            construct_qualification.qualification_id,
            construct_qualification,
        ),
        allowed_scale_cells=(16, 32, 64),
        voltage_unit="V",
        current_unit="A",
        clock_unit="s",
        gauge_id=gauge_id,
        terminal_current_sign_id=terminal_current_sign_id,
        channel_rule_id=channel_rule_id,
        environment_channel_ids=environment_channel_ids,
        environment_channel_units=environment_channel_units,
        status_codebook=status_codebook,
        maximum_manifest_bytes=4 * 1024 * 1024,
        maximum_member_bytes=128 * 1024 * 1024,
        exact_dossier_bound=True,
        source_read_only=True,
        physical_action_supported=False,
    )


__all__ = [
    'RcLadderResponsePhysicalSemanticProfile',
    'RcLadderResponseStatusCodeDefinition',
    'RcLadderResponseStatusMeaning',
    "derive_dossier_bound_semantic_profile",
    "draft_semantic_profile",
]
