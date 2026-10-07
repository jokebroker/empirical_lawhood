"""Fail-closed field projection for Six-matrix response's procedural-blindness claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .contracts import ProtectedAccessClaim, ProtectedFieldAccessRule, ProtectedObservableAccessManifest


_STAGE_ORDER = {
    "six-matrix-response.stage.q4-excluded-qualification": 0,
    "six-matrix-response.stage.structural-prediction-prediction": 1,
    "six-matrix-response.stage.prospective-execution-execution": 2,
    "six-matrix-response.stage.authorized-reveal": 3,
}


@dataclass(frozen=True, slots=True)
class MatrixResponseProtectedFieldValue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-protected-field-value'
    VERSION: ClassVar[str] = "1.0.0"

    field_id: str
    value_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        validate_sha256(self.value_sha256, field_name="value_sha256")


@dataclass(frozen=True, slots=True)
class MatrixResponseProtectedFieldEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-protected-field-envelope'
    VERSION: ClassVar[str] = "1.0.0"

    envelope_id: str
    fields: tuple[MatrixResponseProtectedFieldValue, ...]
    sealed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        require_sorted_unique_ids(self.fields, attribute="field_id", field_name="fields")
        if not self.fields or not self.sealed:
            raise ValueError("Six-matrix response protected field envelope must be nonempty and sealed")


@dataclass(frozen=True, slots=True)
class MatrixResponseProceduralBlindnessMechanism(CanonicalRecord):
    """Exact honest claim for the observed vfat/single-owner environment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-procedural-blindness-mechanism'
    VERSION: ClassVar[str] = "1.0.0"

    mechanism_id: str
    operating_system_boundary_id: str
    filesystem_type: str
    single_owner_environment: bool
    multiprincipal_acl_enforced: bool
    role_separation_procedural: bool
    human_operator_abstention_required: bool
    system_enforced_blinding: bool
    confidentiality_claimed: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("mechanism_id", "operating_system_boundary_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.filesystem_type != "vfat"
            or not self.single_owner_environment
            or self.multiprincipal_acl_enforced
            or not self.role_separation_procedural
            or not self.human_operator_abstention_required
            or self.system_enforced_blinding
            or self.confidentiality_claimed
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response mechanism overclaims the observed procedural boundary")


@dataclass(frozen=True, slots=True)
class MatrixResponseProtectedFieldProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-protected-field-projection'
    VERSION: ClassVar[str] = "1.0.0"

    projection_id: str
    manifest: ObjectIdentity
    mechanism: ObjectIdentity
    envelope: ObjectIdentity
    principal_id: str
    stage_id: str
    fields: tuple[MatrixResponseProtectedFieldValue, ...]
    access_claim: ProtectedAccessClaim
    confidentiality_claimed: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("projection_id", "principal_id", "stage_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.manifest.object_schema != ProtectedObservableAccessManifest.SCHEMA:
            raise ValueError("Six-matrix response field projection binds another manifest schema")
        if self.mechanism.object_schema != MatrixResponseProceduralBlindnessMechanism.SCHEMA:
            raise ValueError("Six-matrix response field projection binds another blindness mechanism")
        if self.envelope.object_schema != MatrixResponseProtectedFieldEnvelope.SCHEMA:
            raise ValueError("Six-matrix response field projection binds another envelope schema")
        require_sorted_unique_ids(self.fields, attribute="field_id", field_name="fields")
        if not self.fields:
            raise ValueError("Six-matrix response allowed field projection cannot be empty")
        if (
            self.access_claim is not ProtectedAccessClaim.PROCEDURAL_BLINDNESS_ONLY
            or self.confidentiality_claimed
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response projection overclaims confidentiality or authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseProtectedWriteScope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-protected-write-scope'
    VERSION: ClassVar[str] = "1.0.0"

    scope_id: str
    manifest: ObjectIdentity
    mechanism: ObjectIdentity
    principal_id: str
    stage_id: str
    field_ids: tuple[str, ...]
    readable: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("scope_id", "principal_id", "stage_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.field_ids, field_name="field_ids", allow_empty=False)
        if self.readable or self.grants_authority:
            raise ValueError("Six-matrix response sealed-writer scope cannot read or grant authority")


class MatrixResponseProtectedFieldProjector:
    def __init__(
        self,
        *,
        manifest: ProtectedObservableAccessManifest,
        mechanism: MatrixResponseProceduralBlindnessMechanism,
    ) -> None:
        if (
            manifest.access_claim is not ProtectedAccessClaim.PROCEDURAL_BLINDNESS_ONLY
            or manifest.operating_system_boundary_id != mechanism.operating_system_boundary_id
            or manifest.confidentiality_claimed
        ):
            raise ValueError("Six-matrix response field projector mechanism differs from its honest manifest")
        if any(value.earliest_stage_id not in _STAGE_ORDER for value in manifest.rules):
            raise ValueError("Six-matrix response manifest contains an unordered access stage")
        self.manifest = manifest
        self.mechanism = mechanism

    def _rule(
        self,
        *,
        principal_id: str,
        access_mode: str,
        stage_id: str,
    ) -> ProtectedFieldAccessRule:
        validate_stable_id(principal_id, field_name="principal_id")
        if stage_id not in _STAGE_ORDER:
            raise PermissionError("Six-matrix response access stage is not declared")
        rules = tuple(
            value
            for value in self.manifest.rules
            if value.principal_id == principal_id and value.access_mode == access_mode
        )
        if len(rules) != 1:
            raise PermissionError("Six-matrix response principal lacks the requested closed access mode")
        rule = rules[0]
        if _STAGE_ORDER[stage_id] < _STAGE_ORDER[rule.earliest_stage_id]:
            raise PermissionError("Six-matrix response protected access precedes its declared stage")
        return rule

    def project_read(
        self,
        *,
        projection_id: str,
        envelope: MatrixResponseProtectedFieldEnvelope,
        principal_id: str,
        stage_id: str,
        requested_field_ids: tuple[str, ...],
    ) -> MatrixResponseProtectedFieldProjection:
        require_sorted_unique_strings(
            requested_field_ids,
            field_name="requested_field_ids",
            allow_empty=False,
        )
        rule = self._rule(principal_id=principal_id, access_mode="read", stage_id=stage_id)
        allowed = set(rule.field_ids)
        if not set(requested_field_ids).issubset(allowed):
            raise PermissionError("Six-matrix response protected projection requests a denied field")
        by_id = {value.field_id: value for value in envelope.fields}
        if not set(requested_field_ids).issubset(by_id):
            raise ValueError("Six-matrix response protected projection requests a missing field")
        return MatrixResponseProtectedFieldProjection(
            projection_id=projection_id,
            manifest=ObjectIdentity.from_record(self.manifest.manifest_id, self.manifest),
            mechanism=ObjectIdentity.from_record(self.mechanism.mechanism_id, self.mechanism),
            envelope=ObjectIdentity.from_record(envelope.envelope_id, envelope),
            principal_id=principal_id,
            stage_id=stage_id,
            fields=tuple(by_id[value] for value in requested_field_ids),
            access_claim=self.manifest.access_claim,
            confidentiality_claimed=False,
            grants_authority=False,
        )

    def authorize_write_scope(
        self,
        *,
        scope_id: str,
        principal_id: str,
        stage_id: str,
        field_ids: tuple[str, ...],
    ) -> MatrixResponseProtectedWriteScope:
        require_sorted_unique_strings(field_ids, field_name="field_ids", allow_empty=False)
        rule = self._rule(principal_id=principal_id, access_mode="write-only", stage_id=stage_id)
        if not set(field_ids).issubset(rule.field_ids):
            raise PermissionError("Six-matrix response sealed writer requests a denied field")
        return MatrixResponseProtectedWriteScope(
            scope_id=scope_id,
            manifest=ObjectIdentity.from_record(self.manifest.manifest_id, self.manifest),
            mechanism=ObjectIdentity.from_record(self.mechanism.mechanism_id, self.mechanism),
            principal_id=principal_id,
            stage_id=stage_id,
            field_ids=field_ids,
            readable=False,
            grants_authority=False,
        )


def build_matrix_response_study_procedural_blindness_projector(
    manifest: ProtectedObservableAccessManifest,
) -> MatrixResponseProtectedFieldProjector:
    mechanism = MatrixResponseProceduralBlindnessMechanism(
        mechanism_id="six-matrix-response.procedural-blindness.single-owner-vfat",
        operating_system_boundary_id=manifest.operating_system_boundary_id,
        filesystem_type="vfat",
        single_owner_environment=True,
        multiprincipal_acl_enforced=False,
        role_separation_procedural=True,
        human_operator_abstention_required=True,
        system_enforced_blinding=False,
        confidentiality_claimed=False,
        grants_authority=False,
    )
    return MatrixResponseProtectedFieldProjector(manifest=manifest, mechanism=mechanism)


__all__ = [
    'MatrixResponseProceduralBlindnessMechanism',
    'MatrixResponseProtectedFieldEnvelope',
    'MatrixResponseProtectedFieldProjection',
    'MatrixResponseProtectedFieldProjector',
    'MatrixResponseProtectedFieldValue',
    'MatrixResponseProtectedWriteScope',
    'build_matrix_response_study_procedural_blindness_projector',
]
