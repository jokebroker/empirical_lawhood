"""Outcome-blind staged preparation and fresh-target issue for Six-matrix response."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .prospective import MatrixResponsePhysicalPanel


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-preparation-request'
    VERSION: ClassVar[str] = "1.0.0"

    request_id: str
    source_config: ObjectIdentity
    panel: ObjectIdentity
    preparation_coordinate_id: str
    permitted_preparation_field_ids: tuple[str, ...]
    stage_index: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("request_id", "preparation_coordinate_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.source_config.object_schema != SixMatrixResponseSixMatrixSourceConfig.SCHEMA
            or self.panel.object_schema != MatrixResponsePhysicalPanel.SCHEMA
        ):
            raise ValueError("Six-matrix response preparation request binds another source/panel schema")
        require_sorted_unique_strings(
            self.permitted_preparation_field_ids,
            field_name="permitted_preparation_field_ids",
            allow_empty=False,
        )
        if self.stage_index != 0:
            raise ValueError("Six-matrix response preparation request must begin the staged issue sequence")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.visibility_ceiling.is_promotable
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response preparation request crosses outcome visibility or authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseRealizedPreparationCheckpoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-realized-preparation-checkpoint'
    VERSION: ClassVar[str] = "1.0.0"

    checkpoint_id: str
    request: ObjectIdentity
    panel: ObjectIdentity
    realized_state_sha256: str
    realized_state_size_bytes: int
    source_contact_count: int
    outcome_field_ids_read: tuple[str, ...]
    stage_index: int
    outcome_access: OutcomeAccess
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        validate_sha256(self.realized_state_sha256, field_name="realized_state_sha256")
        if (
            self.request.object_schema != MatrixResponsePreparationRequest.SCHEMA
            or self.panel.object_schema != MatrixResponsePhysicalPanel.SCHEMA
        ):
            raise ValueError("Six-matrix response realized checkpoint binds another request/panel schema")
        require_sorted_unique_strings(
            self.outcome_field_ids_read,
            field_name="outcome_field_ids_read",
        )
        if (
            self.realized_state_size_bytes < 1
            or self.source_contact_count != 1
            or self.outcome_field_ids_read
            or self.stage_index != 1
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response preparation checkpoint crossed staged/outcome bounds")


@dataclass(frozen=True, slots=True)
class MatrixResponseFreshTargetIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-fresh-target-issue'
    VERSION: ClassVar[str] = "1.0.0"

    issue_id: str
    preparation_request: ObjectIdentity
    realized_checkpoint: ObjectIdentity
    panel: ObjectIdentity
    realized_state_sha256: str
    target_template_id: str
    protected_outcome_field_ids_read: tuple[str, ...]
    stage_index: int
    frozen_before_target_outcomes: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("issue_id", "target_template_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.realized_state_sha256, field_name="realized_state_sha256")
        if (
            self.preparation_request.object_schema != MatrixResponsePreparationRequest.SCHEMA
            or self.realized_checkpoint.object_schema != MatrixResponseRealizedPreparationCheckpoint.SCHEMA
            or self.panel.object_schema != MatrixResponsePhysicalPanel.SCHEMA
        ):
            raise ValueError("Six-matrix response target issue binds another staged-preparation schema")
        require_sorted_unique_strings(
            self.protected_outcome_field_ids_read,
            field_name="protected_outcome_field_ids_read",
        )
        if (
            self.protected_outcome_field_ids_read
            or self.stage_index != 2
            or not self.frozen_before_target_outcomes
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.visibility_ceiling.is_promotable
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response fresh target issue crossed outcome visibility or authority")


def realize_matrix_response_study_preparation_checkpoint(
    *,
    checkpoint_id: str,
    request: MatrixResponsePreparationRequest,
    panel: MatrixResponsePhysicalPanel,
    realized_state: bytes,
) -> MatrixResponseRealizedPreparationCheckpoint:
    if not realized_state:
        raise ValueError("Six-matrix response realized checkpoint cannot hash an empty state")
    panel_identity = ObjectIdentity.from_record(panel.panel_id, panel)
    if request.panel != panel_identity:
        raise ValueError("Six-matrix response preparation checkpoint substitutes its panel")
    return MatrixResponseRealizedPreparationCheckpoint(
        checkpoint_id=checkpoint_id,
        request=ObjectIdentity.from_record(request.request_id, request),
        panel=panel_identity,
        realized_state_sha256=sha256(realized_state).hexdigest(),
        realized_state_size_bytes=len(realized_state),
        source_contact_count=1,
        outcome_field_ids_read=(),
        stage_index=1,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        grants_authority=False,
    )


def issue_matrix_response_study_target(
    *,
    issue_id: str,
    target_template_id: str,
    request: MatrixResponsePreparationRequest,
    checkpoint: MatrixResponseRealizedPreparationCheckpoint,
    panel: MatrixResponsePhysicalPanel,
) -> MatrixResponseFreshTargetIssue:
    panel_identity = ObjectIdentity.from_record(panel.panel_id, panel)
    request_identity = ObjectIdentity.from_record(request.request_id, request)
    if (
        checkpoint.request != request_identity
        or checkpoint.panel != panel_identity
        or request.panel != panel_identity
    ):
        raise ValueError("Six-matrix response target issue substitutes its staged request/checkpoint/panel")
    return MatrixResponseFreshTargetIssue(
        issue_id=issue_id,
        preparation_request=request_identity,
        realized_checkpoint=ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
        panel=panel_identity,
        realized_state_sha256=checkpoint.realized_state_sha256,
        target_template_id=target_template_id,
        protected_outcome_field_ids_read=(),
        stage_index=2,
        frozen_before_target_outcomes=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseFreshTargetIssue',
    'MatrixResponsePreparationRequest',
    'MatrixResponseRealizedPreparationCheckpoint',
    'issue_matrix_response_study_target',
    'realize_matrix_response_study_preparation_checkpoint',
]
