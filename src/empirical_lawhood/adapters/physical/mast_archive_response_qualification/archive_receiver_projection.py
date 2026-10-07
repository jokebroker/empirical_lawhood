"""Typed PF projection contracts over already-custodied MAST-U objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol, runtime_checkable

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding

from .query_source import MASTUArchiveAcquisitionUnitManifest, MASTUArchiveObjectReceipt, MASTUArchiveQueryManifest


class MastArchiveReceiverProjectionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True, slots=True)
class MastArchiveReceiverProjectionView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-receiver-projection-view'

    view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    representation_id: str
    passive_allocation_id: str
    causal_cutoff_id: str

    def __post_init__(self) -> None:
        for name in (
            "view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
            "representation_id",
            "passive_allocation_id",
            "causal_cutoff_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.receiver_ids, field_name="receiver_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.clock_ids, field_name="clock_ids", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MastArchiveReceiverProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-receiver-projection-config'

    config_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    query_manifest: ObjectIdentity
    projection_schema: str
    projection_task_prefix: str
    views: tuple[MastArchiveReceiverProjectionView, ...]
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(
            self.projection_task_prefix,
            field_name="projection_task_prefix",
        )
        if self.experiment_extension_set.object_schema != ResponseExperimentExtensionSet.SCHEMA:
            raise ValueError("PF projection config binds another extension set")
        if self.substrate_binding.object_schema != ResponseSubstrateBinding.SCHEMA:
            raise ValueError("PF projection config binds another substrate")
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("PF projection config binds another query manifest")
        validate_schema(self.projection_schema)
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        if not self.views:
            raise ValueError("PF projection config requires views")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("PF projection config cannot grant authority or reveal outcomes")

    def view(self, view_id: str) -> MastArchiveReceiverProjectionView:
        try:
            return next(value for value in self.views if value.view_id == view_id)
        except StopIteration as error:
            raise KeyError(view_id) from error


@dataclass(frozen=True, slots=True)
class MastArchiveReceiverScientificProjection(CanonicalRecord):
    """Adapter result preserving semantics but owning no law/property verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-receiver-scientific-projection'

    projection_id: str
    view: ObjectIdentity
    object_receipt: ObjectIdentity | None
    acquisition_unit_manifest: ObjectIdentity | None
    projected_artifact: ArtifactIdentity | None
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    disposition: MastArchiveReceiverProjectionDisposition
    reason_codes: tuple[str, ...]
    preserves_native_units_frames_directions: bool
    preserves_causal_cutoff: bool
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        if self.view.object_schema != MastArchiveReceiverProjectionView.SCHEMA:
            raise ValueError("PF projection binds another view")
        require_sorted_unique_strings(
            self.receiver_ids, field_name="receiver_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.clock_ids, field_name="clock_ids", allow_empty=False)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.disposition is MastArchiveReceiverProjectionDisposition.COMPLETE
        complete_lineage = (
            self.object_receipt is not None
            and self.object_receipt.object_schema == MASTUArchiveObjectReceipt.SCHEMA
            and self.acquisition_unit_manifest is not None
            and self.acquisition_unit_manifest.object_schema
            == MASTUArchiveAcquisitionUnitManifest.SCHEMA
        )
        if (
            complete != (self.projected_artifact is not None)
            or complete != complete_lineage
            or complete == bool(self.reason_codes)
        ):
            raise ValueError("PF projection disposition/product/reasons disagree")
        if (
            not self.preserves_native_units_frames_directions
            or not self.preserves_causal_cutoff
            or self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("PF projection loses native semantics or exceeds adapter authority")


@runtime_checkable
class MastArchiveReceiverProjectorPort(Protocol):
    """Pure projection over authoritative custody; never a UDA source port."""

    def project(
        self,
        *,
        receipt: MASTUArchiveObjectReceipt,
        view: MastArchiveReceiverProjectionView,
    ) -> ArtifactIdentity: ...


__all__ = [
    'MastArchiveReceiverProjectionConfig',
    'MastArchiveReceiverProjectionDisposition',
    'MastArchiveReceiverProjectionView',
    'MastArchiveReceiverProjectorPort',
    'MastArchiveReceiverScientificProjection',
]
