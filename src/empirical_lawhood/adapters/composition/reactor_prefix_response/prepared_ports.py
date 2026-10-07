# SPDX-License-Identifier: MPL-2.0

"""Shared prepared-campaign custody over the existing external artifact stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.canonical_record_archive import (
    ExternalCanonicalRecordArchive,
)
from empirical_lawhood.infrastructure.prepared_execution_events import (
    ExternalPreparedExecutionEventStore,
)
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureSlot
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedCommonStartBinding, PreparedDesignBindingReceipt, PreparedExecutionCensus, PreparedForecastParentCommitment, PreparedFutureCompletion, PreparedInstanceBindingReceipt, PreparedParentReturn, PreparedProbeCommitment
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt


@dataclass(frozen=True, slots=True)
class PreparedRootEventClock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/prepared-root-event-clock'

    run_id: str
    root: str
    stage: str
    occurred_at_utc: str

    def __post_init__(self) -> None:
        if self.stage not in ("freeze", "seal", "reveal"):
            raise ValueError(
                "prepared event clock root or stage outside frozen allocation"
            )
        parse_utc_timestamp(self.occurred_at_utc)


@dataclass(frozen=True, slots=True)
class PreparedDecisionEventClock(CanonicalRecord):
    """An additive actual-decision clock, distinct from the initial root freeze."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/composition/reactor-prefix-response/prepared-decision-event-clock'
    )
    run_id: str
    root: str
    decision_id: str
    occurred_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        parse_utc_timestamp(self.occurred_at_utc)


@dataclass(frozen=True)
class PreparedControlCustodyPorts:
    root: GuardedExternalRoot
    run_id: str
    authority: ObjectIdentity
    resources: ObjectIdentity
    issued_study: ObjectIdentity
    approval: ObjectIdentity
    reveal: ObjectIdentity
    assigned_roots: tuple[str, ...]
    direct_record_schemas: tuple[str, ...] = ()

    def event_clock(self, root: str, stage: str) -> str:
        """Save each first event instant once, then replay that exact identity."""
        if root not in self.assigned_roots:
            raise ValueError("prepared event clock root outside frozen allocation")
        if stage not in ("freeze", "seal", "reveal"):
            raise ValueError("prepared event clock stage differs")
        return self._persist_clock(root, stage, None)

    def decision_clock(self, root: str, decision_id: str) -> str:
        """Persist once at the actual callback; recovery reuses that instant."""
        if root not in self.assigned_roots:
            raise ValueError("prepared decision clock root outside frozen allocation")
        validate_stable_id(decision_id, field_name="decision_id")
        return self._persist_clock(root, "decision", decision_id)

    def _persist_clock(self, root: str, stage: str, decision_id: str | None) -> str:
        scope = f"runs/{self.run_id}/control/{root}"
        key = stage if decision_id is None else f"decision-{decision_id}"
        relative = f"{scope}/{key}-clock.json"
        path = self.root.resolve(
            relative, for_write=True, operation_minimum_free_bytes=100 * 1024**3
        )
        sidecar = self.root.resolve(
            f"{relative}.manifest.json",
            for_write=True,
            operation_minimum_free_bytes=100 * 1024**3,
        )
        plane = ExternalArtifactPlane(self.root)
        record: PreparedRootEventClock | PreparedDecisionEventClock
        if path.exists() or sidecar.exists():
            if not path.exists() or not sidecar.exists():
                raise ValueError(
                    "prepared event clock publication is incomplete; recover exact identity"
                )
            manifest = decode_artifact_manifest(sidecar.read_bytes())
            plane.verify_manifest(manifest)
            if decision_id is None:
                record = decode_canonical_bytes(
                    path.read_bytes(), PreparedRootEventClock, maximum_bytes=4096
                )
                matches = (record.root, record.run_id, record.stage) == (
                    root,
                    self.run_id,
                    stage,
                )
            else:
                record = decode_canonical_bytes(
                    path.read_bytes(), PreparedDecisionEventClock, maximum_bytes=4096
                )
                matches = (record.root, record.run_id, record.decision_id) == (
                    root,
                    self.run_id,
                    decision_id,
                )
            if not matches or manifest.logical.content_sha256 != record.fingerprint():
                raise ValueError("prepared event clock changed across recovery")
            return record.occurred_at_utc
        occurred = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        record = (
            PreparedRootEventClock(self.run_id, root, stage, occurred)
            if decision_id is None
            else PreparedDecisionEventClock(self.run_id, root, decision_id, occurred)
        )
        plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=f"{root}.{key}-clock",
                relative_path=relative,
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"{self.run_id}.{root}.control",
                publication_scope_relative_root=scope,
                payload=record.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                minimum_free_bytes=100 * 1024**3,
            )
        )
        return record.occurred_at_utc

    def freeze_clock(self, root: str) -> str:
        return self.event_clock(root, "freeze")

    def __post_init__(self) -> None:
        if (
            self.authority.object_schema
            != 'empirical-lawhood/planning/study-operation-authority'
            or self.resources.object_schema
            not in (
                'empirical-lawhood/runtime/execution-resource-envelope-spec',
                'empirical-lawhood/runtime/run-execution-resource-envelope',
            )
        ):
            raise ValueError(
                "prepared control custody lacks exact typed authority and resources"
            )

    def open_control_store(self, root: str) -> ExternalCanonicalRecordArchive:
        if root not in self.assigned_roots:
            raise ValueError("prepared control archive root outside frozen allocation")
        namespace = f"runs/{self.run_id}/control/{root}"
        return ExternalCanonicalRecordArchive(
            ExternalPreparedExecutionEventStore(
                ExternalArtifactPlane(self.root),
                state_root_relative_path=namespace,
                record_schemas=tuple(
                    sorted(
                        (
                            CanonicalRecordArchive.SCHEMA,
                            CompiledDeliveryControllerStudy.SCHEMA,
                            DeliveryControllerTickReceipt.SCHEMA,
                            *self.direct_record_schemas,
                        )
                    )
                ),
                maximum_record_bytes=64 * 1024**2,
                minimum_free_bytes=100 * 1024**3,
            ),
            root,
        )

    def open_prepared_store(self) -> ExternalPreparedExecutionEventStore:
        "Reconstruct the installed same-realization controller-use event owner."
        return ExternalPreparedExecutionEventStore(
            ExternalArtifactPlane(self.root),
            state_root_relative_path=f"runs/{self.run_id}/prepared-execution-events",
            record_schemas=tuple(
                sorted(
                    (
                        CompiledDeliveryControllerStudy.SCHEMA,
                        DeliveryControllerDecisionCommitment.SCHEMA,
                        DeliveryControllerTickReceipt.SCHEMA,
                        PreparedCommonStartBinding.SCHEMA,
                        PreparedDesignBindingReceipt.SCHEMA,
                        PreparedForecastParentCommitment.SCHEMA,
                        PreparedFutureCompletion.SCHEMA,
                        PreparedFutureSlot.SCHEMA,
                        PreparedExecutionCensus.SCHEMA,
                        PreparedInstanceBindingReceipt.SCHEMA,
                        PreparedParentReturn.SCHEMA,
                        PreparedProbeCommitment.SCHEMA,
                    )
                )
            ),
            maximum_record_bytes=64 * 1024**2,
            minimum_free_bytes=100 * 1024**3,
        )
