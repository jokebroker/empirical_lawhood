# SPDX-License-Identifier: MPL-2.0

"""Reconstruct reactor ports from published, bounded operator context records.

These are installed recipes over the existing external stores. A selector never
contains Python objects or callable names. Publication, operation authority and
the runner's execution closure remain separate from this no-effect resolution.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalStage
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase
from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import NoRegimeRetainedInputs, RegimeCContinuation, RegimeRetainedInput
from empirical_lawhood.adapters.methods.reactor_regime_response.prior import RegimePriorArtifact
from empirical_lawhood.adapters.simulators.reactor_prefix_response.reactor_binding import ReactorResolvedPort, required_upstream_port_keys
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.plans import ArtifactOutputSpec
from empirical_lawhood.runtime.providers import ExternalInputPayload

from .empirical_ports import EmpiricalControlCustodyPorts, EmpiricalPayloadCustodyPorts
from .prepared_ports import PreparedControlCustodyPorts
from .process_limits import LocalCampaignAllocation, LocalCampaignProcessLimits

_NAMESPACE = "operator/reactor-ports"
_MAX_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class ReactorPriorPublication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/reactor-prior-publication'
    manifest: ArtifactManifest
    receipt: CanonicalTaskReceipt
    expected_task_id: str


@dataclass(frozen=True, slots=True)
class ReactorPortContext(CanonicalRecord):
    """One exact installed port, referencing the operator's existing publications.

    Populate this in owner tooling after issue/authority, never in an input
    checker. All seven recipes use the same bounded representation. Fields are
    operands, not executable configuration; route/key select fixed code below.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/reactor-port-context'
    context_id: str
    route: str
    port_key: str
    native_config_sha256: str
    run_id: str
    parent: ArtifactLineageParent
    authority: ObjectIdentity
    resources: ObjectIdentity
    issued_study: ObjectIdentity
    outputs: tuple[ArtifactOutputSpec, ...]
    allocation: LocalCampaignAllocation | None = None
    approval: ObjectIdentity | None = None
    reveal: ObjectIdentity | None = None
    assigned_roots: tuple[str, ...] = ()
    direct_record_schemas: tuple[str, ...] = ()
    phase: FrontierPhase | None = None
    stage: ClassicalStage | None = None
    inputs: tuple[ArtifactManifest, ...] = ()
    prior: tuple[ReactorPriorPublication, ...] = ()
    continuation: RegimeCContinuation | None = None
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id)
        validate_stable_id(self.run_id)
        validate_sha256(self.native_config_sha256)
        if (
            self.port_key not in required_upstream_port_keys(self.route)
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
            or len(self.outputs) > 100_000
            or len(self.inputs) > 4096
            or len(self.prior) > 8
            or self.assigned_roots != tuple(sorted(set(self.assigned_roots)))
        ):
            raise ValueError("REACTOR_PORT_CONTEXT_INVALID")
        if self.allocation is not None and (
            self.allocation.execution_authority != self.authority
            or self.allocation.resource_envelope != self.resources
        ):
            raise ValueError("REACTOR_PORT_ALLOCATION_IDENTITY_MISMATCH")


@dataclass
class _PublishedSource:
    plane: ExternalArtifactPlane
    manifest: ArtifactManifest

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if not 0 < maximum_chunk_bytes <= 1024**2:
            raise ValueError("REACTOR_INPUT_CHUNK_BOUND_INVALID")
        self.plane.verify_manifest(self.manifest)
        path = self.plane.root.resolve(
            self.manifest.materialization.relative_path, for_write=False
        )
        remaining = self.manifest.materialization.size_bytes
        with path.open("rb") as stream:
            while remaining:
                block = stream.read(min(remaining, maximum_chunk_bytes))
                if not block:
                    raise ValueError("REACTOR_INPUT_TRUNCATED")
                remaining -= len(block)
                yield block
            if stream.read(1):
                raise ValueError("REACTOR_INPUT_GREW")

    def close(self) -> None:
        return


@dataclass(frozen=True)
class _RetainedInputs:
    plane: ExternalArtifactPlane
    continuation: RegimeCContinuation

    def create_source(self, declaration: RegimeRetainedInput) -> _PublishedSource:
        if declaration not in self.continuation.inputs:
            raise ValueError("REACTOR_RETAINED_INPUT_UNDECLARED")
        self.plane.verify_manifest(declaration.manifest)
        return _PublishedSource(self.plane, declaration.manifest)


class ReactorExternalPortStore:
    """Only manifested records under the separately selected operator root resolve."""

    def __init__(self, plane: ExternalArtifactPlane) -> None:
        self.plane = plane

    def _load(self, identity: ObjectIdentity) -> ReactorPortContext:
        if identity.object_schema != ReactorPortContext.SCHEMA:
            raise ValueError("REACTOR_PORT_CONTEXT_SCHEMA_MISMATCH")
        relative = f"{_NAMESPACE}/{identity.object_id}.json"
        path = self.plane.root.resolve(relative, for_write=False)
        sidecar = self.plane.root.resolve(relative + ".manifest.json", for_write=False)
        if not path.is_file() or not sidecar.is_file():
            raise KeyError(identity.object_id)
        manifest = decode_artifact_manifest(
            read_bounded_bytes(sidecar, maximum_bytes=_MAX_BYTES)
        )
        if (
            manifest.publication is None
            or manifest.materialization.relative_path != relative
            or manifest.materialization.size_bytes > _MAX_BYTES
            or manifest.logical.logical_artifact_id != identity.object_id
            or manifest.logical.payload_schema != identity.object_schema
            or manifest.logical.content_sha256 != identity.object_fingerprint
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or manifest.logical.visibility_ceiling
            is not VisibilityCeiling.DEVELOPMENT_ONLY
        ):
            raise ValueError("REACTOR_PORT_PUBLICATION_MISMATCH")
        self.plane.verify_manifest(manifest)
        record = decode_canonical_bytes(
            read_bounded_bytes(path, maximum_bytes=_MAX_BYTES),
            ReactorPortContext,
            maximum_bytes=_MAX_BYTES,
        )
        if ObjectIdentity.from_record(record.context_id, record) != identity:
            raise ValueError("REACTOR_PORT_CONTEXT_IDENTITY_MISMATCH")
        authority = ExternalStudyOperationAuthorityStore(self.plane).load(
            record.authority.object_id
        )
        if (
            ObjectIdentity.from_record(authority.authority_id, authority)
            != record.authority
            or not authority.allows_execution
            or authority.subject != record.issued_study
            or authority.prerequisite_authority != record.approval
        ):
            raise ValueError("REACTOR_PORT_EXECUTION_AUTHORITY_MISMATCH")
        if record.reveal is not None:
            reveal = ExternalStudyOperationAuthorityStore(self.plane).load(
                record.reveal.object_id
            )
            if (
                ObjectIdentity.from_record(reveal.authority_id, reveal) != record.reveal
                or not reveal.allows_reveal
                or reveal.subject != record.issued_study
                or reveal.prerequisite_authority != record.authority
            ):
                raise ValueError("REACTOR_PORT_REVEAL_AUTHORITY_MISMATCH")
        # The runtime separately replays issue/source/resource closure and expiry
        # before effects. This read-only operation cannot renew an expired grant.
        return record

    def resolve_port(self, identity: ObjectIdentity) -> ReactorResolvedPort:
        record = self._load(identity)
        return ReactorResolvedPort(identity, self._compose(record))

    def validate_binding(
        self,
        *,
        route: str,
        native_config_sha256: str,
        key: str,
        identity: ObjectIdentity,
    ) -> ObjectIdentity | None:
        record = self._load(identity)
        if (record.route, record.native_config_sha256, record.port_key) != (
            route,
            native_config_sha256,
            key,
        ):
            raise ValueError("REACTOR_PORT_BINDING_CONTEXT_MISMATCH")
        # A context identity includes the exact nested record. The factory sees
        # that record, while the selector identifies its published port recipe.
        nested = (
            record.phase
            if key == "frontier-phase"
            else record.stage
            if key == "classical-stage"
            else None
        )
        return (
            None
            if nested is None
            else ObjectIdentity.from_record(identity.object_id, nested)
        )

    def _payload(self, manifest: ArtifactManifest) -> ExternalInputPayload:
        if (
            manifest.publication is None
            or manifest.materialization.size_bytes > _MAX_BYTES
        ):
            raise ValueError("REACTOR_INPUT_PUBLICATION_REQUIRED")
        self.plane.verify_manifest(manifest)
        logical, material = manifest.logical, manifest.materialization
        return ExternalInputPayload(
            logical.logical_artifact_id,
            logical.payload_schema,
            logical.profile,
            logical.media_type,
            _PublishedSource(self.plane, manifest),
            material.size_bytes,
            material.physical_sha256,
            max(1, material.size_bytes),
            1024**2,
            logical.visibility_ceiling,
            logical.outcome_access,
            logical.parent_visibility_ceilings,
            logical.lineage_parents,
            logical.content_sha256,
        )

    def _compose(self, record: ReactorPortContext) -> object:
        key, root = record.port_key, self.plane.root
        if key == "frontier-phase":
            if record.phase is None or record.route != "finite-control-frontier":
                raise ValueError("REACTOR_FRONTIER_PHASE_REQUIRED")
            return record.phase
        if key == "classical-stage":
            if record.stage is None or record.route != "staged-pulse-response":
                raise ValueError("REACTOR_CLASSICAL_STAGE_REQUIRED")
            return record.stage
        if key.endswith(("-resource-guard", "-limits")):
            if record.allocation is None:
                raise ValueError("REACTOR_ISSUED_ALLOCATION_REQUIRED")
            return LocalCampaignProcessLimits(
                record.allocation, root, f"runs/{record.run_id}/task-cost"
            )
        if (
            key.endswith(("-dependency-reader", "-custody", "-reader"))
            and "control" not in key
        ):
            return EmpiricalPayloadCustodyPorts(
                root,
                record.run_id,
                record.parent,
                record.outputs,
                root.contract.minimum_free_bytes,
            )
        if key.endswith(("-control-custody", "-control")):
            if not record.assigned_roots:
                raise ValueError("REACTOR_CONTROL_ASSIGNED_ROOTS_REQUIRED")
            if record.route == "causal-response-study":
                return EmpiricalControlCustodyPorts(
                    root,
                    record.run_id,
                    record.authority,
                    record.resources,
                    root.contract.minimum_free_bytes,
                )
            if record.approval is None or record.reveal is None:
                raise ValueError("REACTOR_SEPARATE_CONTROL_APPROVAL_REVEAL_REQUIRED")
            return PreparedControlCustodyPorts(
                root,
                record.run_id,
                record.authority,
                record.resources,
                record.issued_study,
                record.approval,
                record.reveal,
                record.assigned_roots,
                record.direct_record_schemas,
            )
        if key in ("frontier-native-inputs", "classical-native-inputs"):
            if not record.inputs:
                raise ValueError("REACTOR_PHASE_INPUTS_REQUIRED")
            return tuple(self._payload(manifest) for manifest in record.inputs)
        if key == "reactor-regime-native-prior-artifacts":
            if record.prior and record.reveal is None:
                raise ValueError("REACTOR_PRIOR_REVEAL_REQUIRED")
            values = []
            for value in record.prior:
                payload = self._payload(value.manifest)
                raw = b"".join(payload.chunks())
                values.append(
                    RegimePriorArtifact(
                        raw, value.manifest, value.receipt, value.expected_task_id
                    )
                )
            return tuple(values)
        if key == "reactor-regime-native-retained-inputs":
            if record.continuation is None:
                return NoRegimeRetainedInputs()
            for value in record.continuation.inputs:
                self.plane.verify_manifest(value.manifest)
            return _RetainedInputs(self.plane, record.continuation)
        raise ValueError("REACTOR_PORT_RECIPE_UNSUPPORTED")


__all__ = [
    "ReactorExternalPortStore",
    'ReactorPortContext',
    'ReactorPriorPublication',
]
