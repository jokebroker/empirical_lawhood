# SPDX-License-Identifier: MPL-2.0

"""Finite response law ports over authenticated input inventories and target stores."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_response_law.runtime_ports import FiniteResponseLawControlRuntimeBinding, FiniteResponseLawPreparedStorePort
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.candidate_payloads import (
    ExternalCandidatePayloadPlane,
)
from empirical_lawhood.infrastructure.candidate_sources import (
    ExternalContentAddressedInputResolver,
)
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort


@dataclass(frozen=True)
class FiniteResponseLawDevelopmentCandidatePort:
    """Spawn-safe publication with the development ceiling of its parent."""

    root: GuardedExternalRoot
    scope_id: str
    parent: ArtifactLineageParent

    def _plane(self) -> ExternalCandidatePayloadPlane:
        return ExternalCandidatePayloadPlane(
            ExternalArtifactPlane(self.root),
            f"development/finite-response-law/{self.scope_id}/candidate-models",
            f"{self.scope_id}.candidate-models",
            VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            (self.parent,),
            self.root.contract.minimum_free_bytes,
        )

    def publish_candidate_payload(self, **kwargs):
        return self._plane().publish_candidate_payload(**kwargs)

    def read_candidate_payload(self, receipt):
        return self._plane().read_candidate_payload(receipt)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRuntimeContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-runtime-context'
    context_id: str
    run_id: str
    source_sha256: str
    issued_study: ObjectIdentity
    prerequisite_authority: ObjectIdentity
    authority: ObjectIdentity
    grantee_id: str
    compiler_release_id: str
    reveal_authority_id: str | None = None

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        validate_sha256(self.source_sha256)
        for value in (
            self.context_id,
            self.run_id,
            self.grantee_id,
            self.compiler_release_id,
        ):
            validate_stable_id(value)


@dataclass
class _Source:
    path: Path
    artifact: ArtifactIdentity
    consumed: bool = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if self.consumed or not 0 < maximum_chunk_bytes <= 1024**2 or any(
            p.is_symlink() for p in (self.path, *self.path.parents)
        ):
            raise ValueError("FINITE_RESPONSE_LAW_RETAINED_SOURCE_BOUND_OR_PATH")
        self.consumed = True
        raw = read_bounded_bytes(self.path, maximum_bytes=self.artifact.size_bytes)
        if (
            len(raw) != self.artifact.size_bytes
            or sha256(raw).hexdigest() != self.artifact.sha256
        ):
            raise ValueError("FINITE_RESPONSE_LAW_RETAINED_SOURCE_CHANGED")
        for start in range(0, len(raw), maximum_chunk_bytes):
            yield raw[start : start + maximum_chunk_bytes]

    def close(self) -> None:
        self.consumed = True


@dataclass(frozen=True)
class FiniteResponseLawAuthenticatedSources:
    artifacts: tuple[ArtifactIdentity, ...]
    paths: tuple[Path, ...]

    def create_source(self, artifact: ArtifactIdentity) -> _Source:
        if artifact not in self.artifacts:
            raise ValueError("FINITE_RESPONSE_LAW_RETAINED_SOURCE_NOT_DECLARED")
        return _Source(self.paths[self.artifacts.index(artifact)], artifact)


def retained_sources(
    artifacts: tuple[ArtifactIdentity, ...], manifests: tuple[Path, ...]
) -> FiniteResponseLawAuthenticatedSources:
    """The caller must first replay every complete inventory and all three grants."""
    roster = [
        entry
        for path in manifests
        for entry in json.loads(read_bounded_bytes(path, maximum_bytes=128 * 1024**2))
    ]
    paths = []
    for artifact in artifacts:
        candidates = []
        for entry in roster:
            if (entry["sha256"], entry["bytes"]) != (
                artifact.sha256,
                artifact.size_bytes,
            ):
                continue
            path = Path(entry["path"])
            sidecar = Path(str(path) + ".manifest.json")
            if not sidecar.is_file() or sidecar.is_symlink():
                continue
            manifest = decode_artifact_manifest(
                read_bounded_bytes(sidecar, maximum_bytes=8 * 1024**2)
            )
            if (
                manifest.logical.logical_artifact_id,
                manifest.logical.payload_schema,
                manifest.logical.content_sha256,
                manifest.logical.media_type,
            ) == (
                artifact.artifact_id,
                artifact.payload_schema,
                artifact.sha256,
                artifact.media_type,
            ):
                candidates.append(path)
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise ValueError("FINITE_RESPONSE_LAW_EXACT_RETAINED_MATERIALIZATION_REQUIRED")
        paths.append(candidates[0])
    return FiniteResponseLawAuthenticatedSources(artifacts, tuple(paths))


def load_control_runtime(
    plane: ExternalArtifactPlane, identity: ObjectIdentity, source_sha256: str
) -> FiniteResponseLawControlRuntimeBinding:
    if identity.object_schema != FiniteResponseLawRuntimeContext.SCHEMA:
        raise ValueError("FINITE_RESPONSE_LAW_RUNTIME_CONTEXT_SCHEMA_MISMATCH")
    relative = f"operator/finite-response-law-contexts/{identity.object_id}.json"
    path = plane.root.resolve(relative, for_write=False)
    sidecar = plane.root.resolve(relative + ".manifest.json", for_write=False)
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("FINITE_RESPONSE_LAW_PUBLISHED_RUNTIME_CONTEXT_REQUIRED")
    manifest = decode_artifact_manifest(
        read_bounded_bytes(sidecar, maximum_bytes=1024**2)
    )
    if (
        manifest.publication is None
        or manifest.materialization.relative_path != relative
        or manifest.logical.content_sha256 != identity.object_fingerprint
        or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
    ):
        raise ValueError("FINITE_RESPONSE_LAW_RUNTIME_PUBLICATION_MISMATCH")
    plane.verify_manifest(manifest)
    config = decode_canonical_bytes(
        read_bounded_bytes(path, maximum_bytes=1024**2),
        FiniteResponseLawRuntimeContext,
        maximum_bytes=1024**2,
    )
    if (
        ObjectIdentity.from_record(config.context_id, config) != identity
        or config.source_sha256 != source_sha256
    ):
        raise ValueError("FINITE_RESPONSE_LAW_RUNTIME_SOURCE_MISMATCH")
    runtime = FiniteResponseLawControlRuntimeBinding(
        plane.root,
        config.run_id,
        FiniteResponseLawPreparedStorePort(
            plane.root,
            f"runs/{config.run_id}/prepared-execution-events",
            plane.root.contract.minimum_free_bytes,
        ),
        config.issued_study,
        config.prerequisite_authority,
        config.authority,
        config.grantee_id,
        config.compiler_release_id,
        config.reveal_authority_id,
    )
    execution = runtime.execution_authority()
    if (
        not execution.allows_execution
        or execution.subject != config.issued_study
        or execution.prerequisite_authority != config.prerequisite_authority
    ):
        raise ValueError("FINITE_RESPONSE_LAW_RUNTIME_EXECUTION_AUTHORITY_MISMATCH")
    return runtime


def consumer_ports(
    *, packet, bundle, plane: ExternalArtifactPlane, manifests: tuple[Path, ...]
) -> tuple[ExecutablePlatformPort, ...]:
    if packet.stage == "preparation-screening":
        raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
    from empirical_lawhood.adapters.methods.finite_response_law.control_ports import CONTROL_RUNTIME_PORT, REVEAL_CONTROL_PORT, SEALED_CONTROL_PORT, SOURCE_CONTROL_PORT
    from empirical_lawhood.adapters.methods.finite_response_law.method_provider import CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT
    from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_EVIDENCE_PORT
    from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation.executable_binding import IMPORT_RESOLVER_PORT
    from empirical_lawhood.adapters.simulators.finite_response_law.provider import RETAINED_SOURCE_PORT
    from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_RETAINED_SOURCE_PORT

    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(bundle.authoring.package_id, bundle.authoring),
        VisibilityCeiling.DEVELOPMENT_ONLY,
        OutcomeAccess.OUTCOME_BLIND,
    )
    custody = FiniteResponseLawDevelopmentCandidatePort(
        plane.root, f"{packet.stage}.{packet.fingerprint()[:32]}", parent
    )
    resolver = ExternalContentAddressedInputResolver(plane.root)
    ports = {
        CANDIDATE_PAYLOAD_PORT: custody,
        INPUT_RESOLVER_PORT: resolver,
        IMPORT_RESOLVER_PORT: resolver,
    }
    if packet.stage == "supplemental-development":
        artifacts = tuple(
            sorted(
                (
                    a
                    for row in packet.source.retained_predecessors
                    for a in (*row.artifacts, row.task_receipt)
                ),
                key=lambda a: a.artifact_id,
            )
        )
        ports[RETAINED_SOURCE_PORT] = retained_sources(artifacts, manifests)
    if packet.stage == "preparation-screening":
        artifacts = tuple(
            sorted(
                (
                    a
                    for row in packet.source.retained_prefixes
                    for a in (*row.declaration.artifacts, row.declaration.task_receipt)
                ),
                key=lambda a: a.artifact_id,
            )
        )
        ports[PREPARATION_POLICY_RETAINED_SOURCE_PORT] = retained_sources(artifacts, manifests)
        screen = packet.screen
        evidence = tuple(
            sorted(
                (
                    screen.lower_artifact,
                    screen.lower_qualification,
                    screen.prospective_adjudication,
                    screen.prospective_closeout,
                ),
                key=lambda a: a.artifact_id,
            )
        )
        ports[PREPARATION_POLICY_EVIDENCE_PORT] = retained_sources(evidence, manifests)
    if packet.control is not None and packet.completion is None:
        if packet.runtime_context is None:
            raise ValueError("FINITE_RESPONSE_LAW_CONTROL_RUNTIME_CONTEXT_REQUIRED")
        runtime = load_control_runtime(
            plane, packet.runtime_context, packet.source.fingerprint()
        )
        ports.update(
            {
                key: runtime
                for key in (
                    CONTROL_RUNTIME_PORT,
                    SOURCE_CONTROL_PORT,
                    SEALED_CONTROL_PORT,
                    REVEAL_CONTROL_PORT,
                )
            }
        )
    return tuple(ExecutablePlatformPort(key, ports[key]) for key in sorted(ports))


__all__ = [
    'FiniteResponseLawAuthenticatedSources',
    'FiniteResponseLawRuntimeContext',
    "consumer_ports",
    "load_control_runtime",
    "retained_sources",
]
