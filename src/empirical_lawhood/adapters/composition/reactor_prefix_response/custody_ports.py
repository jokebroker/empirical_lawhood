"""Spawn-safe reactor custody ports over the target external stores.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import dataclass

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.candidate_payloads import ExternalCandidatePayloadPlane
from empirical_lawhood.infrastructure.dependency_manifests import DependencyManifestReader
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    CanonicalTaskReceipt,
    ArtifactSemanticValidationRegistry,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from empirical_lawhood.runtime.execution import TaskContext, WorkerInputBinding
from empirical_lawhood.runtime.plans import ArtifactOutputSpec


@dataclass(frozen=True)
class ReactorCustodyPorts:
    """Carry only typed configuration through spawn; construct stores in the worker.

    The reactor uses canonical JSON and the default closed validators. No
    configured validator registry or published evidence is serialized as code.
    """

    root: GuardedExternalRoot
    run_id: str
    parent: ArtifactLineageParent
    outputs: tuple[ArtifactOutputSpec, ...]
    minimum_free_bytes: int
    semantic_validations: ArtifactSemanticValidationRegistry | None = None
    model_outcome_access: OutcomeAccess = OutcomeAccess.EVALUATOR_REVEAL

    def _models(self) -> ExternalCandidatePayloadPlane:
        return ExternalCandidatePayloadPlane(
            ExternalArtifactPlane(self.root),
            f"runs/{self.run_id}/candidate-models",
            f"{self.run_id}.candidate-models",
            VisibilityCeiling.PROSPECTIVE,
            self.model_outcome_access,
            (self.parent,),
            self.minimum_free_bytes,
        )

    def publish_candidate_payload(
        self,
        *,
        payload: bytes,
        evaluator: ExecutableReference,
        implementation: ObjectIdentity,
        decoder_schema: str,
        decoder_version: str,
        maximum_decode_bytes: int,
    ) -> CandidatePayloadPublicationReceipt:
        return self._models().publish_candidate_payload(
            payload=payload,
            evaluator=evaluator,
            implementation=implementation,
            decoder_schema=decoder_schema,
            decoder_version=decoder_version,
            maximum_decode_bytes=maximum_decode_bytes,
        )

    def read_candidate_payload(self, receipt: CandidatePayloadPublicationReceipt) -> bytes:
        return self._models().read_candidate_payload(receipt)

    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]:
        return DependencyManifestReader(
            ExternalArtifactPlane(self.root, semantic_validations=self.semantic_validations),
            self.outputs,
        ).read_dependency(context, binding)
