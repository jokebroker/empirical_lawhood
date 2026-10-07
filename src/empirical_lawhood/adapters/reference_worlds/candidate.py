"""Static candidate composition for the non-promotable reference world."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import (
    CandidateCapabilityConfigDecoder,
    SourceReadMode,
)


REFERENCE_SOURCE_ID = "source.reference-medium"
REFERENCE_SOURCE_PAYLOAD_SCHEMA = 'empirical-lawhood/reference-worlds/held-source'
REFERENCE_HELD_SOURCE_CAPABILITY_KEY = "reference.held-source"
REFERENCE_HELD_SOURCE_CAPABILITY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ReferenceHeldSourceCapabilityConfig(CanonicalRecord):
    """Static adapter behavior; materialization identity remains draft-selected."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-held-source-capability-config'

    config_id: str
    adapter_key: str
    adapter_version: str
    read_mode: SourceReadMode
    maximum_chunk_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.adapter_key, field_name="adapter_key")
        validate_semantic_version(self.adapter_version)
        if (
            not isinstance(self.maximum_chunk_bytes, int)
            or isinstance(self.maximum_chunk_bytes, bool)
            or self.maximum_chunk_bytes <= 0
            or self.maximum_chunk_bytes > 1024 * 1024
        ):
            raise ValueError("reference source chunks must be in (0, 1 MiB]")


REFERENCE_HELD_SOURCE_CONFIG = ReferenceHeldSourceCapabilityConfig(
    config_id="config.reference.held-source",
    adapter_key=REFERENCE_HELD_SOURCE_CAPABILITY_KEY,
    adapter_version=REFERENCE_HELD_SOURCE_CAPABILITY_VERSION,
    read_mode=SourceReadMode.CHUNKED_HIGH_VOLUME,
    maximum_chunk_bytes=1024 * 1024,
)


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def reference_held_source_manifest(
    *,
    implementation_sha256: str,
    reference_prepare: CapabilityManifest,
) -> CapabilityManifest:
    """Register a source implementation without modifying shared compiler logic."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if reference_prepare.capability_key != "reference.prepare":
        raise ValueError("reference source replacement requires reference.prepare")
    return replace(
        reference_prepare,
        capability_key=REFERENCE_HELD_SOURCE_CAPABILITY_KEY,
        capability_version=REFERENCE_HELD_SOURCE_CAPABILITY_VERSION,
        kind=CapabilityKind.SOURCE,
        config_schema=ReferenceHeldSourceCapabilityConfig.SCHEMA,
        config_schema_sha256=_schema_sha256(ReferenceHeldSourceCapabilityConfig.SCHEMA),
        input_schema_ids=(REFERENCE_SOURCE_PAYLOAD_SCHEMA,),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            "reference-source-bounded-chunks",
            "reference-source-content-addressed",
            "reference-source-no-acquisition-or-authority",
        ),
        implementation_sha256=implementation_sha256,
    )


def _source_config_ref() -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=REFERENCE_HELD_SOURCE_CONFIG.config_id,
        config_schema=ReferenceHeldSourceCapabilityConfig.SCHEMA,
        config_schema_sha256=_schema_sha256(ReferenceHeldSourceCapabilityConfig.SCHEMA),
        content_sha256=REFERENCE_HELD_SOURCE_CONFIG.fingerprint(),
        artifact_id="config-artifact.reference.held-source",
    )


def _candidate_protocol(
    protocol: ProtocolTemplate,
    source_manifest: CapabilityManifest,
) -> ProtocolTemplate:
    prepare = next(
        (value for value in protocol.steps if value.step_id == "prepare"),
        None,
    )
    if prepare is None or prepare.capability_key != "reference.prepare":
        raise ValueError("reference protocol lacks its replaceable prepare step")
    source_step = replace(
        prepare,
        capability_key=source_manifest.capability_key,
        capability_version=source_manifest.capability_version,
        config=_source_config_ref(),
    )
    steps = tuple(
        sorted(
            (
                source_step if value.step_id == source_step.step_id else value
                for value in protocol.steps
            ),
            key=lambda value: value.step_id,
        )
    )
    return replace(
        protocol,
        template_id="protocol.reference-held-source-prospective",
        steps=steps,
    )


def _candidate_graph(
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    manifests = {value.registry_id: value for value in registry.capabilities}
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=manifests[
                f"{step.capability_key}@{step.capability_version}"
            ].implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    output_by_step = {step.step_id: step.outputs[0] for step in protocol.steps}
    edges: list[CandidateGraphEdge] = [
        CandidateGraphEdge(
            edge_id="edge.source.reference.prepare",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=REFERENCE_SOURCE_ID,
            consumer_node_id="prepare",
            consumer_input_id="source",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id="artifact.reference-source-placeholder",
            payload_schema=REFERENCE_SOURCE_PAYLOAD_SCHEMA,
            media_type="application/octet-stream",
            maximum_size_bytes=64 * 1024**3,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        )
    ]
    for step in protocol.steps:
        for parent_id in step.dependency_step_ids:
            output = output_by_step[parent_id]
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.{parent_id}.{step.step_id}",
                    producer_node_id=parent_id,
                    producer_output_id=output.output_id,
                    external_input_id=None,
                    consumer_node_id=step.step_id,
                    consumer_input_id=f"input-{parent_id}",
                    scientific_role=ScientificInputRole.OUTCOME,
                    logical_artifact_id=f"artifact.{parent_id}.{output.output_id}",
                    payload_schema=output.payload_schema,
                    media_type=output.media_type,
                    maximum_size_bytes=10_000_000,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=step.visibility_ceiling,
                    barrier=step.barrier,
                )
            )
    return CandidateScientificGraph(
        graph_id="graph.reference-held-source-prospective",
        external_inputs=(
            CandidateGraphExternalInput(
                input_id=REFERENCE_SOURCE_ID,
                scientific_role=ScientificInputRole.PREPARED_MEDIUM,
                logical_artifact_id="artifact.reference-source-placeholder",
                content_identity_policy=ContentIdentityPolicy.RECEIPT_BOUND,
                expected_content_sha256=None,
                payload_schema=REFERENCE_SOURCE_PAYLOAD_SCHEMA,
                media_type="application/octet-stream",
                maximum_size_bytes=64 * 1024**3,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            ),
        ),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _coverage(
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation_id,
            proof_owner_node_id=owners.get(
                obligation_id,
                ("report", "report"),
            )[0],
            required_output_id=owners.get(
                obligation_id,
                ("report", "report"),
            )[1],
            contributor_edge_ids=incoming[owners.get(obligation_id, ("report", "report"))[0]],
        )
        for obligation_id in required_candidate_obligation_ids(experiment, protocol)
    )
    return ObligationCoverage(
        coverage_id="coverage.reference-held-source-prospective",
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def reference_candidate_template(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    experiment: ExperimentSpec,
    source_manifest: CapabilityManifest,
) -> StudyTemplate:
    candidate_protocol = _candidate_protocol(protocol, source_manifest)
    manifests = tuple(
        sorted(
            (
                source_manifest,
                *(
                    value
                    for value in registry.capabilities
                    if value.capability_key != "reference.prepare"
                ),
            ),
            key=lambda value: value.registry_id,
        )
    )
    candidate_registry = CapabilityRegistry(
        registry_id="reference-held-source-candidate-registry",
        capabilities=manifests,
    )
    graph = _candidate_graph(candidate_protocol, candidate_registry)
    return StudyTemplate(
        template_key="reference.held-source-prospective",
        template_version="1.0.0",
        protocol=candidate_protocol,
        graph=graph,
        coverage=_coverage(experiment, candidate_protocol, graph),
    )


def reference_candidate_catalog(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    experiment: ExperimentSpec,
    source_implementation_sha256: str,
) -> CandidateCapabilityCatalog:
    prepare = registry.resolve("reference.prepare", "1.0.0")
    source_manifest = reference_held_source_manifest(
        implementation_sha256=source_implementation_sha256,
        reference_prepare=prepare,
    )
    registrations = [
        CandidateCapabilityRegistration(
            manifest=value,
            provider_key=value.capability_key,
            provider_version=value.capability_version,
            config_media_type="text/plain",
            maximum_config_bytes=4096,
        )
        for value in registry.capabilities
    ]
    registrations.append(
        CandidateCapabilityRegistration(
            manifest=source_manifest,
            provider_key=REFERENCE_HELD_SOURCE_CAPABILITY_KEY,
            provider_version=REFERENCE_HELD_SOURCE_CAPABILITY_VERSION,
            config_media_type="application/json",
            maximum_config_bytes=4096,
        )
    )
    return CandidateCapabilityCatalog(
        catalog_id="catalog.reference-candidate-capabilities",
        registrations=tuple(sorted(registrations, key=lambda value: value.registration_id)),
        templates=(
            reference_candidate_template(
                protocol=protocol,
                registry=registry,
                experiment=experiment,
                source_manifest=source_manifest,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ReferenceHeldSourceAdapter:
    """Strict decoder only; acquisition, credentials and authority are absent."""

    manifest: CapabilityManifest

    def __post_init__(self) -> None:
        if (
            self.manifest.capability_key != REFERENCE_HELD_SOURCE_CAPABILITY_KEY
            or self.manifest.kind is not CapabilityKind.SOURCE
        ):
            raise ValueError("reference held-source adapter binds another capability")

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    @staticmethod
    def decode_config(payload: bytes) -> ReferenceHeldSourceCapabilityConfig:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        return decode_canonical_bytes(
            payload,
            ReferenceHeldSourceCapabilityConfig,
            maximum_bytes=4096,
        )

    def validate_config(
        self,
        payload: bytes,
        *,
        expected_schema: str,
    ) -> None:
        if expected_schema != self.manifest.config_schema:
            raise ValueError("held-source config schema differs from its manifest")
        value = self.decode_config(payload)
        if value.adapter_key != self.provider_key or value.adapter_version != self.provider_version:
            raise ValueError("held-source config selects another static adapter")


@dataclass(frozen=True, slots=True)
class ReferenceTextCapabilityAdapter:
    """Strict decoder for the historical reference world's bounded text configs."""

    manifest: CapabilityManifest

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(
        self,
        payload: bytes,
        *,
        expected_schema: str,
    ) -> None:
        if expected_schema != self.manifest.config_schema:
            raise ValueError("reference config schema differs from its manifest")
        if len(payload) > 4096:
            raise ValueError("reference text config exceeds its byte limit")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("reference text config is not UTF-8") from error
        if text != f"config payload for {self.provider_key}":
            raise ValueError("reference text config does not select its exact provider")


def reference_candidate_config_decoders(
    catalog: CandidateCapabilityCatalog,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    """Bind one adapter-local decoder per static capability registration."""

    values: list[CandidateCapabilityConfigDecoder] = []
    for registration in catalog.registrations:
        if registration.manifest.capability_key == REFERENCE_HELD_SOURCE_CAPABILITY_KEY:
            decoder: CandidateCapabilityConfigDecoder = ReferenceHeldSourceAdapter(
                registration.manifest
            )
        else:
            decoder = ReferenceTextCapabilityAdapter(registration.manifest)
        values.append(decoder)
    return tuple(sorted(values, key=lambda value: f"{value.provider_key}@{value.provider_version}"))


__all__ = [
    "REFERENCE_HELD_SOURCE_CAPABILITY_KEY",
    "REFERENCE_HELD_SOURCE_CAPABILITY_VERSION",
    "REFERENCE_HELD_SOURCE_CONFIG",
    "REFERENCE_SOURCE_ID",
    "REFERENCE_SOURCE_PAYLOAD_SCHEMA",
    "ReferenceHeldSourceAdapter",
    "ReferenceHeldSourceCapabilityConfig",
    "ReferenceTextCapabilityAdapter",
    "reference_candidate_config_decoders",
    "reference_candidate_catalog",
    "reference_candidate_template",
    "reference_held_source_manifest",
]
