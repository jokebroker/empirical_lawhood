"""Static runtime-provider ports for capability implementations and inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactSemanticValidation,
    ArtifactSemanticValidationRegistration,
    ArtifactSemanticValidationRegistry,
    lineage_parent_sort_key,
)
from .adjudication import ScientificAdjudicationOutputContract
from .capabilities import CapabilityManifest, CapabilityRegistry
from .execution import TaskRunner
from .plans import ProtocolExecutionPlan


MAX_IN_MEMORY_EXTERNAL_INPUT_BYTES: Final = 8 * 1024 * 1024
MAX_EXTERNAL_INPUT_CHUNK_BYTES: Final = 1024 * 1024


class ExternalInputSource(Protocol):
    """One closeable, sequential provider input source."""

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _BytesExternalInputSource:
    payload: bytes = field(repr=False)
    _consumed: bool = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if maximum_chunk_bytes <= 0:
            raise ValueError("external input chunk bound must be positive")
        if self._consumed:
            raise ValueError("external input source was already consumed")
        self._consumed = True
        for offset in range(0, len(self.payload), maximum_chunk_bytes):
            yield self.payload[offset : offset + maximum_chunk_bytes]

    def close(self) -> None:
        return


@dataclass(frozen=True, slots=True)
class CapabilityOutputSemanticContract:
    """Static profile/schema semantics bound to one capability implementation."""

    capability_key: str
    capability_version: str
    capability_implementation_sha256: str
    payload_schema: str
    profile: ArtifactProfile
    top_level_keys: tuple[str, ...] = ()
    value_keys: tuple[str, ...] = ()
    record_version: str | None = None
    capability_key_field: str | None = None
    task_id_field: str | None = None
    output_id_field: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.capability_implementation_sha256,
            field_name="capability_implementation_sha256",
        )
        validate_schema(self.payload_schema)
        require_sorted_unique_strings(
            self.top_level_keys,
            field_name="semantic top-level keys",
            allow_empty=self.profile is not ArtifactProfile.CANONICAL_JSON,
        )
        require_sorted_unique_strings(
            self.value_keys,
            field_name="semantic value keys",
        )
        if self.record_version is not None:
            validate_semantic_version(self.record_version)
            if (
                self.profile is not ArtifactProfile.CANONICAL_JSON
                or not self.value_keys
                or "version" not in self.top_level_keys
            ):
                raise ValueError(
                    "record version requires a versioned canonical JSON document shape"
                )
        for name, value in (
            ("capability_key_field", self.capability_key_field),
            ("task_id_field", self.task_id_field),
            ("output_id_field", self.output_id_field),
        ):
            if value is not None:
                validate_nonempty(value, field_name=name)
                if value not in self.top_level_keys:
                    raise ValueError(f"{name} is absent from the semantic document shape")
        if self.profile is not ArtifactProfile.CANONICAL_JSON and (
            self.top_level_keys
            or self.value_keys
            or self.capability_key_field is not None
            or self.task_id_field is not None
            or self.output_id_field is not None
        ):
            raise ValueError(
                "non-JSON capability semantics are defined by registered profile/schema"
            )

    @classmethod
    def from_manifest(
        cls,
        manifest: CapabilityManifest,
        *,
        payload_schema: str,
        profile: ArtifactProfile,
        top_level_keys: tuple[str, ...] = (),
        value_keys: tuple[str, ...] = (),
        record_version: str | None = None,
        capability_key_field: str | None = None,
        task_id_field: str | None = None,
        output_id_field: str | None = None,
    ) -> CapabilityOutputSemanticContract:
        """Bind output semantics to the exact registered implementation."""

        if payload_schema not in manifest.output_schema_ids:
            raise ValueError("semantic payload schema is absent from capability outputs")
        return cls(
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            capability_implementation_sha256=manifest.implementation_sha256,
            payload_schema=payload_schema,
            profile=profile,
            top_level_keys=top_level_keys,
            value_keys=value_keys,
            record_version=record_version,
            capability_key_field=capability_key_field,
            task_id_field=task_id_field,
            output_id_field=output_id_field,
        )

    @property
    def key(self) -> tuple[str, str, str, ArtifactProfile]:
        return (
            self.capability_key,
            self.capability_version,
            self.payload_schema,
            self.profile,
        )

    def artifact_validation(
        self,
        *,
        task_id: str,
        output_id: str,
    ) -> ArtifactSemanticValidation:
        """Bind the static contract to one frozen task/output context."""

        validate_stable_id(task_id, field_name="task_id")
        validate_stable_id(output_id, field_name="output_id")
        bindings = tuple(
            sorted(
                (
                    *(
                        (field_name, expected)
                        for field_name, expected in (
                            (self.capability_key_field, self.capability_key),
                            (self.task_id_field, task_id),
                            (self.output_id_field, output_id),
                        )
                        if field_name is not None
                    ),
                    *(
                        (("version", self.record_version),)
                        if self.record_version is not None
                        else ()
                    ),
                ),
                key=lambda value: value[0],
            )
        )
        return ArtifactSemanticValidation(
            validator_key=self.capability_key,
            validator_version=self.capability_version,
            validator_implementation_sha256=self.capability_implementation_sha256,
            payload_schema=self.payload_schema,
            profile=self.profile,
            top_level_keys=self.top_level_keys,
            value_keys=self.value_keys,
            field_bindings=bindings,
        )

    def artifact_validation_registration(
        self,
        *,
        logical_artifact_id: str,
        task_id: str,
        output_id: str,
    ) -> ArtifactSemanticValidationRegistration:
        """Register the exact task/output-bound validation for one logical artifact."""

        return ArtifactSemanticValidationRegistration(
            logical_artifact_id=logical_artifact_id,
            validation=self.artifact_validation(
                task_id=task_id,
                output_id=output_id,
            ),
        )


def capability_semantic_validation_registry(
    *,
    registry_id: str,
    plan: ProtocolExecutionPlan,
    contracts: tuple[CapabilityOutputSemanticContract, ...],
) -> ArtifactSemanticValidationRegistry:
    """Freeze exact capability semantics for every output in one execution plan."""

    contract_by_key = {contract.key: contract for contract in contracts}
    if len(contract_by_key) != len(contracts):
        raise ValueError("capability semantic contracts must be unique")
    registrations: list[ArtifactSemanticValidationRegistration] = []
    for task in plan.tasks:
        for output in task.outputs:
            key = (
                task.capability.capability_key,
                task.capability.capability_version,
                output.payload_schema,
                output.profile,
            )
            contract = contract_by_key.get(key)
            if contract is None:
                raise ValueError("execution output lacks a registered semantic validator")
            if contract.capability_implementation_sha256 != task.capability_implementation_sha256:
                raise ValueError("execution semantic validator differs from the frozen capability")
            registrations.append(
                contract.artifact_validation_registration(
                    logical_artifact_id=output.logical_artifact_id,
                    task_id=task.task_id,
                    output_id=output.output_id,
                )
            )
    return ArtifactSemanticValidationRegistry.from_registrations(
        registry_id=registry_id,
        registrations=tuple(registrations),
    )


@dataclass(frozen=True, slots=True)
class ExternalInputPayload:
    """Bounded source for one frozen external input identity."""

    logical_artifact_id: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    source: ExternalInputSource = field(repr=False, compare=False)
    size_bytes: int
    source_sha256: str
    maximum_bytes: int
    maximum_chunk_bytes: int
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    lineage_parents: tuple[ArtifactLineageParent, ...]
    logical_content_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.size_bytes < 0 or self.maximum_bytes <= 0:
            raise ValueError("external input byte bounds are invalid")
        if self.size_bytes > self.maximum_bytes:
            raise ValueError("external input size exceeds its declared bound")
        if (
            self.maximum_chunk_bytes <= 0
            or self.maximum_chunk_bytes > MAX_EXTERNAL_INPUT_CHUNK_BYTES
            or self.maximum_chunk_bytes > self.maximum_bytes
        ):
            raise ValueError("external input chunk bound is invalid")
        validate_sha256(self.source_sha256, field_name="source_sha256")
        if self.logical_content_sha256 is not None:
            validate_sha256(
                self.logical_content_sha256,
                field_name="logical_content_sha256",
            )
        parent_keys = tuple(lineage_parent_sort_key(parent) for parent in self.lineage_parents)
        if tuple(sorted(set(parent_keys))) != parent_keys:
            raise ValueError("external input parents must have sorted unique identities")
        if tuple(sorted(self.parent_visibility_ceilings, key=lambda value: value.value)) != tuple(
            sorted(
                (parent.visibility_ceiling for parent in self.lineage_parents),
                key=lambda value: value.value,
            )
        ):
            raise ValueError("external input parent identities and ceilings differ")

    @classmethod
    def from_bytes(
        cls,
        *,
        logical_artifact_id: str,
        payload_schema: str,
        profile: ArtifactProfile,
        media_type: str,
        payload: bytes,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
        parent_visibility_ceilings: tuple[VisibilityCeiling, ...],
        lineage_parents: tuple[ArtifactLineageParent, ...],
        logical_content_sha256: str | None = None,
    ) -> ExternalInputPayload:
        """Construct the explicitly capped small-byte convenience source."""

        if not isinstance(payload, bytes):
            raise ValueError("external input convenience payload must be immutable bytes")
        if len(payload) > MAX_IN_MEMORY_EXTERNAL_INPUT_BYTES:
            raise ValueError(
                "external input exceeds the small-byte convenience limit; use a bounded source"
            )
        return cls(
            logical_artifact_id=logical_artifact_id,
            payload_schema=payload_schema,
            profile=profile,
            media_type=media_type,
            source=_BytesExternalInputSource(payload),
            size_bytes=len(payload),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            maximum_bytes=max(1, len(payload)),
            maximum_chunk_bytes=min(
                MAX_EXTERNAL_INPUT_CHUNK_BYTES,
                max(1, len(payload)),
            ),
            visibility_ceiling=visibility_ceiling,
            outcome_access=outcome_access,
            parent_visibility_ceilings=parent_visibility_ceilings,
            lineage_parents=lineage_parents,
            logical_content_sha256=logical_content_sha256,
        )

    def chunks(self) -> Iterator[bytes]:
        """Yield bytes while enforcing the declared source identity and bounds."""

        digest = hashlib.sha256()
        size_bytes = 0
        for chunk in self.source.chunks(self.maximum_chunk_bytes):
            if not isinstance(chunk, bytes):
                raise ValueError("external input chunks must be immutable bytes")
            if len(chunk) > self.maximum_chunk_bytes:
                raise ValueError("external input source exceeded its chunk bound")
            size_bytes += len(chunk)
            if size_bytes > self.maximum_bytes:
                raise ValueError("external input source exceeded its total byte bound")
            digest.update(chunk)
            yield chunk
        if size_bytes != self.size_bytes or digest.hexdigest() != self.source_sha256:
            raise ValueError("external input source bytes differ from their declared identity")

    def close(self) -> None:
        self.source.close()


class CampaignRuntimeProvider:
    """Static capability implementation and source provisioning boundary."""

    registry_sha256: str
    capability_count: int
    issued_source_schema_ids: tuple[str, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        raise NotImplementedError

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        raise NotImplementedError

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        raise NotImplementedError

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        """Return only a static output locator, never a scientific verdict."""

        raise NotImplementedError


class CampaignRuntimeProviderRegistry:
    """Resolve only an explicitly registered capability-registry fingerprint."""

    def __init__(self, providers: tuple[CampaignRuntimeProvider, ...]) -> None:
        fingerprints = tuple(provider.registry_sha256 for provider in providers)
        for fingerprint in fingerprints:
            validate_sha256(fingerprint, field_name="registry_sha256")
        if tuple(sorted(set(fingerprints))) != fingerprints:
            raise ValueError("runtime providers must have sorted unique fingerprints")
        self._providers = providers

    def resolve(self, registry_sha256: str) -> CampaignRuntimeProvider:
        validate_sha256(registry_sha256, field_name="registry_sha256")
        for provider in self._providers:
            if provider.registry_sha256 == registry_sha256:
                return provider
        raise KeyError(f"no runtime provider is registered for {registry_sha256}")

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(provider.registry_sha256 for provider in self._providers)

    @property
    def capability_count(self) -> int:
        return sum(provider.capability_count for provider in self._providers)
