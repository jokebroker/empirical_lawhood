"Decode-only compatibility provider for the retired truth-known runner."

from __future__ import annotations

from dataclasses import fields

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .metatheory_conformance import EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE, METATHEORY_CONFORMANCE_TERMINAL_TYPES, MetatheoryConformanceExecutableConfig, MetatheoryConformanceFixture


class MetatheoryConformanceTaskRunner:
    "Historical current-HEAD tombstone; exact execution is commit-selected."

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: MetatheoryConformanceExecutableConfig,
    ) -> None:
        self.manifest = manifest
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        del context
        raise ValueError("RETIRED_TRUTH_KNOWN_EXECUTION_REQUIRES_EXACT_COMMIT_REPLAY")

    def execute_with_progress(self, context: TaskContext, emitter: object) -> RunnerResult:
        del emitter
        return self.execute(context)


class MetatheoryConformanceProvider(CampaignRuntimeProvider):
    "Preserve decoding/locator metadata without retaining its oracle executor."

    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: MetatheoryConformanceExecutableConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("metatheory provider manifest differs from its registry")
        if (
            config.payload.capability_key != manifest.capability_key
            or config.payload.capability_version != manifest.capability_version
            or config.payload.implementation_sha256 != manifest.implementation_sha256
        ):
            raise ValueError("metatheory provider config differs from its manifest")
        fixture = config.payload.fixture
        fixture_identity = ObjectIdentity.from_record(fixture.fixture_id, fixture)
        if config.payload.campaign.fixture != fixture_identity:
            raise ValueError("metatheory provider config crosses fixtures")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.config = config
        self.fixture = fixture
        self.runner: TaskRunner = MetatheoryConformanceTaskRunner(
            manifest=manifest,
            config=config,
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry:
            raise ValueError("metatheory provider registry differs")
        if source_records:
            raise ValueError("metatheory provider accepts no issued source records")
        return (self.runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records:
            raise ValueError("metatheory provider accepts no issued source records")
        specifications = {
            value.logical_artifact_id: value
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            for value in task.external_inputs
        }
        payloads = []
        for specification in specifications.values():
            if specification.expected_payload_schema == self.config.SCHEMA:
                record: CanonicalRecord = self.config
                record_id = self.config.config_id
            elif specification.expected_payload_schema == MetatheoryConformanceFixture.SCHEMA:
                record = self.fixture
                record_id = self.fixture.fixture_id
            else:
                raise ValueError("metatheory provider lacks one compiled external input")
            if (
                specification.expected_content_sha256 is not None
                and specification.expected_content_sha256 != record.fingerprint()
            ) or specification.expected_media_type not in {
                None,
                EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE,
            }:
                raise ValueError("metatheory external input identity differs")
            visibility = specification.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
            access = specification.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            payloads.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=specification.logical_artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE,
                    payload=record.canonical_bytes(),
                    visibility_ceiling=visibility,
                    outcome_access=access,
                    parent_visibility_ceilings=(visibility,),
                    lineage_parents=(parent,),
                    logical_content_sha256=record.fingerprint(),
                )
            )
        return tuple(sorted(payloads, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("metatheory provider registry differs")
        del execution_plan
        output_schema_ids = set(self.manifest.output_schema_ids)
        record_types: tuple[type[CanonicalRecord], ...] = tuple(
            value
            for value in (*METATHEORY_CONFORMANCE_TERMINAL_TYPES, ScientificAdjudicationRecord)
            if value.SCHEMA in output_schema_ids
        )
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=(
                    ("envelope_id", "terminal")
                    if record_type in METATHEORY_CONFORMANCE_TERMINAL_TYPES
                    else tuple(sorted(value.name for value in fields(ScientificAdjudicationRecord)))
                ),
            )
            for record_type in record_types
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("metatheory provider registry differs")
        del execution_plan
        if ScientificAdjudicationRecord.SCHEMA not in set(self.manifest.output_schema_ids):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id="metatheory-report.report",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=1_000_000,
            fixture_scope_id=self.fixture.fixture_id,
            plumbing_only=True,
        )


def metatheory_conformance_provider(
    *,
    registry: CapabilityRegistry,
    manifest: CapabilityManifest,
    config: MetatheoryConformanceExecutableConfig,
) -> CampaignRuntimeProvider:
    return MetatheoryConformanceProvider(
        registry=registry,
        manifest=manifest,
        config=config,
    )


__all__ = [
    'MetatheoryConformanceProvider',
    'MetatheoryConformanceTaskRunner',
    'metatheory_conformance_provider',
]
