"""Static candidate/runtime bindings for the frozen 2025 Virtual Cell route."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import TaskRunner
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .config import (
    CAPABILITY_VERSION,
    EVALUATOR_CAPABILITY_KEY,
    REPORTER_CAPABILITY_KEY,
    decode_pipeline_config,
)
from .contracts import LeaderboardSnapshot, PlacementAdjudication, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest, VirtualCellSourceObject
from .ports import VirtualCellScratchPort, VirtualCellSourcePort
from .protocol import (
    CONFIG_ARTIFACT_ID,
    DEVELOPMENT_REPORT_STEP,
    LEADERBOARD_SNAPSHOT_INPUT_ID,
    SEALED_TEST_BINDING_INPUT_ID,
    TEST_ROSTER_INPUT_ID,
)
from .records import (
    CONTROL_RESERVOIR_TABLE_SCHEMA,
    FalsifierPanel,
    MODEL_SAFETENSORS_SCHEMA,
    ModelDevelopmentRecord,
    ModelSelectionRecord,
    OFFICIAL_METRICS_TABLE_SCHEMA,
    OfficialEvaluationRecord,
    PREDICTED_MEAN_TABLE_SCHEMA,
    PREDICTION_H5AD_ENVELOPE_SCHEMA,
    PreparedSourceRecord,
    PredictionFreezeRecord,
    RESPONSE_SUMMARY_TABLE_SCHEMA,
    ResponseSummaryRecord,
    TARGET_FEATURE_TABLE_SCHEMA,
    TEST_ROSTER_TEXT_SCHEMA,
    VirtualCellDevelopmentCloseout,
    VirtualCellRunCloseout,
)
from .runtime import VirtualCellTaskRunner, canonical_json_record_value_keys


_JSON_TYPES: tuple[type[CanonicalRecord], ...] = (
    FalsifierPanel,
    PlacementAdjudication,
    ModelDevelopmentRecord,
    ModelSelectionRecord,
    OfficialEvaluationRecord,
    PreparedSourceRecord,
    PredictionFreezeRecord,
    ResponseSummaryRecord,
    ScientificAdjudicationRecord,
    VirtualCellDevelopmentCloseout,
    VirtualCellRunCloseout,
)

_PROFILE_BY_SCHEMA = {
    CONTROL_RESERVOIR_TABLE_SCHEMA: ArtifactProfile.ARROW_IPC,
    MODEL_SAFETENSORS_SCHEMA: ArtifactProfile.SAFETENSORS,
    OFFICIAL_METRICS_TABLE_SCHEMA: ArtifactProfile.ARROW_IPC,
    PREDICTED_MEAN_TABLE_SCHEMA: ArtifactProfile.ARROW_IPC,
    PREDICTION_H5AD_ENVELOPE_SCHEMA: ArtifactProfile.AUDITED_HDF5,
    RESPONSE_SUMMARY_TABLE_SCHEMA: ArtifactProfile.ARROW_IPC,
    TARGET_FEATURE_TABLE_SCHEMA: ArtifactProfile.ARROW_IPC,
}


class VirtualCellConfigDecoder:
    """Exact candidate config decoder; provider selection remains static."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        config: ProvenanceBoundVirtualCellPipelineConfig,
    ) -> None:
        self.manifest = manifest
        self.config = config

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != ProvenanceBoundVirtualCellPipelineConfig.SCHEMA:
            raise ValueError("Virtual Cell config schema differs")
        if decode_pipeline_config(payload) != self.config:
            raise ValueError(
                "Virtual Cell config bytes differ from the registered config"
            )


def virtual_cell_config_decoders(
    *,
    registry: CapabilityRegistry,
    config: ProvenanceBoundVirtualCellPipelineConfig,
) -> tuple[VirtualCellConfigDecoder, ...]:
    return tuple(
        VirtualCellConfigDecoder(value, config) for value in registry.capabilities
    )


class VirtualCellCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind compact external inputs and injected held bytes to the exact registry."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: ProvenanceBoundVirtualCellPipelineConfig,
        source_manifest: VirtualCellSourceManifest,
        target_feature_payload: bytes,
        test_roster_payload: bytes,
        test_source: VirtualCellSourceObject,
        leaderboard_snapshot: LeaderboardSnapshot,
        source_port: VirtualCellSourcePort,
        scratch_port: VirtualCellScratchPort,
    ) -> None:
        if config.source_manifest_sha256 != source_manifest.fingerprint():
            raise ValueError("Virtual Cell source manifest differs from config")
        if sha256(target_feature_payload).hexdigest() != config.target_feature_sha256:
            raise ValueError("Virtual Cell feature payload differs from config")
        if leaderboard_snapshot.fingerprint() != config.leaderboard_snapshot_sha256:
            raise ValueError("Virtual Cell leaderboard snapshot differs from config")
        manifest_tests = tuple(
            value for value in source_manifest.objects if value.sealed
        )
        if manifest_tests != (test_source,):
            raise ValueError(
                "Virtual Cell sealed test binding differs from source manifest"
            )
        if not test_roster_payload or len(test_roster_payload) > 1024**2:
            raise ValueError("Virtual Cell test roster payload is absent or unbounded")
        manifest_rosters = tuple(
            value
            for value in source_manifest.objects
            if value.split.value == "TEST" and value.role == "prefix/target-roster"
        )
        if (
            len(manifest_rosters) != 1
            or manifest_rosters[0].size_bytes != len(test_roster_payload)
            or manifest_rosters[0].sha256 != sha256(test_roster_payload).hexdigest()
        ):
            raise ValueError(
                "Virtual Cell test roster payload differs from source custody"
            )
        self.registry = registry
        self.config = config
        self.source_manifest = source_manifest
        self.target_feature_payload = target_feature_payload
        self.test_roster_payload = test_roster_payload
        self.test_source = test_source
        self.leaderboard_snapshot = leaderboard_snapshot
        self.source_port = source_port
        self.scratch_port = scratch_port
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry != self.registry or registry.fingerprint() != self.registry_sha256:
            raise ValueError("Virtual Cell provider registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._validate_registry(registry)
        if source_records:
            raise ValueError(
                "Virtual Cell runtime accepts no unbound issued source records"
            )
        return tuple(
            VirtualCellTaskRunner(
                manifest=manifest,
                config=self.config,
                source_manifest=self.source_manifest,
                source_port=self.source_port,
                scratch_port=self.scratch_port,
            )
            for manifest in registry.capabilities
        )

    @staticmethod
    def _parent(
        *,
        object_id: str,
        payload_schema: str,
        fingerprint: str,
        visibility: VisibilityCeiling,
        access: OutcomeAccess,
    ) -> ArtifactLineageParent:
        return ArtifactLineageParent(
            identity=ObjectIdentity(
                object_id=object_id,
                object_schema=payload_schema,
                object_version="1.0.0",
                object_fingerprint=fingerprint,
            ),
            visibility_ceiling=visibility,
            outcome_access=access,
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records:
            raise ValueError(
                "Virtual Cell runtime accepts no unbound issued source records"
            )
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("Virtual Cell execution plan registry differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        config_payload = self.config.canonical_bytes()
        manifest_payload = self.source_manifest.canonical_bytes()
        sealed_payload = self.test_source.canonical_bytes()
        leaderboard_payload = self.leaderboard_snapshot.canonical_bytes()
        payloads = {
            CONFIG_ARTIFACT_ID: (
                ProvenanceBoundVirtualCellPipelineConfig.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                "application/json",
                config_payload,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                self.config.config_id,
                self.config.fingerprint(),
            ),
            self.source_manifest.manifest_id: (
                VirtualCellSourceManifest.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                "application/json",
                manifest_payload,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                self.source_manifest.manifest_id,
                self.source_manifest.fingerprint(),
            ),
            self.config.target_feature_artifact_id: (
                TARGET_FEATURE_TABLE_SCHEMA,
                ArtifactProfile.ARROW_IPC,
                "application/vnd.apache.arrow.file",
                self.target_feature_payload,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                self.config.target_feature_artifact_id,
                sha256(self.target_feature_payload).hexdigest(),
            ),
            TEST_ROSTER_INPUT_ID: (
                TEST_ROSTER_TEXT_SCHEMA,
                ArtifactProfile.TEXT_PARAMETERS,
                "text/csv",
                self.test_roster_payload,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                TEST_ROSTER_INPUT_ID,
                sha256(self.test_roster_payload).hexdigest(),
            ),
            SEALED_TEST_BINDING_INPUT_ID: (
                VirtualCellSourceObject.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                "application/json",
                sealed_payload,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
                self.test_source.object_id,
                self.test_source.fingerprint(),
            ),
            LEADERBOARD_SNAPSHOT_INPUT_ID: (
                LeaderboardSnapshot.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                "application/json",
                leaderboard_payload,
                OutcomeAccess.EVALUATION_REVEALED,
                VisibilityCeiling.OUTCOME_VISIBLE,
                self.leaderboard_snapshot.snapshot_id,
                self.leaderboard_snapshot.fingerprint(),
            ),
        }
        if not set(specs).issubset(payloads):
            missing = tuple(sorted(set(specs) - set(payloads)))
            raise ValueError(
                f"Virtual Cell execution external-input set differs: unbound={missing!r}"
            )
        payloads = {key: payloads[key] for key in specs}
        values = []
        for artifact_id in sorted(payloads):
            (
                schema,
                profile,
                media_type,
                payload,
                access,
                visibility,
                object_id,
                fingerprint,
            ) = payloads[artifact_id]
            spec = specs[artifact_id]
            if (
                spec.expected_content_sha256 not in (None, fingerprint)
                or spec.expected_payload_schema not in (None, schema)
                or spec.expected_media_type not in (None, media_type)
                or spec.expected_size_bytes not in (None, len(payload))
                or spec.expected_visibility_ceiling not in (None, visibility)
                or spec.expected_outcome_access not in (None, access)
            ):
                planned = (
                    spec.expected_content_sha256,
                    spec.expected_payload_schema,
                    spec.expected_media_type,
                    spec.expected_size_bytes,
                    None
                    if spec.expected_visibility_ceiling is None
                    else spec.expected_visibility_ceiling.value,
                    None
                    if spec.expected_outcome_access is None
                    else spec.expected_outcome_access.value,
                )
                provided = (
                    fingerprint,
                    schema,
                    media_type,
                    len(payload),
                    visibility.value,
                    access.value,
                )
                raise ValueError(
                    f"Virtual Cell external input contract differs for {artifact_id}: "
                    f"plan={planned!r}, provider={provided!r}"
                )
            parent = self._parent(
                object_id=object_id,
                payload_schema=schema,
                fingerprint=fingerprint,
                visibility=visibility,
                access=access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=schema,
                    profile=profile,
                    media_type=media_type,
                    payload=payload,
                    visibility_ceiling=parent.visibility_ceiling,
                    outcome_access=parent.outcome_access,
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=fingerprint,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._validate_registry(registry)
        json_types: dict[str, type[CanonicalRecord]] = {
            value.SCHEMA: value for value in _JSON_TYPES
        }
        values: list[CapabilityOutputSemanticContract] = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                if schema in json_types:
                    values.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            top_level_keys=("schema", "value", "version"),
                            value_keys=canonical_json_record_value_keys(
                                json_types[schema]
                            ),
                        )
                    )
                elif schema in _PROFILE_BY_SCHEMA:
                    values.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=_PROFILE_BY_SCHEMA[schema],
                        )
                    )
        contracts = tuple(sorted(values, key=lambda value: value.key))
        if execution_plan is None:
            return contracts
        planned = {
            (
                task.capability.capability_key,
                task.capability.capability_version,
                output.payload_schema,
                output.profile,
            )
            for task in execution_plan.tasks
            for output in task.outputs
        }
        return tuple(value for value in contracts if value.key in planned)

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        self._validate_registry(registry)
        development = execution_plan is not None and any(
            value.task_id == DEVELOPMENT_REPORT_STEP for value in execution_plan.tasks
        )
        return ScientificAdjudicationOutputContract(
            capability_key=(
                REPORTER_CAPABILITY_KEY if development else EVALUATOR_CAPABILITY_KEY
            ),
            capability_version=CAPABILITY_VERSION,
            output_id=(
                f"{DEVELOPMENT_REPORT_STEP}.scientific-adjudication"
                if development
                else "vcc-evaluate-official.scientific-adjudication"
            ),
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=1024**2,
            fixture_scope_id=None,
            plumbing_only=False,
        )


__all__ = [
    "VirtualCellCampaignRuntimeProvider",
    "VirtualCellConfigDecoder",
    "virtual_cell_config_decoders",
]
