"One-capability provider for parameterised Gym--TORAX acquisition."

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_native_execution_request
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ExternalInputSpec
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataNativeEpisode
from .source import GYM_TORAX_CANONICAL_MEDIA_TYPE, GYM_TORAX_ACQUISITION_TASK_PREFIX, GYM_TORAX_PARAMETERISED_PROVIDER_KEY, GYM_TORAX_PARAMETERISED_PROVIDER_VERSION, GymToraxFieldMetadataEpisodeAcquirer, GymToraxParameterisedAcquisitionManifest, GymToraxParameterisedEpisodeTask, GymToraxParameterisedSourceConfig, default_gym_torax_acquirer, materialise_gym_torax_parameterised_request, validate_gym_torax_parameterised_source_config


def gym_torax_acquisition_task_id(acquisition_group_id: str) -> str:
    return f"{GYM_TORAX_ACQUISITION_TASK_PREFIX}.{acquisition_group_id}"


def gym_torax_request_artifact_id(acquisition_group_id: str) -> str:
    return f"request.gym-torax.{acquisition_group_id}"


def gym_torax_episode_artifact_id(acquisition_group_id: str) -> str:
    return f"episode.gym-torax.{acquisition_group_id}"


def gym_torax_stage_envelope_artifact_id(acquisition_group_id: str) -> str:
    return f"stage-envelope.gym-torax.{acquisition_group_id}"


class GymToraxParameterisedCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Exact group-task provider; projection remains a separate pure capability."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: GymToraxParameterisedSourceConfig,
        acquisition_manifest: GymToraxParameterisedAcquisitionManifest,
        repository_root: Path,
        acquirer: GymToraxFieldMetadataEpisodeAcquirer | None = None,
    ) -> None:
        validate_gym_torax_parameterised_source_config(
            config=source_config,
            extension_set=extension_set,
            substrate_binding=substrate_binding,
        )
        installed = registry.resolve(
            GYM_TORAX_PARAMETERISED_PROVIDER_KEY,
            GYM_TORAX_PARAMETERISED_PROVIDER_VERSION,
        )
        if (
            installed != manifest
            or manifest.implementation_sha256 != substrate_binding.provider_implementation_sha256
            or manifest.input_schema_ids != (GymToraxParameterisedAcquisitionManifest.SCHEMA,)
            or set(manifest.output_schema_ids)
            != {GymToraxFieldMetadataNativeEpisode.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA}
        ):
            raise ValueError("Gym provider differs from installed substrate semantics")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.extension_set = extension_set
        self.substrate_binding = substrate_binding
        self.source_config = source_config
        self.acquisition_manifest = acquisition_manifest
        self.repository_root = repository_root.resolve(strict=True)
        self._runner = GymToraxParameterisedEpisodeTask(
            manifest=manifest,
            extension_set=extension_set,
            substrate_binding=substrate_binding,
            source_config=source_config,
            acquisition_manifest=acquisition_manifest,
            acquirer=acquirer
            or default_gym_torax_acquirer(
                config=source_config,
                repository_root=self.repository_root,
            ),
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Gym provider registry/source records differ")
        return (cast(TaskRunner, self._runner),)

    def _requests(self) -> tuple[tuple[str, GymToraxFieldMetadataEpisodeRequest], ...]:
        identification_config = self.extension_set.identification_config
        units = {value.physical_independent_unit_id: value for value in identification_config.physical_units}
        views = {value.view_id: value for value in identification_config.nested_views}
        results = []
        for group in identification_config.acquisition_groups:
            unit = units[group.physical_independent_unit_id]
            selected_view = views[group.view_ids[0]]
            if any(
                views[view_id].action_word_id != selected_view.action_word_id
                or views[view_id].numerical_member_id != selected_view.numerical_member_id
                for view_id in group.view_ids
            ):
                raise ValueError("one Gym acquisition group requests heterogeneous episodes")
            intent = materialise_native_execution_request(
                request_id=gym_torax_request_artifact_id(group.acquisition_group_id),
                extension_set=self.extension_set,
                binding=self.substrate_binding,
                installed_executable_binding=self.substrate_binding.installed_executable_binding,
                stage_id='stage.measurement-source-materialization',
                causal_cutoff_id=self.extension_set.identification_config.action_chart.common_horizon_id
                if self.extension_set.identification_config.action_chart is not None
                else "cutoff.gym-torax.unspecified",
                physical_unit=unit,
                acquisition_group=group,
                nested_view=selected_view,
            )
            request = materialise_gym_torax_parameterised_request(
                intent=intent,
                extension_set=self.extension_set,
                substrate_binding=self.substrate_binding,
                config=self.source_config,
            )
            results.append((group.acquisition_group_id, request))
        return tuple(results)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Gym provider plan/source records differ")
        tasks = {
            task.task_id: task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
        }
        requests = self._requests()
        expected_task_ids = {gym_torax_acquisition_task_id(group_id) for group_id, _ in requests}
        if set(tasks) != expected_task_ids:
            raise ValueError("Gym plan tasks differ from acquisition groups")
        for group_id, _request in requests:
            task = tasks[gym_torax_acquisition_task_id(group_id)]
            if task.capability_implementation_sha256 != self.manifest.implementation_sha256 or {
                value.payload_schema for value in task.outputs
            } != set(self.manifest.output_schema_ids):
                raise ValueError("Gym acquisition task identity/output shape differs")
        config_specs = {
            value.logical_artifact_id: value
            for task in tasks.values()
            for value in task.external_inputs
            if value.expected_payload_schema == GymToraxParameterisedSourceConfig.SCHEMA
        }
        if (
            len(config_specs) != 1
            or next(iter(config_specs.values())).expected_content_sha256
            != self.source_config.fingerprint()
        ):
            raise ValueError("Gym acquisition tasks lack the exact shared source config")
        manifest_specs = {
            value.logical_artifact_id: value
            for task in tasks.values()
            for value in task.external_inputs
            if value.expected_payload_schema == GymToraxParameterisedAcquisitionManifest.SCHEMA
        }
        if (
            len(manifest_specs) != 1
            or next(iter(manifest_specs.values())).expected_content_sha256
            != self.acquisition_manifest.fingerprint()
        ):
            raise ValueError("Gym acquisition tasks lack the exact shared acquisition manifest")
        source_identity = ObjectIdentity.from_record(
            self.source_config.config_id,
            self.source_config,
        )
        parent = ArtifactLineageParent(
            identity=source_identity,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        manifest_parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(
                self.acquisition_manifest.manifest_id,
                self.acquisition_manifest,
            ),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )

        def payload(
            record: CanonicalRecord,
            spec: ExternalInputSpec,
            parent_record: ArtifactLineageParent,
        ) -> ExternalInputPayload:
            return ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=GYM_TORAX_CANONICAL_MEDIA_TYPE,
                payload=record.canonical_bytes(),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent_record.visibility_ceiling,),
                lineage_parents=(parent_record,),
                logical_content_sha256=spec.expected_content_sha256,
            )

        return tuple(
            sorted(
                (
                    payload(
                        self.source_config,
                        next(iter(config_specs.values())),
                        parent,
                    ),
                    payload(
                        self.acquisition_manifest,
                        next(iter(manifest_specs.values())),
                        manifest_parent,
                    ),
                ),
                key=lambda value: value.logical_artifact_id,
            )
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Gym provider semantic registry differs")
        del execution_plan
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(record_type))),
            )
            for record_type in (GymToraxFieldMetadataNativeEpisode, LinkedCampaignStageEnvelope)
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("Gym provider adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "GYM_TORAX_ACQUISITION_TASK_PREFIX",
    "GymToraxParameterisedCampaignRuntimeProvider",
    'gym_torax_acquisition_task_id',
    'gym_torax_episode_artifact_id',
    'gym_torax_request_artifact_id',
    'gym_torax_stage_envelope_artifact_id',
]
