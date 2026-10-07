"Parameterised Gym--TORAX acquisition without scientific adjudication."

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest
from empirical_lawhood.runtime.execution import StreamingOutputEmitter, TaskContext
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_native_execution_request
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.response_experiment_ports import NativeExecutionRequest, ResponseSubstrateBinding, SourceReadinessCheck, SourceReadinessDisposition

from .action_word import build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxNumericalMember, GymToraxPreparation
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataNativeEpisode, GymToraxRequestRole
from .extraction_manifest import GymToraxBoundedExtractionManifest, verify_gym_torax_extraction_manifest
from .field_metadata import GymToraxFieldMetadataManifest, verify_installed_gym_torax_metadata_sources
from .runtime import GymToraxRuntimePreflightError, acquire_gym_torax_episode, inspect_gym_torax_runtime
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


GYM_TORAX_PARAMETERISED_PROVIDER_KEY = 'gym-torax-native.parameterised-source-acquisition'
GYM_TORAX_PARAMETERISED_PROVIDER_VERSION = "1.0.0"
GYM_TORAX_CANONICAL_MEDIA_TYPE = "application/json"
GYM_TORAX_MAXIMUM_REQUEST_BYTES = 8 * 1024 * 1024
GYM_TORAX_MAXIMUM_SOURCE_CONFIG_BYTES = 32 * 1024 * 1024
GYM_TORAX_ACQUISITION_TASK_PREFIX = "task.gym-torax-acquire"


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedAcquisitionBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-parameterised-acquisition-binding'

    binding_id: str
    acquisition_group_id: str
    preparation_instance_id: str
    physical_independent_unit_id: str
    preparation_coordinate_id: str
    view_ids: tuple[str, ...]
    preparation_sha256: str
    environment_seed_id: str
    environment_seed: int
    numerical_member_id: str
    preparation: GymToraxPreparation
    numerical_member: GymToraxNumericalMember

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "acquisition_group_id",
            "preparation_instance_id",
            "physical_independent_unit_id",
            "preparation_coordinate_id",
            "environment_seed_id",
            "numerical_member_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if tuple(sorted(set(self.view_ids))) != self.view_ids or not self.view_ids:
            raise ValueError("Gym--TORAX acquisition views must be sorted and nonempty")
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        if (
            self.environment_seed < 0
            or self.environment_seed != self.preparation.environment_seed
            or self.physical_independent_unit_id != self.preparation.physical_independent_unit_id
            or self.preparation_sha256 != self.preparation.fingerprint()
            or self.numerical_member_id != self.numerical_member.member_id
        ):
            raise ValueError("Gym--TORAX adapter realization differs from its native records")


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedSourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-parameterised-source-config'

    config_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    source_qualification: GymToraxNativeSourceQualificationReceipt
    extraction_manifest: GymToraxBoundedExtractionManifest
    field_metadata_manifest: GymToraxFieldMetadataManifest
    acquisition_bindings: tuple[GymToraxParameterisedAcquisitionBinding, ...]
    maximum_episode_bytes: int
    maximum_episode_outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.experiment_extension_set.object_schema != ResponseExperimentExtensionSet.SCHEMA:
            raise ValueError("Gym source config requires a measurement through controller use extension set")
        if self.substrate_binding.object_schema != ResponseSubstrateBinding.SCHEMA:
            raise ValueError("Gym source config requires the open substrate binding")
        require_sorted_unique_ids(
            self.acquisition_bindings,
            attribute="acquisition_group_id",
            field_name="acquisition_bindings",
        )
        if not self.acquisition_bindings:
            raise ValueError("Gym source config requires acquisition bindings")
        all_views = tuple(
            sorted(view for binding in self.acquisition_bindings for view in binding.view_ids)
        )
        if len(set(all_views)) != len(all_views):
            raise ValueError("Gym source config maps a view to multiple acquisitions")
        qualification = self.source_qualification
        if (
            qualification.disposition is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
            or qualification.extraction_manifest
            != ObjectIdentity.from_record(
                self.extraction_manifest.manifest_id,
                self.extraction_manifest,
            )
            or qualification.field_metadata_manifest
            != ObjectIdentity.from_record(
                self.field_metadata_manifest.manifest_id,
                self.field_metadata_manifest,
            )
        ):
            raise ValueError("Gym source config lacks exact source qualification")
        if not 0 < self.maximum_episode_bytes <= 512 * 1024**2:
            raise ValueError("Gym source episode bound is outside 512 MiB")
        if (
            self.maximum_episode_outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.maximum_evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("Gym source config exceeds its source-only authority")

    def binding(self, acquisition_group_id: str) -> GymToraxParameterisedAcquisitionBinding:
        for value in self.acquisition_bindings:
            if value.acquisition_group_id == acquisition_group_id:
                return value
        raise KeyError(acquisition_group_id)


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedAcquisitionManifest(CanonicalRecord):
    """Shared scientific source declaration, distinct from executable config."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-parameterised-acquisition-manifest'

    manifest_id: str
    source_config: ObjectIdentity
    acquisition_group_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if self.source_config.object_schema != GymToraxParameterisedSourceConfig.SCHEMA:
            raise ValueError("Gym acquisition manifest binds another source config")
        require_sorted_unique_strings(
            self.acquisition_group_ids,
            field_name="acquisition_group_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Gym acquisition manifest cannot grant authority or reveal outcomes")


def validate_gym_torax_parameterised_acquisition_manifest(
    *,
    manifest: GymToraxParameterisedAcquisitionManifest,
    extension_set: ResponseExperimentExtensionSet,
    source_config: GymToraxParameterisedSourceConfig,
) -> None:
    if manifest.source_config != ObjectIdentity.from_record(
        source_config.config_id,
        source_config,
    ):
        raise ValueError("Gym acquisition manifest binds another source realization")
    if manifest.acquisition_group_ids != tuple(
        sorted(
            value.acquisition_group_id for value in extension_set.identification_config.acquisition_groups
        )
    ) or manifest.view_ids != tuple(
        sorted(value.view_id for value in extension_set.identification_config.nested_views)
    ):
        raise ValueError("Gym acquisition manifest changes the group/view roster")


def validate_gym_torax_parameterised_source_config(
    *,
    config: GymToraxParameterisedSourceConfig,
    extension_set: ResponseExperimentExtensionSet,
    substrate_binding: ResponseSubstrateBinding,
) -> None:
    """Join adapter realization to the neutral unit/group/view model exactly."""

    if config.experiment_extension_set != ObjectIdentity.from_record(
        extension_set.extension_set_id, extension_set
    ) or config.substrate_binding != ObjectIdentity.from_record(
        substrate_binding.binding_id, substrate_binding
    ):
        raise ValueError("Gym source config changed its generic parents")
    identification_config = extension_set.identification_config
    if (
        config.source_pipeline_profile != identification_config.source_pipeline_profile
        or substrate_binding.provider_key != GYM_TORAX_PARAMETERISED_PROVIDER_KEY
        or substrate_binding.provider_version != GYM_TORAX_PARAMETERISED_PROVIDER_VERSION
        or substrate_binding.native_episode_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA
    ):
        raise ValueError("Gym source config changes source/provider native semantics")
    groups = {value.acquisition_group_id: value for value in identification_config.acquisition_groups}
    units = {value.physical_independent_unit_id: value for value in identification_config.physical_units}
    views = {value.view_id: value for value in identification_config.nested_views}
    if set(groups) != {value.acquisition_group_id for value in config.acquisition_bindings}:
        raise ValueError("Gym acquisition binding roster differs from extension groups")
    for binding in config.acquisition_bindings:
        group = groups[binding.acquisition_group_id]
        unit = units[group.physical_independent_unit_id]
        if (
            binding.preparation_instance_id != group.preparation_instance_id
            or binding.physical_independent_unit_id != group.physical_independent_unit_id
            or binding.preparation_coordinate_id != unit.preparation_coordinate_id
            or binding.view_ids != group.view_ids
            or binding.preparation_sha256 != unit.preparation_sha256
            or binding.environment_seed_id != unit.adapter_realization_id
            or any(
                views[view_id].acquisition_group_id != group.acquisition_group_id
                or views[view_id].numerical_member_id != binding.numerical_member_id
                for view_id in group.view_ids
            )
        ):
            raise ValueError("Gym acquisition binding changes unit/group/view lineage")


def materialise_gym_torax_parameterised_request(
    *,
    intent: NativeExecutionRequest,
    extension_set: ResponseExperimentExtensionSet,
    substrate_binding: ResponseSubstrateBinding,
    config: GymToraxParameterisedSourceConfig,
) -> GymToraxFieldMetadataEpisodeRequest:
    "Lower one neutral interactive intent to the authoritative request."

    validate_gym_torax_parameterised_source_config(
        config=config,
        extension_set=extension_set,
        substrate_binding=substrate_binding,
    )
    if (
        intent.experiment_extension_set != config.experiment_extension_set
        or intent.substrate_binding != config.substrate_binding
        or intent.native_config != substrate_binding.native_config
        or intent.native_request_schema != substrate_binding.native_request_schema
    ):
        raise ValueError("Gym native intent differs from its frozen config/binding")
    binding = config.binding(intent.acquisition_group_id)
    if (
        intent.preparation_instance_id != binding.preparation_instance_id
        or intent.physical_independent_unit_id != binding.physical_independent_unit_id
        or intent.preparation_coordinate_id != binding.preparation_coordinate_id
        or intent.preparation_sha256 != binding.preparation_sha256
        or intent.adapter_realization_id != binding.environment_seed_id
        or intent.nested_view_id not in binding.view_ids
    ):
        raise ValueError("Gym native intent changes adapter realization lineage")
    chart = extension_set.identification_config.action_chart
    if chart is None:
        raise ValueError("interactive Gym acquisition requires a declared action chart")
    try:
        action_word = next(
            value for value in chart.action_words if value.word_id == intent.action_word_id
        )
    except StopIteration as error:
        raise ValueError("Gym intent action word is absent from the issued chart") from error
    return GymToraxFieldMetadataEpisodeRequest(
        request_id=intent.request_id,
        request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
        source_qualification=ObjectIdentity.from_record(
            config.source_qualification.receipt_id,
            config.source_qualification,
        ),
        extraction_manifest=ObjectIdentity.from_record(
            config.extraction_manifest.manifest_id,
            config.extraction_manifest,
        ),
        field_metadata_manifest=ObjectIdentity.from_record(
            config.field_metadata_manifest.manifest_id,
            config.field_metadata_manifest,
        ),
        preparation=binding.preparation,
        numerical_member=binding.numerical_member,
        schedule=build_gym_torax_native_schedule(action_word),
        maximum_output_bytes=config.maximum_episode_bytes,
        outcome_access=config.maximum_episode_outcome_access,
        evidence_ceiling=config.maximum_evidence_ceiling,
    )


def verify_gym_torax_static_readiness(
    *,
    source_pipeline_profile: ObjectIdentity,
    substrate_binding: ResponseSubstrateBinding,
    config: GymToraxParameterisedSourceConfig,
    repository_root: Path,
) -> SourceReadinessCheck:
    """Inspect installed bytes/versions only; never construct or reset an environment."""

    reasons: set[str] = set()
    disposition = SourceReadinessDisposition.READY
    observed_version: str | None = substrate_binding.provider_version
    observed_implementation: str | None = substrate_binding.provider_implementation_sha256
    try:
        inspect_gym_torax_runtime()
        verify_gym_torax_extraction_manifest(
            config.extraction_manifest,
            expected_identity=ObjectIdentity.from_record(
                config.extraction_manifest.manifest_id,
                config.extraction_manifest,
            ),
            repository_root=repository_root,
        )
        verify_installed_gym_torax_metadata_sources(config.field_metadata_manifest)
    except GymToraxRuntimePreflightError as error:
        reasons.update(error.reason_codes)
        disposition = SourceReadinessDisposition.VERSION_MISMATCH
    except (ImportError, ModuleNotFoundError):
        reasons.add("GYM_TORAX_DEPENDENCY_NOT_INSTALLED")
        disposition = SourceReadinessDisposition.NOT_INSTALLED
        observed_version = None
        observed_implementation = None
    except (OSError, ValueError):
        reasons.add("GYM_TORAX_IMPLEMENTATION_SOURCE_DIVERGENCE")
        disposition = SourceReadinessDisposition.IMPLEMENTATION_MISMATCH
    return SourceReadinessCheck(
        check_id=f"source-readiness.{config.config_id}",
        source_pipeline_profile=source_pipeline_profile,
        provider_key=substrate_binding.provider_key,
        expected_provider_version=substrate_binding.provider_version,
        expected_implementation_sha256=substrate_binding.provider_implementation_sha256,
        disposition=disposition,
        observed_provider_version=observed_version,
        observed_implementation_sha256=observed_implementation,
        reason_codes=tuple(sorted(reasons)),
        grants_authority=False,
    )


GymToraxFieldMetadataEpisodeAcquirer = Callable[[GymToraxFieldMetadataEpisodeRequest], GymToraxFieldMetadataNativeEpisode]


class GymToraxParameterisedEpisodeTask:
    """One issued-config/group to one episode; it never loops or retries."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: GymToraxParameterisedSourceConfig,
        acquisition_manifest: GymToraxParameterisedAcquisitionManifest,
        acquirer: GymToraxFieldMetadataEpisodeAcquirer,
    ) -> None:
        if (
            manifest.kind is not CapabilityKind.SIMULATOR
            or manifest.input_schema_ids != (GymToraxParameterisedAcquisitionManifest.SCHEMA,)
            or set(manifest.output_schema_ids)
            != {GymToraxFieldMetadataNativeEpisode.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA}
        ):
            raise ValueError("Gym parameterised task manifest has another contract")
        validate_gym_torax_parameterised_source_config(
            config=source_config,
            extension_set=extension_set,
            substrate_binding=substrate_binding,
        )
        validate_gym_torax_parameterised_acquisition_manifest(
            manifest=acquisition_manifest,
            extension_set=extension_set,
            source_config=source_config,
        )
        self.manifest = manifest
        self._extension_set = extension_set
        self._substrate_binding = substrate_binding
        self._source_config = source_config
        self._acquisition_manifest = acquisition_manifest
        self._acquirer = acquirer

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        if len(context.input_ports) != 2 or len(context.output_ports) != 2:
            raise ValueError("Gym parameterised task requires config/manifest and two outputs")
        record_types: dict[str, type[CanonicalRecord]] = {
            GymToraxParameterisedSourceConfig.SCHEMA: (GymToraxParameterisedSourceConfig),
            GymToraxParameterisedAcquisitionManifest.SCHEMA: (
                GymToraxParameterisedAcquisitionManifest
            ),
        }
        decoded: dict[str, CanonicalRecord] = {}
        for input_port in context.input_ports:
            record_type = record_types.get(input_port.payload_schema)
            if (
                record_type is None
                or input_port.media_type != GYM_TORAX_CANONICAL_MEDIA_TYPE
                or input_port.size_bytes > GYM_TORAX_MAXIMUM_SOURCE_CONFIG_BYTES
            ):
                input_port.close()
                raise ValueError("Gym parameterised config/manifest port differs")
            try:
                payload = input_port.read(input_port.size_bytes + 1)
                if len(payload) != input_port.size_bytes or input_port.read(1):
                    raise ValueError("Gym parameterised config/manifest size differs")
            finally:
                input_port.close()
            value = decode_canonical_bytes(
                payload,
                record_type,
                maximum_bytes=GYM_TORAX_MAXIMUM_SOURCE_CONFIG_BYTES,
            )
            if value.SCHEMA in decoded:
                raise ValueError("Gym parameterised task received a duplicate typed input")
            decoded[value.SCHEMA] = value
        if set(decoded) != set(record_types):
            raise ValueError("Gym parameterised config/manifest roster differs")
        config = decoded[GymToraxParameterisedSourceConfig.SCHEMA]
        acquisition_manifest = decoded[GymToraxParameterisedAcquisitionManifest.SCHEMA]
        if config != self._source_config or acquisition_manifest != self._acquisition_manifest:
            raise ValueError("Gym parameterised task received a substituted issued record")
        assert isinstance(config, GymToraxParameterisedSourceConfig)
        prefix = f"{GYM_TORAX_ACQUISITION_TASK_PREFIX}."
        if not context.task_id.startswith(prefix):
            raise ValueError("Gym parameterised task identity differs")
        group_id = context.task_id[len(prefix) :]
        if group_id not in self._acquisition_manifest.acquisition_group_ids:
            raise ValueError("Gym task acquisition group lies outside its manifest")
        try:
            group = next(
                value
                for value in self._extension_set.identification_config.acquisition_groups
                if value.acquisition_group_id == group_id
            )
        except StopIteration as error:
            raise ValueError("Gym task acquisition group lies outside the extension") from error
        unit = next(
            value
            for value in self._extension_set.identification_config.physical_units
            if value.physical_independent_unit_id == group.physical_independent_unit_id
        )
        view = next(
            value
            for value in self._extension_set.identification_config.nested_views
            if value.view_id == group.view_ids[0]
        )
        if any(
            value.action_word_id != view.action_word_id
            or value.numerical_member_id != view.numerical_member_id
            for value in self._extension_set.identification_config.nested_views
            if value.view_id in group.view_ids
        ):
            raise ValueError("one Gym acquisition group requests heterogeneous episodes")
        intent = materialise_native_execution_request(
            request_id=f"request.gym-torax.{group_id}",
            extension_set=self._extension_set,
            binding=self._substrate_binding,
            installed_executable_binding=(self._substrate_binding.installed_executable_binding),
            stage_id='stage.measurement-source-materialization',
            causal_cutoff_id=(
                self._extension_set.identification_config.action_chart.common_horizon_id
                if self._extension_set.identification_config.action_chart is not None
                else "cutoff.gym-torax.unspecified"
            ),
            physical_unit=unit,
            acquisition_group=group,
            nested_view=view,
        )
        request = materialise_gym_torax_parameterised_request(
            intent=intent,
            extension_set=self._extension_set,
            substrate_binding=self._substrate_binding,
            config=config,
        )
        episode = self._acquirer(request)
        if (
            episode.request != ObjectIdentity.from_record(request.request_id, request)
            or episode.preparation
            != ObjectIdentity.from_record(request.preparation.preparation_id, request.preparation)
            or episode.numerical_member
            != ObjectIdentity.from_record(
                request.numerical_member.member_id, request.numerical_member
            )
            or episode.action_word
            != ObjectIdentity.from_record(
                request.schedule.action_word.word_id, request.schedule.action_word
            )
            or episode.outcome_access is not request.outcome_access
            or episode.evidence_ceiling is not request.evidence_ceiling
        ):
            raise ValueError("Gym acquired episode changed its issued identities")
        episode_output = next(
            value
            for value in context.output_ports
            if value.payload_schema == GymToraxFieldMetadataNativeEpisode.SCHEMA
        )
        envelope_output = next(
            value
            for value in context.output_ports
            if value.payload_schema == LinkedCampaignStageEnvelope.SCHEMA
        )
        envelope = LinkedCampaignStageEnvelope(
            envelope_id=f"stage-envelope.{context.task_id}",
            role=LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
            disposition=LinkedCampaignDisposition.SUPPORTED,
            scientific_product=ObjectIdentity.from_record(episode.episode_id, episode),
            reason_codes=(),
        )
        for output, value in ((episode_output, episode), (envelope_output, envelope)):
            if (
                output.profile is not ArtifactProfile.CANONICAL_JSON
                or output.media_type != GYM_TORAX_CANONICAL_MEDIA_TYPE
            ):
                raise ValueError("Gym parameterised output port differs")
            emitter.write(output.output_id, value.canonical_bytes())
        return (
            ReceiptCheck("bounded-canonical-stream-emitted", True, ()),
            ReceiptCheck("exact-issued-identities-preserved", True, ()),
            ReceiptCheck("one-native-record-resident", True, ()),
        )


def default_gym_torax_acquirer(
    *,
    config: GymToraxParameterisedSourceConfig,
    repository_root: Path,
) -> GymToraxFieldMetadataEpisodeAcquirer:
    return lambda request: acquire_gym_torax_episode(
        request,
        extraction_manifest=config.extraction_manifest,
        field_metadata_manifest=config.field_metadata_manifest,
        repository_root=repository_root,
    )


__all__ = [
    "GYM_TORAX_CANONICAL_MEDIA_TYPE",
    "GYM_TORAX_ACQUISITION_TASK_PREFIX",
    "GYM_TORAX_MAXIMUM_REQUEST_BYTES",
    "GYM_TORAX_MAXIMUM_SOURCE_CONFIG_BYTES",
    "GYM_TORAX_PARAMETERISED_PROVIDER_KEY",
    "GYM_TORAX_PARAMETERISED_PROVIDER_VERSION",
    'GymToraxParameterisedAcquisitionBinding',
    'GymToraxParameterisedAcquisitionManifest',
    'GymToraxParameterisedEpisodeTask',
    'GymToraxParameterisedSourceConfig',
    'default_gym_torax_acquirer',
    'materialise_gym_torax_parameterised_request',
    'validate_gym_torax_parameterised_source_config',
    'validate_gym_torax_parameterised_acquisition_manifest',
    'verify_gym_torax_static_readiness',
]
