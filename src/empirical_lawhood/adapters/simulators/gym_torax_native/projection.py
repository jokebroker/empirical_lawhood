"""Pure per-view projection over one already-materialised Gym--TORAX episode."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.identification_evidence import ObservationDisposition
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .observation import derive_gym_torax_observation_disposition
from .source import GYM_TORAX_CANONICAL_MEDIA_TYPE, GymToraxParameterisedSourceConfig


GYM_TORAX_PARAMETERISED_PROJECTION_KEY = 'gym-torax-native.parameterised-view-projection'
GYM_TORAX_PARAMETERISED_PROJECTION_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_INPUT_BYTES = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class GymToraxProjectionView(CanonicalRecord):
    """Exact neutral view and native-episode identities for one pure projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-projection-view'

    view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    native_preparation_id: str
    action_word_id: str
    numerical_member_id: str
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
            "native_preparation_id",
            "action_word_id",
            "numerical_member_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.receiver_ids,
            field_name="receiver_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.clock_ids,
            field_name="clock_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-parameterised-projection-config'

    config_id: str
    parameterised_source_config: ObjectIdentity
    projection_task_prefix: str
    views: tuple[GymToraxProjectionView, ...]
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.projection_task_prefix, field_name="projection_task_prefix")
        if (
            self.parameterised_source_config.object_schema
            != GymToraxParameterisedSourceConfig.SCHEMA
        ):
            raise ValueError("Gym projection config binds another source config")
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        if not self.views:
            raise ValueError("Gym projection config requires scientific views")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Gym projection config cannot grant authority or reveal outcomes")

    def view(self, view_id: str) -> GymToraxProjectionView:
        try:
            return next(value for value in self.views if value.view_id == view_id)
        except StopIteration as error:
            raise KeyError(view_id) from error


@dataclass(frozen=True, slots=True)
class GymToraxNativeProjection(CanonicalRecord):
    """View-specific native projection that constructs no response-law verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-projection'

    projection_id: str
    view: ObjectIdentity
    native_episode: ObjectIdentity
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    source_disposition: GymToraxSourceDisposition
    delivery_disposition: GymToraxDeliveryDisposition
    numerical_disposition: GymToraxNumericalDisposition
    observation_disposition: GymToraxObservationDisposition
    projection_disposition: ObservationDisposition
    native_reason_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    requested_accepted_applied_realized_preserved: bool
    native_units_frames_directions_preserved: bool
    causal_cutoff_preserved: bool
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        if self.view.object_schema != GymToraxProjectionView.SCHEMA:
            raise ValueError("Gym native projection binds another scientific view")
        if self.native_episode.object_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA:
            raise ValueError("Gym native projection binds another episode")
        require_sorted_unique_strings(
            self.receiver_ids,
            field_name="receiver_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.clock_ids, field_name="clock_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.native_reason_codes,
            field_name="native_reason_codes",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (self.projection_disposition is ObservationDisposition.COMPLETE) == bool(
            self.reason_codes
        ):
            raise ValueError("Gym projection disposition and reasons disagree")
        if (
            not self.requested_accepted_applied_realized_preserved
            or not self.native_units_frames_directions_preserved
            or not self.causal_cutoff_preserved
            or self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("Gym projection loses native semantics or exceeds authority")


def validate_gym_torax_projection_config(
    *,
    config: GymToraxParameterisedProjectionConfig,
    extension_set: ResponseExperimentExtensionSet,
    substrate_binding: ResponseSubstrateBinding,
    source_config: GymToraxParameterisedSourceConfig,
) -> None:
    """Join every view to its issued acquisition and native receiver contracts."""

    if config.parameterised_source_config != ObjectIdentity.from_record(
        source_config.config_id,
        source_config,
    ):
        raise ValueError("Gym projection config binds another source realization")
    if source_config.experiment_extension_set != ObjectIdentity.from_record(
        extension_set.extension_set_id,
        extension_set,
    ) or source_config.substrate_binding != ObjectIdentity.from_record(
        substrate_binding.binding_id,
        substrate_binding,
    ):
        raise ValueError("Gym projection parents differ from source parents")
    authored_views = {value.view_id: value for value in extension_set.identification_config.nested_views}
    projection_views = {value.view_id: value for value in config.views}
    if set(authored_views) != set(projection_views):
        raise ValueError("Gym projection view roster differs from the extension")
    receiver_ids = tuple(sorted(value.receiver_id for value in substrate_binding.receivers))
    clock_ids = tuple(sorted(value.clock_id for value in substrate_binding.clocks))
    for view_id, authored in authored_views.items():
        projected = projection_views[view_id]
        acquisition = source_config.binding(authored.acquisition_group_id)
        if (
            projected.acquisition_group_id != authored.acquisition_group_id
            or projected.physical_independent_unit_id != authored.physical_independent_unit_id
            or projected.native_preparation_id != acquisition.preparation.preparation_id
            or projected.action_word_id != authored.action_word_id
            or projected.numerical_member_id != authored.numerical_member_id
            or projected.receiver_ids != receiver_ids
            or projected.clock_ids != clock_ids
        ):
            raise ValueError("Gym projection view changes native group/view lineage")


def _projection_reasons(episode: GymToraxFieldMetadataNativeEpisode) -> tuple[str, ...]:
    reasons = {
        value
        for value in episode.reason_codes
        if not value.startswith(("OPERATOR_API_", "OPERATOR_SOURCE_"))
    }
    if episode.source_disposition is not GymToraxSourceDisposition.AVAILABLE:
        reasons.add(f"SOURCE_{episode.source_disposition.value}")
    if episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE:
        reasons.add(f"DELIVERY_{episode.delivery_disposition.value}")
    if episode.numerical_disposition is not GymToraxNumericalDisposition.VALID:
        reasons.add(f"NUMERICAL_{episode.numerical_disposition.value}")
    if episode.observation_disposition is not GymToraxObservationDisposition.COMPLETE:
        reasons.add(f"OBSERVATION_{episode.observation_disposition.value}")
    if any(value.nonfinite_value_count for value in episode.blocks):
        reasons.add("NONFINITE_NATIVE_FIELD")
    return tuple(sorted(reasons))


class _GymToraxProjectionRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: GymToraxParameterisedProjectionConfig,
    ) -> None:
        self.manifest = manifest
        self._config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _decode_inputs(
            context,
            (
                GymToraxParameterisedProjectionConfig,
                GymToraxFieldMetadataNativeEpisode,
                LinkedCampaignStageEnvelope,
            ),
        )
        config = cast(
            GymToraxParameterisedProjectionConfig,
            records[GymToraxParameterisedProjectionConfig.SCHEMA],
        )
        episode = cast(GymToraxFieldMetadataNativeEpisode, records[GymToraxFieldMetadataNativeEpisode.SCHEMA])
        upstream = cast(
            LinkedCampaignStageEnvelope,
            records[LinkedCampaignStageEnvelope.SCHEMA],
        )
        if config != self._config:
            raise ValueError("Gym projection task received a substituted config")
        prefix = f"{config.projection_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("Gym projection task identity differs")
        view = config.view(context.task_id[len(prefix) :])
        episode_identity = ObjectIdentity.from_record(episode.episode_id, episode)
        if (
            upstream.role is not LinkedCampaignStageRole.SOURCE_MATERIALIZATION
            or upstream.disposition is not LinkedCampaignDisposition.SUPPORTED
            or upstream.scientific_product != episode_identity
            or episode.preparation.object_id != view.native_preparation_id
            or episode.numerical_member.object_id != view.numerical_member_id
            or episode.action_word.object_id != view.action_word_id
        ):
            raise ValueError("Gym projection input changes acquisition/view lineage")
        reasons = _projection_reasons(episode)
        projection = GymToraxNativeProjection(
            projection_id=f"projection.gym-torax.{view.view_id}",
            view=ObjectIdentity.from_record(view.view_id, view),
            native_episode=episode_identity,
            receiver_ids=view.receiver_ids,
            clock_ids=view.clock_ids,
            source_disposition=episode.source_disposition,
            delivery_disposition=episode.delivery_disposition,
            numerical_disposition=episode.numerical_disposition,
            observation_disposition=episode.observation_disposition,
            projection_disposition=derive_gym_torax_observation_disposition(
                episode,
                reasons,
            ),
            native_reason_codes=episode.reason_codes,
            reason_codes=reasons,
            requested_accepted_applied_realized_preserved=True,
            native_units_frames_directions_preserved=True,
            causal_cutoff_preserved=True,
            scientific_verdict_constructed=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        envelope = LinkedCampaignStageEnvelope(
            envelope_id=f"stage-envelope.{context.task_id}",
            role=LinkedCampaignStageRole.EVIDENCE_PROJECTION,
            disposition=(
                LinkedCampaignDisposition.SUPPORTED
                if projection.projection_disposition is ObservationDisposition.COMPLETE
                else LinkedCampaignDisposition.UNEVALUABLE
            ),
            scientific_product=ObjectIdentity.from_record(projection.projection_id, projection),
            reason_codes=projection.reason_codes,
        )
        outputs = {value.SCHEMA: value for value in (projection, envelope)}
        if set(outputs) != {value.payload_schema for value in context.output_ports}:
            raise ValueError("Gym projection output contract differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(port.output_id, outputs[port.payload_schema].canonical_bytes())
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("one-view-no-reexecution", True, ()),
                ReceiptCheck("native-status-axes-preserved", True, ()),
                ReceiptCheck("no-adapter-scientific-verdict", True, ()),
            ),
        )


def _decode_inputs(
    context: TaskContext,
    record_types: tuple[type[CanonicalRecord], ...],
) -> dict[str, CanonicalRecord]:
    by_schema = {value.SCHEMA: value for value in record_types}
    decoded: dict[str, CanonicalRecord] = {}
    for port in context.input_ports:
        record_type = by_schema.get(port.payload_schema)
        try:
            if record_type is None:
                continue
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes or port.read(1):
                raise ValueError("Gym projection input size differs")
            value = decode_canonical_bytes(
                payload,
                record_type,
                maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
            )
            if value.SCHEMA in decoded:
                raise ValueError("Gym projection received duplicate typed input")
            decoded[value.SCHEMA] = value
        finally:
            port.close()
    if set(decoded) != set(by_schema):
        raise ValueError("Gym projection input roster differs")
    return decoded


class GymToraxParameterisedProjectionProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: GymToraxParameterisedProjectionConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("Gym projection manifest differs from installed registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.config = config
        self.capability_count = 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Gym projection runner inputs differ")
        return (
            cast(
                TaskRunner,
                _GymToraxProjectionRunner(manifest=self.manifest, config=self.config),
            ),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Gym projection plan/source inputs differ")
        tasks = tuple(
            value
            for value in plan.tasks
            if value.capability.capability_key == self.manifest.capability_key
        )
        expected_task_ids = {
            f"{self.config.projection_task_prefix}.{value.view_id}" for value in self.config.views
        }
        if {value.task_id for value in tasks} != expected_task_ids:
            raise ValueError("Gym projection task roster differs from config")
        specs = {
            value.logical_artifact_id: value
            for task in tasks
            for value in task.external_inputs
            if value.expected_payload_schema == self.config.SCHEMA
        }
        if (
            len(specs) != 1
            or next(iter(specs.values())).expected_content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("Gym projection tasks lack the exact shared config")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return (
            ExternalInputPayload.from_bytes(
                logical_artifact_id=next(iter(specs)),
                payload_schema=self.config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=GYM_TORAX_CANONICAL_MEDIA_TYPE,
                payload=self.config.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.fingerprint(),
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Gym projection semantic registry differs")
        del execution_plan
        types = {
            GymToraxNativeProjection.SCHEMA: GymToraxNativeProjection,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(types[schema]))),
            )
            for schema in self.manifest.output_schema_ids
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("Gym projection adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "GYM_TORAX_PARAMETERISED_PROJECTION_KEY",
    "GYM_TORAX_PARAMETERISED_PROJECTION_VERSION",
    'GymToraxNativeProjection',
    'GymToraxParameterisedProjectionConfig',
    'GymToraxParameterisedProjectionProvider',
    'GymToraxProjectionView',
    'validate_gym_torax_projection_config',
]
