"Lossless Gym--TORAX metadata-complete native-to-manifest pre-projection translation."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.planning.identification_evidence import (
    IdentificationManifestObservation,
    ObservationActionDeliveryBinding,
    ObservationDisposition,
)
from empirical_lawhood.runtime.response_experiment_ports import NativeEpisodeEnvelope, NativeExecutionRequest

from .action_word import build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataNativeEpisode
from .source import GymToraxParameterisedSourceConfig


_REGISTERED_NATIVE_CATEGORIES = frozenset(
    ("coordinate", "source-profile", "source-scalar", "source-numerics")
)


@dataclass(frozen=True, slots=True)
class GymToraxNativeFieldRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-field-registration'

    registration_id: str
    category: str
    native_field_id: str
    native_unit: str
    dimension_ids: tuple[str, ...]
    native_frame_id: str
    clock_id: str
    field_metadata_id: str | None
    required: bool
    numerical_axis: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.category, field_name="category")
        if self.category not in _REGISTERED_NATIVE_CATEGORIES:
            raise ValueError("Gym field registration has an unsupported category")
        validate_nonempty(self.native_field_id, field_name="native_field_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if not self.dimension_ids or len(set(self.dimension_ids)) != len(self.dimension_ids):
            raise ValueError("Gym field registration requires ordered unique dimensions")
        validate_stable_id(self.native_frame_id, field_name="native_frame_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        is_source = self.category.startswith("source-")
        if is_source != (self.field_metadata_id is not None):
            raise ValueError("Gym source field registration requires metadata identity")
        if self.field_metadata_id is not None:
            validate_stable_id(self.field_metadata_id, field_name="field_metadata_id")
        if self.numerical_axis != (self.category == "source-numerics"):
            raise ValueError("Gym numerical-axis declaration differs from category")

    @property
    def native_key(self) -> tuple[str, str]:
        return self.category, self.native_field_id


@dataclass(frozen=True, slots=True)
class GymToraxObservationContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-observation-context'

    context_id: str
    source_config: ObjectIdentity
    preparation: ObjectIdentity
    numerical_member: ObjectIdentity
    registered_fields: tuple[GymToraxNativeFieldRegistration, ...]
    nested_coordinate_ids: tuple[str, ...]
    denominator_cell_id: str
    chart_id: str
    split_id: str
    role_id: str
    candidate_version_id: str
    qualification_view_id: str
    receiver_id: str
    receiver_clock_id: str
    native_frame_id: str
    payload_member_locator: str

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id, field_name="context_id")
        if self.source_config.object_schema != GymToraxParameterisedSourceConfig.SCHEMA:
            raise ValueError("Gym observation context requires its source config")
        require_sorted_unique_ids(
            self.registered_fields,
            attribute="registration_id",
            field_name="registered_fields",
        )
        if not self.registered_fields or len(
            {value.native_key for value in self.registered_fields}
        ) != len(self.registered_fields):
            raise ValueError("Gym observation registrations are incomplete or repeated")
        require_sorted_unique_strings(
            self.nested_coordinate_ids,
            field_name="nested_coordinate_ids",
            allow_empty=False,
        )
        for name in (
            "denominator_cell_id",
            "chart_id",
            "split_id",
            "role_id",
            "candidate_version_id",
            "qualification_view_id",
            "receiver_id",
            "receiver_clock_id",
            "native_frame_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_relative_locator(self.payload_member_locator)


@dataclass(frozen=True, slots=True)
class GymToraxNativeObservationAxes(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-observation-axes'

    axes_id: str
    episode_id: str
    observation_id: str
    source_disposition: GymToraxSourceDisposition
    operator_source_disposition: GymToraxSourceDisposition
    delivery_disposition: GymToraxDeliveryDisposition
    numerical_disposition: GymToraxNumericalDisposition
    observation_disposition: GymToraxObservationDisposition
    translation_disposition: ObservationDisposition
    termination: bool
    truncation: bool
    state_clocks: tuple[int, ...]
    missing_required_state_clocks: tuple[int, ...]
    last_valid_state_clock: int | None
    episode_reason_codes: tuple[str, ...]
    operator_reason_codes: tuple[str, ...]
    delivery_reason_codes: tuple[str, ...]
    translation_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("axes_id", "episode_id", "observation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.state_clocks != tuple(sorted(set(self.state_clocks))):
            raise ValueError("Gym observation clocks are not ordered and unique")
        if self.missing_required_state_clocks != tuple(
            sorted(set(self.missing_required_state_clocks))
        ):
            raise ValueError("Gym missing clocks are not ordered and unique")
        if self.last_valid_state_clock != (max(self.state_clocks) if self.state_clocks else None):
            raise ValueError("Gym last-valid clock differs from retained clocks")
        for name in (
            "episode_reason_codes",
            "operator_reason_codes",
            "delivery_reason_codes",
            "translation_reason_codes",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class GymToraxObservationTranslation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-observation-translation'

    translation_id: str
    source_config: ObjectIdentity
    native_envelope: ObjectIdentity
    episode: ObjectIdentity
    request: ObjectIdentity
    generic_intent: ObjectIdentity
    observation: IdentificationManifestObservation
    axes: GymToraxNativeObservationAxes
    registered_fields: tuple[GymToraxNativeFieldRegistration, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.translation_id, field_name="translation_id")
        expected = (
            GymToraxParameterisedSourceConfig.SCHEMA,
            NativeEpisodeEnvelope.SCHEMA,
            GymToraxFieldMetadataNativeEpisode.SCHEMA,
            GymToraxFieldMetadataEpisodeRequest.SCHEMA,
            NativeExecutionRequest.SCHEMA,
        )
        observed = tuple(
            value.object_schema
            for value in (
                self.source_config,
                self.native_envelope,
                self.episode,
                self.request,
                self.generic_intent,
            )
        )
        if observed != expected:
            raise ValueError("Gym translation binds another metadata-complete contract")
        if (
            self.observation.observation_id != self.axes.observation_id
            or self.episode.object_id != self.axes.episode_id
        ):
            raise ValueError("Gym translation observation/axes identity differs")
        require_sorted_unique_ids(
            self.registered_fields,
            attribute="registration_id",
            field_name="registered_fields",
        )


class GymToraxObservationTranslationError(ValueError):
    pass


def _delivery_binding(
    *,
    episode: GymToraxFieldMetadataNativeEpisode,
    request: GymToraxFieldMetadataEpisodeRequest,
    observation_id: str,
) -> tuple[ObservationActionDeliveryBinding, tuple[str, ...]]:
    expected = tuple(
        sorted(value.occurrence_id for value in request.schedule.action_word.occurrences)
    )
    deliveries = tuple(value for value in episode.deliveries if value.occurrence_id is not None)
    by_occurrence = {value.occurrence_id: value for value in deliveries}
    reasons: set[str] = set()
    if len(by_occurrence) != len(deliveries):
        reasons.add("ACTION_OCCURRENCE_REPEATED")
    if set(by_occurrence) != set(expected):
        reasons.add("ACTION_OCCURRENCE_ROSTER_INCOMPLETE")

    def events(field_name: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"event.{occurrence_id}.{field_name}"
                for occurrence_id in expected
                if occurrence_id in by_occurrence
                and getattr(by_occurrence[occurrence_id], field_name) is not None
            )
        )

    requested, accepted, applied, realized = (
        events("requested"),
        events("accepted"),
        events("applied"),
        events("realized"),
    )
    if any(len(value) != len(expected) for value in (requested, accepted, applied, realized)):
        reasons.add("ACTION_DELIVERY_INCOMPLETE")
    return (
        ObservationActionDeliveryBinding(
            binding_id=f"delivery-binding.{observation_id}",
            action_word=ObjectIdentity.from_record(
                request.schedule.action_word.word_id,
                request.schedule.action_word,
            ),
            occurrence_ids=expected,
            requested_event_ids=requested,
            accepted_event_ids=accepted,
            applied_event_ids=applied,
            realized_event_ids=realized,
        ),
        tuple(sorted(reasons)),
    )


def _registered_block_reasons(
    episode: GymToraxFieldMetadataNativeEpisode,
    context: GymToraxObservationContext,
) -> tuple[str, ...]:
    blocks = {
        (value.category, value.native_field_id): value
        for value in episode.blocks
        if value.category in _REGISTERED_NATIVE_CATEGORIES
    }
    registrations = {value.native_key: value for value in context.registered_fields}
    reasons: set[str] = set()
    if len(blocks) != sum(
        value.category in _REGISTERED_NATIVE_CATEGORIES for value in episode.blocks
    ):
        reasons.add("NATIVE_FIELD_REPEATED")
    if set(blocks) - set(registrations):
        reasons.add("UNREGISTERED_NATIVE_FIELD")
    for key, registration in registrations.items():
        block = blocks.get(key)
        if block is None:
            if registration.required:
                reasons.add("REQUIRED_NATIVE_FIELD_MISSING")
            continue
        if (
            block.native_unit != registration.native_unit
            or block.native_frame_id != registration.native_frame_id
            or block.dimension_ids != registration.dimension_ids
            or block.field_metadata_id != registration.field_metadata_id
        ):
            reasons.add("NATIVE_FIELD_SCHEMA_DRIFT")
        expected_clocks = (
            episode.state_clocks if registration.dimension_ids[0] == "state-clock" else ()
        )
        if block.clock_values != expected_clocks:
            reasons.add("NATIVE_FIELD_CLOCK_DRIFT")
        if registration.required and block.nonfinite_value_count:
            reasons.add("NONFINITE_REQUIRED_NATIVE_FIELD")
    source_blocks = tuple(value for value in episode.blocks if value.category.startswith("source-"))
    if episode.state_clocks and not source_blocks:
        reasons.add("SOURCE_BLOCKS_MISSING")
    return tuple(sorted(reasons))


def derive_gym_torax_observation_disposition(
    episode: GymToraxFieldMetadataNativeEpisode,
    reasons: tuple[str, ...],
) -> ObservationDisposition:
    if (
        episode.source_disposition is GymToraxSourceDisposition.SOURCE_UNAVAILABLE
        or episode.numerical_disposition is GymToraxNumericalDisposition.UNEVALUABLE
        or episode.observation_disposition is GymToraxObservationDisposition.UNEVALUABLE
    ):
        return ObservationDisposition.UNEVALUABLE
    if (
        episode.numerical_disposition
        in {GymToraxNumericalDisposition.INVALID, GymToraxNumericalDisposition.SOLVER_FAILURE}
        or episode.delivery_disposition
        in {GymToraxDeliveryDisposition.CLIPPED, GymToraxDeliveryDisposition.REJECTED}
        or any(
            value
            in {
                "NATIVE_FIELD_REPEATED",
                "UNREGISTERED_NATIVE_FIELD",
                "NATIVE_FIELD_SCHEMA_DRIFT",
                "NATIVE_FIELD_CLOCK_DRIFT",
                "NONFINITE_REQUIRED_NATIVE_FIELD",
            }
            for value in reasons
        )
    ):
        return ObservationDisposition.TECHNICAL_INVALID
    if (
        episode.observation_disposition is GymToraxObservationDisposition.PARTIAL
        or episode.delivery_disposition is GymToraxDeliveryDisposition.PARTIAL
        or reasons
    ):
        return ObservationDisposition.PARTIAL
    return ObservationDisposition.COMPLETE


def translate_gym_torax_episode_observation(
    *,
    config: GymToraxParameterisedSourceConfig,
    intent: NativeExecutionRequest,
    envelope: NativeEpisodeEnvelope,
    request: GymToraxFieldMetadataEpisodeRequest,
    episode: GymToraxFieldMetadataNativeEpisode,
    context: GymToraxObservationContext,
) -> GymToraxObservationTranslation:
    """Translate native evidence while retaining every independent status axis."""

    intent_identity = ObjectIdentity.from_record(intent.request_id, intent)
    request_identity = ObjectIdentity.from_record(request.request_id, request)
    episode_identity = ObjectIdentity.from_record(episode.episode_id, episode)
    source_config_identity = ObjectIdentity.from_record(config.config_id, config)
    if (
        context.source_config != source_config_identity
        or envelope.request != intent_identity
        or envelope.episode_id != episode.episode_id
        or envelope.delivery_trace != episode_identity
        or envelope.clock_receipts != (episode_identity,)
        or episode.request != request_identity
        or request.request_id != intent.request_id
        or request.schedule != build_gym_torax_native_schedule(request.schedule.action_word)
        or context.preparation
        != ObjectIdentity.from_record(request.preparation.preparation_id, request.preparation)
        or context.numerical_member
        != ObjectIdentity.from_record(request.numerical_member.member_id, request.numerical_member)
    ):
        raise GymToraxObservationTranslationError("Gym metadata-complete translation identity drifted")
    observation_id = f"observation.{episode.episode_id.removeprefix('episode.')}"
    delivery, delivery_reasons = _delivery_binding(
        episode=episode,
        request=request,
        observation_id=observation_id,
    )
    block_reasons = _registered_block_reasons(episode, context)
    reasons = set(delivery_reasons) | set(block_reasons)
    reasons.update(
        value
        for value in episode.reason_codes
        if not value.startswith(("OPERATOR_API_", "OPERATOR_SOURCE_"))
    )
    if episode.source_disposition is GymToraxSourceDisposition.SOURCE_UNAVAILABLE:
        reasons.add("SOURCE_UNAVAILABLE")
    if episode.observation_disposition is not GymToraxObservationDisposition.COMPLETE:
        reasons.add(f"OBSERVATION_{episode.observation_disposition.value}")
    if episode.numerical_disposition is not GymToraxNumericalDisposition.VALID:
        reasons.add(f"NUMERICAL_{episode.numerical_disposition.value}")
    if episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE:
        reasons.add(f"DELIVERY_{episode.delivery_disposition.value}")
    disposition = derive_gym_torax_observation_disposition(
        episode,
        tuple(sorted(reasons)),
    )
    manifest_reasons = (
        () if disposition is ObservationDisposition.COMPLETE else tuple(sorted(reasons))
    )
    observation = IdentificationManifestObservation(
        observation_id=observation_id,
        physical_unit_instance_id=request.preparation.physical_independent_unit_id,
        nested_coordinate_ids=context.nested_coordinate_ids,
        denominator_cell_id=context.denominator_cell_id,
        chart_id=context.chart_id,
        split_id=context.split_id,
        role_id=context.role_id,
        member_id=request.numerical_member.member_id,
        candidate_version_id=context.candidate_version_id,
        qualification_view_id=context.qualification_view_id,
        receiver_id=context.receiver_id,
        receiver_clock_id=context.receiver_clock_id,
        native_frame_id=context.native_frame_id,
        payload_id=envelope.native_episode_artifact.artifact_id,
        payload_member_locator=context.payload_member_locator,
        action_delivery=delivery,
        disposition=disposition,
        reason_codes=manifest_reasons,
    )
    axes = GymToraxNativeObservationAxes(
        axes_id=f"native-axes.{observation_id}",
        episode_id=episode.episode_id,
        observation_id=observation_id,
        source_disposition=episode.source_disposition,
        operator_source_disposition=episode.operator_source.disposition,
        delivery_disposition=episode.delivery_disposition,
        numerical_disposition=episode.numerical_disposition,
        observation_disposition=episode.observation_disposition,
        translation_disposition=disposition,
        termination=episode.termination,
        truncation=episode.truncation,
        state_clocks=episode.state_clocks,
        missing_required_state_clocks=episode.missing_required_state_clocks,
        last_valid_state_clock=episode.last_valid_state_clock,
        episode_reason_codes=episode.reason_codes,
        operator_reason_codes=episode.operator_source.reason_codes,
        delivery_reason_codes=tuple(
            sorted(
                set(delivery_reasons)
                | {reason for value in episode.deliveries for reason in value.reason_codes}
            )
        ),
        translation_reason_codes=tuple(sorted(reasons)),
    )
    return GymToraxObservationTranslation(
        translation_id=f"translation.{observation_id}",
        source_config=source_config_identity,
        native_envelope=ObjectIdentity.from_record(envelope.episode_id, envelope),
        episode=episode_identity,
        request=request_identity,
        generic_intent=intent_identity,
        observation=observation,
        axes=axes,
        registered_fields=context.registered_fields,
    )


__all__ = [
    'derive_gym_torax_observation_disposition',
    'GymToraxNativeFieldRegistration',
    'GymToraxNativeObservationAxes',
    'GymToraxObservationContext',
    "GymToraxObservationTranslationError",
    'GymToraxObservationTranslation',
    'translate_gym_torax_episode_observation',
]
