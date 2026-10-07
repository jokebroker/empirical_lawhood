"Inward effect ports for parameterised measurement through controller use execution.\n\nThe protocols move exact identities and custody references.  Scientific\nqualification, admission, reachability and controller use adjudication remain with their\nexisting owners and are deliberately absent from this module.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane


class SourceReadinessDisposition(StrEnum):
    READY = "READY"
    NOT_INSTALLED = "NOT_INSTALLED"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    IMPLEMENTATION_MISMATCH = "IMPLEMENTATION_MISMATCH"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class SourceReadinessCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-readiness-check'

    check_id: str
    source_pipeline_profile: ObjectIdentity
    provider_key: str
    expected_provider_version: str
    expected_implementation_sha256: str
    disposition: SourceReadinessDisposition
    observed_provider_version: str | None
    observed_implementation_sha256: str | None
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        validate_stable_id(self.provider_key, field_name="provider_key")
        validate_semantic_version(self.expected_provider_version)
        validate_sha256(
            self.expected_implementation_sha256,
            field_name="expected_implementation_sha256",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.observed_provider_version is not None:
            validate_semantic_version(self.observed_provider_version)
        if self.observed_implementation_sha256 is not None:
            validate_sha256(
                self.observed_implementation_sha256,
                field_name="observed_implementation_sha256",
            )
        if self.disposition is SourceReadinessDisposition.READY:
            if (
                self.reason_codes
                or self.observed_provider_version != self.expected_provider_version
                or self.observed_implementation_sha256 != self.expected_implementation_sha256
            ):
                raise ValueError("ready source check does not match its exact expectation")
        elif not self.reason_codes:
            raise ValueError("non-ready source check requires reason codes")
        if self.grants_authority:
            raise ValueError("source readiness cannot grant authority")


class NativeInteractionKind(StrEnum):
    READ_ONLY_ACQUISITION = "READ_ONLY_ACQUISITION"
    INTERACTIVE_EXECUTION = "INTERACTIVE_EXECUTION"


@dataclass(frozen=True, slots=True)
class NativeReceiverContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-receiver-contract'

    receiver_id: str
    quantity_id: str
    native_unit: str
    frame_id: str
    direction_id: str
    clock_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("receiver_id", "quantity_id", "frame_id", "direction_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        require_sorted_unique_strings(self.clock_ids, field_name="clock_ids", allow_empty=False)
        for clock_id in self.clock_ids:
            validate_stable_id(clock_id, field_name="clock_ids")


@dataclass(frozen=True, slots=True)
class NativeClockContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-clock-contract'

    clock_id: str
    native_unit: str
    frame_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.frame_id, field_name="frame_id")


@dataclass(frozen=True, slots=True)
class NativeActionContract(CanonicalRecord):
    """Adapter-owned action semantics and explicitly observable chain links."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-action-contract'

    action_coordinate_id: str
    requested_observable: bool
    accepted_observable: bool
    applied_observable: bool
    realized_observable: bool
    native_hold_token: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.action_coordinate_id, field_name="action_coordinate_id")
        if self.native_hold_token is not None:
            validate_nonempty(self.native_hold_token, field_name="native_hold_token")


@dataclass(frozen=True, slots=True)
class ResponseSubstrateBinding(CanonicalRecord):
    """Open installed-medium binding; adapter factories own all native constants."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-substrate-binding'

    binding_id: str
    medium_id: str
    interaction_kind: NativeInteractionKind
    provider_key: str
    provider_version: str
    provider_implementation_sha256: str
    installed_executable_binding: ObjectIdentity
    source_profile_schema: str
    native_request_schema: str
    native_object_schema: str | None
    native_episode_schema: str | None
    native_projection_schema: str
    native_config: ObjectIdentity
    receivers: tuple[NativeReceiverContract, ...]
    clocks: tuple[NativeClockContract, ...]
    action_contract: NativeActionContract | None
    external_source_contacted: bool
    scientific_verdict_constructed: bool
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("binding_id", "medium_id", "provider_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.provider_version)
        validate_sha256(
            self.provider_implementation_sha256,
            field_name="provider_implementation_sha256",
        )
        for schema in (
            self.source_profile_schema,
            self.native_request_schema,
            self.native_projection_schema,
        ):
            validate_schema(schema)
        require_sorted_unique_ids(self.receivers, attribute="receiver_id", field_name="receivers")
        require_sorted_unique_ids(self.clocks, attribute="clock_id", field_name="clocks")
        if not self.receivers or not self.clocks:
            raise ValueError("native binding requires bounded receivers and clocks")
        clock_ids = {value.clock_id for value in self.clocks}
        if any(set(value.clock_ids) - clock_ids for value in self.receivers):
            raise ValueError("native receiver references an undeclared clock")
        if self.interaction_kind is NativeInteractionKind.READ_ONLY_ACQUISITION:
            if self.native_object_schema is None or self.native_episode_schema is not None:
                raise ValueError("read-only binding requires an object schema only")
            if self.action_contract is not None:
                raise ValueError("read-only binding cannot fabricate an action contract")
            validate_schema(self.native_object_schema)
        else:
            if self.native_episode_schema is None or self.native_object_schema is not None:
                raise ValueError("interactive binding requires an episode schema only")
            if self.action_contract is None:
                raise ValueError("interactive binding requires adapter-owned action semantics")
            validate_schema(self.native_episode_schema)
        if (
            self.external_source_contacted
            or self.scientific_verdict_constructed
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("native binding exceeds compile-only authority")


@dataclass(frozen=True, slots=True)
class NativeAcquisitionRequest(CanonicalRecord):
    """Read-only source intent; it is neither an action nor an execution grant."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-acquisition-request'

    request_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    stage_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    preparation_coordinate_id: str
    preparation_instance_id: str
    preparation_sha256: str
    adapter_realization_id: str | None
    native_request_schema: str
    native_config: ObjectIdentity
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    causal_cutoff_id: str
    grants_execution_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "stage_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
            "preparation_coordinate_id",
            "preparation_instance_id",
            "causal_cutoff_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.adapter_realization_id is not None:
            validate_stable_id(self.adapter_realization_id, field_name="adapter_realization_id")
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")
        validate_schema(self.native_request_schema)
        for name, values in (("receiver_ids", self.receiver_ids), ("clock_ids", self.clock_ids)):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.grants_execution_authority:
            raise ValueError("native acquisition request cannot grant execution authority")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native request materialisation must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class NativeExecutionRequest(NativeAcquisitionRequest):
    """Interactive native intent with an adapter-owned action contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-execution-request'

    nested_view_id: str
    action_word_id: str
    action_contract: NativeActionContract

    def __post_init__(self) -> None:
        super(NativeExecutionRequest, self).__post_init__()
        validate_stable_id(self.nested_view_id, field_name="nested_view_id")
        validate_stable_id(self.action_word_id, field_name="action_word_id")


@dataclass(frozen=True, slots=True)
class NativeObjectEnvelope(CanonicalRecord):
    """Immediate custody for one read-only native object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-object-envelope'

    object_envelope_id: str
    request: ObjectIdentity
    native_object_artifact: ArtifactIdentity
    delivery_trace: ObjectIdentity
    clock_receipts: tuple[ObjectIdentity, ...]
    sealed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.object_envelope_id, field_name="object_envelope_id")
        if self.request.object_schema != NativeAcquisitionRequest.SCHEMA:
            raise ValueError("native object binds another request schema")
        if not self.clock_receipts:
            raise ValueError("native object requires clock custody")
        if not self.sealed or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("native object must remain sealed before reveal")


@dataclass(frozen=True, slots=True)
class NativeEpisodeEnvelope(CanonicalRecord):
    """Sealed interactive episode custody without a scientific verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-episode-envelope'

    episode_id: str
    request: ObjectIdentity
    native_episode_artifact: ArtifactIdentity
    delivery_trace: ObjectIdentity
    clock_receipts: tuple[ObjectIdentity, ...]
    sealed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        if self.request.object_schema != NativeExecutionRequest.SCHEMA:
            raise ValueError("native episode binds another request schema")
        if not self.clock_receipts:
            raise ValueError("native episode requires clock custody")
        if not self.sealed or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("native episode must remain sealed before reveal")


@dataclass(frozen=True, slots=True)
class ProjectionTranslationReceipt(CanonicalRecord):
    "Custody/semantic translation receipt; no law or controller use status is asserted."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/projection-translation-receipt'

    receipt_id: str
    native_envelope: ObjectIdentity
    projection_contract: ObjectIdentity
    projected_artifact: ArtifactIdentity
    source_unit_id: str
    destination_unit_id: str
    source_frame_id: str
    destination_frame_id: str
    receiver_id: str
    direction_id: str
    causal_cutoff_id: str
    translation_complete: bool
    reason_codes: tuple[str, ...]
    scientific_verdict_asserted: bool

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "source_unit_id",
            "destination_unit_id",
            "source_frame_id",
            "destination_frame_id",
            "receiver_id",
            "direction_id",
            "causal_cutoff_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.translation_complete == bool(self.reason_codes):
            raise ValueError("projection translation completion/reasons are inconsistent")
        if self.scientific_verdict_asserted:
            raise ValueError("projection translator cannot assert a scientific verdict")


class SourceReadinessVerifier(Protocol):
    def verify_source(self, *, source_profile: ObjectIdentity) -> SourceReadinessCheck: ...


class NativeAcquisitionRequestMaterialiser(Protocol):
    def materialise_acquisition_request(
        self,
        *,
        request: NativeAcquisitionRequest,
    ) -> NativeAcquisitionRequest: ...


class NativeExecutionRequestMaterialiser(Protocol):
    def materialise_execution_request(
        self,
        *,
        request: NativeExecutionRequest,
    ) -> NativeExecutionRequest: ...


class NativeObjectProvider(Protocol):
    def acquire_object(
        self,
        *,
        request: NativeAcquisitionRequest,
    ) -> NativeObjectEnvelope: ...


class NativeEpisodeProvider(Protocol):
    def acquire_episode(
        self,
        *,
        request: NativeExecutionRequest,
    ) -> NativeEpisodeEnvelope: ...


class ProjectionTranslator(Protocol):
    def translate_projection(
        self,
        *,
        native_envelope: NativeObjectEnvelope | NativeEpisodeEnvelope,
        projection_contract: ObjectIdentity,
    ) -> ProjectionTranslationReceipt: ...


class StageArtifactPublisher(Protocol):
    def publish_stage_artifact(
        self,
        *,
        stage_id: str,
        payload: bytes,
        payload_schema: str,
    ) -> ArtifactIdentity: ...


__all__ = [
    "CandidatePayloadPlane",
    "NativeAcquisitionRequestMaterialiser",
    'NativeAcquisitionRequest',
    'NativeActionContract',
    'NativeClockContract',
    'NativeEpisodeEnvelope',
    "NativeEpisodeProvider",
    "NativeExecutionRequestMaterialiser",
    'NativeExecutionRequest',
    'NativeInteractionKind',
    'NativeObjectEnvelope',
    "NativeObjectProvider",
    'NativeReceiverContract',
    'ResponseSubstrateBinding',
    'ProjectionTranslationReceipt',
    "ProjectionTranslator",
    'SourceReadinessCheck',
    'SourceReadinessDisposition',
    "SourceReadinessVerifier",
    "StageArtifactPublisher",
]
