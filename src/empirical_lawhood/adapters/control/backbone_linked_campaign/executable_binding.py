"""Executable condition-coordinator binding for linked campaigns."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import cast

from empirical_lawhood.adapters.control.controller_authoring import AdmissionControllerAuthor, AdmissionControllerAuthoringRequest
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.planning.response_experiment import ResponseQualificationConfig, ResponseExperimentExtensionSet, AdmissionExperimentConfig, ProspectiveUseExperimentConfig
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.planning.native_source import NativeSourceProfile, NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeCrossfitConfig, NativeCrossfitExperiment
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignProvider, expand_linked_study_acquisition_views
from empirical_lawhood.runtime.plans import ProtocolStepTemplate, ProtocolTemplate
from empirical_lawhood.runtime.response_experiment import build_response_experiment_decoder_registrations, compile_response_experiment
from empirical_lawhood.runtime.observation_order import compile_observation_order_experiment
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration

from .extension_bundle import (
    EVIDENCE_DERIVED_PROGRAMME_AUTHOR,
    LINKED_CAMPAIGN_PROVIDER,
    OBSERVATION_ORDER_PLAN_COMPILER,
    RESPONSE_EXPERIMENT_CONFIG_DECODERS,
    RESPONSE_SUBSTRATE_BINDING_DECODER,
    PARAMETERISED_RESPONSE_PLAN_COMPILER,
    _RUNTIME_IMPLEMENTATION_SHA256,
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER,
    NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES,
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS,
    NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS,
    NATIVE_CROSSFIT_PLAN_COMPILER,
    NATIVE_CROSSFIT_RECORD_TYPES,
    NATIVE_CROSSFIT_DECODERS,
    NATIVE_CROSSFIT_DECODER_REGISTRATIONS,
)


EVIDENCE_DERIVED_PROGRAMME_AUTHOR_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.evidence-derived-programme-author",
    capability_key=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.component_key,
    capability_version=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.component_version,
    capability_implementation_sha256=(EVIDENCE_DERIVED_PROGRAMME_AUTHOR.implementation_sha256),
    role=ExecutableBindingRole.PROGRAMME_AUTHOR,
    provider_key=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.component_key,
    provider_version=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.component_version,
    provider_implementation_sha256=(EVIDENCE_DERIVED_PROGRAMME_AUTHOR.implementation_sha256),
    capability_backed=False,
    discovery_components=(EVIDENCE_DERIVED_PROGRAMME_AUTHOR,),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            record_schema=AdmissionControllerAuthoringRequest.SCHEMA,
            record_version=AdmissionControllerAuthoringRequest.VERSION,
        ),
    ),
    accepted_config_types=(),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.input_schema_ids,
    output_schema_ids=EVIDENCE_DERIVED_PROGRAMME_AUTHOR.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=False,
)


LINKED_CAMPAIGN_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.linked-campaign-coordinator",
    capability_key=LINKED_CAMPAIGN_PROVIDER.component_key,
    capability_version=LINKED_CAMPAIGN_PROVIDER.component_version,
    capability_implementation_sha256=LINKED_CAMPAIGN_PROVIDER.implementation_sha256,
    role=ExecutableBindingRole.LINKED_CAMPAIGN_COORDINATOR,
    provider_key=LINKED_CAMPAIGN_PROVIDER.component_key,
    provider_version=LINKED_CAMPAIGN_PROVIDER.component_version,
    provider_implementation_sha256=LINKED_CAMPAIGN_PROVIDER.implementation_sha256,
    capability_backed=False,
    discovery_components=(LINKED_CAMPAIGN_PROVIDER,),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            record_schema=LinkedCampaignProfile.SCHEMA,
            record_version=LinkedCampaignProfile.VERSION,
        ),
    ),
    accepted_config_types=(),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=LINKED_CAMPAIGN_PROVIDER.input_schema_ids,
    output_schema_ids=LINKED_CAMPAIGN_PROVIDER.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
)


_RESPONSE_EXPERIMENT_DECODER_REGISTRATIONS = build_response_experiment_decoder_registrations(
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)
RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION = StudyExtensionDecoderRegistration(
    registration_id="decoder-registration.response-substrate-binding",
    decoder_key=RESPONSE_SUBSTRATE_BINDING_DECODER.component_key,
    decoder_version=RESPONSE_SUBSTRATE_BINDING_DECODER.component_version,
    payload_schema=ResponseSubstrateBinding.SCHEMA,
    payload_version=ResponseSubstrateBinding.VERSION,
    config_sha256=hashlib.sha256(
        canonical_json_bytes(
            {
                "decoder_key": RESPONSE_SUBSTRATE_BINDING_DECODER.component_key,
                "decoder_version": RESPONSE_SUBSTRATE_BINDING_DECODER.component_version,
                "mode": "exact-canonical-record",
                "payload_schema": ResponseSubstrateBinding.SCHEMA,
            }
        )
    ).hexdigest(),
    implementation_sha256=RESPONSE_SUBSTRATE_BINDING_DECODER.implementation_sha256,
    maximum_payload_bytes=4 * 1024 * 1024,
)
_RESPONSE_EXPERIMENT_CONFIG_TYPES: tuple[type[CanonicalRecord], ...] = (
    ResponseQualificationConfig,
    ResponseExperimentExtensionSet,
    ResponseSubstrateBinding,
    AdmissionExperimentConfig,
    ProspectiveUseExperimentConfig,
)
PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.parameterised-response-plan-compiler",
    capability_key=PARAMETERISED_RESPONSE_PLAN_COMPILER.component_key,
    capability_version=PARAMETERISED_RESPONSE_PLAN_COMPILER.component_version,
    capability_implementation_sha256=(PARAMETERISED_RESPONSE_PLAN_COMPILER.implementation_sha256),
    role=ExecutableBindingRole.PROFILE_COMPILER,
    provider_key=PARAMETERISED_RESPONSE_PLAN_COMPILER.component_key,
    provider_version=PARAMETERISED_RESPONSE_PLAN_COMPILER.component_version,
    provider_implementation_sha256=(PARAMETERISED_RESPONSE_PLAN_COMPILER.implementation_sha256),
    capability_backed=False,
    discovery_components=tuple(
        sorted(
            (
                *RESPONSE_EXPERIMENT_CONFIG_DECODERS,
                RESPONSE_SUBSTRATE_BINDING_DECODER,
                PARAMETERISED_RESPONSE_PLAN_COMPILER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    EvidenceProfileSelection.SCHEMA,
                    EvidenceProfileSelection.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    LinkedCampaignProfile.SCHEMA,
                    LinkedCampaignProfile.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    SourcePipelineProfile.SCHEMA,
                    SourcePipelineProfile.VERSION,
                ),
            ),
            key=lambda value: value.type_id,
        )
    ),
    accepted_config_types=tuple(
        ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
        for value in sorted(_RESPONSE_EXPERIMENT_CONFIG_TYPES, key=lambda value: value.SCHEMA)
    ),
    required_issued_payload_schemas=tuple(
        sorted(
            (
                ResponseQualificationConfig.SCHEMA,
                ResponseExperimentExtensionSet.SCHEMA,
            )
        )
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (*RESPONSE_EXPERIMENT_CONFIG_DECODERS, RESPONSE_SUBSTRATE_BINDING_DECODER)
            ),
            key=lambda value: value.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *_RESPONSE_EXPERIMENT_DECODER_REGISTRATIONS,
                RESPONSE_SUBSTRATE_BINDING_DECODER_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    input_schema_ids=PARAMETERISED_RESPONSE_PLAN_COMPILER.input_schema_ids,
    output_schema_ids=PARAMETERISED_RESPONSE_PLAN_COMPILER.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
    required_authenticated_record_schemas=tuple(
        sorted(
            (
                EvidenceProfileSelection.SCHEMA,
                LinkedCampaignProfile.SCHEMA,
                SourcePipelineProfile.SCHEMA,
            )
        )
    ),
)

NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING = replace(
    PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
    binding_id="binding.native-response-law-qualification-plan-compiler",
    capability_key=NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.component_key,
    capability_implementation_sha256=NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.implementation_sha256,
    provider_key=NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.component_key,
    provider_implementation_sha256=NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.implementation_sha256,
    discovery_components=tuple(
        sorted(
            (*NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS, NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER),
            key=lambda v: v.registration_id,
        )
    ),
    accepted_profile_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(t.SCHEMA, t.VERSION)
                for t in (
                    EvidenceProfileSelection,
                    CampaignSpec,
                    NativeSourceProfile,
                )
            ),
            key=lambda v: v.type_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(t.SCHEMA, t.VERSION)
                for t in (
                    NativeLawQualificationConfig,
                    NativeLawQualificationExperiment,
                    ResponseSubstrateBinding,
                )
            ),
            key=lambda v: v.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(sorted(t.SCHEMA for t in NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES)),
    codec_registration_identities=tuple(
        sorted(
            (ObjectIdentity.from_record(d.registration_id, d) for d in NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS),
            key=lambda v: v.object_id,
        )
    ),
    issued_decoder_registrations=NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS,
    input_schema_ids=NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.input_schema_ids,
    required_authenticated_record_schemas=tuple(
        sorted(
            (EvidenceProfileSelection.SCHEMA, CampaignSpec.SCHEMA),
        )
    ),
)

_CROSSFIT_DECODERS = tuple(
    sorted(
        (
            *NATIVE_CROSSFIT_DECODERS,
            *(d for d in NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS if d.input_schema_ids == (NativeSourceProfile.SCHEMA,)),
        ),
        key=lambda d: d.registration_id,
    )
)
NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING = replace(
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
    binding_id="binding.native-response-law-qualification-crossfit-plan-compiler",
    capability_key=NATIVE_CROSSFIT_PLAN_COMPILER.component_key,
    capability_implementation_sha256=NATIVE_CROSSFIT_PLAN_COMPILER.implementation_sha256,
    provider_key=NATIVE_CROSSFIT_PLAN_COMPILER.component_key,
    provider_implementation_sha256=NATIVE_CROSSFIT_PLAN_COMPILER.implementation_sha256,
    discovery_components=tuple(
        sorted(
            (*_CROSSFIT_DECODERS, NATIVE_CROSSFIT_PLAN_COMPILER),
            key=lambda d: d.registration_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (ExecutableRecordTypeBinding(t.SCHEMA, t.VERSION)
             for t in (*NATIVE_CROSSFIT_RECORD_TYPES, ResponseSubstrateBinding)),
            key=lambda t: t.type_id,
        )
    ),
    required_issued_payload_schemas=NATIVE_CROSSFIT_PLAN_COMPILER.input_schema_ids,
    codec_registration_identities=tuple(
        ObjectIdentity.from_record(d.registration_id, d) for d in _CROSSFIT_DECODERS
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *NATIVE_CROSSFIT_DECODER_REGISTRATIONS,
                *(d for d in NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS
                  if d.payload_schema == NativeSourceProfile.SCHEMA),
            ),
            key=lambda d: d.registration_id,
        )
    ),
    input_schema_ids=NATIVE_CROSSFIT_PLAN_COMPILER.input_schema_ids,
)

OBSERVATION_ORDER_PLAN_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.observation-order-plan-compiler",
    capability_key=OBSERVATION_ORDER_PLAN_COMPILER.component_key,
    capability_version=OBSERVATION_ORDER_PLAN_COMPILER.component_version,
    capability_implementation_sha256=OBSERVATION_ORDER_PLAN_COMPILER.implementation_sha256,
    role=ExecutableBindingRole.PROFILE_COMPILER,
    provider_key=OBSERVATION_ORDER_PLAN_COMPILER.component_key,
    provider_version=OBSERVATION_ORDER_PLAN_COMPILER.component_version,
    provider_implementation_sha256=OBSERVATION_ORDER_PLAN_COMPILER.implementation_sha256,
    capability_backed=False,
    discovery_components=(OBSERVATION_ORDER_PLAN_COMPILER,),
    accepted_profile_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    EvidenceProfileSelection.SCHEMA,
                    EvidenceProfileSelection.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    SourcePipelineProfile.SCHEMA,
                    SourcePipelineProfile.VERSION,
                ),
            ),
            key=lambda value: value.type_id,
        )
    ),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            ObservationOrderExperimentExtension.SCHEMA,
            ObservationOrderExperimentExtension.VERSION,
        ),
    ),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=OBSERVATION_ORDER_PLAN_COMPILER.input_schema_ids,
    output_schema_ids=OBSERVATION_ORDER_PLAN_COMPILER.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
)


@dataclass(frozen=True, slots=True)
class LinkedCampaignCoordinatorFactory:
    """Build owner/condition delegation only; this is never a task runner."""

    binding: ExecutableCapabilityBinding = LINKED_CAMPAIGN_EXECUTABLE_BINDING

    def build_coordinator(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> LinkedCampaignProvider:
        if platform_ports:
            raise ValueError("linked-campaign coordinator accepts no platform ports")
        if len(records) != 1 or not isinstance(records[0], LinkedCampaignProfile):
            raise ValueError("linked-campaign coordinator requires one exact profile")
        return LinkedCampaignProvider(records[0])


@dataclass(frozen=True, slots=True)
class EvidenceDerivedStudyAuthorFactory:
    """Construct the existing evidence-rederiving author, never a runner."""

    binding: ExecutableCapabilityBinding = EVIDENCE_DERIVED_PROGRAMME_AUTHOR_EXECUTABLE_BINDING

    def build_study_author(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> AdmissionControllerAuthor:
        if records or platform_ports:
            raise ValueError("programme-author construction accepts no instance inputs")
        return AdmissionControllerAuthor()


@dataclass(frozen=True, slots=True)
class ResponseExperimentPlanCompilerFactory:
    """Reconstruct and delegate to the sole parameterised plan compiler."""

    binding: ExecutableCapabilityBinding = PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING

    def _root_types(
        self,
    ) -> tuple[
        type[SourcePipelineProfile] | type[NativeSourceProfile],
        type[LinkedCampaignProfile] | type[CampaignSpec],
        type[ResponseQualificationConfig],
        type[ResponseExperimentExtensionSet],
    ]:
        if self.binding == NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING:
            return (
                NativeSourceProfile,
                CampaignSpec,
                NativeCrossfitConfig,
                NativeCrossfitExperiment,
            )
        if self.binding == NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING:
            return (
                NativeSourceProfile,
                CampaignSpec,
                NativeLawQualificationConfig,
                NativeLawQualificationExperiment,
            )
        if self.binding != PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING:
            raise ValueError("parameterised compiler has an unregistered strict-root binding")
        return (
            SourcePipelineProfile,
            LinkedCampaignProfile,
            ResponseQualificationConfig,
            ResponseExperimentExtensionSet,
        )

    def expand_linked_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
        source_task_prefix: str,
        projection_task_prefix: str,
        source_step_template: ProtocolStepTemplate,
        projection_step_template: ProtocolStepTemplate,
        source_config_by_group: dict[str, CapabilityConfigRef] | None = None,
        projection_config_by_view: dict[str, CapabilityConfigRef] | None = None,
        acquisition_aggregate_step_template: ProtocolStepTemplate | None = None,
        acquisition_aggregate_task_id: str | None = None,
    ) -> ProtocolTemplate:
        """Apply issued parameterisation to adapter-owned production steps.

        This is the installed candidate-authoring seam: the same exact record
        roster accepted by ``build_compiler`` supplies fan-out cardinality,
        while adapters retain complete ownership of their native source and
        projection contracts.  It performs no source access or provider
        construction.
        """

        by_schema = {value.SCHEMA: value for value in records}
        if len(by_schema) != len(records):
            raise ValueError("parameterised protocol expansion received duplicate schemas")
        _, _, qualification_type, extension_type = self._root_types()
        extension = by_schema.get(extension_type.SCHEMA)
        qualification_config = by_schema.get(qualification_type.SCHEMA)
        if not isinstance(extension, ResponseExperimentExtensionSet) or not isinstance(
            qualification_config,
            ResponseQualificationConfig,
        ):
            raise ValueError("parameterised protocol expansion lacks its exact config roots")
        if extension.identification_config != qualification_config:
            raise ValueError("parameterised protocol expansion changes its nested measurement through local law config")
        return expand_linked_study_acquisition_views(
            template=template,
            identification_config=qualification_config,
            source_task_prefix=source_task_prefix,
            projection_task_prefix=projection_task_prefix,
            source_step_template=source_step_template,
            projection_step_template=projection_step_template,
            source_config_by_group=source_config_by_group,
            projection_config_by_view=projection_config_by_view,
            acquisition_aggregate_step_template=acquisition_aggregate_step_template,
            acquisition_aggregate_task_id=acquisition_aggregate_task_id,
        )

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object:
        if platform_ports:
            raise ValueError("parameterised plan compilation accepts no platform ports")
        by_schema = {value.SCHEMA: value for value in records}
        if len(by_schema) != len(records):
            raise ValueError("parameterised plan compiler received duplicate record schemas")
        source_type, campaign_type, qualification_type, extension_type = self._root_types()
        required = {
            EvidenceProfileSelection.SCHEMA,
            source_type.SCHEMA,
            campaign_type.SCHEMA,
            qualification_type.SCHEMA,
            extension_type.SCHEMA,
        }
        if not required.issubset(by_schema):
            raise ValueError("parameterised plan compiler lacks a required strict/config record")
        extension = by_schema[extension_type.SCHEMA]
        qualification_config = by_schema[qualification_type.SCHEMA]
        if not isinstance(extension, ResponseExperimentExtensionSet) or not isinstance(
            qualification_config,
            ResponseQualificationConfig,
        ):
            raise TypeError("parameterised plan records have another canonical type")
        optional_records = {
            AdmissionExperimentConfig.SCHEMA: extension.admission_config,
            ProspectiveUseExperimentConfig.SCHEMA: extension.prospective_evaluation_config,
        }
        expected_schemas = required | {
            schema for schema, value in optional_records.items() if value is not None
        }
        if ResponseSubstrateBinding.SCHEMA in by_schema:
            expected_schemas.add(ResponseSubstrateBinding.SCHEMA)
        if set(by_schema) != expected_schemas:
            raise ValueError("parameterised plan compiler record roster changes applicability")
        if qualification_config != extension.identification_config or any(
            by_schema.get(schema) != value
            for schema, value in optional_records.items()
            if value is not None
        ):
            raise ValueError("standalone parameterised config differs from its extension set")
        evidence = by_schema[EvidenceProfileSelection.SCHEMA]
        source = by_schema[source_type.SCHEMA]
        linked = by_schema[campaign_type.SCHEMA]
        if not isinstance(evidence, EvidenceProfileSelection):
            raise TypeError("parameterised compiler evidence root has another type")
        if (
            not isinstance(source, (SourcePipelineProfile, NativeSourceProfile))
            or type(source) is not source_type
        ):
            raise TypeError("parameterised compiler source root has another type")
        if (
            not isinstance(linked, (LinkedCampaignProfile, CampaignSpec))
            or type(linked) is not campaign_type
        ):
            raise TypeError("parameterised compiler linked root has another type")
        return compile_response_experiment(
            extension_set=extension,
            evidence_profile_selection=evidence,
            source_pipeline_profile=source,
            linked_campaign_profile=linked,
        )


@dataclass(frozen=True, slots=True)
class ObservationOrderPlanCompilerFactory:
    "Compile an action-free observation-order carrier from exact prospective roots."

    binding: ExecutableCapabilityBinding = OBSERVATION_ORDER_PLAN_EXECUTABLE_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> object:
        by_type = {type(value): value for value in records}
        expected = {
            EvidenceProfileSelection,
            ObservationOrderExperimentExtension,
            SourcePipelineProfile,
        }
        if platform_ports or len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("observation-order compiler requires its exact three roots")
        return compile_observation_order_experiment(
            extension=cast(
                ObservationOrderExperimentExtension,
                by_type[ObservationOrderExperimentExtension],
            ),
            evidence_profile_selection=cast(
                EvidenceProfileSelection,
                by_type[EvidenceProfileSelection],
            ),
            source_pipeline_profile=cast(
                SourcePipelineProfile,
                by_type[SourcePipelineProfile],
            ),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.linked-campaign",
    contribution_version="1.0.0",
    bindings=(
        EVIDENCE_DERIVED_PROGRAMME_AUTHOR_EXECUTABLE_BINDING,
        LINKED_CAMPAIGN_EXECUTABLE_BINDING,
        NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING,
        NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING,
        OBSERVATION_ORDER_PLAN_EXECUTABLE_BINDING,
        PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING,
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    EvidenceDerivedStudyAuthorFactory(),
    LinkedCampaignCoordinatorFactory(),
    ResponseExperimentPlanCompilerFactory(NATIVE_CROSSFIT_PLAN_EXECUTABLE_BINDING),
    ResponseExperimentPlanCompilerFactory(NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_EXECUTABLE_BINDING),
    ObservationOrderPlanCompilerFactory(),
    ResponseExperimentPlanCompilerFactory(),
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    *_RESPONSE_EXPERIMENT_CONFIG_TYPES,
    *NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES,
    *NATIVE_CROSSFIT_RECORD_TYPES,
)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "EVIDENCE_DERIVED_PROGRAMME_AUTHOR_EXECUTABLE_BINDING",
    'EvidenceDerivedStudyAuthorFactory',
    "LINKED_CAMPAIGN_EXECUTABLE_BINDING",
    'LinkedCampaignCoordinatorFactory',
    "OBSERVATION_ORDER_PLAN_EXECUTABLE_BINDING",
    'ObservationOrderPlanCompilerFactory',
    "PARAMETERISED_RESPONSE_PLAN_EXECUTABLE_BINDING",
    'ResponseExperimentPlanCompilerFactory',
]
