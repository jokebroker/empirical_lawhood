"""Static linked-campaign provider and evidence-derived author declaration."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.planning.native_source import NativeSourceProfile, NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeCrossfitConfig, NativeCrossfitExperiment
from empirical_lawhood.planning.response_experiment import ResponseQualificationConfig, ResponseExperimentExtensionSet, AdmissionExperimentConfig, ProspectiveUseExperimentConfig
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.response_experiment import CompiledResponseExperiment, ResponseExperimentCompilationReceipt, build_response_experiment_decoder_registrations
from empirical_lawhood.runtime.observation_order import CompiledObservationOrderExperiment, ObservationOrderCompilationReceipt, ObservationOrderSubstrateBinding
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding


_PLANNING_IMPLEMENTATION_SHA256 = (
    "2db15149a355abd504bbb39e3339311cc8306df4adf6e66bad8cb2afb7be095b"
)
_RUNTIME_IMPLEMENTATION_SHA256 = (
    "5abb597589839bf0ac6840535a574d12ff5a6f1e39e04a0f54b9a5b8ae18773f"
)
_AUTHOR_IMPLEMENTATION_SHA256 = (
    "7198337245a3a85fa5c9edf235a7d99130d8ef6a64a4dbe75f3653d5bcaf3e47"
)


_RESPONSE_EXPERIMENT_DECODER_REGISTRATIONS = build_response_experiment_decoder_registrations(
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)
RESPONSE_EXPERIMENT_CONFIG_DECODERS = tuple(
    ExtensionComponentRegistration(
        registration_id=(
            "component."
            + registration.payload_schema.removeprefix("empirical-lawhood/")
            .replace("/", ".")
            .replace("_", "-")
            + ".decoder"
        ),
        component_key=registration.decoder_key,
        component_version=registration.decoder_version,
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(registration.payload_schema,),
        output_schema_ids=(registration.payload_schema,),
        implementation_sha256=registration.implementation_sha256,
    )
    for registration in _RESPONSE_EXPERIMENT_DECODER_REGISTRATIONS
)

RESPONSE_SUBSTRATE_BINDING_DECODER = ExtensionComponentRegistration(
    registration_id="component.response-substrate-binding.decoder",
    component_key="response-substrate-binding-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(ResponseSubstrateBinding.SCHEMA,),
    output_schema_ids=(ResponseSubstrateBinding.SCHEMA,),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

PARAMETERISED_RESPONSE_PLAN_COMPILER = ExtensionComponentRegistration(
    registration_id="component.parameterised-response-plan-compiler",
    component_key="parameterised-response-plan-compiler",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=tuple(
        sorted(
            (
                ResponseQualificationConfig.SCHEMA,
                ResponseExperimentExtensionSet.SCHEMA,
                AdmissionExperimentConfig.SCHEMA,
                ProspectiveUseExperimentConfig.SCHEMA,
                ResponseSubstrateBinding.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                CompiledResponseExperiment.SCHEMA,
                ResponseExperimentCompilationReceipt.SCHEMA,
            )
        )
    ),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

OBSERVATION_ORDER_PLAN_COMPILER = ExtensionComponentRegistration(
    registration_id="component.observation-order-plan-compiler",
    component_key="observation-order-plan-compiler",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(ObservationOrderExperimentExtension.SCHEMA,),
    output_schema_ids=tuple(
        sorted(
            (
                CompiledObservationOrderExperiment.SCHEMA,
                ObservationOrderCompilationReceipt.SCHEMA,
            )
        )
    ),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES = (
    NativeSourceProfile,
    NativeLawQualificationConfig,
    NativeLawQualificationExperiment,
)
_NATIVE_IMPLEMENTATION = hashlib.sha256(
    b"parameterised-native-response-law-qualification-strict-root"
).hexdigest()
NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS = build_response_experiment_decoder_registrations(
    implementation_sha256=_NATIVE_IMPLEMENTATION,
    record_types=NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES,
)
NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS = tuple(
    ExtensionComponentRegistration(
        registration_id=f"component.{r.decoder_key}",
        component_key=r.decoder_key,
        component_version=r.decoder_version,
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(r.payload_schema,),
        output_schema_ids=(r.payload_schema,),
        implementation_sha256=r.implementation_sha256,
    )
    for r in NATIVE_RESPONSE_LAW_QUALIFICATION_DECODER_REGISTRATIONS
)
NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER = replace(
    PARAMETERISED_RESPONSE_PLAN_COMPILER,
    registration_id="component.native-response-law-qualification-plan-compiler",
    component_key="native-response-law-qualification-plan-compiler",
    input_schema_ids=tuple(sorted(r.SCHEMA for r in NATIVE_RESPONSE_LAW_QUALIFICATION_RECORD_TYPES)),
    implementation_sha256=_NATIVE_IMPLEMENTATION,
)

NATIVE_CROSSFIT_RECORD_TYPES = (
    NativeCrossfitConfig,
    NativeCrossfitExperiment,
)
_NATIVE_CROSSFIT_IMPLEMENTATION = hashlib.sha256(
    b"parameterised-native-response-law-qualification-nested-crossfit-all-development"
).hexdigest()
NATIVE_CROSSFIT_DECODER_REGISTRATIONS = build_response_experiment_decoder_registrations(
    implementation_sha256=_NATIVE_CROSSFIT_IMPLEMENTATION,
    record_types=NATIVE_CROSSFIT_RECORD_TYPES,
)
NATIVE_CROSSFIT_DECODERS = tuple(
    ExtensionComponentRegistration(
        registration_id=f"component.{r.decoder_key}",
        component_key=r.decoder_key,
        component_version=r.decoder_version,
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(r.payload_schema,),
        output_schema_ids=(r.payload_schema,),
        implementation_sha256=r.implementation_sha256,
    )
    for r in NATIVE_CROSSFIT_DECODER_REGISTRATIONS
)
NATIVE_CROSSFIT_PLAN_COMPILER = replace(
    NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER,
    registration_id="component.native-response-law-qualification-crossfit-plan-compiler",
    component_key="native-response-law-qualification-crossfit-plan-compiler",
    input_schema_ids=tuple(
        sorted(
            (
                NativeSourceProfile.SCHEMA,
                *(t.SCHEMA for t in NATIVE_CROSSFIT_RECORD_TYPES),
            )
        )
    ),
    implementation_sha256=_NATIVE_CROSSFIT_IMPLEMENTATION,
)

OBSERVATION_ORDER_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.observation-order-config-decoder",
    component_key="observation-order-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(ObservationOrderExperimentExtension.SCHEMA,),
    output_schema_ids=(ObservationOrderExperimentExtension.SCHEMA,),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_SUBSTRATE_DECODER = ExtensionComponentRegistration(
    registration_id="component.observation-order-substrate-decoder",
    component_key="observation-order-substrate-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(ObservationOrderSubstrateBinding.SCHEMA,),
    output_schema_ids=(ObservationOrderSubstrateBinding.SCHEMA,),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

OBSERVATION_ORDER_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.observation-order",
    profile_key='observation-order-relation',
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=ExperimentObjectiveProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        ExperimentObjectiveProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PLANNING_IMPLEMENTATION_SHA256,
)


LINKED_CAMPAIGN_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.linked-campaign",
    profile_key="linked-campaign",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.LINKED_CAMPAIGN,
    profile_schema=LinkedCampaignProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        LinkedCampaignProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PLANNING_IMPLEMENTATION_SHA256,
)

LINKED_CAMPAIGN_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.linked-campaign-runtime-provider",
    component_key="linked-campaign-runtime-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(LinkedCampaignProfile.SCHEMA,),
    output_schema_ids=(ProtocolTemplate.SCHEMA,),
    implementation_sha256=_RUNTIME_IMPLEMENTATION_SHA256,
)

EVIDENCE_DERIVED_PROGRAMME_AUTHOR = ExtensionComponentRegistration(
    registration_id="component.evidence-derived-programme-author",
    component_key="evidence-derived-programme-author",
    component_version="1.0.0",
    kind=ExtensionComponentKind.PROGRAMME_AUTHOR,
    input_schema_ids=tuple(
        sorted(
            (
                'empirical-lawhood/kernel/response-atlas',
                'empirical-lawhood/planning/receipt-admission-spec',
                'empirical-lawhood/planning/controlled-map-reachability-spec',
            )
        )
    ),
    output_schema_ids=(AdmissionControllerStudy.SCHEMA,),
    implementation_sha256=_AUTHOR_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.linked-campaign",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.CAMPAIGN,
    evidence_profile_registries=(),
    capability_manifests=(),
    candidate_capability_registrations=(),
    dataset_capability_registries=(),
    profile_registrations=(
        LINKED_CAMPAIGN_PROFILE_REGISTRATION,
        OBSERVATION_ORDER_PROFILE_REGISTRATION,
    ),
    method_registrations=(),
    config_decoders=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (
                    OBSERVATION_ORDER_CONFIG_DECODER,
                    OBSERVATION_ORDER_SUBSTRATE_DECODER,
                    *RESPONSE_EXPERIMENT_CONFIG_DECODERS,
                    *NATIVE_RESPONSE_LAW_QUALIFICATION_DECODERS,
                    *NATIVE_CROSSFIT_DECODERS,
                    RESPONSE_SUBSTRATE_BINDING_DECODER,
                )
            ),
            key=lambda value: value.object_id,
        )
    ),
    artifact_validators=(),
    runtime_providers=(
        ObjectIdentity.from_record(
            LINKED_CAMPAIGN_PROVIDER.registration_id,
            LINKED_CAMPAIGN_PROVIDER,
        ),
        ObjectIdentity.from_record(
            NATIVE_CROSSFIT_PLAN_COMPILER.registration_id,
            NATIVE_CROSSFIT_PLAN_COMPILER,
        ),
        ObjectIdentity.from_record(
            NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER.registration_id,
            NATIVE_RESPONSE_LAW_QUALIFICATION_PLAN_COMPILER,
        ),
        ObjectIdentity.from_record(
            OBSERVATION_ORDER_PLAN_COMPILER.registration_id,
            OBSERVATION_ORDER_PLAN_COMPILER,
        ),
        ObjectIdentity.from_record(
            PARAMETERISED_RESPONSE_PLAN_COMPILER.registration_id,
            PARAMETERISED_RESPONSE_PLAN_COMPILER,
        ),
    ),
    study_authors=(
        ObjectIdentity.from_record(
            EVIDENCE_DERIVED_PROGRAMME_AUTHOR.registration_id,
            EVIDENCE_DERIVED_PROGRAMME_AUTHOR,
        ),
    ),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
