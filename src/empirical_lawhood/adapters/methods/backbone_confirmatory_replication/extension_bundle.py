"Additive static registrations needed by fixed-panel measurement through controller use replications."

from __future__ import annotations

import hashlib

from empirical_lawhood.adapters.methods.confirmatory_finite_action import ConfirmatoryFiniteActionConfig, ConfirmatoryFiniteActionResult
from empirical_lawhood.adapters.geometry.composite_gate_margin import CompositeGateMarginResult, CompositeGateMarginSpec
from empirical_lawhood.adapters.methods.confirmatory_finite_action_registration import (
    CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE,
    CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
    CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY,
    confirmatory_finite_action_manifest,
)
from empirical_lawhood.adapters.methods.qualification_profiles import ComponentQualificationProfile
from empirical_lawhood.adapters.methods.confirmatory_finite_action import (
    CONFIRMATORY_FINITE_ACTION_KEY,
    CONFIRMATORY_FINITE_ACTION_VERSION,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.nested_controller_evaluation import RepeatedDeliveryControllerEvaluationPlan
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.controller_compiler import CompiledAdmissionControllerStudy
from empirical_lawhood.runtime.controller_evaluation_nested import RepeatedDeliveryControllerCohortAdjudication, RevealedRepeatedDeliveryControllerBundle
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration


_METHOD_IMPLEMENTATION_SHA256 = (
    "afd00527769340fd4b3c375a4223958871d7f4f164e267014382692575e50acc"
)
_REGISTRATION_IMPLEMENTATION_SHA256 = (
    "b207067e90fba2905e36a9bf1159428b040e30793e5c1ec33fedb15dc111d5c9"
)
_NESTED_CONTROLLER_EVALUATION_IMPLEMENTATION_SHA256 = (
    "9b74f17ce642e655097db6180d69d0d9a8fe269f1ca41142e6572774a4a21292"
)
_COMPOSITE_GATE_MARGIN_IMPLEMENTATION_SHA256 = (
    "b82ef6b60a1b9d6187e0e113e3b143af3523d42c402df5ee8113c0415bd803a4"
)
_PROFILE_IMPLEMENTATION_SHA256 = (
    "9ebfec90d816bfbe92cd33c01014019b9561cb83d22b3d3aeb8b4d08af165ab7"
)


CONFIRMATORY_FINITE_ACTION_MANIFEST = confirmatory_finite_action_manifest(
    implementation_sha256=_METHOD_IMPLEMENTATION_SHA256
)

CONFIRMATORY_FINITE_ACTION_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    manifest=CONFIRMATORY_FINITE_ACTION_MANIFEST,
    provider_key=CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY,
    provider_version=CONFIRMATORY_FINITE_ACTION_VERSION,
    config_media_type=CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE,
    maximum_config_bytes=CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
)

CONFIRMATORY_FINITE_ACTION_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    registration_id="profile-registration.confirmatory-finite-action",
    profile_key="confirmatory-finite-action",
    profile_version="1.0.0",
    kind=ExtensionProfileKind.OBJECTIVE_TERMINAL,
    profile_schema=ComponentQualificationProfile.SCHEMA,
    profile_schema_sha256=hashlib.sha256(
        ComponentQualificationProfile.SCHEMA.encode()
    ).hexdigest(),
    implementation_sha256=_PROFILE_IMPLEMENTATION_SHA256,
)

CONFIRMATORY_FINITE_ACTION_METHOD = ExtensionComponentRegistration(
    registration_id="component.confirmatory-finite-action",
    component_key=CONFIRMATORY_FINITE_ACTION_KEY,
    component_version=CONFIRMATORY_FINITE_ACTION_VERSION,
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.input_schema_ids,
    output_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.output_schema_ids,
    implementation_sha256=_METHOD_IMPLEMENTATION_SHA256,
)

CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.confirmatory-finite-action-config-decoder",
    component_key="confirmatory-finite-action-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(ConfirmatoryFiniteActionConfig.SCHEMA,),
    output_schema_ids=(ConfirmatoryFiniteActionConfig.SCHEMA,),
    implementation_sha256=_REGISTRATION_IMPLEMENTATION_SHA256,
)

CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.confirmatory-finite-action-artifact-validator",
    component_key="confirmatory-finite-action-artifact-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(ConfirmatoryFiniteActionResult.SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=_REGISTRATION_IMPLEMENTATION_SHA256,
)

CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.confirmatory-finite-action-runtime-provider",
    component_key=CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY,
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.input_schema_ids,
    output_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.output_schema_ids,
    implementation_sha256=_REGISTRATION_IMPLEMENTATION_SHA256,
)

CONFIRMATORY_NESTED_CONTROLLER_EVALUATOR = ExtensionComponentRegistration(
    registration_id="component.confirmatory-nested-controller-evaluator",
    component_key="controller.confirmatory-nested-controller-evaluation",
    component_version="1.0.0",
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=tuple(
        sorted(
            (
                CompiledAdmissionControllerStudy.SCHEMA,
                RepeatedDeliveryControllerEvaluationPlan.SCHEMA,
                RevealedRepeatedDeliveryControllerBundle.SCHEMA,
            )
        )
    ),
    output_schema_ids=(RepeatedDeliveryControllerCohortAdjudication.SCHEMA,),
    implementation_sha256=_NESTED_CONTROLLER_EVALUATION_IMPLEMENTATION_SHA256,
)

COMPOSITE_GATE_MARGIN_METHOD = ExtensionComponentRegistration(
    registration_id="component.composite-gate-margin",
    component_key="geometry.composite-gate-margin",
    component_version="1.0.0",
    kind=ExtensionComponentKind.METHOD,
    input_schema_ids=(CompositeGateMarginSpec.SCHEMA,),
    output_schema_ids=(CompositeGateMarginResult.SCHEMA,),
    implementation_sha256=_COMPOSITE_GATE_MARGIN_IMPLEMENTATION_SHA256,
)

EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.confirmatory-replication",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.METHOD,
    evidence_profile_registries=(),
    capability_manifests=(CONFIRMATORY_FINITE_ACTION_MANIFEST,),
    candidate_capability_registrations=(
        CONFIRMATORY_FINITE_ACTION_CANDIDATE_REGISTRATION,
    ),
    dataset_capability_registries=(),
    profile_registrations=(CONFIRMATORY_FINITE_ACTION_PROFILE_REGISTRATION,),
    method_registrations=tuple(
        sorted(
            (
                ObjectIdentity.from_record(
                    CONFIRMATORY_FINITE_ACTION_METHOD.registration_id,
                    CONFIRMATORY_FINITE_ACTION_METHOD,
                ),
                ObjectIdentity.from_record(
                    CONFIRMATORY_NESTED_CONTROLLER_EVALUATOR.registration_id,
                    CONFIRMATORY_NESTED_CONTROLLER_EVALUATOR,
                ),
                ObjectIdentity.from_record(
                    COMPOSITE_GATE_MARGIN_METHOD.registration_id,
                    COMPOSITE_GATE_MARGIN_METHOD,
                ),
            ),
            key=lambda value: value.object_id,
        )
    ),
    config_decoders=(
        ObjectIdentity.from_record(
            CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.registration_id,
            CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER,
        ),
    ),
    artifact_validators=(
        ObjectIdentity.from_record(
            CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR.registration_id,
            CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR,
        ),
    ),
    runtime_providers=(
        ObjectIdentity.from_record(
            CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER.registration_id,
            CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER,
        ),
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = ["EXTENSION_BUNDLE_CONTRIBUTION"]
