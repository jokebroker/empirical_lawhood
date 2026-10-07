"""Static Six-matrix response control/config discovery; no programme or prospective evaluation plan is executed."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind

from .contracts import MatrixResponseOuterStudyAuthoringConfig, MatrixResponseProspectiveEvaluationTopology, ProtectedObservableAccessManifest
from .prospective import MatrixResponseOuterProspectiveUmbrella
from .protected_access import MatrixResponseProtectedFieldProjection


_IMPLEMENTATION_SHA256 = sha256(b"six-matrix-response-control-compatibility-services").hexdigest()


def _decoder(
    record_type: type[CanonicalRecord],
    suffix: str,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.six-matrix-response.{suffix}-decoder",
        component_key=f"six-matrix-response-{suffix}-decoder",
        component_version="1.0.0",
        kind=ExtensionComponentKind.CONFIG_DECODER,
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(record_type.SCHEMA,),
        implementation_sha256=_IMPLEMENTATION_SHA256,
    )


MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER = _decoder(
    MatrixResponseOuterStudyAuthoringConfig,
    "outer-programme-authoring-config",
)
MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER = _decoder(
    MatrixResponseProspectiveEvaluationTopology,
    "prospective-evaluation-topology",
)
MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER = _decoder(
    ProtectedObservableAccessManifest,
    "protected-observable-access-manifest",
)

MATRIX_RESPONSE_PROGRAMME_AUTHOR = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.arbitrary-chart-programme-author",
    component_key="six-matrix-response-arbitrary-chart-programme-author",
    component_version="1.0.0",
    kind=ExtensionComponentKind.PROGRAMME_AUTHOR,
    input_schema_ids=(MatrixResponseOuterStudyAuthoringConfig.SCHEMA,),
    output_schema_ids=(AdmissionControllerStudy.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)
MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.prospective-evaluation-coordinator",
    component_key="six-matrix-response-prospective-evaluation-coordinator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(MatrixResponseProspectiveEvaluationTopology.SCHEMA,),
    output_schema_ids=(MatrixResponseOuterProspectiveUmbrella.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)
MATRIX_RESPONSE_PROTECTED_FIELD_PROJECTOR = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.protected-field-projector",
    component_key="six-matrix-response-protected-field-projector",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(ProtectedObservableAccessManifest.SCHEMA,),
    output_schema_ids=(MatrixResponseProtectedFieldProjection.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)


EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.six-matrix-response-control-compatibility",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.CAMPAIGN,
    evidence_profile_registries=(),
    capability_manifests=(),
    candidate_capability_registrations=(),
    dataset_capability_registries=(),
    profile_registrations=(),
    method_registrations=(),
    config_decoders=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER,
                MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER,
                MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    artifact_validators=(),
    runtime_providers=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR, MATRIX_RESPONSE_PROTECTED_FIELD_PROJECTOR),
            key=lambda value: value.registration_id,
        )
    ),
    study_authors=(
        ObjectIdentity.from_record(MATRIX_RESPONSE_PROGRAMME_AUTHOR.registration_id, MATRIX_RESPONSE_PROGRAMME_AUTHOR),
    ),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = [
    "MATRIX_RESPONSE_PROGRAMME_AUTHOR",
    "MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER",
    "MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER",
    "MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR",
    "MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER",
    "MATRIX_RESPONSE_PROTECTED_FIELD_PROJECTOR",
    "EXTENSION_BUNDLE_CONTRIBUTION",
]
