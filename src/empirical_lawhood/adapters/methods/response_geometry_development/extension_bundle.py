"""Import-safe D discovery; finite cutoffs and resources, without source contact."""

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import bundle, manifest
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID, DEVELOPMENT_HDF5_SCHEMA, ResponseGeometryDevelopmentNativeSegmentResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_projection import DEVELOPMENT_DATA_SCHEMA, ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentViewReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentMethodConfig, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import ResponseGeometryDevelopmentQualificationConfig, DEVELOPMENT_VALIDATION_SCHEMA
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentDevelopmentResult, ResponseGeometryDevelopmentCloseoutConfig


_STAGE = LinkedCampaignStageEnvelope.SCHEMA
# Each manifest charges the complete task's scan/output ceilings. The exact
# derived-plan proof must sum these alongside every native segment.
_DEFINITIONS: tuple[
    tuple[
        str,
        type[CanonicalRecord],
        CapabilityKind,
        tuple[str, ...],
        tuple[str, ...],
        str | None,
        int,
        int,
    ],
    ...,
] = (
    (
        "projection",
        ResponseGeometryDevelopmentProjectionConfig,
        CapabilityKind.TRANSFORM,
        (ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA, DEVELOPMENT_HDF5_SCHEMA),
        (ResponseGeometryDevelopmentViewReport.SCHEMA, DEVELOPMENT_DATA_SCHEMA),
        DEVELOPMENT_DATA_SCHEMA,
        120,
        32,
    ),
    (
        "development",
        ResponseGeometryDevelopmentMethodConfig,
        CapabilityKind.ANALYSIS,
        (
            ResponseGeometryDevelopmentViewReport.SCHEMA,
            DEVELOPMENT_DATA_SCHEMA,
            ResponseGeometryDevelopmentFitResult.SCHEMA,
            DEVELOPMENT_FIT_SCHEMA,
            ResponseGeometryDevelopmentCalibrationResult.SCHEMA,
        ),
        (ResponseGeometryDevelopmentFitResult.SCHEMA, DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentCalibrationResult.SCHEMA, ResponseGeometryDevelopmentSupportResult.SCHEMA),
        DEVELOPMENT_FIT_SCHEMA,
        28800,
        72,
    ),
    (
        "assessment",
        ResponseGeometryDevelopmentQualificationConfig,
        CapabilityKind.LAW_IDENTIFIER,
        (
            ResponseGeometryDevelopmentViewReport.SCHEMA,
            DEVELOPMENT_DATA_SCHEMA,
            ResponseGeometryDevelopmentFitResult.SCHEMA,
            DEVELOPMENT_FIT_SCHEMA,
            ResponseGeometryDevelopmentCalibrationResult.SCHEMA,
            ResponseGeometryDevelopmentSupportResult.SCHEMA,
        ),
        (ResponseGeometryDevelopmentContextResult.SCHEMA, DEVELOPMENT_VALIDATION_SCHEMA, ResponseGeometryDevelopmentGeometryReport.SCHEMA),
        DEVELOPMENT_VALIDATION_SCHEMA,
        28800,
        320,
    ),
    (
        "evaluation",
        ResponseGeometryDevelopmentCloseoutConfig,
        CapabilityKind.EVALUATOR,
        (ResponseGeometryDevelopmentContextResult.SCHEMA, DEVELOPMENT_VALIDATION_SCHEMA, ResponseGeometryDevelopmentGeometryReport.SCHEMA),
        (ResponseGeometryDevelopmentDevelopmentResult.SCHEMA, ScientificAdjudicationRecord.SCHEMA),
        None,
        120,
        8,
    ),
)

DEVELOPMENT_CAPABILITIES = tuple(
    manifest(
        role,
        config,
        kind,
        (config.SCHEMA, *inputs, _STAGE),
        (*outputs, _STAGE),
        EvidenceCeiling.LOCAL_LAW
        if role in ("assessment", "evaluation")
        else EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_REVEALED
        if role == "evaluation"
        else OutcomeAccess.DEVELOPMENT_VISIBLE,
        seconds,
        namespace=DEVELOPMENT_PANEL_ID,
        resources=ResourceBudget(
            1,
            8 * 1024**3,
            0,
            seconds,
            (2 if role in ("development", "assessment") else 1) * 1024**3,
            output_mib * 1024**2,
        ),
        read_permission=CapabilityPermission.READ_DEVELOPMENT,
        implementation_id="response-geometry-development-methods",
        conformance_ids=(
            "exact-d-root-role-cutoff",
            "no-native-effect-in-method",
            "retain-unavailable-models-and-independent-roots",
        ),
    )
    for role, config, kind, inputs, outputs, _, seconds, output_mib in _DEFINITIONS
)
_BUILT = tuple(
    bundle(
        role, ExtensionContributionKind.METHOD, capability, (config,), hdf5, namespace=DEVELOPMENT_PANEL_ID
    )
    for (role, config, _, _, _, hdf5, _, _), capability in zip(
        _DEFINITIONS, DEVELOPMENT_CAPABILITIES, strict=True
    )
)
DEVELOPMENT_COMPONENTS = tuple(components for _, components in _BUILT)
DEVELOPMENT_CONFIG_TYPES = tuple(definition[1] for definition in _DEFINITIONS)
DEVELOPMENT_ROLES = tuple(definition[0] for definition in _DEFINITIONS)
_CONTRIBUTIONS = tuple(value for value, _ in _BUILT)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _CONTRIBUTIONS[0],
    contribution_id=f"extension-contribution.{DEVELOPMENT_PANEL_ID}.methods",
    capability_manifests=tuple(sorted(DEVELOPMENT_CAPABILITIES, key=lambda v: v.registry_id)),
    candidate_capability_registrations=tuple(
        sorted(
            (v for c in _CONTRIBUTIONS for v in c.candidate_capability_registrations),
            key=lambda v: v.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (v for c in _CONTRIBUTIONS for v in c.profile_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted((v for c in _CONTRIBUTIONS for v in c.config_decoders), key=lambda v: v.object_id)
    ),
    artifact_validators=tuple(
        sorted(
            (v for c in _CONTRIBUTIONS for v in c.artifact_validators), key=lambda v: v.object_id
        )
    ),
    runtime_providers=tuple(
        sorted((v for c in _CONTRIBUTIONS for v in c.runtime_providers), key=lambda v: v.object_id)
    ),
)
