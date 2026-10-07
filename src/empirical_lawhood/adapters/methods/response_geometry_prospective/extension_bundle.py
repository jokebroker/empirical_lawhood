"""Static finite assay response projection and excluded evaluation discovery."""

from dataclasses import replace

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import bundle, manifest
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeSegmentResult
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA

from .projection import ASSAY_DIAGNOSTICS_SCHEMA
from .qualification import ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig, ResponseGeometryAssayViewReport, ResponseGeometryAssayEvaluation

ASSAY_PROJECTION_CAPABILITY = manifest(
    "projection",
    ResponseGeometryAssayProjectionConfig,
    CapabilityKind.TRANSFORM,
    (
        ResponseGeometryAssayProjectionConfig.SCHEMA,
        ResponseGeometryAssayNativeSegmentResult.SCHEMA,
        ASSAY_HDF5_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (ResponseGeometryAssayViewReport.SCHEMA, ASSAY_DIAGNOSTICS_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_SEALED,
    120,
)
ASSAY_EVALUATION_CAPABILITY = manifest(
    "evaluation",
    ResponseGeometryAssayEvaluationConfig,
    CapabilityKind.EVALUATOR,
    (
        ResponseGeometryAssayEvaluationConfig.SCHEMA,
        ResponseGeometryAssayViewReport.SCHEMA,
        ASSAY_DIAGNOSTICS_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        ResponseGeometryAssayEvaluation.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.RESPONSE,
    OutcomeAccess.EVALUATION_REVEALED,
    120,
)
_projection, ASSAY_PROJECTION_COMPONENTS = bundle(
    "projection",
    ExtensionContributionKind.METHOD,
    ASSAY_PROJECTION_CAPABILITY,
    (ResponseGeometryAssayProjectionConfig,),
    ASSAY_DIAGNOSTICS_SCHEMA,
)
_evaluation, ASSAY_EVALUATION_COMPONENTS = bundle(
    "evaluation",
    ExtensionContributionKind.METHOD,
    ASSAY_EVALUATION_CAPABILITY,
    (ResponseGeometryAssayEvaluationConfig,),
    None,
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _projection,
    contribution_id="extension-contribution.response-geometry-assay.methods",
    capability_manifests=tuple(
        sorted(
            (*_projection.capability_manifests, *_evaluation.capability_manifests),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                *_projection.candidate_capability_registrations,
                *_evaluation.candidate_capability_registrations,
            ),
            key=lambda value: value.manifest.registry_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (*_projection.config_decoders, *_evaluation.config_decoders),
            key=lambda value: value.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (*_projection.runtime_providers, *_evaluation.runtime_providers),
            key=lambda value: value.object_id,
        )
    ),
)
